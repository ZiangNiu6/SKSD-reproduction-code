"""Numerical core for the Gaussian--Bernoulli RBM reproduction study.

The definitions were migrated from
``code/rbm-python/experiments/gaussian_bernoulli_influence_vs_brueck.py``
so the numbered reproduction entry point is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


METHOD_INFLUENCE = "SKSD-influence"
METHOD_BRUECK_CORRECTED = "Brueck-corrected"
BRUECK_CORRECTION_VERSION = "factor-corrected-midpoint-v1"
ALTERNATIVE_ROTATION = "loading-rotation"
ALTERNATIVE_OMITTED = "omitted-hidden-unit"
ALTERNATIVE_TAIL = "heavy-tailed-visible"
ALTERNATIVES = (
    ALTERNATIVE_ROTATION,
    ALTERNATIVE_OMITTED,
    ALTERNATIVE_TAIL,
)
ALTERNATIVE_CODES = {
    ALTERNATIVE_ROTATION: 1,
    ALTERNATIVE_OMITTED: 2,
    ALTERNATIVE_TAIL: 3,
}
SHARED_NULL = "shared-null"
DESIGN_VERSION = "gb-rbm-symmetric-three-alternatives-v4"
OUTPUT_SCHEMA_VERSION = 3
DEFAULT_TRUE_SCALE_MULTIPLIER = 1.5
DEFAULT_ROTATION_GRID = (0.0, 0.069, 0.138, 0.207, 0.276, 0.345)
DEFAULT_OMITTED_GRID = (0.0, 0.302, 0.604, 0.906, 1.208, 1.51)
DEFAULT_TAIL_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


@dataclass(frozen=True)
class Config:
    visible_dimension: int
    hidden_dimension: int
    sample_size: int
    bootstrap_draws: int
    monte_carlo_reps: int
    fixed_bandwidth: float
    alpha: float
    seed: int
    model_seed: int
    signal_grid: tuple[float, ...]
    omitted_unit_grid: tuple[float, ...]
    heavy_tail_grid: tuple[float, ...]
    true_loading_scale_multiplier: float
    loading_scale_lower: float
    loading_scale_upper: float
    optimizer_maxiter: int
    optimizer_gtol: float
    output_dir: str


def validate_config(config: Config) -> None:
    if config.visible_dimension < 2 or config.hidden_dimension < 2:
        raise ValueError("D and M must both be at least 2")
    if config.hidden_dimension > 20:
        raise ValueError("M must be at most 20 for exact hidden-state enumeration")
    if 2 * config.hidden_dimension > config.visible_dimension:
        raise ValueError("2M must not exceed D for Gram-preserving rotations")
    if config.hidden_dimension >= config.visible_dimension:
        raise ValueError("M must be smaller than D for the omitted hidden unit")
    if config.sample_size <= config.hidden_dimension + 2:
        raise ValueError("N must exceed the M-dimensional nuisance by at least two")
    if config.bootstrap_draws < 1 or config.monte_carlo_reps < 1:
        raise ValueError("B and REPS must be positive")
    if config.fixed_bandwidth <= 0.0:
        raise ValueError("H_FIXED must be positive and fixed before simulation")
    if not 0.0 < config.alpha < 1.0:
        raise ValueError("ALPHA must lie strictly between zero and one")
    if not 0.0 < config.loading_scale_lower < config.loading_scale_upper:
        raise ValueError("require 0 < SCALE_LOWER < SCALE_UPPER")
    if config.true_loading_scale_multiplier <= 0.0:
        raise ValueError("TRUE_SCALE_MULTIPLIER must be positive")
    true_scale_min = 0.75 * config.true_loading_scale_multiplier
    true_scale_max = 1.10 * config.true_loading_scale_multiplier
    if not (
        config.loading_scale_lower < true_scale_min
        and true_scale_max < config.loading_scale_upper
    ):
        raise ValueError(
            "the seeded true loading scales must lie strictly inside the "
            "optimization bounds"
        )
    if any(signal >= 1.0 for signal in config.signal_grid):
        raise ValueError("every rotation signal must be smaller than one")
    grids = (
        config.signal_grid,
        config.omitted_unit_grid,
        config.heavy_tail_grid,
    )
    if any(0.0 not in grid for grid in grids):
        raise ValueError("every alternative grid must contain the shared null 0")


def alternative_grids(config: Config) -> dict[str, tuple[float, ...]]:
    return {
        ALTERNATIVE_ROTATION: config.signal_grid,
        ALTERNATIVE_OMITTED: config.omitted_unit_grid,
        ALTERNATIVE_TAIL: config.heavy_tail_grid,
    }


def computational_cells(config: Config) -> list[tuple[str, float]]:
    """Return one common null followed by every nonnull panel cell."""
    cells: list[tuple[str, float]] = [(SHARED_NULL, 0.0)]
    for alternative, grid in alternative_grids(config).items():
        cells.extend(
            (alternative, signal) for signal in grid if signal > 0.0
        )
    return cells


def heavy_tail_degrees_of_freedom(signal: float) -> float:
    if signal < 0.0:
        raise ValueError("heavy-tail signal must be nonnegative")
    return math.inf if signal == 0.0 else 2.0 + 1.0 / signal


@dataclass(frozen=True)
class FitResult:
    theta: np.ndarray
    objective: float
    iterations: int
    evaluations: int
    success: bool
    gradient_norm: float
    boundary_hit: bool
    message: str


@dataclass(frozen=True)
class Geometry:
    kernel: np.ndarray
    difference: np.ndarray
    trace_cross_derivative: np.ndarray
    bandwidth: float


@dataclass(frozen=True)
class PairedResult:
    pvalue_influence: float
    pvalue_brueck_corrected: float
    seconds_influence: float
    seconds_brueck_corrected: float
    shared_preprocess_seconds: float
    observed_v: float
    observed_u: float
    influence_hessian_condition: float
    brueck_refits: int
    brueck_mean_iterations: float
    brueck_max_iterations: int
    brueck_retry_rate: float


class RBMDesign:
    """Fixed directions, true parameter, and orthogonal loading alternative."""

    def __init__(self, config: Config):
        self.config = config
        rng = np.random.default_rng(config.model_seed)
        d, m = config.visible_dimension, config.hidden_dimension

        directions = rng.choice((-1.0, 1.0), size=(d, m))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        self.directions = directions
        # Zero visible and hidden biases with {-1,+1} hidden units make the
        # visible law centrally symmetric. The alternative therefore changes
        # covariance/higher-order orientation, not location.
        self.visible_bias = np.zeros(d)

        raw_perturbation = rng.normal(size=(d, m))
        projection = directions @ np.linalg.pinv(directions)
        complement = raw_perturbation - projection @ raw_perturbation
        complement_q, _ = np.linalg.qr(complement, mode="reduced")
        direction_gram = directions.T @ directions
        gram_factor = np.linalg.cholesky(direction_gram)
        # perturbation' perturbation = direction_gram and
        # directions' perturbation = 0.  Consequently every convex angular
        # rotation below has exactly the same loading Gram matrix.
        perturbation = complement_q @ gram_factor.T
        if np.linalg.norm(directions.T @ perturbation, ord=np.inf) > 1e-10:
            raise RuntimeError("failed to construct orthogonal loading directions")
        if not np.allclose(
            perturbation.T @ perturbation,
            direction_gram,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError("orthogonal loading directions do not preserve Gram")
        self.perturbation = perturbation

        raw_omitted = rng.normal(size=d)
        omitted = raw_omitted - projection @ raw_omitted
        omitted_norm = np.linalg.norm(omitted)
        if omitted_norm <= 1e-10:
            raise RuntimeError("failed to construct an omitted hidden direction")
        self.omitted_direction = omitted / omitted_norm
        if np.linalg.norm(directions.T @ self.omitted_direction, ord=np.inf) > 1e-10:
            raise RuntimeError("omitted hidden direction is not orthogonal")

        self.hidden_bias = np.zeros(m)
        loading_scale = rng.uniform(0.75, 1.10, size=m)
        loading_scale *= config.true_loading_scale_multiplier
        self.true_theta = np.log(loading_scale)

    @property
    def parameter_dimension(self) -> int:
        return self.config.hidden_dimension

    def unpack(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.parameter_dimension,):
            raise ValueError(
                f"theta must have shape ({self.parameter_dimension},)"
            )
        return np.exp(theta)

    def initial_theta(self) -> np.ndarray:
        m = self.config.hidden_dimension
        return np.log(np.full(m, 0.9))

    def bounds(self) -> list[tuple[float, float]]:
        config = self.config
        m = config.hidden_dimension
        return [
            (
                math.log(config.loading_scale_lower),
                math.log(config.loading_scale_upper),
            )
        ] * m

    def rotated_directions(self, signal: float = 0.0) -> np.ndarray:
        if not 0.0 <= signal < 1.0:
            raise ValueError("rotation signal must lie in [0,1)")
        return (
            math.sqrt(1.0 - signal * signal) * self.directions
            + signal * self.perturbation
        )

    def loading(self, theta: np.ndarray, signal: float = 0.0) -> np.ndarray:
        scale = self.unpack(theta)
        return self.rotated_directions(signal) * scale[None, :]

    def hidden_state_distribution(
        self,
        theta: np.ndarray,
        signal: float = 0.0,
        alternative: str = ALTERNATIVE_ROTATION,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all hidden states, their probabilities, and conditional means."""
        if signal < 0.0:
            raise ValueError("alternative signal must be nonnegative")
        m = self.config.hidden_dimension
        if alternative == ALTERNATIVE_ROTATION:
            loading = self.loading(theta, signal)
            hidden_bias = self.hidden_bias
            hidden_dimension = m
        elif alternative == ALTERNATIVE_OMITTED:
            loading = np.column_stack(
                (
                    self.loading(theta, 0.0),
                    signal * self.omitted_direction,
                )
            )
            hidden_bias = np.zeros(m + 1)
            hidden_dimension = m + 1
        elif alternative == ALTERNATIVE_TAIL:
            loading = self.loading(theta, 0.0)
            hidden_bias = self.hidden_bias
            hidden_dimension = m
        else:
            raise ValueError(f"unknown alternative: {alternative}")

        labels = np.arange(1 << hidden_dimension, dtype=np.uint64)[:, None]
        bits = np.arange(hidden_dimension, dtype=np.uint64)[None, :]
        states = 2.0 * ((labels >> bits) & 1).astype(float) - 1.0
        means = self.visible_bias[None, :] + states @ loading.T
        log_weight = (
            states @ hidden_bias
            + 0.5 * np.sum(means * means, axis=1)
            - 0.5 * float(self.visible_bias @ self.visible_bias)
        )
        probabilities = np.exp(log_weight - logsumexp(log_weight))
        return states, probabilities, means

    def sample_exact(
        self,
        rng: np.random.Generator,
        n: int,
        signal: float = 0.0,
        alternative: str = ALTERNATIVE_ROTATION,
    ) -> np.ndarray:
        """Draw iid visible samples by enumerating 2^M or 2^(M+1) states."""
        _, probabilities, means = self.hidden_state_distribution(
            self.true_theta, signal, alternative
        )
        indices = rng.choice(means.shape[0], size=n, p=probabilities)
        noise = rng.normal(size=(n, self.config.visible_dimension))
        if alternative == ALTERNATIVE_TAIL and signal > 0.0:
            # nu = 2 + 1/s makes s=0 the Gaussian limit. Multiplication by
            # sqrt((nu-2)/ChiSq_nu) gives an elliptical t noise with identity
            # covariance, so the path changes tails rather than scale.
            degrees_of_freedom = heavy_tail_degrees_of_freedom(signal)
            radial_scale = np.sqrt(
                (degrees_of_freedom - 2.0)
                / rng.chisquare(degrees_of_freedom, size=(n, 1))
            )
            noise *= radial_scale
        return means[indices] + noise

    def _quantities(self, x: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
        scale = self.unpack(theta)
        projection = x @ self.directions
        logits = self.hidden_bias[None, :] + projection * scale[None, :]
        activation = np.tanh(logits)
        activation_first = 1.0 - activation * activation
        activation_second = -2.0 * activation * activation_first
        activation_third = (
            4.0 * activation * activation * activation_first
            - 2.0 * activation_first * activation_first
        )
        score = (
            -x
            + self.visible_bias[None, :]
            + (activation * scale[None, :]) @ self.directions.T
        )
        score_projection = score @ self.directions
        derivative_rho = (
            scale[None, :] * activation
            + scale[None, :] ** 2 * projection * activation_first
        )
        return {
            "projection": projection,
            "activation": activation,
            "activation_first": activation_first,
            "activation_second": activation_second,
            "activation_third": activation_third,
            "score": score,
            "score_projection": score_projection,
            "derivative_rho": derivative_rho,
            "scale": scale,
        }

    def score(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self._quantities(np.asarray(x, dtype=float), theta)["score"]

    def score_and_parameter_derivative(
        self, x: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return s_theta(x) and D[i,q,a] = partial s_q(X_i)/partial theta_a."""
        x = np.asarray(x, dtype=float)
        quantities = self._quantities(x, theta)
        derivative = (
            self.directions[None, :, :]
            * quantities["derivative_rho"][:, None, :]
        )
        return quantities["score"], derivative

    def score_matching_terms(
        self, x: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-observation Hyvarinen loss and its parameter gradient."""
        x = np.asarray(x, dtype=float)
        q = self._quantities(x, theta)
        scale = q["scale"][None, :]
        norm2 = np.sum(self.directions * self.directions, axis=0)[None, :]
        divergence_hidden = norm2 * scale**2 * q["activation_first"]
        loss = (
            0.5 * np.sum(q["score"] * q["score"], axis=1)
            - self.config.visible_dimension
            + np.sum(divergence_hidden, axis=1)
        )
        gradient_rho = (
            q["derivative_rho"] * q["score_projection"]
            + norm2
            * (
                2.0 * scale**2 * q["activation_first"]
                + scale**3 * q["projection"] * q["activation_second"]
            )
        )
        return loss, gradient_rho

    def score_matching_hessian(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Exact Hessian of the weighted mean score-matching objective."""
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if weights is None:
            weights = np.ones(n)
        normalized = np.asarray(weights, dtype=float).copy()
        normalized /= normalized.sum()

        q = self._quantities(x, theta)
        scale = q["scale"][None, :]
        gram_directions = self.directions.T @ self.directions
        norm2 = np.diag(gram_directions)[None, :]
        dr = q["derivative_rho"]

        h_rr = ((dr * normalized[:, None]).T @ dr) * gram_directions

        second_score_rr = (
            scale * q["activation"]
            + 3.0 * scale**2 * q["projection"] * q["activation_first"]
            + scale**3
            * q["projection"] ** 2
            * q["activation_second"]
        )
        second_div_rr = norm2 * (
            4.0 * scale**2 * q["activation_first"]
            + 5.0
            * scale**3
            * q["projection"]
            * q["activation_second"]
            + scale**4 * q["projection"] ** 2 * q["activation_third"]
        )

        diag_rr = normalized @ (
            q["score_projection"] * second_score_rr + second_div_rr
        )
        diagonal = np.arange(self.config.hidden_dimension)
        h_rr[diagonal, diagonal] += diag_rr
        return 0.5 * (h_rr + h_rr.T)


def fit_score_matching(
    x: np.ndarray,
    design: RBMDesign,
    config: Config,
    weights: np.ndarray | None = None,
    start: np.ndarray | None = None,
    strict: bool = False,
) -> FitResult:
    """Fit positive loading scales by weighted nonlinear score matching."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if weights is None:
        weights = np.ones(n)
    normalized = np.asarray(weights, dtype=float)
    if normalized.shape != (n,) or np.any(normalized < 0.0):
        raise ValueError("weights must be a nonnegative vector of length n")
    if normalized.sum() <= 0.0:
        raise ValueError("weights must have positive sum")
    normalized = normalized / normalized.sum()
    if start is None:
        start = design.initial_theta()

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        losses, gradients = design.score_matching_terms(x, theta)
        return float(normalized @ losses), normalized @ gradients

    result = minimize(
        objective,
        np.asarray(start, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=design.bounds(),
        options={
            "ftol": 0.0 if strict else 1e-12,
            "gtol": config.optimizer_gtol,
            "maxiter": config.optimizer_maxiter * (2 if strict else 1),
            "maxls": 50,
        },
    )
    theta = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(theta)) or not np.isfinite(result.fun):
        raise RuntimeError(f"non-finite score-matching fit: {result.message}")
    lower = np.array([bound[0] for bound in design.bounds()])
    upper = np.array([bound[1] for bound in design.bounds()])
    tolerance = 1e-6 * np.maximum(1.0, upper - lower)
    boundary_hit = bool(
        np.any(theta - lower <= tolerance) or np.any(upper - theta <= tolerance)
    )
    gradient_norm = float(np.linalg.norm(np.asarray(result.jac), ord=np.inf))
    # The influence expansion needs an actual root of the estimating equation,
    # not merely an optimizer exit caused by small relative function change.
    stationarity_tolerance = max(1e-5, 20.0 * config.optimizer_gtol)
    success = bool(gradient_norm <= stationarity_tolerance)
    return FitResult(
        theta=theta,
        objective=float(result.fun),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        success=success,
        gradient_norm=gradient_norm,
        boundary_hit=boundary_hit,
        message=str(result.message),
    )


def rbf_geometry(x: np.ndarray, bandwidth: float) -> Geometry:
    x = np.asarray(x, dtype=float)
    difference = x[:, None, :] - x[None, :, :]
    squared_distance = np.sum(difference * difference, axis=2)
    inverse = 1.0 / (bandwidth * bandwidth)
    kernel = np.exp(-0.5 * squared_distance * inverse)
    trace = (
        x.shape[1] * inverse - squared_distance * inverse * inverse
    ) * kernel
    return Geometry(kernel, difference, trace, bandwidth)


def stein_gram(score: np.ndarray, geometry: Geometry) -> np.ndarray:
    inverse = 1.0 / (geometry.bandwidth * geometry.bandwidth)
    directional = np.einsum(
        "ijd,id->ij", geometry.difference, score, optimize=True
    ) - np.einsum(
        "ijd,jd->ij", geometry.difference, score, optimize=True
    )
    return (
        (score @ score.T) * geometry.kernel
        + directional * inverse * geometry.kernel
        + geometry.trace_cross_derivative
    )


def weighted_u(matrix: np.ndarray, counts: np.ndarray) -> float:
    n = int(np.sum(counts))
    total = counts @ matrix @ counts - counts @ np.diag(matrix)
    return float(total / (n * (n - 1)))


def center_gram(matrix: np.ndarray) -> np.ndarray:
    row_mean = matrix.mean(axis=1, keepdims=True)
    return matrix - row_mean - row_mean.T + matrix.mean()


def gradient_u(
    score: np.ndarray,
    derivative: np.ndarray,
    geometry: Geometry,
    counts: np.ndarray,
) -> np.ndarray:
    """Parameter gradient of a count-weighted Stein U-statistic.

    This contraction avoids materializing an n by n by p gradient tensor.
    """
    n = int(np.sum(counts))
    pair_counts = np.outer(counts, counts)
    diagonal = np.arange(counts.size)
    pair_counts[diagonal, diagonal] -= counts
    weighted_kernel = pair_counts * geometry.kernel
    product = weighted_kernel @ score
    inverse = 1.0 / (geometry.bandwidth * geometry.bandwidth)
    difference_product = inverse * np.einsum(
        "ij,ijd->id",
        weighted_kernel,
        geometry.difference,
        optimize=True,
    )
    common = product + difference_product
    value = 2.0 * np.einsum(
        "idp,id->p", derivative, common, optimize=True
    )
    return value / (n * (n - 1))


def score_matching_influence(
    x: np.ndarray, design: RBMDesign, theta: np.ndarray
) -> tuple[np.ndarray, float]:
    _, estimating_terms = design.score_matching_terms(x, theta)
    estimating_terms -= estimating_terms.mean(axis=0, keepdims=True)
    hessian = design.score_matching_hessian(x, theta)
    condition = float(np.linalg.cond(hessian))
    if not np.isfinite(condition) or condition > 1e12:
        raise RuntimeError(
            f"score-matching Hessian is numerically singular (condition={condition})"
        )
    influence = -np.linalg.solve(hessian, estimating_terms.T).T
    influence -= influence.mean(axis=0, keepdims=True)
    return influence, condition


def adjusted_gram(
    x: np.ndarray,
    score: np.ndarray,
    derivative: np.ndarray,
    stein: np.ndarray,
    geometry: Geometry,
    influence: np.ndarray,
) -> np.ndarray:
    """Gram matrix of the influence-adjusted observation-level Stein feature."""
    n = x.shape[0]
    inverse = 1.0 / (geometry.bandwidth * geometry.bandwidth)
    mixed = (
        score[:, None, :] - geometry.difference * inverse
    ) * geometry.kernel[:, :, None]
    cross = np.einsum(
        "ijd,jdp->ip", mixed, derivative, optimize=True
    ) / n
    derivative_gram = np.einsum(
        "idp,ij,jdq->pq",
        derivative,
        geometry.kernel,
        derivative,
        optimize=True,
    ) / (n * n)
    answer = (
        stein
        + cross @ influence.T
        + influence @ cross.T
        + influence @ derivative_gram @ influence.T
    )
    answer = 0.5 * (answer + answer.T)
    return center_gram(answer)


def paired_empirical_tests(
    x: np.ndarray,
    design: RBMDesign,
    config: Config,
    fitted: FitResult,
    counts_all: np.ndarray,
    weights_all: np.ndarray,
) -> PairedResult:
    """Run the no-refit influence test and corrected refitted Brueck test.

    ``weights_all`` holds the centered Gaussian multipliers for the influence
    test; ``counts_all`` holds the Efron counts used by the Brueck refits.
    """
    if not fitted.success:
        raise RuntimeError(f"observed fit failed: {fitted.message}")
    if fitted.boundary_hit:
        raise RuntimeError(
            "observed fit reached a parameter bound; interior influence theory "
            "does not apply"
        )
    n = x.shape[0]
    if counts_all.ndim != 2 or counts_all.shape[1] != n:
        raise ValueError("counts_all must have shape (B,n)")
    if not np.all(counts_all.sum(axis=1) == n):
        raise ValueError("each Efron count vector must sum to n")
    if weights_all.shape != counts_all.shape:
        raise ValueError("weights_all must match the shape of counts_all")

    shared_started = time.perf_counter()
    geometry = rbf_geometry(x, config.fixed_bandwidth)
    score_hat, derivative_hat = design.score_and_parameter_derivative(
        x, fitted.theta
    )
    stein_hat = stein_gram(score_hat, geometry)
    shared_preprocess_seconds = time.perf_counter() - shared_started

    influence_started = time.perf_counter()
    influence, condition = score_matching_influence(x, design, fitted.theta)
    adjusted = adjusted_gram(
        x,
        score_hat,
        derivative_hat,
        stein_hat,
        geometry,
        influence,
    )
    influence_statistics = np.einsum(
        "bi,ij,bj->b",
        weights_all,
        adjusted,
        weights_all,
        optimize=True,
    ) / (n * n)
    observed_v = float(stein_hat.mean())
    pvalue_influence = (
        1 + np.sum(influence_statistics >= observed_v)
    ) / (counts_all.shape[0] + 1)
    seconds_influence = time.perf_counter() - influence_started

    brueck_started = time.perf_counter()
    ones = np.ones(n)
    observed_u = weighted_u(stein_hat, ones)
    gradient_original = gradient_u(
        score_hat, derivative_hat, geometry, ones
    )
    corrected_statistics = np.empty(counts_all.shape[0])
    iterations: list[int] = []
    retries = 0

    for bootstrap_index, counts in enumerate(counts_all):
        bootstrap_fit = fit_score_matching(
            x,
            design,
            config,
            weights=counts,
            start=fitted.theta,
        )
        optimizer_iterations = bootstrap_fit.iterations
        if bootstrap_fit.boundary_hit or not bootstrap_fit.success:
            retries += 1
            retry_fit = fit_score_matching(
                x,
                design,
                config,
                weights=counts,
                start=design.initial_theta(),
                strict=True,
            )
            optimizer_iterations += retry_fit.iterations
            bootstrap_fit = retry_fit
        iterations.append(optimizer_iterations)
        if bootstrap_fit.boundary_hit:
            raise RuntimeError(
                "bootstrap score-matching refit reached a parameter bound at "
                f"draw {bootstrap_index}; the smooth Brueck/influence expansion "
                "does not apply. Widen the bounds or redesign the setting."
            )
        if not bootstrap_fit.success:
            raise RuntimeError(
                "bootstrap score-matching refit is not sufficiently stationary "
                f"at draw {bootstrap_index}: gradient_norm="
                f"{bootstrap_fit.gradient_norm:.3g}; {bootstrap_fit.message}"
            )
        score_b, derivative_b = design.score_and_parameter_derivative(
            x, bootstrap_fit.theta
        )
        stein_b = stein_gram(score_b, geometry)
        base = weighted_u(center_gram(stein_b), counts)
        gradient_bootstrap_refit = gradient_u(
            score_b, derivative_b, geometry, counts
        )
        delta = bootstrap_fit.theta - fitted.theta
        gradient_bootstrap_original_fit = gradient_u(
            score_hat, derivative_hat, geometry, counts
        )
        corrected_correction = delta @ (
            0.5
            * (gradient_bootstrap_refit + gradient_bootstrap_original_fit)
            - gradient_original
        )
        corrected_statistics[bootstrap_index] = base + corrected_correction

    denominator = counts_all.shape[0] + 1
    pvalue_brueck_corrected = (
        1 + np.sum(corrected_statistics >= observed_u)
    ) / denominator
    seconds_brueck_corrected = time.perf_counter() - brueck_started
    return PairedResult(
        pvalue_influence=float(pvalue_influence),
        pvalue_brueck_corrected=float(pvalue_brueck_corrected),
        seconds_influence=float(seconds_influence),
        seconds_brueck_corrected=float(seconds_brueck_corrected),
        shared_preprocess_seconds=float(shared_preprocess_seconds),
        observed_v=observed_v,
        observed_u=observed_u,
        influence_hessian_condition=condition,
        brueck_refits=int(counts_all.shape[0]),
        brueck_mean_iterations=float(np.mean(iterations)),
        brueck_max_iterations=int(np.max(iterations)),
        brueck_retry_rate=float(retries / counts_all.shape[0]),
    )


def _seed(
    config: Config,
    alternative: str,
    signal: float,
    repetition: int,
    stream: int,
) -> int:
    alternative_code = (
        0 if alternative == SHARED_NULL else ALTERNATIVE_CODES[alternative]
    )
    signal_code = int(round(signal * 10000))
    return int(
        np.random.SeedSequence(
            [config.seed, alternative_code, signal_code, repetition, stream]
        ).generate_state(1)[0]
    )


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials**2)
        )
        / denominator
    )
    return center - half, center + half


def summarize(
    raw_rows: list[dict[str, object]], config: Config
) -> list[dict[str, object]]:
    methods = (
        (
            METHOD_INFLUENCE,
            "pvalue_influence",
            "seconds_influence",
            "V",
            0,
        ),
        (
            METHOD_BRUECK_CORRECTED,
            "pvalue_brueck_corrected",
            "seconds_brueck_corrected",
            "U",
            config.bootstrap_draws,
        ),
    )
    rows: list[dict[str, object]] = []
    for alternative, grid in alternative_grids(config).items():
        for signal in grid:
            source_alternative = SHARED_NULL if signal == 0.0 else alternative
            cell = [
                row
                for row in raw_rows
                if str(row["alternative"]) == source_alternative
                and float(row["signal"]) == signal
            ]
            if len(cell) != config.monte_carlo_reps:
                raise ValueError(
                    f"expected {config.monte_carlo_reps} rows for "
                    f"{source_alternative} at {signal:g}, found {len(cell)}"
                )
            influence_time = float(
                np.mean([float(row["seconds_influence"]) for row in cell])
            )
            for method, pvalue_field, time_field, statistic, refits in methods:
                pvalues = np.array([float(row[pvalue_field]) for row in cell])
                rejections = int(np.sum(pvalues <= config.alpha))
                interval = _wilson(rejections, len(cell))
                calibration_seconds = float(
                    np.mean([float(row[time_field]) for row in cell])
                )
                shared_seconds = float(
                    np.mean(
                        [
                            float(row["observed_fit_seconds"])
                            + float(row["shared_preprocess_seconds"])
                            for row in cell
                        ]
                    )
                )
                total_seconds = shared_seconds + calibration_seconds
                influence_total = shared_seconds + influence_time
                rows.append(
                    {
                        "alternative": alternative,
                        "signal": signal,
                        "shared_null": int(signal == 0.0),
                        "method": method,
                        "statistic": statistic,
                        "rejection_rate": rejections / len(cell),
                        "rejection_ci_lower": interval[0],
                        "rejection_ci_upper": interval[1],
                        "mean_pvalue": float(pvalues.mean()),
                        "calibration_seconds_per_dataset": calibration_seconds,
                        "calibration_speed_ratio_vs_influence": (
                            calibration_seconds / influence_time
                        ),
                        "total_seconds_per_dataset": total_seconds,
                        "total_speed_ratio_vs_influence": (
                            total_seconds / influence_total
                        ),
                        "bootstrap_refits_per_dataset": refits,
                        "mean_brueck_retry_rate": (
                            float(
                                np.mean(
                                    [
                                        float(row["brueck_retry_rate"])
                                        for row in cell
                                    ]
                                )
                            )
                            if refits
                            else 0.0
                        ),
                        "observed_fit_retry_rate": float(
                            np.mean(
                                [
                                    float(row["observed_fit_retried"])
                                    for row in cell
                                ]
                            )
                        ),
                        "n": config.sample_size,
                        "B": config.bootstrap_draws,
                        "reps": config.monte_carlo_reps,
                    }
                )
    return rows


def _validate_resume_rows(
    rows: list[dict[str, object]],
    config: Config,
    source_sha256: str | None = None,
) -> set[tuple[str, float, int]]:
    allowed_cells = set(computational_cells(config))
    required = {
        "alternative",
        "signal",
        "rep",
        "pvalue_influence",
        "pvalue_brueck_corrected",
        "schema_version",
        "design_version",
        "source_sha256",
    }
    keys: set[tuple[str, float, int]] = set()
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"resume checkpoint is missing fields: {sorted(missing)}")
        if int(row["schema_version"]) != OUTPUT_SCHEMA_VERSION:
            raise ValueError("resume checkpoint has a different schema version")
        if str(row["design_version"]) != DESIGN_VERSION:
            raise ValueError("resume checkpoint has a different design version")
        if source_sha256 is not None and str(row["source_sha256"]) != source_sha256:
            raise ValueError("resume checkpoint was created by different source code")
        alternative = str(row["alternative"])
        signal = float(row["signal"])
        repetition = int(row["rep"])
        if (alternative, signal) not in allowed_cells:
            raise ValueError(
                "resume checkpoint contains an unknown cell: "
                f"{alternative}, {signal:g}"
            )
        if not 0 <= repetition < config.monte_carlo_reps:
            raise ValueError("resume checkpoint contains an invalid replication")
        key = (alternative, signal, repetition)
        if key in keys:
            raise ValueError(f"duplicate resume checkpoint row: {key}")
        keys.add(key)
    return keys
