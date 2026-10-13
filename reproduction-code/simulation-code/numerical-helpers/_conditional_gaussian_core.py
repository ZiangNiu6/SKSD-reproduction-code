"""Self-contained numerical core for the conditional-Gaussian simulation.

The definitions below are the minimal scientific implementation used by
03_conditional_gaussian_score_matching.py. Their provenance is
``code/qim-python/experiments/structural_sm_influence_vs_brueck.py`` and the
QIM primitives in ``code/qim-python/lib/qim.py``; neither source is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import time

import numpy as np

METHOD_INFLUENCE = "SKSD-influence"
METHOD_BRUECK_CORRECTED = "Brueck-corrected"
DESIGN_VERSION = "qim-structural-v2-community-closure-matching"
OUTPUT_SCHEMA_VERSION = 2
BRUECK_CORRECTION_VERSION = "factor-corrected-midpoint-v1"
ALTERNATIVES = ("rewire", "community", "matching")
ALTERNATIVE_CODES = {"rewire": 1, "community": 2, "matching": 3}
RAW_FIELDS = (
    "alternative",
    "delta",
    "rep",
    "pvalue_influence",
    # Keep the legacy generic field as the formula-as-printed result.
    "pvalue_brueck",
    "pvalue_brueck_corrected",
    "seconds_influence",
    # Likewise, seconds_brueck is the legacy as-printed runtime.
    "seconds_brueck",
    "seconds_brueck_corrected",
    "score_matching_condition",
    "min_fitted_ring",
    "max_fitted_ring",
    "score_matching_boundary_hit",
    "brueck_boundary_refit_rate",
)


class QIMModel:
    """Simple parameter container with zero diagonal on Theta."""

    __slots__ = ("Theta", "theta1", "theta2", "d")

    def __init__(self, Theta, theta1, theta2):
        Theta = np.asarray(Theta, dtype=float).copy()
        np.fill_diagonal(Theta, 0.0)
        self.Theta = Theta
        self.theta1 = np.asarray(theta1, dtype=float).copy()
        self.theta2 = np.asarray(theta2, dtype=float).copy()
        self.d = self.theta1.size


def qim_model(Theta, theta1, theta2) -> QIMModel:
    return QIMModel(Theta, theta1, theta2)


def gibbs_iid_qim(model: QIMModel, M: int, burn: int = 300, seed=None) -> np.ndarray:
    """
    Vectorised parallel-chain Gibbs (M independent chains, one snapshot
    each). Matches the gibbs_iid used by the manuscript-style scripts:
    much faster than the sequential gibbs_qim when many independent draws
    are needed (parametric bootstrap, MMD reference samples).
    """
    rng = np.random.default_rng(seed)
    d = model.d
    Th = model.Theta
    th1 = model.theta1
    th2 = model.theta2
    X = rng.standard_normal((M, d))
    X2 = X ** 2
    for _ in range(burn):
        for i in range(d):
            precision = -2.0 * (2.0 * (X2 @ Th[:, i]) + th2[i])
            v = 1.0 / precision
            X[:, i] = rng.normal(th1[i] * v, np.sqrt(np.abs(v)))
            X2[:, i] = X[:, i] ** 2
    return X


def build_grad_t(X: np.ndarray) -> np.ndarray:
    """
    G[k, a, :] = grad_x t_a(x_k), where
        t_a(x) = x_a^2 * x_{a+1}^2 (ring sufficient statistics).

    Layout matches gaussian-limit/R/qim.R's `.build_grad_t`: only the
    columns a and ip1[a]=((a+1) mod d) of G[:, a, :] are non-zero.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    G = np.zeros((n, d, d), dtype=float)
    ip = np.r_[1:d, 0]
    for a in range(d):
        G[:, a, a] = 2.0 * X[:, a] * X[:, ip[a]] ** 2
        G[:, a, ip[a]] = 2.0 * X[:, ip[a]] * X[:, a] ** 2
    return G


@dataclass(frozen=True)
class Config:
    dimension: int
    sample_size: int
    bootstrap_draws: int
    monte_carlo_reps: int
    gibbs_burn: int
    fixed_bandwidth: float
    alpha: float
    seed: int
    model_seed: int
    score_matching_ridge: float
    score_matching_lower: float
    score_matching_upper: float
    rewire_grid: tuple[float, ...]
    community_grid: tuple[float, ...]
    matching_grid: tuple[float, ...]
    output_dir: str



def validate_config(config: Config) -> None:
    if config.dimension != 8:
        raise ValueError("D must equal 8 for the three structural designs")
    if config.sample_size <= config.dimension + 2:
        raise ValueError("N must exceed the d+2 score-matching dimension")
    if config.bootstrap_draws < 1 or config.monte_carlo_reps < 1:
        raise ValueError("B and REPS must be positive")
    if config.gibbs_burn < 1 or config.fixed_bandwidth <= 0.0:
        raise ValueError("BURN and H_FIXED must be positive")
    if not 0.0 < config.alpha < 1.0:
        raise ValueError("ALPHA must lie strictly between zero and one")
    if config.score_matching_lower >= config.score_matching_upper:
        raise ValueError("SM_LOWER must be strictly smaller than SM_UPPER")
    if config.score_matching_upper > 0.0:
        raise ValueError(
            "SM_UPPER must be nonpositive so fitted QIMs are normalisable"
        )
    if config.score_matching_ridge != 0.0:
        raise ValueError(
            "SM_RIDGE must be zero: a ridge-dependent influence function "
            "has not been derived"
        )


class StructuralDesign:
    """Null and alternative QIM parameter matrices for a fixed dimension."""

    def __init__(self, config: Config):
        self.config = config
        self.d = config.dimension
        self.ip1 = np.r_[1 : self.d, 0]
        self.ip2 = np.r_[2 : self.d, 0, 1]
        rng = np.random.default_rng(config.model_seed)
        self.ring_weights = rng.uniform(
            2.0, 2.0 + self.d / 2.0, size=self.d
        )
        self.theta1 = np.full(self.d, 2.0)
        self.theta2 = np.full(self.d, -0.5)
        self.null_theta = self._ring_theta(self.ring_weights)
        # Balanced triadic closure within communities {0, 1, 2, 3} and
        # {4, 5, 6, 7}.  Donor and recipient incidence counts agree at
        # every node, so weighted node strengths remain fixed.
        self.community_donor_pairs = (
            (0, 1),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 0),
        )
        self.community_recipient_pairs = (
            (0, 2),
            (1, 3),
            (0, 3),
            (4, 6),
            (5, 7),
            (4, 7),
        )
        # Four diametric matched pairs receive twice the amount removed
        # from each ring edge.  Each node loses 2*delta on the ring and
        # gains 2*delta through its unique matched edge.
        self.matching_donor_pairs = tuple(
            (a, int(self.ip1[a])) for a in range(self.d)
        )
        self.matching_recipient_pairs = tuple(
            (a, a + self.d // 2) for a in range(self.d // 2)
        )

    def _ring_theta(self, weights: np.ndarray) -> np.ndarray:
        theta = np.zeros((self.d, self.d))
        for a in range(self.d):
            theta[a, self.ip1[a]] = -weights[a]
            theta[self.ip1[a], a] = -weights[a]
        return theta

    @staticmethod
    def _reallocate_edges(
        theta: np.ndarray,
        donor_pairs: tuple[tuple[int, int], ...],
        recipient_pairs: tuple[tuple[int, int], ...],
        donor_amount: float,
        recipient_amount: float,
    ) -> np.ndarray:
        answer = theta.copy()
        removed = len(donor_pairs) * donor_amount
        added = len(recipient_pairs) * recipient_amount
        if not np.isclose(removed, added):
            raise ValueError(
                "edge reallocation must preserve the interaction budget"
            )
        for a, b in donor_pairs:
            weakened = answer[a, b] + donor_amount
            if weakened > 0.0:
                raise ValueError(
                    f"signal is too large for donor edge {(a, b)}"
                )
            answer[a, b] = weakened
            answer[b, a] = weakened
        for a, b in recipient_pairs:
            answer[a, b] -= recipient_amount
            answer[b, a] -= recipient_amount
        return answer

    def theta(self, alternative: str, delta: float) -> np.ndarray:
        if delta == 0.0:
            return self.null_theta.copy()
        if alternative == "rewire":
            if delta >= float(np.min(self.ring_weights)):
                raise ValueError(
                    "rewire delta must be smaller than the weakest ring edge "
                    f"({np.min(self.ring_weights):.3f})"
                )
            theta = np.zeros((self.d, self.d))
            for a in range(self.d):
                weakened = self.ring_weights[a] - delta
                theta[a, self.ip1[a]] = -weakened
                theta[self.ip1[a], a] = -weakened
                theta[a, self.ip2[a]] -= delta
                theta[self.ip2[a], a] -= delta
            return theta
        if alternative == "community":
            return self._reallocate_edges(
                self.null_theta,
                self.community_donor_pairs,
                self.community_recipient_pairs,
                donor_amount=delta,
                recipient_amount=delta,
            )
        if alternative == "matching":
            return self._reallocate_edges(
                self.null_theta,
                self.matching_donor_pairs,
                self.matching_recipient_pairs,
                donor_amount=delta,
                recipient_amount=2.0 * delta,
            )
        raise ValueError(f"unknown alternative {alternative!r}")

    def model(self, alternative: str, delta: float):
        return qim_model(
            self.theta(alternative, delta), self.theta1, self.theta2
        )

    def model_from_score_parameter(self, beta: np.ndarray):
        theta = np.zeros((self.d, self.d))
        for a in range(self.d):
            theta[a, self.ip1[a]] = beta[a] / 2.0
            theta[self.ip1[a], a] = beta[a] / 2.0
        return qim_model(theta, self.theta1, self.theta2)


@dataclass
class Geometry:
    basis_gradient: np.ndarray
    laplacian: np.ndarray
    base_score: np.ndarray
    kernel: np.ndarray
    difference: np.ndarray
    trace: np.ndarray


def score_matching_geometry(
    X: np.ndarray, bandwidth: float, design: StructuralDesign
) -> Geometry:
    """Ring score-matching basis and fixed RBF geometry."""
    n, d = X.shape
    G = build_grad_t(X)
    X2 = X * X
    laplacian = 2.0 * (X2 + X2[:, design.ip1])
    base_score = 2.0 * design.theta2 * X + design.theta1

    squared_norm = np.sum(X2, axis=1)
    squared_distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * X @ X.T
    )
    np.clip(squared_distance, 0.0, None, out=squared_distance)
    difference = X[:, None, :] - X[None, :, :]
    kernel = np.exp(-squared_distance / (2.0 * bandwidth * bandwidth))
    trace = (
        d / (bandwidth * bandwidth)
        - squared_distance / bandwidth**4
    ) * kernel
    return Geometry(G, laplacian, base_score, kernel, difference, trace)


def _regularize(
    matrices: np.ndarray, ridge: float
) -> np.ndarray:
    if ridge == 0.0:
        return matrices
    p = matrices.shape[-1]
    scale = np.mean(
        np.abs(np.diagonal(matrices, axis1=-2, axis2=-1)), axis=-1
    )
    return matrices + ridge * scale[..., None, None] * np.eye(p)


def solve_box_quadratic(
    matrices: np.ndarray,
    offsets: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Exact minimizer of .5*x'Ax+b'x over a common coordinate box."""
    matrices = np.asarray(matrices)
    offsets = np.asarray(offsets)
    was_vector = offsets.ndim == 1
    if was_vector:
        matrices = matrices[None, :, :]
        offsets = offsets[None, :]

    unconstrained = -np.linalg.solve(matrices, offsets)
    answer = unconstrained.copy()
    outside = np.any(
        (unconstrained < lower) | (unconstrained > upper), axis=1
    )
    if not np.any(outside):
        return answer[0] if was_vector else answer

    bad_matrices = matrices[outside]
    bad_offsets = offsets[outside]
    batch_size, parameter_dimension = bad_offsets.shape
    best = np.zeros((batch_size, parameter_dimension))
    best_value = np.full(batch_size, np.inf)
    for status in product((-1, 0, 1), repeat=parameter_dimension):
        status_array = np.asarray(status)
        free = np.flatnonzero(status_array == 0)
        fixed = np.flatnonzero(status_array != 0)
        candidate = np.zeros((batch_size, parameter_dimension))
        if fixed.size:
            candidate[:, fixed] = np.array(
                [
                    lower if status[index] == -1 else upper
                    for index in fixed
                ]
            )
        if free.size:
            rhs = -bad_offsets[:, free]
            if fixed.size:
                rhs -= np.einsum(
                    "bfa,ba->bf",
                    bad_matrices[:, free[:, None], fixed],
                    candidate[:, fixed],
                    optimize=True,
                )
            try:
                candidate[:, free] = np.linalg.solve(
                    bad_matrices[:, free[:, None], free], rhs
                )
            except np.linalg.LinAlgError:
                continue
        feasible = np.all(
            (candidate >= lower - 1e-9)
            & (candidate <= upper + 1e-9),
            axis=1,
        )
        objective = (
            0.5
            * np.einsum(
                "bi,bij,bj->b",
                candidate,
                bad_matrices,
                candidate,
                optimize=True,
            )
            + np.einsum(
                "bi,bi->b", bad_offsets, candidate, optimize=True
            )
        )
        improve = feasible & (objective < best_value)
        best[improve] = candidate[improve]
        best_value[improve] = objective[improve]
    if not np.all(np.isfinite(best_value)):
        raise RuntimeError("score-matching box solver found no minimizer")
    answer[outside] = best
    return answer[0] if was_vector else answer


def score_matching_fit(
    geometry: Geometry, counts: np.ndarray, config: Config
) -> tuple[np.ndarray, float]:
    G = geometry.basis_gradient
    n = float(np.sum(counts))
    jacobian = np.einsum(
        "i,ipq,irq->pr", counts, G, G, optimize=True
    ) / n
    per_observation_offset = (
        geometry.laplacian
        + np.einsum(
            "ipq,iq->ip",
            G,
            geometry.base_score,
            optimize=True,
        )
    )
    offset = counts @ per_observation_offset / n
    jacobian = _regularize(
        jacobian, config.score_matching_ridge
    )
    beta = solve_box_quadratic(
        jacobian,
        offset,
        config.score_matching_lower,
        config.score_matching_upper,
    )
    return beta, float(np.linalg.cond(jacobian))


def score_matching_refits(
    geometry: Geometry, counts: np.ndarray, config: Config
) -> np.ndarray:
    G = geometry.basis_gradient
    n = counts.sum(axis=1)
    jacobian = np.einsum(
        "bi,ipq,irq->bpr", counts, G, G, optimize=True
    ) / n[:, None, None]
    per_observation_offset = (
        geometry.laplacian
        + np.einsum(
            "ipq,iq->ip",
            G,
            geometry.base_score,
            optimize=True,
        )
    )
    offset = counts @ per_observation_offset / n[:, None]
    jacobian = _regularize(
        jacobian, config.score_matching_ridge
    )
    return solve_box_quadratic(
        jacobian,
        offset,
        config.score_matching_lower,
        config.score_matching_upper,
    )


def model_score(geometry: Geometry, beta: np.ndarray) -> np.ndarray:
    return geometry.base_score + np.einsum(
        "ipq,p->iq", geometry.basis_gradient, beta, optimize=True
    )


def stein_gram(
    X: np.ndarray, geometry: Geometry, beta: np.ndarray, bandwidth: float
) -> np.ndarray:
    score = model_score(geometry, beta)
    score_x = np.sum(score * X, axis=1)
    return (
        (score @ score.T) * geometry.kernel
        + (
            score_x[:, None] - score @ X.T
        )
        / bandwidth**2
        * geometry.kernel
        + (
            score_x[None, :] - X @ score.T
        )
        / bandwidth**2
        * geometry.kernel
        + geometry.trace
    )


def stein_gram_batch(
    X: np.ndarray,
    geometry: Geometry,
    betas: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    score = geometry.base_score[None, :, :] + np.einsum(
        "ipq,bp->biq", geometry.basis_gradient, betas, optimize=True
    )
    score_x = np.einsum("biq,iq->bi", score, X, optimize=True)
    return (
        np.einsum("biq,bjq->bij", score, score, optimize=True)
        * geometry.kernel[None, :, :]
        + (
            score_x[:, :, None]
            - np.einsum("biq,jq->bij", score, X, optimize=True)
        )
        / bandwidth**2
        * geometry.kernel[None, :, :]
        + (
            score_x[:, None, :]
            - np.einsum("iq,bjq->bij", X, score, optimize=True)
        )
        / bandwidth**2
        * geometry.kernel[None, :, :]
        + geometry.trace[None, :, :]
    )


def weighted_u(matrix: np.ndarray, counts: np.ndarray) -> float:
    n = int(np.sum(counts))
    total = counts @ matrix @ counts - counts @ np.diag(matrix)
    return float(total / (n * (n - 1)))


def score_matching_influence(
    geometry: Geometry, beta: np.ndarray, config: Config
) -> np.ndarray:
    G = geometry.basis_gradient
    A_i = np.einsum("ipq,irq->ipr", G, G, optimize=True)
    phi = (
        np.einsum("ipr,r->ip", A_i, beta, optimize=True)
        + geometry.laplacian
        + np.einsum(
            "ipq,iq->ip",
            G,
            geometry.base_score,
            optimize=True,
        )
    )
    jacobian = _regularize(
        A_i.mean(axis=0), config.score_matching_ridge
    )
    influence = -np.linalg.solve(jacobian, phi.T).T
    return influence - influence.mean(axis=0, keepdims=True)


def adjusted_gram(
    X: np.ndarray,
    geometry: Geometry,
    beta: np.ndarray,
    stein: np.ndarray,
    bandwidth: float,
    config: Config,
) -> np.ndarray:
    """Centered Gram matrix of the influence-adjusted Stein features."""
    n, d = X.shape
    G = geometry.basis_gradient
    score = model_score(geometry, beta)
    influence = score_matching_influence(geometry, beta, config)
    cross = np.zeros((n, d))
    derivative_gram = np.zeros((d, d))
    for q in range(d):
        mixed = (
            score[:, q, None]
            - geometry.difference[:, :, q] / bandwidth**2
        ) * geometry.kernel
        Gq = G[:, :, q]
        cross += mixed @ Gq / n
        derivative_gram += Gq.T @ geometry.kernel @ Gq / (n * n)
    answer = (
        stein
        + cross @ influence.T
        + influence @ cross.T
        + influence @ derivative_gram @ influence.T
    )
    row_mean = answer.mean(axis=1, keepdims=True)
    return answer - row_mean - row_mean.T + answer.mean()


def gradient_u(
    X: np.ndarray,
    geometry: Geometry,
    beta: np.ndarray,
    counts: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    n = int(np.sum(counts))
    pair_counts = np.outer(counts, counts)
    pair_counts[np.diag_indices_from(pair_counts)] -= counts
    weighted_kernel = pair_counts * geometry.kernel
    score = model_score(geometry, beta)
    row_sum = weighted_kernel.sum(axis=1)
    product = weighted_kernel @ score
    value = np.zeros(beta.size)
    for q in range(X.shape[1]):
        Gq = geometry.basis_gradient[:, :, q]
        value += 2.0 * Gq.T @ product[:, q]
        value += (
            2.0
            / bandwidth**2
            * Gq.T
            @ (
                X[:, q] * row_sum
                - weighted_kernel @ X[:, q]
            )
        )
    return value / (n * (n - 1))


def gradient_u_batch(
    X: np.ndarray,
    geometry: Geometry,
    betas: np.ndarray,
    counts: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    n = counts.sum(axis=1)
    score = geometry.base_score[None, :, :] + np.einsum(
        "ipq,bp->biq", geometry.basis_gradient, betas, optimize=True
    )
    pair_counts = counts[:, :, None] * counts[:, None, :]
    diagonal = np.arange(X.shape[0])
    pair_counts[:, diagonal, diagonal] -= counts
    weighted_kernel = pair_counts * geometry.kernel[None, :, :]
    row_sum = weighted_kernel.sum(axis=2)
    product = np.einsum(
        "bij,bjq->biq", weighted_kernel, score, optimize=True
    )
    value = np.zeros_like(betas)
    for q in range(X.shape[1]):
        Gq = geometry.basis_gradient[:, :, q]
        value += 2.0 * np.einsum(
            "ip,bi->bp", Gq, product[:, :, q], optimize=True
        )
        value += (
            2.0
            / bandwidth**2
            * np.einsum(
                "ip,bi->bp",
                Gq,
                X[None, :, q] * row_sum
                - np.einsum(
                    "bij,j->bi",
                    weighted_kernel,
                    X[:, q],
                    optimize=True,
                ),
                optimize=True,
            )
        )
    return value / (n * (n - 1))[:, None]


def brueck_correction_terms(
    beta: np.ndarray,
    beta_bootstrap: np.ndarray,
    gradient_original: np.ndarray,
    gradient_bootstrap_refit: np.ndarray,
    gradient_bootstrap_original_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the as-printed and factor-corrected linearization terms.

    The formula displayed by Brueck et al. uses the endpoint difference

        (beta* - beta)' {grad U*(beta*) - grad U(beta)}.

    A second-order Taylor expansion along the segment from beta to beta*
    instead gives the midpoint/trapezoidal term

        (beta* - beta)' [0.5 {grad U*(beta*) + grad U*(beta)}
                         - grad U(beta)].

    The latter is our diagnostic correction, not an author-confirmed erratum.
    """
    delta = beta_bootstrap - beta
    published = np.einsum(
        "bp,bp->b",
        delta,
        gradient_bootstrap_refit - gradient_original,
        optimize=True,
    )
    midpoint = np.einsum(
        "bp,bp->b",
        delta,
        0.5
        * (
            gradient_bootstrap_refit
            + gradient_bootstrap_original_fit
        )
        - gradient_original,
        optimize=True,
    )
    return published, midpoint


@dataclass
class TestResult:
    pvalue_influence: float
    # Legacy field: formula as printed by Brueck et al.
    pvalue_brueck: float
    pvalue_brueck_corrected: float
    seconds_influence: float
    # Legacy field: formula-as-printed runtime.
    seconds_brueck: float
    seconds_brueck_corrected: float
    score_matching_condition: float
    min_fitted_ring: float
    max_fitted_ring: float
    score_matching_boundary_hit: float
    brueck_boundary_refit_rate: float


def paired_empirical_tests(
    X: np.ndarray,
    design: StructuralDesign,
    config: Config,
    bootstrap_seed: int,
) -> TestResult:
    """Run both tests with a shared fit; influence uses centered Gaussian
    multipliers and Brueck uses the shared Efron counts."""
    n = X.shape[0]
    common_started = time.perf_counter()
    geometry = score_matching_geometry(
        X, config.fixed_bandwidth, design
    )
    beta, condition = score_matching_fit(
        geometry, np.ones(n), config
    )
    boundary_hit = bool(
        np.any(
            np.isclose(beta, config.score_matching_lower)
            | np.isclose(beta, config.score_matching_upper)
        )
    )
    if boundary_hit:
        raise RuntimeError(
            "the observed score-matching fit reached its box constraint; "
            "the interior influence formula is not valid for this dataset"
        )
    stein = stein_gram(
        X, geometry, beta, config.fixed_bandwidth
    )
    rng = np.random.default_rng(bootstrap_seed)
    counts = rng.multinomial(
        n, np.full(n, 1.0 / n), size=config.bootstrap_draws
    )
    weights = rng.standard_normal((config.bootstrap_draws, n))
    weights -= weights.mean(axis=1, keepdims=True)
    common_seconds = time.perf_counter() - common_started

    influence_started = time.perf_counter()
    observed_v = float(stein.mean())
    adjusted = adjusted_gram(
        X,
        geometry,
        beta,
        stein,
        config.fixed_bandwidth,
        config,
    )
    influence_statistics = np.einsum(
        "bi,ij,bj->b", weights, adjusted, weights, optimize=True
    ) / (n * n)
    pvalue_influence = (
        1 + np.sum(influence_statistics >= observed_v)
    ) / (config.bootstrap_draws + 1)
    influence_seconds = (
        common_seconds + time.perf_counter() - influence_started
    )

    brueck_started = time.perf_counter()
    observed_u = weighted_u(stein, np.ones(n))
    gradient_original = gradient_u(
        X,
        geometry,
        beta,
        np.ones(n),
        config.fixed_bandwidth,
    )
    beta_bootstrap = score_matching_refits(
        geometry, counts, config
    )
    stein_bootstrap = stein_gram_batch(
        X, geometry, beta_bootstrap, config.fixed_bandwidth
    )
    row_mean = stein_bootstrap.mean(axis=2, keepdims=True)
    centered = (
        stein_bootstrap
        - row_mean
        - np.swapaxes(row_mean, 1, 2)
        + stein_bootstrap.mean(axis=(1, 2))[:, None, None]
    )
    denominator = n * (n - 1)
    base_bootstrap = (
        np.einsum(
            "bi,bij,bj->b", counts, centered, counts, optimize=True
        )
        - np.einsum(
            "bi,bii->b", counts, centered, optimize=True
        )
    ) / denominator
    gradient_bootstrap_refit = gradient_u_batch(
        X,
        geometry,
        beta_bootstrap,
        counts,
        config.fixed_bandwidth,
    )
    # Compute and retain the formula as printed before doing the extra
    # midpoint-gradient evaluation.  This keeps its p-value and timing
    # available for provenance without making it the production comparator.
    correction_published = np.einsum(
        "bp,bp->b",
        beta_bootstrap - beta,
        gradient_bootstrap_refit - gradient_original,
        optimize=True,
    )
    brueck_statistics = base_bootstrap + correction_published
    pvalue_brueck = (
        1 + np.sum(brueck_statistics >= observed_u)
    ) / (config.bootstrap_draws + 1)
    brueck_seconds = common_seconds + time.perf_counter() - brueck_started

    gradient_bootstrap_original_fit = gradient_u_batch(
        X,
        geometry,
        np.broadcast_to(beta, beta_bootstrap.shape),
        counts,
        config.fixed_bandwidth,
    )
    correction_published_checked, correction_corrected = (
        brueck_correction_terms(
            beta,
            beta_bootstrap,
            gradient_original,
            gradient_bootstrap_refit,
            gradient_bootstrap_original_fit,
        )
    )
    if not np.array_equal(
        correction_published, correction_published_checked
    ):
        raise RuntimeError("internal mismatch in as-printed Brueck correction")
    brueck_statistics_corrected = base_bootstrap + correction_corrected
    pvalue_brueck_corrected = (
        1 + np.sum(brueck_statistics_corrected >= observed_u)
    ) / (config.bootstrap_draws + 1)
    brueck_seconds_corrected = (
        common_seconds + time.perf_counter() - brueck_started
    )

    return TestResult(
        pvalue_influence=float(pvalue_influence),
        pvalue_brueck=float(pvalue_brueck),
        pvalue_brueck_corrected=float(pvalue_brueck_corrected),
        seconds_influence=float(influence_seconds),
        seconds_brueck=float(brueck_seconds),
        seconds_brueck_corrected=float(brueck_seconds_corrected),
        score_matching_condition=condition,
        min_fitted_ring=float(np.min(beta)),
        max_fitted_ring=float(np.max(beta)),
        score_matching_boundary_hit=float(boundary_hit),
        brueck_boundary_refit_rate=float(
            np.mean(
                np.any(
                    np.isclose(
                        beta_bootstrap, config.score_matching_lower
                    )
                    | np.isclose(
                        beta_bootstrap, config.score_matching_upper
                    ),
                    axis=1,
                )
            )
        ),
    )


def seed_for(
    config: Config,
    alternative: str,
    delta: float,
    repetition: int,
    stream: int,
) -> int:
    delta_code = int(round(delta * 10000))
    alternative_code = (
        0 if delta == 0.0 else ALTERNATIVE_CODES[alternative]
    )
    return int(
        np.random.SeedSequence(
            [
                config.seed,
                alternative_code,
                delta_code,
                repetition,
                stream,
            ]
        ).generate_state(1)[0]
    )


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (
        estimate + z * z / (2.0 * trials)
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - half_width, center + half_width


def summarize_rows(rows: list[dict], config: Config) -> list[dict]:
    summary = []
    for alternative in ALTERNATIVES:
        deltas = sorted(
            {
                float(row["delta"])
                for row in rows
                if row["alternative"] == alternative
            }
        )
        for delta in deltas:
            cell = [
                row
                for row in rows
                if row["alternative"] == alternative
                and float(row["delta"]) == delta
            ]
            for method, pvalue_key, time_key in (
                (
                    METHOD_INFLUENCE,
                    "pvalue_influence",
                    "seconds_influence",
                ),
                (
                    METHOD_BRUECK_CORRECTED,
                    "pvalue_brueck_corrected",
                    "seconds_brueck_corrected",
                ),
            ):
                pvalues = np.array(
                    [float(row[pvalue_key]) for row in cell]
                )
                timings = np.array([float(row[time_key]) for row in cell])
                rejections = int(np.sum(pvalues <= config.alpha))
                rate = rejections / len(cell)
                lower, upper = wilson_interval(rejections, len(cell))
                summary.append(
                    {
                        "alternative": alternative,
                        "delta": delta,
                        "method": method,
                        "alpha": config.alpha,
                        "completed_reps": len(cell),
                        "expected_reps": config.monte_carlo_reps,
                        "rejections": rejections,
                        "rejection_rate": rate,
                        "mcse": math.sqrt(
                            rate * (1.0 - rate) / len(cell)
                        ),
                        "wilson95_lower": lower,
                        "wilson95_upper": upper,
                        "mean_pvalue": float(pvalues.mean()),
                        "mean_seconds": float(timings.mean()),
                        "sd_seconds": (
                            float(timings.std(ddof=1))
                            if timings.size > 1
                            else 0.0
                        ),
                        "mean_sm_condition": float(
                            np.mean(
                                [
                                    row["score_matching_condition"]
                                    for row in cell
                                ]
                            )
                        ),
                        "score_matching_boundary_rate": float(
                            np.mean(
                                [
                                    row["score_matching_boundary_hit"]
                                    for row in cell
                                ]
                            )
                        ),
                        "brueck_boundary_refit_rate": float(
                            np.mean(
                                [
                                    row["brueck_boundary_refit_rate"]
                                    for row in cell
                                ]
                            )
                        ),
                    }
                )
    return summary


def grid_for(config: Config, alternative: str) -> tuple[float, ...]:
    return {
        "rewire": config.rewire_grid,
        "community": config.community_grid,
        "matching": config.matching_grid,
    }[alternative]


def computational_cells(config: Config) -> list[tuple[str, float]]:
    """Cells requiring simulation, with the common null represented once."""
    cells: list[tuple[str, float]] = [(ALTERNATIVES[0], 0.0)]
    cells.extend(
        (alternative, float(delta))
        for alternative in ALTERNATIVES
        for delta in grid_for(config, alternative)
        if delta != 0.0
    )
    return cells


def expand_shared_null_rows(rows: list[dict]) -> list[dict]:
    """Clone the single canonical null across the three display panels."""
    canonical = [
        row
        for row in rows
        if row["alternative"] == ALTERNATIVES[0]
        and float(row["delta"]) == 0.0
    ]
    if not canonical:
        return rows
    answer = list(rows)
    present = {
        (row["alternative"], float(row["delta"]), int(row["rep"]))
        for row in answer
    }
    for alternative in ALTERNATIVES[1:]:
        for row in canonical:
            key = (alternative, 0.0, int(row["rep"]))
            if key not in present:
                answer.append({**row, "alternative": alternative})
                present.add(key)
    return answer


def validate_design_grids(
    config: Config, design: StructuralDesign
) -> None:
    """Check every requested signal before starting an expensive run."""
    for alternative in ALTERNATIVES:
        for delta in grid_for(config, alternative):
            design.theta(alternative, delta)
