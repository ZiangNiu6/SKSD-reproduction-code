"""Block-kernel KSD for two conditional RPPA distributions."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import (
    LinAlgError,
    cho_solve,
    cholesky,
    get_lapack_funcs,
)
from scipy.spatial.distance import pdist

from .model import FitResult


@dataclass(frozen=True)
class BlockRBFGeometry:
    kernel: np.ndarray
    difference: np.ndarray
    trace_cross_derivative: np.ndarray
    bandwidth: float
    groups: np.ndarray
    block_scales: dict[int, float]


@dataclass(frozen=True)
class KSDResult:
    v_statistic: float
    u_statistic: float
    ksd: float
    influence_p_value: float | None
    brueck_p_value: float | None
    influence_calibration: str
    brueck_calibration: str
    bootstrap_replicates: int
    influence_seconds: float
    influence_bootstrap_min: float
    influence_bootstrap_q05: float
    influence_bootstrap_median: float
    influence_bootstrap_q95: float
    influence_bootstrap_max: float
    brueck_seconds: float
    brueck_refits: int
    brueck_max_hessian_condition: float
    brueck_max_estimating_residual: float
    brueck_bootstrap_min: float
    brueck_bootstrap_q05: float
    brueck_bootstrap_median: float
    brueck_bootstrap_q95: float
    brueck_bootstrap_max: float


def median_within_group_bandwidth(x: np.ndarray, groups: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups, dtype=int)
    distances = [pdist(x[groups == group]) for group in np.unique(groups)]
    pooled = np.concatenate([value for value in distances if value.size])
    bandwidth = float(np.median(pooled))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Within-group median bandwidth is not positive")
    return bandwidth


def block_rbf_geometry(
    x: np.ndarray, groups: np.ndarray, bandwidth: float
) -> BlockRBFGeometry:
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups, dtype=int)
    if x.ndim != 2 or groups.shape != (x.shape[0],):
        raise ValueError("x and groups have incompatible shapes")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    unique = np.unique(groups)
    n = x.shape[0]
    block_scales = {
        int(group): float(n * n / (unique.size * np.sum(groups == group) ** 2))
        for group in unique
    }
    difference = x[:, None, :] - x[None, :, :]
    squared_distance = np.sum(difference * difference, axis=2)
    inverse = 1.0 / (bandwidth * bandwidth)
    kernel = np.exp(-0.5 * squared_distance * inverse)
    same_group = groups[:, None] == groups[None, :]
    scale = np.array([block_scales[int(group)] for group in groups])
    kernel *= same_group * scale[:, None]
    trace = (
        x.shape[1] * inverse - squared_distance * inverse * inverse
    ) * kernel
    return BlockRBFGeometry(
        kernel=kernel,
        difference=difference,
        trace_cross_derivative=trace,
        bandwidth=float(bandwidth),
        groups=groups,
        block_scales=block_scales,
    )


def stein_gram(score: np.ndarray, geometry: BlockRBFGeometry) -> np.ndarray:
    inverse = 1.0 / (geometry.bandwidth * geometry.bandwidth)
    directional = np.einsum(
        "ijd,id->ij", geometry.difference, score, optimize=True
    ) - np.einsum(
        "ijd,jd->ij", geometry.difference, score, optimize=True
    )
    gram = (
        (score @ score.T) * geometry.kernel
        + directional * inverse * geometry.kernel
        + geometry.trace_cross_derivative
    )
    return 0.5 * (gram + gram.T)


def center_gram(matrix: np.ndarray) -> np.ndarray:
    row_mean = matrix.mean(axis=1, keepdims=True)
    return matrix - row_mean - row_mean.T + matrix.mean()


def influence_adjusted_gram(
    fit: FitResult,
    stein: np.ndarray,
    geometry: BlockRBFGeometry,
) -> np.ndarray:
    if fit.influence is None:
        raise ValueError("Fit has no regular influence representation")
    n = fit.score.shape[0]
    inverse = 1.0 / (geometry.bandwidth * geometry.bandwidth)
    mixed = (
        fit.score[:, None, :] - geometry.difference * inverse
    ) * geometry.kernel[:, :, None]
    cross = np.einsum(
        "ijd,jdp->ip", mixed, fit.derivative, optimize=True
    ) / n
    derivative_gram = np.einsum(
        "idp,ij,jdq->pq",
        fit.derivative,
        geometry.kernel,
        fit.derivative,
        optimize=True,
    ) / (n * n)
    influence = fit.influence
    adjusted = (
        stein
        + cross @ influence.T
        + influence @ cross.T
        + influence @ derivative_gram @ influence.T
    )
    return center_gram(0.5 * (adjusted + adjusted.T))


def stratified_efron_counts(
    groups: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    groups = np.asarray(groups, dtype=int)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    counts = np.zeros((replicates, groups.size), dtype=np.int32)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        counts[:, indices] = rng.multinomial(
            indices.size,
            np.full(indices.size, 1.0 / indices.size),
            size=replicates,
        )
    return counts


def stratified_gaussian_multipliers(
    groups: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw standard Gaussian multipliers centered within each fixed stratum."""
    groups = np.asarray(groups, dtype=int)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    weights = rng.standard_normal((replicates, groups.size))
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        weights[:, indices] -= weights[:, indices].mean(axis=1, keepdims=True)
    return weights


def _quadratic_bootstrap(
    centered: np.ndarray, weights: np.ndarray, *, batch_size: int = 64
) -> np.ndarray:
    replicates, n = weights.shape
    statistics = np.empty(replicates)
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        block = weights[start:stop]
        statistics[start:stop] = np.einsum(
            "bi,ij,bj->b", block, centered, block, optimize=True
        ) / (n * n)
    return statistics


def score_matching_refits(
    fit: FitResult,
    counts: np.ndarray,
    *,
    batch_size: int = 32,
    condition_limit: float = 1e10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exactly refit the quadratic score-matching estimator for Efron counts.

    For observation ``i``, let ``D_i`` be the spatial-score derivative and
    ``psi_i`` the observed estimating term at ``fit.beta``.  The weighted
    score-matching solution satisfies

        H_w (beta_w - beta_hat) = -mean_w(psi_i),
        H_w = mean_w(D_i' D_i).

    The calculation is batched but is algebraically identical to expanding
    each empirical bootstrap sample and calling the ordinary fitter.
    """

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != fit.score.shape[0]:
        raise ValueError("counts must have shape (B,n)")
    if np.any(counts < 0.0) or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite and nonnegative")
    n = fit.score.shape[0]
    if not np.allclose(counts.sum(axis=1), n):
        raise ValueError("each Efron count vector must sum to n")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # This is the batched matrix product D_i' D_i.  ``einsum`` is
    # algebraically equivalent, but on the supported NumPy builds it dispatches
    # this contraction much less efficiently (roughly seconds rather than
    # hundredths of a second for the RPPA dimensions).
    hessian_terms = np.swapaxes(fit.derivative, 1, 2) @ fit.derivative
    refits = np.empty((counts.shape[0], fit.beta.size))
    conditions = np.empty(counts.shape[0])
    maximum_residual = 0.0

    for start in range(0, counts.shape[0], batch_size):
        stop = min(counts.shape[0], start + batch_size)
        block = counts[start:stop]
        hessian = np.einsum(
            "bi,ipq->bpq", block, hessian_terms, optimize=True
        ) / n
        hessian = 0.5 * (hessian + np.swapaxes(hessian, 1, 2))
        rhs = block @ fit.estimating_terms / n
        diagonal_scale = np.sqrt(
            np.maximum(np.diagonal(hessian, axis1=1, axis2=2), 0.0)
        )
        if np.any(~np.isfinite(diagonal_scale)) or np.any(
            diagonal_scale <= 1e-12
        ):
            raise RuntimeError(
                "a Brueck bootstrap score-matching Hessian has a zero column"
            )
        normalized = (
            hessian
            / diagonal_scale[:, :, None]
            / diagonal_scale[:, None, :]
        )
        # Reuse one Cholesky factor for the solve and LAPACK's reciprocal
        # 1-norm condition estimate.  This avoids a separate SVD/eigendecomposition
        # per replicate while retaining a strict no-ridge regularity audit.
        block_conditions = np.empty(stop - start)
        gamma = np.empty_like(rhs)
        scaled_rhs = rhs / diagonal_scale
        pocon = get_lapack_funcs("pocon", (normalized,))
        for local_index, matrix in enumerate(normalized):
            replicate = start + local_index
            try:
                factor = cholesky(
                    matrix, lower=True, check_finite=False
                )
            except LinAlgError as error:
                raise RuntimeError(
                    "Brueck bootstrap refit "
                    f"{replicate} has a non-positive-definite normalized Hessian"
                ) from error
            reciprocal_condition, info = pocon(
                factor,
                float(np.linalg.norm(matrix, ord=1)),
                uplo="L",
            )
            condition = (
                1.0 / float(reciprocal_condition)
                if info == 0
                and np.isfinite(reciprocal_condition)
                and reciprocal_condition > 0.0
                else np.inf
            )
            block_conditions[local_index] = condition
            if not np.isfinite(condition) or condition > condition_limit:
                raise RuntimeError(
                    "Brueck bootstrap refit "
                    f"{replicate} is nonregular: normalized Hessian 1-norm "
                    "condition estimate "
                    f"{condition:.3g} exceeds {condition_limit:.3g}"
                )
            gamma[local_index] = cho_solve(
                (factor, True), scaled_rhs[local_index], check_finite=False
            )
        delta = gamma / diagonal_scale
        beta = fit.beta[None, :] - delta
        residual = np.einsum(
            "bpq,bq->bp", hessian, beta - fit.beta[None, :], optimize=True
        ) + rhs
        maximum_residual = max(
            maximum_residual,
            float(np.max(np.abs(residual))),
        )
        if np.any(~np.isfinite(beta)):
            raise RuntimeError("a Brueck bootstrap score-matching refit failed")
        refits[start:stop] = beta
        conditions[start:stop] = block_conditions

    if maximum_residual > 1e-7:
        raise RuntimeError(
            "a Brueck bootstrap refit has estimating-equation residual "
            f"{maximum_residual:.3g}"
        )
    return refits, conditions, maximum_residual


def _group_u_value_gradient_batch(
    fit: FitResult,
    geometry: BlockRBFGeometry,
    indices: np.ndarray,
    betas: np.ndarray,
    pair_weights: np.ndarray,
    *,
    return_gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate weighted Stein U functionals and parameter gradients."""

    betas = np.asarray(betas, dtype=float)
    pair_weights = np.asarray(pair_weights, dtype=float)
    if betas.ndim != 2 or betas.shape[1] != fit.beta.size:
        raise ValueError("betas must have shape (B,p)")
    if pair_weights.shape != (betas.shape[0], indices.size, indices.size):
        raise ValueError("pair_weights must have shape (B,n_group,n_group)")

    scale = geometry.block_scales[int(geometry.groups[indices[0]])]
    kernel = geometry.kernel[np.ix_(indices, indices)] / scale
    difference = geometry.difference[np.ix_(indices, indices)]
    trace = geometry.trace_cross_derivative[np.ix_(indices, indices)] / scale
    derivative = fit.derivative[indices]
    offset = fit.score[indices] - np.einsum(
        "idp,p->id", derivative, fit.beta, optimize=True
    )
    score = offset[None, :, :] + np.einsum(
        "idp,bp->bid", derivative, betas, optimize=True
    )
    weighted_kernel = pair_weights * kernel[None, :, :]
    interaction = np.einsum(
        "bij,bjd->bid", weighted_kernel, score, optimize=True
    )
    displacement = np.einsum(
        "bij,ijd->bid", weighted_kernel, difference, optimize=True
    ) / (geometry.bandwidth * geometry.bandwidth)
    values = (
        np.einsum("bid,bid->b", score, interaction, optimize=True)
        + 2.0
        * np.einsum("bid,bid->b", score, displacement, optimize=True)
        + np.einsum("bij,ij->b", pair_weights, trace, optimize=True)
    )
    gradient = None
    if return_gradient:
        gradient = 2.0 * np.einsum(
            "idp,bid->bp",
            derivative,
            interaction + displacement,
            optimize=True,
        )
    return values, gradient


def stratified_u_value_gradient(
    fit: FitResult,
    geometry: BlockRBFGeometry,
    beta: np.ndarray,
    counts: np.ndarray | None = None,
    *,
    center_core: bool = False,
) -> tuple[float, np.ndarray]:
    """Equal-stratum Stein U functional for one fit/count vector.

    ``center_core=True`` applies Brueck et al.'s empirical double centering
    within each fixed stratum.  This is the natural fixed-group extension of
    their empirical U-bootstrap; it is not the iid-mixture design in the
    published theorem.
    """

    beta = np.asarray(beta, dtype=float)
    if beta.shape != fit.beta.shape:
        raise ValueError("beta has the wrong shape")
    n = fit.score.shape[0]
    if counts is None:
        counts = np.ones(n)
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (n,):
        raise ValueError("counts must have shape (n,)")

    values = []
    gradients = []
    unique = np.unique(geometry.groups)
    for group in unique:
        indices = np.flatnonzero(geometry.groups == group)
        local = counts[indices]
        if not np.isclose(local.sum(), indices.size):
            raise ValueError("counts must preserve each stratum size")
        pair = np.outer(local, local)
        pair[np.diag_indices(indices.size)] -= local
        pair /= indices.size * (indices.size - 1)
        if center_core:
            pair = center_gram(pair)
        value, gradient = _group_u_value_gradient_batch(
            fit,
            geometry,
            indices,
            beta[None, :],
            pair[None, :, :],
            return_gradient=True,
        )
        values.append(float(value[0]))
        gradients.append(gradient[0])
    return float(np.mean(values)), np.mean(gradients, axis=0)


def brueck_statistics(
    fit: FitResult,
    geometry: BlockRBFGeometry,
    counts: np.ndarray,
    beta_bootstrap: np.ndarray,
    *,
    batch_size: int = 32,
    correction_mode: str = "midpoint",
) -> tuple[np.ndarray, float]:
    """Refitted empirical U-bootstrap with endpoint or midpoint correction.

    The fixed-stratum implementation uses within-stratum empirical centering.
    ``published`` evaluates the endpoint expression printed by Brueck et al.,

        U* h_{beta*,n}
        + (beta* - beta_hat)' {U* grad h_{beta*} - U grad h_{beta_hat}}.

    ``midpoint`` uses this project's factor-corrected trapezoidal diagnostic,

        U* h_{beta*,n}
        + (beta* - beta_hat)' [0.5 {U* grad h_{beta*}
                                    + U* grad h_{beta_hat}}
                               - U grad h_{beta_hat}].

    The midpoint form restores the conventional one-half Hessian coefficient;
    it is not attributed to Brueck et al. as an author-confirmed correction.
    """

    counts = np.asarray(counts, dtype=float)
    beta_bootstrap = np.asarray(beta_bootstrap, dtype=float)
    if counts.shape != (beta_bootstrap.shape[0], fit.score.shape[0]):
        raise ValueError("counts and beta_bootstrap have incompatible shapes")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if correction_mode not in {"published", "midpoint"}:
        raise ValueError("correction_mode must be 'published' or 'midpoint'")

    observed_u, observed_gradient = stratified_u_value_gradient(
        fit, geometry, fit.beta
    )
    statistics = np.zeros(counts.shape[0])
    unique = np.unique(geometry.groups)
    group_weight = 1.0 / unique.size

    for start in range(0, counts.shape[0], batch_size):
        stop = min(counts.shape[0], start + batch_size)
        betas = beta_bootstrap[start:stop]
        base = np.zeros(stop - start)
        gradient_star = np.zeros((stop - start, fit.beta.size))
        # The Stein core is quadratic and its parameter gradient is affine.
        # Hence evaluating at the parameter midpoint is exactly the average
        # of the two endpoint gradients required by the trapezoidal formula.
        gradient_betas = (
            betas
            if correction_mode == "published"
            else 0.5 * (betas + fit.beta[None, :])
        )
        for group in unique:
            indices = np.flatnonzero(geometry.groups == group)
            local = counts[start:stop, :][:, indices]
            if not np.allclose(local.sum(axis=1), indices.size):
                raise ValueError("counts must preserve each stratum size")
            pair = local[:, :, None] * local[:, None, :]
            diagonal = np.arange(indices.size)
            pair[:, diagonal, diagonal] -= local
            pair /= indices.size * (indices.size - 1)
            centered_pair = (
                pair
                - pair.mean(axis=2, keepdims=True)
                - pair.mean(axis=1, keepdims=True)
                + pair.mean(axis=(1, 2))[:, None, None]
            )
            group_base, _ = _group_u_value_gradient_batch(
                fit,
                geometry,
                indices,
                betas,
                centered_pair,
                return_gradient=False,
            )
            _, group_gradient = _group_u_value_gradient_batch(
                fit,
                geometry,
                indices,
                gradient_betas,
                pair,
                return_gradient=True,
            )
            base += group_weight * group_base
            gradient_star += group_weight * group_gradient
        delta = betas - fit.beta[None, :]
        correction = np.einsum(
            "bp,bp->b",
            delta,
            gradient_star - observed_gradient[None, :],
            optimize=True,
        )
        statistics[start:stop] = base + correction
    return statistics, observed_u


def evaluate_ksd(
    fit: FitResult,
    x: np.ndarray,
    groups: np.ndarray,
    bandwidth: float,
    counts: np.ndarray,
    gaussian_weights: np.ndarray,
    *,
    brueck_correction: str = "midpoint",
) -> tuple[KSDResult, np.ndarray]:
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups, dtype=int)
    n = x.shape[0]
    if counts.ndim != 2 or counts.shape[1] != n:
        raise ValueError("counts must have shape (B,n)")
    if gaussian_weights.shape != counts.shape or not np.all(
        np.isfinite(gaussian_weights)
    ):
        raise ValueError("gaussian_weights must be finite and match counts shape")
    for group in np.unique(groups):
        expected = np.sum(groups == group)
        if not np.all(counts[:, groups == group].sum(axis=1) == expected):
            raise ValueError("Each Efron replicate must preserve group sizes")
        if not np.allclose(
            gaussian_weights[:, groups == group].sum(axis=1), 0.0, atol=1e-10
        ):
            raise ValueError("Gaussian multipliers must be centered within each group")

    geometry = block_rbf_geometry(x, groups, bandwidth)
    stein = stein_gram(fit.score, geometry)
    observed_v = float(stein.mean())
    group_u = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        block = stein[np.ix_(indices, indices)] / geometry.block_scales[int(group)]
        group_u.append(
            (block.sum() - np.trace(block)) / (indices.size * (indices.size - 1))
        )
    observed_u = float(np.mean(group_u))
    influence_started = time.perf_counter()
    if fit.influence is not None and fit.regular:
        adjusted = influence_adjusted_gram(fit, stein, geometry)
        statistics = _quadratic_bootstrap(adjusted, gaussian_weights)
        influence_p_value: float | None = float(
            (1 + np.sum(statistics >= observed_v)) / (counts.shape[0] + 1)
        )
        influence_calibration = (
            "influence-adjusted stratified Gaussian multiplier V-bootstrap"
        )
        (
            influence_bootstrap_min,
            influence_bootstrap_q05,
            influence_bootstrap_median,
            influence_bootstrap_q95,
            influence_bootstrap_max,
        ) = (
            float(np.min(statistics)),
            float(np.quantile(statistics, 0.05)),
            float(np.median(statistics)),
            float(np.quantile(statistics, 0.95)),
            float(np.max(statistics)),
        )
    else:
        influence_p_value = None
        influence_calibration = (
            "unavailable: score-matching estimator is nonregular"
        )
        influence_bootstrap_min = float("nan")
        influence_bootstrap_q05 = float("nan")
        influence_bootstrap_median = float("nan")
        influence_bootstrap_q95 = float("nan")
        influence_bootstrap_max = float("nan")
    influence_seconds = time.perf_counter() - influence_started

    brueck_started = time.perf_counter()
    if fit.regular:
        beta_bootstrap, refit_conditions, refit_residual = score_matching_refits(
            fit, counts
        )
        brueck_bootstrap, brueck_observed_u = brueck_statistics(
            fit,
            geometry,
            counts,
            beta_bootstrap,
            correction_mode=brueck_correction,
        )
        if not np.isclose(brueck_observed_u, observed_u, rtol=1e-10, atol=1e-12):
            raise RuntimeError("internal mismatch in the stratified KSD U-statistic")
        brueck_p_value: float | None = float(
            (1 + np.sum(brueck_bootstrap >= observed_u))
            / (counts.shape[0] + 1)
        )
        brueck_calibration = (
            "stratified refitted empirical U-bootstrap with "
            + (
                "project factor-corrected midpoint diagnostic"
                if brueck_correction == "midpoint"
                else "Brueck et al. endpoint expression as printed"
            )
        )
        brueck_refits = int(counts.shape[0])
        brueck_max_condition = float(np.max(refit_conditions))
        brueck_max_residual = float(refit_residual)
        (
            brueck_bootstrap_min,
            brueck_bootstrap_q05,
            brueck_bootstrap_median,
            brueck_bootstrap_q95,
            brueck_bootstrap_max,
        ) = (
            float(np.min(brueck_bootstrap)),
            float(np.quantile(brueck_bootstrap, 0.05)),
            float(np.median(brueck_bootstrap)),
            float(np.quantile(brueck_bootstrap, 0.95)),
            float(np.max(brueck_bootstrap)),
        )
    else:
        brueck_p_value = None
        brueck_calibration = (
            "unavailable: score-matching estimator is nonregular"
        )
        brueck_refits = 0
        brueck_max_condition = float("nan")
        brueck_max_residual = float("nan")
        brueck_bootstrap_min = float("nan")
        brueck_bootstrap_q05 = float("nan")
        brueck_bootstrap_median = float("nan")
        brueck_bootstrap_q95 = float("nan")
        brueck_bootstrap_max = float("nan")
    brueck_seconds = time.perf_counter() - brueck_started

    return (
        KSDResult(
            v_statistic=observed_v,
            u_statistic=observed_u,
            ksd=float(np.sqrt(max(observed_v, 0.0))),
            influence_p_value=influence_p_value,
            brueck_p_value=brueck_p_value,
            influence_calibration=influence_calibration,
            brueck_calibration=brueck_calibration,
            bootstrap_replicates=int(counts.shape[0]),
            influence_seconds=float(influence_seconds),
            influence_bootstrap_min=influence_bootstrap_min,
            influence_bootstrap_q05=influence_bootstrap_q05,
            influence_bootstrap_median=influence_bootstrap_median,
            influence_bootstrap_q95=influence_bootstrap_q95,
            influence_bootstrap_max=influence_bootstrap_max,
            brueck_seconds=float(brueck_seconds),
            brueck_refits=brueck_refits,
            brueck_max_hessian_condition=brueck_max_condition,
            brueck_max_estimating_residual=brueck_max_residual,
            brueck_bootstrap_min=brueck_bootstrap_min,
            brueck_bootstrap_q05=brueck_bootstrap_q05,
            brueck_bootstrap_median=brueck_bootstrap_median,
            brueck_bootstrap_q95=brueck_bootstrap_q95,
            brueck_bootstrap_max=brueck_bootstrap_max,
        ),
        stein,
    )
