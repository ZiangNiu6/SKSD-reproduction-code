"""Heavy-tailed continuous EBMs for T-cell-stratified RPPA networks.

For group g, the unnormalised density is a product Student-t reference times
bounded one-node and pairwise tanh tilts. Every finite parameter vector is a
proper density, the spatial score is affine in the parameters, and an edge
coefficient is zero exactly when that conditional-dependence term is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln

from .data import GROUP_LABELS, PROTEINS


MODEL_LABELS = {
    "independent": "Group-specific marginals; no interactions",
    "shared": "One interaction matrix shared across T-cell groups",
    "varying": "Separate interaction matrices by T-cell group",
}
DEFAULT_MARGINAL_KNOTS = (-2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_MARGINAL_WIDTH = 0.7


@dataclass(frozen=True)
class StudentTBase:
    location: np.ndarray
    scale: np.ndarray
    degrees_of_freedom: float = 5.0

    def __post_init__(self) -> None:
        location = np.asarray(self.location, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        if location.shape != (len(GROUP_LABELS), len(PROTEINS)):
            raise ValueError("location must have shape (2, number of proteins)")
        if scale.shape != location.shape or np.any(scale <= 0.0):
            raise ValueError("scale must match location and be strictly positive")
        if self.degrees_of_freedom <= 0.0:
            raise ValueError("degrees_of_freedom must be positive")
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True)
class LinearDesign:
    model: str
    offset: np.ndarray
    derivative: np.ndarray
    divergence_offset: np.ndarray
    divergence_basis: np.ndarray
    features: np.ndarray
    parameter_names: tuple[str, ...]
    parameter_groups: tuple[int | None, ...]
    edges: tuple[tuple[int, int], ...]
    marginal_knots: tuple[float, ...]
    marginal_width: float
    phi_scale: float


@dataclass
class FitResult:
    model: str
    label: str
    beta: np.ndarray
    parameter_names: tuple[str, ...]
    parameter_groups: tuple[int | None, ...]
    edges: tuple[tuple[int, int], ...]
    marginal_knots: tuple[float, ...]
    marginal_width: float
    phi_scale: float
    base: StudentTBase
    score: np.ndarray
    derivative: np.ndarray
    divergence_offset: np.ndarray
    divergence_basis: np.ndarray
    estimating_terms: np.ndarray
    influence: np.ndarray | None
    hessian: np.ndarray
    hessian_condition: float
    converged: bool
    normalizable: bool
    regular: bool
    message: str
    diagnostics: dict[str, Any]


def _validate_inputs(x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups, dtype=int)
    if x.ndim != 2 or x.shape[1] != len(PROTEINS):
        raise ValueError(f"x must have shape (n,{len(PROTEINS)})")
    if groups.shape != (x.shape[0],) or not set(np.unique(groups)).issubset({0, 1}):
        raise ValueError("groups must be a length-n vector containing only 0/1")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    return x, groups


def _student_t_terms(
    x: np.ndarray, groups: np.ndarray, base: StudentTBase
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    location = base.location[groups]
    scale = base.scale[groups]
    nu = float(base.degrees_of_freedom)
    delta = x - location
    denominator = nu * scale * scale + delta * delta
    score = -(nu + 1.0) * delta / denominator
    second = -(nu + 1.0) * (nu * scale * scale - delta * delta) / (
        denominator * denominator
    )
    log_density = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
        - 0.5 * (nu + 1.0) * np.log1p(delta * delta / (nu * scale * scale))
    )
    return score, second, log_density


def _basis_terms(
    x: np.ndarray,
    phi_scale: float,
    marginal_knots: tuple[float, ...],
    marginal_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(phi_scale) or phi_scale <= 0.0:
        raise ValueError("phi_scale must be finite and positive")
    if not marginal_knots or not np.all(np.isfinite(marginal_knots)):
        raise ValueError("marginal_knots must contain finite values")
    if not np.isfinite(marginal_width) or marginal_width <= 0.0:
        raise ValueError("marginal_width must be finite and positive")
    phi = np.tanh(x / phi_scale)
    first = (1.0 - phi * phi) / phi_scale
    second = -2.0 * phi * (1.0 - phi * phi) / (phi_scale * phi_scale)
    shifted = np.stack(
        [np.tanh((x - knot) / marginal_width) for knot in marginal_knots],
        axis=2,
    )
    gradients = (1.0 - shifted * shifted) / marginal_width
    laplacians = (
        -2.0
        * shifted
        * (1.0 - shifted * shifted)
        / (marginal_width * marginal_width)
    )
    return phi, first, second, shifted, gradients, laplacians


def build_design(
    model: str,
    x: np.ndarray,
    groups: np.ndarray,
    base: StudentTBase,
    edges: tuple[tuple[int, int], ...],
    *,
    marginal_knots: tuple[float, ...] = DEFAULT_MARGINAL_KNOTS,
    marginal_width: float = DEFAULT_MARGINAL_WIDTH,
    phi_scale: float = 1.0,
) -> LinearDesign:
    if model not in MODEL_LABELS:
        raise ValueError(f"Unknown model {model!r}")
    x, groups = _validate_inputs(x, groups)
    marginal_knots = tuple(float(value) for value in marginal_knots)
    edges = tuple(edges)
    n, d = x.shape
    for i, j in edges:
        if not (0 <= i < j < d):
            raise ValueError(f"Invalid undirected edge {(i, j)}")

    offset, divergence_coordinate, _ = _student_t_terms(x, groups, base)
    phi, phi_first, phi_second, basis, basis_first, basis_second = _basis_terms(
        x, phi_scale, marginal_knots, marginal_width
    )

    names: list[str] = []
    parameter_groups: list[int | None] = []
    feature_columns: list[np.ndarray] = []
    derivative_columns: list[np.ndarray] = []
    divergence_columns: list[np.ndarray] = []

    for group, group_label in enumerate(GROUP_LABELS):
        indicator = groups == group
        for j, protein in enumerate(PROTEINS):
            for knot_index, knot in enumerate(marginal_knots):
                feature = np.zeros(n)
                feature[indicator] = basis[indicator, j, knot_index]
                derivative = np.zeros((n, d))
                derivative[indicator, j] = basis_first[indicator, j, knot_index]
                divergence = np.zeros(n)
                divergence[indicator] = basis_second[indicator, j, knot_index]
                names.append(f"a[knot={knot:g};{group_label}:{protein}]")
                parameter_groups.append(group)
                feature_columns.append(feature)
                derivative_columns.append(derivative)
                divergence_columns.append(divergence)

    if model == "shared":
        for i, j in edges:
            feature = phi[:, i] * phi[:, j]
            derivative = np.zeros((n, d))
            derivative[:, i] = phi_first[:, i] * phi[:, j]
            derivative[:, j] = phi_first[:, j] * phi[:, i]
            divergence = phi_second[:, i] * phi[:, j] + phi_second[:, j] * phi[:, i]
            names.append(f"J[shared:{PROTEINS[i]},{PROTEINS[j]}]")
            parameter_groups.append(None)
            feature_columns.append(feature)
            derivative_columns.append(derivative)
            divergence_columns.append(divergence)
    elif model == "varying":
        for group, group_label in enumerate(GROUP_LABELS):
            indicator = groups == group
            for i, j in edges:
                feature = np.zeros(n)
                feature[indicator] = phi[indicator, i] * phi[indicator, j]
                derivative = np.zeros((n, d))
                derivative[indicator, i] = phi_first[indicator, i] * phi[indicator, j]
                derivative[indicator, j] = phi_first[indicator, j] * phi[indicator, i]
                divergence = np.zeros(n)
                divergence[indicator] = (
                    phi_second[indicator, i] * phi[indicator, j]
                    + phi_second[indicator, j] * phi[indicator, i]
                )
                names.append(f"J[{group_label}:{PROTEINS[i]},{PROTEINS[j]}]")
                parameter_groups.append(group)
                feature_columns.append(feature)
                derivative_columns.append(derivative)
                divergence_columns.append(divergence)

    features = np.column_stack(feature_columns)
    derivative = np.stack(derivative_columns, axis=2)
    divergence_basis = np.column_stack(divergence_columns)
    return LinearDesign(
        model=model,
        offset=offset,
        derivative=derivative,
        divergence_offset=divergence_coordinate.sum(axis=1),
        divergence_basis=divergence_basis,
        features=features,
        parameter_names=tuple(names),
        parameter_groups=tuple(parameter_groups),
        edges=edges,
        marginal_knots=marginal_knots,
        marginal_width=float(marginal_width),
        phi_scale=float(phi_scale),
    )


def fit_model(
    model: str,
    x: np.ndarray,
    groups: np.ndarray,
    base: StudentTBase,
    edges: tuple[tuple[int, int], ...],
    *,
    marginal_knots: tuple[float, ...] = DEFAULT_MARGINAL_KNOTS,
    marginal_width: float = DEFAULT_MARGINAL_WIDTH,
    phi_scale: float = 1.0,
    condition_limit: float = 1e10,
) -> FitResult:
    design = build_design(
        model,
        x,
        groups,
        base,
        edges,
        marginal_knots=marginal_knots,
        marginal_width=marginal_width,
        phi_scale=phi_scale,
    )
    n = design.derivative.shape[0]
    hessian = np.einsum(
        "idp,idq->pq", design.derivative, design.derivative, optimize=True
    ) / n
    linear = (
        np.einsum(
            "idp,id->p", design.derivative, design.offset, optimize=True
        )
        / n
        + design.divergence_basis.mean(axis=0)
    )

    diagonal_scale = np.sqrt(np.maximum(np.diag(hessian), 0.0))
    nonsingular_columns = np.all(np.isfinite(diagonal_scale)) and np.all(
        diagonal_scale > 1e-12
    )
    beta = np.full(linear.shape, np.nan)
    normalized_condition = np.inf
    solved = False
    if nonsingular_columns:
        normalized_hessian = hessian / np.outer(diagonal_scale, diagonal_scale)
        normalized_linear = linear / diagonal_scale
        normalized_condition = float(np.linalg.cond(normalized_hessian))
        try:
            gamma = -np.linalg.solve(normalized_hessian, normalized_linear)
            beta = gamma / diagonal_scale
            solved = bool(np.all(np.isfinite(beta)))
        except np.linalg.LinAlgError:
            solved = False

    if solved:
        score = design.offset + np.einsum(
            "idp,p->id", design.derivative, beta, optimize=True
        )
        estimating_terms = (
            np.einsum(
                "idp,id->ip", design.derivative, score, optimize=True
            )
            + design.divergence_basis
        )
    else:
        score = np.full_like(design.offset, np.nan)
        estimating_terms = np.full((n, linear.size), np.nan)

    regular = solved and np.isfinite(normalized_condition) and (
        normalized_condition <= condition_limit
    )
    influence: np.ndarray | None = None
    if regular:
        normalized_hessian = hessian / np.outer(diagonal_scale, diagonal_scale)
        normalized_inverse = np.linalg.inv(normalized_hessian)
        hessian_inverse = (
            normalized_inverse / diagonal_scale[:, None] / diagonal_scale[None, :]
        )
        # Group sizes are fixed by design and by the stratified bootstrap, so
        # remove each stratum's estimating-term mean rather than treating the
        # low/high composition as random.
        centered = estimating_terms.copy()
        for group in np.unique(groups):
            indicator = groups == group
            centered[indicator] -= centered[indicator].mean(axis=0)
        influence = -centered @ hessian_inverse.T

    gradient_residual = (
        float(np.linalg.norm(estimating_terms.mean(axis=0), ord=np.inf))
        if solved
        else np.inf
    )
    converged = solved and gradient_residual <= 1e-7
    if not solved:
        message = "score-matching system is singular"
    elif not converged:
        message = f"estimating-equation residual is {gradient_residual:.3g}"
    elif not regular:
        message = (
            f"fit solved but normalized Hessian condition {normalized_condition:.3g} "
            f"exceeds {condition_limit:.3g}"
        )
    else:
        message = "unconstrained quadratic score-matching solution"

    eigenvalues = (
        np.linalg.eigvalsh(hessian) if np.all(np.isfinite(hessian)) else np.array([np.nan])
    )
    return FitResult(
        model=model,
        label=MODEL_LABELS[model],
        beta=beta,
        parameter_names=design.parameter_names,
        parameter_groups=design.parameter_groups,
        edges=design.edges,
        marginal_knots=design.marginal_knots,
        marginal_width=design.marginal_width,
        phi_scale=design.phi_scale,
        base=base,
        score=score,
        derivative=design.derivative,
        divergence_offset=design.divergence_offset,
        divergence_basis=design.divergence_basis,
        estimating_terms=estimating_terms,
        influence=influence,
        hessian=hessian,
        hessian_condition=normalized_condition,
        converged=converged,
        normalizable=solved,
        regular=regular and converged,
        message=message,
        diagnostics={
            "estimating_equation_residual": gradient_residual,
            "minimum_hessian_eigenvalue": float(eigenvalues[0]),
            "maximum_hessian_eigenvalue": float(eigenvalues[-1]),
            "parameter_count": int(linear.size),
        },
    )
