"""Scientific core for the supplemental one-dimensional KEF score-matching study.

The data-generating density is proportional to

    N(0, sigma**2) * exp(theta0 * phi_0(x) + eta * phi_3(x)),

where ``eta = signal / SD_null(phi_3)``. The fitted null uses the first one,
two, or three KEF basis functions. IA and NWB use the same fitted Stein
V-statistic, data, and centered Gaussian multiplier draws. Decisions use the
nominal p-value cutoff; this module performs no empirical size calibration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import quad

def _load_kef_core():
    """Load the sibling scientific core in a standalone reproduction copy."""
    source = Path(__file__).with_name("_kef_core.py")
    spec = importlib.util.spec_from_file_location("_kef_score_matching_reference", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load KEF scientific core: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kef_core = _load_kef_core()


FITTED_SPANS = ((0,), (0, 1), (0, 1, 2))


def null_basis_sd(theta0: float, sigma: float, basis: int = 3) -> float:
    """Return the standard deviation of ``phi_basis`` under the true null.

    The common factor ``exp(-theta0)`` keeps the quadrature integrand bounded
    without changing its normalized moments.
    """
    if not math.isfinite(theta0) or theta0 <= 0:
        raise ValueError("theta0 must be finite and positive")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not isinstance(basis, int) or basis < 0:
        raise ValueError("basis must be a nonnegative integer")

    def scaled_weight(x: float) -> float:
        return math.exp(
            -x * x / (2 * sigma * sigma)
            + theta0 * math.exp(-x * x / 2) - theta0
        ) / (sigma * math.sqrt(2 * math.pi))

    def moment(power: int) -> float:
        value, _ = quad(
            lambda x: scaled_weight(x) * float(kef_core.phi(x, basis)) ** power,
            -math.inf, math.inf, epsabs=1e-12, epsrel=1e-12, limit=300,
        )
        return float(value)

    normalizer = moment(0)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise RuntimeError("null quadrature produced an invalid normalizer")
    mean = moment(1) / normalizer
    variance = moment(2) / normalizer - mean * mean
    if not math.isfinite(variance) or variance <= 0:
        raise RuntimeError("null quadrature produced an invalid basis variance")
    return math.sqrt(variance)


def _signal_code(signal: float) -> int:
    """Stable seed coordinate independent of the signal grid and its order."""
    if signal == 0.0:
        return 6  # Historical one-dimensional KEF null coordinate.
    canonical = format(signal, ".12g")
    return int.from_bytes(
        hashlib.sha256(canonical.encode("ascii")).digest()[:4], "little"
    )


def _exact_phi3_log_envelope(theta0: float, eta: float) -> float:
    """Global maximum of ``theta0*phi_0 + eta*phi_3`` on the real line."""
    if eta == 0:
        return theta0
    coefficient = eta / math.sqrt(6)
    # In addition to x=0, stationary points solve
    # x**3 - 3*x + theta0/coefficient = 0.
    roots = np.roots([1.0, 0.0, -3.0, theta0 / coefficient])
    values = [theta0, 0.0]  # The potential approaches zero at both tails.
    for root in roots:
        if abs(root.imag) <= 1e-9 * (1 + abs(root.real)):
            x = float(root.real)
            values.append(math.exp(-x * x / 2) * (theta0 + coefficient * x**3))
    return max(values)


def _sample_alternative(
    n: int, sigma: float, theta0: float, eta: float, seed: int
) -> tuple[np.ndarray, int, float]:
    """Exact Gaussian-proposal rejection sampler with an analytic envelope."""
    rng = np.random.RandomState(seed)
    log_envelope = _exact_phi3_log_envelope(theta0, eta) + 1e-3
    out = np.empty(n)
    filled = proposals = 0
    batch = max(64, 8 * n)
    while filled < n:
        x = rng.normal(0.0, sigma, batch)
        potential = theta0 * kef_core.phi(x, 0) + eta * kef_core.phi(x, 3)
        if np.max(potential) > log_envelope + 1e-12:
            raise RuntimeError("rejection envelope violated")
        accepted = x[np.log(rng.uniform(size=batch)) < potential - log_envelope]
        take = min(len(accepted), n - filled)
        out[filled:filled + take] = accepted[:take]
        filled += take
        proposals += batch
    return out, proposals, log_envelope


def _basis_derivatives(
    x: np.ndarray, indices: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Columns of first and second derivatives of the fitted KEF basis."""
    x = np.asarray(x, dtype=float).reshape(-1)
    envelope = np.exp(-0.5 * x * x)
    first = []
    second = []
    for r in indices:
        scale = envelope / math.sqrt(math.factorial(r))
        first.append(scale * ((r * x ** (r - 1) if r else 0) - x ** (r + 1)))
        second.append(scale * (
            (r * (r - 1) * x ** (r - 2) if r >= 2 else 0)
            - (2 * r + 1) * x**r + x ** (r + 2)
        ))
    return np.column_stack(first), np.column_stack(second)


def score_matching_fit(
    x: np.ndarray, sigma: float, fitted_bases: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Ordinary score-matching estimate and observation-level influence."""
    x = np.asarray(x, dtype=float).reshape(-1)
    beta, beta_prime = _basis_derivatives(x, fitted_bases)
    alpha = -x / sigma**2
    matrix = beta.T @ beta / len(x)
    linear = np.mean(alpha[:, None] * beta + beta_prime, axis=0)
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > 1e12:
        raise np.linalg.LinAlgError(
            f"score-matching matrix ill-conditioned: {condition:g}"
        )
    theta = np.linalg.solve(matrix, -linear)
    equation = beta * (alpha + beta @ theta)[:, None] + beta_prime
    residual = float(np.max(np.abs(equation.mean(axis=0))))
    influence = -np.linalg.solve(matrix, equation.T).T
    influence -= influence.mean(axis=0, keepdims=True)
    if not (
        np.all(np.isfinite(theta))
        and np.all(np.isfinite(influence))
        and residual < 1e-8
    ):
        raise RuntimeError(f"invalid score-matching fit, residual={residual:g}")
    return theta, beta, influence, condition, residual


def stein_and_adjusted_grams(
    x: np.ndarray, sigma: float, bandwidth: float, theta: np.ndarray,
    beta: np.ndarray, influence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fitted Stein Gram and centered influence-adjusted IA Gram."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    difference = x[:, None] - x[None, :]
    inv_h2 = 1.0 / bandwidth**2
    kernel = np.exp(-0.5 * difference * difference * inv_h2)
    score = -x / sigma**2 + beta @ theta
    stein = kernel * (
        score[:, None] * score[None, :]
        + (score[:, None] - score[None, :]) * difference * inv_h2
        + inv_h2 - difference * difference * inv_h2**2
    )
    cross = (kernel * (score[:, None] - difference * inv_h2)) @ beta / n
    derivative_gram = beta.T @ kernel @ beta / n**2
    adjusted = (
        stein + cross @ influence.T + influence @ cross.T
        + influence @ derivative_gram @ influence.T
    )
    row_mean = adjusted.mean(axis=1, keepdims=True)
    adjusted = adjusted - row_mean - row_mean.T + adjusted.mean()
    if not (np.all(np.isfinite(stein)) and np.all(np.isfinite(adjusted))):
        raise RuntimeError("nonfinite Stein or IA Gram")
    return stein, adjusted


def run_replication(
    fitted_bases: tuple[int, ...], signal: float, rep: int, *,
    n: int, B: int, alpha: float, theta0: float, sigma: float,
    base_seed: int, phi3_sd: float,
) -> dict:
    """Run one paired nominal SKSD-IA/SKSD-NWB replication.

    The data and bootstrap seeds match the standalone investigation. Efron
    counts are drawn and discarded before the Gaussian multipliers, preserving
    its random-number stream exactly.
    """
    if fitted_bases not in FITTED_SPANS:
        raise ValueError(f"fitted_bases must be one of {FITTED_SPANS}")
    if not math.isfinite(signal) or signal < 0:
        raise ValueError("signal must be finite and nonnegative")
    if not isinstance(rep, int) or rep < 0:
        raise ValueError("rep must be a nonnegative integer")
    if not isinstance(n, int) or n <= len(fitted_bases):
        raise ValueError("n must exceed the number of fitted coefficients")
    if not isinstance(B, int) or B < 1:
        raise ValueError("B must be a positive integer")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if not math.isfinite(theta0) or theta0 <= 0:
        raise ValueError("theta0 must be finite and positive")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if not math.isfinite(phi3_sd) or phi3_sd <= 0:
        raise ValueError("phi3_sd must be finite and positive")

    started = time.perf_counter()
    eta = signal / phi3_sd
    signal_code = _signal_code(signal)
    data_seed = kef_core._cell_seed(base_seed, 1, signal_code, rep, 0)
    bootstrap_seed = kef_core._cell_seed(base_seed, 1, signal_code, rep, 16)
    x, proposals, log_envelope = _sample_alternative(
        n, sigma, theta0, eta, data_seed
    )
    bandwidth = kef_core.ksd_bw_md(x[:, None])
    theta, beta, influence, condition, residual = score_matching_fit(
        x, sigma, fitted_bases
    )
    stein, adjusted = stein_and_adjusted_grams(
        x, sigma, bandwidth, theta, beta, influence
    )
    observed_v = float(stein.mean())

    rng = np.random.default_rng(bootstrap_seed)
    rng.multinomial(n, np.full(n, 1 / n), size=B)
    weights = rng.standard_normal((B, n))
    weights -= weights.mean(axis=1, keepdims=True)

    def pvalue(matrix: np.ndarray) -> float:
        bootstrap = np.einsum(
            "bi,ij,bj->b", weights, matrix, weights, optimize=True
        ) / n**2
        return float((1 + np.count_nonzero(bootstrap >= observed_v)) / (B + 1))

    p_ia, p_nwb = pvalue(adjusted), pvalue(stein)
    return {
        "setup": f"p{len(fitted_bases)}",
        "fitted_bases": ",".join(map(str, fitted_bases)),
        "signal": signal,
        "eta": eta,
        "rep": rep,
        "data_seed": data_seed,
        "bootstrap_seed": bootstrap_seed,
        "theta_hat_0": float(theta[0]),
        "theta_hat_1": float(theta[1]) if len(theta) > 1 else "",
        "theta_hat_2": float(theta[2]) if len(theta) > 2 else "",
        "sm_condition_number": condition,
        "sm_equation_residual": residual,
        "bandwidth": float(bandwidth),
        "observed_v": observed_v,
        "pvalue_ia": p_ia,
        "pvalue_nwb": p_nwb,
        "reject_ia": int(p_ia <= alpha),
        "reject_nwb": int(p_nwb <= alpha),
        "sample_proposals": proposals,
        "log_envelope": log_envelope,
        "seconds": time.perf_counter() - started,
    }
