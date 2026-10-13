"""Scientific core for the kernel exponential-family simulation.

The numerical routines are retained verbatim from the historical KEF producer.
The public :func:`run_replication` adapter fixes the manuscript configuration
used by the reproduction workflow: minimum-KSD nuisance fitting, fitted-model
bootstrap refitting, and the factor-corrected Brück comparison.
"""

from __future__ import annotations

from math import factorial
import time

import numpy as np


TIMING_PROTOCOL = "standalone-test-v1"
TIMING_CLOCK = "perf_counter"
HISTORICAL_SCIENTIFIC_FUNCTION_COUNT = 33
HISTORICAL_SCIENTIFIC_AST_SHA256 = (
    "a8356ce49678f518823aeca3fd8a06c7cac6a48e5ce672427bf83d1b94c912d8"
)

# np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement. Support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def phi(x, i):
    return (x ** i / np.sqrt(factorial(i))) * np.exp(-x ** 2 / 2)

def _logM(theta, sigma, n_grid=4001, x_max_sd=8.0):
    """log(max_x exp(sum theta_i phi_i(x))) on a dense grid."""
    xg = np.linspace(-x_max_sd * sigma, x_max_sd * sigma, n_grid)
    Eg = np.zeros_like(xg)
    for i in range(len(theta)):
        Eg = Eg + theta[i] * phi(xg, i)
    return Eg.max()

def rejection_sampler(n, sigma, theta, rng=None, safety=1e-3):
    """
    Exact i.i.d. sampler for p_theta(x) = N(0,sigma^2)(x) * exp(sum theta_i phi_i(x)) / Z,
    by rejection with proposal N(0, sigma^2) and bound M >= max_x exp(E(x)).
    safety: small log-margin added to logM so the grid-max isn't tight against the true peak.
    """
    if rng is None: rng = np.random
    logM = _logM(theta, sigma) + safety
    p = len(theta)
    out = np.empty(n); filled = 0
    # initial batch big enough so even ~1% acceptance fills in 1-2 rounds
    batch = max(64, 8 * n)
    while filled < n:
        x = rng.normal(0.0, sigma, batch)
        E = np.zeros_like(x)
        for i in range(p):
            E = E + theta[i] * phi(x, i)
        u = rng.uniform(size=batch)
        acc = x[np.log(u) < (E - logM)]                  # log-space avoids exp overflow
        take = min(len(acc), n - filled)
        out[filled:filled + take] = acc[:take]
        filled += take
    return out


# ---------- multidim sampler (factorised: d independent 1D rejection draws) ----------
def rejection_sampler_md(n, sigma, theta, d, rng=None):
    X = np.empty((n, d))
    for j in range(d):
        X[:, j] = rejection_sampler(n, sigma, theta, rng=rng)
    return X


# ---------- multidim bandwidth ----------
def _pairwise_dists(X):
    D2 = np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2), 0)
    return np.sqrt(D2)

def ksd_bw_md(X):                                    # KSD bandwidth: 1.5 * median heuristic
    D = _pairwise_dists(X); n = X.shape[0]
    return 1.5 * np.median(D[np.triu_indices(n, k=1)])

def mmd_bw_md(Z):                                    # MMD bandwidth: plain median heuristic
    D = _pairwise_dists(Z); n = Z.shape[0]
    return np.median(D[np.triu_indices(n, k=1)])


# ---------- d-dim KSD Gram matrix and U-statistic ----------
def _ksd_gram(S, X, K, D2, h, d):
    """Stein-feature Gram matrix from a precomputed score field."""
    inv_h2 = 1.0 / (h * h)
    A   = (S * X).sum(axis=1)                        # (n,)
    SXt = S @ X.T                                    # (n,n), (SXt)[i,j] = <S_i, X_j>
    return (
        (S @ S.T) * K
        + (A[:, None] - SXt) * K * inv_h2
        + (A[None, :] - SXt.T) * K * inv_h2
        + (d * inv_h2 - D2 * inv_h2 * inv_h2) * K
    )


def _ksd_U(S, X, K, D2, h, d):
    """KSD U-statistic, excluding the i=j diagonal."""
    n = X.shape[0]
    M = _ksd_gram(S, X, K, D2, h, d)
    M = M.copy()
    np.fill_diagonal(M, 0.0)
    return M.sum() / (n * (n - 1))


# ---------- min-KSD fit, p=1, closed form (V(theta_0) is quadratic) ----------
def _min_ksd_equation_md(X, h, sigma):
    """Return (B, 2C) for the scalar estimating equation B + 2C theta=0."""
    n, d = X.shape
    alpha = -X / sigma**2                            # (n,d)
    beta  = -X * np.exp(-X**2 / 2)                   # (n,d) = grad_phi_0(X)
    XX = (X * X).sum(axis=1)
    D2 = np.maximum(XX[:, None] + XX[None, :] - 2 * X @ X.T, 0)
    K  = np.exp(-D2 / (2 * h * h))
    V0  = _ksd_U(alpha,          X, K, D2, h, d)
    Vp1 = _ksd_U(alpha + beta,   X, K, D2, h, d)
    Vm1 = _ksd_U(alpha - beta,   X, K, D2, h, d)
    A = V0; B = (Vp1 - Vm1) / 2.0; C = (Vp1 + Vm1) / 2.0 - A
    return B, 2.0 * C


def min_ksd_p1_md(X, h, sigma):
    B, jacobian = _min_ksd_equation_md(X, h, sigma)
    return np.array([-B / jacobian])                 # length-1 theta vector


# ---------- min-SM fit, p=1, closed form (Hyvarinen score matching) ----------
def sm_estimator_p1_md(X, h, sigma):
    """Score-matching fit for the factorised KEF with p=1, pooling all (n*d) coords.
    h is accepted only for a uniform signature with min_ksd_p1_md; it is unused
    (SM has no kernel bandwidth). Closed form: J(theta_0) = sum [s^2/2 + s'], quadratic in theta_0,
    minimiser = -(sum alpha*beta + sum beta') / sum beta^2 over the (n*d) coords.
    """
    x = X.ravel()                                    # treat (n,d) as one big sample
    e = np.exp(-x * x / 2.0)
    alpha  = -x / (sigma * sigma)                    # d/dz log q(z)
    beta   = -x * e                                  # grad phi_0
    bprime = (x * x - 1.0) * e                       # grad^2 phi_0
    return np.array([-(np.sum(alpha * beta) + np.sum(bprime)) / np.sum(beta * beta)])


def _sm_equation_md(X, theta0, sigma):
    """Mean score-matching estimating equation and its scalar Jacobian."""
    x = X.ravel()
    e = np.exp(-x * x / 2.0)
    alpha = -x / (sigma * sigma)
    beta = -x * e
    bprime = (x * x - 1.0) * e
    return np.mean(alpha * beta + bprime + theta0 * beta * beta), np.mean(beta * beta)


def make_one_step_updater(theta_hat, h, sigma, d, est, prep_n):
    """Build a model-Jacobian influence update, with one preparation sample."""
    Xprep = rejection_sampler_md(prep_n, sigma, theta_hat, d)
    if est is min_ksd_p1_md:
        _, jacobian = _min_ksd_equation_md(Xprep, h, sigma)

        def update(Xb):
            B_b, jacobian_b = _min_ksd_equation_md(Xb, h, sigma)
            score_b = B_b + jacobian_b * theta_hat[0]
            return np.array([theta_hat[0] - score_b / jacobian])
    else:
        _, jacobian = _sm_equation_md(Xprep, theta_hat[0], sigma)

        def update(Xb):
            score_b, _ = _sm_equation_md(Xb, theta_hat[0], sigma)
            return np.array([theta_hat[0] - score_b / jacobian])
    return update


# ---------- d-dim KSD test stat at a given theta (p=1) ----------
def ksd_gram_md(X, sigma, theta0, h):
    """Return the RBF kernel and the full Stein-feature Gram matrix."""
    n, d = X.shape
    S = -X / sigma**2 + theta0 * (-X * np.exp(-X**2 / 2))
    XX = (X * X).sum(axis=1)
    D2 = np.maximum(XX[:, None] + XX[None, :] - 2 * X @ X.T, 0)
    K  = np.exp(-D2 / (2 * h * h))
    return K, _ksd_gram(S, X, K, D2, h, d)


def weighted_u_md(matrix, counts):
    n = int(np.sum(counts))
    total = counts @ matrix @ counts - counts @ np.diag(matrix)
    return float(total / (n * (n - 1)))


def weighted_u_batch_md(matrices, counts):
    n = counts.shape[1]
    diagonal = matrices[:, np.arange(n), np.arange(n)]
    total = (
        np.einsum("bi,bij,bj->b", counts, matrices, counts, optimize=True)
        - np.einsum("bi,bi->b", counts, diagonal, optimize=True)
    )
    return total / (n * (n - 1))


def weighted_u_common_md(matrix, counts):
    """Weighted U-statistic for many count vectors and one shared matrix."""
    n = counts.shape[1]
    total = (
        np.einsum(
            "bi,ij,bj->b", counts, matrix, counts, optimize=False
        )
        - counts @ np.diag(matrix)
    )
    return total / (n * (n - 1))


def weighted_u_components_md(matrices, counts):
    """Weighted U-statistics for shared matrix components."""
    n = counts.shape[1]
    total = np.einsum(
        "bi,kij,bj->bk", counts, matrices, counts, optimize=True
    )
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    total -= counts @ diagonal.T
    return total / (n * (n - 1))


def ksd_stat_md(X, sigma, theta0, h):
    _, matrix = ksd_gram_md(X, sigma, theta0, h)
    return weighted_u_md(matrix, np.ones(X.shape[0]))


def ksd_vstat_md(X, sigma, theta0, h):
    _, matrix = ksd_gram_md(X, sigma, theta0, h)
    return float(matrix.mean())


def _ksd_quadratic_grams_md(X, sigma, h):
    """Return H(theta)=H0+theta*H1+theta^2*H2."""
    _, h0 = ksd_gram_md(X, sigma, 0.0, h)
    _, hp = ksd_gram_md(X, sigma, 1.0, h)
    _, hm = ksd_gram_md(X, sigma, -1.0, h)
    h1 = 0.5 * (hp - hm)
    h2 = 0.5 * (hp + hm) - h0
    return h0, h1, h2


def _refit_counts_md(X, sigma, est, counts, h1, h2):
    """Vectorized nuisance refits for empirical-bootstrap counts."""
    if est is min_ksd_p1_md:
        linear = weighted_u_common_md(h1, counts)
        quadratic = weighted_u_common_md(h2, counts)
        return -linear / (2.0 * quadratic)

    x = X
    e = np.exp(-x * x / 2.0)
    alpha = -x / (sigma * sigma)
    beta = -x * e
    bprime = (x * x - 1.0) * e
    numerator_i = np.sum(alpha * beta + bprime, axis=1)
    denominator_i = np.sum(beta * beta, axis=1)
    return -(counts @ numerator_i) / (counts @ denominator_i)


def _nuisance_influence_md(X, sigma, theta_hat, est, h1, h2):
    """Observation-level influence summands for the selected estimator."""
    n, d = X.shape
    if est is min_ksd_p1_md:
        gradient = h1 + 2.0 * theta_hat[0] * h2
        gradient = gradient.copy()
        np.fill_diagonal(gradient, 0.0)
        first_projection = gradient.sum(axis=1) / (n - 1)
        jacobian = 2.0 * weighted_u_md(h2, np.ones(n))
        influence = -2.0 * first_projection / jacobian
    else:
        x = X
        e = np.exp(-x * x / 2.0)
        alpha = -x / (sigma * sigma)
        beta = -x * e
        bprime = (x * x - 1.0) * e
        contribution = np.mean(
            alpha * beta + bprime + theta_hat[0] * beta * beta,
            axis=1,
        )
        jacobian = np.mean(beta * beta)
        influence = -contribution / jacobian
    return influence - influence.mean()


def _adjusted_gram_md(X, sigma, theta_hat, h, est, h1, h2):
    """Centered influence-adjusted Stein-feature Gram matrix."""
    n, _ = X.shape
    kernel, stein = ksd_gram_md(X, sigma, theta_hat[0], h)
    score = (
        -X / sigma**2
        + theta_hat[0] * (-X * np.exp(-X**2 / 2))
    )
    derivative = -X * np.exp(-X**2 / 2)
    diff = X[:, None, :] - X[None, :, :]
    mixed = (
        score[:, None, :] - diff / (h * h)
    ) * kernel[:, :, None]
    cross = np.einsum(
        "ijq,jq->i", mixed, derivative, optimize=True
    ) / n
    derivative_gram = np.einsum(
        "iq,ij,jq->", derivative, kernel, derivative, optimize=True
    ) / (n * n)
    influence = _nuisance_influence_md(
        X, sigma, theta_hat, est, h1, h2
    )
    adjusted = (
        stein
        + np.outer(cross, influence)
        + np.outer(influence, cross)
        + derivative_gram * np.outer(influence, influence)
    )
    row_mean = adjusted.mean(axis=1, keepdims=True)
    return adjusted - row_mean - row_mean.T + adjusted.mean()


def _brueck_statistics_md(
    X,
    sigma,
    theta_hat,
    h0,
    h1,
    h2,
    counts,
    est,
    correction_mode="published",
):
    """Return Brueck-style empirical U-bootstrap statistics.

    ``published`` reproduces the endpoint-gradient expression printed by
    Brueck et al.  ``midpoint`` is our factor-corrected diagnostic: averaging
    the bootstrap gradient at the original and bootstrap nuisance estimates
    restores the conventional one-half coefficient on the quadratic Taylor
    term.  The latter is not an author-confirmed erratum.
    """
    if correction_mode not in {"published", "midpoint"}:
        raise ValueError(
            "correction_mode must be either 'published' or 'midpoint'"
        )
    n = X.shape[0]
    centered_components = []
    for component in (h0, h1, h2):
        row_mean = component.mean(axis=1, keepdims=True)
        centered_components.append(
            component - row_mean - row_mean.T + component.mean()
        )
    component_u = weighted_u_components_md(
        np.stack((*centered_components, h1, h2)), counts
    )
    if est is min_ksd_p1_md:
        theta_b = -component_u[:, 3] / (2.0 * component_u[:, 4])
    else:
        theta_b = _refit_counts_md(X, sigma, est, counts, h1, h2)
    base = (
        component_u[:, 0]
        + theta_b * component_u[:, 1]
        + theta_b**2 * component_u[:, 2]
    )
    gradient_h1_boot_u = component_u[:, 3]
    gradient_h2_boot_u = component_u[:, 4]
    gradient_boot_u = (
        gradient_h1_boot_u + 2.0 * theta_b * gradient_h2_boot_u
    )
    gradient_original_u = weighted_u_md(
        h1 + 2.0 * theta_hat[0] * h2, np.ones(n)
    )
    if correction_mode == "published":
        correction_gradient = gradient_boot_u - gradient_original_u
    else:
        gradient_hat_boot_u = (
            gradient_h1_boot_u
            + 2.0 * theta_hat[0] * gradient_h2_boot_u
        )
        correction_gradient = (
            0.5 * (gradient_boot_u + gradient_hat_boot_u)
            - gradient_original_u
        )
    correction = (
        theta_b - theta_hat[0]
    ) * correction_gradient
    return base + correction


def empirical_ksd_pvalues_md(
    X,
    sigma,
    theta_hat,
    h,
    B,
    est,
    seed,
    brueck_correction="published",
):
    """Return p-values and standalone testing times (excluding the caller's fit).

    Shared Gram/count preparation is measured once and charged in full to
    each method, as it would be needed by either test run on its own.
    """
    shared_started = time.perf_counter()
    n = X.shape[0]
    h0, h1, h2 = _ksd_quadratic_grams_md(X, sigma, h)
    stein_hat = (
        h0 + theta_hat[0] * h1 + theta_hat[0] ** 2 * h2
    )
    observed_v = float(stein_hat.mean())
    observed_u = weighted_u_md(stein_hat, np.ones(n))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B)
    weights = rng.standard_normal((B, n))
    weights -= weights.mean(axis=1, keepdims=True)
    shared_seconds = time.perf_counter() - shared_started

    started = time.perf_counter()
    adjusted = _adjusted_gram_md(
        X, sigma, theta_hat, h, est, h1, h2
    )
    influence_stats = np.einsum(
        "bi,ij,bj->b", weights, adjusted, weights, optimize=True
    ) / (n * n)
    influence_pvalue = float((1 + np.sum(influence_stats >= observed_v)) / (B + 1))
    influence_time = shared_seconds + time.perf_counter() - started

    started = time.perf_counter()
    naive_stats = np.einsum(
        "bi,ij,bj->b", weights, stein_hat, weights, optimize=True
    ) / (n * n)
    naive_pvalue = float((1 + np.sum(naive_stats >= observed_v)) / (B + 1))
    naive_time = shared_seconds + time.perf_counter() - started

    started = time.perf_counter()
    brueck_stats = _brueck_statistics_md(
        X,
        sigma,
        theta_hat,
        h0,
        h1,
        h2,
        counts,
        est,
        correction_mode=brueck_correction,
    )
    brueck_pvalue = float((1 + np.sum(brueck_stats >= observed_u)) / (B + 1))
    brueck_time = shared_seconds + time.perf_counter() - started
    return (
        influence_pvalue,
        brueck_pvalue,
        naive_pvalue,
        influence_time,
        brueck_time,
        naive_time,
    )


# ---------- d-dim MMD V-stat ----------
def mmd2_md(X, Y, h):
    inv = 0.5 / (h * h)
    XX = (X * X).sum(axis=1); YY = (Y * Y).sum(axis=1)
    dxx = np.maximum(XX[:, None] + XX[None, :] - 2 * X @ X.T,  0)
    dyy = np.maximum(YY[:, None] + YY[None, :] - 2 * Y @ Y.T,  0)
    dxy = np.maximum(XX[:, None] + YY[None, :] - 2 * X @ Y.T,  0)
    return (np.exp(-dxx * inv).mean()
          + np.exp(-dyy * inv).mean()
          - 2 * np.exp(-dxy * inv).mean())


# ---------- parametric-bootstrap KSD test (est-agnostic) ----------
def ksd_test_md(X, sigma, theta_hat, h, B, est, update=None):
    """est: callable(Xb, h, sigma) -> theta_b (length-1). Used to refit in each bootstrap."""
    n, d = X.shape
    T_obs = ksd_stat_md(X, sigma, theta_hat[0], h)
    Tb = np.empty(B)
    for b in range(B):
        Xb = rejection_sampler_md(n, sigma, theta_hat, d)
        tb = update(Xb) if update is not None else est(Xb, h, sigma)
        Tb[b] = ksd_stat_md(Xb, sigma, tb[0], h)
    return (1 + np.sum(Tb >= T_obs)) / (B + 1)


# ---------- 1-D KEF cumulative distribution function (numeric) ----------
def _kef_cdf_1d(theta0, sigma, n_grid=4001, x_max_sd=8.0):
    """Grid + CDF for p_theta(z) ~ N(0, sigma^2)(z) * exp(theta0 * phi_0(z)),
    phi_0(z) = exp(-z^2/2). Returns (z_grid, cdf_grid)."""
    z = np.linspace(-x_max_sd * sigma, x_max_sd * sigma, n_grid)
    q_z = np.exp(-z * z / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))
    log_factor = theta0 * np.exp(-z * z / 2.0)
    rho = q_z * np.exp(log_factor - log_factor.max())  # stabilise the exp
    rho = rho / _trapezoid(rho, z)                      # normalise on the grid
    dz = z[1] - z[0]
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * dz)))
    cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
    return z, cdf


def _peacock_stat_md(X, F_X):
    """Peacock (1983) multivariate KS statistic.
    For each anchor a in 1..n and each orthant eps in {-1,+1}^d centred on X[a],
    compare the empirical orthant mass to the null orthant mass under the
    *factorised* CDF F (the latter being a product of 1-D marginal CDF values).
    Returns max_a max_eps |F_emp(a,eps) - F_null(a,eps)|.

    X:   (n, d) data
    F_X: (n, d) with F_X[i, k] = F_k(X[i, k]), the marginal null CDF values.
    """
    n, d = X.shape
    n_orth = 1 << d
    bit_pos = np.arange(d)
    powers  = (1 << bit_pos)                              # (d,)
    # pattern_bits[p, k] = 1 iff orthant p has eps_k = +1 (i.e. y_k > X[a,k])
    pattern_bits = ((np.arange(n_orth)[:, None] >> bit_pos[None, :]) & 1).astype(np.float64)

    D_max = 0.0
    for a in range(n):
        # bit_k(i) = (X[i,k] > X[a,k]); ties (incl. i=a) get bit 0
        bits = (X > X[a]).astype(np.int64)               # (n, d)
        idx  = (bits * powers).sum(axis=1)               # (n,)
        counts = np.bincount(idx, minlength=n_orth).astype(np.float64)
        counts[0] -= 1.0                                  # remove anchor self-count
        emp  = counts / n
        # null orthant masses for this anchor: prod_k (FXa[k] if eps_k=-1 else 1-FXa[k])
        FXa  = F_X[a]
        null = np.prod(pattern_bits * (1.0 - FXa) + (1.0 - pattern_bits) * FXa, axis=1)
        diff = np.max(np.abs(emp - null))
        if diff > D_max:
            D_max = diff
    return D_max


# ---------- parametric-bootstrap Peacock KS test (est-agnostic, any d >= 1) ----------
def ks_test_md(X, sigma, theta_hat, h, B, est, update=None):
    """Multivariate Peacock KS goodness-of-fit with parametric bootstrap.
    In d=1 this reduces to the usual KS statistic up to the 1/n tie convention.
    The factorised null CDF F(x) = prod_k F_k(x_k) is evaluated via the shared
    1-D KEF CDF grid (factorised model: same marginal in every coordinate).
    """
    n, d = X.shape
    z, cdf = _kef_cdf_1d(theta_hat[0], sigma)
    F_X = np.interp(X.ravel(), z, cdf).reshape(n, d)
    D_obs = _peacock_stat_md(X, F_X)
    Db = np.empty(B)
    for b in range(B):
        Xb = rejection_sampler_md(n, sigma, theta_hat, d)
        tb = update(Xb) if update is not None else est(Xb, h, sigma)
        zb, cdfb = _kef_cdf_1d(tb[0], sigma)
        F_Xb = np.interp(Xb.ravel(), zb, cdfb).reshape(n, d)
        Db[b] = _peacock_stat_md(Xb, F_Xb)
    return (1 + np.sum(Db >= D_obs)) / (B + 1)


# ---------- parametric-bootstrap MMD test (est-agnostic) ----------
def mmd_test_md(X, sigma, theta_hat, h_fit, B, est, update=None):
    """est: callable(Xb, h, sigma) -> theta_b. h_fit only used inside est (and ignored by SM)."""
    n, d = X.shape
    Y = rejection_sampler_md(n, sigma, theta_hat, d)
    T_obs = mmd2_md(X, Y, mmd_bw_md(np.vstack([X, Y])))
    Tb = np.empty(B)
    for b in range(B):
        Xb = rejection_sampler_md(n, sigma, theta_hat, d)
        tb = update(Xb) if update is not None else est(Xb, h_fit, sigma)
        Yb = rejection_sampler_md(n, sigma, tb, d)
        Tb[b] = mmd2_md(Xb, Yb, mmd_bw_md(np.vstack([Xb, Yb])))
    return (1 + np.sum(Tb >= T_obs)) / (B + 1)

def _cell_seed(
    seed: int,
    dimension: int,
    signal_index: int,
    replication: int,
    stream: int,
) -> int:
    return int(
        np.random.SeedSequence(
            [seed, dimension, signal_index, replication, stream]
        ).generate_state(1)[0]
    )


def run_replication(
    *,
    n: int,
    dimension: int,
    theta1: float,
    theta2: float,
    bootstrap_draws: int,
    seed: int,
    signal_index: int,
    replication: int,
    sigma: float = 3.0,
) -> dict[str, dict[str, float]]:
    """Run one paired five-method KEF replication.

    Observed-data generation is excluded from the timings. The observed
    bandwidth and minimum-KSD fit are measured once and charged in full to
    every method, matching the historical producer's standalone-test timing
    convention. ``theta2`` is the dimension-scaled model coefficient, while
    ``signal_index`` and ``replication`` are zero-based seed coordinates. The
    legacy global NumPy state is restored before returning.
    """

    random_state = np.random.get_state()
    try:
        theta = np.array([theta1, theta2])

        np.random.seed(
            _cell_seed(seed, dimension, signal_index, replication, 0)
        )
        data = rejection_sampler_md(n, sigma, theta, dimension)

        started = time.perf_counter()
        bandwidth = ksd_bw_md(data)
        bandwidth_seconds = time.perf_counter() - started
        started = time.perf_counter()
        theta_hat = min_ksd_p1_md(data, bandwidth, sigma)
        observed_setup_seconds = (
            bandwidth_seconds + time.perf_counter() - started
        )

        started = time.perf_counter()
        np.random.seed(
            _cell_seed(seed, dimension, signal_index, replication, 10)
        )
        p_ksd = ksd_test_md(
            data,
            sigma,
            theta_hat,
            bandwidth,
            bootstrap_draws,
            est=min_ksd_p1_md,
            update=None,
        )
        ksd_seconds = observed_setup_seconds + time.perf_counter() - started

        (
            p_influence,
            p_brueck,
            p_naive,
            influence_seconds,
            brueck_seconds,
            naive_seconds,
        ) = empirical_ksd_pvalues_md(
            data,
            sigma,
            theta_hat,
            bandwidth,
            bootstrap_draws,
            min_ksd_p1_md,
            _cell_seed(
                seed, dimension, signal_index, replication, 16
            ),
            brueck_correction="midpoint",
        )

        started = time.perf_counter()
        np.random.seed(
            _cell_seed(seed, dimension, signal_index, replication, 12)
        )
        p_mmd = mmd_test_md(
            data,
            sigma,
            theta_hat,
            bandwidth,
            bootstrap_draws,
            est=min_ksd_p1_md,
            update=None,
        )
        mmd_seconds = observed_setup_seconds + time.perf_counter() - started

        started = time.perf_counter()
        np.random.seed(
            _cell_seed(seed, dimension, signal_index, replication, 14)
        )
        p_ks = ks_test_md(
            data,
            sigma,
            theta_hat,
            bandwidth,
            bootstrap_draws,
            est=min_ksd_p1_md,
            update=None,
        )
        ks_seconds = observed_setup_seconds + time.perf_counter() - started

        return {
            "SKSD-parametric": {
                "p_value": float(p_ksd),
                "seconds": float(ksd_seconds),
            },
            "SKSD-influence": {
                "p_value": float(p_influence),
                "seconds": float(
                    observed_setup_seconds + influence_seconds
                ),
            },
            "Brueck-corrected": {
                "p_value": float(p_brueck),
                "seconds": float(observed_setup_seconds + brueck_seconds),
            },
            "SMMD": {
                "p_value": float(p_mmd),
                "seconds": float(mmd_seconds),
            },
            "SKS": {
                "p_value": float(p_ks),
                "seconds": float(ks_seconds),
            },
            "SKSD-naive-wild": {
                "p_value": float(p_naive),
                "seconds": float(observed_setup_seconds + naive_seconds),
            },
        }
    finally:
        np.random.set_state(random_state)
