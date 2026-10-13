"""The repository's corrected Brueck empirical U-bootstrap on S^2.

This follows ``paired_empirical_tests`` in
``code/rbm-python/experiments/gaussian_bernoulli_influence_vs_brueck.py``:
count-weighted refitting, centering under the original empirical law, and
the factor-corrected midpoint gradient term.  The correction is the
project's implementation convention, not an author-confirmed erratum.

Every observed and bootstrap fit is a Watson fit.  Derivatives use the
sign-invariant, trace-free natural matrix A=kappa*(mu mu.T-I/3), for which
s_A(x)=2*(I-xx.T)*A*x.  The smooth symmetric-matrix extension is used only
to take derivatives; it does not replace the Watson fit by a Bingham fit.
The finite-sample correction is explicitly defined in these coordinates,
and is not asserted to be invariant under nonlinear reparameterization.
"""

import math

import numpy as np

from _sphere_core import WatsonFit, fit_watson, stein_gram


BRUECK_CORRECTION_VERSION = "factor-corrected-midpoint-v1"
BRUECK_PARAMETERIZATION = "trace-free-natural-quadratic-matrix"


def _u_unchecked(matrix, counts):
    sample_size = float(np.sum(counts))
    total = counts @ matrix @ counts - counts @ np.diag(matrix)
    return float(total / (sample_size * (sample_size - 1.0)))


def weighted_u(matrix, counts):
    """U-statistic of an explicitly replicated sample, without replication.

    If the original support has m rows and integer multiplicities ``counts``,
    the replicated sample size is N=sum(counts).  The result is

        (counts.T @ matrix @ counts - counts.T @ diag(matrix)) / (N*(N-1)).

    The diagonal subtraction removes only self-pairs of individual sampled
    observations.  Distinct bootstrap copies of the same original row still
    contribute their diagonal kernel values.  The signed result is not
    truncated to zero.
    """
    matrix = np.asarray(matrix, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if counts.shape != (matrix.shape[0],):
        raise ValueError("counts must have one entry per matrix row")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(counts)):
        raise ValueError("matrix and counts must contain only finite values")
    if np.any(counts < 0.0) or not np.all(counts == np.floor(counts)):
        raise ValueError("counts must be nonnegative integers")
    sample_size = float(np.sum(counts))
    if not math.isfinite(sample_size) or sample_size < 2.0:
        raise ValueError("the replicated sample must contain at least two observations")
    result = _u_unchecked(matrix, counts)
    if not math.isfinite(result):
        raise ValueError("the weighted U-statistic is not finite")
    return result


def _natural_matrix(fit):
    return fit.kappa * (np.outer(fit.mu, fit.mu) - np.eye(3) / 3.0)


def _center_original(matrix):
    row_mean = matrix.mean(axis=1, keepdims=True)
    return matrix - row_mean - row_mean.T + matrix.mean()


def _fixed_axis_bootstrap(x, fit, counts_all, bandwidth):
    """Evaluate all scalar refits through the exact quadratic kernel in kappa."""
    n = len(x)
    u = (x @ fit.mu) ** 2
    denominator = 2.0 * (counts_all @ (u * (1.0 - u))) / n
    invalid = ~np.isfinite(denominator) | (denominator <= 0.0)
    if np.any(invalid):
        index = int(np.flatnonzero(invalid)[0])
        raise ValueError(f"Watson bootstrap refit failed at draw {index}: concentration denominator is not positive")
    kappas = (3.0 * (counts_all @ u) / n - 1.0) / denominator
    if not np.all(np.isfinite(kappas)):
        index = int(np.flatnonzero(~np.isfinite(kappas))[0])
        raise ValueError(f"Watson bootstrap refit failed at draw {index}: fitted concentration is not finite")

    # s_kappa is linear, hence H(kappa)=H0+kappa*H1+kappa²*H2 exactly.
    grams = [
        stein_gram(x, WatsonFit(fit.mu, kappa, fit.influence, fixed_axis=True), bandwidth)
        for kappa in (0.0, 1.0, -1.0)
    ]
    coefficients = np.stack((grams[0], (grams[1] - grams[2]) / 2.0,
                             (grams[1] + grams[2]) / 2.0 - grams[0]))
    centered = np.stack([_center_original(coefficient) for coefficient in coefficients])

    def count_u(matrices):
        pairs = np.stack([np.sum((counts_all @ matrix) * counts_all, axis=1)
                          for matrix in matrices], axis=1)
        self_pairs = counts_all @ np.diagonal(matrices, axis1=1, axis2=2).T
        return (pairs - self_pairs) / (n * (n - 1))

    powers = np.column_stack((np.ones(len(kappas)), kappas, kappas**2))
    original_powers = np.array([1.0, fit.kappa, fit.kappa**2])
    bootstrap_u = count_u(coefficients)
    base = np.sum(count_u(centered) * powers, axis=1)
    original_u = (
        coefficients.sum(axis=(1, 2))
        - np.trace(coefficients, axis1=1, axis2=2)
    ) / (n * (n - 1))
    original_derivative = original_u[1] + 2.0 * fit.kappa * original_u[2]
    correction = (
        np.sum(bootstrap_u * (powers - original_powers), axis=1)
        - (kappas - fit.kappa) * original_derivative
    )
    return n * (base + correction)


def brueck_statistics(x, fit, counts_all, bandwidth=1.0):
    """Return observed n*U and corrected empirical-bootstrap n*U draws.

    ``counts_all`` has shape (B,n); each row contains nonnegative integer
    Efron multiplicities summing to n.  Every draw refits the same Watson
    estimator using these counts, preserving a supplied fixed axis.  For a
    fixed axis the kernel is quadratic in the scalar concentration, allowing
    all signed concentration refits and count contractions to be batched.
    Bootstrap degeneracy raises with the draw
    index; no fitted parameters, counts, or statistics are clipped.

    Put H_hat=H(A_hat), H_b=H(A_b), delta=A_b-A_hat and C=I-11.T/n.
    Before multiplication by n, bootstrap draw b is

      U(C H_b C, c_b)
        + <delta, .5*(grad U(H_b,c_b)+grad U(H_hat,c_b))
                          - grad U(H_hat,1)>.

    Centering uses the original equally weighted support, not c_b.  Since
    the spherical score is linear in A, its Stein Gram matrix is quadratic
    in A, and the full correction simplifies exactly to

        U(H_b,c_b) - U(H_hat,c_b) - D_delta U(H_hat,1).

    This identity avoids a parameter-gradient tensor.  The remaining
    directional derivative is evaluated analytically, with the original
    sample's off-diagonal pair weights.  Both outputs retain their signs.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2:
        raise ValueError("x must have shape (n,3), with n at least two")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if not isinstance(fit, WatsonFit):
        raise TypeError("fit must be a WatsonFit")
    if not np.ndim(fit.kappa) == 0 or not math.isfinite(float(fit.kappa)):
        raise ValueError("the observed Watson concentration must be finite")
    if not fit.fixed_axis and fit.kappa <= 0.0:
        raise ValueError("the observed joint Watson concentration must be positive")
    n = len(x)
    counts_all = np.asarray(counts_all, dtype=float)
    if counts_all.ndim != 2 or counts_all.shape[1] != n or len(counts_all) == 0:
        raise ValueError("counts_all must have shape (B,n), with B positive")
    if not np.all(np.isfinite(counts_all)):
        raise ValueError("counts_all must contain only finite values")
    if np.any(counts_all < 0.0) or not np.all(counts_all == np.floor(counts_all)):
        raise ValueError("counts_all must contain nonnegative integers")
    if not np.all(np.sum(counts_all, axis=1) == n):
        raise ValueError("every Efron count vector must sum to n")

    # The core validates unit-sphere observations, the fitted axis, and h.
    stein_hat = stein_gram(x, fit, bandwidth=bandwidth)
    ones = np.ones(n)
    observed = n * _u_unchecked(stein_hat, ones)
    if fit.fixed_axis:
        bootstrap = _fixed_axis_bootstrap(x, fit, counts_all, bandwidth)
        if not math.isfinite(observed) or not np.all(np.isfinite(bootstrap)):
            raise ValueError("the Brueck statistics are not finite")
        return float(observed), bootstrap
    matrix_hat = _natural_matrix(fit)

    inverse_h2 = (1.0 / float(bandwidth)) ** 2
    dot = x @ x.T
    kernel = np.exp((dot - 1.0) * inverse_h2)
    original_pair_kernel = kernel.copy()
    np.fill_diagonal(original_pair_kernel, 0.0)
    axial = x @ fit.mu
    score_hat = 2.0 * fit.kappa * axial[:, None] * (fit.mu - axial[:, None] * x)
    b_hat = score_hat - 2.0 * x
    # Sum_j w_ij*k_ij*[b_hat(x_j) + h^-2*P_(x_j)*x_i],
    # with w_ij=1 for i!=j and zero otherwise.
    original_common = original_pair_kernel @ b_hat + inverse_h2 * (
        original_pair_kernel.sum(axis=1)[:, None] * x
        - (original_pair_kernel * dot) @ x
    )

    bootstrap = np.empty(len(counts_all), dtype=float)
    for bootstrap_index, counts in enumerate(counts_all):
        try:
            refit = fit_watson(x, weights=counts, fixed_axis=fit.mu if fit.fixed_axis else None)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Watson bootstrap refit failed at draw {bootstrap_index}: {exc}") from exc
        stein_refit = stein_gram(x, refit, bandwidth=bandwidth)
        delta = _natural_matrix(refit) - matrix_hat
        delta_ax = x @ delta
        score_direction = 2.0 * (
            delta_ax - np.sum(delta_ax * x, axis=1)[:, None] * x
        )
        original_directional_u = 2.0 * np.sum(score_direction * original_common) / (n * (n - 1))
        midpoint_correction = (
            _u_unchecked(stein_refit, counts)
            - _u_unchecked(stein_hat, counts)
            - original_directional_u
        )
        base = _u_unchecked(_center_original(stein_refit), counts)
        bootstrap[bootstrap_index] = n * (base + midpoint_correction)

    if not math.isfinite(observed) or not np.all(np.isfinite(bootstrap)):
        raise ValueError("the Brueck statistics are not finite")
    return float(observed), bootstrap
