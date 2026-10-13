"""NumPy-only sampling, estimation, and Stein kernels on the unit sphere.

The Watson density is proportional to exp(kappa * (mu.T @ x)**2) relative
to surface area on S^2.  Its axis is unsigned.  Fits may either estimate the
axis and positive concentration jointly, or hold an axis fixed and estimate
an unrestricted real concentration.  Influence functions describe the
selected fitting functional under the empirical law.

The spherical Stein operator is

    A_p f(x) = tr((I - xx.T) Df(x)) + (s_p(x) - 2x).T f(x).

Both Gram matrices below are uncentered.  Bootstrap code should perform any
desired empirical centering after constructing the adjusted Gram matrix.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WatsonFit:
    """A Watson axis, concentration, and n-by-4 influence.

    Influence columns are the three ambient axis coordinates followed by
    concentration.  The first three columns are tangent to ``mu`` and are
    zero when ``fixed_axis`` is true.  Fixed-axis concentrations may be signed.
    """

    mu: np.ndarray
    kappa: float
    influence: np.ndarray
    fixed_axis: bool = False


def _finite_scalar(value, name):
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


def _sample_inputs(rng, n):
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be a nonnegative integer")
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    return int(n)


def _axis(mu):
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (3,) or not np.all(np.isfinite(mu)):
        raise ValueError("mu must be a finite vector of length three")
    if not np.isclose(mu @ mu, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("mu must be a unit vector")
    return mu


def _points(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] == 0:
        raise ValueError("x must have shape (n, 3), with n positive")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if not np.allclose(np.sum(x * x, axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("every observation must lie on the unit sphere")
    return x


def _rejection_sphere(rng, n, log_acceptance):
    """Exact rejection sampling with the uniform surface law as proposal."""
    result = np.empty((n, 3), dtype=float)
    filled = 0
    proposed = 0
    accepted = 0
    rate = 0.125
    while filled < n:
        remaining = n - filled
        batch_size = min(262144, max(256, math.ceil(1.1 * remaining / rate)))
        proposal = rng.normal(size=(batch_size, 3))
        proposal /= np.linalg.norm(proposal, axis=1)[:, None]
        keep = rng.random(batch_size) < np.exp(log_acceptance(proposal))
        batch = proposal[keep]
        take = min(remaining, len(batch))
        result[filled : filled + take] = batch[:take]
        filled += take
        proposed += batch_size
        accepted += len(batch)
        # This changes only proposal batch sizes, never acceptance probabilities.
        rate = (accepted + 1.0) / (proposed + 8.0)
    return result


def sample_bingham(rng, n, kappa, r, matched_odi=False):
    """Draw iid Bingham directions with density exp(kappa*(x3^2+r*x1^2)).

    ``kappa`` is positive and ``0 <= r < 1``.  When ``matched_odi`` is true,
    the actual concentration is kappa/sqrt(1-r), preserving the product of
    transverse concentrations and Tariq et al.'s total-dispersion index.
    The output is an (n, 3) array; n may be zero.
    """
    n = _sample_inputs(rng, n)
    kappa = _finite_scalar(kappa, "kappa")
    r = _finite_scalar(r, "r")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if not 0.0 <= r < 1.0:
        raise ValueError("r must satisfy 0 <= r < 1")
    if not isinstance(matched_odi, (bool, np.bool_)):
        raise ValueError("matched_odi must be boolean")
    if matched_odi:
        kappa /= math.sqrt(1.0 - r)
    if not math.isfinite(kappa):
        raise ValueError("the adjusted concentration is not finite")
    # Subtracting the maximum exponent kappa gives this nonpositive form.
    return _rejection_sphere(
        rng, n, lambda y: -kappa * ((1.0 - r) * y[:, 0] ** 2 + y[:, 1] ** 2)
    )


def sample_watson(rng, n, mu, kappa):
    """Draw iid Watson directions for any finite real concentration.

    The absolute axial coordinate t has density proportional to exp(kappa*t²)
    on [0,1].  For positive kappa, rejection uses a truncated increasing
    exponential proposal; for negative kappa it uses a uniform or truncated
    half-normal proposal.  An independent sign and uniform azimuth complete
    the exact surface draw.  This axial sampler replaces the former uniform-
    sphere rejection sampler, so seeded positive-kappa draws have changed.
    """
    n = _sample_inputs(rng, n)
    mu = _axis(mu)
    kappa = _finite_scalar(kappa, "kappa")
    axial = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        batch_size = min(262144, max(64, 2 * (n - filled)))
        if kappa < -1.0:
            scale = (1.0 / math.sqrt(-kappa)) / math.sqrt(2.0)
            proposal = np.abs(rng.normal(size=batch_size)) * scale
            accepted = proposal[proposal <= 1.0]
        elif kappa > 1e-6:
            # Inverse CDF of exp(kappa*t) on [0,1], written without exp(kappa).
            proposal = 1.0 + np.log1p(
                -rng.random(batch_size) * (-math.expm1(-kappa))
            ) / kappa
            log_acceptance = kappa * proposal * (proposal - 1.0)
            accepted = proposal[rng.random(batch_size) < np.exp(log_acceptance)]
        else:
            proposal = rng.random(batch_size)
            log_acceptance = kappa * proposal**2 - max(kappa, 0.0)
            accepted = proposal[rng.random(batch_size) < np.exp(log_acceptance)]
        take = min(n - filled, len(accepted))
        axial[filled : filled + take] = accepted[:take]
        filled += take

    axial *= 2 * rng.integers(0, 2, size=n) - 1
    azimuth = rng.uniform(0.0, 2.0 * math.pi, size=n)
    reference = np.eye(3)[np.argmin(np.abs(mu))]
    first = np.cross(mu, reference)
    first /= np.linalg.norm(first)
    second = np.cross(mu, first)
    radius = np.sqrt(np.maximum(0.0, 1.0 - axial**2))
    return axial[:, None] * mu + radius[:, None] * (
        np.cos(azimuth)[:, None] * first + np.sin(azimuth)[:, None] * second
    )


def sample_crossing(rng, n, kappa, epsilon, angle_degrees=60):
    """Draw a mixture of Watson axes e3 and (sin(angle), 0, cos(angle)).

    Each observation independently comes from the second component with
    probability ``epsilon``; both components have concentration ``kappa``.
    """
    n = _sample_inputs(rng, n)
    kappa = _finite_scalar(kappa, "kappa")
    epsilon = _finite_scalar(epsilon, "epsilon")
    angle_degrees = _finite_scalar(angle_degrees, "angle_degrees")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between zero and one")
    if not 0.0 <= angle_degrees <= 180.0:
        raise ValueError("angle_degrees must be between zero and 180")
    angle = math.radians(angle_degrees)
    first_axis = np.array([0.0, 0.0, 1.0])
    second_axis = np.array([math.sin(angle), 0.0, math.cos(angle)])
    second = rng.random(n) < epsilon
    result = np.empty((n, 3), dtype=float)
    result[~second] = sample_watson(rng, int(np.sum(~second)), first_axis, kappa)
    result[second] = sample_watson(rng, int(np.sum(second)), second_axis, kappa)
    return result


def fit_watson(x, weights=None, fixed_axis=None):
    """Fit a Watson concentration with an estimated or supplied fixed axis.

    For u_i=(mu.T x_i)^2, kappa=(3 E_n u-1)/(2 E_n[u(1-u)]).
    Optional nonnegative weights specify an empirical probability law after
    normalization.  Returned influence values are contamination derivatives
    at that law, evaluated at each row of x; they are not multiplied by the
    observation weights.

    A supplied unit ``fixed_axis`` skips eigenanalysis, allows signed kappa,
    and gives three zero axis-influence columns.  Otherwise the fitted axis
    is the leading scatter eigenvector and kappa must be positive.

    Under arbitrary Q, the estimated-axis influence uses both eigenvalue gaps.
    The concentration influence additionally includes the derivative of its
    moment ratio through the estimated axis.  No estimates or gaps are
    clipped; unresolved eigenspaces and invalid concentration fits raise.
    """
    x = _points(x)
    n = len(x)
    if weights is None:
        probability = np.full(n, 1.0 / n)
    else:
        probability = np.asarray(weights, dtype=float)
        if probability.shape != (n,) or not np.all(np.isfinite(probability)):
            raise ValueError("weights must be a finite vector with one entry per row")
        if np.any(probability < 0.0):
            raise ValueError("weights must be nonnegative")
        total = np.sum(probability)
        if not math.isfinite(float(total)) or total <= 0.0:
            raise ValueError("weights must have a finite, positive sum")
        probability = probability / total

    axis_is_fixed = fixed_axis is not None
    if axis_is_fixed:
        mu = _axis(fixed_axis).copy()
    else:
        scatter = x.T @ (probability[:, None] * x)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        gaps = eigenvalues[2] - eigenvalues[:2]
        eigen_tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(eigenvalues[2]))
        if np.any(gaps <= eigen_tolerance):
            raise ValueError("the leading scatter eigenvalue is not numerically separated")
        mu = eigenvectors[:, 2].copy()
        if mu[np.argmax(np.abs(mu))] < 0.0:
            mu = -mu
    t = x @ mu
    u = t * t
    denominator = 2.0 * np.dot(probability, u * (1.0 - u))
    if not math.isfinite(float(denominator)) or denominator <= 0.0:
        raise ValueError("the score-matching concentration denominator is not positive")
    kappa = float((3.0 * np.dot(probability, u) - 1.0) / denominator)
    if not math.isfinite(kappa):
        raise ValueError("the fitted concentration must be finite")
    if not axis_is_fixed and kappa <= 0.0:
        raise ValueError("the fitted concentration must be finite and positive")

    concentration_direct = (3.0 * u - 1.0 - 2.0 * kappa * u * (1.0 - u)) / denominator
    if axis_is_fixed:
        axis_influence = np.zeros((n, 3))
        concentration_influence = concentration_direct
    else:
        other_axes = eigenvectors[:, :2]
        axis_influence = (t[:, None] * (x @ other_axes) / gaps) @ other_axes.T
        # Gradient of N(mu)/D(mu), evaluated at the empirical law.  Its tangent
        # part generally does not vanish outside the rotationally symmetric null.
        gradient_weight = probability * t * (6.0 - 4.0 * kappa * (1.0 - 2.0 * u))
        concentration_axis_gradient = (gradient_weight @ x) / denominator
        concentration_influence = concentration_direct + axis_influence @ concentration_axis_gradient
    influence = np.column_stack((axis_influence, concentration_influence))
    if not np.all(np.isfinite(influence)):
        raise ValueError("the fitted influence function is not finite")
    return WatsonFit(mu=mu, kappa=kappa, influence=influence, fixed_axis=axis_is_fixed)


def _kernel_terms(x, fit, bandwidth):
    x = _points(x)
    if not isinstance(fit, WatsonFit):
        raise TypeError("fit must be a WatsonFit")
    mu = _axis(fit.mu)
    kappa = _finite_scalar(fit.kappa, "fit.kappa")
    bandwidth = _finite_scalar(bandwidth, "bandwidth")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    inverse_bandwidth = 1.0 / bandwidth
    inverse_h2 = inverse_bandwidth * inverse_bandwidth
    if not math.isfinite(inverse_h2):
        raise ValueError("the inverse squared bandwidth must be finite")
    dot = x @ x.T
    kernel = np.exp((dot - 1.0) * inverse_h2)
    axial = x @ mu
    projected_mu = mu - axial[:, None] * x
    b = 2.0 * kappa * axial[:, None] * projected_mu - 2.0 * x
    return x, mu, kappa, inverse_h2, dot, kernel, axial, projected_mu, b


def _stein_from_terms(x, inverse_h2, dot, kernel, b):
    b_dot_x = b @ x.T
    b_dot_self = np.sum(b * x, axis=1)
    score_gradient = b_dot_self[:, None] + b_dot_self[None, :] - dot * (b_dot_x + b_dot_x.T)
    mixed_gradient = inverse_h2 * (1.0 + dot * dot) - inverse_h2**2 * dot * (1.0 - dot * dot)
    gram = kernel * (b @ b.T + inverse_h2 * score_gradient + mixed_gradient)
    return (gram + gram.T) / 2.0


def stein_gram(x, fit, bandwidth=1.0):
    """Return the spherical Stein Gram matrix for k=exp((x.T y-1)/h^2).

    The feature is xi(x)=k(x,.)[s(x)-2x]+(I-xx.T) grad_x k(x,.).
    This includes the curvature correction and both tangential projections
    in the mixed kernel derivative.  The returned matrix is uncentered.
    """
    terms = _kernel_terms(x, fit, bandwidth)
    x, _, _, inverse_h2, dot, kernel, _, _, b = terms
    gram = _stein_from_terms(x, inverse_h2, dot, kernel, b)
    if not np.all(np.isfinite(gram)):
        raise ValueError("the Stein Gram matrix is not finite")
    return gram


def adjusted_gram(x, fit, bandwidth=1.0):
    """Return the uncentered influence-adjusted feature Gram matrix.

    With theta=(mu_1,mu_2,mu_3,kappa), let
    A_a=n^{-1} sum_j D_a xi_theta(x_j), and psi_i=xi_theta(x_i)+A IF_i.
    This function computes <psi_i,psi_j> by four-dimensional contractions.
    The derivative D_a xi(x)=k(x,.) D_a s(x) is analytic; axis influence
    directions are tangent to mu, so ambient derivatives suffice.

    The averaging in A is the equally weighted empirical law on x.  The fit
    and influence should therefore describe that law when this matrix is
    used for bootstrap calibration.
    """
    terms = _kernel_terms(x, fit, bandwidth)
    x, mu, kappa, inverse_h2, dot, kernel, axial, projected_mu, b = terms
    n = len(x)
    influence = np.asarray(fit.influence, dtype=float)
    if influence.shape != (n, 4) or not np.all(np.isfinite(influence)):
        raise ValueError("fit.influence must be a finite (n, 4) matrix")
    if not np.allclose(influence[:, :3] @ mu, 0.0, rtol=0.0, atol=1e-8):
        raise ValueError("axis influence directions must be tangent to fit.mu")

    if fit.fixed_axis:
        if np.any(influence[:, :3] != 0.0):
            raise ValueError("fixed-axis fits must have zero axis influence")
        derivative = 2.0 * axial[:, None] * projected_mu
        kernel_derivative = kernel @ derivative
        dot_kernel_derivative = (kernel * dot) @ derivative
        radial_derivative = np.sum(x * derivative, axis=1)
        cross = (
            np.sum(b * kernel_derivative, axis=1)
            + inverse_h2 * (
                kernel @ radial_derivative - np.sum(x * dot_kernel_derivative, axis=1)
            )
        ) / n
        derivative_gram = np.sum(derivative * kernel_derivative) / (n * n)
        concentration_influence = influence[:, 3]
        cross_influence = np.outer(cross, concentration_influence)
        gram = (
            _stein_from_terms(x, inverse_h2, dot, kernel, b)
            + cross_influence + cross_influence.T
            + derivative_gram * np.outer(concentration_influence, concentration_influence)
        )
        if not np.all(np.isfinite(gram)):
            raise ValueError("the adjusted Gram matrix is not finite")
        return (gram + gram.T) / 2.0

    projection = np.eye(3)[None, :, :] - x[:, :, None] * x[:, None, :]
    derivative = np.empty((n, 3, 4), dtype=float)
    derivative[:, :, :3] = 2.0 * kappa * (
        projected_mu[:, :, None] * x[:, None, :] + axial[:, None, None] * projection
    )
    derivative[:, :, 3] = 2.0 * axial[:, None] * projected_mu

    kernel_derivative = (kernel @ derivative.reshape(n, 12)).reshape(n, 3, 4)
    dot_kernel_derivative = ((kernel * dot) @ derivative.reshape(n, 12)).reshape(n, 3, 4)
    radial_derivative = np.einsum("ic,ica->ia", x, derivative)
    cross = (
        np.einsum("ic,ica->ia", b, kernel_derivative)
        + inverse_h2 * (kernel @ radial_derivative - np.einsum("ic,ica->ia", x, dot_kernel_derivative))
    ) / n
    derivative_gram = np.einsum("ica,icb->ab", derivative, kernel_derivative) / (n * n)
    gram = _stein_from_terms(x, inverse_h2, dot, kernel, b)
    cross_influence = cross @ influence.T
    gram = gram + cross_influence + cross_influence.T + influence @ derivative_gram @ influence.T
    if not np.all(np.isfinite(gram)):
        raise ValueError("the adjusted Gram matrix is not finite")
    return (gram + gram.T) / 2.0
