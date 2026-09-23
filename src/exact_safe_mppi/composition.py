"""Composite barriers: exact pointwise minimum vs. log-sum-exp soft minimum.

The two composition operators compared throughout the paper.  Both act on the top-level
cascade values beta_j = b_{j, d_j - 1}.
"""
import numpy as np

_ASSERT_TOL = 1e-9


def softmin_stable(beta, rho, axis=-1):
    """Numerically stabilised log-sum-exp soft minimum.

        softmin_rho(z) = -(1/rho) * log( sum_j exp(-rho * z_j) )
                       = m - (1/rho) * log( sum_j exp(-rho * (z_j - m)) ),   m = min_j z_j

    The stabilised form is mandatory here (see the paper's Remark on stabilisation): the naive
    form underflows for rho * spread >~ 700 and we are not interested in winning on an
    underflow artifact.  Note what the stabilised form actually does -- it computes the exact
    minimum first, and then spends `ell` exponentials and a logarithm blurring it.
    """
    beta = np.asarray(beta, dtype=float)
    m = beta.min(axis=axis, keepdims=True)
    s = np.exp(-rho * (beta - m)).sum(axis=axis, keepdims=True)
    out = m - np.log(s) / rho
    return np.squeeze(out, axis=axis)


def softmin_naive(beta, rho, axis=-1):
    """Unstabilised form. Provided ONLY so tests can demonstrate that we do not use it."""
    beta = np.asarray(beta, dtype=float)
    return -np.log(np.exp(-rho * beta).sum(axis=axis)) / rho


def exact_min(beta, axis=-1):
    """h_ex = min_j beta_j."""
    return np.asarray(beta, dtype=float).min(axis=axis)


def softmin_weights(beta, rho, axis=-1):
    """lambda_i = exp(-rho*(beta_i - h_rho)), the softmin's gradient weights (they sum to 1).

    As rho -> infinity these collapse onto a hard argmax: the soft minimum recovers exactly
    the non-smooth behaviour it was introduced to avoid, at transcendental cost.
    """
    beta = np.asarray(beta, dtype=float)
    m = beta.min(axis=axis, keepdims=True)
    e = np.exp(-rho * (beta - m))
    return e / e.sum(axis=axis, keepdims=True)


def erosion(beta, rho, axis=-1):
    """h_ex - h_rho >= 0, the inward warp of the soft minimum."""
    return exact_min(beta, axis=axis) - softmin_stable(beta, rho, axis=axis)


def active_set(beta, delta, axis=-1):
    """A_delta(x) = {j : beta_j <= h_ex + delta}  (Kamaldar Def. 10) as a boolean mask."""
    beta = np.asarray(beta, dtype=float)
    return beta <= beta.min(axis=axis, keepdims=True) + delta


def tie_multiplicity(beta, tol=0.0, axis=-1):
    """k = |A(x)|, the number of constraints attaining the minimum (within `tol`)."""
    return active_set(beta, tol, axis=axis).sum(axis=axis)


def assert_safe(h, tol=_ASSERT_TOL, what="h"):
    """Fail loudly (brief, ground rule 5) rather than continue with a violated invariant."""
    h = np.asarray(h, dtype=float)
    worst = np.nanmin(h)
    if not (worst >= -tol):
        raise AssertionError(f"safety invariant violated: min {what} = {worst!r} < -{tol}")
    return worst
