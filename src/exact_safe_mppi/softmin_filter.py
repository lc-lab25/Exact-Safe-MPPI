"""GS-MPPI scalar soft-minimum law (baseline).

Carried unchanged from the experiment code that produced the paper results; only the
module layout differs.
"""
"""Safety filters: GS-MPPI soft-minimum closed form, and the exact composite LCP filter.

Both are BATCHED: they accept (..., n) states and (..., m) desired controls and return
(..., m) safe controls.  Phase 2 never calls them per sample.

The two differ in exactly two places -- the barrier composition and the number of constraints
enforced.  Everything else (the jet, the class-K functions, gamma, the integrator) is shared,
so an arm-to-arm difference cannot come from anywhere else.
"""
import itertools

import numpy as np

from ._constants import _TAU_REL, _SF_EPS

_EPS = 2.220446049250313e-16


# --------------------------------------------------------------------------------------- #
#  Arm D: GS-MPPI soft-minimum filter, Eq. (19), stabilised
# --------------------------------------------------------------------------------------- #
def softmin_closed_form(beta, Lf_beta, Lg_beta, v, rho, gamma, alpha):
    """GS-MPPI Eq. (19) with the stabilised log-sum-exp composition.

        h_rho   = m - (1/rho) log sum_j exp(-rho (beta_j - m)),  m = min_j beta_j
        L h_rho = sum_j w_j L beta_j,    w_j = exp(-rho(beta_j - h_rho)) / sum
        omega   = L_f h_rho + L_g h_rho v + alpha(h_rho)
        u*      = v + L_g h_rho^T max{0, -omega} / (L_g h_rho L_g h_rho^T + h_rho^2 / gamma)

    beta (...,L), Lf_beta (...,L), Lg_beta (...,L,m), v (...,m).
    Returns u (...,m) and a dict of diagnostics.
    """
    beta = np.asarray(beta, float)
    mmin = beta.min(-1, keepdims=True)
    e = np.exp(-rho * (beta - mmin))
    ssum = e.sum(-1, keepdims=True)
    h = (mmin - np.log(ssum) / rho)[..., 0]
    w = e / ssum                                       # (...,L), sums to 1
    Lf_h = (w * Lf_beta).sum(-1)                       # (...)
    Lg_h = (w[..., None] * Lg_beta).sum(-2)            # (...,m)
    omega = Lf_h + (Lg_h * v).sum(-1) + alpha(h)
    denom = (Lg_h * Lg_h).sum(-1) + h * h / gamma
    scale = np.maximum(0.0, -omega) / np.where(denom > 0, denom, 1.0)
    scale = np.where(denom > 0, scale, 0.0)
    u = v + Lg_h * scale[..., None]
    # Same slack instrumentation as the exact filter, so arm D and arm E are directly
    # comparable.  GS-MPPI has ONE composite constraint, so k = 1 here.
    lam = scale                                        # the LCP multiplier of Thm 2 at k = 1
    mu = lam * h / gamma
    lin = Lf_h + (Lg_h * u).sum(-1) + alpha(h)         # a + b uhat
    mu_beta = mu * h
    sf = mu_beta / np.maximum(np.abs(lin) + mu_beta, _SF_EPS)
    one = np.ones_like(h, dtype=bool)
    M11 = (Lg_h * Lg_h).sum(-1) + h * h / gamma
    # the pure-uhat halfplane {a + b uhat >= 0} is empty only if L_g h = 0 while a < 0
    empty = (np.linalg.norm(Lg_h, axis=-1) <= 0.0) & (Lf_h + alpha(h) < 0.0)
    return u, dict(h=h, weights=w, lam=lam[..., None], mu=mu[..., None],
                   mu_beta=mu_beta[..., None], lin=lin[..., None],
                   slack_fraction=sf[..., None], in_A=one[..., None],
                   beta_act=h[..., None], n_active=np.ones_like(h, dtype=int),
                   rank_B=(np.linalg.norm(Lg_h, axis=-1) > 0).astype(int),
                   max_lam=np.abs(lam), support_size=(lam > 0).astype(int),
                   polytope_empty=empty, capped=np.zeros_like(h, dtype=bool),
                   lam_min_M=M11,
                   min_Lg_active=np.linalg.norm(Lg_h, axis=-1),
                   Lg_h=Lg_h, omega=omega)


# --------------------------------------------------------------------------------------- #
#  Arm E: exact composite filter -- LCP by bounded support enumeration
# --------------------------------------------------------------------------------------- #
