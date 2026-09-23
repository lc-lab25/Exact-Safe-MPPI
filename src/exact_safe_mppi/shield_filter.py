"""Discrete-time CBF repair baseline (Shield-MPPI style).

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
def shield_repair(beta, Lf_beta, Lg_beta, v, gamma, alpha, Ts, n_grad=10, **kw):
    """OUR REIMPLEMENTATION of the Shield-MPPI repair (Yin et al.), not their code.

    Shield-MPPI repairs the planner's control by a fixed number of gradient-descent steps on a
    soft discrete-time-CBF penalty rather than by solving a constrained program.  We keep that
    mechanism and change nothing else: the arm receives the SAME cascade jet
    (beta, L_f beta, L_g beta) and the SAME class-K function alpha as arms D and E, so the only
    difference from arm E is HOW the constraints are enforced, not WHICH constraints exist.

        g_j(u) = Ts (L_f beta_j + L_g beta_j . u + alpha(beta_j))          (one-step DT-CBF)
        C(u)   = sum_j [max(0, -g_j(u))]^2
        u <- u + 2 eta Ts sum_j max(0, -g_j(u)) L_g beta_j                 (n_grad times)

    THE STEP SIZE IS CHOSEN TO MAKE THIS ARM AS STRONG AS POSSIBLE, so that a poor showing
    cannot be blamed on tuning: eta = 1 / (2 Ts^2 max_j ||L_g beta_j||^2) repairs a SINGLE
    active constraint exactly in one step, i.e. arm C is an exact projection whenever k = 1.
    Its only remaining weakness is the one the paper is about -- a soft penalty summed over
    constraints has no mechanism to satisfy several of them simultaneously.
    """
    beta = np.asarray(beta, float)
    lf = np.asarray(Lf_beta, float)
    lg = np.asarray(Lg_beta, float)
    u = np.asarray(v, float).copy()
    nrm2 = (lg * lg).sum(-1).max(-1)                                   # (...,) max_j ||L_g b_j||^2
    eta = np.where(nrm2 > 0, 1.0 / (2.0 * Ts * Ts * np.maximum(nrm2, 1e-300)), 0.0)
    for _ in range(n_grad):
        g = Ts * (lf + np.einsum("...jm,...m->...j", lg, u) + alpha(beta))
        viol = np.maximum(0.0, -g)                                     # (...,L)
        grad = np.einsum("...j,...jm->...m", viol, lg)
        u = u + (2.0 * eta * Ts)[..., None] * grad
    h = beta.min(-1)
    g = Ts * (lf + np.einsum("...jm,...m->...j", lg, u) + alpha(beta))
    resid = np.maximum(0.0, -g).max(-1)      # how much the repair failed to achieve
    return u, dict(h=h, n_active=(g < 0).sum(-1), shield_resid=resid,
                   max_lam=np.zeros_like(h), support_size=np.zeros_like(h, dtype=int),
                   rank_B=np.zeros_like(h, dtype=int), capped=np.zeros_like(h, dtype=bool),
                   polytope_empty=np.zeros_like(h, dtype=bool),
                   lam_min_M=np.full_like(h, np.nan),
                   min_Lg_active=np.sqrt(np.maximum(nrm2, 0.0)))
