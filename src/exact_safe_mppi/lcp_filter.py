"""Exact-Safe MPPI: closed-form LCP filter (Theorem 1) by bounded support enumeration.

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
def _sigma_activation(gap, delta1, delta2):
    """Lipschitz activation of Theorem 3: 1 on A_{delta1}, 0 outside A_{delta2}."""
    return np.clip((delta2 - gap) / (delta2 - delta1), 0.0, 1.0)




def exact_lcp(beta, Lf_beta, Lg_beta, v, gamma, alpha, delta=0.1, k_max=4,
              tol_rel=1e-9, lipschitz_activation=False,
              delta1=None, delta2=None, K_relax=1e6, check=True):
    """Exact composite filter: h_ex = min_j beta_j, all delta-active constraints enforced.

    Solves the LCP of Theorem 2 by enumerating the 2^k support sets.  Three details are forced
    by the theory rather than chosen for convenience; each was a bug before it was fixed.

    1.  THE CORRECTION IS CONFINED TO range(B^T).  Writing uhat = v + B_S^T lambda_S with
        M_SS lambda_S = -q_S is not merely one algebraic route: it is the numerically correct
        one.  The equivalent primal form (I + A) uhat = v - c, A = sum_j omega_j b_j^T b_j,
        omega_j = gamma/beta_j^2, is unusable whenever A is rank deficient -- with gamma = 1e24
        and small beta_j, `c` reaches 1e30 and its component along the null direction of A,
        exactly zero in real arithmetic, is ~1e14 in float64.  The B^T form has no such
        direction to lose.

    2.  SMALL EIGENVALUES OF M_SS ARE TRUNCATED, AND THAT IS EXACT FOR uhat.  For m = 2 every
        support with |S| >= 3 has singular B_S B_S^T, so M_SS is singular up to the
        gamma^{-1} diag(beta^2) term (~1e-23).  For an eigenvector v_i of M_SS,
        ||B^T v_i||^2 = v_i' M v_i - gamma^{-1} sum beta_j^2 v_ij^2, so a direction with tiny
        eigenvalue has B^T v_i = 0 in exact arithmetic and contributes NOTHING to
        uhat = v + B^T lambda.  Keeping it instead injects ~1e11 of noise.

    3.  FEASIBILITY IS TESTED ONLY ON THE delta-ACTIVE CONSTRAINTS NOT IN THE SUPPORT.
        Constraints in S hold with equality by construction, and their residual after
        truncation is meaningless; constraints outside A_delta have no row in the QP at all.

    The slack is not decorative.  With three active constraints and m = 2 the pure-uhat
    polytope {a_j + b_j uhat >= 0} is routinely EMPTY, and muhat is what restores feasibility
    -- exactly the role Assumption 1 assigns it.  The optimal multipliers are then of order
    gamma, so muhat is recovered from the equality residual on such supports rather than from
    lambda, and from lambda on the well-conditioned ones where the residual is pure roundoff.
    """
    beta = np.asarray(beta, float)
    v = np.asarray(v, float)
    batch = beta.shape[:-1]
    L, m = beta.shape[-1], v.shape[-1]
    Nb = int(np.prod(batch)) if batch else 1
    bt = beta.reshape(Nb, L)
    lf = np.asarray(Lf_beta, float).reshape(Nb, L)
    lg = np.asarray(Lg_beta, float).reshape(Nb, L, m)
    vv = v.reshape(Nb, m)

    h_ex = bt.min(-1)
    gap = bt - h_ex[:, None]
    k = min(k_max, L)
    idx = np.argsort(bt, axis=-1, kind="stable")[:, :k]
    rows = np.arange(Nb)[:, None]
    b_k, lf_k, lg_k, gap_k = bt[rows, idx], lf[rows, idx], lg[rows, idx], gap[rows, idx]
    in_A = gap_k <= delta
    n_active_total = (gap <= delta).sum(-1)
    capped = n_active_total > k

    a_k = lf_k + alpha(b_k)
    if lipschitz_activation:
        d1 = delta if delta1 is None else delta1
        d2 = (2.0 * delta) if delta2 is None else delta2
        a_k = a_k + (1.0 - _sigma_activation(gap_k, d1, d2)) * K_relax

    M = lg_k @ np.swapaxes(lg_k, -1, -2)
    M[:, np.arange(k), np.arange(k)] += b_k ** 2 / gamma
    q = a_k + np.einsum("nkm,nm->nk", lg_k, vv)
    scale = np.maximum(1.0, np.abs(a_k).max(-1))
    tol = tol_rel * scale

    # tier 0 = |S| <= m (multipliers O(1), slack negligible); tier 1 = |S| > m (slack O(1))
    best_key = np.full(Nb, np.inf)      # tier * BIG + objective, so tier 0 always wins
    BIG = 1e100
    best_u = vv.copy()
    best_mu = np.zeros((Nb, k))
    best_lam = np.zeros((Nb, k))
    best_size = np.zeros(Nb, int)
    best_tier = np.zeros(Nb, int)
    found = np.zeros(Nb, bool)

    for r in range(k + 1):
        for S in itertools.combinations(range(k), r):
            S = list(S)
            ok = np.ones(Nb, bool) if not S else in_A[:, S].all(-1)
            if not ok.any():
                continue
            lam = np.zeros((Nb, k))
            if S:
                Ms = M[np.ix_(np.arange(Nb), S, S)]
                wv, P = np.linalg.eigh(Ms)
                tau = _TAU_REL * np.maximum(wv.max(-1, keepdims=True), 1e-300)
                keep = wv > tau
                proj = np.einsum("nij,ni->nj", P, q[:, S])
                inv = np.where(keep, 1.0 / np.where(keep, wv, 1.0), 0.0)
                lam[:, S] = -np.einsum("nij,nj->ni", P, inv * proj)
                # NO dual-feasibility (lambda >= 0) gate here.  On a support where M_SS is
                # numerically singular the truncated lambda omits its null component, whose
                # sign is arbitrary, so the test rejects valid supports.  It is also
                # unnecessary: every candidate retained below is a FEASIBLE point of the QP,
                # the objective is strictly convex, and the optimum is attained at one of the
                # 2^k supports -- so the minimum over feasible candidates IS the optimum.
            u_S = vv + np.einsum("nkm,nk->nm", lg_k, lam)
            resid = a_k + np.einsum("nkm,nm->nk", lg_k, u_S)          # before slack
            mu = np.zeros((Nb, k))
            if S:
                # On the support the constraint holds with EQUALITY, so
                # muhat_j = -(a_j + b_j uhat)/beta_j is always the correct value.  The earlier
                # rule -- take muhat = lambda beta/gamma whenever |S| <= m, on the assumption
                # that the slack is then negligible -- is WRONG: it assumes the pure-uhat
                # polytope is nonempty, and |S| <= m does not imply that.  With m = 2 and two
                # ANTI-PARALLEL active gradients the polytope is empty at |S| = 2 = m, the true
                # slack is O(1), and setting it to ~0 made every support look infeasible.
                # (Observed: a BoxGap state with L_g beta_1 . L_g beta_3 = -0.994 requiring
                # simultaneously u_1 <= -1.763 and u_1 >= -1.086.)
                #
                # Using the residual unconditionally instead is correct but amplifies roundoff
                # by 1/beta_j when the true slack IS ~0, and gamma * muhat^2 then magnifies it.
                # So the residual is floored at the numerical noise level first: genuine O(1)
                # slack survives, roundoff is zeroed.
                rs = resid[:, S]
                rs = np.where(np.abs(rs) <= tol[:, None], 0.0, rs)
                safe_b = np.where(np.abs(b_k[:, S]) > 0, b_k[:, S], np.inf)
                mu[:, S] = -rs / safe_b
            # Feasibility of EVERY delta-active constraint at the candidate point, using the
            # actual residual including the slack (zero off the support by construction).
            res_all = resid + mu * b_k
            ok &= np.where(in_A, res_all, np.inf).min(-1) >= -tol
            if not ok.any():
                continue
            obj = 0.5 * ((u_S - vv) ** 2).sum(-1) + 0.5 * gamma * (mu ** 2).sum(-1)
            key = (0.0 if len(S) <= m else BIG) + obj
            better = ok & (key < best_key)
            best_key = np.where(better, key, best_key)
            best_u = np.where(better[:, None], u_S, best_u)
            best_mu = np.where(better[:, None], mu, best_mu)
            best_lam = np.where(better[:, None], lam, best_lam)
            best_size = np.where(better, len(S), best_size)
            best_tier = np.where(better, 0 if len(S) <= m else 1, best_tier)
            found |= ok

    if check and not found.all():
        raise AssertionError(f"exact_lcp: no feasible support for {int((~found).sum())}/{Nb} "
                             f"states (Assumption 1 feasibility violated?)")
    if check:
        res = a_k + np.einsum("nkm,nm->nk", lg_k, best_u) + best_mu * b_k
        viol = (np.where(in_A, res, np.inf).min(-1) / scale).min()
        if not viol >= -1e-9:
            raise AssertionError(f"exact_lcp: active constraint violated by {viol:.3e}")

    # ---- slack instrumentation (see RESULTS_LOG "empty polytope") --------------------
    # The enforcement of an active constraint splits into a CONTROL part a_j + b_j uhat and a
    # SLACK part muhat_j beta_j, which sum to >= 0.  Logging them separately makes the split
    # recoverable post hoc under any definition.
    lin = a_k + np.einsum("nkm,nm->nk", lg_k, best_u)       # a_j + b_j uhat
    mu_beta = best_mu * b_k                                  # muhat_j beta_j
    slack_fraction = mu_beta / np.maximum(np.abs(lin) + mu_beta, _SF_EPS)
    Bm = lg_k * in_A[..., None]
    sv = np.linalg.svd(Bm, compute_uv=False)
    rank_B = (sv > 1e-10 * np.maximum(sv.max(-1, keepdims=True), 1e-300)).sum(-1)
    # The pure-uhat polytope {a_j + b_j uhat >= 0, j in A_delta} is nonempty iff some support
    # with |S| <= m was feasible: in R^m the projection onto a nonempty polyhedron has an
    # active set of size at most m.  So tier 1 selected  <=>  the polytope is empty.
    polytope_empty = (best_tier == 1)

    def rs(a, tail=()):
        return a.reshape(batch + tail) if batch else (a[0] if not tail else a[0])

    diag = dict(found=rs(found), h=rs(h_ex),
                lam=rs(best_lam, (k,)), mu=rs(best_mu, (k,)),
                mu_beta=rs(mu_beta, (k,)), lin=rs(lin, (k,)),
                slack_fraction=rs(slack_fraction, (k,)),
                in_A=rs(in_A, (k,)), beta_act=rs(b_k, (k,)),
                n_active=rs(n_active_total), rank_B=rs(rank_B),
                max_lam=rs(np.abs(best_lam).max(-1)),
                support_size=rs(best_size), polytope_empty=rs(polytope_empty),
                idx=rs(idx, (k,)), capped=rs(capped),
                lam_min_M=rs(np.linalg.eigvalsh(M).min(-1)),
                min_Lg_active=rs(np.where(in_A, np.linalg.norm(lg_k, axis=-1),
                                          np.inf).min(-1)))
    return best_u.reshape(batch + (m,)), diag


def no_filter(beta, Lf_beta, Lg_beta, v, **kw):
    """Arms A and B: pass the desired control straight through."""
    return np.asarray(v, float), dict(h=np.asarray(beta, float).min(-1))


# --------------------------------------------------------------------------------------- #
#  Arm C: Shield-MPPI style discrete-time CBF repair
# --------------------------------------------------------------------------------------- #
