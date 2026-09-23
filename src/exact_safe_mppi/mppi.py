"""GS-MPPI Algorithms 1 and 2 (Rabiee & Hoagg, arXiv:2410.02154), batched.

DEVIATIONS FROM THE PAPER'S PROSE, recorded because the baseline is what everything else is
diffed against (see RESULTS_LOG.md [V2], [V3]):

  * Eq. (25) sums the running cost from k = 1; Algorithm 1 line 8 accumulates psi(x, v_k)
    from k = 0.  We follow ALGORITHM 1, which is the executable specification.
  * Eq. (41) is written with S(M,E) of Eq. (38); Algorithm 1 lines 13-18 use the raw path cost
    J^(j).  We follow ALGORITHM 1.
  * Algorithm 2 line 13, `Initialize()` for mu_{N-1}, is unspecified.  We use 0.

All arms use the identical convention, so the comparison between arms is unaffected.
"""
import numpy as np

from . import _filters as _filters


# Bin edges for the rollout slack statistics.  Fine near beta = 0, where (C3) predicts the
# slack contribution muhat_j beta_j must vanish; coarse far from the boundary.
BETA_EDGES = np.concatenate([[-np.inf, -1.0, -0.1],
                             np.linspace(0.0, 2.0, 41), [3.0, 5.0, 10.0, np.inf]])
SF_EDGES = np.linspace(0.0, 1.0, 21)


class SlackStats:
    """Aggregates the slack split over EVERY rollout filter evaluation of a run.

    The executed trajectory almost never approaches a constraint boundary, so per-timestep
    logging of the closed loop says nothing about the slack.  The K x N rollout evaluations do
    -- that is where the delta-active sets grow and where the pure-uhat polytope was observed
    to be empty.  Storing all of them (2e6 rows per run) is not worth it, so we bin.
    """

    def __init__(self):
        nb, ns = len(BETA_EDGES) - 1, len(SF_EDGES) - 1
        self.hist = np.zeros((nb, ns), np.int64)        # (beta_j, slack_fraction_j), active j
        self.n_beta = np.zeros(nb, np.int64)
        self.max_mu_beta = np.zeros(nb)                 # sup |muhat_j beta_j| per beta bin
        self.sum_mu_beta = np.zeros(nb)
        self.n_active_hist = np.zeros(16, np.int64)
        self.rank_hist = np.zeros(8, np.int64)
        self.support_hist = np.zeros(16, np.int64)
        self.n_empty = 0
        self.n_eval = 0

    def update(self, d):
        if "slack_fraction" not in d:
            return
        sf = np.asarray(d["slack_fraction"])
        mb = np.asarray(d["mu_beta"])
        ba = np.asarray(d["beta_act"])
        inA = np.asarray(d["in_A"]).astype(bool)
        if sf.ndim == 1:
            sf, mb, ba, inA = sf[None], mb[None], ba[None], inA[None]
        m = inA.ravel()
        b = ba.ravel()[m]
        f = np.clip(sf.ravel()[m], 0.0, 1.0)
        g = np.abs(mb.ravel()[m])
        bi = np.clip(np.digitize(b, BETA_EDGES) - 1, 0, len(BETA_EDGES) - 2)
        fi = np.clip(np.digitize(f, SF_EDGES) - 1, 0, len(SF_EDGES) - 2)
        np.add.at(self.hist, (bi, fi), 1)
        np.add.at(self.n_beta, bi, 1)
        np.add.at(self.sum_mu_beta, bi, g)
        np.maximum.at(self.max_mu_beta, bi, g)
        na = np.asarray(d.get("n_active", 0)).ravel()
        np.add.at(self.n_active_hist, np.clip(na, 0, 15), 1)
        rk = np.asarray(d.get("rank_B", 0)).ravel()
        np.add.at(self.rank_hist, np.clip(rk, 0, 7), 1)
        ss = np.asarray(d.get("support_size", 0)).ravel()
        np.add.at(self.support_hist, np.clip(ss, 0, 15), 1)
        pe = np.asarray(d.get("polytope_empty", False)).ravel()
        self.n_empty += int(pe.sum())
        self.n_eval += int(pe.size)

    def as_dict(self):
        return dict(slack_hist=self.hist, slack_n_beta=self.n_beta,
                    slack_max_mu_beta=self.max_mu_beta,
                    slack_sum_mu_beta=self.sum_mu_beta,
                    slack_n_active_hist=self.n_active_hist,
                    slack_rank_hist=self.rank_hist,
                    slack_support_hist=self.support_hist,
                    slack_beta_edges=BETA_EDGES, slack_sf_edges=SF_EDGES,
                    slack_n_empty=np.int64(self.n_empty),
                    slack_n_eval=np.int64(self.n_eval))


class Cost:
    """phi(x) = 2 (q-qd)'(q-qd);  psi(x,v) = (q-qd)'(q-qd) + 0.05 v'v."""

    def __init__(self, goal, w_ctrl=0.05, penalty=0.0):
        self.goal = np.asarray(goal, float)
        self.w_ctrl = w_ctrl
        self.penalty = penalty          # arm B only: cost added for barrier violation

    def terminal(self, X):
        e = X[..., :2] - self.goal
        return 2.0 * (e * e).sum(-1)

    def running(self, X, V, beta=None):
        e = X[..., :2] - self.goal
        c = (e * e).sum(-1) + self.w_ctrl * (V * V).sum(-1)
        if self.penalty and beta is not None:
            c = c + self.penalty * np.maximum(0.0, -beta.min(-1)) ** 2
        return c


def make_filter(kind, env, rho=None, gamma=None, delta=0.1, k_max=4,
                lipschitz_activation=False):
    """Return `fn(X, V) -> (u, diag)`.  Every arm gets the SAME jet; only this differs."""
    gamma = env.GAMMA if gamma is None else gamma

    def jet(X):
        return env.beta_jet(X)

    if kind == "none":
        def fn(X, V):
            beta, lf, lg = jet(X)
            u, d = _filters.no_filter(beta, lf, lg, V)
            d["beta"] = beta
            return u, d
    elif kind == "softmin":
        def fn(X, V):
            beta, lf, lg = jet(X)
            u, d = _filters.softmin_closed_form(beta, lf, lg, V, rho, gamma, env.alpha)
            d["beta"] = beta
            return u, d
    elif kind == "shield":
        Ts = env.TS
        def fn(X, V):
            beta, lf, lg = jet(X)
            u, d = _filters.shield_repair(beta, lf, lg, V, gamma, env.alpha, Ts)
            d["beta"] = beta
            return u, d
    elif kind == "exact":
        def fn(X, V):
            beta, lf, lg = jet(X)
            u, d = _filters.exact_lcp(beta, lf, lg, V, gamma, env.alpha, delta=delta,
                                      k_max=k_max,
                                      lipschitz_activation=lipschitz_activation)
            d["beta"] = beta
            return u, d
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    return fn


def step_euler(env_dyn, X, U, dt):
    return X + (env_dyn.f_np(X) + np.einsum("...nm,...m->...n", env_dyn.g_np(X), U)) * dt


def mppi_planner(x0, Mseq, dyn, filt, cost, rng, K, N, lam, chol_Sigma, Ts,
                 collect_endpoints=False, slack=None):
    """Algorithm 1, batched over the K samples.

    Returns (M_star, v_best0, info).  `info` carries the weights, the endpoint cloud, and the
    per-sample cost, all of which the HDF5 schema logs.
    """
    n, m = dyn.n, dyn.m
    eps = rng.standard_normal((K, N, m)) @ chol_Sigma.T          # (K,N,m) ~ N(0, Sigma)
    V = Mseq[None, :, :] + eps
    X = np.broadcast_to(np.asarray(x0, float), (K, n)).copy()
    J = np.zeros(K)
    traj = np.empty((K, N + 1, n)) if collect_endpoints else None
    if collect_endpoints:
        traj[:, 0] = X
    for k in range(N):
        u, d = filt(X, V[:, k, :])
        if slack is not None:
            slack.update(d)
        J += cost.running(X, V[:, k, :], d.get("beta"))
        X = step_euler(dyn, X, u, Ts)
        if collect_endpoints:
            traj[:, k + 1] = X
    J += cost.terminal(X)

    xi = J.min()
    jbest = int(np.argmin(J))
    w = np.exp(-(J - xi) / lam)
    w /= w.sum()
    M_star = Mseq + np.einsum("k,knm->nm", w, eps)
    # Effective sample size of the path-integral weights.  ESS = 1 means one sample carries all
    # the weight and MPPI has degenerated to greedy random search: the path-integral weighting
    # is inert and the effective sample count is 1 regardless of K.  This is reported, not
    # incidental -- see RESULTS_LOG [F24].
    ess = float(1.0 / np.sum(w ** 2))
    info = dict(J=J, w=w, jbest=jbest, endpoints=X.copy(), traj=traj, ess=ess,
                J_mean=float(J.mean()), J_std=float(J.std()),
                J_spread=float(J.max() - J.min()))
    return M_star, V[jbest, 0, :].copy(), info


def run_closed_loop(x0, dyn, env, filt, cost, seed, T=10.0, Ts=None, dt=None, N=None,
                    K=None, lam=None, Sigma=None, collect_endpoints_every=0):
    """Algorithm 2.  Deterministic in `seed`: same seed => bit-identical output."""
    Ts = env.TS if Ts is None else Ts
    dt = env.DT_INNER if dt is None else dt
    N = env.N_HORIZON if N is None else N
    K = env.K_SAMPLES if K is None else K
    lam = env.LAMBDA if lam is None else lam
    Sigma = env.SIGMA if Sigma is None else Sigma
    chol = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(seed)

    n_inner = int(round(Ts / dt))
    n_steps = int(round(T / Ts))
    Mseq = np.zeros((N, dyn.m))
    x = np.asarray(x0, float).copy()

    PLANNER = ("ess", "J_mean", "J_std", "J_spread")
    SCALARS = ("h", "n_active", "lam_min_M", "min_Lg_active", "capped", "rank_B",
               "max_lam", "support_size", "polytope_empty")
    VECTORS = ("mu_beta", "lin", "slack_fraction", "in_A", "beta_act", "lam", "mu")
    rec = {kk: [] for kk in ("t", "x", "v", "u", "beta", "hj", "wall") + SCALARS + VECTORS}
    plan = {kk: [] for kk in PLANNER}
    clouds = []
    slack = SlackStats()
    import time as _time
    for it in range(n_steps):
        t0 = _time.perf_counter()
        want_cloud = (collect_endpoints_every and it % collect_endpoints_every == 0)
        Mseq, v, info = mppi_planner(x, Mseq, dyn, filt, cost, rng, K, N, lam, chol, Ts,
                                     collect_endpoints=want_cloud, slack=slack)
        wall = _time.perf_counter() - t0
        for kk in PLANNER:
            plan[kk].append(info[kk])
        if want_cloud:
            clouds.append((it, info["endpoints"].copy()))
        for _ in range(n_inner):
            u, d = filt(x[None, :], v[None, :])
            u = u[0]
            rec["t"].append(it * Ts + _ * dt)
            rec["x"].append(x.copy())
            rec["v"].append(v.copy())
            rec["u"].append(u.copy())
            rec["beta"].append(np.atleast_2d(d["beta"])[0].copy())
            rec["hj"].append(env.h_np(x).copy())
            rec["wall"].append(wall)
            for kk in SCALARS:
                val = d.get(kk, np.nan)
                rec[kk].append(np.atleast_1d(val)[0])
            for kk in VECTORS:
                val = d.get(kk, None)
                rec[kk].append(np.atleast_2d(val)[0].copy() if val is not None
                               else np.zeros(1))
            x = step_euler(dyn, x[None, :], u[None, :], dt)[0]
        Mseq = np.vstack([Mseq[1:], np.zeros((1, dyn.m))])      # shift; Initialize() = 0

    out = {kk: np.asarray(vv) for kk, vv in rec.items()}
    out.update({kk: np.asarray(vv) for kk, vv in plan.items()})
    out.update(slack.as_dict())
    out["clouds"] = clouds
    out["seed"] = seed
    return out
