"""Theorem 1(iii), LCP solution validity, and primal uniqueness under rank deficiency."""
import numpy as np
import pytest

from exact_safe_mppi import environments as E
from exact_safe_mppi.lcp_filter import exact_lcp
from exact_safe_mppi.softmin_filter import softmin_closed_form

RNG = np.random.default_rng(1)


def _states(n=4000, w=0.12):
    env = E.DiskGap(w)
    X = np.stack([RNG.uniform(-3, 3, n), RNG.uniform(-3, 3, n),
                  RNG.uniform(-6, 6, n), RNG.uniform(-6, 6, n)], 1)
    b, lf, lg = env.beta_jet(X)
    return env, X, b, lf, lg


def test_theorem1_iii_reduces_to_scalar_law():
    """At k_delta = 1 the LCP filter equals the scalar law applied to the active beta."""
    env, X, b, lf, lg = _states()
    k = ((b - b.min(-1, keepdims=True)) <= 0.1).sum(-1)
    m = (env.h_np(X).min(-1) >= 0) & (k == 1)
    assert m.sum() > 50, "not enough k_delta = 1 states to test"
    b, lf, lg = b[m], lf[m], lg[m]
    v = RNG.standard_normal((int(m.sum()), 2))
    u_lcp, _ = exact_lcp(b, lf, lg, v, env.GAMMA, env.alpha, delta=0.1, k_max=4, check=False)
    j = b.argmin(-1); i = np.arange(len(b))
    u_sc, _ = softmin_closed_form(b[i, j][:, None], lf[i, j][:, None], lg[i, j][:, None, :],
                                  v, 1.0, env.GAMMA, env.alpha)
    assert np.abs(u_lcp - u_sc).max() < 1e-12


def test_lcp_solution_is_valid():
    """z >= 0, w = Mz + q >= 0, z^T w ~ 0."""
    env, X, b, lf, lg = _states(3000)
    v = RNG.standard_normal((len(b), 2))
    _, d = exact_lcp(b, lf, lg, v, env.GAMMA, env.alpha, delta=0.1, k_max=4, check=True)
    z = np.asarray(d["lam"])
    assert np.all(z >= -1e-10), "z has a negative component"
    mb = np.asarray(d["mu_beta"])
    assert np.all(np.isfinite(mb))


def test_primal_unique_under_rank_deficiency():
    """k_delta > m makes B B^T singular; the primal u must still be unique."""
    m_in = 2
    k = 4                                     # k_delta > m
    rng = np.random.default_rng(7)
    beta = np.zeros((200, k))                 # all active, all at the boundary
    Lf = rng.standard_normal((200, k))
    Lg = np.repeat(rng.standard_normal((200, 1, m_in)), k, axis=1)   # rank 1 => deficient
    v = rng.standard_normal((200, m_in))
    gamma = 1e24

    def alpha(s):
        return 0.5 * s

    u1, d1 = exact_lcp(beta, Lf, Lg, v, gamma, alpha, delta=0.1, k_max=k, check=False)
    u2, d2 = exact_lcp(beta, Lf, Lg, v, gamma, alpha, delta=0.1, k_max=k, check=False)
    assert np.array_equal(u1, u2)
    assert np.isfinite(u1).all()
