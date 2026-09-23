"""Proposition 1: the closed-form rho_crit matches numerical maximisation over the slice."""
import numpy as np
import pytest

from exact_safe_mppi import environments as E
from exact_safe_mppi.barriers import CASCADE_GAIN as A
from exact_safe_mppi.composition import softmin_stable

LN2 = float(np.log(2.0))


def _slice(env, n_q=201, n_v=801, vmax=20.0):
    qy = np.linspace(-env.w, env.w, n_q)
    vy = np.linspace(-vmax, vmax, n_v)
    Q, V = np.meshgrid(qy, vy, indexing="ij")
    Z = np.zeros(Q.size)
    return np.stack([Z, Q.ravel(), Z, V.ravel()], 1)


@pytest.mark.parametrize("w", [0.08, 0.12, 0.20])
def test_rho_crit_closed_form(w):
    env = E.DiskGap(w)
    X = _slice(env)
    b, _, _ = env.beta_jet(X)
    ok = env.h_np(X).min(-1) >= 0
    H0 = A * w * (2 * env.r + w)
    assert abs(float(b.min(-1)[ok].max()) - H0) < 1e-9
    rho_crit = LN2 / H0
    # sup h_rho crosses zero at rho_crit
    assert float(softmin_stable(b, rho_crit * 0.99)[ok].max()) < 0.0
    assert float(softmin_stable(b, rho_crit * 1.01)[ok].max()) > 0.0
