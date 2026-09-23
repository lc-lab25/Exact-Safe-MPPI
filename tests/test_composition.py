"""Lemma 3 identity and bounds, and the two-barrier slice form."""
import numpy as np
import pytest

from exact_safe_mppi.composition import softmin_stable, exact_min, active_set

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("ell", [2, 3, 5, 9])
@pytest.mark.parametrize("rho", [0.5, 1.0, 5.0, 50.0])
def test_lemma3_identity(ell, rho):
    """h_min - h_rho = ln k / rho + (1/rho) ln(1 + (1/k) sum_{j not tied} e^{-rho(b_j-h_min)})."""
    b = RNG.uniform(-1.0, 3.0, (2000, ell))
    b[:, 1] = b[:, 0]                                   # force k >= 2 ties
    hmin = exact_min(b)
    hrho = softmin_stable(b, rho)
    gap = b - hmin[:, None]
    tied = gap <= 1e-12
    k = tied.sum(-1)
    rest = np.where(tied, 0.0, np.exp(-rho * gap)).sum(-1)
    rhs = np.log(k) / rho + np.log1p(rest / k) / rho
    assert np.allclose(hmin - hrho, rhs, atol=1e-12, rtol=0)


@pytest.mark.parametrize("rho", [0.5, 2.0, 20.0])
def test_lemma3_bounds(rho):
    """ln k / rho <= h_min - h_rho <= ln ell / rho."""
    b = RNG.uniform(-1.0, 3.0, (4000, 4))
    b[:, 1] = b[:, 0]
    hmin, hrho = exact_min(b), softmin_stable(b, rho)
    k = (b - hmin[:, None] <= 1e-12).sum(-1)
    e = hmin - hrho
    assert np.all(e >= np.log(k) / rho - 1e-12)
    assert np.all(e <= np.log(b.shape[1]) / rho + 1e-12)


@pytest.mark.parametrize("rho", [0.3, 1.0, 7.0])
def test_two_barrier_slice(rho):
    """For two barriers, h_rho = T/2 - rho^-1 ln(2 cosh(rho s)) with b = T/2 -/+ s."""
    T = RNG.uniform(0.2, 4.0, 3000)
    s = RNG.uniform(-2.0, 2.0, 3000)
    b = np.stack([T / 2 - s, T / 2 + s], 1)
    expect = T / 2 - np.log(2 * np.cosh(rho * s)) / rho
    assert np.allclose(softmin_stable(b, rho), expect, atol=1e-12, rtol=0)


def test_active_set_delta():
    b = np.array([[0.0, 0.05, 0.4], [1.0, 1.0, 1.2]])
    A = active_set(b, 0.1)
    assert A.tolist() == [[True, True, False], [True, True, False]]
