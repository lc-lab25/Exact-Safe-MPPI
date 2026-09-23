"""The same seed gives a bit-identical trajectory, and --workers cannot change results."""
import numpy as np

from exact_safe_mppi import environments as E
from exact_safe_mppi.dynamics import DoubleIntegrator
from exact_safe_mppi.mppi import Cost, make_filter, run_closed_loop
from exact_safe_mppi.utils import run_seed, effective_sample_size


def _run(seed):
    env = E.DiskGap(0.12)
    x0, qg = env.start_goal()
    filt = make_filter("exact", env, delta=0.0666, k_max=4)
    return run_closed_loop(x0, DoubleIntegrator, env, filt, Cost(qg), seed=seed, T=0.4)


def test_same_seed_bit_identical():
    a, b = _run(11), _run(11)
    assert np.array_equal(np.array(a["x"]), np.array(b["x"]))
    assert np.array_equal(np.array(a["u"]), np.array(b["u"]))


def test_different_seed_differs():
    a, b = _run(11), _run(12)
    assert not np.array_equal(np.array(a["x"]), np.array(b["x"]))


def test_run_seed_independent_of_order():
    assert run_seed(0, 7) == 7 and run_seed(100, 7) == 107


def test_effective_sample_size():
    assert abs(effective_sample_size(np.ones(1000)) - 1000.0) < 1e-9
    w = np.zeros(1000); w[0] = 1.0
    assert abs(effective_sample_size(w) - 1.0) < 1e-9
