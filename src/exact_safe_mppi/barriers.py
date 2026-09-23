"""Barrier families and the high-order CBF cascade.

Two families are used in the paper:

  * squared-distance disk barriers   h_j(q) = ||q - o_j||^2 - r^2
  * p-norm box barriers              h_j(q) = ||q - b_j||_p - c        (p = 8)

Both have relative degree two for the planar double integrator, so each is lifted by the
cascade b_{j,0} = h_j,  b_{j,i+1} = L_f b_{j,i} + alpha_{j,i}(b_{j,i}), with the linear gain
alpha_{j,0}(s) = a * s.  The cascade jet is evaluated by forward-mode truncated-polynomial
duals (`_dual`), which is what the experiment code uses; nothing here re-derives it.
"""
from ._dual import TPD, seed_batched, lie_f, value, Lf, Lg          # noqa: F401
from ._envs.gap_scenarios import A0 as CASCADE_GAIN                  # alpha_{j,0}(s) = A0 * s
from ._envs.gap_scenarios import ALPHA_GAIN as FILTER_CLASS_K_GAIN   # alpha(s) in the filter
from ._envs.gap_scenarios import P_NORM                              # box barrier exponent
from ._envs.gap_scenarios import alpha                               # noqa: F401
from ._envs import disk_gap                                          # noqa: F401

__all__ = ["TPD", "seed_batched", "lie_f", "value", "Lf", "Lg", "alpha",
           "CASCADE_GAIN", "FILTER_CLASS_K_GAIN", "P_NORM", "disk_gap"]
