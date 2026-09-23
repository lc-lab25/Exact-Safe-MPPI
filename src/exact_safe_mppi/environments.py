"""Environments: disk gap, rounded-square (p-norm) gap, and the nine-constraint map."""
from ._envs.gap_scenarios import DiskGap, BoxGapDI, sup_slice_hex     # noqa: F401
from ._envs import gs_mppi_map as nine_constraint                     # noqa: F401
from ._envs.gap_scenarios import (GAMMA, SIGMA, LAMBDA, K_SAMPLES, N_HORIZON,  # noqa: F401
                                  TS, DT_INNER, NU_MAX, NU_MIN)

#: the rounded-square gap of the paper is the p-norm box with a double-integrator plant
RoundedSquareGap = BoxGapDI

__all__ = ["DiskGap", "BoxGapDI", "RoundedSquareGap", "nine_constraint", "sup_slice_hex",
           "GAMMA", "SIGMA", "LAMBDA", "K_SAMPLES", "N_HORIZON", "TS", "DT_INNER",
           "NU_MAX", "NU_MIN"]
