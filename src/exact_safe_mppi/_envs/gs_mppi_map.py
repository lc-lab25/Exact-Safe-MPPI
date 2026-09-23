"""The GS-MPPI benchmark map (Rabiee & Hoagg, arXiv:2410.02154, Fig. 2).

    h_j(x) = || [a_x_j*(qx - b_x_j), a_y_j*(qy - b_y_j)] ||_p - c_j     j = 1..6  (obstacles)
    h_7(x) = c_7 - || [a_x_7*qx, a_y_7*qy] ||_p                                   (wall)
    h_8(x) = 9 - nu,   h_9(x) = nu + 1                                            (speed)

    ell = 9,  d_1..d_7 = 2,  d_8 = d_9 = 1.

===============================================================================
RECONSTRUCTED GEOMETRY -- READ THIS BEFORE USING THESE NUMBERS
===============================================================================
The constants b_x_j, b_y_j, a_x_j, a_y_j, c_j, p are NOT given numerically anywhere in
arXiv:2410.02154.  They were reconstructed from Fig. 2 by the following procedure
(reproducible; see RESULTS_LOG.md, Phase 0):

  1. page 7 of the PDF rendered at 600 dpi (ghostscript);
  2. plot axes calibrated from the detected tick marks (max residual < 0.005 data units);
  3. the red constraint curves segmented into connected components;
  4. a superellipse (|u|^p + |v|^p)^(1/p) = 1, with u = (qx-bx)/sx, v = (qy-by)/sy,
     least-squares fitted to the pixels of each component.

The fitted centres land on round numbers to within ~0.015 units, which is good evidence the
extraction is faithful.  The fitted half-extents (sx, sy) are reproduced below.

*** The exponent p is NOT identifiable from the figure. ***  The fitted p is not constant
across obstacles: it scales with obstacle size (p / half-extent = 9.5 +/- 0.5 for all seven
curves).  A single true p in ||.||_p would give the SAME fitted p for every obstacle, because
the normalised shape (|u|^p + |v|^p = 1) is size-independent.  The observed scaling is the
signature of a fixed corner radius -- i.e. Fig. 2's red curves were drawn as rounded-rectangle
patches, not as contours of h_j.  We therefore CHOOSE p = 8: even (so h_j is C-infinity away
from the obstacle centre, which lies outside S_s), > 2 (so h_j has the two continuous
derivatives that relative degree 2 requires), and visually comparable in squareness.

We set a_x_j = a_y_j = 1 so that h_j carries length units, and c_j = fitted half-extent.
This must be stated in the paper.
===============================================================================
"""
import numpy as np

# ===================== FROZEN RECONSTRUCTED PARAMETERS =====================
# Fixed by the joint (p, lambda) sweep of job 1167857 and accepted at gate 2.1 on 2026-09-01.
# Chosen by closed-loop agreement with GS-MPPI Figs. 2 and 3 (shape AND speed), scored over
# 3 x 6 x 4 = 72 runs; see experiments/r0_plambda_choice.py and REPRODUCIBILITY.md Sections
# 2.4-2.6.  DO NOT retune these: they are a reconstruction, not knobs, and every arm uses the
# same values.
#     p      = 8     (exponent; not identifiable from the drawn figure, chosen on trajectory agreement)
#     lambda = 300   (MPPI temperature; the published 1.0 gives ESS = 1.00/1000)
#     dt     = 0.002 (inner hold; the published 0.05 does not inherit the continuous-time guarantee)
# ===========================================================================
P_NORM = 8  # FROZEN -- see above and the module docstring.
# NOTE: every function below resolves `p` from this module attribute AT CALL TIME (p=None
# default), never as a bound default argument.  Binding it as a default silently pinned the
# exponent at 8 and made the p-sweep of job 1167217 a no-op: all three p values produced
# bit-identical trajectories.  Do not reintroduce `p=P_NORM` in a signature.

# (name, bx, by, ax, ay, c)     RECONSTRUCTED from Fig. 2; fitted half-extents in comments
OBSTACLES = (
    ("obs1", -7.0,  5.0, 1.0, 1.0, 2.14),   # fit centre (-7.011,  5.010) half (2.138, 2.138)
    ("obs2", -2.5,  2.5, 1.0, 1.0, 1.39),   # fit centre (-2.513,  2.508) half (1.393, 1.387)
    ("obs3",  2.0,  1.5, 1.0, 1.0, 2.14),   # fit centre ( 1.985,  1.516) half (2.137, 2.141)
    ("obs4",  6.0,  7.0, 1.0, 1.0, 2.14),   # fit centre ( 5.997,  7.014) half (2.139, 2.147)
    ("obs5",  5.0, -6.0, 1.0, 1.0, 3.14),   # fit centre ( 4.994, -5.990) half (3.141, 3.138)
    ("obs6", -5.0, -5.0, 1.0, 1.0, 2.02),   # fit centre (-5.011, -4.993) half (2.020, 2.017)
)
WALL = ("wall", 0.0, 0.0, 1.0, 1.0, 10.15)  # fit centre (-0.010,  0.007) half (10.145, 10.141)

NU_MAX = 9.0
NU_MIN = -1.0

ELL = 9
RELATIVE_DEGREES = (2, 2, 2, 2, 2, 2, 2, 1, 1)

# alpha_{j,0} of the cascade, and the outer alpha of the safety constraint (paper Sec. VI)
ALPHA0_GAINS = (2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1.0)  # j = 1..7; j = 8,9 have no cascade
ALPHA_GAIN = 0.5

# Paper's scenario
X0 = np.array([-1.0, -8.5, 0.0, np.pi / 2])
GOALS = (np.array([3.0, 4.5]), np.array([-7.0, 0.0]),
         np.array([7.0, 1.5]), np.array([-1.0, 7.0]))

# MPPI / filter parameters (all from the paper)
RHO_DEFAULT = 20.0
GAMMA = 1e24
SIGMA = np.diag([1.33, 0.33])
LAMBDA = 300.0   # FROZEN, reconstructed (published value 1.0 gives ESS = 1.00/1000)
K_SAMPLES = 1000
N_HORIZON = 20
TS = 0.1
DT_INNER = 0.002  # FROZEN, adopted program-wide (published value 0.05)


_S_FLOOR = 1e-30


def _pnorm2(u, v, p=None):
    """(|u|^p + |v|^p)^(1/p), for reals or dual numbers.

    P_NORM is EVEN, so |u|^p == u^p and the absolute value can be dropped entirely.  That
    matters for the dual arithmetic: |.| is not differentiable at 0, and a rollout can pass
    exactly through an obstacle centre line (q_x = b_x), whereas u**p is smooth there.

    No max-rescaling: the rescaling trick branches on |u| vs |v| and would make the program
    non-differentiable at |u| = |v|, which is precisely a tie locus.  With p = 8 and the
    coordinate magnitudes of this map (<= ~20) the direct form cannot overflow in float64.

    The sum is floored below `_S_FLOOR` before the fractional power, because S = 0 exactly at
    an obstacle centre, where s^(1/p) has an infinite derivative.  That point is deep inside an
    obstacle (h_j ~ -c_j, wildly unsafe) and is unreachable under any of the filtered arms; the
    floor exists so the unfiltered arms A and B, which may sample there, do not produce inf.
    """
    p = P_NORM if p is None else p
    if p % 2:
        raise ValueError("P_NORM must be even for the abs-free form")
    s = u ** p + v ** p
    if hasattr(s, "c"):                      # dual: floor the real part only
        np.maximum(s.c[..., 0, 0], _S_FLOOR, out=s.c[..., 0, 0])
    else:
        s = np.maximum(s, _S_FLOOR)
    return s ** (1.0 / p)


def h_list(x, p=None):
    """[h_1, ..., h_9](x) for a single state x = [qx, qy, nu, theta].

    Accepts floats or dual numbers componentwise.
    """
    p = P_NORM if p is None else p
    qx, qy, nu = x[0], x[1], x[2]
    out = []
    for _, bx, by, ax, ay, c in OBSTACLES:
        out.append(_pnorm2(ax * (qx - bx), ay * (qy - by), p) - c)
    _, bx, by, ax, ay, c = WALL
    out.append(c - _pnorm2(ax * (qx - bx), ay * (qy - by), p))
    out.append(NU_MAX - nu)
    out.append(nu - NU_MIN)
    return out


def h_np(X, p=None):
    """Vectorised h_1..h_9. X: (..., 4) -> (..., 9)."""
    p = P_NORM if p is None else p
    X = np.asarray(X, dtype=float)
    qx, qy, nu = X[..., 0], X[..., 1], X[..., 2]
    cols = []
    for _, bx, by, ax, ay, c in OBSTACLES:
        cols.append((np.abs(ax * (qx - bx)) ** p + np.abs(ay * (qy - by)) ** p) ** (1.0 / p) - c)
    _, bx, by, ax, ay, c = WALL
    cols.append(c - (np.abs(ax * (qx - bx)) ** p + np.abs(ay * (qy - by)) ** p) ** (1.0 / p))
    cols.append(NU_MAX - nu)
    cols.append(nu - NU_MIN)
    return np.stack(cols, axis=-1)


# --------------------------------------------------------------------------------------- #
#  Batched cascade jet: beta_j, L_f beta_j, L_g beta_j for every j, via the dual seed
# --------------------------------------------------------------------------------------- #
_G_COLS = (np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
D_RING = max(RELATIVE_DEGREES)     # = 2 for this map


def _f_dual(z):
    """Unicycle drift on dual numbers (same straight-line program as Unicycle.f)."""
    zero = z[0] * 0.0
    return [z[2] * z[3].cos(), z[2] * z[3].sin(), zero, zero]


def beta_jet(X, p=None):
    """beta_j, L_f beta_j, L_g beta_j for every j, for a batch of states.

    X : (..., 4)  ->  beta (..., 9), Lf_beta (..., 9), Lg_beta (..., 9, 2)

    One seeded pass per input column (m = 2 passes total, not m+1: the drift jet rides along
    in the eps2^0 coefficients of either pass).  Constraints of different relative degree stop
    at their own d_j and are read off the SAME pass.
    """
    from .._dual import seed_batched, lie_f, value, Lf, Lg

    p = P_NORM if p is None else p
    X = np.asarray(X, dtype=float)
    batch = X.shape[:-1]
    beta = np.zeros(batch + (ELL,))
    Lf_beta = np.zeros(batch + (ELL,))
    Lg_beta = np.zeros(batch + (ELL, 2))
    for c, gcol in enumerate(_G_COLS):
        Z = seed_batched(_f_dual, X, gcol, D_RING)
        hs = h_list(Z, p)
        for j, h in enumerate(hs):
            dj = RELATIVE_DEGREES[j]
            P = h
            for i in range(dj - 1):
                P = lie_f(P) + P * ALPHA0_GAINS[j]      # b_{i+1} = L_f b_i + alpha_{j,i}(b_i)
            if c == 0:
                beta[..., j] = value(P)
                Lf_beta[..., j] = Lf(P, 1)
            Lg_beta[..., j, c] = Lg(P, 0)
    return beta, Lf_beta, Lg_beta


def alpha(s):
    """The outer class-K function alpha(h) = 0.5 h (paper, Sec. VI)."""
    return ALPHA_GAIN * s


STATE_BOX = np.array([[-10.5, 10.5],      # qx
                      [-10.5, 10.5],      # qy
                      [NU_MIN, NU_MAX],   # nu
                      [-np.pi, np.pi]])   # theta
