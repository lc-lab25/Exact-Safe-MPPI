"""Two-disk gap: a NON-DEGENERATE gap environment, adapted from Kamaldar Example 4.

Purpose.  In the parallel-wall corridor (Corollary 2) the unicycle cascade gives
beta_1 + beta_2 == 2*a*w identically -- state independent.  That degeneracy is what makes the
closed-form threshold rho_crit = ln2/(a*w) available, and a referee may reasonably call it
contrived.  Here beta_1 + beta_2 is genuinely state dependent, so the threshold has no closed
form and is predicted numerically instead.  If the one-sided bound survives here too, the
corridor result is not an artifact of that algebra.

Plant (Kamaldar Ex. 4): planar double integrator
    x = [q_x, q_y, v_x, v_y],  qdot = v,  vdot = u
    f(x) = [v_x, v_y, 0, 0],   g = [[0,0],[0,0],[1,0],[0,1]]

Obstacles: two disks of radius r centred at o_1 = (0, r+w), o_2 = (0, -(r+w)), leaving a gap
of half-width w at the origin.  Kamaldar's numbers are r = 0.95, o = (0, +/-1), i.e. w = 0.05.

    h_i(x) = ||q - o_i||^2 - r^2          (relative degree 2: L_g h_i = 0, L_g L_f h_i = 2(q-o_i)^T)
    beta_i = L_f h_i + a*h_i = 2 (q - o_i)^T v + a * (||q - o_i||^2 - r^2)

The gap-crossing slice is  X_gap = {x : q_x = 0, |q_y| <= w}: any trajectory passing from
q_x < 0 to q_x > 0 between the disks must have its state in X_gap at the crossing instant.
"""
import numpy as np

R_DISK = 0.95           # Kamaldar Ex. 4
A_GAIN = 2.5            # alpha_{i,0}(s) = A_GAIN * s, matched to the GS-MPPI map's 2.5


def centres(w, r=R_DISK):
    d = r + w
    return np.array([0.0, d]), np.array([0.0, -d])


def h_np(X, w, r=R_DISK):
    """h_1, h_2 for X of shape (..., 4). Returns (..., 2)."""
    X = np.asarray(X, dtype=float)
    q = X[..., :2]
    o1, o2 = centres(w, r)
    return np.stack([((q - o1) ** 2).sum(-1) - r ** 2,
                     ((q - o2) ** 2).sum(-1) - r ** 2], axis=-1)


def beta_np(X, w, r=R_DISK, a=A_GAIN):
    """beta_1, beta_2 for X of shape (..., 4). Returns (..., 2)."""
    X = np.asarray(X, dtype=float)
    q, v = X[..., :2], X[..., 2:]
    o1, o2 = centres(w, r)
    out = []
    for o in (o1, o2):
        dq = q - o
        out.append(2.0 * (dq * v).sum(-1) + a * ((dq ** 2).sum(-1) - r ** 2))
    return np.stack(out, axis=-1)


def beta_mean_and_half_diff(qy, vy, w, r=R_DISK, a=A_GAIN):
    """On the crossing slice q_x = 0 (v_x drops out of both):

        m = (beta_1+beta_2)/2 = 2*q_y*v_y + a*(q_y^2 + d^2 - r^2)     [state dependent]
        s = (beta_2-beta_1)/2 = 2*d*(v_y + a*q_y)
    """
    d = r + w
    m = 2.0 * qy * vy + a * (qy ** 2 + d ** 2 - r ** 2)
    s = 2.0 * d * (vy + a * qy)
    return m, s


def naive_threshold(w, r=R_DISK, a=A_GAIN):
    """The threshold one would get by assuming the sup sits at the gap centre with v = 0:

        h_rho(0,0) = a*(d^2 - r^2) - ln2/rho = a*w*(2r + w) - ln2/rho

    This is the analogue of the corridor's closed form.  It is NOT the true threshold here --
    see disk_gap_threshold.py, which maximises over the whole slice.
    """
    return np.log(2.0) / (a * w * (2.0 * r + w))
