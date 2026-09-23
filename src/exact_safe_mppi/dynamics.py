"""Control-affine plants  xdot = f(x) + g(x) u.

Every model here is written so that the *same* callables can be evaluated on plain
floats/ndarrays and on the dual/truncated-polynomial-dual numbers of `dual.py`
(operator overloading only, no numpy ufuncs on the state components).
"""
import numpy as np


class Unicycle:
    """Nonholonomic ground robot of GS-MPPI (Rabiee & Hoagg, arXiv:2410.02154, Sec. VI).

        x = [qx, qy, nu, theta],  u = [u1, u2]
        f(x) = [nu*cos(theta), nu*sin(theta), 0, 0]
        g(x) = [[0,0],[0,0],[1,0],[0,1]]
    """

    n = 4
    m = 2
    state_names = ("qx", "qy", "nu", "theta")

    @staticmethod
    def f(x):
        qx, qy, nu, th = x[0], x[1], x[2], x[3]
        # `cos`/`sin` dispatch: dual types expose .cos()/.sin(); floats go through numpy.
        c = th.cos() if hasattr(th, "cos") else np.cos(th)
        s = th.sin() if hasattr(th, "sin") else np.sin(th)
        zero = 0.0 * nu
        return [nu * c, nu * s, zero, zero]

    @staticmethod
    def g(x):
        # constant input matrix; returned as a list of columns
        z = 0.0 * x[0]
        one = z + 1.0
        return [[z, z, one, z], [z, z, z, one]]

    @staticmethod
    def f_np(X):
        """Vectorised drift for arrays of shape (..., 4)."""
        X = np.asarray(X)
        nu, th = X[..., 2], X[..., 3]
        return np.stack([nu * np.cos(th), nu * np.sin(th),
                         np.zeros_like(nu), np.zeros_like(nu)], axis=-1)

    @staticmethod
    def g_np(X):
        X = np.asarray(X)
        G = np.zeros(X.shape[:-1] + (4, 2))
        G[..., 2, 0] = 1.0
        G[..., 3, 1] = 1.0
        return G


class SingleIntegrator:
    n = 2
    m = 2
    state_names = ("x1", "x2")

    @staticmethod
    def f(x):
        return [0.0 * x[0], 0.0 * x[1]]

    @staticmethod
    def g(x):
        z = 0.0 * x[0]
        one = z + 1.0
        return [[one, z], [z, one]]

    @staticmethod
    def f_np(X):
        return np.zeros_like(np.asarray(X, dtype=float))

    @staticmethod
    def g_np(X):
        X = np.asarray(X)
        G = np.zeros(X.shape[:-1] + (2, 2))
        G[..., 0, 0] = 1.0
        G[..., 1, 1] = 1.0
        return G


class DoubleIntegrator:
    """x = [q1, q2, v1, v2], u = acceleration."""

    n = 4
    m = 2
    state_names = ("q1", "q2", "v1", "v2")

    @staticmethod
    def f(x):
        z = 0.0 * x[0]
        return [x[2], x[3], z, z]

    @staticmethod
    def g(x):
        z = 0.0 * x[0]
        one = z + 1.0
        return [[z, z, one, z], [z, z, z, one]]

    @staticmethod
    def f_np(X):
        X = np.asarray(X)
        return np.stack([X[..., 2], X[..., 3],
                         np.zeros_like(X[..., 0]), np.zeros_like(X[..., 0])], axis=-1)

    @staticmethod
    def g_np(X):
        X = np.asarray(X)
        G = np.zeros(X.shape[:-1] + (4, 2))
        G[..., 2, 0] = 1.0
        G[..., 3, 1] = 1.0
        return G
