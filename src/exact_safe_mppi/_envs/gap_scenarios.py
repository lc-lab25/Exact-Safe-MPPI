"""R5 gap-threading scenarios: two geometries with DIFFERENT barrier units.

The point of using two is that they carry different units and different barrier forms, so a
prediction that collapses them onto one parameter-free threshold is much harder to explain away
than either curve alone (theory.tex Remark on non-dimensionalisation).

  A. DISK GAP  -- planar double integrator, squared-distance barriers (Kamaldar Ex. 4 style):
         h_i = ||q - o_i||^2 - r^2,      o_{1,2} = (0, +/-(r + w))
     relative degree 2, barrier in units of LENGTH^2.

  B. BOX GAP   -- unicycle, p-norm distance barriers (the GS-MPPI form):
         h_i = ||[q - b_i]||_p - c,      b_{1,2} = (0, +/-(c + w))
     relative degree 2, barrier in units of LENGTH.

Both are open area -> gap -> open area, so "thread the gap" is a well-posed task: the robot
starts on one side and must reach a goal on the other.  Speed limits are included so the
constraint set is bounded, as in the GS-MPPI map.

The dimensionless group is computed NUMERICALLY per geometry,
    kappa = rho * sup{ h_ex(x) : x in the crossing slice },
so no closed form is needed for either, and kappa_crit = ln 2 is parameter free.
"""
import numpy as np

from .._dual import seed_batched, lie_f, value, Lf, Lg

# shared MPPI parameters -- identical to the frozen GS-MPPI values
GAMMA = 1e24
SIGMA = np.diag([1.33, 0.33])
LAMBDA = 300.0
K_SAMPLES = 1000
N_HORIZON = 20
TS = 0.1
DT_INNER = 0.002
ALPHA_GAIN = 0.5
A0 = 2.5                    # alpha_{j,0}(s) = A0 * s for the gap constraints
NU_MAX, NU_MIN = 9.0, -1.0
P_NORM = 8


def alpha(s):
    return ALPHA_GAIN * s


# --------------------------------------------------------------------------------------- #
class DiskGap:
    """Double integrator through two disks.  x = [q1, q2, v1, v2]."""

    n, m = 4, 2
    RELATIVE_DEGREES = (2, 2, 2, 2)      # 2 gap constraints + 2 speed constraints
    GAMMA = GAMMA
    SIGMA = SIGMA
    LAMBDA = LAMBDA
    K_SAMPLES = K_SAMPLES
    N_HORIZON = N_HORIZON
    TS = TS
    DT_INNER = DT_INNER

    _G = (np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    def __init__(self, w, r=0.95):
        self.w, self.r = float(w), float(r)
        self.d = r + w
        self.RELATIVE_DEGREES = (2, 2, 1)

    @staticmethod
    def alpha(s):
        return ALPHA_GAIN * s

    def h_list(self, x):
        q1, q2, v1, v2 = x[0], x[1], x[2], x[3]
        return [q1 * q1 + (q2 - self.d) ** 2 - self.r ** 2,
                q1 * q1 + (q2 + self.d) ** 2 - self.r ** 2,
                NU_MAX ** 2 - (v1 * v1 + v2 * v2)]

    def h_np(self, X):
        X = np.asarray(X, float)
        q, v = X[..., :2], X[..., 2:]
        o1 = np.array([0.0, self.d]); o2 = np.array([0.0, -self.d])
        sp = (v * v).sum(-1)
        return np.stack([((q - o1) ** 2).sum(-1) - self.r ** 2,
                         ((q - o2) ** 2).sum(-1) - self.r ** 2,
                         NU_MAX ** 2 - sp], -1)

    @staticmethod
    def f_np(X):
        X = np.asarray(X, float)
        return np.stack([X[..., 2], X[..., 3],
                         np.zeros_like(X[..., 0]), np.zeros_like(X[..., 0])], -1)

    @staticmethod
    def g_np(X):
        X = np.asarray(X, float)
        G = np.zeros(X.shape[:-1] + (4, 2)); G[..., 2, 0] = 1.0; G[..., 3, 1] = 1.0
        return G

    @staticmethod
    def _f_dual(z):
        zero = z[0] * 0.0
        return [z[2], z[3], zero, zero]

    def beta_jet(self, X):
        X = np.asarray(X, float); batch = X.shape[:-1]
        L = len(self.RELATIVE_DEGREES)
        beta = np.zeros(batch + (L,)); Lfb = np.zeros(batch + (L,))
        Lgb = np.zeros(batch + (L, 2))
        for c, gcol in enumerate(self._G):
            Z = seed_batched(self._f_dual, X, gcol, 2)
            for j, h in enumerate(self.h_list(Z)):
                P = h
                for _ in range(self.RELATIVE_DEGREES[j] - 1):
                    P = lie_f(P) + P * A0
                if c == 0:
                    beta[..., j] = value(P); Lfb[..., j] = Lf(P, 1)
                Lgb[..., j, c] = Lg(P, 0)
        return beta, Lfb, Lgb

    def start_goal(self):
        return np.array([-3.0, -1.0, 0.0, 0.0]), np.array([3.0, -0.2])

    def crossing_slice(self, n=20000, seed=0):
        """States with q1 = 0 and |q2| <= w -- any threading trajectory passes through here."""
        rng = np.random.default_rng(seed)
        q2 = rng.uniform(-self.w, self.w, n)
        v = rng.uniform(-6, 6, (n, 2))
        return np.stack([np.zeros(n), q2, v[:, 0], v[:, 1]], 1)


# --------------------------------------------------------------------------------------- #
class BoxGap:
    """Unicycle through two p-norm boxes.  x = [qx, qy, nu, theta]."""

    n, m = 4, 2
    GAMMA = GAMMA
    SIGMA = SIGMA
    LAMBDA = LAMBDA
    K_SAMPLES = K_SAMPLES
    N_HORIZON = N_HORIZON
    TS = TS
    DT_INNER = DT_INNER

    _G = (np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    def __init__(self, w, c=1.0, p=P_NORM):
        self.w, self.c, self.p = float(w), float(c), int(p)
        self.d = c + w
        self.RELATIVE_DEGREES = (2, 2, 1, 1)

    def _pn(self, u, v):
        s = u ** self.p + v ** self.p
        if hasattr(s, "c"):
            np.maximum(s.c[..., 0, 0], 1e-30, out=s.c[..., 0, 0])
        else:
            s = np.maximum(s, 1e-30)
        return s ** (1.0 / self.p)

    @staticmethod
    def alpha(s):
        return ALPHA_GAIN * s

    def h_list(self, x):
        qx, qy, nu = x[0], x[1], x[2]
        return [self._pn(qx, qy - self.d) - self.c,
                self._pn(qx, qy + self.d) - self.c,
                NU_MAX - nu, nu - NU_MIN]

    def h_np(self, X):
        X = np.asarray(X, float)
        qx, qy, nu = X[..., 0], X[..., 1], X[..., 2]
        pn = lambda u, v: (np.abs(u) ** self.p + np.abs(v) ** self.p) ** (1.0 / self.p)
        return np.stack([pn(qx, qy - self.d) - self.c, pn(qx, qy + self.d) - self.c,
                         NU_MAX - nu, nu - NU_MIN], -1)

    @staticmethod
    def f_np(X):
        X = np.asarray(X, float); nu, th = X[..., 2], X[..., 3]
        return np.stack([nu * np.cos(th), nu * np.sin(th),
                         np.zeros_like(nu), np.zeros_like(nu)], -1)

    @staticmethod
    def g_np(X):
        X = np.asarray(X, float)
        G = np.zeros(X.shape[:-1] + (4, 2)); G[..., 2, 0] = 1.0; G[..., 3, 1] = 1.0
        return G

    @staticmethod
    def _f_dual(z):
        zero = z[0] * 0.0
        return [z[2] * z[3].cos(), z[2] * z[3].sin(), zero, zero]

    def beta_jet(self, X):
        X = np.asarray(X, float); batch = X.shape[:-1]
        L = len(self.RELATIVE_DEGREES)
        beta = np.zeros(batch + (L,)); Lfb = np.zeros(batch + (L,))
        Lgb = np.zeros(batch + (L, 2))
        for cc, gcol in enumerate(self._G):
            Z = seed_batched(self._f_dual, X, gcol, 2)
            for j, h in enumerate(self.h_list(Z)):
                P = h
                for _ in range(self.RELATIVE_DEGREES[j] - 1):
                    P = lie_f(P) + P * A0
                if cc == 0:
                    beta[..., j] = value(P); Lfb[..., j] = Lf(P, 1)
                Lgb[..., j, cc] = Lg(P, 0)
        return beta, Lfb, Lgb

    def start_goal(self):
        return np.array([-3.0, 0.0, 0.0, 0.0]), np.array([3.0, 0.0])

    def crossing_slice(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        qy = rng.uniform(-self.w, self.w, n)
        return np.stack([np.zeros(n), qy, rng.uniform(0, 7, n),
                         rng.uniform(-np.pi, np.pi, n)], 1)


def sup_slice_hex(env, n=200000, seed=0):
    """sup over the crossing slice of h_ex = min_j beta_j -- the scale in kappa = rho * this."""
    X = env.crossing_slice(n, seed)
    b, _, _ = env.beta_jet(X)
    return float(b.min(-1).max())


# --------------------------------------------------------------------------------------- #
class BoxGapDI:
    """Two p-norm boxes threaded by a DOUBLE INTEGRATOR.  x = [q1, q2, v1, v2].

    Replaces the unicycle BoxGap for R5.  Measured on the unicycle, the ceiling of the exact
    filter -- its success when the barrier is NOT the obstruction -- never exceeds 0.85, and
    reaches that only at w = 0.50 where rho_crit = 0.55, i.e. where the soft minimum is barely a
    barrier.  At the widths whose rho_crit is a value anyone would use (0.05-0.12, rho_crit
    2.3-5.5) the unicycle ceiling is 0.05-0.55, with deadlock 0.05-0.25.  The nonholonomic
    vehicle is MANOEUVRING-limited in gaps narrow enough for erosion to matter, so the threshold
    is not measurable on that plant at any usable rho.

    This keeps the DISTANCE-valued barrier (units of length, sup_slice h_ex = a*w) against the
    disk scenario's SQUARED-distance barrier (units of length^2, sup = a*w*(2r+w)).  The
    collapse therefore rests on two barrier forms and two unit systems sharing ONE plant, and
    the paper says so: it is not two independent plants.
    """

    n, m = 4, 2
    GAMMA = GAMMA
    SIGMA = SIGMA
    LAMBDA = LAMBDA
    K_SAMPLES = K_SAMPLES
    N_HORIZON = N_HORIZON
    TS = TS
    DT_INNER = DT_INNER
    _G = (np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    def __init__(self, w, c=1.0, p=P_NORM):
        self.w, self.c, self.p = float(w), float(c), int(p)
        self.d = c + w
        self.RELATIVE_DEGREES = (2, 2, 1)

    @staticmethod
    def alpha(s):
        return ALPHA_GAIN * s

    def _pn(self, u, v):
        s = u ** self.p + v ** self.p
        if hasattr(s, "c"):
            np.maximum(s.c[..., 0, 0], 1e-30, out=s.c[..., 0, 0])
        else:
            s = np.maximum(s, 1e-30)
        return s ** (1.0 / self.p)

    def h_list(self, x):
        q1, q2, v1, v2 = x[0], x[1], x[2], x[3]
        return [self._pn(q1, q2 - self.d) - self.c,
                self._pn(q1, q2 + self.d) - self.c,
                NU_MAX ** 2 - (v1 * v1 + v2 * v2)]

    def h_np(self, X):
        X = np.asarray(X, float)
        q1, q2, v = X[..., 0], X[..., 1], X[..., 2:]
        pn = lambda u, z: (np.abs(u) ** self.p + np.abs(z) ** self.p) ** (1.0 / self.p)
        return np.stack([pn(q1, q2 - self.d) - self.c, pn(q1, q2 + self.d) - self.c,
                         NU_MAX ** 2 - (v * v).sum(-1)], -1)

    @staticmethod
    def f_np(X):
        X = np.asarray(X, float)
        return np.stack([X[..., 2], X[..., 3],
                         np.zeros_like(X[..., 0]), np.zeros_like(X[..., 0])], -1)

    @staticmethod
    def g_np(X):
        X = np.asarray(X, float)
        G = np.zeros(X.shape[:-1] + (4, 2)); G[..., 2, 0] = 1.0; G[..., 3, 1] = 1.0
        return G

    @staticmethod
    def _f_dual(z):
        zero = z[0] * 0.0
        return [z[2], z[3], zero, zero]

    def beta_jet(self, X):
        X = np.asarray(X, float); batch = X.shape[:-1]
        L = len(self.RELATIVE_DEGREES)
        beta = np.zeros(batch + (L,)); Lfb = np.zeros(batch + (L,))
        Lgb = np.zeros(batch + (L, 2))
        for cc, gcol in enumerate(self._G):
            Z = seed_batched(self._f_dual, X, gcol, 2)
            for j, h in enumerate(self.h_list(Z)):
                P = h
                for _ in range(self.RELATIVE_DEGREES[j] - 1):
                    P = lie_f(P) + P * A0
                if cc == 0:
                    beta[..., j] = value(P); Lfb[..., j] = Lf(P, 1)
                Lgb[..., j, cc] = Lg(P, 0)
        return beta, Lfb, Lgb

    def start_goal(self):
        return np.array([-3.0, 0.0, 0.0, 0.0]), np.array([3.0, 0.0])

    def crossing_slice(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        return np.stack([np.zeros(n), rng.uniform(-self.w, self.w, n),
                         rng.uniform(-6, 6, n), rng.uniform(-6, 6, n)], 1)
