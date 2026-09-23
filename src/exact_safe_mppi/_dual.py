"""Dual-number and truncated-polynomial-dual arithmetic.

Implements the bi-graded ring of Proposition 3,

    D_(d,1)  =  R[eps1, eps2] / (eps1^(d+1), eps2^2),

whose elements are  z = sum_{k=0}^{d} (a_k + b_k eps2) eps1^k,  a_k, b_k in R.

An element is stored as a real array `c` of shape (d+1, 2), with c[k,0] = a_k and c[k,1] = b_k.
Arrays may carry leading batch axes, so one object represents a whole batch of dual numbers and
every operation is vectorised; shape is (..., d+1, 2).

WHY THIS RING.  Seeding it with the Taylor jet of the flow of f, perturbed along one input
column g, makes a single unmodified evaluation of h return

    h~(z_g) = sum_{k=0}^{d} (1/k!) [ L_f^k h(x)  +  L_g L_f^k h(x) eps2 ] eps1^k,

so ALL iterated drift derivatives and ALL control couplings up to order d come out of one
forward pass per input column.  Constraints of different relative degree read off different
coefficients of the same pass, which is what lets mixed d_j be handled together.

The cascade b_{j,i+1} = L_f b_{j,i} + alpha_{j,i}(b_{j,i}) is then evaluated *inside* the ring:
L_f is the coefficient shift `lie_f` below, and alpha is applied by its dual extension.  Each
shift consumes one eps1 order, so d = max_j d_j suffices to deliver beta_j, L_f beta_j and
L_g beta_j for every j.
"""
import math
import numpy as np

_factorial = math.factorial

__all__ = ["TPD", "const", "var_seed", "flow_seed", "lie_f", "value", "Lf", "Lg"]


class TPD:
    """Element(s) of D_(d,1). `c` has shape (..., d+1, 2)."""

    __slots__ = ("c",)
    __array_priority__ = 1000

    def __init__(self, c):
        c = np.asarray(c, dtype=float)
        if c.ndim < 2 or c.shape[-1] != 2:
            raise ValueError(f"TPD coefficient array must have shape (..., d+1, 2), got {c.shape}")
        self.c = c

    # ---- basic structure -------------------------------------------------------------
    @property
    def d(self):
        return self.c.shape[-2] - 1

    @property
    def batch(self):
        return self.c.shape[:-2]

    def real(self):
        """The eps1^0 eps2^0 coefficient: the value of the function itself."""
        return self.c[..., 0, 0]

    def copy(self):
        return TPD(self.c.copy())

    def __repr__(self):
        return f"TPD(d={self.d}, batch={self.batch}, real={np.asarray(self.real())})"

    # ---- ring operations -------------------------------------------------------------
    def _coerce(self, other):
        if isinstance(other, TPD):
            if other.d != self.d:
                raise ValueError(f"order mismatch: {self.d} vs {other.d}")
            return other
        arr = np.asarray(other, dtype=float)
        c = np.zeros(np.broadcast_shapes(self.batch, arr.shape) + (self.d + 1, 2))
        c[..., 0, 0] = arr
        return TPD(c)

    def __add__(self, o):
        o = self._coerce(o)
        return TPD(self.c + o.c)

    __radd__ = __add__

    def __neg__(self):
        return TPD(-self.c)

    def __sub__(self, o):
        return self + (-self._coerce(o))

    def __rsub__(self, o):
        return self._coerce(o) + (-self)

    def __mul__(self, o):
        o = self._coerce(o)
        d = self.d
        a, b = self.c[..., 0], self.c[..., 1]      # (..., d+1)
        p, q = o.c[..., 0], o.c[..., 1]
        shape = np.broadcast_shapes(a.shape[:-1], p.shape[:-1]) + (d + 1,)
        ra = np.zeros(shape)
        rb = np.zeros(shape)
        for i in range(d + 1):                      # truncated Cauchy product in eps1
            for j in range(d + 1 - i):
                ra[..., i + j] += a[..., i] * p[..., j]
                rb[..., i + j] += a[..., i] * q[..., j] + b[..., i] * p[..., j]
        return TPD(np.stack([ra, rb], axis=-1))

    __rmul__ = __mul__

    def _apply(self, derivs):
        """Dual extension of a scalar function phi: phi~(z) = sum_j phi^(j)(a0)/j! * n^j,
        where a0 is the eps1^0 eps2^0 coefficient and n = z - a0 is nilpotent with
        n^(d+2) = 0.  `derivs` is a list [phi(a0), phi'(a0), ...] of length d+2.
        """
        a0 = self.real()
        n = self.copy()
        n.c[..., 0, 0] = 0.0
        out = TPD(np.zeros(np.broadcast_shapes(self.batch, a0.shape) + (self.d + 1, 2)))
        out.c[..., 0, 0] = derivs[0]
        term = None
        fact = 1.0
        for j in range(1, len(derivs)):
            term = n if term is None else term * n
            fact *= j
            out = out + term * (derivs[j] / fact)
        return out

    def __truediv__(self, o):
        if isinstance(o, TPD):
            return self * o.inv()
        return self * (1.0 / np.asarray(o, dtype=float))

    def __rtruediv__(self, o):
        return self.inv() * o

    def inv(self):
        a0 = self.real()
        if np.any(a0 == 0.0):
            raise ZeroDivisionError("TPD.inv: real part must be nonzero (unit condition)")
        # phi(t) = 1/t  =>  phi^(j)(t) = (-1)^j j! t^{-(j+1)}
        return self._apply([((-1.0) ** j) * float(_factorial(j)) * a0 ** (-(j + 1))
                            for j in range(self.d + 2)])

    def __pow__(self, p):
        """z**p.  Guards the case of an INTEGER exponent evaluated at a zero base.

        For integer p the derivatives of order j > p vanish identically, so the coefficient
        `coef` is already 0 there -- but `a0 ** (p - j)` is then `0 ** (negative)` = inf, and
        `0 * inf` is nan.  The whole jet becomes nan.  This is not hypothetical: it fires on any
        barrier of the form (q - b)**2 evaluated exactly at q = b, which is precisely the centre
        line of a symmetric gap, i.e. the states an experiment about gaps samples most.
        """
        p = float(p)
        a0 = self.real()
        derivs, coef = [], 1.0
        for j in range(self.d + 2):
            if coef == 0.0:
                derivs.append(np.zeros_like(np.asarray(a0, dtype=float)))
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    derivs.append(coef * a0 ** (p - j))
            coef *= (p - j)
        return self._apply(derivs)

    # ---- elementary functions ---------------------------------------------------------
    def exp(self):
        e = np.exp(self.real())
        return self._apply([e] * (self.d + 2))

    def log(self):
        a0 = self.real()
        derivs = [np.log(a0)]
        for j in range(1, self.d + 2):
            derivs.append(((-1.0) ** (j - 1)) * float(_factorial(j - 1)) * a0 ** (-j))
        return self._apply(derivs)

    def sqrt(self):
        return self ** 0.5

    def sin(self):
        a0 = self.real()
        cyc = [np.sin(a0), np.cos(a0), -np.sin(a0), -np.cos(a0)]
        return self._apply([cyc[j % 4] for j in range(self.d + 2)])

    def cos(self):
        a0 = self.real()
        cyc = [np.cos(a0), -np.sin(a0), -np.cos(a0), np.sin(a0)]
        return self._apply([cyc[j % 4] for j in range(self.d + 2)])

    def __abs__(self):
        """|z| for units: sign(a0) * z.  Undefined at a0 = 0, where |.| is not differentiable."""
        a0 = self.real()
        if np.any(a0 == 0.0):
            raise ValueError("TPD.__abs__: not differentiable at real part 0")
        return self * np.sign(a0)


# --------------------------------------------------------------------------------------- #
#  Constructors and the seed
# --------------------------------------------------------------------------------------- #
def const(value, d, batch=()):
    c = np.zeros(tuple(batch) + (d + 1, 2))
    c[..., 0, 0] = value
    return TPD(c)


def var_seed(x, gcol, d):
    """c_0 = x + gcol * eps2, embedded at eps1^0.  x, gcol are length-n sequences."""
    n = len(x)
    out = []
    for i in range(n):
        c = np.zeros((d + 1, 2))
        c[0, 0] = x[i]
        c[0, 1] = gcol[i]
        out.append(TPD(c))
    return out


def flow_seed(f, x, gcol, d):
    """Order-d bi-graded flow seed, built by Kamaldar's Remark 11 recursion.

    NO SYMBOLIC DIFFERENTIATION: the Taylor coefficients of the flow of xdot = f(x) obey
        c_0 = x + g eps2,      c_{k+1} = (1/(k+1)) * [ f~( sum_i c_i eps1^i ) ]_{eps1^k},
    with the arithmetic carried out in D_(d,1) on the unmodified straight-line program of f.
    Setting gcol = 0 reproduces the drift-only seed.

    NOTE.  The naive seed  x + f(x) eps1 + g eps2  is NOT sufficient for d >= 2: it omits the
    higher flow iterates f^[k], k >= 2.  It happens to be correct when f^[k] = 0 for k >= 2
    (e.g. the planar double integrator), which is why that case can look deceptively simple.
    """
    n = len(x)
    Z = var_seed(x, gcol, d)                      # currently only the eps1^0 coefficient
    for k in range(d):
        fZ = f(Z)                                  # evaluate f in the ring
        for i in range(n):
            ck1 = fZ[i].c[..., k, :] / (k + 1.0)   # coefficient of eps1^k, divided by k+1
            Z[i].c[..., k + 1, :] = ck1
    return Z


def lie_f(z):
    """Shift realising L_f on a jet: if z holds the jet of phi, lie_f(z) holds that of L_f phi.

    If z = sum_k (1/k!) (L_f^k phi + L_g L_f^k phi eps2) eps1^k, then the jet of L_f phi is
    sum_k (1/k!) (L_f^{k+1} phi + ...) eps1^k, i.e. coefficient k of the result is
    (k+1) times coefficient k+1 of the input.  The top order is lost, so each application
    costs one eps1 order.
    """
    d = z.d
    out = np.zeros_like(z.c)
    for k in range(d):
        out[..., k, :] = (k + 1.0) * z.c[..., k + 1, :]
    return TPD(out)


def value(z):
    return z.c[..., 0, 0]


def Lf(z, k=1):
    """L_f^k phi(x) = k! * (coefficient of eps1^k, eps2^0)."""
    return float(_factorial(k)) * z.c[..., k, 0]


def Lg(z, k=0):
    """L_g L_f^k phi(x) = k! * (coefficient of eps1^k eps2^1), for the seeded input column."""
    return float(_factorial(k)) * z.c[..., k, 1]


def seed_batched(f, X, gcol, d):
    """Order-d bi-graded flow seed for a BATCH of states.

    X    : (..., n) real states
    gcol : (n,) or (..., n) one input column
    Returns a list of n TPD objects, each with batch shape X.shape[:-1].

    Same recursion as `flow_seed` (Kamaldar Rem. 11), vectorised.  Phase 2 evaluates every
    rollout sample through this, never one state at a time: the per-state cost falls by more
    than two orders of magnitude when the batch axis is used.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[-1]
    batch = X.shape[:-1]
    g = np.broadcast_to(np.asarray(gcol, dtype=float), batch + (n,))
    Z = []
    for i in range(n):
        c = np.zeros(batch + (d + 1, 2))
        c[..., 0, 0] = X[..., i]
        c[..., 0, 1] = g[..., i]
        Z.append(TPD(c))
    for k in range(d):
        fZ = f(Z)
        for i in range(n):
            Z[i].c[..., k + 1, :] = fZ[i].c[..., k, :] / (k + 1.0)
    return Z
