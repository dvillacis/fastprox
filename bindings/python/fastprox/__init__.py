"""fastprox: exact proximal operators and projections with derivatives.

Every operator computes

    prox_{gamma g}(x) = argmin_z  g(z) + ||z - x||^2 / (2 gamma)

along one axis of a float32 or float64 array, batched over the other axes.
Projections ignore ``gamma``. One-shot functions (``tv1d``, ``project_simplex``,
...) return the result; operator classes (``TV1D``, ``Simplex``, ...) also give
Jacobian-vector and vector-Jacobian products for bilevel and implicit
differentiation::

    op = fastprox.TV1D(weights=w)
    z, ctx = op.prox(x, gamma, keep=True)
    dz = op.jvp(ctx, dx)
    gx, grads = op.vjp(ctx, g)     # grads.gamma, grads.weights
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _fastprox as _C

__version__ = _C.version()

__all__ = [
    "Info", "Grads", "Context", "Operator",
    "Soft", "ElasticNet", "Huber", "GroupL2", "Box", "NonNeg", "L2Ball", "Simplex",
    "L1Ball", "LInf", "Isotonic", "TV1D",
    "soft_threshold", "elastic_net", "huber", "group_soft_threshold", "prox_l2",
    "project_box", "project_nonneg", "project_l2_ball", "project_simplex",
    "project_l1_ball", "prox_linf", "isotonic", "tv1d", "has_openmp",
]

_TV1D_METHODS = {"auto": 0, "condat": 1, "taut_string": 2}


def has_openmp() -> bool:
    """True when the compiled core runs rows in parallel."""
    return _C.has_openmp()


class Info(NamedTuple):
    """Certificate of a prox call: duality gap (0 for exact operators),
    iterations, and number of entries that sat on a tie."""

    gap: float
    iters: int
    ties: int


class Grads:
    """Parameter gradients returned by :meth:`Operator.vjp`.

    ``gamma`` is the gradient with respect to the step (with respect to sigma
    after ``prox_conj``), ``params`` the flat gradient with respect to the
    operator's parameter vector, and each named parameter is an attribute.
    """

    def __init__(self, gamma: float, params: np.ndarray, fields: dict):
        self.gamma = gamma
        self.params = params
        self._fields = fields
        for k, v in fields.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in {"gamma": self.gamma, **self._fields}.items())
        return f"Grads({items})"


class Context:
    """Structure saved by ``prox(..., keep=True)``: the input to ``jvp`` and
    ``vjp``. Pass it back as ``ctx=`` to reuse its memory in the next call."""

    __slots__ = ("op", "ws", "shape", "dtype", "axis", "threads", "conj", "step", "info")

    def __init__(self, op, ws, shape, dtype, axis, threads):
        self.op = op
        self.ws = ws
        self.shape = shape
        self.dtype = dtype
        self.axis = axis
        self.threads = threads
        self.conj = False
        self.step = None
        self.info = None


def _input(x, copy: bool) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.dtype == np.float64 or x.dtype == np.float32:
            return x
        if copy:
            return x.astype(np.float64)
        raise TypeError(
            f"fastprox operates on float32 and float64 arrays, got {x.dtype}; "
            "pass copy=True to convert"
        )
    return np.asarray(x, dtype=np.float64)


def _output(x: np.ndarray, out) -> np.ndarray:
    if out is None:
        return np.empty(x.shape, dtype=x.dtype)
    if not isinstance(out, np.ndarray) or out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("out must be an ndarray with the shape and dtype of the input")
    return out


def _vec(v):
    if v is None:
        return None
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)


class Operator:
    """Base class: a compiled operator with fixed parameters.

    Subclasses set ``_layout``, a list of ``(name, length, scalar)`` entries
    that split the flat parameter vector into named fields.
    """

    _layout: list = []

    def __init__(self, handle):
        self._op = handle

    @property
    def name(self) -> str:
        return self._op.name

    @property
    def nparams(self) -> int:
        """Length of the parameter vector theta."""
        return self._op.nparams

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"

    # -- forward -----------------------------------------------------------
    def prox(self, x, gamma=1.0, *, axis=-1, out=None, threads=0, keep=False, ctx=None,
             copy=False):
        """prox_{gamma g}(x). With ``keep=True`` returns ``(z, ctx)``."""
        return self._run(x, gamma, axis, out, threads, keep, ctx, copy, conj=False)

    def prox_conj(self, y, sigma=1.0, *, axis=-1, out=None, threads=0, keep=False, ctx=None,
                  copy=False):
        """prox_{sigma g*}(y) = y - sigma prox_{g/sigma}(y/sigma) (Moreau)."""
        return self._run(y, sigma, axis, out, threads, keep, ctx, copy, conj=True)

    def _run(self, x, step, axis, out, threads, keep, ctx, copy, conj):
        x = _input(x, copy)
        z = _output(x, out)
        if not keep:
            _C.prox(self._op, float(step), x, z, axis, threads, False, None, conj)
            return z
        if ctx is None or ctx.op is not self or ctx.shape != x.shape or ctx.dtype != x.dtype \
                or ctx.axis != axis or ctx.threads != threads:
            ctx = Context(self, _C.workspace(self._op, x, axis, threads, True), x.shape,
                          x.dtype, axis, threads)
        ctx.info = Info(*_C.prox(self._op, float(step), x, z, axis, threads, True, ctx.ws, conj))
        ctx.conj = conj
        ctx.step = float(step)
        return z, ctx

    # -- derivatives -------------------------------------------------------
    def jvp(self, ctx: Context, dx=None, dgamma=0.0, dparams=None, *, out=None):
        """J_x dx + (dz/dgamma) dgamma + (dz/dtheta) dparams at the saved point."""
        self._check_ctx(ctx)
        if dx is not None:
            dx = _input(dx, True).astype(ctx.dtype, copy=False)
        dout = np.empty(ctx.shape, ctx.dtype) if out is None else out
        if dparams is not None:
            dparams = self._flatten(dparams)
        _C.jvp(self._op, ctx.ws, dx, float(dgamma), dparams, dout)
        return dout

    def vjp(self, ctx: Context, g, *, out=None):
        """(J_x^T g, Grads) at the saved point."""
        self._check_ctx(ctx)
        g = _input(g, True).astype(ctx.dtype, copy=False)
        gx = np.empty(ctx.shape, ctx.dtype) if out is None else out
        gp = np.zeros(self.nparams, np.float64)
        gg = _C.vjp(self._op, ctx.ws, g, gx, gp if self.nparams else None)
        return gx, Grads(gg, gp, self._fields(gp))

    def _check_ctx(self, ctx):
        if not isinstance(ctx, Context) or ctx.op is not self:
            raise ValueError("ctx must come from prox(..., keep=True) on this operator")

    def _fields(self, gp: np.ndarray) -> dict:
        fields, k = {}, 0
        for name, length, scalar in self._layout:
            part = gp[k:k + length]
            fields[name] = float(part[0]) if scalar else part
            k += length
        return fields

    def _flatten(self, dparams) -> np.ndarray:
        if isinstance(dparams, dict):
            parts = []
            for name, length, _ in self._layout:
                v = np.broadcast_to(np.asarray(dparams.get(name, 0.0), np.float64), (length,))
                parts.append(v)
            return np.concatenate(parts) if parts else np.zeros(0)
        return _vec(dparams)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class Soft(Operator):
    """g(z) = lam * sum_i w_i |z_i| (soft thresholding). Parameters: weights
    (scalar when not given); ``grads.lam`` is also provided."""

    def __init__(self, lam=1.0, weights=None):
        w = _vec(weights)
        super().__init__(_C.soft(float(lam), w))
        self.lam = float(lam)
        self._w = np.ones(1) if w is None else w
        self._layout = [("weights", self._w.size, w is None)]

    def _fields(self, gp):
        f = super()._fields(gp)
        f["lam"] = float(self._w @ gp) / self.lam if self.lam > 0 else float("nan")
        return f


class ElasticNet(Operator):
    """g(z) = l1 ||z||_1 + (l2/2) ||z||^2. Parameters: l1, l2."""

    _layout = [("l1", 1, True), ("l2", 1, True)]

    def __init__(self, l1=1.0, l2=1.0):
        super().__init__(_C.elastic_net(float(l1), float(l2)))


class Huber(Operator):
    """g(z) = lam * sum_i h_delta(z_i), h_delta(t) = t^2/(2 delta) for
    |t| <= delta and |t| - delta/2 otherwise. Parameters: lam, delta."""

    _layout = [("lam", 1, True), ("delta", 1, True)]

    def __init__(self, lam=1.0, delta=1.0):
        super().__init__(_C.huber(float(lam), float(delta)))


class GroupL2(Operator):
    """g(z) = lam * sum_g c_g ||z_g||_2 along the axis. ``groups`` gives a
    group id per entry (None: one group, the l2 norm). Parameters:
    group_weights."""

    def __init__(self, lam=1.0, groups=None, group_weights=None):
        if groups is not None:
            groups = np.ascontiguousarray(groups, dtype=np.int64).reshape(-1)
            ngroups = int(groups.max()) + 1 if groups.size else 0
        else:
            ngroups = 1
        c = _vec(group_weights)
        super().__init__(_C.group_l2(float(lam), groups, ngroups, c))
        self._layout = [("group_weights", 1 if c is None else c.size, c is None)]


class Box(Operator):
    """Projection onto lower <= z <= upper (scalars or per-entry along the
    axis; infinities allowed). Parameters: lower, upper."""

    def __init__(self, lower=-np.inf, upper=np.inf):
        lo = np.atleast_1d(np.asarray(lower, np.float64))
        hi = np.atleast_1d(np.asarray(upper, np.float64))
        super().__init__(_C.box(0.0, 0.0, _vec(lo), _vec(hi)))
        self._layout = [("lower", lo.size, np.ndim(lower) == 0),
                        ("upper", hi.size, np.ndim(upper) == 0)]


class NonNeg(Operator):
    """Projection onto the nonnegative orthant."""

    def __init__(self):
        super().__init__(_C.nonneg())


class L2Ball(Operator):
    """Projection onto ||z||_2 <= radius along the axis. Parameters: radius."""

    _layout = [("radius", 1, True)]

    def __init__(self, radius=1.0):
        super().__init__(_C.l2_ball(float(radius)))


class Simplex(Operator):
    """Projection onto {z >= 0, sum z = radius} along the axis (Condat 2016).
    Parameters: radius."""

    _layout = [("radius", 1, True)]

    def __init__(self, radius=1.0):
        super().__init__(_C.simplex(float(radius)))


class L1Ball(Operator):
    """Projection onto sum_i w_i |z_i| <= radius along the axis.
    Parameters: radius."""

    _layout = [("radius", 1, True)]

    def __init__(self, radius=1.0, weights=None):
        super().__init__(_C.l1_ball(float(radius), _vec(weights)))


class LInf(Operator):
    """g(z) = lam ||z||_inf along the axis. Parameters: lam."""

    _layout = [("lam", 1, True)]

    def __init__(self, lam=1.0):
        super().__init__(_C.linf(float(lam)))


class Isotonic(Operator):
    """Projection onto the monotone cone along the axis (pool-adjacent-
    violators), minimising sum_i w_i (z_i - x_i)^2. Parameters: weights."""

    def __init__(self, weights=None, increasing=True):
        w = _vec(weights)
        super().__init__(_C.isotonic(w, bool(increasing)))
        self._layout = [] if w is None else [("weights", w.size, False)]


class TV1D(Operator):
    """g(z) = sum_i w_i |z_{i+1} - z_i| along the axis (1D total variation).
    ``weights``: scalar or one per edge (axis length - 1). ``method``: "auto"
    (Condat's algorithm, taut string when it backtracks too much), "condat"
    or "taut_string". Parameters: weights."""

    def __init__(self, weights=None, method="auto"):
        w = _vec(weights)
        if method not in _TV1D_METHODS:
            raise ValueError(f"method must be one of {sorted(_TV1D_METHODS)}")
        super().__init__(_C.tv1d(w, _TV1D_METHODS[method]))
        self._layout = [("weights", 1 if w is None else w.size,
                         weights is None or np.ndim(weights) == 0)]


# ---------------------------------------------------------------------------
# One-shot functions
# ---------------------------------------------------------------------------


def _once(handle, x, step, axis, out, threads, copy):
    x = _input(x, copy)
    z = _output(x, out)
    _C.prox(handle, float(step), x, z, axis, threads, False, None, False)
    return z


def soft_threshold(x, lam=1.0, gamma=1.0, *, weights=None, out=None, axis=-1, threads=0,
                   copy=False):
    """prox of gamma * lam * sum_i w_i |z_i|."""
    return _once(_C.soft(float(lam), _vec(weights)), x, gamma, axis, out, threads, copy)


def elastic_net(x, l1=1.0, l2=1.0, gamma=1.0, *, out=None, threads=0, copy=False):
    """prox of gamma * (l1 ||z||_1 + l2/2 ||z||^2)."""
    return _once(_C.elastic_net(float(l1), float(l2)), x, gamma, -1, out, threads, copy)


def huber(x, lam=1.0, delta=1.0, gamma=1.0, *, out=None, threads=0, copy=False):
    """prox of gamma * lam * sum_i h_delta(z_i)."""
    return _once(_C.huber(float(lam), float(delta)), x, gamma, -1, out, threads, copy)


def group_soft_threshold(x, lam=1.0, gamma=1.0, *, groups=None, group_weights=None, axis=-1,
                         out=None, threads=0, copy=False):
    """prox of gamma * lam * sum_g c_g ||z_g||_2 along the axis."""
    op = GroupL2(lam, groups, group_weights)
    return _once(op._op, x, gamma, axis, out, threads, copy)


def prox_l2(x, lam=1.0, gamma=1.0, *, axis=-1, out=None, threads=0, copy=False):
    """prox of gamma * lam * ||z||_2 along the axis."""
    return _once(_C.group_l2(float(lam), None, 1, None), x, gamma, axis, out, threads, copy)


def project_box(x, lower=-np.inf, upper=np.inf, *, axis=-1, out=None, threads=0, copy=False):
    """Projection onto lower <= z <= upper."""
    return _once(Box(lower, upper)._op, x, 1.0, axis, out, threads, copy)


def project_nonneg(x, *, out=None, threads=0, copy=False):
    """Projection onto z >= 0."""
    return _once(_C.nonneg(), x, 1.0, -1, out, threads, copy)


def project_l2_ball(x, radius=1.0, *, axis=-1, out=None, threads=0, copy=False):
    """Projection onto ||z||_2 <= radius along the axis."""
    return _once(_C.l2_ball(float(radius)), x, 1.0, axis, out, threads, copy)


def project_simplex(x, radius=1.0, *, axis=-1, out=None, threads=0, copy=False):
    """Projection onto {z >= 0, sum z = radius} along the axis."""
    return _once(_C.simplex(float(radius)), x, 1.0, axis, out, threads, copy)


def project_l1_ball(x, radius=1.0, *, weights=None, axis=-1, out=None, threads=0, copy=False):
    """Projection onto sum_i w_i |z_i| <= radius along the axis."""
    return _once(_C.l1_ball(float(radius), _vec(weights)), x, 1.0, axis, out, threads, copy)


def prox_linf(x, lam=1.0, gamma=1.0, *, axis=-1, out=None, threads=0, copy=False):
    """prox of gamma * lam * ||z||_inf along the axis."""
    return _once(_C.linf(float(lam)), x, gamma, axis, out, threads, copy)


def isotonic(x, *, weights=None, increasing=True, axis=-1, out=None, threads=0, copy=False):
    """Isotonic regression (projection onto the monotone cone) along the axis."""
    return _once(_C.isotonic(_vec(weights), bool(increasing)), x, 1.0, axis, out, threads, copy)


def tv1d(x, gamma=1.0, *, weights=None, axis=-1, method="auto", out=None, threads=0,
         copy=False):
    """prox of gamma * sum_i w_i |z_{i+1} - z_i| along the axis."""
    if method not in _TV1D_METHODS:
        raise ValueError(f"method must be one of {sorted(_TV1D_METHODS)}")
    return _once(_C.tv1d(_vec(weights), _TV1D_METHODS[method]), x, gamma, axis, out, threads,
                 copy)
