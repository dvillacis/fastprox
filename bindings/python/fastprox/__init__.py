"""fastprox: exact proximal operators and projections with derivatives.

Every operator computes

    prox_{gamma g}(x) = argmin_z  g(z) + ||z - x||^2 / (2 gamma)

along one axis of a float32 or float64 array, batched over the other axes.
Projections ignore ``gamma``. Image operators (``TVAniso``) act on the last
two or three axes instead and are iterative: they report a duality gap.
One-shot functions (``tv1d``, ``project_simplex``,
...) return the result; operator classes (``TV1D``, ``Simplex``, ...) also give
Jacobian-vector and vector-Jacobian products for bilevel and implicit
differentiation::

    op = fastprox.TV1D(weights=w)
    z, ctx = op.prox(x, gamma, keep=True)
    dz = op.jvp(ctx, dx)
    gx, grads = op.vjp(ctx, g)     # grads.gamma, grads.weights
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import numpy as np

from . import _fastprox as _C

__version__ = _C.version()

__all__ = [
    "Info", "IterInfo", "Grads", "Context", "Operator",
    "Soft", "ElasticNet", "Huber", "GroupL2", "Box", "NonNeg", "L2Ball", "Simplex",
    "L1Ball", "LInf", "Isotonic", "TV1D", "TVAniso", "TVIso", "tv_aniso", "tv_iso",
    "L0", "Lq", "SCAD", "MCP", "LogPenalty", "hard_threshold", "lq", "scad", "mcp", "prox_log",
    "soft_threshold", "elastic_net", "huber", "group_soft_threshold", "prox_l2",
    "project_box", "project_nonneg", "project_l2_ball", "project_simplex",
    "project_l1_ball", "prox_linf", "isotonic", "tv1d", "default_threads",
    "Max", "SumLargest", "Hyperplane", "HalfSpace", "prox_max", "prox_sum_largest",
    "project_hyperplane", "project_halfspace",
]

_TV1D_METHODS = {"auto": 0, "condat": 1, "taut_string": 2}
_TIES = {"nonzero": 0, "zero": 1}


def _tie(tie):
    if tie not in _TIES:
        raise ValueError(f"tie must be one of {sorted(_TIES)}")
    return _TIES[tie]


def default_threads() -> int:
    """Threads used when ``threads=0``: the environment variable
    ``FASTPROX_NUM_THREADS``, else the first number in ``OMP_NUM_THREADS``, else
    the CPUs this process may run on (read once). 1 for a serial build."""
    return _C.default_threads()


class Info(NamedTuple):
    """Certificate of a prox call: duality gap (0 for exact operators),
    iterations, and number of entries that sat on a tie."""

    gap: float
    iters: int
    ties: int


class IterInfo(NamedTuple):
    """Certificate of an iterative operator: the duality gap of the prox
    problem at the returned point (summed over the batch; |z - z*|^2 <= 2 gap
    per image), the iterations (the maximum over the batch), whether every
    image reached the tolerance, and the dual of shape (ndim, *x.shape), which
    ``warm_start=`` accepts back."""

    gap: float
    iters: int
    ties: int
    converged: bool
    dual: np.ndarray | None


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

    __slots__ = ("op", "ws", "shape", "dtype", "axis", "threads", "conj", "step", "info", "deriv")
    # info is an Info, or an IterInfo for iterative operators

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
        self.deriv = None  # (residual, iterations) of the last iterative derivative solve


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
        ctx = self._context(x, axis, threads, ctx)
        ctx.info = Info(*_C.prox(self._op, float(step), x, z, axis, threads, True, ctx.ws,
                                 conj)[:3])
        ctx.conj = conj
        ctx.step = float(step)
        return z, ctx

    def _context(self, x, axis, threads, ctx):
        if ctx is None or ctx.op is not self or ctx.shape != x.shape or ctx.dtype != x.dtype \
                or ctx.axis != axis or ctx.threads != threads:
            ctx = Context(self, _C.workspace(self._op, x, axis, threads, True), x.shape,
                          x.dtype, axis, threads)
        return ctx

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


class Max(Operator):
    """g(z) = lam * max_i z_i along the axis: z = min(x, tau) by the Moreau
    identity with the simplex. Parameters: lam."""

    _layout = [("lam", 1, True)]

    def __init__(self, lam=1.0):
        super().__init__(_C.max(float(lam)))


class SumLargest(Operator):
    """g(z) = lam * (sum of the k largest z_i) along the axis, k at most the
    axis length; with absolute=True the k largest |z_i| (any k; k >= n is
    lam * ||z||_1). Parameters: lam."""

    _layout = [("lam", 1, True)]

    def __init__(self, lam=1.0, k=1, absolute=False):
        super().__init__(_C.sum_largest(float(lam), operator.index(k), bool(absolute)))


class Hyperplane(Operator):
    """Projection onto {a^T z = b, lower <= z <= upper} along the axis (Beck,
    Theorem 6.27): a is a scalar or one entry per element (None: all ones),
    the box as in Box (unbounded by default). Raises ValueError when the set
    is empty for the axis length. Parameters: b, a, lower, upper."""

    _inequality = False

    def __init__(self, a=None, b=0.0, lower=-np.inf, upper=np.inf):
        av = None if a is None else np.atleast_1d(np.asarray(a, np.float64))
        lo = np.atleast_1d(np.asarray(lower, np.float64))
        hi = np.atleast_1d(np.asarray(upper, np.float64))
        super().__init__(_C.hyperplane(_vec(av), float(b), _vec(lo), _vec(hi), self._inequality))
        self._layout = [("b", 1, True),
                        ("a", 1 if av is None else av.size, a is None or np.ndim(a) == 0),
                        ("lower", lo.size, np.ndim(lower) == 0),
                        ("upper", hi.size, np.ndim(upper) == 0)]


class HalfSpace(Hyperplane):
    """Projection onto {a^T z <= b, lower <= z <= upper} along the axis; the
    box projection when it satisfies the constraint, else as Hyperplane.
    Parameters: b, a, lower, upper."""

    _inequality = True


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


class _ImageOperator(Operator):
    """Iterative operator on the last ``ndim`` axes (see :class:`TVAniso`)."""

    def __init__(self, handle, ndim):
        super().__init__(handle)
        self.ndim = int(ndim)
        self._layout = [("weights", self.ndim, False)]  # one gradient per axis

    def prox(self, x, gamma=1.0, *, out=None, threads=0, keep=False, ctx=None, copy=False,
             tol=1e-6, max_iter=1000, warm_start=None, warm=False):
        """prox_{gamma g}(x); see the class docstring."""
        z, ctx, _ = self._iter(x, gamma, out, threads, keep, ctx, copy, False, tol, max_iter,
                               warm_start, warm)
        return (z, ctx) if keep else z

    def prox_conj(self, y, sigma=1.0, *, out=None, threads=0, keep=False, ctx=None, copy=False,
                  tol=1e-6, max_iter=1000, warm_start=None, warm=False):
        """prox_{sigma g*}(y) = y - sigma prox_{g/sigma}(y/sigma) (Moreau)."""
        z, ctx, _ = self._iter(y, sigma, out, threads, keep, ctx, copy, True, tol, max_iter,
                               warm_start, warm)
        return (z, ctx) if keep else z

    def solve(self, x, gamma=1.0, *, out=None, threads=0, copy=False, tol=1e-6, max_iter=1000,
              warm_start=None, ctx=None, warm=False):
        """``(z, info)`` with ``info`` an :class:`IterInfo` (gap and dual)."""
        z, _, info = self._iter(x, gamma, out, threads, False, ctx, copy, False, tol, max_iter,
                                warm_start, warm)
        return z, info

    def _iter(self, x, step, out, threads, keep, ctx, copy, conj, tol, max_iter, warm_start,
              warm):
        x = _input(x, copy)
        if x.ndim < self.ndim:
            raise ValueError(f"{type(self).__name__}(ndim={self.ndim}) needs arrays with at "
                             f"least {self.ndim} dimensions")
        z = _output(x, out)
        if keep or ctx is not None:
            ctx = self._context(x, -1, threads, ctx)
            ws = ctx.ws
        else:
            ws = _C.workspace(self._op, x, -1, threads, False)
        _C.set_warm(ws, bool(warm))
        if warm_start is not None:
            _C.set_dual(ws, np.asarray(warm_start, np.float64))
        gap, iters, ties, conv = _C.prox(self._op, float(step), x, z, -1, threads, keep, ws, conj,
                                         float(tol), int(max_iter))
        dual = np.empty((self.ndim, *x.shape), np.float64)
        _C.get_dual(ws, dual)
        info = IterInfo(gap, iters, ties, conv, dual)
        if ctx is not None:
            ctx.info = info
            ctx.conj = conj
            ctx.step = float(step)
        return z, ctx, info


class TVAniso(_ImageOperator):
    """Anisotropic total variation of 2D or 3D images, on the last ``ndim``
    axes and batched over the others:

        g(z) = sum_a w_a * (sum of |z_{p + e_a} - z_p| along image axis a)

    Iterative (accelerated dual block methods with exact 1D TV steps): each
    image stops when its duality gap is at most ``tol * ||x||^2 / 2``.
    ``prox`` returns ``z`` (or ``(z, ctx)`` with ``keep=True``, where
    ``ctx.info`` is an :class:`IterInfo`). ``warm_start=dual`` starts from a
    dual (``IterInfo.dual``); with ``ctx=`` and ``warm=True`` a call starts from
    the dual of the previous call on that context. Parameters: weights (one
    per axis, or a scalar for all)."""

    def __init__(self, ndim=2, weights=None):
        if ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        super().__init__(_C.tv_aniso(int(ndim), _vec(weights)), ndim)


class TVIso(_ImageOperator):
    """Isotropic total variation of 2D or 3D images, on the last ``ndim`` axes
    and batched over the others:

        g(z) = sum_p |(w_0 D_0 z, ..., w_{ndim-1} D_{ndim-1} z)_p|_2

    with forward differences (zero past the last entry); the weights scale
    each axis inside the norm (e.g. voxel spacing). Iterative (fast gradient
    projection on the dual), with the gap, warm starts and ``keep`` as in
    :class:`TVAniso`; the dual ``u`` (``IterInfo.dual``) has |u_p| <= gamma and
    z = x - sum_a w_a D_a^T u_a. ``jvp`` and ``vjp`` solve a linear system by
    conjugate gradients to relative residual ``tol`` and record the residual
    and iterations reached in ``ctx.deriv``. Parameters: weights."""

    def __init__(self, ndim=2, weights=None):
        if ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        super().__init__(_C.tv_iso(int(ndim), _vec(weights)), ndim)

    def jvp(self, ctx, dx=None, dgamma=0.0, dparams=None, *, out=None, tol=1e-10,
            max_iter=10000):
        self._check_ctx(ctx)
        _C.set_deriv_tol(ctx.ws, float(tol), int(max_iter))
        dz = super().jvp(ctx, dx, dgamma, dparams, out=out)
        ctx.deriv = _C.deriv_info(ctx.ws)
        return dz

    def vjp(self, ctx, g, *, out=None, tol=1e-10, max_iter=10000):
        self._check_ctx(ctx)
        _C.set_deriv_tol(ctx.ws, float(tol), int(max_iter))
        res = super().vjp(ctx, g, out=out)
        ctx.deriv = _C.deriv_info(ctx.ws)
        return res


class _Pointwise(Operator):
    """Nonconvex pointwise map: ``prox`` and ``prox_conj`` take ``tie`` ("nonzero",
    the default, or "zero"), the minimiser returned where the prox is
    set-valued; with ``keep=True``, ``ctx.info.ties`` counts the entries that
    sat on a tie. ``prox_conj`` is the Moreau formula y - sigma prox(y/sigma),
    not the prox of the conjugate, for these nonconvex functions."""

    def prox(self, x, gamma=1.0, *, axis=-1, out=None, threads=0, keep=False, ctx=None,
             copy=False, tie="nonzero"):
        return self._run_tie(x, gamma, axis, out, threads, keep, ctx, copy, False, tie)

    def prox_conj(self, y, sigma=1.0, *, axis=-1, out=None, threads=0, keep=False, ctx=None,
                  copy=False, tie="nonzero"):
        return self._run_tie(y, sigma, axis, out, threads, keep, ctx, copy, True, tie)

    def _run_tie(self, x, step, axis, out, threads, keep, ctx, copy, conj, tie):
        t = _tie(tie)
        x = _input(x, copy)
        z = _output(x, out)
        if not keep:
            _C.prox(self._op, float(step), x, z, axis, threads, False, None, conj, tie=t)
            return z
        ctx = self._context(x, axis, threads, ctx)
        ctx.info = Info(*_C.prox(self._op, float(step), x, z, axis, threads, True, ctx.ws, conj,
                                 tie=t)[:3])
        ctx.conj = conj
        ctx.step = float(step)
        return z, ctx


class L0(_Pointwise):
    """g(z) = lam * sum_i w_i [z_i != 0] (hard thresholding at sqrt(2 gamma lam
    w_i)). Parameters: weights (the map is piecewise constant in them)."""

    def __init__(self, lam=1.0, weights=None):
        w = _vec(weights)
        super().__init__(_C.l0(float(lam), w))
        self._layout = [("weights", 1 if w is None else w.size, w is None or w.size == 1)]


class Lq(_Pointwise):
    """g(z) = lam * sum_i w_i |z_i|^q with 0 < q < 1 (closed form for q = 1/2
    and 2/3, safeguarded Newton otherwise). Parameters: weights;
    ``grads.lam`` is also provided."""

    def __init__(self, lam=1.0, q=0.5, weights=None):
        w = _vec(weights)
        super().__init__(_C.lq(float(lam), float(q), w))
        self.lam = float(lam)
        self._w = np.ones(1) if w is None else w
        self._layout = [("weights", self._w.size, w is None or w.size == 1)]

    def _fields(self, gp):
        f = super()._fields(gp)
        f["lam"] = float(self._w @ gp) / self.lam if self.lam > 0 else float("nan")
        return f


class SCAD(_Pointwise):
    """SCAD penalty (Fan and Li 2001) with threshold lam and concavity a > 1;
    the prox needs a > 1 + gamma (ValueError otherwise). Parameters: lam, a."""

    _layout = [("lam", 1, True), ("a", 1, True)]

    def __init__(self, lam=1.0, a=3.7):
        super().__init__(_C.scad(float(lam), float(a)))


class MCP(_Pointwise):
    """Minimax concave penalty (Zhang 2010) with threshold lam and concavity
    b > 0; the prox needs b > gamma (ValueError otherwise). Parameters: lam, b."""

    _layout = [("lam", 1, True), ("b", 1, True)]

    def __init__(self, lam=1.0, b=3.0):
        super().__init__(_C.mcp(float(lam), float(b)))


class LogPenalty(_Pointwise):
    """g(z) = lam * sum_i log(1 + |z_i| / eps), eps > 0. Parameters: lam, eps."""

    _layout = [("lam", 1, True), ("eps", 1, True)]

    def __init__(self, lam=1.0, eps=1.0):
        super().__init__(_C.log_penalty(float(lam), float(eps)))


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


def prox_max(x, lam=1.0, gamma=1.0, *, axis=-1, out=None, threads=0, copy=False):
    """prox of gamma * lam * max_i z_i along the axis."""
    return _once(_C.max(float(lam)), x, gamma, axis, out, threads, copy)


def prox_sum_largest(x, lam=1.0, k=1, gamma=1.0, *, absolute=False, axis=-1, out=None, threads=0,
                     copy=False):
    """prox of gamma * lam * (sum of the k largest z_i, or |z_i|) along the axis."""
    return _once(SumLargest(lam, k, absolute)._op, x, gamma, axis, out, threads, copy)


def project_hyperplane(x, a=None, b=0.0, lower=-np.inf, upper=np.inf, *, axis=-1, out=None,
                       threads=0, copy=False):
    """Projection onto {a^T z = b, lower <= z <= upper} along the axis."""
    return _once(Hyperplane(a, b, lower, upper)._op, x, 1.0, axis, out, threads, copy)


def project_halfspace(x, a=None, b=0.0, lower=-np.inf, upper=np.inf, *, axis=-1, out=None,
                      threads=0, copy=False):
    """Projection onto {a^T z <= b, lower <= z <= upper} along the axis."""
    return _once(HalfSpace(a, b, lower, upper)._op, x, 1.0, axis, out, threads, copy)


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


def tv_aniso(x, gamma=1.0, *, weights=None, ndim=2, tol=1e-6, max_iter=1000, warm_start=None,
             out=None, threads=0, copy=False):
    """prox of gamma * sum_a w_a TV_a on the last ``ndim`` (2 or 3) axes.
    Returns ``(z, info)``; ``info.gap`` certifies the result and ``info.dual``
    can be passed back as ``warm_start``."""
    return TVAniso(ndim, weights).solve(x, gamma, out=out, threads=threads, copy=copy, tol=tol,
                                        max_iter=max_iter, warm_start=warm_start)


def tv_iso(x, gamma=1.0, *, weights=None, ndim=2, tol=1e-6, max_iter=1000, warm_start=None,
           out=None, threads=0, copy=False):
    """prox of gamma * sum_p |(w_a D_a z)_p|_2 (isotropic TV) on the last ``ndim``
    (2 or 3) axes. Returns ``(z, info)`` as :func:`tv_aniso`."""
    return TVIso(ndim, weights).solve(x, gamma, out=out, threads=threads, copy=copy, tol=tol,
                                      max_iter=max_iter, warm_start=warm_start)


def hard_threshold(x, lam=1.0, gamma=1.0, *, weights=None, tie="nonzero", out=None, axis=-1,
                   threads=0, copy=False):
    """prox of gamma * lam * sum_i w_i [z_i != 0]."""
    return L0(lam, weights).prox(x, gamma, axis=axis, out=out, threads=threads, copy=copy,
                                 tie=tie)


def lq(x, lam=1.0, q=0.5, gamma=1.0, *, weights=None, tie="nonzero", out=None, axis=-1,
       threads=0, copy=False):
    """prox of gamma * lam * sum_i w_i |z_i|^q, 0 < q < 1."""
    return Lq(lam, q, weights).prox(x, gamma, axis=axis, out=out, threads=threads, copy=copy,
                                    tie=tie)


def scad(x, lam=1.0, a=3.7, gamma=1.0, *, out=None, threads=0, copy=False):
    """prox of gamma * SCAD(lam, a); needs a > 1 + gamma."""
    return SCAD(lam, a).prox(x, gamma, out=out, threads=threads, copy=copy)


def mcp(x, lam=1.0, b=3.0, gamma=1.0, *, out=None, threads=0, copy=False):
    """prox of gamma * MCP(lam, b); needs b > gamma."""
    return MCP(lam, b).prox(x, gamma, out=out, threads=threads, copy=copy)


def prox_log(x, lam=1.0, eps=1.0, gamma=1.0, *, tie="nonzero", out=None, threads=0, copy=False):
    """prox of gamma * lam * sum_i log(1 + |z_i| / eps)."""
    return LogPenalty(lam, eps).prox(x, gamma, out=out, threads=threads, copy=copy, tie=tie)
