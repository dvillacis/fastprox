"""JAX wrappers (since 0.3): the operators as ``jax.custom_vjp`` functions
over ``jax.pure_callback``, differentiated by the library's own VJP rather
than through any iterations::

    import fastprox as fp
    import fastprox.jax as fpj

    z = fpj.prox(fp.TV1D(), x, gamma)                 # gradients for x and gamma
    z = fpj.prox(fp.TV1D, x, gamma, weights=w)        # and for the weights

``op`` is an operator, or an operator class with its constructor arguments
as keywords: JAX float arrays among them are traced and receive the
gradient of the parameter of that name (``Grads`` fields, see
:meth:`fastprox.Operator.vjp`), NaN for a parameter without one (L0's
``lam``, for instance); Python and NumPy values are constants, and integer
arrays (groups, edges) must be concrete under ``jit``. Works under
``jit``, ``grad`` and ``vmap`` (one call per batch element); reverse mode
only. The backward pass recomputes the prox to keep its structure, so a
gradient costs one prox and one VJP. Float64 needs ``jax_enable_x64``.

:func:`tv_iso` and :func:`tv_aniso` instead solve isotropic and anisotropic
TV in JAX operations on the input's device, GPU included, under ``jit`` and
``vmap``, with a backward on the flat regions of the solution (see
``fastprox._tv_jax``).

Spectral operators (since 1.0) go through :func:`prox` too, as in
``fastprox.torch``. The decomposition and the matrix products run in JAX on
the input's device; the prox of the spectrum and the k x k core of the
derivative run in callbacks on the CPU.
"""

from __future__ import annotations

import functools
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from . import _ACCURACY, _MAX_ITER, IterInfo, Operator, Spectral, _call_tol, _sq_tol, _warn_unconverged
from . import _tv_jax as _tvj
from ._interop import Spec, check_dtype, check_step, is_float, spectral_step_grad, steps

__all__ = ["prox", "prox_conj", "tv_iso", "tv_aniso"]


def prox(op, x, gamma=1.0, *, axis=None, threads=0, tol=None, ztol=None, max_iter=None,
         deriv_tol=None, deriv_max_iter=10000, **params):
    r"""$\operatorname{prox}_{\gamma g}(x)$. ``gamma`` is a scalar or (since 1.0) one step per
    row, image or matrix, broadcasting to the batch shape, as in
    :func:`fastprox.torch.prox`. ``axis`` (row operators), ``tol``, ``ztol``
    and ``max_iter`` (iterative operators) are passed on when given;
    ``deriv_tol`` and ``deriv_max_iter`` (since 1.0) as in
    :func:`fastprox.torch.prox`."""
    return _apply(op, x, gamma, False, axis, threads, tol, ztol, max_iter, params,
                  {"tol": deriv_tol, "max_iter": deriv_max_iter})


def prox_conj(op, y, sigma=1.0, *, axis=None, threads=0, tol=None, ztol=None, max_iter=None,
              deriv_tol=None, deriv_max_iter=10000, **params):
    r"""$\operatorname{prox}_{\sigma g^*}(y)$ (Moreau's identity), with a scalar ``sigma``."""
    return _apply(op, y, sigma, True, axis, threads, tol, ztol, max_iter, params,
                  {"tol": deriv_tol, "max_iter": deriv_max_iter})


def _apply(op, x, step, conj, axis, threads, tol, ztol, max_iter, params, deriv):
    x = jnp.asarray(x)
    check_dtype(x.dtype)
    check_step(step, conj)
    names, values, static = [], [], {}
    for k, v in params.items():
        if isinstance(v, jax.Array) and is_float(v):
            names.append(k)
            values.append(v.astype(x.dtype))
        elif isinstance(v, jax.Array):
            static[k] = np.asarray(v)  # integer arrays must be concrete
        else:
            static[k] = v
    spec = Spec(op, static, names, conj, threads,
                {"axis": axis, "tol": tol, "ztol": ztol, "max_iter": max_iter}, deriv)
    # The step's sum only when the step can carry a gradient (audit P11).
    want_gamma = isinstance(step, jax.core.Tracer)
    if spec.spectral:
        if x.ndim < 2:
            raise ValueError("spectral operators act on matrices: the last two axes")
        if spec.kind == "eigh" and x.shape[-1] != x.shape[-2]:
            raise ValueError("kind='eigh' needs square matrices")
        return _jitted(spec, _spectral, want_gamma)(x, jnp.asarray(step, x.dtype), *values)
    return _jitted(spec, _function, want_gamma)(x, jnp.asarray(step, x.dtype), *values)


# Eager calls traced the custom_vjp and its callbacks on every call: 2.5 ms
# for a soft threshold of 100 entries (audit P9). A jit of the built function
# is kept on the operator instance, or for a class in a small LRU keyed on its
# static arguments, and JAX's own cache retraces it for new shapes and dtypes.
# Inside a user's jit the inner jit is inlined. Operators are immutable once
# built, so a key never goes stale.
_CLASS_CACHE: OrderedDict = OrderedDict()
_CLASS_CACHE_SIZE = 128
_CACHE_ATTR = "_fpj_jit"


def _jitted(spec, build, want_gamma):
    key = (build.__name__, spec.spectral, spec.conj, spec.threads, tuple(sorted(spec.call.items())),
           spec.names, tuple(sorted(spec.deriv.items())), spec.accuracy, want_gamma)
    if isinstance(spec.op, (Operator, Spectral)):
        cache = spec.op.__dict__.setdefault(_CACHE_ATTR, {})
    else:
        frozen = _freeze(spec.static)
        if frozen is None:  # an unhashable constant: no cache
            return build(spec, want_gamma)
        key = (spec.op, frozen) + key
        cache = _CLASS_CACHE
    f = cache.get(key)
    if f is None:
        f = cache[key] = jax.jit(build(spec, want_gamma))
        if cache is _CLASS_CACHE and len(cache) > _CLASS_CACHE_SIZE:
            cache.popitem(last=False)
    elif cache is _CLASS_CACHE:
        cache.move_to_end(key)
    return f


def _freeze(static):
    """The static arguments as a hashable key, or None when one cannot be."""
    try:
        out = tuple(sorted((k, _fz(v)) for k, v in static.items()))
    except _Unhashable:
        return None
    return out


class _Unhashable(Exception):
    pass


def _fz(v):
    if isinstance(v, np.ndarray):
        return ("nd", v.dtype.str, v.shape, v.tobytes())
    if isinstance(v, (list, tuple)):
        return (type(v).__name__,) + tuple(_fz(a) for a in v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return (type(v).__name__, v)
    raise _Unhashable


def _function(spec, want_gamma=True):
    def forward(x, step, values):
        def cb(x, step, *values):
            op = spec.build(values)
            return spec.run(op, np.asarray(x), steps(step), keep=False)[0]  # audit P10

        return jax.pure_callback(cb, jax.ShapeDtypeStruct(x.shape, x.dtype), x, step, *values,
                                 vmap_method="sequential")

    @jax.custom_vjp
    def f(x, step, *values):
        return forward(x, step, values)

    def fwd(x, step, *values):
        return forward(x, step, values), (x, step, values)

    def bwd(res, g):
        x, step, values = res

        def cb(x, step, g, *values):
            op = spec.build(values)
            _, ctx = spec.run(op, np.asarray(x), steps(step))
            g = np.asarray(g, x.dtype)
            per_row = np.size(step) > 1  # its gradient from a jvp, not grads.gamma
            gx, grads = op._vjp(ctx, g, None, spec.deriv.get("tol"), spec.deriv.get("max_iter", _MAX_ITER),
                                None, None, 0.0, want_gamma and not per_row, bool(values))
            gs = spec.step_grad(op, ctx, g, grads, np.shape(step)) if want_gamma else np.zeros(np.shape(step))
            out = [gx, gs.astype(x.dtype)]
            out += [v.astype(x.dtype) for v in spec.param_grads(op, grads, [np.shape(v) for v in values])]
            return tuple(out)

        shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(step.shape, step.dtype)]
        shapes += [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
        return tuple(jax.pure_callback(cb, tuple(shapes), x, step, g, *values, vmap_method="sequential"))

    f.defvjp(fwd, bwd)
    return f


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------


def _mT(a):
    return jnp.swapaxes(a, -1, -2)


def _spectral(spec, want_gamma=True):
    """The decomposition and the products in JAX, on x's device; the prox of
    the spectrum and the k x k core of the derivative in callbacks on the
    CPU. The backward recomputes the prox of the spectrum for its structure.
    The shape checks are in _apply; the traced bodies read x's static shape,
    so one jit serves every shape."""
    eig = spec.kind == "eigh"

    def prox_s(op, s, step, keep=True):
        return op.op.prox(np.ascontiguousarray(s), steps(step), threads=spec.threads, keep=keep)

    def forward(x, step, values):
        if eig:
            s, u = jnp.linalg.eigh((x + _mT(x)) / 2)
            vt = _mT(u)
        else:
            u, s, vt = jnp.linalg.svd(x, full_matrices=False)

        def cb(s, step, *values):
            return prox_s(spec.build(values), s, step, keep=False)  # audit P10

        z = jax.pure_callback(cb, jax.ShapeDtypeStruct(s.shape, s.dtype), s, step, *values,
                              vmap_method="sequential")
        return (u * z[..., None, :]) @ vt, (u, s, vt, step, values)

    @jax.custom_vjp
    def f(x, step, *values):
        return forward(x, step, values)[0]

    def fwd(x, step, *values):
        return forward(x, step, values)

    def bwd(res, g):
        u, s, vt, step, values = res
        m, n = u.shape[-2], vt.shape[-1]
        rect = not eig and m != n
        if eig:
            g = (g + _mT(g)) / 2
        a = _mT(u) @ g @ _mT(vt)

        def cb(s, step, a, *values):
            op = spec.build(values)
            z, inner = prox_s(op, s, step)
            a = np.asarray(a)
            b, r = op._coefficients(np.ascontiguousarray(s), z, inner, a, rect=rect)
            grads = op._grads(inner, a)
            out = [b, spectral_step_grad(op, inner, a, grads, np.shape(step)).astype(s.dtype)]
            if rect:
                out.append(r)
            out += [v.astype(s.dtype) for v in spec.param_grads(op, grads, [np.shape(v) for v in values])]
            return tuple(out)

        shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype), jax.ShapeDtypeStruct(step.shape, step.dtype)]
        if rect:
            shapes.append(jax.ShapeDtypeStruct(s.shape, s.dtype))
        shapes += [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
        out = jax.pure_callback(cb, tuple(shapes), s, step, a, *values, vmap_method="sequential")
        b, gstep, rest = out[0], out[1], out[2:]
        gx = u @ b @ vt
        if rect:
            r, rest = rest[0][..., None, :], rest[1:]
            if m > n:  # (I - U U^T) G V diag(r) V^T
                gx = gx + ((g @ _mT(vt) - u @ a) * r) @ vt
            else:      # U diag(r) U^T G (I - V V^T)
                gx = gx + (u * r) @ (_mT(u) @ g - a @ vt)
        return (gx, gstep, *rest)

    f.defvjp(fwd, bwd)
    return f


# ---------------------------------------------------------------------------
# Device-native image TV
# ---------------------------------------------------------------------------


def tv_iso(x, gamma=1.0, *, ndim=2, weights=None, pixel_weights=None, tol=None, ztol=None,
           max_iter=1000, warm_start=None, return_info=False, deriv_tol=None,
           deriv_max_iter=10000, warn=True):
    r"""Isotropic TV of ``TVIso(ndim, weights)`` on the last ``ndim`` axes of
    ``x``, in JAX operations on ``x``'s device; differentiable (reverse
    mode) in ``x``, ``gamma`` and ``weights``. ``gamma`` is a number or one
    step per image, broadcasting to the batch shape ``x.shape[:-ndim]``. Each image stops when its
    duality gap is at most $\mathrm{tol} \cdot \lVert x \rVert_2^2 / 2$ (``tol`` defaults to
    1e-6), which certifies $\lVert z - z^* \rVert_2 \le \sqrt{\mathrm{tol}}\,\lVert x \rVert_2$;
    ``ztol`` (since 1.0) asks for that relative error of $z$ instead (``tol`` $= \mathrm{ztol}^2$).
    Without either, :func:`fastprox.inner_accuracy` sets ``ztol`` (read when traced). A solve that
    stops at ``max_iter`` raises :class:`fastprox.ConvergenceWarning` from the host unless
    ``warn=False``.
    ``tol``, ``ztol`` and ``max_iter`` are static under ``jit``. ``warm_start`` takes the ``dual`` of an earlier
    call (shape (ndim, *x.shape)). With ``return_info=True`` returns
    ``(z, info)``, an :class:`IterInfo` of JAX arrays. The backward is the C
    kernel's: one unknown per flat region, solved by conjugate gradients to
    relative residual ``deriv_tol``; by default each image's solve stops at
    the relative error of $z$ that its gap certifies,
    $\sqrt{2\,\mathrm{gap}} / \lVert x \rVert_2$, and at
    $\max(10^{-10}, 100\,\varepsilon)$ at the tightest. ``pixel_weights`` (since
    0.3), as in :func:`fastprox.torch.tv_iso`: $\alpha \ge 0$ broadcasting to x.shape,
    differentiable."""
    tol = _call_tol(tol, ztol, 1e-6, _sq_tol, warn, stacklevel=3)
    return _device_tv(True, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter, pixel_weights, warn and "fastprox.jax.tv_iso")


def tv_aniso(x, gamma=1.0, *, ndim=2, weights=None, edge_weights=None, tol=None, ztol=None,
             max_iter=1000, warm_start=None, return_info=False, deriv_tol=None,
             deriv_max_iter=10000, warn=True):
    r"""Anisotropic TV of ``TVAniso(ndim, weights)``, as :func:`tv_iso`: a
    plain dual method, with the dual in the C kernel's form; its derivatives
    are exact region averages. ``edge_weights`` (since 0.3), as in
    :func:`fastprox.torch.tv_aniso`: $\beta \ge 0$ of shape (ndim, ...)
    broadcasting to (ndim, *x.shape), differentiable."""
    tol = _call_tol(tol, ztol, 1e-6, _sq_tol, warn, stacklevel=3)
    return _device_tv(False, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter, edge_weights, warn and "fastprox.jax.tv_aniso")


# Compiled once (audit P9): eagerly, the solve's loops were traced on every
# call (70 ms for one 32 x 32 image). iso and the tolerances are static.
_tv_jit = jax.jit(_tvj.tv, static_argnums=(5, 6, 7, 8, 9))
_dual_in_jit = jax.jit(_tvj.dual_in, static_argnums=(2,))
_dual_out_jit = jax.jit(_tvj.dual_out, static_argnums=(2,))


def _device_tv(iso, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info, deriv_tol,
               deriv_max_iter, spatial=None, warn=False):
    """warn: the function's name for a ConvergenceWarning, or False."""
    x = jnp.asarray(x)
    check_dtype(x.dtype)
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")
    if x.ndim < ndim:
        raise ValueError(f"needs an array with at least {ndim} dimensions")
    if not tol >= 0 or max_iter < 0:
        raise ValueError("tol must be >= 0 and max_iter >= 0")
    img = x.shape[-ndim:]
    if x.size == 0:  # an empty batch (static): an empty result, in the graph of every input (audit P8)
        z = x
        for t in (gamma, weights, spatial):
            if t is not None:
                z = z + 0 * jnp.sum(jnp.asarray(t, x.dtype))
        if not return_info:
            return z
        return z, IterInfo(jnp.zeros((), x.dtype), 0, 0, jnp.asarray(True),
                           jnp.zeros((ndim,) + x.shape, x.dtype), jnp.zeros((), x.dtype))
    try:  # under jit values cannot be checked, and an Inf image is a NaN image
        has_inf = bool(jnp.isinf(x).any())
    except jax.errors.ConcretizationTypeError:
        has_inf = False
    if has_inf:
        raise ValueError("x contains ±Inf; image operators reject infinite values")
    xb = x.reshape((-1,) + img)
    w = jnp.ones(ndim, x.dtype) if weights is None else jnp.asarray(weights, x.dtype).reshape(-1)
    if w.shape[0] == 1:
        w = jnp.broadcast_to(w, (ndim,))
    if w.shape[0] != ndim:
        raise ValueError("weights must be one or ndim values")
    batch = x.shape[:-ndim]
    gamma = jnp.asarray(gamma, x.dtype)
    if gamma.size == 1:
        gamma = gamma.reshape(())
    try:
        ok = np.broadcast_shapes(gamma.shape, batch) == tuple(batch)
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"gamma must be a number or broadcast to the batch shape {tuple(batch)}, "
                         f"got shape {gamma.shape}")
    gamma = jnp.broadcast_to(gamma, batch).reshape(-1)  # one per image; its gradient sums back
    if warm_start is None:
        u0 = jnp.zeros((ndim,) + xb.shape, x.dtype)
    else:
        q = jnp.asarray(warm_start, x.dtype)
        if q.shape != (ndim,) + x.shape:
            raise ValueError(f"warm_start must have shape {(ndim,) + x.shape}")
        u0 = _dual_in_jit(q.reshape((ndim,) + xb.shape), w, iso)
    if deriv_tol is not None:
        if not deriv_tol > 0:
            raise ValueError("deriv_tol must be > 0 or None")
        deriv_tol = float(deriv_tol)
    sw = None if spatial is None else _spatial(spatial, x, ndim, iso)
    z, (u, gap, it, conv) = _tv_jit(xb, gamma, w, u0, sw, iso, float(tol), int(max_iter), deriv_tol,
                                    int(deriv_max_iter))
    z = z.reshape(x.shape)
    # A report, not a differentiable output: no gradients through it.
    xx = jnp.sum(jnp.square(jax.lax.stop_gradient(xb)).reshape(xb.shape[0], -1), axis=1)
    g = jax.lax.stop_gradient(gap)
    zerr = jnp.max(jnp.sqrt(2 * jnp.maximum(g, 0) / jnp.where(xx > 0, xx, 1)))  # NaN for a NaN image
    if warn:  # conv is traced: the host reduces it and warns, under jit too
        jax.debug.callback(functools.partial(_unconverged, warn, int(max_iter), _ACCURACY.get()),
                           conv, zerr)
    if not return_info:
        return z
    dual = _dual_out_jit(u, w, iso).reshape((ndim,) + x.shape)
    return z, IterInfo(gap.sum(), it, 0, jnp.all(conv), dual, zerr)


def _unconverged(who, max_iter, need, converged, zerr):
    if not np.all(converged):
        _warn_unconverged(who, float(zerr), max_iter, stacklevel=2, need=need)


def _spatial(sw, x, ndim, iso):
    """Pixel weights broadcasting to x.shape, as (B, *img), or edge weights
    (ndim, ...) whose other axes broadcast to x.shape, as (ndim, B, *img), as
    in fastprox.torch; the gradient sums back to sw's own shape. Values are
    not checked here (they may be traced): they must be finite and >= 0."""
    name = "pixel_weights" if iso else "edge_weights"
    t = jnp.asarray(sw, x.dtype)
    full = tuple(x.shape) if iso else (ndim,) + tuple(x.shape)
    ok = True
    if not iso:
        ok = t.ndim >= 1 and t.shape[0] == ndim and t.ndim - 1 <= x.ndim
        if ok:
            t = t.reshape((ndim,) + (1,) * (x.ndim + 1 - t.ndim) + t.shape[1:])
    try:
        ok = ok and np.broadcast_shapes(t.shape, full) == full
    except ValueError:
        ok = False
    if not ok:
        what = "broadcast to" if iso else f"have shape ({ndim}, ...) broadcasting to"
        raise ValueError(f"{name} must {what} {full}, got shape {jnp.shape(sw)}")
    img = tuple(x.shape[-ndim:])
    return jnp.broadcast_to(t, full).reshape((-1,) + img if iso else (ndim, -1) + img)
