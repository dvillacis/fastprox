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
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import IterInfo
from . import _tv_jax as _tvj
from ._interop import Spec, check_dtype, is_float

__all__ = ["prox", "prox_conj", "tv_iso", "tv_aniso"]


def prox(op, x, gamma=1.0, *, axis=None, threads=0, tol=None, max_iter=None, **params):
    """prox_{gamma g}(x). ``axis`` (row operators), ``tol`` and ``max_iter``
    (iterative operators) are passed on when given."""
    return _apply(op, x, gamma, False, axis, threads, tol, max_iter, params)


def prox_conj(op, y, sigma=1.0, *, axis=None, threads=0, tol=None, max_iter=None, **params):
    """prox_{sigma g*}(y) (Moreau's identity)."""
    return _apply(op, y, sigma, True, axis, threads, tol, max_iter, params)


def _apply(op, x, step, conj, axis, threads, tol, max_iter, params):
    x = jnp.asarray(x)
    check_dtype(x.dtype)
    names, values, static = [], [], {}
    for k, v in params.items():
        if isinstance(v, jax.Array) and is_float(v):
            names.append(k)
            values.append(v.astype(x.dtype))
        elif isinstance(v, jax.Array):
            static[k] = np.asarray(v)  # integer arrays must be concrete
        else:
            static[k] = v
    spec = Spec(op, static, names, conj, threads, {"axis": axis, "tol": tol, "max_iter": max_iter})
    return _function(spec)(x, jnp.asarray(step, x.dtype), *values)


def _function(spec):
    def forward(x, step, values):
        def cb(x, step, *values):
            op = spec.build(values)
            return spec.run(op, np.asarray(x), step)[0]

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
            _, ctx = spec.run(op, np.asarray(x), step)
            gx, grads = op.vjp(ctx, np.asarray(g, x.dtype))
            out = [gx, np.asarray(grads.gamma, x.dtype)]
            out += [v.astype(x.dtype) for v in spec.param_grads(op, grads, [np.shape(v) for v in values])]
            return tuple(out)

        shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(step.shape, step.dtype)]
        shapes += [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
        return tuple(jax.pure_callback(cb, tuple(shapes), x, step, g, *values, vmap_method="sequential"))

    f.defvjp(fwd, bwd)
    return f


# ---------------------------------------------------------------------------
# Device-native image TV
# ---------------------------------------------------------------------------


def tv_iso(x, gamma=1.0, *, ndim=2, weights=None, tol=1e-6, max_iter=1000, warm_start=None,
           return_info=False, deriv_tol=None, deriv_max_iter=10000):
    """Isotropic TV of ``TVIso(ndim, weights)`` on the last ``ndim`` axes of
    ``x``, in JAX operations on ``x``'s device; differentiable (reverse
    mode) in ``x``, ``gamma`` and ``weights``. Each image stops when its
    duality gap is at most ``tol * |x|^2 / 2``; ``tol`` and ``max_iter`` are
    static under ``jit``. ``warm_start`` takes the ``dual`` of an earlier
    call (shape (ndim, *x.shape)). With ``return_info=True`` returns
    ``(z, info)``, an :class:`IterInfo` of JAX arrays. The backward is the C
    kernel's: one unknown per flat region, solved by conjugate gradients to
    ``deriv_tol`` (default 1e-10 in float64, 100 eps in float32)."""
    return _device_tv(True, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter)


def tv_aniso(x, gamma=1.0, *, ndim=2, weights=None, tol=1e-6, max_iter=1000, warm_start=None,
             return_info=False, deriv_tol=None, deriv_max_iter=10000):
    """Anisotropic TV of ``TVAniso(ndim, weights)``, as :func:`tv_iso`: a
    plain dual method, with the dual in the C kernel's form; its derivatives
    are exact region averages."""
    return _device_tv(False, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter)


def _device_tv(iso, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info, deriv_tol,
               deriv_max_iter):
    x = jnp.asarray(x)
    check_dtype(x.dtype)
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")
    if x.ndim < ndim:
        raise ValueError(f"needs an array with at least {ndim} dimensions")
    if not tol >= 0 or max_iter < 0:
        raise ValueError("tol must be >= 0 and max_iter >= 0")
    img = x.shape[-ndim:]
    xb = x.reshape((-1,) + img)
    w = jnp.ones(ndim, x.dtype) if weights is None else jnp.asarray(weights, x.dtype).reshape(-1)
    if w.shape[0] == 1:
        w = jnp.broadcast_to(w, (ndim,))
    if w.shape[0] != ndim:
        raise ValueError("weights must be one or ndim values")
    gamma = jnp.asarray(gamma, x.dtype)
    if warm_start is None:
        u0 = jnp.zeros((ndim,) + xb.shape, x.dtype)
    else:
        q = jnp.asarray(warm_start, x.dtype)
        if q.shape != (ndim,) + x.shape:
            raise ValueError(f"warm_start must have shape {(ndim,) + x.shape}")
        u0 = _tvj.dual_in(q.reshape((ndim,) + xb.shape), w, iso)
    if deriv_tol is None:
        deriv_tol = max(1e-10, 100 * float(jnp.finfo(x.dtype).eps))
    z, (u, gap, it, conv) = _tvj.tv(xb, gamma, w, u0, iso, float(tol), int(max_iter), float(deriv_tol),
                                    int(deriv_max_iter))
    z = z.reshape(x.shape)
    if not return_info:
        return z
    dual = _tvj.dual_out(u, w, iso).reshape((ndim,) + x.shape)
    return z, IterInfo(gap.sum(), it, 0, jnp.all(conv), dual)
