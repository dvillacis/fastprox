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
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._interop import Spec, check_dtype, is_float

__all__ = ["prox", "prox_conj"]


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
