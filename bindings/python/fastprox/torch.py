"""PyTorch wrappers (since 0.3): the operators as ``torch.autograd.Function``
s, differentiated by the library's own JVP and VJP rather than through any
iterations::

    import fastprox as fp
    import fastprox.torch as fpt

    z = fpt.prox(fp.TV1D(), x, gamma)                 # gradients for x and gamma
    z = fpt.prox(fp.TV1D, x, gamma, weights=w)        # and for the weights

``op`` is an operator, or an operator class with its constructor arguments
as keywords: float tensors among them that require grad receive the
gradient of the parameter of that name (``Grads`` fields, see
:meth:`fastprox.Operator.vjp`); other arguments are constants. Reverse mode
(``backward``, ``torch.autograd.grad``) and forward mode
(``torch.autograd.forward_ad``) are supported; not double backward or the
``torch.func`` transforms, which hand the derivatives wrapped tensors that
cannot be read as NumPy arrays. The
computation runs on the CPU in float32 or float64; tensors on another device
are copied to the CPU and the result back.
"""

from __future__ import annotations

import numpy as np
import torch

from ._interop import Spec, check_dtype

__all__ = ["prox", "prox_conj"]


def prox(op, x, gamma=1.0, *, axis=None, threads=0, tol=None, max_iter=None, **params):
    """prox_{gamma g}(x) as a differentiable tensor. ``gamma`` may be a tensor
    (0-d) that requires grad; ``axis`` (row operators), ``tol`` and
    ``max_iter`` (iterative operators) are passed on when given."""
    return _apply(op, x, gamma, False, axis, threads, tol, max_iter, params)


def prox_conj(op, y, sigma=1.0, *, axis=None, threads=0, tol=None, max_iter=None, **params):
    """prox_{sigma g*}(y) as a differentiable tensor (Moreau's identity);
    ``sigma`` as ``gamma`` in :func:`prox`."""
    return _apply(op, y, sigma, True, axis, threads, tol, max_iter, params)


def _apply(op, x, step, conj, axis, threads, tol, max_iter, params):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    check_dtype(_np_dtype(x.dtype))
    names, values, static = [], [], {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            names.append(k)
            values.append(v)
        elif isinstance(v, torch.Tensor):
            static[k] = v.detach().cpu().numpy()
        else:
            static[k] = v
    spec = Spec(op, static, names, conj, threads, {"axis": axis, "tol": tol, "max_iter": max_iter})
    return _Prox.apply(spec, x, step, *values)


def _np_dtype(dt):
    return {torch.float32: np.float32, torch.float64: np.float64}.get(dt, dt)


def _numpy(t):
    return t.detach().cpu().resolve_conj().resolve_neg().numpy()


def _scalar(v):
    return float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)


class _Prox(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spec, x, step, *values):
        op = spec.build([_numpy(v) for v in values])
        spec.check_names(op, [k for k, need in zip(spec.names, ctx.needs_input_grad[3:]) if need])
        z, fctx = spec.run(op, _numpy(x), _scalar(step))
        ctx.spec, ctx.op, ctx.fctx = spec, op, fctx
        ctx.step_like = step if isinstance(step, torch.Tensor) else None
        ctx.shapes = [tuple(v.shape) for v in values]
        ctx.like = [(v.dtype, v.device) for v in values]
        ctx.x_like = (x.dtype, x.device)
        return torch.from_numpy(z).to(x.device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        spec, op = ctx.spec, ctx.op
        gx, grads = op.vjp(ctx.fctx, _numpy(g))
        dtype, device = ctx.x_like
        out_x = torch.from_numpy(gx).to(device=device, dtype=dtype) if ctx.needs_input_grad[1] else None
        out_step = None
        if ctx.step_like is not None and ctx.needs_input_grad[2]:
            out_step = torch.full_like(ctx.step_like, grads.gamma)
        out_p = [None] * len(ctx.shapes)
        if any(ctx.needs_input_grad[3:]):
            vals = spec.param_grads(op, grads, ctx.shapes)
            for i, (v, (dt, dev)) in enumerate(zip(vals, ctx.like)):
                if ctx.needs_input_grad[3 + i]:
                    out_p[i] = torch.from_numpy(v).to(device=dev, dtype=dt)
        return (None, out_x, out_step, *out_p)

    @staticmethod
    def jvp(ctx, _spec, dx, dstep, *dvalues):
        op = ctx.op
        dparams = ctx.spec.dparams(op, [None if t is None else _numpy(t) for t in dvalues])
        dz = op.jvp(ctx.fctx, None if dx is None else _numpy(dx),
                    0.0 if dstep is None else _scalar(dstep), dparams)
        dtype, device = ctx.x_like
        return torch.from_numpy(dz).to(device=device, dtype=dtype)
