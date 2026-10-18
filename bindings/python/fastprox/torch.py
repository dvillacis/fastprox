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

:func:`tv_iso` and :func:`tv_aniso` instead solve isotropic and anisotropic
TV in tensor operations on the input's device, GPU included, with a backward
on the flat regions of the solution (see ``fastprox._tv_torch``).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import _tv_torch as _tvt
from ._interop import Spec, check_dtype

__all__ = ["prox", "prox_conj", "tv_iso", "tv_aniso"]


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


# ---------------------------------------------------------------------------
# Device-native image TV
# ---------------------------------------------------------------------------


def tv_iso(x, gamma=1.0, *, ndim=2, weights=None, tol=1e-6, max_iter=1000, warm_start=None,
           return_info=False, deriv_tol=None, deriv_max_iter=10000):
    """Isotropic TV of ``TVIso(ndim, weights)`` on the last ``ndim`` axes of
    ``x``, computed in tensor operations on ``x``'s device and differentiable
    in ``x``, ``gamma`` and ``weights`` (tensors that require grad), forward
    and reverse. ``gamma`` is a number or one step per image: a tensor that
    broadcasts to the batch shape ``x.shape[:-ndim]``. Each image stops when its duality gap is at most
    ``tol * |x|^2 / 2``. ``warm_start`` takes the ``dual`` of an earlier
    call (shape (ndim, *x.shape), the C kernel's form). With
    ``return_info=True`` returns ``(z, info)``, an :class:`IterInfo` whose
    ``dual`` is a tensor. The derivatives are the C kernel's: one unknown per
    flat region, solved by conjugate gradients to relative residual
    ``deriv_tol``; by default each image's solve stops at the relative error
    of z that its gap certifies, sqrt(2 gap) / |x|, and at max(1e-10,
    100 eps) at the tightest."""
    return _device_tv(True, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter)


def tv_aniso(x, gamma=1.0, *, ndim=2, weights=None, tol=1e-6, max_iter=1000, warm_start=None,
             return_info=False, deriv_tol=None, deriv_max_iter=10000):
    """Anisotropic TV of ``TVAniso(ndim, weights)``, as :func:`tv_iso`. A plain
    dual method on the device (the C kernel's exact 1D steps and polish stay
    on the CPU), so tight tolerances take more iterations than ``TVAniso``;
    the dual is in the C kernel's form (q_a with x - z = sum_a q_a). Its
    derivatives are exact region averages."""
    return _device_tv(False, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter)


class _TVConfig:
    def __init__(self, iso, ndim, w, tol, max_iter, warm, deriv_tol, deriv_max_iter):
        self.iso, self.k, self.w = iso, ndim, w
        self.tol, self.max_iter, self.warm = float(tol), int(max_iter), warm
        self.deriv_tol, self.deriv_max_iter = deriv_tol, int(deriv_max_iter)
        self.info = None
        self.deriv = None


def _device_tv(iso, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info, deriv_tol,
               deriv_max_iter):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    check_dtype(_np_dtype(x.dtype))
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")
    if x.dim() < ndim:
        raise ValueError(f"needs a tensor with at least {ndim} dimensions")
    if not tol >= 0 or max_iter < 0:
        raise ValueError("tol must be >= 0 and max_iter >= 0")
    wt = None
    if weights is None:
        w = [1.0] * ndim
    elif isinstance(weights, torch.Tensor) and weights.is_floating_point():
        wt = weights
        w = [float(a) for a in weights.detach().cpu().reshape(-1)]
    else:
        w = [float(a) for a in np.atleast_1d(np.asarray(weights, np.float64))]
    if len(w) == 1:
        w = w * ndim
    if len(w) != ndim or min(w) < 0 or not all(math.isfinite(a) for a in w):
        raise ValueError("weights must be one or ndim finite values >= 0")
    gb = _per_image(gamma, x, ndim)
    warm = None
    if warm_start is not None:
        warm = torch.as_tensor(warm_start).to(dtype=x.dtype, device=x.device)
        if tuple(warm.shape) != (ndim, *x.shape):
            raise ValueError(f"warm_start must have shape {(ndim, *x.shape)}")
    if deriv_tol is not None:
        if not deriv_tol > 0:
            raise ValueError("deriv_tol must be > 0 or None")
        deriv_tol = float(deriv_tol)
    cfg = _TVConfig(iso, ndim, w, tol, max_iter, warm, deriv_tol, deriv_max_iter)
    z = _DeviceTV.apply(cfg, x, gb, wt)
    return (z, cfg.info) if return_info else z


def _per_image(gamma, x, ndim):
    """gamma as one value per image, (B,): a number, or a tensor or array
    that broadcasts to the batch shape x.shape[:-ndim]. Tensors stay in the
    graph, so the gradient sums back to gamma's own shape."""
    batch = tuple(x.shape[:-ndim])
    g = gamma if isinstance(gamma, torch.Tensor) else torch.as_tensor(np.asarray(gamma, np.float64))
    if not g.is_floating_point():
        g = g.to(torch.float64)
    g = g.to(dtype=x.dtype, device=x.device)
    if g.numel() == 1:
        g = g.reshape(())
    try:
        if torch.broadcast_shapes(g.shape, batch) != batch:
            raise RuntimeError
    except RuntimeError:
        raise ValueError(f"gamma must be a number or broadcast to the batch shape {batch}, "
                         f"got shape {tuple(g.shape)}") from None
    gd = g.detach()
    if bool(((gd < 0) | ~torch.isfinite(gd)).any()):
        raise ValueError("gamma must be finite and >= 0")
    return g.expand(batch).reshape(-1)


class _DeviceTV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cfg, x, gamma, wt):
        k = cfg.k
        img = tuple(x.shape[-k:])
        xb = x.detach().reshape((-1,) + img)
        gam = gamma.detach()  # (B,), one per image
        u0 = None
        if cfg.warm is not None:
            u0 = _tvt.dual_in(cfg.warm.reshape((k, -1) + img), cfg.w, cfg.iso, k, gam)
        with torch.no_grad():
            z, u, gap, it, conv = _tvt.solve(xb, gam, cfg.w, cfg.iso, cfg.tol, cfg.max_iter, u0)
            dual = _tvt.dual_out(cfg.w, u, cfg.iso, k).reshape((k,) + tuple(x.shape))
        cfg.info = _tvt.info(gap, it, conv, dual)
        ctx.cfg, ctx.xb, ctx.z, ctx.u, ctx.gap, ctx.gam = cfg, xb, z, u, gap, gam
        ctx.shape = tuple(x.shape)
        ctx.wt = wt
        ctx.S = None
        return z.reshape(x.shape)

    @staticmethod
    def _structure(ctx):
        if ctx.S is None:
            cfg = ctx.cfg
            with torch.no_grad():
                ctx.S = _tvt.Structure(ctx.xb, ctx.z, ctx.u, ctx.gap, ctx.gam, cfg.w, cfg.iso)
        return ctx.S

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, c):
        cfg = ctx.cfg
        S = _DeviceTV._structure(ctx)
        with torch.no_grad():
            r, ggamma, gw, cfg.deriv = S.vjp(c.reshape(ctx.z.shape), cfg.deriv_tol, cfg.deriv_max_iter)
        out_g = ggamma if ctx.needs_input_grad[2] else None
        out_w = None
        if ctx.wt is not None and ctx.needs_input_grad[3]:
            v = torch.tensor(gw, dtype=ctx.wt.dtype, device=ctx.wt.device)
            out_w = v.sum().reshape(ctx.wt.shape) if ctx.wt.numel() == 1 else v.reshape(ctx.wt.shape)
        out_x = r.reshape(ctx.shape) if ctx.needs_input_grad[1] else None
        return None, out_x, out_g, out_w

    @staticmethod
    def jvp(ctx, _cfg, dx, dgamma, dwt):
        cfg = ctx.cfg
        S = _DeviceTV._structure(ctx)
        dw = None
        if dwt is not None:
            vals = [float(a) for a in dwt.detach().cpu().reshape(-1)]
            dw = vals * cfg.k if len(vals) == 1 else vals
        with torch.no_grad():
            dz, cfg.deriv = S.jvp(None if dx is None else dx.reshape(ctx.z.shape),
                                  dgamma, dw,
                                  cfg.deriv_tol, cfg.deriv_max_iter)
        return dz.reshape(ctx.shape)
