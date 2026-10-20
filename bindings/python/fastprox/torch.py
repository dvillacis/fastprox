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

Spectral operators (since 1.1) go through :func:`prox` too, as an operator
(``fp.NuclearNorm(0.5)``) or a class with its parameter
(``fp.NuclearNorm, lam=t``). The SVD or eigendecomposition and the matrix
products run in torch on the input's device; the prox of the spectrum and the
k x k core of the derivative run on the CPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import _tv_torch as _tvt
from ._interop import Spec, check_dtype

__all__ = ["prox", "prox_conj", "tv_iso", "tv_aniso"]


def prox(op, x, gamma=1.0, *, axis=None, threads=0, tol=None, max_iter=None, **params):
    r"""$\operatorname{prox}_{\gamma g}(x)$ as a differentiable tensor. ``gamma`` may be a tensor
    (0-d) that requires grad; ``axis`` (row operators), ``tol`` and
    ``max_iter`` (iterative operators) are passed on when given."""
    return _apply(op, x, gamma, False, axis, threads, tol, max_iter, params)


def prox_conj(op, y, sigma=1.0, *, axis=None, threads=0, tol=None, max_iter=None, **params):
    r"""$\operatorname{prox}_{\sigma g^*}(y)$ as a differentiable tensor (Moreau's identity);
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
    if spec.spectral:
        return _SpectralProx.apply(spec, x, step, *values)
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
# Spectral operators
# ---------------------------------------------------------------------------


def _symmetric(t):
    return (t + t.mT) / 2


class _SpectralProx(torch.autograd.Function):
    """The decomposition and the products in torch, on x's device; the prox
    of the spectrum and the k x k core of the derivative on the CPU."""

    @staticmethod
    def forward(ctx, spec, x, step, *values):
        op = spec.build([_numpy(v) for v in values])
        spec.check_names(op, [k for k, need in zip(spec.names, ctx.needs_input_grad[3:]) if need])
        if x.dim() < 2:
            raise ValueError("spectral operators act on matrices: the last two axes")
        xd = x.detach()
        if op.kind == "eigh":
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("kind='eigh' needs square matrices")
            s, u = torch.linalg.eigh(_symmetric(xd))
            vh = u.mT
        else:
            u, s, vh = torch.linalg.svd(xd, full_matrices=False)
        sn = np.ascontiguousarray(_numpy(s))
        z, inner = op.op.prox(sn, _scalar(step), threads=spec.threads, keep=True)
        ctx.spec, ctx.op, ctx.u, ctx.vh, ctx.s, ctx.z, ctx.inner = spec, op, u, vh, sn, z, inner
        ctx.step_like = step if isinstance(step, torch.Tensor) else None
        ctx.shapes = [tuple(v.shape) for v in values]
        ctx.like = [(v.dtype, v.device) for v in values]
        ctx.x_like = (x.dtype, x.device)
        return (u * torch.from_numpy(z).to(x.device)[..., None, :]) @ vh

    @staticmethod
    def _derivative(ctx, h, dgamma, dparams):
        """D(h) and A = U^T h V (NumPy), as Spectral._core."""
        op, u, vh = ctx.op, ctx.u, ctx.vh
        h = h.to(dtype=u.dtype, device=u.device)
        if op.kind == "eigh":
            h = _symmetric(h)
        a = u.mT @ h @ vh.mT
        an = _numpy(a)
        m, n = h.shape[-2:]
        b, r = op._coefficients(ctx.s, ctx.z, ctx.inner, an, dgamma, dparams, op.kind == "svd" and m != n)
        out = u @ torch.from_numpy(b).to(u.device) @ vh
        if r is not None:
            rt = torch.from_numpy(r).to(u.device)[..., None, :]
            if m > n:  # (I - U U^T) H V diag(r) V^T
                out = out + ((h @ vh.mT - u @ a) * rt) @ vh
            else:      # U diag(r) U^T H (I - V V^T)
                out = out + (u * rt) @ (u.mT @ h - a @ vh)
        return out, an

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        spec, op = ctx.spec, ctx.op
        gx, a = _SpectralProx._derivative(ctx, g, 0.0, None)
        dtype, device = ctx.x_like
        out_x = gx.to(device=device, dtype=dtype) if ctx.needs_input_grad[1] else None
        out_step = None
        out_p = [None] * len(ctx.shapes)
        if (ctx.step_like is not None and ctx.needs_input_grad[2]) or any(ctx.needs_input_grad[3:]):
            grads = op._grads(ctx.inner, a)
            if ctx.step_like is not None and ctx.needs_input_grad[2]:
                out_step = torch.full_like(ctx.step_like, grads.gamma)
            vals = spec.param_grads(op, grads, ctx.shapes)
            for i, (v, (dt, dev)) in enumerate(zip(vals, ctx.like)):
                if ctx.needs_input_grad[3 + i]:
                    out_p[i] = torch.from_numpy(v).to(device=dev, dtype=dt)
        return (None, out_x, out_step, *out_p)

    @staticmethod
    def jvp(ctx, _spec, dx, dstep, *dvalues):
        op = ctx.op
        dparams = ctx.spec.dparams(op, [None if t is None else _numpy(t) for t in dvalues])
        dtype, device = ctx.x_like
        h = torch.zeros(ctx.u.shape[:-1] + ctx.vh.shape[-1:], dtype=dtype, device=device) if dx is None else dx
        dz, _ = _SpectralProx._derivative(ctx, h, 0.0 if dstep is None else _scalar(dstep), dparams)
        return dz.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Device-native image TV
# ---------------------------------------------------------------------------


def tv_iso(x, gamma=1.0, *, ndim=2, weights=None, pixel_weights=None, tol=1e-6, max_iter=1000,
           warm_start=None, return_info=False, deriv_tol=None, deriv_max_iter=10000):
    r"""Isotropic TV of ``TVIso(ndim, weights)`` on the last ``ndim`` axes of
    ``x``, computed in tensor operations on ``x``'s device and differentiable
    in ``x``, ``gamma`` and ``weights`` (tensors that require grad), forward
    and reverse. ``gamma`` is a number or one step per image: a tensor that
    broadcasts to the batch shape ``x.shape[:-ndim]``. Each image stops when its duality gap is at most
    $\mathrm{tol} \cdot \lVert x \rVert_2^2 / 2$. ``warm_start`` takes the ``dual`` of an earlier
    call (shape (ndim, *x.shape), the C kernel's form). With
    ``return_info=True`` returns ``(z, info)``, an :class:`IterInfo` whose
    ``dual`` is a tensor. The derivatives are the C kernel's: one unknown per
    flat region, solved by conjugate gradients to relative residual
    ``deriv_tol``; by default each image's solve stops at the relative error
    of $z$ that its gap certifies, $\sqrt{2\,\mathrm{gap}} / \lVert x \rVert_2$, and at
    $\max(10^{-10}, 100\,\varepsilon)$ at the tightest. ``pixel_weights`` (since
    0.3): $\alpha \ge 0$ in $\gamma \sum_p \alpha_p \lVert (w D z)_p \rVert_2$, a
    tensor that broadcasts to x.shape (one map for every image, or one per image, e.g. predicted by a network),
    differentiable like gamma."""
    return _device_tv(True, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter, pixel_weights)


def tv_aniso(x, gamma=1.0, *, ndim=2, weights=None, edge_weights=None, tol=1e-6, max_iter=1000,
             warm_start=None, return_info=False, deriv_tol=None, deriv_max_iter=10000):
    r"""Anisotropic TV of ``TVAniso(ndim, weights)``, as :func:`tv_iso`. A plain
    dual method on the device (the C kernel's exact 1D steps and polish stay
    on the CPU), so tight tolerances take more iterations than ``TVAniso``;
    the dual is in the C kernel's form ($q_a$ with $x - z = \sum_a q_a$). Its
    derivatives are exact region averages. ``edge_weights`` (since 0.3):
    $\beta \ge 0$ in
    $\gamma \sum_a w_a \sum_p \beta[a, \dots, p] \lvert z_{p + e_a} - z_p \rvert$, a
    tensor that broadcasts to (ndim, *x.shape) ($\beta[a]$ at the last entry
    along $a$ unused), differentiable like gamma."""
    return _device_tv(False, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info,
                      deriv_tol, deriv_max_iter, edge_weights)


class _TVConfig:
    def __init__(self, iso, ndim, w, tol, max_iter, warm, deriv_tol, deriv_max_iter):
        self.iso, self.k, self.w = iso, ndim, w
        self.tol, self.max_iter, self.warm = float(tol), int(max_iter), warm
        self.deriv_tol, self.deriv_max_iter = deriv_tol, int(deriv_max_iter)
        self.info = None
        self.deriv = None


def _device_tv(iso, x, gamma, ndim, weights, tol, max_iter, warm_start, return_info, deriv_tol,
               deriv_max_iter, spatial=None):
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
    sw = None if spatial is None else _spatial(spatial, x, ndim, iso)
    z = _DeviceTV.apply(cfg, x, gb, wt, sw)
    return (z, cfg.info) if return_info else z


def _spatial(sw, x, ndim, iso):
    """Pixel weights that broadcast to x.shape, as (B, *img), or edge
    weights of shape (ndim, *s) with s broadcasting to x.shape, as (ndim, B,
    *img). Tensors stay in the graph, so their gradient sums back to their
    own shape."""
    name = "pixel_weights" if iso else "edge_weights"
    full = tuple(x.shape) if iso else (ndim, *x.shape)
    t = sw if isinstance(sw, torch.Tensor) else torch.as_tensor(np.asarray(sw, np.float64))
    t = t.to(dtype=x.dtype, device=x.device)
    try:
        if not iso:  # the leading axis is the image axis; the rest broadcasts to x
            if t.dim() < 1 or t.shape[0] != ndim or t.dim() - 1 > x.dim():
                raise RuntimeError
            t = t.reshape((ndim,) + (1,) * (x.dim() + 1 - t.dim()) + tuple(t.shape[1:]))
        if torch.broadcast_shapes(t.shape, full) != full:
            raise RuntimeError
    except RuntimeError:
        what = "broadcast to" if iso else f"have shape ({ndim}, ...) broadcasting to"
        raise ValueError(f"{name} must {what} {full}, got shape {tuple(sw.shape) if hasattr(sw, 'shape') else ()}") \
            from None
    td = t.detach()
    if bool(((td < 0) | ~torch.isfinite(td)).any()):
        raise ValueError(f"{name} must be finite and >= 0")
    img = tuple(x.shape[-ndim:])
    return t.expand(full).reshape((-1,) + img if iso else (ndim, -1) + img)


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
    def forward(ctx, cfg, x, gamma, wt, sw):
        k = cfg.k
        img = tuple(x.shape[-k:])
        xb = x.detach().reshape((-1,) + img)
        gam = gamma.detach()  # (B,), one per image
        u0 = None
        if cfg.warm is not None:
            u0 = _tvt.dual_in(cfg.warm.reshape((k, -1) + img), cfg.w, cfg.iso, k, gam)
        with torch.no_grad():
            ctx.sw = None if sw is None else sw.detach()
            z, u, gap, it, conv = _tvt.solve(xb, gam, cfg.w, cfg.iso, cfg.tol, cfg.max_iter, u0, ctx.sw)
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
                ctx.S = _tvt.Structure(ctx.xb, ctx.z, ctx.u, ctx.gap, ctx.gam, cfg.w, cfg.iso, ctx.sw)
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
        out_s = S.gsw if ctx.sw is not None and ctx.needs_input_grad[4] else None
        return None, out_x, out_g, out_w, out_s

    @staticmethod
    def jvp(ctx, _cfg, dx, dgamma, dwt, dsw):
        cfg = ctx.cfg
        S = _DeviceTV._structure(ctx)
        dw = None
        if dwt is not None:
            vals = [float(a) for a in dwt.detach().cpu().reshape(-1)]
            dw = vals * cfg.k if len(vals) == 1 else vals
        with torch.no_grad():
            dz, cfg.deriv = S.jvp(None if dx is None else dx.reshape(ctx.z.shape),
                                  dgamma, dw,
                                  cfg.deriv_tol, cfg.deriv_max_iter,
                                  None if ctx.sw is None else dsw)
        return dz.reshape(ctx.shape)
