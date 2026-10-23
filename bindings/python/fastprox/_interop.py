"""Shared parts of the PyTorch and JAX wrappers: an operator (or an operator
class and its parameters) and the call options, run on NumPy arrays with the
library's own derivatives."""

from __future__ import annotations

import math

import numpy as np

from . import Operator, Spectral, _ImageOperator

_CALL = ("axis", "tol", "ztol", "max_iter")


class Spec:
    """What to compute: ``op`` is an Operator, or an Operator class built from
    the static parameters and the differentiable ones (``names``, given in
    order at each call)."""

    def __init__(self, op, static, names, conj, threads, call):
        kinds = (Operator, Spectral)
        if isinstance(op, kinds):
            if static or names:
                raise TypeError("parameters are given to an operator class, not to an operator; "
                                f"build {type(op).__name__}(...) with them or pass the class")
        elif not (isinstance(op, type) and issubclass(op, kinds)):
            raise TypeError("op must be a fastprox Operator or an Operator class")
        cls = op if isinstance(op, type) else type(op)
        if issubclass(cls, _ImageOperator) and call.get("axis") is not None:
            raise TypeError("image operators act on their last axes; axis is not accepted")
        # Spectral operators act on the last two axes and are direct.
        self.spectral = issubclass(cls, Spectral)
        if self.spectral:
            if conj:
                raise TypeError("spectral operators have no prox_conj")
            if any(call.get(k) is not None for k in _CALL):
                raise TypeError("spectral operators act on the last two axes; axis, tol, ztol and "
                                "max_iter are not accepted")
            if cls is Spectral and not isinstance(op, Spectral):
                raise TypeError("pass Spectral(op, kind) built, or a subclass such as NuclearNorm")
        self.op = op
        self.static = dict(static)
        self.names = tuple(names)
        self.conj = bool(conj)
        self.threads = int(threads)
        self.call = {k: v for k, v in call.items() if v is not None}

    def build(self, values):
        """The operator for the differentiable parameters' values (NumPy)."""
        if isinstance(self.op, (Operator, Spectral)):
            return self.op
        kw = dict(self.static)
        for k, v in zip(self.names, values):
            v = np.asarray(v, np.float64)
            kw[k] = float(v) if v.ndim == 0 else v
        return self.op(**kw)

    @property
    def kind(self):
        """'svd' or 'eigh' for a spectral operator, known before it is built."""
        return self.op.kind

    def run(self, op, x, step):
        """(z, ctx) with the structure kept for jvp and vjp."""
        f = op.prox_conj if self.conj else op.prox
        return f(x, steps(step), threads=self.threads, keep=True, **self.call)

    def row_axes(self, op, ndim):
        """The axes of x that a step does not vary over: the operator's axis,
        or the image axes."""
        if isinstance(op, _ImageOperator):
            return tuple(range(ndim - op.ndim, ndim))
        return (self.call.get("axis", -1) % ndim,) if ndim else ()

    def step_grad(self, op, ctx, g, grads, shape):
        """The gradient of a step of the given shape: grads.gamma for a
        scalar, else each row's <g, dz/dgamma> (fp_jvp with dgamma = 1)
        summed to the step's shape."""
        if math.prod(shape) == 1:
            return np.full(shape, grads.gamma)
        t = op.jvp(ctx, None, 1.0)
        return sum_to(np.sum(g * t, axis=self.row_axes(op, g.ndim)), shape)

    def step_tangent(self, op, ctx, dstep):
        """dz from a step tangent of the step's shape, one per row."""
        t = op.jvp(ctx, None, 1.0)
        axes = self.row_axes(op, t.ndim)
        batch = tuple(n for i, n in enumerate(t.shape) if i not in axes)
        return t * np.expand_dims(np.broadcast_to(dstep, batch), axes)
    def param_grads(self, op, grads, shapes):
        """The gradient of each differentiable parameter, shaped like it; NaN
        for a parameter without one."""
        out = []
        for name, shape in zip(self.names, shapes):
            if not hasattr(grads, name) or name == "gamma":
                out.append(np.full(shape, np.nan))
                continue
            v = np.asarray(getattr(grads, name), np.float64)
            size = int(np.prod(shape))
            if v.size == size:
                v = v.reshape(shape)
            elif size == 1:
                v = np.asarray(v.sum()).reshape(shape)
            else:
                raise ValueError(f"the gradient of {name!r} has {v.size} entries, the parameter {size}")
            out.append(v)
        return out

    def check_names(self, op, names=None):
        """Raise for a parameter (of names, default all) without a gradient."""
        fields = {n for n, _, _ in op._layout} | _derived(op)
        for name in self.names if names is None else names:
            if name not in fields:
                raise ValueError(f"{type(op).__name__} has no gradient with respect to {name!r}; "
                                 "pass it as a constant")

    def dparams(self, op, tangents):
        """Flat theta tangent from per-parameter tangents (None: zero)."""
        given = {k: np.asarray(t, np.float64) for k, t in zip(self.names, tangents) if t is not None}
        if not given:
            return None
        layout = {n for n, _, _ in op._layout}
        for k in given:
            if k not in layout:
                raise ValueError(f"forward-mode derivatives of {type(op).__name__} with respect to "
                                 f"{k!r} are not available")
        return op._flatten(given)


def check_step(step, conj):
    """Steps per row are for prox only, as in the NumPy API."""
    n = step.numel() if hasattr(step, "numel") else np.size(step)  # torch, or arrays and numbers
    if conj and n > 1:
        raise ValueError("prox_conj takes a scalar sigma")


def steps(step):
    """A step for the NumPy API: a float (from any single value), or (since
    1.1) an array of one step per row."""
    s = np.asarray(step, np.float64)
    return float(s.reshape(())) if s.size == 1 else s


def sum_to(a, shape):
    """a summed over the axes that broadcasting added or stretched to reach
    its shape from shape: the gradient of a broadcast step."""
    a = np.asarray(a)
    lead = a.ndim - len(shape)
    a = a.sum(axis=tuple(range(lead))) if lead else a
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and a.shape[i] != 1)
    return a.sum(axis=stretched, keepdims=True) if stretched else a


def spectral_step_grad(op, inner, a, grads, shape):
    """step_grad for a spectral operator: each matrix's <diag(A), ds/dgamma>,
    with ds/dgamma of the spectrum from the inner operator's JVP."""
    if math.prod(shape) == 1:
        return np.full(shape, grads.gamma)
    t = op.op.jvp(inner, None, 1.0)
    return sum_to(np.sum(np.diagonal(a, axis1=-2, axis2=-1) * t, axis=-1), shape)


def _derived(op):
    """Gradient fields computed from theta rather than part of it."""
    probe = op._fields(np.zeros(op.nparams))
    return set(probe) - {n for n, _, _ in op._layout}


def is_float(v) -> bool:
    """A parameter that can carry a gradient: a float scalar or float array."""
    if isinstance(v, (bool, np.bool_, str)) or v is None:
        return False
    if isinstance(v, float):
        return True
    dt = getattr(v, "dtype", None)
    return dt is not None and np.issubdtype(np.dtype(dt), np.floating)


def check_dtype(dtype):
    if np.dtype(dtype) not in (np.float32, np.float64):
        raise TypeError(f"fastprox operates on float32 and float64 arrays, got {dtype}")
