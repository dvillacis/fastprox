"""Shared parts of the PyTorch and JAX wrappers: an operator (or an operator
class and its parameters) and the call options, run on NumPy arrays with the
library's own derivatives."""

from __future__ import annotations

import numpy as np

from . import Operator, _ImageOperator

_CALL = ("axis", "tol", "max_iter")


class Spec:
    """What to compute: ``op`` is an Operator, or an Operator class built from
    the static parameters and the differentiable ones (``names``, given in
    order at each call)."""

    def __init__(self, op, static, names, conj, threads, call):
        if isinstance(op, Operator):
            if static or names:
                raise TypeError("parameters are given to an operator class, not to an operator; "
                                f"build {type(op).__name__}(...) with them or pass the class")
        elif not (isinstance(op, type) and issubclass(op, Operator)):
            raise TypeError("op must be a fastprox Operator or an Operator class")
        image = issubclass(op if isinstance(op, type) else type(op), _ImageOperator)
        if image and call.get("axis") is not None:
            raise TypeError("image operators act on their last axes; axis is not accepted")
        self.op = op
        self.static = dict(static)
        self.names = tuple(names)
        self.conj = bool(conj)
        self.threads = int(threads)
        self.call = {k: v for k, v in call.items() if v is not None}

    def build(self, values):
        """The operator for the differentiable parameters' values (NumPy)."""
        if isinstance(self.op, Operator):
            return self.op
        kw = dict(self.static)
        for k, v in zip(self.names, values):
            v = np.asarray(v, np.float64)
            kw[k] = float(v) if v.ndim == 0 else v
        return self.op(**kw)

    def run(self, op, x, step):
        """(z, ctx) with the structure kept for jvp and vjp."""
        f = op.prox_conj if self.conj else op.prox
        return f(x, float(step), threads=self.threads, keep=True, **self.call)

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
