r"""PyProximal adapter (since 1.0): a fastprox operator as a
``pyproximal.ProxOperator``, for PyProximal's solvers (proximal gradient,
ADMM, primal-dual, ...).

    import pyproximal
    import fastprox as fp
    from fastprox.pyproximal import ProxOp

    g = ProxOp(fp.TVIso(), dims=(64, 64), tol=1e-8)       # isotropic TV of an image
    x = pyproximal.optimization.primal.ProximalGradient(f, g, x0, tau=1.0, epsg=0.1)

PyProximal passes flat vectors; ``dims`` gives the array shape the operator
acts on (by default the vector itself: one row). The operator is then
applied as it would be to an array of that shape, so ``dims=(B, n)`` is a
batch of ``B`` rows of length ``n`` for a row operator, and ``(B, H, W)`` a
batch of images for an image operator. The value $g(x)$ is the sum over the
batch.

PyProximal is a separate package (LGPL-3.0), not a dependency of fastprox:
install it to use this module. The adapter is written against PyProximal's
documented interface.
"""

from __future__ import annotations

import numbers

import numpy as np

try:
    from pyproximal import ProxOperator
except ImportError as e:  # pragma: no cover - exercised without PyProximal
    raise ImportError("fastprox.pyproximal needs PyProximal: pip install pyproximal") from e

from . import Operator, _ImageOperator

__all__ = ["ProxOp"]


class ProxOp(ProxOperator):
    r"""``op`` as a PyProximal operator.

    - ``op(x)`` is $g(x)$ (``op.value``, summed over the batch), with the
      indicator of a set taken to tolerance ``value_tol``.
    - ``prox(x, tau)`` is $\operatorname{prox}_{\tau g}(x)$ and
      ``proxdual(x, tau)`` is $\operatorname{prox}_{\tau g^\ast}(x)$ (fastprox's
      ``prox_conj``, by the Moreau identity), both for a scalar ``tau > 0``.

    ``dims``: the shape of the array the operator acts on (default: the
    flat vector). ``warm``: for the iterative image operators, start each
    solve from the dual of the previous one, which saves iterations when a
    solver calls the prox at nearby points. The other keywords (``axis``,
    ``tol``, ``max_iter``, ``threads``, ...) are passed to ``op.prox`` and
    ``op.prox_conj``; ``axis`` also to ``op.value``.
    """

    def __init__(self, op: Operator, dims=None, *, warm: bool = False, value_tol: float = 1e-6,
                 **prox_kwargs):
        if not isinstance(op, Operator):
            raise TypeError(f"expected a fastprox operator, got {type(op).__name__}")
        super().__init__(None, False)
        self.op = op
        self.dims = None if dims is None else tuple(int(d) for d in np.atleast_1d(dims))
        self.warm = bool(warm) and isinstance(op, _ImageOperator)
        self.value_tol = float(value_tol)
        self.prox_kwargs = dict(prox_kwargs)
        self._ctx = {False: None, True: None}  # warm-start contexts of prox and proxdual

    def __repr__(self) -> str:
        return f"ProxOp({self.op!r}, dims={self.dims})"

    def _shaped(self, x):
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)
        return x.reshape(self.dims) if self.dims is not None else x

    def __call__(self, x) -> float:
        kw = {"axis": self.prox_kwargs["axis"]} if "axis" in self.prox_kwargs else {}
        return float(np.sum(self.op.value(self._shaped(x), tol=self.value_tol, **kw)))

    def _apply(self, x, tau, conj: bool):
        if not isinstance(tau, numbers.Real) and np.ndim(tau) != 0:
            raise TypeError("fastprox operators take one scalar tau per call")
        tau = float(tau)
        if not tau > 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        shape = np.shape(x)
        y = self._shaped(x)
        run = self.op.prox_conj if conj else self.op.prox
        if self.warm:
            z, self._ctx[conj] = run(y, tau, keep=True, ctx=self._ctx[conj],
                                     warm=self._ctx[conj] is not None, **self.prox_kwargs)
        else:
            z = run(y, tau, **self.prox_kwargs)
        return z.reshape(shape)

    def prox(self, x, tau):
        r"""$\operatorname{prox}_{\tau g}(x)$, of the shape of ``x``."""
        return self._apply(x, tau, False)

    def proxdual(self, x, tau):
        r"""$\operatorname{prox}_{\tau g^\ast}(x) = x - \tau \operatorname{prox}_{g/\tau}(x/\tau)$,
        of the shape of ``x``."""
        return self._apply(x, tau, True)
