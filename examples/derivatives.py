"""Derivatives of a prox: JVP, VJP and parameter gradients.

``op.prox(x, gamma, keep=True)`` saves the structure of the solution in a
context. From it, ``op.jvp`` applies the Jacobian of z = prox_{gamma g}(x) to
directions in x, gamma and the operator's parameters, and ``op.vjp`` applies
its transpose. Both are exact for the saved structure (an element of the
generalised Jacobian at kinks, see docs/api.md) and cost about one pass over
the data. This script checks them against finite differences.

    uv run python examples/derivatives.py
"""

import numpy as np

import fastprox as fp


def main():
    rng = np.random.default_rng(0)
    n = 100
    x = np.cumsum(rng.normal(size=(8, n)), axis=1)    # 8 random walks, one per row
    w = rng.uniform(0.5, 2.0, size=n - 1)             # one TV weight per edge
    gamma = 1.5

    op = fp.TV1D(weights=w)
    z, ctx = op.prox(x, gamma, keep=True)
    print(op, "parameters:", op.nparams, " certificate:", ctx.info)

    # -- JVP: derivative of z along a direction in (x, gamma, w) ------------
    dx = rng.normal(size=x.shape)
    dgamma = 0.3
    dw = rng.normal(size=w.shape)
    dz = op.jvp(ctx, dx=dx, dgamma=dgamma, dparams={"weights": dw})

    eps = 1e-6
    def prox_at(s):
        return fp.TV1D(weights=w + s * dw).prox(x + s * dx, gamma + s * dgamma)
    fd = (prox_at(eps) - prox_at(-eps)) / (2 * eps)
    print(f"JVP vs finite differences:  max error {np.abs(dz - fd).max():.1e}")

    # -- VJP: gradients of a scalar loss L(z) -------------------------------
    target = rng.normal(size=x.shape)
    g = z - target                                    # dL/dz for L = ||z - target||^2 / 2
    gx, grads = op.vjp(ctx, g=g)
    print(grads.weights.shape, "edge-weight gradients; dL/dgamma =", f"{grads.gamma:.4f}")

    # Adjointness: <g, J d> = <J^T g, d> for the same direction d.
    lhs = np.vdot(g, dz)
    rhs = np.vdot(gx, dx) + grads.gamma * dgamma + np.vdot(grads.weights, dw)
    print(f"<g, J d> = {lhs:.10f}\n<J^T g, d> = {rhs:.10f}")

    # -- Repeated calls: pass ctx back to reuse its memory ------------------
    for gamma in (0.5, 1.0, 2.0):
        z, ctx = op.prox(x, gamma, keep=True, ctx=ctx)
        _, grads = op.vjp(ctx, g=z - target)
        print(f"gamma = {gamma}: loss {0.5 * np.sum((z - target) ** 2):9.3f}, "
              f"dL/dgamma {grads.gamma:9.3f}")

    # -- Other operators: the same calls, named parameter gradients ---------
    v = rng.normal(size=(4, 20))
    for op in (fp.Simplex(radius=2.0), fp.Soft(lam=0.5), fp.Huber(lam=1.0, delta=0.5)):
        z, ctx = op.prox(v, 1.0, keep=True)
        _, grads = op.vjp(ctx, g=np.ones_like(v))
        print(f"{op!r:28} {grads}")


if __name__ == "__main__":
    main()
