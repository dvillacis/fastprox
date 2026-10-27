"""The hypergradient of 1D TV deblurring, three ways (since 1.0.2).

The lower level recovers a piecewise-constant signal from a blurred, noisy
copy b = A x + noise,

    z(lam) = argmin_z  ||A z - b||^2 / 2 + lam TV(z),

by forward-backward splitting, whose fixed point is

    z = T(z, lam) = prox_{tau lam TV}(z - tau A^T (A z - b)),

and the upper level measures the error L(lam) = ||z(lam) - x||^2 / 2. Its
derivative dL/dlam is computed three ways:

1. The fixed-point adjoint: p = g + (I - tau A^T A) J^T p, with g = z - x and
   J^T from ``vjp``, then dL/dlam = tau <p, d prox/d gamma>, also from ``vjp``.
2. The reduced system from the structure of the derivative. J is the
   projector onto the signals constant on the segments of z (``kind`` is
   "projector"), so dz/dlam = M c, with M the segment matrix, and
   M^T A^T A M y = M^T g has one unknown per segment instead of one per
   sample. Then dL/dlam = <M y, d prox/d gamma>: tau drops out.
3. Central finite differences of L, solving the lower level again.

    uv run python examples/bilevel_deblurring.py
"""

import numpy as np

import fastprox as fp

N = 120
LAM = 0.05


def blur(n):
    """The blur [0.2, 0.6, 0.2] with zero boundary: 0.04 <= eig(A^T A) <= 1."""
    return 0.6 * np.eye(n) + 0.2 * (np.eye(n, k=1) + np.eye(n, k=-1))


def lower(op, A, b, lam, tau, z=None, iters=5000):
    """The forward-backward fixed point, to rounding; returns (z, ctx) with
    ctx kept at the last prox, whose input is z - tau A^T (A z - b)."""
    z = np.zeros(A.shape[1]) if z is None else z
    for _ in range(iters):
        z_new, ctx = op.prox(z - tau * (A.T @ (A @ z - b)), tau * lam, keep=True)
        if np.array_equal(z_new, z):
            break
        z = z_new
    return z_new, ctx


def cg(apply, rhs, tol=1e-15):
    """Conjugate gradients for a small symmetric positive definite system."""
    y = np.zeros_like(rhs)
    r = rhs.copy()
    d = r.copy()
    rr = r @ r
    for _ in range(4 * len(rhs)):
        if np.sqrt(rr) <= tol * np.linalg.norm(rhs):
            break
        Ad = apply(d)
        a = rr / (d @ Ad)
        y += a * d
        r -= a * Ad
        rr, rr_old = r @ r, rr
        d = r + (rr / rr_old) * d
    return y


def main():
    rng = np.random.default_rng(3)
    edges = np.sort(rng.choice(np.arange(5, N - 5), 7, replace=False))
    x = np.repeat(rng.normal(size=8), np.diff(np.r_[0, edges, N]))
    A = blur(N)
    b = A @ x + 0.05 * rng.normal(size=N)
    tau = 1.0 / np.linalg.norm(A, 2) ** 2
    op = fp.TV1D()

    z, ctx = lower(op, A, b, LAM, tau)
    g = z - x
    print(f"lower level: n = {N}, L(lam) = {0.5 * g @ g:.6f}")

    # 1. The fixed-point adjoint, by iteration (a contraction: |I - tau A^T A| < 1).
    p = g.copy()
    for _ in range(5000):
        p_new = g + (p - tau * (A.T @ (A @ op.vjp(ctx, p)[0])))
        if np.array_equal(p_new, p):
            break
        p = p_new
    _, grads = op.vjp(ctx, p)
    adjoint = tau * float(grads.gamma)

    # 2. The reduced system, on one unknown per segment (matrix-free CG).
    s = ctx.structure()
    K = int(s.offsets[-1])
    y = cg(lambda c: s.reduce(A.T @ (A @ s.expand(c))), s.reduce(g))
    dz_dgamma = op.jvp(ctx, np.zeros(N), dgamma=1.0)
    reduced = float(s.expand(y) @ dz_dgamma)
    print(f"structure:   kind {s.kind!r}, {K} segments, so {K} unknowns instead of {N}")

    # 3. Central differences. z is affine in lam while the segments stay, so
    #    L is quadratic there and the difference is exact up to rounding.
    h = 1e-4 * LAM
    zp, cp = lower(op, A, b, LAM + h, tau, z)
    zm, cm = lower(op, A, b, LAM - h, tau, z)
    assert np.array_equal(cp.structure().labels, s.labels)
    assert np.array_equal(cm.structure().labels, s.labels)
    fd = (0.5 * np.sum((zp - x) ** 2) - 0.5 * np.sum((zm - x) ** 2)) / (2 * h)

    print(f"dL/dlam:     fixed-point adjoint {adjoint: .15e}")
    print(f"             reduced system      {reduced: .15e}")
    print(f"             finite differences  {fd: .15e}")
    e_red = abs(reduced - adjoint) / abs(adjoint)
    e_fd = abs(fd - adjoint) / abs(adjoint)
    print(f"relative differences: reduced {e_red:.1e}, finite differences {e_fd:.1e}")
    return adjoint, reduced, fd


if __name__ == "__main__":
    main()
