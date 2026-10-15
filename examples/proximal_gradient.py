"""Proximal gradient methods with fastprox as the prox step.

1. Lasso,  min_x  ||A x - b||^2 / 2 + lam ||x||_1,  by FISTA with soft thresholding.
2. Least squares over the simplex,  min_{x in simplex}  ||A x - b||^2 / 2,
   by accelerated projected gradient, solving a batch of problems at once:
   each row of X is a separate problem and one call projects all of them.

Both stop on the fixed-point residual ||x - prox_{t g}(x - t grad f(x))|| / t,
which is zero exactly at a solution.

    uv run python examples/proximal_gradient.py
"""

import numpy as np

import fastprox as fp


def fista(grad, prox, x0, t, iters=500, tol=1e-8):
    """FISTA (Beck and Teboulle, 2009) with step t <= 1/L."""
    x = y = x0
    s = 1.0
    for k in range(iters):
        x_new = prox(y - t * grad(y), t)
        s_new = (1 + np.sqrt(1 + 4 * s * s)) / 2
        y = x_new + ((s - 1) / s_new) * (x_new - x)
        x, s = x_new, s_new
        res = np.linalg.norm(x - prox(x - t * grad(x), t)) / t
        if res < tol:
            break
    return x, k + 1, res


def lasso(rng):
    m, n, k = 100, 300, 10
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    x_true[rng.choice(n, k, replace=False)] = rng.normal(scale=3.0, size=k)
    b = A @ x_true + 0.01 * rng.normal(size=m)
    lam = 0.02

    t = 1 / np.linalg.norm(A, 2) ** 2
    x, iters, res = fista(lambda x: A.T @ (A @ x - b),
                          lambda v, t: fp.soft_threshold(v, lam=lam, gamma=t),
                          np.zeros(n), t, iters=5000)
    print(f"lasso: {iters} iterations, residual {res:.1e}")
    print(f"  support found {np.flatnonzero(x).size} entries, "
          f"true support recovered: {set(np.flatnonzero(x_true)) <= set(np.flatnonzero(x))}")
    print(f"  relative error {np.linalg.norm(x - x_true) / np.linalg.norm(x_true):.3f}")


def simplex_least_squares(rng):
    # 32 problems sharing A: each row of B is a mixture to unmix into weights on
    # the 20 columns of A (nonnegative, summing to 1).
    m, n, p = 50, 20, 32
    A = rng.uniform(size=(m, n))
    W_true = fp.project_simplex(rng.normal(scale=2.0, size=(p, n)))
    B = W_true @ A.T + 0.01 * rng.normal(size=(p, m))

    t = 1 / np.linalg.norm(A, 2) ** 2
    W, iters, res = fista(lambda W: (W @ A.T - B) @ A,
                          lambda V, t: fp.project_simplex(V),
                          np.full((p, n), 1 / n), t, iters=5000, tol=1e-6)
    print(f"simplex least squares, {p} problems at once: {iters} iterations, "
          f"residual {res:.1e}")
    print(f"  rows sum to 1: {np.allclose(W.sum(axis=1), 1)}, min entry {W.min():.1e}, "
          f"max error vs true weights {np.abs(W - W_true).max():.3f}")


def main():
    rng = np.random.default_rng(0)
    lasso(rng)
    simplex_least_squares(rng)


if __name__ == "__main__":
    main()
