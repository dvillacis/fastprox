"""SCD semismooth* Newton steps from the structure of the derivative (since 1.0.2).

For phi = f + g with f(x) = ||A x - b||^2 / 2 and g convex, Algorithm 4.2 of
Gfrerer, Hubmer and Ramlau (Inverse Problems 41 (2025) 075002; Algorithm 3.8
with their sum rule, Lemma 4.1) alternates two steps:

- AS, a forward-backward step: x_hat = prox_{t g}(x - t grad f(x)) and
  y_hat = (x - x_hat) / t - (grad f(x) - grad f(x_hat)), an element of
  d phi(x_hat);
- NS, a stationary point s of the quadratic program
  min_s s^T (P H P + W) s / 2 + <y_hat, s> over s in rng P, with H = A^T A,
  and x = x_hat + s.

(P, W) describes a subspace of the SC derivative of d g at the prox point.
When the prox's Jacobian J is the orthogonal projector onto a subspace L
(``kind`` "projector"), it is P = P_L and W = I - P (docs/api.md, "Structure
of the derivative"): the structure alone gives the step, which then solves
P (H s + y_hat) = 0 on rng P. Once the structure of the solution is found the
next step lands on the solution, up to rounding.

Two problems: the lasso (g = alpha ||x||_1, P keeps the support) and 1D TV
deblurring (g = alpha TV, P averages over the segments). Each Newton step is
checked against the objective and replaced by the plain forward-backward step
when it does not decrease it, a simple safeguard in place of the paper's
globalisation (Algorithm 4.5).

    uv run python examples/scd_newton_lasso.py
"""

import numpy as np

import fastprox as fp


def scd_newton(op, A, b, alpha, value, iters=30, tol=1e-12):
    """Algorithm 4.2 with P from the structure and W = I - P; returns the
    solution and the history (residual |y_hat|, entries fixed, regions, and
    whether the Newton step was taken)."""
    n = A.shape[1]
    H, Atb = A.T @ A, A.T @ b
    t = 1.0 / np.linalg.norm(A, 2) ** 2
    phi = lambda x: 0.5 * np.sum((A @ x - b) ** 2) + alpha * value(x)
    x, hist = np.zeros(n), []
    for _ in range(iters):
        grad = H @ x - Atb
        x_hat, ctx = op.prox(x - t * grad, t * alpha, keep=True)        # AS
        y_hat = (x - x_hat) / t - (grad - (H @ x_hat - Atb))
        s = ctx.structure()
        r = np.linalg.norm(y_hat)
        if r <= tol * np.linalg.norm(Atb):
            hist.append((r, s.nfixed, int(s.offsets[-1]), None))
            return x_hat, hist
        P = np.column_stack([s.project(e) for e in np.eye(n)])          # NS
        W = np.eye(n) - P
        step = np.linalg.solve(P @ H @ P + W, -P @ y_hat)              # Lemma 3.7 (ii)
        newton = bool(phi(x_hat + step) <= phi(x_hat))
        hist.append((r, s.nfixed, int(s.offsets[-1]), newton))
        x = x_hat + step if newton else x_hat
    return x_hat, hist


def fista_iterations(op, A, b, alpha, x_ref, iters=100000):
    """FISTA iterations to the same distance from x_ref, for comparison."""
    H, Atb = A.T @ A, A.T @ b
    t = 1.0 / np.linalg.norm(A, 2) ** 2
    x = v = np.zeros(A.shape[1])
    q = 1.0
    goal = 1e-10 * np.linalg.norm(x_ref)
    for k in range(1, iters + 1):
        x_new = op.prox(v - t * (H @ v - Atb), t * alpha)
        q_new = 0.5 * (1 + np.sqrt(1 + 4 * q * q))
        v = x_new + ((q - 1) / q_new) * (x_new - x)
        x, q = x_new, q_new
        if np.linalg.norm(x - x_ref) <= goal:
            return k
    return iters


def report(name, op, A, b, alpha, value):
    x, hist = scd_newton(op, A, b, alpha, value)
    print(f"{name}: n = {A.shape[1]}")
    for k, (r, nfixed, nreg, newton) in enumerate(hist):
        took = {True: "Newton step", False: "forward-backward step", None: "converged"}[newton]
        print(f"  {k}: |y_hat| = {r:8.2e}, fixed entries {nfixed:3d}, regions {nreg:3d}, {took}")
    print(f"  FISTA to the same point (1e-10 relative): {fista_iterations(op, A, b, alpha, x)} iterations")
    return x, hist


def main():
    rng = np.random.default_rng(1)

    # The lasso: 300 observations, 200 unknowns (A^T A positive definite, so
    # every Newton step is defined), 10 nonzero.
    m, n = 300, 200
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    x0 = np.zeros(n)
    x0[rng.choice(n, 10, replace=False)] = 3 * rng.normal(size=10)
    b = A @ x0 + 0.05 * rng.normal(size=m)
    alpha = 0.1 * np.abs(A.T @ b).max()
    lasso, h1 = report("lasso", fp.Soft(1.0), A, b, alpha, lambda x: np.abs(x).sum())
    # KKT: A^T (b - A x) = alpha sign(x) on the support, within [-alpha, alpha] off it.
    c = A.T @ (b - A @ lasso)
    on = lasso != 0
    assert np.allclose(c[on], alpha * np.sign(lasso[on]), rtol=0, atol=1e-10 * alpha)
    assert np.all(np.abs(c[~on]) <= alpha * (1 + 1e-10))
    print(f"  support {int(on.sum())}, KKT holds")

    # 1D TV deblurring with the blur [0.2, 0.6, 0.2].
    n = 150
    A = 0.6 * np.eye(n) + 0.2 * (np.eye(n, k=1) + np.eye(n, k=-1))
    edges = np.sort(rng.choice(np.arange(5, n - 5), 5, replace=False))
    x0 = np.repeat(rng.normal(size=6), np.diff(np.r_[0, edges, n]))
    b = A @ x0 + 0.02 * rng.normal(size=n)
    tv, h2 = report("1D TV deblurring", fp.TV1D(), A, b, 0.02, lambda x: np.abs(np.diff(x)).sum())
    return h1, h2


if __name__ == "__main__":
    main()
