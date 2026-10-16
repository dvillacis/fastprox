"""Recover sparse signals with convex and nonconvex penalties (since 0.2).

Each of 64 problems observes b = A x0 + noise with A of size 80 x 256 and a
16-sparse x0, and solves

    min_x  ||A x - b||^2 / 2 + g(x)

by proximal gradient, x <- prox_{t g}(x - t A^T (A x - b)) with t = 1/||A||^2.
The l1 norm keeps every true entry but shrinks each amplitude by lam; at a
lam small enough to limit that bias it also keeps many spurious entries.
The nonconvex penalties
(MCP, SCAD, l0, lq, log) stop shrinking large entries and get close to the
oracle that knows the support. They are not convex, so proximal gradient
finds a stationary point; starting from the l1 solution, as is common, gives
a good one. All 64 problems run as one batch: the maps are elementwise.

    uv run python examples/sparse_recovery.py
"""

import numpy as np

import fastprox as fp

M, N, K, TRIALS = 80, 256, 16, 64


def make_problems(rng):
    A = rng.normal(size=(M, N)) / np.sqrt(M)
    x0 = np.zeros((TRIALS, N))
    for row in x0:
        idx = rng.choice(N, K, replace=False)
        row[idx] = rng.choice([-1, 1], K) * rng.uniform(1, 3, K)
    b = x0 @ A.T + 0.05 * rng.normal(size=(TRIALS, M))
    return A, x0, b


def proximal_gradient(op, A, b, x, iters=500):
    t = 1.0 / np.linalg.norm(A, 2) ** 2
    for _ in range(iters):
        x = op.prox(x - t * ((x @ A.T - b) @ A), t)
    return x


def report(name, x, x0):
    found, true = x != 0, x0 != 0
    exact = np.mean(np.all(found == true, axis=1))
    err = np.median(np.linalg.norm(x - x0, axis=1) / np.linalg.norm(x0, axis=1))
    extra = np.mean(np.sum(found & ~true, axis=1))
    missed = np.mean(np.sum(~found & true, axis=1))
    print(f"{name:22s} {exact:6.0%}  {err:9.3f}  {extra:6.1f}  {missed:6.1f}")
    return exact, err


def main():
    rng = np.random.default_rng(0)
    A, x0, b = make_problems(rng)

    # Weights chosen per penalty from a small grid; each has its own scale.
    print(f"{'penalty':22s} {'exact':>6s}  {'rel. err':>9s}  {'extra':>6s}  {'missed':>6s}")
    x1 = proximal_gradient(fp.Soft(0.1), A, b, np.zeros_like(x0))
    results = {"l1": report("l1, lam 0.1", x1, x0)}
    for name, op in (("MCP, lam 0.2, b 3", fp.MCP(0.2, b=3.0)),
                     ("SCAD, lam 0.2, a 3.7", fp.SCAD(0.2, a=3.7)),
                     ("l0, lam 0.15", fp.L0(0.15)),
                     ("l1/2, lam 0.1", fp.Lq(0.1, q=0.5)),
                     ("log, lam 0.1, eps 0.1", fp.LogPenalty(0.1, eps=0.1))):
        results[name.split(",")[0]] = report(name, proximal_gradient(op, A, b, x1), x0)

    # The oracle: least squares on the true support of each problem.
    oracle = np.zeros_like(x0)
    for xo, xt, bt in zip(oracle, x0, b):
        s = xt != 0
        xo[s] = np.linalg.lstsq(A[:, s], bt, rcond=None)[0]
    report("oracle (known support)", oracle, x0)

    # Where the prox is set-valued, tie= picks the minimiser and ctx.info.ties
    # counts the entries that sat on the tie. Hard thresholding with lam = 1/2
    # and step 1 keeps |x| > 1, zeroes |x| < 1, and |x| = 1 is a tie.
    v = np.array([2.0, 1.0, -1.0, 0.5, -3.0])
    for tie in ("nonzero", "zero"):
        z, ctx = fp.L0(0.5).prox(v, 1.0, keep=True, tie=tie)
        print(f"hard threshold of {v}, tie={tie!r:9s}: {z}, {ctx.info.ties} ties")
    return results


if __name__ == "__main__":
    main()
