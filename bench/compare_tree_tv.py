"""Tree TV (dynamic programming) on four kinds of tree, single thread.

  uv run python bench/compare_tree_tv.py [reps]

* random recursive trees (each node joins a uniformly random earlier one,
  labels shuffled), against graph TV on the same edges up to 10^5 nodes;
* combs: an h x h grid's rows joined by its first column, the spanning tree
  of Kolmogorov, Pock and Rolinek's image experiments, against graph TV;
* stars (one hub), against graph TV on 3000 nodes only: its max-flow is
  quadratic there;
* paths, in order and with shuffled labels, against TV1D on the ordered
  signal.

Signals are the pattern of bench/compare_graph_tv.py (or a random walk on
paths) plus Gaussian noise 0.1. Medians of `reps` runs.
"""

import sys
import time

import numpy as np

import fastprox as fp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_graph_tv import pattern  # noqa: E402


def timed(f, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def random_tree(n, rng):
    i = np.arange(1, n)
    e = np.stack([i, (rng.random(n - 1) * i).astype(np.int64)], 1)
    return rng.permutation(n)[e]


def comb(h):
    idx = np.arange(h * h).reshape(h, h)
    return np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                           np.stack([idx[:-1, 0], idx[1:, 0]], 1)])


def row(label, op, x, gamma, reps, ref=None):
    t = timed(lambda: op.solve(x, gamma, threads=1), reps)
    out = f"  {label} gamma {gamma:<4}: tree {t:8.1f}"
    if ref is not None:
        name, f = ref
        out += f" | {name} {timed(f, max(1, reps if name != 'graph' else 1)):8.1f}"
    print(out)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rng = np.random.default_rng(7)
    gammas = (0.02, 0.1, 0.5)
    print("random trees (ms)")
    for n in (10**4, 10**5, 10**6):
        e = random_tree(n, rng)
        p = rng.random((n, 2))
        x = pattern(p[:, 0], p[:, 1]) + 0.1 * rng.standard_normal(n)
        op, g = fp.TreeTV(n, e), fp.GraphTV(n, e)
        for gamma in gammas:
            ref = ("graph", lambda: g.solve(x, gamma, threads=1)) if n <= 10**5 else None
            row(f"n {n:>7}", op, x, gamma, reps, ref)
    print("combs (ms)")
    for h in (316, 1000):
        a, b = np.mgrid[0:h, 0:h] / h
        x = (pattern(a, b) + 0.1 * rng.standard_normal((h, h))).ravel()
        e = comb(h)
        op, g = fp.TreeTV(h * h, e), fp.GraphTV(h * h, e)
        for gamma in gammas:
            row(f"{h}^2", op, x, gamma, reps, ("graph", lambda: g.solve(x, gamma, threads=1)))
    print("stars (ms)")
    for n in (3000, 10**5):
        e = np.stack([np.zeros(n - 1, np.int64), np.arange(1, n)], 1)
        x = pattern(*rng.random((2, n))) + 0.1 * rng.standard_normal(n)
        op, g = fp.TreeTV(n, e), fp.GraphTV(n, e)
        for gamma in gammas:
            ref = ("graph", lambda: g.solve(x, gamma, threads=1)) if n <= 3000 else None
            row(f"n {n:>7}", op, x, gamma, reps, ref)
    print("paths (ms)")
    for n in (10**5, 10**6):
        x = np.cumsum(0.1 * rng.standard_normal(n)) + 0.1 * rng.standard_normal(n)
        perm = rng.permutation(n)
        xs = np.empty(n)
        xs[perm] = x
        ordered = fp.TreeTV(n, np.stack([np.arange(n - 1), np.arange(1, n)], 1))
        shuffled = fp.TreeTV(n, np.stack([perm[:-1], perm[1:]], 1))
        tv = fp.TV1D()
        for gamma in gammas:
            row(f"n {n:>7} in order", ordered, x, gamma, reps, ("TV1D", lambda: tv.prox(x, gamma, threads=1)))
            row(f"n {n:>7} shuffled", shuffled, xs, gamma, reps)


if __name__ == "__main__":
    main()
