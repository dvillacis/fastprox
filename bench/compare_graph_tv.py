"""Graph TV (parametric max-flow) on three kinds of graph, single thread.

  uv run python bench/compare_graph_tv.py [reps]

* 4-neighbour grids, where graph TV is anisotropic TV: against TVAniso at
  gap tolerances 1e-6 and 1e-10, with the largest error against the exact
  graph TV solution;
* 8-neighbour grids (diagonals weighted 1/sqrt(2)), graph TV only;
* k-nearest-neighbour graphs of random points in the unit square (k = 8,
  each pair once, weights exp(-d^2 / h^2)), graph TV only;
* graphs with hubs (since 0.3): stars, against tree TV, and a 4-neighbour
  grid plus five hubs each joined to a tenth of the pixels.

The images are the synthetic ones of bench/bench_maxflow.cpp (a ramp, disks
and a square, Gaussian noise 0.1); the point signals are the same pattern
sampled at the points. Graph TV is timed exact (tol = 0) and at tol = 1e-4,
with the number of max-flow solves. Medians of `reps` runs.
"""

import sys
import time

import numpy as np

import fastprox as fp


def pattern(a, b):
    v = 0.3 * b
    v = np.where((a - 0.35) ** 2 + (b - 0.3) ** 2 < 0.04, 0.8, v)
    v = np.where((np.abs(a - 0.7) < 0.15) & (np.abs(b - 0.65) < 0.2), 0.15, v)
    return np.where((a - 0.6) ** 2 + (b - 0.25) ** 2 < 0.01, 0.95, v)


def image(h, rng):
    a, b = np.mgrid[0:h, 0:h] / h
    return pattern(a, b) + 0.1 * rng.standard_normal((h, h))


def grid(h, diagonals):
    idx = np.arange(h * h).reshape(h, h)
    e = [np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
         np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)]
    w = [np.ones(h * (h - 1)), np.ones(h * (h - 1))]
    if diagonals:
        e += [np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], 1),
              np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], 1)]
        w += [np.full((h - 1) ** 2, np.sqrt(0.5))] * 2
    return np.concatenate(e), np.concatenate(w)


def knn_graph(n, k, rng):
    """k nearest neighbours by a uniform cell grid (no SciPy needed)."""
    p = rng.random((n, 2))
    c = max(1, int(np.sqrt(n / 4)))
    cell = np.minimum((p * c).astype(int), c - 1)
    key = cell[:, 0] * c + cell[:, 1]
    order = np.argsort(key, kind="stable")
    start = np.searchsorted(key[order], np.arange(c * c + 1))
    pairs = []
    for i in range(n):
        ci, cj = cell[i]
        cand = np.concatenate([order[start[a * c + b]:start[a * c + b + 1]]
                               for a in range(max(ci - 1, 0), min(ci + 2, c))
                               for b in range(max(cj - 1, 0), min(cj + 2, c))])
        cand = cand[cand != i]
        d = ((p[cand] - p[i]) ** 2).sum(1)
        for j in cand[np.argsort(d)[:k]]:
            pairs.append((min(i, j), max(i, j)))
    e = np.unique(np.array(pairs), axis=0)
    d2 = ((p[e[:, 0]] - p[e[:, 1]]) ** 2).sum(1)
    w = np.exp(-d2 / np.median(d2))
    x = pattern(p[:, 0], p[:, 1]) + 0.1 * rng.standard_normal(n)
    return e, w, x


def timed(f, reps):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rng = np.random.default_rng(7)
    print("4-neighbour grids: graph TV and TVAniso (ms; error: max |z - z_exact|)")
    for h in (256, 512):
        x = image(h, rng)
        e, w = grid(h, False)
        op = fp.GraphTV(h * h, e)
        for gamma in (0.02, 0.1, 0.5):
            t0, (z0, i0) = timed(lambda: op.solve(x.ravel(), gamma, threads=1), reps)
            t4, (z4, i4) = timed(lambda: op.solve(x.ravel(), gamma, tol=1e-4, threads=1), reps)
            row = [f"{h}^2 gamma {gamma:<4}: exact {t0:8.1f} ({i0.iters} solves)",
                   f"tol 1e-4 {t4:8.1f} ({i4.iters}, err {np.abs(z4 - z0).max():.0e})"]
            for tol in (1e-6, 1e-10):
                ta, (za, _) = timed(lambda: fp.tv_aniso(x, gamma, tol=tol, max_iter=10**6, threads=1), reps)
                row.append(f"TVAniso {tol:g} {ta:7.1f} (err {np.abs(za.ravel() - z0).max():.0e})")
            print("  " + " | ".join(row))
    print("8-neighbour grids: graph TV (ms)")
    for h in (256, 512):
        x = image(h, rng)
        e, w = grid(h, True)
        op = fp.GraphTV(h * h, e, w)
        for gamma in (0.02, 0.1, 0.5):
            t0, (_, i0) = timed(lambda: op.solve(x.ravel(), gamma, threads=1), reps)
            t4, (_, i4) = timed(lambda: op.solve(x.ravel(), gamma, tol=1e-4, threads=1), reps)
            print(f"  {h}^2 gamma {gamma:<4}: exact {t0:8.1f} ({i0.iters} solves) | tol 1e-4 {t4:8.1f} ({i4.iters})")
    print("k-nearest-neighbour graphs, k = 8: graph TV (ms)")
    for n in (10_000, 100_000):
        e, w, x = knn_graph(n, 8, rng)
        op = fp.GraphTV(n, e, w)
        for gamma in (0.02, 0.1, 0.5):
            t0, (_, i0) = timed(lambda: op.solve(x, gamma, threads=1), reps)
            t4, (_, i4) = timed(lambda: op.solve(x, gamma, tol=1e-4, threads=1), reps)
            print(f"  n {n} ({len(e)} edges) gamma {gamma:<4}: exact {t0:8.1f} ({i0.iters} solves)"
                  f" | tol 1e-4 {t4:8.1f} ({i4.iters})")
    print("graphs with hubs: graph TV (ms)")
    for n in (10_000, 100_000, 1_000_000):
        e = np.stack([np.zeros(n - 1, int), np.arange(1, n)], 1)
        w = rng.uniform(0.5, 1.5, n - 1)
        x = rng.standard_normal(n)
        op, tree = fp.GraphTV(n, e, w), fp.TreeTV(n, e, w)
        t0, (z0, i0) = timed(lambda: op.solve(x, 0.3, threads=1), reps)
        tt, zt = timed(lambda: tree.prox(x, 0.3), reps)
        print(f"  star, {n - 1} leaves, gamma 0.3: exact {t0:8.1f} ({i0.iters} solves) | tree TV {tt:6.1f}"
              f" (max diff {np.abs(z0 - zt).max():.0e})")
    h = 256
    eg, wg = grid(h, False)
    hubs = rng.choice(h * h, 5, replace=False)
    eh = [np.stack([np.full(h * h // 10, c), rng.choice(np.delete(np.arange(h * h), c), h * h // 10, replace=False)], 1)
          for c in hubs]
    e = np.concatenate([eg, *eh])
    w = np.concatenate([wg, np.full(len(e) - len(eg), 0.05)])
    x = image(h, rng).ravel()
    op = fp.GraphTV(h * h, e, w)
    for gamma in (0.02, 0.1, 0.5):
        t0, (_, i0) = timed(lambda: op.solve(x, gamma, threads=1), reps)
        print(f"  {h}^2 grid + 5 hubs of {h * h // 10} edges, gamma {gamma:<4}: exact {t0:8.1f} ({i0.iters} solves)")


if __name__ == "__main__":
    main()
