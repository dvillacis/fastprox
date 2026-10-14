"""Benchmarks for the 0.1 targets in docs/spec.md (Section 10).

    uv run python bench/bench_python.py

Measures per-call overhead from Python (target: < 5 us at n = 100), 1D TV
throughput for both paths, and parallel scaling over rows (target: >= 0.7 x
threads on batches of >= 4 x threads rows of length >= 1e4). Comparisons
against proxTV and Condat's C code live in separate scripts that fetch those
codes; nothing copyleft is vendored here.
"""

import os
import time

import numpy as np

import fastprox as fp


def best(f, repeat=7, number=None):
    if number is None:
        number = 1
        while True:
            t = time.perf_counter()
            for _ in range(number):
                f()
            if time.perf_counter() - t > 0.05:
                break
            number *= 4
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(number):
            f()
        times.append((time.perf_counter() - t) / number)
    return min(times)


def overhead():
    print("Per-call time from Python, n = 100 (target < 5 us)")
    x = np.random.default_rng(0).normal(size=100)
    out = np.empty_like(x)
    op = fp.TV1D()
    calls = {
        "fp.soft_threshold(x)": lambda: fp.soft_threshold(x, 0.1),
        "fp.project_simplex(x)": lambda: fp.project_simplex(x),
        "fp.tv1d(x)": lambda: fp.tv1d(x, 0.1),
        "TV1D().prox(x, out=out)": lambda: op.prox(x, 0.1, out=out),
    }
    for name, f in calls.items():
        print(f"  {name:28s} {best(f) * 1e6:7.2f} us")


def tv1d_throughput():
    print("1D TV, one thread, ns per sample")
    rng = np.random.default_rng(1)
    for n in (10**3, 10**4, 10**5, 10**6, 10**7):
        x = np.cumsum(rng.normal(size=n)) + rng.normal(size=n)
        row = [f"  n = {n:>8d}"]
        for m in ("condat", "taut_string"):
            t = best(lambda: fp.tv1d(x, 1.0, method=m, threads=1), repeat=3)
            row.append(f"{m} {t / n * 1e9:6.2f}")
        print("  ".join(row))
    n = 10**6
    x = (-1.0) ** np.arange(n) * (1 + 1e-3 * np.arange(n))
    for m in ("auto", "condat", "taut_string"):
        t = best(lambda: fp.tv1d(x, 1e-3, method=m, threads=1), repeat=3)
        print(f"  alternating n = 1e6, {m:12s} {t / n * 1e9:8.2f} ns/sample")


def simplex_throughput():
    print("Simplex projection, one thread, ns per sample")
    rng = np.random.default_rng(2)
    for n in (10**3, 10**5, 10**7):
        x = rng.normal(size=n)
        t = best(lambda: fp.project_simplex(x, threads=1), repeat=3)
        print(f"  n = {n:>8d}  {t / n * 1e9:6.2f}")


def scaling():
    print("Parallel scaling, tv1d on 64 rows of 1e4 (target >= 0.7 x threads)")
    x = np.random.default_rng(3).normal(size=(64, 10**4))
    out = np.empty_like(x)
    t1 = best(lambda: fp.tv1d(x, 1.0, out=out, threads=1))
    ncpu = os.cpu_count() or 1
    counts = sorted({t for t in (1, 2, 4, 8, 16, 32, 64) if t <= ncpu} | {ncpu})
    print(f"  {ncpu} logical CPUs")
    for t in counts:
        tt = best(lambda: fp.tv1d(x, 1.0, out=out, threads=t))
        print(f"  threads {t:2d}: {tt * 1e3:7.3f} ms  speed-up {t1 / tt:5.2f}  "
              f"efficiency {t1 / tt / t:4.2f}")


if __name__ == "__main__":
    print(f"fastprox {fp.__version__}, default threads {fp.default_threads()}")
    overhead()
    tv1d_throughput()
    simplex_throughput()
    scaling()
