"""Benchmarks for the 0.1 targets in docs/spec.md (Section 10).

    uv run python bench/bench_python.py

Measures per-call overhead from Python (target: < 5 us at n = 100), 1D TV
throughput for both paths, and parallel scaling over rows (target: >= 0.7 x
threads on batches of >= 4 x threads rows of length >= 1e4).
``uv run --group interop python bench/bench_python.py`` adds the PyTorch and
JAX wrapper sections. Comparisons
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
    soft = fp.Soft(0.4)
    _, sc = soft.prox(x, 0.8, keep=True)
    calls = {
        "fp.soft_threshold(x)": lambda: fp.soft_threshold(x, 0.1),
        "fp.project_simplex(x)": lambda: fp.project_simplex(x),
        "fp.tv1d(x)": lambda: fp.tv1d(x, 0.1),
        "TV1D().prox(x, out=out)": lambda: op.prox(x, 0.1, out=out),
        "Soft().prox(x, out=out)": lambda: soft.prox(x, 0.8, out=out),
        "Soft vjp(ctx, x)": lambda: soft.vjp(sc, x),
        "Soft jvp(ctx, x)": lambda: soft.jvp(sc, x),
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


def torch_wrappers():
    try:
        import torch

        import fastprox.torch as fpt
    except ImportError:
        print("PyTorch wrappers: torch not installed, skipped")
        return
    torch.set_num_threads(1)
    print("PyTorch wrappers (us)")
    rng = np.random.default_rng(0)
    soft = fp.Soft(0.4)
    xt = torch.from_numpy(rng.normal(size=100))
    xr = xt.clone().requires_grad_(True)
    lines = {
        "fpt.prox(Soft) no grad": lambda: fpt.prox(soft, xt, 0.8),
        "fpt.prox(Soft) fwd+bwd": lambda: fpt.prox(soft, xr, 0.8).sum().backward(),
    }
    X = torch.from_numpy(rng.normal(size=(64, 10000)))
    Xn = X.numpy()
    for name, op in (("L1Ball", fp.L1Ball(1.0)), ("Simplex", fp.Simplex(1.0)), ("Huber", fp.Huber(0.4, 0.5)),
                     ("NonNeg", fp.NonNeg())):
        def nograd(op=op):
            with torch.no_grad():
                fpt.prox(op, X, 0.8, threads=1)

        lines[f"fpt.prox({name}) no_grad 64x1e4"] = nograd
        lines[f"{name}().prox 64x1e4"] = lambda op=op: op.prox(Xn, 0.8, threads=1)
    for name, f in lines.items():
        print(f"  {name:34s} {best(f) * 1e6:9.1f}")


def jax_wrappers():
    try:
        import jax
        import jax.numpy as jnp

        import fastprox.jax as fpj
    except ImportError:
        print("JAX wrappers: jax not installed, skipped")
        return
    jax.config.update("jax_enable_x64", True)
    print("JAX wrappers (us)")
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=100))
    x2 = jnp.asarray(rng.normal(size=(32, 32)))
    X = jnp.asarray(rng.normal(size=(64, 10000)))
    soft, simplex = fp.Soft(0.4), fp.Simplex(1.0)
    f = jax.jit(lambda x: fpj.prox(soft, x, 0.8))
    gf = jax.jit(jax.grad(lambda x: fpj.prox(soft, x, 0.8).sum()))
    fs = jax.jit(lambda x: fpj.prox(simplex, x, 1.0, threads=1))
    gs = jax.jit(jax.grad(lambda x: (fpj.prox(simplex, x, 1.0, threads=1) * x).sum()))
    for g, a in ((f, x), (gf, x), (fs, X), (gs, X)):
        g(a).block_until_ready()  # compiled before timing
    lines = {
        "fpj.prox(Soft) eager": lambda: fpj.prox(soft, x, 0.8).block_until_ready(),
        "jax.grad(fpj.prox(Soft)) eager":
            lambda: jax.grad(lambda x: fpj.prox(soft, x, 0.8).sum())(x).block_until_ready(),
        "fpj.prox(Soft) jit": lambda: f(x).block_until_ready(),
        "jit(grad(fpj.prox(Soft)))": lambda: gf(x).block_until_ready(),
        "fpj.tv_iso 32x32 eager": lambda: fpj.tv_iso(x2, 0.3).block_until_ready(),
        "fpj.prox(Simplex) jit 64x1e4": lambda: fs(X).block_until_ready(),
        "jit(grad(fpj.prox(Simplex))) 64x1e4": lambda: gs(X).block_until_ready(),
    }
    for name, g in lines.items():
        print(f"  {name:38s} {best(g) * 1e6:9.1f}")
def _test_image(n, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    return ((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.1) + 0.5 * (xx > 0.7) + 0.1 * rng.standard_normal((n, n))


def _grid_edges(h, w):
    idx = np.arange(h * w).reshape(h, w)
    return np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                           np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])


def edge_tv_batches():
    print("Graph and tree TV, small batches (ms)")
    n = 512
    op = fp.GraphTV(n * n, _grid_edges(n, n))
    x = _test_image(n, 4).ravel() + 0.01 * np.random.default_rng(4).standard_normal((4, n * n))
    for gamma in (0.1, 1.0):
        row = [f"  GraphTV 512^2 grid, batch 4, gamma {gamma:g}:"]
        for threads in (1, 0):
            row.append(f"threads {threads} {best(lambda: op.prox(x, gamma, threads=threads), repeat=3, number=1) * 1e3:8.1f}")
        print("  ".join(row))
    m = 200_000
    rng = np.random.default_rng(6)
    parent = np.floor(np.arange(1, m) * rng.random(m - 1)).astype(np.int64)
    tree = fp.TreeTV(m, np.stack([np.arange(1, m), parent], 1))
    for b in (4, 16):
        y = 0.1 * np.cumsum(rng.standard_normal((b, m)), axis=1) + rng.standard_normal((b, m))
        row = [f"  TreeTV n = 2e5, batch {b:2d}, gamma 1:"]
        for threads in (1, 0):
            row.append(f"threads {threads} {best(lambda: tree.prox(y, 1.0, threads=threads), repeat=3, number=1) * 1e3:8.1f}")
        print("  ".join(row))
def image_tv_derivatives():
    print("Isotropic TV derivatives, one image (ms; CG iterations)")
    op = fp.TVIso(2)
    for n, seed, gamma, tol in ((512, 0, 0.1, 1e-6), (512, 0, 1.0, 1e-6), (2048, 11, 0.1, 1e-4)):
        x = _test_image(n, seed)
        dx = np.random.default_rng(1).standard_normal(x.shape)
        for threads in (1, 0):
            tp = best(lambda: op.prox(x, gamma, keep=True, tol=tol, max_iter=100000, threads=threads),
                      repeat=3, number=1)
            _, ctx = op.prox(x, gamma, keep=True, tol=tol, max_iter=100000, threads=threads)
            tj = best(lambda: op.jvp(ctx, dx), repeat=3, number=1)
            it = ctx.deriv.iters
            tv = best(lambda: op.vjp(ctx, dx), repeat=3, number=1)
            print(f"  D({n}, {seed}) gamma {gamma:g} tol {tol:g} threads {threads}:"
                  f" prox {tp * 1e3:7.1f}  jvp {tj * 1e3:7.1f}  vjp {tv * 1e3:7.1f}  ({it})")
def kernel_throughput():
    print("Row and elementwise kernels, one thread, ns per entry")
    rng = np.random.default_rng(4)
    n = 10**6
    x = rng.normal(size=n)
    w = rng.uniform(0, 2, n)
    w[::10] = 0.0
    setups = {"soft weighted": lambda: fp.Soft(0.3, w).prox(x, threads=1)}
    for q in (0.9, 0.999):
        beta = (2 * (1 - q)) ** (1 / (2 - q))
        tau = beta + q * beta ** (q - 1)
        op = fp.Lq(1.0, q)
        far = tau * rng.uniform(3, 10, n)
        near = tau * (1 + rng.uniform(1e-9, 1e-3, n))
        setups[f"lq q={q} far"] = lambda op=op, v=far: op.prox(v, threads=1)
        setups[f"lq q={q} near"] = lambda op=op, v=near: op.prox(v, threads=1)
    ball = fp.L2Ball(1.0)
    _, ctx = ball.prox(x, keep=True, threads=1)
    dx, g = rng.normal(size=n), rng.normal(size=n)
    setups["l2_ball prox"] = lambda: ball.prox(x, threads=1)
    setups["l2_ball jvp"] = lambda: ball.jvp(ctx, dx)
    setups["l2_ball vjp"] = lambda: ball.vjp(ctx, g)
    groups = np.arange(n) // 8
    setups["prox_l2 groups of 8"] = lambda: fp.group_soft_threshold(x, 0.5, groups=groups, threads=1)
    soc = fp.SOC()
    setups["soc"] = lambda: soc.prox(x, threads=1)
    a = rng.uniform(0.5, 1.5, n)
    setups["hyperplane box"] = lambda: fp.project_hyperplane(x, a, 0.3 * n, 0.0, 1.0, threads=1)
    setups["sum_largest k=10"] = lambda: fp.prox_sum_largest(x, 1.0, 10, threads=1)
    walk = np.cumsum(rng.normal(size=n))
    setups["isotonic random walk"] = lambda: fp.isotonic(walk, threads=1)
    for name, f in setups.items():
        print(f"  {name:22s} {best(f, repeat=3) / n * 1e9:8.2f}")


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
    kernel_throughput()
    scaling()
    torch_wrappers()
    jax_wrappers()
    edge_tv_batches()
    image_tv_derivatives()
