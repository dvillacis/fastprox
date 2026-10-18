"""Device-native image TV against DeepInverse's TV denoisers.

  uv run --group interop --with deepinv python bench/compare_deepinv.py [size] [batch] [iso|aniso]

The float64 reference is cached in deepinv_ref_<mode>_<size>_<batch>.npz in
the working directory (see main), so a machine with few cores can be handed
one computed elsewhere.

DeepInverse's ``TVDenoiser`` solves the same problem as ``TVIso`` (checked
here to 1e-12), and ``TVL1Denoiser`` (the same method with the dual clamped
per component) that of ``TVAniso``: an over-relaxed Chambolle-Pock method
(Condat's form) that stops on the relative change of x, differentiated by
autograd through its iterations. fastprox runs FISTA on the dual with a
certified gap and an implicit backward (``fastprox.torch.tv_iso`` /
``tv_aniso``), or the C kernel behind a host round trip
(``fastprox.torch.prox(fp.TVIso(), x)``; the anisotropic C kernel adds exact
1D TV steps and a polish, so it needs far fewer iterations).

Images: a batch of disk-and-band images plus noise 0.1 (float32, shape
(batch, 1, size, size)), gamma = 0.1. The reference is the C kernel in
float64 to a gap of 1e-13; errors are |z - z*| / |z*| over the batch, and
gradient errors compare the gradient of <z, c> for a random c with the C
kernel's exact VJP at the reference.

Reported, on every PyTorch device available:
  1. forward: time and error for DeepInverse at fixed iteration counts
     (crit = 0; and its defaults, crit = 1e-5 with at most 1000 iterations),
     for fastprox's device path at gap tolerances, and for the C path;
  2. time to reach errors of 1e-2, 1e-3 and 1e-4 by each;
  3. the host round trip of the C path (to the CPU and back);
  4. backward: time, memory kept for the backward (MPS: allocated memory
     after the forward minus before), and gradient error. The unrolled
     backward stops at 400 iterations: its memory grows with them.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

import fastprox as fp
import fastprox.torch as fpt

GAMMA = 0.1


def images(size, batch, rng):
    a, b = np.mgrid[0:size, 0:size] / size
    clean = ((a - 0.5) ** 2 + (b - 0.45) ** 2 < 0.08) + 0.5 * (b > 0.7)
    return np.stack([clean + 0.1 * rng.normal(size=(size, size)) for _ in range(batch)])[:, None]


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def timed(f, dev, reps=3):
    f()
    sync(dev)
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), out


MODES = {  # (fastprox operator, device function, DeepInverse class)
    "iso": (fp.TVIso, fpt.tv_iso, "TVDenoiser"),
    "aniso": (fp.TVAniso, fpt.tv_aniso, "TVL1Denoiser"),
}


def deepinv_tv(y, n, crit=0.0, cls="TVDenoiser"):
    import deepinv.models
    return getattr(deepinv.models, cls)(n_it_max=n, crit=crit)(y, GAMMA)  # a new denoiser: no warm restart


def rel(z, ref):
    z = z.detach().cpu().double().numpy()
    return float(np.linalg.norm(z - ref) / np.linalg.norm(ref))


def mem_after(dev, f):
    """Bytes allocated by f() that stay alive with its result (MPS, CUDA)."""
    if dev == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
        m0 = torch.mps.current_allocated_memory()
        out = f()
        torch.mps.synchronize()
        return out, torch.mps.current_allocated_memory() - m0
    if dev == "cuda":
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        out = f()
        torch.cuda.synchronize()
        return out, torch.cuda.memory_allocated() - m0
    return f(), None


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mode = sys.argv[3] if len(sys.argv) > 3 else "iso"
    Op, dev_tv, cls = MODES[mode]
    dinv = lambda y, n, crit=0.0: deepinv_tv(y, n, crit, cls)
    rng = np.random.default_rng(0)
    x64 = images(size, batch, rng)
    c64 = rng.normal(size=x64.shape)
    # The float64 reference depends only on (mode, size, batch): it is read from
    # deepinv_ref_<mode>_<size>_<batch>.npz in the working directory when that
    # file exists (computing it takes minutes on two cores, e.g. on Colab), and
    # written there otherwise.
    cache = Path(f"deepinv_ref_{mode}_{size}_{batch}.npz")
    if cache.exists():
        # The images and cotangent are taken from the file, so that they match
        # the reference exactly: regenerated elsewhere they can differ in the
        # last bit (NumPy's normal sampler calls the platform's exp and log).
        with np.load(cache) as f:
            ref, gref = f["ref"], f["gref"]
            assert np.allclose(f["x"], x64, rtol=0, atol=1e-12) and np.allclose(f["c"], c64, rtol=0, atol=1e-12), \
                f"{cache} is for other data"
            x64, c64 = f["x"], f["c"]
        print(f"reference read from {cache}", flush=True)
    else:
        op = Op(2)
        ref, ctx = op.prox(x64, GAMMA, keep=True, tol=1e-13, max_iter=10 ** 6)
        if mode == "iso":
            gref, _ = op.vjp(ctx, c64, tol=1e-10, max_iter=10 ** 5)  # 18 548 iterations here; the default cap is 10 000
        else:
            gref, _ = op.vjp(ctx, c64)  # exact region averages
        np.savez_compressed(cache, ref=ref, gref=gref, x=x64, c=c64)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
        (["cuda"] if torch.cuda.is_available() else [])
    name = "isotropic" if mode == "iso" else "anisotropic"
    print(f"{name} TV, {batch} x {size}^2 float32, gamma {GAMMA}; reference: C kernel, float64, gap 1e-13",
          flush=True)
    for dev in devices:
        x = torch.tensor(x64, dtype=torch.float32, device=dev)
        c = torch.tensor(c64, dtype=torch.float32, device=dev)
        print(f"\n== {dev} ==")
        rows = []  # (method, setting, ms, error)
        for n in (25, 50, 100, 200, 400, 800, 1600):
            t, z = timed(lambda: dinv(x, n), dev, reps=1 if n >= 800 else 3)
            rows.append(("deepinv", f"{n} it", t, rel(z, ref)))
        t, z = timed(lambda: dinv(x, 1000, 1e-5), dev, reps=1)
        rows.append(("deepinv", "defaults", t, rel(z, ref)))
        for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):  # 1e-8 is below float32 rounding: never met
            t, (z, info) = timed(lambda: dev_tv(x, GAMMA, tol=tol, max_iter=100000, return_info=True), dev)
            rows.append(("fastprox", f"tol {tol:g} ({info.iters} it)", t, rel(z, ref)))
        if dev != "cpu":
            for tol in (1e-4, 1e-6, 1e-8):
                t, z = timed(lambda: fpt.prox(Op(), x, GAMMA, tol=tol, max_iter=100000), dev)
                rows.append(("C via host", f"tol {tol:g}", t, rel(z, ref)))
        for m, s, t, e in rows:
            print(f"  {m:10s} {s:24s} {t:9.1f} ms   error {e:.1e}", flush=True)
        print("  time to reach an error of:")
        for target in (1e-2, 1e-3, 1e-4):
            best = {}
            for m, s, t, e in rows:
                if e <= target and (m not in best or t < best[m][0]):
                    best[m] = (t, s)
            print(f"    {target:g}: " + " | ".join(f"{m} {t:.0f} ms ({s})" for m, (t, s) in best.items()))
        if dev != "cpu":
            t, _ = timed(lambda: x.cpu().to(dev), dev)
            print(f"  host round trip of the batch: {t:.1f} ms")
        # Backward: gradient of <z, c>. Unrolled autograd keeps every
        # iteration's tensors, so its memory grows with the iterations.
        print("  backward (gradient of <z, c>):", flush=True)
        runs = [(f"deepinv {n} it", (lambda n=n: lambda xx: dinv(xx, n))()) for n in (50, 100, 200, 400)]
        runs += [(f"fastprox tol {tol:g}", (lambda tol=tol: lambda xx: dev_tv(xx, GAMMA, tol=tol, max_iter=100000))())
                 for tol in (1e-4, 1e-6, 1e-7)]
        for name, f in runs:
            xg = x.clone().requires_grad_()
            z, mem = mem_after(dev, lambda: f(xg))
            sync(dev)
            t0 = time.perf_counter()
            (gx,) = torch.autograd.grad(z, xg, c)
            sync(dev)
            tb = 1e3 * (time.perf_counter() - t0)
            ge = float(np.linalg.norm(gx.cpu().double().numpy() - gref) / np.linalg.norm(gref))
            ms = f"{mem / 2 ** 20:7.0f} MiB kept" if mem is not None else "  (memory: GPU only)"
            print(f"    {name:22s} backward {tb:8.1f} ms {ms}   gradient error {ge:.1e}", flush=True)
            del z, gx, xg


if __name__ == "__main__":
    main()
