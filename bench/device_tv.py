"""Device-native image TV (fastprox.torch / fastprox.jax) against the C kernels.

  uv run --group interop python bench/device_tv.py [reps]

A batch of 8 images of 512^2 (the disk-and-band image of
examples/autodiff_frameworks.py plus noise 0.1, float32), gamma = 0.1, to gap
tolerances 1e-4 and 1e-6: the forward solve and one backward (the VJP of a
random cotangent; the gradient of a plain sum is trivial, as constants are
in the null space of the differences)
on every PyTorch device available (CPU tensors, MPS, CUDA), JAX on its
default device (jit, compiled before timing), and the C kernels
(TVIso/TVAniso, float32 input) on one thread and on all of them. Medians of
`reps` runs; iteration counts in parentheses.
"""

import sys
import time

import numpy as np

import fastprox as fp


def image(h, rng):
    a, b = np.mgrid[0:h, 0:h] / h
    clean = ((a - 0.5) ** 2 + (b - 0.45) ** 2 < 0.08) + 0.5 * (b > 0.7)
    return (clean + 0.1 * rng.normal(size=(h, h))).astype(np.float32)


def med(f, reps, sync=lambda: None):
    f()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        sync()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rng = np.random.default_rng(0)
    x = np.stack([image(512, rng) for _ in range(8)])
    c = rng.normal(size=x.shape).astype(np.float32)
    rows = []
    import torch
    import fastprox.torch as fpt
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
        (["cuda"] if torch.cuda.is_available() else [])
    for iso, name in ((True, "isotropic"), (False, "anisotropic")):
        print(f"{name} TV, 8 x 512^2 float32, gamma 0.1 (ms)")
        cls = fp.TVIso if iso else fp.TVAniso
        for tol in (1e-4, 1e-6):
            out = []
            for dev in devices:
                sync = (torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize if dev == "mps"
                        else (lambda: None))
                f = fpt.tv_iso if iso else fpt.tv_aniso
                xt = torch.tensor(x, device=dev)
                _, info = f(xt, 0.1, tol=tol, return_info=True)
                t = med(lambda: f(xt, 0.1, tol=tol), reps, sync)
                xg = xt.clone().requires_grad_()
                ct = torch.tensor(c, device=dev)

                def fb():
                    f(xg, 0.1, tol=tol).backward(ct)
                tb = med(fb, reps, sync) - t
                out.append(f"torch {dev} {t:7.0f} ({info.iters}) + bwd {tb:5.0f}")
            import jax
            import jax.numpy as jnp
            import fastprox.jax as fpj
            fj = fpj.tv_iso if iso else fpj.tv_aniso
            run = jax.jit(lambda x: fj(x, 0.1, tol=tol))
            cj = jnp.asarray(c)
            grad = jax.jit(jax.grad(lambda x: jnp.vdot(fj(x, 0.1, tol=tol), cj)))
            xj = jnp.asarray(x)
            t = med(lambda: run(xj).block_until_ready(), reps)
            tg = med(lambda: grad(xj).block_until_ready(), reps) - t
            out.append(f"jax {jax.default_backend()} {t:7.0f} + bwd {tg:5.0f}")
            _, ic = cls(2).solve(x, 0.1, tol=tol)
            t1 = med(lambda: cls(2).solve(x, 0.1, tol=tol, threads=1), reps)
            ta = med(lambda: cls(2).solve(x, 0.1, tol=tol), reps)
            out.append(f"C 1 thread {t1:7.0f} ({ic.iters}) | C {fp.default_threads()} threads {ta:6.0f}")
            print(f"  tol {tol:g}: " + " | ".join(out))


if __name__ == "__main__":
    main()
