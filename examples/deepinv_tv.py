"""fastprox's device TV in place of DeepInverse's TV denoisers (0.3).

DeepInverse's ``TVDenoiser`` and ``TVL1Denoiser`` compute the prox of
isotropic and anisotropic TV of batches (B, C, H, W) with the step ``ths``,
as ``fastprox.torch.tv_iso`` and ``tv_aniso`` do (docs/deepinv.md). This
script checks, on two noisy images with one step per sample:

1. that both give the same image when run to convergence;
2. warm starts inside an iterative algorithm (proximal gradient for
   inpainting): DeepInverse keeps its last iterates in the denoiser and
   stops on the relative change of x, fastprox takes the dual of the
   previous call and stops on the duality gap. Against the same loop with
   exact proxes, DeepInverse's defaults end a few hundredths away (the
   relative-change test stops early from a warm start), fastprox at
   tol = 1e-6 about 1e-3;
3. gradients in the image and in the per-sample step, which fastprox
   computes by implicit differentiation instead of through the iterations.

Needs PyTorch and DeepInverse (``uv run --group interop --with deepinv python
examples/deepinv_tv.py``); without them it says so and stops.
"""

import numpy as np

import fastprox as fp


def images(rng, n=32, b=2):
    a, c = np.mgrid[0:n, 0:n] / n
    clean = ((a - 0.5) ** 2 + (c - 0.45) ** 2 < 0.08) + 0.5 * (c > 0.7)
    return np.stack([clean + 0.1 * rng.normal(size=(n, n)) for _ in range(b)])[:, None]


def main():
    try:
        import torch
        from deepinv.models import TVDenoiser, TVL1Denoiser
    except ImportError:
        print("PyTorch and DeepInverse are not installed: "
              "uv run --group interop --with deepinv python examples/deepinv_tv.py")
        return None
    import fastprox.torch as fpt

    rng = np.random.default_rng(0)
    y = torch.tensor(images(rng))                      # (B, C, H, W), float64
    ths = torch.tensor([0.05, 0.1], dtype=y.dtype)     # one step per sample

    # 1. The same problems.
    diffs = {}
    for name, den, tv in (("TVDenoiser / tv_iso", TVDenoiser, fpt.tv_iso),
                          ("TVL1Denoiser / tv_aniso", TVL1Denoiser, fpt.tv_aniso)):
        zd = den(n_it_max=5000, crit=1e-10)(y, ths)
        z = tv(y, ths[:, None], tol=1e-10, max_iter=10 ** 6)   # ths broadcasts to the batch shape (B, C)
        diffs[name] = float((z - zd).abs().max())
        print(f"{name:24s} max difference {diffs[name]:.1e}")

    # 2. Warm starts in proximal gradient for inpainting, x <- prox(x - (M x - b)),
    # against the same loop with exact proxes (the C kernel, image by image).
    mask = torch.tensor(rng.uniform(size=y.shape) < 0.5)
    b = torch.where(mask, y, 0.0)
    step = lambda x: x - torch.where(mask, x - b, 0.0)
    den = TVDenoiser()                                  # defaults; keeps x2, u2 between calls
    exact = fp.TVIso()
    x_e, x_d, x_f, dual_e, dual, iters = b.clone(), b.clone(), b.clone(), [None, None], None, []
    for _ in range(30):
        v = step(x_e).numpy()
        for i in range(2):
            z, inf = exact.solve(v[i, 0], float(ths[i]), tol=1e-12, max_iter=10 ** 6, warm_start=dual_e[i])
            x_e[i, 0], dual_e[i] = torch.from_numpy(z), inf.dual
        x_d = den(step(x_d), ths)
        x_f, info = fpt.tv_iso(step(x_f), ths[:, None], tol=1e-6, warm_start=dual, return_info=True)
        dual = info.dual                                # (2, B, C, H, W): the next call's start
        iters.append(info.iters)
    err_d, err_f = float((x_d - x_e).abs().max()), float((x_f - x_e).abs().max())
    print(f"inpainting, 30 steps, distance to exact proxes: DeepInverse {err_d:.1e}, fastprox {err_f:.1e} "
          f"({iters[0]} iterations in the first prox, {iters[-1]} in the last)")

    # 3. Gradients of <prox(y), c> in y and in the per-sample step.
    c = torch.tensor(rng.normal(size=y.shape))
    grads = {}
    for name, f in (("fastprox", lambda yy, g: fpt.tv_iso(yy, g[:, None], tol=1e-12, max_iter=10 ** 6)),
                    ("DeepInverse", lambda yy, g: TVDenoiser(n_it_max=1500, crit=0.0)(yy, g))):
        yy, g = y.clone().requires_grad_(), ths.clone().requires_grad_()
        gy, gg = torch.autograd.grad((f(yy, g) * c).sum(), (yy, g))
        grads[name] = (gy, gg)
    (gy, gg), (gy_d, gg_d) = grads["fastprox"], grads["DeepInverse"]
    rel = float((gy - gy_d).norm() / gy_d.norm())
    print(f"gradients: in y, relative difference {rel:.1e}; in ths, fastprox {gg.tolist()} "
          f"against unrolled {gg_d.tolist()}")
    return diffs, (err_d, err_f), rel, (gg, gg_d)


if __name__ == "__main__":
    main()
