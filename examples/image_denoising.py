"""Denoise images with isotropic and anisotropic total variation (since 0.2).

    z = argmin_z  gamma * TV(z) + ||z - y||^2 / 2

Anisotropic TV sums |differences| along each axis; isotropic TV sums the
length of the gradient at each pixel, so it does not favour edges aligned
with the axes. Both are iterative: each call returns a duality gap that
certifies the result, ||z - z*||^2 <= 2 * gap, and a dual that warm-starts
the next call. The last part learns gamma from clean/noisy pairs with the
VJP of the prox, which solves for the derivative of the solution instead of
differentiating through the iterations. Plots the results if matplotlib is
installed:

    uv run --group examples python examples/image_denoising.py
"""

import time

import numpy as np

import fastprox as fp

SIGMA = 0.3     # noise level


def make_image(rng, n=128):
    """A tilted square and a disk with random positions and levels."""
    i, j = np.mgrid[0:n, 0:n] / n
    img = np.zeros((n, n))
    ci, cj = rng.uniform(0.3, 0.7, size=2)
    u, v = (i - ci + j - cj) / np.sqrt(2), (i - ci - j + cj) / np.sqrt(2)
    img[(np.abs(u) < 0.18) & (np.abs(v) < 0.18)] = rng.uniform(0.5, 1.0)
    di, dj = rng.uniform(0.2, 0.8, size=2)
    img[(i - di) ** 2 + (j - dj) ** 2 < 0.12 ** 2] = rng.uniform(-1.0, -0.5)
    return img


def psnr(z, x):
    return 10 * np.log10(np.ptp(x) ** 2 / np.mean((z - x) ** 2))


def loss(z, x):
    return 0.5 * np.mean((z - x) ** 2)


def main(plot=True):
    rng = np.random.default_rng(0)
    x = make_image(rng)
    y = x + SIGMA * rng.normal(size=x.shape)
    gamma = 0.3
    print(f"noisy:        PSNR {psnr(y, x):5.2f} dB")

    # 1. One image, both models. The gap bounds the distance to the exact
    #    solution; divided by the pixel count it bounds the RMS error.
    results = {}
    for name, solve in (("isotropic", fp.tv_iso), ("anisotropic", fp.tv_aniso)):
        t = time.perf_counter()
        z, info = solve(y, gamma=gamma, tol=1e-5)
        t = time.perf_counter() - t
        results[name] = z
        print(f"{name:12s}  PSNR {psnr(z, x):5.2f} dB   {info.iters:4d} iterations, "
              f"{1e3 * t:4.0f} ms   RMS error <= {np.sqrt(2 * info.gap / z.size):.0e}, "
              f"converged: {info.converged}")

    # 2. Nearby problems, e.g. a fine sweep over gamma: starting each call from
    #    the previous dual saves iterations (the dual is projected onto the new
    #    constraint |u_p| <= gamma, so any dual is a valid start).
    gammas = np.linspace(0.25, 0.35, 8)
    cold = sum(fp.tv_iso(y, gamma=g, tol=1e-5)[1].iters for g in gammas)
    dual, warm = None, 0
    for g in gammas:
        _, info = fp.tv_iso(y, gamma=g, tol=1e-5, warm_start=dual)
        dual, warm = info.dual, warm + info.iters
    print(f"sweep of {len(gammas)} gammas: {cold} iterations cold, {warm} warm-started")

    # 3. A batch: leading axes are independent images, solved in parallel.
    batch = np.stack([make_image(rng) for _ in range(16)])
    noisy = batch + SIGMA * rng.normal(size=batch.shape)
    t = time.perf_counter()
    zb, info = fp.tv_iso(noisy, gamma=gamma, tol=1e-5)
    t = time.perf_counter() - t
    print(f"batch of {len(batch)} images in {1e3 * t:.0f} ms: "
          f"PSNR {psnr(noisy, batch):5.2f} -> {psnr(zb, batch):5.2f} dB, "
          f"converged: {info.converged}")

    # 4. Learn gamma from training pairs (bilevel): minimise L(gamma) =
    #    mean (z(gamma) - x)^2 / 2 by gradient descent on log(gamma). One VJP
    #    gives dL/dgamma; warm=True starts each prox from the previous dual.
    train = np.stack([make_image(rng, 64) for _ in range(4)])
    ytrain = train + SIGMA * rng.normal(size=train.shape)
    op = fp.TVIso(ndim=2)
    s, ctx = np.log(0.05), None
    for _ in range(10):
        z, ctx = op.prox(ytrain, np.exp(s), keep=True, ctx=ctx, tol=1e-6, max_iter=10 ** 5, warm=True)
        _, grads = op.vjp(ctx, g=(z - train) / z.size)
        s -= 100.0 * grads.gamma * np.exp(s)        # dL/dlog(gamma) = gamma dL/dgamma
    learned = float(np.exp(s))
    residual, cg_iters = ctx.deriv
    grid = learned * np.geomspace(0.5, 2.0, 15)
    losses = [loss(fp.tv_iso(ytrain, gamma=g, tol=1e-6, max_iter=10 ** 5)[0], train) for g in grid]
    best = grid[np.argmin(losses)]
    print(f"learned gamma {learned:.3f} (grid search: {best:.3f}); last VJP: "
          f"{cg_iters} CG iterations, relative residual {residual:.0e}")

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(install matplotlib, e.g. `uv run --group examples ...`, to see a plot)")
            return
        panels = [("clean", x), ("noisy", y), *((f"{k} TV", v) for k, v in results.items())]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
        for ax, (title, im) in zip(axes, panels):
            ax.imshow(im, cmap="gray", vmin=-1.2, vmax=1.2)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        plt.show()
    return learned, best


if __name__ == "__main__":
    main()
