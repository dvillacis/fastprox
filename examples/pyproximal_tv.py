"""fastprox operators inside PyProximal's solvers (since 0.5).

``fastprox.pyproximal.ProxOp`` makes any operator a ``pyproximal.ProxOperator``.
Here isotropic TV deblurs an image with PyProximal's accelerated proximal
gradient:

    min_x 1/2 ||H x - b||^2 + lam TV(x),

H a Gaussian blur (PyLops), b the blurred image plus noise. The TV prox is
fastprox's, stopped on its duality gap and warm-started from one iteration
to the next; the same solve with PyProximal's own ``TV`` (a fixed number of
inner iterations) is shown for comparison.

Needs PyProximal (``uv run --group interop python examples/pyproximal_tv.py``);
without it the script says so and stops.
"""

import time

import numpy as np

import fastprox as fp

H, W, LAM, ITERS = 64, 64, 0.02, 100


def image():
    a, b = np.mgrid[0:H, 0:W] / H
    x = np.where((a - 0.4) ** 2 + (b - 0.5) ** 2 < 0.06, 1.0, 0.2)
    return np.where((np.abs(a - 0.75) < 0.12) & (np.abs(b - 0.3) < 0.2), 0.6, x)


def psnr(x, ref):
    return 10 * np.log10(np.ptp(ref) ** 2 / np.mean((x - ref) ** 2))


def main():
    import pylops
    import pyproximal

    from fastprox.pyproximal import ProxOp

    rng = np.random.default_rng(0)
    clean = image()
    t = np.arange(-4, 5)
    k = np.exp(-(t[:, None] ** 2 + t[None, :] ** 2) / (2 * 1.5 ** 2))
    blur = pylops.signalprocessing.Convolve2D((H, W), h=k / k.sum(), offset=(4, 4))
    b = blur @ clean.ravel() + 0.01 * rng.normal(size=H * W)
    f = pyproximal.L2(Op=blur, b=b)
    tau = 0.95  # the blur has norm at most 1, so 1/L >= 1

    def solve(g):
        t0 = time.perf_counter()
        x = pyproximal.optimization.primal.ProximalGradient(
            f, g, np.zeros(H * W), tau=tau, epsg=LAM, niter=ITERS, acceleration="fista")
        return x.reshape(H, W), time.perf_counter() - t0

    x_fp, t_fp = solve(ProxOp(fp.TVIso(), dims=(H, W), tol=1e-6, warm=True))
    x_pp, t_pp = solve(pyproximal.TV(dims=(H, W), sigma=1.0, niter=20))
    blurred = b.reshape(H, W)
    print(f"blurred and noisy        PSNR {psnr(blurred, clean):5.1f} dB")
    print(f"TV, fastprox ProxOp      PSNR {psnr(x_fp, clean):5.1f} dB  {t_fp:.2f} s")
    print(f"TV, pyproximal.TV        PSNR {psnr(x_pp, clean):5.1f} dB  {t_pp:.2f} s")
    return psnr(blurred, clean), psnr(x_fp, clean), psnr(x_pp, clean)


if __name__ == "__main__":
    try:
        import pyproximal  # noqa: F401
    except ImportError:
        print("Needs PyProximal: uv run --group interop python examples/pyproximal_tv.py")
    else:
        main()
