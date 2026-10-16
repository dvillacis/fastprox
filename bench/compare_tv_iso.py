"""Isotropic 2D and 3D TV: time to reach a given accuracy, fastprox against
scikit-image's denoise_tv_chambolle (Chambolle's 2004 projection algorithm,
NumPy, BSD; only called here), single thread.

    uv run --with scikit-image python bench/compare_tv_iso.py [--quick]

Both solve min_z |z - x|^2 / 2 + gamma sum_p |(grad z)_p|_2 with forward
differences (scikit-image's ``weight`` is gamma). Each solver is run with a
range of settings (fastprox: the gap tolerance; scikit-image: the iteration
cap, with its own stopping turned off), and the table gives, for each target
error |z - z*| / |z*| against a reference solved to a gap of 1e-15, the
fastest run that reached it ("-" when none did). scikit-image is vectorised
NumPy, not compiled code, so this is a baseline, not a like-for-like race.
"""

import argparse
import platform
import time

import numpy as np

import fastprox as fp


def image(shape, rng):
    idx = np.indices(shape)
    blocks = sum((i // max(1, s // 6)) for i, s in zip(idx, shape)) % 3
    return blocks.astype(float) + 0.3 * rng.normal(size=shape)


def best_time(f, reps):
    ts, z = [], None
    for _ in range(reps):
        t = time.perf_counter()
        z = f()
        ts.append(time.perf_counter() - t)
    return min(ts), z


def table(k, shape, gamma, reps, targets, skimage_fn):
    rng = np.random.default_rng(1)
    x = image(shape, rng)
    zref, info = fp.tv_iso(x, gamma, ndim=k, tol=1e-15, max_iter=10**7, threads=1)
    assert info.converged
    nref = np.linalg.norm(zref)
    err = lambda z: np.linalg.norm(z - zref) / nref  # noqa: E731
    curves = {}
    pts = []
    for tol in 10.0 ** -np.arange(2, 13):
        t, z = best_time(lambda: fp.tv_iso(x, gamma, ndim=k, tol=tol, max_iter=10**7,
                                           threads=1)[0], reps)
        pts.append((t, err(z)))
    curves["fastprox (FDPG)"] = pts
    if skimage_fn is not None:
        pts = []
        for it in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            t, z = best_time(lambda: skimage_fn(x, weight=gamma, eps=0.0, max_num_iter=it), reps)
            pts.append((t, err(z)))
        curves["scikit-image denoise_tv_chambolle"] = pts
    print(f"\n{k}D isotropic TV, {'x'.join(map(str, shape))}, gamma = {gamma}: "
          f"ms to reach |z - z*| / |z*| <= target")
    print("  " + " " * 36 + "".join(f"{t:>10.0e}" for t in targets))
    for name, pts in curves.items():
        row = []
        for tgt in targets:
            ok = [t for t, e in pts if e <= tgt]
            row.append(f"{1e3 * min(ok):10.2f}" if ok else f"{'-':>10}")
        print(f"  {name:36s}" + "".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller images, fewer repetitions")
    args = ap.parse_args()
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None
        print("scikit-image not installed: fastprox only")
    print(f"fastprox {fp.__version__}, {platform.machine()}, single thread")
    reps = 1 if args.quick else 3
    targets = (1e-2, 1e-3, 1e-4, 1e-6)
    sizes2 = [(128, 128)] if args.quick else [(256, 256)]
    sizes3 = [(32, 32, 32)] if args.quick else [(64, 64, 64)]
    for shape in sizes2:
        for gamma in (0.1, 0.3):
            table(2, shape, gamma, reps, targets, denoise_tv_chambolle)
    for shape in sizes3:
        for gamma in (0.1, 0.3):
            table(3, shape, gamma, reps, targets, denoise_tv_chambolle)


if __name__ == "__main__":
    main()
