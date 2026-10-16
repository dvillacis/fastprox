"""Anisotropic 2D and 3D TV: time to reach a given accuracy, fastprox against
proxTV's solvers, single thread.

    uv run python bench/compare_tv_aniso.py [--quick]

The solvers stop on different criteria, so each one is run with a range of
settings (fastprox: the gap tolerance; proxTV: the iteration cap) and the
error |z - z*| / |z*| of every run is measured against a reference solved to
a gap of 1e-15 relative. The table gives, for each target error, the fastest
run that reached it ("-" when none did).

proxTV is fetched and compiled into build/external/ as in compare_external.py
(BSD; only compiled and called here), without OpenMP, so it runs on one
thread. All weights are equal, so the layout proxTV assumes for its arrays
does not change the problem.
"""

import argparse
import ctypes
import pathlib
import platform
import sys
import time

import numpy as np

import fastprox as fp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare_external as ce  # noqa: E402

SHIM = r"""
#include "TVopt.h"
extern "C" {
int shim_dr2(size_t M, size_t N, double *y, double lam, double *x, int maxit, double *info) {
    return DR2_TV(M, N, y, lam, lam, 1, 1, x, 1, maxit, info);
}
int shim_kolmogorov2(size_t M, size_t N, double *y, double lam, double *x, int maxit, double *info) {
    return Kolmogorov2_TV(M, N, y, lam, x, maxit, info);
}
int shim_yang2(size_t M, size_t N, double *y, double lam, double *x, int maxit, double *info) {
    return Yang2_TV(M, N, y, lam, x, maxit, info);
}
int shim_yang3(size_t M, size_t N, size_t O, double *y, double lam, double *x, int maxit, double *info) {
    return Yang3_TV(M, N, O, y, lam, x, maxit, info);
}
// Parallel Dykstra (pd = 1) or parallel Douglas-Rachford (pd = 0) over the
// nd axes, all with weight lam.
int shim_nd(int pd, int nd, const int *ns, double *y, double lam, double *x, int maxit, double *info) {
    double lams[3], norms[3], dims[3];
    int n[3];
    for (int i = 0; i < nd; ++i) { lams[i] = lam; norms[i] = 1; dims[i] = i + 1; n[i] = ns[i]; }
    return pd ? PD_TV(y, lams, norms, dims, x, info, n, nd, nd, 1, maxit)
              : PDR_TV(y, lams, norms, dims, x, info, n, nd, nd, 1, maxit);
}
}
"""

def build():
    ce.build_proxtv()  # fetches the sources
    lib = ce.EXT / f"libproxtv_tvnd{ce.SUFFIX}"
    if not lib.exists():
        shim = ce.EXT / "proxtv_shim_nd.cpp"
        shim.write_text(SHIM)
        src = ce.EXT / "proxTV" / "src"
        files = [str(p) for p in sorted(src.glob("*.cpp"))]
        link = ["-framework", "Accelerate"] if platform.system() == "Darwin" else ["-llapack"]
        ce.run([ce.CXX, *ce.OPT, "-std=c++11", "-w", "-DNOMATLAB", f"-I{src}", "-o", str(lib),
                str(shim), *files, *link])
    return ctypes.CDLL(str(lib))


def image(shape, rng):
    """Piecewise-constant blocks plus Gaussian noise."""
    idx = np.indices(shape)
    blocks = sum((i // max(1, s // 6)) for i, s in zip(idx, shape)) % 3
    return blocks.astype(float) + 0.3 * rng.normal(size=shape)


def solvers(lib, k, shape):
    D = ctypes.POINTER(ctypes.c_double)
    p = lambda a: a.ctypes.data_as(D)  # noqa: E731
    info = np.zeros(8)
    S = ctypes.c_size_t
    out = {}

    def ptv(fn):
        return lambda x, lam, maxit: fn(x, lam, maxit)

    if k == 2:
        M, N = shape
        for name, f in (("proxTV DR2 (alternating reflections)", lib.shim_dr2),
                        ("proxTV Kolmogorov2", lib.shim_kolmogorov2),
                        ("proxTV Yang2 (ADMM)", lib.shim_yang2)):
            def run(x, lam, it, f=f):
                z = np.empty_like(x)
                f(S(M), S(N), p(x), ctypes.c_double(lam), p(z), it, p(info))
                return z
            out[name] = run
    else:
        M, N, O = shape

        def run3(x, lam, it):
            z = np.empty_like(x)
            lib.shim_yang3(S(M), S(N), S(O), p(x), ctypes.c_double(lam), p(z), it, p(info))
            return z
        out["proxTV Yang3 (ADMM)"] = run3
    ns = (ctypes.c_int * k)(*shape)
    for pd, name in ((1, "proxTV parallel Dykstra"), (0, "proxTV parallel DR")):
        def runnd(x, lam, it, pd=pd):
            z = np.empty_like(x)
            lib.shim_nd(pd, k, ns, p(x), ctypes.c_double(lam), p(z), it, p(info))
            return z
        out[name] = runnd
    return out


def best_time(f, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        z = f()
        ts.append(time.perf_counter() - t)
    return min(ts), z


def table(lib, k, shape, lam, reps, targets):
    rng = np.random.default_rng(1)
    x = image(shape, rng)
    zref, info = fp.tv_aniso(x, lam, ndim=k, tol=1e-15, max_iter=10**6, threads=1)
    assert info.converged
    nref = np.linalg.norm(zref)
    err = lambda z: np.linalg.norm(z - zref) / nref  # noqa: E731
    curves = {}
    pts = []
    for tol in 10.0 ** -np.arange(2, 13):
        t, z = best_time(lambda: fp.tv_aniso(x, lam, ndim=k, tol=tol, max_iter=10**6,
                                             threads=1)[0], reps)
        pts.append((t, err(z)))
    curves["fastprox (ADBPG)" if k == 2 else "fastprox (FDPG, 3 blocks)"] = pts
    for name, f in solvers(lib, k, shape).items():
        pts = []
        for it in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            t, z = best_time(lambda: f(x, lam, it), reps)
            if np.all(np.isfinite(z)):
                e = min(err(z), err(z.T) if k == 2 else np.inf)  # orientation-proof
                pts.append((t, e))
        curves[name] = pts
    n = int(np.prod(shape))
    print(f"\n{k}D anisotropic TV, {'x'.join(map(str, shape))}, lambda = {lam}: "
          f"ms to reach |z - z*| / |z*| <= target")
    print("  " + " " * 40 + "".join(f"{t:>10.0e}" for t in targets))
    for name, pts in curves.items():
        row = []
        for tgt in targets:
            ok = [t for t, e in pts if e <= tgt]
            row.append(f"{1e3 * min(ok):10.2f}" if ok else f"{'-':>10}")
        print(f"  {name:40s}" + "".join(row))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller images, fewer repetitions")
    args = ap.parse_args()
    lib = build()
    print(f"fastprox {fp.__version__}, {platform.machine()}, single thread")
    reps = 1 if args.quick else 3
    targets = (1e-2, 1e-3, 1e-4, 1e-6)
    sizes2 = [(128, 128)] if args.quick else [(256, 256), (512, 512)]
    sizes3 = [(32, 32, 32)] if args.quick else [(64, 64, 64)]
    for shape in sizes2:
        for lam in (0.1, 0.5):
            table(lib, 2, shape, lam, reps, targets)
    for shape in sizes3:
        for lam in (0.1, 0.5):
            table(lib, 3, shape, lam, reps, targets)


if __name__ == "__main__":
    main()
