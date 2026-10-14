"""Single-thread speed of fastprox against proxTV and Condat's C code.

    uv run python bench/compare_external.py [--quick]

The external codes are fetched and compiled into build/external/ when the
script runs; they are never vendored or shipped (docs/spec.md, Section 12).
Clean-room rule: this script only compiles and calls them. Condat's files
(CeCILL) and proxTV's bundled copies of Condat's and Kolmogorov's codes are
not to be read by anyone working on fastprox's kernels.

Timing uses a fresh input for every repetition, shared by all solvers in turn:
repeating one input lets the branch predictor learn it and makes the direct
algorithms look several times faster, and interleaving the solvers keeps the
drift of a laptop (frequency, migration between core types) out of the
comparison.
Every external result is checked against fastprox before it is timed.
"""

import argparse
import ctypes
import pathlib
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np

import fastprox as fp

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXT = ROOT / "build" / "external"
SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
CXX = "c++"
CC = "cc"
OPT = ["-O3", "-fPIC", "-shared", "-DNDEBUG"]

CONDAT_TV = "https://lcondat.github.io/download/condat_fast_tv.c"
CONDAT_SIMPLEX = "https://lcondat.github.io/download/condat_simplexproj.c"
PROXTV = "https://github.com/albarji/proxTV"

# extern "C" shim over proxTV's C++ functions (declarations from TVopt.h).
PROXTV_SHIM = r'''
#include "TVopt.h"
extern "C" {
int shim_classic_taut(double *y, int n, double lam, double *x) { return classicTautString_TV1(y, n, lam, x); }
int shim_linearized_taut(double *y, int n, double lam, double *x) { return linearizedTautString_TV1(y, lam, x, n); }
void shim_hybrid_taut(double *y, int n, double lam, double *x) { hybridTautString_TV1(y, n, lam, x); }
int shim_weighted_taut(double *y, double *w, double *x, int n) { return tautString_TV1_Weighted(y, w, x, n); }
}
'''


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def exported(lib: pathlib.Path):
    out = subprocess.run(["nm", "-g", str(lib)], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "T":
            names.add(parts[2].lstrip("_") if platform.system() == "Darwin" else parts[2])
    return names


def gsl_flags():
    """Condat's files include GSL for their demo main(); find its headers."""
    for prefix in ("/opt/homebrew/opt/gsl", "/usr/local/opt/gsl"):
        if pathlib.Path(prefix, "include").exists():
            return [f"-I{prefix}/include", f"-L{prefix}/lib", "-lgsl", "-lgslcblas"]
    return ["-lgsl", "-lgslcblas"]


def build_condat():
    d = EXT / "condat"
    d.mkdir(parents=True, exist_ok=True)
    libs = {}
    for url, name in ((CONDAT_TV, "condat_tv"), (CONDAT_SIMPLEX, "condat_simplex")):
        src = d / pathlib.Path(url).name
        if not src.exists():
            urllib.request.urlretrieve(url, src)
        lib = d / f"lib{name}{SUFFIX}"
        if not lib.exists():
            # Its entry points are static; -Dstatic= exports them unchanged.
            extra = ["-Dstatic="] if name == "condat_simplex" else []
            run([CC, *OPT, "-w", "-Dmain=condat_unused_main", *extra, *gsl_flags(), "-o",
                 str(lib), str(src), "-lm"])
        libs[name] = lib
    return libs


def build_proxtv():
    d = EXT / "proxTV"
    if not d.exists():
        run(["git", "clone", "-q", "--depth", "1", PROXTV, str(d)])
    lib = EXT / f"libproxtv_tv1{SUFFIX}"
    if not lib.exists():
        shim = EXT / "proxtv_shim.cpp"
        shim.write_text(PROXTV_SHIM)
        src = d / "src"
        files = ["TVL1opt_tautstring.cpp", "TVL1opt_hybridtautstring.cpp", "TVL1Wopt.cpp",
                 "TVL1opt.cpp", "condat_fast_tv.cpp", "utils.cpp"]
        run([CXX, *OPT, "-std=c++11", "-w", "-DNOMATLAB", f"-I{src}", "-o", str(lib), str(shim),
             *[str(src / f) for f in files], "-llapack" if platform.system() != "Darwin"
             else "-framework", *([] if platform.system() != "Darwin" else ["Accelerate"])])
    return lib


DBL = ctypes.POINTER(ctypes.c_double)


def ptr(a):
    return a.ctypes.data_as(DBL)


def tv_solvers():
    """name -> (f(x, lam, out), dtype) for every available 1D TV implementation."""
    f64, f32 = np.float64, np.float32
    solvers = {
        "fastprox auto": (lambda x, lam, out: fp.tv1d(x, lam, out=out, threads=1), f64),
        "fastprox condat": (lambda x, lam, out: fp.tv1d(x, lam, method="condat", out=out,
                                                        threads=1), f64),
        "fastprox taut string": (lambda x, lam, out: fp.tv1d(x, lam, method="taut_string",
                                                             out=out, threads=1), f64),
    }
    try:
        ptv = ctypes.CDLL(str(build_proxtv()))
        for sym, label in (("shim_classic_taut", "classic"), ("shim_linearized_taut", "linearized"),
                           ("shim_hybrid_taut", "hybrid")):
            f = getattr(ptv, sym)
            f.argtypes = [DBL, ctypes.c_int, ctypes.c_double, DBL]
            solvers[f"proxTV {label} taut string"] = (
                (lambda f: lambda x, lam, out: f(ptr(x), x.size, lam, ptr(out)))(f), f64)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"proxTV unavailable: {e}", file=sys.stderr)
    solvers["fastprox auto (float32)"] = (lambda x, lam, out: fp.tv1d(x, lam, out=out, threads=1),
                                          f32)
    try:
        lib_path = build_condat()["condat_tv"]
        ctv = ctypes.CDLL(str(lib_path))
        flt = ctypes.POINTER(ctypes.c_float)
        for sym in sorted(s for s in exported(lib_path) if s.startswith("TV1D_denoise")):
            f = getattr(ctv, sym)
            f.argtypes = [flt, flt, ctypes.c_int, ctypes.c_float]
            f.restype = None
            solvers[f"Condat {sym} (float32)"] = (
                (lambda f: lambda x, lam, out: f(x.ctypes.data_as(flt), out.ctypes.data_as(flt),
                                                 x.size, lam))(f), f32)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Condat's TV code unavailable: {e}", file=sys.stderr)
    return solvers


def weighted_tv_solvers():
    """name -> (f(x, lam_per_edge, out), dtype) for weighted 1D TV."""
    f64 = np.float64
    solvers = {"fastprox": (lambda x, lam, out: fp.tv1d(x, 1.0, weights=lam, out=out, threads=1),
                            f64)}
    try:
        ptv = ctypes.CDLL(str(build_proxtv()))
        ptv.shim_weighted_taut.argtypes = [DBL, DBL, DBL, ctypes.c_int]
        solvers["proxTV weighted taut string"] = (
            lambda x, lam, out: ptv.shim_weighted_taut(ptr(x), ptr(lam), ptr(out), x.size), f64)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"proxTV unavailable: {e}", file=sys.stderr)
    return solvers


def simplex_solvers():
    solvers = {"fastprox": lambda x, r, out: fp.project_simplex(x, r, out=out, threads=1)}
    try:
        lib_path = build_condat()["condat_simplex"]
        lib = ctypes.CDLL(str(lib_path))
        for sym in ("simplexproj_Condat",):
            f = getattr(lib, sym)
            f.argtypes = [DBL, DBL, ctypes.c_uint, ctypes.c_double]
            f.restype = None
            solvers[f"Condat {sym}"] = (lambda f: lambda x, r, out: f(ptr(x), ptr(out), x.size, r))(f)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Condat's simplex code unavailable: {e}", file=sys.stderr)
    return solvers


# Signal families: piecewise constant plus noise (the BenchOpt TV-1D setting),
# random walk, white noise.
def signal(kind, n, rng):
    if kind == "blocks":
        return np.repeat(rng.normal(scale=3, size=n // 50 + 1), 50)[:n] + 0.5 * rng.normal(size=n)
    if kind == "random walk":
        return np.cumsum(rng.normal(size=n))
    return rng.normal(size=n)


def check(solvers, x, arg):
    """Names of solvers whose output differs from fastprox in their precision."""
    arg = arg(x.size) if callable(arg) else arg
    ref = {}
    bad = []
    for name, (f, dt) in solvers.items():
        xx = x.astype(dt)
        if dt not in ref:
            ref[dt] = np.empty_like(xx)
            next(g for g, d in solvers.values() if d == dt)(xx, arg, ref[dt])
        out = np.empty_like(xx)
        f(xx, arg, out)
        tol = (1e-4 if dt == np.float32 else 1e-9) * (1 + np.abs(x).max())
        if not np.allclose(out, ref[dt], rtol=0, atol=tol):
            bad.append(name)
    return bad


def table(solvers, make_input, arg, sizes, quick):
    """One row per solver. For every size, each repetition draws a fresh
    input and runs every solver on it in turn, so drift of the machine
    (frequency, core migration) affects all solvers alike; the median over
    repetitions is reported."""
    bad = check(solvers, make_input(2000, np.random.default_rng(0)), arg)
    if bad:
        print(f"  disagree with fastprox (excluded): {', '.join(bad)}")
    names = [k for k in solvers if k not in bad]
    best = {k: [] for k in names}
    for n in sizes:
        reps = 3 if quick else max(5, min(30, 10**6 // n))
        rng = np.random.default_rng(n)
        a = arg(n) if callable(arg) else arg
        times = {k: [] for k in names}
        for r in range(reps + 1):  # the first round warms up
            x64 = make_input(n, rng)
            for k in names:
                f, dt = solvers[k]
                x = x64.astype(dt)
                out = np.empty_like(x)
                t = time.perf_counter()
                f(x, a, out)
                if r:
                    times[k].append(time.perf_counter() - t)
        for k in names:
            best[k].append(np.median(times[k]) / n * 1e9)
    print("  " + " " * 34 + "".join(f"{n:>10.0e}" for n in sizes))
    for k in names:
        print(f"  {k:34s}" + "".join(f"{v:10.2f}" for v in best[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sizes, fewer repetitions")
    args = ap.parse_args()
    sizes = (10**3, 10**5) if args.quick else (10**3, 10**4, 10**5, 10**6, 10**7)

    print(f"fastprox {fp.__version__}, {platform.machine()}, single thread, ns per sample")
    tv = tv_solvers()
    for kind in ("blocks", "random walk", "white noise"):
        for lam in (0.1, 1.0, 10.0):
            print(f"\n1D TV, {kind}, lambda = {lam}")
            table(tv, lambda n, rng: signal(kind, n, rng), lam, sizes, args.quick)

    wtv = weighted_tv_solvers()
    for kind in ("blocks", "random walk", "white noise"):
        for lam in (0.1, 1.0, 10.0):
            print(f"\nWeighted 1D TV, {kind}, lambda = {lam} times edge weights in [0.5, 1.5]")
            weights = lambda n, lam=lam: lam * np.random.default_rng(7).uniform(0.5, 1.5, n - 1)
            table(wtv, lambda n, rng: signal(kind, n, rng), weights, sizes, args.quick)

    print("\nSimplex projection, radius 1")
    sx = {k: (f, np.float64) for k, f in simplex_solvers().items()}
    for kind, make in (("Gaussian", lambda n, rng: rng.normal(size=n)),
                       ("uniform", lambda n, rng: rng.uniform(size=n)),
                       ("one large entry", lambda n, rng: np.r_[n, rng.normal(size=n - 1)])):
        print(f"  input: {kind}")
        table(sx, make, 1.0, sizes, args.quick)


if __name__ == "__main__":
    main()
