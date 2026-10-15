# Roadmap

Planned work, release by release, following [docs/spec.md](docs/spec.md)
(Section 13). Each release ships only when its gate passes. Finished work
moves to [CHANGELOG.md](CHANGELOG.md); decisions that change the spec are
logged at the end of this file.

Status: `[x]` done · `[~]` in progress or partly done · `[ ]` not started

## 0.1 — core and 1D (in progress)

**Done**

- [x] C API with try/catch boundary, opaque handles, workspaces, `fp_options`/`fp_info`
- [x] Separable convex maps: soft thresholding (weighted), elastic net, Huber, group ℓ2 and ℓ2 norm
- [x] Projections: box, nonnegative orthant, ℓ2 ball, simplex and weighted ℓ1 ball (Condat), ℓ∞ prox
- [x] Monotone cone by pool-adjacent-violators, weighted
- [x] 1D TV: Condat's algorithm, taut string (segment chains in local coordinates), weights with zero-weight splitting, automatic fallback
- [x] JVP/VJP in x, γ and parameters for all of the above; conjugate prox via Moreau
- [x] Batching over any axis and stride, an internal thread pool with bit-identical results (OpenMP until 0.1.0rc1), no allocation with a workspace
- [x] nanobind bindings, abi3 wheel built locally with uv and scikit-build-core
- [x] C/C++ tests (certificates, finite differences, adjointness, determinism, allocation counter) and Python tests (references, properties)

- [x] Benchmarks against proxTV and Condat's C code (`bench/compare_external.py`, fetched and built at run time, fresh input per call); results in [docs/benchmarks.md](docs/benchmarks.md)
- [x] Condat backtracking budget tuned from measured work per sample (now `3n + 1024` rescanned steps; typical signals rescan 0–3n, ramps up to 3000n); ramp regression test
- [x] Fuzz harness `fuzz/fuzz_ops.cpp`: libFuzzer target where available plus a standalone driver; 400k inputs clean under ASan/UBSan; nightly workflow `.github/workflows/fuzz.yml`
- [x] CVXPY + Clarabel fixtures (180 problems, n ≤ 200) in `tests/fixtures/`; agreement within 4.2e-10
- [x] Valgrind job in CI
- [x] Single source for the version (`project()` in CMakeLists.txt; header macros checked at configure time)
- [x] `fastprox` is free on conda-forge (no package, feedstock or staged-recipes PR, 2026-10-14) and on PyPI
- [x] API reference: [docs/api.md](docs/api.md)

**Open**

- [~] Reach the speed gate on 1D TV ([docs/benchmarks.md](docs/benchmarks.md)). Now fastest or within 2% in all 9 uniform float64 settings and 8 of 9 weighted ones. Remaining: weighted white noise at λ = 10 (8% after the 8-lane range pass of 2026-10-15, was 13%; the range pass is down to about the cost of reading the row, so closing the rest means folding it into the scan, which needs a speculative run and a redo when the radii turn out to need clamping), and float32 at λ = 0.1 (1–5% behind Condat's single-precision code, was 6–12%; closing it would need single-precision running sums, which the spec rules out, so this is proposed as an accepted deviation)
- [~] Simplex and ℓ1 ball: finiteness test folded into the pivot pass (0.81× the previous time). Now 8–11% behind Condat's code on Gaussian input in one program, tied on uniform input, faster at small n
- [~] Benchmarks on homogeneous cores: a manual `bench` workflow runs both benchmark scripts on GitHub's x86_64 and arm64 runners ([docs/benchmarks.md](docs/benchmarks.md)). Python overhead confirmed (1.6–3.4 µs at n = 100, target 5 µs). Parallel scaling confirmed at 4 physical cores (3.90×, efficiency 0.97 on Neoverse-N2, 2026-10-15, now that the public repository gets 4-vCPU runners); scaling to 8–16 cores still needs a larger runner or a dedicated machine before the spec target can be confirmed
- [~] Tune for Linux arm64 / GCC: on Neoverse-N2 with GCC 13, fastprox trailed at λ = 0.1 (9–17%), weighted at λ = 10 (about 20%) and on the simplex (24–31%). Found (2026-10-15): GCC compiled the short-segment selects of Condat's algorithm as branches, and the range pass as a scalar chain twice as slow as Clang's; with both fixed, the arm64 runner went from 10 to 20 of 30 comparisons won or within 2% ([docs/benchmarks.md](docs/benchmarks.md)). Remaining on Neoverse-N2: weighted white noise at λ = 10 (25%), uniform random walk at λ = 10 (20%, where GCC's branches had helped), λ = 0.1 on noise and random walk (6–9%), and the simplex (22–23%; it compiles as well with GCC as with Clang on Apple M5, so this is the core or GCC 13). Next: per-architecture `kBranchFree`, or branches beyond the first samples at large λ
- [x] CI matrix green on GitHub (private repo dvillacis/fastprox, 2026-10-14): Linux GCC/Clang with and without threads, sanitizers (ASan/UBSan and ThreadSanitizer), Valgrind, Windows MSVC, macOS arm64 and x86_64, Python on all three. The first Windows run hung on a "DLL not found" dialog; executables and DLLs now share an output directory there, and ctest has a 300 s timeout.
- [x] Wheels with cibuildwheel (manual dispatch or tags): abi3 for CPython ≥ 3.12 on manylinux_2_28 x86_64/aarch64, macOS arm64 (11.0) and x86_64 (10.15), Windows amd64; built and tested in CI. Since 0.1.0rc2 no OpenMP runtime is bundled; the repair tools (delocate, auditwheel, delvewheel) still run
- [x] Published 0.1.0rc1 to PyPI (2026-10-15): https://pypi.org/project/fastprox/0.1.0rc1/, 5 abi3 wheels and the sdist, uploaded by trusted publishing from the `v0.1.0rc1` tag after a TestPyPI dry run; the `pypi` environment accepts only `v*` tags and needs approval. The repository is public
- [x] Published 0.1.0rc2 to PyPI (2026-10-15): thread pool instead of OpenMP, which fixes the abort next to PyTorch on macOS; checked by installing from TestPyPI and PyPI next to PyTorch 2.14 and JAX 0.11
- [ ] Final 0.1.0 once the gate passes (set `FASTPROX_VERSION_SUFFIX` to empty)

**Gate (proposed, from spec Section 10):** all accuracy, determinism, special-value and allocation requirements pass on the full CI matrix; 1D TV, simplex and ℓ1 ball no slower than proxTV and Condat's code on one thread; overhead and scaling targets confirmed or revised; wheels install and pass tests on Linux, macOS (arm64, x86_64) and Windows.

## 0.2 — images

- [ ] Anisotropic TV in 2D and 3D: parallel Douglas–Rachford or Dykstra over axes with exact 1D TV inside, duality-gap stopping, warm starts
- [ ] Isotropic TV in 2D and 3D: fast gradient projection on the dual, fused single pass, gap stopping, dual field returned and accepted for warm starts
- [ ] 1D TV with ℓ2 of differences: projected Newton on the dual, gap
- [ ] Nonconvex pointwise maps: ℓ0, ℓ1/2, ℓ2/3, general ℓq (threshold test, safeguarded Newton), SCAD and MCP (`FP_EPARAM` below the concavity bound), log penalty
- [ ] Tie rule (`fp_options.tie`, nonzero branch by default) and tie counts in `fp_info.ties`
- [ ] Derivatives for the nonconvex maps (diagonal, support-restricted)
- [ ] Python: `fp.lq`, `fp.tv_iso` returning `(z, info)` with `info.gap` and `info.dual`, and the other new operators
- [ ] Check the citations marked "to check" in the spec (Beck–Teboulle; Xu et al.; Cao, Sun and Xu; Fan–Li; Zhang) before they go into kernel headers

**Gate:** iterative operators meet `gap ≤ tol · ½‖x‖²` with the gap recomputed independently in tests; nonconvex maps verified against a dense grid with refinement, tie flags exact at thresholds.

## 0.3 — graph TV

- [ ] Max-flow engine (Boykov–Kolmogorov style, tree reuse between levels), written from the paper (clean room)
- [ ] Graph TV by parametric max-flow, exact to a user-set precision (Chambolle–Darbon, Hochbaum)
- [ ] TV on trees and chains (Kolmogorov, Pock and Rolínek)
- [ ] VJPs for the iterative operators by conjugate gradients on the dual active set, returning the residual
- [ ] PyTorch (`torch.autograd.Function`) and JAX (`custom_vjp` + `pure_callback`) wrappers, checked with `gradcheck`

## 1.0 — stable

- [ ] ABI freeze; after this, only additions that keep the ABI until 2.0
- [ ] conda-forge packages for the C library and the Python package
- [ ] Documentation
- [ ] PyProximal adapter (`ProxOperator` subclass)
- [ ] BenchOpt solvers for TV-1D and TV-2D, and a new simplex-projection benchmark

## After 1.0

- [ ] Julia bindings (ccall through a JLL package) and a ProximalOperators.jl adapter
- [ ] MATLAB mex gateways, with a gate that they load on Linux next to MATLAB's own libstdc++
- [ ] Rust `fastprox-sys` plus a thin safe crate

## Version 2 and later

- [ ] Matrix-valued prox maps (nuclear norm, spectral functions)
- [ ] GPU backend (the `fp_device` field is reserved in `fp_array` from 0.1)

## Decisions and deviations from the spec

| Date | Decision | Where |
| --- | --- | --- |
| 2026-10-14 | `fp_jvp` takes a `dparams` argument, so JVPs cover the parameters as the spec requires | `fastprox.h` |
| 2026-10-14 | `fp_op_tv1d` takes an `fp_tv1d_method` (auto, Condat, taut string) for the manual override | `fastprox.h` |
| 2026-10-14 | Every operator has a parameter vector θ (`fp_op_nparams`); for soft thresholding θ is the weights, and Python also reports `grads.lam` | `fastprox.h`, Python |
| 2026-10-14 | At the end of the signal, Condat's algorithm restarts after a jump with the plain post-jump bounds instead of the paper's tightened ones (measured: identical work over 200k signals, and simpler) | `src/ops/tv1d.cpp` |
| 2026-10-14 | OpenMP uses dynamic scheduling over rows (balances performance and efficiency cores); determinism holds because rows are independent and vjp reductions use fixed blocks. The thread pool that replaced OpenMP (2026-10-15) keeps this scheduling | `src/core/driver.hpp` |
| 2026-10-14 | The automatic 1D TV path lets Condat's algorithm rescan at most 3n + 1024 steps before switching to the taut string (measured: typical signals rescan 0–3n, ramps up to 3000n; past that the taut string is also faster) | `src/ops/tv1d.cpp` |
| 2026-10-14 | Condat's bound updates use branch-free selects and a static table of 2¹⁵ reciprocals: faster on noisy signals, about 4% slower on smooth ones at large λ; an adaptive switch on clamp frequency was tried and mispredicts | `src/ops/tv1d.cpp` |
| 2026-10-14 | Weighted 1D TV runs Condat's algorithm generalised to per-edge radii, precomputed per piece (evaluating them inside the scan doubled its cost) (new segment after a jump at edge e starts from y_k + σ r_e ∓ r_k; the last sample has radius 0), checked against the taut string on 40k problems | `src/ops/tv1d.cpp` |
| 2026-10-14 | 1D TV clamps radii above twice the dual bound min(e+1, n−1−e)·(max x − min x), which are never active, and returns the constant mean when every radius exceeds the bound; this keeps the 10 n ε ‖x‖∞ accuracy for extreme weights and steps | `src/ops/tv1d.cpp` |
| 2026-10-14 | The taut string stores its chains as segments in coordinates local to the current knot (after proxTV's classic taut string, credited in the kernel header) instead of absolute double-double heights: 17–26% faster, error at most 4e-4 of the spec bound | `src/ops/tv1d.cpp` |
| 2026-10-14 | macOS wheels target the macOS version of the build runner's Homebrew libomp (14.0 arm64 on macos-14, 15.0 x86_64 on macos-15-intel, which replaced the retired macos-13 runners), since delocate refuses older targets. Superseded on 2026-10-15: with no libomp bundled, the wheels target 11.0 (arm64) and 10.15 (x86_64) | `pyproject.toml`, CI |
| 2026-10-14 | Condat's 1D TV uses branch-free bound updates only in the first 8 samples of a segment and branches after (bounds settle in long segments and branches predict well); 1/m is fetched only when a clamp happens in long segments | `src/ops/tv1d.cpp` |
| 2026-10-14 | Weighted ℓ1-ball weights are stored normalised by their maximum, and weight ratios below 1e-100 are rejected with `FP_EPARAM` (their squares underflow; found by fuzzing) | `src/ops/projections.cpp`, `fastprox.h` |
| 2026-10-15 | The first PyPI upload is the release candidate 0.1.0rc1, before the 0.1 gate passes (open speed items), so users can install it with `--pre` while 0.1.0 stays reserved for the gated release; the repository is made public | `CMakeLists.txt`, CI |
| 2026-10-15 | Rows run on fastprox's own thread pool instead of OpenMP (spec: "OpenMP over rows"). Each wheel had to bundle an OpenMP runtime, and two copies of LLVM's libomp in one process abort it; 0.1.0rc1 crashed next to PyTorch on macOS. The pool keeps dynamic scheduling and bit-identical results, spins 50 µs before sleeping, runs concurrent callers serially, and rebuilds itself after fork. `threads = 0` reads `FASTPROX_NUM_THREADS`, then `OMP_NUM_THREADS`, then the available CPUs. `fp_has_openmp` became `fp_default_threads` (the ABI is not frozen before 1.0) | `src/core/parallel.cpp`, `fastprox.h` |
