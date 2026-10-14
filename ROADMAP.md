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
- [x] Batching over any axis and stride, OpenMP with bit-identical results, no allocation with a workspace
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

- [~] Reach the speed gate on 1D TV ([docs/benchmarks.md](docs/benchmarks.md)). Done: branch-free Condat bound updates, segment-based taut string after proxTV (17–26% faster), weighted rows on Condat's algorithm with precomputed per-edge radii (was 1.4–2.8× slower than proxTV, now faster in 5 of 9 settings). Remaining gaps, n = 10⁶: uniform λ = 10 on random walks (16%) and white noise (12%) against proxTV's linearized taut string; float32 at λ = 0.1 (8–16% against Condat's code); weighted at λ = 0.1 (12–20%) and white noise at λ = 10 (about 1.6×). The small-λ gap is the branch-free updates costing a little where branches predict well
- [ ] Simplex and ℓ1 ball: 13–28% slower than Condat's code at n ≥ 10⁵ (faster at 10³). The kernel alone is within 15%; the rest is the separate finiteness pass and the driver
- [ ] Benchmark on a homogeneous-core runner (separate runs on this laptop differ by up to 1.6×), and measure the design targets there: parallel scaling ≥ 0.7 × threads (measured locally: 0.86 at 2 threads, 0.69 at 4 on a 5P+10E machine) and Python overhead < 5 µs (measured 0.9–2.3 µs at n = 100); confirm or revise them in the spec
- [ ] Run the CI matrix (needs a GitHub remote); expect fixes for MSVC (OpenMP 2.0 loop variables, `/W4`). The macOS wheel runners are pinned to `macos-14`/`macos-13` because delocate raises the wheel's minimum macOS to that of Homebrew's libomp; a libomp built for older targets would widen support
- [ ] Release wheels with cibuildwheel

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
| 2026-10-14 | OpenMP uses dynamic scheduling over rows (balances performance and efficiency cores); determinism holds because rows are independent and vjp reductions use fixed blocks | `src/core/driver.hpp` |
| 2026-10-14 | The automatic 1D TV path lets Condat's algorithm rescan at most 3n + 1024 steps before switching to the taut string (measured: typical signals rescan 0–3n, ramps up to 3000n; past that the taut string is also faster) | `src/ops/tv1d.cpp` |
| 2026-10-14 | Condat's bound updates use branch-free selects and a static table of 2¹⁵ reciprocals: faster on noisy signals, about 4% slower on smooth ones at large λ; an adaptive switch on clamp frequency was tried and mispredicts | `src/ops/tv1d.cpp` |
| 2026-10-14 | Weighted 1D TV runs Condat's algorithm generalised to per-edge radii, precomputed per piece (evaluating them inside the scan doubled its cost) (new segment after a jump at edge e starts from y_k + σ r_e ∓ r_k; the last sample has radius 0), checked against the taut string on 40k problems | `src/ops/tv1d.cpp` |
| 2026-10-14 | 1D TV clamps radii above twice the dual bound min(e+1, n−1−e)·(max x − min x), which are never active, and returns the constant mean when every radius exceeds the bound; this keeps the 10 n ε ‖x‖∞ accuracy for extreme weights and steps | `src/ops/tv1d.cpp` |
| 2026-10-14 | The taut string stores its chains as segments in coordinates local to the current knot (after proxTV's classic taut string, credited in the kernel header) instead of absolute double-double heights: 17–26% faster, error at most 4e-4 of the spec bound | `src/ops/tv1d.cpp` |
| 2026-10-14 | macOS wheels target the macOS version of the build runner's Homebrew libomp (14.0 arm64, 13.0 x86_64), since delocate refuses older targets | `pyproject.toml`, CI |
