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
- [x] 1D TV: Condat's algorithm, taut string (compensated sums), weights with zero-weight splitting, automatic fallback
- [x] JVP/VJP in x, γ and parameters for all of the above; conjugate prox via Moreau
- [x] Batching over any axis and stride, OpenMP with bit-identical results, no allocation with a workspace
- [x] nanobind bindings, abi3 wheel built locally with uv and scikit-build-core
- [x] C/C++ tests (certificates, finite differences, adjointness, determinism, allocation counter) and Python tests (references, properties)

- [x] Benchmarks against proxTV and Condat's C code (`bench/compare_external.py`, fetched and built at run time, fresh input per call); results in [docs/benchmarks.md](docs/benchmarks.md)
- [x] Condat backtracking budget tuned to `4n + 1024` from measured work per sample (typical signals 1–4n, ramps up to 3000n); ramp regression test
- [x] Fuzz harness `fuzz/fuzz_ops.cpp`: libFuzzer target where available plus a standalone driver; 400k inputs clean under ASan/UBSan; nightly workflow `.github/workflows/fuzz.yml`
- [x] CVXPY + Clarabel fixtures (180 problems, n ≤ 200) in `tests/fixtures/`; agreement within 4.2e-10
- [x] Valgrind job in CI
- [x] Single source for the version (`project()` in CMakeLists.txt; header macros checked at configure time)
- [x] `fastprox` is free on conda-forge (no package, feedstock or staged-recipes PR, 2026-10-14) and on PyPI
- [x] API reference: [docs/api.md](docs/api.md)

**Open**

- [ ] Reach the speed gate on 1D TV: fastprox's Condat path is 10–30% slower than proxTV's linearized/hybrid taut string at λ = 1, and Condat's own float32 code is up to 40% faster there. The cost is per step (work per sample is already minimal: the paper's tighter end-of-signal restart saves nothing, checked over 200k signals). Also bring the taut string closer to proxTV's classic one (now ~1.5× slower, mostly the compensated sums)
- [ ] Simplex and ℓ1 ball: fastprox is 10–30% slower than Condat's code at n ≥ 10⁵ (faster at 10³); the remaining gap is mostly the separate finiteness pass
- [ ] Benchmark on a homogeneous-core runner (this laptop varies ±20% run to run), and measure the design targets there: parallel scaling ≥ 0.7 × threads (measured locally: 0.86 at 2 threads, 0.69 at 4 on a 5P+10E machine) and Python overhead < 5 µs (measured 0.9–2.3 µs at n = 100); confirm or revise them in the spec
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
| 2026-10-14 | At the end of the signal, Condat's algorithm restarts after a jump with the plain post-jump bounds instead of the paper's tightened ones (more rescanning, cannot be wrong) | `src/ops/tv1d.cpp` |
| 2026-10-14 | Taut-string heights are carried as compensated double-double sums to meet the 10 n ε ‖x‖∞ accuracy on long rows | `src/ops/tv1d.cpp` |
| 2026-10-14 | OpenMP uses dynamic scheduling over rows (balances performance and efficiency cores); determinism holds because rows are independent and vjp reductions use fixed blocks | `src/core/driver.hpp` |
| 2026-10-14 | The automatic 1D TV path gives Condat's algorithm a budget of 4n + 1024 steps before switching to the taut string (measured: typical signals need 1–4n, ramps up to 3000n; past 4n the taut string is also faster) | `src/ops/tv1d.cpp` |
| 2026-10-14 | macOS wheels target the macOS version of the build runner's Homebrew libomp (14.0 arm64, 13.0 x86_64), since delocate refuses older targets | `pyproject.toml`, CI |
