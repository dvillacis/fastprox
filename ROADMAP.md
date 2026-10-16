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
- [x] Examples in `examples/` (2026-10-15), run by the Python tests

**Open**

- [x] Reach the speed gate on 1D TV ([docs/benchmarks.md](docs/benchmarks.md)) on Apple M5: fastest or within 2% in all 9 uniform and all 9 weighted float64 settings. Weighted white noise at λ = 10, which trailed by 8%, went to 0.43–0.45 of proxTV's time with the running-sum Condat kernel (2026-10-15), so on Apple M5 every float64 setting is now ahead. Remaining: float32 at λ = 0.1 (1–5% behind Condat's single-precision code, was 6–12%; accepted as a deviation on 2026-10-15, see the decisions table)
- [~] Simplex and ℓ1 ball: finiteness test folded into the pivot pass (0.81× the previous time), and blocks at or below the running pivot skipped whole (2026-10-15). On Apple M5 now faster than Condat's code on Gaussian input and with one large entry at every size from 10³ to 10⁷ (0.50 against 0.56 and 0.48 against 0.60 ns per sample at 10⁶), tied on uniform input at 10⁶ (7% behind at 10⁵, ahead at 10³ and 10⁷). On the CI runners (n = 10⁵) 0.71–1.01 of Condat's time on Neoverse-N2 and 0.70–0.73 on x86_64
- [x] Benchmarks on homogeneous cores: a manual `bench` workflow runs both benchmark scripts on GitHub's x86_64 and arm64 runners ([docs/benchmarks.md](docs/benchmarks.md)). Python overhead confirmed (1.6–3.4 µs at n = 100, target 5 µs). Parallel scaling confirmed at 4 physical cores (3.90×, efficiency 0.97 on Neoverse-N2, 2026-10-15, now that the public repository gets 4-vCPU runners); at 16 physical cores on a Google Cloud c4a-standard-16 (Neoverse-V2, 2026-10-16): 15.86× on 64 rows of 10⁴ (efficiency 0.99, target 0.7), 0.97–1.02 on larger batches
- [~] Tune for Linux arm64 / GCC: on Neoverse-N2 with GCC 13, fastprox trailed at λ = 0.1 (9–17%), weighted at λ = 10 (about 20%) and on the simplex (24–31%). Found (2026-10-15): GCC compiled the short-segment selects of Condat's algorithm as branches, and the range pass as a scalar chain twice as slow as Clang's; with both fixed, the arm64 runner went from 10 to 20 of 30 comparisons won or within 2% ([docs/benchmarks.md](docs/benchmarks.md)). Remaining on Neoverse-N2: weighted white noise at λ = 10 (25%), uniform random walk at λ = 10 (20%, where GCC's branches had helped), λ = 0.1 on noise and random walk (6–9%), and the simplex (22–23%; it compiles as well with GCC as with Clang on Apple M5, so this is the core or GCC 13). Per-architecture or per-λ tuning of `kBranchFree` was set aside (2026-10-15) as not a clean fix; the remaining gaps need a kernel-level change instead. That change (2026-10-15): long segments of Condat's algorithm are tracked by their running sum, with the bounds as a running max/min of candidates; on Apple M5 weighted white noise at λ = 10 went from 1.03–1.09 of proxTV's time to 0.43–0.45 and uniform random walk at λ = 10 from 0.99 to 0.64. On the runners: weighted white noise at λ = 10 now 0.92 on Neoverse-N2, 0.73–0.78 on Intel and 0.55–0.62 on AMD; still behind on Neoverse-N2: uniform random walk at λ = 10 (1.04–1.06, was 1.19; against proxTV's classic taut string, 0.74 against its linearized one) and λ = 0.1 (1.03–1.12). For the random walk (2026-10-16, `variants` workflow) the kernel alone is at 0.95–0.96; starting the running sum at the first sample of a segment reaches 0.82 but is 13–29% slower at λ = 0.1, and no variant tried kept both ([docs/benchmarks.md](docs/benchmarks.md)). λ = 0.1 (1.03–1.13 on Neoverse-N2, 1.01–1.12 on Intel) set aside for now (2026-10-16) after two attempts that did not hold on the runners ([docs/benchmarks.md](docs/benchmarks.md)); next time, compare variants side by side in one binary on the runners rather than on Apple M5
- [~] 1D TV on Intel x86_64: on Xeon runners (GCC 13.3) fastprox trailed proxTV by 7–24% in most settings at λ ≤ 1 while it led on AMD EPYC. The `profile` workflow showed GCC compiling the branch-free bound updates into compare-and-branch; with SSE2 `minsd`/`maxsd` (2026-10-15) it is fastest or within 2% in 12–13 of 18 settings on Intel (was 3–6) and 0.82–1.03 of proxTV's time at λ = 1. With the running-sum Condat kernel (2026-10-15) every λ ≥ 1 setting is ahead on Intel (uniform random walk at λ = 10 0.83–0.84, weighted white noise 0.73–0.78); still behind: λ = 0.1 random walk and white noise, uniform 1.06–1.09 and weighted 1.04–1.12 ([docs/benchmarks.md](docs/benchmarks.md))
- [x] CI matrix green on GitHub (private repo dvillacis/fastprox, 2026-10-14): Linux GCC/Clang with and without threads, sanitizers (ASan/UBSan and ThreadSanitizer), Valgrind, Windows MSVC, macOS arm64 and x86_64, Python on all three. The first Windows run hung on a "DLL not found" dialog; executables and DLLs now share an output directory there, and ctest has a 300 s timeout.
- [x] Wheels with cibuildwheel (manual dispatch or tags): abi3 for CPython ≥ 3.12 on manylinux_2_28 x86_64/aarch64, macOS arm64 (11.0) and x86_64 (10.15), Windows amd64; built and tested in CI. Since 0.1.0rc2 no OpenMP runtime is bundled; the repair tools (delocate, auditwheel, delvewheel) still run
- [x] Published 0.1.0rc1 to PyPI (2026-10-15): https://pypi.org/project/fastprox/0.1.0rc1/, 5 abi3 wheels and the sdist, uploaded by trusted publishing from the `v0.1.0rc1` tag after a TestPyPI dry run; the `pypi` environment accepts only `v*` tags and needs approval. The repository is public
- [x] Published 0.1.0rc2 to PyPI (2026-10-15): thread pool instead of OpenMP, which fixes the abort next to PyTorch on macOS; checked by installing from TestPyPI and PyPI next to PyTorch 2.14 and JAX 0.11
- [x] Document the kink choices per operator in `docs/api.md` (2026-10-15, "Derivatives at kinks"), with their justification: every choice is an element of the B-subdifferential (Scholtes, *Introduction to Piecewise Differentiable Equations*, Prop. 2.4.3–2.4.5 and 4.3.1; Clason–Valkonen Ex. 14.10, where soft thresholding at |x| = τ admits both 0 and 1), and x, γ and θ derivatives come from one saved structure, so they form a joint B-element and a VJP composed with a smooth loss is a Clarke subgradient. Note that the ℓ∞ prox treats |xᵢ| = τ as unclipped (the Moreau identity flips the ℓ1-ball rule), unlike soft thresholding
- [x] Final 0.1.0 once the gate passes (set `FASTPROX_VERSION_SUFFIX` to empty)
- [x] Published 0.1.0 to PyPI (2026-10-16): https://pypi.org/project/fastprox/0.1.0/, 5 abi3 wheels and the sdist, uploaded by trusted publishing from the `v0.1.0` tag after a TestPyPI dry run; checked by installing from TestPyPI and from PyPI into fresh environments and running the examples

**Gate (proposed, from spec Section 10):** all accuracy, determinism, special-value and allocation requirements pass on the full CI matrix; 1D TV, simplex and ℓ1 ball no slower than proxTV and Condat's code on one thread; overhead and scaling targets confirmed or revised; wheels install and pass tests on Linux, macOS (arm64, x86_64) and Windows.

## 0.2 — images

- [x] Anisotropic TV in 2D and 3D (2026-10-16): `fp_op_tv_aniso`, `TVAniso`, `tv_aniso`. Two active axes: Beck's accelerated dual block method (ADBPG, §12.5.4, Thm 12.19; Chambolle–Pock 2015) with exact 1D TV steps; three: Beck's fast dual proximal gradient for three blocks (Algorithm 9), which beat parallel proximal Dykstra at every accuracy. Certified gap stopping, warm starts through the workspace dual, polish to the exact solution on the identified regions, and O(n) derivatives from those regions. On Apple M5 fastest from 1e-3 relative error on in 2D against proxTV's DR2, Kolmogorov2, Yang2 and parallel Dykstra, 3.0–3.7× faster than the best proxTV solver at 1e-6 ([docs/benchmarks.md](docs/benchmarks.md))
- [ ] Anisotropic TV in 3D at high accuracy: the three-block method is 1.6–9.6× faster than the best proxTV solver up to 1e-4 relative error but slower than Yang's ADMM at 1e-6 (2.5× at λ = 0.1, 1.1× at λ = 0.5, 64³). Try ADBPG with two blocks, the contiguous axis exactly and the other two by the 2D method per plane, warm-started (proximity stacking as in Barbero–Sra §4.1, accelerated); also DR2's faster start to 1e-2 in 2D at small λ
- [x] Isotropic TV in 2D and 3D (2026-10-16): `fp_op_tv_iso`, `TVIso`, `tv_iso`. Fast gradient projection on the dual (Beck–Teboulle 2009; Beck 2017, §12.4.4) with step 1/L, L = 4 Σ wₐ² (8 in 2D, 12 in 3D), primal from the last dual iterate (12.19), gap stopping certifying ‖z − z*‖² ≤ 2·gap (Lemma 12.7), dual returned and accepted for warm starts. Per-axis weights inside the norm. Derivatives by conjugate gradients on the flat regions, with the residual reported. Against scikit-image's Chambolle projection on Apple M5: 2.6–11× faster from 1e-3 relative error on, and the only one of the two to reach 1e-6; scikit-image is 25–45% faster to 1e-2 at small γ ([docs/benchmarks.md](docs/benchmarks.md))
- [ ] Isotropic TV at high accuracy: FISTA needs about 1/√tol iterations (tens of thousands at tol 1e-12). Try adaptive restarts (O'Donoghue and Candès, Found. Comput. Math. 15, 2015), which become linearly convergent near the solution, and a preconditioner for the derivative CG from the diagonal of the reduced system
- [ ] 1D TV with ℓ2 of differences: projected Newton on the dual, gap
- [x] Nonconvex pointwise maps (2026-10-16): ℓ0, ℓq (closed forms for q = 1/2 and 2/3, Newton from |x| otherwise, monotone on the nonzero branch), SCAD and MCP for any step (a > 1 + γ, b > γ, else `FP_EPARAM`), log penalty. Verified against minimisation on a refined grid in `tests/python/test_nonconvex.py`
- [x] Tie rule (`fp_options.tie`, nonzero branch by default) and tie counts in `fp_info.ties` (2026-10-16): exact at the double-precision threshold, the same for any thread count
- [x] Derivatives for the nonconvex maps (diagonal, support-restricted, from the scalar equation; the selected branch at ties) (2026-10-16)
- [x] Cheap exact operators found in the literature review of 2026-10-15, each reusing an existing kernel or closed form with a simple Jacobian (pages are the book's printed pages):
  - [x] prox of max(x) as x − λ P_Δ(x/λ) (Beck Ex. 6.49, p. 162; Parikh and Boyd, *Proximal Algorithms*, §6.4.1, whose bisection interval is right only for λ = 1), and of the sum of the k largest entries or absolute values through the capped simplex (Beck Ex. 6.50–6.51) (2026-10-16: `Max`, `SumLargest`, on the set scaled by γλ)
  - [x] projection onto a hyperplane intersected with a box, P_Box(x − μa) with μ a root of a monotone piecewise-linear function (Beck Thm 6.27, pp. 147–148), which contains the simplex and the capped simplex; Jacobian I_F − a_F a_Fᵀ/‖a_F‖² on the free set (2026-10-16: `Hyperplane`, `HalfSpace`; Newton steps in a bracket with a median fallback, 3–9 ns per entry)
  - [x] projection onto a level set {f ≤ α}, e.g. a 1D TV ball, as prox_{γf} with γ the root of f(prox_{γf}(x)) = α (Beck Thm 6.30), using ∂z/∂γ for Newton steps and the implicit derivative (2026-10-16: `TV1DBall`, 4–15 1D TV solves per row; other level sets when a need appears)
  - [x] group Huber, (1 − λ / max(‖x‖, μ + λ)) x (Beck Ex. 6.66), and radial lifting of any scalar prox to a group one (Beck Thm 6.18), e.g. group SCAD and MCP (2026-10-16: `GroupHuber`, `GroupSCAD`, `GroupMCP`, one kernel around the scalar operator)
  - [x] half-space, hyperplane, second-order cone (Beck Ex. 6.37), −λ Σ log xᵢ as (x + √(x² + 4λ))/2 (Beck Lemma 6.5), ℓ1 over a box (Beck Ex. 6.23) (half-space and hyperplane done 2026-10-16, with an optional box; `SOC`, `LogBarrier`, `L1Box` 2026-10-16)
  - [x] sparse group lasso and tree-structured group norms by composing soft thresholding and group proxes in tree order (Bach, Jenatton, Mairal and Obozinski, *Optimization with Sparsity-Inducing Penalties*, (3.8)) (2026-10-16: `TreeL2` and `TreeL2.sparse_group`, groups in any order)
- [x] Python: `fp.lq`, `fp.tv_iso` returning `(z, info)` with `info.gap` and `info.dual`, and the other new operators (2026-10-16; `fp.tv_aniso`, `TVAniso`, `TVIso`, `L0`, `Lq`, `SCAD`, `MCP`, `LogPenalty` and their one-shot functions)
- [x] Check the citations marked "to check" in the spec (2026-10-16): Beck–Teboulle against Beck (2017), Appendix D; Xu et al., Cao–Sun–Xu, Fan–Li and Zhang against the bibliographies of Wen et al. (IEEE Access 2018) and a second citing paper, now listed as checked in docs/spec.md with volume and pages. The originals were not opened, so the kernel derives every closed form (SCAD and MCP also for γ ≠ 1) and the tests verify it against grid minimisation

**Gate:** iterative operators meet `gap ≤ tol · ½‖x‖²` with the gap recomputed independently in tests; nonconvex maps verified against a dense grid with refinement, tie flags exact at thresholds.

## 0.3 — graph TV

- [ ] Max-flow engine (Boykov–Kolmogorov style, tree reuse between levels), written from the paper (clean room)
- [ ] Graph TV by parametric max-flow, exact to a user-set precision (Chambolle–Darbon, Hochbaum)
- [ ] TV on trees and chains (Kolmogorov, Pock and Rolínek)
- [ ] VJPs for the iterative operators by conjugate gradients on the dual active set, returning the residual. Choose the active set from the strict support of the computed primal and handle ties within the tolerance explicitly: picking an index set by rule can give a matrix outside the generalised Jacobian (Outrata, Kočvara and Zowe, *Nonsmooth Approach to Optimization Problems with Equilibrium Constraints*, Ex. 7.3)
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
| 2026-10-15 | float32 1D TV may trail Condat's single-precision `TV1D_denoise` by a few percent at small λ (1–5% at λ = 0.1 on Apple M5, n = 10⁶) and still meets the 0.1 speed gate. fastprox keeps its running sums in float64 (spec Section 10: "float32 accumulates sums in float64 where a sum spans a row"), which costs a conversion per sample; single-precision sums would lose the accuracy bound on long rows | `src/ops/tv1d.cpp`, gate |
| 2026-10-16 | Uniform 1D TV on random walks at λ = 10 may be about even with proxTV's classic taut string on Neoverse-N2 (0.97 estimated from the kernel, 1.04–1.06 in `bench`, n = 10⁵) and still meets the 0.1 speed gate; it is ahead on every other machine (0.64 on Apple M5, 0.73 on AMD EPYC, 0.80–0.84 on Intel) and ahead of proxTV's other solvers there (0.74). The classic taut string makes one pass where Condat's algorithm visits each sample about 3 times in this setting; the kernel variants that win it lose 13–29% at λ = 0.1 ([docs/benchmarks.md](docs/benchmarks.md)). Beating it needs fewer rescans, an algorithmic change | gate, `src/ops/tv1d.cpp` |
| 2026-10-16 | 1D TV at λ = 0.1 may trail proxTV by up to 13% on Neoverse-N2 and Intel Xeon (1.01–1.13 of its time at n = 10⁵, GCC 13) and still meets the 0.1 speed gate; it is ahead on Apple M5 and AMD EPYC. Segments there are mostly 1 to 3 samples long and the time goes to mispredicted jumps; two attempts did not hold on the runners ([docs/benchmarks.md](docs/benchmarks.md)). To be revisited with the `variants` workflow | gate, `src/ops/tv1d.cpp` |
| 2026-10-16 | Derivatives of anisotropic TV come from the constant regions of the solution (averaging per region, γ and weight terms from the jump signs on its boundary), exact for the identified structure and O(n), instead of implicit differentiation of the dual conditions by conjugate gradients (spec Section 7). Anisotropic TV is an ℓ1 norm of differences, so its prox Jacobian has this closed form; CG stays the plan for isotropic TV | `src/ops/tv_aniso.cpp`, docs/api.md |
| 2026-10-16 | The dual of iterative image operators crosses the C API through the workspace: `fp_workspace_get_dual`, `fp_workspace_set_dual` (seed the next call) and `fp_workspace_set_warm` (every call starts from the previous dual). `fp_prox` keeps its signature (the spec's ABI rule), and the spec's sketch did not say how the dual is passed | `fastprox.h` |
| 2026-10-16 | Image operators act on the last 2 or 3 axes and ignore `fp_options.axis` (the spec says "the last two axes" for 2D TV; 3D takes the last three) | `fastprox.h` |
| 2026-10-16 | Iterative operators return an `IterInfo` (gap, iters, ties, converged, dual) in Python instead of extending `Info`, which 0.1 code may unpack as a 3-tuple; reaching `max_iter` sets `converged = False` rather than raising, since the iterate is written | Python package |
| 2026-10-16 | Isotropic TV takes one weight per axis inside the norm, Σ_p ‖(wₐ Dₐ z)_p‖₂ (the spec's Σ‖(∇z)_i‖₂ with unit weights), so anisotropic voxel spacing is expressible and θ has one entry per axis as for anisotropic TV | `fastprox.h` |
| 2026-10-16 | The tolerance and the residual of the derivative solves (spec Section 7: CG "stopped at a user tolerance", "the returned residual") cross the C API through the workspace, `fp_workspace_set_deriv_tol` and `fp_workspace_deriv_info`, since `fp_jvp` and `fp_vjp` have no options or info argument and their signatures stay fixed | `fastprox.h` |
| 2026-10-16 | For the nonconvex maps `fp_prox_conj` returns the Moreau formula y − σ prox_{g/σ}(y/σ), which is not the prox of g* when g is not convex (spec Section 4 states the identity for every operator). The SCAD and MCP conditions "concavity parameter large enough relative to γ" are a > 1 + γ and b > γ, checked against the kernel's step (1/σ after `fp_prox_conj`) | `fastprox.h`, docs/api.md |
| 2026-10-16 | Operators beyond the spec's catalogue of 20 (Section 5) join 0.2 from the literature review of 2026-10-15: the maximum, the sum of the k largest entries or absolute values, and the hyperplane and half-space projections with an optional box. Each is exact and reuses the simplex kernel or one plane solver. An empty constraint set for the axis length is `FP_EPARAM` at call time, since it depends on n | `fastprox.h`, docs/api.md |
| 2026-10-16 | More operators beyond the catalogue: the log barrier, ℓ1 over a box (any interval, not only Beck's [−α, α]), the second-order cone, and group Huber, SCAD and MCP. The cone puts t first in the row, as conic solvers do (Beck's Example 6.37 puts it last). Group SCAD and MCP are nonconvex but single-valued under their step conditions, so they take no tie rule | `fastprox.h`, docs/api.md |
| 2026-10-16 | The level-set projection of Beck's Theorem 6.30 lands as the 1D TV ball only, with θ = (α, w…). It is the one level set the roadmap named, and a generic form would need the value of f in the operator interface. Its cost is several 1D TV solves (4–15 in measurements), the price of exactness without a closed form. Tree-structured group norms take groups in any order and sort them by size, which is a valid composition order for nested or disjoint groups | `fastprox.h`, docs/api.md |
