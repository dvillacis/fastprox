# Changelog

All notable changes to fastprox. Planned work is in [ROADMAP.md](ROADMAP.md).
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Anisotropic TV of 2D and 3D images (`fp_op_tv_aniso`, `fastprox.TVAniso`, `fastprox.tv_aniso`): the prox of γ Σₐ wₐ TVₐ on the last 2 or 3 axes, batched over the others. Dual methods with exact 1D TV steps along each axis: the accelerated dual block proximal gradient method for two axes (Beck 2017, §12.5.4; Chambolle and Pock 2015) and the fast dual proximal gradient method for three (Beck, Algorithm 9). It stops on a certified duality gap (`tol`·‖x‖²/2 per image, `FP_ENOTCONV` after `max_iter`), and recovers the solution's constant regions to polish it to the exact solution when they are identified. JVP and VJP in x, γ and the axis weights come from those regions in O(n). Bit-identical for any thread count, also inside a single image.
- Image operators in the core: a third operator kind (`ImageOp`) acting on the last k axes, with its own driver, working sets in the workspace and threads inside an image when the batch is small.
- Warm starts: `fp_workspace_get_dual`, `fp_workspace_set_dual` and `fp_workspace_set_warm`; in Python, `warm_start=` and `warm=True`, and `IterInfo` (gap, iters, ties, converged, dual) for iterative operators.
- `fp_options.tol` and `max_iter` are now used (by the iterative operators), and the Python `prox` binding takes them.
- CVXPY reference fixtures for images (`tests/fixtures/reference_images.npz`, `generate.py images`), Python tests (`tests/python/test_tv_aniso.py`), C tests with an independently recomputed certificate, and fuzzing of the new operator.
- `bench/compare_tv_aniso.py`: time to reach a given accuracy against proxTV's 2D and 3D solvers.
- Isotropic TV of 2D and 3D images (`fp_op_tv_iso`, `fastprox.TVIso`, `fastprox.tv_iso`): the prox of γ Σ_p ‖(wₐ Dₐ z)_p‖₂ with forward differences on the last 2 or 3 axes, the weights scaling each axis inside the norm. FISTA on the dual (Beck and Teboulle 2009; Beck 2017, §12.4.4) with a certified gap, warm starts (a dual for another γ is projected onto the new ball) and an exact 1D path for one active axis; rescaled internally so extreme weights and steps stay finite (found by fuzzing: 4w² underflowed to 0 for a denormal weight). JVP and VJP by implicit differentiation, one unknown per flat region, solved by conjugate gradients.
- `fp_workspace_set_deriv_tol` and `fp_workspace_deriv_info`: tolerance and iteration cap of the derivative solves, and the residual and iterations of the last one; in Python, `tol` and `max_iter` on `TVIso.jvp`/`vjp` and `ctx.deriv`.
- `bench/compare_tv_iso.py`: time to reach a given accuracy against scikit-image's `denoise_tv_chambolle`. CVXPY fixtures for isotropic TV, `tests/python/test_tv_iso.py`, C tests with an independent certificate, and fuzzing.
- `src/ops/image_util.hpp`: the fiber passes, geometry and fixed-order row sums shared by the image operators.
- Nonconvex pointwise maps: ℓ0 (`fp_op_l0`, `L0`, `hard_threshold`), ℓq for 0 < q < 1 (`fp_op_lq`, `Lq`, `lq`; closed forms for q = 1/2 and 2/3, Newton from |x| otherwise), SCAD (`fp_op_scad`, `SCAD`, `scad`), MCP (`fp_op_mcp`, `MCP`, `mcp`) and the log penalty (`fp_op_log`, `LogPenalty`, `prox_log`), with derivatives in x, γ and their parameters. The SCAD and MCP maps are given for any step and need a > 1 + γ and b > γ (`FP_EPARAM` otherwise, through a new step check). Verified against minimisation on a refined grid, ties included; ℓ0, SCAD and MCP cost about 0.4–0.5 ns per entry, ℓq 12–13 ns in closed form and about 40 ns by Newton (Apple M5).
- The tie rule: `fp_options.tie` (`FP_TIE_NONZERO` by default, `FP_TIE_ZERO`) and `fp_info.ties`, the number of entries exactly on a threshold, the same for any thread count; in Python `tie="nonzero"|"zero"` and `ctx.info.ties`. Derivatives follow the branch the rule picked.
- `tests/python/test_nonconvex.py` (global minimisers on a refined grid, exact tie flags, closed forms against Newton, step conditions) and the nonconvex maps in `CASES`, the C derivative cases and the fuzzer. `CASES` entries may now be marked nonconvex, which skips the firm-nonexpansiveness property.
- Support functions through the Moreau identity: the prox of λ max (`fp_op_max`, `Max`, `prox_max`), which is min(x, τ) with Condat's simplex threshold, and of λ times the sum of the k largest entries or absolute values (`fp_op_sum_largest`, `SumLargest`, `prox_sum_largest`; Beck 2017, Examples 6.49–6.51). Both use the set scaled by γλ, so small steps do not overflow.
- Projection onto a hyperplane or half-space intersected with a box (`fp_op_hyperplane`, `Hyperplane`, `HalfSpace`, `project_hyperplane`, `project_halfspace`; Beck 2017, Theorem 6.27), with derivatives in x, b, a and the bounds. a may be a scalar or per entry, and a = 1 on [0, 1] is the capped simplex. Its multiplier comes from Newton steps on the piecewise-linear constraint function inside a bracket, with a median-of-breakpoints fallback, starting long rows from a strided sample's root: 3–9 ns per entry on Apple M5. An empty set for the row length returns `FP_EPARAM`.
- `tests/python/test_planes.py`: KKT conditions up to 20 000 entries, agreement with the bisection reference, the closed forms and the operators these generalise (simplex, ℓ∞, soft thresholding), conjugates, extreme steps, long structured rows, `a` over 90 orders of magnitude, and errors. The new operators are also in `CASES`, the C derivative cases and the fuzzer.
- The log barrier −λ Σ log zᵢ (`fp_op_log_barrier`, `LogBarrier`, `prox_log_barrier`; Beck 2017, Lemma 6.5), stable for x < 0 and huge |x| or γλ; ℓ1 over a box (`fp_op_l1_box`, `L1Box`, `prox_l1_box`; Beck, Example 6.23, here for any interval and per-entry weights and bounds); and the projection onto the second-order cone with t first (`fp_op_soc`, `SOC`, `project_soc`; Beck, Example 6.37). All with derivatives; 0.7–0.9 ns per entry on Apple M5.
- Group Huber, group SCAD and group MCP by norm composition (`fp_op_group_huber`, `fp_op_group_scad`, `fp_op_group_mcp`, `GroupHuber`, `GroupSCAD`, `GroupMCP` and their functions; Beck, Theorem 6.18 and Example 6.66): each group is scaled by the scalar prox of its norm at the step γc_g, with groups and weights as for group ℓ2 and derivatives in x, γ, the scalar parameters and the group weights. The kernel wraps the scalar operator, so its formulas and kink rules are shared. Group SCAD and MCP check a > 1 + γ max c and b > γ max c at call time. Cost is that of group ℓ2 (about 3.5 ns per entry with group ids, 1 ns per entry on rows).
- `tests/python/test_radial.py`: stationarity of the log barrier over 9 orders of magnitude, ℓ1 over a box and the group penalties against minimisation on refined grids, the cone's KKT conditions and polar conjugate, Beck's closed form for group Huber, kink rules, NaN and infinity handling and errors. The new operators are also in `CASES`, the C derivative cases and the fuzzer.

## [0.1.0] — 2026-10-16

First release, on PyPI (`pip install fastprox`). Changes since 0.1.0rc2:

### Added

- `variants` workflow and `bench/tv_variants.cpp`: variants of the 1D TV kernel side by side against proxTV (classic, linearized, hybrid and weighted taut string), compiled into one binary and timed on the same inputs with the order rotated, on the arm64 runner and on x86_64 jobs. It runs on pushes to `variants/*` branches or by hand. `condat` takes the length of its first, paper-form phase as a template parameter (`kShort` by default) so variants need no copy of the kernel.

- `profile` workflow (manual dispatch): several identical jobs on GitHub's x86_64 runner, each labelled with the CPU it landed on. Each job profiles single-thread 1D TV against proxTV with `bench/profile_kernels.cpp`, which records hardware counters where the VM exposes them and rotates the solver order so that none benefits from branch history. It also summarises GCC's assembly of the kernels (conditional jumps against selects) with `bench/asm_summary.py`.

### Documentation

- `examples/`: basic usage, 1D TV denoising, proximal gradient (lasso and batched simplex least squares), derivatives checked against finite differences, and bilevel learning of TV weights through the VJP. `tests/python/test_examples.py` runs them all; matplotlib for the plot is in the `examples` dependency group.
- `docs/api.md` lists, for every operator, which Jacobian the derivatives use at kinks and ties. It explains why each choice is an element of the B-subdifferential, and that the x, γ and parameter derivatives together form one. The ℓ∞ prox and conjugate proxes reverse the thresholding rule through the Moreau identity.

### Changed

- `bench/compare_external.py` rotates the order of the solvers with every repetition (fastprox's default solver always ran first, on an input just written, and its Condat path right after it), and quick mode takes the median of 7 repetitions instead of 3.
- 1D TV: the per-call finiteness and range pass runs in 8 independent lanes with `fminnm`/`fmaxnm` on arm64 (compare-selects elsewhere) instead of one dependent chain: 0.25 instead of 0.48 ns per sample with Clang and 0.89 with GCC on Apple M5 (float64, n = 10⁶). Weighted rows on white noise at λ = 10 went from 5.52 to 5.29 ns per sample, and float32 rows at λ = 0.1 are now 1–5% behind Condat's single-precision code (were 6–12%).
- 1D TV: the branch-free bound updates in short segments of Condat's algorithm are written as min/max of the excess instead of selects on a comparison, which GCC compiled back into branches. Same arithmetic up to the sign of zero. With GCC 16 on Apple M5, 4–12% faster at λ = 0.1 and 5–27% faster at λ = 1 (uniform and weighted), within 3% of the Clang build at λ ≤ 1; at λ = 10, uniform rows lose 1–19% under GCC (random walk most), where the branches it emitted happened to predict well, and weighted rows gain up to 10%. Clang's code is unchanged.
- Simplex and ℓ1 ball: the first pass of Condat's method skips blocks of 8 entries whose maximum is at or below the running pivot (the pivot can only change on an entry above it, so results are bit-identical). The ℓ1 norm that decides whether x is inside the ball is summed in 8 lanes instead of one dependent chain. On Apple M5 at n = 10⁶: simplex projection 0.50 instead of 0.60 ns per sample on Gaussian input, now faster than Condat's code (0.56); unweighted ℓ1 ball 0.64 instead of 1.08, weighted 1.08 instead of 1.37 (Clang) and 1.05 instead of 1.68 (GCC). The ℓ∞ prox shares the ℓ1-ball kernel.

- x86-64: the min/max helpers behind 1D TV's bound updates and range pass and the simplex block maximum use SSE2 `minsd`/`maxsd` instead of a compare-select, which GCC compiled into compare-and-branch. Same values. On GitHub's x86_64 runners (GCC 13.3, n = 10⁵), 1D TV went from fastest or within 2% of proxTV in 3–6 of 18 settings to 12–13 on Intel Xeons (8573C, 6973P-C) and from 14–17 to 17–18 on AMD EPYC; at λ = 1 fastprox now takes 0.82–1.03 of proxTV's time on Intel (was 1.01–1.16). Uniform random walk at λ = 10 lost 5–11% relative to proxTV, where the branches predicted well.

- 1D TV: after the first 3 samples of a segment, Condat's algorithm now tracks the segment by its running sum. The lower bound is the running maximum of the candidates (s − λₖ)/m and the upper bound the running minimum of (s + λₖ)/m, so the clamps are min/max and the only value carried between steps is the sum, instead of a dependency chain through every clamp (same algorithm and structure; values differ at rounding level). On Apple M5 (n = 10⁵, single thread) the kernel is 1.3–2.9× faster at λ ≥ 1 and unchanged at λ = 0.1. Weighted white noise at λ = 10, the one setting where fastprox trailed proxTV on every machine, went from 1.03–1.09 of proxTV's time to 0.43–0.45 (Clang) and from 1.14 to 0.60 (GCC 16), and uniform random walk at λ = 10 from 0.99 to 0.64. On GitHub's runners (GCC 13.3): weighted white noise at λ = 10 from 1.12–1.14 to 0.73–0.78 on Intel Xeons, from 1.00–1.08 to 0.55–0.62 on AMD EPYC and from 1.22 to 0.92 on Neoverse-N2; uniform random walk at λ = 10 from 1.05–1.09 to 0.83–0.84 on Intel and from 1.19 to 1.06 on Neoverse-N2. At λ = 0.1 weighted rows lost about 3% on Intel. Long segments no longer use branches for the clamps, so `kBranchFree` (8) is replaced by `kShort` (3).

### Fixed

- Weighted ℓ1-ball projection with weights spanning many orders of magnitude could return outputs far larger than the input (z = [5.5e7, 2.4e70] for x = [11.0, 2.0], weights [1.8e-308, 7.7e-246] and radius 1e-300; found by fuzzing). Removing an entry within rounding of the pivot that dominated the running sums cancelled them. The threshold is now kept at least at the largest breakpoint left out of the support, as it is in exact arithmetic. The same guard is in the unweighted kernel.

## [0.1.0rc2] — 2026-10-15

### Fixed

- The macOS wheel of 0.1.0rc1 aborted the process (`OMP: Error #15`) when PyTorch, or any package bundling a recent LLVM OpenMP runtime, was used in the same process: each wheel carried its own libomp, and the copies refuse to coexist.

### Changed

- Rows run on fastprox's own thread pool (`src/core/parallel.cpp`) instead of OpenMP. No OpenMP runtime is linked or bundled on any platform; the wheels depend only on the system and NumPy, and macOS wheels now support macOS 11.0 (arm64) and 10.15 (x86_64). Same results bit for bit; dynamic scheduling as before. Measured against the OpenMP build on Apple M5 in one process: equal on one thread, 1–8% slower at 2–4 threads on small batches, 8–31% faster at 8–15 threads.
- `threads = 0` means `fp_default_threads()`: `FASTPROX_NUM_THREADS`, else the first number in `OMP_NUM_THREADS`, else the CPUs the process may use (CPU affinity on Linux). Concurrent callers do not oversubscribe: one uses the pool, the others run serially. A child after `fork()` builds its own pool.
- `fp_has_openmp()` and `fastprox.has_openmp()` are replaced by `fp_default_threads()` and `fastprox.default_threads()`; the CMake option `FASTPROX_OPENMP` is now `FASTPROX_THREADS`.
- CI: a ThreadSanitizer job; AddressSanitizer and Valgrind now run with threads.

## [0.1.0rc1] — 2026-10-15

First release candidate, on PyPI (`pip install --pre fastprox`).

### Added

- Plain-C API (`include/fastprox.h`) over a C++17 core: opaque operator and workspace handles, `fp_prox`, `fp_prox_conj`, `fp_jvp`, `fp_vjp`, error codes with every exception caught at the boundary.
- Operators: soft thresholding (weighted), elastic net, Huber, group ℓ2 and ℓ2 norm, box, nonnegative orthant, ℓ2 ball, simplex, weighted ℓ1 ball, ℓ∞ norm, isotonic regression, and 1D TV (Condat's direct algorithm and taut string, edge weights).
- JVP and VJP in the input, the step and the operator parameters for every operator.
- Batching over any axis of strided n-d arrays, OpenMP parallelism with bit-identical results for any thread count, and allocation-free calls with a workspace.
- Python package built with nanobind (abi3, CPython ≥ 3.12): one-shot functions and operator classes with `prox`, `prox_conj`, `jvp` and `vjp`.
- C/C++ and Python test suites, a benchmark script, and a CI workflow.
- Release pipeline: sdist job, trusted publishing to TestPyPI (manual dispatch with `publish`) and to PyPI (tags `v*`, checked against the package version); PyPI metadata and a README for the project page; pre-release tags through `FASTPROX_VERSION_SUFFIX` in `CMakeLists.txt`; Python tests also run on CPython 3.14.
- Fuzz harness for the whole API (libFuzzer target and standalone driver) with a nightly workflow; CVXPY + Clarabel reference fixtures; benchmarks against proxTV and Condat's code (`bench/compare_external.py`, `docs/benchmarks.md`); API reference (`docs/api.md`); Valgrind CI job.

### Changed

- 1D TV: the Condat path falls back to the taut string after 3n + 1024 rescanned steps (was 16n + 1024 steps in total), which makes adversarial inputs such as ramps about 3.5× cheaper without affecting typical signals.
- 1D TV: the bound updates of Condat's algorithm are branch-free selects with a table of reciprocals, 25–35% faster on noisy signals (same arithmetic, so the same results).
- 1D TV: weighted problems now use Condat's direct algorithm, generalised to per-edge weights (radii precomputed per piece), with the same budgeted fallback to the taut string; `FP_TV1D_CONDAT` accepts weights. Weighted rows went from 1.4–2.8× slower than proxTV's weighted taut string to faster in most settings.
- 1D TV: radii that can never be active (above the dual bound j · (max x − min x)) are clamped, and when no edge can be active the result is the constant mean. Extreme weights or steps no longer cost accuracy: the residual stays within 10 n ε ‖x‖∞.
- 1D TV: the taut string stores its two chains as segments (length, rise, cached slope) in coordinates local to the current knot, an idea taken from proxTV's classic taut string. It is 17–26% faster and needs no compensated sums: the cumulative sum of the signal is never formed.
- Unweighted simplex and ℓ1-ball projections keep candidate values instead of indices (about 30% faster). The simplex also tests finiteness inside its pivot pass instead of a separate pass (a further 19%), and the ℓ1 ball and ℓ∞ prox reuse their ℓ1 norm as the test.
- 1D TV: bound updates are branch-free only in the first 8 samples of a segment; weighted rows read their radii directly when none can need clamping; operator construction makes one pass over the weights.
- Workspace scratch is no longer zero-filled, and the finiteness check is a branch-free pass: one-shot calls on large rows are noticeably faster.
- The version is defined once, in `CMakeLists.txt`; the Python package reads it from there.

### Fixed

- Weighted ℓ1-ball projection returned infinities for tiny weights (around 1e-279) because sums of squared weights underflowed (found by fuzzing). Weights are now normalised by their maximum, and ratios below 1e-100 are rejected.
- `fp_prox_conj` with σ = 0 on arrays with negative strides computed byte offsets in unsigned arithmetic (undefined behaviour, found by fuzzing).
- Static Windows builds marked API symbols `dllexport`.
- The taut string kept cumulative sums in plain doubles and missed the 10 n ε ‖x‖∞ accuracy bound on long rows; its chains now use local coordinates.
- A test assumed weighted isotonic regression is firmly nonexpansive in the Euclidean norm; it is so in the weighted norm, which the test now uses.
