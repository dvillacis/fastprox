# Changelog

All notable changes to fastprox. Planned work is in [ROADMAP.md](ROADMAP.md).
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Documentation

- `examples/`: basic usage, 1D TV denoising, proximal gradient (lasso and batched simplex least squares), derivatives checked against finite differences, and bilevel learning of TV weights through the VJP. `tests/python/test_examples.py` runs them all; matplotlib for the plot is in the `examples` dependency group.
- `docs/api.md` lists, for every operator, which Jacobian the derivatives use at kinks and ties. It explains why each choice is an element of the B-subdifferential, and that the x, γ and parameter derivatives together form one. The ℓ∞ prox and conjugate proxes reverse the thresholding rule through the Moreau identity.

### Changed

- 1D TV: the per-call finiteness and range pass runs in 8 independent lanes with `fminnm`/`fmaxnm` on arm64 (compare-selects elsewhere) instead of one dependent chain: 0.25 instead of 0.48 ns per sample with Clang and 0.89 with GCC on Apple M5 (float64, n = 10⁶). Weighted rows on white noise at λ = 10 went from 5.52 to 5.29 ns per sample, and float32 rows at λ = 0.1 are now 1–5% behind Condat's single-precision code (were 6–12%).
- 1D TV: the branch-free bound updates in short segments of Condat's algorithm are written as min/max of the excess instead of selects on a comparison, which GCC compiled back into branches. Same arithmetic up to the sign of zero. With GCC 16 on Apple M5, 4–12% faster at λ = 0.1 and 5–27% faster at λ = 1 (uniform and weighted), within 3% of the Clang build at λ ≤ 1; at λ = 10, uniform rows lose 1–19% under GCC (random walk most), where the branches it emitted happened to predict well, and weighted rows gain up to 10%. Clang's code is unchanged.
- Simplex and ℓ1 ball: the first pass of Condat's method skips blocks of 8 entries whose maximum is at or below the running pivot (the pivot can only change on an entry above it, so results are bit-identical). The ℓ1 norm that decides whether x is inside the ball is summed in 8 lanes instead of one dependent chain. On Apple M5 at n = 10⁶: simplex projection 0.50 instead of 0.60 ns per sample on Gaussian input, now faster than Condat's code (0.56); unweighted ℓ1 ball 0.64 instead of 1.08, weighted 1.08 instead of 1.37 (Clang) and 1.05 instead of 1.68 (GCC). The ℓ∞ prox shares the ℓ1-ball kernel.

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
