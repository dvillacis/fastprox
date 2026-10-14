# Changelog

All notable changes to fastprox. Planned work is in [ROADMAP.md](ROADMAP.md).
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — 0.1.0

### Added

- Plain-C API (`include/fastprox.h`) over a C++17 core: opaque operator and workspace handles, `fp_prox`, `fp_prox_conj`, `fp_jvp`, `fp_vjp`, error codes with every exception caught at the boundary.
- Operators: soft thresholding (weighted), elastic net, Huber, group ℓ2 and ℓ2 norm, box, nonnegative orthant, ℓ2 ball, simplex, weighted ℓ1 ball, ℓ∞ norm, isotonic regression, and 1D TV (Condat's direct algorithm and taut string, edge weights).
- JVP and VJP in the input, the step and the operator parameters for every operator.
- Batching over any axis of strided n-d arrays, OpenMP parallelism with bit-identical results for any thread count, and allocation-free calls with a workspace.
- Python package built with nanobind (abi3, CPython ≥ 3.12): one-shot functions and operator classes with `prox`, `prox_conj`, `jvp` and `vjp`.
- C/C++ and Python test suites, a benchmark script, and a CI workflow.
- Fuzz harness for the whole API (libFuzzer target and standalone driver) with a nightly workflow; CVXPY + Clarabel reference fixtures; benchmarks against proxTV and Condat's code (`bench/compare_external.py`, `docs/benchmarks.md`); API reference (`docs/api.md`); Valgrind CI job.

### Changed

- 1D TV: the Condat path falls back to the taut string after 3n + 1024 rescanned steps (was 16n + 1024 steps in total), which makes adversarial inputs such as ramps about 3.5× cheaper without affecting typical signals.
- 1D TV: the bound updates of Condat's algorithm are branch-free selects with a table of reciprocals, 25–35% faster on noisy signals (same arithmetic, so the same results).
- 1D TV: weighted problems now use Condat's direct algorithm, generalised to per-edge weights (radii precomputed per piece), with the same budgeted fallback to the taut string; `FP_TV1D_CONDAT` accepts weights. Weighted rows went from 1.4–2.8× slower than proxTV's weighted taut string to faster in most settings.
- 1D TV: radii that can never be active (above the dual bound j · (max x − min x)) are clamped, and when no edge can be active the result is the constant mean. Extreme weights or steps no longer cost accuracy: the residual stays within 10 n ε ‖x‖∞.
- 1D TV: the taut string stores its two chains as segments (length, rise, cached slope) in coordinates local to the current knot, an idea taken from proxTV's classic taut string. It is 17–26% faster and needs no compensated sums: the cumulative sum of the signal is never formed.
- Unweighted simplex and ℓ1-ball projections keep candidate values instead of indices (about 30% faster).
- Workspace scratch is no longer zero-filled, and the finiteness check is a branch-free pass: one-shot calls on large rows are noticeably faster.
- The version is defined once, in `CMakeLists.txt`; the Python package reads it from there.

### Fixed

- `fp_prox_conj` with σ = 0 on arrays with negative strides computed byte offsets in unsigned arithmetic (undefined behaviour, found by fuzzing).
- Static Windows builds marked API symbols `dllexport`.
- The taut string kept cumulative sums in plain doubles and missed the 10 n ε ‖x‖∞ accuracy bound on long rows; its chains now use local coordinates.
- A test assumed weighted isotonic regression is firmly nonexpansive in the Euclidean norm; it is so in the weighted norm, which the test now uses.
