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

- 1D TV: the Condat path falls back to the taut string after 4n + 1024 steps (was 16n + 1024), which halves the cost on adversarial inputs such as ramps without affecting typical signals.
- Unweighted simplex and ℓ1-ball projections keep candidate values instead of indices (about 30% faster).
- Workspace scratch is no longer zero-filled, and the finiteness check is a branch-free pass: one-shot calls on large rows are noticeably faster.
- The version is defined once, in `CMakeLists.txt`; the Python package reads it from there.

### Fixed

- `fp_prox_conj` with σ = 0 on arrays with negative strides computed byte offsets in unsigned arithmetic (undefined behaviour, found by fuzzing).
- Static Windows builds marked API symbols `dllexport`.
- The taut string kept cumulative sums in plain doubles and missed the 10 n ε ‖x‖∞ accuracy bound on long rows; it now uses compensated sums.
