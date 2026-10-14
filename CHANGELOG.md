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
