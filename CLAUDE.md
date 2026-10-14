# fastprox

Exact proximal operators with JVP/VJP. C++17 core behind the plain-C header
`include/fastprox.h`; Python bindings via nanobind. The design spec is
`docs/spec.md` — read it before changing the API or adding operators.

## Commands

- C library and tests: `cmake -S . -B build/c -G Ninja && cmake --build build/c && ./build/c/tests/c/test_core && ./build/c/tests/c/test_header`
- Sanitizers: configure a second tree with `-DFASTPROX_SANITIZE=ON -DFASTPROX_OPENMP=OFF -DCMAKE_BUILD_TYPE=Debug`
- Python (always through uv): `uv sync`, `uv run pytest -q`. uv rebuilds the extension when C++ sources change (`tool.uv.cache-keys`).
- Benchmarks: `uv run python bench/bench_python.py`

## Layout

- `src/core/op.hpp` — `fp_op` base class; `ElemOp<D>` (elementwise, needs `prox1`/`deriv1`) and `RowOp<D>` (row kernels `prox`/`jvp`/`vjp`) CRTP helpers.
- `src/core/driver.hpp` — batching over strided n-d arrays, OpenMP, saved structure; `src/core/layout.hpp` — row and elementwise iteration plans.
- `src/ops/*.cpp` — one file per operator family, each with a `make_*` factory declared in `src/ops/ops.hpp`.
- `src/api/api.cpp` — the only `extern "C"` code: validation, dtype dispatch, try/catch.
- `bindings/python/src/module.cpp` calls only the C API; `bindings/python/fastprox/__init__.py` is the user-facing package.

## Rules

- Clean room: never read GPL, CeCILL or unlicensed code (Condat's C files, Kolmogorov's maxflow/TVTree, Chambolle's graphTV, PyMaxflow, gco). Implement from papers and cite them in the kernel's header comment. proxTV (BSD) may be consulted with attribution.
- No `-ffast-math`. Results must be bit-identical for any thread count: rows are independent, and cross-row sums (vjp parameter gradients) use fixed blocks from `reduction_blocks`, never thread-dependent partitions.
- No allocation inside `fp_prox`/`fp_jvp`/`fp_vjp` when a workspace is given; size everything in `build_workspace` (tested by an allocation counter in `tests/c/test_core.cpp`).
- Adding an operator: kernel + derivatives in `src/ops/`, factory in `ops.hpp`, constructor in `fastprox.h` and `api.cpp`, nanobind wrapper, Python class with `_layout`, and an entry in `CASES` in `tests/python/test_operators.py` (which runs reference, finite-difference, adjointness, nonexpansiveness, threading and stride tests on it) plus `derivative_cases` in `tests/c/test_core.cpp`.
