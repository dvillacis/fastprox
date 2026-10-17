# fastprox

Exact proximal operators and projections with derivatives. A C++17 core
behind a plain-C API, with Python bindings.

- **Exact or certified.** Direct algorithms (Condat's 1D TV and simplex
  projection, the taut string, pool-adjacent-violators, closed forms) where
  they exist. The image TV solvers stop on a certified duality gap, and
  anisotropic TV is polished to the exact solution once its regions are
  found.
- **Differentiable.** Jacobian-vector and vector-Jacobian products in the
  input, the step and the operator's parameters, for bilevel learning and
  implicit differentiation.
- **Batched.** Applied along any axis of a float32 or float64 array, with rows
  processed in parallel on fastprox's own threads (no OpenMP runtime to clash
  with PyTorch or others) and results bit-identical for any number of threads.
- **Permissive.** MIT licensed; every kernel is written from the papers.

## Install

```sh
pip install fastprox
```

Wheels are abi3 for CPython 3.12 and later on Linux (x86_64, aarch64), macOS
(arm64 from 11.0, x86_64 from 10.15) and Windows (amd64), and depend on
nothing but NumPy.

## Use

```python
import numpy as np
import fastprox as fp

x = np.random.default_rng(0).normal(size=(64, 1000))

z = fp.tv1d(x, gamma=0.5)                 # 1D total variation along the last axis
p = fp.project_simplex(x, radius=1.0)     # each row onto the simplex
s = fp.soft_threshold(x, lam=0.1)

# Operators with derivatives
op = fp.TV1D(weights=0.8)
z, ctx = op.prox(x, 0.5, keep=True)
dz = op.jvp(ctx, dx=np.ones_like(x))      # Jacobian-vector product
gx, grads = op.vjp(ctx, g=x)              # vector-Jacobian product
print(grads.gamma, grads.weights)         # gradients in the step and the weights
```

## Operators

- **Elementwise:** soft thresholding (weighted), elastic net, Huber, box,
  nonnegative orthant, log barrier, ℓ1 over a box.
- **Nonconvex:** ℓ0, ℓq (0 < q < 1), SCAD and MCP (weighted), log penalty, with an explicit tie rule.
- **Groups and trees:** group ℓ2, group Huber, group SCAD and MCP, tree-structured group norms and the sparse group lasso.
- **Projections:** ℓ2 ball, simplex, weighted ℓ1 ball, hyperplane or
  half-space intersected with a box, second-order cone. Also the ℓ∞ norm and
  the support functions max and the sum of the k largest entries.
- **Signals:** isotonic regression, 1D total variation with per-edge weights, its ℓ2-of-differences
  variant, and the projection onto a 1D TV ball.
- **Images:** anisotropic and isotropic total variation in 2D and 3D, with warm starts (since 0.2).
- **Graphs:** total variation on any graph, exact, by parametric max-flow (since 0.3).

Each one has a conjugate prox through the Moreau identity. The elementwise maps with kinks report
which entries sit at a breakpoint, and their derivatives can be taken on either side of it.

## Documentation

- [API reference](https://github.com/dvillacis/fastprox/blob/master/docs/api.md) (C and Python)
- [Examples](https://github.com/dvillacis/fastprox/tree/master/examples): denoising, image denoising, proximal gradient, sparse recovery, group sparsity, the weighted graphical lasso, derivatives, bilevel learning
- [Benchmarks](https://github.com/dvillacis/fastprox/blob/master/docs/benchmarks.md) against proxTV and Condat's C code
- [Design specification](https://github.com/dvillacis/fastprox/blob/master/docs/spec.md)
- [Roadmap](https://github.com/dvillacis/fastprox/blob/master/ROADMAP.md) and [changelog](https://github.com/dvillacis/fastprox/blob/master/CHANGELOG.md)

## C library

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build && cmake --install build --prefix /usr/local
```

Then `find_package(fastprox)` and link `fastprox::fastprox`, or use
`pkg-config --cflags --libs fastprox`. The header is
[`include/fastprox.h`](https://github.com/dvillacis/fastprox/blob/master/include/fastprox.h).

## License

MIT
