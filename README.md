# fastprox

Fast, exact proximal operators and projections, with derivatives.

fastprox computes prox<sub>γg</sub>(x) = argmin<sub>z</sub> g(z) + ‖z − x‖² / (2γ)
for over thirty functions g used in optimisation, signal and image processing
and statistics, along any axis of a batch of arrays. It also differentiates
them: Jacobian-vector and vector-Jacobian products in the input, the step γ
and the function's own parameters. The core is C++17 behind a plain-C API,
with Python bindings, PyTorch and JAX wrappers, and image TV that runs on GPUs.

## Why fastprox

- **Exact where an exact algorithm exists, certified where it doesn't.** 1D
  TV, the simplex and ℓ1 ball, isotonic regression, graph TV and tree TV are
  solved exactly. The image TV solvers stop on a duality gap, which bounds
  the error of the result, and anisotropic TV is then polished to the exact
  solution.
- **Derivatives without unrolling.** Every operator has a JVP and a VJP,
  exact or by implicit differentiation on the solution's structure, so the
  backward pass of an iterative solver costs one linear solve and constant
  memory, not autograd through its iterations. At kinks, where the prox is
  not differentiable, the library can tell you which points sit there and
  differentiate either side.
- **Fast.** Measured in [docs/benchmarks.md](https://github.com/dvillacis/fastprox/blob/master/docs/benchmarks.md):
  - 1D TV and simplex projection run as fast as proxTV and Condat's
    reference C code, or faster (1D TV in about 6–15 ns per sample).
  - Anisotropic TV reaches an error of 1e-6 about 3× faster than the best
    proxTV solver on 256² and 512² images.
  - Isotropic TV against DeepInverse's `TVDenoiser` on an NVIDIA T4: 7.5×
    faster to an error of 1e-2, and at the same gradient accuracy a backward
    pass of 281 ms with 6 MiB kept, against 1002 ms and 4 GiB through 400
    unrolled iterations.
  - Graph TV on large grids is up to 2.7× faster in 0.4, on a new max-flow
    engine (EIBFS).
- **Batched and deterministic.** float32 or float64, along any axis, with
  rows processed in parallel on fastprox's own thread pool (no OpenMP runtime
  to clash with PyTorch's). Results are bit-identical for any number of
  threads.
- **Light and permissive.** The wheels depend only on NumPy. MIT licensed,
  and every kernel is written from the published papers.

## Install

```sh
pip install fastprox
```

The 1.0 release candidate, with the spectral operators and whose C ABI holds
for every 1.x: `pip install --pre fastprox`. A tagged release can also be
installed from GitHub,
`pip install "fastprox @ git+https://github.com/dvillacis/fastprox@v1.0.0rc2"`,
and unreleased work from a commit of master (`@<commit>`); CI builds and tests
every push to master. Build from a commit, never from a working tree.

Wheels are abi3 for CPython 3.12 and later on Linux (x86_64, aarch64), macOS
(arm64 from 11.0, x86_64 from 10.15) and Windows (amd64). The PyTorch and JAX
wrappers use PyTorch and JAX when they are installed, and never require them.
Free-threaded CPython builds (3.13t, 3.14t) are not supported yet: they cannot
load abi3 wheels, and built from source the extension re-enables the GIL at
import, with a RuntimeWarning.

## Quick start

**One-shot functions** apply an operator along the last axis by default:

```python
import numpy as np
import fastprox as fp

x = np.random.default_rng(0).normal(size=(64, 1000))

z = fp.tv1d(x, gamma=0.5)                  # 1D total variation, row by row
p = fp.project_simplex(x, radius=1.0)      # each row onto the simplex
s = fp.soft_threshold(x, lam=0.1)
```

**Operators** keep what the derivatives need:

```python
op = fp.TV1D(weights=0.8)
z, ctx = op.prox(x, 0.5, keep=True)
dz = op.jvp(ctx, dx=np.ones_like(x))       # Jacobian-vector product
gx, grads = op.vjp(ctx, g=z - x)           # vector-Jacobian product
print(grads.gamma, grads.weights)          # gradients in the step and the weights
```

**Images** are batched over the leading axes; the result comes with its
certificate:

```python
images = np.random.default_rng(1).normal(size=(8, 256, 256))
z, info = fp.tv_iso(images, gamma=0.1, tol=1e-6)    # isotropic TV of each image
print(info.gap, info.iters)                         # duality gap reached, iterations
```

**PyTorch and JAX.** Image TV runs on the tensor's own device (CUDA, MPS or
the CPU), and gradients come from implicit differentiation:

```python
import torch
import fastprox.torch as fpt

x = noisy.cuda().requires_grad_()               # (batch, channels, H, W)
gamma = torch.tensor(0.1, device="cuda", requires_grad=True)
z = fpt.tv_iso(x, gamma, tol=1e-6)              # TV over the last two axes
((z - clean.cuda()) ** 2).mean().backward()     # x.grad and gamma.grad

w = torch.full((999,), 0.8, dtype=torch.float64, requires_grad=True)
y = fpt.prox(fp.TV1D, torch.randn(64, 1000, dtype=torch.float64), 0.5, weights=w)
```

`fastprox.jax` has the same functions, under `jit`, `grad` and `vmap`.

## Operators

| Family | Operators |
| --- | --- |
| Elementwise | soft thresholding (weighted), elastic net, Huber, box, nonnegative orthant, log barrier, ℓ1 over a box |
| Nonconvex | ℓ0, ℓq (0 < q < 1), SCAD and MCP (weighted), log penalty, with an explicit rule at ties |
| Groups and trees | group ℓ2, group Huber, group SCAD and MCP, tree-structured group norms, sparse group lasso |
| Projections | ℓ2 ball, simplex, weighted ℓ1 ball, hyperplane or half-space intersected with a box, second-order cone; also the ℓ∞ norm, max and the sum of the k largest entries |
| Signals | isotonic regression, the sorted ℓ1 norm (OWL, SLOPE), 1D TV with per-edge weights, its ℓ2-of-differences variant, projection onto a 1D TV ball |
| Images | anisotropic and isotropic TV in 2D and 3D, with per-axis and per-pixel weights and warm starts; on GPUs through PyTorch and JAX |
| Graphs | TV on any graph, exact, by parametric max-flow; TV on trees and paths by dynamic programming |

Every operator also has its conjugate prox, through the Moreau identity. The
operators with kinks report which entries sit at one, and differentiate
either side of it on request: the thresholding and nonconvex maps, the box
and orthant, the simplex and ℓ1 ball, group ℓ2, and 1D, image, graph and
tree TV (for graph and tree TV the kinks are edges; isotropic TV reports
them but has one side only).

## Documentation

The documentation site is at <https://dvillacis.github.io/fastprox/>; its
pages are also in the repository:

- [API reference](https://github.com/dvillacis/fastprox/blob/master/docs/api.md), for C and Python
- [Examples](https://github.com/dvillacis/fastprox/tree/master/examples): 1D and image denoising, proximal gradient, sparse
  recovery, group sparsity, the weighted graphical lasso, derivatives,
  bilevel learning, reweighted TV, PyTorch and JAX
- [Benchmarks](https://github.com/dvillacis/fastprox/blob/master/docs/benchmarks.md) against proxTV, Condat's C code and
  DeepInverse
- [Using fastprox in place of DeepInverse's TV denoisers](https://github.com/dvillacis/fastprox/blob/master/docs/deepinv.md)
- [Design specification](https://github.com/dvillacis/fastprox/blob/master/docs/spec.md), [roadmap](https://github.com/dvillacis/fastprox/blob/master/ROADMAP.md) and
  [changelog](https://github.com/dvillacis/fastprox/blob/master/CHANGELOG.md)

## C library

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build && cmake --install build --prefix /usr/local
```

Then `find_package(fastprox)` and link `fastprox::fastprox`, or use
`pkg-config --cflags --libs fastprox`. Everything is in one header,
[`include/fastprox.h`](https://github.com/dvillacis/fastprox/blob/master/include/fastprox.h):

```c
#include <fastprox.h>

double x[1000], z[1000];                      /* filled by the caller */
const int64_t shape[1] = {1000};
fp_array xa = {x, FP_F64, 1, shape, NULL, FP_DEVICE_CPU};  /* NULL strides: C order */
fp_array za = {z, FP_F64, 1, shape, NULL, FP_DEVICE_CPU};

fp_op *op;
fp_status s = fp_op_tv1d(NULL, FP_TV1D_AUTO, &op);        /* unit edge weights */
if (s == FP_OK) {
    s = fp_prox(op, 0.5, &xa, &za, NULL, NULL, NULL);    /* default options */
    fp_op_free(op);
}
if (s != FP_OK) fprintf(stderr, "fastprox: %s\n", fp_strerror(s));
```

## License

MIT
