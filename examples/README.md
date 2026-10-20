# Examples

Short scripts that use the Python package. Each one prints what it checks
and runs in about a second. From the repository root:

```sh
uv run python examples/basic_usage.py
uv run --group examples python examples/tv_denoising.py   # with a plot (matplotlib)
```

| Script | Shows |
| --- | --- |
| [basic_usage.py](basic_usage.py) | One-shot functions, operator objects, `axis`, dtypes, `out=`, the conjugate prox, threads |
| [tv_denoising.py](tv_denoising.py) | 1D TV denoising for a range of γ, and a batch of signals in one call |
| [proximal_gradient.py](proximal_gradient.py) | FISTA for the lasso, and a batch of simplex-constrained least-squares problems |
| [derivatives.py](derivatives.py) | JVP and VJP in x, γ and the parameters, checked against finite differences and for adjointness |
| [bilevel_learning.py](bilevel_learning.py) | Learning γ, then one TV weight per edge, from clean/noisy pairs by gradients through the prox |
| [image_denoising.py](image_denoising.py) | Isotropic and anisotropic TV of images (0.2): gap certificates, warm starts, a batch, and learning γ through the VJP of an iterative operator |
| [group_sparsity.py](group_sparsity.py) | Lasso, group lasso, sparse group lasso, group MCP and SCAD (0.2) on grouped regression with λ chosen on a validation set, and hierarchical sparsity with tree-structured groups |
| [weighted_glasso.py](weighted_glasso.py) | Weighted graphical lasso by ADMM (0.2): the −log det step as `eigh` plus `LogBarrier`, the penalty step as weighted `Soft` or weighted `MCP` on the packed upper triangle; the graphical lasso against adaptive weights and weighted MCP, with a KKT check |
| [autodiff_frameworks.py](autodiff_frameworks.py) | The operators in PyTorch and JAX (0.3): one TV weight per edge as an `nn.Parameter` trained by Adam, and γ of isotropic TV learned with `jax.jit(jax.value_and_grad(...))` (needs `--group interop`) |
| [deepinv_tv.py](deepinv_tv.py) | `fastprox.torch.tv_iso`/`tv_aniso` in place of DeepInverse's `TVDenoiser`/`TVL1Denoiser` (0.3): the same results, warm starts in an inpainting loop, gradients in the image and a per-sample step (needs `--group interop --with deepinv`; about 6 s) |
| [pyproximal_tv.py](pyproximal_tv.py) | fastprox operators in PyProximal's solvers (since 0.5): isotropic TV deblurring by accelerated proximal gradient, with `ProxOp(fp.TVIso(), dims, warm=True)` against `pyproximal.TV` (needs `--group interop`) |
| [reweighted_tv.py](reweighted_tv.py) | Nonconvex TV^q denoising by iteratively reweighted ℓ1 (Ochs et al. 2015; 0.3): each step a `TVIso` prox with pixel weights, warm-started from the last dual, decreasing the objective to a stationary point, 6 dB above TV |
| [sparse_recovery.py](sparse_recovery.py) | Proximal gradient with ℓ1 against MCP, SCAD, ℓ0, ℓ1/2 and the log penalty (0.2) on a batch of sparse recovery problems, and the tie rule |

`tests/python/test_examples.py` runs every script here, so they are kept in
step with the API.
