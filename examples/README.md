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

`tests/python/test_examples.py` runs every script here, so they are kept in
step with the API.
