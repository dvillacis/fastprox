"""The operators inside PyTorch and JAX programs (0.3).

``fastprox.torch.prox`` and ``fastprox.jax.prox`` make any operator a
differentiable function of its input, its step and its parameters, with the
gradients from the library's VJP (no unrolled iterations). Two small
learning problems:

1. PyTorch: one TV weight per edge for denoising piecewise-constant signals
   (the setting of bilevel_learning.py), as an ``nn.Parameter`` trained by
   Adam on minibatches; w = softplus(theta) keeps the weights positive.
2. JAX: the step gamma of isotropic TV for denoising images, by gradient
   descent on log(gamma) with ``jax.jit(jax.value_and_grad(...))``.

Needs PyTorch and JAX (``uv run --group interop python
examples/autodiff_frameworks.py``); without them it says so and stops.
"""

import numpy as np

import fastprox as fp

N, JUMPS = 120, [30, 55, 90]


def signals(rng, m, sigma=0.3):
    edges = [0, *JUMPS, N]
    levels = rng.normal(size=(m, len(edges) - 1))
    x = np.concatenate([np.repeat(levels[:, [k]], edges[k + 1] - edges[k], 1) for k in range(len(edges) - 1)], 1)
    return x, x + sigma * rng.normal(size=x.shape)


def torch_part(rng):
    import torch

    import fastprox.torch as fpt

    x_train, y_train = (torch.tensor(a) for a in signals(rng, 512))
    x_test, y_test = (torch.tensor(a) for a in signals(rng, 512))
    theta = torch.nn.Parameter(torch.full((N - 1,), -1.0, dtype=torch.float64))
    opt = torch.optim.Adam([theta], lr=0.05)

    def loss(x, y):
        z = fpt.prox(fp.TV1D, y, 1.0, weights=torch.nn.functional.softplus(theta))
        return 0.5 * ((z - x) ** 2).sum(1).mean()

    with torch.no_grad():
        before = loss(x_test, y_test).item()
    for step in range(200):
        idx = torch.randint(0, 512, (64,), generator=torch.Generator().manual_seed(step))
        opt.zero_grad()
        loss(x_train[idx], y_train[idx]).backward()
        opt.step()
    with torch.no_grad():
        after = loss(x_test, y_test).item()
    w = torch.nn.functional.softplus(theta).detach().numpy()
    found = sorted(int(i) + 1 for i in np.argsort(w)[:len(JUMPS)])
    print(f"PyTorch: test loss {before:.4f} -> {after:.4f}; lowest weights at {found} (jumps at {JUMPS})")
    return before, after, found


def jax_part(rng):
    import jax
    import jax.numpy as jnp

    import fastprox.jax as fpj

    h = 48
    a, b = np.mgrid[0:h, 0:h] / h
    clean = ((a - 0.5) ** 2 + (b - 0.45) ** 2 < 0.08) + 0.5 * (b > 0.7)
    x = jnp.asarray(np.stack([clean, clean.T]))
    y = x + 0.2 * jnp.asarray(rng.normal(size=x.shape), jnp.float32)
    x = x.astype(jnp.float32)

    @jax.jit
    def loss_and_grad(t):
        f = lambda t: 0.5 * jnp.sum((fpj.prox(fp.TVIso(), y, jnp.exp(t), tol=1e-8, max_iter=10 ** 5) - x) ** 2)
        return jax.value_and_grad(f)(t)

    t = jnp.log(0.02).astype(jnp.float32)
    first, _ = loss_and_grad(t)
    for _ in range(60):
        val, g = loss_and_grad(t)
        t = t - 5e-3 * g
    gamma = float(jnp.exp(t))
    grid = np.geomspace(gamma / 2, gamma * 2, 21)
    best = grid[np.argmin([float(0.5 * jnp.sum((fp.TVIso().prox(np.asarray(y), g, tol=1e-7,
                                                                 max_iter=10 ** 5) - x) ** 2))
                           for g in grid])]
    print(f"JAX:     loss {float(first):.2f} -> {float(val):.2f}; gamma {gamma:.4f} (grid search {best:.4f})")
    return gamma, best


def main():
    try:
        import jax  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch and JAX are not installed: uv run --group interop python examples/autodiff_frameworks.py")
        return None
    rng = np.random.default_rng(0)
    return torch_part(rng), jax_part(rng)


if __name__ == "__main__":
    main()
