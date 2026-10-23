# Empty batches and NaN or Inf images: one contract on the C, PyTorch and JAX
# paths (audit P8). An empty batch gives an empty result with gap 0, 0
# iterations and converged True; an image holding a NaN is NaN, with a NaN gap
# and no iterations, and counts as finished; ±Inf is rejected, except under
# jax.jit, where it is treated as a NaN.
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

import fastprox as fp

EMPTY = [(0, 8, 8), (2, 0, 8)]
CHAIN = np.stack([np.arange(15), np.arange(1, 16)], 1)


def _batch(seed=0):
    return np.random.default_rng(seed).normal(size=(3, 8, 8))


@pytest.mark.parametrize("shape", EMPTY)
@pytest.mark.parametrize("Op", [fp.TVIso, fp.TVAniso])
def test_empty_numpy(Op, shape):
    op = Op(2)
    x = np.zeros(shape)
    z, info = op.solve(x, 0.5)
    assert z.shape == shape and info.gap == 0.0 and info.iters == 0 and info.converged
    assert info.dual.shape == (2, *shape)
    z, ctx = op.prox(x, 0.5, keep=True)
    gx, grads = op.vjp(ctx, x)
    assert gx.shape == shape and grads.gamma == 0.0 and not np.any(grads.params)
    assert op.jvp(ctx, x).shape == shape


def test_empty_graph():
    z, info = fp.GraphTV(16, CHAIN).solve(np.zeros((0, 16)), 0.5)
    assert z.shape == (0, 16) and info.gap == 0.0 and info.converged


@pytest.mark.parametrize("shape", EMPTY)
@pytest.mark.parametrize("name", ["tv_iso", "tv_aniso"])
def test_empty_torch(name, shape):
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    x = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    z, info = getattr(fpt, name)(x, gamma, return_info=True)
    assert tuple(z.shape) == shape and info.gap == 0.0 and info.iters == 0 and info.converged
    z.sum().backward()
    assert x.grad.shape == x.shape and gamma.grad is not None and float(gamma.grad) == 0.0


@pytest.mark.parametrize("shape", EMPTY)
@pytest.mark.parametrize("jit", [False, True])
def test_empty_jax(shape, jit):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import fastprox.jax as fpj
    f = lambda x, g: fpj.tv_iso(x, g, return_info=True)  # noqa: E731
    if jit:
        f = jax.jit(f)
    z, info = f(jnp.zeros(shape), 0.5)
    assert z.shape == shape and float(info.gap) == 0.0 and bool(info.converged)
    gx, gg = jax.grad(lambda x, g: fpj.tv_iso(x, g).sum(), argnums=(0, 1))(jnp.zeros(shape), 0.5)
    assert gx.shape == shape and float(gg) == 0.0


def test_nan_image_numpy():
    x = _batch()
    x[1, 3, 4] = np.nan
    for op in (fp.TVIso(2), fp.TVAniso(2)):
        z, info = op.solve(x, 0.5, tol=1e-8, max_iter=100000)
        assert np.isnan(z[1]).all() and np.isnan(info.gap) and info.converged
        clean = [op.solve(x[i], 0.5, tol=1e-8, max_iter=100000) for i in (0, 2)]
        assert info.iters == max(c[1].iters for c in clean)
        for i, (zi, _) in zip((0, 2), clean):
            assert np.array_equal(z[i], zi)


def test_nan_image_torch():
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    x = _batch()
    x[1, 3, 4] = np.nan
    for name in ("tv_iso", "tv_aniso"):
        f = getattr(fpt, name)
        xt = torch.tensor(x, requires_grad=True)
        z, info = f(xt, 0.5, tol=1e-8, max_iter=100000, return_info=True)
        assert torch.isnan(z[1]).all() and np.isnan(info.gap) and info.converged
        solo = [f(torch.tensor(x[i]), 0.5, tol=1e-8, max_iter=100000, return_info=True) for i in (0, 2)]
        assert info.iters == max(s[1].iters for s in solo)
        for i, (zi, _) in zip((0, 2), solo):
            np.testing.assert_allclose(z[i].detach().numpy(), zi.numpy(), atol=1e-6)
        z[[0, 2]].sum().backward()
        assert torch.isfinite(xt.grad[[0, 2]]).all()


@pytest.mark.parametrize("jit", [False, True])
def test_nan_image_jax(jit):
    """JAX computes in float32 by default, so the tolerance is 1e-6, and the
    reference is the same 3-image program with image 1 a copy of image 0: a
    batch of another shape compiles to other float32 reductions."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import fastprox.jax as fpj
    x = _batch()
    x[1, 3, 4] = np.nan
    ref = _batch()
    ref[1] = ref[0]
    f = lambda x: fpj.tv_iso(x, 0.5, tol=1e-6, max_iter=100000, return_info=True)  # noqa: E731
    if jit:
        f = jax.jit(f)
    z, info = f(jnp.asarray(x))
    zr, ir = f(jnp.asarray(ref))
    assert bool(jnp.isnan(z[1]).all()) and np.isnan(float(info.gap)) and bool(info.converged)
    assert int(info.iters) == int(ir.iters)
    z, zr = np.asarray(z), np.asarray(zr)
    np.testing.assert_array_equal(z[[0, 2]], zr[[0, 2]])


def test_inf_images():
    x = _batch()
    x[1, 3, 4] = np.inf
    with pytest.raises(ValueError):
        fp.TVIso(2).solve(x, 0.5)
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    with pytest.raises(ValueError, match="Inf"):
        fpt.tv_iso(torch.tensor(x), 0.5)
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import fastprox.jax as fpj
    with pytest.raises(ValueError, match="Inf"):
        fpj.tv_iso(jnp.asarray(x), 0.5)
    z = np.asarray(jax.jit(lambda x: fpj.tv_iso(x, 0.5))(jnp.asarray(x)))  # treated as a NaN image
    assert np.isnan(z[1]).all() and np.isfinite(z[[0, 2]]).all()
