"""The JAX wrappers: reverse-mode gradients through the library's VJP against
finite differences, jit, vmap, agreement with the NumPy API, dtypes and
errors."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

import fastprox as fp  # noqa: E402
import fastprox.jax as fpj  # noqa: E402

from interop_cases import cases  # noqa: E402

CASES = cases()
IDS = [c[0] for c in CASES]


@pytest.fixture(autouse=True)
def x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_grads(case):
    _, cls, const, params, x, step, conj, call = case
    f = fpj.prox_conj if conj else fpj.prox
    names = list(params)

    def fn(x, step, *vals):
        return f(cls, x, step, **const, **dict(zip(names, vals)), **call)

    args = (jnp.asarray(x), jnp.asarray(step), *[jnp.asarray(v) for v in params.values()])
    check_grads(fn, args, order=1, modes=["rev"], eps=1e-6)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_matches_numpy_and_jit(case):
    _, cls, const, params, x, step, conj, call = case
    op = cls(**const, **params)
    ref = (op.prox_conj if conj else op.prox)(x, step, **call)
    f = fpj.prox_conj if conj else fpj.prox
    z = f(op, jnp.asarray(x), step, **call)
    assert z.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(z), ref)
    g = jax.jit(lambda x, s, *v: f(cls, x, s, **const, **dict(zip(params, v)), **call))
    vals = [jnp.asarray(v) for v in params.values()]
    np.testing.assert_array_equal(np.asarray(g(jnp.asarray(x), step, *vals)), ref)


def test_grad_matches_vjp_under_jit():
    rng = np.random.default_rng(1)
    x, w, g = rng.normal(size=(4, 20)).cumsum(-1), rng.uniform(0.5, 1.5, 19), rng.normal(size=(4, 20))
    op = fp.TV1D(weights=w)
    _, ctx = op.prox(x, 0.6, keep=True)
    gx, grads = op.vjp(ctx, g)
    loss = lambda x, s, w: jnp.vdot(fpj.prox(fp.TV1D, x, s, weights=w), g)
    dx, ds, dw = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(jnp.asarray(x), 0.6, jnp.asarray(w))
    np.testing.assert_array_equal(np.asarray(dx), gx)
    np.testing.assert_array_equal(np.asarray(dw), grads.weights)
    assert float(ds) == grads.gamma


def test_vmap():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 25)).cumsum(-1)
    steps = jnp.array([0.1, 0.5, 2.0])
    z = jax.vmap(lambda xi, s: fpj.prox(fp.TV1D(), xi, s))(jnp.asarray(x), steps)
    for i in range(3):
        np.testing.assert_array_equal(np.asarray(z[i]), fp.tv1d(x[i], float(steps[i])))
    gs = jax.vmap(jax.grad(lambda xi, s: fpj.prox(fp.TV1D(), xi, s).sum(), argnums=1))(jnp.asarray(x), steps)
    assert gs.shape == (3,)


def test_float32_and_constants():
    jax.config.update("jax_enable_x64", False)
    x = jnp.asarray(np.random.default_rng(3).normal(size=(2, 6)), jnp.float32)
    z = fpj.prox(fp.TreeTV, x, 0.5, n=6, edges=np.array([[0, 1], [1, 2], [3, 4]]))
    assert z.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(z), fp.TreeTV(6, [[0, 1], [1, 2], [3, 4]]).prox(np.asarray(x), 0.5))
    gx = jax.grad(lambda x: fpj.prox(fp.Soft(0.2), x).sum())(x)
    assert gx.dtype == jnp.float32


def test_errors():
    x = jnp.zeros(10)
    with pytest.raises(TypeError):
        fpj.prox(fp.TV1D(), jnp.arange(10), 0.5)
    with pytest.raises(TypeError):
        fpj.prox(fp.TV1D(), x, 0.5, weights=jnp.ones(9))
    with pytest.raises(TypeError):
        fpj.prox(fp.TVIso(), jnp.zeros((4, 4)), 0.5, axis=0)
    # A traced parameter without a gradient: NaN, and the others are unaffected.
    gl, gx = jax.grad(lambda lam, x: fpj.prox(fp.L0, x, 0.5, lam=lam).sum(), argnums=(0, 1))(
        jnp.asarray(0.3), x + 1.0)
    assert np.isnan(gl) and np.all(np.asarray(gx) == 1.0)
