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

from interop_cases import built, cases, row_steps, spectral_cases  # noqa: E402

CASES = cases()
IDS = [c[0] for c in CASES]
SPECTRAL = spectral_cases()
SPECTRAL_IDS = [c[0] for c in SPECTRAL]


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


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_steps_grads(case):
    """One step per row (fp_prox_steps), each row's step gradient from a JVP
    in gamma; also under jit."""
    _, cls, const, params, x, step, conj, call = case
    s = row_steps(case)
    if conj:
        with pytest.raises(ValueError, match="scalar sigma"):
            fpj.prox_conj(cls, jnp.asarray(x), jnp.asarray(s), **const, **params, **call)
        return
    fn = lambda x, s: fpj.prox(cls, x, s, **const, **params, **call)  # noqa: E731
    check_grads(fn, (jnp.asarray(x), jnp.asarray(s)), order=1, modes=["rev"], eps=1e-6)
    op = cls(**const, **params)
    np.testing.assert_array_equal(np.asarray(jax.jit(fn)(jnp.asarray(x), jnp.asarray(s))),
                                  op.prox(x, s, **call))


def test_steps_broadcast_and_spectral():
    rng = np.random.default_rng(6)
    x, g = rng.normal(size=(4, 3, 20)).cumsum(-1), rng.normal(size=(4, 3, 20))
    s = rng.uniform(0.2, 1.0, (4, 1))
    op = fp.TV1D()
    gs = jax.jit(jax.grad(lambda s: jnp.vdot(fpj.prox(op, jnp.asarray(x), s), g)))(jnp.asarray(s))
    assert gs.shape == (4, 1)
    for i in range(4):
        gr = [op.vjp(op.prox(x[i, j], s[i, 0], keep=True)[1], g[i, j])[1].gamma for j in range(3)]
        assert float(gs[i, 0]) == pytest.approx(sum(gr), rel=1e-11, abs=1e-12)
    xm, sm = rng.normal(size=(2, 3, 5, 4)) * 1.5, rng.uniform(0.4, 1.0, (2, 3))
    fn = lambda x, s, lam: fpj.prox(fp.NuclearNorm, x, s, lam=lam)  # noqa: E731
    check_grads(fn, (jnp.asarray(xm), jnp.asarray(sm), jnp.asarray(0.6)), order=1, modes=["rev"], eps=1e-6)
    check_grads(lambda x, s: fpj.prox(fp.PSD(), x, s), (jnp.asarray(xm[:, :, :4]), jnp.asarray(sm)),
                order=1, modes=["rev"], eps=1e-6)


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


def test_weighted_isotonic_has_no_conj():
    # Math audit M2: the weighted fit projects in the metric W.
    w = np.array([1.0, 3.0, 2.0])
    y = jnp.array([2.0, 0.0, 1.0], dtype=jnp.float64)
    with pytest.raises(ValueError, match="no Euclidean conjugate"):
        fpj.prox_conj(fp.Isotonic(w), y)


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


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SPECTRAL, ids=SPECTRAL_IDS)
def test_spectral_grads(case):
    _, op, params, x, step = case
    names = list(params)

    def fn(x, step, *vals):
        return fpj.prox(op, x, step, **dict(zip(names, vals)))

    args = (jnp.asarray(x), jnp.asarray(step), *[jnp.asarray(v) for v in params.values()])
    check_grads(fn, args, order=1, modes=["rev"], eps=1e-6)


@pytest.mark.parametrize("case", SPECTRAL, ids=SPECTRAL_IDS)
def test_spectral_matches_numpy_under_jit(case):
    _, op, params, x, step = case
    ref_op = op(**params) if isinstance(op, type) else op
    g = np.random.default_rng(2).normal(size=x.shape)
    z, ctx = ref_op.prox(x, step, keep=True)
    gx, grads = ref_op.vjp(ctx, g)
    names = list(params)

    def loss(x, step, *vals):
        return jnp.vdot(fpj.prox(op, x, step, **dict(zip(names, vals))), g)

    args = (jnp.asarray(x), jnp.asarray(step), *[jnp.asarray(v) for v in params.values()])
    zj = jax.jit(lambda x, s, *v: fpj.prox(op, x, s, **dict(zip(names, v))))(*args)
    assert zj.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(zj), z, atol=1e-13)
    out = jax.jit(jax.grad(loss, argnums=tuple(range(len(args)))))(*args)
    np.testing.assert_allclose(np.asarray(out[0]), gx, atol=1e-12)
    assert float(out[1]) == pytest.approx(grads.gamma, abs=1e-12)
    for k, v in zip(names, out[2:]):
        assert float(v) == pytest.approx(getattr(grads, k), abs=1e-12)


def test_spectral_ties_and_vmap():
    rng = np.random.default_rng(3)
    x = built(7, 5, (2.0, 2.0, 2.0, 1.5, 1.5), rng)
    g = rng.normal(size=x.shape)
    op = fp.NuclearNorm(0.7)
    gx = np.asarray(jax.grad(lambda x: jnp.vdot(fpj.prox(op, x), g))(jnp.asarray(x)))
    assert np.isfinite(gx).all()
    for _ in range(3):
        h = rng.normal(size=x.shape)
        fd = (np.sum(g * op.prox(x + 1e-6 * h)) - np.sum(g * op.prox(x - 1e-6 * h))) / 2e-6
        assert np.sum(gx * h) == pytest.approx(fd, rel=1e-6)
    xb = rng.normal(size=(3, 5, 4))
    zv = jax.vmap(lambda xi: fpj.prox(op, xi))(jnp.asarray(xb))
    np.testing.assert_allclose(np.asarray(zv), op.prox(xb), atol=1e-13)


def test_spectral_float32_and_errors():
    jax.config.update("jax_enable_x64", False)
    x = jnp.asarray(np.random.default_rng(4).normal(size=(2, 5, 5)), jnp.float32)
    z = fpj.prox(fp.PSD(), x)
    assert z.dtype == jnp.float32
    assert jax.grad(lambda x: fpj.prox(fp.PSD(), x).sum())(x).dtype == jnp.float32
    with pytest.raises(TypeError, match="prox_conj"):
        fpj.prox_conj(fp.NuclearNorm(), x)
    with pytest.raises(TypeError, match="axis"):
        fpj.prox(fp.NuclearNorm(), x, axis=0)
    with pytest.raises(ValueError, match="matrices"):
        fpj.prox(fp.NuclearNorm(), jnp.zeros(4))
    with pytest.raises(ValueError, match="square"):
        fpj.prox(fp.PSD(), jnp.zeros((3, 4)))


def test_jit_cache():
    # A jit of the built function per operator instance, and per class with
    # equal static arguments (audit P9).
    x = jnp.asarray(np.random.default_rng(30).normal(size=50))
    op = fp.Soft(0.4)
    z1 = fpj.prox(op, x, 0.8)
    z2 = fpj.prox(op, x, 0.8)
    assert len(op.__dict__["_fpj_jit"]) == 1
    ref = op.prox(np.asarray(x), 0.8)
    assert np.asarray(z1).tobytes() == np.asarray(z2).tobytes() == ref.tobytes()
    fpj.prox(op, x[:20], 0.8)  # a new shape: JAX's cache, the same entry
    assert len(op.__dict__["_fpj_jit"]) == 1
    fpj._CLASS_CACHE.clear()
    w = np.linspace(0.5, 1.5, 50)
    a = fpj.prox(fp.Soft, x, 0.8, lam=0.4, weights=w)
    b = fpj.prox(fp.Soft, x, 0.8, lam=0.4, weights=w.copy())
    assert len(fpj._CLASS_CACHE) == 1
    c = fpj.prox(fp.Soft, x, 0.8, lam=0.4, weights=w[::-1].copy())
    assert len(fpj._CLASS_CACHE) == 2
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes() == fp.Soft(0.4, w).prox(np.asarray(x), 0.8).tobytes()
    assert np.asarray(c).tobytes() == fp.Soft(0.4, w[::-1].copy()).prox(np.asarray(x), 0.8).tobytes()
    # The gradient is the library's vjp, bit for bit.
    g = jax.grad(lambda x: (fpj.prox(op, x, 0.8) * jnp.arange(50.0)).sum())(x)
    _, ctx = op.prox(np.asarray(x), 0.8, keep=True)
    assert np.asarray(g).tobytes() == op.vjp(ctx, np.arange(50.0))[0].tobytes()


def test_jit_cache_shapes():
    # One jit serves every shape: spectral operators and the device TV.
    rng = np.random.default_rng(31)
    op = fp.NuclearNorm(0.5)
    for shape in ((4, 3), (3, 5), (2, 4, 4)):
        a = rng.normal(size=shape)
        np.testing.assert_allclose(fpj.prox(op, jnp.asarray(a), 0.7), op.prox(a, 0.7), rtol=1e-12, atol=1e-12)
    for shape in ((16, 16), (12, 20)):
        a = rng.normal(size=shape)
        z = fpj.tv_iso(jnp.asarray(a), 0.3, tol=1e-10, max_iter=100000)
        ref = fp.TVIso(2).prox(a, 0.3, tol=1e-12, max_iter=100000)
        np.testing.assert_allclose(z, ref, atol=1e-4 * np.linalg.norm(a))


def test_forward_without_structure_and_grads():
    # The forward callback runs without the derivative structure (audit P10):
    # the same bits, and the gradient unchanged.
    rng = np.random.default_rng(32)
    x = rng.normal(size=(8, 40))
    for op in (fp.Simplex(1.0), fp.L1Ball(2.0), fp.Huber(0.4, 0.5)):
        z = fpj.prox(op, jnp.asarray(x), 0.8)
        assert np.asarray(z).tobytes() == op.prox(x, 0.8).tobytes()
        g = jax.grad(lambda t: (fpj.prox(op, t, 0.8) * t).sum())(jnp.asarray(x))
        zz, ctx = op.prox(x, 0.8, keep=True)
        np.testing.assert_array_equal(g, zz + op.vjp(ctx, x)[0])
