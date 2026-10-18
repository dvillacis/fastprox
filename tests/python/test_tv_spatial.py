"""Spatially varying weights for image TV (since 0.3): pixel weights for
isotropic TV, edge weights for anisotropic TV. The CVXPY fixtures are in
test_fixtures.py (cases *_pixel and *_edge)."""

import numpy as np
import pytest

import fastprox as fp

SHAPES = [(2, 9, 11), (4, 5, 6)]


def image(shape, seed):
    rng = np.random.default_rng(seed)
    k = 2 if len(shape) == 3 and shape[0] == 2 else len(shape)
    img = shape[-k:]
    return rng.normal(scale=0.4, size=shape) + (np.indices(img).sum(0) // 3 % 2)


def spatial(iso, img, seed, zeros=True):
    rng = np.random.default_rng(seed)
    size = img if iso else (len(img), *img)
    v = rng.uniform(0.3, 1.7, size)
    if zeros:
        v[rng.random(size) < 0.1] = 0.0
    return v


def make(iso, k, w, sw):
    return fp.TVIso(k, w, pixel_weights=sw) if iso else fp.TVAniso(k, w, edge_weights=sw)


def ndim_of(shape):
    return 2 if shape == SHAPES[0] else 3


@pytest.mark.parametrize("iso", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_unit_weights_are_the_unweighted_operator(iso, shape):
    k = ndim_of(shape)
    img = shape[-k:]
    x = image(shape, 1)
    w = np.linspace(0.8, 1.2, k)
    plain = (fp.TVIso if iso else fp.TVAniso)(k, w)
    ones = make(iso, k, w, np.ones(img if iso else (k, *img)))
    np.testing.assert_array_equal(ones.prox(x, 0.4, tol=1e-10, max_iter=10 ** 5),
                                  plain.prox(x, 0.4, tol=1e-10, max_iter=10 ** 5))


@pytest.mark.parametrize("iso", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_threads_and_batches(iso, shape):
    k = ndim_of(shape)
    img = shape[-k:]
    x = image(shape, 2)
    op = make(iso, k, None, spatial(iso, img, 3))
    runs = [op.prox(x, 0.4, tol=1e-10, max_iter=10 ** 5, threads=t) for t in (1, 3, 8)]
    for z in runs[1:]:
        np.testing.assert_array_equal(z, runs[0])
    if len(shape) > k:  # each image of the batch as if alone
        for b in range(shape[0]):
            np.testing.assert_array_equal(op.prox(x[b], 0.4, tol=1e-10, max_iter=10 ** 5), runs[0][b])


@pytest.mark.parametrize("iso", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("conj", [False, True])
def test_derivatives(iso, shape, conj):
    """Adjointness of the JVP and VJP in every parameter, and the JVP against
    finite differences; gradients have the weights' shapes."""
    k = ndim_of(shape)
    img = shape[-k:]
    rng = np.random.default_rng(4)
    x = image(shape, 5)
    w = rng.uniform(0.6, 1.4, k)
    sw = spatial(iso, img, 6)
    key = "pixel_weights" if iso else "edge_weights"
    op = make(iso, k, w, sw)
    step = 2.0 if conj else 0.4
    run = (lambda o, v, s, **kw: o.prox_conj(v, s, **kw)) if conj else (lambda o, v, s, **kw: o.prox(v, s, **kw))
    tight = dict(tol=1e-14, max_iter=10 ** 6)
    _, ctx = run(op, x, step, keep=True, **tight)
    dx, dw, dg = rng.normal(size=shape), rng.normal(size=k), 0.2
    dsw = rng.normal(size=sw.shape) * (sw > 0)    # a zero weight cannot step below 0
    dt = {"tol": 1e-12} if iso else {}
    jv = op.jvp(ctx, dx, dg, {"weights": dw, key: dsw}, **dt)
    g = rng.normal(size=shape)
    gx, grads = op.vjp(ctx, g, **dt)
    assert grads.weights.shape == (k,) and getattr(grads, key).shape == sw.shape
    lhs = float((g * jv).sum())
    rhs = float((gx * dx).sum() + grads.gamma * dg + grads.weights @ dw + (getattr(grads, key) * dsw).sum())
    assert abs(lhs - rhs) <= 1e-8 * (1 + abs(lhs))
    h = 1e-6 if iso else 1e-8  # anisotropic TV is piecewise affine and solved exactly
    zp = run(make(iso, k, w + h * dw, sw + h * dsw), x + h * dx, step + h * dg, **tight)
    zm = run(make(iso, k, w - h * dw, sw - h * dsw), x - h * dx, step - h * dg, **tight)
    fd = (zp - zm) / (2 * h)
    assert np.abs(fd - jv).max() <= 1e-4 * max(1.0, np.abs(fd).max())


def test_zero_weights_decouple():
    """Zero edge weights across a column split anisotropic TV into two
    independent problems; a zero pixel weight leaves the pixel's dual at 0."""
    x = image((9, 12), 7)
    b = np.ones((2, 9, 12))
    b[1, :, 4] = 0.0                               # edges from column 4 to 5
    z = fp.TVAniso(2, edge_weights=b).prox(x, 0.5, tol=1e-12, max_iter=10 ** 5)
    left = fp.TVAniso(2).prox(x[:, :5], 0.5, tol=1e-12, max_iter=10 ** 5)
    right = fp.TVAniso(2).prox(x[:, 5:], 0.5, tol=1e-12, max_iter=10 ** 5)
    np.testing.assert_allclose(z, np.concatenate([left, right], 1), atol=1e-6)
    a = np.ones((9, 12))
    a[3, 4] = 0.0
    _, info = fp.TVIso(2, pixel_weights=a).solve(x, 0.5, tol=1e-10, max_iter=10 ** 5)
    assert np.all(info.dual[:, 3, 4] == 0.0)
    assert np.sqrt((info.dual ** 2).sum(0)).max() <= 0.5 * (1 + 1e-12)


def test_errors():
    with pytest.raises(ValueError):
        fp.TVIso(2, pixel_weights=np.ones(5))              # not an image
    with pytest.raises(ValueError):
        fp.TVIso(2, pixel_weights=-np.ones((3, 4)))
    with pytest.raises(ValueError):
        fp.TVAniso(2, edge_weights=np.ones((3, 4, 5)))     # needs (2, ...)
    with pytest.raises(ValueError):
        fp.TVAniso(3, edge_weights=np.full((3, 2, 3, 4), np.nan))
    op = fp.TVIso(2, pixel_weights=np.ones((3, 4)))
    with pytest.raises(ValueError):
        op.prox(np.zeros((3, 5)), 0.5)                     # another image shape
    np.testing.assert_array_equal(op.prox(np.zeros((2, 3, 4)), 0.5), 0.0)  # batches are fine


@pytest.mark.parametrize("iso", [True, False])
def test_torch_and_jax_wrappers(iso):
    """The C path of fastprox.torch.prox / fastprox.jax.prox: gradients in the
    spatial weights by gradcheck and check_grads."""
    k, img = 2, (5, 6)
    x = image((2, *img), 8)
    sw = spatial(iso, img, 9, zeros=False)
    key = "pixel_weights" if iso else "edge_weights"
    cls = fp.TVIso if iso else fp.TVAniso
    opts = dict(tol=1e-14, max_iter=10 ** 6)
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    xt = torch.tensor(x)
    st = torch.tensor(sw, requires_grad=True)
    assert torch.autograd.gradcheck(lambda s: fpt.prox(cls, xt, 0.4, ndim=k, **{key: s}, **opts), (st,), atol=1e-5)
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.test_util import check_grads

    import fastprox.jax as fpj
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        check_grads(lambda s: fpj.prox(cls, jnp.asarray(x), 0.4, ndim=k, **{key: s}, **opts), (jnp.asarray(sw),),
                    order=1, modes=["rev"], eps=1e-6 if iso else 1e-8)
    finally:
        jax.config.update("jax_enable_x64", old)
