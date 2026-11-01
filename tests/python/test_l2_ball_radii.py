"""The l2 ball with one radius per row (since 1.1): ``L2Ball(radius=array)``
and ``project_l2_ball(x, radius=array)``.

The operator is built for one batch shape, so it cannot join ``CASES`` in
test_operators.py (whose tests use several batch shapes); this file runs the
same checks on it: reference, finite differences in x and the radii,
adjointness, the conjugate, threads, axes and strides, float32, and the
value and structure, plus the bits of the single radius for equal radii.
The PyTorch and JAX wrappers run it from interop_cases.py."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref

N = 7
BATCH = (4, 3)


def radii(seed=0):
    r = np.random.default_rng(seed).uniform(0.2, 2.5, BATCH)
    r[1, 2] = 0.0  # the zero vector
    return r


def data(seed=1, scale=1.0):
    x = np.random.default_rng(seed).normal(scale=scale, size=BATCH + (N,))
    x[0, 0] *= 1e-3  # inside
    return x


def per_row(f, x, r):
    return np.stack([f(xr, rr) for xr, rr in zip(x.reshape(-1, N), r.reshape(-1))]).reshape(x.shape)


def test_matches_reference_and_rows():
    r = radii()
    op = fp.L2Ball(r)
    assert op.nparams == r.size and op.batch_shape == BATCH
    for scale in (0.1, 1.0, 10.0):
        x = data(scale=scale)
        z = op.prox(x)
        np.testing.assert_allclose(z, per_row(ref.l2_ball, x, r), rtol=1e-14, atol=1e-15 * scale)
        # Each row is the single radius on that row, bit for bit.
        np.testing.assert_array_equal(z, per_row(lambda xr, rr: fp.project_l2_ball(xr, rr), x, r))
        np.testing.assert_array_equal(fp.project_l2_ball(x, r), z)


def test_tvq_dual_step():
    """The pixelwise projection s lam / max(lam, |s|) of a TV dual, as in a
    PDHG with a weight per pixel: the same bits as the formula and, for a
    constant weight, as the scalar radius."""
    rng = np.random.default_rng(3)
    s = rng.normal(size=(32, 40, 2))
    lam = rng.uniform(0.1, 1.5, (32, 40))
    norms = np.sqrt(s[..., 0] ** 2 + s[..., 1] ** 2)
    np.testing.assert_array_equal(fp.project_l2_ball(s, lam), s * (lam / np.maximum(lam, norms))[..., None])
    np.testing.assert_array_equal(fp.project_l2_ball(s, np.full((32, 40), 0.7)), fp.project_l2_ball(s, 0.7))


@pytest.mark.parametrize("conj", [False, True])
def test_equal_radii_are_the_single_radius(conj):
    x = data(scale=2.0)
    rng = np.random.default_rng(4)
    dx, g = rng.normal(size=x.shape), rng.normal(size=x.shape)
    ops = (fp.L2Ball(np.full(BATCH, 0.9)), fp.L2Ball(0.9))
    out = []
    for op, dth in zip(ops, (np.full(BATCH, 0.3), 0.3)):
        prox = op.prox_conj if conj else op.prox
        z, ctx = prox(x, 0.6, keep=True)
        dz = op.jvp(ctx, dx, 0.37, {"radius": dth})
        gx, grads = op.vjp(ctx, g)
        out.append((z, dz, gx, grads.gamma, np.sum(grads.radius), op.value(x)))
    for a, b in zip(out[0][:4], out[1][:4]):
        np.testing.assert_array_equal(a, b)
    assert out[0][4] == pytest.approx(out[1][4], rel=1e-13)
    np.testing.assert_array_equal(out[0][5], out[1][5])


@pytest.mark.parametrize("conj", [False, True])
def test_derivatives(conj):
    r = radii()
    r[1, 2] = 0.4  # central differences need every radius > 0
    op = fp.L2Ball(r)
    h = 1e-6
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = rng.normal(scale=1.5, size=BATCH + (N,))
        prox = op.prox_conj if conj else op.prox
        z, ctx = prox(x, 0.8, keep=True)
        dx, g = rng.normal(size=x.shape), rng.normal(size=x.shape)
        dr = rng.normal(size=BATCH)
        dg = 0.37
        dz = op.jvp(ctx, dx, dg, dr)
        gx, grads = op.vjp(ctx, g)
        assert grads.radius.shape == BATCH
        np.testing.assert_array_equal(grads.radius.reshape(-1), grads.params)
        # Adjointness, one tangent per radius.
        lhs = np.vdot(g, dz)
        rhs = np.vdot(gx, dx) + grads.gamma * dg + np.vdot(grads.params, dr.reshape(-1))
        assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)

        # Central differences through rebuilt operators.
        def f(t):
            o = fp.L2Ball(r + t * dr)
            p = o.prox_conj if conj else o.prox
            return p(x + t * dx, 0.8 + t * dg)

        fd = (f(h) - f(-h)) / (2 * h)
        np.testing.assert_allclose(dz, fd, rtol=1e-5, atol=1e-5 * (1 + np.abs(fd).max()))
        # The gradient of one row's radius is that row alone.
        k = int(np.argmax(np.abs(grads.params)))
        e = np.zeros(r.size)
        e[k] = 1.0
        assert grads.params[k] == pytest.approx(np.vdot(g, op.jvp(ctx, None, 0.0, e)), rel=1e-12, abs=1e-14)


def test_named_dparams():
    r = radii()
    op = fp.L2Ball(r)
    x = data(scale=3.0)
    z, ctx = op.prox(x, keep=True)
    np.testing.assert_array_equal(op.jvp(ctx, None, 0.0, {"radius": np.ones(BATCH)}),
                                  op.jvp(ctx, None, 0.0, np.ones(r.size)))
    # A number broadcasts over the radii.
    np.testing.assert_array_equal(op.jvp(ctx, None, 0.0, {"radius": 1.0}),
                                  op.jvp(ctx, None, 0.0, np.ones(r.size)))
    with pytest.raises(ValueError):
        op.jvp(ctx, None, 0.0, np.ones(r.size + 1))


def test_threads_bitwise():
    rng = np.random.default_rng(5)
    r = rng.uniform(0.5, 3.5, 4000)
    op = fp.L2Ball(r)
    x = rng.normal(size=(4000, N))
    g = np.sin(np.arange(x.size)).reshape(x.shape)
    outs = []
    for t in (1, 3, 8):
        z, ctx = op.prox(x, keep=True, threads=t)
        gx, gr = op.vjp(ctx, g)
        outs.append(z.tobytes() + gx.tobytes() + gr.params.tobytes() + np.float64(gr.gamma).tobytes())
    assert outs[0] == outs[1] == outs[2]


def test_axis_and_strides():
    rng = np.random.default_rng(6)
    r = rng.uniform(0.5, 2.0, (5, 4))
    x = rng.normal(size=(5, N, 4))
    op = fp.L2Ball(r)
    z1 = op.prox(x, axis=1)
    z2 = np.moveaxis(op.prox(np.ascontiguousarray(np.moveaxis(x, 1, -1))), -1, 1)
    np.testing.assert_array_equal(z1, z2)
    v = x[::-1, :, ::-1]
    opv = fp.L2Ball(r[::-1, ::-1])  # strided radii
    np.testing.assert_array_equal(opv.prox(v, axis=1), opv.prox(np.ascontiguousarray(v), axis=1))
    y = x.copy()
    op.prox(y, axis=1, out=y)
    np.testing.assert_array_equal(y, z1)
    # Another batch shape: the operator is built for (5, 4).
    with pytest.raises(ValueError):
        op.prox(x)  # batch (5, 7)
    with pytest.raises(ValueError):
        op.prox(x[:4], axis=1)


def test_float32():
    r = radii()
    x = data()
    op32 = fp.L2Ball(r.astype(np.float32))  # float32 radii are read as given
    z32 = op32.prox(x.astype(np.float32))
    assert z32.dtype == np.float32
    np.testing.assert_allclose(z32, fp.L2Ball(r.astype(np.float32).astype(np.float64)).prox(
        x.astype(np.float32).astype(np.float64)), rtol=1e-6, atol=1e-6)


def test_value_and_structure():
    r = radii()
    x = data(scale=2.0)
    op = fp.L2Ball(r)
    norms = np.linalg.norm(x, axis=-1)
    np.testing.assert_array_equal(op.value(x), np.where(norms <= r, 0.0, np.inf))
    z = op.prox(x)
    np.testing.assert_array_equal(op.value(z), 0.0)
    _, ctx = op.prox(x, keep=True)
    s = ctx.structure()
    assert s.kind == "curved"
    np.testing.assert_array_equal(s.active.reshape(BATCH), norms > r)


def test_project_broadcasts():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 5, 3))
    col = rng.uniform(0.2, 1.0, (6, 1))
    np.testing.assert_array_equal(fp.project_l2_ball(x, col), fp.project_l2_ball(x, np.repeat(col, 5, 1)))
    np.testing.assert_array_equal(fp.project_l2_ball(x, [0.5]), fp.project_l2_ball(x, 0.5))
    z = fp.project_l2_ball(x, rng.uniform(0.2, 1.0, (6, 3)), axis=1)
    assert z.shape == x.shape
    with pytest.raises(ValueError, match="does not broadcast"):
        fp.project_l2_ball(x, np.ones(4))
    with pytest.raises(ValueError):
        fp.project_l2_ball(x, -np.ones((6, 5)))


def test_errors():
    with pytest.raises(ValueError):
        fp.L2Ball(np.array([1.0, -0.5]))
    with pytest.raises(ValueError):
        fp.L2Ball(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        fp.Spectral(fp.L2Ball(np.ones(3)))

