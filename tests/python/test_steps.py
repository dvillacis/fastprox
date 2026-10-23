"""One step per row or image (fp_prox_steps, since 1.0): every operator against
row-by-row calls with that row's step, for the prox and its derivatives."""

import numpy as np
import pytest

import fastprox as fp

from test_operators import CASES, N, NAMES

ROWS = 6


def _steps(gamma, size, seed=0):
    # Steps up to the case's own, so that step conditions (SCAD, MCP) still hold.
    return gamma * np.random.default_rng(seed).uniform(0.4, 1.0, size)


def _rows(f, x, steps):
    return np.stack([f(x[i], steps[i]) for i in range(len(steps))])


@pytest.mark.parametrize("name", NAMES)
def test_prox_matches_rows(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    x = np.random.default_rng(1).normal(scale=1.5, size=(ROWS, N))
    s = _steps(gamma, ROWS)
    np.testing.assert_array_equal(op.prox(x, s), _rows(op.prox, x, s))
    # Along the first axis, where an elementwise run crosses rows.
    np.testing.assert_array_equal(op.prox(x.T, s, axis=0), op.prox(x, s).T)
    # Broadcast steps over a (2, 3) batch, and float32.
    x3 = x.reshape(2, 3, N)
    s3 = s[:2, None]
    np.testing.assert_array_equal(op.prox(x3, s3),
                                  op.prox(x3, np.broadcast_to(s3, (2, 3)).copy()))
    xf = x.astype(np.float32)
    np.testing.assert_array_equal(op.prox(xf, s), _rows(op.prox, xf, s))


@pytest.mark.parametrize("name", NAMES)
def test_derivatives_match_rows(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    rng = np.random.default_rng(2)
    x = rng.normal(scale=1.5, size=(ROWS, N))
    s = _steps(gamma, ROWS, 3)
    dx, g = rng.normal(size=x.shape), rng.normal(size=x.shape)
    dth = rng.normal(size=theta.size) if theta.size else None
    z, ctx = op.prox(x, s, keep=True)
    dz = op.jvp(ctx, dx, 0.37, dth)
    gx, grads = op.vjp(ctx, g)
    t = op.jvp(ctx, None, 1.0)  # dz/dgamma of each row at its own step
    for i in range(ROWS):
        zi, ci = op.prox(x[i], s[i], keep=True)
        np.testing.assert_array_equal(z[i], zi)
        np.testing.assert_allclose(dz[i], op.jvp(ci, dx[i], 0.37, dth), rtol=1e-13, atol=1e-13)
        gxi, gri = op.vjp(ci, g[i])
        np.testing.assert_allclose(gx[i], gxi, rtol=1e-13, atol=1e-13)
        assert np.dot(g[i], t[i]) == pytest.approx(gri.gamma, rel=1e-11, abs=1e-12)
    sums = [op.vjp(op.prox(x[i], s[i], keep=True)[1], g[i])[1] for i in range(ROWS)]
    assert grads.gamma == pytest.approx(sum(r.gamma for r in sums), rel=1e-11, abs=1e-12)
    np.testing.assert_allclose(grads.params, sum(r.params for r in sums), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_threads_bitwise(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    x = np.random.default_rng(5).normal(size=(4000, N))
    s = _steps(gamma, 4000)
    g = np.sin(np.arange(x.size)).reshape(x.shape)
    outs = []
    for t in (1, 3, 8):
        z, ctx = op.prox(x, s, keep=True, threads=t)
        gx, gr = op.vjp(ctx, g)
        outs.append(z.tobytes() + gx.tobytes() + np.append(gr.params, gr.gamma).tobytes())
    assert outs[0] == outs[1] == outs[2]


IMAGE = {
    "tv_aniso": (lambda: fp.TVAniso(2, [1.0, 0.6]), (3, 9, 8)),
    "tv_aniso_3d": (lambda: fp.TVAniso(3), (2, 5, 4, 6)),
    "tv_iso": (lambda: fp.TVIso(2, [1.0, 0.6]), (3, 9, 8)),
    "graph_tv": (lambda: fp.GraphTV(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [2, 3]],
                                    [0.5, 1.0, 0.7, 0.3, 0.9, 0.6]), (4, 6)),
    "tree_tv": (lambda: fp.TreeTV(6, [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3]]), (4, 6)),
}


@pytest.mark.parametrize("name", sorted(IMAGE))
def test_images_match_one_by_one(name):
    make, shape = IMAGE[name]
    op = make()
    rng = np.random.default_rng(7)
    x = rng.normal(size=shape)
    s = rng.uniform(0.2, 1.0, shape[0])
    kw = dict(tol=1e-9, max_iter=5000)
    # Isotropic TV's derivatives solve a system by CG: tightly, so that the
    # JVP and the VJP agree.
    dk = dict(tol=1e-13) if name == "tv_iso" else {}
    z, ctx = op.prox(x, s, keep=True, **kw)
    assert ctx.info.converged
    g = rng.normal(size=shape)
    gx, grads = op.vjp(ctx, g, **dk)
    t = op.jvp(ctx, None, 1.0, **dk)
    total = 0.0
    for i in range(shape[0]):
        zi, ci = op.prox(x[i], s[i], keep=True, **kw)
        np.testing.assert_array_equal(z[i], zi)
        gxi, gri = op.vjp(ci, g[i], **dk)
        np.testing.assert_allclose(gx[i], gxi, rtol=1e-12, atol=1e-12)
        assert np.vdot(g[i], t[i]) == pytest.approx(gri.gamma, rel=1e-9, abs=1e-10)
        total += gri.gamma
    assert grads.gamma == pytest.approx(total, rel=1e-9, abs=1e-10)
    assert ctx.info.gap == pytest.approx(
        sum(op.prox(x[i], s[i], keep=True, **kw)[1].info.gap for i in range(shape[0])))


def test_spectral_steps():
    op = fp.NuclearNorm(0.7)
    x = np.random.default_rng(8).normal(size=(4, 5, 3))
    s = np.array([0.2, 0.5, 1.0, 1.7])
    z = op.prox(x, s)
    for i in range(4):
        np.testing.assert_allclose(z[i], op.prox(x[i], s[i]), rtol=1e-13, atol=1e-13)


def test_errors():
    op = fp.Soft(0.5)
    x = np.zeros((3, 4))
    with pytest.raises(ValueError, match="broadcast"):
        op.prox(x, np.ones(4))
    with pytest.raises(ValueError):
        op.prox(x, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        op.prox(x, np.array([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="scalar sigma"):
        op.prox_conj(x, np.ones(3))
    with pytest.raises(ValueError):  # SCAD needs a > 1 + gamma for every step
        fp.SCAD(0.5, 3.0).prox(x, np.array([0.5, 2.5, 0.5]))
    tv = fp.TVIso()
    with pytest.raises(ValueError, match="broadcast"):
        tv.prox(np.zeros((2, 4, 4)), np.ones(3))
    # A scalar array is still a scalar step.
    np.testing.assert_array_equal(op.prox(x + 1, np.float64(0.5)), op.prox(x + 1, 0.5))
