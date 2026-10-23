# Derivatives of iterative TV at loose gaps (feedback U4): ctx.deriv says
# whether the derivative at the saved point is exact, ctx.refine and refine=
# solve the kept call again from its dual to a tighter tolerance, and an
# inexact derivative at a loose point warns once.
#
# SPDX-License-Identifier: MIT
import warnings

import numpy as np
import pytest

import fastprox as fp


def _blocks(seed=0, shape=(2, 48, 48), noise=0.1):
    rng = np.random.default_rng(seed)
    b = shape[-1] // 6
    return np.kron(rng.random(shape[:-2] + (6, 6)), np.ones((b, b))) + noise * rng.normal(size=shape)


def _grid(n):
    e = [(i * n + j, i * n + j + 1) for i in range(n) for j in range(n - 1)]
    e += [(i * n + j, (i + 1) * n + j) for i in range(n - 1) for j in range(n)]
    return np.array(e)


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return f()


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.filterwarnings("ignore::fastprox.AccuracyWarning")
def test_refine_isotropic():
    """A 1e-6 solve's derivative is biased; refined to ztol 1e-5 it is within
    0.7% of the reference, and bit for bit that of a fresh solve seeded with
    the loose solve's dual."""
    x = _blocks()
    g = np.random.default_rng(1).normal(size=x.shape)
    op = fp.TVIso()
    _, cr = op.prox(x, 0.2, keep=True, tol=1e-14, max_iter=10 ** 6)
    ref = op.vjp(cr, g, tol=1e-12)[0]
    _, c = op.prox(x, 0.2, keep=True, tol=1e-6, max_iter=10 ** 6)
    loose = op.vjp(c, g)[0]
    assert not c.deriv.exact and c.deriv.smoothing > 0
    dual = c.info.dual
    gx = op.vjp(c, g, refine=1e-5, max_iter=10 ** 6)[0]
    assert _rel(gx, ref) < 0.007 < _rel(loose, ref)
    assert c.info.zerr <= 1e-5
    _, cf = op.prox(x, 0.2, keep=True, tol=1e-6, max_iter=10 ** 6)
    zf, cf = op.prox(x, 0.2, keep=True, ctx=cf, tol=1e-10, max_iter=10 ** 6, warm_start=dual)
    assert np.array_equal(gx, op.vjp(cf, g, max_iter=10 ** 6)[0])
    assert c.info.gap == cf.info.gap and c.info.iters == cf.info.iters
    assert c.deriv == cf.deriv


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("Op", [fp.TVIso, fp.TVAniso])
def test_refine_returns_the_fresh_result(Op, conj):
    x = _blocks(2)
    op = Op()
    run = op.prox_conj if conj else op.prox
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fp.AccuracyWarning)
        _, c = run(x, 0.2, keep=True, tol=1e-4)
        dual = c.info.dual
        z = c.refine(1e-4, max_iter=10 ** 6)
        _, cf = run(x, 0.2, keep=True, tol=1e-4)
        zf, cf = run(x, 0.2, keep=True, ctx=cf, tol=1e-8, max_iter=10 ** 6, warm_start=dual)
    assert np.array_equal(z, zf) and c.info.zerr == cf.info.zerr and np.array_equal(c.info.dual, cf.info.dual)
    x32 = x.astype(np.float32)
    _, c32 = run(x32, 0.2, keep=True, tol=1e-4)
    assert c32.refine(1e-3).dtype == np.float32


def test_exact_flags():
    x = _blocks(3)
    g = np.random.default_rng(4).normal(size=x.shape)
    aniso = fp.TVAniso()
    _, c = aniso.prox(x, 0.2, keep=True, tol=1e-12, max_iter=10 ** 6)
    aniso.vjp(c, g)
    assert c.deriv.exact and c.deriv.smoothing == 0.0
    noisy = np.random.default_rng(5).normal(size=(2, 48, 48))
    _, c = aniso.prox(noisy, 0.5, keep=True, tol=1e-4)
    with pytest.warns(fp.AccuracyWarning):
        aniso.vjp(c, noisy)
    assert not c.deriv.exact
    graph = fp.GraphTV(48 * 48, _grid(48))
    xg = x.reshape(2, -1)
    _, c = graph.prox(xg, 0.2, keep=True)
    graph.vjp(c, xg)
    assert c.deriv.exact
    _, c = graph.prox(xg, 0.2, keep=True, tol=0.2)
    with pytest.warns(fp.AccuracyWarning):
        graph.vjp(c, xg)
    assert not c.deriv.exact
    zr = c.refine(1e-6)  # graph TV refines to the exact solve
    graph.vjp(c, xg)
    assert c.deriv.exact and np.array_equal(zr, graph.prox(xg, 0.2))
    chain = np.stack([np.arange(48 * 48 - 1), np.arange(1, 48 * 48)], 1)
    tree = fp.TreeTV(48 * 48, chain)
    _, c = tree.prox(xg, 0.2, keep=True)
    tree.vjp(c, xg)
    assert c.deriv.exact
    soft = fp.Soft(0.3)
    _, c = soft.prox(xg, 0.2, keep=True)
    soft.vjp(c, xg, refine=1e-8)  # ignored: closed form
    assert c.deriv == fp.DerivInfo(0.0, 0, True, 0.0)


def test_warning_once_per_point():
    x = _blocks(6)
    op = fp.TVIso()
    _, c = op.prox(x, 0.2, keep=True)  # tol 1e-6: z to 1e-3
    with pytest.warns(fp.AccuracyWarning, match="refine=") as rec:
        op.vjp(c, x)
    assert rec[0].filename == __file__
    _quiet(lambda: op.vjp(c, x))  # once per point
    _quiet(lambda: op.jvp(c, x))
    _, c = op.prox(x, 0.2, keep=True, ctx=c)  # a new point warns again
    with pytest.warns(fp.AccuracyWarning):
        op.jvp(c, x)
    _, c = op.prox(x, 0.2, keep=True, tol=1e-10, max_iter=10 ** 6)  # z to 1e-5: silent
    _quiet(lambda: op.vjp(c, x))
    _, c = op.prox(x, 0.2, keep=True, tol=1e-8, max_iter=10 ** 6)  # z to 1e-4
    with fp.inner_accuracy(1e-5, factor=1.0):  # the block's ztol is the threshold
        with pytest.warns(fp.AccuracyWarning):
            op.vjp(c, x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", fp.AccuracyWarning)
        op.vjp(op.prox(x, 0.2, keep=True)[1], x)


def test_refine_errors():
    x = _blocks(7, (2, 24, 24))
    _, c = fp.Soft(0.3).prox(x, 0.2, keep=True)
    with pytest.raises(TypeError):
        c.refine(1e-4)
    op = fp.TVAniso()
    _, c = op.prox(x, 0.2, keep=True)
    with pytest.raises(ValueError):
        c.refine()
    with pytest.raises(ValueError):
        c.refine(1e-4, tol=1e-8)
    with pytest.raises(ValueError):
        c.refine(1e-4, max_iter=0)
    with fp.inner_accuracy(1e-3):
        z = c.refine()  # the block's ztol, 1e-4
    assert c.info.zerr <= 1e-4 and z.shape == x.shape
    with pytest.warns(fp.ConvergenceWarning):
        fp.TVIso().prox(x, 0.2, keep=True)[1].refine(tol=1e-30, max_iter=2)
