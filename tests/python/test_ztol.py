# Accuracy in the units of z (feedback U1): ztol= asks for
# |z - z*| <= ztol |x| per image, and IterInfo.zerr reports the error the gaps
# certify, max_i sqrt(2 gap_i) / |x_i|.
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

import fastprox as fp


def _grid(n):
    e = [(i * n + j, i * n + j + 1) for i in range(n) for j in range(n - 1)]
    e += [(i * n + j, (i + 1) * n + j) for i in range(n - 1) for j in range(n)]
    return np.array(e)


def _batch(seed=0, shape=(2, 3, 24, 24)):
    rng = np.random.default_rng(seed)
    blocks = np.kron(rng.random(shape[:-2] + (4, 4)), np.ones((shape[-2] // 4, shape[-1] // 4)))
    return blocks + 0.1 * rng.normal(size=shape)


def _relerr(z, ref, x, axes):
    return np.max(np.sqrt(np.sum((z - ref) ** 2, axis=axes) / np.sum(x * x, axis=axes)))


@pytest.mark.parametrize("Op", [fp.TVIso, fp.TVAniso])
def test_ztol_is_tol_squared(Op):
    x = _batch()
    op = Op(2)
    kw = dict(max_iter=100000)
    z1, i1 = op.solve(x, 0.4, ztol=1e-5, **kw)
    z2, i2 = op.solve(x, 0.4, tol=1e-10, **kw)
    assert np.array_equal(z1, z2) and i1.gap == i2.gap and i1.zerr == i2.zerr
    assert np.array_equal(op.prox(x, 0.4, ztol=1e-4, **kw), op.prox(x, 0.4, tol=1e-8, **kw))
    assert np.array_equal(op.prox_conj(x, 0.4, ztol=1e-4, **kw), op.prox_conj(x, 0.4, tol=1e-8, **kw))
    f = fp.tv_iso if Op is fp.TVIso else fp.tv_aniso
    assert np.array_equal(f(x, 0.4, ztol=1e-4, **kw)[0], f(x, 0.4, tol=1e-8, **kw)[0])
    # The default is still tol = 1e-6.
    assert np.array_equal(op.prox(x, 0.4, **kw), op.prox(x, 0.4, tol=1e-6, **kw))


@pytest.mark.parametrize("Op", [fp.TVIso, fp.TVAniso])
@pytest.mark.parametrize("ztol", [1e-2, 1e-3, 1e-4])
def test_zerr_bounds_the_error(Op, ztol):
    """|z - ref| <= |z - z*| + |ref - z*| <= zerr + zerr_ref: the reference
    carries its own certificate, as isotropic TV need not reach 1e-20."""
    x = np.random.default_rng(1).normal(size=(2, 3, 24, 24))  # isotropic TV is slow on flat blocks
    op = Op(2)
    ref, ri = op.solve(x, 0.4, tol=1e-20, max_iter=20000, warn=False)
    assert ri.zerr < 1e-2 * ztol
    z, info = op.solve(x, 0.4, ztol=ztol, max_iter=100000)
    assert info.converged and info.zerr <= ztol
    assert _relerr(z, ref, x, (-2, -1)) <= (info.zerr + ri.zerr) * (1 + 1e-6)
    # Stopped early: zerr reports the shortfall, still a bound.
    with pytest.warns(fp.ConvergenceWarning):
        z, info = op.solve(x, 0.4, tol=1e-14, max_iter=3)
    assert not info.converged and info.zerr > 1e-7
    assert _relerr(z, ref, x, (-2, -1)) <= (info.zerr + ri.zerr) * (1 + 1e-6)


@pytest.mark.parametrize("offset", [0.0, 50.0])
@pytest.mark.parametrize("ztol", [1e-1, 1e-2, 1e-3, 1e-4])
def test_graph_ztol(ztol, offset):
    """Graph TV maps ztol through its per-node bound, so the guarantee holds
    whatever the scale and offset of x; zerr bounds the error too."""
    n = 24
    g = fp.GraphTV(n * n, _grid(n))
    x = _batch(2).reshape(6, n * n) + offset
    x[1] *= 30.0
    ref, _ = g.solve(x, 0.4)
    z, info = g.solve(x, 0.4, ztol=ztol)
    err = _relerr(z, ref, x, -1)
    assert err <= ztol and err <= info.zerr * (1 + 1e-6) + 1e-12
    z, info = fp.graph_tv(x, _grid(n), 0.4, ztol=ztol)
    assert _relerr(z, ref, x, -1) <= ztol
    # Constant signals are exact at any tolerance; tree TV ignores the tolerance.
    assert np.array_equal(g.prox(np.ones((2, n * n)), 0.4, ztol=ztol), np.ones((2, n * n)))
    chain = np.stack([np.arange(n * n - 1), np.arange(1, n * n)], 1)
    t = fp.TreeTV(n * n, chain)
    assert np.array_equal(t.prox(x, 0.4, ztol=ztol), t.prox(x, 0.4))


def test_zerr_special_cases():
    x = _batch(3, (3, 16, 16))
    x[1] = 0.0  # an image of zeros: exact, error 0
    for op in (fp.TVIso(2), fp.TVAniso(2)):
        _, info = op.solve(x, 0.4, ztol=1e-3)
        _, alone = op.solve(x[[0, 2]], 0.4, ztol=1e-3)
        assert info.zerr == alone.zerr
        x[2, 3, 3] = np.nan
        _, info = op.solve(x, 0.4, ztol=1e-3)
        assert np.isnan(info.zerr) and np.isnan(info.gap)
        x[2, 3, 3] = 0.0
        _, info = op.solve(np.zeros((0, 16, 16)), 0.4, ztol=1e-3)
        assert info.zerr == 0.0
        _, ctx = op.prox(x, 0.4, ztol=1e-3, keep=True)
        assert ctx.info.zerr == op.solve(x, 0.4, ztol=1e-3)[1].zerr
    # float32 input: the norms in float64, the bound still holds.
    x32 = _batch(4, (2, 16, 16)).astype(np.float32)
    ref, _ = fp.TVIso(2).solve(x32.astype(np.float64), 0.4, tol=1e-20, max_iter=100000, warn=False)
    z, info = fp.TVIso(2).solve(x32, 0.4, ztol=1e-3)
    assert _relerr(z.astype(np.float64), ref, x32.astype(np.float64), (-2, -1)) <= 1e-3


@pytest.mark.parametrize("bad", [dict(tol=1e-8, ztol=1e-4), dict(ztol=1e-200), dict(ztol=-1.0),
                                 dict(ztol=np.inf), dict(ztol=np.nan)])
def test_ztol_errors(bad):
    x = _batch(5, (16, 16))
    for f in (fp.TVIso(2).prox, fp.TVAniso(2).solve, fp.GraphTV(256, _grid(16)).prox):
        with pytest.raises(ValueError):
            f(x if f.__self__.ndim == 2 else x.reshape(-1), 0.4, **bad)
    with pytest.warns(fp.ConvergenceWarning):
        fp.TVIso(2).prox(x, 0.4, ztol=0.0, max_iter=5)  # 0 is allowed: run to max_iter


def test_ztol_torch():
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    x = torch.tensor(_batch(6))
    for f, Op in ((fpt.tv_iso, fp.TVIso), (fpt.tv_aniso, fp.TVAniso)):
        z1, i1 = f(x, 0.4, ztol=1e-4, max_iter=100000, return_info=True)
        z2, i2 = f(x, 0.4, tol=1e-8, max_iter=100000, return_info=True)
        assert torch.equal(z1, z2) and i1.zerr == i2.zerr and i1.zerr <= 1e-4
        ref, ri = Op(2).solve(x.numpy(), 0.4, tol=1e-20, max_iter=20000, warn=False)
        assert _relerr(z1.numpy(), ref, x.numpy(), (-2, -1)) <= (i1.zerr + ri.zerr) * (1 + 1e-6)
        _, e = f(torch.zeros(0, 8, 8, dtype=torch.float64), 0.4, ztol=1e-3, return_info=True)
        assert e.zerr == 0.0
        with pytest.raises(ValueError):
            f(x, 0.4, tol=1e-8, ztol=1e-4)
    a = fpt.prox(fp.TVIso(2), x, 0.4, ztol=1e-4, max_iter=100000)
    b = fpt.prox(fp.TVIso(2), x, 0.4, tol=1e-8, max_iter=100000)
    assert torch.equal(a, b)


@pytest.fixture
def x64():
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("jit", [False, True])
def test_ztol_jax(jit, x64):
    """In float64: in float32 the gap, a difference of two large sums, is
    itself rounded to about 1e-3 of its value on these images."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import fastprox.jax as fpj
    x = jnp.asarray(_batch(7))
    for f, Op in ((fpj.tv_iso, fp.TVIso), (fpj.tv_aniso, fp.TVAniso)):
        g = (lambda x, **kw: f(x, 0.4, max_iter=100000, return_info=True, **kw))  # noqa: E731
        if jit:
            g = jax.jit(g, static_argnames=("tol", "ztol"))
        z1, i1 = g(x, ztol=1e-3)
        z2, i2 = g(x, tol=1e-6)
        assert np.array_equal(np.asarray(z1), np.asarray(z2))
        assert float(i1.zerr) == float(i2.zerr) and float(i1.zerr) <= 1e-3
        ref, ri = Op(2).solve(np.asarray(x), 0.4, tol=1e-20, max_iter=20000, warn=False)
        assert _relerr(np.asarray(z1), ref, np.asarray(x), (-2, -1)) <= (float(i1.zerr) + ri.zerr) * (1 + 1e-6)
    a = fpj.prox(fp.TVAniso(2), np.asarray(x, np.float64), 0.4, ztol=1e-4)
    b = fpj.prox(fp.TVAniso(2), np.asarray(x, np.float64), 0.4, tol=1e-8)
    assert np.array_equal(np.asarray(a), np.asarray(b))
