# One set of derivative keywords on every operator (feedback U3): jvp and vjp
# take tol and max_iter everywhere. They control isotropic TV's conjugate
# gradients and are accepted and ignored by the closed-form operators, and
# ctx.deriv is set after every derivative.
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

import fastprox as fp

from test_operators import CASES, N, NAMES


def _image_ops():
    chain = np.stack([np.arange(63), np.arange(1, 64)], 1)
    return {"tv_aniso": (fp.TVAniso(2), (2, 8, 8)), "graph_tv": (fp.GraphTV(64, chain), (2, 64)),
            "tree_tv": (fp.TreeTV(64, chain), (2, 64))}


def _same_with_keywords(op, ctx, x, rng):
    dx, g = rng.normal(size=x.shape), rng.normal(size=x.shape)
    dth = rng.normal(size=op.nparams) if op.nparams else None
    kw = dict(tol=1e-8, max_iter=50)
    ctx.deriv = None
    assert np.array_equal(op.jvp(ctx, dx, 0.3, dth), op.jvp(ctx, dx, 0.3, dth, **kw))
    assert ctx.deriv == (0.0, 0)
    ctx.deriv = None
    (a, ga), (b, gb) = op.vjp(ctx, g), op.vjp(ctx, g, **kw)
    assert np.array_equal(a, b) and ga.gamma == gb.gamma and np.array_equal(ga.params, gb.params)
    assert ctx.deriv == (0.0, 0)


@pytest.mark.parametrize("name", NAMES)
def test_closed_form_accept_and_ignore(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=1.5, size=(3, N))
    _, ctx = op.prox(x, gamma, keep=True)
    _same_with_keywords(op, ctx, x, rng)


@pytest.mark.parametrize("name", sorted(_image_ops()))
def test_image_closed_form(name):
    op, shape = _image_ops()[name]
    rng = np.random.default_rng(1)
    x = rng.normal(size=shape)
    _, ctx = op.prox(x, 0.4, keep=True)
    _same_with_keywords(op, ctx, x, rng)


def test_spectral_accepts():
    op = fp.NuclearNorm(0.5)
    x = np.random.default_rng(2).normal(size=(3, 6, 4))
    _, ctx = op.prox(x, 1.0, keep=True)
    h = np.random.default_rng(3).normal(size=x.shape)
    assert ctx.deriv is None
    assert np.array_equal(op.jvp(ctx, h), op.jvp(ctx, h, tol=1e-8, max_iter=50))
    a, b = op.vjp(ctx, h)[0], op.vjp(ctx, h, tol=1e-8, max_iter=50)[0]
    assert np.array_equal(a, b) and ctx.deriv == (0.0, 0)


def test_tviso_uses_them():
    op = fp.TVIso(2)
    x = np.random.default_rng(4).normal(size=(2, 16, 16))
    _, ctx = op.prox(x, 0.4, keep=True, tol=1e-10, max_iter=100000)
    g = np.random.default_rng(5).normal(size=x.shape)
    loose = op.vjp(ctx, g, tol=1e-2)[0]
    res, its = ctx.deriv
    assert 0.0 < res <= 1e-2 and its > 0
    tight = op.vjp(ctx, g, tol=1e-12)[0]
    assert ctx.deriv[0] <= 1e-12 and not np.array_equal(loose, tight)
    op.jvp(ctx, g, max_iter=1, tol=1e-14)
    assert ctx.deriv[1] == 1


@pytest.mark.parametrize("bad", [dict(tol=0.0), dict(tol=-1.0), dict(max_iter=0), dict(max_iter=2.5),
                                 dict(max_iter=True)])
def test_errors_everywhere(bad):
    x = np.random.default_rng(6).normal(size=(2, 8, 8))
    for op in (fp.Soft(0.3), fp.TVAniso(2), fp.TVIso(2)):
        _, ctx = op.prox(x, 0.4, keep=True)
        with pytest.raises(ValueError):
            op.jvp(ctx, x, **bad)
        with pytest.raises(ValueError):
            op.vjp(ctx, x, **bad)
    op = fp.NuclearNorm(0.5)
    _, ctx = op.prox(x, 1.0, keep=True)
    with pytest.raises(ValueError):
        op.vjp(ctx, x, **bad)


def _spy(monkeypatch):
    """Records ctx.deriv after each TVIso.vjp."""
    seen = []
    real = fp.Operator.vjp

    def vjp(self, ctx, g, **kw):
        out = real(self, ctx, g, **kw)
        if isinstance(self, fp.TVIso):
            seen.append((kw.get("tol"), ctx.deriv))
        return out

    monkeypatch.setattr(fp.Operator, "vjp", vjp)
    return seen


def test_torch_generic(monkeypatch):
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    seen = _spy(monkeypatch)
    x = torch.tensor(np.random.default_rng(7).normal(size=(2, 16, 16)), requires_grad=True)
    fpt.prox(fp.TVIso(2), x, 0.4, tol=1e-10, max_iter=100000, deriv_tol=1e-4).sum().backward()
    assert seen and seen[-1][0] == 1e-4 and seen[-1][1][0] <= 1e-4
    fpt.prox(fp.Soft(0.3), x, 0.4, deriv_tol=1e-4, deriv_max_iter=7).sum().backward()  # ignored


def test_jax_generic(monkeypatch):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import fastprox.jax as fpj
    seen = _spy(monkeypatch)
    x = jnp.asarray(np.random.default_rng(8).normal(size=(2, 16, 16)), jnp.float32)
    f = lambda x: fpj.prox(fp.TVIso(2), x, 0.4, tol=1e-10, max_iter=100000, deriv_tol=1e-4).sum()  # noqa: E731
    jax.block_until_ready(jax.grad(f)(x))
    assert seen and seen[-1][0] == 1e-4 and seen[-1][1][0] <= 1e-4
