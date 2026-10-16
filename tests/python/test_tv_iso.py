"""Isotropic TV of 2D and 3D images (fp.TVIso, fp.tv_iso): certified gaps
recomputed independently, determinism, strides, derivatives by conjugate
gradients, warm starts and errors. Agreement with CVXPY is in
test_fixtures.py."""

import numpy as np
import pytest

import fastprox as fp

SHAPES = [((18, 23), 2), ((3, 11, 13), 2), ((6, 7, 8), 3), ((2, 5, 6, 7), 3)]


def image(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.3 * rng.normal(size=shape)
    return x + (np.indices(shape)[-1] // 4 % 2)


def certificate(x, z, u, gamma, w, k):
    """Gap per image and the dual's infeasibility: |u_p| <= gamma, u_a = 0 at
    the last entry along a, and z = x - sum_a w_a D_a^T u_a."""
    w = np.broadcast_to(np.asarray(w, float), (k,))
    axes = list(range(x.ndim - k, x.ndim))
    img = tuple(axes)
    zz = x.copy()
    grads, feas = [], 0.0
    for a, ax in enumerate(axes):
        ua = u[a]
        last = np.take(ua, [ua.shape[ax] - 1], axis=ax)
        feas = max(feas, np.abs(last).max())
        shifted = np.concatenate([np.zeros_like(last), np.take(ua, range(ua.shape[ax] - 1), axis=ax)],
                                 axis=ax)
        zz = zz + w[a] * (ua - shifted)
        d = np.diff(z, axis=ax)
        grads.append(w[a] * np.concatenate([d, np.zeros_like(np.take(z, [0], axis=ax))], axis=ax))
    g = np.stack(grads)
    feas = max(feas, (np.sqrt((u ** 2).sum(axis=0)) - gamma).max())
    gap = (gamma * np.sqrt((g ** 2).sum(axis=0)) - (g * u).sum(axis=0)).sum(axis=img)
    return gap, feas, np.abs(zz - z).max()


@pytest.mark.parametrize("shape,k", SHAPES)
@pytest.mark.parametrize("gamma", [0.1, 0.6])
def test_certified_gap(shape, k, gamma):
    x = image(shape)
    w = np.linspace(0.7, 1.3, k)
    z, info = fp.tv_iso(x, gamma, weights=w, ndim=k, tol=1e-9, max_iter=200000)
    assert info.converged and info.dual.shape == (k, *shape)
    gap, feas, resid = certificate(x, z, info.dual, gamma, w, k)
    half = 0.5 * (x ** 2).reshape(*gap.shape, -1).sum(axis=-1) if gap.ndim else 0.5 * (x ** 2).sum()
    assert feas <= 1e-12 and resid <= 1e-12 * (1 + np.abs(x).max())
    assert np.all(gap <= 1e-9 * half * (1 + 1e-6) + 1e-13 * half)
    assert np.all(gap >= -1e-12 * half)


@pytest.mark.parametrize("shape,k", SHAPES)
def test_threads_bitwise_and_batching(shape, k):
    x = image(shape, 1)
    op = fp.TVIso(k, 1.2)
    outs = [op.prox(x, 0.4, threads=t, tol=1e-8, max_iter=100000) for t in (1, 2, 8)]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])
    if len(shape) > k:
        for b in range(shape[0]):
            assert np.array_equal(op.prox(x[b], 0.4, tol=1e-8, max_iter=100000), outs[0][b])


def test_strided_float32_and_special_cases():
    op = fp.TVIso(2)
    big = image((30, 80), 2)
    view = big[:, ::2]
    assert np.array_equal(op.prox(view, 0.5), op.prox(np.ascontiguousarray(view), 0.5))
    z64 = op.prox(view, 0.5, tol=1e-10, max_iter=100000)
    z32 = op.prox(view.astype(np.float32), 0.5, tol=1e-8, max_iter=100000)
    assert z32.dtype == np.float32 and np.abs(z32 - z64).max() <= 1e-3 * (1 + np.abs(view).max())
    x = image((1, 40), 3)  # one active axis: exactly 1D TV
    z, info = fp.tv_iso(x, 0.6)
    assert np.allclose(z, fp.tv1d(x, 0.6), rtol=0, atol=1e-13) and info.gap == 0
    assert np.array_equal(fp.tv_iso(x, 0.0)[0], x)


def test_isotropic_differs_from_anisotropic():
    x = image((20, 20), 4)
    zi, _ = fp.tv_iso(x, 0.5, tol=1e-10, max_iter=100000)
    za, _ = fp.tv_aniso(x, 0.5, tol=1e-10)
    assert np.abs(zi - za).max() > 1e-3


@pytest.mark.parametrize("shape,k", [((8, 9), 2), ((2, 6, 7), 2), ((4, 4, 5), 3)])
@pytest.mark.parametrize("conj", [False, True])
def test_derivatives(shape, k, conj):
    rng = np.random.default_rng(5)
    x = image(shape, 5)
    w = rng.uniform(0.6, 1.4, k)
    step = 2.0 if conj else 0.4
    op = fp.TVIso(k, w)
    run = (lambda o, v, s, **kw: o.prox_conj(v, s, **kw)) if conj else \
        (lambda o, v, s, **kw: o.prox(v, s, **kw))
    tight = dict(tol=1e-13, max_iter=3000000)
    z, ctx = run(op, x, step, keep=True, **tight)
    dx, dw, dg = rng.normal(size=shape), rng.normal(size=k), 0.2
    g = rng.normal(size=shape)
    jv = op.jvp(ctx, dx, dg, dw)
    res, its = ctx.deriv
    assert res <= 1e-10 and its > 0
    gx, grads = op.vjp(ctx, g)
    lhs = float((g * jv).sum())
    rhs = float((gx * dx).sum() + grads.gamma * dg + grads.weights @ dw)
    assert abs(lhs - rhs) <= 1e-8 * (1 + abs(lhs))
    h = 1e-5  # the solve error bounds the finite-difference accuracy
    zp = run(fp.TVIso(k, w + h * dw), x + h * dx, step + h * dg, **tight)
    zm = run(fp.TVIso(k, w - h * dw), x - h * dx, step - h * dg, **tight)
    fd = (zp - zm) / (2 * h)
    assert np.abs(fd - jv).max() <= 3e-3 * max(1.0, np.abs(fd).max())


def test_derivative_tolerance_controls():
    op = fp.TVIso(2)
    z, ctx = op.prox(image((20, 25), 6), 0.5, keep=True, tol=1e-10, max_iter=100000)
    op.vjp(ctx, np.ones_like(z), tol=1e-3, max_iter=3)
    assert ctx.deriv[1] <= 3
    with pytest.raises(ValueError):
        op.vjp(ctx, np.ones_like(z), tol=0.0)


def test_warm_starts():
    op = fp.TVIso(2)
    x = image((30, 40), 7)
    z, info = op.solve(x, 0.5, tol=1e-8, max_iter=100000)
    z2, info2 = op.solve(x, 0.5, tol=1e-8, max_iter=100000, warm_start=info.dual)
    # The seed is projected onto the ball again, which may move a dual on its
    # boundary by an ulp.
    assert info2.iters == 0 and np.allclose(z, z2, rtol=0, atol=1e-12)
    _, cold = op.solve(x + 0.01 * np.sin(7 * x), 0.5, tol=1e-8, max_iter=100000)
    _, warm = op.solve(x + 0.01 * np.sin(7 * x), 0.5, tol=1e-8, max_iter=100000,
                       warm_start=info.dual)
    assert warm.iters < cold.iters
    # A dual for a larger gamma is projected onto the smaller ball.
    z3, info3 = op.solve(x, 0.2, tol=1e-8, max_iter=100000, warm_start=info.dual)
    gap, feas, _ = certificate(x, z3, info3.dual, 0.2, 1.0, 2)
    assert feas <= 1e-12 and gap <= 1e-8 * 0.5 * (x ** 2).sum() * (1 + 1e-6)


def test_not_converged_and_errors():
    z, info = fp.tv_iso(image((30, 40), 8), 0.5, tol=1e-14, max_iter=3)
    assert not info.converged and info.iters == 3 and np.isfinite(z).all()
    with pytest.raises(ValueError):
        fp.TVIso(1)
    with pytest.raises(ValueError):
        fp.TVIso(2, [1.0, -1.0])
    with pytest.raises(ValueError):
        fp.TVIso(3).prox(np.zeros((4, 5)))
    xb = image((3, 10, 12), 9)
    xb[1, 2, 3] = np.nan
    zb, _ = fp.tv_iso(xb, 0.5)
    assert np.isnan(zb[1]).all() and not np.isnan(zb[[0, 2]]).any()
