"""Anisotropic TV of 2D and 3D images (fp.TVAniso, fp.tv_aniso): certified
gaps recomputed independently, determinism, batching and strides, derivatives,
warm starts and errors. Agreement with CVXPY is in test_fixtures.py."""

import numpy as np
import pytest

import fastprox as fp

SHAPES = [((24, 31), 2), ((3, 17, 20), 2), ((9, 10, 11), 3), ((2, 6, 7, 8), 3)]


def image(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.3 * rng.normal(size=shape)
    return x + (np.indices(shape)[-1] // 5 % 2)


def diffs(z, axis):
    return np.diff(z, axis=axis)


def certificate(x, z, dual, gamma, w, k):
    """Independent gap P(z) - D(q) per image, and the dual's infeasibility:
    q_a must be D_a^T p_a with |p_a| <= gamma w_a (running sums along the axis
    stay within the radius and end at 0)."""
    w = np.broadcast_to(np.asarray(w, float), (k,))
    axes = range(x.ndim - k, x.ndim)
    feas = 0.0
    P = 0.5 * ((z - x) ** 2)
    s = dual.sum(axis=0)
    D = x * s - 0.5 * s * s
    img = tuple(axes)
    P = P.sum(axis=img)
    D = D.sum(axis=img)
    for a, ax in enumerate(axes):
        lam = gamma * w[a]
        P = P + lam * np.abs(diffs(z, ax)).sum(axis=img)
        p = np.cumsum(dual[a], axis=ax)
        inner = np.take(p, range(p.shape[ax] - 1), axis=ax)
        last = np.take(p, [p.shape[ax] - 1], axis=ax)
        feas = max(feas, (np.abs(inner) - lam).max(initial=0.0), np.abs(last).max())
    return P - D, feas


@pytest.mark.parametrize("shape,k", SHAPES)
@pytest.mark.parametrize("gamma", [0.1, 0.7])
def test_certified_gap(shape, k, gamma):
    x = image(shape)
    w = np.linspace(0.5, 1.5, k)
    z, info = fp.tv_aniso(x, gamma, weights=w, ndim=k, tol=1e-10, max_iter=100000)
    assert info.converged and info.dual.shape == (k, *shape)
    gap, feas = certificate(x, z, info.dual, gamma, w, k)
    half = 0.5 * (x.reshape(-1, *shape[-k:]) ** 2).reshape(gap.shape + (-1,)).sum(axis=-1) \
        if gap.ndim else 0.5 * (x ** 2).sum()
    assert feas <= 1e-12 * (1 + np.abs(x).max())
    assert np.all(gap <= 1e-10 * half * (1 + 1e-6) + 1e-13 * half)
    assert np.all(gap >= -1e-12 * half)
    assert info.gap <= 1e-10 * 0.5 * (x ** 2).sum()


@pytest.mark.parametrize("shape,k", SHAPES)
def test_threads_bitwise_and_batching(shape, k):
    x = image(shape, 1)
    op = fp.TVAniso(k, 1.3)
    outs = [op.prox(x, 0.4, threads=t, tol=1e-9) for t in (1, 2, 8)]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])
    if len(shape) > k:
        for b in range(shape[0]):
            assert np.array_equal(op.prox(x[b], 0.4, tol=1e-9), outs[0][b])


def test_strided_and_float32():
    op = fp.TVAniso(2)
    big = image((40, 100), 2)
    view = big[:, ::2]
    assert np.array_equal(op.prox(view, 0.5), op.prox(np.ascontiguousarray(view), 0.5))
    rev = big[::-1, ::-3]
    assert np.array_equal(op.prox(rev, 0.5), op.prox(np.ascontiguousarray(rev), 0.5))
    z64 = op.prox(view, 0.5, tol=1e-10)
    z32 = op.prox(view.astype(np.float32), 0.5, tol=1e-8)
    assert z32.dtype == np.float32
    assert np.abs(z32 - z64).max() <= 1e-4 * (1 + np.abs(view).max())


def test_special_cases():
    x = image((7, 30), 3)
    z, info = fp.tv_aniso(x, 0.6, weights=[0.0, 1.0])
    assert np.array_equal(z, fp.tv1d(x, 0.6)) and info.gap == 0 and info.iters == 0
    assert np.array_equal(fp.tv_aniso(x, 0.0)[0], x)
    xb = image((3, 10, 12), 4)
    xb[1, 4, 5] = np.nan
    zb, _ = fp.tv_aniso(xb, 0.5)
    assert np.isnan(zb[1]).all() and not np.isnan(zb[[0, 2]]).any()
    xb[1, 4, 5] = np.inf
    with pytest.raises(ValueError):
        fp.tv_aniso(xb, 0.5)


@pytest.mark.parametrize("shape,k,w", [((20, 24, 40), 3, 0.8), ((96, 120), 2, 0.8), ((37, 21, 30), 3, 0.8),
                                       ((131, 77), 2, 0.8), ((37, 21, 30), 3, [0.0, 0.8, 0.8])])
def test_tight_solves_threads_and_warm(shape, k, w):
    """Tight solves on images large enough that the passes and the polish
    split over threads (and its slabs along axis 0 over several, uneven or
    with no axis-0 edges): the same bits for any thread count, an
    independent certificate, and a warm start from the dual returned stops
    at once."""
    x = image(shape, 10)
    op = fp.TVAniso(k, w)
    runs = [op.solve(x, 0.3, tol=1e-12, max_iter=100000, threads=t) for t in (1, 3, 8)]
    z, info = runs[0]
    assert info.converged
    for zt, it in runs[1:]:
        assert np.array_equal(z, zt) and np.array_equal(info.dual, it.dual) and it.iters == info.iters
    gap, feas = certificate(x, z, info.dual, 0.3, w, k)
    half = 0.5 * (x ** 2).sum()
    assert feas <= 1e-12 * (1 + np.abs(x).max())
    assert -1e-12 * half <= gap <= 1e-12 * half * (1 + 1e-6)
    z2, info2 = op.solve(x, 0.3, tol=1e-12, warm_start=info.dual)
    assert info2.iters == 0 and np.array_equal(z, z2)


def _fd_case(shape, k, conj, seed):
    rng = np.random.default_rng(seed)
    x = image(shape, seed)
    w = rng.uniform(0.5, 1.5, k)
    return x, w, rng


@pytest.mark.parametrize("shape,k", SHAPES)
@pytest.mark.parametrize("conj", [False, True])
def test_derivatives(shape, k, conj):
    x, w, rng = _fd_case(shape, k, conj, 5)
    step = 2.0 if conj else 0.4
    op = fp.TVAniso(k, w)
    run = (lambda o, v, s, **kw: o.prox_conj(v, s, **kw)) if conj else \
        (lambda o, v, s, **kw: o.prox(v, s, **kw))
    z, ctx = run(op, x, step, keep=True, tol=1e-14, max_iter=200000)
    dx, dw, dg = rng.normal(size=shape), rng.normal(size=k), 0.3
    g = rng.normal(size=shape)
    jv = op.jvp(ctx, dx, dg, dw)
    gx, grads = op.vjp(ctx, g)
    lhs = float((g * jv).sum())
    rhs = float((gx * dx).sum() + grads.gamma * dg + grads.weights @ dw)
    assert abs(lhs - rhs) <= 1e-10 * (1 + abs(lhs))
    h = 1e-6
    zp = run(fp.TVAniso(k, w + h * dw), x + h * dx, step + h * dg, tol=1e-14, max_iter=200000)
    zm = run(fp.TVAniso(k, w - h * dw), x - h * dx, step - h * dg, tol=1e-14, max_iter=200000)
    fd = (zp - zm) / (2 * h)
    assert np.abs(fd - jv).max() <= 1e-6 * max(1.0, np.abs(fd).max())


@pytest.mark.parametrize("shape,k", SHAPES[:3])
def test_firmly_nonexpansive(shape, k):
    op = fp.TVAniso(k)
    x, y = image(shape, 6), image(shape, 7)
    px, py = op.prox(x, 0.5, tol=1e-12), op.prox(y, 0.5, tol=1e-12)
    d = px - py
    assert (d * d).sum() <= (d * (x - y)).sum() + 1e-9 * (x * x).sum()


def test_warm_starts():
    op = fp.TVAniso(2)
    x = image((40, 50), 8)
    z, info = op.solve(x, 0.5, tol=1e-10)
    z2, info2 = op.solve(x, 0.5, tol=1e-10, warm_start=info.dual)
    assert info2.iters == 0 and np.array_equal(z, z2)
    x2 = x + 0.01 * np.sin(7 * x)
    _, cold = op.solve(x2, 0.5, tol=1e-10)
    _, warm = op.solve(x2, 0.5, tol=1e-10, warm_start=info.dual)
    assert warm.iters < cold.iters
    # A context reuses its workspace; warm=True starts from its last dual.
    z3, ctx = op.prox(x, 0.5, keep=True, tol=1e-10)
    z4, ctx = op.prox(x, 0.5, keep=True, ctx=ctx, warm=True, tol=1e-10)
    assert ctx.info.iters == 0 and np.array_equal(z3, z4)
    with pytest.raises(ValueError):
        op.solve(x, 0.5, warm_start=np.zeros((3, 40, 50)))


@pytest.mark.parametrize("shape,k", [((30, 40), 2), ((10, 12, 14), 3)])
def test_infeasible_warm_start(shape, k):
    """A warm dual is made feasible before use. Unchecked, q = (x - mean(x),
    0, ...) gives z = mean(x) with gap 0, a wrong point certified at once. Random and huge duals must not break the result either, nor the NaN
    dual a NaN image leaves for the next warm call; a non-finite seed is
    rejected."""
    x = image(shape, 11)
    op = fp.TVAniso(k)
    z0, _ = op.solve(x, 0.4, tol=1e-12, max_iter=100000)
    half = 0.5 * (x ** 2).sum()
    rng = np.random.default_rng(12)

    def check(z, info):
        assert info.converged and np.isfinite(z).all()
        gap, feas = certificate(x, z, info.dual, 0.4, 1.0, k)
        assert feas <= 1e-12 * (1 + np.abs(x).max())
        assert -1e-12 * half <= gap <= 1e-12 * half * (1 + 1e-6)
        assert np.linalg.norm(z - z0) <= 2e-6 * np.linalg.norm(x)

    bad = np.zeros((k, *shape))
    bad[0] = x - x.mean()
    for warm in (bad, 50.0 * rng.normal(size=(k, *shape))):
        z, info = op.solve(x, 0.4, tol=1e-12, max_iter=100000, warm_start=warm)
        assert info.iters > 0
        check(z, info)
    nonfinite = rng.normal(size=(k, *shape))
    nonfinite[0, 2, 3] = np.nan
    with pytest.raises(ValueError):
        op.solve(x, 0.4, warm_start=nonfinite)
    xn = x.copy()
    xn[2, 3] = np.nan
    _, ctx = op.prox(xn, 0.4, keep=True, tol=1e-12, max_iter=100000)
    z, ctx = op.prox(x, 0.4, keep=True, ctx=ctx, warm=True, tol=1e-12, max_iter=100000)
    check(z, ctx.info)


def test_not_converged_is_reported():
    z, info = fp.tv_aniso(image((40, 50), 9), 0.5, tol=1e-14, max_iter=2)
    assert not info.converged and info.iters == 2 and np.isfinite(z).all() and info.gap > 0


def test_errors():
    with pytest.raises(ValueError):
        fp.TVAniso(4)
    with pytest.raises(ValueError):
        fp.TVAniso(2, [1.0, -1.0])
    with pytest.raises(ValueError):
        fp.TVAniso(2, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        fp.TVAniso(3).prox(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        fp.tv_aniso(np.zeros((4, 5)), 0.5, tol=-1.0)
    with pytest.raises(TypeError):
        fp.tv_aniso(np.zeros((4, 5), dtype=np.int64))
