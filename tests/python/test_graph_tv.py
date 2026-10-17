"""Graph TV (parametric max-flow): optimality by an independent KKT check,
agreement with 1D TV on chains and anisotropic TV on grids, the precision
and gap it reports, derivatives, batching, threads and errors."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref


def random_graph(n, m, seed, zero_frac=0.1, parallel=True):
    """A ring (so the graph is connected) plus m random chords, with weights
    in [0.2, 2] and some zeros; repeated edges when parallel."""
    rng = np.random.default_rng(seed)
    ring = np.stack([np.arange(n), (np.arange(n) + 1) % n], 1) if n > 2 else np.array([[0, 1]])[: n - 1]
    a = rng.integers(0, n, m)
    b = (a + rng.integers(1, n, m)) % n if n > 1 else a
    e = np.concatenate([ring, np.stack([a, b], 1)]) if n > 1 else np.zeros((0, 2), int)
    if parallel and len(e) > 3:
        e = np.concatenate([e, e[:2]])
    w = rng.uniform(0.2, 2.0, len(e))
    w[rng.random(len(e)) < zero_frac] = 0.0
    return e, w


def grid_edges(shape):
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    parts = []
    for a in range(len(shape)):
        lo = np.take(idx, range(shape[a] - 1), axis=a).ravel()
        hi = np.take(idx, range(1, shape[a]), axis=a).ravel()
        parts.append(np.stack([lo, hi], 1))
    return np.concatenate(parts)


def blocky(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=scale, size=n)
    return x + scale * (np.arange(n) // 4 % 3)


@pytest.mark.parametrize("n,m", [(1, 0), (2, 0), (3, 2), (12, 10), (40, 80), (120, 300)])
@pytest.mark.parametrize("gamma", [0.05, 0.6, 5.0])
def test_kkt(n, m, gamma):
    for seed in range(3):
        e, w = random_graph(n, m, seed)
        for scale in (0.1, 1.0, 30.0):
            x = blocky(n, seed + 7, scale)
            z, info = fp.GraphTV(n, e, w).solve(x, gamma)
            assert info.converged
            assert ref.graph_tv_kkt_residual(x, z, gamma, e, w) < 1e-10
            assert info.gap <= 1e-20 + 1e-13 * np.dot(x, x)
            # The dual: z = x - q.
            np.testing.assert_allclose(info.dual[0], x - z, atol=1e-11 * (1 + np.abs(x).max()))


def test_scalar_and_unit_weights():
    e, _ = random_graph(30, 40, 1)
    x = blocky(30, 2)
    z1 = fp.GraphTV(30, e).prox(x, 0.7)
    z2 = fp.GraphTV(30, e, 1.0).prox(x, 0.7)
    z3 = fp.GraphTV(30, e, np.ones(len(e))).prox(x, 0.7)
    z4 = fp.GraphTV(30, e, 0.5).prox(x, 1.4)
    np.testing.assert_array_equal(z1, z2)
    np.testing.assert_array_equal(z1, z3)
    np.testing.assert_allclose(z1, z4, atol=1e-14)
    assert ref.graph_tv_kkt_residual(x, z1, 0.7, e, 1.0) < 1e-10


@pytest.mark.parametrize("n", [2, 7, 100, 3000])
@pytest.mark.parametrize("gamma", [1e-3, 0.3, 4.0])
def test_chains_match_tv1d(n, gamma):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n).cumsum()
    w = rng.uniform(0.0, 2.0, n - 1)
    e = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
    z, info = fp.GraphTV(n, e, w).solve(x, gamma)
    np.testing.assert_allclose(z, fp.tv1d(x, gamma, weights=w), atol=1e-12 * (1 + np.abs(x).max()))
    # Edges in either direction and any order give the same problem.
    perm = rng.permutation(n - 1)
    e2 = e[perm][:, ::-1]
    np.testing.assert_allclose(fp.GraphTV(n, e2, w[perm]).prox(x, gamma), z, atol=1e-12 * (1 + np.abs(x).max()))


@pytest.mark.parametrize("shape", [(17, 23), (40, 31), (6, 7, 9)])
@pytest.mark.parametrize("gamma", [0.05, 0.4])
def test_grids_match_tv_aniso(shape, gamma):
    x = np.random.default_rng(3).normal(size=shape) + (np.indices(shape).sum(0) // 5 % 2)
    k = len(shape)
    z, info = fp.GraphTV(x.size, grid_edges(shape)).solve(x.ravel(), gamma)
    za, ia = fp.tv_aniso(x, gamma, ndim=k, tol=1e-15, max_iter=10**6)
    assert np.abs(z.reshape(shape) - za).max() < 1e-7


def test_precision():
    """tol bounds the error at every node by tol * (max x - min x)."""
    shape = (48, 64)
    x = np.random.default_rng(4).normal(size=shape).ravel() + np.repeat(np.arange(6), 512)
    op = fp.GraphTV(x.size, grid_edges(shape))
    z0, i0 = op.solve(x, 0.3)
    rng = np.ptp(x)
    for tol in (1e-10, 1e-6, 1e-3, 1e-1):
        z, info = op.solve(x, 0.3, tol=tol)
        assert info.converged
        assert np.abs(z - z0).max() <= tol * rng
        assert info.iters <= i0.iters
        assert info.gap >= 0


def test_max_iter_is_reported():
    shape = (30, 30)
    x = np.random.default_rng(5).normal(size=shape).ravel()
    op = fp.GraphTV(x.size, grid_edges(shape))
    z, info = op.solve(x, 0.3, max_iter=1)
    assert not info.converged and info.iters == 1
    assert np.all(np.isfinite(z))
    z0, i0 = op.solve(x, 0.3)
    assert info.gap > i0.gap


def test_batching_threads_strides_float32():
    e, w = random_graph(50, 90, 6)
    op = fp.GraphTV(50, e, w)
    x = np.random.default_rng(7).normal(size=(4, 9, 50)) * 3
    rows = np.stack([op.prox(r, 0.8) for r in x.reshape(-1, 50)]).reshape(x.shape)
    outs = [op.prox(x, 0.8, threads=t) for t in (1, 2, 8)]
    for o in outs:
        assert o.tobytes() == rows.tobytes()
    v = x[::-1, ::2]
    np.testing.assert_array_equal(op.prox(v, 0.8), op.prox(np.ascontiguousarray(v), 0.8))
    y = x.copy()
    op.prox(y, 0.8, out=y)
    np.testing.assert_array_equal(y, rows)
    z32 = op.prox(x.astype(np.float32), 0.8)
    assert z32.dtype == np.float32
    np.testing.assert_allclose(z32, op.prox(x.astype(np.float32).astype(np.float64), 0.8), atol=1e-5)
    # Gradients do not depend on the thread count either.
    g = np.sin(np.arange(x.size)).reshape(x.shape)
    res = []
    for t in (1, 3, 8):
        _, ctx = op.prox(x, 0.8, keep=True, threads=t)
        gx, gr = op.vjp(ctx, g)
        res.append(gx.tobytes() + np.append(gr.params, gr.gamma).tobytes())
    assert res[0] == res[1] == res[2]


def test_special_cases():
    e, w = random_graph(20, 30, 8)
    op = fp.GraphTV(20, e, w)
    x = np.random.default_rng(9).normal(size=20)
    np.testing.assert_array_equal(op.prox(x, 0.0), x)
    # No edges, isolated nodes and a single node.
    np.testing.assert_array_equal(fp.GraphTV(5, np.zeros((0, 2), int)).prox(x[:5], 3.0), x[:5])
    np.testing.assert_array_equal(fp.GraphTV(1, np.zeros((0, 2), int)).prox(x[:1], 3.0), x[:1])
    z = fp.GraphTV(6, [[0, 1], [1, 2]]).prox(x[:6], 0.4)
    np.testing.assert_array_equal(z[3:], x[3:6])
    assert ref.graph_tv_kkt_residual(x[:6], z, 0.4, [[0, 1], [1, 2]], 1.0) < 1e-10
    # Large gamma: the mean of each component.
    z = op.prox(x, 1e6)
    np.testing.assert_allclose(z, np.full(20, x.mean()), atol=1e-9)
    # Constant input.
    np.testing.assert_array_equal(op.prox(np.full(20, 2.5), 1.0), np.full(20, 2.5))
    # NaN anywhere makes the signal NaN; infinities are rejected.
    xn = x.copy()
    xn[3] = np.nan
    assert np.all(np.isnan(op.prox(xn, 1.0)))
    xi = x.copy()
    xi[3] = np.inf
    with pytest.raises(ValueError):
        op.prox(xi, 1.0)


def _fd_case(seed, conj, scalar):
    e, w = random_graph(14, 12, seed, zero_frac=0.0, parallel=False)
    theta = np.array([0.9]) if scalar else w
    build = lambda t: fp.GraphTV(14, e, t[0] if scalar else t)
    x = np.random.default_rng(seed).normal(scale=1.5, size=(3, 14))
    return build, theta, x


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
def test_derivatives(conj, scalar):
    h = 1e-6
    for seed in range(6):
        build, theta, x = _fd_case(seed, conj, scalar)
        op = build(theta)
        gamma = 0.6
        prox = op.prox_conj if conj else op.prox
        z, ctx = prox(x, gamma, keep=True)
        rng = np.random.default_rng(seed + 100)
        dx = rng.normal(size=x.shape)
        dth = rng.normal(size=theta.size)
        dg = 0.37
        dz = op.jvp(ctx, dx, dg, dth)
        g = rng.normal(size=x.shape)
        gx, grads = op.vjp(ctx, g)
        lhs = np.vdot(g, dz)
        rhs = np.vdot(gx, dx) + grads.gamma * dg + np.vdot(grads.params, dth)
        assert lhs == pytest.approx(rhs, rel=1e-11, abs=1e-11)

        def f(t):
            o = build(theta + t * dth)
            p = o.prox_conj if conj else o.prox
            return p(x + t * dx, gamma + t * dg)

        fd = (f(h) - f(-h)) / (2 * h)
        np.testing.assert_allclose(dz, fd, rtol=1e-5, atol=1e-5 * (1 + np.abs(fd).max()))


def test_derivatives_on_regions():
    """The Jacobian averages over the regions where z is constant."""
    e = grid_edges((12, 12))
    x = np.random.default_rng(10).normal(size=144) + np.repeat([0.0, 3.0], 72)
    op = fp.GraphTV(144, e)
    z, ctx = op.prox(x, 0.8, keep=True)
    J = np.stack([op.jvp(ctx, np.eye(144)[i]) for i in range(144)], 1)
    vals, lab = np.unique(np.round(z, 9), return_inverse=True)
    for r in range(len(vals)):
        idx = np.flatnonzero(lab == r)
        sub = J[np.ix_(idx, idx)]
        np.testing.assert_allclose(sub, np.full(sub.shape, 1.0 / idx.size), atol=1e-14)
    np.testing.assert_allclose(J, J.T, atol=1e-14)


def test_firmly_nonexpansive():
    e, w = random_graph(40, 60, 11)
    op = fp.GraphTV(40, e, w)
    rng = np.random.default_rng(12)
    for _ in range(20):
        a, b = rng.normal(size=(2, 40)) * 2
        za, zb = op.prox(a, 0.7), op.prox(b, 0.7)
        assert np.dot(za - zb, za - zb) <= np.dot(za - zb, a - b) + 1e-12


def test_errors():
    with pytest.raises(ValueError):
        fp.GraphTV(3, [[0, 3]])  # out of range
    with pytest.raises(ValueError):
        fp.GraphTV(3, [[1, 1]])  # self-loop
    with pytest.raises(ValueError):
        fp.GraphTV(3, [[0, 1], [1, 2]], [1.0, -1.0])
    with pytest.raises(ValueError):
        fp.GraphTV(3, [[0, 1], [1, 2]], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        fp.GraphTV(3, [0, 1, 2])  # not (m, 2)
    op = fp.GraphTV(3, [[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        op.prox(np.zeros(4), 1.0)
    with pytest.raises(ValueError):
        op.prox(np.zeros(3), 1.0, tol=-1.0)
