"""Tree TV (dynamic programming on a forest): optimality by an independent KKT
check, agreement with graph TV and with 1D TV on paths, the gap it reports,
derivatives, batching, threads and errors."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref


def random_forest(n, seed, kind="random", drop=0, zero_frac=0.1):
    """A tree on n nodes (each node joins an earlier one, the one before it
    for a path, node 0 for a star) with shuffled labels, ends and edge
    order, minus `drop` edges; weights in [0.2, 2] with some zeros."""
    rng = np.random.default_rng(seed)
    i = np.arange(1, n)
    p = {"random": (rng.random(n - 1) * i).astype(int), "path": i - 1, "star": np.zeros(n - 1, int)}[kind]
    e = np.stack([i, p], 1)
    flip = rng.random(n - 1) < 0.5
    e[flip] = e[flip][:, ::-1]
    e = rng.permutation(n)[e][rng.permutation(n - 1)][: max(n - 1 - drop, 0)]
    w = rng.uniform(0.2, 2.0, len(e))
    w[rng.random(len(e)) < zero_frac] = 0.0
    return e.reshape(-1, 2), w


def blocky(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=scale, size=n)
    return x + scale * (np.arange(n) // 4 % 3)


@pytest.mark.parametrize("kind", ["random", "path", "star"])
@pytest.mark.parametrize("n,drop", [(1, 0), (2, 0), (3, 1), (12, 0), (40, 3), (300, 0), (300, 20)])
@pytest.mark.parametrize("gamma", [0.05, 0.6, 5.0])
def test_kkt_and_graph_tv(kind, n, drop, gamma):
    for seed in range(3):
        e, w = random_forest(n, seed, kind, drop)
        for scale in (0.1, 1.0, 30.0):
            x = blocky(n, seed + 7, scale)
            z, info = fp.TreeTV(n, e, w).solve(x, gamma)
            assert info.converged and info.iters == 1
            assert ref.graph_tv_kkt_residual(x, z, gamma, e, w) < 1e-10
            assert info.gap <= 1e-20 + 1e-13 * np.dot(x, x)
            np.testing.assert_allclose(info.dual[0], x - z, atol=1e-11 * (1 + np.abs(x).max()))
            zg = fp.GraphTV(n, e, w).prox(x, gamma)
            np.testing.assert_allclose(z, zg, rtol=0, atol=1e-12 * (1 + np.abs(x).max()))


@pytest.mark.parametrize("n", [2, 7, 100, 3000])
@pytest.mark.parametrize("gamma", [1e-3, 0.3, 4.0])
def test_paths_match_tv1d(n, gamma):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n).cumsum()
    w = rng.uniform(0.0, 2.0, n - 1)
    w[rng.random(n - 1) < 0.05] = 0.0
    z1 = fp.tv1d(x, gamma, weights=w)
    # The path through the nodes in a shuffled order, edges in any direction and order.
    perm = rng.permutation(n)
    e = np.stack([perm[:-1], perm[1:]], 1)
    flip = rng.random(n - 1) < 0.5
    e[flip] = e[flip][:, ::-1]
    k = rng.permutation(n - 1)
    xs = np.empty(n)
    xs[perm] = x
    z, info = fp.TreeTV(n, e[k], w[k]).solve(xs, gamma)
    np.testing.assert_allclose(z[perm], z1, rtol=0, atol=1e-12 * (1 + np.abs(x).max()))


def test_scalar_and_unit_weights():
    e, _ = random_forest(30, 1)
    x = blocky(30, 2)
    z1 = fp.TreeTV(30, e).prox(x, 0.7)
    np.testing.assert_array_equal(z1, fp.TreeTV(30, e, 1.0).prox(x, 0.7))
    np.testing.assert_array_equal(z1, fp.TreeTV(30, e, np.ones(len(e))).prox(x, 0.7))
    np.testing.assert_allclose(z1, fp.TreeTV(30, e, 0.5).prox(x, 1.4), atol=1e-14)
    np.testing.assert_allclose(z1, fp.tree_tv(x, e, 0.7)[0], atol=0)


def test_large_trees():
    """A long path in shuffled order against 1D TV; a wide star and a comb (a
    spine with long teeth) against exact graph TV."""
    rng = np.random.default_rng(13)
    n = 100_000
    x = np.sin(np.arange(n) / 500.0) + 0.3 * rng.normal(size=n)
    perm = rng.permutation(n)
    xs = np.empty(n)
    xs[perm] = x
    op = fp.TreeTV(n, np.stack([perm[:-1], perm[1:]], 1))
    for gamma in (0.1, 3.0):
        z, info = op.solve(xs, gamma)
        np.testing.assert_allclose(z[perm], fp.tv1d(x, gamma), rtol=0, atol=1e-11)
        assert info.gap <= 1e-13 * np.dot(x, x)
    h = 200
    idx = np.arange(h * h).reshape(h, h)
    comb = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                           np.stack([idx[:-1, 0], idx[1:, 0]], 1)])
    # Graph TV is slow on a star (its max-flow rescans the hub), so a smaller one.
    for e, w in [random_forest(3000, 2, "star", zero_frac=0.0), (comb, None)]:
        n = int(e.max()) + 1
        x = np.sin(np.arange(n) / 500.0) + 0.3 * rng.normal(size=n)
        for gamma in (0.1, 3.0):
            z, info = fp.TreeTV(n, e, w).solve(x, gamma)
            np.testing.assert_allclose(z, fp.GraphTV(n, e, w).prox(x, gamma), rtol=0, atol=1e-11)
            assert info.gap <= 1e-13 * np.dot(x, x)


def test_batching_threads_strides_float32():
    e, w = random_forest(50, 6, drop=2)
    op = fp.TreeTV(50, e, w)
    x = np.random.default_rng(7).normal(size=(4, 9, 50)) * 3
    rows = np.stack([op.prox(r, 0.8) for r in x.reshape(-1, 50)]).reshape(x.shape)
    for t in (1, 2, 8):
        assert op.prox(x, 0.8, threads=t).tobytes() == rows.tobytes()
    v = x[::-1, ::2]
    np.testing.assert_array_equal(op.prox(v, 0.8), op.prox(np.ascontiguousarray(v), 0.8))
    y = x.copy()
    op.prox(y, 0.8, out=y)
    np.testing.assert_array_equal(y, rows)
    z32 = op.prox(x.astype(np.float32), 0.8)
    assert z32.dtype == np.float32
    np.testing.assert_allclose(z32, op.prox(x.astype(np.float32).astype(np.float64), 0.8), atol=1e-5)
    g = np.sin(np.arange(x.size)).reshape(x.shape)
    res = []
    for t in (1, 3, 8):
        _, ctx = op.prox(x, 0.8, keep=True, threads=t)
        gx, gr = op.vjp(ctx, g)
        res.append(gx.tobytes() + np.append(gr.params, gr.gamma).tobytes())
    assert res[0] == res[1] == res[2]


def test_special_cases():
    e, w = random_forest(20, 8)
    op = fp.TreeTV(20, e, w)
    x = np.random.default_rng(9).normal(size=20)
    np.testing.assert_array_equal(op.prox(x, 0.0), x)
    np.testing.assert_array_equal(fp.TreeTV(5, np.zeros((0, 2), int)).prox(x[:5], 3.0), x[:5])
    np.testing.assert_array_equal(fp.TreeTV(1, np.zeros((0, 2), int)).prox(x[:1], 3.0), x[:1])
    z = fp.TreeTV(6, [[0, 1], [1, 2]]).prox(x[:6], 0.4)
    np.testing.assert_array_equal(z[3:], x[3:6])
    # Large gamma: the mean of each tree (no zero weights here).
    e1, _ = random_forest(20, 3, zero_frac=0.0)
    np.testing.assert_allclose(fp.TreeTV(20, e1).prox(x, 1e6), np.full(20, x.mean()), atol=1e-9)
    np.testing.assert_array_equal(op.prox(np.full(20, 2.5), 1.0), np.full(20, 2.5))
    # tol and max_iter have no effect.
    np.testing.assert_array_equal(op.solve(x, 0.6, tol=0.1, max_iter=1)[0], op.prox(x, 0.6))
    xn = x.copy()
    xn[3] = np.nan
    assert np.all(np.isnan(op.prox(xn, 1.0)))
    xi = x.copy()
    xi[3] = np.inf
    with pytest.raises(ValueError):
        op.prox(xi, 1.0)


def _fd_case(seed, scalar):
    e, w = random_forest(14, seed, drop=seed % 3, zero_frac=0.0)
    theta = np.array([0.9]) if scalar else w
    build = lambda t: fp.TreeTV(14, e, t[0] if scalar else t)
    x = np.random.default_rng(seed).normal(scale=1.5, size=(3, 14))
    return build, theta, x


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
def test_derivatives(conj, scalar):
    h = 1e-6
    for seed in range(6):
        build, theta, x = _fd_case(seed, scalar)
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


def test_derivatives_match_graph_tv():
    e, w = random_forest(60, 4, drop=4, zero_frac=0.0)
    x = blocky(60, 5)
    rng = np.random.default_rng(6)
    dx, g = rng.normal(size=(2, 60))
    out = []
    for cls in (fp.TreeTV, fp.GraphTV):
        op = cls(60, e, w)
        _, ctx = op.prox(x, 0.5, keep=True)
        gx, gr = op.vjp(ctx, g)
        out.append((op.jvp(ctx, dx, 0.2, w), gx, gr.params, gr.gamma))
    for a, b in zip(*out):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_firmly_nonexpansive():
    e, w = random_forest(40, 11, drop=3)
    op = fp.TreeTV(40, e, w)
    rng = np.random.default_rng(12)
    for _ in range(20):
        a, b = rng.normal(size=(2, 40)) * 2
        za, zb = op.prox(a, 0.7), op.prox(b, 0.7)
        assert np.dot(za - zb, za - zb) <= np.dot(za - zb, a - b) + 1e-12


def test_errors():
    with pytest.raises(ValueError):
        fp.TreeTV(3, [[0, 1], [1, 2], [2, 0]])  # a cycle
    with pytest.raises(ValueError):
        fp.TreeTV(3, [[0, 1], [1, 0]])  # a repeated edge
    with pytest.raises(ValueError):
        fp.TreeTV(3, [[0, 3]])
    with pytest.raises(ValueError):
        fp.TreeTV(3, [[1, 1]])
    with pytest.raises(ValueError):
        fp.TreeTV(3, [[0, 1], [1, 2]], [1.0, -1.0])
    with pytest.raises(ValueError):
        fp.TreeTV(3, [0, 1, 2])
    op = fp.TreeTV(3, [[0, 1], [1, 2]])
    assert isinstance(op, fp.GraphTV)
    with pytest.raises(ValueError):
        op.prox(np.zeros(4), 1.0)
