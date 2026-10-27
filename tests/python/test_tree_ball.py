"""Tree-structured group norms (and the sparse group lasso) and the projection
onto the 1D TV ball: agreement with independent solvers, the operators these
generalise, projection inequalities, special cases and errors. Derivatives,
threads and strides run in test_operators.py."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref


def random_tree(rng, n, depth=3):
    """Nested groups over 0..n-1: recursive random splits, keeping a random
    subset of the nodes as groups, plus one group outside the root's span."""
    groups = []

    def split(lo, hi, d):
        if rng.random() < 0.7:
            groups.append(list(range(lo, hi)))
        if d == 0 or hi - lo < 2:
            return
        cuts = np.sort(rng.choice(np.arange(lo + 1, hi), size=min(2, hi - lo - 1), replace=False))
        for a, b in zip([lo, *cuts], [*cuts, hi]):
            split(a, b, d - 1)

    split(0, n - 3, depth)
    groups.append([n - 2, n - 1])
    order = rng.permutation(len(groups))
    return [groups[k] for k in order]


# ---------------------------------------------------------------- tree norms


@pytest.mark.parametrize("seed", range(8))
def test_tree_matches_dual_coordinate_descent(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 40))
    groups = random_tree(rng, n)
    eta = rng.uniform(0.0, 1.0, len(groups))
    l1 = [0.0, 0.3][seed % 2]
    for scale in (0.3, 1.0, 5.0):
        x = rng.normal(scale=scale, size=n)
        z = fp.tree_soft_threshold(x, groups, eta, l1, 0.8)
        np.testing.assert_allclose(z, ref.tree_l2(x, 0.8, l1, groups, eta), rtol=1e-11,
                                   atol=1e-12 * scale)


def test_tree_group_order_is_irrelevant():
    rng = np.random.default_rng(1)
    groups = random_tree(rng, 30)
    eta = rng.uniform(0.1, 1.0, len(groups))
    x = rng.normal(size=(5, 30))
    z = fp.tree_soft_threshold(x, groups, eta, 0.1)
    for _ in range(3):
        p = rng.permutation(len(groups))
        np.testing.assert_allclose(fp.tree_soft_threshold(x, [groups[k] for k in p], eta[p], 0.1),
                                   z, rtol=1e-14, atol=1e-15)


def test_tree_special_cases():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 12))
    ids = np.arange(12) % 4
    parts = [np.flatnonzero(ids == g) for g in range(4)]
    c = np.array([0.5, 1.0, 1.5, 0.2])
    # A partition without l1 is group l2; singletons are soft thresholding.
    np.testing.assert_allclose(fp.tree_soft_threshold(x, parts, c, 0.0, 0.7),
                               fp.group_soft_threshold(x, 1.0, 0.7, groups=ids, group_weights=c),
                               rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(fp.tree_soft_threshold(x, [[i] for i in range(12)], 0.4),
                               fp.soft_threshold(x, 0.4), rtol=1e-14, atol=1e-16)
    np.testing.assert_array_equal(fp.tree_soft_threshold(x, [], None, 0.4), fp.soft_threshold(x, 0.4))
    # The sparse group lasso is soft then group soft thresholding (Bach et al.).
    np.testing.assert_allclose(
        fp.sparse_group_soft_threshold(x, 0.3, 0.8, 1.2, groups=ids, group_weights=c),
        fp.group_soft_threshold(fp.soft_threshold(x, 0.3, 1.2), 0.8, 1.2, groups=ids,
                                group_weights=c), rtol=1e-14, atol=1e-16)
    # A repeated group adds its weights; entries outside every group are only soft-thresholded.
    np.testing.assert_allclose(fp.tree_soft_threshold(x, [[0, 1, 2], [0, 1, 2]], [0.3, 0.4]),
                               fp.tree_soft_threshold(x, [[0, 1, 2]], 0.7), rtol=1e-14, atol=1e-16)
    z = fp.tree_soft_threshold(x, [[0, 1]], 5.0, 0.2)
    np.testing.assert_array_equal(z[:, :2], 0.0)
    np.testing.assert_array_equal(z[:, 2:], fp.soft_threshold(x[:, 2:], 0.2))


def test_tree_named_grads():
    op = fp.TreeL2([[0, 1], [0, 1, 2]], [0.4, 0.3], l1=0.1)
    z, ctx = op.prox(np.array([2.0, -1.0, 0.5]), keep=True)
    _, grads = op.vjp(ctx, np.ones(3))
    assert np.shape(grads.group_weights) == (2,) and isinstance(grads.l1, float)


def test_tree_errors():
    x = np.ones(4)
    for bad in (lambda: fp.TreeL2([[0, 1], [1, 2]]),              # overlap, not nested
                lambda: fp.TreeL2([[0, 1, 2], [2, 3], [0, 1, 2, 3, 4]]),
                lambda: fp.TreeL2([[0, 0]]),                      # repeated entry
                lambda: fp.TreeL2([[-1]]),
                lambda: fp.TreeL2([[0], [1]], [1.0, 2.0, 3.0]),   # weights of another length
                lambda: fp.TreeL2([[0]], -1.0), lambda: fp.TreeL2([[0]], None, -0.1),
                lambda: fp.TreeL2([[0, 5]]).prox(x)):             # index beyond the axis
        with pytest.raises(ValueError):
            bad()
    z = fp.TreeL2([[0, 1], [0, 1, 2, 3]]).prox(np.array([1.0, np.nan, 2.0, 3.0]))
    assert np.all(np.isnan(z))


# ---------------------------------------------------------------- TV ball


def tv(z, w=1.0):
    return float(np.sum(w * np.abs(np.diff(z))))


@pytest.mark.parametrize("n", [2, 3, 10, 1000, 100000])
@pytest.mark.parametrize("kind", ["noise", "walk", "steps"])
def test_tv_ball_on_the_sphere(n, kind):
    rng = np.random.default_rng(n)
    x = {"noise": rng.normal(size=n), "walk": np.cumsum(rng.normal(size=n)),
         "steps": np.repeat(rng.normal(size=max(n // 50, 1)), 50)[:n] + 0.01 * rng.normal(size=n)}[kind]
    for frac in (1e-3, 0.1, 0.5, 0.999):
        a = frac * tv(x)
        z = fp.project_tv1d_ball(x, a)
        assert tv(z) == pytest.approx(a, rel=1e-11, abs=1e-12)
        if n <= 1000:
            np.testing.assert_allclose(z, ref.tv1d_ball(x, a), rtol=1e-10, atol=1e-10)


def test_tv_ball_weighted_with_zeros():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=500))
    w = rng.uniform(0.0, 2.0, 499)
    w[::37] = 0.0
    for a in (0.0, 1.0, 30.0):
        z = fp.project_tv1d_ball(x, a, weights=w)
        assert tv(z, w) == pytest.approx(a, abs=1e-10 * (1 + a))
        np.testing.assert_allclose(z, ref.tv1d_ball(x, a, w), rtol=1e-10, atol=1e-10)


def test_tv_ball_is_a_projection():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=200))
    a = 0.2 * tv(x)
    z = fp.project_tv1d_ball(x, a)
    # <x - z, y - z> <= 0 for every y in the ball: here y shrinks a random
    # signal towards its mean until it fits.
    for _ in range(200):
        v = np.cumsum(rng.normal(size=200)) * rng.uniform(0.1, 3)
        t = min(1.0, a / max(tv(v), 1e-300))
        y = v.mean() + t * (v - v.mean()) + rng.normal() * 3
        assert tv(y) <= a * (1 + 1e-12)
        assert (x - z) @ (y - z) <= 1e-9 * (1 + np.abs(x).max() ** 2)
    np.testing.assert_allclose(fp.project_tv1d_ball(z, a), z, rtol=1e-12, atol=1e-12)


def test_tv_ball_special_cases():
    x = np.array([[0.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(fp.project_tv1d_ball(x, 4.0), x)   # TV = 4: on the sphere, kept
    np.testing.assert_array_equal(fp.project_tv1d_ball(x, 1e9), x)
    np.testing.assert_allclose(fp.project_tv1d_ball(x, 0.0), np.repeat(x.mean(1, keepdims=True), 4, 1))
    # Length-1 rows are always inside.
    np.testing.assert_array_equal(fp.project_tv1d_ball(np.array([5.0]), 0.0), [5.0])
    # Inside, the Jacobian is the identity and the parameter gradients vanish.
    op = fp.TV1DBall(10.0)
    z, ctx = op.prox(x[0], keep=True)
    gx, grads = op.vjp(ctx, np.arange(4.0))
    np.testing.assert_array_equal(gx, np.arange(4.0))
    assert grads.alpha == 0.0
    # float32 lands on the sphere to single precision.
    z32 = fp.project_tv1d_ball(np.cumsum(np.ones(1000, np.float32)), 10.0)
    assert z32.dtype == np.float32 and tv(z32.astype(float)) == pytest.approx(10.0, rel=1e-5)


def test_tv_ball_gradient_in_alpha():
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(size=300))
    op = fp.TV1DBall(15.0)
    z, ctx = op.prox(x, keep=True)
    g = rng.normal(size=300)
    _, grads = op.vjp(ctx, g)
    h = 1e-6
    fd = (g @ fp.project_tv1d_ball(x, 15.0 + h) - g @ fp.project_tv1d_ball(x, 15.0 - h)) / (2 * h)
    assert grads.alpha == pytest.approx(fd, rel=1e-6)


def test_tv_ball_extreme_scales():
    # The nightly fuzzer: the conjugate at sigma = 1.8e-306 projects y / sigma,
    # with entries up to 4.9e307, and the piece sums overflowed: NaN. At
    # alpha = 0 the projection is the mean, so the conjugate is y - mean(y).
    y = np.array([-0.00390625, -22, 0, 0, 0, -90, -89.9882812, -89.3515625, -89.3515625])
    z = fp.TV1DBall(0.0).prox_conj(y, 1.8198309184996924e-306)
    np.testing.assert_allclose(z, y - y.mean(), rtol=0, atol=1e-13)
    # P_{c alpha}(c x) = c P_alpha(x), with the same derivatives (dz/dw scales with c).
    rng = np.random.default_rng(9)
    x, w = np.cumsum(rng.normal(size=40)), rng.uniform(0.5, 2.0, 39)
    a = 0.3 * np.sum(w * np.abs(np.diff(x)))
    dx, g, dp = rng.normal(size=40), rng.normal(size=40), rng.normal(size=40)
    op1 = fp.TV1DBall(a, w)
    z1, c1 = op1.prox(x, keep=True)
    for c in (2.0**1000, 2.0**-1000):
        op = fp.TV1DBall(a * c, w)
        zc, cc = op.prox(x * c, keep=True)
        np.testing.assert_allclose(zc / c, z1, rtol=0, atol=1e-13 * np.abs(x).max())
        np.testing.assert_allclose(op.jvp(cc, dx), op1.jvp(c1, dx), rtol=1e-12, atol=1e-12)
        gx, grads = op.vjp(cc, g)
        gx1, grads1 = op1.vjp(c1, g)
        np.testing.assert_allclose(gx, gx1, rtol=1e-12, atol=1e-12)
        assert grads.alpha == pytest.approx(grads1.alpha, rel=1e-12)
        np.testing.assert_allclose(grads.params[1:] / c, grads1.params[1:], rtol=1e-12, atol=1e-12)
        jw = op.jvp(cc, np.zeros(40), dparams=np.r_[0.0, dp[1:]])
        np.testing.assert_allclose(jw / c, op1.jvp(c1, np.zeros(40), dparams=np.r_[0.0, dp[1:]]), rtol=1e-12, atol=1e-12)
    # Inside the ball the row comes back exactly, even an entry that the
    # rescaling by 2^-1017 would take below the smallest subnormal.
    x = np.array([1e306, -1e306, 3e-310, 0.0])
    np.testing.assert_array_equal(fp.project_tv1d_ball(x, 1e308), x)


def test_tv_ball_derivatives_at_extreme_weights():
    # <q, z_gamma> grows as w^2: it overflowed from w ~ 1e155 and underflowed
    # below 1e-155, and the derivatives lost the constraint's term (fuzzing
    # guarantee 1, 1.0.2). Scaling w and alpha by c leaves the projection.
    rng = np.random.default_rng(10)
    x, w = np.cumsum(rng.normal(size=30)), rng.uniform(0.5, 2.0, 29)
    a = 0.3 * np.sum(w * np.abs(np.diff(x)))
    dx = rng.normal(size=30)
    op1 = fp.TV1DBall(a, w)
    z1, c1 = op1.prox(x, keep=True)
    for c in (2.0**600, 2.0**-600):
        op = fp.TV1DBall(a * c, w * c)
        z, cc = op.prox(x, keep=True)
        np.testing.assert_allclose(z, z1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(op.jvp(cc, dx), op1.jvp(c1, dx), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(op.vjp(cc, dx)[0], op1.vjp(c1, dx)[0], rtol=1e-10, atol=1e-10)
    # The fuzz reproducer: the conjugate at sigma = 1.1e298 works on
    # dx / sigma, whose products with the small weight's q underflowed.
    w = np.array([4.0582731154107899e147, 0, 0, 1.2105675798371893e20, 0])
    op = fp.TV1DBall(1e-200, w)
    y, dx = np.array([-44.0, 99, -61, -70, 54, 31]), np.arange(1.0, 7.0)
    _, ctx = op.prox_conj(y, 1.1444946670995897e298, keep=True)
    s = ctx.structure()
    assert s.conj and s.active.all()
    J = dx - op.jvp(ctx, dx)  # the inner prox's J dx (Moreau)
    np.testing.assert_allclose(s.project(J), J, rtol=0, atol=1e-12 * np.abs(dx).max())


def test_tv_ball_large_weight_inside_a_segment():
    # The derivatives' scale was taken from max w, here an edge inside a
    # segment (7e157), which left the jump's terms subnormal and <q, z_gamma>
    # with 8 digits (fuzzing guarantee 1, 1.0.2): now from the jumps' weights.
    rng = np.random.default_rng(12)
    w = np.array([0.36, 3.08, 0.47, 2.34, 1.14, 7.27e157, 0.5, 2.66])
    x = 10 * rng.normal(size=9)
    op = fp.TV1DBall(1.0, w)
    _, ctx = op.prox(x, keep=True)
    s = ctx.structure()
    assert s.active.all()
    for dx in rng.normal(size=(4, 9)):
        J = op.jvp(ctx, dx)
        np.testing.assert_allclose(s.project(J), J, rtol=0, atol=1e-13 * np.abs(dx).max())


def test_tv_ball_errors():
    for bad in (lambda: fp.TV1DBall(-1.0), lambda: fp.TV1DBall(np.nan),
                lambda: fp.TV1DBall(1.0, [-1.0]), lambda: fp.TV1DBall(1.0, np.ones(3)).prox(np.ones(5))):
        with pytest.raises(ValueError):
            bad()
    z = fp.project_tv1d_ball(np.array([1.0, np.nan, 2.0]), 0.5)
    assert np.all(np.isnan(z))
    with pytest.raises(ValueError):
        fp.project_tv1d_ball(np.array([1.0, np.inf]), 0.5)
