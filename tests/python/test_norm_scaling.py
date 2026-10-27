"""Euclidean norms far from unit scale (audit K2, K5): the l2 ball, group l2,
the tree-structured norms, the group radial maps and the second-order cone
took sqrt(sum x_i^2), whose squares overflow beyond |x| ~ 1e154 and underflow
below ~1e-154. They now rescale by a power of two where the plain sum leaves
its safe range, and keep a scale when the norm itself exceeds the largest
double. Each map below is positively homogeneous once its thresholds or
radii scale with x, so prox(c x; c theta) = c prox(x; theta) for c = 2^k."""

import numpy as np
import pytest

import fastprox as fp

GROUPS = [0, 0, 1, 1, 1, 2, 2, 2]


def ops(c):
    """name: operator with its thresholds and radii scaled by c."""
    return {
        "l2_ball": fp.L2Ball(0.7 * c),
        "prox_l2": fp.GroupL2(0.4 * c),
        "group_l2": fp.GroupL2(0.5 * c, GROUPS, [1.0, 0.6, 1.4]),
        "tree_l2": fp.TreeL2([[0, 1], [2, 3, 4], [0, 1, 2, 3, 4]], [0.2 * c, 0.3 * c, 0.4 * c], l1=0.1 * c),
        "soc": fp.SOC(),
        "group_huber": fp.GroupHuber(0.5 * c, 0.8 * c, GROUPS),
        "group_scad": fp.GroupSCAD(0.5 * c, 3.7, GROUPS),
        "group_mcp": fp.GroupMCP(0.5 * c, 3.0, GROUPS),
    }


NAMES = sorted(ops(1.0))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("k", [-1000, -600, -400, 400, 600, 900])  # 900: gamma lam in the step range
def test_homogeneous(name, k):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 8))
    dx = rng.standard_normal(x.shape)
    c = 2.0 ** k
    op1, opc = ops(1.0)[name], ops(c)[name]
    z1, ctx1 = op1.prox(x, keep=True)
    zc, ctxc = opc.prox(x * c, keep=True)
    np.testing.assert_allclose(zc / c, z1, rtol=1e-14, atol=1e-14 * np.abs(z1).max())
    np.testing.assert_allclose(opc.jvp(ctxc, dx), op1.jvp(ctx1, dx), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(opc.vjp(ctxc, dx)[0], op1.vjp(ctx1, dx)[0], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("k", [-1000, -600, 600])
def test_tangents_at_the_scale_of_x(name, k):
    """x and the tangent both scaled by c (fuzzing guarantee 1, 1.0.2): the
    dots <x, dx> are of order c^2 and left the double range while ||x|| did
    not, so the derivatives lost their term along u (by 27% at 1e-200, inf at
    1e154). The derivatives are linear in the tangent: J(c x) (c dx) / c =
    J(x) dx."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((5, 8))
    dx = rng.standard_normal(x.shape)
    c = 2.0 ** k
    op1, opc = ops(1.0)[name], ops(c)[name]
    _, ctx1 = op1.prox(x, keep=True)
    _, ctxc = opc.prox(x * c, keep=True)
    np.testing.assert_allclose(opc.jvp(ctxc, dx * c) / c, op1.jvp(ctx1, dx), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(opc.vjp(ctxc, dx * c)[0] / c, op1.vjp(ctx1, dx)[0], rtol=1e-12, atol=1e-13)


def test_group_huber_subnormal_norm():
    # On Huber's quadratic piece phi(s) = s / (1 + K): with a subnormal group
    # norm phi(s) underflowed to 0 while phi'(s) did not, and the ratio
    # phi(s) / s was taken as 0, so the derivative kept only the slope along u
    # and the structure held the group at 0 (fuzzing guarantee 1, 1.0.2).
    # J = I / (1 + K) there, as at a normal norm on the same piece.
    op = fp.GroupHuber(1.0, 1e-5)
    dx = np.array([0.7, -1.3])
    _, ref = op.prox(np.array([3e-6, 4e-6]), keep=True)
    _, ctx = op.prox(np.array([3e-320, 4e-320]), keep=True)
    np.testing.assert_allclose(op.jvp(ctx, dx), op.jvp(ref, dx), rtol=1e-12)
    np.testing.assert_allclose(op.vjp(ctx, dx)[0], op.vjp(ref, dx)[0], rtol=1e-12)
    assert (ctx.structure().state == 0).all()  # FREE


@pytest.mark.parametrize("k", [-960, 900])
def test_l1_ball_tangent_scales(k):
    # Weights down to 1e-24 of the largest: a_i dx_i was subnormal for dx near
    # 1e-289 and the projection along a lost digits (fuzzing guarantee 1,
    # 1.0.2). J is linear in the tangent: J (c dx) / c = J dx.
    # The largest weight is off the support (x_0 = 0), so every a_i is near
    # 1e-24 of it.
    rng = np.random.default_rng(13)
    w = np.r_[1.0, 1e-24 * rng.uniform(1, 3, 7)]
    x = np.r_[0.0, rng.normal(size=7) / w[1:]]
    op = fp.L1Ball(0.2 * np.sum(w * np.abs(x)), w)
    _, ctx = op.prox(x, keep=True)
    dx, c = rng.normal(size=8), 2.0**k
    np.testing.assert_allclose(op.jvp(ctx, dx * c) / c, op.jvp(ctx, dx), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(op.vjp(ctx, dx * c)[0] / c, op.vjp(ctx, dx)[0], rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("cls", ["Hyperplane", "HalfSpace"])
@pytest.mark.parametrize("k", [-960, 900])
def test_hyperplane_tangent_scales(cls, k):
    # The largest |a_i| sits at a bound, so the free entries' a_i are near
    # 1e-75 of it: a_i dx_i was subnormal or 0 for dx near 1e-289, and the
    # jvp lost the constraint (fuzzing guarantee 1, 1.0.2, after prox_conj
    # at sigma near 1e263). J is linear in the tangent: J (c dx) / c = J dx.
    rng = np.random.default_rng(14)
    a = np.r_[1.0, 1e-75 * rng.uniform(1, 3, 7)]
    op = getattr(fp, cls)(a, b=1e-75, lower=np.r_[0.0, np.full(7, -np.inf)], upper=np.r_[0.0, np.full(7, np.inf)])
    _, ctx = op.prox(rng.normal(size=8), keep=True)
    s = ctx.structure()
    assert s.active and s.state[0] != 0 and (s.state[1:] == 0).all()
    dx, c = rng.normal(size=8), 2.0**k
    np.testing.assert_allclose(op.jvp(ctx, dx * c) / c, op.jvp(ctx, dx), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(op.vjp(ctx, dx * c)[0] / c, op.vjp(ctx, dx)[0], rtol=1e-13, atol=1e-15)


def test_reproducers():
    s2 = np.sqrt(2.0)
    np.testing.assert_allclose(fp.project_l2_ball(np.array([1e200, 1e200]), 1.0), [1 / s2] * 2, rtol=1e-15)
    x = np.array([1e-200, 1e-200])
    np.testing.assert_allclose(fp.project_l2_ball(x, 1e-201), x * (1e-201 / (s2 * 1e-200)), rtol=1e-15)
    np.testing.assert_allclose(fp.prox_l2(x, 1e-201), x * (1 - 1e-201 / (s2 * 1e-200)), rtol=1e-15)
    y = np.array([1e-170, 1e-170])
    np.testing.assert_array_equal(fp.GroupL2(1.0, None, [0.0]).prox(y), y)  # the identity
    soc = fp.SOC()
    np.testing.assert_allclose(soc.prox(np.array([0.0, 1e200, 1e200])), [s2 * 1e200 / 2, 5e199, 5e199], rtol=1e-15)
    inside = np.array([1e300, 1e200, 1e200])
    np.testing.assert_array_equal(soc.prox(inside), inside)
    al = 0.5 * (s2 * 1e-200 - 1e-201)
    np.testing.assert_allclose(soc.prox(np.array([-1e-201, 1e-200, 1e-200])),
                               [al, al / s2, al / s2], rtol=1e-15)
    big = np.array([1e200, 1e200])
    for op in (fp.GroupHuber(1.0, 1.0), fp.GroupSCAD(1.0, 3.7)):
        np.testing.assert_allclose(op.prox(big), big, rtol=1e-15)
    np.testing.assert_allclose(fp.GroupHuber(1.0, 1.0).prox(y), y / 2, rtol=1e-15)


def test_norm_above_the_largest_double():
    # ||x|| = 1.84e308 for finite entries: the norm keeps a scale.
    x = np.array([1.3e308, 1.3e308])
    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    op = fp.L2Ball(1.0)
    z, ctx = op.prox(x, keep=True)
    np.testing.assert_allclose(z, u, rtol=1e-15)
    np.testing.assert_allclose(op.jvp(ctx, np.zeros(2), 0.0, [1.0]), u, rtol=1e-15)  # dz/dr = u
    z = fp.SOC().prox(np.array([0.0, 1.3e308, 1.3e308]))
    np.testing.assert_allclose(z, [1.3e308 / np.sqrt(2.0), 6.5e307, 6.5e307], rtol=1e-15)
    for op in (fp.GroupL2(1.0), fp.GroupHuber(1.0, 1.0), fp.GroupSCAD(1.0, 3.7)):
        z, ctx = op.prox(x, keep=True)
        np.testing.assert_allclose(z, x, rtol=1e-15)
        assert np.all(np.isfinite(op.jvp(ctx, np.ones(2))))


@pytest.mark.parametrize("name", NAMES)
def test_subnormal_inputs(name):
    # A subnormal largest entry: its scale 2^-e would overflow (found by the
    # fuzz harness: group Huber of 4.3e-316 gave NaN). The entries are exact
    # in the subnormal range, so homogeneity at 2^-1060 holds to rounding.
    x = np.array([[3.0, -4.0, 1.0, 2.0, -1.0, 5.0, 2.0, -3.0]])
    c = 2.0 ** -1060
    opc = ops(c)[name]
    z1 = ops(1.0)[name].prox(x)
    zc, ctxc = opc.prox(x * c, keep=True)
    assert np.all(np.isfinite(zc))
    np.testing.assert_allclose(zc / c, z1, rtol=0, atol=1e-3 * np.abs(z1).max())  # subnormal outputs round
    assert np.all(np.isfinite(opc.jvp(ctxc, np.ones_like(x))))


@pytest.mark.parametrize("sigma", [1.0, 1e-110, 1e-150, 1e150])
def test_tree_l2_scale_free_chain(sigma):
    # tau / s^3 overflowed or underflowed where s did not (audit K5).
    x = sigma * np.array([0.6, 0.8])
    op = fp.TreeL2([[0, 1]], [0.1 * sigma], 0.0)
    _, ctx = op.prox(x, keep=True)
    np.testing.assert_allclose(op.jvp(ctx, np.array([1.0, 0.0])), [0.936, 0.048], rtol=1e-13)
    np.testing.assert_allclose(op.vjp(ctx, np.array([1.0, 0.0]))[0], [0.936, 0.048], rtol=1e-13)


def test_thresholds_where_gamma_lambda_overflows():
    """Soft thresholding, l1 over a box and group l2 formed gamma lambda w as
    (gamma lambda) w (fuzz, 2026-10-25): with every weight 0 the step range
    does not apply, gamma lambda overflowed and inf * 0 gave NaN, where the
    map is the identity; and with gamma lambda = 2^1100, w = 2^-200 the
    threshold 2^900 came out infinite. The product is rounded as written
    from the mantissas there (math audit M3), and unchanged elsewhere."""
    x = np.array([-30.0, 0.5, 2.0**899, -(2.0**901)])
    lam, gamma = 2.0**600, 2.0**500
    for op in (fp.Soft(lam, np.zeros(4)), fp.L1Box(lam, -np.inf, np.inf, np.zeros(4)),
               fp.GroupL2(lam, np.arange(4), np.zeros(4))):
        np.testing.assert_array_equal(op.prox(x, gamma), x)
    w = np.full(4, 2.0**-200)
    want = np.array([0.0, 0.0, 0.0, -(2.0**901 - 2.0**900)])
    for op in (fp.Soft(lam, w), fp.L1Box(lam, -np.inf, np.inf, w), fp.GroupL2(lam, np.arange(4), w)):
        np.testing.assert_array_equal(op.prox(x, gamma), want)
        _, ctx = op.prox(x, gamma, keep=True)
        np.testing.assert_array_equal(op.jvp(ctx, np.ones(4)), [0.0, 0.0, 0.0, 1.0])
    # Ordinary scales: bit for bit the plain product.
    rng = np.random.default_rng(2)
    x, w = rng.normal(size=50), rng.uniform(0.1, 2.0, 50)
    tau = (0.7 * 0.9) * w
    np.testing.assert_array_equal(fp.Soft(0.9, w).prox(x, 0.7), np.sign(x) * np.maximum(np.abs(x) - tau, 0))
