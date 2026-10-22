"""1D TV with the l2 norm of the differences, prox of gamma ||W D z||_2:
agreement with a dense solver, the duality gap of an independently built dual
point, optimality at large n, invariances, special cases and errors.
Derivatives, threads and strides run in test_operators.py."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref


def signal(rng, kind, n):
    return {"noise": lambda: rng.normal(size=n),
            "walk": lambda: np.cumsum(rng.normal(size=n)),
            "steps": lambda: np.repeat(rng.normal(size=n // 5 + 1), 5)[:n] + 0.01 * rng.normal(size=n),
            "offset": lambda: np.sin(2.9 * np.arange(n)) + 100.0}[kind]()


def dual(x, z, w):
    """u with (W D)^T u = x - z, built by cumulative sums from x and z alone
    (zero-weight edges get u = 0), and the gap gamma ||W D z|| - <W D z, u>."""
    w = np.broadcast_to(np.asarray(w, float), (x.size - 1,))
    r = np.cumsum(x - z)[:-1]
    u = np.where(w > 0, -r / np.where(w > 0, w, 1.0), 0.0)
    return u, w * np.diff(z)


def rho0(x, w):
    """||u0||, the threshold on gamma below which z is not the piecewise mean."""
    return np.linalg.norm(dual(x, ref.tv1d_l2(x, 1e300, w), w)[0])


@pytest.mark.parametrize("n", [2, 3, 10, 60])
@pytest.mark.parametrize("kind", ["noise", "walk", "steps", "offset"])
@pytest.mark.parametrize("wkind", ["one", "rand", "zeros"])
def test_matches_dense_solver(n, kind, wkind):
    rng = np.random.default_rng(n)
    x = signal(rng, kind, n)
    w = {"one": 1.0, "rand": rng.uniform(0.1, 2.0, n - 1),
         "zeros": np.where(rng.random(n - 1) < 0.25, 0.0, rng.uniform(0.1, 2.0, n - 1))}[wkind]
    r0 = rho0(x, w)
    for frac in (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1.0, 2.0):
        g = frac * r0
        z = fp.tv1d_l2(x, g, weights=None if np.ndim(w) == 0 else w)
        np.testing.assert_allclose(z, ref.tv1d_l2(x, g, w), rtol=0, atol=1e-11 * (1 + np.abs(x).max()))


@pytest.mark.parametrize("n", [10, 1000, 100000])
@pytest.mark.parametrize("kind", ["noise", "walk"])
def test_gap_and_optimality(n, kind):
    rng = np.random.default_rng(n + 1)
    x = signal(rng, kind, n)
    w = rng.uniform(0.5, 1.5, n - 1)
    r0 = np.linalg.norm(np.cumsum(x - x.mean())[:-1] / w)
    for frac in (1e-4, 0.1, 0.5, 0.99):
        g = frac * r0
        z = fp.tv1d_l2(x, g, weights=w)
        u, wdz = dual(x, z, w)
        s = np.linalg.norm(wdz)
        # The dual point is feasible and certifies z: zero gap to rounding.
        # Without FMA contraction (x86-64 builds) the dual taken from x - z
        # carries about 20 eps |x| per entry at small gamma, the cancellation
        # in x - z: up to 6e-13 relative on the walk at n = 1e5, 2e-14 with FMA.
        assert np.linalg.norm(u) <= g * (1 + 1e-9)
        assert g * s - wdz @ u <= 4e-12 * (1 + g * s)
        # Stationarity: x - z = gamma (W D)^T W D z / s, i.e. u = gamma W D z / s.
        # The right side multiplies the rounding of z by gamma / s.
        np.testing.assert_allclose(u, g * wdz / s, rtol=0,
                                   atol=1e-9 * g + 1e-14 * (g / s) * np.abs(x).max())


def test_two_entries_is_1d_tv():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 2)) * 3
    for w in (1.0, 0.4):
        for g in (0.1, 1.0, 10.0):
            np.testing.assert_allclose(fp.tv1d_l2(x, g, weights=[w]), fp.tv1d(x, g, weights=[w]),
                                       rtol=1e-14, atol=1e-15)


def test_invariances():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=(4, 200)), axis=1)
    z = fp.tv1d_l2(x, 5.0)
    np.testing.assert_allclose(fp.tv1d_l2(x + 7.5, 5.0), z + 7.5, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fp.tv1d_l2(3.0 * x, 15.0), 3.0 * z, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fp.tv1d_l2(x[:, ::-1], 5.0), z[:, ::-1], rtol=1e-12, atol=1e-12)
    # Sum preserved per piece; weights scale gamma.
    np.testing.assert_allclose(z.sum(1), x.sum(1), rtol=1e-12)
    np.testing.assert_allclose(fp.tv1d_l2(x, 2.5, weights=2.0), z, rtol=1e-12, atol=1e-12)


def test_special_cases():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=(3, 40)), axis=1)
    np.testing.assert_array_equal(fp.tv1d_l2(x, 0.0), x)
    np.testing.assert_array_equal(fp.tv1d_l2(np.array([4.0]), 3.0), [4.0])
    np.testing.assert_array_equal(fp.tv1d_l2(np.full(5, 2.0), 1.0), np.full(5, 2.0))
    # Inside: the piecewise mean, with the pieces between zero weights.
    w = np.ones(39)
    w[[9, 24]] = 0.0
    z = fp.tv1d_l2(x, 1e6, weights=w)
    for lo, hi in ((0, 10), (10, 25), (25, 40)):
        np.testing.assert_allclose(z[:, lo:hi], np.repeat(x[:, lo:hi].mean(1, keepdims=True), hi - lo, 1),
                                   rtol=1e-13)
    # There the Jacobian is the piecewise averaging and the gradients vanish.
    op = fp.TV1DL2(w)
    zr, ctx = op.prox(x[0], 1e6, keep=True)
    g = rng.normal(size=40)
    gx, grads = op.vjp(ctx, g)
    np.testing.assert_allclose(gx[:10], g[:10].mean(), rtol=1e-13)
    assert grads.gamma == 0.0 and np.all(grads.weights == 0.0)
    assert np.shape(grads.weights) == (39,)
    op1 = fp.TV1DL2()
    assert isinstance(op1.vjp(op1.prox(x[0], 2.0, keep=True)[1], g)[1].weights, float)
    # In place.
    y = x.copy()
    fp.tv1d_l2(y, 3.0, out=y)
    np.testing.assert_array_equal(y, fp.tv1d_l2(x, 3.0))


def test_tiny_weights_on_constant_pieces():
    # The nightly fuzzer: constant pieces whose running mean rounds, with a
    # weight of 2e-53 inside one of them, gave NaN (the rounding residual over
    # w^2 posed as a dual of 1e38). Constant pieces add nothing to the norm,
    # so they stay as they are and the varying piece is solved on its own.
    w = np.array([0.46875] * 5 + [0, 0, 2.2334456681412216e-53] + [0.46875] * 5 + [3.3125])
    x = np.full(15, 29.1132812)
    x[6] = -0.00390625
    for g in (0.515625, 1.0, 1e3):
        np.testing.assert_array_equal(fp.tv1d_l2(x, g, weights=w), x)
    assert np.isfinite(fp.TV1DL2(w).prox_conj(x, 0.515625)).all()
    x[:6] = np.random.default_rng(8).normal(size=6) + 29.0
    for g in (0.1, 0.5, 2.0):
        z = fp.tv1d_l2(x, g, weights=w)
        np.testing.assert_array_equal(z[6:], x[6:])
        np.testing.assert_allclose(z[:6], fp.tv1d_l2(x[:6], g, weights=w[:5]), rtol=1e-14, atol=1e-13)


def test_tiny_weight_between_tiny_entries():
    # The nightly fuzzer: in a conjugate prox, a piece of entries near 1e-147
    # whose only active edge has w = 4e-149 puts the root at mu = 2e-295, and
    # sigma x underflowed in the solves: NaN. The edges of weight 0.53 and 4
    # hold the blocks 29..32 and 33..35 rigid, so the weak edge moves gamma w
    # between their means.
    w = np.array([0.0] * 29 + [0.53125] * 3 + [4.1871721532216448e-149, 4.0, 4.0])
    x = np.zeros(36)
    x[[20, 21, 23, 28]] = [33.12890625, 33.12890625, 33.0, 88.49609375]
    x[29], x[31] = 4.18634103e-149, 1.60559213e-146
    assert np.isfinite(fp.TV1DL2(w).prox_conj(x, 4.0)).all()
    y, g = x / 4, 0.25
    z = fp.tv1d_l2(y, g, weights=w)
    np.testing.assert_array_equal(z[:29], y[:29])
    m = y[29:33].mean()
    np.testing.assert_allclose(z[29:33], m - g * w[32] / 4, rtol=1e-12)
    np.testing.assert_allclose(z[33:], g * w[32] / 3, rtol=1e-12)
    # The same problem at other scales: z(t y, t g) = t z(y, g). At 2^480,
    # ||y0||^2 overflowed and Newton started from the bisection: differences
    # of z then carried its rounding, 1e-17 across the edges of weight 0.53
    # against 1e-150 across the weak edge, and the iteration returned the
    # block means unshrunk.
    for k in (-300, -100, 100, 300, 480):
        t = 2.0**k
        np.testing.assert_allclose(fp.tv1d_l2(y * t, g * t, weights=w) / t, z, rtol=1e-12, atol=0)


def test_scale_invariance_with_extreme_weights():
    # z(t x, t gamma) = t z(x, gamma) for weights over 150 orders of magnitude.
    # Before the inside test and the iteration ran on a rescaled row and the
    # differences, every one of these rows broke it at some scale, by up to
    # 100% of max |x|; the results now agree with a 700-digit solver to 2e-16
    # (checked offline).
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(2, 40))
        w = 10.0 ** rng.uniform(-150, 2, n - 1)
        w[rng.random(n - 1) < 0.15] = 0.0
        x = rng.normal(size=n) * 10.0 ** rng.uniform(-3, 3, n) * (rng.random(n) < 0.8)
        g = rho0(x, w) * 10.0 ** rng.uniform(-3, 0)
        if not (np.isfinite(g) and g > 0):
            continue
        z = fp.tv1d_l2(x, g, weights=w)
        for k in (-600, -100, 100, 400):
            t = 2.0**k / np.abs(x).max()
            np.testing.assert_allclose(fp.tv1d_l2(x * t, g * t, weights=w) / t, z, rtol=0,
                                       atol=1e-12 * np.abs(x).max())


def test_float32():
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(size=(8, 1000)), axis=1)
    z64 = fp.tv1d_l2(x, 20.0)
    z32 = fp.tv1d_l2(x.astype(np.float32), 20.0)
    assert z32.dtype == np.float32
    np.testing.assert_allclose(z32, z64, rtol=0, atol=1e-5 * np.abs(x).max())


def test_gradient_in_gamma_large_n():
    rng = np.random.default_rng(6)
    x = np.cumsum(rng.normal(size=5000))
    op = fp.TV1DL2()
    z, ctx = op.prox(x, 40.0, keep=True)
    g = rng.normal(size=5000)
    _, grads = op.vjp(ctx, g)
    h = 1e-5
    fd = (g @ fp.tv1d_l2(x, 40.0 + h) - g @ fp.tv1d_l2(x, 40.0 - h)) / (2 * h)
    assert grads.gamma == pytest.approx(fd, rel=1e-6)


def test_errors():
    for bad in (lambda: fp.TV1DL2(-1.0), lambda: fp.TV1DL2(np.nan), lambda: fp.TV1DL2(1e200),
                lambda: fp.TV1DL2(np.ones(3)).prox(np.ones(5))):
        with pytest.raises(ValueError):
            bad()
    z = fp.tv1d_l2(np.array([1.0, np.nan, 2.0]), 0.5)
    assert np.all(np.isnan(z))
    with pytest.raises(ValueError):
        fp.tv1d_l2(np.array([1.0, np.inf]), 0.5)


@pytest.mark.parametrize("k", [-560, -365, 365, 530])
def test_derivative_scale_invariance(k):
    """H is unchanged when z and gamma both scale, so the derivatives in x and
    gamma are scale-free and those in w scale with z (audit K4). Setup on raw z
    under- and overflowed: NaN, -inf, a silent 2x error at 1e110, or the
    identity Jacobian beyond 1e160."""
    sigma = 2.0 ** k
    rng = np.random.default_rng(8)
    for x0, w in [(np.array([0.0, 1, 3, 2, 5]), None), (rng.normal(size=50).cumsum(), rng.uniform(0.3, 2.0, 49))]:
        op = fp.TV1DL2(w)
        n = x0.size
        u, v = rng.normal(size=n), rng.normal(size=n)
        dw = rng.normal(size=op.nparams)
        _, c1 = op.prox(x0, 0.8, keep=True)
        _, cs = op.prox(x0 * sigma, 0.8 * sigma, keep=True)
        np.testing.assert_allclose(op.jvp(cs, u), op.jvp(c1, u), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(op.jvp(cs, None, 1.0), op.jvp(c1, None, 1.0), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(op.jvp(cs, None, 0.0, dw) / sigma, op.jvp(c1, None, 0.0, dw), rtol=1e-12, atol=1e-13)
        g1, r1 = op.vjp(c1, v)
        gs, rs = op.vjp(cs, v)
        np.testing.assert_allclose(gs, g1, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(rs.params / sigma, r1.params, rtol=1e-12, atol=1e-13)
        lhs = v @ op.jvp(cs, u)
        assert abs(lhs - gs @ u) <= 1e-12 * (1 + abs(lhs))
