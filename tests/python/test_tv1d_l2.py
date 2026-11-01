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
    for bad in (lambda: fp.TV1DL2(-1.0), lambda: fp.TV1DL2(np.nan),
                lambda: fp.TV1DL2(np.array([1.0, 2.0**-512])),  # its square, scaled, not normal (M5)
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


def test_weight_scale():
    """A weight whose square underflowed dropped its edge, where gamma made
    its effect of order one: TV1DL2(1e-180) at gamma = 1e180 is two-node TV
    with gamma w = 1, whose solution at (1, -1) is 0, and the input came back
    (math audit M5). The weights are stored over a power of two and the step
    scaled by it: prox(gamma, w) = prox(gamma / c, c w)."""
    np.testing.assert_array_equal(fp.TV1DL2(1e-180).prox(np.array([1.0, -1.0]), 1e180), [0.0, 0.0])
    # 1e200, whose square overflowed, is a valid weight now.
    np.testing.assert_allclose(fp.TV1DL2(1e200).prox(np.array([1.0, -1.0]), 0.25e-200), [0.75, -0.75],
                               rtol=1e-15)
    # The smallest ratio accepted, 2^-511.
    fp.TV1DL2(np.array([1.0, 2.0**-511]))
    rng = np.random.default_rng(11)
    for trial in range(60):
        n = int(rng.integers(2, 30))
        # Weights over 100 orders for the prox; for the derivatives, whose
        # conditioning grows with the weights' ratio (to 1e80 here), within
        # one order, at a common scale far from 1.
        wide = trial % 2 == 0
        w = 10.0 ** rng.uniform(-100, 2, n - 1) if wide else rng.uniform(0.3, 3.0, n - 1) * 10.0 ** rng.uniform(-150, 150)
        w[rng.random(n - 1) < 0.15] = 0.0
        x = rng.normal(size=n).cumsum()
        g = rho0(x, w) * 10.0 ** rng.uniform(-2, 0)
        if not (np.isfinite(g) and g > 0):
            continue
        op = fp.TV1DL2(w)
        z, ctx = op.prox(x, g, keep=True)
        u, v = rng.normal(size=n), rng.normal(size=n)
        dw = rng.normal(size=op.nparams)
        _, r1 = op.vjp(ctx, v)
        for k in (-1000, -600, -200, 200, 600, 900):
            c = 2.0**k
            with np.errstate(over="ignore"):
                pos = w[w > 0] * c
            # g / c subnormal is rounded, a different problem (k = 900 at g =
            # 1.2e-37, where NumPy's pow on Linux drew it).
            if not (pos.size and -1022 <= np.log2(g) - k < 1023 and pos.min() >= 2.0**-1022
                    and np.isfinite(pos.max())):
                continue  # the scaled problem itself is not representable
            opc = fp.TV1DL2(w * c)
            zc, cc = opc.prox(x, g / c, keep=True)
            np.testing.assert_array_equal(zc, z)
            if wide:
                continue
            # The same Jacobian in x; d/dgamma scales by c, d/dw by 1 / c.
            np.testing.assert_allclose(opc.jvp(cc, u), op.jvp(ctx, u), rtol=1e-13, atol=1e-14)
            dg, dzw = op.jvp(ctx, None, 1.0), op.jvp(ctx, None, 0.0, dw)
            np.testing.assert_allclose(opc.jvp(cc, None, 1.0) / c, dg, rtol=1e-13, atol=1e-14 * np.abs(dg).max())
            np.testing.assert_allclose(opc.jvp(cc, None, 0.0, dw) * c, dzw, rtol=1e-13,
                                       atol=1e-14 * np.abs(dzw).max())
            _, rc = opc.vjp(cc, v)
            assert rc.gamma / c == pytest.approx(r1.gamma, rel=1e-13, abs=1e-14 * np.abs(dg).max())
            np.testing.assert_allclose(rc.params * c, r1.params, rtol=1e-13, atol=1e-14 * np.abs(r1.params).max())


def test_entries_near_overflow():
    """Rows with an entry near the largest double (fuzz, 2026-10-25). The
    piece mean summed offsets from the first entry, which overflowed for
    x_0 = 9.2e307 among small entries (the row came back -inf); and gamma
    times the row's scale 2^-1024 underflowed to 0, which returned the
    piecewise mean for gamma = 3.6e-97, where z = x to rounding."""
    op = fp.TV1DL2()
    col = np.array([9.18953224e+307, -64.140625, 25.09375, -122.35546875, 0, 0, 0, 44.48828125, -32.92578125,
                    67.01171875, 102, 76, 1.93257735e+204, -22.52734375, 76.21875, -28, 0, 86.3203125, 11.015625])
    z = op.prox(col, 5.639754439021782e+218)
    assert np.all(np.isfinite(z))
    # |z - x| <= 2 gamma, below the rounding of |x| here.
    assert np.linalg.norm(z - col) <= 2 * 5.64e218
    x = np.array([-123.29296875, 1.0584055116233093e+308, 61, -22.6015625, -17, 21.51171875, 16.421875,
                  -1.9236206764566657e+149, 84.03125, -13.2265625, -47.890625])
    np.testing.assert_array_equal(op.prox(x, 3.5833485176407953e-97), x)
    # Near overflow and at unit scale, the same answer (prox(c x; c gamma) = c prox(x; gamma)).
    rng = np.random.default_rng(3)
    for k in (900, 1000):
        y = rng.normal(size=19)
        np.testing.assert_allclose(op.prox(y * 2.0**k, 0.7 * 2.0**(k - 100)) / 2.0**k,
                                   op.prox(y, 0.7 * 2.0**-100), rtol=0, atol=1e-15)


# ------------------------------------------- the boundary and scales (SCD review S2)

def _dense(op, x, g, dtype=np.float64):
    """J densely (rows J e_i, J symmetric) and the context of x itself."""
    n = x.size
    _, cb = op.prox(np.tile(x.astype(dtype), (n, 1)), g, keep=True)
    J = op.jvp(cb, np.eye(n, dtype=dtype)).astype(float)
    return J, op.prox(x.astype(dtype), g, keep=True)[1]


def _inside(ctx, n):
    """The selected side, from the descriptor: inside has the pieces as regions."""
    s = ctx.structure()
    return np.unique(s.labels).size < n or n == 1


def _mean_matrix(w, n):
    M = np.zeros((n, n))
    s = 0
    for e in range(n):
        if e == n - 1 or w[e] == 0:
            M[s:e + 1, s:e + 1] = 1.0 / (e + 1 - s)
            s = e + 1
    return M


def _limit_projector(x, w):
    """The mu -> 0 outside limit: the projector onto {d : W D d in span n},
    n = u0 / ||u0|| (spec S2, item 5), from the piece indicators and a d with
    W D d = n."""
    n = x.size
    u0 = dual(x, ref.tv1d_l2(x, 1e300, w), w)[0]
    nd = u0 / np.linalg.norm(u0)
    d = np.concatenate([[0.0], np.cumsum(np.where(w > 0, nd / np.where(w > 0, w, 1), 0.0))])
    V = [d] + [np.where(np.cumsum(np.concatenate([[0], w == 0])) == k, 1.0, 0.0) for k in range((w == 0).sum() + 1)]
    Q, _ = np.linalg.qr(np.column_stack(V))
    return Q @ Q.T


def _boundary_rows(rng, count):
    """(x, gamma, w, distance) on and just outside the boundary ||u0|| = gamma."""
    out = []
    for k in range(count):
        n = (5, 17, 64)[k % 3]
        w = rng.uniform(0.5, 2.0, n - 1)
        if k % 2:
            w[rng.random(n - 1) < 0.2] = 0.0
        x = rng.normal(size=n).cumsum()
        dist = (0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3)[(k // 6) % 6]
        out.append((x, rho0(x, w) / (1 + dist), w, dist))
    return out


def test_boundary_spectrum():
    """The report's counterexample: on the boundary to rounding, 1.0.2 gave
    eigenvalues of J up to 7 (54 of 200 rows outside [0, 1]), and just
    outside it, above 1 by up to 5e-5. Now J is symmetric with its spectrum
    in [0, 1], and its descriptor follows the side it selects: inside the
    piecewise mean and its pieces, outside singletons."""
    rng = np.random.default_rng(20261101)
    for x, g, w, _ in _boundary_rows(rng, 200):
        op = fp.TV1DL2(w)
        n = x.size
        J, ctx = _dense(op, x, g)
        assert np.abs(J - J.T).max() <= 1e-13
        ev = np.linalg.eigvalsh((J + J.T) / 2)
        assert ev.min() >= -1e-12 and ev.max() <= 1 + 1e-12, ev
        if _inside(ctx, n):
            np.testing.assert_allclose(J, _mean_matrix(w, n), atol=1e-15)
        else:
            assert np.unique(ctx.structure().labels).size == n


def test_against_high_precision():
    """J, the parameter tangents and the VJP against central differences of an
    80-digit prox from the original inputs (reference.tv1d_l2_hp), on and near
    the boundary and at moderate distances, with and without zero weights.
    Rows within the rounding of the inside test may select either side: they
    are counted apart, and the side selected is the documented one."""
    rng = np.random.default_rng(7)
    rows = _boundary_rows(rng, 36)
    for k in range(9):  # moderate distances, and weights over four orders
        n = (5, 17, 5)[k % 3]
        w = rng.uniform(0.5, 2.0, n - 1) if k < 6 else 10.0 ** rng.uniform(-2, 2, n - 1)
        x = rng.normal(size=n).cumsum()
        rows.append((x, rho0(x, w) * rng.uniform(0.2, 0.8), w, None))
    ambiguous = 0
    for x, g, w, _ in rows:
        n = x.size
        if n == 64 and rng.random() < 0.75:
            continue  # a few long rows: the reference costs O(n) solves per direction
        op = fp.TV1DL2(w)
        _, ctx = op.prox(x, g, keep=True)
        margin = ref.tv1d_l2_hp_margin(x, g, w)
        if _inside(ctx, n) != (margin <= 0):
            assert abs(margin) <= 1e-14  # the rounding of the inside test
            ambiguous += 1
            continue
        if margin <= 0:
            continue  # inside: the piecewise mean (test_boundary_spectrum)
        V = np.eye(n) if n == 5 else rng.normal(size=(3, n))
        g_out = rng.normal(size=n)
        gx, grads = op.vjp(ctx, g_out)
        for v in V:
            jr = ref.tv1d_l2_hp_dir(x, g, w, v)
            np.testing.assert_allclose(op.jvp(ctx, v), jr, rtol=0, atol=1e-12 * np.abs(v).max())
            assert abs(gx @ v - g_out @ jr) <= 1e-12 * np.abs(g_out).sum() * np.abs(v).max()
        dg = ref.tv1d_l2_hp_dir(x, g, w, dgamma=1.0)
        np.testing.assert_allclose(op.jvp(ctx, None, 1.0), dg, rtol=0, atol=1e-12 * (1 + np.abs(dg).max()))
        assert grads.gamma == pytest.approx(g_out @ dg, rel=1e-11, abs=1e-12 * np.abs(g_out).sum())
        # Positive weights only: a zero weight's direction changes the support
        # (the continuous extension gives it zero, spec S2 item 3).
        dw = rng.normal(size=n - 1) * (w > 0)
        dzw = ref.tv1d_l2_hp_dir(x, g, w, dw=dw)
        np.testing.assert_allclose(op.jvp(ctx, None, 0.0, dw), dzw, rtol=0, atol=1e-12 * (1 + np.abs(dzw).max()))
        assert grads.params @ dw == pytest.approx(g_out @ dzw, rel=1e-11, abs=1e-12 * np.abs(g_out).sum())
    assert ambiguous <= 8


def test_high_precision_reference_is_stable():
    """The reference at 80 digits and h = 1e-30 against 110 digits and
    h = 1e-45, just outside the boundary, where the prox is least smooth."""
    rng = np.random.default_rng(3)
    x, w = rng.normal(size=5).cumsum(), rng.uniform(0.5, 2.0, 4)
    g = rho0(x, w) / (1 + 1e-12)
    v = rng.normal(size=5)
    a = ref.tv1d_l2_hp_dir(x, g, w, v)
    b = ref.tv1d_l2_hp_dir(x, g, w, v, h="1e-45", prec=110)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-20)


@pytest.mark.parametrize("x, w", [
    (np.array([-1.0, -1, 0, -2, 4]), np.ones(4)),  # u0 = (1, 2, 2, 4), ||u0|| = 5
    (np.array([-3.0, -5, 8]), np.array([1.0, 2])),  # u0 = (3, 4)
    (np.array([-3.0, 3, -4, 4, 0]), np.array([1.0, 0, 1, 1])),  # pieces; u0 = (3, 0, 4, 0)
])
def test_exact_boundary(x, w):
    """||u0|| = gamma = 5 exactly: inside (the documented side), J the piecewise
    mean and the pieces as regions. One ulp below, outside: mu is a rounding
    above 0, J is the limit projector onto {d : W D d in span n} to O(mu), and
    the descriptor singletons (J is positive definite at a finite root)."""
    op = fp.TV1DL2(w)
    n = x.size
    J, ctx = _dense(op, x, 5.0)
    assert _inside(ctx, n)
    np.testing.assert_allclose(J, _mean_matrix(w, n), atol=1e-15)
    J, ctx = _dense(op, x, np.nextafter(5.0, 0.0))
    assert not _inside(ctx, n)
    np.testing.assert_allclose(J, _limit_projector(x, w), atol=1e-13)
    ev = np.linalg.eigvalsh((J + J.T) / 2)
    assert ev.min() >= -1e-12 and ev.max() <= 1 + 1e-12


def test_derivatives_at_degenerate_scales():
    """Outside points where the saved z cannot give the direction n, which
    1.0.2 read from its differences: n and mu come from the root solve."""
    rng = np.random.default_rng(12)
    op = fp.TV1DL2()
    D = np.diff(np.eye(17), axis=0)
    # gamma so small that the saved z is x: J = I to rounding, dz/dgamma = -D^T n.
    x = rng.normal(size=17).cumsum()
    nd = D @ x / np.linalg.norm(D @ x)
    for g in (1e-20, 1e-310):  # a regular root, and gamma t below the normal range
        z, ctx = op.prox(x, g, keep=True)
        np.testing.assert_array_equal(z, x)
        v = rng.normal(size=17)
        np.testing.assert_allclose(op.jvp(ctx, v), v, rtol=0, atol=1e-15)
        np.testing.assert_allclose(op.jvp(ctx, None, 1.0), -D.T @ nd, rtol=0, atol=1e-14)
    np.testing.assert_allclose(op.jvp(op.prox(x, 1e-20, keep=True)[1], None, 1.0),
                               ref.tv1d_l2_hp_dir(x, 1e-20, np.ones(16), dgamma=1.0), rtol=0, atol=1e-14)
    # A root beyond the range of doubles (gamma near its bottom, a long row).
    xa = np.where(np.arange(64) % 2, 1.0, -1.0)
    Da = np.diff(np.eye(64), axis=0)
    _, ctx = op.prox(xa, 4.6e-308, keep=True)
    np.testing.assert_allclose(op.jvp(ctx, None, 1.0), -Da.T @ (Da @ xa) / np.linalg.norm(Da @ xa), atol=1e-15)
    np.testing.assert_allclose(op.jvp(ctx, np.ones(64)), np.ones(64), atol=1e-15)
    # A large common offset, just outside: the saved z is constant (its
    # differences are below the offset's rounding) though mu > 0. 1.0.2 took
    # the identity there; J is that of the centred row (translation invariance).
    # The pattern's mean is exact, so the inside test is exact at the offset.
    k = np.array([0.0, 1, 3, 2, 4])
    g = rho0(16 * k, np.ones(4)) / (1 + 1e-6)
    z, ctx = op.prox(1e17 + 16 * k, g, keep=True)
    assert np.all(np.diff(z) == 0)
    _, c0 = op.prox(16 * k, g, keep=True)
    assert not _inside(ctx, 5)
    for v in np.eye(5):
        np.testing.assert_allclose(op.jvp(ctx, v), op.jvp(c0, v), rtol=0, atol=1e-15)
        np.testing.assert_allclose(op.jvp(ctx, v), ref.tv1d_l2_hp_dir(16 * k, g, np.ones(4), v), rtol=0, atol=1e-12)
    # All-zero weights: every entry its own piece, J = I.
    opz = fp.TV1DL2(np.zeros(4))
    J, cz = _dense(opz, rng.normal(size=5), 1.0)
    np.testing.assert_array_equal(J, np.eye(5))
    assert np.unique(cz.structure().labels).size == 5


def test_float32_near_boundary():
    """float32 rows on and near the boundary: J against the float64 reference
    of the same (float32) inputs, symmetric with its spectrum in [0, 1] at
    float32 accuracy."""
    rng = np.random.default_rng(32)
    for x, g, w, dist in _boundary_rows(rng, 30):
        x, w = x.astype(np.float32), w.astype(np.float32)
        g = float(np.float32(g))
        op = fp.TV1DL2(w)
        J, ctx = _dense(op, x, g, np.float32)
        assert np.abs(J - J.T).max() <= 1e-6
        ev = np.linalg.eigvalsh((J + J.T) / 2)
        assert ev.min() >= -1e-5 and ev.max() <= 1 + 1e-5
        if dist and dist >= 1e-6 and not _inside(ctx, x.size):
            v = rng.normal(size=x.size).astype(np.float32)
            jr = ref.tv1d_l2_hp_dir(x.astype(float), g, w.astype(float), v.astype(float))
            np.testing.assert_allclose(op.jvp(ctx, v), jr, rtol=0, atol=1e-5 * np.abs(v).max())


def test_conj_near_boundary():
    """After prox_conj (positively homogeneous: y - prox(y) at step 1), J_conj
    = I - J_inner and the parameter tangents are the inner's negated, near the
    boundary too; the tangent in sigma is 0."""
    rng = np.random.default_rng(13)
    for x, g, w, _ in _boundary_rows(rng, 12):
        op = fp.TV1DL2(w * g)  # the boundary at step 1
        _, cc = op.prox_conj(x, 2.5, keep=True)
        _, ci = op.prox(x, 1.0, keep=True)
        v, dw = rng.normal(size=x.size), rng.normal(size=x.size - 1)
        np.testing.assert_allclose(op.jvp(cc, v), v - op.jvp(ci, v), rtol=0, atol=1e-13)
        np.testing.assert_allclose(op.jvp(cc, None, 0.0, dw), -op.jvp(ci, None, 0.0, dw), rtol=0, atol=1e-13)
        np.testing.assert_array_equal(op.jvp(cc, None, 1.0), 0.0)
