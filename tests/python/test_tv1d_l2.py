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
        assert np.linalg.norm(u) <= g * (1 + 1e-9)
        assert g * s - wdz @ u <= 1e-12 * (1 + g * s)
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
