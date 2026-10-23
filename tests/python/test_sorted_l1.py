# The sorted l1 norm (OWL, SLOPE) and the weighted nuclear norm (feedback
# U5): the prox against cvxpy where pooling merges blocks, soft thresholding
# bit for bit at equal weights, derivatives against finite differences, and
# the spectral operator against its optimality conditions.
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

import fastprox as fp

import reference as ref


def _clustered(rng, rows, n):
    """Magnitudes in tight clusters, so that most blocks pool."""
    centres = rng.uniform(0.5, 3.0, size=(rows, 3))
    pick = rng.integers(0, 3, size=(rows, n))
    mag = np.take_along_axis(centres, pick, 1) + rng.uniform(-0.05, 0.05, size=(rows, n))
    return mag * rng.choice([-1.0, 1.0], size=(rows, n))


def _weights(n, seed=0):
    return np.sort(np.random.default_rng(seed).uniform(0.1, 1.5, n))[::-1].copy()


def test_against_cvxpy():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(1)
    n = 12
    w = _weights(n)
    x = _clustered(rng, 5, n)
    z = fp.SortedL1(w).prox(x, 0.7)
    dw = np.append(w[:-1] - w[1:], w[-1])  # sum_k w_k |z|_(k) = sum_k dw_k S_k(|z|)
    for xr, zr in zip(x, z):
        v = cp.Variable(n)
        h = sum(dw[k] * cp.sum_largest(cp.abs(v), k + 1) for k in range(n))
        cp.Problem(cp.Minimize(0.5 * cp.sum_squares(v - xr) + 0.7 * h)).solve(solver=cp.CLARABEL)
        np.testing.assert_allclose(zr, v.value, atol=1e-6)
        np.testing.assert_allclose(zr, ref.sorted_l1(xr, 0.7, w), rtol=1e-12, atol=1e-12)
    # The pooling did merge blocks here: tied magnitudes in the output.
    assert any(len(np.unique(np.abs(zr[zr != 0]))) < np.count_nonzero(zr) for zr in z)


def test_equal_weights_are_soft_bitwise():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 50))
    g, dx = rng.normal(size=x.shape), rng.normal(size=x.shape)
    for weights in (0.7, np.full(50, 0.7)):
        a, b = fp.SortedL1(weights), fp.Soft(0.7)
        za, ca = a.prox(x, 0.8, keep=True)
        zb, cb = b.prox(x, 0.8, keep=True)
        assert np.array_equal(za, zb)
        assert np.array_equal(a.jvp(ca, dx, 0.3), b.jvp(cb, dx, 0.3))
        (ga, ra), (gb, rb) = a.vjp(ca, g), b.vjp(cb, g)
        assert np.array_equal(ga, gb)
        # The step's gradient sums block by block in sorted order, Soft's entry
        # by entry: equal to rounding.
        assert ra.gamma == pytest.approx(rb.gamma, rel=1e-14)
        assert np.sum(ra.weights) == pytest.approx(rb.lam, rel=1e-14)
        assert np.array_equal(a.prox_conj(x, 0.8), b.prox_conj(x, 0.8))
    np.testing.assert_allclose(fp.SortedL1(0.7).value(x), fp.Soft(0.7).value(x), rtol=1e-15)


def test_derivatives_where_blocks_pool():
    rng = np.random.default_rng(3)
    n = 16
    w = _weights(n, 4)
    x = _clustered(rng, 4, n)
    op = fp.SortedL1(w)
    gamma = 0.6
    z, ctx = op.prox(x, gamma, keep=True)
    dx, dw = rng.normal(size=x.shape), rng.normal(size=n) * 1e-3  # small: w stays sorted
    dz = op.jvp(ctx, dx, 0.4, dw)
    h = 1e-6
    fd = (fp.SortedL1(w + h * dw).prox(x + h * dx, gamma + 0.4 * h)
          - fp.SortedL1(w - h * dw).prox(x - h * dx, gamma - 0.4 * h)) / (2 * h)
    np.testing.assert_allclose(dz, fd, rtol=1e-6, atol=1e-7)
    g = rng.normal(size=x.shape)
    gx, grads = op.vjp(ctx, g)
    assert np.vdot(g, dz) == pytest.approx(np.vdot(gx, dx) + 0.4 * grads.gamma + np.vdot(grads.weights, dw),
                                           rel=1e-12)


def test_value_and_errors():
    x = np.array([[0.5, -3.0, 2.0, 0.0]])
    w = np.array([2.0, 1.0, 1.0, 0.5])
    assert fp.SortedL1(w).value(x)[0] == pytest.approx(2 * 3 + 1 * 2 + 1 * 0.5 + 0.5 * 0)
    for bad in ([1.0, 2.0], [1.0, -0.5], [np.nan], [np.inf, 1.0]):
        with pytest.raises(ValueError):
            fp.SortedL1(bad)
    with pytest.raises(ValueError):
        fp.SortedL1([2.0, 1.0, 0.5]).prox(x, 1.0)  # weights for 3 entries, rows of 4
    with pytest.raises(ValueError, match="WeightedNuclearNorm"):
        fp.Spectral(fp.Soft(1.0, [1.0, 2.0, 3.0]))


def _objective(z, x, gamma, w):
    return 0.5 * np.sum((z - x) ** 2) + gamma * np.sum(w * np.linalg.svd(z, compute_uv=False))


def test_weighted_nuclear_norm_optimality():
    """6 x 5 matrices: no perturbation of the prox lowers the objective (a
    convex function, so local optimality is global)."""
    rng = np.random.default_rng(5)
    w = np.array([1.2, 0.9, 0.9, 0.4, 0.1])
    op = fp.WeightedNuclearNorm(w)
    for _ in range(4):
        x = rng.normal(size=(6, 5)) * 2
        z = op.prox(x, 0.8)
        f = _objective(z, x, 0.8, w)
        for _ in range(200):
            d = rng.normal(size=z.shape) * 1e-4
            assert f <= _objective(z + d, x, 0.8, w) + 1e-15
        s = np.linalg.svd(z, compute_uv=False)
        np.testing.assert_allclose(s, ref.sorted_l1(np.linalg.svd(x, compute_uv=False), 0.8, w), atol=1e-12)


def test_weighted_nuclear_norm_derivatives_at_ties():
    """Repeated singular values: the JVP against central differences in a
    generic direction, and the VJP adjoint to it."""
    rng = np.random.default_rng(6)
    u, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    v, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    s = np.array([3.0, 2.0, 2.0, 2.0, 0.5])
    x = (u[:, :5] * s) @ v.T
    w = np.array([1.0, 0.8, 0.5, 0.5, 0.2])
    op = fp.WeightedNuclearNorm(w)
    z, ctx = op.prox(x, 0.5, keep=True)
    d = rng.normal(size=x.shape)
    dz = op.jvp(ctx, d)
    h = 1e-6
    fd = (op.prox(x + h * d, 0.5) - op.prox(x - h * d, 0.5)) / (2 * h)
    np.testing.assert_allclose(dz, fd, atol=1e-6)
    g = rng.normal(size=x.shape)
    gx, _ = op.vjp(ctx, g)
    assert np.vdot(g, dz) == pytest.approx(np.vdot(gx, d), rel=1e-10)


def test_spectral_check_and_eigh():
    w = _weights(5)
    assert fp.WeightedNuclearNorm(w).kind == "svd"
    a = np.random.default_rng(7).normal(size=(5, 5))
    a = a + a.T
    z = fp.Spectral(fp.SortedL1(w), kind="eigh").prox(a, 0.3)
    lam, q = np.linalg.eigh(a)
    want = (q * ref.sorted_l1(lam, 0.3, w)) @ q.T
    np.testing.assert_allclose(z, want, atol=1e-12)
