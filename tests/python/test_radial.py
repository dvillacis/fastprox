"""The log barrier, l1 over a box, the second-order cone and the group
penalties by norm composition (group Huber, SCAD, MCP): optimality conditions,
independent minimisation, kink rules, conjugates, extreme inputs and errors.
Derivatives, threads and strides run in test_operators.py."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref


# ---------------------------------------------------------------- log barrier


@pytest.mark.parametrize("gamma", [1e-8, 0.3, 1.0, 50.0])
@pytest.mark.parametrize("lam", [1e-6, 0.5, 20.0])
def test_log_barrier_optimality(gamma, lam):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(scale=s, size=200) for s in (1e-3, 1.0, 1e3)])
    z = fp.prox_log_barrier(x, lam, gamma)
    c = gamma * lam
    assert np.all(z > 0)
    # Stationarity z - x - c / z = 0, to rounding of the terms.
    r = z - x - c / z
    assert np.all(np.abs(r) <= 4e-16 * (np.abs(x) + z + c / z))


def test_log_barrier_extremes():
    x = np.array([-1e300, -1e200, -1e20, 0.0, 1e20, 1e200, 1e300, np.inf, -np.inf])
    z = fp.prox_log_barrier(x, 2.0, 1.5)
    assert np.all(z[:7] > 0) and np.all(np.isfinite(z[:7]))
    np.testing.assert_allclose(z[:3], 3.0 / -x[:3], rtol=1e-15)  # c / |x| for x -> -inf
    assert z[3] == pytest.approx(np.sqrt(3.0), rel=1e-15)
    np.testing.assert_allclose(z[4:7], x[4:7], rtol=1e-15)
    assert z[7] == np.inf and z[8] == 0.0
    assert np.isnan(fp.prox_log_barrier(np.array([np.nan]))[0])
    # A huge step: the square of x overflows; 4 gamma lam stays in the step
    # range (gamma lam <= 2^960, since 1.0), beyond which the call fails.
    assert fp.prox_log_barrier(np.array([-1e200]), 1e280, 1e8)[0] == pytest.approx(1e88, rel=1e-14)
    with pytest.raises(ValueError):
        fp.prox_log_barrier(np.array([-1e200]), 1e300, 1e8)
    z32 = fp.prox_log_barrier(np.array([-1e30, 3.0], np.float32), 0.5)
    assert z32.dtype == np.float32 and np.all(z32 > 0)


def test_log_barrier_derivative_at_gamma_zero():
    op = fp.LogBarrier(1.0)
    x = np.array([-1.0, 0.0, 2.0])
    z, ctx = op.prox(x, 0.0, keep=True)
    np.testing.assert_array_equal(z, [0.0, 0.0, 2.0])
    np.testing.assert_array_equal(op.jvp(ctx, np.ones(3)), [0.0, 1.0, 1.0])


# ---------------------------------------------------------------- l1 over a box


@pytest.mark.parametrize("bounds", [(-0.5, 0.8), (0.2, 0.9), (-0.9, -0.2), (0.0, np.inf),
                                    (-np.inf, np.inf), (0.3, 0.3)])
def test_l1_box_minimises(bounds):
    lo, hi = bounds
    rng = np.random.default_rng(1)
    x = rng.normal(scale=1.5, size=60)
    w = rng.uniform(0.0, 2.0, 60)
    gamma, lam = 0.7, 0.6
    z = fp.prox_l1_box(x, lam, lo, hi, gamma, weights=w)
    assert np.all(z >= lo) and np.all(z <= hi)
    # Entry by entry against a refined grid on the interval.
    for xi, wi, zi in zip(x, w, z):
        a, b = max(lo, -abs(xi) - 1), min(hi, abs(xi) + 1)
        for _ in range(6):
            g = np.append(np.linspace(a, b, 2001), [a, b])
            f = 0.5 * (g - xi) ** 2 + gamma * lam * wi * np.abs(g)
            best = g[np.argmin(f)]
            h = (b - a) / 2000
            a, b = max(lo, best - 4 * h), min(hi, best + 4 * h)
        F = lambda v: 0.5 * (v - xi) ** 2 + gamma * lam * wi * abs(v)
        assert F(zi) <= F(best) + 1e-15 * (1 + xi * xi)  # no better point on the grid
        assert abs(zi - best) <= 1e-7 * (1 + abs(xi))    # a flat minimum: ~sqrt(eps)
    # The unbounded box is soft thresholding.
    if bounds == (-np.inf, np.inf):
        np.testing.assert_array_equal(z, fp.soft_threshold(x, gamma * lam, weights=w))


def test_l1_box_kinks():
    op = fp.L1Box(1.0, -0.5, 0.5)
    x = np.array([1.5, -1.5, 1.0, 0.3, -1.6])  # at the upper bound, the lower bound, the threshold
    z, ctx = op.prox(x, 1.0, keep=True)
    np.testing.assert_array_equal(z, [0.5, -0.5, 0.0, 0.0, -0.5])
    # At a bound the box is interior (soft's Jacobian); at |x| = tau thresholded.
    np.testing.assert_array_equal(op.jvp(ctx, np.ones(5)), [1.0, 1.0, 0.0, 0.0, 0.0])
    gx, grads = op.vjp(ctx, np.ones(5))
    assert grads.upper == 0.0 and grads.lower == 1.0


# ---------------------------------------------------------------- second-order cone


def soc_kkt(x, z):
    """z in K, x - z in the polar cone -K, <z, x - z> = 0 (Moreau)."""
    tol = 1e-13 * (1 + np.abs(x).max()) * np.sqrt(x.size)
    r = z - x
    assert np.linalg.norm(z[1:]) <= z[0] + tol
    assert np.linalg.norm(r[1:]) <= r[0] + tol
    assert abs(z @ r) <= tol * (1 + np.abs(x).max())


@pytest.mark.parametrize("n", [1, 2, 3, 10, 1000])
def test_soc_kkt(n):
    rng = np.random.default_rng(n)
    for scale in (1e-3, 1.0, 1e3):
        x = rng.normal(scale=scale, size=(40, n))
        x[:, 0] *= np.linspace(-3, 3, 40) * np.sqrt(n)  # all three branches
        z = fp.project_soc(x)
        for xr, zr in zip(x, z):
            soc_kkt(xr, zr)
        np.testing.assert_allclose(z, np.stack([ref.soc(r) for r in x]), rtol=1e-13,
                                   atol=1e-14 * scale)
        np.testing.assert_allclose(fp.project_soc(z), z, rtol=1e-14, atol=1e-15 * scale)


def test_soc_branches_and_kinks():
    op = fp.SOC()
    x = np.array([[5.0, 3.0, 4.0],    # on the cone: kept, identity Jacobian
                  [-5.0, 3.0, 4.0],   # on the polar cone: 0, zero Jacobian
                  [0.0, 0.0, 0.0],    # the origin: identity
                  [1.0, 3.0, 4.0]])   # strictly between
    z, ctx = op.prox(x, keep=True)
    np.testing.assert_array_equal(z[:3], x[:3] * [[1], [0], [1]])
    np.testing.assert_allclose(z[3], [3.0, 1.8, 2.4], rtol=1e-15)
    dx = np.arange(1.0, 13.0).reshape(4, 3)
    dz = op.jvp(ctx, dx)
    np.testing.assert_array_equal(dz[:3], dx[:3] * [[1], [0], [1]])
    # The Jacobian of the middle branch is symmetric.
    J = np.stack([op.jvp(ctx, np.eye(3)[k] * np.array([[0], [0], [0], [1]]))[3] for k in range(3)])
    np.testing.assert_allclose(J, J.T, atol=1e-15)


def test_soc_conjugate_is_polar_projection():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(50, 6))
    # The polar of the self-dual cone is -K: P_{K°}(y) = -P_K(-y).
    np.testing.assert_allclose(fp.SOC().prox_conj(y, 0.7), -fp.project_soc(-y), atol=1e-15)


# ---------------------------------------------------------------- group penalties


def radius_min(s, step, psi):
    """argmin over r >= 0 of psi(r) + (r - s)^2 / (2 step), on a refined grid."""
    a, b = 0.0, s + 1.0
    for _ in range(8):
        g = np.append(np.linspace(a, b, 4001), [0.0])
        f = psi(g) + (g - s) ** 2 / (2 * step)
        best = g[np.argmin(f)]
        h = (b - a) / 4000
        a, b = max(0.0, best - 4 * h), best + 4 * h
    return best


@pytest.mark.parametrize("kind", ["huber", "scad", "mcp"])
def test_group_penalties_minimise(kind):
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(6), [1, 2, 3, 4, 5, 6])
    c = rng.uniform(0.2, 1.5, 6)
    gamma = 0.9
    if kind == "huber":
        op, psi = fp.GroupHuber(0.8, 0.6, groups, c), lambda r: 0.8 * np.where(
            r <= 0.6, r * r / 1.2, r - 0.3)
    elif kind == "scad":
        op, psi = fp.GroupSCAD(1.0, 3.7, groups, c), ref.scad_psi(1.0, 3.7)
    else:
        op, psi = fp.GroupMCP(1.0, 3.0, groups, c), ref.mcp_psi(1.0, 3.0)
    for scale in (0.3, 1.0, 3.0):
        x = rng.normal(scale=scale, size=groups.size)
        z = op.prox(x, gamma)
        for g in range(6):
            m = groups == g
            s = np.linalg.norm(x[m])
            r = radius_min(s, gamma * c[g], psi)
            # z_g is r times the direction of x_g.
            np.testing.assert_allclose(z[m], r * x[m] / s, atol=1e-9 * (1 + s))


def test_group_huber_closed_form():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 7)) * np.linspace(0.01, 3, 30)[:, None]
    lam, delta, gamma = 0.7, 0.4, 1.3
    z = fp.group_huber(x, lam, delta, gamma)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    beck = (1 - gamma * lam / np.maximum(nrm, delta + gamma * lam)) * x  # Example 6.66
    np.testing.assert_allclose(z, beck, rtol=1e-14, atol=1e-16)
    # One-entry groups are the scalar Huber map.
    np.testing.assert_allclose(fp.group_huber(x, lam, delta, gamma, groups=np.arange(7)),
                               fp.huber(x, lam, delta, gamma), rtol=1e-15, atol=1e-16)
    np.testing.assert_allclose(fp.group_scad(x, 1.0, 3.7, 0.8, groups=np.arange(7)),
                               fp.scad(x, 1.0, 3.7, 0.8), rtol=1e-15, atol=1e-16)


def test_group_penalties_at_zero_and_weights():
    op = fp.GroupHuber(0.5, 1.0, np.array([0, 0, 1, 1]), np.array([1.0, 0.0]))
    x = np.array([0.0, 0.0, 0.3, -0.4])
    z, ctx = op.prox(x, 2.0, keep=True)
    np.testing.assert_array_equal(z, [0.0, 0.0, 0.3, -0.4])  # c = 0: identity
    # At x_g = 0 the Jacobian is phi'(0) I = delta / (delta + gamma lam) I.
    dz = op.jvp(ctx, np.ones(4))
    np.testing.assert_allclose(dz, [0.5, 0.5, 1.0, 1.0], rtol=1e-15)
    gx, grads = op.vjp(ctx, np.ones(4))
    assert grads.group_weights.shape == (2,) and set(grads._fields) >= {"lam", "delta"}


def test_group_penalties_nan_and_inf():
    for op in (fp.GroupHuber(), fp.GroupSCAD(), fp.GroupMCP(), fp.SOC()):
        z = op.prox(np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]]))
        assert np.all(np.isnan(z[0])) and np.all(np.isfinite(z[1]))
        with pytest.raises((ValueError, FloatingPointError, RuntimeError)):
            op.prox(np.array([1.0, np.inf]))


def test_errors():
    x = np.ones(4)
    for bad in (lambda: fp.LogBarrier(0.0), lambda: fp.LogBarrier(-1.0),
                lambda: fp.L1Box(1.0, 1.0, 0.0),                         # lo > hi
                lambda: fp.L1Box(-1.0), lambda: fp.L1Box(1.0, weights=[-1.0]),
                lambda: fp.L1Box(1.0, np.zeros(3)).prox(x),              # bounds of another length
                lambda: fp.GroupHuber(1.0, 0.0),
                lambda: fp.GroupSCAD(1.0, 1.0), lambda: fp.GroupMCP(1.0, 0.0),
                lambda: fp.GroupSCAD(1.0, 3.0, None, [2.0]).prox(x, 1.0),  # a <= 1 + gamma c
                lambda: fp.GroupMCP(1.0, 3.0, None, [2.0]).prox(x, 1.5),   # b <= gamma c
                lambda: fp.GroupHuber(1.0, 1.0, np.array([0, 1])).prox(x),
                lambda: fp.GroupHuber(1.0, 1.0, None, [-1.0])):
        with pytest.raises(ValueError):
            bad()
    fp.GroupSCAD(1.0, 3.0, None, [2.0]).prox(x, 0.9)


def test_group_step_in_range():
    """The scalar map's step is gamma c_g, which h then multiplies by its own
    factors (since 1.0.1). Group Huber's step range counted only gamma lambda
    c_g: at lambda = 1e-280, gamma lambda c_g = 1.1e273 but gamma c_g
    overflowed, and the group came out NaN (fuzz, 2026-10-25). The range is
    now gamma max(lambda, 1) c_max <= 2^960."""
    x = np.array([-84.0, 3.0, 1e-30])
    op = fp.GroupHuber(1e-280, 1.0, None, np.array([2.4e275]))
    with pytest.raises(ValueError):
        op.prox(x, 4.6e277)
    z = op.prox(x, 1e10)  # gamma c = 2.4e285, gamma lambda c = 2.4e5
    assert np.all(np.isfinite(z)) and np.all(np.abs(z) <= np.abs(x))
    # lambda >= 1: the range as before.
    op = fp.GroupHuber(1e10, 1.0, None, np.array([1.0]))
    assert np.all(np.isfinite(op.prox(x, 2.0**920)))
