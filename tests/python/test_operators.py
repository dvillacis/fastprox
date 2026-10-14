"""Operator tests through the Python bindings: reference agreement,
optimality certificates, derivatives, prox properties and batching."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import fastprox as fp

import reference as ref

N = 10
_rng = np.random.default_rng(2026)
W = _rng.uniform(0.5, 1.5, N)
WE = _rng.uniform(0.2, 1.5, N - 1)
LO = _rng.uniform(-1.0, 0.0, N)
HI = _rng.uniform(0.0, 1.0, N)
GROUPS = np.arange(N) * 7 % 3


# name: (build(theta) -> Operator, theta0, reference(x_row, gamma, theta) or None,
#        gamma, convex)
CASES = {
    "soft": (lambda t: fp.Soft(0.4, t), W, lambda x, g, t: ref.soft(x, g * 0.4 * t), 0.8),
    # theta is the weight vector (here a broadcast scalar), not lam.
    "soft_scalar": (lambda t: fp.Soft(0.4, t), np.array([1.3]),
                    lambda x, g, t: ref.soft(x, g * 0.4 * t[0]), 0.8),
    "elastic_net": (lambda t: fp.ElasticNet(*t), np.array([0.3, 0.7]),
                    lambda x, g, t: ref.elastic_net(x, g, *t), 0.9),
    "huber": (lambda t: fp.Huber(*t), np.array([0.6, 0.5]),
              lambda x, g, t: ref.huber(x, g, *t), 0.7),
    "group_l2": (lambda t: fp.GroupL2(0.6, GROUPS, t), np.array([0.8, 1.0, 1.2]),
                 lambda x, g, t: ref.group_l2(x, g, 0.6, GROUPS, t), 0.9),
    "l2": (lambda t: fp.GroupL2(0.5, None, t), np.array([1.2]),
           lambda x, g, t: ref.group_l2(x, g, 0.5, np.zeros(x.size, int), t), 0.9),
    "box": (lambda t: fp.Box(t[:N], t[N:]), np.concatenate([LO, HI]),
            lambda x, g, t: np.clip(x, t[:N], t[N:]), 1.0),
    "nonneg": (lambda t: fp.NonNeg(), np.zeros(0), lambda x, g, t: np.maximum(x, 0), 1.0),
    "l2_ball": (lambda t: fp.L2Ball(t[0]), np.array([0.7]),
                lambda x, g, t: ref.l2_ball(x, t[0]), 1.0),
    "simplex": (lambda t: fp.Simplex(t[0]), np.array([1.3]),
                lambda x, g, t: ref.simplex(x, t[0]), 1.0),
    "l1_ball": (lambda t: fp.L1Ball(t[0]), np.array([1.1]),
                lambda x, g, t: ref.l1_ball(x, t[0]), 1.0),
    "l1_ball_w": (lambda t: fp.L1Ball(t[0], W), np.array([1.1]),
                  lambda x, g, t: ref.l1_ball(x, t[0], W), 1.0),
    "linf": (lambda t: fp.LInf(t[0]), np.array([0.9]),
             lambda x, g, t: ref.linf(x, g, t[0]), 1.0),
    "isotonic": (lambda t: fp.Isotonic(), np.zeros(0), lambda x, g, t: ref.isotonic(x), 1.0),
    "isotonic_w": (lambda t: fp.Isotonic(t, increasing=False), W,
                   lambda x, g, t: ref.isotonic(x, t, increasing=False), 1.0),
    "tv1d": (lambda t: fp.TV1D(t), np.array([1.0]), None, 0.6),
    "tv1d_w": (lambda t: fp.TV1D(t), WE, None, 0.6),
    "tv1d_taut": (lambda t: fp.TV1D(t, method="taut_string"), np.array([1.0]), None, 0.6),
}
NAMES = sorted(CASES)


def apply_rows(f, x):
    return np.stack([f(r) for r in x.reshape(-1, x.shape[-1])]).reshape(x.shape)


# ---------------------------------------------------------------- forward


@pytest.mark.parametrize("name", NAMES)
def test_matches_reference(name):
    build, theta, reference, gamma = CASES[name]
    op = build(theta)
    rng = np.random.default_rng(1)
    for scale in (0.1, 1.0, 10.0):
        x = rng.normal(scale=scale, size=(7, N))
        z = op.prox(x, gamma)
        if reference is None:
            w = theta if theta.size > 1 else theta[0]
            for xr, zr in zip(x, z):
                assert ref.tv1d_kkt_residual(xr, zr, gamma, w) < 1e-12
        else:
            np.testing.assert_allclose(
                z, apply_rows(lambda r: reference(r, gamma, theta), x), rtol=1e-12,
                atol=1e-12 * scale)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 100, 1000, 20000])
@pytest.mark.parametrize("gamma", [1e-3, 0.3, 2.0, 100.0])
def test_tv1d_optimality(n, gamma):
    rng = np.random.default_rng(n)
    signals = [
        rng.normal(size=n),
        np.repeat(rng.normal(scale=5, size=n // 10 + 1), 10)[:n] + 0.2 * rng.normal(size=n),
        np.cumsum(rng.normal(size=n)),
        (-1.0) ** np.arange(n) * (1 + 1e-3 * np.arange(n)),
    ]
    for x in signals:
        zs = [fp.tv1d(x, gamma, method=m) for m in ("auto", "condat", "taut_string")]
        for z in zs:
            assert ref.tv1d_kkt_residual(x, z, gamma, 1.0) < 1e-11
        np.testing.assert_allclose(zs[1], zs[2], rtol=0, atol=1e-10 * (1 + np.abs(x).max()))


def test_tv1d_weighted_with_zeros():
    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(2, 400))
        w = rng.uniform(0, 2, n - 1)
        w[rng.random(n - 1) < 0.1] = 0.0
        x = np.cumsum(rng.normal(size=n))
        gamma = rng.uniform(0.01, 5)
        z = fp.tv1d(x, gamma, weights=w)
        assert ref.tv1d_kkt_residual(x, z, gamma, w) < 1e-11


def test_tv1d_gamma_zero_is_identity():
    x = np.random.default_rng(0).normal(size=(3, 50))
    np.testing.assert_array_equal(fp.tv1d(x, 0.0), x)


def test_edge_cases():
    x = np.random.default_rng(0).normal(size=20)
    np.testing.assert_array_equal(fp.project_simplex(x, 0.0), 0)
    np.testing.assert_array_equal(fp.project_l1_ball(x, 0.0), 0)
    assert fp.tv1d(np.empty(0)).shape == (0,)
    np.testing.assert_array_equal(fp.tv1d(np.array([2.5]), 3.0), [2.5])
    np.testing.assert_array_equal(fp.project_simplex(np.array([7.0]), 2.0), [2.0])


# ---------------------------------------------------------------- derivatives


def _derivative_setup(name, conj, seed):
    build, theta, _, gamma = CASES[name]
    rng = np.random.default_rng(seed)
    op = build(theta)
    x = rng.normal(scale=1.5, size=(3, N))
    prox = op.prox_conj if conj else op.prox
    z, ctx = prox(x, gamma, keep=True)
    return build, theta, gamma, op, x, z, ctx, rng


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_adjointness(name, conj):
    for seed in range(5):
        _, theta, _, op, x, _, ctx, rng = _derivative_setup(name, conj, seed)
        dx = rng.normal(size=x.shape)
        dth = rng.normal(size=theta.size)
        dg = 0.37
        g = rng.normal(size=x.shape)
        dz = op.jvp(ctx, dx, dg, dth if theta.size else None)
        gx, grads = op.vjp(ctx, g)
        lhs = np.vdot(g, dz)
        rhs = np.vdot(gx, dx) + grads.gamma * dg + np.vdot(grads.params, dth)
        assert lhs == pytest.approx(rhs, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_jvp_finite_differences(name, conj):
    h = 1e-6
    for seed in range(5):
        build, theta, gamma, op, x, _, ctx, rng = _derivative_setup(name, conj, seed)
        dx = rng.normal(size=x.shape)
        dth = rng.normal(size=theta.size)
        dg = 0.37

        def f(t):
            o = build(theta + t * dth)
            prox = o.prox_conj if conj else o.prox
            return prox(x + t * dx, gamma + t * dg)

        fd = (f(h) - f(-h)) / (2 * h)
        dz = op.jvp(ctx, dx, dg, dth if theta.size else None)
        np.testing.assert_allclose(dz, fd, rtol=1e-5, atol=1e-5 * (1 + np.abs(fd).max()))


def test_named_grads():
    w = np.linspace(0.5, 1.5, N - 1)
    op = fp.TV1D(w)
    x = np.random.default_rng(0).normal(size=N)
    z, ctx = op.prox(x, 0.5, keep=True)
    gx, grads = op.vjp(ctx, np.ones(N))
    assert grads.weights.shape == (N - 1,)
    np.testing.assert_array_equal(grads.weights, grads.params)
    # dparams by name.
    dz = op.jvp(ctx, None, 0.0, {"weights": np.ones(N - 1)})
    np.testing.assert_allclose(dz, op.jvp(ctx, None, 0.0, np.ones(N - 1)))
    # Soft also reports the gradient in lam.
    op = fp.Soft(0.3)
    z, ctx = op.prox(x, 1.0, keep=True)
    _, grads = op.vjp(ctx, np.ones(N))
    h = 1e-7
    fd = (fp.soft_threshold(x, 0.3 + h).sum() - fp.soft_threshold(x, 0.3 - h).sum()) / (2 * h)
    assert grads.lam == pytest.approx(fd, rel=1e-6)


def test_derivative_requires_structure():
    op = fp.TV1D()
    with pytest.raises(ValueError):
        op.jvp(None, np.ones(3))


# ---------------------------------------------------------------- properties

finite = st.floats(-100, 100, allow_nan=False, allow_infinity=False)
rows = hnp.arrays(np.float64, st.integers(1, 30), elements=finite)


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(NAMES), st.data())
def test_firm_nonexpansive(name, data):
    build, theta, _, gamma = CASES[name]
    n = N
    x1 = data.draw(hnp.arrays(np.float64, n, elements=finite))
    x2 = data.draw(hnp.arrays(np.float64, n, elements=finite))
    op = build(theta)
    z1, z2 = op.prox(x1, gamma), op.prox(x2, gamma)
    d = z1 - z2
    tol = 1e-10 * (1 + np.abs(x1).max() + np.abs(x2).max()) ** 2
    assert d @ d <= d @ (x1 - x2) + tol


@settings(max_examples=60, deadline=None)
@given(rows, st.floats(0.0, 10.0))
def test_projections_idempotent(x, r):
    for op in (fp.Simplex(r), fp.L1Ball(r), fp.L2Ball(r), fp.Isotonic(), fp.Box(-r, r)):
        z = op.prox(x)
        np.testing.assert_allclose(op.prox(z), z, rtol=1e-12, atol=1e-12 * (1 + r))


@settings(max_examples=60, deadline=None)
@given(rows, st.floats(0.01, 10.0), st.floats(0.01, 5.0))
def test_moreau_conjugates(y, sigma, lam):
    # (lam ||.||_1)* is the indicator of the l-inf ball of radius lam.
    np.testing.assert_allclose(fp.Soft(lam).prox_conj(y, sigma), np.clip(y, -lam, lam),
                               rtol=1e-12, atol=1e-12 * (1 + np.abs(y).max()))
    # (lam ||.||_2)* is the indicator of the l2 ball of radius lam.
    np.testing.assert_allclose(fp.GroupL2(lam).prox_conj(y, sigma), ref.l2_ball(y, lam),
                               rtol=1e-11, atol=1e-11 * (1 + np.abs(y).max()))
    # (lam ||.||_inf)* is the indicator of the l1 ball of radius lam.
    np.testing.assert_allclose(fp.LInf(lam).prox_conj(y, sigma), ref.l1_ball(y, lam),
                               rtol=1e-10, atol=1e-10 * (1 + np.abs(y).max()))


# ---------------------------------------------------------------- batching


@pytest.mark.parametrize("name", NAMES)
def test_threads_bitwise(name):
    build, theta, _, gamma = CASES[name]
    op = build(theta)
    x = np.random.default_rng(5).normal(size=(4000, N))
    outs = [op.prox(x, gamma, threads=t) for t in (1, 2, 8)]
    assert outs[0].tobytes() == outs[1].tobytes() == outs[2].tobytes()
    grads = []
    g = np.sin(np.arange(x.size)).reshape(x.shape)
    for t in (1, 3, 8):
        z, ctx = op.prox(x, gamma, keep=True, threads=t)
        gx, gr = op.vjp(ctx, g)
        grads.append(np.append(gr.params, gr.gamma).tobytes() + gx.tobytes())
    assert grads[0] == grads[1] == grads[2]


@pytest.mark.parametrize("name", NAMES)
def test_axis_and_strides(name):
    build, theta, _, gamma = CASES[name]
    op = build(theta)
    x = np.random.default_rng(6).normal(size=(5, N, 4))
    z1 = op.prox(x, gamma, axis=1)
    z2 = np.moveaxis(op.prox(np.ascontiguousarray(np.moveaxis(x, 1, -1)), gamma), -1, 1)
    np.testing.assert_array_equal(z1, z2)
    # Reversed, sliced views give the same result as copies.
    v = x[::-1, :, ::2]
    np.testing.assert_array_equal(op.prox(v, gamma, axis=1),
                                  op.prox(np.ascontiguousarray(v), gamma, axis=1))
    # In place.
    y = x.copy()
    op.prox(y, gamma, axis=1, out=y)
    np.testing.assert_array_equal(y, z1)


@pytest.mark.parametrize("name", NAMES)
def test_float32(name):
    build, theta, _, gamma = CASES[name]
    op = build(theta)
    x = np.random.default_rng(7).normal(size=(20, N))
    z64 = op.prox(x.astype(np.float32).astype(np.float64), gamma)
    z32 = op.prox(x.astype(np.float32), gamma)
    assert z32.dtype == np.float32
    np.testing.assert_allclose(z32, z64, rtol=1e-5, atol=1e-5)


def test_dtype_rules():
    x = np.arange(5)
    with pytest.raises(TypeError):
        fp.tv1d(x)
    np.testing.assert_array_equal(fp.tv1d(x, 0.0, copy=True), x)
    assert fp.tv1d([1.0, 2.0, 3.0], 0.1).dtype == np.float64


def test_errors():
    x = np.ones(5)
    with pytest.raises(ValueError):
        fp.tv1d(x, -1.0)
    with pytest.raises(ValueError):
        fp.tv1d(x, weights=np.ones(3))
    with pytest.raises(ValueError):
        fp.project_simplex(x, -1.0)
    with pytest.raises(ValueError):
        fp.tv1d(np.array([1.0, np.inf, 2.0]))
    with pytest.raises(ValueError):
        fp.TV1D(np.array([1.0, 2.0]), method="condat")
    z = fp.tv1d(np.array([1.0, np.nan, 2.0]))
    assert np.isnan(z).all()
    z = fp.soft_threshold(np.array([np.nan, 3.0]))
    assert np.isnan(z[0]) and z[1] == 2.0
