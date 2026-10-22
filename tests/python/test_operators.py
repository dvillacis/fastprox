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
AV = _rng.uniform(0.5, 1.5, N) * np.where(np.arange(N) % 3 == 1, -1.0, 1.0)
AV[4] = 0.0  # an entry outside the constraint
UNBOUNDED = np.array([-np.inf, np.inf])
# A tree over the axis: leaves, a subtree, a singleton, and the root.
TREE = [[0, 1], [2, 3, 4], [0, 1, 2, 3, 4], [5], [6, 7, 8, 9], list(range(N))]


# name: (build(theta) -> Operator, theta0, reference, gamma[, convex]).
# reference: reference(x_row, gamma, theta) -> z, None (1D TV: optimality
# residual), or for the nonconvex maps ("objective", psi(theta)) or, with
# per-entry parameters, ("objective_entry", psi(theta, j)) (the minimum on a
# refined grid, see reference.grid_prox). convex defaults to True; nonconvex
# maps are not firmly nonexpansive.
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
    "max": (lambda t: fp.Max(t[0]), np.array([0.7]), lambda x, g, t: ref.prox_max(x, g * t[0]), 0.9),
    "sum_largest": (lambda t: fp.SumLargest(t[0], 3), np.array([0.6]),
                    lambda x, g, t: ref.prox_sum_largest(x, g * t[0], 3), 0.9),
    # gamma lam > 1: Beck's form c P_C(x / c) rather than the scaled set.
    "sum_largest_abs": (lambda t: fp.SumLargest(t[0], 3, absolute=True), np.array([2.5]),
                        lambda x, g, t: ref.prox_sum_largest(x, g * t[0], 3, True), 0.9),
    # theta = (b, a..., lower..., upper...).
    "hyperplane": (lambda t: fp.Hyperplane(t[1:N + 1], t[0], t[N + 1], t[N + 2]),
                   np.concatenate([[0.4], AV, UNBOUNDED]),
                   lambda x, g, t: ref.hyperplane_box(x, t[1:N + 1], t[0]), 1.0),
    "hyperplane_box": (lambda t: fp.Hyperplane(t[1:N + 1], t[0], t[N + 1:2 * N + 1], t[2 * N + 1:]),
                       np.concatenate([[0.4], AV, LO, HI]),
                       lambda x, g, t: ref.hyperplane_box(x, t[1:N + 1], t[0], t[N + 1:2 * N + 1],
                                                          t[2 * N + 1:]), 1.0),
    "capped_simplex": (lambda t: fp.Hyperplane(t[1], t[0], t[2], t[3]), np.array([3.0, 1.0, 0.0, 1.0]),
                       lambda x, g, t: ref.hyperplane_box(x, t[1], t[0], t[2], t[3]), 1.0),
    "halfspace": (lambda t: fp.HalfSpace(t[1:N + 1], t[0], t[N + 1], t[N + 2]),
                  np.concatenate([[0.2], AV, UNBOUNDED]),
                  lambda x, g, t: ref.hyperplane_box(x, t[1:N + 1], t[0], inequality=True), 1.0),
    "halfspace_box": (lambda t: fp.HalfSpace(t[1:N + 1], t[0], t[N + 1:2 * N + 1], t[2 * N + 1:]),
                      np.concatenate([[0.2], AV, LO, HI]),
                      lambda x, g, t: ref.hyperplane_box(x, t[1:N + 1], t[0], t[N + 1:2 * N + 1],
                                                         t[2 * N + 1:], inequality=True), 1.0),
    "log_barrier": (lambda t: fp.LogBarrier(t[0]), np.array([0.6]),
                    lambda x, g, t: ref.log_barrier(x, g * t[0]), 0.8),
    # theta = (weights..., lower..., upper...).
    "l1_box": (lambda t: fp.L1Box(0.4, t[N:2 * N], t[2 * N:], t[:N]), np.concatenate([W, LO, HI]),
               lambda x, g, t: ref.l1_box(x, g * 0.4 * t[:N], t[N:2 * N], t[2 * N:]), 0.8),
    # An interval away from 0: the penalty is linear on it.
    "l1_box_shifted": (lambda t: fp.L1Box(0.4, t[1], t[2], t[:1]), np.array([1.2, 0.2, 0.9]),
                       lambda x, g, t: ref.l1_box(x, g * 0.4 * t[0], t[1], t[2]), 0.8),
    "soc": (lambda t: fp.SOC(), np.zeros(0), lambda x, g, t: ref.soc(x), 1.0),
    # theta = (lam, delta | a | b, group_weights...).
    "group_huber": (lambda t: fp.GroupHuber(t[0], t[1], GROUPS, t[2:]),
                    np.array([0.6, 2.0, 0.8, 1.0, 1.2]),
                    lambda x, g, t: ref.group_radial(x, g, fp.Huber(t[0], t[1]), GROUPS, t[2:]), 0.9),
    "huber_l2": (lambda t: fp.GroupHuber(t[0], t[1], None, t[2:]), np.array([0.6, 2.0, 1.1]),
                 lambda x, g, t: ref.group_radial(x, g, fp.Huber(t[0], t[1]), None, t[2:]), 0.9),
    "group_scad": (lambda t: fp.GroupSCAD(t[0], t[1], GROUPS, t[2:]),
                   np.array([1.0, 3.7, 0.8, 1.0, 1.2]),
                   lambda x, g, t: ref.group_radial(x, g, fp.SCAD(t[0], t[1]), GROUPS, t[2:]), 0.8,
                   False),
    "group_mcp": (lambda t: fp.GroupMCP(t[0], t[1], GROUPS, t[2:]),
                  np.array([1.0, 3.0, 0.8, 1.0, 1.2]),
                  lambda x, g, t: ref.group_radial(x, g, fp.MCP(t[0], t[1]), GROUPS, t[2:]), 0.8,
                  False),
    # theta = (l1, eta...).
    "tree_l2": (lambda t: fp.TreeL2(TREE, t[1:], t[0]), np.array([0.2, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4]),
                lambda x, g, t: ref.tree_l2(x, g, t[0], TREE, t[1:]), 0.9),
    "sparse_group": (lambda t: fp.TreeL2.sparse_group(t[0], 1.0, GROUPS, t[1:]),
                     np.array([0.3, 0.8, 1.0, 1.2]),
                     lambda x, g, t: ref.group_l2(ref.soft(x, g * t[0]), g, 1.0, GROUPS, t[1:]), 0.9),
    # theta = (alpha, weights...).
    "tv1d_ball": (lambda t: fp.TV1DBall(t[0], t[1:]), np.array([2.0, 1.0]),
                  lambda x, g, t: ref.tv1d_ball(x, t[0], t[1]), 1.0),
    "tv1d_ball_w": (lambda t: fp.TV1DBall(t[0], t[1:]), np.concatenate([[1.5], WE]),
                    lambda x, g, t: ref.tv1d_ball(x, t[0], t[1:]), 1.0),
    # theta = weights.
    "tv1d_l2": (lambda t: fp.TV1DL2(t), np.array([1.0]),
                lambda x, g, t: ref.tv1d_l2(x, g, t[0]), 0.6),
    "tv1d_l2_w": (lambda t: fp.TV1DL2(t), WE, lambda x, g, t: ref.tv1d_l2(x, g, t), 0.6),
    "l0": (lambda t: fp.L0(0.3, t), W,
           ("objective", lambda t: lambda z: 0.3 * t[0] * (z != 0)), 0.8, False),
    "lq_half": (lambda t: fp.Lq(0.4, 0.5, t), W,
                ("objective", lambda t: lambda z: 0.4 * t[0] * np.abs(z) ** 0.5), 0.8, False),
    "lq_two_thirds": (lambda t: fp.Lq(0.4, 2 / 3, t), np.array([1.0]),
                      ("objective", lambda t: lambda z: 0.4 * t[0] * np.abs(z) ** (2 / 3)), 0.8,
                      False),
    "lq_0.3": (lambda t: fp.Lq(0.4, 0.3, t), np.array([1.0]),
               ("objective", lambda t: lambda z: 0.4 * t[0] * np.abs(z) ** 0.3), 0.8, False),
    "scad": (lambda t: fp.SCAD(*t), np.array([0.5, 3.7]),
             ("objective", lambda t: ref.scad_psi(*t)), 0.8, False),
    "mcp": (lambda t: fp.MCP(*t), np.array([0.5, 3.0]),
            ("objective", lambda t: ref.mcp_psi(*t)), 0.8, False),
    # theta = (lam, a or b, w...): per-entry weights, a > 1 + gamma max w.
    "scad_w": (lambda t: fp.SCAD(t[0], t[1], t[2:]), np.r_[0.5, 3.7, W],
               ("objective_entry", lambda t, j: lambda z: t[2 + j] * ref.scad_psi(t[0], t[1])(z)),
               0.8, False),
    "mcp_w": (lambda t: fp.MCP(t[0], t[1], t[2:]), np.r_[0.5, 3.0, W],
              ("objective_entry", lambda t, j: lambda z: t[2 + j] * ref.mcp_psi(t[0], t[1])(z)),
              0.8, False),
    "log": (lambda t: fp.LogPenalty(*t), np.array([0.8, 0.5]),
            ("objective", lambda t: lambda z: t[0] * np.log1p(np.abs(z) / t[1])), 0.8, False),
}
NAMES = sorted(CASES)


def apply_rows(f, x):
    return np.stack([f(r) for r in x.reshape(-1, x.shape[-1])]).reshape(x.shape)


# ---------------------------------------------------------------- forward


@pytest.mark.parametrize("name", NAMES)
def test_matches_reference(name):
    build, theta, reference, gamma = CASES[name][:4]
    op = build(theta)
    rng = np.random.default_rng(1)
    for scale in (0.1, 1.0, 10.0):
        x = rng.normal(scale=scale, size=(7, N))
        z = op.prox(x, gamma)
        if isinstance(reference, tuple):  # nonconvex: a global minimiser, entry by entry
            # theta is the per-entry weight vector for l0 and lq_half, else the
            # map's parameters; psi(z) for one entry j.
            # ("objective_entry", f) gives psi for entry j as f(theta, j).
            per_entry = theta.size == N and name in ("l0", "lq_half")
            for row in range(2):  # the grid reference is slow: two rows per scale
                for j in range(N):
                    if reference[0] == "objective_entry":
                        psi = reference[1](theta, j)
                    else:
                        psi = reference[1](np.array([theta[j]]) if per_entry else theta)
                    xj, zj = x[row, j], z[row, j]
                    zg = ref.grid_prox(np.array([xj]), gamma, psi)[0]
                    fz = ref.nonconvex_objective(zj, xj, gamma, psi)
                    fg = ref.nonconvex_objective(zg, xj, gamma, psi)
                    assert fz <= fg + 1e-13 * (1 + xj ** 2)  # no better point on the grid
                    # The flat minimum locates the grid's argmin to ~sqrt(eps).
                    assert abs(zj - zg) <= 1e-7 * (1 + abs(xj))
        elif reference is None:
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


@pytest.mark.parametrize("gamma", [1e-3, 1.0, 1e3])
def test_tv1d_adversarial_inputs_stay_fast(gamma):
    # A linear ramp makes Condat's algorithm quadratic (about 3000 n steps at
    # gamma = 1e3); the automatic path must switch to the taut string.
    import time

    n = 10**5
    for x in (np.arange(n) / n, np.arange(n) / n * 10 + 0.1 * np.sin(np.arange(n))):
        t = time.perf_counter()
        z = fp.tv1d(x, gamma, threads=1)
        assert time.perf_counter() - t < 0.5  # forced Condat takes seconds
        # The spec's accuracy bound: 10 n eps, relative to |x|_inf.
        assert ref.tv1d_kkt_residual(x, z, gamma, 1.0) < 10 * n * np.finfo(float).eps


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


@pytest.mark.parametrize("method", ["auto", "condat", "taut_string"])
def test_tv1d_extreme_weights_within_spec_bound(method):
    # Radii far above n |x| never bind; they must not cost accuracy either
    # (spec: optimality residual <= 10 n eps |x|_inf).
    rng = np.random.default_rng(8)
    for t in range(3000):
        n = int(rng.integers(2, 200))
        x = [rng.normal(size=n), np.cumsum(rng.normal(size=n))][t % 2]
        gamma = float(np.exp(4 * rng.normal()))
        w = np.exp(3 * rng.normal(size=n - 1)) * (rng.random(n - 1) > 0.1)
        bound = 10 * n * np.finfo(float).eps
        z = fp.tv1d(x, gamma, weights=w, method=method)
        assert ref.tv1d_kkt_residual(x, z, gamma, w) < bound
        z = fp.tv1d(x, gamma, method=method)
        assert ref.tv1d_kkt_residual(x, z, gamma, 1.0) < bound


def test_l1_ball_tiny_weights():
    # Only weight ratios matter: uniformly tiny weights must behave like any
    # other scale (their squares would underflow without normalisation).
    rng = np.random.default_rng(4)
    x = rng.normal(size=50)
    w = rng.uniform(0.5, 1.5, 50)
    for s in (1e-200, 1.0, 1e200):
        z = fp.project_l1_ball(x, 2.0 * s, weights=w * s)
        np.testing.assert_allclose(z, ref.l1_ball(x, 2.0, w), rtol=1e-12, atol=1e-12)
    # Ratios whose squares underflow are rejected.
    with pytest.raises(ValueError):
        fp.L1Ball(1.0, np.r_[1.0, 1e-200])


def test_l1_ball_extreme_weight_ratios():
    # An entry within rounding of the pivot that dominates the running sums
    # used to cancel them when removed (s2 = 1 + 1e-126 - 1 = 0) and return
    # z = [5.5e7, 2.4e70] for this input (found by fuzzing).
    w = np.array([1.8127637189163722e-308, 7.7265702135305315e-246])
    for dt in (np.float64, np.float32):
        z = fp.project_l1_ball(np.array([11.0351562, 2.0390625], dt), 1e-300, weights=w)
        np.testing.assert_allclose(z, [11.0351562, 0.0], rtol=1e-6, atol=1e-30)
    # Random weights spanning 60 orders of magnitude and tiny radii, against
    # the projection in exact rational arithmetic (the float reference has
    # the same cancellation): a projection never grows an entry or flips a
    # sign, and it is accurate to rounding of x.
    from fractions import Fraction

    def exact_l1_ball(x, r, w):
        a = [Fraction(abs(float(v))) for v in x]
        wf = [Fraction(float(v)) for v in w]
        order = sorted(range(len(a)), key=lambda i: -a[i] / wf[i])
        s1 = s2 = Fraction(0)
        for k, i in enumerate(order):
            s1 += wf[i] * a[i]
            s2 += wf[i] * wf[i]
            tau = (s1 - Fraction(r)) / s2
            nxt = a[order[k + 1]] / wf[order[k + 1]] if k + 1 < len(a) else Fraction(0)
            if tau >= nxt:
                break
        return np.sign(x) * np.array([float(max(a[i] - tau * wf[i], 0)) for i in range(len(a))])

    rng = np.random.default_rng(11)
    for _ in range(2000):
        n = int(rng.integers(2, 12))
        x = rng.normal(size=n) * 10.0 ** rng.integers(-3, 4)
        w = 10.0 ** rng.uniform(-60, 0, n)
        r = float(np.sum(w * np.abs(x))) * 10.0 ** rng.uniform(-60, 0)
        z = fp.project_l1_ball(x, r, weights=w)
        assert np.all(np.abs(z) <= np.abs(x) * (1 + 1e-12))
        assert np.all(z * x >= 0)
        tol = 10 * n * np.finfo(float).eps * np.abs(x).max()
        np.testing.assert_allclose(z, exact_l1_ball(x, r, w), rtol=0, atol=tol)


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
    build, theta, _, gamma = CASES[name][:4]
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


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_jacobian_diag(name, conj):
    """ctx.jacobian_diag() is jvp(ctx, ones) for the elementwise operators,
    and then jvp(ctx, dx) = diag * dx for any dx; the others refuse it."""
    _, _, _, op, x, _, ctx, rng = _derivative_setup(name, conj, 0)
    if not op._elementwise:
        with pytest.raises(ValueError):
            ctx.jacobian_diag()
        return
    d = ctx.jacobian_diag()
    assert d.shape == x.shape and d.dtype == x.dtype
    np.testing.assert_array_equal(d, op.jvp(ctx, np.ones_like(x)))
    dx = rng.normal(size=x.shape)
    if conj:  # dx - sigma p'(t) dx / sigma: linear in dx up to rounding
        np.testing.assert_allclose(op.jvp(ctx, dx), d * dx, rtol=1e-14, atol=1e-15 * np.abs(dx).max())
    else:
        np.testing.assert_array_equal(op.jvp(ctx, dx), d * dx)


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
    build, theta, _, gamma = CASES[name][:4]
    if len(CASES[name]) > 4 and not CASES[name][4]:
        return  # nonconvex maps are not firmly nonexpansive
    n = N
    x1 = data.draw(hnp.arrays(np.float64, n, elements=finite))
    x2 = data.draw(hnp.arrays(np.float64, n, elements=finite))
    op = build(theta)
    z1, z2 = op.prox(x1, gamma), op.prox(x2, gamma)
    d = z1 - z2
    # Weighted isotonic regression projects in the norm sum_i w_i (.)^2, so
    # it is firmly nonexpansive in that norm, not the Euclidean one.
    m = theta if name == "isotonic_w" else np.ones(n)
    tol = 1e-10 * (1 + np.abs(x1).max() + np.abs(x2).max()) ** 2
    assert d @ (m * d) <= d @ (m * (x1 - x2)) + tol


@settings(max_examples=60, deadline=None)
@given(rows, st.floats(0.0, 10.0))
def test_projections_idempotent(x, r):
    for op in (fp.Simplex(r), fp.L1Ball(r), fp.L2Ball(r), fp.Isotonic(), fp.Box(-r, r)):
        z = op.prox(x)
        np.testing.assert_allclose(op.prox(z), z, rtol=1e-12, atol=1e-12 * (1 + r))


@settings(max_examples=60, deadline=None)
@given(rows, st.floats(1e-320, 1e300), st.floats(0.01, 5.0))
def test_moreau_conjugates(y, sigma, lam):
    # Positively homogeneous g: y - prox_g(y), exact for every sigma > 0 (since
    # 1.0; it overflowed below sigma ~ 1e-300).
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
    build, theta, _, gamma = CASES[name][:4]
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
    build, theta, _, gamma = CASES[name][:4]
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
    build, theta, _, gamma = CASES[name][:4]
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
    z = fp.tv1d(np.array([1.0, np.nan, 2.0]))
    assert np.isnan(z).all()
    z = fp.soft_threshold(np.array([np.nan, 3.0]))
    assert np.isnan(z[0]) and z[1] == 2.0
    # Row operators: NaN fills the row, an infinity is an error, in either
    # position of the row and on every code path (radius zero included).
    for f in (lambda v: fp.project_simplex(v, 1.0), lambda v: fp.project_simplex(v, 0.0),
              lambda v: fp.project_l1_ball(v, 1.0), lambda v: fp.prox_linf(v, 1.0),
              lambda v: fp.project_l1_ball(v, 1.0, weights=np.arange(1.0, 5.0))):
        for pos in (0, 3):
            v = np.array([5.0, -2.0, 0.5, 1.0])
            v[pos] = np.nan
            assert np.isnan(f(v)).all()
            v[pos] = np.inf
            with pytest.raises(ValueError):
                f(v)


def test_version_matches_metadata():
    from importlib.metadata import version
    assert fp.__version__ == version("fastprox")
