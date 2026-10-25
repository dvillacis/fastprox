"""Hyperplane and half-space projections (with a box) and the support-function
proxes max and sum of the k largest: optimality conditions over many sizes,
agreement with the operators they generalise, conjugates, extreme steps and
errors. Derivatives, threads and strides run in test_operators.py."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref


def plane_problem(rng, n, box=True, zeros=True):
    a = rng.normal(size=n)
    if zeros and n > 2:
        a[rng.integers(n)] = 0.0
    lo = -rng.uniform(0.0, 2.0, n) if box else np.full(n, -np.inf)
    hi = rng.uniform(0.0, 2.0, n) if box else np.full(n, np.inf)
    z0 = rng.uniform(lo, hi) if box else rng.normal(size=n)
    return a, float(a @ z0), lo, hi  # b = a^T z0 with z0 in the box: feasible


def check_kkt(x, z, a, b, lo, hi, inequality):
    """z = clip(x - mu a) with a^T z = b (or <= b and mu >= 0, complementary)."""
    scale = 1 + np.abs(x).max() + np.abs(a).max() * (1 + np.abs(x).max())
    tol = 1e-12 * scale * np.sqrt(x.size)
    assert np.all(z >= lo) and np.all(z <= hi)
    r = a @ z - b
    assert (r <= tol) if inequality else (abs(r) <= tol)
    free = (z > lo) & (z < hi) & (a != 0)
    if not free.any():
        return
    mus = (x - z)[free] / a[free]
    mu = np.median(mus)
    assert np.all(np.abs(mus - mu) <= tol * (1 + abs(mu)) / np.abs(a[free]).min())
    v = x - mu * a
    on_lo, on_hi = (z == lo) & ~free, (z == hi) & ~free
    assert np.all(v[on_lo] <= lo[on_lo] + tol) and np.all(v[on_hi] >= hi[on_hi] - tol)
    zero = (a == 0) & (z > lo) & (z < hi)
    assert np.array_equal(z[zero], x[zero])  # entries outside the constraint stay
    if inequality:
        assert mu >= -tol and (abs(r) <= tol or abs(mu) <= tol)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 50, 1000, 20000])
@pytest.mark.parametrize("inequality", [False, True])
@pytest.mark.parametrize("box", [False, True])
def test_kkt_and_reference(n, inequality, box):
    rng = np.random.default_rng(n + 10 * inequality + 100 * box)
    for trial in range(4):
        a, b, lo, hi = plane_problem(rng, n, box, zeros=trial % 2 == 0)
        if inequality:
            b -= rng.uniform(0, 1) * abs(b)  # moves b inside the range, still feasible
            if box:
                b = max(b, (np.minimum(a * lo, a * hi)).sum())
        if not np.any(a):
            continue
        x = rng.normal(scale=2.0, size=n)
        op = (fp.HalfSpace if inequality else fp.Hyperplane)(a, b, lo, hi)
        z = op.prox(x)
        check_kkt(x, z, a, b, lo, hi, inequality)
        if n <= 20000:
            zr = ref.hyperplane_box(x, a, b, lo, hi, inequality)
            np.testing.assert_allclose(z, zr, rtol=1e-11, atol=1e-11 * (1 + np.abs(x).max()))


def test_long_structured_rows():
    # Rows of 8192 entries or more start Newton from a sample's root; the
    # result must not depend on how good that start is.
    rng = np.random.default_rng(7)
    n = 50000
    walk = np.cumsum(rng.normal(size=n))
    rows = [rng.normal(size=n), np.sort(rng.normal(size=n)), walk, walk[::-1].copy(),
            np.repeat(rng.normal(size=n // 100), 100), rng.standard_cauchy(size=n)]
    for x in rows:
        for k in (1, 7, n // 3):
            z = fp.project_hyperplane(x, None, float(k), 0.0, 1.0)
            np.testing.assert_allclose(z, ref.hyperplane_box(x, 1.0, k, 0.0, 1.0), rtol=1e-10,
                                       atol=1e-10 * (1 + np.abs(x).max()))
    # A sample (every 16th entry) whose own problem is infeasible: no start.
    hi = np.ones(n)
    hi[::16] = 1e-3
    x = rng.normal(size=n)
    b = 0.9 * hi.sum()
    z = fp.project_hyperplane(x, None, b, 0.0, hi)
    np.testing.assert_allclose(z, ref.hyperplane_box(x, 1.0, b, 0.0, hi), rtol=1e-10, atol=1e-10)


def test_wide_range_of_a():
    # Nearly flat pieces of phi (only tiny a_i free) must not throw mu off.
    rng = np.random.default_rng(8)
    for trial in range(30):
        n = int(rng.integers(2, 40))
        a = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-90, 0, n)
        lo, hi = -rng.uniform(0, 2, n), rng.uniform(0, 2, n)
        b = float(a @ rng.uniform(lo, hi))
        x = rng.normal(scale=3.0, size=n)
        for inequality in (False, True):
            op = (fp.HalfSpace if inequality else fp.Hyperplane)(a, b, lo, hi)
            z = op.prox(x)
            assert np.all(z >= lo) and np.all(z <= hi)
            assert (a @ z - b <= 1e-12) if inequality else (abs(a @ z - b) <= 1e-12)
            zr = ref.hyperplane_box(x, a, b, lo, hi, inequality)
            fz, fr = np.sum((z - x) ** 2), np.sum((zr - x) ** 2)
            assert fz <= fr * (1 + 1e-12) + 1e-24  # no feasible point is closer


def test_flat_piece_near_the_root():
    # At mu = 0 phi - b is at rounding level while only the tiny entry is free:
    # the step to that piece's root (mu ~ 1e160) would leave the root's piece.
    a, lo, hi = np.array([1.0, 1e-90]), np.array([0.0, -1.0]), np.array([1.0, 1.0])
    z = fp.project_hyperplane(np.array([5.0, 0.0]), a, 1.0 - 1e-15, lo, hi)
    np.testing.assert_allclose(z, [1.0 - 1e-15, 0.0], rtol=0, atol=4e-16)


def test_closed_forms_and_special_cases():
    rng = np.random.default_rng(1)
    x = rng.normal(scale=3.0, size=(6, 12))
    a = rng.normal(size=12)
    # Lemma 6.26: no box.
    for b in (-2.0, 0.0, 3.5):
        expect = x - np.outer(x @ a - b, a) / (a @ a)
        np.testing.assert_allclose(fp.project_hyperplane(x, a, b), expect, rtol=1e-13, atol=1e-13)
        expect = x - np.outer(np.maximum(x @ a - b, 0), a) / (a @ a)
        np.testing.assert_allclose(fp.project_halfspace(x, a, b), expect, rtol=1e-13, atol=1e-13)
    # The simplex and the box projection.
    np.testing.assert_allclose(fp.project_hyperplane(x, None, 1.7, 0.0), fp.project_simplex(x, 1.7),
                               rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(fp.project_halfspace(x, a, 1e9, -1.0, 1.0), np.clip(x, -1, 1))
    # The support functions: k = 1 is max (and the l-inf norm for |x|),
    # k = n is the sum, and for |x| any k >= n is the l1 norm.
    for g in (0.1, 1.0, 7.0):
        np.testing.assert_allclose(fp.prox_sum_largest(x, 0.8, 1, g), fp.prox_max(x, 0.8, g),
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fp.prox_sum_largest(x, 0.8, 1, g, absolute=True),
                                   fp.prox_linf(x, 0.8, g), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fp.prox_sum_largest(x, 0.8, 12, g), x - 0.8 * g,
                                   rtol=1e-13, atol=1e-13)
        for k in (12, 40):
            np.testing.assert_allclose(fp.prox_sum_largest(x, 0.8, k, g, absolute=True),
                                       fp.soft_threshold(x, 0.8, g), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("which", ["max", "sum3", "sum3_abs"])
def test_support_minimises_the_prox_objective(which):
    rng = np.random.default_rng(2)
    k = 1 if which == "max" else 3
    absolute = which.endswith("abs")
    for gamma in (0.05, 1.0, 20.0):
        for x in (rng.normal(size=9), np.array([2.0, 2.0, 2.0, 1.0, -3.0, -3.0, 0.5, 2.0, 0.0])):
            z = fp.prox_sum_largest(x, 0.7, k, gamma, absolute=absolute)
            np.testing.assert_allclose(z, ref.prox_sum_largest(x, 0.7 * gamma, k, absolute),
                                       rtol=1e-12, atol=1e-12)

            def obj(v):
                return 0.5 * np.sum((v - x) ** 2) + 0.7 * gamma * ref.sum_largest(v, k, absolute)

            f = obj(z)
            for _ in range(300):
                d = rng.normal(size=x.size) * 10.0 ** rng.uniform(-6, 0)
                assert f <= obj(z + d) + 1e-12 * (1 + f)


def test_support_ties_derivatives():
    # Tied entries at the k-th largest value: the derivatives stay adjoint.
    x = np.array([[3.0, 1.0, 3.0, 3.0, -1.0, 0.5, 3.0, 2.0]])
    for op in (fp.Max(0.5), fp.SumLargest(0.5, 2), fp.SumLargest(0.5, 3, absolute=True)):
        z, ctx = op.prox(x, 1.0, keep=True)
        rng = np.random.default_rng(3)
        dx, g = rng.normal(size=x.shape), rng.normal(size=x.shape)
        dz = op.jvp(ctx, dx, 0.3, np.array([0.7]))
        gx, gr = op.vjp(ctx, g)
        assert np.vdot(g, dz) == pytest.approx(np.vdot(gx, dx) + 0.3 * gr.gamma + 0.7 * gr.lam,
                                               rel=1e-12, abs=1e-12)


def test_conjugates():
    rng = np.random.default_rng(4)
    y = rng.normal(scale=2.0, size=(5, 9))
    for lam, sigma in ((0.6, 0.5), (2.0, 3.0)):
        # (lam max)* is the indicator of the simplex of radius lam.
        np.testing.assert_allclose(fp.Max(lam).prox_conj(y, sigma), fp.project_simplex(y, lam),
                                   rtol=1e-12, atol=1e-12)
        # (lam sum_k)* is the indicator of {sum = lam k, 0 <= v <= lam}.
        np.testing.assert_allclose(fp.SumLargest(lam, 4).prox_conj(y, sigma),
                                   fp.project_hyperplane(y, None, 4 * lam, 0.0, lam),
                                   rtol=1e-12, atol=1e-12)


def test_extreme_steps_and_scales():
    x = np.array([1e300, -1e300, 3.0, 2.0, 1e-300])
    for g in (1e-310, 1e-300, 1e280):  # no x / (gamma lam) is formed
        for f in (lambda v: fp.prox_max(v, 1.0, g), lambda v: fp.prox_sum_largest(v, 1.0, 2, g),
                  lambda v: fp.prox_sum_largest(v, 1.0, 2, g, absolute=True)):
            z = f(x)
            assert np.all(np.isfinite(z))
    # Beyond the step range, gamma lam > 2^960 (since 1.0): their threshold sums overflowed.
    with pytest.raises(ValueError):
        fp.prox_max(x, 1.0, 1e300)
    np.testing.assert_array_equal(fp.prox_max(x, 1.0, 1e-310), x)  # x_0 - 1e-310 rounds to x_0
    np.testing.assert_array_equal(fp.prox_max(np.array([1e-300, 0.0]), 1.0, 5e-301), [5e-301, 0.0])
    # The set does not change when (a, b) is scaled, nor does the projection.
    x = np.random.default_rng(6).normal(size=(4, 7))
    a = np.array([1.0, -2.0, 0.5, 0.0, 3.0, 1e-30, 1.0])
    z = fp.project_hyperplane(x, a, 0.7, -1.0, 1.0)
    for s in (1e-200, 1e200):
        np.testing.assert_allclose(fp.project_hyperplane(x, s * a, s * 0.7, -1.0, 1.0), z,
                                   rtol=1e-14, atol=1e-14)
    # A ratio below 1e-100 between nonzero entries would underflow.
    with pytest.raises(ValueError):
        fp.Hyperplane(np.array([1.0, 1e-101]), 0.0)


def test_projections_idempotent():
    rng = np.random.default_rng(5)
    for _ in range(20):
        a, b, lo, hi = plane_problem(rng, 15)
        for op in (fp.Hyperplane(a, b, lo, hi), fp.HalfSpace(a, b, lo, hi)):
            z = op.prox(rng.normal(scale=3, size=15))
            np.testing.assert_allclose(op.prox(z), z, rtol=1e-12, atol=1e-12)


def test_errors():
    x = np.ones(3)
    for bad in (lambda: fp.Hyperplane(np.ones(3), 3.5, 0.0, 1.0).prox(x),  # b > n hi
                lambda: fp.HalfSpace(np.ones(3), -0.1, 0.0, 1.0).prox(x),  # below the box
                lambda: fp.SumLargest(1.0, 4).prox(x),                     # k > n
                lambda: fp.Hyperplane(np.ones(4), 1.0).prox(x),            # a of another length
                lambda: fp.Hyperplane(np.zeros(3), 0.0),                   # a = 0
                lambda: fp.Hyperplane(0.0, 0.0),
                lambda: fp.Hyperplane(1.0, 0.0, 1.0, 0.0),                 # lo > hi
                lambda: fp.Hyperplane(np.array([1.0, np.inf]), 0.0),
                lambda: fp.Hyperplane(1.0, np.nan),
                lambda: fp.Hyperplane(1.0, 0.0, np.inf),
                lambda: fp.SumLargest(1.0, 0), lambda: fp.Max(-1.0)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError):
        fp.SumLargest(1.0, 2.5)
    # Feasibility depends on the row length; the boundary b = n hi is feasible.
    np.testing.assert_array_equal(fp.project_hyperplane(x, np.ones(3), 3.0, 0.0, 1.0), np.ones(3))
    np.testing.assert_array_equal(fp.project_hyperplane(np.zeros(5), None, 3.5, 0.0, 1.0)[:1], [0.7])
    fp.SumLargest(1.0, 4, absolute=True).prox(x)
    for f in (lambda v: fp.prox_max(v, 1.0), lambda v: fp.prox_max(v, 0.0),
              lambda v: fp.prox_sum_largest(v, 1.0, 2), lambda v: fp.prox_sum_largest(v, 0.0, 2),
              lambda v: fp.prox_sum_largest(v, 1.0, 2, absolute=True),
              lambda v: fp.project_hyperplane(v, None, 1.0, 0.0, 1.0),
              lambda v: fp.project_halfspace(v, np.arange(4.0), 1e9)):
        for pos in (0, 3):
            v = np.array([5.0, -2.0, 0.5, 1.0])
            v[pos] = np.nan
            assert np.isnan(f(v)).all()
            v[pos] = np.inf
            with pytest.raises(ValueError):
                f(v)


@pytest.mark.parametrize("box", [True, False])
@pytest.mark.parametrize("k", [-1000, 980, 1000])
def test_scaled_planes(box, k):
    """P_{tC}(t x) = t P_C(x): the sums a^T x overflowed for entries near
    DBL_MAX / n (audit K7); the solver now rescales by a power of two."""
    rng = np.random.default_rng(11)
    t = 2.0 ** k
    for _ in range(5):
        a, b, lo, hi = plane_problem(rng, 9, box=box)
        x = rng.normal(scale=3, size=9)
        for op, opt in ((fp.Hyperplane(a, b, lo, hi), fp.Hyperplane(a, b * t, lo * t, hi * t)),
                        (fp.HalfSpace(a, b, lo, hi), fp.HalfSpace(a, b * t, lo * t, hi * t))):
            z = op.prox(x)
            np.testing.assert_allclose(opt.prox(x * t) / t, z, rtol=8 * 2.0 ** -52, atol=8 * 2.0 ** -52 * np.abs(z).max())
    x = np.array([1.4e308, 1.4e308])
    np.testing.assert_array_equal(fp.project_hyperplane(x, np.ones(2), 0.0), [0.0, 0.0])
    np.testing.assert_array_equal(fp.project_hyperplane(x, np.ones(2), 0.0, -1.5e308, 1.5e308), [0.0, 0.0])


def test_sum_largest_near_max():
    y = np.array([1e308, 5e307, -3.0])
    np.testing.assert_array_equal(fp.prox_sum_largest(y, 1.0, 1), [1e308 - 1.0, 5e307, -3.0])


def test_degenerate_bounds():
    """An entry with lo = hi is fixed: its Jacobian is 0, also on the bound
    (audit K11); the free piece does not exist there."""
    x = np.array([0.25, 0.5, 0.25])
    lo, hi = np.array([0.0, 0.5, 0.0]), np.array([1.0, 0.5, 1.0])
    op = fp.Hyperplane(np.ones(3), 1.0, lo, hi)
    z, ctx = op.prox(x, keep=True)
    np.testing.assert_array_equal(op.jvp(ctx, np.array([0.0, 1.0, 0.0])), [0.0, 0.0, 0.0])
    h = 1e-7
    for side in (1, -1):  # one-sided differences in x_1
        e = np.array([0.0, side * h, 0.0])
        np.testing.assert_allclose((op.prox(x + e) - z) / h, 0.0, atol=1e-9)
    g = np.random.default_rng(3).normal(size=3)
    dx = np.random.default_rng(4).normal(size=3)
    assert abs(g @ op.jvp(ctx, dx) - op.vjp(ctx, g)[0] @ dx) <= 1e-14 * (1 + abs(g @ op.jvp(ctx, dx)))


def test_coefficients_of_disparate_size():
    """Breakpoints (x_i - bound) / a_i with a_i far below max |a_i| (fuzz,
    2026-10-25): a_i = 6.6e-76 of a normalised a and |x| = 2e239 put them at
    3e314, and the projection came out NaN and inf. Since 1.0.1 the scale
    also counts min |a_i|; it is unchanged when the a_i are of one size.
    P_{tC}(t x) = t P_C(x) for t = 2^k checks it."""
    a = np.array([2.09375, 1.640625, 0, -0.0, -1.578125, 3.078125, -0.53125, 2.890625, 3.1732573283419783e+75])
    y = np.array([-28, 0, -76.4179688, 0, 0, -0.000194345913, 82.8125, -8.578125, 89.21875])
    for k in (790, 600, 0, -600):
        t = 2.0**k
        z = fp.Hyperplane(a, 1.0, -np.inf, 3.0).prox(y * t)
        assert np.all(np.isfinite(z)) and np.all(z <= 3.0)
        zt = fp.Hyperplane(a, 2.0**-k, -np.inf, 3.0 * 2.0**-k).prox(y)
        np.testing.assert_allclose(z, zt * t, rtol=1e-12, atol=0)
