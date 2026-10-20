"""Operator.value (fp_value, since 0.5): g(x) per row or image.

The main check ties each value to its prox: z = prox_{gamma g}(x) minimises
    Phi(y) = g(y) + ||y - x||^2 / (2 gamma),
so Phi(z) <= Phi(y) for every y; for the iterative image operators up to
the duality gap of z. A wrong formula, weight or index breaks it for some y.
"""

import math

import numpy as np
import pytest

import fastprox as fp
from test_operators import CASES, N

# The sets: their prox is a projection, and g is the set's indicator.
SETS = {"box", "nonneg", "l2_ball", "simplex", "l1_ball", "l1_ball_w", "isotonic", "hyperplane",
        "hyperplane_box", "capped_simplex", "halfspace", "halfspace_box", "soc", "tv1d_ball",
        "tv1d_ball_w"}
# Weighted isotonic regression minimises sum_i w_i (z_i - x_i)^2: not the
# Euclidean prox of an indicator, so the inequality below does not apply.
SKIP = {"isotonic_w"}


def _phi(op, y, x, gamma, tol=1e-6):
    return op.value(y, tol=tol) + ((y - x) ** 2).sum(axis=-1) / (2 * gamma)


@pytest.mark.parametrize("name", sorted(set(CASES) - SKIP))
def test_prox_minimises_value(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    rng = np.random.default_rng(3)
    for scale in (0.1, 1.0, 10.0):
        x = rng.normal(scale=scale, size=(6, N))
        z = op.prox(x, gamma)
        best = _phi(op, z, x, gamma)
        assert np.all(np.isfinite(best)), name
        if name in SETS:
            np.testing.assert_array_equal(op.value(z), 0.0)
        tol = 1e-9 * (1 + np.abs(best))
        trials = [x, rng.normal(scale=scale, size=x.shape)]
        for h in (1e-1, 1e-3, 1e-6):
            trials += [z + h * scale * rng.normal(size=x.shape) for _ in range(4)]
        for y in trials:  # judged exactly: a point just off a set is off it
            assert np.all(best <= _phi(op, y, x, gamma, tol=0.0) + tol), name


# Built inside the test: operators made at collection time outlive the
# interpreter's teardown and trip nanobind's leak check.
IMAGE_OPS = {"iso2": lambda: fp.TVIso(2, [0.7, 1.3]), "aniso2": lambda: fp.TVAniso(2, [0.7, 1.3]),
             "iso3": lambda: fp.TVIso(3), "aniso3": lambda: fp.TVAniso(3, [1.0, 0.5, 2.0])}


@pytest.mark.parametrize("name", sorted(IMAGE_OPS))
def test_image_prox_minimises_value(name):
    op = IMAGE_OPS[name]()
    rng = np.random.default_rng(4)
    shape = (2,) + (9, 8, 7)[:op.ndim]
    x = rng.normal(size=shape)
    gamma = 0.4
    z = op.prox(x, gamma, tol=1e-12, max_iter=100000)
    axes = tuple(range(1, x.ndim))
    phi = lambda y: op.value(y) + ((y - x) ** 2).sum(axis=axes) / (2 * gamma)  # noqa: E731
    best = phi(z)
    for y in [x, z + 1e-3 * rng.normal(size=x.shape), z + 1e-6 * rng.normal(size=x.shape)]:
        assert np.all(best <= phi(y) + 1e-9 * (1 + np.abs(best)))


def _tv_aniso(x, w, beta=None):
    s = np.zeros(x.shape[:-2] if x.ndim > 2 else ())
    for a in range(2):
        d = np.abs(np.diff(x, axis=x.ndim - 2 + a))
        if beta is not None:
            b = beta[a]
            b = b[:-1, :] if a == 0 else b[:, :-1]
            d = d * b
        s = s + w[a] * d.sum(axis=(-2, -1))
    return s


def _tv_iso(x, w, alpha=None):
    gy = np.zeros_like(x)
    gx = np.zeros_like(x)
    gy[..., :-1, :] = w[0] * np.diff(x, axis=-2)
    gx[..., :, :-1] = w[1] * np.diff(x, axis=-1)
    m = np.sqrt(gy ** 2 + gx ** 2)
    if alpha is not None:
        m = m * alpha
    return m.sum(axis=(-2, -1))


def test_formulas():
    """Direct NumPy formulas for a few operators of each kind."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, N))
    w = rng.uniform(0.5, 1.5, N)
    np.testing.assert_allclose(fp.Soft(0.3, w).value(x), 0.3 * (w * np.abs(x)).sum(-1), rtol=1e-14)
    we = rng.uniform(0.5, 1.5, N - 1)
    np.testing.assert_allclose(fp.TV1D(we).value(x), (we * np.abs(np.diff(x))).sum(-1), rtol=1e-14)
    np.testing.assert_allclose(fp.TV1DL2(we).value(x), np.sqrt((we ** 2 * np.diff(x) ** 2).sum(-1)),
                               rtol=1e-14)
    np.testing.assert_allclose(fp.SumLargest(0.5, 3).value(x), 0.5 * np.sort(x)[:, -3:].sum(-1), rtol=1e-14)
    np.testing.assert_allclose(fp.SumLargest(0.5, 3, absolute=True).value(x),
                               0.5 * np.sort(np.abs(x))[:, -3:].sum(-1), rtol=1e-14)
    np.testing.assert_allclose(fp.LInf(0.7).value(x), 0.7 * np.abs(x).max(-1), rtol=1e-14)
    np.testing.assert_allclose(fp.Max(0.7).value(x), 0.7 * x.max(-1), rtol=1e-14)
    groups = np.arange(N) % 3
    c = np.array([0.5, 1.0, 2.0])
    norms = np.stack([np.linalg.norm(x[:, groups == g], axis=-1) for g in range(3)], -1)
    np.testing.assert_allclose(fp.GroupL2(0.4, groups, c).value(x), 0.4 * (norms * c).sum(-1), rtol=1e-14)
    lam, a, b = 0.6, 3.7, 2.5
    t = np.abs(x)
    scad = np.where(t <= lam, lam * t, np.where(t <= a * lam, (2 * a * lam * t - t ** 2 - lam ** 2) / (2 * (a - 1)),
                                                (a + 1) * lam ** 2 / 2))
    np.testing.assert_allclose(fp.SCAD(lam, a).value(x), scad.sum(-1), rtol=1e-14)
    mcp = np.where(t <= b * lam, lam * t - t ** 2 / (2 * b), b * lam ** 2 / 2)
    np.testing.assert_allclose(fp.MCP(lam, b, w).value(x), (w * mcp).sum(-1), rtol=1e-14)
    np.testing.assert_allclose(fp.LogPenalty(0.5, 0.2).value(x), 0.5 * np.log1p(t / 0.2).sum(-1), rtol=1e-14)
    np.testing.assert_allclose(fp.L0(0.5, w).value(np.where(t < 0.5, 0.0, x)),
                               0.5 * (w * (t >= 0.5)).sum(-1), rtol=1e-14)
    # Images, with weights.
    img = rng.normal(size=(3, 7, 9))
    beta = rng.uniform(0.0, 2.0, (2, 7, 9))
    alpha = rng.uniform(0.0, 2.0, (7, 9))
    np.testing.assert_allclose(fp.TVAniso(2, [0.7, 1.3]).value(img), _tv_aniso(img, [0.7, 1.3]), rtol=1e-13)
    np.testing.assert_allclose(fp.TVAniso(2, [0.7, 1.3], edge_weights=beta).value(img),
                               _tv_aniso(img, [0.7, 1.3], beta), rtol=1e-13)
    np.testing.assert_allclose(fp.TVIso(2, [0.7, 1.3]).value(img), _tv_iso(img, [0.7, 1.3]), rtol=1e-13)
    np.testing.assert_allclose(fp.TVIso(2, [0.7, 1.3], pixel_weights=alpha).value(img),
                               _tv_iso(img, [0.7, 1.3], alpha), rtol=1e-13)
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4]])
    ew = np.array([0.5, 1.0, 2.0, 0.3])
    g = rng.normal(size=(2, 5))
    ref = (ew * np.abs(g[:, edges[:, 0]] - g[:, edges[:, 1]])).sum(-1)
    np.testing.assert_allclose(fp.GraphTV(5, edges, ew).value(g), ref, rtol=1e-14)
    np.testing.assert_allclose(fp.TreeTV(5, edges[[0, 1, 3]], ew[[0, 1, 3]]).value(g),
                               (ew[[0, 1, 3]] * np.abs(g[:, edges[[0, 1, 3], 0]] - g[:, edges[[0, 1, 3], 1]])).sum(-1),
                               rtol=1e-14)


def test_shapes_axis_and_special_values():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(3, 4, 5))
    op = fp.TV1D()
    v = op.value(x, axis=1)
    assert v.shape == (3, 5)
    np.testing.assert_allclose(v, np.abs(np.diff(x, axis=1)).sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(op.value(np.asfortranarray(x), axis=1), v, rtol=1e-14)
    np.testing.assert_allclose(op.value(x[:, ::-1, ::2], axis=1), op.value(np.ascontiguousarray(x[:, ::-1, ::2]), axis=1))
    assert isinstance(op.value(x[0, 0]), float)              # one row: a float
    assert isinstance(fp.Soft(1.0).value(2.0), float)        # a scalar is a row of one
    assert fp.Soft(1.0).value(-2.0) == 2.0
    np.testing.assert_allclose(op.value(x.astype(np.float32)), op.value(x), rtol=1e-6)
    assert fp.TVIso().value(rng.normal(size=(6, 7))) > 0     # one image: a float
    assert fp.TVIso().value(rng.normal(size=(2, 3, 6, 7))).shape == (2, 3)
    y = x.copy()
    y[1, 2, 3] = np.nan
    v = op.value(y)
    assert np.isnan(v[1, 2]) and np.isfinite(np.delete(v.ravel(), 1 * 4 + 2)).all()
    assert op.value(np.array([[1, 2, 4]])) == 3.0            # integers are converted
    with pytest.raises(ValueError):
        op.value(x, axis=3)
    with pytest.raises(ValueError):
        fp.TVIso(3).value(rng.normal(size=(4, 5)))


def test_set_tolerance():
    """A point off the set by less than tol * max(1, max |x|) counts as in it."""
    op = fp.Simplex(1.0)
    z = op.prox(np.array([0.3, 0.9, -0.2]))
    assert op.value(z) == 0.0
    assert op.value(z + 1e-9) == 0.0                 # sum off by 3e-9, within 1e-6
    assert math.isinf(op.value(z + 1e-3))
    assert math.isinf(op.value(z + 1e-9, tol=1e-12))
    assert op.value(z + 1e-3, tol=1e-2) == 0.0
    assert fp.Box(-1, 1).value(np.array([1 + 5e-7, 0.0])) == 0.0
    assert math.isinf(fp.Box(-1, 1).value(np.array([1.1, 0.0])))
    assert math.isinf(fp.LogBarrier(1.0).value(np.array([1.0, 0.0])))   # outside the domain
    with pytest.raises(Exception):
        op.value(z, tol=-1.0)
