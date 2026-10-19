"""fastprox.pyproximal: fastprox operators inside PyProximal (since 1.0)."""

import numpy as np
import pytest

pyproximal = pytest.importorskip("pyproximal")
pylops = pytest.importorskip("pylops")

import fastprox as fp  # noqa: E402
from fastprox.pyproximal import ProxOp  # noqa: E402
from pyproximal.utils import moreau  # noqa: E402


def test_matches_pyproximal_l1():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    ours, theirs = ProxOp(fp.Soft(0.3)), pyproximal.L1(sigma=0.3)
    for tau in (0.1, 1.0, 3.0):
        np.testing.assert_allclose(ours.prox(x, tau), theirs.prox(x, tau), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(ours.proxdual(x, tau), theirs.proxdual(x, tau), rtol=1e-12, atol=1e-12)
    assert ours(x) == pytest.approx(float(theirs(x)), rel=1e-14)


@pytest.mark.parametrize("name", ["soft", "tv1d", "simplex", "l1_ball", "group_l2", "huber", "tv_iso", "graph_tv"])
def test_moreau_identity(name):
    """PyProximal's own check: prox(x, tau) + tau proxdual(x / tau, 1 / tau) = x."""
    rng = np.random.default_rng(1)
    n = 24
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
    op = {"soft": lambda: ProxOp(fp.Soft(0.4)),
          "tv1d": lambda: ProxOp(fp.TV1D(rng.uniform(0.5, 1.5, n - 1))),
          "simplex": lambda: ProxOp(fp.Simplex(2.0)),
          "l1_ball": lambda: ProxOp(fp.L1Ball(1.5)),
          "group_l2": lambda: ProxOp(fp.GroupL2(0.5, np.arange(n) % 4)),
          "huber": lambda: ProxOp(fp.Huber(0.7, 0.3)),
          "tv_iso": lambda: ProxOp(fp.TVIso(), dims=(4, 6), tol=1e-12, max_iter=100000),
          "graph_tv": lambda: ProxOp(fp.GraphTV(n, edges))}[name]()
    x = rng.normal(size=n)
    for tau in (0.3, 1.0, 2.5):
        assert moreau(op, x, tau, tol=1e-8)


def test_shapes_and_values():
    rng = np.random.default_rng(2)
    rows = rng.normal(size=(5, 8))
    g = ProxOp(fp.Simplex(1.0), dims=(5, 8))
    z = g.prox(rows.ravel(), 1.0)
    assert z.shape == (40,)
    np.testing.assert_array_equal(z, fp.project_simplex(rows).ravel())
    assert g(z) == 0.0 and g(rows.ravel()) == np.inf
    imgs = rng.normal(size=(3, 7, 9))
    t = ProxOp(fp.TVAniso(), dims=imgs.shape, tol=1e-10)
    np.testing.assert_allclose(t.prox(imgs.ravel(), 0.2), fp.TVAniso().prox(imgs, 0.2, tol=1e-10).ravel())
    assert t(imgs.ravel()) == pytest.approx(fp.TVAniso().value(imgs).sum(), rel=1e-14)
    cols = ProxOp(fp.TV1D(), dims=(6, 5), axis=0)       # along the first axis
    x = rng.normal(size=(6, 5))
    np.testing.assert_allclose(cols.prox(x.ravel(), 0.5), fp.tv1d(x, 0.5, axis=0).ravel())
    assert cols(x.ravel()) == pytest.approx(np.abs(np.diff(x, axis=0)).sum(), rel=1e-14)


def test_proximal_gradient_lasso():
    """The same lasso through PyProximal's L1 and through fastprox: the same
    iterates, so the same solution."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(40, 60))
    x_true = np.zeros(60)
    x_true[rng.choice(60, 6, replace=False)] = rng.normal(size=6) * 3
    b = A @ x_true + 0.01 * rng.normal(size=40)
    f = pyproximal.L2(Op=pylops.MatrixMult(A), b=b)
    tau = 0.9 / np.linalg.norm(A, 2) ** 2
    run = lambda g: pyproximal.optimization.primal.ProximalGradient(  # noqa: E731
        f, g, np.zeros(60), tau=tau, epsg=1.0, niter=300, acceleration="fista")
    x_fp, x_pp = run(ProxOp(fp.Soft(0.5))), run(pyproximal.L1(sigma=0.5))
    np.testing.assert_allclose(x_fp, x_pp, rtol=1e-10, atol=1e-12)
    assert np.count_nonzero(np.abs(x_fp) > 1e-8) <= 20


def test_warm_starts_agree():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 10))
    cold = ProxOp(fp.TVIso(), dims=x.shape, tol=1e-12, max_iter=100000)
    warm = ProxOp(fp.TVIso(), dims=x.shape, tol=1e-12, max_iter=100000, warm=True)
    for tau in (0.3, 0.31, 0.32):
        np.testing.assert_allclose(warm.prox(x.ravel(), tau), cold.prox(x.ravel(), tau), atol=1e-8)
        np.testing.assert_allclose(warm.proxdual(x.ravel(), tau), cold.proxdual(x.ravel(), tau), atol=1e-8)


def test_errors():
    g = ProxOp(fp.Soft(1.0))
    with pytest.raises(ValueError):
        g.prox(np.ones(3), 0.0)
    with pytest.raises(TypeError):
        g.prox(np.ones(3), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError):
        ProxOp(pyproximal.L1())
    assert repr(g).startswith("ProxOp(Soft(")
