"""Nonconvex pointwise maps (l0, lq, SCAD, MCP, log): global minimisers on a
dense grid with refinement, tie rules and counts exact at thresholds, the
closed forms for q = 1/2 and 2/3, the step conditions of SCAD and MCP, and
errors (the 0.2 gate in ROADMAP.md). Derivatives, threads and strides run in
test_operators.py through CASES."""

import numpy as np
import pytest

import fastprox as fp

import reference as ref

MAPS = {
    "l0": (lambda: fp.L0(0.7), lambda z: 0.7 * (z != 0)),
    "lq_half": (lambda: fp.Lq(0.6, 0.5), lambda z: 0.6 * np.abs(z) ** 0.5),
    "lq_two_thirds": (lambda: fp.Lq(0.6, 2 / 3), lambda z: 0.6 * np.abs(z) ** (2 / 3)),
    "lq_0.2": (lambda: fp.Lq(0.6, 0.2), lambda z: 0.6 * np.abs(z) ** 0.2),
    "lq_0.95": (lambda: fp.Lq(0.6, 0.95), lambda z: 0.6 * np.abs(z) ** 0.95),
    "scad": (lambda: fp.SCAD(0.8, 3.7), ref.scad_psi(0.8, 3.7)),
    "mcp": (lambda: fp.MCP(0.8, 3.0), ref.mcp_psi(0.8, 3.0)),
    "log_nonconvex": (lambda: fp.LogPenalty(1.0, 0.3), lambda z: np.log1p(np.abs(z) / 0.3)),
    "log_convex": (lambda: fp.LogPenalty(0.05, 0.5), lambda z: 0.05 * np.log1p(np.abs(z) / 0.5)),
}


def threshold(op, gamma, hi=20.0, tie="nonzero"):
    """Smallest positive double where the map is nonzero with this tie rule."""
    lo = 0.0
    while np.nextafter(lo, hi) < hi:
        mid = 0.5 * (lo + hi)
        if op.prox(np.array([mid]), gamma, tie=tie)[0] == 0.0:
            lo = mid
        else:
            hi = mid
    return hi


@pytest.mark.parametrize("name", sorted(MAPS))
@pytest.mark.parametrize("gamma", [0.3, 1.0, 2.0])
def test_global_minimiser_on_a_refined_grid(name, gamma):
    make, psi = MAPS[name]
    op = make()
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(scale=2.0, size=60), [0.0, 1e-9, -1e-300, 25.0]])
    t = threshold(op, gamma)
    x = np.concatenate([x, [t, np.nextafter(t, 0), -t, 1.0001 * t, 0.9999 * t]])  # around the jump
    z = op.prox(x, gamma)
    zg = ref.grid_prox(x, gamma, psi)
    fz = ref.nonconvex_objective(z, x, gamma, psi)
    fg = ref.nonconvex_objective(zg, x, gamma, psi)
    assert np.all(fz <= fg + 1e-13 * (1 + x ** 2))
    # Away from the jump the minimiser is unique and both agree; at a tie the
    # grid may pick the other minimiser.
    far = np.abs(np.abs(x) - t) > 1e-6 * t
    assert np.all(np.abs(z - zg)[far] <= 1e-7 * (1 + np.abs(x[far])))
    assert np.all(np.sign(z) * np.sign(x) >= 0) and np.all(np.abs(z) <= np.abs(x) + 1e-15)


@pytest.mark.parametrize("name", ["l0", "lq_half", "lq_two_thirds", "lq_0.2"])
def test_ties_exact_at_thresholds(name):
    op = MAPS[name][0]()
    gamma = 0.9
    t = threshold(op, gamma)
    below = np.nextafter(t, 0)
    x = np.array([below, t, -t, 2 * t])
    zn, cn = op.prox(x, gamma, keep=True, tie="nonzero")
    zz, cz = op.prox(x, gamma, keep=True, tie="zero")
    assert zn[0] == zz[0] == 0.0 and zn[3] == zz[3] != 0.0
    if cn.info.ties:  # the threshold is representable: t sits on the tie
        assert cn.info.ties == cz.info.ties == 2
        assert zn[1] != 0.0 and zz[1] == 0.0 and zn[2] == -zn[1] and zz[2] == 0.0
    else:  # it falls between doubles: no tie, both rules agree
        assert cz.info.ties == 0 and np.array_equal(zn, zz)
    if name == "l0":  # sqrt(2 gamma lam) is exact: always a tie, z = x there
        assert t == np.sqrt(2 * gamma * 0.7) and cn.info.ties == 2 and zn[1] == t
    # Derivatives follow the branch the rule picked.
    jn = op.jvp(cn, np.ones(4))
    jz = op.jvp(cz, np.ones(4))
    if cn.info.ties:
        assert jn[1] > 0 and jz[1] == 0
    # A map without ties never reports one.
    _, c = fp.SCAD(0.8, 3.7).prox(x, 1.0, keep=True)
    assert c.info.ties == 0


def test_closed_forms_match_newton():
    # q = 1/2 and 2/3 use closed forms; a q one ulp away takes the Newton path.
    x = np.random.default_rng(4).normal(scale=3.0, size=2000)
    for q in (0.5, 2 / 3):
        z = fp.lq(x, 0.7, q, 1.3)
        zn = fp.lq(x, 0.7, np.nextafter(q, 1.0), 1.3)
        on = (z != 0) & (zn != 0)
        assert on.sum() > 500
        np.testing.assert_allclose(z[on], zn[on], rtol=1e-12, atol=0)


def test_step_conditions_and_known_values():
    x = np.array([-3.0, -1.0, -0.2, 0.0, 0.5, 1.2, 4.0])
    # Fan and Li's rule with gamma = 1 (a = 3.7, lambda = 1).
    lam, a = 1.0, 3.7
    expect = np.where(np.abs(x) <= 2 * lam, np.sign(x) * np.maximum(np.abs(x) - lam, 0),
                      np.where(np.abs(x) <= a * lam, ((a - 1) * x - np.sign(x) * a * lam) / (a - 2), x))
    np.testing.assert_allclose(fp.scad(x, lam, a, 1.0), expect, rtol=1e-15, atol=1e-15)
    # Zhang's firm rule with gamma = 1 (b = 3, lambda = 1).
    b = 3.0
    expect = np.where(np.abs(x) <= lam, 0.0,
                      np.where(np.abs(x) <= b * lam, np.sign(x) * (np.abs(x) - lam) / (1 - 1 / b), x))
    np.testing.assert_allclose(fp.mcp(x, lam, b, 1.0), expect, rtol=1e-15, atol=1e-15)
    with pytest.raises(ValueError):
        fp.scad(x, 1.0, 3.7, 2.7)  # needs a > 1 + gamma
    with pytest.raises(ValueError):
        fp.mcp(x, 1.0, 3.0, 3.0)  # needs b > gamma
    with pytest.raises(ValueError):
        fp.SCAD(1.0, 3.7).prox_conj(x, 1.0 / 2.7)  # the kernel's step is 1/sigma
    # l0 keeps x above sqrt(2 gamma lam) = 1, and on it with the default tie
    # rule (x = -1 sits on the tie); gamma = 0 is the identity for all.
    np.testing.assert_array_equal(fp.hard_threshold(x, 0.5, 1.0), np.where(np.abs(x) >= 1.0, x, 0.0))
    np.testing.assert_array_equal(fp.hard_threshold(x, 0.5, 1.0, tie="zero"),
                                  np.where(np.abs(x) > 1.0, x, 0.0))
    for f in (lambda v: fp.lq(v, 1.0, 0.5, 0.0), lambda v: fp.prox_log(v, 1.0, 0.1, 0.0),
              lambda v: fp.hard_threshold(v, 1.0, 0.0)):
        np.testing.assert_array_equal(f(x), x)


def test_weights_and_errors():
    x = np.linspace(-3, 3, 7)
    w = np.array([0.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.0])
    z = fp.lq(x, 0.6, 0.5, 1.0, weights=w)
    assert z[0] == x[0] and z[-1] == x[-1]  # zero weight: identity
    np.testing.assert_allclose(z[1], fp.lq(x[1:2], 0.6, 0.5, 1.0)[0])
    for bad in (lambda: fp.Lq(1.0, 0.0), lambda: fp.Lq(1.0, 1.0), lambda: fp.SCAD(1.0, 1.0),
                lambda: fp.MCP(1.0, 0.0), lambda: fp.LogPenalty(1.0, 0.0), lambda: fp.L0(-1.0)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        fp.lq(x, tie="maybe")
    v = np.array([1.0, np.nan, 3.0])
    for f in (fp.hard_threshold, fp.lq, fp.scad, fp.mcp, fp.prox_log):
        r = f(v)
        assert np.isnan(r[1]) and not np.isnan(r[[0, 2]]).any()


def test_lq_lam_gradient():
    x = np.random.default_rng(8).normal(scale=2.0, size=40)
    op = fp.Lq(0.4, 0.5)
    _, ctx = op.prox(x, 1.0, keep=True)
    _, grads = op.vjp(ctx, np.ones_like(x))
    h = 1e-7
    fd = (fp.lq(x, 0.4 + h, 0.5).sum() - fp.lq(x, 0.4 - h, 0.5).sum()) / (2 * h)
    assert grads.lam == pytest.approx(fd, rel=1e-6)
