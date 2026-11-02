"""Structure of the derivative (since 1.0.2; docs/spec.md, Section 7): the
guarantees against jvp for every operator, the Structure helpers, layout,
threads, branches and edge cases."""

import numpy as np
import pytest

import fastprox as fp

from test_operators import CASES, N, NAMES, NO_CONJ

FREE, ZERO = 0, 1


def _J(op, ctx, conj, **kw):
    """J v of the prox the structure describes: after prox_conj the inner
    prox, I - J_conj."""
    def f(v):
        out = op.jvp(ctx, np.asarray(v, ctx.dtype), **kw)
        return v - out if conj else out
    return f


def _units(s, a):
    """a as (count, size), the units in their own order (axis -1 here)."""
    return np.asarray(a).reshape(s._count, s._size)


def check_structure(op, ctx, s, conj, rng, rank=True, **kw):
    """Fields, helpers and guarantees 1 and 2 against jvp (with kw: a branch)."""
    J = _J(op, ctx, conj, **kw)
    shape = ctx.shape
    assert s.kind == op.struct_kind and s.conj == conj
    assert s.state.dtype == np.int8 and s.sign.dtype == np.int8 and s.labels.dtype == np.int32
    assert s.state.shape == s.labels.shape == s.sign.shape == shape
    assert s.nfixed == int(np.count_nonzero(s.state))
    nreg = np.asarray(s.nregions).reshape(-1)
    free = s.state == FREE
    # Not free: label -1 and sign 0; free: sign +-1 unless the unit holds a NaN.
    assert np.all(s.labels[~free] == -1) and np.all(s.sign[~free] == 0)
    for u, (lab, sg, k) in enumerate(zip(_units(s, s.labels), _units(s, s.sign), nreg)):
        if k < 0:
            assert np.all(lab == -1) and np.all(sg == 0)
            continue
        seen = lab[lab >= 0]
        first = np.unique(seen, return_index=True)[1]  # canonical: 0, 1, ... by first entry
        assert np.array_equal(seen[np.sort(first)], np.arange(k))
        assert np.all(np.abs(sg[lab >= 0]) == 1)
    if s.coef is None:
        assert s.active is None
    else:
        assert s.coef.dtype == ctx.dtype and s.active.dtype == bool
        assert np.all(s.coef[~free] == 0)
        assert np.all(_units(s, s.coef)[~s.active.reshape(-1)] == 0)
    # Helpers: offsets, global labels, M^T and M adjoint, matrix(unit).
    off = s.offsets
    assert off[0] == 0 and np.all(np.diff(off) == np.maximum(nreg, 0))
    gl = s.global_labels()
    assert np.array_equal(gl >= 0, s.labels >= 0)
    v = rng.normal(size=shape)
    c = rng.normal(size=off[-1])
    assert np.vdot(s.reduce(v), c) == pytest.approx(np.vdot(v, s.expand(c)), rel=1e-12, abs=1e-12)
    for u in range(s._count):
        if nreg[u] >= 0:
            np.testing.assert_allclose(s.matrix(u) @ c[off[u]:off[u + 1]], _units(s, s.expand(c))[u],
                                       atol=1e-14)
    # Guarantee 1: J v lies in L; guarantee 2 by kind.
    ok = np.repeat(nreg >= 0, s._size).reshape(shape)  # units without a NaN
    dx = rng.normal(size=shape)
    Jdx = J(dx)
    scale = 1e-10 * (1.0 + np.abs(dx).max())
    np.testing.assert_allclose(np.where(ok, s.project(Jdx), 0), np.where(ok, Jdx, 0), atol=scale)
    if s.kind in ("projector", "weighted"):
        np.testing.assert_allclose(np.where(ok, Jdx, 0), np.where(ok, s.project(dx), 0), atol=scale)
    else:
        w = s.project(rng.normal(size=shape))
        Jw = _units(s, J(w))
        wu = _units(s, w)
        vjv, vv = np.einsum("ij,ij->i", wu, Jw), np.einsum("ij,ij->i", wu, wu)
        assert np.all((vv == 0) | (vjv > 1e-8 * vv) | (nreg < 0)), (vjv, vv)
    # Range: rank J = dim L on every unit, from the size JVPs of the unit
    # vectors (the same position in every unit at once).
    if rank:
        cols = []
        for i in range(s._size):
            e = np.zeros((s._count, s._size))
            e[:, i] = 1.0
            cols.append(_units(s, J(e.reshape(shape))))
        Jm = np.stack(cols, axis=-1)  # (count, size, size)
        dim = np.asarray(s.dim).reshape(-1)
        for u in range(s._count):
            if nreg[u] >= 0:
                assert np.linalg.matrix_rank(Jm[u], tol=1e-9) == dim[u], (u, dim[u])


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_guarantees(name, conj):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    if conj and name in NO_CONJ:
        pytest.skip("no Euclidean conjugate")
    for seed in range(3):
        rng = np.random.default_rng(seed)
        # Rows inside and outside the sets, and an all-zero row.
        x = rng.normal(scale=1.5, size=(4, N)) * np.array([[0.05], [1.0], [4.0], [0.0]])
        prox = op.prox_conj if conj else op.prox
        _, ctx = prox(x, gamma, keep=True)
        check_structure(op, ctx, ctx.structure(), conj, rng)


def test_kinds():
    expect = {"soft": "projector", "elastic_net": "curved", "huber": "curved", "box": "projector",
              "group_l2": "curved", "l2_ball": "curved", "simplex": "projector", "isotonic": "projector",
              "isotonic_w": "weighted", "tv1d": "projector", "scad": "curved", "l0": "projector"}
    for name, kind in expect.items():
        build, theta = CASES[name][:2]
        assert build(theta).struct_kind == kind, name
    assert fp.TVAniso(2).struct_kind == "projector" and fp.TVIso(2).struct_kind == "curved"
    assert fp.NuclearNorm(1.0).struct_kind is None


def test_by_hand():
    # l-infinity at (3, -2.5, 0.5, 2.5): tau = 7/3 clips entries 0, 1 and 3.
    _, ctx = fp.LInf(1.0).prox(np.array([3.0, -2.5, 0.5, 2.5]), keep=True)
    s = ctx.structure()
    assert s.labels.tolist() == [0, 0, 1, 0] and s.sign.tolist() == [1, -1, 1, 1]
    assert int(s.nregions) == 2 and s.coef is None and int(s.dim) == 2
    np.testing.assert_array_equal(s.matrix(), [[1, 0], [-1, 0], [0, 1], [1, 0]])
    # The simplex at (0.2, 1.0, -0.5, 0.6): support {1, 3}, constraint 1 there.
    _, ctx = fp.Simplex(1.0).prox(np.array([[0.2, 1.0, -0.5, 0.6]]), keep=True)
    s = ctx.structure()
    assert s.state.tolist() == [[ZERO, FREE, ZERO, FREE]] and s.nfixed == 2
    assert s.coef.tolist() == [[0, 1, 0, 1]] and s.active.tolist() == [True]
    assert s.dim.tolist() == [1] and s.offsets.tolist() == [0, 2]
    # Weighted isotonic regression: the projection orthogonal in diag(w).
    w = np.array([1.0, 3.0, 1.0, 2.0])
    _, ctx = fp.Isotonic(w).prox(np.array([2.0, 1.0, 0.0, 5.0]), keep=True)
    s = ctx.structure()
    assert s.kind == "weighted" and s.labels.tolist() == [0, 0, 0, 1]
    v = np.array([1.0, -2.0, 0.5, 3.0])
    p = s.project(v)
    np.testing.assert_allclose(p[:3], (w[:3] @ v[:3]) / w[:3].sum())
    for c in np.eye(2):  # W-orthogonal to L
        assert np.vdot(w * (v - p), s.expand(c)) == pytest.approx(0.0, abs=1e-14)


def test_coefficient_scale():
    # Each unit's coefficients come scaled so the largest |a_i| is 1. In
    # float32 the hyperplane's a_F at 5e-45 of max |a| (an entry at its
    # bound) were 4, 5 and 7 subnormals, and a^T (J dx) lost the constraint
    # (fuzzing guarantee 1, 1.0.2); the l1 ball's weights of 1e-50 were 0.
    a = np.array([0.765625, 1.109375, -1.5625, 1.5227227580871691e44])
    lo, hi = np.r_[-np.inf, -np.inf, -np.inf, 0.0], np.r_[np.inf, np.inf, np.inf, 0.0]
    x = np.array([[2.0, -1.0, 0.5, 3.0], [0.25, 4.0, -2.0, -1.0]], np.float32)
    dx = np.array([[1.0, -2.0, 3.0, 0.5], [-1.0, 0.5, 2.0, 1.0]], np.float32)
    _, ctx = fp.Hyperplane(a, b=0.0, lower=lo, upper=hi).prox(x, keep=True)
    s = ctx.structure()
    assert s.active.all() and s.coef.dtype == np.float32
    np.testing.assert_allclose(s.coef[:, :3], np.broadcast_to(a[:3] / 1.5625, (2, 3)), rtol=1e-7)
    assert (s.coef[:, 3] == 0).all()
    J = ctx.op.jvp(ctx, dx)
    assert (np.abs(np.sum(s.coef * J, axis=1)) <= 1e-5 * np.abs(dx).max(axis=1)).all()
    w = np.array([1e-50, 2e-50, 4e-50])
    _, ctx = fp.L1Ball(1e-50, w).prox(np.array([3.0, -1.0, 2.0], np.float32), keep=True)
    s = ctx.structure()
    assert s.active.all() and np.abs(s.coef).max() == 1.0
    np.testing.assert_array_equal(s.coef != 0, s.state == FREE)


@pytest.mark.parametrize("name", NAMES)
def test_threads_axis_strides(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)

    def fields(s):
        return [s.state, s.labels, s.sign, s.nregions] + ([] if s.coef is None else [s.coef, s.active])

    x = np.random.default_rng(5).normal(size=(4000, N))
    ref = None
    for t in (1, 2, 8):
        _, ctx = op.prox(x, gamma, keep=True, threads=t)
        f = [a.tobytes() for a in fields(ctx.structure())]
        ref = f if ref is None else ref
        assert f == ref
    # Along axis 1 of a (5, N, 4) array: the moved array's structure.
    x = np.random.default_rng(6).normal(size=(5, N, 4))
    _, c1 = op.prox(x, gamma, axis=1, keep=True)
    _, c2 = op.prox(np.ascontiguousarray(np.moveaxis(x, 1, -1)), gamma, keep=True)
    s1, s2 = c1.structure(), c2.structure()
    for a, b in zip(fields(s1), fields(s2)):
        if op._elementwise and a is s1.labels:
            continue  # one unit, numbered in C order of the array: the axis order shows
        if a.shape == x.shape:
            b = np.moveaxis(b, -1, 1)
        np.testing.assert_array_equal(a, b)
    v = np.random.default_rng(7).normal(size=x.shape)
    np.testing.assert_allclose(s1.project(v), np.moveaxis(s2.project(np.moveaxis(v, 1, -1)), -1, 1),
                               atol=1e-14)
    # A reversed, sliced view: the same as its copy.
    view = x[::-1, :, ::2]
    _, c3 = op.prox(view, gamma, axis=1, keep=True)
    _, c4 = op.prox(np.ascontiguousarray(view), gamma, axis=1, keep=True)
    for a, b in zip(fields(c3.structure()), fields(c4.structure())):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_branches(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    rng = np.random.default_rng(11)
    x = rng.normal(scale=1.5, size=(4, N))
    _, ctx = op.prox(x, gamma, keep=True)
    tol = 0.3  # loose: many entries count as kinks
    try:
        mask = ctx.kinks(tol)
    except ValueError:
        pytest.skip("no kink masks")
    if not mask.any():
        pytest.skip("no kink within tol")
    try:
        side = rng.choice([-1, 1], size=mask.shape).astype(np.int8)
        s = ctx.structure(branch=side, kink_tol=tol)
    except ValueError as e:
        assert "branch" in str(e) or "fp_workspace_set_branch" in str(e)
        pytest.skip("no branch overrides")
    check_structure(op, ctx, s, False, rng, branch=side, kink_tol=tol)
    # One call only: the default structure afterwards.
    d = ctx.structure()
    check_structure(op, ctx, d, False, rng)


@pytest.mark.parametrize("name", NAMES)
def test_nan_empty_and_identity(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    x = np.random.default_rng(3).normal(size=(3, N))
    x[1, 3] = np.nan
    _, ctx = op.prox(x, gamma, keep=True)
    s = ctx.structure()
    if op._elementwise:  # only the NaN entry is marked
        assert s.state[1, 3] == FREE and s.labels[1, 3] == -1 and s.sign[1, 3] == 0
        assert int(s.nregions) == np.count_nonzero(s.labels >= 0)
    else:  # the whole row
        assert s.nregions[1] == -1 and s.dim[1] == -1
        assert np.all(s.labels[1] == -1) and np.all(s.sign[1] == 0) and np.all(s.state[1] == FREE)
        assert s.active is None or not s.active[1]
        assert s.nregions[0] >= 0 and s.offsets[2] == s.offsets[1]
    # Empty: no units.
    _, ctx = op.prox(np.zeros((0, N)), gamma, keep=True)
    s = ctx.structure()
    assert s.offsets.tolist() == [0] and s.nfixed == 0
    assert s.nregions.shape == (() if op._elementwise else (0,))
    # After prox_conj at sigma = 0, J = I: the inner prox's I - J is 0.
    if name not in NO_CONJ:
        y = np.random.default_rng(4).normal(size=(3, N))
        _, ctx = op.prox_conj(y, 0.0, keep=True)
        s = ctx.structure()
        assert np.all(s.state == ZERO) and np.all(np.asarray(s.nregions) == 0) and s.conj


IMAGES = {
    "tv_aniso": (lambda: fp.TVAniso(2), (3, 6, 7)),
    "tv_aniso_3d": (lambda: fp.TVAniso(3), (2, 3, 4, 5)),
    "tv_iso": (lambda: fp.TVIso(2), (3, 6, 7)),
    "graph_tv": (lambda: fp.GraphTV(12, np.array([[i, i + 1] for i in range(11)] + [[0, 5], [3, 9]])), (3, 12)),
    "tree_tv": (lambda: fp.TreeTV(12, np.array([[(i - 1) // 2, i] for i in range(1, 12)])), (3, 12)),
}


@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.parametrize("name", sorted(IMAGES))
def test_images(name, conj):
    make, shape = IMAGES[name]
    op = make()
    rng = np.random.default_rng(8)
    m = int(np.prod(shape[1:]))
    x = rng.normal(scale=0.2, size=shape) + (np.arange(m) * 3 // m - 1.0).reshape(shape[1:])
    prox = op.prox_conj if conj else op.prox
    _, ctx = prox(x, 0.6, keep=True, tol=1e-10, max_iter=20000)
    s = ctx.structure()
    assert s.nregions.shape == shape[:1] and s._size == m
    assert np.any((s.nregions > 0) & (s.nregions < m))  # regions formed
    check_structure(op, ctx, s, conj, rng, rank=False)
    # After a refine (isotropic TV), the refined point's structure.
    if name == "tv_iso":
        ctx.refine(tol=1e-13)
        check_structure(op, ctx, ctx.structure(), conj, rng, rank=False)


def test_spectral_has_none():
    op = fp.NuclearNorm(0.5)
    _, ctx = op.prox(np.random.default_rng(0).normal(size=(4, 3)), keep=True)
    with pytest.raises(TypeError, match="no structure query"):
        ctx.structure()


def test_errors():
    _, ctx = fp.Soft(0.5).prox(np.linspace(-2, 2, 5), keep=True)
    with pytest.raises(ValueError):
        ctx.structure(branch=np.full(5, 2))
    with pytest.raises(ValueError):
        ctx.structure(kink_tol=-1.0)
    s = ctx.structure()
    with pytest.raises(IndexError):
        s.matrix(1)
    assert "projector" in repr(s)


# ------------------------------------------- group l2 kept live at its kink (S1)

G0 = np.array([0.25, 0.375, 0.75])  # norm 0.875, exactly
U0 = np.array([2.0, 3.0, 6.0]) / 7
GIDS = np.array([0, 0, 0, 1, 1, 1])


def _dense(op, ctx, n, **kw):
    return np.stack([op.jvp(ctx, np.eye(n)[j], **kw) for j in range(n)], 1)


@pytest.mark.parametrize("scale, tol", [(1.0, 0.0), (1 - 1e-6, 1e-5), (1 + 1e-6, 1e-5)])
def test_group_l2_live_at_kink(scale, tol):
    """Branch +1 on a group at ||x_g|| = tau (or within kink_tol below):
    J_g = u u^T, the live piece at the kink, with dz/dtheta = -gamma u, and
    the structure refused; above tau the live formula and a structure."""
    op = fp.GroupL2(1.0, GIDS)
    x = np.concatenate([G0 * scale, [2.0, 1.0, 2.0]])
    gamma = 0.875
    _, ctx = op.prox(x, gamma, keep=True)
    J = _dense(op, ctx, 6, branch=1, kink_tol=tol)
    c0 = 1.0 if scale <= 1 else 1 / scale
    np.testing.assert_allclose(J[:3, :3], (1 - c0) * np.eye(3) + c0 * np.outer(U0, U0), atol=1e-12)
    np.testing.assert_allclose(J, J.T, atol=1e-15)
    w = np.linalg.eigvalsh(J)
    assert w.min() >= -1e-12 and w.max() <= 1 + 1e-12
    dz = op.jvp(ctx, None, 0.0, np.ones(1), branch=1, kink_tol=tol)
    np.testing.assert_allclose(dz[:3], -gamma * U0, atol=1e-12)
    if scale <= 1:
        with pytest.raises(ValueError, match="span"):
            ctx.structure(branch=1, kink_tol=tol)
    else:
        assert ctx.structure(branch=1, kink_tol=tol).state.tolist() == [FREE] * 6
    # Side -1 and the default: a structure.
    ctx.structure(branch=-1, kink_tol=tol)
    ctx.structure()


def test_group_l2_live_at_kink_parameters_match_the_live_side():
    """The parameter tangent at the kink is the live side's: one-sided
    differences in the weight from x_g = tau u, where z_g = (1 - tau/||x_g||) x_g."""
    op = fp.GroupL2(1.0, GIDS)
    x = np.concatenate([G0, [2.0, 1.0, 2.0]])
    _, ctx = op.prox(x, 0.875, keep=True)
    dz = op.jvp(ctx, None, 0.0, np.ones(1), branch=1)
    h = 1e-7  # decreasing the weight keeps group 0 live
    fd = (fp.GroupL2(1.0, GIDS, [1.0]).prox(x, 0.875) - fp.GroupL2(1.0, GIDS, [1.0 - h]).prox(x, 0.875)) / h
    np.testing.assert_allclose(dz, fd, atol=1e-6)


def test_group_l2_zero_norm_cannot_be_kept_live():
    t = 0.625 * 2.0 ** -20
    op = fp.GroupL2(1.0, GIDS)
    x = np.array([0, 0, 0, 0.375 * 2.0 ** -20, 0.5 * 2.0 ** -20, 0])
    _, ctx = op.prox(x, t, keep=True)
    with pytest.raises(ValueError, match="no direction"):
        op.jvp(ctx, np.ones(6), branch=1, kink_tol=1e-5)
    J = _dense(op, ctx, 6, branch=[-1, -1, -1, 1, 1, 1], kink_tol=1e-5)
    np.testing.assert_allclose(J[3:5, 3:5], np.outer([0.6, 0.8], [0.6, 0.8]), atol=1e-15)
    np.testing.assert_array_equal(J[:3], 0.0)


def test_group_l2_live_at_kink_conj():
    """After prox_conj the inner prox at y (step 1, tau = lambda): J_conj =
    I - u u^T on the group, and the structure (of J_inner) is refused."""
    op = fp.GroupL2(0.875, GIDS)
    y = np.concatenate([G0, [2.0, 1.0, 2.0]])
    _, ctx = op.prox_conj(y, 2.0, keep=True)
    J = _dense(op, ctx, 6, branch=1)
    np.testing.assert_allclose(J[:3, :3], np.eye(3) - np.outer(U0, U0), atol=1e-12)
    with pytest.raises(ValueError, match="span"):
        ctx.structure(branch=1)
    ctx.structure()


def test_weighted_isotonic_euclidean_pair():
    """SCD review S4: weighted isotonic regression is the projection onto the
    monotone cone C in the metric D_w = diag(w), and its J is the projector
    onto L orthogonal in D_w, not symmetric (s.project returns it). The SCD
    pair of N_C at (z, v), v = D_w (x - z), is instead the Euclidean P_L =
    expand(reduce(.) / reduce(sign)) with W = I - P (docs/api.md, "SCD
    subspaces"): P symmetric and idempotent with range J, and every
    (J u, D_w (I - J) u) in rge(P, I - P) = L x L^perp."""
    rng = np.random.default_rng(4)
    n = 9
    for t in range(20):
        w = rng.uniform(0.2, 5.0, size=n)
        x = rng.normal(size=(3, n)) + np.linspace(1.0, -1.0, n)  # pools of several sizes
        op = fp.Isotonic(w)
        _, ctx = op.prox(x, keep=True)
        s = ctx.structure()
        assert s.kind == "weighted"
        I = np.eye(n)
        for r in range(3):
            def unit(v):  # the row r of a batch whose other rows are 0
                out = np.zeros_like(x)
                out[r] = v
                return out
            J = np.column_stack([op.jvp(ctx, unit(e))[r] for e in I])
            P = np.column_stack([s.expand(s.reduce(unit(e)) / s.reduce(s.sign))[r] for e in I])
            Pw = np.column_stack([s.project(unit(e))[r] for e in I])
            np.testing.assert_allclose(P, P.T, atol=1e-15)
            np.testing.assert_allclose(P @ P, P, atol=1e-14)
            np.testing.assert_allclose(P @ J, J, atol=1e-13)  # range J within L
            assert np.linalg.matrix_rank(J, tol=1e-9) == np.linalg.matrix_rank(P, tol=1e-9)
            np.testing.assert_allclose(Pw, J, atol=1e-13)  # s.project is the D_w projector
            if t == 0 and r == 0:
                assert np.abs(Pw - Pw.T).max() > 1e-3  # which is not symmetric
            # The graph of N_C's SCD subspace: J u in L, D_w (I - J) u in L^perp.
            U = rng.normal(size=(n, 5))
            np.testing.assert_allclose(P @ (np.diag(w) @ (I - J) @ U), 0, atol=1e-12)


# ------------------------------------- exact kinks and boundaries (S6, since 1.0.3)
#
# Every operator with kink masks, at inputs built exactly on its kinks from
# representable values (steps are powers of two, so that the conjugate's
# y = x / gamma and step 1 / sigma = gamma are exact too): at kink_tol 0 on
# the kink, and a 2^-30 on either side of it within kink_tol 2^-26, for the
# sides -1, 0 and +1, after prox and prox_conj. On each side whose structure
# query succeeds, guarantees 1 and 2 and the rank (where J's positive
# singular values are resolved); a refused query fails atomically. J's
# spectrum: symmetric in [0, 1] for the convex operators (weighted isotonic
# after the change of metric D_w^1/2 J D_w^-1/2), [0, slope bound] for SCAD
# and MCP, nothing for l0 (its jump is a convention). After prox_conj, J_conj
# = I - J_inner against a separate inner call at y / sigma with step
# 1 / sigma, the Moreau relation and the chain rule in (y, sigma, theta).
# B-Jacobian membership by analytic limits: each side's J at the kink is the
# J of nearby points on that side's piece, and the default is one of them.

KINK_DELTA, KINK_TOL = 2.0 ** -30, 2.0 ** -26


def _tv_open(x):
    """For rows of 1D TV with edges at every jump: the direction that moves
    the two ends of each edge apart (the first along the row's step)."""
    d = np.sign(np.diff(x, axis=-1))
    t = np.zeros_like(x)
    t[..., :-1] -= d
    t[..., 1:] += d
    return t


def kink_fixtures():
    """name: dict(make, x on the kinks at gamma, gamma, d: {side: a direction
    into that side's piece}, spec: "convex", "jump", ("slope", bound) or
    ("weighted", w), and flags: overrides (kink masks and branches), conj,
    live (side +1 has no structure: group l2 kept live at its kink, S1),
    curved (J varies within a piece), tol0 (the kink tolerance at x itself))."""
    one = lambda x: {1: np.ones_like(x), -1: -np.ones_like(x)}  # noqa: E731
    pm = lambda d: {1: d, -1: -d}  # noqa: E731
    f = {}
    x = np.array([[0.5, -0.5, 1.0, 0.25], [-0.5, 0.75, 0.5, -2.0]])
    f["soft"] = dict(make=lambda: fp.Soft(0.5), x=x, gamma=1.0, d=one(x), spec="convex")
    x = np.array([[0.5, -0.5, -1.25, 1.5]])  # |x| = gamma lam, soft(x) = lower, soft(x) = upper
    f["l1_box"] = dict(make=lambda: fp.L1Box(0.5, -0.75, 1.0), x=x, gamma=1.0, d=one(x), spec="convex")
    x = np.array([[-1.0, 2.0, 0.5, 3.0]])
    f["box"] = dict(make=lambda: fp.Box(-1.0, 2.0), x=x, gamma=1.0, d=one(x), spec="convex")
    x = np.array([[0.0, 1.0, -1.0, 0.0]])
    f["nonneg"] = dict(make=lambda: fp.NonNeg(), x=x, gamma=1.0, d=one(x), spec="convex")
    x = np.array([[0.5, -1.0, 2.0, 1.5]])  # gamma lam, (1 + gamma) lam, a lam
    f["scad"] = dict(make=lambda: fp.SCAD(0.5, 4.0), x=x, gamma=1.0, d=one(x), spec=("slope", 3.0 / 2.0),
                     curved=True)
    x = np.array([[0.5, -1.5, 1.0, -0.5]])  # gamma lam, b lam
    f["mcp"] = dict(make=lambda: fp.MCP(0.5, 3.0), x=x, gamma=1.0, d=one(x), spec=("slope", 3.0 / 2.0),
                    curved=True)
    x = np.array([[1.0, -1.0, 0.5, 2.0]])  # the tie sqrt(2 gamma lam) = 1
    f["l0"] = dict(make=lambda: fp.L0(0.5), x=x, gamma=1.0, d=one(x), spec="jump", conj=False)
    # The simplex of radius 3 at (3, 2, 1, 1): tau = 1, entries 2 and 3 on it.
    x = np.array([[3.0, 2.0, 1.0, 1.0]])
    f["simplex"] = dict(make=lambda: fp.Simplex(3.0), x=x, gamma=1.0, d=pm(np.array([[0, 0, 1.0, 1.0]])),
                        spec="convex")
    # The l1 ball of radius 2 at (3, 1, -1, 0.5): tau = 1, entries 1 and 2 on it.
    x = np.array([[3.0, 1.0, -1.0, 0.5]])
    f["l1_ball_tau"] = dict(make=lambda: fp.L1Ball(2.0), x=x, gamma=1.0, d=pm(np.array([[0, 1.0, -1.0, 0]])),
                            spec="convex")
    # On the sphere |x|_1 = 4: inside (-1) or projected with tau = 0 (+1).
    x = np.array([[1.0, 2.0, -1.0]])
    f["l1_ball_sphere"] = dict(make=lambda: fp.L1Ball(4.0), x=x, gamma=1.0, d=pm(x.copy()), spec="convex")
    # Group l2 with tau = 3 at ||(1, 2, 2)|| = 3, beside a live group.
    x = np.array([[1.0, 2.0, 2.0, 4.0, 4.0, 2.0]])
    f["group_l2"] = dict(make=lambda: fp.GroupL2(3.0, GIDS), x=x, gamma=1.0,
                         d=pm(np.array([[1.0, 2.0, 2.0, 0, 0, 0]])), spec="convex", live=True, curved=True)
    # 1D TV at gamma = 1/2: (0, 1/2, 3, 7/2) solves to (1/2, 1/2, 3, 3), with
    # the duals of edges 0 and 2 on their bound and no jump there; the
    # second row mirrored. Side +1 opens those edges, -1 closes them.
    x = np.array([[0.0, 0.5, 3.0, 3.5], [3.5, 3.0, 0.5, 0.0]])
    f["tv1d"] = dict(make=lambda: fp.TV1D(), x=x, gamma=0.5, d=pm(np.array([[-1.0, 1, -1, 1], [1, -1, 1, -1]])),
                     spec="convex")
    x = np.array([[0.0, 1.0], [0.0, 1.0]])
    f["tv_aniso"] = dict(make=lambda: fp.TVAniso(2), x=x, gamma=0.5, d={1: _tv_open(x), -1: -_tv_open(x)},
                         spec="convex")
    # Tree TV on the chain 0-1-2 at (2, -1, 5): the flow on edge (0, 1) at its bound.
    x = np.array([2.0, -1.0, 5.0])
    f["tree_tv"] = dict(make=lambda: fp.TreeTV(3, np.array([[0, 1], [1, 2]])), x=x, gamma=1.0,
                        d=pm(np.array([1.0, -1.0, 0.0])), spec="convex")
    # Graph TV on the 4-cycle at (1, 1, -1, -1): two edges saturated; the
    # solver's flow is exact only to its accuracy.
    x = np.array([1.0, 1.0, -1.0, -1.0])
    f["graph_tv"] = dict(make=lambda: fp.GraphTV(4, np.array([[0, 1], [1, 2], [2, 3], [3, 0]])), x=x,
                         gamma=1.0, d=pm(x.copy()), spec="convex", tol0=1e-10)
    # No kink masks or overrides: the default piece at an exact kink. The
    # hyperplane sum z = 1 over [0, 1]^3 at (1, 1/2, 1/4): mu = 1/4 puts
    # entry 2 exactly on its bound; weighted isotonic regression at a tie of
    # two pools, (1, 0) of mean 1/2 beside 1/2.
    x = np.array([[1.0, 0.5, 0.25]])
    f["hyperplane_box"] = dict(make=lambda: fp.Hyperplane(np.ones(3), 1.0, np.zeros(3), np.ones(3)), x=x,
                               gamma=1.0, d=pm(np.array([[0, 0, 1.0]])), spec="convex", overrides=False)
    w = np.array([1.0, 1.0, 2.0])
    x = np.array([[1.0, 0.0, 0.5]])
    f["isotonic_w"] = dict(make=lambda: fp.Isotonic(w), x=x, gamma=1.0, d=pm(np.array([[0, 0, 1.0]])),
                           spec=("weighted", w), overrides=False, conj=False)
    return f


KINKS = sorted(kink_fixtures())


def _dense_J(f, shape):
    n = int(np.prod(shape))
    return np.column_stack([np.ravel(f(np.eye(n)[j].reshape(shape))) for j in range(n)])


def _check_spectrum(J, spec, conj, shape):
    """J of the prox (conj False) or of the conjugate's prox (True)."""
    eps = 1e-12
    if spec == "jump":
        return
    if spec[0] == "slope":  # diagonal, J in [0, b]; the conjugate's I - J in [1 - b, 1]
        np.testing.assert_array_equal(J, np.diag(np.diag(J)))
        lo, hi = (1 - spec[1], 1.0) if conj else (0.0, spec[1])
        assert np.all(np.diag(J) >= lo - eps) and np.all(np.diag(J) <= hi + eps), np.diag(J)
        return
    if spec[0] == "weighted":
        sw = np.sqrt(np.broadcast_to(spec[1], shape)).ravel()
        J = sw[:, None] * J / sw[None, :]
    np.testing.assert_allclose(J, J.T, atol=eps)
    ev = np.linalg.eigvalsh(0.5 * (J + J.T))
    assert ev.min() >= -eps and ev.max() <= 1 + eps, ev


def _check_rank(Jin, s):
    """rank J_inner = dim L, when J's positive singular values are resolved
    (none between 1e-10 and 1e-6: 2^-30 from a curved kink they are not)."""
    sv = np.linalg.svd(Jin, compute_uv=False)
    if np.any((sv > 1e-10) & (sv < 1e-6)):
        return
    ok = np.asarray(s.nregions).reshape(-1) >= 0
    assert np.linalg.matrix_rank(Jin, tol=1e-8) == int(np.asarray(s.dim).reshape(-1)[ok].sum())


def _sides(fx):
    return (-1, 0, 1) if fx.get("overrides", True) else (None,)


def _kw(side, tol):
    return {} if side is None else {"branch": side, "kink_tol": tol}


@pytest.mark.filterwarnings("ignore::fastprox.ConvergenceWarning")
@pytest.mark.parametrize("name", KINKS)
def test_exact_kinks(name):
    fx = kink_fixtures()[name]
    op, x0, gamma = fx["make"](), fx["x"], fx["gamma"]
    shape = x0.shape
    solve = {"tol": 1e-14, "max_iter": 10 ** 5} if name == "tv_aniso" else {}
    rng = np.random.default_rng(12)
    tol0 = fx.get("tol0", 0.0)
    points = [(x0, tol0, 0)] + [(x0 + KINK_DELTA * fx["d"][s], KINK_TOL, s) for s in (1, -1)]
    for x, tol, at in points:
        _, ctx = op.prox(x, gamma, keep=True, **solve)
        if fx.get("overrides", True):
            assert ctx.kinks(tol).any(), name
        else:
            with pytest.raises(ValueError):
                ctx.kinks(tol)
        for conj in (False, True) if fx.get("conj", True) else (False,):
            sigma = 1.0 / gamma
            if conj:  # y = sigma x: the inner prox at x with step 1 / sigma = gamma
                y = sigma * x
                zc, ctx = op.prox_conj(y, sigma, keep=True, **solve)
                p, inner = op.prox(x, gamma, keep=True, **solve)
                np.testing.assert_allclose(zc, y - sigma * p, atol=1e-13)  # Moreau
            for side in _sides(fx):
                kw = _kw(side, sigma * tol if conj else tol)
                ikw = _kw(side, tol)
                J = _dense_J(lambda v: op.jvp(ctx, v, **kw), shape)
                _check_spectrum(J, fx["spec"], conj, shape)
                Jin = np.eye(J.shape[0]) - J if conj else J
                if conj:
                    Jinner = _dense_J(lambda v: op.jvp(inner, v, **ikw), shape)
                    np.testing.assert_allclose(Jin, Jinner, atol=1e-12)
                    # Off the kink under an override, the piece's linearisation
                    # and p itself differ by O(distance) in the dsigma p term.
                    slack = 0.0 if at == 0 or not side else 4 * KINK_TOL
                    _check_conj_chain_rule(op, ctx, inner, x, sigma, kw, ikw, rng, slack)
                before = op.jvp(ctx, x)
                if side == 1 and fx.get("live") and at != 1:
                    # The live piece at or below the kink has no descriptor
                    # (S1): refused, and nothing changes.
                    with pytest.raises(ValueError, match="span"):
                        ctx.structure(**kw)
                    np.testing.assert_array_equal(op.jvp(ctx, x), before)
                    check_structure(op, ctx, ctx.structure(), conj, rng, rank=False)
                    continue
                s = ctx.structure(**kw)
                check_structure(op, ctx, s, conj, rng, rank=False, **kw)
                _check_rank(Jin, s)
                np.testing.assert_array_equal(op.jvp(ctx, x), before)  # one call only


def _check_conj_chain_rule(op, ctx, inner, x, sigma, kw, ikw, rng, slack):
    """z_c = y - sigma p(y / sigma, 1 / sigma, theta) at y = sigma x:
    dz_c = dy - dsigma p - sigma Dp(dy / sigma - y dsigma / sigma^2,
    -dsigma / sigma^2, dtheta)."""
    y = sigma * x
    dy, ds = rng.normal(size=x.shape), 0.37
    dth = rng.normal(size=op.nparams) if op.nparams else None
    p = op.prox(x, 1.0 / sigma)
    lhs = op.jvp(ctx, dy, ds, dth, **kw)
    rhs = dy - ds * p - sigma * op.jvp(inner, dy / sigma - y * ds / sigma ** 2, -ds / sigma ** 2, dth, **ikw)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12 * (1 + np.abs(rhs).max()) + slack)


@pytest.mark.filterwarnings("ignore::fastprox.ConvergenceWarning")
@pytest.mark.parametrize("name", KINKS)
def test_exact_kinks_are_limits(name):
    """B-Jacobian membership: at the kink, side s's J (and parameter and step
    tangents) is the default J at x + t d_s, strictly inside s's piece, in the
    limit t -> 0 (exactly on a polyhedral piece, to O(t) on a curved one);
    the default is one of the sides (entry by entry for the elementwise
    maps, whose kinks are independent). Without overrides, the default is
    the J of one side."""
    fx = kink_fixtures()[name]
    op, x0, gamma = fx["make"](), fx["x"], fx["gamma"]
    shape = x0.shape
    solve = {"tol": 1e-14, "max_iter": 10 ** 5} if name == "tv_aniso" else {}
    tol0 = fx.get("tol0", 0.0)
    t = 2.0 ** -20
    atol = 4 * t if fx.get("curved") else 1e-9

    def tangents(ctx, **kw):
        J = _dense_J(lambda v: op.jvp(ctx, v, **kw), shape)
        dg = op.jvp(ctx, None, 1.0, None, **kw)
        dp = op.jvp(ctx, None, 0.0, np.ones(op.nparams), **kw) if op.nparams else np.zeros(shape)
        return J, dg, dp

    _, ctx = op.prox(x0, gamma, keep=True, **solve)
    limits = {}
    for s in (1, -1):
        _, c = op.prox(x0 + t * fx["d"][s], gamma, keep=True, **solve)
        limits[s] = tangents(c)
    default = tangents(ctx)
    if op._elementwise:
        flat = [np.stack([np.diag(a), np.ravel(b), np.ravel(c)]) for a, b, c in (default, limits[1], limits[-1])]
        near = [np.all(np.isclose(flat[0], flat[k], atol=atol), axis=0) for k in (1, 2)]
        assert np.all(near[0] | near[1]), name
    else:
        assert any(all(np.allclose(a, b, atol=atol) for a, b in zip(default, limits[s])) for s in (1, -1)), name
    if not fx.get("overrides", True):
        return
    for s in (1, -1):
        got = tangents(ctx, branch=s, kink_tol=tol0)
        for a, b in zip(got, limits[s]):
            np.testing.assert_allclose(a, b, atol=atol, err_msg=f"{name} side {s}")
    if fx["spec"] == "jump":  # the convention: the zero piece and the branch
        J = {s: tangents(ctx, branch=s, kink_tol=tol0)[0] for s in (1, -1)}
        np.testing.assert_array_equal(np.diag(J[-1]), [0, 1, 0, 1])  # x = (1, -1, 1/2, 2)
        np.testing.assert_array_equal(np.diag(J[1]), [1, 0, 0, 1])
