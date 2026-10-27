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
