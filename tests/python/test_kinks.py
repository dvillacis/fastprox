"""Kink masks and branch overrides of the elementwise maps with kinks (soft
thresholding, l1 over a box, SCAD, MCP; since 0.3 the ties of l0, lq and the
log penalty), and the kink masks of 1D, anisotropic and isotropic TV (since
0.3): codes at exact and near kinks,
the tolerance, and the gate in ROADMAP.md: with each side's override the
derivative matches the one-sided finite difference from that side, in x and
in the parameters, and the mask does not depend on the thread count."""

import numpy as np
import pytest

import fastprox as fp

GAMMA = 0.8
W = np.linspace(0.6, 1.4, 6)


def cases():
    """name: (operator, breakpoints as a function of gamma: one row of six
    positive positions per code, in the operator's numbering)."""
    lam = 0.5
    return {
        "soft": (fp.Soft(lam, W), lambda g: [g * lam * W]),
        "l1_box": (fp.L1Box(lam, -0.7, 1.1, W),
                   lambda g: [g * lam * W, -0.7 - g * lam * W, 1.1 + g * lam * W]),
        "scad": (fp.SCAD(lam, 3.7, W), lambda g: [g * W * lam, (1 + g * W) * lam, np.full(6, 3.7 * lam)]),
        "mcp": (fp.MCP(lam, 3.0, W), lambda g: [g * W * lam, np.full(6, 3.0 * lam)]),
    }


NAMES = sorted(cases())


@pytest.mark.parametrize("name", NAMES)
def test_codes_at_exact_and_near_kinks(name):
    op, bp = cases()[name]
    b = np.array(bp(GAMMA))  # (codes, 6)
    for code, row in enumerate(b, start=1):
        for sign in (1.0, -1.0):
            if name == "l1_box" and code > 1:
                sign = 1.0  # the bounds are not symmetric
            x = sign * row
            _, ctx = op.prox(x, GAMMA, keep=True)
            assert np.all(ctx.kinks() == code), (code, ctx.kinks())
            # Off by 1e-9: flagged only with a tolerance at least that large.
            _, ctx = op.prox(x + 1e-9, GAMMA, keep=True)
            assert np.all(ctx.kinks(1e-8) == code) and np.all(ctx.kinks(1e-10) == 0)
    # Away from every breakpoint.
    far = np.full(6, 0.73 * b[0].max() if name != "l1_box" else 0.05)
    _, ctx = op.prox(far, GAMMA, keep=True)
    assert np.all(ctx.kinks(1e-3) == 0)


@pytest.mark.parametrize("name", NAMES)
def test_identity_maps_have_no_kinks(name):
    op, _ = cases()[name]
    x = np.linspace(-3, 3, 6)
    if name == "l1_box":  # gamma = 0 leaves only the bounds
        _, ctx = op.prox(np.array([-0.7, 0.0, 1.1, 0.3, 2.0, -2.0]), 0.0, keep=True)
        assert list(ctx.kinks()) == [2, 0, 3, 0, 0, 0]
        return
    _, ctx = op.prox(x, 0.0, keep=True)
    assert not ctx.kinks(10.0).any()


def _piece_direction(op, rng):
    """A direction (dx, dgamma, dtheta) whose x part dominates how fast the
    breakpoints move, so x + h d stays on one side of each kink."""
    return 0.05, 0.05 * rng.uniform(-1, 1, op.nparams)


@pytest.mark.parametrize("name", NAMES)
def test_each_side_matches_its_one_sided_difference(name):
    op, bp = cases()[name]
    rng = np.random.default_rng(3)
    b = np.array(bp(GAMMA))
    x = np.concatenate([b[c] * (1.0 if name == "l1_box" and c else s)
                        for c in range(len(b)) for s in (1.0, -1.0)])
    x = x.reshape(-1, 6)  # every breakpoint, each on all six entries
    _, ctx = op.prox(x, GAMMA, keep=True)
    assert np.all(ctx.kinks() > 0) and x.shape[0] >= 2 * len(b) - 2
    dg, dth = _piece_direction(op, rng)
    h = 1e-7
    for side in (1, -1):
        dx = side * np.ones_like(x)
        jv = op.jvp(ctx, dx, dg * side, dth * side, branch=side)
        # The same piece along x + t d for small t > 0 (dx dominates).
        th = op_theta(op)

        def at(t):
            return rebuild(name, th + t * dth * side).prox(x + t * dx, GAMMA + t * dg * side)

        fd = (at(h) - at(0.0)) / h
        assert np.abs(fd - jv).max() <= 1e-5 * (1 + np.abs(fd).max()), (side, fd - jv)
        # The vjp with the same override is its adjoint, parameters included.
        g = rng.normal(size=x.shape)
        gx, grads = op.vjp(ctx, g, branch=side)
        lhs = float((g * jv).sum())
        rhs = float((gx * dx).sum() + grads.gamma * dg * side + grads.params @ (dth * side))
        assert abs(lhs - rhs) <= 1e-12 * (1 + abs(lhs))
    # The two sides differ at every one of these kinks, and the default is one
    # of them.
    jp, jm, j0 = (op.jvp(ctx, np.ones_like(x), branch=s) for s in (1, -1, 0))
    assert np.all((jp != jm) | (np.abs(x) < 1e-300))
    assert np.all((j0 == jp) | (j0 == jm))


def op_theta(op):
    return {"Soft": lambda: W.copy(), "L1Box": lambda: np.r_[W, -0.7, 1.1],
            "SCAD": lambda: np.r_[0.5, 3.7, W], "MCP": lambda: np.r_[0.5, 3.0, W]}[type(op).__name__]()


def rebuild(name, th):
    if name == "soft":
        return fp.Soft(0.5, th)
    if name == "l1_box":
        return fp.L1Box(0.5, th[6], th[7], th[:6])
    if name == "scad":
        return fp.SCAD(th[0], th[1], th[2:])
    return fp.MCP(th[0], th[1], th[2:])


@pytest.mark.parametrize("name", NAMES)
def test_near_kink_uses_the_other_piece_at_x(name):
    """Within the tolerance but not on the kink, the other side's derivative
    is that piece's formula at x, which a difference across the kink sees."""
    op, bp = cases()[name]
    b = np.array(bp(GAMMA))[-1]
    x = (b - 1e-6).reshape(1, 6)  # just below the last breakpoint
    _, ctx = op.prox(x, GAMMA, keep=True)
    assert np.all(ctx.kinks(1e-5) > 0)
    jup = op.jvp(ctx, np.ones_like(x), branch=1, kink_tol=1e-5)
    h = 1e-4  # crosses the kink, then stays on the upper piece
    fd = (op.prox(x + 2 * h, GAMMA) - op.prox(x + h, GAMMA)) / h
    np.testing.assert_allclose(jup, fd, rtol=1e-6, atol=1e-9)
    # Without the tolerance the override does not apply.
    np.testing.assert_array_equal(op.jvp(ctx, np.ones_like(x), branch=1), op.jvp(ctx, np.ones_like(x)))


def test_conjugate_mode_and_axis():
    op = fp.MCP(0.5, 3.0, W)
    sigma = 2.0  # the kernel's step 1/sigma, the breakpoints at y = sigma b
    b = (1 / sigma) * W * 0.5
    y = np.stack([sigma * b, sigma * 1.5 * np.ones(6), sigma * b + 2e-9])
    _, ctx = op.prox_conj(y, sigma, keep=True)
    assert list(ctx.kinks(1e-9).ravel()) == [1] * 6 + [2] * 6 + [0] * 6  # tol in units of y
    assert np.all(ctx.kinks(4e-9)[2] == 1)
    # Per-entry parameters along axis 0.
    _, ctx = op.prox(np.ascontiguousarray((GAMMA * W * 0.5)[:, None] * np.ones((6, 3))), GAMMA, axis=0,
                     keep=True)
    assert np.all(ctx.kinks() == 1)


def test_threads_float32_and_errors():
    op = fp.Soft(0.5)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 500))
    x[::7, ::3] = 0.4  # gamma lam
    masks = []
    for t in (1, 3, 8):
        _, ctx = op.prox(x, GAMMA, keep=True, threads=t)
        masks.append(ctx.kinks(1e-12))
    assert all(np.array_equal(masks[0], m) for m in masks[1:]) and masks[0].sum() > 0
    z32, ctx32 = op.prox(x.astype(np.float32), GAMMA, keep=True)
    assert np.array_equal(ctx32.kinks(1e-6) > 0, np.abs(np.abs(x.astype(np.float32)) - np.float32(0.4))
                          <= 1e-6)
    _, ctx = op.prox(x, GAMMA, keep=True)
    with pytest.raises(ValueError):
        op.jvp(ctx, x, branch=2)
    with pytest.raises(ValueError):
        op.jvp(ctx, x, branch=1, kink_tol=-1.0)
    # The override holds for one call only.
    d0 = op.jvp(ctx, np.ones_like(x))
    op.jvp(ctx, np.ones_like(x), branch=1)
    np.testing.assert_array_equal(op.jvp(ctx, np.ones_like(x)), d0)
    for other in (fp.Huber(0.5, 0.3), fp.ElasticNet(0.3, 0.2)):
        _, c2 = other.prox(x, 0.5, keep=True)
        with pytest.raises(ValueError):
            c2.kinks()
        with pytest.raises(ValueError):
            other.jvp(c2, x, branch=1)


def test_jacobian_diag_reads_the_scd_subspaces():
    """MCP: 0 on the thresholded piece, b / (b - gamma w) on the concave one,
    1 on the identity; at a kink the branch override picks the side."""
    op = fp.MCP(0.5, 3.0, W)
    c = GAMMA * W * 0.5  # gamma w lambda
    x = np.stack([0.5 * c, 0.5 * (c + 1.5), np.full(6, 2.0), c])
    _, ctx = op.prox(x, GAMMA, keep=True)
    d = ctx.jacobian_diag()
    slope = 3.0 / (3.0 - GAMMA * W)
    np.testing.assert_array_equal(d[0], 0.0)
    np.testing.assert_allclose(d[1], slope, rtol=1e-15)
    np.testing.assert_array_equal(d[2], 1.0)
    np.testing.assert_array_equal(d[3], 0.0)  # the default rule at |x| = gamma w lambda
    np.testing.assert_allclose(ctx.jacobian_diag(branch=1)[3], slope, rtol=1e-15)
    np.testing.assert_array_equal(ctx.jacobian_diag(branch=-1)[3], 0.0)


# ---- ties of the nonconvex maps (since 0.3) ---------------------------------


def _lq_tau(lam, q):
    beta = (2 * lam * (1 - q)) ** (1 / (2 - q))
    return beta + lam * q * beta ** (q - 1)


def _log_breakpoint(lam, eps):
    if lam <= eps * eps:
        return lam / eps

    def df(a):
        r = 0.5 * ((a - eps) + np.sqrt((a + eps) ** 2 - 4 * lam))
        return r * (0.5 * r - a) + lam * np.log1p(r / eps)
    lo, hi = 2 * np.sqrt(lam) - eps, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if df(mid) > 0 else (lo, mid)
    return hi


def tie_cases():
    """name: (operator class and arguments, breakpoints |x| at gamma (six), continuous)."""
    return {
        "l0": (lambda **kw: fp.L0(0.5, W, **kw), np.sqrt(2 * GAMMA * 0.5 * W), False),
        "lq": (lambda **kw: fp.Lq(0.5, 0.5, W, **kw), np.array([_lq_tau(GAMMA * 0.5 * w, 0.5) for w in W]), False),
        "log": (lambda **kw: fp.LogPenalty(0.5, 0.2, **kw), np.full(6, _log_breakpoint(GAMMA * 0.5, 0.2)), False),
        "log_continuous": (lambda **kw: fp.LogPenalty(0.1, 0.5, **kw),
                           np.full(6, _log_breakpoint(GAMMA * 0.1, 0.5)), True),
    }


TIES = sorted(tie_cases())


@pytest.mark.parametrize("name", TIES)
def test_tie_kinks(name):
    """The breakpoint of each nonconvex map (the tie, or for the log penalty
    with gamma lam <= eps^2 the start of its continuous branch) is a kink:
    code 1 within the tolerance, and each side's override is the derivative
    of the one-sided difference from that side. The points sit a relative
    1e-10 on either side (the breakpoints here are computed independently of
    the library's, so not to the last bit), with entries of both signs."""
    make, b, _ = tie_cases()[name]
    op = make()
    _, ctx = op.prox(np.stack([b, -b]), GAMMA, keep=True)
    assert np.all(ctx.kinks(1e-9 * (1 + b.max())) == 1)
    _, far = op.prox(0.5 * b, GAMMA, keep=True)
    assert not far.kinks(1e-3).any()
    h, tol = 1e-7, 1e-8 * (1 + b.max())
    g = np.random.default_rng(5).normal(size=(2, 6))
    for side in (1, -1):
        x = np.stack([b * (1 + side * 1e-10), -b * (1 - side * 1e-10)])  # on the x + side eps piece
        _, c = op.prox(x, GAMMA, keep=True)
        assert np.all(c.kinks(tol) == 1)
        fd = (op.prox(x + side * h, GAMMA) - op.prox(x, GAMMA)) / (side * h)
        jv = op.jvp(c, np.ones_like(x), branch=side, kink_tol=tol)
        np.testing.assert_allclose(jv, fd, rtol=1e-5, atol=1e-6)
        gx, _ = op.vjp(c, g, branch=side, kink_tol=tol)
        assert abs(float((g * jv).sum()) - float(gx.sum())) <= 1e-12 * (1 + abs(float((g * jv).sum())))


# ---- 1D, anisotropic and isotropic TV (since 0.3) ----------------------------


def _tv1d_definition(x, z, gamma, w, tol):
    """Code 1 at entry e when the edge (e, e + 1) has a jump within tol of 0
    and its dual u_e = sum_{i <= e} (x_i - z_i) within tol of gamma w_e."""
    u = np.cumsum(x - z, axis=-1)[..., :-1]
    r = gamma * np.broadcast_to(w, u.shape)
    m = np.zeros(x.shape, np.int8)
    m[..., :-1] = (r > 0) & (np.abs(np.diff(z, axis=-1)) <= tol) & (np.abs(u) >= r - tol)
    return m


def _image_definition(iso, z, q, gamma, w, sw, tol):
    k = q.shape[0]
    img = z.shape[-k:]
    if iso:
        g = np.zeros_like(q)
        for a in range(k):
            sl = [slice(None)] * z.ndim
            sl[z.ndim - k + a] = slice(0, -1)
            g[a][tuple(sl)] = w[a] * np.diff(z, axis=z.ndim - k + a)
        r = gamma * (1.0 if sw is None else sw)
        return ((r > 0) & (np.sqrt((g ** 2).sum(0)) <= tol) & (np.sqrt((q ** 2).sum(0)) >= r - tol)).astype(np.int8)
    m = np.zeros(z.shape, np.int8)
    for a in range(k):
        ax = z.ndim - k + a
        p = -np.cumsum(q[a], axis=ax)
        n = img[a]
        pe = np.take(p, range(n - 1), axis=ax)
        b = 1.0 if sw is None else np.take(sw[a], range(n - 1), axis=a)
        r = gamma * w[a] * b
        bit = (r > 0) & (np.abs(np.diff(z, axis=ax)) <= tol) & (np.abs(pe) >= r - tol)
        pad = [(0, 0)] * z.ndim
        pad[ax] = (0, 1)
        m |= (np.pad(bit, pad) << a).astype(np.int8)
    return m


def test_tv1d_kinks():
    # Two entries 1 apart at gamma = 1/2: the mean, with the dual exactly on its bound.
    _, ctx = fp.TV1D().prox(np.array([[0.0, 1.0], [0.0, 3.0]]), 0.5, keep=True)
    assert ctx.kinks().tolist() == [[1, 0], [0, 0]]
    rng = np.random.default_rng(6)
    x = np.repeat(rng.normal(size=(4, 7)), 5, axis=1) + 0.05 * rng.normal(size=(4, 35))
    for w in (None, rng.uniform(0.5, 1.5, 34) * (rng.random(34) > 0.1)):
        op = fp.TV1D(w) if w is not None else fp.TV1D()
        z, ctx = op.prox(x, 0.3, keep=True)
        ww = 1.0 if w is None else w
        counts = []
        for tol in (0.0, 1e-12, 1e-6, 1e-2):
            m = ctx.kinks(tol)
            np.testing.assert_array_equal(m, _tv1d_definition(x, z, 0.3, ww, tol))
            counts.append(int(m.sum()))
        assert counts == sorted(counts) and counts[-1] > 0
    # Along another axis, in the conjugate's units (tol in units of y), and in float32.
    y = np.ascontiguousarray(x.T)
    _, ca = fp.TV1D().prox(y, 0.3, axis=0, keep=True)
    _, cr = fp.TV1D().prox(x, 0.3, keep=True)
    np.testing.assert_array_equal(ca.kinks(1e-2), cr.kinks(1e-2).T)
    sigma = 2.0  # the kernel solves at the step 1 / sigma on y / sigma
    _, cc = fp.TV1D().prox_conj(x * sigma, sigma, keep=True)
    _, cs = fp.TV1D().prox(x, 1 / sigma, keep=True)
    np.testing.assert_array_equal(cc.kinks(sigma * 1e-2), cs.kinks(1e-2))
    _, c32 = fp.TV1D().prox(x.astype(np.float32), 0.3, keep=True)
    assert c32.kinks(1e-2).dtype == np.int8


@pytest.mark.parametrize("iso", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_image_tv_kinks(iso, weighted):
    """Pixels (isotropic) or edges (anisotropic, bit a for the edge along axis
    a) whose dual is on its bound with no jump; the mask is the definition
    recomputed from the returned z and dual, for any thread count."""
    rng = np.random.default_rng(7)
    shape = (2, 12, 14)
    x = rng.normal(size=shape) + (np.indices(shape[1:]).sum(0) // 4 % 2)
    w = np.array([1.0, 0.7])
    sw = None
    if weighted:
        sw = rng.uniform(0.3, 1.5, shape[1:] if iso else (2, *shape[1:]))
        sw[rng.random(sw.shape) < 0.1] = 0.0
    kw = {} if sw is None else ({"pixel_weights": sw} if iso else {"edge_weights": sw})
    op = (fp.TVIso if iso else fp.TVAniso)(2, w, **kw)
    z, ctx = op.prox(x, 0.4, keep=True, tol=1e-12, max_iter=10 ** 6)
    for tol in (0.0, 1e-8, 1e-3, 3e-2):
        m = ctx.kinks(tol)
        np.testing.assert_array_equal(m, _image_definition(iso, z, ctx.info.dual, 0.4, w, sw, tol))
    m = ctx.kinks(3e-2)
    assert m.any()
    for t in (1, 4):
        _, c2 = op.prox(x, 0.4, keep=True, tol=1e-12, max_iter=10 ** 6, threads=t)
        np.testing.assert_array_equal(c2.kinks(3e-2), m)
    if iso:  # isotropic TV has no branch overrides yet
        with pytest.raises(ValueError):
            op.jvp(ctx, x, branch=1)


def test_image_tv_exact_kinks():
    # One active axis: each row is the 1D TV of (0, 1) at gamma = 1/2, the
    # mean, with its dual exactly on the bound.
    x = np.array([[0.0, 1.0], [0.0, 1.0]])
    _, ca = fp.TVAniso(2, [0.0, 1.0]).prox(x, 0.5, keep=True)
    assert ca.kinks().tolist() == [[2, 0], [2, 0]]  # bit 1: the edge along axis 1
    _, ci = fp.TVIso(2, [0.0, 1.0]).prox(x, 0.5, keep=True)
    assert ci.kinks().tolist() == [[1, 0], [1, 0]]


# ---- branch overrides of 1D and anisotropic TV (since 0.3) ------------------


def _tv_ops():
    """(name, operator, input with exactly biactive edges at gamma = 1/2): rows
    (0, 1) solve to their mean with the dual on its bound; an anisotropic TV
    of identical rows has the same solution in 2D."""
    rows = np.array([[0.0, 1.0, 1.0, 3.0], [2.0, 1.0, 5.0, 4.0]])  # pairs (0,1), (2,1), (5,4)
    return [("tv1d", fp.TV1D(), rows),
            ("aniso_1axis", fp.TVAniso(2, [0.0, 1.0]), rows),
            ("aniso_2d", fp.TVAniso(2), np.array([[0.0, 1.0], [0.0, 1.0]]))]


@pytest.mark.parametrize("name", ["tv1d", "aniso_1axis", "aniso_2d"])
def test_tv_branch_matches_one_sided_differences(name):
    """At a biactive edge side -1 joins its two sides and +1 keeps them
    apart: the derivative along a direction that closes the gap (the edge
    stays joined) or opens it (the edge comes apart), in x and in gamma, is
    that of the one-sided difference."""
    _, op, x = next(c for c in _tv_ops() if c[0] == name)  # built here: nanobind objects held by
    g, tol = 0.5, 1e-9                                     # pytest's parameters outlive the module
    z, ctx = op.prox(x, g, keep=True, tol=1e-14, max_iter=10 ** 5) if name != "tv1d" else op.prox(x, g, keep=True)
    m = ctx.kinks(tol)
    assert m.any()
    kink = m != 0
    # Directions: move each flagged edge's two values towards (close) or away
    # from (open) each other; x's step at an edge (i, i + 1) is the sign of
    # x_{i+1} - x_i.
    step = np.zeros_like(x)
    ax = -1
    d = np.sign(np.diff(x, axis=ax))
    t = np.zeros_like(x)
    t[..., :-1] -= d * (np.take(m, range(x.shape[ax] - 1), axis=ax) != 0)
    t[..., 1:] += d * (np.take(m, range(x.shape[ax] - 1), axis=ax) != 0)
    opening, closing = t, -t
    h = 1e-7
    run = (lambda xx, gg: op.prox(xx, gg, tol=1e-14, max_iter=10 ** 5)) if name != "tv1d" else op.prox
    side = np.where(kink, 1, 0).astype(np.int8)
    for s, dx, dg in ((1, opening, 0.0), (-1, closing, 0.0), (1, 0 * x, -1.0), (-1, 0 * x, 1.0)):
        fd = (run(x + h * dx, g + h * dg) - z) / h
        jv = op.jvp(ctx, dx, dg, branch=s * side, kink_tol=tol)
        np.testing.assert_allclose(jv, fd, rtol=1e-6, atol=1e-6)
        r = np.random.default_rng(8).normal(size=x.shape)
        gx, grads = op.vjp(ctx, r, branch=s * side, kink_tol=tol)
        lhs, rhs = float((r * jv).sum()), float((gx * dx).sum() + grads.gamma * dg)
        assert abs(lhs - rhs) <= 1e-12 * (1 + abs(lhs))
    # The two sides differ, and the override holds for one call only.
    dx = np.random.default_rng(9).normal(size=x.shape)
    jp, jm, j0 = (op.jvp(ctx, dx, branch=b, kink_tol=tol) for b in (side, -side, None))
    assert not np.allclose(jp, jm)
    np.testing.assert_array_equal(op.jvp(ctx, dx), j0)
    assert np.allclose(j0, jp) or np.allclose(j0, jm)


# ---- box, orthant, simplex, l1 ball, group l2 (since 0.3) -----------------


def _one_sided(op, x, d, side, gamma=1.0, tol=1e-12, branch=None):
    """The derivative along d with the override `branch` (default side at
    every flagged entry) against the difference (prox(x + h d) - prox(x)) / h;
    and the VJP's adjointness."""
    _, ctx = op.prox(x, gamma, keep=True)
    b = (side * (ctx.kinks(tol) != 0)).astype(np.int8) if branch is None else branch
    jv = op.jvp(ctx, d, branch=b, kink_tol=tol)
    h = 1e-7
    fd = (op.prox(x + h * d, gamma) - op.prox(x, gamma)) / h
    np.testing.assert_allclose(jv, fd, rtol=1e-6, atol=1e-6)
    g = np.random.default_rng(11).normal(size=x.shape)
    gx, _ = op.vjp(ctx, g, branch=b, kink_tol=tol)
    assert abs(float((g * jv).sum()) - float((gx * d).sum())) <= 1e-12 * (1 + float(np.abs(g).sum()))
    return jv


def test_box_and_orthant_kinks():
    op = fp.Box(-0.5, 0.75)
    x = np.array([-0.5, 0.75, 0.1, -2.0, 3.0])
    _, ctx = op.prox(x, 1.0, keep=True)
    assert ctx.kinks().tolist() == [1, 2, 0, 0, 0]  # 1: on the lower bound, 2: on the upper
    for side in (1, -1):  # x + eps: inside at the lower bound, clipped at the upper
        jv = _one_sided(op, x, side * np.ones_like(x), side)
        assert jv[:2].tolist() == ([1.0, 0.0] if side > 0 else [0.0, -1.0])  # J times dx = side
    op = fp.NonNeg()
    x = np.array([0.0, 1.0, -1.0])
    _, ctx = op.prox(x, 1.0, keep=True)
    assert ctx.kinks().tolist() == [1, 0, 0]
    for side in (1, -1):
        assert _one_sided(op, x, side * np.ones_like(x), side)[0] == (1.0 if side > 0 else 0.0)


def test_simplex_and_l1_ball_ties():
    """Entries exactly at the threshold (x = (3, 2, 1, 1) on the simplex of
    radius 3: tau = 1): +1 puts them on the support, -1 off it, each the
    derivative along a direction that moves them to that side."""
    x = np.array([3.0, 2.0, 1.0, 1.0])
    op = fp.Simplex(3.0)
    _, ctx = op.prox(x, 1.0, keep=True)
    assert ctx.kinks().tolist() == [0, 0, 1, 1]
    tie = np.array([0.0, 0.0, 1.0, 1.0])
    up = _one_sided(op, x, tie, 1)
    np.testing.assert_allclose(up, [-0.5, -0.5, 0.5, 0.5])
    np.testing.assert_allclose(_one_sided(op, x, -tie, -1), 0.0, atol=1e-15)
    # The l1 ball of radius 3: the same with signs, and a row on the sphere.
    op = fp.L1Ball(3.0)
    x = np.array([3.0, -2.0, 1.0, -1.0])
    _, ctx = op.prox(x, 1.0, keep=True)
    assert ctx.kinks().tolist() == [0, 0, 1, 1]
    s = np.sign(x) * (np.abs(x) == 1)
    _one_sided(op, x, s, 1)
    _one_sided(op, x, -s, -1)
    x = np.array([1.0, 2.0, -1.0])  # |x|_1 = 4 = r
    op = fp.L1Ball(4.0)
    _, ctx = op.prox(x, 1.0, keep=True)
    assert ctx.kinks().tolist() == [2, 2, 2]
    first = lambda v: np.array([v, 0, 0], np.int8)  # the side at the first entry decides
    np.testing.assert_allclose(_one_sided(op, x, -x, -1, branch=first(-1)), -x)       # inside
    np.testing.assert_allclose(_one_sided(op, x, x, 1, branch=first(1)), x - np.sign(x) * 4 / 3)


def test_group_l2_kinks():
    """A group whose norm equals its threshold (0.6, 0.8) at gamma lambda = 1:
    +1 keeps it live (its shrink taken at x), -1 kills it; the side at the
    group's first entry decides."""
    op = fp.GroupL2(1.0, groups=np.array([0, 1, 0, 1]))
    x = np.array([0.6, 3.0, 0.8, 4.0])
    _, ctx = op.prox(x, 1.0, keep=True)
    assert ctx.kinks().tolist() == [1, 0, 1, 0]
    g0 = np.array([0.6, 0.0, 0.8, 0.0])
    np.testing.assert_allclose(_one_sided(op, x, g0, 1), g0, atol=1e-12)
    np.testing.assert_allclose(_one_sided(op, x, -g0, -1), 0.0, atol=1e-15)
    # The default is the killed group, as before.
    np.testing.assert_allclose(op.jvp(ctx, g0), 0.0, atol=1e-15)


# ---- tree TV: kinks are edges (since 0.4) ----------------------------------


def test_tree_tv_edge_kinks():
    """x = (2, -1, 5) on the chain 0-1-2 with gamma w = 1: nodes 0 and 1 meet
    at 1, (2 - 1 + 1) / 2 with the push from the jump on edge (1, 2), and the
    flow on edge (0, 1) is x_0 - z_0 = 1, at its bound: a kink. -1 joins the kink's ends (moving them apart is not first order),
    +1 keeps them apart; masks and branches are shaped (..., m)."""
    op = fp.TreeTV(3, np.array([[0, 1], [1, 2]]))
    x = np.array([2.0, -1.0, 5.0])
    z, ctx = op.prox(x, 1.0, keep=True)
    np.testing.assert_allclose(z, [1.0, 1.0, 4.0])
    assert ctx.kinks(1e-12).tolist() == [1, 0]
    apart, joined = np.array([1.0, -1.0, 0.0]), np.array([-1.0, 1.0, 0.0])
    np.testing.assert_allclose(_one_sided(op, x, apart, 1), apart, atol=1e-12)
    np.testing.assert_allclose(_one_sided(op, x, joined, -1), 0.0, atol=1e-12)
    # The default joins, as before; a cleared override restores it.
    dz0 = op.jvp(ctx, apart)
    np.testing.assert_allclose(dz0, 0.0, atol=1e-15)
    op.jvp(ctx, apart, branch=np.array([1, 0], np.int8), kink_tol=1e-12)
    np.testing.assert_array_equal(op.jvp(ctx, apart), dz0)
    # Batched, along the edges: a second row without kinks.
    X = np.stack([x, [3.0, 0.0, -3.0]])
    _, ctx = op.prox(X, 1.0, keep=True)
    assert ctx.kinks(1e-12).tolist() == [[1, 0], [0, 0]]
    b = np.array([[1, 0], [0, 0]], np.int8)
    jv = op.jvp(ctx, np.stack([apart, apart]), branch=b, kink_tol=1e-12)
    np.testing.assert_allclose(jv[0], apart, atol=1e-12)
    with pytest.raises(ValueError):
        op.jvp(ctx, np.stack([apart, apart]), branch=np.zeros((2, 3), np.int8))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tree_tv_restore_is_default(dtype):
    """A cleared override gives back the prox call's own derivatives, bit for
    bit, also in float32 where the structure is rebuilt from the rounded z."""
    rng = np.random.default_rng(5)
    n = 200
    par = rng.integers(0, np.arange(1, n))  # a random tree
    op = fp.TreeTV(n, np.stack([np.arange(1, n), par], 1))
    X = (np.round(2 * rng.normal(size=(3, n))) / 2).astype(dtype)  # halves at gamma 1/2: many kinks
    _, ctx = op.prox(X, 0.5, keep=True)
    d = rng.normal(size=X.shape).astype(dtype)
    before = op.jvp(ctx, d)
    k = ctx.kinks(1e-5)
    assert (k != 0).sum() > 10
    split = op.jvp(ctx, d, branch=np.where(k != 0, 1, 0).astype(np.int8), kink_tol=1e-5)
    assert np.abs(split - before).max() > 0.1
    np.testing.assert_array_equal(op.jvp(ctx, d), before)


# ---- graph TV: kinks on the solver's flow (since 0.4) ----------------------


def test_graph_tv_cycle_kinks():
    """x = (1, 1, -1, -1) on the 4-cycle 0-1-2-3-0 at gamma w = 1: z = 0 and
    the only feasible flow saturates the two edges across {0, 1} | {2, 3},
    (1, 2) and (3, 0). Splitting needs both kept apart; one alone leaves the
    region joined through the other."""
    op = fp.GraphTV(4, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    x = np.array([1.0, 1.0, -1.0, -1.0])
    z, ctx = op.prox(x, 1.0, keep=True)
    np.testing.assert_allclose(z, 0.0, atol=1e-14)
    assert ctx.kinks(1e-10).tolist() == [0, 1, 0, 1]
    np.testing.assert_allclose(_one_sided(op, x, x, 1, tol=1e-10), x, atol=1e-10)
    np.testing.assert_allclose(_one_sided(op, x, -x, -1, tol=1e-10), 0.0, atol=1e-12)
    one = op.jvp(ctx, x, branch=np.array([0, 1, 0, 0], np.int8), kink_tol=1e-10)
    np.testing.assert_allclose(one, 0.0, atol=1e-12)


def test_graph_tv_matches_tree_tv_on_a_forest():
    rng = np.random.default_rng(8)
    n = 120
    par = rng.integers(0, np.arange(1, n))
    edges = np.stack([np.arange(1, n), par], 1)
    g, t = fp.GraphTV(n, edges), fp.TreeTV(n, edges)
    X = np.round(2 * rng.normal(size=(2, n))) / 2
    _, cg = g.prox(X, 0.5, keep=True)
    _, ct = t.prox(X, 0.5, keep=True)
    kg, kt = cg.kinks(1e-9), ct.kinks(1e-9)
    np.testing.assert_array_equal(kg, kt)
    assert (kg != 0).sum() > 10
    d = rng.normal(size=X.shape)
    b = np.where(kg != 0, rng.choice([-1, 1], size=kg.shape), 0).astype(np.int8)
    np.testing.assert_allclose(g.jvp(cg, d, branch=b, kink_tol=1e-9), t.jvp(ct, d, branch=b, kink_tol=1e-9),
                               atol=1e-10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_graph_tv_grid_overrides(dtype):
    """On a 12 x 12 grid with many kinks: overrides change the derivative, the
    VJP stays the adjoint of the JVP, and clearing gives back the default bit
    for bit."""
    rng = np.random.default_rng(9)
    s = 12
    idx = np.arange(s * s).reshape(s, s)
    edges = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                            np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])
    op = fp.GraphTV(s * s, edges)
    X = (np.round(2 * rng.normal(size=(2, s * s))) / 2).astype(dtype)
    _, ctx = op.prox(X, 0.5, keep=True, tol=1e-12)
    d = rng.normal(size=X.shape).astype(dtype)
    g = rng.normal(size=X.shape).astype(dtype)
    before = op.jvp(ctx, d)
    k = ctx.kinks(1e-5)
    assert k.shape == (2, len(edges)) and (k != 0).sum() > 10
    b = np.where(k != 0, 1, 0).astype(np.int8)
    jv = op.jvp(ctx, d, branch=b, kink_tol=1e-5)
    gx, _ = op.vjp(ctx, g, branch=b, kink_tol=1e-5)
    assert np.abs(jv - before).max() > 0.1
    lhs, rhs = float((g.astype(float) * jv).sum()), float((gx.astype(float) * d).sum())
    assert abs(lhs - rhs) <= (1e-4 if dtype == np.float32 else 1e-10) * (1 + abs(lhs))
    np.testing.assert_array_equal(op.jvp(ctx, d), before)


def zero_threshold_cases():
    """name: (operator, entries of x = [0, 0.5] whose threshold is zero, the
    identity's Jacobian there). At GAMMA every threshold that is not zero
    exceeds 0.5."""
    return {
        "soft_w0": (fp.Soft(1.0, [0.0, 1.0]), [0], 1.0),
        "soft_lam0": (fp.Soft(0.0), [0, 1], 1.0),
        "elastic_net_l1_0": (fp.ElasticNet(0.0, 0.5), [0, 1], 1.0 / (1.0 + GAMMA * 0.5)),
        "l1_box_lam0": (fp.L1Box(0.0, -1.0, 1.0), [0, 1], 1.0),
        "l0_lam0": (fp.L0(0.0), [0, 1], 1.0),
        "lq_lam0": (fp.Lq(0.0, 0.5), [0, 1], 1.0),
        "scad_lam0": (fp.SCAD(0.0, 3.7), [0, 1], 1.0),
        "mcp_lam0": (fp.MCP(0.0, 3.0), [0, 1], 1.0),
        "scad_w0": (fp.SCAD(1.0, 3.7, [0.0, 1.0]), [0], 1.0),
        "mcp_w0": (fp.MCP(1.0, 3.0, [0.0, 1.0]), [0], 1.0),
        "log_lam0": (fp.LogPenalty(0.0, 1.0), [0, 1], 1.0),
    }


@pytest.mark.parametrize("name", sorted(zero_threshold_cases()))
def test_zero_threshold_is_identity(name):
    """A zero threshold is not a kink: the map is the identity there, also at
    x = 0, so its Jacobian is 1 (1/(1 + gamma l2) for the elastic net) and z
    does not move with gamma or the parameters at x = 0."""
    op, idx, j = zero_threshold_cases()[name]
    x = np.array([0.0, 0.5])
    z, ctx = op.prox(x, GAMMA, keep=True)
    assert np.all(ctx.jacobian_diag()[idx] == j)
    if not isinstance(op, fp.ElasticNet):  # no kink masks
        assert np.all(ctx.kinks(1e-12)[idx] == 0)
    # Adjointness of (dx, dgamma, dtheta) -> dz.
    rng = np.random.default_rng(3)
    dx, g = rng.standard_normal(2), rng.standard_normal(2)
    dp = rng.standard_normal(op.nparams)
    dz = op.jvp(ctx, dx, 0.37, dp)
    gx, grads = op.vjp(ctx, g)
    rhs = gx @ dx + 0.37 * grads.gamma + grads.params @ dp
    assert abs(g @ dz - rhs) <= 1e-14 * (1 + abs(g @ dz))
    # At x = 0, z = 0 for every step and parameter.
    dz = op.jvp(ctx, np.zeros(2), 1.0, np.ones(op.nparams))
    assert dz[0] == 0.0
    e0 = np.array([1.0, 0.0])
    gx, grads = op.vjp(ctx, e0)
    assert gx[0] == j and grads.gamma == 0.0 and np.all(grads.params == 0.0)


def test_zero_threshold_groups_are_identity():
    # The unpenalised group of GroupL2 at x_g = 0, beside a live group.
    op = fp.GroupL2(1.0, [0, 0, 1, 1], [0.0, 1.0])
    _, ctx = op.prox(np.array([0.0, 0.0, 3.0, 4.0]), 1.0, keep=True)
    np.testing.assert_allclose(op.jvp(ctx, np.ones(4)), [1.0, 1.0, 0.968, 1.024], rtol=1e-14)
    gx, grads = op.vjp(ctx, np.array([1.0, -2.0, 0.0, 0.0]))
    assert list(gx) == [1.0, -2.0, 0.0, 0.0] and grads.gamma == 0.0
    assert grads.group_weights[0] == 0.0
    # TreeL2 with l1 = 0 and a zero group weight is the identity at x = 0.
    op = fp.TreeL2([[0, 1]], [0.0], 0.0)
    _, ctx = op.prox(np.zeros(2), GAMMA, keep=True)
    assert list(op.jvp(ctx, np.ones(2))) == [1.0, 1.0]
    gx, grads = op.vjp(ctx, np.array([1.0, -2.0]))
    assert list(gx) == [1.0, -2.0] and grads.gamma == 0.0 and np.all(grads.params == 0.0)
