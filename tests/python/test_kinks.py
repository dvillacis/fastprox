"""Kink masks and branch overrides of the elementwise maps with kinks (soft
thresholding, l1 over a box, SCAD, MCP): codes at exact and near kinks,
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
    for other in (fp.Huber(0.5, 0.3), fp.L0(0.3)):
        _, c2 = other.prox(x, 0.5, keep=True)
        with pytest.raises(ValueError):
            c2.kinks()
        with pytest.raises(ValueError):
            other.jvp(c2, x, branch=1)
