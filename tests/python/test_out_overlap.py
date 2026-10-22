"""Outputs that overlap their input (audit C1, P2). The C library accepts an
output that is exactly its input and rejects any other overlap with
FP_EINVAL (since 1.0); before, such calls returned wrong results silently,
some depending on the thread count. The Python package copies an
overlapping input first, so every view below gives the result of a fresh
output, bit for bit."""

import numpy as np
import pytest

import fastprox as fp
from test_operators import CASES, N, NAMES

IMAGE = {
    "tv_aniso": fp.TVAniso(2),
    "tv_iso": fp.TVIso(2),
}


def views(buf):
    """name: (x, out) as views of buf, a (2N, 2N) square buffer."""
    a = buf[:N, :N]
    return {
        "transpose": (a, a.T),
        "reversed": (a, a[::-1]),
        "reversed_cols": (a, a[:, ::-1]),
        "shifted": (buf[1:N + 1, :N], buf[:N, :N]),
        "interleaved": (buf[:2 * N:2, :N], buf[1:2 * N:2, :N]),
        "same_view": (a, a.view()),
    }


VIEWS = sorted(views(np.zeros((2 * N, 2 * N))))


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("name", NAMES)
def test_prox_into_an_overlapping_out(name, view):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    buf = np.random.default_rng(3).normal(size=(2 * N, 2 * N))
    x, out = views(buf)[view]
    ref = op.prox(x.copy(), gamma)
    res = op.prox(x, gamma, out=out)
    assert res is out
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("name", sorted(IMAGE))
def test_image_prox_into_an_overlapping_out(name, view):
    op = IMAGE[name]
    buf = np.random.default_rng(4).normal(size=(2 * N, 2 * N))
    x, out = views(buf)[view]
    ref = op.prox(x.copy(), 0.3, tol=1e-10)
    op.prox(x, 0.3, out=out, tol=1e-10)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("name", ["soft", "tv1d", "group_l2"])
def test_derivatives_into_an_overlapping_out(name):
    build, theta, _, gamma = CASES[name][:4]
    op = build(theta)
    rng = np.random.default_rng(5)
    _, ctx = op.prox(rng.normal(size=(N, N)), gamma, keep=True)
    t = rng.normal(size=(N, N))
    ref_j, ref_v = op.jvp(ctx, t.copy()), op.vjp(ctx, t.copy())[0]
    for layout in (lambda b: b[::-1], lambda b: b.T, lambda b: b):
        b = t.copy()
        out = layout(b)
        op.jvp(ctx, b, out=out)  # dx and its output share memory
        np.testing.assert_array_equal(out, ref_j)
        b = t.copy()
        out = layout(b)
        op.vjp(ctx, b, out=out)  # g and its output share memory
        np.testing.assert_array_equal(out, ref_v)


def test_one_shot_and_per_row_steps():
    x = np.random.default_rng(6).normal(size=(N, N))
    ref = fp.soft_threshold(x, 0.4)
    y = x.copy()
    fp.soft_threshold(y, 0.4, out=y.T)
    np.testing.assert_array_equal(y.T, ref)
    # Per-row steps that are a view of out: out's first row holds them.
    op = fp.Soft(0.4)
    buf = np.empty((N, N))
    buf[0] = np.linspace(0.2, 1.5, N)
    ref = op.prox(x, buf[0].copy())
    op.prox(x, buf[0], out=buf)
    np.testing.assert_array_equal(buf, ref)


def test_bad_outputs():
    op = fp.Soft(0.4)
    x = np.random.default_rng(8).normal(size=(N, N))
    ro = np.empty_like(x)
    ro.flags.writeable = False
    zero = np.lib.stride_tricks.as_strided(np.empty(N), shape=(N, N), strides=(0, 8))
    for out, msg in [(ro, "writeable"), (np.empty((N, N), np.float32), "dtype"), (zero, "zero strides")]:
        with pytest.raises(ValueError, match=msg):
            op.prox(x, out=out)
    _, ctx = op.prox(x, keep=True)
    for f in (lambda o: op.jvp(ctx, x, out=o), lambda o: op.vjp(ctx, x, out=o)):
        with pytest.raises(ValueError, match="writeable"):
            f(ro)
        with pytest.raises(ValueError, match="zero strides"):
            f(zero)
