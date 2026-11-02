"""The l2 ball's short rows in blocks (since 1.0.5).

Without ``keep``, conjugate or steps, the prox of ``L2Ball`` on rows shorter
than 8 runs whole runs of rows per call (``fp_op::prox_block``, with a loop
over rows of 2 that the compiler vectorizes) instead of one kernel call per
row. With ``keep=True`` the driver still goes row by row, so the two must agree
bit for bit: on every layout, dtype, radius, thread count and in place, and on
the rows that leave the fast loop (a zero, tiny, huge, NaN or infinite row)."""

import numpy as np
import pytest

import fastprox as fp


def awkward(x, rng, tiny, huge, tinier):
    """x with rows of every kind along axis -1: inside, on the sphere, zero,
    tiny, huge and NaN. In float64 the tiny rows' sum of squares is below
    2^-968 and the huge rows' overflows (both leave the fast loop); float32's
    range keeps every finite row's sum, taken in double, in range."""
    v = x.reshape(-1, x.shape[-1])
    m = v.shape[0]
    kinds = rng.integers(0, 8, m)
    v[kinds == 1] *= 1e-3
    v[kinds == 2] = 0.0
    v[kinds == 3] *= tiny
    v[kinds == 4] *= huge
    v[kinds == 5] *= tinier
    nan = np.flatnonzero(kinds == 6)
    v[nan, 0] = np.nan
    on = np.flatnonzero(kinds == 7)
    v[on] /= np.linalg.norm(v[on], axis=1, keepdims=True)
    return x


def rows(rng, batch, n, dtype):
    f = (1e-170, 1e160, 1e-300) if dtype == np.float64 else (1e-30, 1e37, 1e-44)
    return awkward(rng.normal(size=batch + (n,)), rng, *f).astype(dtype)


def views(x):
    """(name, array, axis) for x (batch..., n): the axis last, first and in
    the middle, transposed, reversed and stepped."""
    yield "last", x, -1
    yield "first", np.ascontiguousarray(np.moveaxis(x, -1, 0)), 0
    yield "middle", np.ascontiguousarray(np.moveaxis(x, -1, 1)), 1
    yield "transposed", np.moveaxis(x, -1, 0).T, -1
    yield "reversed", np.moveaxis(x, -1, 0)[::-1, ::-1, ::-1], 0
    yield "stepped", np.moveaxis(x, -1, 0)[:, ::2, 1::3], 0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 8])
@pytest.mark.parametrize("per_row", [False, True])
def test_blocks_have_the_rows_bits(dtype, n, per_row):
    rng = np.random.default_rng(n)
    x = rows(rng, (24, 33), n, dtype)
    for name, v, axis in views(x):
        batch = np.delete(v.shape, axis % v.ndim)
        op = fp.L2Ball(rng.uniform(0.0, 2.0, batch) if per_row else 0.9)
        z = op.prox(v, axis=axis, threads=1)
        zr, _ = op.prox(v, axis=axis, keep=True, threads=1)
        assert z.dtype == v.dtype
        np.testing.assert_array_equal(z, zr, err_msg=name)
        # In place, and into an output of another layout.
        y = v.copy(order="K")
        op.prox(y, axis=axis, out=y)
        np.testing.assert_array_equal(y, zr, err_msg=name)
        o = np.empty(v.shape[::-1], v.dtype).T
        op.prox(v, axis=axis, out=o)
        np.testing.assert_array_equal(o, zr, err_msg=name)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("shape,axis", [((2, 300, 301), 0), ((300, 301, 2), -1), ((97, 3, 701), 1)])
def test_threads_bitwise(dtype, shape, axis):
    """Above the serial threshold (2^17 entries) the runs are cut at the
    pieces' bounds; the bits do not depend on where."""
    rng = np.random.default_rng(1)
    x = rows(rng, tuple(np.delete(shape, axis % 3)), shape[axis], dtype)
    x = np.ascontiguousarray(np.moveaxis(x, -1, axis))
    for op in (fp.L2Ball(1.1), fp.L2Ball(rng.uniform(0.5, 1.5, np.delete(shape, axis % 3)))):
        zs = [op.prox(x, axis=axis, threads=t) for t in (1, 2, 3, 8)]
        zr, _ = op.prox(x, axis=axis, keep=True, threads=1)
        for z in zs:
            np.testing.assert_array_equal(z, zr)


def test_projection_formula():
    """The pixelwise projection of a TV dual, s r / max(r, |s|), bit for bit,
    in both layouts of the gradient field."""
    rng = np.random.default_rng(2)
    s = rng.normal(size=(2, 64, 80))
    norms = np.sqrt(s[0] ** 2 + s[1] ** 2)
    for r in (0.7, rng.uniform(0.1, 1.5, (64, 80))):
        want = s * (r / np.maximum(r, norms))
        np.testing.assert_array_equal(fp.project_l2_ball(s, r, axis=0), want)
        sl = np.ascontiguousarray(np.moveaxis(s, 0, -1))
        np.testing.assert_array_equal(fp.project_l2_ball(sl, r), np.moveaxis(want, 0, -1))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("axis", [0, -1])
def test_infinite_row(dtype, axis):
    """An infinite entry fails the call (FP_EINVAL), as row by row."""
    x = np.random.default_rng(3).normal(size=(2, 50, 40)).astype(dtype)
    x[1, 7, 9] = -np.inf
    if axis == -1:
        x = np.ascontiguousarray(np.moveaxis(x, 0, -1))
    op = fp.L2Ball(0.8)
    for keep in (False, True):
        with pytest.raises(ValueError):
            op.prox(x, axis=axis, keep=keep)
