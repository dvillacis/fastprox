"""Image TV far from unit scale (audit T5): the solvers rescale an image by a
power of two when its largest entry is outside [2^-64, 2^65), so that the
squares in the stopping target and the gap neither overflow (|x| ~ 1e154)
nor underflow. TV is positively homogeneous, so prox(c x, c gamma) =
c prox(x, gamma)."""

import numpy as np
import pytest

import fastprox as fp

OPS = {
    "aniso2": (fp.TVAniso, 2, (16, 16)),
    "aniso3": (fp.TVAniso, 3, (4, 8, 8)),
    "iso2": (fp.TVIso, 2, (16, 16)),
    "iso3": (fp.TVIso, 3, (4, 8, 8)),
}
KW = dict(tol=1e-12, max_iter=100000)


def setup(name):
    cls, k, shape = OPS[name]
    return cls(k), np.random.default_rng(2).standard_normal(shape)


@pytest.mark.parametrize("name", sorted(OPS))
def test_scaled_images(name):
    op, x = setup(name)
    zref, _ = op.solve(x, 0.5, **KW)
    out = {}
    for k in (300, 600, 900, -300, -600, -1000):  # 900: gamma w in the step range, <= 2^960
        z, info = op.solve(x * 2.0 ** k, 0.5 * 2.0 ** k, **KW)
        # The gap scales by 4^k: +inf past the largest double, never NaN.
        assert info.converged and info.gap >= 0.0, (k, info.gap)
        if abs(k) <= 300:
            assert np.isfinite(info.gap)
        np.testing.assert_allclose(z / 2.0 ** k, zref, rtol=0, atol=1e-10)
        out[k] = z / 2.0 ** k
    # Both are solved on the same rescaled image.
    np.testing.assert_array_equal(out[600], out[900])
    np.testing.assert_array_equal(out[-600], out[-1000])


@pytest.mark.parametrize("name", sorted(OPS))
def test_conjugate_at_tiny_sigma(name):
    # The conjugate of a positively homogeneous g is the indicator of a set,
    # so prox_conj is the same projection for every sigma; at 1e-156 the
    # image the solver sees, y / sigma, is near 1e156.
    op, x = setup(name)
    ref = op.prox_conj(x, 1e-3, **KW)
    np.testing.assert_allclose(op.prox_conj(x, 1e-156, **KW), ref, rtol=0, atol=1e-10)


def test_derivatives_at_scale():
    # The saved structure is scale-free, but for isotropic TV's z, dual and
    # curvature floor, which scale back with the image: J is the same.
    rng = np.random.default_rng(3)
    for name in sorted(OPS):
        op, x = setup(name)
        dx = rng.standard_normal(x.shape)
        _, c1 = op.prox(x, 0.5, keep=True, **KW)
        _, c2 = op.prox(x * 2.0 ** 700, 0.5 * 2.0 ** 700, keep=True, **KW)
        np.testing.assert_allclose(op.jvp(c2, dx), op.jvp(c1, dx), rtol=0, atol=1e-8)
        # In x and gamma the vjp is the same; in the weights it scales with z.
        g1, r1 = op.vjp(c1, dx)
        g2, r2 = op.vjp(c2, dx)
        np.testing.assert_allclose(g2, g1, rtol=0, atol=1e-8)
        np.testing.assert_allclose(r2.gamma, r1.gamma, rtol=1e-10)
        np.testing.assert_allclose(r2.params / 2.0 ** 700, r1.params, rtol=1e-10)


def test_huge_aniso_converges_quickly():
    # The gap was NaN, so the solve ran to max_iter (100000) for a right z.
    # It is now +inf (1e600 times that of the unit-scale problem), not NaN.
    op, x = setup("aniso2")
    _, info = op.solve(x * 1e280, 1e280, **KW)  # 1e280: in the step range
    assert info.converged and info.gap >= 0.0 and info.iters < 100


def _edges(name, n):
    if name == "tree_star":
        return np.stack([np.zeros(n - 1, int), np.arange(1, n)], 1)
    return np.stack([np.arange(n - 1), np.arange(1, n)], 1)


@pytest.mark.parametrize("name", sorted(OPS) + ["graph", "tree_path", "tree_star"])
def test_tiny_image_heavy_weights(name):
    """An image near the smallest normal double scales up by 2^1020; that
    checked gamma, not gamma w, so heavy weights overflowed there (fuzz,
    2026-10-23): TreeTV returned inf and NaN, GraphTV a NaN gap, TVAniso
    stopped unconverged. The scale-up now stops where gamma w reaches the
    step range. The step is far above the image's variation, so the prox is
    the image's mean."""
    if name in OPS:
        cls, k, shape = OPS[name]
        op = cls(k, weights=[64.0] * k)
    else:
        shape = (16,)
        op = (fp.GraphTV if name == "graph" else fp.TreeTV)(16, _edges(name, 16), weights=64.0)
    x = np.random.default_rng(4).standard_normal(shape) * 2.0**-1021
    z, info = op.solve(x, 1.0, **KW)
    assert info.converged and np.isfinite(info.gap) and info.gap >= 0.0
    np.testing.assert_allclose(z, np.full(shape, x.mean()), rtol=1e-9, atol=0)


@pytest.mark.parametrize("name", sorted(OPS) + ["graph", "tree_path", "tree_star"])
def test_tiny_image_light_weights(name):
    """The converse (fuzz, 2026-10-24): with every weight below 1, gamma
    itself bounds the scale-up. A tree TV edge of weight 2.7e-315 let an
    image of 5e-312 scale up by 2^1034, where gamma = 1.9 overflowed: inf.
    Here gamma = 16 with weights 2^-1026 on an image near 2^-1021, so the
    capacities match the image: the result is c times that of the same
    problem at unit scale, x / c with weights 16 w / c."""
    c, w, gamma = 2.0**-1021, 2.0**-1026, 16.0

    def make(weight):
        if name in OPS:
            cls, k, shape = OPS[name]
            return cls(k, weights=[weight] * k), shape
        return (fp.GraphTV if name == "graph" else fp.TreeTV)(16, _edges(name, 16), weights=weight), (16,)

    op, shape = make(w)
    unit, _ = make(gamma * w / c)
    x = np.random.default_rng(5).standard_normal(shape) * c
    z, info = op.solve(x, gamma, **KW)
    assert info.converged and np.isfinite(info.gap) and info.gap >= 0.0
    z1, _ = unit.solve(x / c, 1.0, **KW)
    np.testing.assert_allclose(z / c, z1, rtol=0, atol=1e-9 * np.abs(x / c).max())


def test_tree_tv_fuzz_light_edge():
    # The fuzz input: one edge of weight 2.7e-315 between two zeros, and a
    # subnormal off the tree. Untouched entries leave the conjugate at 0.
    x = np.zeros(31)
    x[18] = float.fromhex("0x1.0000000021p-1034")
    op = fp.TreeTV(31, [[1, 0]], [2.7461121994333263e-315])
    np.testing.assert_array_equal(op.prox_conj(x, 0.53125), np.zeros(31))
    np.testing.assert_array_equal(op.prox(x, 1.0 / 0.53125), x)


# 1D TV on rows beyond 2^960 (since 1.0.6). The local sums of the taut string
# and of Condat's algorithm reach n max |x|, and max x - min x alone overflows
# near 1e308: the taut string gave inf on 2.4e307 next to 1.7e308 at
# gamma = 2.5 (fuzz seed 103550005, 2026-11-02). Such rows now run scaled by
# a power of two, which commutes with rounding: the bits are those of the
# same row scaled by hand, and rows below 2^960 are not touched.
TV1D_METHODS = ("auto", "condat", "taut_string")


@pytest.mark.parametrize("method", TV1D_METHODS)
def test_tv1d_overflow_row(method):
    op = fp.TV1D(method=method)
    for x in ([2.4e307, 1.7e308], [2.4e307, 1.7e308, -1.7e308, 1.0e308, 2.4e307]):
        x = np.array(x)
        # gamma = 2.5 is far below an ulp of these entries.
        np.testing.assert_array_equal(op.prox(x, 2.5), x)
        np.testing.assert_array_equal(op.prox(x, 2.5, keep=True)[0], x)


def huge_rows(rng, count):
    for t in range(count):
        n = int(rng.integers(2, 60))
        e = int(rng.integers(961, 1024))
        if t % 2:  # entries of every size up to 2^e
            y = rng.standard_normal(n) * rng.choice([1.0, 10.0, 1e3], n)
            y *= np.ldexp(rng.uniform(0.5, 0.999), e) / np.abs(y).max()
            g = float(np.ldexp(rng.uniform(0.5, 1.0), int(rng.integers(900, 958))))
        else:  # an offset near 2^e, with variations that TV moves
            s = int(rng.integers(30, 60))
            y = np.ldexp(rng.uniform(0.5, 0.9) * rng.choice([-1, 1]), e) + np.ldexp(rng.standard_normal(n), e - s)
            g = float(np.ldexp(rng.uniform(0.1, 3.0), min(e - s, 956)))
        yield y, g


@pytest.mark.parametrize("method", TV1D_METHODS)
@pytest.mark.parametrize("weights", ["none", "positive", "zeros"])
def test_tv1d_huge_rows_bitwise(method, weights):
    rng = np.random.default_rng(["none", "positive", "zeros"].index(weights))
    moved = 0
    for y, g in huge_rows(rng, 150):
        n = y.size
        w = None
        if weights != "none":
            w = rng.uniform(0.1, 2.0, n - 1)
            if weights == "zeros":
                w[rng.random(n - 1) < 0.2] = 0.0
        op = fp.TV1D(weights=w, method=method)
        z = op.prox(y, g)
        assert np.isfinite(z).all()
        for k in (64, 300):  # below 2^960 once scaled by hand
            zh = np.ldexp(op.prox(np.ldexp(y, -k), np.ldexp(g, -k)), k)
            assert z.tobytes() == zh.tobytes(), (method, n, k)
        zk, _ = op.prox(y, g, keep=True)
        assert zk.tobytes() == z.tobytes()
        moved += not np.array_equal(z, y)
    assert moved > 75  # TV moves at least half of them


@pytest.mark.parametrize("method", TV1D_METHODS)
def test_tv1d_rows_at_dbl_max(method):
    M = np.finfo(np.float64).max
    rng = np.random.default_rng(7)
    for t in range(300):
        n = int(rng.integers(2, 40))
        y = rng.choice([M, -M, 0.9 * M, -0.9 * M, 1e307, 0.0], n) if t % 2 else M * rng.uniform(-1, 1, n)
        g = float(np.ldexp(rng.uniform(0.5, 1.0), int(rng.integers(800, 959))))
        w = None if t % 3 else rng.uniform(0.0, 1.0, n - 1)
        z = fp.TV1D(weights=w, method=method).prox(y, g)
        # Finite, and within [min y, max y] but for an ulp of max |y|.
        tol = np.spacing(np.abs(y).max())
        assert np.isfinite(z).all() and z.min() >= y.min() - tol and z.max() <= y.max() + tol
