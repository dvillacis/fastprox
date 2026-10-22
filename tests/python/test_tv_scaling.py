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
