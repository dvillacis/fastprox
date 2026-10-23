# Array-likes at the Python boundary (audit P5): float32 is kept for
# non-NumPy inputs, other real types become float64, complex raises.
#
# SPDX-License-Identifier: MIT
import array

import numpy as np
import pytest

import fastprox as fp

X32 = np.random.default_rng(0).normal(size=50).astype(np.float32)


def _check(like):
    z = fp.tv1d(like, 0.5)
    assert z.dtype == np.float32
    assert np.array_equal(z, fp.tv1d(X32, 0.5))


def test_buffers_keep_float32():
    _check(memoryview(X32))
    _check(array.array("f", X32.tolist()))


def test_torch_keeps_float32():
    torch = pytest.importorskip("torch")
    _check(torch.from_numpy(X32.copy()))


def test_jax_keeps_float32():
    jnp = pytest.importorskip("jax.numpy")
    _check(jnp.asarray(X32))


def test_lists_and_complex():
    assert fp.tv1d([1, 2, 3], 0.5).dtype == np.float64
    assert fp.tv1d([True, False, True], 0.5).dtype == np.float64
    with pytest.raises(TypeError, match="complex"):
        fp.tv1d([1 + 1j, 2.0], 0.5)


def test_pyproximal_inputs():
    pytest.importorskip("pyproximal")
    from fastprox.pyproximal import ProxOp
    op = ProxOp(fp.TV1D(0.5))
    assert op.prox(np.arange(5, dtype=np.float16), 1.0).dtype == np.float64
    with pytest.raises(TypeError, match="complex"):
        op.prox(np.ones(5, complex), 1.0)
