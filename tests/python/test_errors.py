# Clear errors at the Python boundary (audit P7).
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

import fastprox as fp

X = np.random.default_rng(0).normal(size=(6, 50))


@pytest.mark.parametrize("dtype", ["float16", "int32", "complex64"])
def test_torch_dtypes(dtype):
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    x = torch.randn(4, 20).to(getattr(torch, dtype))
    with pytest.raises(TypeError, match="float32 and float64 tensors"):
        fpt.prox(fp.TV1D(0.6), x, 0.5)
    with pytest.raises(TypeError, match="float32 and float64 tensors"):
        fpt.tv_iso(x, 0.5)


def test_out_axis_threads():
    op = fp.TV1D(0.6)
    ro = np.empty_like(X)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        op.prox(X, 0.5, out=ro)
    with pytest.raises(ValueError, match="writeable"):
        op.prox(X, 0.5, out=X.tolist())
    with pytest.raises(TypeError, match="integers"):
        op.prox(X, 0.5, axis=1.0)
    with pytest.raises(ValueError, match="threads must be >= 0"):
        op.prox(X, 0.5, threads=-1)
    with pytest.raises(ValueError, match="threads must be >= 0"):
        fp.TVIso(2).prox(X, 0.5, threads=-1)


def test_one_shot_functions():
    with pytest.raises(TypeError, match="scalar gamma"):
        fp.tv1d(X, np.full(6, 0.5))
    with pytest.raises(ValueError, match="axis 5 out of range"):
        fp.tv1d(X, 0.5, axis=5)


def test_branch_values_and_c_names():
    op = fp.Soft(0.4)
    _, c = op.prox(X, 1.0, keep=True)
    with pytest.raises(ValueError, match="branch entries must be -1, 0 or 1"):
        op.jvp(c, X, branch=2)
    # Errors from the C library name the entry point.
    with pytest.raises(ValueError, match="^fp_prox: "):
        fp.SCAD(0.5, 1.5).prox(X, 10.0)
