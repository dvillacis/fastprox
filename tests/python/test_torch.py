"""The PyTorch wrappers: gradcheck in reverse and forward mode through the
library's derivatives, agreement with the NumPy API, dtypes, devices and
errors."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import fastprox as fp  # noqa: E402
import fastprox.torch as fpt  # noqa: E402

from interop_cases import cases  # noqa: E402

CASES = cases()
IDS = [c[0] for c in CASES]


def _t(a, grad=True):
    return torch.tensor(np.asarray(a), dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_gradcheck(case):
    _, cls, const, params, x, step, conj, call = case
    f = fpt.prox_conj if conj else fpt.prox
    names = list(params)

    def fn(x, step, *vals):
        return f(cls, x, step, **const, **dict(zip(names, vals)), **call)

    args = (_t(x), _t(step), *[_t(v) for v in params.values()])
    # Forward mode reaches the parameters of theta, not derived ones (Soft's lam).
    layout = {n for n, _, _ in cls(**const, **params)._layout}
    assert torch.autograd.gradcheck(fn, args, check_forward_ad=set(names) <= layout)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_matches_numpy(case):
    _, cls, const, params, x, step, conj, call = case
    op = cls(**const, **params)
    ref = (op.prox_conj if conj else op.prox)(x, step, **call)
    f = fpt.prox_conj if conj else fpt.prox
    z = f(op, torch.tensor(x), step, **call)
    assert isinstance(z, torch.Tensor) and z.dtype == torch.float64
    np.testing.assert_array_equal(z.numpy(), ref)
    # Parameters as tensors give the same operator.
    z2 = f(cls, torch.tensor(x), torch.tensor(step, dtype=torch.float64), **const,
           **{k: torch.tensor(v) for k, v in params.items()}, **call)
    np.testing.assert_array_equal(z2.detach().numpy(), ref)


def test_backward_matches_vjp():
    rng = np.random.default_rng(1)
    x, w, g = rng.normal(size=(4, 20)).cumsum(-1), rng.uniform(0.5, 1.5, 19), rng.normal(size=(4, 20))
    op = fp.TV1D(weights=w)
    _, ctx = op.prox(x, 0.6, keep=True)
    gx, grads = op.vjp(ctx, g)
    xt, wt, st = _t(x), _t(w), _t(0.6)
    fpt.prox(fp.TV1D, xt, st, weights=wt).backward(torch.tensor(g))
    np.testing.assert_array_equal(xt.grad.numpy(), gx)
    np.testing.assert_array_equal(wt.grad.numpy(), grads.weights)
    assert st.grad.item() == grads.gamma


def test_float32_strides_no_grad():
    x = torch.randn(5, 30).cumsum(-1)
    z = fpt.prox(fp.TV1D(), x, 0.5)
    assert z.dtype == torch.float32
    np.testing.assert_array_equal(z.numpy(), fp.tv1d(x.numpy(), 0.5))
    v = torch.randn(30, 5, dtype=torch.float64).t()  # not contiguous
    np.testing.assert_array_equal(fpt.prox(fp.TV1D(), v, 0.5).numpy(), fp.tv1d(v.numpy().copy(), 0.5))
    with torch.no_grad():
        z = fpt.prox(fp.TV1D(), x.requires_grad_(), 0.5)
    assert not z.requires_grad
    x32 = torch.randn(3, 10, requires_grad=True)
    fpt.prox(fp.Soft(0.3), x32).sum().backward()
    assert x32.grad.dtype == torch.float32


def test_static_parameters():
    """Integer tensors and Python values are constants."""
    x = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    e = torch.tensor([[0, 1], [1, 2], [3, 4]])
    z = fpt.prox(fp.TreeTV, x, 0.5, n=6, edges=e)
    np.testing.assert_array_equal(z.detach().numpy(), fp.TreeTV(6, e.numpy()).prox(x.detach().numpy(), 0.5))
    z.sum().backward()
    assert x.grad is not None


def test_errors():
    x = torch.randn(10, dtype=torch.float64)
    with pytest.raises(TypeError):
        fpt.prox(fp.TV1D(), torch.arange(10), 0.5)
    with pytest.raises(TypeError):
        fpt.prox(fp.TV1D(), x.half(), 0.5)
    with pytest.raises(TypeError):
        fpt.prox(fp.TV1D(), x.numpy(), 0.5)
    with pytest.raises(TypeError):
        fpt.prox(fp.TV1D(), x, 0.5, weights=torch.ones(9))
    with pytest.raises(TypeError):
        fpt.prox(fp.TVIso(), torch.randn(4, 4, dtype=torch.float64), 0.5, axis=0)
    with pytest.raises(ValueError, match="no gradient"):
        fpt.prox(fp.L0, x, 0.5, lam=_t(0.3))
    # A float tensor that does not require grad is a constant.
    z = fpt.prox(fp.GroupL2, x, 0.5, lam=torch.tensor(0.8, dtype=torch.float64), groups=np.repeat(np.arange(5), 2))
    np.testing.assert_array_equal(z.numpy(), fp.GroupL2(0.8, np.repeat(np.arange(5), 2)).prox(x.numpy(), 0.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_round_trip():
    x = torch.randn(3, 20, device="cuda", dtype=torch.float64, requires_grad=True)
    z = fpt.prox(fp.TV1D(), x, 0.5)
    assert z.device == x.device
    z.sum().backward()
    assert x.grad.device == x.device
