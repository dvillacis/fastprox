"""The PyTorch wrappers: gradcheck in reverse and forward mode through the
library's derivatives, agreement with the NumPy API, dtypes, devices and
errors."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import fastprox as fp  # noqa: E402
import fastprox.torch as fpt  # noqa: E402

from interop_cases import built, cases, spectral_cases  # noqa: E402

CASES = cases()
IDS = [c[0] for c in CASES]
SPECTRAL = spectral_cases()
SPECTRAL_IDS = [c[0] for c in SPECTRAL]


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


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------


def _spectral_op(op, params):
    return op(**params) if isinstance(op, type) else op


@pytest.mark.parametrize("case", SPECTRAL, ids=SPECTRAL_IDS)
def test_spectral_gradcheck(case):
    _, op, params, x, step = case
    names = list(params)

    def fn(x, step, *vals):
        return fpt.prox(op, x, step, **dict(zip(names, vals)))

    args = (_t(x), _t(step), *[_t(v) for v in params.values()])
    # Forward mode reaches theta, not the derived lam and radius.
    assert torch.autograd.gradcheck(fn, args, check_forward_ad=not names)


@pytest.mark.parametrize("case", SPECTRAL, ids=SPECTRAL_IDS)
def test_spectral_matches_numpy(case):
    _, op, params, x, step = case
    ref_op = _spectral_op(op, params)
    g = np.random.default_rng(2).normal(size=x.shape)
    z, ctx = ref_op.prox(x, step, keep=True)
    gx, grads = ref_op.vjp(ctx, g)
    xt, st = _t(x), _t(step)
    pt = {k: _t(v) for k, v in params.items()}
    zt = fpt.prox(op, xt, st, **pt)
    assert zt.dtype == torch.float64
    np.testing.assert_allclose(zt.detach().numpy(), z, atol=1e-13)
    (zt * torch.tensor(g)).sum().backward()
    np.testing.assert_allclose(xt.grad.numpy(), gx, atol=1e-12)
    assert st.grad.item() == pytest.approx(grads.gamma, abs=1e-12)
    for k, v in pt.items():
        assert v.grad.item() == pytest.approx(getattr(grads, k), abs=1e-12)


def test_spectral_ties():
    """At tied singular values autograd through torch.linalg.svd fails; the
    wrapper's gradient is the derivative, against finite differences."""
    rng = np.random.default_rng(3)
    x = built(7, 5, (2.0, 2.0, 2.0, 1.5, 1.5), rng)
    g = rng.normal(size=x.shape)
    op = fp.NuclearNorm(0.7)
    xt = _t(x)
    (fpt.prox(op, xt) * torch.tensor(g)).sum().backward()
    gx = xt.grad.numpy()
    assert np.isfinite(gx).all()
    for _ in range(3):
        h = rng.normal(size=x.shape)
        fd = (np.sum(g * op.prox(x + 1e-6 * h)) - np.sum(g * op.prox(x - 1e-6 * h))) / 2e-6
        assert np.sum(gx * h) == pytest.approx(fd, rel=1e-6)


def test_spectral_float32_and_eigh_input():
    rng = np.random.default_rng(4)
    x = torch.tensor(rng.normal(size=(3, 6, 6)), dtype=torch.float32, requires_grad=True)
    z = fpt.prox(fp.PSD(), x)
    assert z.dtype == torch.float32
    xs = (x.detach().numpy() + x.detach().numpy().transpose(0, 2, 1)) / 2
    np.testing.assert_allclose(z.detach().numpy(), fp.PSD().prox(xs), atol=1e-5)
    z.sum().backward()
    assert x.grad.dtype == torch.float32


def test_spectral_errors():
    x = torch.randn(4, 4, dtype=torch.float64)
    with pytest.raises(TypeError, match="prox_conj"):
        fpt.prox_conj(fp.NuclearNorm(), x)
    with pytest.raises(TypeError, match="axis"):
        fpt.prox(fp.NuclearNorm(), x, axis=0)
    with pytest.raises(TypeError, match="Spectral"):
        fpt.prox(fp.Spectral, x)
    with pytest.raises(ValueError, match="matrices"):
        fpt.prox(fp.NuclearNorm(), torch.randn(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="square"):
        fpt.prox(fp.PSD(), torch.randn(3, 4, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_spectral_cuda():
    x = torch.randn(8, 5, 4, device="cuda", dtype=torch.float64, requires_grad=True)
    z = fpt.prox(fp.NuclearNorm(0.5), x)
    assert z.device == x.device
    z.sum().backward()
    assert x.grad.device == x.device
