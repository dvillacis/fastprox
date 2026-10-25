"""The PyTorch wrappers: gradcheck in reverse and forward mode through the
library's derivatives, agreement with the NumPy API, dtypes, devices and
errors."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import fastprox as fp  # noqa: E402
import fastprox.torch as fpt  # noqa: E402

from interop_cases import built, cases, row_steps, spectral_cases  # noqa: E402

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


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_steps_gradcheck(case):
    """One step per row (fp_prox_steps): each row's step gradient from a JVP
    in gamma, forward mode through each row's dz/dgamma."""
    _, cls, const, params, x, step, conj, call = case
    s = row_steps(case)
    if conj:
        with pytest.raises(ValueError, match="scalar sigma"):
            fpt.prox_conj(cls, _t(x), _t(s), **const, **params, **call)
        return
    fn = lambda x, s: fpt.prox(cls, x, s, **const, **params, **call)  # noqa: E731
    assert torch.autograd.gradcheck(fn, (_t(x), _t(s)), check_forward_ad=True)


def test_steps_match_rows():
    rng = np.random.default_rng(6)
    x, g = rng.normal(size=(4, 3, 20)).cumsum(-1), rng.normal(size=(4, 3, 20))
    s = rng.uniform(0.2, 1.0, (4, 1))  # broadcast over the second axis
    op = fp.TV1D(weights=rng.uniform(0.5, 1.5, 19))
    xt, st = _t(x), _t(s)
    z = fpt.prox(op, xt, st)
    np.testing.assert_array_equal(z.detach().numpy(), op.prox(x, np.broadcast_to(s, (4, 3))))
    (z * torch.tensor(g)).sum().backward()
    assert st.grad.shape == (4, 1)
    for i in range(4):
        gr = [op.vjp(op.prox(x[i, j], s[i, 0], keep=True)[1], g[i, j])[1].gamma for j in range(3)]
        assert st.grad[i, 0].item() == pytest.approx(sum(gr), rel=1e-11, abs=1e-12)
    # Along the first axis, steps over the remaining one.
    xa = _t(x[:, 0].T.copy())
    sa = _t(rng.uniform(0.2, 1.0, 4))
    fpt.prox(op, xa, sa, axis=0).sum().backward()
    ref = [op.vjp(op.prox(x[i, 0], sa[i].item(), keep=True)[1], np.ones(20))[1].gamma for i in range(4)]
    np.testing.assert_allclose(sa.grad.numpy(), ref, rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match="broadcast"):
        fpt.prox(op, xt, _t(np.ones(4)))
    # A single value in any shape is a scalar step, as before 1.1.
    x1 = _t(x[0, 0])
    s1 = _t([0.5])
    fpt.prox(op, x1, s1).sum().backward()
    assert s1.grad.shape == (1,)
    assert s1.grad.item() == op.vjp(op.prox(x[0, 0], 0.5, keep=True)[1], np.ones(20))[1].gamma


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


@pytest.mark.parametrize("op, params, shape", [
    (fp.NuclearNorm, {"lam": 0.6}, (4, 5, 3)),
    (fp.SpectralNormBall, {"radius": 1.2}, (2, 3, 3, 6)),
    (fp.PSD(), {}, (3, 4, 4)),
], ids=["nuclear", "spectral_ball", "psd"])
def test_spectral_steps(op, params, shape):
    """One step per matrix: gradcheck in both modes, and the prox against
    matrix-by-matrix calls."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=shape) * 1.5
    s = rng.uniform(0.4, 1.0, shape[:-2])
    pt = {k: _t(v) for k, v in params.items()}
    fn = lambda x, s, *v: fpt.prox(op, x, s, **dict(zip(pt, v)))  # noqa: E731
    assert torch.autograd.gradcheck(fn, (_t(x), _t(s), *pt.values()), check_forward_ad=not pt)
    ref = _spectral_op(op, params)
    z = fpt.prox(op, torch.tensor(x), torch.tensor(s), **params).numpy()
    for i in np.ndindex(shape[:-2]):
        np.testing.assert_allclose(z[i], ref.prox(x[i], s[i]), atol=1e-13)


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


def test_weighted_isotonic_has_no_conj():
    # Math audit M2: the weighted fit projects in the metric W.
    w = np.array([1.0, 3.0, 2.0])
    y = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64)
    for f in (lambda: fpt.prox_conj(fp.Isotonic(w), y),
              lambda: fpt.prox_conj(fp.Isotonic, y, weights=_t(w))):
        with pytest.raises(ValueError, match="no Euclidean conjugate"):
            f()


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


def _second_order_cases():
    rng = np.random.default_rng(4)
    return {
        "soft": (lambda x: fpt.prox(fp.Soft(0.4), x, 0.8), rng.normal(size=6)),
        "tv1d": (lambda x: fpt.prox(fp.TV1D(), x, 0.6), rng.normal(size=(2, 8)).cumsum(-1)),
        "nuclear": (lambda x: fpt.prox(fp.NuclearNorm(0.6), x, 0.9), rng.normal(size=(4, 4)) * 1.5),
        "tv_iso": (lambda x: fpt.tv_iso(x, 0.3, tol=1e-8), rng.normal(size=(1, 6, 6))),
        "tv_aniso": (lambda x: fpt.tv_aniso(x, 0.3, tol=1e-8), rng.normal(size=(1, 6, 6))),
    }


@pytest.mark.parametrize("name", sorted(_second_order_cases()))
def test_second_order_raises(name):
    """Second-order reverse mode raises until 1.0.x instead of returning a
    Hessian without the prox's own second derivatives (torch 2.14's
    once_differentiable does not refuse); first order is unchanged."""
    f, x = _second_order_cases()[name]
    xt = _t(x)

    def loss(x):
        return (f(x) ** 2).sum() + (x ** 3).sum()

    with pytest.raises(RuntimeError, match="second-order"):
        torch.autograd.grad(loss(xt), xt, create_graph=True)
    with pytest.raises(RuntimeError, match="second-order"):
        torch.autograd.functional.hvp(loss, xt.detach(), torch.ones_like(xt))
    g1, = torch.autograd.grad(loss(xt), xt)
    loss(xt).backward()
    assert torch.equal(xt.grad, g1) and torch.isfinite(g1).all()


def test_structure_only_when_a_derivative_can_follow(monkeypatch):
    # Under no_grad, or for inputs without grad, the forward keeps no
    # structure (audit P10); the bits are the same.
    from fastprox import _interop
    keeps = []
    real = _interop.Spec.run

    def run(self, op, x, step, keep=True):
        keeps.append(keep)
        return real(self, op, x, step, keep)

    monkeypatch.setattr(_interop.Spec, "run", run)
    x = np.random.default_rng(33).normal(size=(6, 30))
    op = fp.L1Ball(1.5)
    ref = op.prox(x, 0.8)
    xr = _t(x)
    with torch.no_grad():
        z = fpt.prox(op, xr, 0.8)
    assert keeps[-1] is False and z.numpy().tobytes() == ref.tobytes()
    z = fpt.prox(op, _t(x, grad=False), 0.8)
    assert keeps[-1] is False and z.numpy().tobytes() == ref.tobytes()
    z = fpt.prox(op, xr, 0.8)
    assert keeps[-1] is True
    z.sum().backward()
    _, ctx = op.prox(x, 0.8, keep=True)
    assert xr.grad.numpy().tobytes() == op.vjp(ctx, np.ones_like(x))[0].tobytes()
    # Forward mode: a dual input keeps the structure, and the tangent is jvp's.
    import torch.autograd.forward_ad as fwAD
    dx = np.random.default_rng(34).normal(size=x.shape)
    with fwAD.dual_level():
        zt = fpt.prox(op, fwAD.make_dual(_t(x, grad=False), torch.from_numpy(dx)), 0.8)
        tangent = fwAD.unpack_dual(zt).tangent
    assert keeps[-1] is True
    assert tangent.numpy().tobytes() == op.jvp(ctx, dx).tobytes()
