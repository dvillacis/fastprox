"""Device-native image TV in the PyTorch and JAX wrappers: agreement with the
C kernels within the certified gap, the gap recomputed here from the
returned dual, derivatives against the C kernels and by gradcheck /
check_grads, warm starts, reproducibility, special cases and errors."""

import numpy as np
import pytest

import fastprox as fp

C = {True: fp.TVIso, False: fp.TVAniso}


def _d(z, ax):
    out = np.zeros_like(z)
    sl = [slice(None)] * z.ndim
    sl[ax] = slice(0, -1)
    out[tuple(sl)] = np.diff(z, axis=ax)
    return out


def _dt(p, ax):
    n = p.shape[ax]
    q = np.take(p, range(n - 1), axis=ax)
    pad_b = [(0, 0)] * p.ndim
    pad_a = [(0, 0)] * p.ndim
    pad_b[ax], pad_a[ax] = (1, 0), (0, 1)
    return np.pad(q, pad_b) - np.pad(q, pad_a)


def recomputed_gap(x, z, dual, gamma, w, iso):
    """The duality gap of (z, dual) from the definitions, in float64; also
    checks the dual's feasibility and z = x - (its divergence)."""
    k = len(w)
    axes = [x.ndim - k + a for a in range(k)]
    if iso:
        u = dual
        assert np.sqrt((u ** 2).sum(0)).max() <= gamma * (1 + 1e-6) + 1e-12
        zu = x - sum(w[a] * _dt(u[a], axes[a]) for a in range(k))
        np.testing.assert_allclose(z, zu, atol=1e-6 * (1 + np.abs(x).max()))
        g = np.stack([w[a] * _d(z, axes[a]) for a in range(k)])
        return float((gamma * np.sqrt((g ** 2).sum(0)) - (g * u).sum(0)).sum())
    q = dual
    np.testing.assert_allclose(x - z, q.sum(0), atol=1e-6 * (1 + np.abs(x).max()))
    for a in range(k):
        p = -np.cumsum(q[a], axis=axes[a])
        assert np.abs(p).max() <= gamma * w[a] * (1 + 1e-6) + 1e-12
    tv = sum(w[a] * np.abs(_d(z, axes[a])).sum() for a in range(k))
    return float(gamma * tv - (z * (x - z)).sum())


def images(shape, seed):
    rng = np.random.default_rng(seed)
    k = len(shape) - 1
    blocks = np.indices(shape[1:]).sum(0) // 4 % 2
    return rng.normal(scale=0.5, size=shape) + blocks


CASES = [((3, 16, 18), [1.0, 1.0], 0.3), ((2, 12, 13), [0.7, 1.4], 0.5), ((1, 9, 1), [1.0, 1.0], 0.4),
         ((2, 6, 7, 8), [1.0, 0.5, 1.2], 0.25), ((2, 6, 7, 5), [1.0, 0.0, 0.8], 0.3)]


class Framework:
    def __init__(self, name):
        self.name = name
        if name == "torch":
            self.mod = pytest.importorskip("torch")
            import fastprox.torch as f
        else:
            self.mod = pytest.importorskip("jax")
            self.mod.config.update("jax_enable_x64", True)
            import fastprox.jax as f
        self.f = f

    def arr(self, a, dtype=np.float64):
        if self.name == "torch":
            return self.mod.tensor(np.asarray(a, dtype))
        import jax.numpy as jnp
        return jnp.asarray(np.asarray(a, dtype))

    def tv(self, iso):
        return self.f.tv_iso if iso else self.f.tv_aniso


@pytest.fixture(params=["torch", "jax"])
def fw(request):
    if request.param == "torch":
        yield Framework("torch")
        return
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    yield Framework("jax")
    jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("iso", [True, False])
@pytest.mark.parametrize("shape,w,gamma", CASES)
@pytest.mark.parametrize("tol", [1e-4, 1e-9])
def test_against_c_within_the_gap(fw, iso, shape, w, gamma, tol):
    x = images(shape, len(shape) + int(10 * gamma))
    k = len(w)
    z, info = fw.tv(iso)(fw.arr(x), gamma, ndim=k, weights=w, tol=tol, max_iter=100000, return_info=True)
    z, dual = np.asarray(z), np.asarray(info.dual)
    assert bool(info.converged)
    gap = recomputed_gap(x, z, dual, gamma, w, iso)
    assert gap == pytest.approx(float(info.gap), rel=1e-6, abs=1e-12)
    xx = (x.reshape(-1, *x.shape[-k:]) ** 2).sum()
    assert gap <= tol * xx / 2 * 1.0001 + 1e-12
    zc, ic = C[iso](k, w).solve(x, gamma, tol=1e-13, max_iter=10 ** 6)
    assert ((z - zc) ** 2).sum() <= 2 * (gap + ic.gap) * 1.0001 + 1e-20


@pytest.mark.parametrize("iso", [True, False])
@pytest.mark.parametrize("shape,w,gamma", CASES[:2] + CASES[3:4])
def test_derivatives_match_c(fw, iso, shape, w, gamma):
    x = images(shape, 3)
    k = len(w)
    rng = np.random.default_rng(4)
    c = rng.normal(size=x.shape)
    op = C[iso](k, w)
    _, ctx = op.prox(x, gamma, keep=True, tol=1e-14, max_iter=10 ** 6)
    gxc, gr = op.vjp(ctx, c)
    tv = fw.tv(iso)
    if fw.name == "torch":
        torch = fw.mod
        xt = torch.tensor(x, requires_grad=True)
        gt = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
        wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
        tv(xt, gt, ndim=k, weights=wt, tol=1e-14, max_iter=10 ** 6).backward(torch.tensor(c))
        gx, gg, gw = xt.grad.numpy(), gt.grad.item(), wt.grad.numpy()
    else:
        import jax
        import jax.numpy as jnp
        loss = lambda x, g, w: jnp.vdot(tv(x, g, ndim=k, weights=w, tol=1e-14, max_iter=10 ** 6), c)
        gx, gg, gw = (np.asarray(v) for v in jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(
            jnp.asarray(x), gamma, jnp.asarray(w)))
    scale = 1 + np.abs(gxc).max()
    np.testing.assert_allclose(gx, gxc, atol=1e-7 * scale)
    assert float(gg) == pytest.approx(gr.gamma, rel=1e-6, abs=1e-7)
    np.testing.assert_allclose(gw, gr.weights, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("iso", [True, False])
def test_gradcheck(fw, iso):
    x = images((1, 5, 6), 5)
    w = np.array([0.9, 1.2])
    tv = fw.tv(iso)
    if fw.name == "torch":
        torch = fw.mod
        f = lambda x, g, w: tv(x, g, weights=w, tol=1e-15, max_iter=10 ** 6)
        args = (torch.tensor(x, requires_grad=True), torch.tensor(0.35, dtype=torch.float64, requires_grad=True),
                torch.tensor(w, requires_grad=True))
        assert torch.autograd.gradcheck(f, args, check_forward_ad=True, atol=1e-5)
    else:
        import jax.numpy as jnp
        from jax.test_util import check_grads
        check_grads(lambda x, g, w: tv(x, g, weights=w, tol=1e-15, max_iter=10 ** 6),
                    (jnp.asarray(x), 0.35, jnp.asarray(w)), order=1, modes=["rev"], eps=1e-6)


@pytest.mark.parametrize("iso", [True, False])
def test_warm_starts(fw, iso):
    x = images((2, 14, 15), 6)
    tv = fw.tv(iso)
    z, info = tv(fw.arr(x), 0.4, tol=1e-8, return_info=True)
    z2, info2 = tv(fw.arr(x), 0.4, tol=1e-8, return_info=True, warm_start=info.dual)
    assert int(info2.iters) <= 8 and int(info.iters) > 8
    np.testing.assert_allclose(np.asarray(z2), np.asarray(z), atol=1e-6)
    _, ic = C[iso](2).solve(x, 0.4, tol=1e-12, max_iter=10 ** 6)
    z3, info3 = tv(fw.arr(x), 0.4, tol=1e-8, return_info=True, warm_start=ic.dual)
    assert int(info3.iters) < int(info.iters)


@pytest.mark.parametrize("iso", [True, False])
def test_same_iterations_as_c_and_reproducible(fw, iso):
    """Isotropic TV is the C kernel's method; both frameworks repeat exactly."""
    x = images((2, 20, 22), 7)
    tv = fw.tv(iso)
    z1, i1 = tv(fw.arr(x), 0.3, tol=1e-6, return_info=True)
    z2, i2 = tv(fw.arr(x), 0.3, tol=1e-6, return_info=True)
    assert np.array_equal(np.asarray(z1), np.asarray(z2)) and int(i1.iters) == int(i2.iters)
    if iso:
        _, ic = fp.TVIso(2).solve(x, 0.3, tol=1e-6)
        assert abs(int(i1.iters) - ic.iters) <= 8


def test_frameworks_agree():
    pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    t, j = Framework("torch"), Framework("jax")
    x = images((2, 12, 14), 8)
    for iso in (True, False):
        zt, it = t.tv(iso)(t.arr(x), 0.3, tol=1e-8, return_info=True)
        zj, ij = j.tv(iso)(j.arr(x), 0.3, tol=1e-8, return_info=True)
        assert int(it.iters) == int(ij.iters)
        np.testing.assert_allclose(zt.numpy(), np.asarray(zj), atol=1e-12)
    jax.config.update("jax_enable_x64", old)


def test_jax_vmap_of_grad():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import fastprox.jax as fpj
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(11)
    x, c = jnp.asarray(rng.normal(size=(3, 10, 11))), jnp.asarray(rng.normal(size=(3, 10, 11)))
    for f in (fpj.tv_iso, fpj.tv_aniso):
        loss = lambda x, c: jnp.vdot(f(x, 0.4, tol=1e-12, max_iter=10 ** 5), c)
        np.testing.assert_allclose(np.asarray(jax.vmap(jax.grad(loss))(x, c)),
                                   np.asarray(jax.grad(loss)(x, c)), atol=1e-12)
    jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("iso", [True, False])
def test_special_cases(fw, iso):
    x = images((2, 9, 10), 9)
    tv = fw.tv(iso)
    np.testing.assert_array_equal(np.asarray(tv(fw.arr(x), 0.0)), x)
    np.testing.assert_array_equal(np.asarray(tv(fw.arr(x), 0.5, weights=[0.0, 0.0])), x)
    np.testing.assert_array_equal(np.asarray(tv(fw.arr(x), 0.5, max_iter=0)), x)
    z, info = tv(fw.arr(x), 0.5, max_iter=3, return_info=True)
    assert not bool(info.converged) and int(info.iters) == 3
    z32 = tv(fw.arr(x, np.float32), 0.5, tol=1e-4)
    assert np.asarray(z32).dtype == np.float32
    np.testing.assert_allclose(np.asarray(z32), np.asarray(tv(fw.arr(x), 0.5, tol=1e-4)), atol=1e-3)


def test_errors(fw):
    tv = fw.f.tv_iso
    with pytest.raises(ValueError):
        tv(fw.arr(np.zeros(5)), 0.5)
    with pytest.raises(ValueError):
        tv(fw.arr(np.zeros((4, 4))), 0.5, ndim=4)
    with pytest.raises(ValueError):
        tv(fw.arr(np.zeros((4, 4))), 0.5, weights=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        tv(fw.arr(np.zeros((4, 4))), 0.5, warm_start=np.zeros((2, 4, 5)))
    with pytest.raises(TypeError):
        tv(fw.arr(np.zeros((4, 4)), np.int32), 0.5)


def _gpu(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.parametrize("iso", [True, False])
def test_torch_gpu(iso):
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    dev = _gpu(torch)
    if dev is None:
        pytest.skip("no GPU")
    x = images((3, 40, 44), 10).astype(np.float32)
    tv = fpt.tv_iso if iso else fpt.tv_aniso
    xt = torch.tensor(x, device=dev, requires_grad=True)
    z, info = tv(xt, 0.3, tol=1e-5, return_info=True)
    assert z.device.type == dev and info.dual.device.type == dev
    z2 = tv(xt, 0.3, tol=1e-5)
    assert torch.equal(z, z2)                                  # run to run on one device
    zc, _ = C[iso](2).solve(x.astype(np.float64), 0.3, tol=1e-12, max_iter=10 ** 6)
    assert ((z.detach().cpu().numpy() - zc) ** 2).sum() <= 2 * info.gap * 1.01 + 1e-6
    z.sum().backward()
    assert xt.grad.device.type == dev and torch.isfinite(xt.grad).all()
