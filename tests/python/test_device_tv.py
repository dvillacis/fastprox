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



def disk_and_band(n, seed):
    rng = np.random.default_rng(seed)
    a, b = np.mgrid[0:n, 0:n] / n
    return (((a - 0.5) ** 2 + (b - 0.45) ** 2 < 0.08) + 0.5 * (b > 0.7)) + 0.1 * rng.normal(size=(n, n))


@pytest.mark.parametrize("iso", [True, False])
def test_derivatives_at_loose_tolerances(fw, iso):
    """Flat pixels and edges from the dual, as in the C kernels: close to
    the exact derivative after a loose solve (isotropic: 2.4e-2 at a gap
    tolerance of 1e-6 and 2.6e-3 at 1e-8, against 0.49 and 0.15 from a
    threshold on |G z|); anisotropic: exact here."""
    x = disk_and_band(32, 0)[None]
    c = np.random.default_rng(9).normal(size=x.shape)
    op = C[iso](2)
    _, ctx = op.prox(x, 0.1, keep=True, tol=1e-15, max_iter=10 ** 7)
    gref, _ = op.vjp(ctx, c, max_iter=10 ** 5) if iso else op.vjp(ctx, c)
    tv = fw.tv(iso)
    for tol, bound in ((1e-6, 0.08), (1e-8, 1e-2)):
        if fw.name == "torch":
            torch = fw.mod
            xt = torch.tensor(x, requires_grad=True)
            (g,) = torch.autograd.grad(tv(xt, 0.1, tol=tol, max_iter=10 ** 6), xt, torch.tensor(c))
            g = g.numpy()
        else:
            import jax
            import jax.numpy as jnp
            g = np.asarray(jax.grad(lambda x: jnp.vdot(tv(x, 0.1, tol=tol, max_iter=10 ** 6), c))(jnp.asarray(x)))
        err = np.linalg.norm(g - gref) / np.linalg.norm(gref)
        assert err <= (bound if iso else 1e-12)


@pytest.mark.parametrize("shape,w", [((2, 9, 11), [1.0, 0.7]), ((1, 5, 6, 4), [1.0, 0.0, 2.0])])
def test_jacobi_preconditioner(fw, shape, w):
    """The isotropic backward's preconditioner is the diagonal of its
    region-space system, column by column."""
    rng = np.random.default_rng(3)
    x = np.round(rng.normal(size=shape) * 2) / 2 + 0.05 * rng.normal(size=shape)
    if fw.name == "torch":
        torch = fw.mod
        import fastprox._tv_torch as T
        xt = torch.tensor(x)
        z, u, gap, _, _ = T.solve(xt, 0.3, w, True, 1e-8, 10 ** 5, None)
        S = T.Structure(xt, z, u, gap, 0.3, w, True)
        R, dinv, K = S.R, S.jacobi(torch.float64).numpy(), lambda e: S.K(torch.tensor(e)).numpy()
        end = R.end.numpy()
    else:
        import jax.numpy as jnp
        import fastprox._tv_jax as J
        wj = jnp.asarray(w)
        g = jnp.full((shape[0],), 0.3)
        z, u, gap, _, _ = J.solve(jnp.asarray(x), g, wj, jnp.zeros((len(w),) + shape), True, 1e-8, 10 ** 5)
        ops, n, h, R = J._structure(jnp.asarray(x), z, u, gap, g, wj, True)
        dinv = np.asarray(J._jacobi(wj, n, h, R, jnp.float64))
        K = lambda e: np.asarray(R.sum(R.spread(jnp.asarray(e)) + ops.Gt(J._H(n, h, ops.G(R.spread(jnp.asarray(e)))))))
        end = np.asarray(R.end)
    B, N = end.shape
    assert (end.sum(1) < N).all()  # some regions have more than one pixel
    diag = np.zeros((B, N))
    for j in range(N):
        e = np.zeros((B, N))
        e[:, j] = end[:, j]
        diag[:, j] = K(e)[:, j]
    ref = np.where(end, 1 / np.where(end, diag, 1.0), 0.0)
    np.testing.assert_allclose(dinv, ref, rtol=1e-13, atol=0)

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
    with pytest.raises(ValueError):  # gamma must broadcast to the batch shape (3,)
        tv(fw.arr(np.zeros((3, 4, 4))), fw.arr([0.1, 0.2]))
    if fw.name == "torch":  # values are checked where they are concrete
        with pytest.raises(ValueError):
            tv(fw.arr(np.zeros((2, 4, 4))), fw.arr([0.1, -0.2]))


@pytest.mark.parametrize("iso", [True, False])
def test_gamma_per_image(fw, iso):
    """One step per image, as DeepInverse's denoisers take one sigma per
    sample: each image's result, gap and gradients are those of a call with
    its own scalar step, and a step of shape (B, 1) broadcasts over the
    channels of a (B, C, H, W) batch, its gradient summed over them."""
    x = images((2, 3, 10, 11), 12)  # B = 2, C = 3
    g = np.array([[0.15], [0.4]])
    c = np.random.default_rng(13).normal(size=x.shape)
    tv = fw.tv(iso)
    kw = dict(tol=1e-12, max_iter=10 ** 6)
    if fw.name == "torch":
        torch = fw.mod
        xt = torch.tensor(x, requires_grad=True)
        gt = torch.tensor(g, requires_grad=True)
        z = tv(xt, gt, **kw)
        z.backward(torch.tensor(c))
        z, gx, gg = z.detach().numpy(), xt.grad.numpy(), gt.grad.numpy()
    else:
        import jax
        import jax.numpy as jnp
        f = lambda x, g: jnp.vdot(tv(x, g, **kw), c)
        z = np.asarray(tv(jnp.asarray(x), jnp.asarray(g), **kw))
        gx, gg = (np.asarray(v) for v in jax.grad(f, argnums=(0, 1))(jnp.asarray(x), jnp.asarray(g)))
    assert gg.shape == g.shape
    for b in range(2):
        if fw.name == "torch":
            xb = torch.tensor(x[b], requires_grad=True)
            gb = torch.tensor(float(g[b, 0]), dtype=torch.float64, requires_grad=True)
            zb = tv(xb, gb, **kw)
            zb.backward(torch.tensor(c[b]))
            zb, gxb, ggb = zb.detach().numpy(), xb.grad.numpy(), gb.grad.item()
        else:
            zb = np.asarray(tv(jnp.asarray(x[b]), float(g[b, 0]), **kw))
            fb = lambda x, g, cb=c[b]: jnp.vdot(tv(x, g, **kw), cb)
            gxb, ggb = jax.grad(fb, argnums=(0, 1))(jnp.asarray(x[b]), float(g[b, 0]))
            gxb, ggb = np.asarray(gxb), float(ggb)
        np.testing.assert_allclose(z[b], zb, atol=1e-10)
        np.testing.assert_allclose(gx[b], gxb, atol=1e-7 * (1 + np.abs(gxb).max()))
        assert float(gg[b, 0]) == pytest.approx(ggb, rel=1e-6, abs=1e-8)
    # a zero step leaves its image unchanged
    z0 = np.asarray(tv(fw.arr(x), fw.arr([[0.0], [0.4]]), **kw))
    np.testing.assert_allclose(z0[0], x[0], atol=1e-12)
    np.testing.assert_allclose(z0[1], z[1], atol=1e-10)


@pytest.mark.parametrize("iso", [True, False])
def test_gamma_per_image_gradcheck(fw, iso):
    x = images((2, 5, 6), 14)
    tv = fw.tv(iso)
    if fw.name == "torch":
        torch = fw.mod
        f = lambda x, g: tv(x, g, tol=1e-15, max_iter=10 ** 6)
        args = (torch.tensor(x, requires_grad=True),
                torch.tensor([0.3, 0.45], dtype=torch.float64, requires_grad=True))
        assert torch.autograd.gradcheck(f, args, check_forward_ad=True, atol=1e-5)
    else:
        import jax.numpy as jnp
        from jax.test_util import check_grads
        check_grads(lambda x, g: tv(x, g, tol=1e-15, max_iter=10 ** 6),
                    (jnp.asarray(x), jnp.asarray([0.3, 0.45])), order=1, modes=["rev"], eps=1e-6)


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
