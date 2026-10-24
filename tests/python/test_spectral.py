"""Spectral operators (since 1.0): prox maps on singular values and
eigenvalues, with derivatives from fp_spectral_jvp."""

import numpy as np
import pytest

import fastprox as fp

RNG = np.random.default_rng(0)


def fd(f, x, h, e=1e-6):
    return (f(x + e * h) - f(x - e * h)) / (2 * e)


def built(m, n, s, rng):
    """U diag(s) V^T with random orthonormal U (m x k) and V (n x k)."""
    k = len(s)
    u, _ = np.linalg.qr(rng.normal(size=(m, k)))
    v, _ = np.linalg.qr(rng.normal(size=(n, k)))
    return u @ np.diag(s) @ v.T


def sym(x):
    return (x + x.T) / 2


SVD_OPS = {
    "nuclear": lambda: fp.Spectral(fp.Soft(0.7)),
    "nuclear_ball": lambda: fp.Spectral(fp.L1Ball(3.0)),
    "spectral_norm": lambda: fp.Spectral(fp.LInf(0.8)),
    "spectral_ball": lambda: fp.SpectralNormBall(1.5),
    "frobenius_ball": lambda: fp.Spectral(fp.L2Ball(2.0)),
    "huber": lambda: fp.Spectral(fp.Huber(0.6, 0.5)),
    "scad": lambda: fp.Spectral(fp.SCAD(0.5, 3.7)),
    "mcp": lambda: fp.Spectral(fp.MCP(0.5, 3.0)),
    "log": lambda: fp.Spectral(fp.LogPenalty(0.4, 1.0)),
}
EIG_OPS = {
    "psd": lambda: fp.PSD(),
    "spectraplex": lambda: fp.Spectral(fp.Simplex(1.0), "eigh"),
    "logdet": lambda: fp.Spectral(fp.LogBarrier(0.4), "eigh"),
    "bounds": lambda: fp.Spectral(fp.Box(-0.5, 1.0), "eigh"),
    "soft": lambda: fp.Spectral(fp.Soft(0.3), "eigh"),
}


def test_references():
    x = RNG.normal(size=(7, 5)) * 2
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    np.testing.assert_allclose(fp.NuclearNorm(0.7).prox(x, 1.5), u @ np.diag(np.maximum(s - 1.05, 0)) @ vt,
                               atol=1e-13)
    np.testing.assert_allclose(fp.SpectralNormBall(1.5).prox(x), u @ np.diag(np.minimum(s, 1.5)) @ vt,
                               atol=1e-13)
    y = sym(RNG.normal(size=(6, 6)))
    lam, q = np.linalg.eigh(y)
    np.testing.assert_allclose(fp.PSD().prox(y), q @ np.diag(np.maximum(lam, 0)) @ q.T, atol=1e-13)
    assert fp.NuclearNorm(0.7).value(x) == pytest.approx(0.7 * s.sum(), rel=1e-14)
    assert fp.PSD().value(y) == np.inf and fp.PSD().value(fp.PSD().prox(y)) == 0.0


@pytest.mark.parametrize("name", list(SVD_OPS) + list(EIG_OPS))
def test_optimality(name):
    """prox_{gamma g}(X) minimises g(Y) + |Y - X|^2 / (2 gamma) against
    perturbed points (the nonconvex maps included: their prox is global)."""
    op = (SVD_OPS | EIG_OPS)[name]()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 5)) * 1.5
    if op.kind == "eigh":
        x = sym(x)
    gamma = 0.8
    z = op.prox(x, gamma)
    obj = lambda y: op.value(y, tol=1e-9) + np.sum((y - x) ** 2) / (2 * gamma)  # noqa: E731
    f0 = obj(z)
    assert np.isfinite(f0)
    for _ in range(30):
        e = rng.normal(size=x.shape) * 10.0 ** rng.uniform(-4, -1)
        y = z + (sym(e) if op.kind == "eigh" else e)
        assert obj(y) >= f0 - 1e-10


@pytest.mark.parametrize("name", list(SVD_OPS) + list(EIG_OPS))
@pytest.mark.parametrize("shape", [(6, 6), (8, 5), (5, 8), (3, 4, 4)])
def test_jvp_finite_differences(name, shape):
    op = (SVD_OPS | EIG_OPS)[name]()
    if op.kind == "eigh" and shape[-1] != shape[-2]:
        pytest.skip("eigenvalues need square matrices")
    rng = np.random.default_rng(2)
    x = rng.normal(size=shape) * 1.5
    h = rng.normal(size=shape)
    if op.kind == "eigh":
        x, h = ((a + np.swapaxes(a, -1, -2)) / 2 for a in (x, h))
    _, ctx = op.prox(x, 0.9, keep=True)
    np.testing.assert_allclose(op.jvp(ctx, h), fd(lambda y: op.prox(y, 0.9), x, h), atol=2e-7)


@pytest.mark.parametrize("name", list(SVD_OPS) + list(EIG_OPS))
def test_adjoint_and_vjp(name):
    op = (SVD_OPS | EIG_OPS)[name]()
    rng = np.random.default_rng(3)
    shape = (6, 6) if op.kind == "eigh" else (7, 4)
    x, h, g = (rng.normal(size=shape) for _ in range(3))
    if op.kind == "eigh":
        x, h, g = sym(x), sym(h), sym(g)
    _, ctx = op.prox(x, 1.1, keep=True)
    lhs = np.sum(g * op.jvp(ctx, h))
    gx, grads = op.vjp(ctx, g)
    assert lhs == pytest.approx(np.sum(gx * h), rel=1e-12, abs=1e-12)
    # The gamma gradient against finite differences of <g, prox_gamma(x)>.
    e = 1e-6
    dgam = (np.sum(g * op.prox(x, 1.1 + e)) - np.sum(g * op.prox(x, 1.1 - e))) / (2 * e)
    assert grads.gamma == pytest.approx(dgam, rel=1e-6, abs=1e-8)
    # dgamma in the jvp reaches the same derivative.
    assert np.sum(g * op.jvp(ctx, np.zeros(shape), dgamma=1.0)) == pytest.approx(dgam, rel=1e-6, abs=1e-8)


def test_parameter_gradient():
    """d/dw of <g, prox> for the nuclear norm's weight (Soft's theta = w)."""
    rng = np.random.default_rng(4)
    x, g = rng.normal(size=(6, 4)) * 2, rng.normal(size=(6, 4))
    op = fp.Spectral(fp.Soft(0.6, weights=[1.3]))
    _, ctx = op.prox(x, 1.0, keep=True)
    _, grads = op.vjp(ctx, g)
    e = 1e-6
    f = lambda w: np.sum(g * fp.Spectral(fp.Soft(0.6, weights=[w])).prox(x))  # noqa: E731
    assert grads.weights == pytest.approx((f(1.3 + e) - f(1.3 - e)) / (2 * e), rel=1e-6)
    dz = op.jvp(ctx, np.zeros_like(x), dparams={"weights": 1.0})
    assert np.sum(g * dz) == pytest.approx(grads.weights, rel=1e-10)


@pytest.mark.parametrize("s", [(2.0, 2.0, 2.0, 1.5, 1.5), (3.0, 1.2, 0.0, 0.0, 0.0), (5.0, 2.5, 2.5, 0.3, 0.3),
                               (1.3, 1.3, 1.3, 1.3, 1.3)])
@pytest.mark.parametrize("name", ["nuclear", "nuclear_ball", "spectral_norm", "huber", "scad"])
def test_ties_and_zeros(name, s):
    """Repeated and zero singular values: the map is differentiable when the
    scalar map is, and the JVP matches finite differences there."""
    op = SVD_OPS[name]()
    rng = np.random.default_rng(5)
    x = built(7, 5, s, rng)
    _, ctx = op.prox(x, 1.0, keep=True)
    for _ in range(3):
        h = rng.normal(size=x.shape)
        np.testing.assert_allclose(op.jvp(ctx, h), fd(lambda y: op.prox(y), x, h), atol=2e-7)


def test_eigenvalue_ties():
    rng = np.random.default_rng(6)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    for name in EIG_OPS:
        op = EIG_OPS[name]()
        x = q @ np.diag([0.9, 0.9, 0.9, -0.4, -0.4]) @ q.T
        _, ctx = op.prox(x, 1.0, keep=True)
        h = sym(rng.normal(size=(5, 5)))
        np.testing.assert_allclose(op.jvp(ctx, h), fd(lambda y: op.prox(y), x, h), atol=2e-7, err_msg=name)


@pytest.mark.parametrize("s", [(2.0, 2.0, 2.0, 1.2, 1.2), (3.0, 1.2, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.2, 0.2),
                               (0.9, 0.9, 0.7, 0.7, 0.1)])
def test_mcp_ties(s):
    """Spectral MCP at repeated singular values on each piece of firm
    thresholding (0, (s - gamma lam) b / (b - gamma), s), away from its kinks
    at gamma lam = 0.5 and b lam = 1.5 (feedback U6)."""
    op = SVD_OPS["mcp"]()
    rng = np.random.default_rng(5)
    x = built(7, 5, s, rng)
    _, ctx = op.prox(x, 1.0, keep=True)
    for _ in range(3):
        h = rng.normal(size=x.shape)
        np.testing.assert_allclose(op.jvp(ctx, h), fd(lambda y: op.prox(y), x, h), atol=2e-7)


_KINKS = [("mcp", fp.MCP(0.5, 3.0), 0.5), ("mcp", fp.MCP(0.5, 3.0), 1.5),
          ("scad", fp.SCAD(0.5, 3.7), 1.0), ("scad", fp.SCAD(0.5, 3.7), 1.85)]


@pytest.mark.parametrize("name,scalar,t", _KINKS)
@pytest.mark.parametrize("tied", [False, True])
def test_kinks_and_ties(name, scalar, t, tied):
    """Singular values on a kink of the scalar map (MCP: gamma lam, b lam;
    SCAD: (1 + gamma) lam, a lam), alone and tied (feedback U6). There the
    map is not differentiable and the JVP is one element of its
    B-subdifferential, adjoint to the VJP:
    - at values exactly on the kink, the scalar map's branch below for each,
      so a tie's block is that slope times A's, the limit from below with the
      tie kept;
    - at a tie that rounding splits across the kink, as an SVD of the matrix
      does, the limit of the Jacobian at points with the pair split by delta
      in the same basis, which depends on that basis within the tie."""
    op = fp.Spectral(scalar)
    rng = np.random.default_rng(11)
    U, _ = np.linalg.qr(rng.normal(size=(7, 5)))
    V, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    H, G = rng.normal(size=(7, 5)), rng.normal(size=(7, 5))
    on = [2, 3] if tied else [2]

    def jvp(sv):
        _, ctx = op.prox_from(U, sv, V.T, keep=True)
        return op.jvp(ctx, H), ctx

    s0 = np.array([2.6, 2.2, t, t if tied else 0.3, 0.1])
    j0, ctx = jvp(s0)
    assert np.isfinite(j0).all()
    gx, _ = op.vjp(ctx, G)
    assert np.sum(G * j0) == pytest.approx(np.sum(gx * H), rel=1e-12)
    below = s0.copy()
    below[on] = t - 1e-4  # beyond the tie band, so the quotients are used
    np.testing.assert_allclose(j0, jvp(below)[0], atol=1e-3 * np.abs(j0).max())
    if tied:
        split = s0.copy()
        split[2], split[3] = t + 4.4e-16, t - 8.9e-16
        js, _ = jvp(split)
        apart = s0.copy()
        apart[2], apart[3] = t + 1e-4, t - 1e-4
        np.testing.assert_allclose(js, jvp(apart)[0], atol=1e-3 * np.abs(js).max())


@pytest.mark.parametrize("b", [3.0, 1.1, 1.001])
def test_mcp_gain(b):
    """On firm thresholding's middle piece, gamma lam < s < b lam, the
    derivative is b / (b - gamma): unbounded as b falls to gamma, so a
    hypergradient through it is amplified as much (feedback U6). gamma >= b
    is refused."""
    op = fp.Spectral(fp.MCP(0.5, b))
    s0 = 0.25 * (1 + b)  # midway between gamma lam and b lam at gamma = 1
    _, ctx = op.prox(np.diag([s0, 0.9 * s0, 0.3]), 1.0, keep=True)
    e = np.zeros((3, 3))
    e[0, 0] = 1.0
    assert op.jvp(ctx, e)[0, 0] == pytest.approx(b / (b - 1.0), rel=1e-12)
    with pytest.raises(ValueError):
        op.prox(np.eye(3), b)


def test_mcp_proximal_gradient():
    """Proximal gradient on spectral MCP, 50% matrix completion of a rank-4
    30 x 25 matrix, at a step 0.99 / L (L = 1): the objective decreases at
    every iteration and the iterates stay finite (feedback U6; a step below
    1/L is the condition for a nonconvex penalty, Bolte, Sabach and Teboulle
    2014)."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(30, 4)) @ rng.normal(size=(4, 25))
    mask = rng.random(M.shape) < 0.5
    op = fp.Spectral(fp.MCP(0.5, 3.0))
    X, t, f = np.zeros_like(M), 0.99, []
    for _ in range(200):
        X = op.prox(X - t * mask * (X - M), t)
        f.append(0.5 * np.sum((mask * (X - M)) ** 2) + op.value(X))
    assert np.isfinite(X).all()
    assert all(b <= a * (1 + 1e-12) for a, b in zip(f, f[1:]))


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_autograd_contrast(seed):
    """At tied singular values autograd through torch.linalg.svd gives NaN
    (a computed gap of exactly 0) or a gradient off by tens of percent (gaps
    of a few ulps); fastprox matches finite differences."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    x = built(7, 5, (2.0, 2.0, 2.0, 1.5, 1.5), rng)
    g = rng.normal(size=x.shape)
    xt = torch.tensor(x, requires_grad=True)
    u, s, vh = torch.linalg.svd(xt, full_matrices=False)
    (u @ torch.diag(torch.clamp(s - 0.7, min=0)) @ vh * torch.tensor(g)).sum().backward()
    op = fp.NuclearNorm(0.7)
    _, ctx = op.prox(x, keep=True)
    gx, _ = op.vjp(ctx, g)
    tg = xt.grad.numpy()
    assert not np.isfinite(tg).all() or np.abs(tg - gx).max() > 0.05 * np.abs(gx).max()
    for _ in range(3):
        h = rng.normal(size=x.shape)
        assert np.sum(gx * h) == pytest.approx(np.sum(g * fd(lambda y: op.prox(y), x, h)), rel=1e-6)


@pytest.mark.parametrize("shape", [(6, 6), (9, 5), (5, 9)])
def test_divergence(shape):
    """The trace of the JVP against the closed form of Candes, Sing-Long and
    Trzasko (2013) for singular value thresholding."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=shape) * 2
    m, n = shape
    op = fp.NuclearNorm(0.7)
    _, ctx = op.prox(x, keep=True)
    tr = 0.0
    for i in range(m):
        for j in range(n):
            e = np.zeros(shape)
            e[i, j] = 1.0
            tr += op.jvp(ctx, e)[i, j]
    s = np.linalg.svd(x, compute_uv=False)
    z = np.maximum(s - 0.7, 0)
    k = len(s)
    closed = abs(m - n) * np.sum(z / s) + np.sum(s > 0.7) + 2 * sum(
        s[i] * z[i] / (s[i] ** 2 - s[j] ** 2) for i in range(k) for j in range(k) if i != j)
    assert tr == pytest.approx(closed, rel=1e-12)


def test_prox_from_truncated():
    """Singular value thresholding from the leading triplets only."""
    rng = np.random.default_rng(9)
    x = built(8, 6, (5.0, 3.0, 2.0, 0.4, 0.2, 0.1), rng)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    op = fp.NuclearNorm(0.5)
    np.testing.assert_allclose(op.prox_from(u[:, :3], s[:3], vt[:3]), op.prox(x), atol=1e-13)
    _, ctx = op.prox_from(u[:, :3], s[:3], vt[:3], keep=True)
    with pytest.raises(ValueError, match="full thin"):
        op.jvp(ctx, x)


def test_batches_threads_float32():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(64, 24, 20))
    h = rng.normal(size=x.shape)
    op = fp.Spectral(fp.L1Ball(4.0))
    z1, c1 = op.prox(x, threads=1, keep=True)
    z8, c8 = op.prox(x, threads=8, keep=True)
    np.testing.assert_array_equal(z1, z8)
    np.testing.assert_array_equal(op.jvp(c1, h), op.jvp(c8, h))
    for b in (0, 37):
        _, cb = op.prox(x[b], keep=True)
        np.testing.assert_allclose(op.jvp(cb, h[b]), op.jvp(c1, h)[b], atol=1e-13)
    x32 = x.astype(np.float32)
    z32, c32 = op.prox(x32, keep=True)
    assert z32.dtype == np.float32 and op.jvp(c32, h.astype(np.float32)).dtype == np.float32
    np.testing.assert_allclose(op.jvp(c32, h.astype(np.float32)), op.jvp(c1, h), atol=2e-4)


def test_errors():
    with pytest.raises(ValueError, match="absolutely symmetric"):
        fp.Spectral(fp.NonNeg())
    with pytest.raises(ValueError, match="weights must be equal"):
        fp.Spectral(fp.Soft(1.0, weights=[1.0, 2.0, 1.0]))
    with pytest.raises(ValueError):
        fp.Spectral(fp.Isotonic(), "eigh")
    with pytest.raises(ValueError):
        fp.Spectral(fp.Box(0.0, 1.0))  # not even: eigenvalues only
    fp.Spectral(fp.Box(0.0, 1.0), "eigh")
    with pytest.raises(TypeError):
        fp.Spectral(fp.TVAniso())
    with pytest.raises(ValueError, match="kind"):
        fp.Spectral(fp.Soft(), "qr")
    op = fp.NuclearNorm()
    with pytest.raises(ValueError, match="matrices"):
        op.prox(np.ones(4))
    with pytest.raises(ValueError, match="square"):
        fp.PSD().prox(np.ones((3, 4)))
    _, ctx = op.prox(np.eye(3), keep=True)
    with pytest.raises(ValueError, match="shape"):
        op.jvp(ctx, np.ones((3, 4)))
    with pytest.raises(ValueError, match="ctx"):
        fp.NuclearNorm().jvp(ctx, np.ones((3, 3)))
    assert repr(op).startswith("NuclearNorm(Soft(")


def _nuclear_jvp_exact(U, s, V, lam, H):
    """Ding, Sun, Sun and Toh's formula with quotients, for singular value
    soft thresholding at distinct s."""
    z = np.maximum(s - lam, 0.0)
    A = U.T @ H @ V
    with np.errstate(divide="ignore", invalid="ignore"):
        Gp = (z[:, None] - z[None, :]) / (s[:, None] - s[None, :])
    Gm = (z[:, None] + z[None, :]) / (s[:, None] + s[None, :])
    B = Gp * (A + A.T) / 2 + Gm * (A - A.T) / 2
    np.fill_diagonal(B, (s > lam) * np.diag(A))
    return U @ B @ V.T


@pytest.mark.parametrize("s, lam", [
    (1 + 5e-6 * np.arange(200), 1 + 5.125e-4),
    (np.concatenate([1 + 5e-6 * np.arange(100), 1.01 + 1e-3 * np.arange(100)]), 1 + 2.475e-4),
])
def test_chained_near_ties(s, lam):
    """Values 5e-6 apart, under the band eps^(1/3) max|s|, chain into one
    cluster 1e-3 wide; only pairs within the band take the averaged limit
    (audit T7). The averaged limit gave 0.23 and 0.06 relative error. The
    thresholds sit midway between two values: a breakpoint of the scalar
    map within the band of a pair still takes the average there."""
    s = np.sort(s)[::-1]
    rng = np.random.default_rng(7)
    U, _ = np.linalg.qr(rng.standard_normal((200, 200)))
    V, _ = np.linalg.qr(rng.standard_normal((200, 200)))
    H = np.random.default_rng(7).standard_normal((200, 200))
    op = fp.NuclearNorm(lam)
    _, ctx = op.prox_from(U, s, V.T, keep=True)
    ref = _nuclear_jvp_exact(U, s, V, lam, H)
    assert np.linalg.norm(op.jvp(ctx, H) - ref) <= 1e-8 * np.linalg.norm(ref)
