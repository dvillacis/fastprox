"""Device-native isotropic and anisotropic TV in JAX (since 0.3): the method
of ``fastprox._tv_torch`` (FISTA on the dual with restarts and gap stopping;
a backward on the flat regions, found by label propagation and summed by a
sort and a segmented scan), written with ``lax`` loops so that it runs under
``jit`` and ``vmap`` on any JAX device."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

CHECK = 4


def _d(z, dim):
    """Forward difference along dim, zero past the last entry."""
    n = z.shape[dim]
    if n < 2:
        return jnp.zeros_like(z)
    pad = [(0, 0)] * z.ndim
    pad[dim] = (0, 1)
    return jnp.pad(jnp.diff(z, axis=dim), pad)


def _dt(p, dim):
    """Adjoint of _d: (D^T p)_i = p_{i-1} - p_i, the last entry of p unused."""
    n = p.shape[dim]
    if n < 2:
        return jnp.zeros_like(p)
    q = lax.slice_in_dim(p, 0, n - 1, axis=dim)
    after = [(0, 0)] * p.ndim
    before = [(0, 0)] * p.ndim
    after[dim] = (0, 1)
    before[dim] = (1, 0)
    return jnp.pad(q, before) - jnp.pad(q, after)


class Ops:
    """Operators for images (B, *img), weights w (k,), traced or not."""

    def __init__(self, k, w, iso):
        self.k, self.w, self.iso = k, w, iso

    def G(self, z):
        return jnp.stack([self.w[a] * _d(z, 1 + a) for a in range(self.k)])

    def Gt(self, u):
        out = self.w[0] * _dt(u[0], 1)
        for a in range(1, self.k):
            out = out + self.w[a] * _dt(u[a], 1 + a)
        return out

    @staticmethod
    def pnorm(g):
        out = jnp.abs(g[0]) if g.shape[0] == 1 else jnp.hypot(g[0], g[1])
        for a in range(2, g.shape[0]):
            out = jnp.hypot(out, g[a])
        return out

    def proj(self, y, gamma):
        if self.iso:
            nrm = self.pnorm(y)
            big = nrm > gamma
            return jnp.where(big[None], y * (gamma / jnp.where(big, nrm, 1.0))[None], y)
        return jnp.clip(y, -gamma, gamma)

    @staticmethod
    def isum(t):
        return t.reshape(t.shape[0], -1).sum(1)

    @staticmethod
    def dsum(t):
        return jnp.swapaxes(t, 0, 1).reshape(t.shape[1], -1).sum(1)

    def nsum(self, g):
        return self.isum(self.pnorm(g)) if self.iso else self.dsum(jnp.abs(g))

    def gap(self, x, u, gamma):
        z = x - self.Gt(u)
        g = self.G(z)
        return z, gamma * self.nsum(g) - self.dsum(g * u)


def _bimg(v, nd):
    return v.reshape(v.shape + (1,) * nd)


def solve(x, gamma, w, u0, iso, tol, max_iter):
    """(z, u, gap per image, iterations) for x (B, *img); tol, max_iter static."""
    k = w.shape[0]
    ops = Ops(k, w, iso)
    L = 4.0 * jnp.sum(w * w)
    tau = jnp.where(L > 0, 1.0 / jnp.where(L > 0, L, 1.0), 0.0)
    target = tol * ops.isum(x * x) / 2
    B, nd = x.shape[0], x.ndim - 1
    u = ops.proj(u0, gamma)

    def step(state, check):
        it, u, v, t, live = state
        up = ops.proj(v + tau * ops.G(x - ops.Gt(v)), gamma)
        tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
        coef = (t - 1) / tn
        if check:
            restart = ops.dsum((v - up) * (up - u)) > 0
            coef = jnp.where(restart, 0.0, coef)
            tn = jnp.where(restart, 1.0, tn)
        vn = up + _bimg(coef, nd)[None] * (up - u)
        lv = _bimg(live, nd)[None]
        return it + 1, jnp.where(lv, up, u), jnp.where(lv, vn, v), jnp.where(live, tn, t), live

    def body(carry):
        state, gap = carry
        it = state[0]
        n_plain = jnp.minimum(CHECK - 1, max_iter - it - 1)
        state = lax.fori_loop(0, n_plain, lambda _, s: step(s, False), state)
        state = step(state, True)
        it, u, v, t, live = state
        _, gnew = ops.gap(x, u, gamma)
        gap = jnp.where(live, gnew, gap)
        live = live & ~(gap <= target)
        return (it, u, v, t, live), gap

    def cond(carry):
        state, _ = carry
        return (state[0] < max_iter) & jnp.any(state[4])

    zero = jnp.zeros((B,), x.dtype)
    state = (jnp.asarray(0), u, u, jnp.ones((B,), x.dtype), jnp.ones((B,), bool))
    if max_iter > 0:
        state, _ = lax.while_loop(cond, body, (state, zero))
    it, u = state[0], state[1]
    z, gap = ops.gap(x, u, gamma)
    return z, u, gap, it, gap <= target


class Regions:
    """Connected regions of a batch (B, *img) under joins between p and
    p + e_a, in a sorted layout of fixed size (B, N); see _tv_torch.Regions."""

    def __init__(self, join, B, img):
        N = 1
        for n in img:
            N *= n
        self.B, self.N, self.img = B, N, img
        # Hooking and pointer jumping (see _tv_torch.Regions).
        shape = (B,) + tuple(img)
        idx = jnp.arange(B * N).reshape(shape)
        ends = []
        for a, j in enumerate(join):
            dim, n = 1 + a, img[a]
            if n < 2:
                continue
            lo = lax.slice_in_dim(idx, 0, n - 1, axis=dim)
            hi = lax.slice_in_dim(idx, 1, n, axis=dim)
            p = jnp.where(lax.slice_in_dim(j, 0, n - 1, axis=dim), lo, hi)
            ends.append((p.reshape(-1), hi.reshape(-1)))

        def jump(lab):
            return lax.while_loop(lambda c: c[1], lambda c: (c[0][c[0]], jnp.any(c[0][c[0]] != c[0])),
                                  (lab, jnp.asarray(True)))[0]

        def body(c):
            lab, _ = c
            new = lab
            for p, q in ends:
                rp, rq = new[p], new[q]
                new = new.at[jnp.maximum(rp, rq)].min(jnp.minimum(rp, rq))
            new = jump(new)
            return new, jnp.any(new != lab)

        lab = lax.while_loop(lambda c: c[1], body, (idx.reshape(-1), jnp.asarray(True)))[0].reshape(shape)
        perm = jnp.argsort(lab.reshape(B, N), axis=1, stable=True)
        keys = jnp.take_along_axis(lab.reshape(B, N), perm, axis=1)
        start = jnp.concatenate([jnp.ones((B, 1), bool), keys[:, 1:] != keys[:, :-1]], axis=1)
        end = jnp.concatenate([start[:, 1:], jnp.ones((B, 1), bool)], axis=1)
        idx = jnp.broadcast_to(jnp.arange(N), (B, N))
        last = lax.cummin(jnp.where(end, idx, N), axis=1, reverse=True)
        first = lax.cummax(jnp.where(start, idx, -1), axis=1)
        self.perm, self.keys, self.end = perm, keys, end
        self.size = jnp.where(end, last - first + 1, 0)
        self.at = jnp.take_along_axis(last, jnp.argsort(perm, axis=1), axis=1)
        self.same = []
        sh = 1
        while sh < N:
            self.same.append(keys[:, sh:] == keys[:, :-sh])
            sh *= 2

    def sum(self, v):
        """Region sums, at each region's last position (segmented scan by doubling)."""
        vs = jnp.take_along_axis(v.reshape(self.B, self.N), self.perm, axis=1)
        for i, same in enumerate(self.same):
            sh = 1 << i
            vs = vs + jnp.pad(jnp.where(same, vs[:, :-sh], 0.0), ((0, 0), (sh, 0)))
        return jnp.where(self.end, vs, 0.0)

    def spread(self, c):
        return jnp.take_along_axis(c, self.at, axis=1).reshape((self.B,) + tuple(self.img))


def _structure(x, z, u, gap, gamma, w, iso):
    """Flat entries from the dual, as in _tv_torch.Structure."""
    k = w.shape[0]
    ops = Ops(k, w, iso)
    B, nd = z.shape[0], z.ndim - 1
    g = ops.G(z)
    eps = jnp.finfo(z.dtype).eps
    inside = gamma * (1 - max(1e-9, 64 * float(eps)))
    live = [w[a] != 0 for a in range(k)]
    if iso:
        gn = ops.pnorm(g)
        floor = jnp.max(w) * 64 * eps * jnp.abs(x).reshape(B, -1).max(1)
        flat = (gn <= _bimg(floor, nd)) | (ops.pnorm(u) < inside)
        safe = jnp.where(flat, 1.0, gn)
        join = [flat & live[a] for a in range(k)]
        n = jnp.where(flat[None], 0.0, g / safe[None])
        mu = _bimg(0.1 * jnp.max(w) * jnp.sqrt(2 * jnp.maximum(gap, 0) / z[0].size), nd)  # _tv_torch.CURV_RMS
        h = jnp.where(flat | (gamma <= 0), 0.0, gamma / jnp.maximum(safe, mu))
    else:
        flat = jnp.abs(u) < inside
        join = [flat[a] & live[a] for a in range(k)]
        n = jnp.where(flat, 0.0, jnp.sign(u))
        h = None
    return ops, n, h, Regions(join, B, z.shape[1:])


def _H(n, h, v):
    return h[None] * (v - n * (n * v).sum(0, keepdims=True))


def _jacobi(w, n, h, R, dtype):
    """1 / diag(K) in region space (see _tv_torch.Structure.jacobi)."""
    k = w.shape[0]
    lab = R.at.reshape((R.B,) + tuple(R.img))
    nxt = []
    for a in range(k):
        dim, m = 1 + a, lab.shape[1 + a]
        t = jnp.full_like(lab, -1)
        if m > 1:
            pad = [(0, 0)] * lab.ndim
            pad[dim] = (0, 1)
            t = jnp.pad(lax.slice_in_dim(lab, 1, m, axis=dim), pad, constant_values=-1)
        nxt.append(jnp.where(w[a] != 0, t, -1))

    def term(e):
        ee = sum(t * t for t in e)
        ne = sum(n[a] * e[a] for a in range(k))
        return h * (ee - ne * ne)

    has = [t >= 0 for t in nxt]
    own = term([-w[a] * (has[a] & (nxt[a] != lab)).astype(dtype) for a in range(k)])
    for a in range(k):
        C = nxt[a]
        new = has[a] & (C != lab)
        for b in range(a):
            new = new & (nxt[b] != C)
        t = jnp.where(new, term([w[b] * (nxt[b] == C).astype(dtype) for b in range(k)]), 0.0)
        dim, m = 1 + a, lab.shape[1 + a]
        if m > 1:
            pad = [(0, 0)] * lab.ndim
            pad[dim] = (1, 0)
            own = own + jnp.pad(lax.slice_in_dim(t, 0, m - 1, axis=dim), pad)
    d = R.size.astype(dtype) + R.sum(own)
    return jnp.where(R.end, 1.0 / jnp.where(R.end, d, 1.0), 0.0).astype(dtype)


def _region_solve(ops, n, h, R, b, tol, max_iter):
    """K c = b in region space (see _tv_torch.Structure.solve)."""
    if h is None:
        return b * jnp.where(R.end, 1.0 / jnp.where(R.end, R.size, 1), 0.0).astype(b.dtype)
    inv = _jacobi(ops.w, n, h, R, b.dtype)

    def K(c):
        v = R.spread(c)
        return R.sum(v + ops.Gt(_H(n, h, ops.G(v))))

    bb = (b * b).sum(1)

    def body(st):
        it, c, r, p, rz, rr, done = st
        Kp = K(p)
        pKp = (p * Kp).sum(1)
        ok = ~done & (pKp > 0)
        alpha = jnp.where(ok, rz / jnp.where(pKp > 0, pKp, 1.0), 0.0)
        c = c + alpha[:, None] * p
        r = r - alpha[:, None] * Kp
        zr = r * inv
        rzn = (r * zr).sum(1)
        beta = jnp.where(ok, rzn / jnp.where(rz > 0, rz, 1.0), 0.0)
        p = zr + beta[:, None] * p
        rz = jnp.where(ok, rzn, rz)
        rr = jnp.where(ok, (r * r).sum(1), rr)
        done = done | (rr <= (tol * tol) * bb) | ~(pKp > 0)
        return it + 1, c, r, p, rz, rr, done

    zr = b * inv
    st = (jnp.asarray(0), jnp.zeros_like(b), b, zr, (b * zr).sum(1), bb, bb <= (tol * tol) * bb)
    st = lax.while_loop(lambda s: (s[0] < max_iter) & ~jnp.all(s[6]), body, st)
    return st[1]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def tv(x, gamma, w, u0, iso, tol, max_iter, dtol, dmax):
    z, u, gap, it, conv = solve(x, gamma, w, u0, iso, tol, max_iter)
    return z, (u, gap, it, conv)


def _fwd(x, gamma, w, u0, iso, tol, max_iter, dtol, dmax):
    z, u, gap, it, conv = solve(x, gamma, w, u0, iso, tol, max_iter)
    return (z, (u, gap, it, conv)), (x, z, u, gap, gamma, w, u0)


def _bwd(iso, tol, max_iter, dtol, dmax, res, cts):
    x, z, u, gap, gamma, w, u0 = res
    ops, n, h, R = _structure(x, z, u, gap, gamma, w, iso)
    hz = R.spread(_region_solve(ops, n, h, R, R.sum(cts[0]), dtol, dmax))
    Gh = ops.G(hz)
    ggamma = -jnp.sum(Gh * n)
    HGh = _H(n, h, Gh) if h is not None else None
    gw = []
    for a in range(w.shape[0]):
        v = -jnp.sum(_d(hz, 1 + a) * u[a])
        if HGh is not None:
            v = v - jnp.sum(HGh[a] * _d(z, 1 + a))
        gw.append(v)
    return hz, ggamma.astype(jnp.result_type(gamma)), jnp.stack(gw).astype(w.dtype), jnp.zeros_like(u0)


tv.defvjp(_fwd, _bwd)


def dual_out(u, w, iso):
    if iso:
        return u
    return jnp.stack([w[a] * _dt(u[a], 1 + a) for a in range(w.shape[0])])


def dual_in(q, w, iso):
    if iso:
        return q
    us = []
    for a in range(w.shape[0]):
        safe = jnp.where(w[a] > 0, w[a], 1.0)
        us.append(jnp.where(w[a] > 0, -jnp.cumsum(q[a], axis=1 + a) / safe, 0.0))
    return jnp.stack(us)
