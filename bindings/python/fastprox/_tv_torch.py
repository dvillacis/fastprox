"""Device-native isotropic and anisotropic TV in PyTorch tensor operations
(since 0.3): the problems of ``TVIso`` and ``TVAniso`` solved on the input's
device, GPU included.

Forward: FISTA on the dual (Beck and Teboulle 2009; Beck 2017, Section
12.4.4), u in the pixelwise ball |u_p| <= gamma (isotropic) or the box
|u_e| <= gamma (anisotropic), z = x - G^T u with G = W D (forward
differences, zero past the last entry). Step 1/L with L = 4 sum_a w_a^2, the
gradient restart of O'Donoghue and Candes tested with the gap every 4
iterations, and an image stops when its gap is at most tol |x|^2 / 2, as in
the C kernels (src/ops/tv_iso.cpp, tv_aniso.cpp). The anisotropic C kernel's
exact 1D steps and polish stay on the CPU, so this path is a plain dual
method: slower to tight tolerances, certified by the same gap.

Backward: the C kernels' derivative systems on the flat regions. As in
the C kernels, an entry is flat where its dual is strictly inside its bound,
|u| < gamma (1 - eta) (complementary slackness), with eta = max(1e-9,
64 eps) above the projection's rounding; isotropic TV also counts pixels
with |g| = |G z| at the rounding level of x, w_max 64 eps max|x|. A flat
pixel (isotropic) or flat edge (anisotropic) joins its pixels into a region,
and dz is constant on each region. The isotropic curvature gamma / |g_p| is
capped at gamma / mu, mu = 0.1 w_max rms with rms = sqrt(2 gap / N), as in
the C kernel. Anisotropic: dz is the region mean of
    r = dx - dG^T u - dgamma G_A^T n     (n = sign u on the non-flat edges),
exactly. Isotropic: with H_p = gamma / |g_p| (I - n_p n_p^T) on non-flat
pixels and M the map from region values to pixels,
    (M^T M + M^T G_A^T H G_A M) c = M^T (r - G_A^T H (dG z)_A),   dz = M c,
by conjugate gradients in region space preconditioned with the region sizes,
as in src/ops/tv_iso.cpp. The VJPs are the transposes.

Regions: hooking and pointer jumping over the joins (the only scatter is a
minimum, the same in any order), a stable sort of the labels within each
image, and region sums by a segmented scan by doubling (log2 N shifted adds
within segments). Every step is deterministic, so results repeat run to run
on one device; atomic scatter-adds would not (index_add_ on MPS does not).
Region vectors are kept at fixed size (B, N) in sorted order, the value of a
region at its last position.
"""

from __future__ import annotations

import math

import torch

from . import IterInfo

CHECK = 4  # iterations between gap and restart tests


class Problem:
    """Geometry and operators for a batch of images x of shape (B, *img)."""

    def __init__(self, x, w, iso):
        self.k = len(w)
        self.w = [float(a) for a in w]
        self.iso = iso
        self.B = x.shape[0]
        self.N = math.prod(x.shape[1:])
        self.positive = False  # every radius > 0 (set by solve; projections in place)

    def d(self, z, a):
        dim = 1 + a
        out = torch.zeros_like(z)
        n = z.shape[dim]
        if n > 1:
            torch.sub(z.narrow(dim, 1, n - 1), z.narrow(dim, 0, n - 1), out=out.narrow(dim, 0, n - 1))
        return out

    def dt(self, p, a):
        dim = 1 + a
        out = torch.zeros_like(p)
        n = p.shape[dim]
        if n > 1:
            q = p.narrow(dim, 0, n - 1)
            out.narrow(dim, 0, n - 1).sub_(q)
            out.narrow(dim, 1, n - 1).add_(q)
        return out

    def G(self, z):
        return torch.stack([self.w[a] * self.d(z, a) for a in range(self.k)])

    def Gt(self, u):
        out = torch.zeros_like(u[0])
        for a in range(self.k):
            if self.w[a] != 0.0:
                out.add_(self.dt(u[a], a), alpha=self.w[a])
        return out

    def primal(self, x, v):
        """x - G^T v."""
        z = x.clone()
        for a in range(self.k):
            dim, n = 1 + a, x.shape[1 + a]
            if self.w[a] == 0.0 or n < 2:
                continue
            q = v[a].narrow(dim, 0, n - 1)
            z.narrow(dim, 0, n - 1).add_(q, alpha=self.w[a])
            z.narrow(dim, 1, n - 1).sub_(q, alpha=self.w[a])
        return z

    def ascent(self, v, z, tau):
        """v + tau G z."""
        y = v.clone()
        for a in range(self.k):
            dim, n = 1 + a, z.shape[1 + a]
            if self.w[a] == 0.0 or n < 2:
                continue
            y[a].narrow(dim, 0, n - 1).add_(z.narrow(dim, 1, n - 1) - z.narrow(dim, 0, n - 1),
                                            alpha=tau * self.w[a])
        return y

    @staticmethod
    def pnorm(g):
        """Pixelwise 2-norm over the components (hypot: fast on every device;
        vector_norm over the leading dimension was 100x slower on MPS)."""
        out = g[0].abs() if g.shape[0] == 1 else torch.hypot(g[0], g[1])
        for a in range(2, g.shape[0]):
            out = torch.hypot(out, g[a])
        return out

    def norm(self, g):
        """|g| per entry: pixelwise 2-norm (isotropic) or per edge (anisotropic)."""
        return self.pnorm(g) if self.iso else g.abs()

    def proj(self, y, gamma):
        """Onto |u_p| <= gamma (isotropic) or |u_e| <= gamma (anisotropic), for
        one radius (a float) or one per image (a tensor (B,)); may overwrite y."""
        if not isinstance(gamma, torch.Tensor):  # the common case, in place
            if gamma <= 0:
                return torch.zeros_like(y)
            if self.iso:
                return y.div_(self.pnorm(y).div_(gamma).clamp_(min=1.0))
            return y.clamp_(-gamma, gamma)
        g = _img(self, gamma)
        if self.iso:
            n = self.pnorm(y)
            if self.positive:  # in place, as for one radius
                return y.div_(n.div_(g).clamp_(min=1.0))
            return y.mul_(torch.where(n > g, g / torch.where(n > g, n, torch.ones_like(n)), torch.ones_like(n)))
        return y.clamp_(min=-g, max=g)

    def isum(self, t):
        """Per-image sums of an image batch (B, ...)."""
        return t.reshape(self.B, -1).sum(1)

    def dsum(self, t):
        """Per-image sums of a dual field (k, B, ...)."""
        return t.transpose(0, 1).reshape(self.B, -1).sum(1)

    def nsum(self, g):
        """Per-image sums of |g|."""
        return self.isum(self.norm(g)) if self.iso else self.dsum(g.abs())

    def gap(self, x, u, gamma):
        z = self.primal(x, u)
        g = self.G(z)
        return z, g, gamma * self.nsum(g) - self.dsum(g * u)


CURV_RMS = 0.1  # the curvature cap, as kCurvRms in src/ops/tv_iso.cpp


def dual_eta(dtype):
    """Relative distance below the dual's bound that still counts as on it:
    the C kernels' 1e-9, or 64 eps where rounding is coarser (float32)."""
    return max(1e-9, 64 * torch.finfo(dtype).eps)


def _img(prob, v):
    return v.reshape((prob.B,) + (1,) * (prob.k))


def per_image(gamma, B, x):
    """gamma (a number or a tensor of B values) as a tensor (B,) like x."""
    if isinstance(gamma, torch.Tensor):
        return gamma.to(dtype=x.dtype, device=x.device).expand(B)
    return torch.full((B,), float(gamma), dtype=x.dtype, device=x.device)


def solve(x, gamma, w, iso, tol, max_iter, u0):
    """(z, u, gap per image, iterations, converged) for x of shape (B, *img)
    and gamma a number or one value per image."""
    prob = Problem(x, w, iso)
    gamma = per_image(gamma, prob.B, x)
    # The projections take one float when every image has the same radius.
    radius = float(gamma[0]) if bool((gamma == gamma[0]).all()) else gamma
    prob.positive = bool((gamma > 0).all())
    L = 4.0 * sum(a * a for a in prob.w)
    u = torch.zeros((prob.k,) + tuple(x.shape), dtype=x.dtype, device=x.device) if u0 is None \
        else prob.proj(u0.to(dtype=x.dtype, device=x.device).clone(), radius)
    none = not bool((gamma > 0).any())
    if L == 0.0 or none or max_iter <= 0:
        u = torch.zeros_like(u) if L == 0.0 or none else u
        z, g, gap = prob.gap(x, u, gamma)
        return z, u, gap, 0, bool((gap <= tol * prob.isum(x * x) / 2).all())
    tau = 1.0 / L
    target = tol * prob.isum(x * x) / 2
    v = u.clone()
    t = torch.ones(prob.B, dtype=x.dtype, device=x.device)
    live = torch.ones(prob.B, dtype=torch.bool, device=x.device)
    gap = None
    it = 0
    all_live = True
    while it < max_iter:
        up = prob.proj(prob.ascent(v, prob.primal(x, v), tau), radius)
        it += 1
        tn = (1 + torch.sqrt(1 + 4 * t * t)) / 2
        coef = (t - 1) / tn
        check = it % CHECK == 0 or it == max_iter
        if check:  # restart from up where the momentum points uphill
            restart = prob.dsum((v - up) * (up - u)) > 0
            coef = torch.where(restart, torch.zeros_like(coef), coef)
            tn = torch.where(restart, torch.ones_like(tn), tn)
        vn = torch.addcmul(up, _img(prob, coef).unsqueeze(0), up - u)
        if all_live:
            u, v, t = up, vn, tn
        else:
            lv = _img(prob, live).unsqueeze(0)
            u, v, t = torch.where(lv, up, u), torch.where(lv, vn, v), torch.where(live, tn, t)
        if check:
            _, _, gnew = prob.gap(x, u, gamma)
            gap = gnew if gap is None else torch.where(live, gnew, gap)
            live = live & ~(gap <= target)
            all_live = bool(live.all())
            if not all_live and not bool(live.any()):
                break
    z, _, gap = prob.gap(x, u, gamma)
    return z, u, gap, it, bool((gap <= target).all())


class Regions:
    """The connected regions of a batch (B, *img) under joins between p and
    p + e_a (join[a][p] true), in a sorted layout."""

    def __init__(self, join, B, img):
        dev = join[0].device
        N = math.prod(img)
        self.B, self.N, self.img = B, N, img
        # Hooking and pointer jumping: every joined edge hooks the larger of
        # its two roots under the smaller (a scatter of minima, the same in
        # any order), then every pixel jumps to its root; O(log N) rounds.
        # Labels are int32 where they fit: atomic minima on 64-bit integers do
        # not compile on some Metal GPUs (GitHub's macOS runners).
        shape = (B,) + tuple(img)
        ltype = torch.int32 if B * N < 2 ** 31 else torch.int64
        idx = torch.arange(B * N, device=dev, dtype=ltype).reshape(shape)
        ends = []
        for a, j in enumerate(join):
            dim, n = 1 + a, img[a]
            if n < 2:
                continue
            p = torch.where(j.narrow(dim, 0, n - 1), idx.narrow(dim, 0, n - 1), idx.narrow(dim, 1, n - 1))
            ends.append((p.reshape(-1).long(), idx.narrow(dim, 1, n - 1).reshape(-1).long()))
        lab = idx.reshape(-1).clone()
        while True:
            old = lab
            lab = lab.clone()
            for p, q in ends:  # unjoined edges have p = q: no change
                rp, rq = lab[p], lab[q]
                lab.scatter_reduce_(0, torch.maximum(rp, rq).long(), torch.minimum(rp, rq), reduce="amin")
            while True:
                nxt = lab[lab.long()]
                if torch.equal(nxt, lab):
                    break
                lab = nxt
            if torch.equal(lab, old):
                break
        lab = lab.reshape(shape)
        keys, self.perm = torch.sort(lab.reshape(B, N), dim=1, stable=True)
        self.keys = keys
        start = torch.ones_like(keys, dtype=torch.bool)
        start[:, 1:] = keys[:, 1:] != keys[:, :-1]
        self.end = torch.ones_like(start)
        self.end[:, :-1] = start[:, 1:]
        idx = torch.arange(N, device=dev, dtype=torch.int32).expand(B, N)
        last = torch.flip(torch.cummin(torch.flip(torch.where(self.end, idx, N), [1]), 1).values, [1])
        first = torch.cummax(torch.where(start, idx, -1), 1).values
        self.size = torch.where(self.end, (last - first + 1), 0)  # region sizes at their last position
        self.at = torch.gather(last.long(), 1, torch.argsort(self.perm, dim=1))  # pixel -> its region's position
        # Masks of the segmented scan, one per doubling step (log2 N bytes per pixel).
        self.same = []
        sh = 1
        while sh < N:
            self.same.append(keys[:, sh:] == keys[:, :-sh])
            sh *= 2

    def sum(self, v):
        """Region sums of an image batch: (B, N), at each region's last position."""
        vs = torch.gather(v.reshape(self.B, self.N), 1, self.perm)
        for i, same in enumerate(self.same):
            sh = 1 << i
            nxt = vs.clone()
            nxt[:, sh:].add_(vs[:, :-sh] * same)
            vs = nxt
        return vs * self.end

    def spread(self, c):
        """Region values (B, N) to the pixels (B, *img)."""
        return torch.gather(c, 1, self.at).reshape((self.B,) + tuple(self.img))

    def mean(self, v):
        return self.spread(self.sum(v) / torch.where(self.end, self.size, 1).to(v.dtype))


class Structure:
    """What the derivatives need at a solution (z, u) of shape (B, *img)."""

    def __init__(self, x, z, u, gap, gamma, w, iso):
        prob = Problem(z, w, iso)
        self.prob, self.z, self.u = prob, z, u
        self.gamma = per_image(gamma, prob.B, z)
        gimg = _img(prob, self.gamma)
        g = prob.G(z)
        eps = torch.finfo(z.dtype).eps
        inside = gimg * (1 - dual_eta(z.dtype))
        live = [prob.w[a] != 0.0 for a in range(prob.k)]
        if iso:
            gn = prob.pnorm(g)
            floor = max(prob.w) * 64 * eps * x.reshape(prob.B, -1).abs().amax(1)
            flat = (gn <= _img(prob, floor)) | (prob.pnorm(u) < inside)
            join = [flat & live[a] for a in range(prob.k)]
            safe = torch.where(flat, torch.ones_like(gn), gn)
            self.n = torch.where(flat.unsqueeze(0), torch.zeros_like(g), g / safe.unsqueeze(0))
            mu = _img(prob, CURV_RMS * max(prob.w) * torch.sqrt(2 * torch.clamp(gap, min=0) / prob.N))
            self.h = torch.where(flat, torch.zeros_like(gn), gimg / torch.maximum(safe, mu)) \
                if bool((self.gamma > 0).any()) else None
        else:
            flat = u.abs() < inside
            join = [flat[a] & live[a] for a in range(prob.k)]
            self.n = torch.where(flat, torch.zeros_like(u), torch.sign(u))
            self.h = None
        self.R = Regions(join, prob.B, tuple(z.shape[1:]))

    def H(self, v):
        """H_p v_p on the non-flat pixels (isotropic), for a field v (k, B, ...)."""
        return self.h.unsqueeze(0) * (v - self.n * (self.n * v).sum(0, keepdim=True))

    def K(self, c):
        """M^T (I + G_A^T H G_A) M c in region space."""
        prob, R = self.prob, self.R
        v = R.spread(c)
        return R.sum(v + prob.Gt(self.H(prob.G(v))))

    def jacobi(self, dtype):
        """1 / diag(K) in region space, as in src/ops/tv_iso.cpp: |R| plus, for
        every non-flat pixel p, h_p (|e|^2 - (n_p . e)^2) with e = (G 1_R)_p,
        nonzero for the regions of p and of its forward neighbours. Each
        neighbour's term is moved onto that neighbour's pixel, so that the
        region sums take it."""
        prob, R = self.prob, self.R
        lab = R.at.reshape((prob.B,) + tuple(R.img))
        nxt = []  # the label at p + e_a, or -1 where there is no such edge
        for a in range(prob.k):
            t = torch.full_like(lab, -1)
            n = lab.shape[1 + a]
            if prob.w[a] != 0.0 and n > 1:
                t.narrow(1 + a, 0, n - 1).copy_(lab.narrow(1 + a, 1, n - 1))
            nxt.append(t)

        def term(e):  # h_p (|e|^2 - (n . e)^2) for e = (e_0, ..., e_{k-1})
            ee = sum(t * t for t in e)
            ne = sum(self.n[a] * e[a] for a in range(prob.k))
            return self.h * (ee - ne * ne)

        has = [t >= 0 for t in nxt]
        own = term([-prob.w[a] * (has[a] & (nxt[a] != lab)).to(dtype) for a in range(prob.k)])
        for a in range(prob.k):
            # The region of p + e_a, unless it is p's or an earlier neighbour's.
            C = nxt[a]
            new = has[a] & (C != lab)
            for b in range(a):
                new = new & (nxt[b] != C)
            t = torch.where(new, term([prob.w[b] * (nxt[b] == C).to(dtype) for b in range(prob.k)]),
                            torch.zeros_like(own))
            n = lab.shape[1 + a]
            if n > 1:
                own.narrow(1 + a, 1, n - 1).add_(t.narrow(1 + a, 0, n - 1))
        d = R.size.to(dtype) + R.sum(own)
        return torch.where(R.end, 1.0 / torch.where(R.end, d, torch.ones_like(d)), torch.zeros_like(d))

    def solve(self, b, tol, max_iter):
        """Region values c with K c = b (isotropic: conjugate gradients per
        image, preconditioned by the inverse diagonal of K; anisotropic:
        b / |R|); (c, iterations,
        largest relative residual)."""
        R = self.R
        if self.h is None:
            inv = torch.where(R.end, 1.0 / torch.where(R.end, R.size, 1).to(b.dtype), torch.zeros_like(b))
            return b * inv, 0, 0.0
        inv = self.jacobi(b.dtype)
        c = torch.zeros_like(b)
        r = b.clone()
        zr = r * inv
        p = zr.clone()
        rz = (r * zr).sum(1)
        bb = (b * b).sum(1)
        rr = bb.clone()
        done = rr <= (tol * tol) * bb
        it = 0
        while it < max_iter:
            if it % 8 == 0 and bool(done.all()):
                break
            Kp = self.K(p)
            pKp = (p * Kp).sum(1)
            ok = ~done & (pKp > 0)
            alpha = torch.where(ok, rz / torch.where(pKp > 0, pKp, torch.ones_like(pKp)), torch.zeros_like(rz))
            c = c + alpha[:, None] * p
            r = r - alpha[:, None] * Kp
            zr = r * inv
            rzn = (r * zr).sum(1)
            beta = torch.where(ok, rzn / torch.where(rz > 0, rz, torch.ones_like(rz)), torch.zeros_like(rz))
            p = zr + beta[:, None] * p
            rz = torch.where(ok, rzn, rz)
            rr = torch.where(ok, (r * r).sum(1), rr)
            done = done | (rr <= (tol * tol) * bb) | ~(pKp > 0)
            it += 1
        rel = torch.sqrt(rr / torch.where(bb > 0, bb, torch.ones_like(bb)))
        return c, it, float(rel.max()) if rel.numel() else 0.0

    def jvp(self, dx, dgamma, dw, tol, max_iter):
        prob, R = self.prob, self.R
        r = torch.zeros_like(self.z) if dx is None else dx.clone()
        if dgamma is not None:
            r = r - _img(prob, dgamma) * prob.Gt(self.n)
        if dw is not None:
            for a in range(prob.k):
                if dw[a] != 0.0:
                    r = r - dw[a] * prob.dt(self.u[a], a)
            if self.h is not None:
                q = torch.stack([dw[a] * prob.d(self.z, a) for a in range(prob.k)])
                r = r - prob.Gt(self.H(q))
        c, it, res = self.solve(R.sum(r), tol, max_iter)
        return R.spread(c), (res, it)

    def vjp(self, gz, tol, max_iter):
        """(gx, d/dgamma per image, d/dw_a summed over the batch) for the
        cotangent gz."""
        prob, R = self.prob, self.R
        c, it, res = self.solve(R.sum(gz), tol, max_iter)
        h = R.spread(c)
        Gh = prob.G(h)
        ggamma = -prob.dsum(Gh * self.n)
        HGh = self.H(Gh) if self.h is not None else None
        gw = []
        for a in range(prob.k):
            v = -float((prob.d(h, a) * self.u[a]).sum())
            if HGh is not None:
                v -= float((HGh[a] * prob.d(self.z, a)).sum())
            gw.append(v)
        return h, ggamma, gw, (res, it)


def dual_out(prob_w, u, iso, k):
    """The dual in the C kernels' form: u (isotropic) or q_a = w_a D_a^T u_a."""
    if iso:
        return u
    prob = Problem(u[0], prob_w, False)
    return torch.stack([prob.w[a] * prob.dt(u[a], a) for a in range(k)])


def dual_in(q, w, iso, k, gamma):
    """A warm start in the C kernels' form back to u."""
    if iso:
        return q
    us = []
    for a in range(k):
        if w[a] > 0:
            us.append(-torch.cumsum(q[a], dim=1 + a) / w[a])
        else:
            us.append(torch.zeros_like(q[a]))
    return torch.stack(us)


def info(gap, iters, converged, dual):
    return IterInfo(float(gap.sum()), int(iters), 0, bool(converged), dual)
