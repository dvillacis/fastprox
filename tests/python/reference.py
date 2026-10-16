"""Independent NumPy reference implementations and optimality certificates."""

import numpy as np


def soft(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def elastic_net(x, gamma, l1, l2):
    return soft(x, gamma * l1) / (1.0 + gamma * l2)


def huber(x, gamma, lam, delta):
    c = delta + gamma * lam
    return np.where(np.abs(x) <= c, x * delta / c, x - gamma * lam * np.sign(x))


def group_l2(x, gamma, lam, groups, c):
    z = np.zeros_like(x)
    for g in np.unique(groups):
        m = groups == g
        nrm = np.linalg.norm(x[m])
        tau = gamma * lam * c[g if c.size > 1 else 0]
        if nrm > tau:
            z[m] = (1 - tau / nrm) * x[m]
    return z


def l2_ball(x, r):
    n = np.linalg.norm(x)
    return x if n <= r else x * (r / n)


def simplex(x, r):
    """Sort-based projection (Held, Wolfe and Crowder)."""
    u = np.sort(x)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, x.size + 1)
    rho = np.nonzero(u - (css - r) / j > 0)[0][-1]
    tau = (css[rho] - r) / (rho + 1)
    return np.maximum(x - tau, 0.0)


def l1_ball(x, r, w=None):
    w = np.ones_like(x) if w is None else np.broadcast_to(w, x.shape)
    a = np.abs(x)
    if (w * a).sum() <= r:
        return x.copy()
    if r == 0:
        return np.zeros_like(x)
    order = np.argsort(-a / w)
    s1 = np.cumsum((w * a)[order])
    s2 = np.cumsum((w * w)[order])
    taus = (s1 - r) / s2
    t = a[order] / w[order]
    k = np.nonzero(t > taus)[0][-1]
    tau = taus[k]
    return np.sign(x) * np.maximum(a - tau * w, 0.0)


def linf(x, gamma, lam):
    return x - l1_ball(x, gamma * lam)


def isotonic(x, w=None, increasing=True):
    s = 1.0 if increasing else -1.0
    y = s * np.asarray(x, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    blocks = []  # [sum w y, sum w, count]
    for yi, wi in zip(y, w):
        blocks.append([wi * yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            a, b = blocks[-2], blocks.pop()
            a[0] += b[0]
            a[1] += b[1]
            a[2] += b[2]
    return s * np.concatenate([np.full(c, sy / sw) for sy, sw, c in blocks])


def tv1d_kkt_residual(x, z, gamma, w):
    """Largest violation of the TV1D optimality conditions, relative to scale.

    With u_e = sum_{i<=e} (x_i - z_i): |u_e| <= gamma w_e, u_{n-1} = 0, and
    u_e = -gamma w_e where z jumps up, +gamma w_e where it jumps down.
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    n = x.size
    w = np.broadcast_to(np.asarray(w, float), (max(n - 1, 0),))
    u = np.cumsum(x - z)
    lam = gamma * w
    scale = 1.0 + np.abs(x).max()
    tol = 1e-12 * scale * n
    res = abs(u[-1])
    if n > 1:
        ue = u[:-1]
        res = max(res, np.max(np.abs(ue) - lam, initial=0.0))
        jump = np.diff(z)
        up = jump > tol
        down = jump < -tol
        res = max(res, np.max(np.abs(ue[up] + lam[up]), initial=0.0))
        res = max(res, np.max(np.abs(ue[down] - lam[down]), initial=0.0))
    return res / scale


def grid_prox(x, gamma, psi, n=4001, rounds=7):
    """Minimiser of (z - x)^2 / 2 + gamma psi(z) for each entry of x, by a dense
    grid on [-|x| - 1, |x| + 1] (plus 0) refined around the best point:
    an independent check of the nonconvex maps, accurate to about
    1e-12 relative after the refinement rounds."""
    x = np.asarray(x, float)
    flat = x.reshape(-1)
    out = np.empty_like(flat)
    for i, xi in enumerate(flat):
        lo, hi = -abs(xi) - 1.0, abs(xi) + 1.0
        best = 0.0
        for r in range(rounds):
            g = np.append(np.linspace(lo, hi, n), 0.0)
            f = 0.5 * (g - xi) ** 2 + gamma * psi(g)
            best = g[np.argmin(f)]
            h = (hi - lo) / (n - 1)
            lo, hi = best - 4 * h, best + 4 * h
        out[i] = best
    return out.reshape(x.shape)


def nonconvex_objective(z, x, gamma, psi):
    return 0.5 * (z - x) ** 2 + gamma * psi(z)


def scad_psi(lam, a):
    def psi(z):
        t = np.abs(z)
        return np.where(t <= lam, lam * t,
                        np.where(t <= a * lam, (2 * a * lam * t - t * t - lam * lam) / (2 * (a - 1)),
                                 (a + 1) * lam * lam / 2))
    return psi


def mcp_psi(lam, b):
    def psi(z):
        t = np.abs(z)
        return np.where(t <= b * lam, lam * t - t * t / (2 * b), b * lam * lam / 2)
    return psi


def hyperplane_box(x, a, b, lo=-np.inf, hi=np.inf, inequality=False):
    """Projection onto {a^T z = b (or <= b), lo <= z <= hi} by bisection on the
    multiplier mu of z = clip(x - mu a), down to adjacent doubles."""
    a, lo, hi = (np.broadcast_to(np.asarray(v, np.float64), x.shape) for v in (a, lo, hi))
    z0 = np.clip(x, lo, hi)
    if inequality and a @ z0 <= b:
        return z0

    def phi(mu):
        return a @ np.clip(x - mu * a, lo, hi) - b

    l, h = -1.0, 1.0
    while phi(l) < 0:
        l *= 2
    while phi(h) > 0:
        h *= 2
    while True:
        m = 0.5 * (l + h)
        if not l < m < h:
            break
        if phi(m) > 0:
            l = m
        else:
            h = m
    mu = l if abs(phi(l)) <= abs(phi(h)) else h
    return np.clip(x - mu * a, lo, hi)


def prox_max(x, c):
    """Beck, Example 6.49: x - c P_simplex(x / c)."""
    return x - c * simplex(x / c, 1.0)


def prox_sum_largest(x, c, k, absolute=False):
    """Beck, Examples 6.50 and 6.51: x - c P_C(x / c)."""
    y = x / c
    if absolute:
        return x - c * np.sign(y) * hyperplane_box(np.abs(y), 1.0, k, 0.0, 1.0, inequality=True)
    return x - c * hyperplane_box(y, 1.0, k, 0.0, 1.0)


def sum_largest(z, k, absolute=False):
    v = np.abs(z) if absolute else z
    return np.sort(v)[::-1][:k].sum()
