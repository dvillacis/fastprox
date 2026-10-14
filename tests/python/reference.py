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
