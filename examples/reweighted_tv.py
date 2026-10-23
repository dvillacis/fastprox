"""Nonconvex TV^q denoising by iteratively reweighted l1 (0.3).

    minimise  1/2 |z - y|^2 + gamma sum_p (|(D z)_p| + eps)^q,   0 < q < 1,

by the iteratively reweighted l1 method of P. Ochs, A. Dosovitskiy, T. Brox
and T. Pock, "On iteratively reweighted algorithms for nonsmooth nonconvex
optimization in computer vision", SIAM J. Imaging Sci. 8 (2015) 331-372:
t -> (t + eps)^q is concave, so its tangent at the current |(D z_k)_p|
majorises it, and each step is a weighted isotropic TV prox,

    z_{k+1} = prox of gamma sum_p alpha_p |(D z)_p|,
    alpha_p = q (|(D z_k)_p| + eps)^(q - 1),

which ``fp.TVIso(pixel_weights=alpha)`` computes, warm-started from the
previous dual (projected onto the new balls). The objective decreases at
every step, and the iterates settle at a stationary point: a fixed point of
the step. Small jumps get large weights and are flattened, large ones small
weights and keep their height, so TV^q loses less contrast than TV: on the
64^2 test image (three flat regions plus noise 0.1), 42.6 dB against 36.7 dB
for TV, each at its best step, after 40 steps of a relative size 3e-8.
"""

import numpy as np

import fastprox as fp

Q, EPS = 0.5, 1e-2


def images(rng, n=64):
    a, b = np.mgrid[0:n, 0:n] / n
    clean = ((a - 0.5) ** 2 + (b - 0.45) ** 2 < 0.08) + 0.5 * (b > 0.7) - 0.4 * ((a < 0.25) & (b < 0.3))
    return clean, clean + 0.1 * rng.normal(size=clean.shape)


def grad_norm(z):
    g0 = np.zeros_like(z)
    g1 = np.zeros_like(z)
    g0[:-1] = z[1:] - z[:-1]
    g1[:, :-1] = z[:, 1:] - z[:, :-1]
    return np.hypot(g0, g1)


def objective(z, y, gamma):
    return 0.5 * ((z - y) ** 2).sum() + gamma * ((grad_norm(z) + EPS) ** Q).sum()


def psnr(z, clean):
    return 10 * np.log10(np.ptp(clean) ** 2 / np.mean((z - clean) ** 2))


def reweighted(y, gamma, iters=40):
    """IRL1 from the TV solution; returns z, the objectives and the last step size."""
    z = fp.TVIso(2).prox(y, gamma, tol=1e-8, max_iter=10 ** 5)
    dual, objs, step = None, [objective(z, y, gamma)], np.inf
    for _ in range(iters):
        alpha = Q * (grad_norm(z) + EPS) ** (Q - 1)
        zn, info = fp.TVIso(2, pixel_weights=alpha).solve(y, gamma, tol=1e-10, max_iter=10 ** 5, warm_start=dual)
        dual = info.dual
        step = np.linalg.norm(zn - z) / np.linalg.norm(z)
        z = zn
        objs.append(objective(z, y, gamma))
    return z, np.array(objs), step


def main():
    rng = np.random.default_rng(0)
    clean, y = images(rng)
    # Each method at its best step on a small grid (an oracle, for comparison).
    tv = max(((psnr(fp.TVIso(2).prox(y, g, tol=1e-8, max_iter=10 ** 5), clean), g) for g in (0.06, 0.08, 0.1, 0.12, 0.15)))
    runs = [(reweighted(y, g), g) for g in (0.05, 0.08, 0.12)]
    (z, objs, step), g = max(runs, key=lambda r: psnr(r[0][0], clean))
    rise = float(np.max(np.diff(objs) / np.abs(objs[1:])))
    print(f"TV:   PSNR {tv[0]:.2f} dB (gamma {tv[1]})")
    print(f"TV^q: PSNR {psnr(z, clean):.2f} dB (q = {Q}, gamma {g}); objective {objs[0]:.3f} -> {objs[-1]:.3f}, "
          f"largest relative rise {max(rise, 0.0):.1e}, last step |z_k+1 - z_k| / |z_k| = {step:.1e}")
    return tv[0], psnr(z, clean), objs, step


if __name__ == "__main__":
    main()
