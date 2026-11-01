"""Review of the SCD subspaces of 1.0.2 (2026-10-28); diagnostic findings, not
regression tests. The specs that address them are in docs/scd-review-2026-10.md.

Run from the repository:
  uv run python reports/scd_review_2026_10_28.py

For each case it builds the dense Jacobian J from `jvp`, the projector P_L
from `ctx.structure().project`, and compares range J with L (guarantee 1),
the spectrum of J with [0, 1] (a Jacobian of a convex prox), and, for the
image operators, the piece the structure selects with the Jacobians at nearby
points.
"""
import warnings

import fastprox as fp
import numpy as np

warnings.simplefilter("ignore")


def dense(op, ctx, shape, **kw):
    n = int(np.prod(shape))
    return np.column_stack([op.jvp(ctx, e.reshape(shape), **kw).ravel() for e in np.eye(n)])


def projector(ctx, shape, **kw):
    s = ctx.structure(**kw)
    n = int(np.prod(shape))
    return np.column_stack([s.project(e.reshape(shape)).ravel() for e in np.eye(n)]), s


def range_projector(J, tol=1e-14):
    U, S, _ = np.linalg.svd(J)
    Q = U[:, S > tol * max(S.max(), 1.0)]
    return Q @ Q.T


def report(name, op, x, gamma=1.0, **kw):
    z, ctx = op.prox(x, gamma, keep=True)
    J = dense(op, ctx, x.shape, **kw)
    P, s = projector(ctx, x.shape, **kw)
    ev = np.linalg.eigvals(J).real
    rank = np.linalg.matrix_rank(J, tol=1e-9)
    gap = np.abs(range_projector(J) - P).max()
    ok = gap < 1e-8 and ev.min() > -1e-12 and ev.max() < 1 + 1e-12
    print(f"  {name:44s} rank J {rank:2d}  dim L {int(np.ravel(s.dim)[0]):2d}  "
          f"|P_rgeJ - P_L| {gap:.1e}  eig [{ev.min():+.2e}, {ev.max():.4f}]  {'ok' if ok else 'FAIL'}")
    return ok


def grid_edges(h, w):
    i = np.arange(h * w).reshape(h, w)
    return np.r_[np.c_[i[:, :-1].ravel(), i[:, 1:].ravel()], np.c_[i[:-1].ravel(), i[1:].ravel()]]


def main():
    rng = np.random.default_rng(20261028)
    u = rng.normal(size=4)
    u /= np.linalg.norm(u)

    print("S1. Group l2 forced live at its kink (tau = gamma lam = 1)")
    g = fp.GroupL2(1.0)
    report("||x|| = tau, branch +1", g, 1.0 * u, branch=1, kink_tol=1e-12)
    report("||x|| = tau (1 - 1e-6), branch +1, kink_tol 1e-5", g, (1 - 1e-6) * u, branch=1, kink_tol=1e-5)
    report("||x|| = tau (1 + 1e-6), default (live)", g, (1 + 1e-6) * u)
    report("||x|| = tau, branch -1", g, 1.0 * u, branch=-1, kink_tol=1e-12)

    print("S2. 1D TV with l2 of differences on its inside/outside boundary")
    op = fp.TV1DL2()
    D = np.diff(np.eye(5), axis=0)
    worst, bad = 0.0, 0
    for k in range(200):
        xb = rng.normal(size=5)
        xb -= xb.mean()
        xb /= np.linalg.norm(np.linalg.solve(D @ D.T, D @ xb))  # ||u0|| = gamma = 1 to rounding
        _, ctx = op.prox(xb, 1.0, keep=True)
        ev = np.linalg.eigvals(dense(op, ctx, (5,))).real
        worst = max(worst, ev.max())
        bad += ev.max() > 1 + 1e-12 or ev.min() < -1e-12
        if bad == 1 and (ev.max() > 1 + 1e-12 or ev.min() < -1e-12):
            report("first such x (seeded)", op, xb)
            bad += 1
    print(f"  {bad - 1 if bad else 0}/200 points on the boundary give eigenvalues outside [0, 1]; largest {worst:.4f}")
    for f in (1 + 1e-12, 1 + 1e-9, 1 + 1e-6):
        report(f"just outside, x scaled by {f:.12f}", op, xb * f)

    print("S3. Weighted isotonic regression: s.project is the diag(w) projector")
    w = np.array([1.0, 3.0, 2.0, 1.0, 1.0])
    iso = fp.Isotonic(weights=w)
    x = np.array([3.0, 1.0, 2.0, 0.0, 5.0])
    _, ctx = iso.prox(x, 1.0, keep=True)
    P, s = projector(ctx, x.shape)
    print(f"  asymmetry of s.project: {np.abs(P - P.T).max():.2f} (the SCD pair needs the Euclidean P_L)")

    print("S4. Anisotropic TV and graph TV on the same 6 x 6 grid")
    h = 6
    aniso, graph = fp.TVAniso(), fp.GraphTV(h * h, grid_edges(h, h))
    differ = {"noisy": 0, "tied": 0}
    for t in range(150):
        x = rng.integers(0, 3, size=(h, h)).astype(float) + 0.3 * rng.normal(size=(h, h))
        kind = "noisy"
        if t % 3 == 0:
            x, kind = np.round(x), "tied"
        gamma = float(rng.choice([0.1, 0.3, 0.7]))
        _, ca = aniso.prox(x, gamma, keep=True, tol=1e-20, max_iter=100000)
        _, cg = graph.prox(x.ravel(), gamma, keep=True)
        differ[kind] += np.abs(dense(aniso, ca, (h, h)) - dense(graph, cg, (h * h,))).max() > 1e-8
    print(f"  Jacobians differ on {differ['noisy']}/100 noisy and {differ['tied']}/50 tied images")
    # A point where the prox is differentiable (every nearby point has one
    # Jacobian) and the anisotropic piece is not that Jacobian.
    x = np.array([[2, 2, 2, 2, 1, 0], [2, -1, 1, 0, 1, 0], [0, 1, 2, 1, 1, 1],
                  [1, 0, 1, 1, 1, 0], [1, 2, 0, 2, 1, 2], [2, 1, 1, 2, 0, 0]], dtype=float)
    gamma = 0.3
    za, ca = aniso.prox(x, gamma, keep=True, tol=1e-20, max_iter=100000)
    zg, cg = graph.prox(x.ravel(), gamma, keep=True)
    err = {"aniso": 0.0, "graph": 0.0}
    for _ in range(20):
        d = rng.normal(size=(h, h))
        fd = (graph.prox(x.ravel() + 1e-6 * d.ravel(), gamma) - graph.prox(x.ravel() - 1e-6 * d.ravel(), gamma)) / 2e-6
        err["aniso"] = max(err["aniso"], np.abs(aniso.jvp(ca, d).ravel() - fd).max())
        err["graph"] = max(err["graph"], np.abs(graph.jvp(cg, d.ravel()) - fd).max())
    nearby = {dense(graph, graph.prox(x.ravel() + 1e-7 * rng.normal(size=h * h), gamma, keep=True)[1],
                    (h * h,)).round(6).tobytes() for _ in range(200)}
    print(f"  counterexample: |z_aniso - z_graph| {np.abs(za.ravel() - zg).max():.1e}, regions "
          f"{int(np.ravel(ca.structure().nregions)[0])} and {int(np.ravel(cg.structure().nregions)[0])}, "
          f"{len(nearby)} distinct Jacobian(s) at 200 points within 1e-7, deriv.exact {ca.deriv.exact}")
    print(f"  max |jvp - central difference| over 20 directions: anisotropic {err['aniso']:.2f}, "
          f"graph {err['graph']:.1e}")


if __name__ == "__main__":
    main()
