"""The reproducer of feedback U6: spectral MCP for matrix completion.

    uv run python bench/repro_u6_spectral_mcp.py [--iters 8000] [--zip ml-100k.zip]

The setup of the report's experiment (docs/feedback-2026-10.md, U6): MovieLens
100k restricted to the 300 most active users and the 300 most rated movies,
70% of their ratings for training, centred by the training mean, and

    min_W  (1/2) sum_{(i,j) in train} (W_ij - R_ij)^2 + lam g(W),

with g the nuclear norm or MCP of the singular values (threshold 1, concavity
b, so the prox at step tau thresholds at tau lam and b). FISTA with gradient
restart at tau = 1/L, as the report ran it, and proximal gradient at 0.99/L.

With MCP the objective keeps falling while the iterates grow without bound,
for both methods: the penalty is constant above b and the data term does not
see the unobserved entries, so the objective is not coercive and the lowest
values lie at ever larger W. With the nuclear norm both settle. The data are
downloaded into build/external/ on the first run.
"""

import argparse
import pathlib
import urllib.request
import warnings
import zipfile

import numpy as np

import fastprox as fp

ML100K = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def ratings(path):
    with zipfile.ZipFile(path) as z:
        raw = np.loadtxt(z.open("ml-100k/u.data"), dtype=int)
    u, m, r = raw[:, 0], raw[:, 1], raw[:, 2].astype(float)
    top_u, top_m = np.argsort(-np.bincount(u))[:300], np.argsort(-np.bincount(m))[:300]
    ui, mi = {v: i for i, v in enumerate(top_u)}, {v: i for i, v in enumerate(top_m)}
    keep = np.array([a in ui and b in mi for a, b in zip(u, m)])
    rows, cols, r = np.array([ui[a] for a in u[keep]]), np.array([mi[b] for b in m[keep]]), r[keep]
    train = np.random.default_rng(0).permutation(len(r))[: int(0.7 * len(r))]
    R, M = np.zeros((300, 300)), np.zeros((300, 300))
    R[rows[train], cols[train]] = r[train] - r[train].mean()
    M[rows[train], cols[train]] = 1.0
    return R, M


def run(op, R, M, lam, tau, fista, iters, marks):
    obj = lambda W: 0.5 * np.sum((M * (W - R)) ** 2) + lam * op.value(W)  # noqa: E731
    W = M * R
    V, t, out = W, 1.0, []
    for k in range(1, iters + 1):
        Wn = op.prox(V - tau * M * (V - R), tau * lam)
        if not fista:
            V = Wn
        elif np.vdot(V - Wn, Wn - W) > 0:  # gradient restart
            t, V = 1.0, Wn
        else:
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            V, t = Wn + (t - 1) / tn * (Wn - W), tn
        W = Wn
        if k in marks:
            out.append((k, obj(W), np.linalg.norm(W), np.abs(W[M == 0]).max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--zip", type=pathlib.Path, help="ml-100k.zip, if already downloaded")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--b", type=float, default=30.0)
    a = ap.parse_args()
    path = a.zip
    if path is None:
        path = pathlib.Path(__file__).resolve().parents[1] / "build" / "external" / "ml-100k.zip"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(ML100K, path)
    R, M = ratings(path)
    warnings.simplefilter("ignore", fp.AccuracyWarning)
    marks = sorted({a.iters // 8, a.iters // 4, a.iters // 2, a.iters} - {0})
    print(f"MovieLens 300 x 300, {int(M.sum())} training ratings; lam {a.lam}, MCP b {a.b}")
    print("objective / |W|_F / largest unobserved |W_ij| (the centred ratings lie between -2.7 and 1.3)")
    for name, op in (("MCP", fp.Spectral(fp.MCP(1.0, a.b))), ("nuclear", fp.NuclearNorm(1.0))):
        for method, tau, fista in (("FISTA + restart", 1.0, True), ("prox. gradient", 0.99, False)):
            cells = "  ".join(f"k={k}: {f:.6g} / {n:.4g} / {u:.3g}"
                              for k, f, n, u in run(op, R, M, a.lam, tau, fista, a.iters, marks))
            print(f"  {name:8s} {method:16s} tau {tau}:  {cells}", flush=True)


if __name__ == "__main__":
    main()
