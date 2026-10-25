"""Independent mathematical audit; diagnostic findings, not regression tests.

Run from the repository:
  uv run --no-sync python reports/mathematical_audit_2026_10_25.py

Requires the existing reference environment (NumPy, CVXPY/Clarabel, mpmath).
No library internals or test reference implementations are used.
"""
import json
import platform
import subprocess
import warnings
from importlib.metadata import version
from pathlib import Path

import cvxpy as cp
import fastprox as fp
import mpmath as mp
import numpy as np
from scipy.optimize import minimize_scalar


def ordinary_scale_references():
    rng = np.random.default_rng(252026)
    records = []
    for trial in range(5):
        x = rng.normal(size=6)
        w = rng.uniform(0.3, 1.5, 6)
        ew = rng.uniform(0.3, 1.5, 5)
        gamma = float(rng.uniform(0.1, 0.8))
        v = cp.Variable(6)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 4], [0, 5]])
        gw = rng.uniform(0.3, 1.5, len(edges))
        a = rng.normal(size=6)
        ow = np.linspace(1.2, 0.2, 6)
        # sum_k w_k |v|_(k) = w_last ||v||_1 + sum_k (w_k-w_{k+1}) sum_largest(|v|, k).
        owl = ow[-1] * cp.norm1(v) + sum(
            (ow[k - 1] - ow[k]) * cp.sum_largest(cp.abs(v), k) for k in range(1, 6)
        )
        families = [
            ("soft", fp.Soft(0.7, w), 0.7 * cp.sum(cp.multiply(w, cp.abs(v))), []),
            ("elastic_net", fp.ElasticNet(0.4, 0.8), 0.4 * cp.norm1(v) + 0.4 * cp.sum_squares(v), []),
            ("huber", fp.Huber(0.7, 0.6), 0.7 / 1.2 * cp.sum(cp.huber(v, 0.6)), []),
            ("group_l2", fp.GroupL2(0.7), 0.7 * cp.norm(v, 2), []),
            ("tree_l2", fp.TreeL2([[0, 1], [0, 1, 2, 3, 4, 5]], [0.4, 0.8]),
             0.4 * cp.norm(v[:2], 2) + 0.8 * cp.norm(v, 2), []),
            ("simplex", fp.Simplex(1.3), 0, [v >= 0, cp.sum(v) == 1.3]),
            ("weighted_l1_ball", fp.L1Ball(1.3, w), 0, [cp.sum(cp.multiply(w, cp.abs(v))) <= 1.3]),
            ("l2_ball", fp.L2Ball(1.3), 0, [cp.norm(v, 2) <= 1.3]),
            ("isotonic", fp.Isotonic(), 0, [v[:-1] <= v[1:]]),
            ("sorted_l1", fp.SortedL1(ow), owl, []),
            ("tv1d", fp.TV1D(ew), cp.sum(cp.multiply(ew, cp.abs(cp.diff(v)))), []),
            ("tv1d_l2", fp.TV1DL2(ew), cp.norm(cp.multiply(ew, cp.diff(v)), 2), []),
            ("tv_ball", fp.TV1DBall(0.8, ew), 0, [cp.sum(cp.multiply(ew, cp.abs(cp.diff(v)))) <= 0.8]),
            ("hyperplane_box", fp.Hyperplane(a, 0.0, -0.8, 0.9), 0,
             [a @ v == 0, v >= -0.8, v <= 0.9]),
            ("soc", fp.SOC(), 0, [cp.norm(v[1:], 2) <= v[0]]),
            ("log_barrier", fp.LogBarrier(0.7), -0.7 * cp.sum(cp.log(v)), []),
            ("graph_tv", fp.GraphTV(6, edges, gw),
             cp.sum(cp.multiply(gw, cp.abs(v[edges[:, 0]] - v[edges[:, 1]]))), []),
        ]
        for name, op, penalty, constraints in families:
            objective = 0.5 * cp.sum_squares(v - x) + gamma * penalty
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9,
                       max_iter=500)
            if prob.status != cp.OPTIMAL:
                prob.solve(solver="SCS", eps=1e-8, max_iters=100000)
            assert prob.status == cp.OPTIMAL, (name, prob.status)
            z = op.prox(x, gamma)
            err = float(np.linalg.norm(z - v.value) / (1 + np.linalg.norm(x)))
            # Conic solver coordinate errors need not scale as its objective gap.
            assert err < 2e-5, (name, trial, err)
            records.append({"family": name, "trial": trial, "normalized_l2_error": err,
                            "solver": prob.solver_stats.solver_name, "status": prob.status})
    # Independently model forward differences with zero Neumann boundaries.
    for trial in range(3):
        x = rng.normal(size=(3, 4))
        v = cp.Variable(x.shape)
        d0 = cp.vstack([v[1:] - v[:-1], np.zeros((1, 4))])
        d1 = cp.hstack([v[:, 1:] - v[:, :-1], np.zeros((3, 1))])
        flat = cp.vstack([cp.reshape(d0, (12,), order="C"), cp.reshape(d1, (12,), order="C")])
        for name, op, penalty in [
            ("tv_aniso", fp.TVAniso(2), cp.norm1(d0) + cp.norm1(d1)),
            ("tv_iso", fp.TVIso(2), cp.sum(cp.norm(flat, 2, axis=0))),
        ]:
            prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(v - x) + 0.3 * penalty))
            prob.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
            assert prob.status == cp.OPTIMAL
            z, info = op.solve(x, 0.3, tol=1e-14, max_iter=100000, warn=False)
            assert info.converged
            err = float(np.linalg.norm(z - v.value) / (1 + np.linalg.norm(x)))
            assert err < 2e-5, (name, trial, err)
            records.append({"family": name, "trial": trial, "normalized_l2_error": err})
    return records


def derivative_checks():
    rng = np.random.default_rng(252027)
    x = rng.normal(size=6) * 2
    ops = [fp.Soft(0.4), fp.ElasticNet(0.3, 0.8), fp.Huber(0.5, 0.7), fp.GroupL2(0.7),
           fp.L2Ball(1.3), fp.Simplex(1.4), fp.L1Ball(1.5), fp.Isotonic(),
           fp.SortedL1(np.linspace(1.2, 0.2, 6)), fp.TV1D(), fp.TV1DL2(),
           fp.TV1DBall(1.5), fp.SOC(), fp.LogBarrier(0.7), fp.SCAD(0.6, 3.7),
           fp.MCP(0.6, 3.0), fp.Lq(0.3, 0.5), fp.Lq(0.3, 0.67), fp.LogPenalty(0.3, 0.5)]
    records = []
    for op in ops:
        _, ctx = op.prox(x, 0.4, keep=True)
        columns = [op.jvp(ctx, d) for d in np.eye(6)]
        J = np.column_stack(columns)
        h = rng.normal(size=6)
        g = rng.normal(size=6)
        eps = 1e-6
        fd = (op.prox(x + eps * h, 0.4) - op.prox(x - eps * h, 0.4)) / (2 * eps)
        tangent = op.jvp(ctx, h)
        gx, _ = op.vjp(ctx, g)
        fd_error = float(np.linalg.norm(tangent - fd) / (1 + np.linalg.norm(fd)))
        adjoint_error = float(abs(g @ tangent - gx @ h))
        symmetry_error = float(np.max(np.abs(J - J.T)))
        assert fd_error < 1e-6, (type(op).__name__, fd_error)
        assert adjoint_error < 1e-10, (type(op).__name__, adjoint_error)
        assert symmetry_error < 1e-10, (type(op).__name__, symmetry_error)
        eigenvalues = np.linalg.eigvalsh(J)
        assert eigenvalues.min() >= -1e-10
        if not isinstance(op, (fp.SCAD, fp.MCP, fp.Lq, fp.LogPenalty)):
            assert eigenvalues.max() <= 1 + 1e-10
        records.append({"family": repr(op), "fd_error": fd_error,
                        "adjoint_error": adjoint_error, "symmetry_error": symmetry_error,
                        "jacobian_eigenvalues": eigenvalues.tolist()})
    return records


def nonconvex_scalar_checks():
    """Minimize independently on intervals of convexity/concavity.

    For an even nonnegative increasing penalty a minimizer has the sign of x
    and lies in [0, |x|]. Concave intervals attain minima at their ends;
    the remaining intervals below are convex and can be minimized directly.
    This checks objectives rather than requiring agreement of tie choices.
    """
    rng = np.random.default_rng(252028)
    records = []
    for trial in range(20):
        x = float(rng.uniform(0.01, 5))
        lam = float(rng.uniform(0.1, 1))
        gamma = float(rng.uniform(0.1, 0.8))
        eps, a, b = 0.6, 3.7, 3.0
        cases = []
        for q in (0.1, 0.5, 2 / 3, 0.9):
            cut = (gamma * lam * q * (1 - q)) ** (1 / (2 - q))
            cases.append((f"lq_{q}", fp.Lq(lam, q), lambda z, q=q: lam * z ** q, [cut]))
        cases += [
            ("log", fp.LogPenalty(lam, eps), lambda z: lam * np.log1p(z / eps),
             [max(0., np.sqrt(gamma * lam) - eps)]),
            ("scad", fp.SCAD(lam, a),
             lambda z: lam * z if z <= lam else
             ((2 * a * lam * z - z * z - lam * lam) / (2 * (a - 1)) if z <= a * lam else
              (a + 1) * lam * lam / 2), [lam, a * lam]),
            ("mcp", fp.MCP(lam, b),
             lambda z: lam * z - z * z / (2 * b) if z <= b * lam else b * lam * lam / 2,
             [b * lam]),
        ]
        for name, op, penalty, cuts in cases:
            objective = lambda z: 0.5 * (z - x) ** 2 + gamma * penalty(z)
            bounds = sorted(set([0., x] + [t for t in cuts if 0 < t < x]))
            candidates = list(bounds)
            for lo, hi in zip(bounds, bounds[1:]):
                solution = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                                           options={"xatol": 1e-13})
                assert solution.success
                candidates.append(float(solution.x))
            ref = min(objective(t) for t in candidates)
            z = float(op.prox(np.array([x]), gamma)[0])
            discrepancy = float(objective(abs(z)) - ref)
            assert abs(discrepancy) <= 1e-10 * (1 + abs(ref)), (name, trial, discrepancy)
            records.append({"family": name, "trial": trial, "objective_discrepancy": discrepancy})
    return records


def findings():
    records = []
    h = np.array([[0., 1.], [1., 0.]])
    x = np.diag([1 + 1e-7, 1 - 2e-7])
    op = fp.NuclearNorm(1.)
    _, ctx = op.prox(x, keep=True)
    exact = float((ctx.z[0] - ctx.z[1]) / (ctx.s[0] - ctx.s[1]))
    got = float(op.jvp(ctx, h)[0, 1])
    records.append({"id": "M1", "observed": got, "expected": exact,
                    "reported_derivative_exact": bool(ctx.deriv.exact),
                    "reproduced": abs(got - exact) > 0.1})
    x32 = np.diag(np.array([1.001, 0.998], dtype=np.float32))
    _, ctx32 = op.prox(x32, keep=True)
    got32 = float(op.jvp(ctx32, h.astype(np.float32))[0, 1])
    exact32 = float((ctx32.z[0] - ctx32.z[1]) / (ctx32.s[0] - ctx32.s[1]))
    records.append({"id": "M1_float32", "observed": got32, "expected": exact32,
                    "reproduced": abs(got32 - exact32) > 0.1})
    op = fp.Isotonic([1., 3.])
    got = op.prox_conj(np.array([2., 0.]))
    records.append({"id": "M2", "observed": got.tolist(), "expected": [1., -1.],
                    "polar_sum_violation": float(got.sum()),
                    "reproduced": bool(abs(got.sum()) > 0.1)})
    for name, op, x, gamma, expected in [
        ("M3", fp.L0(1e-308), [2.], 1e308, [2.]),
        ("M4", fp.LogBarrier(1e280), [-1e308], 1., [1e-28]),
        ("M5", fp.TV1DL2(1e-180), [1., -1.], 1e180, [0., 0.]),
        ("M6_scad", fp.SCAD(1., 1e308), [3.], 1., [2.]),
        ("M6_mcp", fp.MCP(1., 1e308), [3.], 1., [2.]),
        ("M6_group_scad", fp.GroupSCAD(1., 1e308), [3.], 1., [2.]),
        ("M6_group_mcp", fp.GroupMCP(1., 1e308), [3.], 1., [2.]),
    ]:
        z, ctx = op.prox(np.array(x), gamma, keep=True)
        records.append({"id": name, "observed": z.tolist(), "expected": expected,
                        "reported_gap": float(ctx.info.gap),
                        "reproduced": not np.allclose(z, expected, rtol=1e-10, atol=0.)})
    # A 700-digit independent root avoids cancellation of x + sqrt(x*x+4 gl).
    with mp.workdps(700):
        x, gl = mp.mpf(-1e308), mp.mpf(1e280)
        root = 2 * gl / (mp.sqrt(x * x + 4 * gl) - x)
        assert abs(float(root) / 1e-28 - 1) < 1e-14
    # Distinguish rounding of the returned dtype from the internal solver gap.
    for Op in (fp.TVIso, fp.TVAniso):
        x = np.array([[0., 1.]], dtype=np.float32)
        z, info = Op(2).solve(x, 0.1, ztol=1e-12, max_iter=1000, warn=False)
        err = float(np.linalg.norm(z.astype(float) - np.array([[0.1, 0.9]])))
        records.append({"id": "M7_" + Op.__name__, "reported_zerr": float(info.zerr),
                        "returned_relative_error": err,  # ||x|| = 1
                        "reproduced": bool(err > info.zerr + 1e-12)})
    return records


def serializable(value):
    # JSON has no IEEE infinity; retain diagnostic values as strings.
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error", fp.AccuracyWarning)
        result = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "platform": platform.platform(),
            "fastprox": fp.__version__, "numpy": np.__version__, "cvxpy": cp.__version__,
            "extension_path": fp._C.__file__,
            "scipy": version("scipy"), "mpmath": version("mpmath"),
            "clarabel": version("clarabel"), "scs": version("scs"),
            "references": ordinary_scale_references(), "derivatives": derivative_checks(),
            "nonconvex_scalar_checks": nonconvex_scalar_checks(),
            "findings": findings(),
        }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(serializable(result), indent=2, allow_nan=False) + "\n")
    print(f"{len(result['references'])} independent convex solves passed")
    print(f"{len(result['derivatives'])} derivative/adjoint/Jacobian checks passed")
    print(f"{len(result['nonconvex_scalar_checks'])} scalar nonconvex objective checks passed")
    for record in result["findings"]:
        print(record)
    print(f"Results: {path}")
