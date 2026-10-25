"""Small checks accompanying the theoretical audit; no library changes.

uv run --no-sync python reports/theoretical_audit_2026_10_25.py
The proofs and qualifications are in docs/theoretical-audit-2026-10-25.md.
"""
import json
from pathlib import Path

import fastprox as fp
import numpy as np


def main():
    out = {}
    # F(t) = ReLU(t) - ReLU(-t) = t. A valid ambient Jacobian
    # selection need not restrict to a valid derivative of this composition.
    op = fp.NonNeg()
    _, ctx = op.prox(np.zeros(2), keep=True)
    a = np.array([1., -1.])
    gx, _ = op.vjp(ctx, a)
    propagated = float(gx @ a)
    h = 1e-6
    f = lambda t: float(a @ op.prox(a * t))
    difference = (f(h) - f(-h)) / (2 * h)
    assert propagated == 2 and abs(difference - 1) < 1e-12
    out["restricted_chain_rule"] = {"propagated": propagated, "actual": difference}

    # At gamma*lambda = eps^2 = 1, the positive branch is exactly
    # (h + sqrt(h^2 + 4h))/2 for x = 1+h.
    op = fp.LogPenalty(1., 1.)
    rows = []
    for h in (1e-2, 1e-4, 1e-6):
        z = float(op.prox(np.array([1 + h]))[0])
        exact = 0.5 * (h + np.sqrt(h * h + 4 * h))
        assert abs(z - exact) < 1e-9 * exact
        rows.append({"h": h, "prox": z, "right_difference_quotient": z / h})
    out["critical_log_penalty"] = rows

    # Triangle flow: a strict feasible dual proves smooth full fusion.
    # Another optimal dual can have a saturated edge, triggering a mask.
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    K = np.zeros((3, 3))
    for i, (u, v) in enumerate(edges):
        K[i, u], K[i, v] = 1., -1.
    x = np.array([1., 0., -1.])
    strict_flow = np.array([1 / 3, 1 / 3, 2 / 3])
    assert np.allclose(K.T @ strict_flow, x)
    assert np.max(np.abs(strict_flow)) < 1
    op = fp.GraphTV(3, edges)
    z, ctx = op.prox(x, 1., keep=True)
    mask = ctx.kinks()
    J = np.column_stack([op.jvp(ctx, d) for d in np.eye(3)])
    assert np.allclose(z, 0) and np.allclose(J, np.ones((3, 3)) / 3)
    rng = np.random.default_rng(20261025)
    for _ in range(20):
        dx = 0.01 * rng.normal(size=3)
        assert np.max(np.abs(op.prox(x + dx) - np.mean(x + dx))) < 1e-12
    out["graph_dual_dependent_mask"] = {
        "mask": mask.tolist(), "strict_dual_flow": strict_flow.tolist(),
        "strict_dual_divergence": (K.T @ strict_flow).tolist(),
        "primal_jacobian": J.tolist(), "nearby_mean_checks": 20,
    }

    # A primal certificate gives no uniform Jacobian error bound.
    # Model: .5||z-x||^2 + |z0-z1|; feasible dual f=1.
    rows = []
    for eps in (1e-2, 1e-4, 1e-6):
        x = np.array([1 + eps, -1 - eps])
        candidate = np.zeros(2)
        q = np.array([1., -1.])
        residual = candidate - x + q
        gap = float(0.5 * (residual @ residual))
        exact = fp.TV1D().prox(x, 1.)
        assert np.allclose(exact, [eps, -eps], atol=1e-15, rtol=1e-9)
        true_J = np.eye(2)
        fused_J = np.ones((2, 2)) / 2
        discrepancy = float(np.linalg.norm(true_J - fused_J, ord=2))
        assert abs(discrepancy - 1) < 1e-12
        rows.append({"eps": eps, "gap": gap, "jacobian_error_norm": discrepancy})
    out["small_gap_wrong_region"] = rows

    # All the listed whole-space nonconvex penalties are nonnegative and
    # sublinear at infinity, hence their Fenchel conjugates are delta_{0}.
    # The API intentionally exposes a Moreau residual instead.
    residual = float(fp.MCP(1., 3.).prox_conj(np.array([2.]), 1.)[0])
    assert abs(residual - 0.5) < 1e-12
    out["nonconvex_conjugate"] = {"moreau_residual": residual, "true_fenchel_prox": 0.}

    # A completely well-defined prox need not make a lower solution isolated.
    op = fp.Soft(0.)
    _, ctx = op.prox(np.ones(2), 1., keep=True)
    J = np.column_stack([op.jvp(ctx, d) for d in np.eye(2)])
    fixed_point_matrix = np.eye(2) - J
    assert np.linalg.matrix_rank(fixed_point_matrix) == 0
    out["singular_bilevel_system"] = {"fixed_point_matrix": fixed_point_matrix.tolist()}

    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Results: {path}")


if __name__ == "__main__":
    main()
