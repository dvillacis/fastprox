"""Regenerate the CVXPY + Clarabel reference fixtures (offline).

    uv run --group ref python tests/fixtures/generate.py

Writes tests/fixtures/reference.npz: for every case in
tests/python/fixture_cases.py, every row length in SIZES and every input
scale in SCALES, the input x, the step gamma, the parameters and the
solver's solution z. The fixtures are checked into the repository so the
test suite does not need CVXPY.
"""

import pathlib
import sys

import cvxpy as cp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from fixture_cases import CASES, SCALES, SIZES  # noqa: E402

# Clarabel at one tolerance is not reliably accurate to 1e-10 on every
# problem: tight tolerances sometimes stop "optimal_inaccurate", loose ones
# leave a larger error. Solve at several and keep the best feasible point by
# objective value (an independent criterion: fastprox is never consulted).
TOLERANCES = (1e-14, 1e-12, 1e-10)


def solve(problem, x, gamma, params):
    z = cp.Variable(x.size)
    out = problem(cp, z, x, gamma, params)
    if len(out) == 3:
        objective = out[2]
    else:
        objective = (out[0] if out[0] is not None else 0) + 0.5 * cp.sum_squares(z - x)
    prob = cp.Problem(cp.Minimize(objective), out[1])
    best = None
    for tol in TOLERANCES:
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
                       max_iter=500)
        except cp.SolverError:
            continue
        if z.value is None:
            continue
        viol = max((c.violation().max() for c in prob.constraints), default=0.0)
        if viol > 1e-9 * (1 + np.abs(x).max()):
            continue
        cand = (prob.objective.value, z.value.copy(), prob.status)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        raise RuntimeError("no accurate solution")
    return best[1], best[2]


def main():
    rng = np.random.default_rng(20261014)
    data = {}
    k = 0
    for name, (sample, _, problem) in CASES.items():
        for n in SIZES:
            for scale in SCALES:
                x = rng.normal(scale=scale, size=n)
                gamma = float(rng.uniform(0.2, 2.0))
                params = sample(rng, n)
                z, status = solve(problem, x, gamma, params)
                key = f"{k:04d}"
                data[f"{key}/case"] = np.array(name)
                data[f"{key}/x"] = x
                data[f"{key}/gamma"] = np.array(gamma)
                data[f"{key}/z"] = z
                for pk, pv in params.items():
                    data[f"{key}/param/{pk}"] = pv
                if status != "optimal":
                    print(f"{key} {name} n={n}: {status}")
                k += 1
    np.savez_compressed(HERE / "reference.npz", **data)
    print(f"wrote {k} problems to {HERE / 'reference.npz'}")


if __name__ == "__main__":
    main()
