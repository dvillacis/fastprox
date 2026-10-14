"""Agreement with a generic solver (CVXPY + Clarabel) on stored fixtures for
row lengths up to 200 (docs/spec.md, Section 11). Regenerate the fixtures
with tests/fixtures/generate.py."""

import pathlib

import numpy as np
import pytest

from fixture_cases import CASES

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "reference.npz"


def load():
    data = np.load(FIXTURES)
    problems = {}
    for key in data.files:
        k, field = key.split("/", 1)
        problems.setdefault(k, {"param": {}})
        if field.startswith("param/"):
            problems[k]["param"][field[6:]] = data[key]
        else:
            problems[k][field] = data[key]
    return [problems[k] for k in sorted(problems)]


PROBLEMS = load()


def discrepancy(p):
    op = CASES[str(p["case"])][1](p["param"])
    z = op.prox(p["x"], float(p["gamma"]))
    return np.abs(z - p["z"]).max() / (1 + np.abs(p["x"]).max())


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_generic_solver(case):
    probs = [p for p in PROBLEMS if str(p["case"]) == case]
    assert probs, f"no fixtures for {case}"
    worst = max(discrepancy(p) for p in probs)
    assert worst < 1e-8
