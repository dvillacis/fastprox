"""Run every script in examples/ so they keep working as the API changes."""

import os
import pathlib
import subprocess
import sys

import pytest

DIR = pathlib.Path(__file__).parents[2] / "examples"
EXAMPLES = sorted(DIR.glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(script):
    env = {**os.environ, "MPLBACKEND": "Agg"}       # plt.show() does nothing
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_bilevel_finds_the_jumps():
    sys.path.insert(0, str(DIR))
    try:
        import bilevel_learning as ex
    finally:
        sys.path.pop(0)
    w = ex.main()
    assert sorted(int(i) + 1 for i in w.argsort()[:len(ex.JUMPS)]) == ex.JUMPS


def _load(name):
    sys.path.insert(0, str(DIR))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_image_denoising_learns_gamma():
    learned, best = _load("image_denoising").main(plot=False)
    assert abs(learned / best - 1) < 0.05        # grid spacing is 10%


def test_nonconvex_penalties_find_the_support():
    res = _load("sparse_recovery").main()
    l1_exact, l1_err = res["l1"]
    for name in ("MCP", "SCAD"):
        exact, err = res[name]
        assert exact == 1.0 and err < 0.5 * l1_err
    assert l1_exact < 0.5


def test_group_penalties_select_the_groups():
    res = _load("group_sparsity").main()
    for name in ("group MCP", "group SCAD"):
        extra_groups, _, err = res[name]
        assert extra_groups == 0 and err < res["lasso"][2]
    assert res["sparse group lasso"][1] < res["group lasso"][1]   # fewer extra features


def test_weighted_glasso_beats_the_graphical_lasso():
    res = _load("weighted_glasso").main()
    g_extra, g_missed, g_err = res["glasso 0.2"]
    a_extra, a_missed, a_err = res["adaptive"]
    _, m_missed, m_err = res["mcp"]
    # The weights cost a few missed edges in some problems (see the docstring).
    assert a_extra < g_extra / 2 and a_missed <= g_missed + 2 and m_missed <= a_missed
    assert m_err < a_err < g_err
    assert max(res["kkt"].values()) < 1e-5     # the convex solves are optimal


def test_autodiff_frameworks_learn():
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    (before, after, found), (gamma, best) = _load("autodiff_frameworks").main()
    ex = _load("autodiff_frameworks")
    assert found == ex.JUMPS and after < before / 3
    assert abs(gamma / best - 1) < 0.1           # grid spacing is 7%
