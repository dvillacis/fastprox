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
