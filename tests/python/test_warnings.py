# Unconverged and too-loose iterative calls warn (feedback U2):
# ConvergenceWarning when a call stops at max_iter, and inside
# fp.inner_accuracy(target) a default ztol of a tenth of the target, with
# AccuracyWarning for a call that asks for less.
#
# SPDX-License-Identifier: MIT
import re
import threading
import warnings

import numpy as np
import pytest

import fastprox as fp


def _x(shape=(64, 64), seed=0):
    return np.random.default_rng(seed).normal(size=shape)


def _grid(n):
    e = [(i * n + j, i * n + j + 1) for i in range(n) for j in range(n - 1)]
    e += [(i * n + j, (i + 1) * n + j) for i in range(n - 1) for j in range(n)]
    return np.array(e)


def _quiet(f):
    """f() with every warning an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return f()


def test_classes():
    assert issubclass(fp.ConvergenceWarning, fp.AccuracyWarning)
    assert issubclass(fp.AccuracyWarning, UserWarning)


def test_convergence_warning_numpy():
    x = _x()
    op = fp.TVIso()
    with pytest.warns(fp.ConvergenceWarning, match="max_iter=2") as rec:
        z, info = op.solve(x, 0.5, max_iter=2)
    assert not info.converged
    zerr = float(re.search(r"error of z is (\S+)", str(rec[0].message)).group(1))
    assert zerr == pytest.approx(info.zerr, rel=1e-2)
    assert rec[0].filename == __file__  # points at the caller
    for f in (lambda: op.prox(x, 0.5, max_iter=2), lambda: op.prox_conj(x, 0.5, max_iter=2),
              lambda: op.prox(x, 0.5, max_iter=2, keep=True),
              lambda: fp.tv_iso(x, 0.5, max_iter=2), lambda: fp.TVAniso().solve(x, 0.5, max_iter=1, tol=1e-14),
              lambda: fp.GraphTV(64 * 64, _grid(64)).prox(x.reshape(-1), 0.5, max_iter=1)):
        with pytest.warns(fp.ConvergenceWarning):
            f()
    # Converged calls and warn=False are silent.
    _quiet(lambda: op.solve(x, 0.5))
    _quiet(lambda: op.prox(x, 0.5))
    _quiet(lambda: op.solve(x, 0.5, max_iter=2, warn=False))
    _quiet(lambda: fp.tv_iso(x, 0.5, max_iter=2, warn=False))
    # The standard filters silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", fp.ConvergenceWarning)
        op.prox(x, 0.5, max_iter=2)


def test_inner_accuracy_default():
    x = _x((2, 32, 32))
    for op in (fp.TVIso(), fp.TVAniso()):
        with fp.inner_accuracy(1e-3) as ztol:
            assert ztol == pytest.approx(1e-4)
            z1 = _quiet(lambda: op.prox(x, 0.5, max_iter=100000))
            z3, _ = _quiet(lambda: op.solve(x, 0.5, max_iter=100000))
        assert np.array_equal(z1, op.prox(x, 0.5, ztol=1e-4, max_iter=100000))
        assert np.array_equal(z3, z1)
        with fp.inner_accuracy(1e-2, factor=0.5):
            z2 = op.prox(x, 0.5, max_iter=100000)
        assert np.array_equal(z2, op.prox(x, 0.5, ztol=5e-3, max_iter=100000))
    g = fp.GraphTV(32 * 32, _grid(32))
    xg = x.reshape(2, -1)
    with fp.inner_accuracy(1e-2):
        zg = _quiet(lambda: g.prox(xg, 0.5))
    assert np.array_equal(zg, g.prox(xg, 0.5, ztol=1e-3))
    # Outside the block: the defaults again.
    assert np.array_equal(fp.TVIso().prox(x, 0.5), fp.TVIso().prox(x, 0.5, tol=1e-6))


def test_inner_accuracy_looser_warns():
    x = _x((32, 32))
    op = fp.TVIso()
    with fp.inner_accuracy(1e-3):  # ztol 1e-4, tol 1e-8
        with pytest.warns(fp.AccuracyWarning, match="tol=1e-06") as rec:
            op.prox(x, 0.5, tol=1e-6)
        assert rec[0].filename == __file__
        with pytest.warns(fp.AccuracyWarning, match="ztol=0.001"):
            op.prox(x, 0.5, ztol=1e-3)
        with pytest.warns(fp.AccuracyWarning):
            fp.tv_aniso(x, 0.5, tol=1e-6)
        _quiet(lambda: op.prox(x, 0.5, tol=1e-10, max_iter=100000))  # tighter: silent
        _quiet(lambda: op.prox(x, 0.5, ztol=1e-4, max_iter=100000))
        _quiet(lambda: op.solve(x, 0.5, tol=1e-6, warn=False))
        with pytest.warns(fp.ConvergenceWarning, match="inner_accuracy asks for 0.0001"):
            op.prox(x, 0.5, max_iter=2)
        # Tree TV is exact: no tolerance is too loose.
        chain = np.stack([np.arange(32 * 32 - 1), np.arange(1, 32 * 32)], 1)
        _quiet(lambda: fp.TreeTV(32 * 32, chain).prox(x.reshape(-1), 0.5, tol=0.5))


def test_inner_accuracy_nesting_and_errors():
    with fp.inner_accuracy(1e-2):
        with fp.inner_accuracy(1e-6, factor=1.0) as inner:
            assert inner == 1e-6
            assert fp._ACCURACY.get() == 1e-6
        assert fp._ACCURACY.get() == pytest.approx(1e-3)
        with pytest.raises(RuntimeError):
            with fp.inner_accuracy(1e-4):
                raise RuntimeError
        assert fp._ACCURACY.get() == pytest.approx(1e-3)
    assert fp._ACCURACY.get() is None
    for bad in [(0.0,), (-1.0,), (np.inf,), (np.nan,), (1e-3, 0.0), (1e-3, -1.0), (1e-200,)]:
        with pytest.raises(ValueError):
            with fp.inner_accuracy(*bad):
                pass
    with pytest.raises(ValueError):
        with fp.inner_accuracy(1e-3):
            fp.TVIso().prox(_x((8, 8)), 0.5, tol=1e-6, ztol=1e-3)


def test_inner_accuracy_threads():
    """Each thread has its own block: two threads with different targets,
    interleaved, get the results of their own ztol."""
    x = _x((32, 32))
    op = fp.TVAniso()
    want = {t: op.prox(x, 0.5, ztol=t, max_iter=100000) for t in (1e-2, 1e-5)}
    got, errors = {}, []
    start = threading.Barrier(2)

    def run(target):
        try:
            with fp.inner_accuracy(target, factor=1.0):
                start.wait()
                for _ in range(20):
                    got[target] = op.prox(x, 0.5, max_iter=100000)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(t,)) for t in want]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    for t in want:
        assert np.array_equal(got[t], want[t])
    assert fp._ACCURACY.get() is None


def test_torch():
    torch = pytest.importorskip("torch")
    import fastprox.torch as fpt
    x = torch.tensor(_x())
    with pytest.warns(fp.ConvergenceWarning) as rec:
        fpt.tv_iso(x, 0.5, max_iter=2)
    assert rec[0].filename == __file__
    with pytest.warns(fp.ConvergenceWarning):
        fpt.tv_aniso(x, 0.5, max_iter=2)
    with pytest.warns(fp.ConvergenceWarning):
        fpt.prox(fp.TVIso(), x, 0.5, max_iter=2)
    _quiet(lambda: fpt.tv_iso(x, 0.5, max_iter=2, warn=False))
    _quiet(lambda: fpt.tv_iso(x, 0.5))
    xs = x[:32, :32].contiguous()
    with fp.inner_accuracy(1e-2):
        a = _quiet(lambda: fpt.tv_iso(xs, 0.5, max_iter=100000))
        b = _quiet(lambda: fpt.prox(fp.TVIso(), xs, 0.5, max_iter=100000))
        with pytest.warns(fp.AccuracyWarning):
            fpt.tv_iso(xs, 0.5, tol=1e-4)
        with pytest.warns(fp.AccuracyWarning):
            fpt.prox(fp.TVIso(), xs, 0.5, tol=1e-4)
    assert torch.equal(a, fpt.tv_iso(xs, 0.5, ztol=1e-3, max_iter=100000))
    assert torch.equal(b, fpt.prox(fp.TVIso(), xs, 0.5, ztol=1e-3, max_iter=100000))


@pytest.mark.parametrize("jit", [False, True])
def test_jax(jit):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import fastprox.jax as fpj
    x = jnp.asarray(_x(), jnp.float32)

    def run(f):
        g = jax.jit(f) if jit else f
        out = g(x)
        jax.block_until_ready(out)
        jax.effects_barrier()  # the callbacks have run
        return out

    for f in (lambda x: fpj.tv_iso(x, 0.5, max_iter=2), lambda x: fpj.tv_aniso(x, 0.5, max_iter=2),
              lambda x: fpj.prox(fp.TVIso(), x, 0.5, max_iter=2)):
        with pytest.warns(fp.ConvergenceWarning, match="max_iter=2"):
            run(f)
    _quiet(lambda: run(lambda x: fpj.tv_iso(x, 0.5, max_iter=2, warn=False)))
    _quiet(lambda: run(lambda x: fpj.tv_iso(x, 0.5)))
    # The block in force when the function is traced, also in the host callback.
    with fp.inner_accuracy(1e-2):
        a = _quiet(lambda: run(lambda x: fpj.prox(fp.TVIso(), x, 0.5, max_iter=100000)))
        with pytest.warns(fp.AccuracyWarning):
            run(lambda x: fpj.prox(fp.TVIso(), x, 0.5, tol=1e-4))
        with pytest.warns(fp.AccuracyWarning):
            run(lambda x: fpj.tv_iso(x, 0.5, tol=1e-4))
    assert np.array_equal(np.asarray(a), np.asarray(run(lambda x: fpj.prox(fp.TVIso(), x, 0.5, ztol=1e-3,
                                                                          max_iter=100000))))
