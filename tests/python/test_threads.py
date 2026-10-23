# Python threads and the bindings: calls on one Context are serialised
# (audit P3), and the data-sized C calls release the GIL (audit P4).
#
# SPDX-License-Identifier: MIT
import threading
import time

import numpy as np
import pytest

import fastprox as fp


def _hammer(work, n_threads=4, calls=30):
    """Runs work(t, i) from n_threads threads, calls times each, with a
    barrier so that they overlap; returns {(t, i): result}."""
    results, errors = {}, []
    start = threading.Barrier(n_threads)

    def run(t):
        start.wait()
        try:
            for i in range(calls):
                results[(t, i)] = work(t, i)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    return results


def test_context_concurrent():
    """4 threads x 30 calls on one context give the serial results, bit for
    bit. Unserialised, isotropic TV's vjp was wrong in 80 of 80 calls (by up to
    3.9e262) and soft thresholding with a branch in 118 of 120 (audit P3)."""
    rng = np.random.default_rng(3)
    gs = rng.normal(size=(4, 30, 4, 64, 64))
    iso = fp.TVIso(2, 0.3)
    _, ci = iso.prox(rng.normal(size=(4, 64, 64)), 0.8, keep=True, threads=1, tol=1e-8, max_iter=100000)

    def iso_work(t, i):
        g = gs[t, i]
        return iso.vjp(ci, g)[0] if i % 2 else iso.jvp(ci, g)

    serial = {(t, i): iso_work(t, i) for t in range(4) for i in range(30)}
    for k, v in _hammer(iso_work).items():
        assert np.array_equal(v, serial[k]), k

    soft = fp.Soft(0.4)
    xs = rng.normal(size=(256, 4000))
    xs[::7, ::5] = 0.4  # entries on the kink
    _, cs = soft.prox(xs, 1.0, keep=True, threads=1)

    def soft_work(t, i):  # each call's tangent from its own seed: 8 MB, not 1 GB for all
        h = np.random.default_rng(100 * t + i).normal(size=xs.shape)
        return soft.jvp(cs, h, branch=1 if i % 2 else -1, kink_tol=0.5)

    serial = {(t, i): soft_work(t, i) for t in range(4) for i in range(30)}
    for k, v in _hammer(soft_work).items():
        assert np.array_equal(v, serial[k]), k

    nuc = fp.NuclearNorm(0.5)
    _, cn = nuc.prox(rng.normal(size=(16, 20, 12)), 1.0, keep=True)
    ms = rng.normal(size=(4, 30, 16, 20, 12))

    def nuc_work(t, i):
        gx, gr = nuc.vjp(cn, ms[t, i])
        return np.concatenate([gx.ravel(), [gr.gamma]])

    serial = {(t, i): nuc_work(t, i) for t in range(4) for i in range(30)}
    for k, v in _hammer(nuc_work).items():
        assert np.array_equal(v, serial[k]), k


def test_constructor_releases_gil():
    """Building a large graph TV operator lets other Python threads run (audit
    P4): it held the GIL for its whole 255 ms on 2e6 nodes before."""
    rng = np.random.default_rng(0)
    n = 500_000
    e = rng.integers(0, n, (4 * n, 2))
    e = e[e[:, 0] != e[:, 1]]
    ticks, stop = [0], threading.Event()

    def ticker():
        while not stop.is_set():
            ticks[0] += 1
            time.sleep(0.001)

    th = threading.Thread(target=ticker)
    th.start()
    time.sleep(0.01)
    t0, k0 = time.perf_counter(), ticks[0]
    fp.GraphTV(n, e)
    ms, k1 = 1e3 * (time.perf_counter() - t0), ticks[0]
    stop.set()
    th.join()
    if ms < 20:
        pytest.skip(f"construction took {ms:.1f} ms, too short to measure")
    # One tick per 1 ms sleep at best; a quarter of that allows for the
    # scheduler's granularity. With the GIL held there are none.
    assert k1 - k0 >= 0.25 * ms, (ms, k1 - k0)
