"""Problems for the CVXPY reference fixtures (docs/spec.md, Section 11).

Each case draws parameters for a row of length n, builds the fastprox
operator from them, and states the same prox problem for CVXPY. The fixtures
are generated offline by ``tests/fixtures/generate.py`` and checked by
``test_fixtures.py``; CVXPY is needed only to regenerate them.
"""

import numpy as np

import fastprox as fp


def _pos(rng, size, lo=0.2, hi=2.0):
    return rng.uniform(lo, hi, size)


def _soft(rng, n):
    return {"lam": np.array([rng.uniform(0.1, 1.0)]), "weights": _pos(rng, n)}


def _groups(rng, n):
    g = rng.integers(0, 3, n)
    g[: min(n, 3)] = np.arange(min(n, 3))  # every group id present
    return {"lam": np.array([0.7]), "groups": g.astype(float),
            "c": _pos(rng, int(g.max()) + 1)}


def _box(rng, n):
    return {"lower": -_pos(rng, n, 0.0, 1.0), "upper": _pos(rng, n, 0.0, 1.0)}


def _iso(rng, n):
    return {"weights": _pos(rng, n), "increasing": np.array([float(rng.integers(0, 2))])}


def _tvw(rng, n):
    w = _pos(rng, max(n - 1, 1), 0.0, 2.0)
    w[rng.random(w.size) < 0.1] = 0.0
    return {"weights": w}


# name: (sample(rng, n) -> params, make(params) -> Operator,
#        problem(cp, z, x, gamma, params) -> (g term, constraints[, full objective]))
CASES = {
    "soft": (
        _soft,
        lambda p: fp.Soft(p["lam"][0], p["weights"]),
        lambda cp, z, x, g, p: (g * p["lam"][0] * cp.sum(cp.multiply(p["weights"], cp.abs(z))), []),
    ),
    "elastic_net": (
        lambda rng, n: {"l1": np.array([0.4]), "l2": np.array([0.8])},
        lambda p: fp.ElasticNet(p["l1"][0], p["l2"][0]),
        lambda cp, z, x, g, p: (g * (p["l1"][0] * cp.norm1(z) + p["l2"][0] / 2 * cp.sum_squares(z)), []),
    ),
    "huber": (
        lambda rng, n: {"lam": np.array([0.6]), "delta": np.array([0.5])},
        lambda p: fp.Huber(p["lam"][0], p["delta"][0]),
        # h_delta(t) = huber(t, delta) / (2 delta) in CVXPY's convention.
        lambda cp, z, x, g, p: (g * p["lam"][0] * cp.sum(cp.huber(z, p["delta"][0])) / (2 * p["delta"][0]), []),
    ),
    "group_l2": (
        _groups,
        lambda p: fp.GroupL2(p["lam"][0], p["groups"].astype(np.int64), p["c"]),
        lambda cp, z, x, g, p: (g * p["lam"][0] * sum(
            p["c"][k] * cp.norm(z[np.flatnonzero(p["groups"] == k)], 2)
            for k in range(p["c"].size)), []),
    ),
    "box": (
        _box,
        lambda p: fp.Box(p["lower"], p["upper"]),
        lambda cp, z, x, g, p: (0, [z >= p["lower"], z <= p["upper"]]),
    ),
    "l2_ball": (
        lambda rng, n: {"radius": np.array([rng.uniform(0.2, 3.0)])},
        lambda p: fp.L2Ball(p["radius"][0]),
        lambda cp, z, x, g, p: (0, [cp.norm(z, 2) <= p["radius"][0]]),
    ),
    "simplex": (
        lambda rng, n: {"radius": np.array([rng.uniform(0.2, 3.0)])},
        lambda p: fp.Simplex(p["radius"][0]),
        lambda cp, z, x, g, p: (0, [z >= 0, cp.sum(z) == p["radius"][0]]),
    ),
    "l1_ball": (
        lambda rng, n: {"radius": np.array([rng.uniform(0.2, 3.0)]), "weights": _pos(rng, n)},
        lambda p: fp.L1Ball(p["radius"][0], p["weights"]),
        lambda cp, z, x, g, p: (0, [cp.sum(cp.multiply(p["weights"], cp.abs(z))) <= p["radius"][0]]),
    ),
    "linf": (
        lambda rng, n: {"lam": np.array([rng.uniform(0.2, 3.0)])},
        lambda p: fp.LInf(p["lam"][0]),
        lambda cp, z, x, g, p: (g * p["lam"][0] * cp.norm_inf(z), []),
    ),
    "isotonic": (
        _iso,
        lambda p: fp.Isotonic(p["weights"], increasing=bool(p["increasing"][0])),
        # Weighted least squares replaces the prox term: a third entry is the
        # whole objective.
        lambda cp, z, x, g, p: (
            None,
            [cp.diff(z) >= 0] if p["increasing"][0] else [cp.diff(z) <= 0],
            0.5 * cp.sum(cp.multiply(p["weights"], cp.square(z - x)))),
    ),
    "tv1d": (
        lambda rng, n: {},
        lambda p: fp.TV1D(),
        lambda cp, z, x, g, p: (g * cp.norm1(cp.diff(z)), []),
    ),
    "tv1d_weighted": (
        _tvw,
        lambda p: fp.TV1D(p["weights"]),
        lambda cp, z, x, g, p: (g * cp.sum(cp.multiply(p["weights"], cp.abs(cp.diff(z)))), []),
    ),
}

SIZES = (2, 5, 17, 60, 200)
SCALES = (0.1, 1.0, 10.0)


# Image operators: name -> (sample(rng, shape) -> params, make(params) -> Operator,
# problem(cp, z, x, gamma, params, shape) -> g term), z the flattened image.
def _diff(shape, axis):
    """Sparse forward differences along `axis` of a C-ordered image."""
    import scipy.sparse as sp

    idx = np.arange(int(np.prod(shape))).reshape(shape)
    a = np.take(idx, range(shape[axis] - 1), axis=axis).ravel()
    b = np.take(idx, range(1, shape[axis]), axis=axis).ravel()
    m = a.size
    rows = np.concatenate([np.arange(m), np.arange(m)])
    return sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (rows, np.r_[a, b])),
                         shape=(m, idx.size))


def _aniso_term(cp, z, g, p, shape):
    k = len(shape)
    return g * sum(p["weights"][a] * cp.norm1(_diff(shape, a) @ z)
                   for a in range(k) if shape[a] > 1)


IMAGE_CASES = {
    "tv_aniso_2d": (
        lambda rng, shape: {"weights": _pos(rng, 2, 0.0, 2.0)},
        lambda p: fp.TVAniso(2, p["weights"]),
        _aniso_term,
    ),
    "tv_aniso_3d": (
        lambda rng, shape: {"weights": _pos(rng, 3, 0.0, 2.0)},
        lambda p: fp.TVAniso(3, p["weights"]),
        _aniso_term,
    ),
}

IMAGE_SHAPES = {
    "tv_aniso_2d": ((1, 7), (2, 3), (5, 7), (12, 15), (20, 9)),
    "tv_aniso_3d": ((2, 3, 4), (4, 5, 6), (3, 1, 8), (6, 6, 5)),
}
