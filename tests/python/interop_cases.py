"""Cases shared by the PyTorch and JAX wrapper tests: an operator class, its
constant and differentiable parameters, an input and a step, chosen away from
kinks so that finite differences are valid."""

import numpy as np

import fastprox as fp


def _tree(n, seed):
    rng = np.random.default_rng(seed)
    i = np.arange(1, n)
    return np.stack([i, (rng.random(n - 1) * i).astype(np.int64)], 1)


def _ring(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], 1)


def cases():
    """(name, op class, constants, differentiable parameters, x, step, conj, call options)."""
    rng = np.random.default_rng(0)
    walk = rng.normal(size=(3, 16)).cumsum(-1)
    img = rng.normal(size=(2, 7, 8)) + np.indices((7, 8)).sum(0) // 4 % 2
    edges = _ring(12)
    out = [
        ("soft", fp.Soft, {}, {"lam": np.array(0.4), "weights": rng.uniform(0.5, 1.5, 16)},
         rng.normal(size=(3, 16)) * 2, 0.8, False, {}),
        ("elastic_net", fp.ElasticNet, {}, {"l1": np.array(0.3), "l2": np.array(0.7)},
         rng.normal(size=(4, 9)), 1.1, False, {}),
        ("box", fp.Box, {}, {"lower": np.full(9, -0.5), "upper": np.full(9, 0.6)},
         rng.normal(size=(4, 9)), 1.0, False, {}),
        ("tv1d", fp.TV1D, {}, {"weights": rng.uniform(0.5, 1.5, 15)}, walk, 0.7, False, {}),
        ("tv1d_conj", fp.TV1D, {}, {"weights": rng.uniform(0.5, 1.5, 15)}, walk, 0.9, True, {}),
        ("simplex", fp.Simplex, {}, {"radius": np.array(1.3)}, rng.normal(size=(3, 10)), 1.0, False,
         {"axis": 0}),
        ("group_l2", fp.GroupL2, {"lam": 0.8, "groups": np.repeat(np.arange(4), 3)},
         {"group_weights": rng.uniform(0.5, 1.5, 4)}, rng.normal(size=(2, 12)), 0.6, False, {}),
        ("tv_aniso", fp.TVAniso, {"ndim": 2}, {"weights": np.array([0.8, 1.2])}, img, 0.3, False,
         {"tol": 1e-14, "max_iter": 100000}),
        ("tv_iso", fp.TVIso, {"ndim": 2}, {"weights": np.array([0.9, 1.1])}, img, 0.3, False,
         {"tol": 1e-14, "max_iter": 200000}),
        ("graph_tv", fp.GraphTV, {"n": 12, "edges": edges}, {"weights": rng.uniform(0.5, 1.5, 12)},
         rng.normal(size=(2, 12)) * 2, 0.4, False, {}),
        ("tree_tv", fp.TreeTV, {"n": 12, "edges": _tree(12, 1)}, {"weights": rng.uniform(0.5, 1.5, 11)},
         rng.normal(size=(2, 12)) * 2, 0.4, False, {}),
    ]
    return out
