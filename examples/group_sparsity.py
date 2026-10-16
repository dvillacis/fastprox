"""Select groups of features: lasso, group lasso, sparse group lasso, group
MCP and SCAD, and hierarchical (tree) sparsity (since 0.2).

The 400 features of a regression come in 40 groups of 10 (the levels of a
categorical variable, the genes of a pathway, the lags of a series). Only 4
groups matter, and within each only 4 of the 10 features. Each penalty g is
fitted by

    min_x  ||A x - b||^2 / 2 + g(x)

with FISTA (plain proximal gradient for the nonconvex ones) along a path of
lam, and lam is chosen on a validation set of 1000 samples:

- the lasso finds the features but spreads extras over many groups;
- the group lasso selects whole groups, so every feature of a group enters;
- the sparse group lasso, l1 + group l2 (``TreeL2.sparse_group``), is sparse
  at both levels;
- group MCP and SCAD, started from the group lasso, drop the spurious groups
  and the shrinkage bias, though like the group lasso they keep every
  feature of a selected group.

The last part uses nested groups: with one group per node of a tree holding
the node and its descendants, a node can be nonzero only if its parent is
(Jenatton, Mairal, Obozinski and Bach, JMLR 12, 2011).

    uv run python examples/group_sparsity.py
"""

import numpy as np

import fastprox as fp

N, NVAL, G, S = 150, 1000, 40, 10   # samples, validation samples, groups, group size
GROUPS = np.arange(G * S) // S  # group id of each feature
LAMS = np.geomspace(3.0, 0.005, 20)


def make_problem(rng):
    x0 = np.zeros(G * S)
    for g in rng.choice(G, 4, replace=False):
        idx = g * S + rng.choice(S, 4, replace=False)
        x0[idx] = rng.choice([-1, 1], 4) * rng.uniform(1, 2, 4)
    A = rng.normal(size=(N, G * S)) / np.sqrt(N)
    Av = rng.normal(size=(NVAL, G * S)) / np.sqrt(N)
    b = A @ x0 + 0.1 * rng.normal(size=N)
    bv = Av @ x0 + 0.1 * rng.normal(size=NVAL)
    return A, b, Av, bv, x0


def fit(op, A, b, x, iters=400, momentum=True):
    """FISTA, or proximal gradient with momentum=False."""
    t = 1.0 / np.linalg.norm(A, 2) ** 2
    y, s = x.copy(), 1.0
    for _ in range(iters):
        x_new = op.prox(y - t * (A.T @ (A @ y - b)), t)
        s_new = (1 + np.sqrt(1 + 4 * s * s)) / 2 if momentum else 1.0
        y = x_new + (s - 1) / s_new * (x_new - x)
        x, s = x_new, s_new
    return x


def path(make_op, A, b, Av, bv, x, momentum=True, warm=True):
    """Fit along LAMS, each from the previous solution (warm) or from x; return
    (lam, x) with the best validation error."""
    best = (np.inf, None, None)
    start = x
    for lam in LAMS:
        x = fit(make_op(lam), A, b, x if warm else start.copy(), momentum=momentum)
        err = np.mean((Av @ x - bv) ** 2)
        if err < best[0]:
            best = (err, lam, x.copy())
    return best[1], best[2]


def report(name, lam, x, x0):
    true = np.unique(GROUPS[x0 != 0])
    found = np.unique(GROUPS[x != 0])
    extra_groups = len(np.setdiff1d(found, true))
    extra = np.sum((x != 0) & (x0 == 0))
    missed = np.sum((x == 0) & (x0 != 0))
    err = np.linalg.norm(x - x0) / np.linalg.norm(x0)
    print(f"{name:20s} {lam:6.3f}  {extra_groups:6d}  {extra:7d}  {missed:6d}  {err:8.3f}")
    return extra_groups, extra, err


def main():
    rng = np.random.default_rng(0)
    A, b, Av, bv, x0 = make_problem(rng)
    zero = np.zeros_like(x0)

    print(f"{'penalty':20s} {'lam':>6s}  {'extra':>6s}  {'extra':>7s}  {'missed':>6s}  "
          f"{'rel. err':>8s}")
    print(f"{'':20s} {'':6s}  {'groups':>6s}  {'feat.':>7s}  {'feat.':>6s}")
    results = {}
    convex = {
        "lasso": lambda lam: fp.Soft(lam),
        "group lasso": lambda lam: fp.GroupL2(lam, GROUPS),
        "sparse group lasso": lambda lam: fp.TreeL2.sparse_group(l1=lam / 2, lam=lam / 2,
                                                                  groups=GROUPS),
    }
    for name, make_op in convex.items():
        lam, x = path(make_op, A, b, Av, bv, zero)
        results[name] = report(name, lam, x, x0)
        if name == "group lasso":
            x_group = x

    # Nonconvex: each lam starts from the group lasso solution.
    nonconvex = {
        "group MCP": lambda lam: fp.GroupMCP(lam, b=3.0, groups=GROUPS),
        "group SCAD": lambda lam: fp.GroupSCAD(lam, a=3.7, groups=GROUPS),
    }
    for name, make_op in nonconvex.items():
        lam, x = path(make_op, A, b, Av, bv, x_group, momentum=False, warm=False)
        results[name] = report(name, lam, x, x0)

    # Hierarchical sparsity on a binary tree of 15 nodes in heap order (the
    # children of node i are 2i + 1 and 2i + 2), e.g. wavelet coefficients.
    # One group per node: the node and all its descendants. The weights are
    # set so both keep about the same share of nodes.
    def subtree(i):
        return [i] if i >= 15 else [i, *subtree(2 * i + 1), *subtree(2 * i + 2)]
    tree = [[j for j in subtree(i) if j < 15] for i in range(15)]
    parent = (np.arange(1, 15) - 1) // 2

    x = rng.normal(size=(10000, 15))                  # 10000 rows, one call each
    z_tree = fp.tree_soft_threshold(x, tree, group_weights=0.8)
    z_l1 = fp.soft_threshold(x, lam=1.0)
    print("\nnodes kept, and rows where a kept node has a zero parent:")
    for name, z in (("l1", z_l1), ("tree", z_tree)):
        orphans = np.any((z[:, 1:] != 0) & (z[:, parent] == 0), axis=1)
        print(f"  {name:5s} {np.mean(z != 0):5.1%} kept, {np.mean(orphans):6.1%} of rows")
    return results


if __name__ == "__main__":
    main()
