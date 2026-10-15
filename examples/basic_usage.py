"""Basic usage: one-shot functions, operator objects, axes, dtypes and threads.

Every function computes prox_{gamma g}(x) = argmin_z g(z) + ||z - x||^2 / (2 gamma)
along one axis of an array (the last by default), independently for each row.

    uv run python examples/basic_usage.py
"""

import numpy as np

import fastprox as fp


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 6))

    # -- One-shot functions --------------------------------------------------
    print("soft thresholding (lam = 0.5):\n", fp.soft_threshold(x, lam=0.5).round(3))

    p = fp.project_simplex(x, radius=1.0)
    print("simplex projection: rows sum to", p.sum(axis=1), " min entry", p.min())

    b = fp.project_l2_ball(x, radius=1.0)
    print("l2 ball projection: row norms", np.linalg.norm(b, axis=1).round(6))

    print("isotonic regression of [3, 1, 2, 5, 4]:", fp.isotonic([3.0, 1, 2, 5, 4]))

    print("1D TV of [0, 0, 5, 5, 5] with gamma = 1:", fp.tv1d([0.0, 0, 5, 5, 5], gamma=1.0))

    # -- Axis: the same operator along columns instead of rows ---------------
    pc = fp.project_simplex(x, axis=0)
    print("simplex along axis 0: columns sum to", pc.sum(axis=0))

    # Strided views are used as they are, without a copy.
    print("every other column:", fp.project_simplex(x[:, ::2]).sum(axis=1))

    # -- dtypes and output buffers --------------------------------------------
    x32 = x.astype(np.float32)
    print("float32 in, float32 out:", fp.tv1d(x32, gamma=0.5).dtype)

    try:
        fp.tv1d(np.arange(5))                        # integers are not converted silently
    except TypeError as e:
        print("TypeError:", e)
    print("with copy=True:", fp.tv1d(np.arange(5), gamma=0.5, copy=True))

    out = np.empty_like(x)
    fp.soft_threshold(x, lam=0.5, out=out)            # write into a preallocated buffer
    y = x.copy()
    fp.soft_threshold(y, lam=0.5, out=y)              # or in place
    print("out= and in place agree:", np.array_equal(out, y))

    # -- Operator objects: fixed parameters, prox and conjugate prox ----------
    op = fp.TV1D(weights=0.8)
    z = op.prox(x, 0.5)
    print(op, "matches tv1d:", np.array_equal(z, fp.tv1d(x, 0.5, weights=0.8)))

    # Moreau identity: x = prox_{gamma g}(x) + gamma prox_{g*/gamma}(x / gamma).
    gamma = 0.5
    r = op.prox(x, gamma) + gamma * op.prox_conj(x / gamma, 1 / gamma)
    print("Moreau identity residual:", np.abs(r - x).max())

    # -- Threads: rows run in parallel, with identical results for any count --
    big = rng.normal(size=(256, 4096))
    one = fp.tv1d(big, gamma=2.0, threads=1)
    many = fp.tv1d(big, gamma=2.0, threads=4)
    print(f"default threads: {fp.default_threads()}; "
          f"1 vs 4 threads bit-identical: {np.array_equal(one, many)}")


if __name__ == "__main__":
    main()
