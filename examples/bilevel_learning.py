"""Learn TV regularisation weights from data (bilevel learning).

Training pairs are noisy and clean piecewise-constant signals whose jumps sit
at the same positions in every signal (a sensor whose state switches at known
times, say). The lower level denoises each signal with weighted 1D TV,

    z(w) = argmin_z  sum_i w_i |z_{i+1} - z_i| + ||z - y||^2 / 2,

and the upper level fits the weights to the clean signals,

    min_{w >= 0}  L(w) = mean over signals of ||z(w) - x||^2 / 2.

The gradient of L comes from one VJP of the prox: with g = dL/dz, ``op.vjp``
returns g^T dz/dw summed over the batch. We first learn one scalar step (as a
user would tune gamma), then one weight per edge by projected gradient.

    uv run python examples/bilevel_learning.py
"""

import numpy as np

import fastprox as fp

N, M = 200, 512          # signal length, training signals
JUMPS = [40, 75, 120, 160]


def make_signals(rng, m, sigma=0.3):
    """m clean signals with jumps at JUMPS (random levels) and noisy copies."""
    edges = [0, *JUMPS, N]
    levels = rng.normal(size=(m, len(edges) - 1))
    x = np.concatenate([np.repeat(levels[:, [k]], edges[k + 1] - edges[k], axis=1)
                        for k in range(len(edges) - 1)], axis=1)
    return x, x + sigma * rng.normal(size=x.shape)


def loss(z, x):
    return 0.5 * np.mean(np.sum((z - x) ** 2, axis=1))


def loss_and_grad(op, gamma, y, x):
    """L at (op, gamma) and its VJP: (L, grads) with grads.gamma, grads.weights."""
    z, ctx = op.prox(y, gamma, keep=True)
    _, grads = op.vjp(ctx, g=(z - x) / len(x))
    return loss(z, x), grads


def main():
    rng = np.random.default_rng(0)
    x_train, y_train = make_signals(rng, M)
    x_test, y_test = make_signals(rng, M)
    print(f"noisy input:       test loss {loss(y_test, x_test):8.4f}")

    # 1. One scalar step, gradient descent on t = log(gamma) to keep gamma > 0.
    op = fp.TV1D()
    t = np.log(0.1)
    for _ in range(30):
        _, grads = loss_and_grad(op, np.exp(t), y_train, x_train)
        t -= 0.5 * grads.gamma * np.exp(t)           # dL/dt = dL/dgamma * gamma
    gamma = float(np.exp(t))
    test_scalar = loss(op.prox(y_test, gamma), x_test)
    print(f"learned gamma:     test loss {test_scalar:8.4f}   (gamma = {gamma:.3f})")

    # Check against a grid search on the training loss.
    grid = np.linspace(0.5 * gamma, 1.5 * gamma, 101)
    best = grid[np.argmin([loss(fp.tv1d(y_train, g), x_train) for g in grid])]
    print(f"grid search:       best gamma {best:.3f}")

    # 2. One weight per edge, starting from the scalar, projected onto w >= 0.
    w = np.full(N - 1, gamma)
    for _ in range(300):
        _, grads = loss_and_grad(fp.TV1D(weights=w), 1.0, y_train, x_train)
        w = fp.project_nonneg(w - 2.0 * grads.weights)
    test_edges = loss(fp.TV1D(weights=w).prox(y_test, 1.0), x_test)
    print(f"learned weights:   test loss {test_edges:8.4f}")

    # The smallest weights should sit on the jumps (edge i joins samples i, i+1).
    found = sorted(int(i) + 1 for i in np.argsort(w)[:len(JUMPS)])
    print(f"lowest weights at: {found}   (jumps at {JUMPS})")
    print(f"median weight {np.median(w):.3f}, on the jumps {w[np.array(JUMPS) - 1].round(3)}")
    return w


if __name__ == "__main__":
    main()
