"""Denoise a piecewise-constant signal with 1D total variation.

    z = argmin_z  gamma * sum_i |z_{i+1} - z_i| + ||z - y||^2 / 2

Small gamma leaves noise in; large gamma merges steps and shrinks their
height. Plots the result if matplotlib is installed:

    uv run --group examples python examples/tv_denoising.py
"""

import numpy as np

import fastprox as fp


def make_signal(rng, n=1000, sigma=0.5):
    jumps = np.sort(rng.choice(np.arange(50, n - 50, 25), size=8, replace=False))
    x = np.zeros(n)
    for j in jumps:
        x[j:] += rng.normal(scale=2.0)
    return x, x + sigma * rng.normal(size=n)


def main(plot=True):
    rng = np.random.default_rng(1)
    x, y = make_signal(rng)

    print(" gamma     RMSE   pieces")
    results = {}
    for gamma in (0.5, 2.0, 8.0, 32.0, 128.0):
        z = fp.tv1d(y, gamma=gamma)
        pieces = 1 + np.count_nonzero(np.diff(z))
        rmse = np.sqrt(np.mean((z - x) ** 2))
        results[gamma] = z
        print(f"{gamma:6.1f}  {rmse:7.4f}  {pieces:7d}")
    print(f"noisy   {np.sqrt(np.mean((y - x) ** 2)):7.4f}  "
          f"(clean signal has {1 + np.count_nonzero(np.diff(x))} pieces)")

    # Many signals at once: each row is denoised independently, in parallel.
    batch = np.stack([make_signal(rng)[1] for _ in range(64)])
    zb = fp.tv1d(batch, gamma=8.0)
    print("batch of", zb.shape[0], "signals: row 0 matches a single call:",
          np.array_equal(zb[0], fp.tv1d(batch[0], gamma=8.0)))

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(install matplotlib, e.g. `uv run --group examples ...`, to see a plot)")
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y, color="0.75", lw=0.8, label="noisy")
        ax.plot(x, "k", lw=1.5, label="clean")
        for gamma in (2.0, 32.0):
            ax.plot(results[gamma], lw=1.2, label=f"TV, gamma = {gamma:g}")
        ax.legend()
        ax.set_title("1D total variation denoising")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
