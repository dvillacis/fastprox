// Projection onto the monotone cone (isotonic regression) by
// pool-adjacent-violators: one pass with a stack of blocks, merging while the
// previous block mean exceeds the current one. Minimises
// sum_i w_i (z_i - x_i)^2 subject to z nondecreasing (or nonincreasing).
//
// Saved structure (code): 1 at the first entry of every pooled block.
// The Jacobian averages over blocks with the weights:
//   (J v)_i = sum_{j in B} w_j v_j / W_B, dz_B/dw_j = (x_j - z_B) / W_B.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <cstring>
#include <utility>

namespace fp {
namespace {

struct Isotonic final : RowOp<Isotonic> {
    Vec w;  // empty: unit weights
    bool increasing;
    Isotonic(Vec ww, bool inc) : w(std::move(ww)), increasing(inc) {}

    const char *name() const noexcept override { return "isotonic"; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.empty() || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    size_t scratch_bytes(int64_t n) const noexcept override {
        return 3 * static_cast<size_t>(n) * sizeof(double);
    }
    double wi(int64_t i) const noexcept { return w.empty() ? 1.0 : w[i]; }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double, void *scr, uint8_t *code) const noexcept {
        bool nan;
        if (scan_row(x, n, nan) != kRowOk) return kRowNonFinite;
        if (code) std::memset(code, 0, static_cast<size_t>(n));
        if (nan) {
            fill_nan(z, n);
            return kRowOk;
        }
        const double sg = increasing ? 1.0 : -1.0;
        int64_t *start = static_cast<int64_t *>(scr);
        double *bw = reinterpret_cast<double *>(start + n);
        double *bs = bw + n;
        int64_t top = 0;  // number of blocks
        for (int64_t i = 0; i < n; ++i) {
            const double wv = wi(i);
            start[top] = i;
            bw[top] = wv;
            bs[top] = wv * sg * static_cast<double>(x[i]);
            ++top;
            while (top > 1 && bs[top - 2] / bw[top - 2] > bs[top - 1] / bw[top - 1]) {
                bw[top - 2] += bw[top - 1];
                bs[top - 2] += bs[top - 1];
                --top;
            }
        }
        for (int64_t b = 0; b < top; ++b) {
            const int64_t e = b + 1 < top ? start[b + 1] : n;
            const T m = static_cast<T>(sg * (bs[b] / bw[b]));
            for (int64_t i = start[b]; i < e; ++i) z[i] = m;
            if (code) code[start[b]] = 1;
        }
        return kRowOk;
    }

    // Calls f(s, e) for every block [s, e).
    template <class F>
    static void blocks(const uint8_t *code, int64_t n, F &&f) {
        int64_t s = 0;
        for (int64_t i = 1; i <= n; ++i)
            if (i == n || code[i]) {
                f(s, i);
                s = i;
            }
    }

    template <class T>
    void jvp(const T *x, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n, double,
             double, const double *dp, void *) const noexcept {
        blocks(code, n, [&](int64_t s, int64_t e) {
            double W = 0.0, num = 0.0;
            for (int64_t i = s; i < e; ++i) {
                W += wi(i);
                num += wi(i) * static_cast<double>(dx[i]);
                if (dp) num += (static_cast<double>(x[i]) - static_cast<double>(z[s])) * dp[i];
            }
            const T v = static_cast<T>(num / W);
            for (int64_t i = s; i < e; ++i) dz[i] = v;
        });
    }

    template <class T>
    void vjp(const T *x, const T *z, const uint8_t *code, const T *g, T *gx, int64_t n, double,
             double *acc, void *) const noexcept {
        blocks(code, n, [&](int64_t s, int64_t e) {
            double W = 0.0, G = 0.0;
            for (int64_t i = s; i < e; ++i) {
                W += wi(i);
                G += static_cast<double>(g[i]);
            }
            const double m = G / W;
            for (int64_t i = s; i < e; ++i) {
                gx[i] = static_cast<T>(wi(i) * m);
                if (!w.empty())
                    acc[i] += (static_cast<double>(x[i]) - static_cast<double>(z[s])) * m;
            }
        });
    }
};

}  // namespace

fp_status make_isotonic(Vec w, bool increasing, fp_op **out) {
    for (double v : w)
        if (!finite_nonneg(v) || v == 0.0) return FP_EPARAM;
    *out = new Isotonic(std::move(w), increasing);
    return FP_OK;
}

}  // namespace fp
