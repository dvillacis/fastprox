// Projection onto the monotone cone (isotonic regression) by
// pool-adjacent-violators: one pass with a stack of blocks, merging while the
// previous block mean exceeds the current one. Minimises
// sum_i w_i (z_i - x_i)^2 subject to z nondecreasing (or nonincreasing).
//
// The stack keeps block sums and weights; if a sum overflows (audit K8), the
// row is pooled again on block means, which cannot. Weights are stored over 2^k with their
// maximum in [1/2, 1), an exact scaling, so block weights stay below n; the
// derivative terms in the weights carry the 2^-k back. k is kept as an
// exponent: 2^-k overflowed for weights below 2^-1023 (nightly fuzz, NaN from
// weights of 1e-315), so the terms scale their weight factor with ldexp. Weight ratios below
// 1e-100 are rejected, as for the weighted l1 ball, so none underflows.
//
// Saved structure (code): 1 at the first entry of every pooled block.
// The Jacobian averages over blocks with the weights:
//   (J v)_i = sum_{j in B} w_j v_j / W_B, dz_B/dw_j = (x_j - z_B) / W_B.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace fp {
namespace {

struct Isotonic final : RowOp<Isotonic> {
    static constexpr int kStruct = FP_STRUCT_PROJECTOR;
    Vec w;  // empty: unit weights; else the weights over 2^k
    int wk = 0;  // k
    bool increasing;
    bool euclidean = true;  // unit or equal weights: the Euclidean projection
    Isotonic(Vec ww, bool inc) : w(std::move(ww)), increasing(inc) {
        if (w.empty()) return;
        std::frexp(vmax(w), &wk);
        for (double &v : w) {
            v = std::ldexp(v, -wk);
            euclidean = euclidean && v == w[0];
        }
    }

    const char *name() const noexcept override { return "isotonic"; }
    double step_scale() const noexcept override { return 0.0; }
    bool conj_homogeneous() const noexcept override { return true; }
    // Unequal weights project in the metric W, where y - P_C^W(y) is not the
    // Euclidean conjugate's prox: W (y - P_C^W(y)) lies in the polar cone,
    // y - P_C^W(y) need not (math audit M2).
    bool conj_supported() const noexcept override { return euclidean; }
    // Unequal weights average in the metric diag(w).
    int struct_kind() const noexcept override { return euclidean ? FP_STRUCT_PROJECTOR : FP_STRUCT_WEIGHTED; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.empty() || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    size_t scratch_bytes(int64_t n) const noexcept override {
        return 3 * static_cast<size_t>(n) * sizeof(double);
    }
    double wi(int64_t i) const noexcept { return w.empty() ? 1.0 : w[i]; }

    bool has_value() const noexcept override { return true; }
    // The indicator of the monotone cone.
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        double v = 0.0;
        for (int64_t i = 0; i + 1 < n; ++i) v = std::max(v, increasing ? x[i] - x[i + 1] : x[i + 1] - x[i]);
        return indicator(v, value_atol(x, n, tol));
    }

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
        // Block sums, as before; if one overflows (entries near DBL_MAX,
        // audit K8), the row again with block means, one division per merge:
        // slower here, as the division sits on each merge's critical path.
        int64_t top = 0;  // number of blocks
        bool finite = true;
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
        // An overflowed sum stays inf (or NaN), so the final blocks tell.
        for (int64_t b = 0; b < top; ++b) finite = finite && std::isfinite(bs[b]);
        if (finite) {
            for (int64_t b = 0; b < top; ++b) bs[b] /= bw[b];  // the means
        } else {
            top = 0;
            for (int64_t i = 0; i < n; ++i) {
                start[top] = i;
                bw[top] = wi(i);
                bs[top] = sg * static_cast<double>(x[i]);
                ++top;
                while (top > 1 && bs[top - 2] > bs[top - 1]) {
                    const double W = bw[top - 2] + bw[top - 1], f = bw[top - 1] / W;
                    const double m2 = bs[top - 2], m1 = bs[top - 1], d = m1 - m2;
                    // The update; the convex form where opposite extreme
                    // means make the difference overflow.
                    bs[top - 2] = std::isfinite(d) ? m2 + f * d : (1.0 - f) * m2 + f * m1;
                    bw[top - 2] = W;
                    --top;
                }
            }
        }
        for (int64_t b = 0; b < top; ++b) {
            const int64_t e = b + 1 < top ? start[b + 1] : n;
            const T m = static_cast<T>(sg * bs[b]);
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

    // Structure (since 1.0.2): every entry FREE, the pooled blocks as regions.
    template <class T>
    int64_t structure(const T *, const T *, const uint8_t *code, int64_t n, double, RowStruct &s,
                      void *) const noexcept {
        return runs(s, n, [code](int64_t i) { return code[i] != 0; });
    }

    template <class T>
    void jvp(const T *x, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n, double,
             double, const double *dp, void *) const noexcept {
        blocks(code, n, [&](int64_t s, int64_t e) {
            double W = 0.0, num = 0.0;
            for (int64_t i = s; i < e; ++i) {
                W += wi(i);
                num += wi(i) * static_cast<double>(dx[i]);
                if (dp) num += (static_cast<double>(x[i]) - static_cast<double>(z[s])) * std::ldexp(dp[i], -wk);
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
                    acc[i] += (static_cast<double>(x[i]) - static_cast<double>(z[s])) * std::ldexp(m, -wk);
            }
        });
    }
};

}  // namespace

fp_status make_isotonic(Vec w, bool increasing, fp_op **out) {
    double wmin = 0.0;
    for (double v : w) {
        if (!finite_nonneg(v) || v == 0.0) return FP_EPARAM;
        wmin = wmin == 0.0 ? v : std::min(wmin, v);
    }
    if (!w.empty() && wmin / vmax(w) < 1e-100) return FP_EPARAM;  // stored over 2^k, they would underflow
    *out = new Isotonic(std::move(w), increasing);
    return FP_OK;
}

}  // namespace fp
