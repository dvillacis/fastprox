// The sorted l1 norm (OWL, SLOPE): g(z) = sum_k w_k |z|_(k), with |z|_(1) >=
// |z|_(2) >= ... the magnitudes in decreasing order and w_1 >= w_2 >= ... >= 0.
// It is a norm, convex and absolutely symmetric, so on singular values it
// gives the weighted nuclear norm sum_k w_k sigma_k(X) (feedback U5).
//
// Prox (M. Bogdan, E. van den Berg, C. Sabatti, W. Su and E. J. Candes,
// SLOPE: adaptive variable selection via convex optimization, Ann. Appl.
// Stat. 9(3), 2015, Algorithm 3; X. Zeng and M. A. T. Figueiredo, The
// ordered weighted l1 norm: atomic formulation, projections, and algorithms,
// 2014): sort |x| in decreasing order, subtract gamma w, project onto the
// nonincreasing cone by pool-adjacent-violators, clip at 0, and restore the
// order and the signs. Ties in |x| are ordered by index, so the result is
// deterministic, and blocks merge only on a strict violation: with equal
// weights nothing pools and the map is soft thresholding, bit for bit.
//
// Derivatives: on a pooled block B of positive value, every |z_i| is the
// mean over B of |x|_(k) - gamma w_k. So the Jacobian averages the signed
// tangents within the block, d|z|/dgamma is minus the block's mean weight,
// and d|z|/dw_k is -gamma / |B| for each weight position k in B; the block
// at value 0 has zero derivatives (|x| at the threshold counts as
// thresholded, as for soft thresholding). The saved code marks the first
// entry of every block in sorted order (1), and the derivatives sort the
// saved x again into their scratch.
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

struct SortedL1 final : RowOp<SortedL1> {
    Vec w;  // nonincreasing, >= 0; one entry for all positions
    explicit SortedL1(Vec ww) : w(std::move(ww)) {}

    const char *name() const noexcept override { return "sorted_l1"; }
    double step_scale() const noexcept override { return w[0]; }  // the largest weight
    bool conj_homogeneous() const noexcept override { return true; }
    int spectral() const noexcept override { return kSpectralBoth; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    // The order, as (|x_i|, i) pairs (2 n words), and for the prox the stack
    // (start, mean, count): 5 n words.
    size_t scratch_bytes(int64_t n) const noexcept override {
        return 5 * static_cast<size_t>(n) * sizeof(double);
    }
    double wk(int64_t k) const noexcept { return w.size() == 1 ? w[0] : w[k]; }

    struct Key {
        double a;   // |x_i|
        int64_t i;  // its position
    };
    // The positions in decreasing order of |x|, ties by index: keys sorted in
    // place, contiguous, rather than indices compared through x.
    template <class T>
    static void order(const T *x, int64_t n, Key *key) noexcept {
        for (int64_t i = 0; i < n; ++i) key[i] = {std::fabs(static_cast<double>(x[i])), i};
        std::sort(key, key + n, [](const Key &p, const Key &q) { return p.a > q.a || (p.a == q.a && p.i < q.i); });
    }

    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double, double *scr) const noexcept override {
        for (int64_t i = 0; i < n; ++i) scr[i] = std::fabs(x[i]);
        std::sort(scr, scr + n, [](double a, double b) { return a > b; });
        double v = 0.0;
        for (int64_t k = 0; k < n; ++k) v += wk(k) * scr[k];
        return v;
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *code) const noexcept {
        bool nan;
        if (scan_row(x, n, nan) != kRowOk) return kRowNonFinite;
        if (code) std::memset(code, 0, static_cast<size_t>(n));
        if (nan) {
            fill_nan(z, n);
            return kRowOk;
        }
        Key *key = static_cast<Key *>(scr);
        int64_t *start = reinterpret_cast<int64_t *>(key + n);
        double *mean = reinterpret_cast<double *>(start + n);
        double *cnt = mean + n;
        order(x, n, key);
        // Pool-adjacent-violators for a nonincreasing fit, on block means
        // (no sums to overflow): merge while a block's mean is below the next.
        int64_t top = 0;
        for (int64_t k = 0; k < n; ++k) {
            const double tau = gamma * wk(k);
            start[top] = k;
            mean[top] = key[k].a - tau;
            cnt[top] = 1.0;
            ++top;
            while (top > 1 && mean[top - 2] < mean[top - 1]) {
                const double c = cnt[top - 2] + cnt[top - 1], f = cnt[top - 1] / c;
                mean[top - 2] += f * (mean[top - 1] - mean[top - 2]);
                cnt[top - 2] = c;
                --top;
            }
        }
        for (int64_t b = 0; b < top; ++b) {
            const int64_t e = b + 1 < top ? start[b + 1] : n;
            const double v = std::max(mean[b], 0.0);
            for (int64_t k = start[b]; k < e; ++k) {
                const int64_t i = key[k].i;
                z[i] = static_cast<T>(std::copysign(v, static_cast<double>(x[i])));
            }
            if (code) code[key[start[b]].i] = 1;
        }
        return kRowOk;
    }

    // Calls f(s, e) for every block of sorted positions [s, e) whose value is
    // positive; the order is rebuilt from the saved x into scr.
    template <class T, class F>
    static void active_blocks(const T *x, const T *z, const uint8_t *code, int64_t n, void *scr, F &&f) {
        Key *key = static_cast<Key *>(scr);
        order(x, n, key);
        int64_t s = 0;
        for (int64_t k = 1; k <= n; ++k)
            if (k == n || code[key[k].i]) {
                if (z[key[s].i] != 0) f(key, s, k);
                s = k;
            }
    }
    static double sgn(double v) noexcept { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); }

    template <class T>
    void jvp(const T *x, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dp, void *scr) const noexcept {
        for (int64_t i = 0; i < n; ++i) dz[i] = T(0);
        active_blocks(x, z, code, n, scr, [&](const Key *key, int64_t s, int64_t e) {
            double num = 0.0;
            for (int64_t k = s; k < e; ++k) {
                const int64_t i = key[k].i;
                num += sgn(static_cast<double>(x[i])) * static_cast<double>(dx[i]) - dgamma * wk(k);
                if (dp) num -= gamma * (w.size() == 1 ? dp[0] : dp[k]);
            }
            const double m = num / static_cast<double>(e - s);
            for (int64_t k = s; k < e; ++k) {
                const int64_t i = key[k].i;
                dz[i] = static_cast<T>(sgn(static_cast<double>(x[i])) * m);
            }
        });
    }

    template <class T>
    void vjp(const T *x, const T *z, const uint8_t *code, const T *g, T *gx, int64_t n, double gamma,
             double *acc, void *scr) const noexcept {
        for (int64_t i = 0; i < n; ++i) gx[i] = T(0);
        const int64_t P = nparams();
        active_blocks(x, z, code, n, scr, [&](const Key *key, int64_t s, int64_t e) {
            double G = 0.0, W = 0.0;
            for (int64_t k = s; k < e; ++k) {
                G += sgn(static_cast<double>(x[key[k].i])) * static_cast<double>(g[key[k].i]);
                W += wk(k);
            }
            const double len = static_cast<double>(e - s), m = G / len;
            for (int64_t k = s; k < e; ++k) {
                const int64_t i = key[k].i;
                gx[i] = static_cast<T>(sgn(static_cast<double>(x[i])) * m);
                acc[w.size() == 1 ? 0 : k] -= gamma * m;
            }
            acc[P] -= G * (W / len);  // the step: minus the block's mean weight
        });
    }
};

}  // namespace

fp_status make_sorted_l1(Vec w, fp_op **out) {
    if (w.empty()) return FP_EPARAM;
    for (size_t k = 0; k < w.size(); ++k) {
        if (!finite_nonneg(w[k])) return FP_EPARAM;
        if (k > 0 && w[k] > w[k - 1]) return FP_EPARAM;  // nonincreasing
    }
    *out = new SortedL1(std::move(w));
    return FP_OK;
}

}  // namespace fp
