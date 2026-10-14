// 1D total variation: prox of gamma sum_e w_e |z_{e+1} - z_e| along the axis.
//
// Two exact paths:
//  * L. Condat, "A direct algorithm for 1-D total variation denoising", IEEE
//    Signal Processing Letters 20(11), 2013, for uniform weights. At the end
//    of the signal, a restart after a jump uses the plain post-jump bounds
//    instead of the paper's tightened ones (measured over 200k signals: same
//    work, simpler). A budget on rescanned steps bounds the backtracking, which
//    is quadratic on ramps: past it, the row is redone by the taut string.
//  * The taut string (P. L. Davies and A. Kovac, Ann. Statist. 29(1), 2001;
//    weighted form as in A. Barbero and S. Sra, JMLR 19, 2018): the cumulative
//    sum of z is the shortest path through the tube S_k -+ gamma w_k, found by
//    a funnel over the upper and lower convex chains in O(n).
// Zero weights split the row into independent pieces.
//
// Saved structure (code[i] for i >= 1, the edge between i-1 and i):
//   0  i continues the segment of i-1;
//   1  new segment, dual u_{i-1} = +gamma w_{i-1} (a downward jump);
//   2  new segment, dual u_{i-1} = -gamma w_{i-1} (an upward jump);
//   3  new segment with a free edge (zero weight), u_{i-1} = 0.
// With u_e = sum_{i <= e} (x_i - z_i), a segment B = [s, t] satisfies
//   |B| z_B = sum_B x + u_{s-1} - u_t,
// which gives every derivative in O(n).
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <cstring>
#include <utility>

namespace fp {
namespace {

constexpr uint8_t kSame = 0, kDown = 1, kUp = 2, kFree = 3;

inline double sigma_of(uint8_t c) noexcept { return c == kDown ? 1.0 : (c == kUp ? -1.0 : 0.0); }

// Reciprocals 1/m for segment lengths m < kRecip, filled once on first use
// (thread-safe static initialisation). Longer segments divide.
constexpr int64_t kRecip = int64_t(1) << 15;

struct RecipTable {
    double r[kRecip];
    RecipTable() noexcept {
        r[0] = 0.0;
        for (int64_t m = 1; m < kRecip; ++m) r[m] = 1.0 / static_cast<double>(m);
    }
};

inline const double *recip_table() noexcept {
    static const RecipTable t;
    return t.r;
}

// Condat's direct algorithm with uniform threshold lam > 0. Returns false
// when more than `budget` steps had to be rescanned after jumps (the output
// is then incomplete).
//
// The bound updates inside a segment are branch-free selects: whether a bound
// is clamped is close to random on noisy signals, and mispredicted branches
// cost more than always computing the update. The arithmetic is the same
// either way, so this changes speed, not results.
template <class T>
bool condat(const T *y, T *z, int64_t n, double lam, uint8_t *code, int64_t budget) noexcept {
    const double *recip = recip_table();
    int64_t k = 0, k0 = 0, km = 0, kp = 0;
    const double twolam = 2.0 * lam;
    double vmin = static_cast<double>(y[0]) - lam, vmax = static_cast<double>(y[0]) + lam;
    double umin = lam, umax = -lam;
    int64_t rescanned = 0;

    auto jump_down = [&]() {
        const T v = static_cast<T>(vmin);
        for (int64_t i = k0; i <= km; ++i) z[i] = v;
        if (code) code[km + 1] = kDown;
        rescanned += k - km;
        k = k0 = kp = km = km + 1;
        vmin = static_cast<double>(y[k]);
        vmax = vmin + twolam;
        umin = lam;
        umax = -lam;
    };
    auto jump_up = [&]() {
        const T v = static_cast<T>(vmax);
        for (int64_t i = k0; i <= kp; ++i) z[i] = v;
        if (code) code[kp + 1] = kUp;
        rescanned += k - kp;
        k = k0 = km = kp = kp + 1;
        vmax = static_cast<double>(y[k]);
        vmin = vmax - twolam;
        umin = lam;
        umax = -lam;
    };

    for (;;) {
        while (k < n - 1) {
            const double yn = static_cast<double>(y[k + 1]);
            if (yn + umin < vmin - lam) {
                jump_down();
                if (rescanned > budget) return false;
            } else if (yn + umax > vmax + lam) {
                jump_up();
                if (rescanned > budget) return false;
            } else {
                ++k;
                umin += yn - vmin;
                umax += yn - vmax;
                const int64_t m = k - k0 + 1;
                const double r = m < kRecip ? recip[m] : 1.0 / static_cast<double>(m);
                const bool cm = umin >= lam, cp = umax <= -lam;
                vmin = cm ? vmin + (umin - lam) * r : vmin;
                umin = cm ? lam : umin;
                km = cm ? k : km;
                vmax = cp ? vmax + (umax + lam) * r : vmax;
                umax = cp ? -lam : umax;
                kp = cp ? k : kp;
            }
        }
        // End of the signal: the last segment must close with u_{n-1} = 0.
        if (umin < 0.0) {
            jump_down();
        } else if (umax > 0.0) {
            jump_up();
        } else {
            const T v = static_cast<T>(vmin + umin / static_cast<double>(k - k0 + 1));
            for (int64_t i = k0; i < n; ++i) z[i] = v;
            return true;
        }
        if (rescanned > budget) return false;
    }
}

// Tube points carry their height as an unevaluated sum hi + lo: the
// cumulative sums grow like n |x|, and plain doubles would lose the
// 10 n eps |x|_inf accuracy the spec asks for on long rows.
struct Pt {
    int64_t i;
    double hi, lo;
};

inline void two_sum(double a, double b, double &s, double &e) noexcept {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

inline Pt shifted(int64_t i, double shi, double slo, double add) noexcept {
    Pt p;
    p.i = i;
    double e;
    two_sum(shi, add, p.hi, e);
    p.lo = slo + e;
    return p;
}

inline double rise(const Pt &a, const Pt &b) noexcept { return (b.hi - a.hi) + (b.lo - a.lo); }

// slope(a -> b) <= slope(a -> c), abscissae b.i, c.i > a.i.
inline bool slope_le(const Pt &a, const Pt &b, const Pt &c) noexcept {
    return rise(a, b) * static_cast<double>(c.i - a.i) <= rise(a, c) * static_cast<double>(b.i - a.i);
}
inline bool slope_ge(const Pt &a, const Pt &b, const Pt &c) noexcept {
    return rise(a, b) * static_cast<double>(c.i - a.i) >= rise(a, c) * static_cast<double>(b.i - a.i);
}

inline size_t taut_scratch(int64_t n) noexcept { return 2 * static_cast<size_t>(n + 1) * sizeof(Pt); }

// Taut string on y[0..m) with thresholds gamma * w[e] (w == nullptr: gamma * wu),
// all positive. Writes z[0..m) and code[1..m).
template <class T>
void taut_string(const T *y, T *z, int64_t m, double gamma, const double *w, double wu,
                 uint8_t *code, void *scr) noexcept {
    if (m == 1) {
        z[0] = y[0];
        return;
    }
    // The upper and lower chains are deques stored as [head, tail) in arrays
    // of m + 1 points; both heads are the current apex of the funnel.
    Pt *up = static_cast<Pt *>(scr);
    Pt *lo = up + (m + 1);
    int64_t uhd = 0, utl = 1, lhd = 0, ltl = 1;
    up[0] = lo[0] = Pt{0, 0.0, 0.0};

    auto emit = [&](const Pt &a, const Pt &b, uint8_t c) {
        const T v = static_cast<T>(rise(a, b) / static_cast<double>(b.i - a.i));
        for (int64_t i = a.i; i < b.i; ++i) z[i] = v;
        if (code && b.i < m) code[b.i] = c;
    };

    double shi = 0.0, slo = 0.0;  // cumulative sum S_k = shi + slo
    for (int64_t k = 1; k < m; ++k) {
        double e;
        two_sum(shi, static_cast<double>(y[k - 1]), shi, e);
        slo += e;
        const double lam = gamma * (w ? w[k - 1] : wu);
        const Pt U = shifted(k, shi, slo, lam), L = shifted(k, shi, slo, -lam);

        // Upper tube point: if it falls below the lower chain, the string
        // wraps over lower points, which become fixed.
        if (ltl - lhd >= 2 && slope_le(lo[lhd], U, lo[lhd + 1])) {
            do {
                emit(lo[lhd], lo[lhd + 1], kDown);
                ++lhd;
            } while (ltl - lhd >= 2 && slope_le(lo[lhd], U, lo[lhd + 1]));
            up[0] = lo[lhd];
            up[1] = U;
            uhd = 0;
            utl = 2;
        } else {
            while (utl - uhd >= 2 && slope_le(up[utl - 2], U, up[utl - 1])) --utl;
            up[utl++] = U;
        }

        // Lower tube point, symmetrically.
        if (utl - uhd >= 2 && slope_ge(up[uhd], L, up[uhd + 1])) {
            do {
                emit(up[uhd], up[uhd + 1], kUp);
                ++uhd;
            } while (utl - uhd >= 2 && slope_ge(up[uhd], L, up[uhd + 1]));
            lo[0] = up[uhd];
            lo[1] = L;
            lhd = 0;
            ltl = 2;
        } else {
            while (ltl - lhd >= 2 && slope_ge(lo[ltl - 2], L, lo[ltl - 1])) --ltl;
            lo[ltl++] = L;
        }
    }
    double e;
    two_sum(shi, static_cast<double>(y[m - 1]), shi, e);
    const Pt E{m, shi, slo + e};  // the end point is pinned at (m, S_m)

    bool walked = false;
    while (ltl - lhd >= 2 && slope_le(lo[lhd], E, lo[lhd + 1])) {
        emit(lo[lhd], lo[lhd + 1], kDown);
        ++lhd;
        walked = true;
    }
    if (walked) {
        up[0] = lo[lhd];
        uhd = 0;
        utl = 1;
    }
    while (utl - uhd >= 2 && slope_ge(up[uhd], E, up[uhd + 1])) {
        emit(up[uhd], up[uhd + 1], kUp);
        ++uhd;
    }
    emit(up[uhd], E, kSame);
}

struct TV1D final : RowOp<TV1D> {
    Vec w;
    bool uniform;
    fp_tv1d_method method;

    TV1D(Vec ww, bool uni, fp_tv1d_method m) : w(std::move(ww)), uniform(uni), method(m) {}

    const char *name() const noexcept override { return "tv1d"; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n - 1) ? FP_OK : FP_ESHAPE;
    }
    size_t scratch_bytes(int64_t n) const noexcept override { return taut_scratch(n); }
    bool needs_distinct() const noexcept override { return true; }

    double we(int64_t e) const noexcept { return w.size() == 1 ? w[0] : w[e]; }

    // z = x with the structure of the limit gamma -> 0+.
    template <class T>
    void identity(const T *x, T *z, int64_t n, uint8_t *code) const noexcept {
        std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
        if (!code) return;
        for (int64_t i = 1; i < n; ++i) {
            if (we(i - 1) == 0.0) code[i] = kFree;
            else if (x[i] < x[i - 1]) code[i] = kDown;
            else if (x[i] > x[i - 1]) code[i] = kUp;
            else code[i] = kSame;
        }
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
        if (gamma == 0.0) {
            identity(x, z, n, code);
            return kRowOk;
        }
        if (uniform) {
            const double lam = gamma * w[0];
            if (lam == 0.0) {
                identity(x, z, n, code);
                return kRowOk;
            }
            if (method != FP_TV1D_TAUT_STRING) {
                // Typical signals rescan 0 to 3 steps per sample; ramps and
                // noisy ramps with large gamma rescan 10 to 3000 (quadratic).
                // Past about 3n rescanned steps the taut string is faster.
                const int64_t budget = method == FP_TV1D_CONDAT ? INT64_MAX : 3 * n + 1024;
                if (condat(x, z, n, lam, code, budget)) return kRowOk;
                if (code) std::memset(code, 0, static_cast<size_t>(n));
            }
            taut_string(x, z, n, gamma, nullptr, w[0], code, scr);
            return kRowOk;
        }
        // Weighted: independent pieces between zero-weight edges.
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e == n - 1 || w[e] == 0.0) {
                taut_string(x + s, z + s, e - s + 1, gamma, w.data() + s, 0.0, code ? code + s : nullptr,
                            scr);
                if (code) {
                    if (s > 0) code[s] = kFree;
                    if (e < n - 1) code[e + 1] = kFree;
                }
                s = e + 1;
            }
        }
        return kRowOk;
    }

    // Calls f(s, t, sl, sr) for every segment [s, t] with the dual signs at
    // its left edge (s-1) and right edge (t); 0 at the ends of the row.
    template <class F>
    static void segments(const uint8_t *code, int64_t n, F &&f) {
        int64_t s = 0;
        double sl = 0.0;
        for (int64_t i = 1; i <= n; ++i) {
            if (i == n || code[i] != kSame) {
                const double sr = i == n ? 0.0 : sigma_of(code[i]);
                f(s, i - 1, sl, sr);
                s = i;
                sl = sr;
            }
        }
    }

    template <class T>
    void jvp(const T *, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dp, void *) const noexcept {
        segments(code, n, [&](int64_t s, int64_t t, double sl, double sr) {
            double num = 0.0;
            for (int64_t i = s; i <= t; ++i) num += static_cast<double>(dx[i]);
            if (s > 0) {
                num += sl * we(s - 1) * dgamma;
                if (dp) num += gamma * sl * dp[w.size() == 1 ? 0 : s - 1];
            }
            if (t < n - 1) {
                num -= sr * we(t) * dgamma;
                if (dp) num -= gamma * sr * dp[w.size() == 1 ? 0 : t];
            }
            const T v = static_cast<T>(num / static_cast<double>(t - s + 1));
            for (int64_t i = s; i <= t; ++i) dz[i] = v;
        });
    }

    template <class T>
    void vjp(const T *, const T *, const uint8_t *code, const T *g, T *gx, int64_t n, double gamma,
             double *acc, void *) const noexcept {
        const int64_t P = nparams();
        segments(code, n, [&](int64_t s, int64_t t, double sl, double sr) {
            double G = 0.0;
            for (int64_t i = s; i <= t; ++i) G += static_cast<double>(g[i]);
            const double m = G / static_cast<double>(t - s + 1);
            for (int64_t i = s; i <= t; ++i) gx[i] = static_cast<T>(m);
            if (s > 0 && sl != 0.0) {
                acc[P] += sl * we(s - 1) * m;
                acc[w.size() == 1 ? 0 : s - 1] += gamma * sl * m;
            }
            if (t < n - 1 && sr != 0.0) {
                acc[P] -= sr * we(t) * m;
                acc[w.size() == 1 ? 0 : t] -= gamma * sr * m;
            }
        });
    }
};

}  // namespace

fp_status make_tv1d(Vec w, fp_tv1d_method method, fp_op **out) {
    if (method != FP_TV1D_AUTO && method != FP_TV1D_CONDAT && method != FP_TV1D_TAUT_STRING)
        return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    bool uniform = true;
    for (double v : w) {
        if (!finite_nonneg(v)) return FP_EPARAM;
        uniform = uniform && v == w[0];
    }
    if (method == FP_TV1D_CONDAT && !uniform) return FP_EPARAM;
    *out = new TV1D(std::move(w), uniform, method);
    return FP_OK;
}

}  // namespace fp
