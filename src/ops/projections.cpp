// Projections onto the l2 ball, the simplex and the (weighted) l1 ball, and
// the prox of the l-infinity norm through the Moreau identity.
//
// Simplex and l1 ball: L. Condat, "Fast projection onto the simplex and the
// l1 ball", Math. Programming 158, 2016 (the filtered pivot method of its
// Section 3). The weighted variant replaces the pivot (sum_V a - r)/|V| by
// (sum_V w a - r)/sum_V w^2 and the breakpoints a_i by a_i/w_i; the pivot of
// any candidate set is a lower bound on the final threshold, and it only
// increases, so the method stays exact.
//
// Saved structure (code): 1 on the support of the projection, 0 elsewhere,
// and 3 on every entry when the input was already inside the l1 ball.
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

constexpr uint8_t kInside = 3;

// Threshold tau with sum_i w_i max(a_i - tau w_i, 0) = r, for r > 0. a and w
// are accessors; v and vt are scratch lists of n indices each.
template <bool Weighted, class A, class W>
double condat_tau(int64_t n, A a, W w, double r, int64_t *v, int64_t *vt) noexcept {
    int64_t nv = 0, nvt = 0;
    double s1, s2, rho;
    {
        const double w0 = Weighted ? w(0) : 1.0;
        v[nv++] = 0;
        s1 = w0 * a(0);
        s2 = w0 * w0;
        rho = (s1 - r) / s2;
    }
    // Step 1: single pass with a running pivot.
    for (int64_t i = 1; i < n; ++i) {
        const double ai = a(i), wi = Weighted ? w(i) : 1.0;
        if (ai > rho * wi) {
            const double t1 = s1 + wi * ai, t2 = s2 + wi * wi;
            const double rn = (t1 - r) / t2;
            if (rn > (wi * ai - r) / (wi * wi)) {
                v[nv++] = i;
                s1 = t1;
                s2 = t2;
                rho = rn;
            } else {
                for (int64_t j = 0; j < nv; ++j) vt[nvt++] = v[j];
                nv = 0;
                v[nv++] = i;
                s1 = wi * ai;
                s2 = wi * wi;
                rho = (s1 - r) / s2;
            }
        }
    }
    // Step 2: recover candidates set aside by the resets. bmax is the
    // largest breakpoint a_i / w_i of a candidate left out in steps 2 and 3:
    // in exact arithmetic the threshold is at least bmax. When an entry sits
    // within rounding of the pivot and dominates the sums, removing it can
    // cancel them (s2 = 1 + 1e-126 - 1 = 0) and leave a threshold far below
    // bmax, with outputs larger than the inputs; clamping to bmax restores
    // a solution that is correct to rounding (found by fuzzing).
    double bmax = -HUGE_VAL;
    for (int64_t j = 0; j < nvt; ++j) {
        const int64_t i = vt[j];
        const double ai = a(i), wi = Weighted ? w(i) : 1.0;
        if (ai > rho * wi) {
            v[nv++] = i;
            s1 += wi * ai;
            s2 += wi * wi;
            rho = (s1 - r) / s2;
        } else {
            bmax = std::max(bmax, ai / wi);
        }
    }
    // Step 3: remove candidates below the pivot until none is left.
    bool changed = true;
    while (changed && nv > 1) {
        changed = false;
        int64_t m = 0;
        for (int64_t j = 0; j < nv; ++j) {
            const int64_t i = v[j];
            const double ai = a(i), wi = Weighted ? w(i) : 1.0;
            if (ai > rho * wi || nv - (j - m) == 1) {
                v[m++] = i;
            } else {
                bmax = std::max(bmax, ai / wi);
                s1 -= wi * ai;
                s2 -= wi * wi;
                rho = (s1 - r) / s2;
                changed = true;
            }
        }
        nv = m;
    }
    // Recompute the pivot from the final set without running-sum drift.
    s1 = s2 = 0.0;
    for (int64_t j = 0; j < nv; ++j) {
        const int64_t i = v[j];
        const double wi = Weighted ? w(i) : 1.0;
        s1 += wi * a(i);
        s2 += wi * wi;
    }
    return std::max((s1 - r) / s2, bmax);
}

// The same method with unit weights, keeping candidate values rather than
// indices: no indirect loads, and the pivot is (sum_V a - r) / |V|.
//
// If finite is not null, step 1 also tests every value for NaN or infinity
// (an integer test on the exponent bits, off the floating-point dependency
// chain): it reads every entry anyway, which saves a separate pass. On a
// non-finite value *finite is false and the result is meaningless.
template <class A>
double condat_tau_unit(int64_t n, A a, double r, double *v, double *vt,
                       bool *finite = nullptr) noexcept {
    constexpr uint64_t kExp = 0x7ff0000000000000ull;
    uint64_t bad = 0;
    auto check = [&](double d) {
        uint64_t b;
        std::memcpy(&b, &d, sizeof b);
        bad |= static_cast<uint64_t>((b & kExp) == kExp);
    };
    int64_t nv = 0, nvt = 0;
    v[nv++] = a(0);
    if (finite) check(v[0]);
    double s1 = v[0];
    double rho = s1 - r;
    auto visit = [&](double ai) {
        if (ai > rho) {
            const double t1 = s1 + ai;
            const double rn = (t1 - r) / static_cast<double>(nv + 1);
            if (rn > ai - r) {
                v[nv++] = ai;
                s1 = t1;
                rho = rn;
            } else {
                for (int64_t j = 0; j < nv; ++j) vt[nvt++] = v[j];
                nv = 0;
                v[nv++] = ai;
                s1 = ai;
                rho = ai - r;
            }
        }
    };
    // Entries at or below the running pivot are skipped, and the pivot only
    // changes on an entry above it, so a block whose maximum is at or below
    // the pivot can be skipped whole: one compare per block instead of one
    // per entry, with the same result. Once the pivot has grown, almost every
    // block is skipped. The block maximum ignores NaN; with `finite` the
    // result is discarded for such rows anyway.
    // Blocks of 8 (4 and 16 were slower with one of Clang or GCC), held in
    // locals: a scratch array made GCC on x86_64 reload the values with
    // wider loads than it stored them, which stalls, and doubled the time.
    constexpr int64_t kBlock = 8;
    int64_t i = 1;
    for (; i + kBlock <= n; i += kBlock) {
        const double a0 = a(i), a1 = a(i + 1), a2 = a(i + 2), a3 = a(i + 3);
        const double a4 = a(i + 4), a5 = a(i + 5), a6 = a(i + 6), a7 = a(i + 7);
        if (finite) {
            check(a0), check(a1), check(a2), check(a3);
            check(a4), check(a5), check(a6), check(a7);
        }
        const double m = max_nonan(max_nonan(max_nonan(a0, a1), max_nonan(a2, a3)),
                                   max_nonan(max_nonan(a4, a5), max_nonan(a6, a7)));
        if (m > rho) {
            visit(a0), visit(a1), visit(a2), visit(a3);
            visit(a4), visit(a5), visit(a6), visit(a7);
        }
    }
    for (; i < n; ++i) {
        const double ai = a(i);
        if (finite) check(ai);
        visit(ai);
    }
    if (finite) {
        *finite = bad == 0;
        if (bad) return 0.0;
    }
    double bmax = -HUGE_VAL;  // as in condat_tau
    for (int64_t j = 0; j < nvt; ++j) {
        const double ai = vt[j];
        if (ai > rho) {
            v[nv++] = ai;
            s1 += ai;
            rho = (s1 - r) / static_cast<double>(nv);
        } else {
            bmax = std::max(bmax, ai);
        }
    }
    bool changed = true;
    while (changed && nv > 1) {
        changed = false;
        int64_t m = 0;
        for (int64_t j = 0; j < nv; ++j) {
            const double ai = v[j];
            if (ai > rho || nv - (j - m) == 1) {
                v[m++] = ai;
            } else {
                bmax = std::max(bmax, ai);
                s1 -= ai;
                rho = (s1 - r) / static_cast<double>(nv - (j - m) - 1);
                changed = true;
            }
        }
        nv = m;
    }
    s1 = 0.0;
    for (int64_t j = 0; j < nv; ++j) s1 += v[j];
    return std::max((s1 - r) / static_cast<double>(nv), bmax);
}

inline size_t condat_scratch(int64_t n) noexcept {
    return 2 * static_cast<size_t>(n) * sizeof(int64_t);  // n indices or n doubles, twice
}

// ---------------------------------------------------------------------------

struct L2Ball final : RowOp<L2Ball> {
    double r;
    explicit L2Ball(double rr) : r(rr) {}
    const char *name() const noexcept override { return "l2_ball"; }
    double step_scale() const noexcept override { return 0.0; }
    int spectral() const noexcept override { return kSpectralBoth; }  // Frobenius ball
    int64_t nparams() const noexcept override { return 1; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    // f(m, t) with ||x|| = m / t (with_norm in core/op.hpp), t = 1 unless the
    // norm exceeds the largest double or is subnormal. The formulas read it
    // through r t / m, (x_i t) / m and <t x, v> / m, which are those of the
    // plain norm, bit for bit, when t = 1.
    template <class T, class F>
    static auto with_norm(const T *x, int64_t n, F f) noexcept {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        return fp::with_norm(s, n, [x](int64_t i) { return x[i]; }, f);
    }

    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        return with_norm(x, n, [=](double m, double t) { return indicator(m / t - r, value_atol(x, n, tol)); });
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double, void *, uint8_t *) const noexcept {
        for (int64_t i = 0; i < n; ++i)
            if (std::isinf(x[i])) return kRowNonFinite;
        return with_norm(x, n, [=](double m, double t) {
            if (m <= r * t) {
                if (z != x) std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
                return kRowOk;
            }
            const double c = r / m;  // NaN when the norm is NaN
            for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(static_cast<double>(x[i]) * t * c);
            return kRowOk;
        });
    }
    template <class T>
    void jvp(const T *x, const T *, const uint8_t *, const T *dx, T *dz, int64_t n, double, double,
             const double *dp, void *) const noexcept {
        with_norm(x, n, [=](double m, double t) {
            if (m <= r * t) {
                for (int64_t i = 0; i < n; ++i) dz[i] = dx[i];
                return;
            }
            double ud = 0.0;
            for (int64_t i = 0; i < n; ++i) ud += static_cast<double>(x[i]) * t * static_cast<double>(dx[i]);
            ud /= m;
            const double c = r * t / m, dr = dp ? dp[0] : 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double ui = static_cast<double>(x[i]) * t / m;
                dz[i] = static_cast<T>(c * (static_cast<double>(dx[i]) - ui * ud) + dr * ui);
            }
        });
    }
    template <class T>
    void vjp(const T *x, const T *, const uint8_t *, const T *g, T *gx, int64_t n, double,
             double *acc, void *) const noexcept {
        with_norm(x, n, [=](double m, double t) {
            if (m <= r * t) {
                for (int64_t i = 0; i < n; ++i) gx[i] = g[i];
                return;
            }
            double ug = 0.0;
            for (int64_t i = 0; i < n; ++i) ug += static_cast<double>(x[i]) * t * static_cast<double>(g[i]);
            ug /= m;
            const double c = r * t / m;
            for (int64_t i = 0; i < n; ++i)
                gx[i] = static_cast<T>(c * (static_cast<double>(g[i]) - static_cast<double>(x[i]) * t / m * ug));
            acc[0] += ug;
        });
    }
};

// ---------------------------------------------------------------------------

struct Simplex final : RowOp<Simplex> {
    double r;
    explicit Simplex(double rr) : r(rr) {}
    const char *name() const noexcept override { return "simplex"; }
    double step_scale() const noexcept override { return 0.0; }
    int spectral() const noexcept override { return kSpectralEig; }  // the spectraplex
    int64_t nparams() const noexcept override { return 1; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }
    size_t scratch_bytes(int64_t n) const noexcept override { return condat_scratch(n); }

    bool has_value() const noexcept override { return true; }
    // The indicator of x >= 0, sum x = r.
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        double s = 0.0, v = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            s += x[i];
            v = std::max(v, -x[i]);
        }
        return indicator(std::max(v, std::fabs(s - r)), value_atol(x, n, tol));
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double, void *scr, uint8_t *code) const noexcept {
        // Non-finite input: an error for infinities, NaN in the whole row.
        auto nonfinite = [&]() {
            bool nan;
            if (scan_row(x, n, nan) != kRowOk) return static_cast<int>(kRowNonFinite);
            if (nan) fill_nan(z, n); else std::memset(z, 0, static_cast<size_t>(n) * sizeof(T));
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            return static_cast<int>(kRowOk);
        };
        if (r == 0.0) return nonfinite();  // z = 0 unless the row has NaN or an infinity
        double *v = static_cast<double *>(scr);
        bool finite;
        const double tau = condat_tau_unit(
            n, [x](int64_t i) { return static_cast<double>(x[i]); }, r, v, v + n, &finite);
        if (!finite) return nonfinite();
        if (code) {
            for (int64_t i = 0; i < n; ++i) {
                const double d = static_cast<double>(x[i]) - tau;
                z[i] = static_cast<T>(d > 0 ? d : 0.0);
                code[i] = d > 0;
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {  // vectorises
                const double d = static_cast<double>(x[i]) - tau;
                z[i] = static_cast<T>(d > 0 ? d : 0.0);
            }
        }
        return kRowOk;
    }
    // J = P_S - 1_S 1_S^T / |S|, dz/dr = 1_S / |S|.
    template <class T>
    void jvp(const T *, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n, double,
             double, const double *dp, void *) const noexcept {
        double sum = 0.0;
        int64_t k = 0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                sum += static_cast<double>(dx[i]);
                ++k;
            }
        const double shift = k ? (sum - (dp ? dp[0] : 0.0)) / static_cast<double>(k) : 0.0;
        for (int64_t i = 0; i < n; ++i)
            dz[i] = code[i] ? static_cast<T>(static_cast<double>(dx[i]) - shift) : T(0);
    }
    template <class T>
    void vjp(const T *, const T *, const uint8_t *code, const T *g, T *gx, int64_t n, double,
             double *acc, void *) const noexcept {
        double sum = 0.0;
        int64_t k = 0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                sum += static_cast<double>(g[i]);
                ++k;
            }
        const double mean = k ? sum / static_cast<double>(k) : 0.0;
        for (int64_t i = 0; i < n; ++i)
            gx[i] = code[i] ? static_cast<T>(static_cast<double>(g[i]) - mean) : T(0);
        acc[0] += mean;
    }

    // Kinks (since 0.3): code 1 at an entry within tol of the threshold tau
    // (read off a support entry, tau = x_i - z_i), on or off the support.
    // Overrides: side +1 puts it on the support (the x + eps piece), -1 off.
    static double threshold(const double *x, const double *z, int64_t n) noexcept {
        for (int64_t i = 0; i < n; ++i)
            if (z[i] > 0.0) return x[i] - z[i];
        return HUGE_VAL;  // no support (r = 0): no kinks
    }
    bool has_kinks() const noexcept override { return true; }
    bool has_branches() const noexcept override { return true; }
    int64_t kink_row(const double *x, const double *z, int64_t n, double, double tol, int8_t *mask,
                     int64_t ms, void *) const noexcept override {
        const double tau = threshold(x, z, n);
        int64_t count = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int8_t c = std::fabs(x[i] - tau) <= tol;
            if (mask) mask[i * ms] = c;
            count += c;
        }
        return count;
    }
    void branch_row(const double *x, const double *z, uint8_t *code, int64_t n, double, double tol,
                    const int8_t *side, void *) const noexcept override {
        const double tau = threshold(x, z, n);
        for (int64_t i = 0; i < n; ++i)
            if (side[i] != 0 && std::fabs(x[i] - tau) <= tol) code[i] = side[i] > 0;
    }
};

// ---------------------------------------------------------------------------

constexpr double kL1Inside = -1.0, kL1Infinite = -2.0, kL1NaN = -3.0;

// Projection of x onto {sum_i w_i |z_i| <= radius}; shared by the l1 ball and
// the l-infinity prox. Returns tau with z_i = sign(x_i) max(|x_i| - tau w_i, 0),
// kL1Inside when x is inside the ball, or kL1Infinite / kL1NaN for rows with
// an infinity or a NaN. The l1 norm doubles as the finiteness test: it is
// non-finite exactly when an entry is, or when it overflows.
template <class T>
double l1_threshold(const T *x, int64_t n, const Vec &w, double radius, void *scr) noexcept {
    const bool weighted = w.size() > 1;
    // Eight partial sums in a fixed order: one running sum is a chain of
    // dependent additions (3 to 4 cycles per entry). The norm only decides
    // whether x is inside the ball and whether the row is finite.
    constexpr int64_t kLanes = 8;
    double acc[kLanes] = {};
    int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes)
        for (int64_t j = 0; j < kLanes; ++j)
            acc[j] += (weighted ? w[i + j] : w[0]) * std::fabs(static_cast<double>(x[i + j]));
    for (; i < n; ++i) acc[0] += (weighted ? w[i] : w[0]) * std::fabs(static_cast<double>(x[i]));
    for (int64_t h = kLanes / 2; h > 0; h /= 2)
        for (int64_t j = 0; j < h; ++j) acc[j] += acc[j + h];
    const double norm1 = acc[0];
    if (!(norm1 <= 1.7976931348623157e308)) {
        bool nan;
        if (scan_row(x, n, nan) != kRowOk) return kL1Infinite;
        if (nan) return kL1NaN;
        // Finite entries whose norm overflows: certainly outside the ball.
    }
    if (norm1 <= radius) return kL1Inside;
    if (radius == 0.0) return HUGE_VAL;
    auto a = [x](int64_t i) { return std::fabs(static_cast<double>(x[i])); };
    if (weighted) {
        int64_t *v = static_cast<int64_t *>(scr);
        return condat_tau<true>(n, a, [&w](int64_t i) { return w[i]; }, radius, v, v + n);
    }
    // A uniform weight c: project onto sum |z| <= radius / c.
    double *v = static_cast<double *>(scr);
    return condat_tau_unit(n, a, radius / w[0], v, v + n) / w[0];
}

// Weights are stored divided by their maximum `scale` (the projection
// depends only on their ratios, with radius r / scale), so that sums of
// squared weights cannot underflow for tiny but valid weights.
struct L1Ball final : RowOp<L1Ball> {
    double r;
    Vec w;         // normalised: max w = 1
    double scale;  // the largest weight given
    L1Ball(double rr, Vec ww, double sc) : r(rr), w(std::move(ww)), scale(sc) {}
    const char *name() const noexcept override { return "l1_ball"; }
    double step_scale() const noexcept override { return 0.0; }
    int spectral() const noexcept override { return uniform(w) ? kSpectralBoth : 0; }  // nuclear-norm ball
    int64_t nparams() const noexcept override { return 1; }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    size_t scratch_bytes(int64_t n) const noexcept override { return condat_scratch(n); }
    double wi(int64_t i) const noexcept { return w.size() == 1 ? w[0] : w[i]; }

    bool has_value() const noexcept override { return true; }
    // The indicator of sum_i w_i |x_i| <= r, in the weights given (w scale).
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += wi(i) * std::fabs(x[i]);
        return indicator(scale * s - r, value_atol(x, n, tol));
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double, void *scr, uint8_t *code) const noexcept {
        const double tau = l1_threshold(x, n, w, r / scale, scr);
        if (tau == kL1Infinite) return kRowNonFinite;
        if (tau == kL1NaN) {
            fill_nan(z, n);
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            return kRowOk;
        }
        if (tau == kL1Inside) {
            if (z != x) std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
            if (code) std::memset(code, kInside, static_cast<size_t>(n));
            return kRowOk;
        }
        for (int64_t i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);
            const double d = std::fabs(xi) - tau * wi(i);
            z[i] = static_cast<T>(d > 0 ? std::copysign(d, xi) : 0.0);
            if (code) code[i] = d > 0;
        }
        return kRowOk;
    }
    // Active: J = P_S - a a^T / W2 with a_i = s_i w_i, W2 = sum_S w_i^2;
    // dz/dr = a / W2.
    template <class T>
    void jvp(const T *x, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n, double,
             double, const double *dp, void *) const noexcept {
        if (code[0] == kInside) {
            for (int64_t i = 0; i < n; ++i) dz[i] = dx[i];
            return;
        }
        double dot = 0.0, w2 = 0.0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                const double a = std::copysign(wi(i), static_cast<double>(x[i]));
                dot += a * static_cast<double>(dx[i]);
                w2 += wi(i) * wi(i);
            }
        const double c = w2 > 0 ? (dot - (dp ? dp[0] / scale : 0.0)) / w2 : 0.0;
        for (int64_t i = 0; i < n; ++i) {
            if (code[i]) {
                const double a = std::copysign(wi(i), static_cast<double>(x[i]));
                dz[i] = static_cast<T>(static_cast<double>(dx[i]) - a * c);
            } else {
                dz[i] = T(0);
            }
        }
    }
    template <class T>
    void vjp(const T *x, const T *, const uint8_t *code, const T *g, T *gx, int64_t n, double,
             double *acc, void *) const noexcept {
        if (code[0] == kInside) {
            for (int64_t i = 0; i < n; ++i) gx[i] = g[i];
            return;
        }
        double dot = 0.0, w2 = 0.0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                const double a = std::copysign(wi(i), static_cast<double>(x[i]));
                dot += a * static_cast<double>(g[i]);
                w2 += wi(i) * wi(i);
            }
        const double c = w2 > 0 ? dot / w2 : 0.0;
        for (int64_t i = 0; i < n; ++i) {
            if (code[i]) {
                const double a = std::copysign(wi(i), static_cast<double>(x[i]));
                gx[i] = static_cast<T>(static_cast<double>(g[i]) - a * c);
            } else {
                gx[i] = T(0);
            }
        }
        acc[0] += c / scale;
    }

    // Kinks (since 0.3): a row on the sphere, sum_i w_i |x_i| within tol max w
    // of r, has code 2 at every entry (inside or projected); in a projected
    // row, code 1 at an entry with |x_i| within tol of tau w_i (tau read off
    // a support entry). Overrides: at a code-1 entry side +1 puts it on the
    // support, -1 off; for a row on the sphere the side at its first entry
    // picks the inside piece (-1) or the projection with tau = 0 (+1, the
    // nonzero entries on the support).
    double sphere_gap(const double *x, int64_t n) const noexcept {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += wi(i) * std::fabs(x[i]);
        return std::fabs(s - r / scale);
    }
    double threshold(const double *x, const double *z, int64_t n) const noexcept {
        for (int64_t i = 0; i < n; ++i)
            if (z[i] != 0.0 && wi(i) > 0.0) return (std::fabs(x[i]) - std::fabs(z[i])) / wi(i);
        return HUGE_VAL;
    }
    bool has_kinks() const noexcept override { return true; }
    bool has_branches() const noexcept override { return true; }
    int64_t kink_row(const double *x, const double *z, int64_t n, double, double tol, int8_t *mask,
                     int64_t ms, void *) const noexcept override {
        int64_t count = 0;
        const bool sphere = sphere_gap(x, n) <= tol;
        bool inside = true;
        for (int64_t i = 0; i < n && inside; ++i) inside = x[i] == z[i];
        const double tau = inside ? HUGE_VAL : threshold(x, z, n);
        for (int64_t i = 0; i < n; ++i) {
            const int8_t c = sphere ? 2 : (std::fabs(std::fabs(x[i]) - tau * wi(i)) <= tol ? 1 : 0);
            if (mask) mask[i * ms] = c;
            count += c != 0;
        }
        return count;
    }
    void branch_row(const double *x, const double *z, uint8_t *code, int64_t n, double, double tol,
                    const int8_t *side, void *) const noexcept override {
        if (sphere_gap(x, n) <= tol) {
            if (side[0] < 0) std::memset(code, kInside, static_cast<size_t>(n));
            if (side[0] > 0)
                for (int64_t i = 0; i < n; ++i) code[i] = x[i] != 0.0;
            return;
        }
        if (code[0] == kInside) return;
        const double tau = threshold(x, z, n);
        for (int64_t i = 0; i < n; ++i)
            if (side[i] != 0 && std::fabs(std::fabs(x[i]) - tau * wi(i)) <= tol) code[i] = side[i] > 0;
    }
};

// ---------------------------------------------------------------------------

// prox of gamma lambda ||.||_inf: z = x - P_{B1(gamma lambda)}(x), which clips
// x to [-tau, tau] when x is outside the ball and is 0 inside it.
struct LInf final : RowOp<LInf> {
    double lambda;
    Vec one{1.0};
    explicit LInf(double l) : lambda(l) {}
    const char *name() const noexcept override { return "linf"; }
    double step_scale() const noexcept override { return lambda; }
    bool conj_homogeneous() const noexcept override { return true; }
    int spectral() const noexcept override { return kSpectralBoth; }  // spectral norm
    int64_t nparams() const noexcept override { return 1; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }
    size_t scratch_bytes(int64_t n) const noexcept override { return condat_scratch(n); }

    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double, double *) const noexcept override {
        double m = 0.0;
        for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
        return lambda * m;
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *code) const noexcept {
        const double tau = l1_threshold(x, n, one, gamma * lambda, scr);
        if (tau == kL1Infinite) return kRowNonFinite;
        if (tau == kL1NaN) {
            fill_nan(z, n);
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            return kRowOk;
        }
        if (tau == kL1Inside) {
            std::memset(z, 0, static_cast<size_t>(n) * sizeof(T));
            if (code) std::memset(code, kInside, static_cast<size_t>(n));
            return kRowOk;
        }
        for (int64_t i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);
            const bool clip = std::fabs(xi) > tau;
            z[i] = clip ? static_cast<T>(std::copysign(tau, xi)) : x[i];
            if (code) code[i] = clip;
        }
        return kRowOk;
    }
    // Active: (J v)_i = v_i off S and s_i <s_S, v_S> / |S| on S;
    // dz_i/dgamma = -lambda s_i / |S| and dz_i/dlambda = -gamma s_i / |S| on S.
    template <class T>
    void jvp(const T *x, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n,
             double gamma, double dgamma, const double *dp, void *) const noexcept {
        if (code[0] == kInside) {
            for (int64_t i = 0; i < n; ++i) dz[i] = T(0);
            return;
        }
        double dot = 0.0;
        int64_t k = 0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                dot += std::copysign(1.0, static_cast<double>(x[i])) * static_cast<double>(dx[i]);
                ++k;
            }
        const double c =
            k ? (dot - lambda * dgamma - gamma * (dp ? dp[0] : 0.0)) / static_cast<double>(k) : 0.0;
        for (int64_t i = 0; i < n; ++i)
            dz[i] = code[i] ? static_cast<T>(static_cast<double>(x[i]) < 0 ? -c : c) : dx[i];
    }
    template <class T>
    void vjp(const T *x, const T *, const uint8_t *code, const T *g, T *gx, int64_t n,
             double gamma, double *acc, void *) const noexcept {
        if (code[0] == kInside) {
            for (int64_t i = 0; i < n; ++i) gx[i] = T(0);
            return;
        }
        double dot = 0.0;
        int64_t k = 0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                dot += (static_cast<double>(x[i]) < 0 ? -1.0 : 1.0) * static_cast<double>(g[i]);
                ++k;
            }
        const double m = k ? dot / static_cast<double>(k) : 0.0;
        for (int64_t i = 0; i < n; ++i)
            gx[i] = code[i] ? static_cast<T>(static_cast<double>(x[i]) < 0 ? -m : m) : g[i];
        acc[0] += -gamma * m;
        acc[1] += -lambda * m;
    }
};

// ---------------------------------------------------------------------------

// prox of gamma lambda max_i x_i. max is the support function of the unit
// simplex (Beck 2017, Example 6.49), so by the Moreau identity
// z = x - P_{c Delta}(x) with c = gamma lambda and c Delta the simplex of
// radius c: with tau its threshold, z = min(x, tau). The scaled form avoids
// Beck's x / c, which overflows for small c.
struct Max final : RowOp<Max> {
    double lambda;
    explicit Max(double l) : lambda(l) {}
    const char *name() const noexcept override { return "max"; }
    double step_scale() const noexcept override { return lambda; }
    bool conj_homogeneous() const noexcept override { return true; }
    int64_t nparams() const noexcept override { return 1; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }
    size_t scratch_bytes(int64_t n) const noexcept override { return condat_scratch(n); }

    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double, double *) const noexcept override {
        if (n == 0) return -lambda * std::numeric_limits<double>::infinity();  // the empty maximum
        double m = x[0];
        for (int64_t i = 1; i < n; ++i) m = std::max(m, x[i]);
        return lambda * m;
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *code) const noexcept {
        auto nonfinite = [&]() {
            bool nan;
            if (scan_row(x, n, nan) != kRowOk) return static_cast<int>(kRowNonFinite);
            if (nan) fill_nan(z, n);
            else std::memmove(z, x, static_cast<size_t>(n) * sizeof(T));  // z may be x
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            return static_cast<int>(kRowOk);
        };
        const double c = gamma * lambda;
        if (c == 0.0) return nonfinite();  // the identity unless the row has NaN or an infinity
        double *v = static_cast<double *>(scr);
        bool finite;
        const double tau = condat_tau_unit(
            n, [x](int64_t i) { return static_cast<double>(x[i]); }, c, v, v + n, &finite);
        if (!finite) return nonfinite();
        for (int64_t i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);
            const bool top = xi > tau;
            z[i] = top ? static_cast<T>(tau) : x[i];
            if (code) code[i] = top;
        }
        return kRowOk;
    }
    // On S = {x_i > tau}: z_i = tau with sum_S (x_i - tau) = c, so
    // dz_S = (sum_S dx - dc) / |S|; dz = dx off S.
    template <class T>
    void jvp(const T *, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n,
             double gamma, double dgamma, const double *dp, void *) const noexcept {
        double sum = 0.0;
        int64_t k = 0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                sum += static_cast<double>(dx[i]);
                ++k;
            }
        const double dc = lambda * dgamma + gamma * (dp ? dp[0] : 0.0);
        const double dt = k ? (sum - dc) / static_cast<double>(k) : 0.0;
        for (int64_t i = 0; i < n; ++i) dz[i] = code[i] ? static_cast<T>(dt) : dx[i];
    }
    template <class T>
    void vjp(const T *, const T *, const uint8_t *code, const T *g, T *gx, int64_t n,
             double gamma, double *acc, void *) const noexcept {
        double sum = 0.0;
        int64_t k = 0;
        for (int64_t i = 0; i < n; ++i)
            if (code[i]) {
                sum += static_cast<double>(g[i]);
                ++k;
            }
        const double m = k ? sum / static_cast<double>(k) : 0.0;
        for (int64_t i = 0; i < n; ++i) gx[i] = code[i] ? static_cast<T>(m) : g[i];
        acc[0] += -gamma * m;
        acc[1] += -lambda * m;
    }
};

}  // namespace

fp_status make_l2_ball(double radius, fp_op **out) {
    if (!finite_nonneg(radius)) return FP_EPARAM;
    *out = new L2Ball(radius);
    return FP_OK;
}

fp_status make_simplex(double radius, fp_op **out) {
    if (!finite_nonneg(radius)) return FP_EPARAM;
    *out = new Simplex(radius);
    return FP_OK;
}

fp_status make_l1_ball(double radius, Vec w, fp_op **out) {
    if (!finite_nonneg(radius)) return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    double wmax = 0.0, wmin = HUGE_VAL;
    for (double v : w) {
        if (!finite_nonneg(v) || v == 0.0) return FP_EPARAM;
        wmax = std::max(wmax, v);
        wmin = std::min(wmin, v);
    }
    // Squares of weight ratios below 1e-100 underflow; such weights are
    // rejected rather than silently mishandled.
    if (wmin / wmax < 1e-100) return FP_EPARAM;
    for (double &v : w) v /= wmax;
    *out = new L1Ball(radius, std::move(w), wmax);
    return FP_OK;
}

fp_status make_linf(double lambda, fp_op **out) {
    if (!finite_nonneg(lambda)) return FP_EPARAM;
    *out = new LInf(lambda);
    return FP_OK;
}

fp_status make_max(double lambda, fp_op **out) {
    if (!finite_nonneg(lambda)) return FP_EPARAM;
    *out = new Max(lambda);
    return FP_OK;
}

}  // namespace fp
