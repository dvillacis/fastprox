// Projection onto a hyperplane or half-space intersected with a box, and the
// prox of the sum of the k largest entries (or absolute values).
//
// Plane: P_C(x) for C = {z : a^T z = b (or <= b), lo <= z <= hi}, by A. Beck,
// First-Order Methods in Optimization, SIAM 2017, Theorem 6.27: P_C(x) =
// P_box(x - mu a) with mu a root of the nonincreasing piecewise linear
// phi(mu) = a^T P_box(x - mu a) - b. On the piece holding mu, the free set F
// gives phi's linear form and its root,
//   mu* = (sum_F a_i x_i + sum_lo a_i lo_i + sum_hi a_i hi_i - b) / sum_F a_i^2,
// which is exact once mu* lies on that piece. The solver (derived here)
// takes these Newton steps from mu = 0 inside a bracket (L, R) that each pass
// shrinks; it ends when a step returns mu exactly (the same states give the
// same sums: mu is on the root's piece). A step that leaves the bracket, or
// past 40 of them, is replaced by the median of the breakpoints in (L, R),
// where entries reach a bound, which halves them, so the search ends; with
// none left phi is linear on (L, R). Rows of 8192 entries or more start
// Newton (after mu = 0) from the root of the problem on every 16th entry, so
// the search takes 3 to 5 passes on the inputs of bench/; from 0 alone it
// creeps up long tails (15 passes for k = 1 of 10^6 normal entries).
// With "<=" the box projection is kept when it satisfies the constraint, and
// mu >= 0 otherwise (Beck's table of projections in Section 6.4). An infinite
// box gives the hyperplane and half-space projections of Lemma 6.26.
//
// Derivatives: F holds the entries strictly inside the box after the shift
// (an entry exactly on a bound counts as free, as for the box). From
// sum_i (da_i z_i + a_i dz_i) = db,
//   dmu = (sum_F a_i (dx_i - mu da_i) + sum_lo a_i dlo_i + sum_hi a_i dhi_i
//          + sum_i da_i z_i - db) / |a_F|^2,
//   dz_F = dx_F - dmu a_F - mu da_F, dz_i = dlo_i or dhi_i at a bound,
// so J = I_F - a_F a_F^T / |a_F|^2 on F and 0 off it. theta = (b, a..., lo...,
// hi...), where a, lo and hi are scalars or one entry per row element. C does
// not change when (a, b) is scaled, so the constructor divides both by
// s = max |a_i| (keeping mu and |a_F|^2 in range) and the derivatives in a
// and b carry a factor 1 / s.
//
// SumLargest: g(x) = sum of the k largest x_i is the support function of
// C = {e^T y = k, 0 <= y <= 1} (Beck, Example 6.50), and the sum of the k
// largest |x_i| that of {||y||_1 <= k, -1 <= y <= 1} (Example 6.51). By the
// Moreau identity z = x - P_{cC}(x) with c = gamma lambda: for c <= 1 on the
// scaled set (sum ck, cap c), so no x / c overflows, and for c > 1 as Beck's
// c P_C(x / c), so ck does not overflow. For the absolute values
// P_{cC}(x) = sign(x) P_+(|x|) with P_+ onto {e^T v <= ck, 0 <= v <= c}.
// Derivatives with u = S dx (S the signs, or I) and dc = lambda dgamma +
// gamma dlambda: dP_F = u_F - mean_F(u) - dc (|Hi| - k) / |F|, dP_Hi = dc,
// dP_Lo = 0 (dP_F = u_F when the constraint is inactive), dz = dx - S dP.
// theta = (lambda).
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace fp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Entry states in code: free, at the lower bound, at the upper bound. Bit
// kInactive on entry 0 marks a "<=" constraint met by the box projection.
constexpr uint8_t kFreeE = 0, kLoE = 1, kHiE = 2, kStateMask = 3, kInactive = 4;

// a, lo and hi each hold one value or one per entry.
// Scaled solves (audit K7) read the bounds times sc, a power of two: P_C is
// homogeneous, P_{tC}(t x) = t P_C(x) with tC = {a^T z = t b, t lo <= z <= t hi}.
struct Plane {
    const double *a, *lo, *hi;
    bool sa, sl, sh;  // scalar
    double b;
    bool unbounded = false;  // every bound infinite
    double bmax = 0.0;       // the largest finite |lo_i|, |hi_i|
    double sc = 1.0;         // the bounds' scale in a Scaled solve
    int ea = 1;              // the exponent of the smallest nonzero |a_i| (frexp; a = 1: 1)
    double A(int64_t i) const noexcept { return sa ? a[0] : a[i]; }
    double L(int64_t i) const noexcept { return sl ? lo[0] : lo[i]; }
    double H(int64_t i) const noexcept { return sh ? hi[0] : hi[i]; }
    template <bool S>
    double Ls(int64_t i) const noexcept { return S ? sc * L(i) : L(i); }
    template <bool S>
    double Hs(int64_t i) const noexcept { return S ? sc * H(i) : H(i); }
};

// The largest finite |v_i| (0 for none).
inline double finite_max(const Vec &v) noexcept {
    double m = 0.0;
    for (double x : v)
        if (std::isfinite(x)) m = std::max(m, std::fabs(x));
    return m;
}

inline double clip(double v, double l, double h) noexcept { return std::min(std::max(v, l), h); }

// The range [bot, top] of a^T z over the box (ends may be infinite).
template <bool S = false>
inline void plane_range(const Plane &p, int64_t n, double &bot, double &top) noexcept {
    bot = top = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double ai = p.A(i);
        if (ai == 0.0) continue;
        top += ai > 0 ? ai * p.Hs<S>(i) : ai * p.Ls<S>(i);
        bot += ai > 0 ? ai * p.Ls<S>(i) : ai * p.Hs<S>(i);
    }
}

// A point strictly inside (L, R).
inline double inside(double L, double R) noexcept {
    if (L == -kInf) return R == kInf ? 0.0 : R - 1.0 - std::fabs(R);
    if (R == kInf) return L + 1.0 + std::fabs(L);
    return 0.5 * L + 0.5 * R;
}

// mu with phi(mu) = b for a feasible plane (bp holds 2n doubles), from
// mu = 0 and then start if finite. With ineq, returns false when the box
// projection meets a^T z <= b (mu = 0).
template <bool S>
inline bool plane_solve(const Plane &p, bool ineq, const double *x, int64_t n, double *bp, double &mu,
                        double start = kInf) noexcept {
    constexpr int kNewtonMax = 40;
    const double b = p.b;
    double L = -kInf, R = kInf;
    bool linear = false;  // no breakpoint left in (L, R)
    mu = 0.0;
    if (p.unbounded) {  // Lemma 6.26: mu = (a^T x - b) / |a|^2
        double s1 = 0.0, s2 = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double ai = p.A(i);
            s1 += ai * x[i];
            s2 += ai * ai;
        }
        if (ineq && !(s1 > b)) return false;
        mu = (s1 - b) / s2;
        return true;
    }
    for (int pass = 0;; ++pass) {
        // phi on the piece holding mu: c + s1 - mu s2, with root mstar.
        // Branch-free (the states are unpredictable): a bound that is reached
        // is finite, so a_i times it is never 0 * inf, and a_i = 0 adds 0.
        double c = 0.0, s1 = 0.0, s2 = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double ai = p.A(i), xi = x[i], l = p.Ls<S>(i), h = p.Hs<S>(i);
            const double v = xi - mu * ai;
            const bool lo = v < l, hi = v > h, fr = !(lo || hi);
            c += fr ? 0.0 : ai * (lo ? l : h);
            s1 += fr ? ai * xi : 0.0;
            s2 += fr ? ai * ai : 0.0;
        }
        const double ph = c + s1 - mu * s2;
        if (pass == 0 && ineq && !(ph > b)) return false;
        const double mstar = s2 > 0.0 ? (c + s1 - b) / s2 : mu;
        // Done on the root's piece (the step returns mu) or when phi is
        // linear on the bracket.
        if (ph == b) return true;
        if (s2 > 0.0 && (linear || mstar == mu)) {
            mu = clip(mstar, L, R);
            return true;
        }
        if (linear) {  // phi constant on (L, R): b sits at an end
            mu = L != -kInf ? L : R;
            return true;
        }
        if (ph > b) L = mu;
        else R = mu;
        if (pass == 0 && start > L && start < R) {
            mu = start;
            continue;
        }
        if (pass < kNewtonMax && s2 > 0.0 && mstar > L && mstar < R) {
            mu = mstar;
            continue;
        }
        // The median of the breakpoints in (L, R), where entries reach a
        // bound, halves them; with none left phi is linear there.
        int64_t m = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double ai = p.A(i), l = p.Ls<S>(i), h = p.Hs<S>(i);
            if (ai == 0.0 || l == h) continue;  // l = h never changes state
            const double e = (x[i] - (ai > 0 ? h : l)) / ai, f = (x[i] - (ai > 0 ? l : h)) / ai;
            if (e > L && e < R) bp[m++] = e;
            if (f > L && f < R) bp[m++] = f;
        }
        if (m == 0) {
            linear = true;
            mu = inside(L, R);
        } else {
            std::nth_element(bp, bp + m / 2, bp + m);
            mu = bp[m / 2];
        }
    }
}

constexpr int64_t kSampleMin = 8192, kSampleStride = 16;

// A start for Newton on a long row: the root of the problem restricted to
// every 16th entry, with b scaled to the sample (solved the same way, so
// recursively), or inf when that problem is infeasible. A start near the
// root saves passes where Newton from 0 creeps up a long tail. buf holds 2n
// doubles.
template <bool S>
inline double plane_start(const Plane &p, const double *x, int64_t n, double *buf) noexcept {
    const int64_t m = n / kSampleStride;
    double *xs = buf, *as = xs + m, *ls = as + m, *hs = ls + m;
    for (int64_t j = 0; j < m; ++j) {
        const int64_t i = j * kSampleStride;
        xs[j] = x[i];
        if (!p.sa) as[j] = p.a[i];
        if (!p.sl) ls[j] = p.lo[i];
        if (!p.sh) hs[j] = p.hi[i];
    }
    Plane q{p.sa ? p.a : as, p.sl ? p.lo : ls, p.sh ? p.hi : hs, p.sa, p.sl, p.sh,
            p.b * static_cast<double>(m) / static_cast<double>(n), false, p.bmax, p.sc};  // raw bounds, p's scale
    double bot, top;
    plane_range<S>(q, m, bot, top);
    if (!(q.b > bot && q.b < top)) return kInf;
    double *rest = hs + m;  // 2n - 4m >= 2m doubles
    double mu = 0.0;
    if (m >= kSampleMin) plane_solve<S>(q, false, xs, m, rest, mu, plane_start<S>(q, xs, m, rest));
    else plane_solve<S>(q, false, xs, m, rest, mu);
    return std::isfinite(mu) ? mu : kInf;
}

template <bool S, class T>
inline void plane_project_at(const Plane &p, bool ineq, const double *x, T *z, int64_t n, double *bp,
                             uint8_t *code, int k) noexcept {
    double mu = 0.0;
    const double start = n >= kSampleMin && !p.unbounded ? plane_start<S>(p, x, n, bp) : kInf;
    const bool active = plane_solve<S>(p, ineq, x, n, bp, mu, start);
    for (int64_t i = 0; i < n; ++i) {
        const double v = x[i] - mu * p.A(i), l = p.Ls<S>(i), h = p.Hs<S>(i);
        z[i] = static_cast<T>(S ? std::ldexp(clip(v, l, h), k) : clip(v, l, h));  // back by 2^k
        if (code) code[i] = (v < l || (v == l && l == h)) ? kLoE : (v > h ? kHiE : kFreeE);
    }
    if (code && !active) code[0] |= kInactive;
}

// The exponent k of the scale 2^-k for a problem of magnitude M (max |x|, |b|
// and the finite bounds), or 0 when M is in [2^-960, 2^960], where no sum of
// n < 2^62 terms over- or underflows (audit K7), and the breakpoints
// (x_i - bound) / a_i, below M / min |a_i| = M 2^(1 - ea), are at most about
// 2^960 too (since 1.0.1: a_i = 6.6e-76 beside max |a_i| = 1 put them at
// 3e314 for |x| = 2e239, a NaN projection; fuzz). Scaled, M is near 1, or
// near 2^(ea + 960) where that is smaller (at least 2^-114). 2^k itself may
// overflow (k = 1024 near DBL_MAX), so results go back with ldexp.
inline int plane_scale(double M, int ea) noexcept {
    if (!(M > 0.0)) return 0;
    int k;
    std::frexp(M, &k);
    if (M >= 0x1p-960 && M <= 0x1p960 && k - ea <= 960) return 0;  // ea = 1: the rule before 1.0.1
    return k - std::min(0, ea + 960);
}

// z = P_C(x) with the states in code (if given); x in double, scratch that
// is rescaled in place when xmax = max |x_i| is far from unit scale.
template <class T>
inline void plane_project(const Plane &p, bool ineq, double *x, T *z, int64_t n, double *bp, uint8_t *code,
                          double xmax) noexcept {
    const int k = plane_scale(std::max({xmax, std::fabs(p.b), p.bmax}), p.ea);
    if (k == 0) {
        plane_project_at<false>(p, ineq, x, z, n, bp, code, 0);
        return;
    }
    for (int64_t i = 0; i < n; ++i) x[i] = std::ldexp(x[i], -k);  // exact but for subnormal results
    Plane ps = p;
    ps.b = std::ldexp(p.b, -k);
    ps.sc = std::ldexp(1.0, -k);
    plane_project_at<true>(ps, ineq, x, z, n, bp, code, k);
}

// Copies a row into double; kRowNonFinite on an infinity, nan set on NaN.
template <class T>
inline int load_row(const T *x, double *v, int64_t n, bool &nan, double &vmax) noexcept {
    nan = false;
    vmax = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double xi = static_cast<double>(x[i]);
        if (std::isinf(xi)) return kRowNonFinite;
        nan = nan || std::isnan(xi);
        vmax = std::max(vmax, std::fabs(xi));
        v[i] = xi;
    }
    return kRowOk;
}

// ---------------------------------------------------------------------------

struct HyperBox final : RowOp<HyperBox> {
    static constexpr int kStruct = FP_STRUCT_PROJECTOR;
    Vec a, lo, hi;  // a and b divided by s
    double b, inv_s;
    bool ineq;
    HyperBox(Vec aa, double bb, Vec l, Vec h, bool in, double s)
        : a(std::move(aa)), lo(std::move(l)), hi(std::move(h)), b(bb), inv_s(1.0 / s), ineq(in) {
        unbounded = std::all_of(lo.begin(), lo.end(), [](double v) { return v == -kInf; }) &&
                    std::all_of(hi.begin(), hi.end(), [](double v) { return v == kInf; });
        bmax = std::max(finite_max(lo), finite_max(hi));
        double amin = 1.0;
        for (double v : a)
            if (v != 0.0) amin = std::min(amin, std::fabs(v));
        std::frexp(amin, &ea);
    }

    bool unbounded;
    double bmax;
    int ea;
    Plane plane() const noexcept {
        return Plane{a.data(), lo.data(), hi.data(), a.size() == 1, lo.size() == 1, hi.size() == 1, b, unbounded,
                     bmax, 1.0, ea};
    }
    const char *name() const noexcept override { return ineq ? "halfspace" : "hyperplane"; }
    double step_scale() const noexcept override { return 0.0; }
    int64_t nparams() const noexcept override {
        return 1 + static_cast<int64_t>(a.size() + lo.size() + hi.size());
    }
    // Shapes, then C nonempty: b within the range of a^T z over the box (or
    // above its lower end for "<=").
    fp_status check_n(int64_t n) const noexcept override {
        for (const Vec *v : {&a, &lo, &hi})
            if (v->size() != 1 && static_cast<int64_t>(v->size()) != n) return FP_ESHAPE;
        double bot, top;
        plane_range(plane(), n, bot, top);
        if (!(b >= bot) || (!ineq && !(b <= top))) return FP_EPARAM;
        return FP_OK;
    }
    // The row in double and the breakpoints.
    size_t scratch_bytes(int64_t n) const noexcept override { return 3 * static_cast<size_t>(n) * sizeof(double); }

    bool has_value() const noexcept override { return true; }
    // The indicator of a^T x = b (or <=) and lo <= x <= hi; the plane's
    // violation in its normalised form (max |a_i| = 1).
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        const Plane p = plane();
        double s = 0.0, v = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            s += p.A(i) * x[i];
            v = std::max(v, std::max(p.L(i) - x[i], x[i] - p.H(i)));
        }
        v = std::max(v, ineq ? s - p.b : std::fabs(s - p.b));
        return indicator(v, value_atol(x, n, tol));
    }

    template <class T>
    int prox(const T *xin, T *z, int64_t n, double, void *scr, uint8_t *code) const noexcept {
        double *x = static_cast<double *>(scr);
        bool nan;
        double xmax;
        if (load_row(xin, x, n, nan, xmax) != kRowOk) return kRowNonFinite;
        if (nan) {
            fill_nan(z, n);
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            return kRowOk;
        }
        plane_project(plane(), ineq, x, z, n, x + n, code, xmax);
        return kRowOk;
    }

    // mu and |a_F|^2 from the saved states (the forward formula).
    template <class T>
    void saved_mu(const T *x, const uint8_t *code, int64_t n, double &mu, double &den) const noexcept {
        const Plane p = plane();
        mu = den = 0.0;
        if (code[0] & kInactive) return;
        // The sums in units scaled by t, as in plane_project (t = 1: exact).
        double xmax = 0.0;
        for (int64_t i = 0; i < n; ++i) xmax = std::max(xmax, std::fabs(static_cast<double>(x[i])));
        const int k = plane_scale(std::max({xmax, std::fabs(b), bmax}), ea);
        double num = -std::ldexp(b, -k);
        for (int64_t i = 0; i < n; ++i) {
            const double ai = p.A(i);
            if (ai == 0.0) continue;
            const uint8_t st = code[i] & kStateMask;
            if (st == kLoE) num += ai * std::ldexp(p.L(i), -k);
            else if (st == kHiE) num += ai * std::ldexp(p.H(i), -k);
            else {
                num += ai * std::ldexp(static_cast<double>(x[i]), -k);
                den += ai * ai;
            }
        }
        if (den > 0.0) mu = std::ldexp(num / den, k);
    }

    // Structure (since 1.0.2): LOWER and UPPER at the bounds (lo = hi:
    // LOWER), FREE inside, the saved states; the constraint a_F, unless the
    // half-space's box projection met it or no free entry has a_i != 0
    // (|a_F|^2 = 0, where the jvp is the identity on F).
    template <class T>
    int64_t structure(const T *, const T *, const uint8_t *code, int64_t n, double, RowStruct &s,
                      void *) const noexcept {
        const Plane p = plane();
        double den = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t st = code[i] & kStateMask;
            s.state[i] = st == kLoE ? FP_ENTRY_LOWER : (st == kHiE ? FP_ENTRY_UPPER : FP_ENTRY_FREE);
            if (st == kFreeE) den += p.A(i) * p.A(i);
        }
        const int64_t k = singletons(s, n);
        if (!s.constraint || (code[0] & kInactive) || !(den > 0.0)) return k;
        if (s.coef)
            for (int64_t i = 0; i < n; ++i) s.coef[i] = s.state[i] == FP_ENTRY_FREE ? p.A(i) : 0.0;
        s.active = true;
        return k;
    }

    template <class T>
    void jvp(const T *x, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n, double,
             double, const double *dp, void *) const noexcept {
        const Plane p = plane();
        const size_t na = a.size(), nl = lo.size(), nh = hi.size();
        auto da = [&](int64_t i) { return dp ? inv_s * dp[1 + (na == 1 ? 0 : i)] : 0.0; };
        auto dl = [&](int64_t i) { return dp ? dp[1 + na + (nl == 1 ? 0 : i)] : 0.0; };
        auto dh = [&](int64_t i) { return dp ? dp[1 + na + nl + (nh == 1 ? 0 : i)] : 0.0; };
        double mu, den;
        saved_mu(x, code, n, mu, den);
        double dmu = 0.0;
        if (den > 0.0) {
            double num = -(dp ? inv_s * dp[0] : 0.0);
            for (int64_t i = 0; i < n; ++i) {
                const double ai = p.A(i);
                const uint8_t st = code[i] & kStateMask;
                if (st == kFreeE) num += ai * (static_cast<double>(dx[i]) - mu * da(i));
                else num += ai * (st == kLoE ? dl(i) : dh(i));
                num += da(i) * static_cast<double>(z[i]);
            }
            dmu = num / den;
        }
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t st = code[i] & kStateMask;
            if (st == kLoE) dz[i] = static_cast<T>(dl(i));
            else if (st == kHiE) dz[i] = static_cast<T>(dh(i));
            else dz[i] = static_cast<T>(static_cast<double>(dx[i]) - dmu * p.A(i) - mu * da(i));
        }
    }

    template <class T>
    void vjp(const T *x, const T *z, const uint8_t *code, const T *g, T *gx, int64_t n, double,
             double *acc, void *) const noexcept {
        const Plane p = plane();
        const size_t na = a.size(), nl = lo.size(), nh = hi.size();
        double mu, den;
        saved_mu(x, code, n, mu, den);
        double s = 0.0;  // <a_F, g_F> / |a_F|^2
        if (den > 0.0) {
            double ag = 0.0;
            for (int64_t i = 0; i < n; ++i)
                if ((code[i] & kStateMask) == kFreeE) ag += p.A(i) * static_cast<double>(g[i]);
            s = ag / den;
        }
        acc[0] += inv_s * s;
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t st = code[i] & kStateMask;
            const double r = static_cast<double>(g[i]) - s * p.A(i);
            acc[1 + (na == 1 ? 0 : i)] += inv_s * ((st == kFreeE ? -mu * r : 0.0) - s * static_cast<double>(z[i]));
            if (st == kLoE) acc[1 + na + (nl == 1 ? 0 : i)] += r;
            else if (st == kHiE) acc[1 + na + nl + (nh == 1 ? 0 : i)] += r;
            gx[i] = static_cast<T>(st == kFreeE ? r : 0.0);
        }
    }
};

// ---------------------------------------------------------------------------

struct SumLargest final : RowOp<SumLargest> {
    static constexpr int kStruct = FP_STRUCT_PROJECTOR;
    double lambda;
    int64_t k;
    bool absolute;
    SumLargest(double l, int64_t kk, bool abs) : lambda(l), k(kk), absolute(abs) {}

    const char *name() const noexcept override { return absolute ? "sum_largest_abs" : "sum_largest"; }
    double step_scale() const noexcept override { return lambda; }
    bool conj_homogeneous() const noexcept override { return true; }
    int64_t nparams() const noexcept override { return 1; }
    // The sum of the k largest entries needs k <= n; for the absolute values
    // k >= n is the l1 norm.
    fp_status check_n(int64_t n) const noexcept override { return absolute || k <= n ? FP_OK : FP_EPARAM; }
    // The row (|x| for the absolute values), the projection, the breakpoints.
    size_t scratch_bytes(int64_t n) const noexcept override { return 4 * static_cast<size_t>(n) * sizeof(double); }

    bool has_value() const noexcept override { return true; }
    // lambda times the sum of the k largest x_i (|x_i| when absolute); scr: n.
    double value_row(const double *x, int64_t n, double, double *scr) const noexcept override {
        for (int64_t i = 0; i < n; ++i) scr[i] = absolute ? std::fabs(x[i]) : x[i];
        const int64_t m = std::min(k, n);
        std::nth_element(scr, scr + (n - m), scr + n);
        double s = 0.0;
        for (int64_t i = n - m; i < n; ++i) s += scr[i];
        return lambda * s;
    }

    template <class T>
    int prox(const T *xin, T *z, int64_t n, double gamma, void *scr, uint8_t *code) const noexcept {
        double *x = static_cast<double *>(scr);
        bool nan;
        double xmax;
        if (load_row(xin, x, n, nan, xmax) != kRowOk) return kRowNonFinite;
        const double c = gamma * lambda;
        if (nan || c == 0.0) {
            if (nan) fill_nan(z, n);
            else std::memmove(z, xin, static_cast<size_t>(n) * sizeof(T));
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            return kRowOk;
        }
        if (absolute)
            for (int64_t i = 0; i < n; ++i) x[i] = std::fabs(x[i]);
        const bool big = c > 1.0;  // project x / c onto C, else x onto cC
        if (big) {
            for (int64_t i = 0; i < n; ++i) x[i] /= c;
            xmax /= c;  // an upper bound is enough for the scale
        }
        const double one = 1.0, zero = 0.0, cap = big ? 1.0 : c;
        const Plane p{&one, &zero, &cap, true, true, true, cap * static_cast<double>(k), false, cap};
        double *proj = x + n;
        plane_project(p, absolute, x, proj, n, x + 2 * n, code, xmax);
        for (int64_t i = 0; i < n; ++i) {
            const double xi = static_cast<double>(xin[i]), pi = big ? c * proj[i] : proj[i];
            z[i] = static_cast<T>(xi - (absolute ? std::copysign(pi, xi) : pi));
        }
        return kRowOk;
    }

    struct Free {
        int64_t nf = 0, nh = 0;
        bool active = true;
    };
    Free count(const uint8_t *code, int64_t n) const noexcept {
        Free f;
        f.active = !(code[0] & kInactive);
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t st = code[i] & kStateMask;
            f.nf += st == kFreeE;
            f.nh += st == kHiE;
        }
        return f;
    }
    template <class T>
    static double sgn(const T *x, int64_t i, bool absolute) noexcept {
        return absolute && static_cast<double>(x[i]) < 0.0 ? -1.0 : 1.0;
    }

    // Structure (since 1.0.2) of z = x - S P: the entries below the shared
    // level and at the cap move with x (singletons); the free set F of the
    // projection, at the shared level, is one region with signs S_i when the
    // constraint is active, and ZERO when the absolute values' "<=" is met
    // there (dz_F = 0). gamma lambda = 0: the identity.
    template <class T>
    int64_t structure(const T *x, const T *, const uint8_t *code, int64_t n, double gamma, RowStruct &s,
                      void *) const noexcept {
        const bool id = gamma * lambda == 0.0, active = id || !(code[0] & kInactive);
        for (int64_t i = 0; i < n; ++i)
            s.state[i] = !id && !active && (code[i] & kStateMask) == kFreeE ? FP_ENTRY_ZERO : FP_ENTRY_FREE;
        return one_region(s, n, [=](int64_t i) { return !id && (code[i] & kStateMask) == kFreeE; },
                          [=](int64_t i) { return sgn(x, i, absolute); });
    }

    template <class T>
    void jvp(const T *x, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dp, void *) const noexcept {
        if (gamma * lambda == 0.0) {
            for (int64_t i = 0; i < n; ++i) dz[i] = dx[i];
            return;
        }
        const double dc = lambda * dgamma + gamma * (dp ? dp[0] : 0.0);
        const Free f = count(code, n);
        double shift = 0.0;  // mean_F(u) + dc (|Hi| - k) / |F|
        if (f.active && f.nf > 0) {
            double su = 0.0;
            for (int64_t i = 0; i < n; ++i)
                if ((code[i] & kStateMask) == kFreeE) su += sgn(x, i, absolute) * static_cast<double>(dx[i]);
            shift = (su + dc * static_cast<double>(f.nh - k)) / static_cast<double>(f.nf);
        }
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t st = code[i] & kStateMask;
            const double si = sgn(x, i, absolute), dxi = static_cast<double>(dx[i]);
            const double dP = st == kFreeE ? si * dxi - shift : (st == kHiE ? dc : 0.0);
            dz[i] = static_cast<T>(dxi - si * dP);
        }
    }

    template <class T>
    void vjp(const T *x, const T *, const uint8_t *code, const T *g, T *gx, int64_t n, double gamma,
             double *acc, void *) const noexcept {
        if (gamma * lambda == 0.0) {
            for (int64_t i = 0; i < n; ++i) gx[i] = g[i];
            return;
        }
        const Free f = count(code, n);
        // With h = S g: <h, dP> = sum_F (h_i - hbar) u_i + dc (sum_Hi h_i -
        // (|Hi| - k) hbar), hbar = mean_F(h) when the constraint is active.
        double hf = 0.0, hh = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t st = code[i] & kStateMask;
            const double hi = sgn(x, i, absolute) * static_cast<double>(g[i]);
            if (st == kFreeE) hf += hi;
            else if (st == kHiE) hh += hi;
        }
        const double hbar = f.active && f.nf > 0 ? hf / static_cast<double>(f.nf) : 0.0;
        for (int64_t i = 0; i < n; ++i)
            gx[i] = (code[i] & kStateMask) == kFreeE ? static_cast<T>(sgn(x, i, absolute) * hbar) : g[i];
        const double gc = static_cast<double>(f.nh - k) * hbar - hh;  // d<g, z>/dc
        acc[0] += gamma * gc;
        acc[1] += lambda * gc;
    }
};

}  // namespace

fp_status make_hyperplane(Vec a, double b, Vec lo, Vec hi, bool inequality, fp_op **out) {
    if (a.empty()) a.assign(1, 1.0);
    if (lo.empty()) lo.assign(1, -kInf);
    if (hi.empty()) hi.assign(1, kInf);
    if (!std::isfinite(b)) return FP_EPARAM;
    double amax = 0.0;
    for (double v : a) {
        if (!std::isfinite(v)) return FP_EPARAM;
        amax = std::max(amax, std::fabs(v));
    }
    if (amax == 0.0) return FP_EPARAM;
    // Nonzero ratios below 1e-100 would underflow in |a_F|^2 (as for the
    // weighted l1 ball); such a is rejected.
    for (double &v : a) {
        if (v != 0.0 && std::fabs(v) / amax < 1e-100) return FP_EPARAM;
        v /= amax;
    }
    b /= amax;
    for (double v : lo)
        if (std::isnan(v) || v == kInf) return FP_EPARAM;
    for (double v : hi)
        if (std::isnan(v) || v == -kInf) return FP_EPARAM;
    if (lo.size() > 1 && hi.size() > 1 && lo.size() != hi.size()) return FP_ESHAPE;
    for (size_t i = 0; i < std::max(lo.size(), hi.size()); ++i)
        if (lo[lo.size() == 1 ? 0 : i] > hi[hi.size() == 1 ? 0 : i]) return FP_EPARAM;
    *out = new HyperBox(std::move(a), b, std::move(lo), std::move(hi), inequality, amax);
    return FP_OK;
}

fp_status make_sum_largest(double lambda, int64_t k, bool absolute, fp_op **out) {
    if (!finite_nonneg(lambda) || k < 1) return FP_EPARAM;
    *out = new SumLargest(lambda, k, absolute);
    return FP_OK;
}

}  // namespace fp
