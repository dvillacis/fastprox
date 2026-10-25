// Nonconvex pointwise maps: the prox of gamma sum_i psi(z_i) for
//   l0      psi(t) = lambda w_i [t != 0]               (hard thresholding)
//   lq      psi(t) = lambda w_i |t|^q, 0 < q < 1
//   SCAD    J. Fan and R. Li, "Variable selection via nonconcave penalized
//           likelihood and its oracle properties", JASA 96(456) (2001) 1348-1360
//   MCP     C.-H. Zhang, "Nearly unbiased variable selection under minimax
//           concave penalty", Ann. Statist. 38(2) (2010) 894-942
//   log     psi(t) = lambda log(1 + |t| / eps)  (E. J. Candes, M. B. Wakin and
//           S. P. Boyd, J. Fourier Anal. Appl. 14 (2008) 877-905)
// Every map acts on |x| and keeps the sign; below, x >= 0 and lam = gamma
// lambda (w_i). The forms below were derived here for a general step gamma
// and are checked against minimisation on a refined grid in the tests.
//
// l0. z = x for x > sqrt(2 lam), 0 below; at x = sqrt(2 lam) both minimise.
//
// lq (G. Marjanovic and V. Solo, IEEE Trans. Signal Process. 60(11) (2012)
// 5714-5724; the form of F. Wen, L. Chu, P. Liu and R. Qiu, IEEE Access 6
// (2018), Table I). The nonzero branch is the root y in [beta, x] of
//   h(y) = y + lam q y^{q-1} - x,   beta = (2 lam (1 - q))^{1/(2-q)},
// and it beats 0 exactly for x > tau = beta + lam q beta^{q-1}; at x = tau both
// 0 and beta minimise. h is increasing and convex on [beta, inf), so Newton
// from any point above the root (x, or the tangent bound in root) decreases
// monotonically to it. For q = 1/2 and 2/3 the root is
// in closed form (Z. Xu, X. Chang, F. Xu and H. Zhang, IEEE Trans. Neural
// Netw. Learn. Syst. 23(7) (2012) 1013-1027; W. Cao, J. Sun and Z. Xu,
// J. Vis. Commun. Image Represent. 24(1) (2013) 31-41), rederived here:
//   q = 1/2: y = s^2 with s the largest root of s^3 - x s + lam/2 = 0,
//            s = 2 sqrt(x/3) cos(acos(-(3 lam / (4 x)) sqrt(3/x)) / 3);
//   q = 2/3: y = s^3 with s the largest root of s^4 - x s + c = 0, c = 2 lam/3,
//            by Ferrari: m the largest root of m^3 - c m - x^2/8 = 0, then
//            s = (sqrt(2m) + sqrt(2x / sqrt(2m) - 2m)) / 2;
// each followed by one Newton step.
//
// SCAD (a > 1 + gamma, otherwise the prox is not single-valued: FP_EPARAM):
//   z = max(x - lam, 0)                              for x <= (1 + gamma) lambda,
//   z = ((a - 1) x - gamma a lambda) / (a - 1 - gamma)   up to a lambda,
//   z = x                                            beyond;
// with gamma = 1 this is Fan and Li's rule. MCP (b > gamma): z = 0 for
// x <= lam, b (x - lam) / (b - gamma) up to b lambda, x beyond. In both the
// objective is convex when the condition holds, so the map is continuous.
// The middle pieces are evaluated as bounded ratios, SCAD's as
// x + gamma (x / D - lambda a / D) with D = a - 1 - gamma and
// a / D = 1 / (1 - (1 + gamma) / a), MCP's as (x - lam) / (1 - gamma / b):
// the products with a or b overflowed at a = b = 1e308, where both answers
// round to 2 at x = 3, gamma = lambda = 1 (math audit M6).
//
// log. The nonzero candidate solves z - x + lam / (eps + z) = 0,
//   r = (x - eps + sqrt((x + eps)^2 - 4 lam)) / 2,
// and is kept when it beats 0: F(r) - F(0) = r (r/2 - x) + lam log1p(r/eps)
// < 0 (for lam <= eps^2 the objective is convex and this is the soft rule).
//
// Ties (l0, lq, log): the nonzero branch by default, 0 with FP_TIE_ZERO; the
// entries on a tie are counted. Derivatives (spec Section 7) are those of the
// selected branch: 0 off the support, else from the scalar equation
// z + gamma lambda psi'(z) = x, i.e. 1 / (1 + gamma lambda psi''(z)).
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace fp {
namespace {

template <class Op, class... A>
fp_status emit(fp_op **out, A &&...a) {
    *out = new Op(std::forward<A>(a)...);
    return FP_OK;
}

inline double sgn(double x) noexcept { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

// sqrt(2 a b c) without the underflow of a b c where the root is normal: an
// l0 threshold sqrt(2 gamma lambda w) of 1e-160 has gamma lambda w = 1e-320.
inline double sqrt2prod3(double a, double b, double c) noexcept {
    const double p = prod3(a, b, c);
    if (p >= std::numeric_limits<double>::min() || (p == 0.0 && (a == 0.0 || b == 0.0 || c == 0.0)))
        return std::sqrt(2.0 * p);
    int ea, eb, ec;
    const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb), mc = std::frexp(c, &ec);
    int e = ea + eb + ec;  // 2 a b c = (2 m) 2^e, m = ma mb mc in [1/8, 1)
    double m = 2.0 * ma * mb * mc;
    if (e & 1) {
        m *= 2.0;
        --e;
    }
    return std::ldexp(std::sqrt(m), e / 2);
}

// of 0; 0 if none.
inline int nearest_kink(double x, const double *t, int m, double tol, Kink &k) noexcept {
    const double ax = std::fabs(x);
    int code = 0;
    double best = tol;
    for (int j = 0; j < m; ++j) {
        const double dist = std::fabs(ax - t[j]);
        if (dist <= best && (code == 0 || dist < best)) {
            code = j + 1;
            best = dist;
            k.b = std::copysign(t[j], x);
            k.scale = t[j];
        }
    }
    return code;
}

// ---- l0 -------------------------------------------------------------------------------

struct L0 final : ElemOp<L0> {
    static constexpr bool kTies = true;
    double lambda;
    Vec w;
    double wmax_ = vmax(w);  // its largest entry, for step_scale: once, not per call (it scanned every weight on each prox call)
    L0(double l, Vec ww) : lambda(l), w(std::move(ww)) {}
    const char *name() const noexcept override { return "l0"; }
    double step_scale() const noexcept override { return lambda * wmax_; }
    int spectral() const noexcept override { return uniform(w) ? kSpectralBoth : 0; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    bool axis_params() const noexcept override { return w.size() > 1; }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    int64_t wi(int64_t i) const noexcept { return w.size() == 1 ? 0 : i; }
    double thr(double gamma, int64_t i) const noexcept { return sqrt2prod3(gamma, lambda, w[wi(i)]); }

    double value1(double x, int64_t i, double) const noexcept { return x != 0.0 ? lambda * w[wi(i)] : 0.0; }

    template <class T>
    T prox1(T x, double gamma, int64_t i, int tie, bool &tied) const noexcept {
        const double a = std::fabs(static_cast<double>(x)), t = thr(gamma, i);
        tied = a == t && t > 0.0;
        if (a > t || (tied && tie == FP_TIE_NONZERO)) return x;
        return std::isnan(a) ? x : static_cast<T>(0);
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i, int tie) const noexcept {
        ElemDeriv e;
        const double a = std::fabs(x), t = thr(gamma, i);
        e.j = (!(t > 0.0) || a > t || (a == t && tie == FP_TIE_NONZERO)) ? 1.0 : 0.0;
        return e;  // piecewise constant in gamma and w
    }
    // Pieces: 0 zero, 1 the identity (everywhere when t = 0); the kink (since
    // 0.3) is the tie |x| = t > 0.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double gamma, int64_t i) const noexcept {
        const double t = thr(gamma, i);
        return {!(t > 0.0) || std::fabs(x) > t ? 1 : 0, sgn(x)};
    }
    ElemDeriv deriv_piece(double, double, int64_t, Piece pc) const noexcept {
        ElemDeriv e;
        e.j = pc.id;
        return e;
    }
    int kink1(double x, double gamma, int64_t i, double tol, Kink &k) const noexcept {
        const double t[1] = {thr(gamma, i)};
        return t[0] > 0.0 ? nearest_kink(x, t, 1, tol, k) : 0;
    }
};

// ---- lq -------------------------------------------------------------------------------

struct Lq final : ElemOp<Lq> {
    static constexpr bool kTies = true;
    double lambda, q;
    Vec w;
    double wmax_ = vmax(w);  // its largest entry, for step_scale: once, not per call (it scanned every weight on each prox call)
    int form;  // 1: q = 1/2, 2: q = 2/3 (closed forms and cheap powers), 0: general
    double inv_slope;  // 1 / h'(beta) = 1 / (1 - q/2), for root's start
    Lq(double l, double qq, Vec ww)
        : lambda(l), q(qq), w(std::move(ww)), form(qq == 0.5 ? 1 : (qq == 2.0 / 3.0 ? 2 : 0)),
          inv_slope(1.0 / (1.0 - 0.5 * qq)) {}
    const char *name() const noexcept override { return "lq"; }
    double step_scale() const noexcept override { return lambda * wmax_; }
    int spectral() const noexcept override { return uniform(w) ? kSpectralBoth : 0; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    bool axis_params() const noexcept override { return w.size() > 1; }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    int64_t wi(int64_t i) const noexcept { return w.size() == 1 ? 0 : i; }

    // y^{q-1}, cheaply for the closed-form cases.
    double pw(double y) const noexcept {
        if (form == 1) return 1.0 / std::sqrt(y);
        if (form == 2) return 1.0 / std::cbrt(y);
        return std::pow(y, q - 1.0);
    }
    double h(double y, double lam, double x) const noexcept { return y + lam * q * pw(y) - x; }
    double dh(double y, double lam) const noexcept { return 1.0 + lam * q * (q - 1.0) * pw(y) / y; }

    // beta and tau depend on (q, lam) only: cached per thread on those values
    // (not on the operator's address, which a new operator may reuse), so a
    // scalar weight computes them once per call.
    void thresholds(double lam, double &beta, double &tau) const noexcept {
        thread_local double c_q = -1.0, c_lam = -1.0, c_beta = 0.0, c_tau = 0.0;
        if (c_q != q || c_lam != lam) {
            c_q = q;
            c_lam = lam;
            c_beta = std::pow(2.0 * lam * (1.0 - q), 1.0 / (2.0 - q));
            c_tau = c_beta + lam * q * pw(c_beta);
        }
        beta = c_beta;
        tau = c_tau;
    }

    // The root of h on [beta, x] for x > tau.
    double root(double x, double lam, double beta, double tau) const noexcept {
        double y;
        int newton = 1;  // polish of the closed forms
        if (form == 1) {
            const double arg = -(3.0 * lam / (4.0 * x)) * std::sqrt(3.0 / x);
            const double s = 2.0 * std::sqrt(x / 3.0) * std::cos(std::acos(std::fmax(-1.0, std::fmin(1.0, arg))) / 3.0);
            y = s * s;
        } else if (form == 2) {
            const double c = 2.0 * lam / 3.0;
            // m^3 - c m - x^2/8 = 0, largest real root.
            const double P = -c, Q = -x * x / 8.0;
            const double disc = Q * Q / 4.0 + P * P * P / 27.0;
            double m;
            if (disc >= 0.0) {
                const double sq = std::sqrt(disc);
                m = std::cbrt(-Q / 2.0 + sq) + std::cbrt(-Q / 2.0 - sq);
            } else {
                const double r = 2.0 * std::sqrt(-P / 3.0);
                const double arg = (3.0 * Q / (2.0 * P)) * std::sqrt(-3.0 / P);
                m = r * std::cos(std::acos(std::fmax(-1.0, std::fmin(1.0, arg))) / 3.0);
            }
            const double r2m = std::sqrt(2.0 * m);
            const double s = 0.5 * (r2m + std::sqrt(std::fmax(0.0, 2.0 * x / r2m - 2.0 * m)));
            y = s * s * s;
        } else {
            newton = 100;
        }
        if (newton == 100 || !(y >= beta) || !(y <= x)) {  // the general q, or the closed forms' safeguard
            // h lies above its tangent at beta, where h' = 1 - q/2 for every
            // lam, so the tangent's zero bounds the root from above. Just
            // above tau it is close to the root, where starting from x took
            // up to 9 iterations; from about 2 tau on it exceeds x.
            // Both starts are valid; the test on x alone lets the far entries
            // start without waiting for the bound.
            y = x < 2.0 * tau ? std::fmin(x, beta + (x - tau) * inv_slope) : x;
            newton = 100;
        }
        auto step = [&](double v) {
            double vn = v - h(v, lam, x) / dh(v, lam);
            if (!(vn >= beta)) vn = 0.5 * (v + beta);  // keep to the branch
            return vn > x ? x : vn;
        };
        if (newton == 1) return step(y);  // the closed forms' polish may step either way
        // From above the root the iterates decrease to it. Stop when the step
        // is below 4e-16 y, or when it does not decrease y: that is the
        // rounding floor of h, which near tau, where y is far below x,
        // exceeds 4e-16 y (the relative test alone let the iterates oscillate
        // up to the cap).
        for (int it = 0; it < newton; ++it) {
            const double yn = step(y), d = y - yn;
            if (!(d > 4e-16 * y)) {
                // A bound below the root by rounding: the step goes up, past
                // the root (h is convex), and the descent follows.
                if (it == 0 && d < -4e-16 * y) {
                    y = yn;
                    continue;
                }
                if (d > 0.0) y = yn;
                break;
            }
            y = yn;
        }
        return y;
    }

    // |z| for |x| = a, the branch chosen and the tie flag.
    double mag(double a, double lam, int tie, bool &tied, bool &on) const noexcept {
        tied = false;
        on = false;
        if (lam == 0.0) {
            on = a > 0.0;
            return a;
        }
        double beta, tau;
        thresholds(lam, beta, tau);
        if (a < tau) return 0.0;
        if (a == tau) {
            tied = true;
            on = tie == FP_TIE_NONZERO;
            return on ? beta : 0.0;
        }
        on = true;
        return root(a, lam, beta, tau);
    }

    double value1(double x, int64_t i, double) const noexcept {
        return lambda * w[wi(i)] * std::pow(std::fabs(x), q);
    }

    template <class T>
    T prox1(T x, double gamma, int64_t i, int tie, bool &tied) const noexcept {
        const double xd = static_cast<double>(x);
        if (std::isnan(xd)) {
            tied = false;
            return x;
        }
        bool on;
        const double m = mag(std::fabs(xd), prod3(gamma, lambda, w[wi(i)]), tie, tied, on);
        return static_cast<T>(std::copysign(m, xd));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i, int tie) const noexcept {
        ElemDeriv e;
        const double lam = prod3(gamma, lambda, w[wi(i)]);
        bool tied, on;
        const double m = mag(std::fabs(x), lam, tie, tied, on);
        if (lam == 0.0 && m == 0.0) {  // the identity
            e.j = 1.0;
            return e;
        }
        if (!on || m == 0.0) return e;
        return branch_deriv(m, sgn(x), gamma, i);
    }
    // The derivative on the nonzero branch at |z| = m, sign s.
    ElemDeriv branch_deriv(double m, double s, double gamma, int64_t i) const noexcept {
        ElemDeriv e;
        const double lam = prod3(gamma, lambda, w[wi(i)]), mq = pw(m);
        const double den = 1.0 + lam * q * (q - 1.0) * mq / m;
        e.j = 1.0 / den;
        e.dg = -prod3(lambda, w[wi(i)], q * mq) * s / den;
        e.np = 1;
        e.pi[0] = wi(i);
        e.pv[0] = -prod3(gamma, lambda, q * mq) * s / den;
        return e;
    }
    // Pieces: 0 zero, 1 the nonzero branch, taken at max(|x|, tau) (at tau
    // its root is beta); the kink (since 0.3) is the tie |x| = tau.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double gamma, int64_t i) const noexcept {
        const double lam = prod3(gamma, lambda, w[wi(i)]);
        if (lam == 0.0) return {1, sgn(x)};
        double beta, tau;
        thresholds(lam, beta, tau);
        return {std::fabs(x) > tau ? 1 : 0, sgn(x)};
    }
    ElemDeriv deriv_piece(double x, double gamma, int64_t i, Piece pc) const noexcept {
        ElemDeriv e;
        const double lam = prod3(gamma, lambda, w[wi(i)]);
        if (pc.id == 0) return e;
        if (lam == 0.0) {
            e.j = 1.0;
            return e;
        }
        double beta, tau;
        thresholds(lam, beta, tau);
        const double a = std::fabs(x);
        return branch_deriv(a > tau ? root(a, lam, beta, tau) : beta, pc.s, gamma, i);
    }
    int kink1(double x, double gamma, int64_t i, double tol, Kink &k) const noexcept {
        const double lam = prod3(gamma, lambda, w[wi(i)]);
        if (!(lam > 0.0)) return 0;
        double beta, tau;
        thresholds(lam, beta, tau);
        const double t[1] = {tau};
        return nearest_kink(x, t, 1, tol, k);
    }
};

// ---- SCAD and MCP ---------------------------------------------------------------------
//
// Weighted: g(z) = sum_i w_i phi(z_i). The prox of gamma w_i phi is the
// unweighted map at the step c = gamma w_i (both forms above depend on gamma
// only through c), so the condition is a > 1 + gamma max w (SCAD) or
// b > gamma max w (MCP), and the derivatives follow from those in c:
// dz/dgamma = w_i dz/dc, dz/dw_i = gamma dz/dc. For MCP this is the identity
// w MCP_{lam, b} = MCP_{w lam, b / w}. theta = (lambda, a or b) without
// weights, as before they existed (and as the group penalties wrap it), and
// (lambda, a or b, w...) with them.

// The breakpoint +-t[j] (j + 1 the code) nearest to x within tol, on x's side

// Shared by both: lambda, the concavity, the weights and their maximum.
// Weighted is a template parameter, so the unweighted maps load no weight per
// entry and compile as before weights existed.
template <bool Weighted>
struct WeightedPair {
    double lambda, p;
    Vec w;
    double wmax = 1.0;
    WeightedPair(double l, double pp, Vec ww) : lambda(l), p(pp), w(std::move(ww)) {
        if (Weighted) wmax = 0.0;
        for (double v : w) wmax = std::fmax(wmax, v);
    }
    int64_t nw() const noexcept { return Weighted ? static_cast<int64_t>(w.size()) : 0; }
    int64_t wi(int64_t i) const noexcept { return w.size() == 1 ? 0 : i; }
    double step(double gamma, int64_t i) const noexcept {
        if constexpr (Weighted) return gamma * w[wi(i)];
        else return gamma;
    }
    fp_status check_len(int64_t n) const noexcept { return (nw() <= 1 || nw() == n) ? FP_OK : FP_ESHAPE; }
    // dz/dgamma and dz/dw_i from dz/dc.
    void chain(ElemDeriv &e, double dzdc, double gamma, int64_t i) const noexcept {
        e.np = 2;
        e.pi[0] = 0;
        e.pi[1] = 1;
        if constexpr (Weighted) {
            e.dg = w[wi(i)] * dzdc;
            e.np = 3;
            e.pi[2] = 2 + wi(i);
            e.pv[2] = gamma * dzdc;
        } else {
            e.dg = dzdc;
        }
    }
};

template <bool Weighted>
struct Scad final : ElemOp<Scad<Weighted>>, WeightedPair<Weighted> {
    using P = WeightedPair<Weighted>;
    using P::chain, P::check_len, P::lambda, P::nw, P::p, P::step, P::wmax;
    Scad(double l, double aa, Vec ww) : P(l, aa, std::move(ww)) {}
    const char *name() const noexcept override { return "scad"; }
    double step_scale() const noexcept override { return lambda * wmax; }
    int spectral() const noexcept override { return uniform(this->w) ? kSpectralBoth : 0; }
    int64_t nparams() const noexcept override { return 2 + nw(); }  // (lambda, a, w...)
    bool axis_params() const noexcept override { return nw() > 1; }
    fp_status check_n(int64_t n) const noexcept override { return check_len(n); }
    fp_status check_step(double gamma) const noexcept override {
        return p > 1.0 + gamma * wmax ? FP_OK : FP_EPARAM;
    }

    // w_i SCAD(x): lambda |t| up to lambda, the quadratic piece up to a lambda,
    // then (a + 1) lambda^2 / 2 (Fan and Li 2001).
    double value1(double x, int64_t i, double) const noexcept {
        const double l = this->lambda, a = this->p, t = std::fabs(x);
        const double v = t <= l ? l * t : (t <= a * l ? (2.0 * a * l * t - t * t - l * l) / (2.0 * (a - 1.0))
                                                      : 0.5 * (a + 1.0) * l * l);
        return this->step(1.0, i) * v;
    }

    template <class T>
    T prox1(T x, double gamma, int64_t i) const noexcept {
        const double xd = static_cast<double>(x), ax = std::fabs(xd), c = step(gamma, i), a = p;
        // Branches vectorise the unweighted loop, and every branch evaluated
        // and selected the weighted one (Apple clang 21, arm64: 0.5 and
        // 1.2 ns per entry; the other way round, 1.3 and 3.7).
        double m;
        if constexpr (Weighted) {
            const double m1 = std::fmax(ax - c * lambda, 0.0);
            const double m2 = linear(ax, c, a);
            m = ax <= a * lambda ? m2 : ax;
            m = ax <= (1.0 + c) * lambda ? m1 : m;
        } else {
            if (ax <= (1.0 + c) * lambda) m = std::fmax(ax - c * lambda, 0.0);
            else if (ax <= a * lambda) m = linear(ax, c, a);
            else m = ax;
        }
        if (c == 0.0) m = ax;  // zero weight or step: the identity, exactly
        return std::isnan(xd) ? x : static_cast<T>(std::copysign(m, xd));
    }
    // The linear piece ((a - 1) x - c a lambda) / (a - 1 - c) as x + c (x / D
    // - lambda r), D = a - 1 - c, r = a / D, so that no product with a
    // overflows (M6); bitwise it can differ from the quotient by an ulp. It is
    // at most x on its piece (x <= a lambda), and taken so: rounding put it an
    // ulp above (since 1.0.1, as for MCP).
    static double ratio(double c, double a) noexcept { return 1.0 / (1.0 - (1.0 + c) / a); }
    double linear(double ax, double c, double a) const noexcept {
        return std::fmin(ax + c * (ax / (a - 1.0 - c) - lambda * ratio(c, a)), ax);
    }
    // Pieces (in |x|, with the sign s): 0 thresholded, 1 soft, 2 linear, 3
    // identity. Kinks: 1 at |x| = c lambda, 2 at (1 + c) lambda, 3 at a lambda.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double gamma, int64_t i) const noexcept {
        const double ax = std::fabs(x), c = step(gamma, i), a = p;
        int id;
        if (ax <= c * lambda && c * lambda > 0.0) id = 0;  // c lambda = 0 is the identity
        else if (ax <= (1.0 + c) * lambda) id = 1;
        else if (ax <= a * lambda) id = 2;
        else id = 3;
        return {id, sgn(x)};
    }
    // The piece's formula at x; ax = s x is |x| unless x lies across a kink.
    ElemDeriv deriv_piece(double x, double gamma, int64_t i, Piece pc) const noexcept {
        ElemDeriv e;
        const double s = pc.s, ax = s * x, c = step(gamma, i), a = p;
        double dzdc = 0.0;
        if (pc.id == 0) {
            // thresholded (the soft rule's kink choice)
        } else if (pc.id == 1) {
            e.j = 1.0;
            dzdc = -s * lambda;
            e.pv[0] = -s * c;
        } else if (pc.id == 2) {
            // With D = a - 1 - c and r = a / D: z = s (x + c (x / D - lambda r)),
            // j = (a - 1) / D, dz/dc = z / D - s lambda r, dz/dlambda = -s c r,
            // dz/da = -s (c / D) (x - (1 + c) lambda) / D (M6).
            const double D = a - 1.0 - c, r = ratio(c, a);
            const double z = s * linear(ax, c, a);
            e.j = 1.0 / (1.0 - c / (a - 1.0));
            dzdc = z / D - s * lambda * r;
            e.pv[0] = -s * c * r;
            e.pv[1] = -s * (c / D) * ((ax - (1.0 + c) * lambda) / D);
        } else {
            e.j = 1.0;
        }
        chain(e, dzdc, gamma, i);
        return e;
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i) const noexcept {
        return deriv_piece(x, gamma, i, piece_of(x, gamma, i));
    }
    int kink1(double x, double gamma, int64_t i, double tol, Kink &k) const noexcept {
        const double c = step(gamma, i);
        if (!(c > 0.0) || !(lambda > 0.0)) return 0;  // the identity
        const double t[3] = {c * lambda, (1.0 + c) * lambda, p * lambda};
        return nearest_kink(x, t, 3, tol, k);
    }
};

template <bool Weighted>
struct Mcp final : ElemOp<Mcp<Weighted>>, WeightedPair<Weighted> {
    using P = WeightedPair<Weighted>;
    using P::chain, P::check_len, P::lambda, P::nw, P::p, P::step, P::wmax;
    Mcp(double l, double bb, Vec ww) : P(l, bb, std::move(ww)) {}
    const char *name() const noexcept override { return "mcp"; }
    double step_scale() const noexcept override { return lambda * wmax; }
    int spectral() const noexcept override { return uniform(this->w) ? kSpectralBoth : 0; }
    int64_t nparams() const noexcept override { return 2 + nw(); }  // (lambda, b, w...)
    bool axis_params() const noexcept override { return nw() > 1; }
    fp_status check_n(int64_t n) const noexcept override { return check_len(n); }
    fp_status check_step(double gamma) const noexcept override { return p > gamma * wmax ? FP_OK : FP_EPARAM; }

    // w_i MCP(x): lambda |t| - t^2 / (2 b) up to b lambda, then b lambda^2 / 2
    // (Zhang 2010).
    double value1(double x, int64_t i, double) const noexcept {
        const double l = this->lambda, b = this->p, t = std::fabs(x);
        const double v = t <= b * l ? l * t - t * t / (2.0 * b) : 0.5 * b * l * l;
        return this->step(1.0, i) * v;
    }

    template <class T>
    T prox1(T x, double gamma, int64_t i) const noexcept {
        const double xd = static_cast<double>(x), ax = std::fabs(xd), c = step(gamma, i), b = p;
        double m;  // as in Scad::prox1
        // The linear piece is at most x where it applies (x <= b lambda), and
        // taken so: x = -2.98828125 at gamma = 1.296875 came back an ulp
        // above |x| (fuzz, 2026-10-25), moving an entry away from 0.
        if constexpr (Weighted) {
            const double m2 = std::fmin((ax - c * lambda) / (1.0 - c / b), ax);
            m = ax <= b * lambda ? m2 : ax;
            m = ax <= c * lambda ? 0.0 : m;
        } else {
            // b (x - c lambda) / (b - c), without the product b x (M6).
            if (ax <= c * lambda) m = 0.0;
            else if (ax <= b * lambda) m = std::fmin((ax - c * lambda) / (1.0 - c / b), ax);
            else m = ax;
        }
        if (c == 0.0) m = ax;  // zero weight or step: the identity, exactly (b x / b is not)
        return std::isnan(xd) ? x : static_cast<T>(std::copysign(m, xd));
    }
    // Pieces (in |x|, with the sign s): 0 thresholded, 1 linear, 2 identity.
    // Kinks: 1 at |x| = c lambda, 2 at b lambda.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double gamma, int64_t i) const noexcept {
        const double ax = std::fabs(x), c = step(gamma, i), b = p;
        int id;
        // lambda = 0 is the identity; c = 0 (lambda > 0) is the linear piece
        // with j = 1, which keeps its one-sided dz/dc.
        if (!(lambda > 0.0)) id = 2;
        else if (ax <= c * lambda && c > 0.0) id = 0;
        else if (ax <= b * lambda) id = 1;
        else id = 2;
        return {id, sgn(x)};
    }
    ElemDeriv deriv_piece(double x, double gamma, int64_t i, Piece pc) const noexcept {
        ElemDeriv e;
        const double s = pc.s, ax = s * x, c = step(gamma, i), b = p;
        double dzdc = 0.0;
        if (pc.id == 0) {
            // thresholded
        } else if (pc.id == 1) {
            // With D = b - c: z = s (x - c lambda) j, j = b / D = 1 / (1 - c / b),
            // dz/dc = z / D - s lambda j, dz/dlambda = -s c j, dz/db = -s (c / D)
            // (x - c lambda) / D: no product with b (M6).
            const double D = b - c, u = ax - c * lambda;
            e.j = 1.0 / (1.0 - c / b);
            const double z = s * u * e.j;
            dzdc = z / D - s * lambda * e.j;
            e.pv[0] = -s * c * e.j;
            e.pv[1] = -s * (c / D) * (u / D);
        } else {
            e.j = 1.0;
        }
        chain(e, dzdc, gamma, i);
        return e;
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i) const noexcept {
        return deriv_piece(x, gamma, i, piece_of(x, gamma, i));
    }
    int kink1(double x, double gamma, int64_t i, double tol, Kink &k) const noexcept {
        const double c = step(gamma, i);
        if (!(c > 0.0) || !(lambda > 0.0)) return 0;  // the identity
        const double t[2] = {c * lambda, p * lambda};
        return nearest_kink(x, t, 2, tol, k);
    }
};

// ---- log ------------------------------------------------------------------------------

struct LogPen final : ElemOp<LogPen> {
    static constexpr bool kTies = true;
    double lambda, eps;
    bool eps_mid;  // eps in [2^-500, 2^499]: log_root's usual path needs only a <= 2^499
    LogPen(double l, double e) : lambda(l), eps(e), eps_mid(e >= 0x1p-500 && e <= 0x1p499) {}
    const char *name() const noexcept override { return "log"; }
    double step_scale() const noexcept override { return lambda; }  // the kernel forms gamma lambda only
    int spectral() const noexcept override { return kSpectralBoth; }
    int64_t nparams() const noexcept override { return 2; }  // (lambda, eps)
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    // The positive root r of z^2 + (eps - a) z + (lam - a eps) = 0, the
    // stationarity condition z - a + lam / (eps + z) = 0 (0 when there is none
    // or it is not positive; a, lam >= 0), and df = F(r) - F(0) = r (r/2 - a) +
    // lam log(1 + r/eps), of which only the sign is read (audit K3). Far from
    // unit scale both run on (a, eps, r) 2^-k and lam 2^-2k, which leaves them
    // unchanged but for a positive factor: (a + eps)^2 overflowed beyond 1e154
    // and r (r/2 - a) underflowed into a spurious tie. For a < eps the root is
    // 2 (a eps - lam) / ((eps - a) + sqrt(D)), the same root without the
    // cancellation of (a - eps) + sqrt(D); fma is correctly rounded. The root
    // is below a for lam > 0, and taken as at most a: rounding put it an ulp
    // above at lam << a eps (fuzz, 2026-10-25), and the prox moved an entry
    // away from 0.
    struct Root {
        double r, df;
    };
    FP_INLINE Root log_root(double a, double lam) const noexcept {
        // a + eps in [2^-500, 2^500]; lam <= 2^960 by the step range (fp_prox).
        if (!(eps_mid && a <= 0x1p499)) return log_root_scaled(a, lam);
        const double S = a + eps;
        // root_at at t = 1, written out.
        const double D = S * S - 4.0 * lam;
        if (!(D >= 0.0)) return {0.0, 0.0};
        const double q = std::sqrt(D);
        double r;
        if (a >= eps) {
            r = std::fmin(0.5 * ((a - eps) + q), a);
        } else {
            const double num = std::fma(a, eps, -lam);
            if (!(num > 0.0)) return {0.0, 0.0};
            r = std::fmin(2.0 * num / ((eps - a) + q), a);
        }
        if (!(r > 0.0)) return {0.0, 0.0};
        return {r, r * (0.5 * r - a) + lam * std::log1p(r / eps)};
    }
    // Out of line, so that the usual path stays small and inlined.
    FP_COLD Root log_root_scaled(double a, double lam) const noexcept {
        int k;
        std::frexp(std::max(a, eps), &k);
        // lam 2^-2k may be +inf: no root then, as 4 lam > S^2.
        Root rt = root_at(std::ldexp(a, -k), std::ldexp(eps, -k), std::ldexp(lam, -2 * k), std::ldexp(1.0, -k));
        rt.r = std::fmin(rt.r, a);  // a 2^-k can be subnormal, and the root then an ulp above a
        return rt;
    }
    // The root and df in units scaled by t (a power of two; 1 unscaled).
    static Root root_at(double ah, double eh, double lh, double t) noexcept {
        const double Sh = ah + eh, D = Sh * Sh - 4.0 * lh;
        if (!(D >= 0.0)) return {0.0, 0.0};
        const double q = std::sqrt(D);
        double rh;
        if (ah >= eh) {
            rh = std::fmin(0.5 * ((ah - eh) + q), ah);
        } else {
            const double num = std::fma(ah, eh, -lh);  // a eps - lam: a positive root needs > 0
            if (!(num > 0.0)) return {0.0, 0.0};      // (and skips the division below the breakpoint)
            rh = std::fmin(2.0 * num / ((eh - ah) + q), ah);
        }
        if (!(rh > 0.0)) return {0.0, 0.0};
        return {t == 1.0 ? rh : rh / t, rh * (0.5 * rh - ah) + lh * std::log1p(rh / eh)};
    }

    // Per entry; inlined, or tied and on go through memory (40% here).
    FP_INLINE double mag(double ax, double lam, int tie, bool &tied, bool &on) const noexcept {
        tied = false;
        on = false;
        if (lam == 0.0) {
            on = ax > 0.0;
            return ax;
        }
        const Root rt = log_root(ax, lam);
        const double r = rt.r, df = rt.df;  // df: F(r) - F(0), up to a positive factor
        if (!(r > 0.0)) return 0.0;
        if (df < 0.0) {
            on = true;
            return r;
        }
        if (df == 0.0) {
            tied = true;
            on = tie == FP_TIE_NONZERO;
            return on ? r : 0.0;
        }
        return 0.0;
    }

    double value1(double x, int64_t, double) const noexcept { return lambda * std::log1p(std::fabs(x) / eps); }

    template <class T>
    FP_INLINE T prox1(T x, double gamma, int64_t, int tie, bool &tied) const noexcept {
        const double xd = static_cast<double>(x);
        if (std::isnan(xd)) {
            tied = false;
            return x;
        }
        bool on;
        return static_cast<T>(std::copysign(mag(std::fabs(xd), gamma * lambda, tie, tied, on), xd));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t, int tie) const noexcept {
        ElemDeriv e;
        e.np = 2;
        e.pi[0] = 0;
        e.pi[1] = 1;
        const double lam = gamma * lambda;
        // lam = 0 is the identity, also at x = 0, where s = 0 zeroes the
        // parameter terms.
        if (lam == 0.0) return branch_deriv(std::fabs(x), sgn(x), gamma);
        bool tied, on;
        const double r = mag(std::fabs(x), lam, tie, tied, on);
        if (!on || r == 0.0) return e;
        return branch_deriv(r, sgn(x), gamma);
    }
    // The derivative on the nonzero branch at z = s r.
    ElemDeriv branch_deriv(double r, double s, double gamma) const noexcept {
        ElemDeriv e;
        e.np = 2;
        e.pi[0] = 0;
        e.pi[1] = 1;
        // kappa = lam / u^2 as (lam / u) / u: u^2 over- or underflows where
        // kappa (<= 1 on this branch) does not.
        const double lam = gamma * lambda, u = eps + r, kappa = lam / u / u;
        const double den = 1.0 - kappa;
        e.j = 1.0 / den;
        e.dg = -lambda * s / u / den;
        e.pv[0] = -gamma * s / u / den;
        e.pv[1] = kappa * s / den;
        return e;
    }
    // The root r(a) of the stationarity condition on the nonzero branch, a >=
    // 2 sqrt(lam) - eps (0 below that).
    double branch(double a, double lam) const noexcept { return log_root(a, lam).r; }
    // The breakpoint in |x| (since 0.3): lam / eps where the map is
    // continuous (lam <= eps^2, the branch starts at 0), else the tie, where
    // F(r(a)) - F(0) = r (r / 2 - a) + lam log(1 + r / eps), decreasing in a
    // (its derivative is -r), crosses 0; by bisection, cached per thread.
    double breakpoint(double lam) const noexcept {
        if (lam / eps <= eps) return lam / eps;  // lam <= eps^2, without its underflow
        thread_local double c_lam = -1.0, c_eps = -1.0, c_t = 0.0;
        if (c_lam == lam && c_eps == eps) return c_t;
        auto df = [&](double a) { return log_root(a, lam).df; };
        double lo = 2.0 * std::sqrt(lam) - eps, hi = std::fmax(2.0 * lo, lo + eps);
        for (int it = 0; it < 200 && df(hi) > 0.0; ++it) hi *= 2.0;
        for (int it = 0; it < 200 && hi - lo > 2.220446049250313e-16 * hi; ++it) {
            const double mid = 0.5 * (lo + hi);
            (df(mid) > 0.0 ? lo : hi) = mid;
        }
        c_lam = lam;
        c_eps = eps;
        c_t = hi;
        return hi;
    }
    // Pieces: 0 zero, 1 the nonzero branch at |x| (at max(|x|, the
    // breakpoint) for a tie); the kink is the breakpoint.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double gamma, int64_t) const noexcept {
        const double lam = gamma * lambda;
        if (lam == 0.0) return {1, sgn(x)};
        return {std::fabs(x) > breakpoint(lam) ? 1 : 0, sgn(x)};
    }
    ElemDeriv deriv_piece(double x, double gamma, int64_t, Piece pc) const noexcept {
        const double lam = gamma * lambda;
        if (pc.id == 0) {
            ElemDeriv e;
            e.np = 2;
            e.pi[0] = 0;
            e.pi[1] = 1;
            return e;
        }
        if (lam == 0.0) return branch_deriv(std::fabs(x), pc.s, gamma);  // as deriv1
        const double a = std::fmax(std::fabs(x), lam / eps > eps ? breakpoint(lam) : 0.0);
        return branch_deriv(branch(a, lam), pc.s, gamma);
    }
    int kink1(double x, double gamma, int64_t, double tol, Kink &k) const noexcept {
        const double lam = gamma * lambda;
        if (!(lam > 0.0)) return 0;
        const double t[1] = {breakpoint(lam)};
        return nearest_kink(x, t, 1, tol, k);
    }
};

}  // namespace

fp_status make_l0(double lambda, Vec w, fp_op **out) {
    if (!finite_nonneg(lambda)) return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    return emit<L0>(out, lambda, std::move(w));
}

fp_status make_lq(double lambda, double q, Vec w, fp_op **out) {
    if (!finite_nonneg(lambda) || !(q > 0.0 && q < 1.0)) return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    return emit<Lq>(out, lambda, q, std::move(w));
}

fp_status make_scad(double lambda, double a, Vec w, fp_op **out) {
    if (!finite_nonneg(lambda) || !(a > 1.0) || !std::isfinite(a)) return FP_EPARAM;
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    if (w.empty()) return emit<Scad<false>>(out, lambda, a, Vec{1.0});
    return emit<Scad<true>>(out, lambda, a, std::move(w));
}

fp_status make_mcp(double lambda, double b, Vec w, fp_op **out) {
    if (!finite_nonneg(lambda) || !(b > 0.0) || !std::isfinite(b)) return FP_EPARAM;
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    if (w.empty()) return emit<Mcp<false>>(out, lambda, b, Vec{1.0});
    return emit<Mcp<true>>(out, lambda, b, std::move(w));
}

fp_status make_log(double lambda, double eps, fp_op **out) {
    if (!finite_nonneg(lambda) || !(eps > 0.0) || !std::isfinite(eps)) return FP_EPARAM;
    return emit<LogPen>(out, lambda, eps);
}

}  // namespace fp
