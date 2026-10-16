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
// from x decreases monotonically to the root. For q = 1/2 and 2/3 the root is
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

// ---- l0 -------------------------------------------------------------------------------

struct L0 final : ElemOp<L0> {
    static constexpr bool kTies = true;
    double lambda;
    Vec w;
    L0(double l, Vec ww) : lambda(l), w(std::move(ww)) {}
    const char *name() const noexcept override { return "l0"; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    bool axis_params() const noexcept override { return w.size() > 1; }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    int64_t wi(int64_t i) const noexcept { return w.size() == 1 ? 0 : i; }
    double thr(double gamma, int64_t i) const noexcept { return std::sqrt(2.0 * gamma * lambda * w[wi(i)]); }

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
        e.j = (a > t || (a == t && t > 0.0 && tie == FP_TIE_NONZERO)) ? 1.0 : 0.0;
        return e;  // piecewise constant in gamma and w
    }
};

// ---- lq -------------------------------------------------------------------------------

struct Lq final : ElemOp<Lq> {
    static constexpr bool kTies = true;
    double lambda, q;
    Vec w;
    int form;  // 1: q = 1/2, 2: q = 2/3 (closed forms and cheap powers), 0: general
    Lq(double l, double qq, Vec ww)
        : lambda(l), q(qq), w(std::move(ww)), form(qq == 0.5 ? 1 : (qq == 2.0 / 3.0 ? 2 : 0)) {}
    const char *name() const noexcept override { return "lq"; }
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
    double root(double x, double lam, double beta) const noexcept {
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
            y = x;
            newton = 100;
        }
        if (!(y >= beta) || !(y <= x)) {  // safeguard the closed forms
            y = x;
            newton = 100;
        }
        for (int it = 0; it < newton; ++it) {
            const double step = h(y, lam, x) / dh(y, lam);
            double yn = y - step;
            if (!(yn >= beta)) yn = 0.5 * (y + beta);  // keep to the branch
            if (yn > x) yn = x;
            const bool done = std::fabs(yn - y) <= 4e-16 * y;
            y = yn;
            if (done) break;
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
        return root(a, lam, beta);
    }

    template <class T>
    T prox1(T x, double gamma, int64_t i, int tie, bool &tied) const noexcept {
        const double xd = static_cast<double>(x);
        if (std::isnan(xd)) {
            tied = false;
            return x;
        }
        bool on;
        const double m = mag(std::fabs(xd), gamma * lambda * w[wi(i)], tie, tied, on);
        return static_cast<T>(std::copysign(m, xd));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i, int tie) const noexcept {
        ElemDeriv e;
        const double lw = lambda * w[wi(i)], lam = gamma * lw;
        bool tied, on;
        const double m = mag(std::fabs(x), lam, tie, tied, on);
        if (lam == 0.0 && m == 0.0) {  // the identity
            e.j = 1.0;
            return e;
        }
        if (!on || m == 0.0) return e;
        const double s = sgn(x), mq = pw(m);
        const double den = 1.0 + lam * q * (q - 1.0) * mq / m;
        e.j = 1.0 / den;
        e.dg = -lw * q * mq * s / den;
        e.np = 1;
        e.pi[0] = wi(i);
        e.pv[0] = -gamma * lambda * q * mq * s / den;
        return e;
    }
};

// ---- SCAD -----------------------------------------------------------------------------

struct Scad final : ElemOp<Scad> {
    double lambda, a;
    Scad(double l, double aa) : lambda(l), a(aa) {}
    const char *name() const noexcept override { return "scad"; }
    int64_t nparams() const noexcept override { return 2; }  // (lambda, a)
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }
    fp_status check_step(double gamma) const noexcept override { return a > 1.0 + gamma ? FP_OK : FP_EPARAM; }

    template <class T>
    T prox1(T x, double gamma, int64_t) const noexcept {
        const double xd = static_cast<double>(x), ax = std::fabs(xd);
        double m;
        if (ax <= (1.0 + gamma) * lambda) m = std::fmax(ax - gamma * lambda, 0.0);
        else if (ax <= a * lambda) m = ((a - 1.0) * ax - gamma * a * lambda) / (a - 1.0 - gamma);
        else m = ax;
        return std::isnan(xd) ? x : static_cast<T>(std::copysign(m, xd));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t) const noexcept {
        ElemDeriv e;
        const double ax = std::fabs(x), s = sgn(x);
        e.np = 2;
        e.pi[0] = 0;
        e.pi[1] = 1;
        if (ax <= gamma * lambda) return e;  // thresholded (the soft rule's kink choice)
        if (ax <= (1.0 + gamma) * lambda) {
            e.j = 1.0;
            e.dg = -s * lambda;
            e.pv[0] = -s * gamma;
        } else if (ax <= a * lambda) {
            const double D = a - 1.0 - gamma;
            const double z = s * ((a - 1.0) * ax - gamma * a * lambda) / D;
            e.j = (a - 1.0) / D;
            e.dg = (z - s * a * lambda) / D;
            e.pv[0] = -s * gamma * a / D;
            e.pv[1] = (s * (ax - gamma * lambda) - z) / D;
        } else {
            e.j = 1.0;
        }
        return e;
    }
};

// ---- MCP ------------------------------------------------------------------------------

struct Mcp final : ElemOp<Mcp> {
    double lambda, b;
    Mcp(double l, double bb) : lambda(l), b(bb) {}
    const char *name() const noexcept override { return "mcp"; }
    int64_t nparams() const noexcept override { return 2; }  // (lambda, b)
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }
    fp_status check_step(double gamma) const noexcept override { return b > gamma ? FP_OK : FP_EPARAM; }

    template <class T>
    T prox1(T x, double gamma, int64_t) const noexcept {
        const double xd = static_cast<double>(x), ax = std::fabs(xd);
        double m;
        if (ax <= gamma * lambda) m = 0.0;
        else if (ax <= b * lambda) m = b * (ax - gamma * lambda) / (b - gamma);
        else m = ax;
        return std::isnan(xd) ? x : static_cast<T>(std::copysign(m, xd));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t) const noexcept {
        ElemDeriv e;
        const double ax = std::fabs(x), s = sgn(x);
        e.np = 2;
        e.pi[0] = 0;
        e.pi[1] = 1;
        if (ax <= gamma * lambda) return e;
        if (ax <= b * lambda) {
            const double D = b - gamma;
            const double z = s * b * (ax - gamma * lambda) / D;
            e.j = b / D;
            e.dg = (z - s * b * lambda) / D;
            e.pv[0] = -s * gamma * b / D;
            e.pv[1] = -s * gamma * (ax - gamma * lambda) / (D * D);
        } else {
            e.j = 1.0;
        }
        return e;
    }
};

// ---- log ------------------------------------------------------------------------------

struct LogPen final : ElemOp<LogPen> {
    static constexpr bool kTies = true;
    double lambda, eps;
    LogPen(double l, double e) : lambda(l), eps(e) {}
    const char *name() const noexcept override { return "log"; }
    int64_t nparams() const noexcept override { return 2; }  // (lambda, eps)
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    double mag(double ax, double lam, int tie, bool &tied, bool &on) const noexcept {
        tied = false;
        on = false;
        if (lam == 0.0) {
            on = ax > 0.0;
            return ax;
        }
        const double disc = (ax + eps) * (ax + eps) - 4.0 * lam;
        if (!(disc >= 0.0)) return 0.0;
        const double r = 0.5 * ((ax - eps) + std::sqrt(disc));
        if (!(r > 0.0)) return 0.0;
        const double df = r * (0.5 * r - ax) + lam * std::log1p(r / eps);  // F(r) - F(0)
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

    template <class T>
    T prox1(T x, double gamma, int64_t, int tie, bool &tied) const noexcept {
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
        bool tied, on;
        const double r = mag(std::fabs(x), lam, tie, tied, on);
        if (!on || r == 0.0) return e;
        const double s = sgn(x), u = eps + r;
        const double den = 1.0 - lam / (u * u);
        e.j = 1.0 / den;
        e.dg = -lambda * s / u / den;
        e.pv[0] = -gamma * s / u / den;
        e.pv[1] = lam * s / (u * u) / den;
        return e;
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

fp_status make_scad(double lambda, double a, fp_op **out) {
    if (!finite_nonneg(lambda) || !(a > 1.0) || !std::isfinite(a)) return FP_EPARAM;
    return emit<Scad>(out, lambda, a);
}

fp_status make_mcp(double lambda, double b, fp_op **out) {
    if (!finite_nonneg(lambda) || !(b > 0.0) || !std::isfinite(b)) return FP_EPARAM;
    return emit<Mcp>(out, lambda, b);
}

fp_status make_log(double lambda, double eps, fp_op **out) {
    if (!finite_nonneg(lambda) || !(eps > 0.0) || !std::isfinite(eps)) return FP_EPARAM;
    return emit<LogPen>(out, lambda, eps);
}

}  // namespace fp
