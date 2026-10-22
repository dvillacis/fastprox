// Separable convex maps: soft thresholding, elastic net, Huber, box,
// nonnegative orthant, the log barrier and l1 over a box. All closed form;
// derivatives are diagonal.
//
// Log barrier, g(z) = -lambda sum_i log z_i (A. Beck, First-Order Methods in
// Optimization, SIAM 2017, Lemma 6.5, p. 133): z = (x + r) / 2 with
// r = sqrt(x^2 + 4 gamma lambda), evaluated as 2 gamma lambda / (r - x) for
// x < 0, where the sum cancels. dz/dx = z / r, dz/dgamma = lambda / r.
//
// l1 over a box, g(z) = lambda sum_i w_i |z_i| + indicator(lo <= z <= hi)
// (Beck, Example 6.23, pp. 145-146, for the symmetric box -alpha <= z <= alpha):
// z = clamp(soft(x, gamma lambda w), lo, hi). The same holds for any interval,
// since a strictly convex function of one variable is minimised over an
// interval at the projection of its unconstrained minimiser.
//
// Kink convention (spec Section 7): an entry exactly at a threshold or bound
// takes the Jacobian of the branch that contains the boundary, i.e. soft
// thresholding treats |x| = tau as thresholded, and the box treats
// x in [lo, hi] as interior. A zero threshold is not a kink: the map is the
// identity there (soft) or a scaling (elastic net), also at x = 0.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace fp {
namespace {

inline double sgn(double x) noexcept { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }

// std::max(a, 0.0) returns a when a is NaN, which keeps NaN propagating.
inline double pos(double a) noexcept { return std::max(a, 0.0); }

// g(z) = lambda sum_i w_i |z_i|.
struct Soft final : ElemOp<Soft> {
    double lambda;
    Vec w;
    Soft(double l, Vec ww) : lambda(l), w(std::move(ww)) {}

    const char *name() const noexcept override { return "soft"; }
    double step_scale() const noexcept override { return lambda * vmax(w); }
    bool conj_homogeneous() const noexcept override { return true; }
    int spectral() const noexcept override { return uniform(w) ? kSpectralBoth : 0; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    bool axis_params() const noexcept override { return w.size() > 1; }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    int64_t wi(int64_t i) const noexcept { return w.size() == 1 ? 0 : i; }

    double value1(double x, int64_t i, double) const noexcept { return lambda * w[wi(i)] * std::fabs(x); }

    template <class T>
    T prox1(T x, double gamma, int64_t i) const noexcept {
        const double xd = static_cast<double>(x);
        const double tau = gamma * lambda * w[wi(i)];
        return static_cast<T>(std::copysign(pos(std::fabs(xd) - tau), xd));
    }
    // Pieces: 0 thresholded, 1 shifted by -s tau (the identity when tau = 0,
    // where s = 0 at x = 0). Kink 1 at |x| = tau > 0.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double gamma, int64_t i) const noexcept {
        const double tau = gamma * lambda * w[wi(i)];
        return {!(tau > 0.0) || std::fabs(x) > tau ? 1 : 0, sgn(x)};
    }
    ElemDeriv deriv_piece(double, double gamma, int64_t i, Piece p) const noexcept {
        ElemDeriv e;
        const int64_t k = wi(i);
        if (p.id == 1) {
            const double s = p.s;
            e.j = 1.0;
            e.dg = -s * lambda * w[k];
            e.np = 1;
            e.pi[0] = k;
            e.pv[0] = -s * gamma * lambda;
        }
        return e;
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i) const noexcept {
        return deriv_piece(x, gamma, i, piece_of(x, gamma, i));
    }
    int kink1(double x, double gamma, int64_t i, double tol, Kink &k) const noexcept {
        const double tau = gamma * lambda * w[wi(i)];
        if (!(tau > 0.0) || !(std::fabs(std::fabs(x) - tau) <= tol)) return 0;
        k.b = std::copysign(tau, x);
        k.scale = tau;
        return 1;
    }
};

// g(z) = l1 ||z||_1 + (l2/2) ||z||^2: z = soft(x, gamma l1) / (1 + gamma l2).
struct ElasticNet final : ElemOp<ElasticNet> {
    double l1, l2;
    ElasticNet(double a, double b) : l1(a), l2(b) {}

    const char *name() const noexcept override { return "elastic_net"; }
    double step_scale() const noexcept override { return std::max(l1, l2); }
    int spectral() const noexcept override { return kSpectralBoth; }
    int64_t nparams() const noexcept override { return 2; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    double value1(double x, int64_t, double) const noexcept { return l1 * std::fabs(x) + 0.5 * l2 * x * x; }

    template <class T>
    T prox1(T x, double gamma, int64_t) const noexcept {
        const double xd = static_cast<double>(x);
        return static_cast<T>(std::copysign(pos(std::fabs(xd) - gamma * l1), xd) /
                              (1.0 + gamma * l2));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t) const noexcept {
        ElemDeriv e;
        const double tau = gamma * l1;
        if (!(tau > 0.0) || std::fabs(x) > tau) {
            const double s = sgn(x);
            const double d = 1.0 + gamma * l2;
            const double z = (x - gamma * l1 * s) / d;
            e.j = 1.0 / d;
            e.dg = -(l1 * s + l2 * z) / d;
            e.np = 2;
            e.pi[0] = 0;
            e.pv[0] = -gamma * s / d;
            e.pi[1] = 1;
            e.pv[1] = -gamma * z / d;
        }
        return e;
    }
};

// g(z) = lambda sum_i h_delta(z_i), the Moreau envelope of |.| with parameter
// delta: h(t) = t^2/(2 delta) for |t| <= delta, |t| - delta/2 otherwise.
struct Huber final : ElemOp<Huber> {
    double lambda, delta;
    Huber(double l, double d) : lambda(l), delta(d) {}

    const char *name() const noexcept override { return "huber"; }
    double step_scale() const noexcept override { return lambda; }  // the kernel forms gamma lambda only
    int spectral() const noexcept override { return kSpectralBoth; }
    int64_t nparams() const noexcept override { return 2; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    double value1(double x, int64_t, double) const noexcept {
        const double a = std::fabs(x);
        return lambda * (a <= delta ? x * x / (2.0 * delta) : a - 0.5 * delta);
    }

    template <class T>
    T prox1(T x, double gamma, int64_t) const noexcept {
        const double xd = static_cast<double>(x);
        const double c = delta + gamma * lambda;
        if (std::fabs(xd) <= c) return static_cast<T>(xd * (delta / c));
        return static_cast<T>(xd - gamma * lambda * std::copysign(1.0, xd));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t) const noexcept {
        ElemDeriv e;
        const double c = delta + gamma * lambda;
        e.np = 2;
        e.pi[0] = 0;
        e.pi[1] = 1;
        if (std::fabs(x) <= c) {
            // With q = x / c (|q| <= 1 here), delta / c and gamma lambda / c (in
            // [0, 1]): no product grows past its factors, where c^2 and
            // x delta lambda over- or underflow (audit K10).
            const double q = x / c, rd = delta / c, rl = gamma * lambda / c;
            e.j = rd;
            e.dg = -q * rd * lambda;
            e.pv[0] = -q * rd * gamma;
            e.pv[1] = q * rl;
        } else {
            const double s = sgn(x);
            e.j = 1.0;
            e.dg = -lambda * s;
            e.pv[0] = -gamma * s;
            e.pv[1] = 0.0;
        }
        return e;
    }
};

// Projection onto lo <= z <= hi. theta = (lo..., hi...).
struct Box final : ElemOp<Box> {
    Vec lo, hi;
    Box(Vec a, Vec b) : lo(std::move(a)), hi(std::move(b)) {}

    const char *name() const noexcept override { return "box"; }
    double step_scale() const noexcept override { return 0.0; }
    // Eigenvalue bounds; on singular values only the symmetric box [-c, c].
    int spectral() const noexcept override {
        if (!uniform(lo) || !uniform(hi)) return 0;
        return lo[0] == -hi[0] ? kSpectralBoth : kSpectralEig;
    }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(lo.size() + hi.size()); }
    bool axis_params() const noexcept override { return lo.size() > 1 || hi.size() > 1; }
    fp_status check_n(int64_t n) const noexcept override {
        const bool ok = (lo.size() == 1 || static_cast<int64_t>(lo.size()) == n) &&
                        (hi.size() == 1 || static_cast<int64_t>(hi.size()) == n);
        return ok ? FP_OK : FP_ESHAPE;
    }
    int64_t li(int64_t i) const noexcept { return lo.size() == 1 ? 0 : i; }
    int64_t hj(int64_t i) const noexcept { return hi.size() == 1 ? 0 : i; }

    double value1(double x, int64_t i, double atol) const noexcept {
        return indicator(std::max(lo[li(i)] - x, x - hi[hj(i)]), atol);
    }

    template <class T>
    T prox1(T x, double, int64_t i) const noexcept {
        // Written so that NaN passes through: comparisons with NaN are false.
        const double xd = static_cast<double>(x);
        const double l = lo[li(i)], h = hi[hj(i)];
        return static_cast<T>(xd < l ? l : (xd > h ? h : xd));
    }
    // Pieces (since 0.3 explicit): 0 clipped to the lower bound, 1 inside, 2
    // clipped to the upper bound. Kinks 1 at x = lower, 2 at x = upper.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double, int64_t i) const noexcept {
        // An entry with lo = hi is fixed: no inside piece (audit K11).
        const double l = lo[li(i)], h = hi[hj(i)];
        return {(x < l || (x == l && l == h)) ? 0 : (x > h ? 2 : 1), 1.0};
    }
    ElemDeriv deriv_piece(double, double, int64_t i, Piece p) const noexcept {
        ElemDeriv e;
        if (p.id == 0) {
            e.np = 1;
            e.pi[0] = li(i);
            e.pv[0] = 1.0;
        } else if (p.id == 2) {
            e.np = 1;
            e.pi[0] = static_cast<int64_t>(lo.size()) + hj(i);
            e.pv[0] = 1.0;
        } else {
            e.j = 1.0;
        }
        return e;
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i) const noexcept {
        return deriv_piece(x, gamma, i, piece_of(x, gamma, i));
    }
    int kink1(double x, double, int64_t i, double tol, Kink &k) const noexcept {
        const double l = lo[li(i)], h = hi[hj(i)];
        const double dl = std::fabs(x - l), dh = std::fabs(x - h);
        const int code = dl <= tol && (dl <= dh || !(dh <= tol)) ? 1 : (dh <= tol ? 2 : 0);
        if (code) {
            k.b = code == 1 ? l : h;
            k.scale = std::fabs(k.b);
        }
        return code;
    }
};

struct NonNeg final : ElemOp<NonNeg> {
    const char *name() const noexcept override { return "nonneg"; }
    double step_scale() const noexcept override { return 0.0; }
    bool conj_homogeneous() const noexcept override { return true; }
    int spectral() const noexcept override { return kSpectralEig; }  // the PSD cone
    int64_t nparams() const noexcept override { return 0; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    double value1(double x, int64_t, double atol) const noexcept { return indicator(-x, atol); }

    template <class T>
    T prox1(T x, double, int64_t) const noexcept {
        return x < T(0) ? T(0) : x;
    }
    // Pieces (since 0.3 explicit): 0 clipped, 1 inside; kink 1 at x = 0.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double, int64_t) const noexcept { return {x < 0 ? 0 : 1, 1.0}; }
    ElemDeriv deriv_piece(double, double, int64_t, Piece p) const noexcept {
        ElemDeriv e;
        e.j = p.id;
        return e;
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i) const noexcept {
        return deriv_piece(x, gamma, i, piece_of(x, gamma, i));
    }
    int kink1(double x, double, int64_t, double tol, Kink &k) const noexcept {
        if (!(std::fabs(x) <= tol)) return 0;
        k.b = 0.0;
        k.scale = 0.0;
        return 1;
    }
};

// g(z) = -lambda sum_i log z_i. theta = (lambda).
struct LogBarrier final : ElemOp<LogBarrier> {
    double lambda;
    explicit LogBarrier(double l) : lambda(l) {}

    const char *name() const noexcept override { return "log_barrier"; }
    double step_scale() const noexcept override { return lambda; }
    int spectral() const noexcept override { return kSpectralEig; }  // -log det
    int64_t nparams() const noexcept override { return 1; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    // r = sqrt(x^2 + 4 gamma lambda), through hypot where the square overflows.
    static double root(double x, double gl) noexcept {
        const double r = std::sqrt(x * x + 4.0 * gl);
        return r == HUGE_VAL ? std::hypot(x, 2.0 * std::sqrt(gl)) : r;
    }
    static double value(double x, double r, double gl) noexcept {
        // NaN fails x < 0 and stays NaN; -inf gives 0, +inf stays inf.
        return x < 0.0 ? gl / (0.5 * (r - x)) : 0.5 * x + 0.5 * r;
    }

    // -lambda log x; INFINITY at x <= 0, outside the domain.
    double value1(double x, int64_t, double) const noexcept {
        return x > 0.0 ? -lambda * std::log(x) : std::numeric_limits<double>::infinity();
    }

    template <class T>
    T prox1(T x, double gamma, int64_t) const noexcept {
        const double xd = static_cast<double>(x), gl = gamma * lambda;
        return static_cast<T>(value(xd, root(xd, gl), gl));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t) const noexcept {
        ElemDeriv e;
        const double gl = gamma * lambda, r = root(x, gl);
        e.np = 1;
        e.pi[0] = 0;
        if (r > 0.0) {
            e.j = value(x, r, gl) / r;
            e.dg = lambda / r;
            e.pv[0] = gamma / r;
        } else {
            // gamma = 0 and x = 0: the nonnegative projection's rule (the
            // derivatives in gamma and lambda are infinite there).
            e.j = 1.0;
        }
        return e;
    }
};

// g(z) = lambda sum_i w_i |z_i| + indicator(lo <= z <= hi).
// theta = (w..., lo..., hi...).
struct L1Box final : ElemOp<L1Box> {
    double lambda;
    Vec w, lo, hi;
    L1Box(double l, Vec ww, Vec a, Vec b)
        : lambda(l), w(std::move(ww)), lo(std::move(a)), hi(std::move(b)) {}

    const char *name() const noexcept override { return "l1_box"; }
    double step_scale() const noexcept override { return lambda * vmax(w); }
    int64_t nparams() const noexcept override {
        return static_cast<int64_t>(w.size() + lo.size() + hi.size());
    }
    bool axis_params() const noexcept override {
        return w.size() > 1 || lo.size() > 1 || hi.size() > 1;
    }
    fp_status check_n(int64_t n) const noexcept override {
        auto fits = [n](const Vec &v) { return v.size() == 1 || static_cast<int64_t>(v.size()) == n; };
        return fits(w) && fits(lo) && fits(hi) ? FP_OK : FP_ESHAPE;
    }
    static int64_t at(const Vec &v, int64_t i) noexcept { return v.size() == 1 ? 0 : i; }

    double value1(double x, int64_t i, double atol) const noexcept {
        return lambda * w[at(w, i)] * std::fabs(x) + indicator(std::max(lo[at(lo, i)] - x, x - hi[at(hi, i)]), atol);
    }

    template <class T>
    T prox1(T x, double gamma, int64_t i) const noexcept {
        const double xd = static_cast<double>(x);
        // tau as its own statement, as in Soft and deriv1: a compiler may
        // contract a - b * c into one rounding within an expression.
        const double tau = gamma * lambda * w[at(w, i)];
        const double y = std::copysign(pos(std::fabs(xd) - tau), xd);
        const double l = lo[at(lo, i)], h = hi[at(hi, i)];
        return static_cast<T>(y < l ? l : (y > h ? h : y));
    }
    // Pieces: 0 thresholded (inside the bounds), 1 soft's shifted branch, 2
    // clipped at lower, 3 clipped at upper. Kinks: 1 at |x| = tau, 2 where
    // soft(x) = lower, 3 where soft(x) = upper.
    static constexpr bool kKinks = true;
    Piece piece_of(double x, double gamma, int64_t i) const noexcept {
        const double tau = gamma * lambda * w[at(w, i)];
        const double y = std::copysign(pos(std::fabs(x) - tau), x);
        const double l = lo[at(lo, i)], h = hi[at(hi, i)];
        if (y < l || (y == l && l == h)) return {2, sgn(x)};  // lo = hi: fixed (audit K11)
        if (y > h) return {3, sgn(x)};
        return {!(tau > 0.0) || std::fabs(x) > tau ? 1 : 0, sgn(x)};
    }
    ElemDeriv deriv_piece(double, double gamma, int64_t i, Piece p) const noexcept {
        ElemDeriv e;
        const int64_t k = at(w, i), a = at(lo, i), b = at(hi, i);
        e.np = 1;
        e.pv[0] = 1.0;
        if (p.id == 2) {
            e.pi[0] = static_cast<int64_t>(w.size()) + a;
        } else if (p.id == 3) {
            e.pi[0] = static_cast<int64_t>(w.size() + lo.size()) + b;
        } else if (p.id == 1) {
            const double s = p.s;
            e.j = 1.0;
            e.dg = -s * lambda * w[k];
            e.pi[0] = k;
            e.pv[0] = -s * gamma * lambda;
        } else {
            e.np = 0;
        }
        return e;
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i) const noexcept {
        return deriv_piece(x, gamma, i, piece_of(x, gamma, i));
    }
    // The point where soft(x) = v (v finite): v + tau above 0, v - tau below,
    // and the threshold on v's side at 0 (-tau for a lower bound, +tau for an
    // upper one, where the clipped piece begins).
    static double soft_inverse(double v, double tau, double at0) noexcept {
        return v > 0.0 ? v + tau : (v < 0.0 ? v - tau : at0);
    }
    int kink1(double x, double gamma, int64_t i, double tol, Kink &k) const noexcept {
        const double tau = gamma * lambda * w[at(w, i)];
        const double l = lo[at(lo, i)], h = hi[at(hi, i)];
        int code = 0;
        double best = tol;
        auto consider = [&](int c, double b, double scale) {
            const double dist = std::fabs(x - b);
            if (dist <= best && (code == 0 || dist < best)) {
                code = c;
                best = dist;
                k.b = b;
                k.scale = scale;
            }
        };
        if (tau > 0.0) consider(1, std::copysign(tau, x), tau);
        if (std::isfinite(l)) consider(2, soft_inverse(l, tau, -tau), std::fmax(std::fabs(l), tau));
        if (std::isfinite(h)) consider(3, soft_inverse(h, tau, tau), std::fmax(std::fabs(h), tau));
        return code;
    }
};

template <class Op, class... A>
fp_status emit(fp_op **out, A &&...a) {
    *out = new Op(std::forward<A>(a)...);
    return FP_OK;
}

}  // namespace

fp_status make_soft(double lambda, Vec w, fp_op **out) {
    if (!finite_nonneg(lambda)) return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    return emit<Soft>(out, lambda, std::move(w));
}

fp_status make_elastic_net(double l1, double l2, fp_op **out) {
    if (!finite_nonneg(l1) || !finite_nonneg(l2)) return FP_EPARAM;
    return emit<ElasticNet>(out, l1, l2);
}

fp_status make_huber(double lambda, double delta, fp_op **out) {
    if (!finite_nonneg(lambda) || !finite_nonneg(delta) || delta == 0.0) return FP_EPARAM;
    return emit<Huber>(out, lambda, delta);
}

fp_status make_box(Vec lo, Vec hi, fp_op **out) {
    if (lo.empty() || hi.empty()) return FP_ESHAPE;
    for (double v : lo)
        if (std::isnan(v) || v == HUGE_VAL) return FP_EPARAM;
    for (double v : hi)
        if (std::isnan(v) || v == -HUGE_VAL) return FP_EPARAM;
    if (lo.size() == hi.size() || lo.size() == 1 || hi.size() == 1) {
        const size_t m = std::max(lo.size(), hi.size());
        for (size_t i = 0; i < m; ++i)
            if (lo[lo.size() == 1 ? 0 : i] > hi[hi.size() == 1 ? 0 : i]) return FP_EPARAM;
    } else {
        return FP_ESHAPE;
    }
    return emit<Box>(out, std::move(lo), std::move(hi));
}

fp_status make_nonneg(fp_op **out) { return emit<NonNeg>(out); }

fp_status make_log_barrier(double lambda, fp_op **out) {
    if (!finite_nonneg(lambda) || lambda == 0.0) return FP_EPARAM;
    return emit<LogBarrier>(out, lambda);
}

fp_status make_l1_box(double lambda, Vec w, Vec lo, Vec hi, fp_op **out) {
    if (!finite_nonneg(lambda)) return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    // The box rules of make_box: shapes, bounds, lo <= hi.
    fp_op *box = nullptr;
    if (fp_status s = make_box(lo, hi, &box)) return s;
    delete box;
    return emit<L1Box>(out, lambda, std::move(w), std::move(lo), std::move(hi));
}

}  // namespace fp
