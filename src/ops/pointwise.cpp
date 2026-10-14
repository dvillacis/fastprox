// Separable convex maps: soft thresholding, elastic net, Huber, box and
// nonnegative orthant. All closed form; derivatives are diagonal.
//
// Kink convention (spec Section 7): an entry exactly at a threshold or bound
// takes the Jacobian of the branch that contains the boundary, i.e. soft
// thresholding treats |x| = tau as thresholded, and the box treats
// x in [lo, hi] as interior.
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
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    bool axis_params() const noexcept override { return w.size() > 1; }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    int64_t wi(int64_t i) const noexcept { return w.size() == 1 ? 0 : i; }

    template <class T>
    T prox1(T x, double gamma, int64_t i) const noexcept {
        const double xd = static_cast<double>(x);
        const double tau = gamma * lambda * w[wi(i)];
        return static_cast<T>(std::copysign(pos(std::fabs(xd) - tau), xd));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t i) const noexcept {
        ElemDeriv e;
        const int64_t k = wi(i);
        if (std::fabs(x) > gamma * lambda * w[k]) {
            const double s = sgn(x);
            e.j = 1.0;
            e.dg = -s * lambda * w[k];
            e.np = 1;
            e.pi[0] = k;
            e.pv[0] = -s * gamma * lambda;
        }
        return e;
    }
};

// g(z) = l1 ||z||_1 + (l2/2) ||z||^2: z = soft(x, gamma l1) / (1 + gamma l2).
struct ElasticNet final : ElemOp<ElasticNet> {
    double l1, l2;
    ElasticNet(double a, double b) : l1(a), l2(b) {}

    const char *name() const noexcept override { return "elastic_net"; }
    int64_t nparams() const noexcept override { return 2; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    template <class T>
    T prox1(T x, double gamma, int64_t) const noexcept {
        const double xd = static_cast<double>(x);
        return static_cast<T>(std::copysign(pos(std::fabs(xd) - gamma * l1), xd) /
                              (1.0 + gamma * l2));
    }
    ElemDeriv deriv1(double x, double gamma, int64_t) const noexcept {
        ElemDeriv e;
        if (std::fabs(x) > gamma * l1) {
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
    int64_t nparams() const noexcept override { return 2; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

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
            const double c2 = c * c;
            e.j = delta / c;
            e.dg = -x * delta * lambda / c2;
            e.pv[0] = -x * delta * gamma / c2;
            e.pv[1] = x * gamma * lambda / c2;
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
    int64_t nparams() const noexcept override { return static_cast<int64_t>(lo.size() + hi.size()); }
    bool axis_params() const noexcept override { return lo.size() > 1 || hi.size() > 1; }
    fp_status check_n(int64_t n) const noexcept override {
        const bool ok = (lo.size() == 1 || static_cast<int64_t>(lo.size()) == n) &&
                        (hi.size() == 1 || static_cast<int64_t>(hi.size()) == n);
        return ok ? FP_OK : FP_ESHAPE;
    }
    int64_t li(int64_t i) const noexcept { return lo.size() == 1 ? 0 : i; }
    int64_t hj(int64_t i) const noexcept { return hi.size() == 1 ? 0 : i; }

    template <class T>
    T prox1(T x, double, int64_t i) const noexcept {
        // Written so that NaN passes through: comparisons with NaN are false.
        const double xd = static_cast<double>(x);
        const double l = lo[li(i)], h = hi[hj(i)];
        return static_cast<T>(xd < l ? l : (xd > h ? h : xd));
    }
    ElemDeriv deriv1(double x, double, int64_t i) const noexcept {
        ElemDeriv e;
        const int64_t a = li(i), b = hj(i);
        if (x < lo[a]) {
            e.np = 1;
            e.pi[0] = a;
            e.pv[0] = 1.0;
        } else if (x > hi[b]) {
            e.np = 1;
            e.pi[0] = static_cast<int64_t>(lo.size()) + b;
            e.pv[0] = 1.0;
        } else {
            e.j = 1.0;
        }
        return e;
    }
};

struct NonNeg final : ElemOp<NonNeg> {
    const char *name() const noexcept override { return "nonneg"; }
    int64_t nparams() const noexcept override { return 0; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    template <class T>
    T prox1(T x, double, int64_t) const noexcept {
        return x < T(0) ? T(0) : x;
    }
    ElemDeriv deriv1(double x, double, int64_t) const noexcept {
        ElemDeriv e;
        e.j = x < 0 ? 0.0 : 1.0;
        return e;
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

}  // namespace fp
