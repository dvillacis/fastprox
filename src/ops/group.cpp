// Group soft thresholding: g(z) = lambda sum_g c_g ||z_g||_2 along the axis,
// with the l2 norm as the single-group case.
//
// On a live group (||x_g|| > tau_g = gamma lambda c_g) the prox is
// z_g = (1 - tau_g/||x_g||) x_g and its Jacobian is (1 - c) I + c u u^T with
// c = tau_g/||x_g||, u = x_g/||x_g||; killed groups have zero Jacobian.
//
// Tree-structured group norms: g(z) = l1 ||z||_1 + sum_g eta_g ||z_g||_2 for
// groups that are pairwise nested or disjoint. The prox is the composition of
// the group shrinks, each the identity off its group, applied from the
// smallest group up, with soft thresholding (the singletons) first:
// F. Bach, R. Jenatton, J. Mairal and G. Obozinski, "Optimization with
// sparsity-inducing penalties", Found. Trends Mach. Learn. 4(1) (2012) 1-106,
// (3.8), p. 48, after R. Jenatton, J. Mairal, G. Obozinski and F. Bach,
// ICML 2010. Any order in which a group follows the groups inside it works,
// and sorting by size (stably) gives one. The sparse group lasso is the case
// of a partition. The cost is the total size of the groups.
//
// Derivatives follow the chain. A live shrink w_g <- a w_g, a = 1 - tau/s,
// s = ||w_g||, tau = gamma eta_g, maps a tangent t_g to
//   a t_g + ((tau / s^3) <w_g, t_g> - dtau / s) w_g,
// a symmetric map of t_g; a killed group (s <= tau) maps it to 0. The JVP runs
// it forward alongside the prox; the VJP keeps each group's input w_g and runs
// it backwards.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace fp {
namespace {

struct GroupL2 final : RowOp<GroupL2> {
    double lambda;
    std::vector<int64_t> gid;  // empty: one group
    int64_t G;
    Vec c;  // group weights, length 1 or G

    GroupL2(double l, std::vector<int64_t> g, int64_t ng, Vec cc)
        : lambda(l), gid(std::move(g)), G(ng), c(std::move(cc)) {}

    const char *name() const noexcept override { return gid.empty() ? "l2" : "group_l2"; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(c.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (gid.empty() || static_cast<int64_t>(gid.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    size_t scratch_bytes(int64_t) const noexcept override {
        return 2 * static_cast<size_t>(G) * sizeof(double);
    }

    int64_t group(int64_t i) const noexcept { return gid.empty() ? 0 : gid[i]; }
    int64_t ci(int64_t g) const noexcept { return c.size() == 1 ? 0 : g; }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *) const noexcept {
        double *scale = static_cast<double *>(scr);
        for (int64_t g = 0; g < G; ++g) scale[g] = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            if (std::isinf(v)) return kRowNonFinite;
            scale[group(i)] += v * v;
        }
        for (int64_t g = 0; g < G; ++g) {
            const double s = std::sqrt(scale[g]);
            const double tau = gamma * lambda * c[ci(g)];
            // NaN norms give NaN scales, so the whole group becomes NaN.
            scale[g] = s > tau ? 1.0 - tau / s : (s == s ? 0.0 : s);
        }
        for (int64_t i = 0; i < n; ++i)
            z[i] = static_cast<T>(static_cast<double>(x[i]) * scale[group(i)]);
        return kRowOk;
    }

    // Fills nrm[g] = ||x_g|| and dot[g] = <x_g, v_g>.
    template <class T>
    void norms_dots(const T *x, const T *v, int64_t n, double *nrm, double *dot) const noexcept {
        for (int64_t g = 0; g < G; ++g) nrm[g] = dot[g] = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);
            nrm[group(i)] += xi * xi;
            dot[group(i)] += xi * static_cast<double>(v[i]);
        }
        for (int64_t g = 0; g < G; ++g) nrm[g] = std::sqrt(nrm[g]);
    }

    template <class T>
    void jvp(const T *x, const T *, const uint8_t *, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dparams, void *scr) const noexcept {
        double *a = static_cast<double *>(scr);  // coefficient of dx
        double *b = a + G;                       // coefficient of x
        norms_dots(x, dx, n, a, b);
        for (int64_t g = 0; g < G; ++g) {
            const double s = a[g];
            const double tau = gamma * lambda * c[ci(g)];
            if (s > tau) {
                const double cc = tau / s;
                double coef = cc * b[g] / s - lambda * c[ci(g)] * dgamma;
                if (dparams) coef -= gamma * lambda * dparams[ci(g)];
                a[g] = 1.0 - cc;
                b[g] = coef / s;
            } else {
                a[g] = b[g] = 0.0;
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = group(i);
            dz[i] = static_cast<T>(a[g] * static_cast<double>(dx[i]) +
                                   b[g] * static_cast<double>(x[i]));
        }
    }

    template <class T>
    void vjp(const T *x, const T *, const uint8_t *, const T *gv, T *gx, int64_t n, double gamma,
             double *acc, void *scr) const noexcept {
        const int64_t P = nparams();
        double *a = static_cast<double *>(scr);
        double *b = a + G;
        norms_dots(x, gv, n, a, b);
        for (int64_t g = 0; g < G; ++g) {
            const double s = a[g];
            const double tau = gamma * lambda * c[ci(g)];
            if (s > tau) {
                const double cc = tau / s;
                const double ug = b[g] / s;  // <u, g_g>
                acc[ci(g)] += -gamma * lambda * ug;
                acc[P] += -lambda * c[ci(g)] * ug;
                a[g] = 1.0 - cc;
                b[g] = cc * ug / s;
            } else {
                a[g] = b[g] = 0.0;
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = group(i);
            gx[i] = static_cast<T>(a[g] * static_cast<double>(gv[i]) +
                                   b[g] * static_cast<double>(x[i]));
        }
    }
};

struct TreeL2 final : RowOp<TreeL2> {
    double l1;
    std::vector<int64_t> ptr, idx;  // groups in the caller's order
    Vec eta;                        // length 1 or G
    std::vector<int64_t> order;     // processing order: by size, stable
    int64_t G, total, need;         // need: the smallest valid axis length

    TreeL2(double a, std::vector<int64_t> p, std::vector<int64_t> i, Vec e,
           std::vector<int64_t> o, int64_t nd)
        : l1(a), ptr(std::move(p)), idx(std::move(i)), eta(std::move(e)), order(std::move(o)),
          G(static_cast<int64_t>(ptr.size()) - 1), total(static_cast<int64_t>(idx.size())),
          need(nd) {}

    const char *name() const noexcept override { return "tree_l2"; }
    int64_t nparams() const noexcept override { return 1 + static_cast<int64_t>(eta.size()); }
    fp_status check_n(int64_t n) const noexcept override { return n >= need ? FP_OK : FP_ESHAPE; }
    size_t scratch_bytes(int64_t n) const noexcept override {
        return static_cast<size_t>(2 * n + total) * sizeof(double);
    }
    int64_t ei(int64_t g) const noexcept { return eta.size() == 1 ? 0 : g; }

    // w = soft(x, gamma l1) in double; false on an infinity.
    template <class T>
    bool start(const T *x, int64_t n, double gamma, double *w) const noexcept {
        const double tau = gamma * l1;
        for (int64_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            if (std::isinf(v)) return false;
            w[i] = std::copysign(std::max(std::fabs(v) - tau, 0.0), v);
        }
        return true;
    }
    double norm(const double *w, int64_t g) const noexcept {
        double q = 0.0;
        for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) q += w[idx[k]] * w[idx[k]];
        return std::sqrt(q);
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *) const noexcept {
        double *w = static_cast<double *>(scr);
        if (!start(x, n, gamma, w)) return kRowNonFinite;
        for (int64_t g : order) {
            const double s = norm(w, g), tau = gamma * eta[ei(g)];
            // A NaN norm gives a NaN scale, so the group becomes NaN.
            const double a = s > tau ? 1.0 - tau / s : (s == s ? 0.0 : s);
            for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) w[idx[k]] *= a;
        }
        for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(w[i]);
        return kRowOk;
    }

    template <class T>
    void jvp(const T *x, const T *, const uint8_t *, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dp, void *scr) const noexcept {
        double *w = static_cast<double *>(scr);
        double *t = w + n;
        const double tau1 = gamma * l1, dtau1 = l1 * dgamma + (dp ? gamma * dp[0] : 0.0);
        for (int64_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            if (std::fabs(v) > tau1) {
                const double sg = v > 0 ? 1.0 : -1.0;
                w[i] = v - sg * tau1;
                t[i] = static_cast<double>(dx[i]) - sg * dtau1;
            } else {
                w[i] = t[i] = 0.0;
            }
        }
        for (int64_t g : order) {
            const double s = norm(w, g), e = eta[ei(g)], tau = gamma * e;
            if (s > tau) {
                double wt = 0.0;
                for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) wt += w[idx[k]] * t[idx[k]];
                const double dtau = e * dgamma + (dp ? gamma * dp[1 + ei(g)] : 0.0);
                const double a = 1.0 - tau / s, c = tau / (s * s * s) * wt - dtau / s;
                for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) {
                    const int64_t i = idx[k];
                    t[i] = a * t[i] + c * w[i];
                    w[i] *= a;
                }
            } else {
                for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) w[idx[k]] = t[idx[k]] = 0.0;
            }
        }
        for (int64_t i = 0; i < n; ++i) dz[i] = static_cast<T>(t[i]);
    }

    template <class T>
    void vjp(const T *x, const T *, const uint8_t *, const T *gv, T *gx, int64_t n, double gamma,
             double *acc, void *scr) const noexcept {
        const int64_t P = nparams();
        double *w = static_cast<double *>(scr);
        double *u = w + n;
        double *before = u + n;  // each group's input, in processing order
        start(x, n, gamma, w);
        int64_t off = 0;
        for (int64_t g : order) {
            const double s = norm(w, g), tau = gamma * eta[ei(g)];
            const double a = s > tau ? 1.0 - tau / s : 0.0;
            for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) {
                before[off++] = w[idx[k]];
                w[idx[k]] *= a;
            }
        }
        for (int64_t i = 0; i < n; ++i) u[i] = static_cast<double>(gv[i]);
        for (int64_t r = G - 1; r >= 0; --r) {
            const int64_t g = order[r], len = ptr[g + 1] - ptr[g];
            off -= len;
            const double *wb = before + off;
            double q = 0.0;
            for (int64_t k = 0; k < len; ++k) q += wb[k] * wb[k];
            const double s = std::sqrt(q), e = eta[ei(g)], tau = gamma * e;
            if (s > tau) {
                double wu = 0.0;
                for (int64_t k = 0; k < len; ++k) wu += wb[k] * u[idx[ptr[g] + k]];
                acc[1 + ei(g)] += -gamma * wu / s;
                acc[P] += -e * wu / s;
                const double a = 1.0 - tau / s, c = tau / (s * s * s) * wu;
                for (int64_t k = 0; k < len; ++k) {
                    double &ui = u[idx[ptr[g] + k]];
                    ui = a * ui + c * wb[k];
                }
            } else {
                for (int64_t k = 0; k < len; ++k) u[idx[ptr[g] + k]] = 0.0;
            }
        }
        const double tau1 = gamma * l1;
        for (int64_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            if (std::fabs(v) > tau1) {
                const double sg = v > 0 ? 1.0 : -1.0;
                acc[0] += -sg * gamma * u[i];
                acc[P] += -sg * l1 * u[i];
                gx[i] = static_cast<T>(u[i]);
            } else {
                gx[i] = T(0);
            }
        }
    }
};

}  // namespace

fp_status make_tree_l2(double l1, std::vector<int64_t> ptr, std::vector<int64_t> idx, Vec eta,
                       fp_op **out) {
    if (!finite_nonneg(l1)) return FP_EPARAM;
    if (ptr.empty() || ptr[0] != 0 || ptr.back() != static_cast<int64_t>(idx.size()))
        return FP_EPARAM;
    const int64_t G = static_cast<int64_t>(ptr.size()) - 1;
    for (int64_t g = 0; g < G; ++g)
        if (ptr[g + 1] < ptr[g]) return FP_EPARAM;
    if (eta.empty()) eta.assign(1, 1.0);
    if (eta.size() != 1 && static_cast<int64_t>(eta.size()) != G) return FP_ESHAPE;
    for (double v : eta)
        if (!finite_nonneg(v)) return FP_EPARAM;
    int64_t m = 0;
    for (int64_t i : idx) {
        if (i < 0) return FP_EPARAM;
        m = std::max(m, i + 1);
    }
    std::vector<int64_t> order(static_cast<size_t>(G));
    std::iota(order.begin(), order.end(), int64_t(0));
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return ptr[a + 1] - ptr[a] < ptr[b + 1] - ptr[b];
    });
    // Tree check, in processing order. last[i] is the latest group holding i.
    // The latest earlier groups met by g must lie inside g: all of their
    // entries have them as latest group (a later group meeting one contains
    // it), so g must hold as many entries with last == h as h has.
    std::vector<int64_t> last(static_cast<size_t>(m), -1), cnt(static_cast<size_t>(G), 0),
        stamp(static_cast<size_t>(m), -1), met;
    for (int64_t g : order) {
        met.clear();
        for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) {
            const int64_t i = idx[k];
            if (stamp[i] == g) return FP_EPARAM;  // repeated entry
            stamp[i] = g;
            const int64_t h = last[i];
            if (h >= 0 && cnt[h]++ == 0) met.push_back(h);
        }
        bool nested = true;
        for (int64_t h : met) {
            nested = nested && cnt[h] == ptr[h + 1] - ptr[h];
            cnt[h] = 0;
        }
        if (!nested) return FP_EPARAM;
        for (int64_t k = ptr[g]; k < ptr[g + 1]; ++k) last[idx[k]] = g;
    }
    *out = new TreeL2(l1, std::move(ptr), std::move(idx), std::move(eta), std::move(order), m);
    return FP_OK;
}

fp_status make_group_l2(double lambda, std::vector<int64_t> groups, int64_t ngroups, Vec c,
                        fp_op **out) {
    if (!finite_nonneg(lambda)) return FP_EPARAM;
    if (groups.empty()) {
        ngroups = 1;
    } else {
        if (ngroups < 1) return FP_EPARAM;
        for (int64_t g : groups)
            if (g < 0 || g >= ngroups) return FP_EPARAM;
    }
    if (c.empty()) c.assign(1, 1.0);
    if (c.size() != 1 && static_cast<int64_t>(c.size()) != ngroups) return FP_ESHAPE;
    for (double v : c)
        if (!finite_nonneg(v)) return FP_EPARAM;
    *out = new GroupL2(lambda, std::move(groups), ngroups, std::move(c));
    return FP_OK;
}

}  // namespace fp
