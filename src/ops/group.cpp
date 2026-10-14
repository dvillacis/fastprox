// Group soft thresholding: g(z) = lambda sum_g c_g ||z_g||_2 along the axis,
// with the l2 norm as the single-group case.
//
// On a live group (||x_g|| > tau_g = gamma lambda c_g) the prox is
// z_g = (1 - tau_g/||x_g||) x_g and its Jacobian is (1 - c) I + c u u^T with
// c = tau_g/||x_g||, u = x_g/||x_g||; killed groups have zero Jacobian.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <cmath>
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

}  // namespace

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
