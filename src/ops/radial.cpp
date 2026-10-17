// Radial maps along the axis: group penalties sum_g c_g h(||z_g||_2) for a
// scalar penalty h (Huber, SCAD, MCP), and the projection onto the
// second-order cone.
//
// Norm composition (A. Beck, First-Order Methods in Optimization, SIAM 2017,
// Theorem 6.18, p. 142). For a fixed radius r = ||z_g||, the closest point to
// x_g is r u with u = x_g / ||x_g|| (Cauchy-Schwarz), at distance |r - s|,
// s = ||x_g||. So z_g = phi(s) u, where phi(s) minimises
// h(r) + (r - s)^2 / (2 gamma c_g) over r >= 0. When h is even and its scalar
// prox maps [0, inf) into itself, as for these three, phi is that prox at the
// step gamma c_g. Beck states the convex case; the argument above does not
// need convexity, and for SCAD and MCP the scalar prox is single-valued under
// the conditions of the scalar operators (a > 1 + gamma c_g, b > gamma c_g).
// Group Huber, (1 - gamma lambda / max(||x||, delta + gamma lambda)) x, is
// Beck's Example 6.66 (p. 169).
//
// Derivatives. With j = phi'(s) and the scalar map's derivatives in its step
// and parameters (taken from the scalar operator itself),
//   dz_g = (phi/s) dx_g + [(j - phi/s) <u, dx_g> + dphi] u,
// with dphi = phi_step (c_g dgamma + gamma dc_g) + phi_theta . dtheta. At
// x_g = 0 the map is z_g = 0 and its Jacobian j(0) I, the limit of phi(s)/s.
//
// Second-order cone {(t, y) : ||y||_2 <= t}, with t the first entry of the
// row (the order of conic solvers; Beck, Example 6.37, pp. 154-155, puts it
// last). With s = ||y||: (t, y) when s <= t, 0 when s <= -t, else
// alpha (1, y / s) with alpha = (s + t) / 2. On the last branch the Jacobian is
//   [[1/2, u^T/2], [u/2, (alpha/s) I + (1/2 - alpha/s) u u^T]],  u = y / s.
// Kinks follow the box and soft thresholding (spec Section 7): the cone's
// boundary s = t, and the origin, take the identity; s = -t > 0 takes 0.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

namespace fp {
namespace {

// ---- group penalties by norm composition ----------------------------------------------

struct GroupRadial final : RowOp<GroupRadial> {
    std::unique_ptr<fp_op> h;  // the elementwise scalar map, with scalar parameters
    const char *nm;
    int64_t Ph;                // h's parameter count; theta = (h's theta..., c...)
    std::vector<int64_t> gid;  // empty: one group
    int64_t G;
    Vec c;  // group weights, length 1 or G
    double cmax;

    GroupRadial(fp_op *hh, const char *name, std::vector<int64_t> g, int64_t ng, Vec cc)
        : h(hh), nm(name), Ph(hh->nparams()), gid(std::move(g)), G(ng), c(std::move(cc)), cmax(0.0) {
        for (double v : c) cmax = v > cmax ? v : cmax;
    }

    const char *name() const noexcept override { return nm; }
    int64_t nparams() const noexcept override { return Ph + static_cast<int64_t>(c.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (gid.empty() || static_cast<int64_t>(gid.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    fp_status check_step(double gamma) const noexcept override { return h->check_step(gamma * cmax); }
    size_t scratch_bytes(int64_t) const noexcept override {
        return 2 * static_cast<size_t>(G) * sizeof(double);
    }

    int64_t group(int64_t i) const noexcept { return gid.empty() ? 0 : gid[i]; }
    int64_t ci(int64_t g) const noexcept { return c.size() == 1 ? 0 : g; }

    double phi(double s, double step) const noexcept {
        double z;
        h->prox_run(&s, 1, &z, 1, 1, step, 0, 0, nullptr, false, 1.0, FP_TIE_NONZERO);
        return z;
    }
    // phi(s), j = phi'(s), d[0..Ph) = dphi/dtheta and d[Ph] = dphi/dstep,
    // through one vjp of the scalar map with cotangent 1.
    double deriv(double s, double step, double &j, double *d) const noexcept {
        const double one = 1.0;
        for (int64_t k = 0; k < Ph + 2; ++k) d[k] = 0.0;
        h->vjp_run(&s, &one, 1, &j, 1, 1, step, d, 0, 0, false, 1.0, FP_TIE_NONZERO, nullptr, 0.0);
        return phi(s, step);
    }

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
            // A NaN norm gives a NaN scale, so the whole group becomes NaN.
            scale[g] = s > 0.0 ? phi(s, gamma * c[ci(g)]) / s : (s == s ? 0.0 : s);
        }
        for (int64_t i = 0; i < n; ++i)
            z[i] = static_cast<T>(static_cast<double>(x[i]) * scale[group(i)]);
        return kRowOk;
    }

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
        double d[4];
        for (int64_t g = 0; g < G; ++g) {
            const double s = a[g], cg = c[ci(g)];
            double j;
            const double p = deriv(s, gamma * cg, j, d);
            if (s > 0.0) {
                double dphi = d[Ph] * cg * dgamma;
                if (dparams) {
                    for (int64_t k = 0; k < Ph; ++k) dphi += d[k] * dparams[k];
                    dphi += d[Ph] * gamma * dparams[Ph + ci(g)];
                }
                const double r = p / s;
                a[g] = r;
                b[g] = ((j - r) * (b[g] / s) + dphi) / s;
            } else {
                a[g] = j;
                b[g] = 0.0;
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
        double d[4];
        for (int64_t g = 0; g < G; ++g) {
            const double s = a[g], cg = c[ci(g)];
            double j;
            const double p = deriv(s, gamma * cg, j, d);
            if (s > 0.0) {
                const double ug = b[g] / s;  // <u, g_g>
                for (int64_t k = 0; k < Ph; ++k) acc[k] += d[k] * ug;
                acc[Ph + ci(g)] += d[Ph] * gamma * ug;
                acc[P] += d[Ph] * cg * ug;
                const double r = p / s;
                a[g] = r;
                b[g] = (j - r) * ug / s;
            } else {
                a[g] = j;
                b[g] = 0.0;
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = group(i);
            gx[i] = static_cast<T>(a[g] * static_cast<double>(gv[i]) +
                                   b[g] * static_cast<double>(x[i]));
        }
    }
};

// ---- second-order cone ----------------------------------------------------------------

struct Soc final : RowOp<Soc> {
    const char *name() const noexcept override { return "soc"; }
    int64_t nparams() const noexcept override { return 0; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    enum Region { kKeep, kZero, kMid };

    // s = ||x[1:]||; the branch (NaN falls through to kMid, which stays NaN).
    template <class T>
    static Region region(const T *x, int64_t n, double &t, double &s) noexcept {
        t = static_cast<double>(x[0]);
        double q = 0.0;
        for (int64_t i = 1; i < n; ++i) q += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        s = std::sqrt(q);
        if (s <= t) return kKeep;
        if (s <= -t) return kZero;
        return kMid;
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double, void *, uint8_t *) const noexcept {
        for (int64_t i = 0; i < n; ++i)
            if (std::isinf(x[i])) return kRowNonFinite;
        double t, s;
        switch (region(x, n, t, s)) {
        case kKeep:
            if (z != x) std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
            break;
        case kZero:
            for (int64_t i = 0; i < n; ++i) z[i] = T(0);
            break;
        case kMid: {
            const double al = 0.5 * (s + t), r = al / s;
            z[0] = static_cast<T>(al);
            for (int64_t i = 1; i < n; ++i) z[i] = static_cast<T>(static_cast<double>(x[i]) * r);
            break;
        }
        }
        return kRowOk;
    }

    // The Jacobian is symmetric: jvp and vjp apply the same map.
    template <class T>
    static void apply(const T *x, const T *v, T *out, int64_t n) noexcept {
        double t, s;
        switch (region(x, n, t, s)) {
        case kKeep:
            for (int64_t i = 0; i < n; ++i) out[i] = v[i];
            return;
        case kZero:
            for (int64_t i = 0; i < n; ++i) out[i] = T(0);
            return;
        case kMid:
            break;
        }
        double uv = 0.0;
        for (int64_t i = 1; i < n; ++i) uv += static_cast<double>(x[i]) * static_cast<double>(v[i]);
        uv /= s;
        const double r = 0.5 * (s + t) / s;
        const double da = 0.5 * (uv + static_cast<double>(v[0]));
        out[0] = static_cast<T>(da);
        for (int64_t i = 1; i < n; ++i) {
            const double ui = static_cast<double>(x[i]) / s;
            out[i] = static_cast<T>(da * ui + r * (static_cast<double>(v[i]) - uv * ui));
        }
    }
    template <class T>
    void jvp(const T *x, const T *, const uint8_t *, const T *dx, T *dz, int64_t n, double,
             double, const double *, void *) const noexcept {
        apply(x, dx, dz, n);
    }
    template <class T>
    void vjp(const T *x, const T *, const uint8_t *, const T *g, T *gx, int64_t n, double,
             double *, void *) const noexcept {
        apply(x, g, gx, n);
    }
};

}  // namespace

fp_status make_group_radial(fp_op *scalar, const char *name, std::vector<int64_t> groups,
                            int64_t ngroups, Vec c, fp_op **out) {
    std::unique_ptr<fp_op> h(scalar);
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
    *out = new GroupRadial(h.release(), name, std::move(groups), ngroups, std::move(c));
    return FP_OK;
}

fp_status make_soc(fp_op **out) {
    *out = new Soc();
    return FP_OK;
}

}  // namespace fp
