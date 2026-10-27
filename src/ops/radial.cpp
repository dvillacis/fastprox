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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace fp {
namespace {

// ---- group penalties by norm composition ----------------------------------------------

struct GroupRadial final : RowOp<GroupRadial> {
    static constexpr int kStruct = FP_STRUCT_CURVED;
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
    // The kernel forms the scalar map's step gamma c_g, which h multiplies by
    // its own factors: both c and h's scale times c are factors (since 1.0.1;
    // group Huber at lambda = 1e-200 had gamma c_g overflow, a NaN group).
    double step_scale() const noexcept override { return std::max(h->step_scale(), 1.0) * cmax; }
    int64_t nparams() const noexcept override { return Ph + static_cast<int64_t>(c.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (gid.empty() || static_cast<int64_t>(gid.size()) == n) ? FP_OK : FP_ESHAPE;
    }
    fp_status check_step(double gamma) const noexcept override { return h->check_step(gamma * cmax); }
    // Six arrays of G: norms, dots, the scales of safe_group_norms, and the
    // derivatives' largest |x_i| and |v_i| and sums of squares (norms_dots).
    size_t scratch_bytes(int64_t) const noexcept override {
        return 6 * static_cast<size_t>(G) * sizeof(double);
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

    // phi(s) / s, from p = phi(s) and j = phi'(s). Near 0 every scalar map
    // here is linear through 0 (Huber's quadratic piece) or 0 (a dead zone,
    // where j = 0 too), so p = 0 with j != 0 means that j s underflowed, as for
    // a subnormal norm, and the ratio is j (found by fuzzing guarantee 1 in
    // 1.0.2: group Huber's derivative and structure took it as 0).
    static double ratio(double p, double s, double j) noexcept {
        const double r = p / s;
        return r == 0.0 && j != 0.0 ? j : r;
    }

    bool has_value() const noexcept override { return true; }
    int64_t value_doubles(int64_t n) const noexcept override { return n > 2 * G ? n : 2 * G; }
    // sum_g c_g h(||x_g||), h the scalar map (its value at one entry).
    double value_row(const double *x, int64_t n, double tol, double *scr) const noexcept override {
        for (int64_t g = 0; g < G; ++g) scr[g] = 0.0;
        for (int64_t i = 0; i < n; ++i) scr[group(i)] += x[i] * x[i];
        safe_group_norms(n, G, [x](int64_t i) { return x[i]; }, [this](int64_t i) { return group(i); }, scr, scr + G);
        double s = 0.0;
        for (int64_t g = 0; g < G; ++g) {
            const double t = scr[g] / scr[G + g];
            s += c[ci(g)] * h->value_row(&t, 1, tol, nullptr);
        }
        return s;
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
        safe_group_norms(n, G, [x](int64_t i) { return x[i]; }, [this](int64_t i) { return group(i); }, scale,
                         scale + G);
        for (int64_t g = 0; g < G; ++g) {
            const double s = norm(scale[g], scale[G + g]);
            // A NaN norm gives a NaN scale, so the whole group becomes NaN.
            scale[g] = s > 0.0 ? phi(s, gamma * c[ci(g)]) / s : (s == s ? 0.0 : s);
        }
        for (int64_t i = 0; i < n; ++i)
            z[i] = static_cast<T>(static_cast<double>(x[i]) * scale[group(i)]);
        return kRowOk;
    }

    // ||x_g|| from safe_group_norms' m and t, capped at the largest double:
    // beyond it phi(s) = s - kappa (the last piece of the scalar maps), so
    // phi(s) / s is 1 - kappa / s to rounding unless kappa is near the cap.
    static double norm(double m, double t) noexcept {
        return std::min(m / t, std::numeric_limits<double>::max());
    }
    // nrm[g] = m_g and M[g] = t_g, ||x_g|| = m_g / t_g, and ud[g] = <u_g, v_g>
    // with u_g = (t_g x_g) / m_g (group_unit_dots in core/op.hpp); s holds the
    // three arrays of G after M that it needs.
    template <class T>
    void norms_dots(const T *x, const T *v, int64_t n, double *nrm, double *ud, double *M) const noexcept {
        group_unit_dots(n, G, [x](int64_t i) { return x[i]; }, [v](int64_t i) { return v[i]; },
                        [this](int64_t i) { return group(i); }, nrm, ud, M, M + G, M + 2 * G, M + 3 * G);
    }

    // Structure (since 1.0.2): ZERO on a group set to 0, where the jvp's
    // phi(s) / s (or j at s = 0) is 0; on the others J_g has the eigenvalues
    // phi(s) / s and j > 0, positive definite.
    template <class T>
    int64_t structure(const T *x, const T *, const uint8_t *, int64_t n, double gamma, RowStruct &s,
                      void *scr) const noexcept {
        double *nrm = static_cast<double *>(scr), *M = nrm + G;
        for (int64_t g = 0; g < G; ++g) nrm[g] = 0.0;
        for (int64_t i = 0; i < n; ++i) nrm[group(i)] += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        safe_group_norms(n, G, [x](int64_t i) { return x[i]; }, [this](int64_t i) { return group(i); }, nrm, M);
        double d[4];
        for (int64_t g = 0; g < G; ++g) {
            const double sg = norm(nrm[g], M[g]);
            double j;
            const double p = deriv(sg, gamma * c[ci(g)], j, d);
            nrm[g] = (sg > 0.0 ? ratio(p, sg, j) != 0.0 : j != 0.0) ? 1.0 : 0.0;
        }
        for (int64_t i = 0; i < n; ++i) s.state[i] = nrm[group(i)] != 0.0 ? FP_ENTRY_FREE : FP_ENTRY_ZERO;
        return singletons(s, n);
    }

    template <class T>
    void jvp(const T *x, const T *, const uint8_t *, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dparams, void *scr) const noexcept {
        double *a = static_cast<double *>(scr);  // the norm, then the coefficient of dx
        double *b = a + G;                       // <u, dx>, then the coefficient of t x
        double *M = b + G;
        norms_dots(x, dx, n, a, b, M);
        double d[4];
        for (int64_t g = 0; g < G; ++g) {
            const double m = a[g], t = M[g], s = norm(m, t), cg = c[ci(g)];
            double j;
            const double p = deriv(s, gamma * cg, j, d);
            if (s > 0.0) {
                double dphi = d[Ph] * cg * dgamma;
                if (dparams) {
                    for (int64_t k = 0; k < Ph; ++k) dphi += d[k] * dparams[k];
                    dphi += d[Ph] * gamma * dparams[Ph + ci(g)];
                }
                const double r = ratio(p, s, j);
                a[g] = r;
                b[g] = ((j - r) * b[g] + dphi) / m;  // times t x_i below: u_i = (t x_i) / m
            } else {
                a[g] = j;
                b[g] = 0.0;
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = group(i);
            dz[i] = static_cast<T>(a[g] * static_cast<double>(dx[i]) +
                                   b[g] * (static_cast<double>(x[i]) * M[g]));
        }
    }

    template <class T>
    void vjp(const T *x, const T *, const uint8_t *, const T *gv, T *gx, int64_t n, double gamma,
             double *acc, void *scr) const noexcept {
        const int64_t P = nparams();
        double *a = static_cast<double *>(scr);
        double *b = a + G;
        double *M = b + G;
        norms_dots(x, gv, n, a, b, M);
        double d[4];
        for (int64_t g = 0; g < G; ++g) {
            const double m = a[g], t = M[g], s = norm(m, t), cg = c[ci(g)];
            double j;
            const double p = deriv(s, gamma * cg, j, d);
            if (s > 0.0) {
                const double ug = b[g];  // <u, g_g>
                for (int64_t k = 0; k < Ph; ++k) acc[k] += d[k] * ug;
                acc[Ph + ci(g)] += d[Ph] * gamma * ug;
                acc[P] += d[Ph] * cg * ug;
                const double r = ratio(p, s, j);
                a[g] = r;
                b[g] = (j - r) * ug / m;  // times t x_i below, as in jvp
            } else {
                a[g] = j;
                b[g] = 0.0;
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = group(i);
            gx[i] = static_cast<T>(a[g] * static_cast<double>(gv[i]) +
                                   b[g] * (static_cast<double>(x[i]) * M[g]));
        }
    }
};

// ---- second-order cone ----------------------------------------------------------------

struct Soc final : RowOp<Soc> {
    static constexpr int kStruct = FP_STRUCT_CURVED;
    const char *name() const noexcept override { return "soc"; }
    double step_scale() const noexcept override { return 0.0; }
    bool conj_homogeneous() const noexcept override { return true; }
    int64_t nparams() const noexcept override { return 0; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

    enum Region { kKeep, kZero, kMid };

    // f(s, sc) with ||x[1:]|| = s / sc (with_norm in core/op.hpp), sc = 1
    // unless the norm exceeds the largest double or is subnormal. The
    // formulas read it through t sc, (x_i sc) / s and <sc y, v> / s, which
    // are those of the plain norm, bit for bit, when sc = 1.
    template <class T, class F>
    static auto with_ynorm(const T *x, int64_t n, F f) noexcept {
        double q = 0.0;
        for (int64_t i = 1; i < n; ++i) q += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        return with_norm(q, n - 1, [x](int64_t i) { return x[i + 1]; }, f);
    }
    // The branch for t = x_0 (NaN falls through to kMid, which stays NaN).
    static Region region(double t, double s, double sc) noexcept {
        if (s <= t * sc) return kKeep;
        if (s <= -t * sc) return kZero;
        return kMid;
    }

    bool has_value() const noexcept override { return true; }
    // The indicator of ||y||_2 <= t, t = x_0.
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        if (n == 0) return 0.0;
        return with_ynorm(x, n, [=](double s, double sc) { return indicator(s / sc - x[0], value_atol(x, n, tol)); });
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double, void *, uint8_t *) const noexcept {
        for (int64_t i = 0; i < n; ++i)
            if (std::isinf(x[i])) return kRowNonFinite;
        const double t = static_cast<double>(x[0]);
        with_ynorm(x, n, [=](double s, double sc) {
            switch (region(t, s, sc)) {
            case kKeep:
                if (z != x) std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
                break;
            case kZero:
                for (int64_t i = 0; i < n; ++i) z[i] = T(0);
                break;
            case kMid: {
                const double al = 0.5 * (s + t * sc), r = al / s;
                z[0] = static_cast<T>(al / sc);
                for (int64_t i = 1; i < n; ++i) z[i] = static_cast<T>(static_cast<double>(x[i]) * r);
                break;
            }
            }
        });
        return kRowOk;
    }

    // Structure (since 1.0.2): inside, every entry FREE; in the polar cone,
    // every entry ZERO; on the boundary, every entry FREE with the
    // constraint a = (-1, u), u = y / ||y||, the kernel of J there, whose
    // eigenvalues on a^perp are 1 (along (1, u)) and alpha / s.
    template <class T>
    int64_t structure(const T *x, const T *, const uint8_t *, int64_t n, double, RowStruct &s,
                      void *) const noexcept {
        if (n == 0) return 0;
        const double t = static_cast<double>(x[0]);
        const Region reg = with_ynorm(x, n, [=](double sn, double sc) { return region(t, sn, sc); });
        for (int64_t i = 0; i < n; ++i) s.state[i] = reg == kZero ? FP_ENTRY_ZERO : FP_ENTRY_FREE;
        const int64_t k = singletons(s, n);
        if (reg == kMid && s.constraint) {
            if (s.coef)
                with_ynorm(x, n, [=, &s](double sn, double sc) {
                    s.coef[0] = -1.0;
                    for (int64_t i = 1; i < n; ++i) s.coef[i] = static_cast<double>(x[i]) * sc / sn;
                });
            s.active = true;
        }
        return k;
    }

    // The Jacobian is symmetric: jvp and vjp apply the same map.
    template <class T>
    static void apply(const T *x, const T *v, T *out, int64_t n) noexcept {
        const double t = static_cast<double>(x[0]);
        with_ynorm(x, n, [=](double s, double sc) {
            switch (region(t, s, sc)) {
            case kKeep:
                for (int64_t i = 0; i < n; ++i) out[i] = v[i];
                return;
            case kZero:
                for (int64_t i = 0; i < n; ++i) out[i] = T(0);
                return;
            case kMid:
                break;
            }
            // <u, v[1:]> (unit_dot in core/op.hpp: scaled where x_i v_i would
            // leave the range; found by fuzzing guarantee 1 in 1.0.2).
            const double uv = unit_dot(n - 1, [x](int64_t k) { return x[k + 1]; },
                                       [v](int64_t k) { return v[k + 1]; }, s, sc);
            const double r = 0.5 * (s + t * sc) / s;
            const double da = 0.5 * (uv + static_cast<double>(v[0]));
            out[0] = static_cast<T>(da);
            for (int64_t i = 1; i < n; ++i) {
                const double ui = static_cast<double>(x[i]) * sc / s;
                out[i] = static_cast<T>(da * ui + r * (static_cast<double>(v[i]) - uv * ui));
            }
        });
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
