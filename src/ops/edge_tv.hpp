// Shared parts of the operators that take the prox of
//     gamma sum_e w_e |z_u - z_v|
// over a given edge list (graph TV, tree TV): the edges and weights, the
// duality gap of a solution with an edge flow, and the exact region
// derivatives.
//
// Certificate. A flow f_e (net u -> v, |f_e| <= gamma w_e) is a dual, with
// divergence q_p = sum of f_e out of p. The duality gap of (z, f) is
//     gap = 1/2 |z - x + q|^2 + sum_e (gamma w_e |z_u - z_v| - f_e (z_u - z_v)),
// both terms nonnegative; q is written as the workspace dual.
//
// Derivatives. The solution is constant on the regions R of the graph of
// edges with gamma w_e > 0 and equal ends (up to rounding), and there
//     z_R = (sum_R x_p - gamma sum_{e in dR} w_e s_{e,R}) / |R|,
// s_{e,R} the sign of z_R minus the value across e. So the Jacobian averages
// over regions and the parameter derivatives are O(n + m).
//
// SPDX-License-Identifier: MIT
#pragma once

#include "core/op.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace fp {
namespace edge_tv {

constexpr double kEps = 2.220446049250313e-16;

inline size_t round64(size_t b) noexcept { return (b + 63) / 64 * 64; }

// Checks n, m, the edges and the weights of a constructor; fills u, v and
// sets an empty w to the unit weight.
inline fp_status parse(int64_t n, int64_t m, const int64_t *edges, Vec &w, std::vector<int32_t> &u,
                       std::vector<int32_t> &v) {
    if (n < 0 || m < 0 || (m > 0 && !edges)) return FP_EINVAL;
    if (n >= 2147483647 || m > 1073741823) return FP_ESHAPE;
    if (w.empty()) w.assign(1, 1.0);
    if (w.size() != 1 && static_cast<int64_t>(w.size()) != m) return FP_ESHAPE;
    for (double x : w)
        if (!finite_nonneg(x)) return FP_EPARAM;
    u.resize(static_cast<size_t>(m));
    v.resize(static_cast<size_t>(m));
    for (int64_t e = 0; e < m; ++e) {
        const int64_t a = edges[2 * e], b = edges[2 * e + 1];
        if (a < 0 || a >= n || b < 0 || b >= n || a == b) return FP_EPARAM;
        u[static_cast<size_t>(e)] = static_cast<int32_t>(a);
        v[static_cast<size_t>(e)] = static_cast<int32_t>(b);
    }
    return FP_OK;
}

// Union-find with roots at the smallest index, so labels are deterministic.
inline int32_t find(int32_t *uf, int32_t i) noexcept {
    while (uf[i] != i) {
        uf[i] = uf[uf[i]];
        i = uf[i];
    }
    return i;
}
inline bool unite(int32_t *uf, int32_t a, int32_t b) noexcept {
    a = find(uf, a);
    b = find(uf, b);
    if (a == b) return false;
    if (a < b) uf[b] = a;
    else uf[a] = b;
    return true;
}

template <class D>
struct Base : ImageOp<D> {
    int64_t n, m;
    std::vector<int32_t> eu, ev;
    Vec w;  // one shared weight or one per edge

    Base(int64_t nn, std::vector<int32_t> u, std::vector<int32_t> v, Vec ww)
        : n(nn), m(static_cast<int64_t>(u.size())), eu(std::move(u)), ev(std::move(v)), w(std::move(ww)) {}

    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    int image_ndim() const noexcept override { return 1; }
    fp_status check_image(const int64_t *shape) const noexcept override {
        return shape[0] == n ? FP_OK : FP_ESHAPE;
    }
    double we(int64_t e) const noexcept { return w.size() == 1 ? w[0] : w[static_cast<size_t>(e)]; }

    // Saved structure: the number of regions, a region label per node, 1/|R|
    // per region, and the sign of z_u - z_v per edge (0 inside a region).
    size_t struct_bytes(const int64_t *) const noexcept override {
        return 8 + round64(4 * static_cast<size_t>(n)) + 8 * static_cast<size_t>(n) + static_cast<size_t>(m);
    }
    struct View {
        int64_t *nreg;
        int32_t *label;
        double *inv;
        int8_t *sgn;
    };
    View view(char *save) const noexcept {
        View v;
        v.nreg = reinterpret_cast<int64_t *>(save);
        v.label = reinterpret_cast<int32_t *>(save + 8);
        v.inv = reinterpret_cast<double *>(save + 8 + round64(4 * static_cast<size_t>(n)));
        v.sgn = reinterpret_cast<int8_t *>(v.inv + n);
        return v;
    }

    // Copies x into d. Returns kRowNonFinite on Inf; on NaN fills z and the
    // dual with NaN, saves singleton regions and returns kRowOk with done set.
    template <class T>
    int read_input(const T *x, T *z, double *d, ImageCall &c, bool &done) const noexcept {
        done = false;
        bool nan = false;
        for (int64_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            if (std::isinf(v)) return kRowNonFinite;
            nan = nan || std::isnan(v);
            d[i] = v;
        }
        if (nan) {  // NaN anywhere: the whole signal is NaN, with singleton regions
            for (int64_t i = 0; i < n; ++i) z[i] = std::numeric_limits<T>::quiet_NaN();
            std::fill(c.dual, c.dual + n, std::numeric_limits<double>::quiet_NaN());
            if (c.save) {
                const View v = view(c.save);
                for (int64_t i = 0; i < n; ++i) {
                    v.label[i] = static_cast<int32_t>(i);
                    v.inv[i] = 1.0;
                }
                std::fill(v.sgn, v.sgn + m, int8_t(0));
                *v.nreg = n;
            }
            done = true;
        } else if (n == 0) {
            if (c.save) *view(c.save).nreg = 0;
            done = true;
        }
        return kRowOk;
    }

    // Gap of z with the flow f(e), and its divergence q into c.dual.
    template <class F>
    void certificate(const double *d, const double *zv, F &&flow, ImageCall &c) const noexcept {
        const double gamma = c.gamma;
        double *q = c.dual;
        std::fill(q, q + n, 0.0);
        double edge = 0.0;
        for (int64_t e = 0; e < m; ++e) {
            const double cap = gamma * we(e);
            const double f = flow(e);
            const int32_t u = eu[e], v = ev[e];
            q[u] += f;
            q[v] -= f;
            const double dz = zv[u] - zv[v];
            edge += cap * std::fabs(dz) - f * dz;
        }
        double node = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double r = zv[i] - d[i] + q[i];
            node += r * r;
        }
        c.gap = 0.5 * node + std::max(edge, 0.0);
    }

    // Regions of the solution zv into c.save; deg (n doubles) and uf (n) are scratch.
    // With side (since 0.4; one per edge, with the edge flows), an edge at a
    // kink within stol, its jump within stol of 0 and its flow within stol of
    // its capacity, is joined for side -1 and kept apart for +1 with the
    // flow's sign (f_e = gamma w_e sign(z_u - z_v) across a jump).
    void structure(const double *d, const double *zv, double *deg, int32_t *uf, ImageCall &c,
                   const int8_t *side = nullptr, double stol = 0.0, const double *flow = nullptr) const noexcept {
        const View v = view(c.save);
        const double gamma = c.gamma;
        // Values are means of x plus boundary terms, within a few ulps of
        // max |x| + gamma (largest weighted degree).
        std::fill(deg, deg + n, 0.0);
        double scale = 0.0;
        for (int64_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(d[i]));
        for (int64_t e = 0; e < m; ++e) {
            const double cap = gamma * we(e);
            deg[eu[e]] += cap;
            deg[ev[e]] += cap;
        }
        double dmax = 0.0;
        for (int64_t i = 0; i < n; ++i) dmax = std::max(dmax, deg[i]);
        const double tau = 64.0 * kEps * (scale + dmax);
        auto forced = [&](int64_t e) {  // the side at a kink, else 0
            if (!side || side[e] == 0) return 0;
            const double cap = gamma * we(e);
            const bool kink = cap > 0.0 && std::fabs(zv[eu[e]] - zv[ev[e]]) <= stol && std::fabs(flow[e]) >= cap - stol;
            return kink ? static_cast<int>(side[e]) : 0;
        };
        for (int64_t i = 0; i < n; ++i) uf[i] = static_cast<int32_t>(i);
        for (int64_t e = 0; e < m; ++e) {
            const int f = forced(e);
            if (f < 0 || (f == 0 && gamma * we(e) > 0.0 && std::fabs(zv[eu[e]] - zv[ev[e]]) <= tau))
                unite(uf, eu[e], ev[e]);
        }
        int64_t nr = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t r = find(uf, static_cast<int32_t>(i));
            v.label[i] = r == i ? static_cast<int32_t>(nr++) : v.label[r];
        }
        std::fill(v.inv, v.inv + nr, 0.0);
        for (int64_t i = 0; i < n; ++i) v.inv[v.label[i]] += 1.0;
        for (int64_t R = 0; R < nr; ++R) v.inv[R] = 1.0 / v.inv[R];
        for (int64_t e = 0; e < m; ++e) {
            const int32_t a = v.label[eu[e]], bb = v.label[ev[e]];
            const double dz = zv[eu[e]] - zv[ev[e]];
            v.sgn[e] = a == bb ? 0 : (dz > 0.0 ? 1 : (dz < 0.0 ? -1 : 0));
            if (a != bb && forced(e) > 0) v.sgn[e] = flow[e] > 0.0 ? 1 : (flow[e] < 0.0 ? -1 : 0);
        }
        *v.nreg = nr;
    }

    // sum_e w_e |x_u - x_v| (since 0.5).
    bool has_value() const noexcept override { return true; }
    double value_image(const double *x, const int64_t *, double) const noexcept override {
        double s = 0.0;
        for (int64_t e = 0; e < m; ++e) s += we(e) * std::fabs(x[eu[e]] - x[ev[e]]);
        return s;
    }

    // Kinks (since 0.4) are edges (kink_sites, so masks and sides have the
    // shape (..., m)): code 1 at an edge with no jump, |z_u - z_v| <= tol,
    // whose flow f_e is within tol of its capacity gamma w_e > 0.
    int64_t kink_sites() const noexcept override { return m; }
    bool has_kinks() const noexcept override { return true; }
    bool has_branches() const noexcept override { return true; }
    int64_t kink_edges(const double *z, const double *f, double gamma, double tol, int8_t *mask,
                       int64_t ms) const noexcept {
        int64_t count = 0;
        for (int64_t e = 0; e < m; ++e) {
            const double cap = gamma * we(e);
            const int8_t k = cap > 0.0 && std::fabs(z[eu[e]] - z[ev[e]]) <= tol && std::fabs(f[e]) >= cap - tol;
            mask[e * ms] = k;
            count += k;
        }
        return count;
    }

    // dz_R = mean_R(dx) - (1/|R|) sum_{e in dR} (dgamma w_e + gamma dw_e) s_{e,R}.
    template <class T>
    void jvp_img(const char *save, const T *dx, T *dz, const int64_t *, double gamma, double dgamma,
                 const double *dparams, char *work, DerivCtl &) const noexcept {
        const View v = view(const_cast<char *>(save));
        const int64_t nr = *v.nreg;
        double *acc = reinterpret_cast<double *>(work);
        std::fill(acc, acc + nr, 0.0);
        for (int64_t i = 0; i < n; ++i) acc[v.label[i]] += static_cast<double>(dx[i]);
        for (int64_t e = 0; e < m; ++e) {
            if (!v.sgn[e]) continue;
            const double dwe = dparams ? dparams[w.size() == 1 ? 0 : e] : 0.0;
            const double t = (dgamma * we(e) + gamma * dwe) * v.sgn[e];
            acc[v.label[eu[e]]] -= t;
            acc[v.label[ev[e]]] += t;
        }
        for (int64_t R = 0; R < nr; ++R) acc[R] *= v.inv[R];
        for (int64_t i = 0; i < n; ++i) dz[i] = static_cast<T>(acc[v.label[i]]);
    }

    template <class T>
    void vjp_img(const char *save, const T *gin, T *gx, const int64_t *, double gamma, double *acc, char *work,
                 DerivCtl &) const noexcept {
        const View v = view(const_cast<char *>(save));
        const int64_t nr = *v.nreg;
        double *G = reinterpret_cast<double *>(work);
        std::fill(G, G + nr, 0.0);
        for (int64_t i = 0; i < n; ++i) G[v.label[i]] += static_cast<double>(gin[i]);
        for (int64_t R = 0; R < nr; ++R) G[R] *= v.inv[R];
        const int64_t P = nparams();
        for (int64_t e = 0; e < m; ++e) {
            if (!v.sgn[e]) continue;
            const double h = v.sgn[e] * (G[v.label[eu[e]]] - G[v.label[ev[e]]]);
            acc[w.size() == 1 ? 0 : e] -= gamma * h;
            acc[P] -= we(e) * h;
        }
        for (int64_t i = 0; i < n; ++i) gx[i] = static_cast<T>(G[v.label[i]]);
    }
};

}  // namespace edge_tv
}  // namespace fp
