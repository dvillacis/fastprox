// Anisotropic TV of 2D and 3D images: the prox of
//   gamma sum_a w_a TV_a(z),
// TV_a the sum of |z_{p + e_a} - z_p| along image axis a, by dual methods
// whose steps are exact 1D TV proxes along one axis (src/ops/tv1d.cpp) on
// every fiber of the image.
//
//  * Two active axes: the accelerated dual block proximal gradient method
//    (ADBPG) of A. Beck, First-Order Methods in Optimization, SIAM 2017,
//    Section 12.5.4 and Theorem 12.19, a form of the accelerated alternating
//    minimisation of A. Chambolle and T. Pock, "A remark on accelerated block
//    coordinate descent for computing the proximity operators of a sum of
//    convex functions", SMAI J. Comput. Math. 1 (2015) 29-54. With
//    f = |z - d|^2 / 2, g1 and g2 the two axis terms and a dual y:
//        u    = prox_{g1}(d + w)
//        y+   = w - u + prox_{g2}(u - w)
//        w    = y+ + (t_k - 1) / t_{k+1} (y+ - y),   t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
//    and the primal z = prox_{g1}(d + y), with |z - z*|^2 <= 4 |y0 - y*|^2 / (k+1)^2.
//  * Three active axes: the fast dual proximal gradient method for a sum of
//    p = 3 terms, Beck (2017) Algorithm 9 (Section 12.5.1): with u = d + sum w_i,
//        y_i+ = w_i - u / p + prox_{p g_i}(u - p w_i) / p,
//    then the same extrapolation. The primal returned is the block step
//    z = prox_{g_c}(d - sum_{i != c} q_i) on the contiguous axis c.
//
// Certificate. Every iterate carries duals q_a, each feasible for the
// conjugate of its axis term (q_a = D_a^T p_a with |p_a| <= gamma w_a, the
// exact 1D solves guarantee it), with d - z = sum_a q_a. The duality gap of
// the prox problem at (z, q) is then
//     gap = sum_a gamma w_a TV_a(z) - <z, d - z>   (>= 0),
// and |z - z*|^2 <= 2 gap (Beck, Lemma 12.7). An image stops when its gap is at
// most tol |d|^2 / 2. The dual returned is (q_0, ..., q_{k-1}).
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "core/parallel.hpp"
#include "core/workspace.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace fp {
namespace {

constexpr int64_t kBlock = 8;  // strided fibers gathered together
constexpr int kCheck = 4;      // iterations between gap checks

struct Geom {
    int k = 0;
    int64_t L[3] = {1, 1, 1};  // extents of the image axes
    int64_t S[3] = {1, 1, 1};  // strides (C order)
    int64_t N = 1;
    int64_t Lmax = 1;
    explicit Geom(int kk, const int64_t *shape) : k(kk) {
        for (int a = 0; a < k; ++a) L[a] = shape[a];
        S[k - 1] = 1;
        for (int a = k - 2; a >= 0; --a) S[a] = S[a + 1] * L[a + 1];
        N = S[0] * L[0];
        for (int a = 0; a < k; ++a) Lmax = std::max(Lmax, L[a]);
    }
    int64_t rows() const noexcept { return N / L[k - 1]; }
};

inline double fista_t(double t) noexcept { return 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t)); }

struct TVAniso final : ImageOp<TVAniso> {
    int k;
    double w[3];
    std::unique_ptr<fp_op> tv;  // 1D TV with unit weight; gamma carries the radius

    TVAniso(int kk, const double *ww, std::unique_ptr<fp_op> t) : k(kk), tv(std::move(t)) {
        for (int a = 0; a < 3; ++a) w[a] = a < k ? ww[a] : 0.0;
    }

    const char *name() const noexcept override { return "tv_aniso"; }
    int64_t nparams() const noexcept override { return k; }
    int image_ndim() const noexcept override { return k; }
    // Component labels are 32-bit.
    fp_status check_image(const int64_t *shape) const noexcept override {
        double n = 1.0;
        for (int a = 0; a < k; ++a) n *= static_cast<double>(shape[a]);
        return n < 2147483647.0 ? FP_OK : FP_ESHAPE;
    }

    // Working set: d, z, u, y_0..2, w_0..2 (doubles), and per-row partial sums.
    size_t image_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        return (9 * static_cast<size_t>(g.N) + static_cast<size_t>(g.rows()) + 8) * sizeof(double);
    }
    // Saved structure: the number of components, a component label per entry,
    // and per component 1/|C| and the boundary sign sums S_{C,a} (k of them).
    size_t struct_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        return 8 + round8(4 * static_cast<size_t>(g.N)) + (1 + static_cast<size_t>(k)) * static_cast<size_t>(g.N) * 8;
    }
    static size_t round8(size_t b) noexcept { return (b + 7) / 8 * 8; }
    struct View {
        int64_t *ncomp;
        int32_t *label;
        double *inv, *S;  // S[a * N + C]
    };
    View view(char *save, int64_t N) const noexcept {
        View v;
        v.ncomp = reinterpret_cast<int64_t *>(save);
        v.label = reinterpret_cast<int32_t *>(save + 8);
        v.inv = reinterpret_cast<double *>(save + 8 + round8(4 * static_cast<size_t>(N)));
        v.S = v.inv + N;
        return v;
    }

    size_t image_thread_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        return 2 * static_cast<size_t>(kBlock * g.Lmax) * sizeof(double) + 64 + tv->scratch_bytes(g.Lmax);
    }

    // ---- fiber passes ------------------------------------------------------

    // For every fiber along axis a: z = prox_{lam TV}(v) with v(o) = in(o) at
    // each entry o of the fiber, then out(o, z, v). lam == 0 is the identity.
    template <class In, class Out>
    void pass(const Geom &g, int a, double lam, const ImageCall &c, In &&in, Out &&out) const {
        const int64_t L = g.L[a], s = g.S[a];
        const int64_t nf = g.N / L;
        const int64_t outer = nf / s;
        const int64_t bpo = s == 1 ? 1 : (s + kBlock - 1) / kBlock;
        const int64_t nblk = outer * bpo;
        const bool ident = lam == 0.0 || L == 1;
        const int64_t width = s == 1 ? 1 : kBlock;
        const int64_t grain = std::max<int64_t>(1, kChunk / std::max<int64_t>(1, L * width));
        parallel_for(c.nt, nblk, grain, [&](int tid, int64_t b0, int64_t b1) {
            double *bin = reinterpret_cast<double *>(c.thread_scratch(tid));
            double *bout = bin + kBlock * g.Lmax;
            void *scr = bout + kBlock * g.Lmax;
            for (int64_t blk = b0; blk < b1; ++blk) {
                const int64_t oo = blk / bpo;
                const int64_t q0 = (blk % bpo) * kBlock;
                const int64_t W = s == 1 ? 1 : std::min<int64_t>(kBlock, s - q0);
                const int64_t base = oo * L * s + q0;
                for (int64_t i = 0; i < L; ++i)
                    for (int64_t q = 0; q < W; ++q) bin[q * L + i] = in(base + i * s + q);
                for (int64_t q = 0; q < W; ++q) {
                    if (ident) std::memcpy(bout + q * L, bin + q * L, static_cast<size_t>(L) * sizeof(double));
                    else tv->prox_row(bin + q * L, bout + q * L, L, lam, scr, nullptr);
                }
                for (int64_t i = 0; i < L; ++i)
                    for (int64_t q = 0; q < W; ++q) out(base + i * s + q, bout[q * L + i], bin[q * L + i]);
            }
        });
    }

    // sum over rows of f(r) for the rows of length L[k-1], in a fixed order.
    template <class F>
    double row_sum(const Geom &g, const ImageCall &c, double *part, F &&f) const {
        const int64_t R = g.rows(), n = g.L[g.k - 1];
        parallel_for(c.nt, R, std::max<int64_t>(1, kChunk / n), [&](int, int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) part[r] = f(r);
        });
        double s = 0.0;
        for (int64_t r = 0; r < R; ++r) s += part[r];
        return s;
    }

    // sum_a lam_a TV_a(z) - <z, d - z>.
    double gap(const Geom &g, const ImageCall &c, const double *lam, const double *d, const double *z,
               double *part) const {
        const int64_t n = g.L[g.k - 1];
        return row_sum(g, c, part, [&](int64_t r) {
            const int64_t base = r * n;
            double s = 0.0;
            const double ll = lam[g.k - 1];
            if (ll > 0.0)
                for (int64_t j = 0; j + 1 < n; ++j) s += ll * std::fabs(z[base + j + 1] - z[base + j]);
            for (int a = 0; a + 1 < g.k; ++a) {
                if (lam[a] == 0.0 || (base / g.S[a]) % g.L[a] == g.L[a] - 1) continue;
                const double *z1 = z + base + g.S[a];
                double t = 0.0;
                for (int64_t j = 0; j < n; ++j) t += std::fabs(z1[j] - z[base + j]);
                s += lam[a] * t;
            }
            for (int64_t j = 0; j < n; ++j) s -= z[base + j] * (d[base + j] - z[base + j]);
            return s;
        });
    }

    // ---- prox --------------------------------------------------------------

    template <class T>
    int prox_img(const T *x, T *z, ImageCall &c) const noexcept {
        const Geom g(k, c.shape);
        const int64_t N = g.N;
        double *base = reinterpret_cast<double *>(c.work);
        double *d = base, *zz = d + N, *u = zz + N, *y = u + N, *wv = y + 3 * N;
        double *part = wv + 3 * N;
        c.gap = 0.0;
        c.iters = 0;
        c.converged = true;

        bool nan = false;
        for (int64_t i = 0; i < N; ++i) {
            const double v = static_cast<double>(x[i]);
            if (std::isinf(v)) return kRowNonFinite;
            nan = nan || std::isnan(v);
            d[i] = v;
        }
        double *q = c.dual;
        if (nan) {  // NaN anywhere in the image: the whole image is NaN
            for (int64_t i = 0; i < N; ++i) z[i] = std::numeric_limits<T>::quiet_NaN();
            std::fill(q, q + k * N, std::numeric_limits<double>::quiet_NaN());
            if (c.save) {  // singleton components, so that jvp/vjp stay defined
                const double none[3] = {0.0, 0.0, 0.0};
                structure(g, c, none, -1, false, d, d, u, y, part);
            }
            return kRowOk;
        }

        double lam[3] = {0.0, 0.0, 0.0};
        int act[3], na = 0;
        for (int a = 0; a < k; ++a) {
            lam[a] = c.gamma * w[a];
            if (lam[a] > 0.0 && g.L[a] > 1) act[na++] = a;
            else lam[a] = 0.0;
        }
        // Inactive axes carry a zero dual (the warm start reads the active ones).
        for (int a = 0; a < k; ++a)
            if (lam[a] == 0.0 || na < 2) std::fill(q + a * N, q + (a + 1) * N, 0.0);

        if (na == 0) {
            for (int64_t i = 0; i < N; ++i) z[i] = x[i];
            if (c.save) structure(g, c, lam, -1, false, d, d, u, y, part);
            return kRowOk;
        }
        if (na == 1) {  // exact
            const int a = act[0];
            double *qa = q + a * N;
            pass(g, a, lam[a], c, [&](int64_t o) { return d[o]; },
                 [&](int64_t o, double zo, double v) { zz[o] = zo; qa[o] = v - zo; });
            if (c.save) structure(g, c, lam, a, false, d, zz, u, y, part);
            for (int64_t i = 0; i < N; ++i) z[i] = static_cast<T>(zz[i]);
            return kRowOk;
        }

        const double dn2 = row_sum(g, c, part, [&](int64_t r) {
            const int64_t n = g.L[k - 1];
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += d[r * n + j] * d[r * n + j];
            return s;
        });
        const double target = c.tol * 0.5 * dn2;
        int exact_axis;
        if (na == 2) {
            adbpg(g, c, act, lam, d, zz, u, y, wv, part, target);
            exact_axis = g.S[act[0]] < g.S[act[1]] ? act[0] : act[1];
        } else {
            fdpg3(g, c, lam, d, zz, u, y, wv, part, target);
            exact_axis = 2;
        }
        structure(g, c, lam, exact_axis, true, d, zz, u, y, part);
        for (int64_t i = 0; i < N; ++i) z[i] = static_cast<T>(zz[i]);
        return kRowOk;
    }

    // Two active axes a1 (primal axis, the one with the smaller stride) and a2.
    void adbpg(const Geom &g, ImageCall &c, const int *act, const double *lam, const double *d,
               double *z, double *u, double *y, double *wv, double *part, double target) const {
        const int64_t N = g.N;
        const int a1 = g.S[act[0]] < g.S[act[1]] ? act[0] : act[1];
        const int a2 = a1 == act[0] ? act[1] : act[0];
        double *q1 = c.dual + a1 * N, *q2 = c.dual + a2 * N;
        // Dual of the a2 term: y = -q2.
        if (c.warm)
            for (int64_t i = 0; i < N; ++i) y[i] = -q2[i];
        else
            std::fill(y, y + N, 0.0);
        std::memcpy(wv, y, static_cast<size_t>(N) * sizeof(double));

        // z = prox_{g1}(d + y), and the gap of (z, q1 = d + y - z, q2 = -y).
        auto primal = [&]() {
            pass(g, a1, lam[a1], c, [&](int64_t o) { return d[o] + y[o]; },
                 [&](int64_t o, double zo, double) { z[o] = zo; });
            return gap(g, c, lam, d, z, part);
        };
        double t = 1.0;
        int it = 0;
        double gp = c.warm ? primal() : std::numeric_limits<double>::infinity();
        while (!(gp <= target) && it < c.max_iter) {
            pass(g, a1, lam[a1], c, [&](int64_t o) { return d[o] + wv[o]; },
                 [&](int64_t o, double zo, double) { u[o] = zo; });
            const double tn = fista_t(t);
            const double beta = (t - 1.0) / tn;
            t = tn;
            pass(g, a2, lam[a2], c, [&](int64_t o) { return u[o] - wv[o]; },
                 [&](int64_t o, double zo, double v) {
                     const double yn = zo - v;
                     wv[o] = yn + beta * (yn - y[o]);
                     y[o] = yn;
                 });
            ++it;
            if (it % kCheck == 0 || it == c.max_iter) gp = primal();
        }
        if (!std::isfinite(gp)) gp = primal();
        for (int64_t i = 0; i < N; ++i) {
            q1[i] = d[i] + y[i] - z[i];
            q2[i] = -y[i];
        }
        c.gap = std::max(gp, 0.0);
        c.iters = it;
        c.converged = gp <= target;
    }

    // Three active axes (k == 3). y holds y_0..2 (y_i = -q_i), wv the
    // extrapolated w_0..2.
    void fdpg3(const Geom &g, ImageCall &c, const double *lam, const double *d, double *z, double *u,
               double *y, double *wv, double *part, double target) const {
        const int64_t N = g.N;
        constexpr double p = 3.0;
        const int ac = 2;  // contiguous axis, used for the primal block step
        if (c.warm)
            for (int64_t i = 0; i < 3 * N; ++i) y[i] = -c.dual[i];
        else
            std::fill(y, y + 3 * N, 0.0);
        std::memcpy(wv, y, static_cast<size_t>(3 * N) * sizeof(double));

        // z = prox_{g_c}(d - q_a - q_b) = prox_{g_c}(d + y_a + y_b) for the
        // other two axes, then the gap.
        auto primal = [&]() {
            const double *ya = y, *yb = y + N;
            pass(g, ac, lam[ac], c, [&](int64_t o) { return d[o] + ya[o] + yb[o]; },
                 [&](int64_t o, double zo, double) { z[o] = zo; });
            return gap(g, c, lam, d, z, part);
        };
        double t = 1.0;
        int it = 0;
        double gp = c.warm ? primal() : std::numeric_limits<double>::infinity();
        while (!(gp <= target) && it < c.max_iter) {
            for (int64_t i = 0; i < N; ++i) u[i] = d[i] + wv[i] + wv[N + i] + wv[2 * N + i];
            const double tn = fista_t(t);
            const double beta = (t - 1.0) / tn;
            t = tn;
            for (int a = 0; a < 3; ++a) {
                double *ya = y + a * N, *wa = wv + a * N;
                pass(g, a, p * lam[a], c, [&](int64_t o) { return u[o] - p * wa[o]; },
                     [&](int64_t o, double zo, double v) {
                         // y+ = w - u/p + zo/p with v = u - p w, i.e. (zo - v) / p.
                         const double yn = (zo - v) / p;
                         wa[o] = yn + beta * (yn - ya[o]);
                         ya[o] = yn;
                     });
            }
            ++it;
            if (it % kCheck == 0 || it == c.max_iter) gp = primal();
        }
        if (!std::isfinite(gp)) gp = primal();
        double *q = c.dual;
        for (int64_t i = 0; i < N; ++i) {
            q[i] = -y[i];
            q[N + i] = -y[N + i];
            q[2 * N + i] = d[i] + y[i] + y[N + i] - z[i];
        }
        c.gap = std::max(gp, 0.0);
        c.iters = it;
        c.converged = gp <= target;
    }

    // ---- structure ----------------------------------------------------------

    // Components of the graph of flat edges of z (|z_j - z_i| <= tau along an
    // active axis; exactly equal along exact_axis, where z is piecewise
    // constant by construction), the boundary sign sums S_{C,a} on every axis,
    // and, with polish, the closed form
    //     z_C = (sum_C d - sum_a lam_a S_{C,a}) / |C|
    // kept when its gap with the same dual is no larger. Saved for jvp/vjp
    // when c.save. Uses u and the y/wv region as scratch (the dual is already
    // written out).
    void structure(const Geom &g, ImageCall &c, const double *lam, int exact_axis, bool polish,
                   const double *d, double *z, double *u, double *scratch, double *part) const {
        const int64_t N = g.N;
        View v{};
        int32_t *label;
        double *inv, *S;
        if (c.save) {
            v = view(c.save, N);
            label = v.label;
            inv = v.inv;
            S = v.S;
        } else {
            label = reinterpret_cast<int32_t *>(scratch);
            inv = scratch + N;
            S = inv + N;  // k * N <= 3N fits in y/wv (6N)
        }
        int32_t *parent = reinterpret_cast<int32_t *>(u);
        double *sumd = scratch + (1 + 3) * N + N;  // after label/inv/S in the 6N region

        // Union-find, roots at the smallest index (deterministic labels).
        for (int64_t i = 0; i < N; ++i) parent[i] = static_cast<int32_t>(i);
        auto find = [&](int64_t i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        double scale = 0.0;
        for (int64_t i = 0; i < N; ++i) scale = std::max(scale, std::fabs(d[i]));
        const double rms = std::sqrt(2.0 * std::max(c.gap, 0.0) / static_cast<double>(N));
        for (int a = 0; a < g.k; ++a) {
            if (lam[a] == 0.0) continue;
            const double tau = a == exact_axis ? 0.0 : 8.0 * rms + 64.0 * 2.2e-16 * scale;
            const int64_t s = g.S[a], L = g.L[a];
            for (int64_t i = 0; i < N; ++i) {
                if ((i / s) % L == L - 1) continue;
                if (!(std::fabs(z[i + s] - z[i]) <= tau)) continue;
                int64_t ri = find(i), rj = find(i + s);
                if (ri == rj) continue;
                if (ri < rj) parent[rj] = static_cast<int32_t>(ri);
                else parent[ri] = static_cast<int32_t>(rj);
            }
        }
        int64_t nc = 0;
        for (int64_t i = 0; i < N; ++i) {
            const int64_t r = find(i);
            label[i] = r == i ? static_cast<int32_t>(nc++) : label[r];
        }
        std::fill(inv, inv + nc, 0.0);
        std::fill(S, S + g.k * N, 0.0);
        std::fill(sumd, sumd + nc, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            inv[label[i]] += 1.0;
            sumd[label[i]] += d[i];
        }
        for (int a = 0; a < g.k; ++a) {
            const int64_t s = g.S[a], L = g.L[a];
            if (L == 1) continue;
            double *Sa = S + a * N;
            for (int64_t i = 0; i < N; ++i) {
                if ((i / s) % L == L - 1) continue;
                const int32_t ci = label[i], cj = label[i + s];
                if (ci == cj) continue;
                const double diff = z[i] - z[i + s];
                const double sg = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
                Sa[ci] += sg;
                Sa[cj] -= sg;
            }
        }
        for (int64_t C = 0; C < nc; ++C) inv[C] = 1.0 / inv[C];
        if (c.save) *v.ncomp = nc;
        if (!polish) return;

        // Polished point, into u (parent is no longer needed).
        double *zp = u;
        for (int64_t i = 0; i < N; ++i) {
            const int32_t C = label[i];
            double t = sumd[C];
            for (int a = 0; a < g.k; ++a) t -= lam[a] * S[a * N + C];
            zp[i] = t * inv[C];
        }
        // gap(zp, q) = P(zp) - D(q), with s = sum_a q_a = d - z and
        // D(q) = <d, s> - |s|^2 / 2.
        const int64_t n = g.L[g.k - 1];
        const double pz = row_sum(g, c, part, [&](int64_t r) {
            const int64_t b = r * n;
            double acc = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                const double e = zp[b + j] - d[b + j];
                const double sj = d[b + j] - z[b + j];
                acc += 0.5 * e * e - (d[b + j] * sj - 0.5 * sj * sj);
            }
            if (lam[g.k - 1] > 0.0)
                for (int64_t j = 0; j + 1 < n; ++j) acc += lam[g.k - 1] * std::fabs(zp[b + j + 1] - zp[b + j]);
            for (int a = 0; a + 1 < g.k; ++a) {
                if (lam[a] == 0.0 || (b / g.S[a]) % g.L[a] == g.L[a] - 1) continue;
                double t = 0.0;
                for (int64_t j = 0; j < n; ++j) t += std::fabs(zp[b + j + g.S[a]] - zp[b + j]);
                acc += lam[a] * t;
            }
            return acc;
        });
        if (pz <= c.gap) {
            std::memcpy(z, zp, static_cast<size_t>(N) * sizeof(double));
            c.gap = std::max(pz, 0.0);
        }
    }

    // ---- derivatives (structure saved by prox_img) --------------------------

    // dz_C = mean_C(dx) - (1/|C|) sum_a (dgamma w_a + gamma dw_a) S_{C,a}.
    template <class T>
    void jvp_img(const char *save, const T *dx, T *dz, const int64_t *shape, double gamma, double dgamma,
                 const double *dparams, char *work) const noexcept {
        const Geom g(k, shape);
        const int64_t N = g.N;
        const View v = view(const_cast<char *>(save), N);
        const int64_t nc = *v.ncomp;
        double *accc = reinterpret_cast<double *>(work);
        std::fill(accc, accc + nc, 0.0);
        for (int64_t i = 0; i < N; ++i) accc[v.label[i]] += static_cast<double>(dx[i]);
        for (int64_t C = 0; C < nc; ++C) {
            double t = 0.0;
            for (int a = 0; a < k; ++a)
                t += (dgamma * w[a] + (dparams ? gamma * dparams[a] : 0.0)) * v.S[a * N + C];
            accc[C] = (accc[C] - t) * v.inv[C];
        }
        for (int64_t i = 0; i < N; ++i) dz[i] = static_cast<T>(accc[v.label[i]]);
    }

    template <class T>
    void vjp_img(const char *save, const T *gin, T *gx, const int64_t *shape, double gamma, double *acc,
                 char *work) const noexcept {
        const Geom g(k, shape);
        const int64_t N = g.N;
        const View v = view(const_cast<char *>(save), N);
        const int64_t nc = *v.ncomp;
        double *G = reinterpret_cast<double *>(work);
        std::fill(G, G + nc, 0.0);
        for (int64_t i = 0; i < N; ++i) G[v.label[i]] += static_cast<double>(gin[i]);
        const int64_t P = k;
        for (int64_t C = 0; C < nc; ++C) {
            const double gc = G[C] * v.inv[C];
            for (int a = 0; a < k; ++a) {
                acc[a] -= gamma * v.S[a * N + C] * gc;
                acc[P] -= w[a] * v.S[a * N + C] * gc;
            }
            G[C] = gc;
        }
        for (int64_t i = 0; i < N; ++i) gx[i] = static_cast<T>(G[v.label[i]]);
    }
};

}  // namespace

fp_status make_tv_aniso(int ndim, Vec w, fp_op **out) {
    if (ndim != 2 && ndim != 3) return FP_EPARAM;
    double ww[3] = {1.0, 1.0, 1.0};
    if (w.size() == 1) {
        for (int a = 0; a < ndim; ++a) ww[a] = w[0];
    } else if (!w.empty()) {
        if (static_cast<int>(w.size()) != ndim) return FP_ESHAPE;
        for (int a = 0; a < ndim; ++a) ww[a] = w[static_cast<size_t>(a)];
    }
    for (int a = 0; a < ndim; ++a)
        if (!finite_nonneg(ww[a])) return FP_EPARAM;
    fp_op *t = nullptr;
    if (fp_status s = make_tv1d(Vec{1.0}, FP_TV1D_AUTO, &t)) return s;
    std::unique_ptr<fp_op> tv(t);
    *out = new TVAniso(ndim, ww, std::move(tv));
    return FP_OK;
}

}  // namespace fp
