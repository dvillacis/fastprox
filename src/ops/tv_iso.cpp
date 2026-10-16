// Isotropic TV of 2D and 3D images: the prox of
//   gamma sum_p | (w_0 D_0 z, ..., w_{k-1} D_{k-1} z)_p |_2
// with forward differences D_a along the last k axes (zero past the last
// entry), by the fast dual proximal gradient method (FISTA on the dual):
// A. Beck and M. Teboulle, "Fast gradient-based algorithms for constrained
// total variation image denoising and deblurring problems", IEEE Trans. Image
// Process. 18 (2009) 2419-2434, after A. Chambolle, J. Math. Imaging Vis. 20
// (2004) 89-97; in the form of A. Beck, First-Order Methods in Optimization,
// SIAM 2017, Section 12.4.4 and Theorem 12.10. With the dual u (|u_p| <= gamma)
// and z(u) = d - sum_a w_a D_a^T u_a:
//     z   = z(v)
//     u+  = proj_{|.| <= gamma}(v + tau W D z)       (pixelwise)
//     v   = u+ + (t_k - 1) / t_{k+1} (u+ - u)
// with tau = 1 / L, L = |W D|^2 <= 4 sum_a w_a^2 (8 in 2D for unit weights,
// 12 in 3D), and the primal z(u) from the last dual iterate, not the
// extrapolated one (Beck, (12.19)).
//
// Certificate: for feasible u, the duality gap of the prox problem at z(u) is
//     sum_p ( gamma |g_p| - <g_p, u_p> ),   g = W D z(u),
// every term >= 0, and |z - z*|^2 <= 2 gap (Beck, Lemma 12.7).
//
// Derivatives by implicit differentiation of the optimality conditions
// z - d + D^T W u = 0, u_p = gamma g_p / |g_p| where g_p != 0. Entries with
// g_p ~ 0 (flat) keep z constant across their edges, so dz lies in the space
// of images constant on the connected flat regions (one unknown per region);
// elsewhere u_p moves along the tangent of the sphere, du_p = H_p (W D dz)_p
// with H_p = gamma / |g_p| (I - n_p n_p^T), n_p = g_p / |g_p|. Then
//     (I + G_A^T H G_A) dz = rhs  on that space  (G = W D, A the non-flat set),
// a symmetric positive definite system solved by conjugate gradients.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "core/parallel.hpp"
#include "core/workspace.hpp"
#include "ops/image_util.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace fp {
namespace {

constexpr int kCheck = 4;  // iterations between gap checks

struct TVIso final : ImageOp<TVIso> {
    int k;
    double w[3];
    std::unique_ptr<fp_op> tv;  // exact 1D TV, for a single active axis

    TVIso(int kk, const double *ww, std::unique_ptr<fp_op> t) : k(kk), tv(std::move(t)) {
        for (int a = 0; a < 3; ++a) w[a] = a < k ? ww[a] : 0.0;
    }

    const char *name() const noexcept override { return "tv_iso"; }
    int64_t nparams() const noexcept override { return k; }
    int image_ndim() const noexcept override { return k; }
    fp_status check_image(const int64_t *shape) const noexcept override {
        double n = 1.0;
        for (int a = 0; a < k; ++a) n *= static_cast<double>(shape[a]);
        return n < 2147483647.0 ? FP_OK : FP_ESHAPE;  // 32-bit region labels
    }

    // prox: d, z (N), u, v (k N), per-row sums; derivatives: the gradient of z
    // and a scratch field (k N each), then seven vectors of N (image, rhs and
    // the CG vectors in region space).
    size_t image_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        const size_t N = static_cast<size_t>(g.N), K = static_cast<size_t>(k);
        return (std::max((2 + 2 * K) * N, (7 + 2 * K) * N) + static_cast<size_t>(g.rows()) + 8) * sizeof(double);
    }
    size_t image_thread_bytes(const int64_t *shape) const noexcept override {
        return fiber_thread_bytes(*tv, Geom(k, shape));
    }
    // Saved: number of regions, labels, flat flags, z, u (k blocks), 1/|C|.
    static size_t r8(size_t b) noexcept { return (b + 7) / 8 * 8; }
    size_t struct_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        const size_t N = static_cast<size_t>(g.N);
        return 8 + r8(4 * N) + r8(N) + (2 + static_cast<size_t>(k)) * N * 8;
    }
    struct View {
        int64_t *ncomp;
        int32_t *label;
        uint8_t *flat;
        double *z, *u, *inv;
    };
    View view(char *s, int64_t N) const noexcept {
        View v;
        const size_t n = static_cast<size_t>(N);
        v.ncomp = reinterpret_cast<int64_t *>(s);
        v.label = reinterpret_cast<int32_t *>(s + 8);
        v.flat = reinterpret_cast<uint8_t *>(s + 8 + r8(4 * n));
        v.z = reinterpret_cast<double *>(s + 8 + r8(4 * n) + r8(n));
        v.u = v.z + N;
        v.inv = v.u + k * N;
        return v;
    }

    // ---- passes ---------------------------------------------------------------

    // z = d - sum_a w_a D_a^T u_a = d + sum_a w_a (u_a[p] - u_a[p - e_a]),
    // with u_a = 0 at the last entry along a.
    void divergence(const Geom &g, const ImageCall &c, const double *wa, const double *d, const double *u,
                    double *z) const {
        const int64_t n = g.L[k - 1], N = g.N;
        row_for(g, c, [&](int64_t r) {
            const int64_t b = r * n;
            for (int64_t j = 0; j < n; ++j) z[b + j] = d[b + j];
            for (int a = 0; a < k; ++a) {
                if (wa[a] == 0.0) continue;
                const double *ua = u + a * N;
                if (a == k - 1) {
                    for (int64_t j = 0; j < n; ++j)
                        z[b + j] += wa[a] * (ua[b + j] - (j > 0 ? ua[b + j - 1] : 0.0));
                } else {
                    const bool prev = g.coord(b, a) > 0;
                    const double *up = ua + b - g.S[a];
                    for (int64_t j = 0; j < n; ++j) z[b + j] += wa[a] * (ua[b + j] - (prev ? up[j] : 0.0));
                }
            }
        });
    }

    // f(r, j, gvec) with gvec[a] = w_a (z[p + e_a] - z[p]) (0 past the end) for
    // every entry p = r n + j of row r.
    template <class F>
    void grad_rows(const Geom &g, const double *wa, const double *z, int64_t r, F &&f) const {
        const int64_t n = g.L[k - 1], b = r * n;
        bool nxt[3];
        for (int a = 0; a + 1 < k; ++a) nxt[a] = g.has_next(b, a);
        double gv[3] = {0.0, 0.0, 0.0};
        for (int64_t j = 0; j < n; ++j) {
            const int64_t p = b + j;
            for (int a = 0; a < k; ++a) {
                const bool has = a == k - 1 ? j + 1 < n : nxt[a];
                gv[a] = (wa[a] != 0.0 && has) ? wa[a] * (z[p + g.S[a]] - z[p]) : 0.0;
            }
            f(p, gv);
        }
    }

    double gap(const Geom &g, const ImageCall &c, const double *wa, double gamma, const double *z,
               const double *u, double *part) const {
        const int64_t N = g.N;
        return row_sum(g, c, part, [&](int64_t r) {
            double s = 0.0;
            grad_rows(g, wa, z, r, [&](int64_t p, const double *gv) {
                double nn = 0.0, dot = 0.0;
                for (int a = 0; a < k; ++a) {
                    nn += gv[a] * gv[a];
                    dot += gv[a] * u[a * N + p];
                }
                s += gamma * std::sqrt(nn) - dot;
            });
            return s;
        });
    }

    // ---- prox -------------------------------------------------------------------

    template <class T>
    int prox_img(const T *x, T *zout, ImageCall &c) const noexcept {
        const Geom g(k, c.shape);
        const int64_t N = g.N;
        double *d = reinterpret_cast<double *>(c.work);
        double *z = d + N, *u = z + N, *v = u + k * N, *part = v + k * N;
        c.gap = 0.0;
        c.iters = 0;
        c.converged = true;
        bool nan = false;
        for (int64_t i = 0; i < N; ++i) {
            const double t = static_cast<double>(x[i]);
            if (std::isinf(t)) return kRowNonFinite;
            nan = nan || std::isnan(t);
            d[i] = t;
        }
        // Active axes: positive weight, extent > 1, and not negligible next to
        // the largest weight (ratios below 1e-100, as for the weighted l1 ball).
        double wa[3] = {0.0, 0.0, 0.0}, wmax = 0.0;
        for (int a = 0; a < k; ++a)
            if (g.L[a] > 1) wmax = std::max(wmax, w[a]);
        int act[3], na = 0;
        for (int a = 0; a < k; ++a)
            if (c.gamma > 0.0 && w[a] > 0.0 && g.L[a] > 1 && w[a] >= 1e-100 * wmax) {
                wa[a] = w[a];
                act[na++] = a;
            }
        double *q = c.dual;
        if (nan) {
            for (int64_t i = 0; i < N; ++i) zout[i] = std::numeric_limits<T>::quiet_NaN();
            std::fill(q, q + k * N, std::numeric_limits<double>::quiet_NaN());
            if (c.save) save(g, c, wa, d, d, nullptr, 0.0);
            return kRowOk;
        }
        if (na == 0) {
            std::memcpy(z, d, static_cast<size_t>(N) * sizeof(double));
            std::fill(q, q + k * N, 0.0);
        } else if (na == 1) {  // exactly 1D TV along the active axis
            const int a = act[0];
            const double lam = c.gamma * w[a];
            std::fill(q, q + k * N, 0.0);
            double *qa = q + a * N;
            fiber_pass(*tv, g, a, lam, c, [&](int64_t o) { return d[o]; },
                       [&](int64_t o, double zo, double) { z[o] = zo; });
            // u_a[p] = -(1/w_a) sum_{i <= p} (d - z) along the fiber; zero at its end.
            const int64_t L = g.L[a], s = g.S[a];
            for (int64_t st = 0; st < N; ++st) {
                if (g.coord(st, a) != 0) continue;
                double cs = 0.0;
                for (int64_t i = 0; i < L; ++i) {
                    const int64_t o = st + i * s;
                    cs += d[o] - z[o];
                    qa[o] = i + 1 < L ? -cs / w[a] : 0.0;
                }
            }
        } else {
            fdpg(g, c, wa, d, z, u, v, part);
        }
        if (c.save) save(g, c, wa, d, z, na >= 2 ? c.dual : (na == 1 ? c.dual : nullptr), na == 1 ? 0.0 : -1.0);
        for (int64_t i = 0; i < N; ++i) zout[i] = static_cast<T>(z[i]);
        return kRowOk;
    }

    // Runs on the rescaled problem: |(w_a D_a z)| = wmax |(wh_a D_a z)| with
    // wh = w / wmax in [0, 1], so the radius is gamma wmax and the dual
    // uh = wmax u; the step 1/L stays in [1/12, 1/4] for any weights, and the
    // gap is the same in both scalings. The radius is clamped where the
    // solution is the constant mean anyway: a spanning-tree flow carries at
    // most the total imbalance sum |d - mean|, so a radius of that over the
    // smallest active weight is never binding beyond it.
    void fdpg(const Geom &g, ImageCall &c, const double *wa_in, const double *d, double *z, double *u, double *v,
              double *part) const {
        const int64_t N = g.N, n = g.L[k - 1];
        double wmax = 0.0, whmin = 1.0;
        for (int a = 0; a < k; ++a) wmax = std::max(wmax, wa_in[a]);
        double wa[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < k; ++a)
            if (wa_in[a] != 0.0) {
                wa[a] = wa_in[a] / wmax;
                whmin = std::min(whmin, wa[a]);
            }
        const double mean = row_sum(g, c, part, [&](int64_t r) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += d[r * n + j];
            return s;
        }) / static_cast<double>(N);
        const double imbalance = row_sum(g, c, part, [&](int64_t r) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += std::fabs(d[r * n + j] - mean);
            return s;
        });
        double gamma = c.gamma * wmax;
        const double bound = 2.0 * imbalance / whmin + std::numeric_limits<double>::min();
        if (!(gamma <= bound)) gamma = bound;  // also catches gamma * wmax = inf
        double L2 = 0.0;
        for (int a = 0; a < k; ++a) L2 += 4.0 * wa[a] * wa[a];
        const double tau = 1.0 / L2;
        // Start: zero, or the given dual (rescaled) projected onto the
        // current ball (inactive components and the last entry along each
        // axis are zero).
        if (c.warm)
            for (int64_t i = 0; i < k * N; ++i) u[i] = c.dual[i] * wmax;
        else
            std::fill(u, u + k * N, 0.0);
        row_for(g, c, [&](int64_t r) {
            const int64_t b = r * n;
            for (int64_t j = 0; j < n; ++j) {
                const int64_t p = b + j;
                double nn = 0.0;
                for (int a = 0; a < k; ++a) {
                    const bool has = a == k - 1 ? j + 1 < n : g.has_next(b, a);
                    double &ua = u[a * N + p];
                    if (wa[a] == 0.0 || !has || !std::isfinite(ua)) ua = 0.0;
                    nn += ua * ua;
                }
                if (std::sqrt(nn) > gamma) {  // not nn > gamma^2: gamma may be ~1e300
                    const double sc = gamma / std::sqrt(nn);
                    for (int a = 0; a < k; ++a) u[a * N + p] *= sc;
                }
            }
        });
        std::memcpy(v, u, static_cast<size_t>(k * N) * sizeof(double));
        const double target = c.tol * 0.5 * row_sum(g, c, part, [&](int64_t r) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += d[r * n + j] * d[r * n + j];
            return s;
        });
        auto primal = [&]() {
            divergence(g, c, wa, d, u, z);
            return gap(g, c, wa, gamma, z, u, part);
        };
        double t = 1.0;
        int it = 0;
        double gp = c.warm ? primal() : std::numeric_limits<double>::infinity();
        while (!(gp <= target) && it < c.max_iter) {
            divergence(g, c, wa, d, v, z);
            const double tn = fista_t(t);
            const double beta = (t - 1.0) / tn;
            t = tn;
            row_for(g, c, [&](int64_t r) {
                grad_rows(g, wa, z, r, [&](int64_t p, const double *gv) {
                    double cv[3], nn = 0.0;
                    // v is zero past the last entry along each axis and gv is
                    // too, so the dual stays zero there.
                    for (int a = 0; a < k; ++a) {
                        cv[a] = wa[a] != 0.0 ? v[a * N + p] + tau * gv[a] : 0.0;
                        nn += cv[a] * cv[a];
                    }
                    const double nr = std::sqrt(nn);
                    const double sc = nr > gamma ? gamma / nr : 1.0;
                    for (int a = 0; a < k; ++a) {
                        const double un = cv[a] * sc;
                        double &ua = u[a * N + p];
                        v[a * N + p] = un + beta * (un - ua);
                        ua = un;
                    }
                });
            });
            ++it;
            if (it % kCheck == 0 || it == c.max_iter) gp = primal();
        }
        if (!std::isfinite(gp)) gp = primal();
        for (int64_t i = 0; i < k * N; ++i) c.dual[i] = u[i] / wmax;
        c.gap = std::max(gp, 0.0);
        c.iters = it;
        c.converged = gp <= target;
    }

    // ---- structure ----------------------------------------------------------------

    // Flat entries (|g_p| <= tau) and the regions they connect; z and u for
    // the derivative system. dual null: u = 0. tau_fixed >= 0 overrides the
    // threshold (0: z is exactly piecewise constant).
    void save(const Geom &g, ImageCall &c, const double *wa, const double *d, const double *z, const double *dual,
              double tau_fixed) const {
        const int64_t N = g.N;
        View s = view(c.save, N);
        std::memcpy(s.z, z, static_cast<size_t>(N) * sizeof(double));
        if (dual) std::memcpy(s.u, dual, static_cast<size_t>(k * N) * sizeof(double));
        else std::fill(s.u, s.u + k * N, 0.0);
        double scale = 0.0, wmax = 0.0;
        for (int64_t i = 0; i < N; ++i) scale = std::max(scale, std::fabs(d[i]));
        for (int a = 0; a < k; ++a) wmax = std::max(wmax, wa[a]);
        const double rms = std::sqrt(2.0 * std::max(c.gap, 0.0) / static_cast<double>(N));
        const double thr = tau_fixed >= 0.0 ? tau_fixed : wmax * (8.0 * rms + 64.0 * 2.2e-16 * scale);
        // Union-find over the edges of flat entries, roots at the smallest index.
        int32_t *parent = s.label;  // relabelled below, after a copy of the roots
        for (int64_t i = 0; i < N; ++i) parent[i] = static_cast<int32_t>(i);
        auto find = [&](int64_t i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        for (int64_t p = 0; p < N; ++p) {
            double nn = 0.0;
            bool any = false;
            for (int a = 0; a < k; ++a) {
                if (wa[a] == 0.0 || !g.has_next(p, a)) continue;
                const double t = wa[a] * (z[p + g.S[a]] - z[p]);
                nn += t * t;
                any = true;
            }
            const bool flat = !any || std::sqrt(nn) <= thr;
            s.flat[p] = flat ? 1 : 0;
            if (!flat || !any) continue;
            for (int a = 0; a < k; ++a) {
                if (wa[a] == 0.0 || !g.has_next(p, a)) continue;
                int64_t ri = find(p), rj = find(p + g.S[a]);
                if (ri == rj) continue;
                if (ri < rj) parent[rj] = static_cast<int32_t>(ri);
                else parent[ri] = static_cast<int32_t>(rj);
            }
        }
        // Roots to labels: compress fully, then number the roots in order.
        for (int64_t i = 0; i < N; ++i) parent[i] = static_cast<int32_t>(find(i));
        double *idmap = s.inv;  // root index -> label, reused for 1/|C| below
        int64_t nc = 0;
        for (int64_t i = 0; i < N; ++i)
            if (parent[i] == i) idmap[i] = static_cast<double>(nc++);
        for (int64_t i = 0; i < N; ++i) parent[i] = static_cast<int32_t>(idmap[parent[i]]);
        std::fill(s.inv, s.inv + nc, 0.0);
        for (int64_t i = 0; i < N; ++i) s.inv[s.label[i]] += 1.0;
        for (int64_t C = 0; C < nc; ++C) s.inv[C] = 1.0 / s.inv[C];
        *s.ncomp = nc;
    }

    // ---- derivatives ----------------------------------------------------------------

    struct Sys {
        const TVIso &op;
        const Geom &g;
        View s;
        double gamma;
        double wa[3];
        double *gz;   // k N: W D z at each entry
        double *pix;  // N
        double *hbuf; // k N scratch
        int64_t nc;

        // y_p = (G x)_p, the scaled forward differences of the image x.
        void grad(const double *x, double *y) const {
            const int64_t N = g.N;
            for (int a = 0; a < op.k; ++a) {
                double *ya = y + a * N;
                for (int64_t p = 0; p < N; ++p)
                    ya[p] = (wa[a] != 0.0 && g.has_next(p, a)) ? wa[a] * (x[p + g.S[a]] - x[p]) : 0.0;
            }
        }
        // x -= G^T h (h zero past the ends), i.e. x += sum_a w_a (h_a[p] - h_a[p - e_a]).
        void sub_gradT(const double *h, double *x) const {
            const int64_t N = g.N;
            for (int a = 0; a < op.k; ++a) {
                if (wa[a] == 0.0) continue;
                const double *ha = h + a * N;
                for (int64_t p = 0; p < N; ++p) x[p] += wa[a] * (ha[p] - (g.coord(p, a) > 0 ? ha[p - g.S[a]] : 0.0));
            }
        }
        // H_p y_p on non-flat entries, 0 on flat ones (in place).
        void hess(double *y) const {
            const int64_t N = g.N;
            for (int64_t p = 0; p < N; ++p) {
                if (s.flat[p]) {
                    for (int a = 0; a < op.k; ++a) y[a * N + p] = 0.0;
                    continue;
                }
                double nn = 0.0, dot = 0.0;
                for (int a = 0; a < op.k; ++a) nn += gz[a * N + p] * gz[a * N + p];
                const double nr = std::sqrt(nn);
                for (int a = 0; a < op.k; ++a) dot += gz[a * N + p] * y[a * N + p];
                const double h = gamma / nr;
                for (int a = 0; a < op.k; ++a) {
                    const double na = gz[a * N + p] / nr;
                    y[a * N + p] = h * (y[a * N + p] - na * dot / nr);
                }
            }
        }
        // out = M^T (I + G^T H G) M c (component space).
        void apply(const double *cv, double *out) const {
            const int64_t N = g.N;
            for (int64_t p = 0; p < N; ++p) pix[p] = cv[s.label[p]];
            grad(pix, hbuf);
            hess(hbuf);
            // pix + G^T H G pix: G^T h = -sum_a w_a (h_a[p] - h_a[p - e_a]).
            for (int64_t p = 0; p < N; ++p) pix[p] = -pix[p];
            sub_gradT(hbuf, pix);
            std::fill(out, out + nc, 0.0);
            for (int64_t p = 0; p < N; ++p) out[s.label[p]] -= pix[p];
        }
    };

    // Solves S c = b by CG, preconditioned with |C| (the diagonal of M^T M).
    static void cg(const Sys &S, const double *b, double *x, double *r, double *pv, double *qv, DerivCtl &dc) {
        const int64_t nc = S.nc;
        double bn = 0.0;
        for (int64_t C = 0; C < nc; ++C) bn += b[C] * b[C] * S.s.inv[C];
        std::fill(x, x + nc, 0.0);
        if (bn == 0.0) {
            dc.residual = 0.0;
            dc.iters = 0;
            return;
        }
        std::memcpy(r, b, static_cast<size_t>(nc) * sizeof(double));
        double rz = 0.0;
        for (int64_t C = 0; C < nc; ++C) {
            pv[C] = r[C] * S.s.inv[C];
            rz += r[C] * pv[C];
        }
        int it = 0;
        double res = std::sqrt(rz / bn);
        while (res > dc.tol && it < dc.max_iter) {
            S.apply(pv, qv);
            double pq = 0.0;
            for (int64_t C = 0; C < nc; ++C) pq += pv[C] * qv[C];
            if (!(pq > 0.0)) break;
            const double alpha = rz / pq;
            double rzn = 0.0;
            for (int64_t C = 0; C < nc; ++C) {
                x[C] += alpha * pv[C];
                r[C] -= alpha * qv[C];
                rzn += r[C] * r[C] * S.s.inv[C];
            }
            const double beta = rzn / rz;
            rz = rzn;
            for (int64_t C = 0; C < nc; ++C) pv[C] = r[C] * S.s.inv[C] + beta * pv[C];
            ++it;
            res = std::sqrt(rz / bn);
        }
        dc.residual = res;
        dc.iters = it;
    }

    Sys system(const char *save, const Geom &g, double gamma, char *work) const {
        Sys S{*this, g, view(const_cast<char *>(save), g.N), gamma, {0, 0, 0}, nullptr, nullptr, nullptr, 0};
        for (int a = 0; a < k; ++a) S.wa[a] = (gamma > 0.0 && w[a] > 0.0 && g.L[a] > 1) ? w[a] : 0.0;
        double *base = reinterpret_cast<double *>(work);
        S.gz = base;
        S.hbuf = base + k * g.N;
        S.pix = base + 2 * k * g.N;
        S.nc = *S.s.ncomp;
        S.grad(S.s.z, S.gz);
        return S;
    }

    template <class T>
    void jvp_img(const char *save, const T *dx, T *dz, const int64_t *shape, double gamma, double dgamma,
                 const double *dparams, char *work, DerivCtl &dc) const noexcept {
        const Geom g(k, shape);
        const int64_t N = g.N;
        Sys S = system(save, g, gamma, work);
        double *rhs = S.pix + N, *cb = rhs + N, *x = cb + N, *r = x + N, *pv = r + N, *q2 = pv + N;
        // rhs = dx + sum_a dw_a (u_a[p] - u_a[p - e_a]) - G_A^T c_A,
        // c_p = H_p (dW D z)_p + dgamma n_p on non-flat entries.
        for (int64_t p = 0; p < N; ++p) rhs[p] = static_cast<double>(dx[p]);
        if (dparams) {
            for (int a = 0; a < k; ++a) {
                if (dparams[a] == 0.0 || g.L[a] == 1) continue;
                const double *ua = S.s.u + a * N;
                for (int64_t p = 0; p < N; ++p) rhs[p] += dparams[a] * (ua[p] - (g.coord(p, a) > 0 ? ua[p - g.S[a]] : 0.0));
            }
        }
        double *cA = S.hbuf;
        for (int a = 0; a < k; ++a) {
            double *ca = cA + a * N;
            for (int64_t p = 0; p < N; ++p) {
                const double wd = S.wa[a] != 0.0 ? S.gz[a * N + p] / S.wa[a] : 0.0;  // (D_a z)_p
                ca[p] = (dparams && S.wa[a] != 0.0) ? dparams[a] * wd : 0.0;
            }
        }
        S.hess(cA);  // H (dW D z)
        for (int64_t p = 0; p < N; ++p) {
            if (S.s.flat[p]) continue;
            double nn = 0.0;
            for (int a = 0; a < k; ++a) nn += S.gz[a * N + p] * S.gz[a * N + p];
            const double nr = std::sqrt(nn);
            for (int a = 0; a < k; ++a) cA[a * N + p] += dgamma * S.gz[a * N + p] / nr;
        }
        S.sub_gradT(cA, rhs);  // rhs -= G^T c
        std::fill(cb, cb + S.nc, 0.0);
        for (int64_t p = 0; p < N; ++p) cb[S.s.label[p]] += rhs[p];
        cg(S, cb, x, r, pv, q2, dc);
        for (int64_t p = 0; p < N; ++p) dz[p] = static_cast<T>(x[S.s.label[p]]);
    }

    template <class T>
    void vjp_img(const char *save, const T *gin, T *gx, const int64_t *shape, double gamma, double *acc,
                 char *work, DerivCtl &dc) const noexcept {
        const Geom g(k, shape);
        const int64_t N = g.N;
        Sys S = system(save, g, gamma, work);
        double *cb = S.pix + N, *x = cb + N, *r = x + N, *pv = r + N, *q2 = pv + N;
        std::fill(cb, cb + S.nc, 0.0);
        for (int64_t p = 0; p < N; ++p) cb[S.s.label[p]] += static_cast<double>(gin[p]);
        cg(S, cb, x, r, pv, q2, dc);
        double *vpix = S.pix;
        for (int64_t p = 0; p < N; ++p) {
            vpix[p] = x[S.s.label[p]];
            gx[p] = static_cast<T>(vpix[p]);
        }
        // Parameter gradients from v = J^T g:
        //   d/dw_a:   -<D_a v, u_a> - sum_A [H_p (G v)_p]_a (D_a z)_p
        //   d/dgamma: -sum_A <(G v)_p, n_p>
        const int64_t P = k;
        double *gv = S.hbuf;
        S.grad(vpix, gv);  // G v
        for (int a = 0; a < k; ++a) {
            if (g.L[a] == 1) continue;
            const double *ua = S.s.u + a * N;
            double t = 0.0;
            for (int64_t p = 0; p < N; ++p)
                if (g.has_next(p, a)) t += (vpix[p + g.S[a]] - vpix[p]) * ua[p];
            acc[a] -= t;
        }
        double ggam = 0.0;
        for (int64_t p = 0; p < N; ++p) {
            if (S.s.flat[p]) continue;
            double nn = 0.0;
            for (int a = 0; a < k; ++a) nn += S.gz[a * N + p] * S.gz[a * N + p];
            const double nr = std::sqrt(nn);
            for (int a = 0; a < k; ++a) ggam += gv[a * N + p] * S.gz[a * N + p] / nr;
        }
        acc[P] -= ggam;
        S.hess(gv);  // H G v
        for (int a = 0; a < k; ++a) {
            if (S.wa[a] == 0.0) continue;
            double t = 0.0;
            for (int64_t p = 0; p < N; ++p)
                if (!S.s.flat[p]) t += gv[a * N + p] * (S.gz[a * N + p] / S.wa[a]);
            acc[a] -= t;
        }
    }
};

}  // namespace

fp_status make_tv_iso(int ndim, Vec w, fp_op **out) {
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
    *out = new TVIso(ndim, ww, std::move(tv));
    return FP_OK;
}

}  // namespace fp
