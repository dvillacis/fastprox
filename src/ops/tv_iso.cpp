// Isotropic TV of 2D and 3D images: the prox of
//   gamma sum_p alpha_p | (w_0 D_0 z, ..., w_{k-1} D_{k-1} z)_p |_2
// with axis weights w and, since 0.3, optional pixel weights alpha >= 0
// (alpha = 1 without them), for which the operator is built for one image
// shape. The pixel weights enter only the dual bound, |u_p| <= gamma alpha_p
// (the radius r_p below); everything else is unchanged. With them, a single
// active axis runs the 2D method rather than the exact 1D TV pass.
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
// Restarts: when <v - u+, u+ - u> > 0 (the gradient mapping (v - u+) / tau and
// the step disagree: the momentum points uphill) the iteration restarts from
// u+ with t = 1, the gradient scheme of B. O'Donoghue and E. Candes, "Adaptive
// restart for accelerated gradient schemes", Found. Comput. Math. 15 (2015)
// 715-732. The dual is degenerate, so restarts do not make the method
// linearly convergent here, but they cut the iterations to tolerances of
// 1e-8 and below by up to 15x; at gamma <= 0.3 they first fire near a gap of
// 1e-6, so looser solves are unchanged. The test runs
// every kCheck-th pass (1-4% per iteration), and the interval since the last
// restart must not shrink: restarting at every opportunity slowed some
// images down (blocks at gamma = 1 by 1.5x to 1e-8).
//
// Certificate: for feasible u, the duality gap of the prox problem at z(u) is
//     sum_p ( gamma |g_p| - <g_p, u_p> ),   g = W D z(u),
// every term >= 0, and |z - z*|^2 <= 2 gap (Beck, Lemma 12.7).
//
// Derivatives by implicit differentiation of the optimality conditions
// z - d + D^T W u = 0, u_p = gamma g_p / |g_p| where g_p != 0. Entries with
// g_p = 0 (flat) keep z constant across their edges, so dz lies in the space
// of images constant on the connected flat regions (one unknown per region);
// elsewhere u_p moves along the tangent of the sphere, du_p = H_p (W D dz)_p
// with H_p = gamma / |g_p| (I - n_p n_p^T), n_p = g_p / |g_p|. Then
//     (I + G_A^T H G_A) dz = rhs  on that space  (G = W D, A the non-flat set),
// a symmetric positive definite system solved by conjugate gradients,
// preconditioned with its diagonal (|C| plus the H_p terms of the boundary
// entries of each region; 1.4-3x fewer iterations than |C| alone in most
// expensive solves, up to 40% more in cheap ones at small gamma).
//
// Flat entries are read off the dual: |u_p| < gamma forces g_p = 0
// (complementary slackness), so an entry strictly inside its ball,
// |u_p| < gamma (1 - kDualEta), is flat, as are entries with g_p at the
// rounding level. A threshold on |g_p| instead (8 times the rms error that
// the gap certifies) also flattened entries with small true jumps: against
// the exact derivative on 2 x 128^2 images at gamma = 0.03 and 0.1, errors
// of 0.18-0.57 at a gap tolerance of 1e-6 and 0.05-0.24 at 1e-8, against
// 0.04-0.05 and 0.007-0.016 now.
//
// Entries on the sphere with small |g_p| make the system stiff, and stiffer
// as the gap shrinks (|g_p| near 1e-10 at a gap of 1e-13: a curvature near
// 1e9), although such gradients are far below the error of z that the gap
// certifies. The curvature gamma / |g_p| is therefore capped at gamma / mu,
// mu = kCurvRms w_max rms(gap): the Huber smoothing of TV at that level, as
// in the semismooth Newton methods for TV of M. Hintermueller and G.
// Stadler, SIAM J. Sci. Comput. 28 (2006) 1-23, and in bilevel learning,
// J. C. De los Reyes, C.-B. Schoenlieb and T. Valkonen, J. Math. Imaging
// Vis. 57 (2017) 1-25. It vanishes with the gap. On those images at
// gamma = 0.1: errors 0.041, 0.016 and 0.0019 at 1e-6, 1e-8 and 1e-10 with
// 536, 1272 and 3600 iterations, against 0.048, 0.0074, 0.0026 and 960,
// 3560, 12832 uncapped; a cap at mu = rms, or flat entries below it, lost
// accuracy (up to 0.04 and 0.29).
//
// Conjugate gradients stop, by default (a tolerance of 0), at the relative
// accuracy that the gap certifies for z, sqrt(2 gap) / |d|, and no tighter
// than kDerivFloor: the derivative's error from a solve to a gap tolerance tol
// is 50-100 sqrt(tol) on these images, and residuals below sqrt(tol) did not
// change it (2 x 128^2 at gamma = 0.1, tol 1e-4 to 1e-10: 34, 172, 545 and
// about 1900 iterations against 182, 502, 1346 and 3369 to 1e-10).
//
// Pixel weights: the radius r_p = gamma alpha_p replaces gamma in the
// projection, the gap, the flat test and the curvature, and dr_p = dgamma
// alpha_p + gamma dalpha_p enters the right-hand side as dgamma did (c_p =
// dr_p n_p); the VJP's gradient in alpha_p is -gamma <(G v)_p, n_p> on
// non-flat pixels and 0 on flat ones. A pixel with alpha_p = 0 has u_p = 0:
// no coupling, its jumps are free.
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
#include <type_traits>
#include <utility>

namespace fp {
namespace {

constexpr int kCheck = 4;  // iterations between gap checks
// Entries whose dual is within this relative distance of the ball's radius
// count as on the sphere (the projection leaves them there to rounding).
constexpr double kDualEta = 1e-9;
// The curvature gamma / |g_p| of the derivative system is capped at
// gamma / mu with mu = kCurvRms w_max rms, rms = sqrt(2 gap / N) the error of z
// that the gap certifies (a Huber smoothing at the level of that error).
constexpr double kCurvRms = 0.1;
// The tightest default tolerance of the derivative's conjugate gradients.
constexpr double kDerivFloor = 1e-10;
// Fixed blocks of the threaded derivative solves (audit T2): region sums in
// chunks of kSeg positions of the region-ordered pixel list, region-space
// dot products in blocks of kRegBlk regions.
constexpr int64_t kSeg = 4096;
constexpr int64_t kRegBlk = 4096;
constexpr double kInf = std::numeric_limits<double>::infinity();

// The derivative loops run over rows b = r n, ..., b + n - 1 (n the last
// extent): along an axis a < k - 1 a row has its neighbours p +- e_a for all
// of its entries or for none, so they test (b / S_a) % L_a once per row
// rather than per entry and axis. The arithmetic per entry is unchanged.
//
// x[p] += c (h[p] - h[p - e_a]) on the row at b, with h taken as 0 before
// the first entry along a.
void add_diff_row(const Geom &g, int a, double c, const double *h, double *x, int64_t b) noexcept {
    const int64_t n = g.L[g.k - 1], Sa = g.S[a];
    if (a == g.k - 1) {
        x[b] += c * h[b];
        for (int64_t j = 1; j < n; ++j) x[b + j] += c * (h[b + j] - h[b + j - 1]);
    } else if (g.coord(b, a) > 0) {
        for (int64_t j = 0; j < n; ++j) x[b + j] += c * (h[b + j] - h[b + j - Sa]);
    } else {
        for (int64_t j = 0; j < n; ++j) x[b + j] += c * h[b + j];
    }
}

// nxt[a]: whether the entries of the row at b have a forward neighbour along
// a < k - 1 with a nonzero weight; along k - 1 all but the last do.
inline void row_next(const Geom &g, const double *wa, int64_t b, bool *nxt) noexcept {
    for (int a = 0; a < g.k; ++a) nxt[a] = wa[a] != 0.0 && (a == g.k - 1 || g.has_next(b, a));
}

struct TVIso final : ImageOp<TVIso> {
    int k;
    double w[3];
    std::unique_ptr<fp_op> tv;  // exact 1D TV, for a single active axis
    Vec alpha;                  // pixel weights (C order over ashape), or empty
    double alphamax_ = vmax(alpha);  // its largest entry, for step_scale: once, not per call (it scanned every weight on each prox call)
    int64_t ashape[3] = {1, 1, 1};

    TVIso(int kk, const double *ww, std::unique_ptr<fp_op> t, Vec al = {}, const int64_t *ash = nullptr)
        : k(kk), tv(std::move(t)), alpha(std::move(al)) {
        for (int a = 0; a < 3; ++a) w[a] = a < k ? ww[a] : 0.0;
        if (ash)
            for (int a = 0; a < k; ++a) ashape[a] = ash[a];
    }

    const char *name() const noexcept override { return "tv_iso"; }
    double step_scale() const noexcept override { return std::max({w[0], w[1], w[2]}) * (alpha.empty() ? 1.0 : alphamax_); }
    bool conj_homogeneous() const noexcept override { return true; }
    int64_t nparams() const noexcept override { return k + static_cast<int64_t>(alpha.size()); }
    int image_ndim() const noexcept override { return k; }
    bool splits_image() const noexcept override { return true; }  // rows in parallel (row_for)
    bool threads_derivatives() const noexcept override { return true; }  // audit T2
    fp_status check_image(const int64_t *shape) const noexcept override {
        double n = 1.0;
        for (int a = 0; a < k; ++a) n *= static_cast<double>(shape[a]);
        if (!alpha.empty())
            for (int a = 0; a < k; ++a)
                if (shape[a] != ashape[a]) return FP_ESHAPE;
        return n < 2147483647.0 ? FP_OK : FP_ESHAPE;  // 32-bit region labels
    }
    const double *pixel_weights() const noexcept { return alpha.empty() ? nullptr : alpha.data(); }
    // The radius gamma alpha_p, capped at bound (gamma = inf with alpha_p = 0 is 0).
    static double radius(double gamma, const double *al, int64_t p, double bound) noexcept {
        if (!al) return gamma;
        if (al[p] == 0.0) return 0.0;
        const double r = gamma * al[p];
        return r <= bound ? r : bound;
    }

    // prox: d, z (N), u, v (k N), per-row sums; derivatives: the gradient of z
    // and a scratch field (k N each), then eight vectors of N (image, rhs, the
    // CG vectors and the preconditioner in region space), per-row sums, and
    // the pieces of the region sums and the block partials of the dot
    // products (2 each per chunk of kSeg).
    size_t image_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        const size_t N = static_cast<size_t>(g.N), K = static_cast<size_t>(k);
        const size_t nch = (N + kSeg - 1) / kSeg;
        return (std::max((2 + 2 * K) * N, (8 + 2 * K) * N) + static_cast<size_t>(g.rows()) + 4 * nch + 8) *
               sizeof(double);
    }
    size_t image_thread_bytes(const int64_t *shape) const noexcept override {
        return fiber_thread_bytes(*tv, Geom(k, shape));
    }
    // Saved: number of regions, the curvature floor mu, the certified relative
    // error of z, the exponent e of the rescaling (z, u and mu are those of
    // the problem solved, 2^-e times the input's; see prox_img), labels, flat
    // flags, z, u (k blocks), 1/|C|, and the regions' pixels: those of C in
    // increasing order at members[rstart[C]] .. members[rstart[C + 1] - 1].
    static size_t r8(size_t b) noexcept { return (b + 7) / 8 * 8; }
    size_t struct_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        const size_t N = static_cast<size_t>(g.N);
        return 32 + r8(4 * N) + r8(N) + (2 + static_cast<size_t>(k)) * N * 8 + r8(4 * N) + r8(4 * (N + 1));
    }
    struct View {
        int64_t *ncomp, *ex;
        double *mu, *rel;
        int32_t *label;
        uint8_t *flat;
        double *z, *u, *inv;
        int32_t *members, *rstart;
    };
    // Structure (since 1.0.2): the saved regions; J = M (M^T (I + G^T H G)
    // M)^{-1} M^T, with M their membership matrix, has range span M and is
    // positive definite there.
    int struct_kind() const noexcept override { return FP_STRUCT_CURVED; }
    const int32_t *image_labels(const char *save, const int64_t *shape, int64_t *nreg) const noexcept override {
        const View v = view(const_cast<char *>(save), Geom(k, shape).N);
        *nreg = *v.ncomp;
        return v.label;
    }
    View view(char *s, int64_t N) const noexcept {
        View v;
        const size_t n = static_cast<size_t>(N);
        v.ncomp = reinterpret_cast<int64_t *>(s);
        v.mu = reinterpret_cast<double *>(s + 8);
        v.rel = reinterpret_cast<double *>(s + 16);
        v.ex = reinterpret_cast<int64_t *>(s + 24);
        v.label = reinterpret_cast<int32_t *>(s + 32);
        v.flat = reinterpret_cast<uint8_t *>(s + 32 + r8(4 * n));
        v.z = reinterpret_cast<double *>(s + 32 + r8(4 * n) + r8(n));
        v.u = v.z + N;
        v.inv = v.u + k * N;
        v.members = reinterpret_cast<int32_t *>(v.inv + N);
        v.rstart = v.members + r8(4 * n) / 4;
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

    double gap(const Geom &g, const ImageCall &c, const double *wa, double gamma, double bound, const double *z,
               const double *u, double *part) const {
        const int64_t N = g.N;
        const double *al = pixel_weights();
        return row_sum(g, c, part, [&](int64_t r) {
            double s = 0.0;
            grad_rows(g, wa, z, r, [&](int64_t p, const double *gv) {
                double nn = 0.0, dot = 0.0;
                for (int a = 0; a < k; ++a) {
                    nn += gv[a] * gv[a];
                    dot += gv[a] * u[a * N + p];
                }
                s += radius(gamma, al, p, bound) * std::sqrt(nn) - dot;
            });
            return s;
        });
    }

    // ---- prox -------------------------------------------------------------------

    // sum_p alpha_p ||(w_a (x_{p+e_a} - x_p))_a||_2, differences zero past the
    // last entry along each axis.
    bool has_value() const noexcept override { return true; }
    double value_image(const double *x, const int64_t *shape, double) const noexcept override {
        int64_t stride[3], N = 1;
        for (int a = k - 1; a >= 0; --a) {
            stride[a] = N;
            N *= shape[a];
        }
        double s = 0.0;
        for (int64_t p = 0; p < N; ++p) {
            double q = 0.0;
            for (int a = 0; a < k; ++a) {
                if ((p / stride[a]) % shape[a] == shape[a] - 1) continue;
                const double d = w[a] * (x[p + stride[a]] - x[p]);
                q += d * d;
            }
            s += (alpha.empty() ? 1.0 : alpha[static_cast<size_t>(p)]) * std::sqrt(q);
        }
        return s;
    }

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
            if (c.save) save(g, c, wa, d, d, nullptr);
            c.gap = std::numeric_limits<double>::quiet_NaN();  // not solved (audit A8)
            return kRowOk;
        }
        // Far from unit scale, solve at a power-of-two scale (image_util.hpp),
        // unless an active step would underflow there.
        const double gamma0 = c.gamma;
        int e = na ? rescale_exponent(d, N, c.gamma, c.gamma * step_scale()) : 0;
        for (int a = 0; a < k && e; ++a)
            if (wa[a] > 0.0 && !(std::ldexp(gamma0 * wa[a], -e) > 0.0)) e = 0;
        if (e) {
            scale2(d, N, -e);
            if (c.warm) scale2(q, k * N, -e);
            c.gamma = std::ldexp(gamma0, -e);
        }
        if (na == 0) {
            std::memcpy(z, d, static_cast<size_t>(N) * sizeof(double));
            std::fill(q, q + k * N, 0.0);
        } else if (na == 1 && alpha.empty()) {  // exactly 1D TV along the active axis
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
        if (c.save) save(g, c, wa, d, z, na >= 1 ? c.dual : nullptr, na >= 2);
        if (e) {  // back to the input's scale; the saved structure stays at the solved one
            scale2(z, N, e);
            scale2(q, k * N, e);
            c.scale_exp = e;
            c.scaled_gap = c.gap;
            c.gap = std::ldexp(c.gap, 2 * e);
            c.smoothing = std::ldexp(c.smoothing, e);  // mu scales with z
            c.gamma = gamma0;
            if (c.save) *view(c.save, N).ex = e;
        }
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
        if (!(gamma <= bound) && alpha.empty()) gamma = bound;  // also catches gamma * wmax = inf
        // With pixel weights each radius gamma alpha_p is capped at bound instead.
        const double *al = pixel_weights();
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
                const double rp = radius(gamma, al, p, bound);
                if (std::sqrt(nn) > rp) {  // not nn > rp^2: rp may be ~1e300
                    const double sc = rp / std::sqrt(nn);
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
            return gap(g, c, wa, gamma, bound, z, u, part);
        };
        double t = 1.0;
        int it = 0, since = 0, last = 0;
        double gp = c.warm ? primal() : std::numeric_limits<double>::infinity();
        while (!(gp <= target) && it < c.max_iter) {
            divergence(g, c, wa, d, v, z);
            const double tn = fista_t(t);
            const double beta = (t - 1.0) / tn;
            t = tn;
            // Every kCheck-th pass also returns <v - u+, u+ - u>, the
            // gradient-mapping restart test, in fixed row blocks.
            const bool test = it % kCheck == 0;
            auto update = [&](int64_t r, auto with_test) {
                double s = 0.0;
                grad_rows(g, wa, z, r, [&](int64_t p, const double *gv) {
                    double cv[3], nn = 0.0;
                    // v is zero past the last entry along each axis and gv is
                    // too, so the dual stays zero there.
                    for (int a = 0; a < k; ++a) {
                        cv[a] = wa[a] != 0.0 ? v[a * N + p] + tau * gv[a] : 0.0;
                        nn += cv[a] * cv[a];
                    }
                    const double nr = std::sqrt(nn), rp = radius(gamma, al, p, bound);
                    const double sc = nr > rp ? rp / nr : 1.0;
                    for (int a = 0; a < k; ++a) {
                        const double un = cv[a] * sc;
                        double &ua = u[a * N + p];
                        double &va = v[a * N + p];
                        if constexpr (decltype(with_test)::value) s += (va - un) * (un - ua);
                        va = un + beta * (un - ua);
                        ua = un;
                    }
                });
                return s;
            };
            double up = 0.0;
            if (test) up = row_sum(g, c, part, [&](int64_t r) { return update(r, std::true_type{}); });
            else row_for(g, c, [&](int64_t r) { update(r, std::false_type{}); });
            // Momentum pointing uphill: restart from u+ (O'Donoghue and Candes),
            // but never sooner after the last restart than that one came after
            // its predecessor.
            ++since;
            if (test && up > 0.0 && since >= last) {
                std::memcpy(v, u, static_cast<size_t>(k * N) * sizeof(double));
                t = 1.0;
                last = since;
                since = 0;
            }
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

    // Flat entries and the regions they connect; z and u for the derivative
    // system. dual null: u = 0. from_dual: flat where the dual is strictly
    // inside its ball or |g_p| is at the rounding level of the data; else
    // where g_p = 0 exactly (z is exactly piecewise constant).
    void save(const Geom &g, ImageCall &c, const double *wa, const double *d, const double *z, const double *dual,
              bool from_dual = false) const {
        const int64_t N = g.N;
        View s = view(c.save, N);
        *s.ex = 0;
        std::memcpy(s.z, z, static_cast<size_t>(N) * sizeof(double));
        if (dual) std::memcpy(s.u, dual, static_cast<size_t>(k * N) * sizeof(double));
        else std::fill(s.u, s.u + k * N, 0.0);
        double scale = 0.0, wmax = 0.0, dd = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            scale = std::max(scale, std::fabs(d[i]));
            dd += d[i] * d[i];
        }
        for (int a = 0; a < k; ++a) wmax = std::max(wmax, wa[a]);
        const double thr = from_dual ? wmax * 64.0 * 2.2e-16 * scale : 0.0;
        const double *al = pixel_weights();
        *s.rel = dd > 0.0 ? std::sqrt(2.0 * std::max(c.gap, 0.0) / dd) : 0.0;
        *s.mu = from_dual ? kCurvRms * wmax * std::sqrt(2.0 * std::max(c.gap, 0.0) / static_cast<double>(N)) : 0.0;
        c.smoothing = *s.mu;  // at the solved scale; scaled back with z
        c.exact = !(*s.mu > 0.0);
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
        const int64_t n = g.L[k - 1];
        bool nxt[3];
        for (int64_t p = 0, j = 0; p < N; ++p, j = j + 1 == n ? 0 : j + 1) {  // j: p's column (add_diff)
            if (j == 0) row_next(g, wa, p, nxt);
            const auto has = [&](int a) { return nxt[a] && (a + 1 < k || j + 1 < n); };
            double nn = 0.0;
            bool any = false;
            for (int a = 0; a < k; ++a) {
                if (!has(a)) continue;
                const double t = wa[a] * (z[p + g.S[a]] - z[p]);
                nn += t * t;
                any = true;
            }
            bool flat = !any || std::sqrt(nn) <= thr;
            if (from_dual && !flat) {
                double uu = 0.0;
                for (int a = 0; a < k; ++a) uu += s.u[a * N + p] * s.u[a * N + p];
                flat = std::sqrt(uu) < radius(c.gamma, al, p, kInf) * (1.0 - kDualEta);
            }
            s.flat[p] = flat ? 1 : 0;
            if (!flat || !any) continue;
            for (int a = 0; a < k; ++a) {
                if (!has(a)) continue;
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
        // The regions' pixels by a counting sort: rstart[C + 1] counts C, its
        // prefix sums start the lists, and filling in index order advances
        // rstart[C] to the start of C + 1, which the shift undoes.
        int32_t *rs = s.rstart;
        std::fill(rs, rs + nc + 1, 0);
        for (int64_t i = 0; i < N; ++i) ++rs[s.label[i] + 1];
        for (int64_t C = 0; C < nc; ++C) rs[C + 1] += rs[C];
        for (int64_t i = 0; i < N; ++i) s.members[rs[s.label[i]]++] = static_cast<int32_t>(i);
        for (int64_t C = nc; C > 0; --C) rs[C] = rs[C - 1];
        rs[0] = 0;
    }

    // ---- kinks (since 0.3) -----------------------------------------------------------

    // Code 1 at a pixel whose gradient |(W D z)_p| is within tol of 0 while its
    // dual is within tol of the sphere, |u_p| >= gamma alpha_p - tol (pixels
    // of zero radius are free, never kinks).
    bool has_kinks() const noexcept override { return true; }
    int64_t kink_image(const char *, const double *, const double *z, const double *u, const int64_t *shape, double gamma,
                       double tol, int8_t *mask, const int64_t *ms, char *) const noexcept override {
        const Geom g(k, shape);
        const int64_t N = g.N;
        const double *al = pixel_weights();
        int64_t count = 0;
        for (int64_t p = 0; p < N; ++p) {
            double gg = 0.0, uu = 0.0;
            for (int a = 0; a < k; ++a) {
                if (w[a] > 0.0 && g.has_next(p, a)) {
                    const double t = w[a] * (z[p + g.S[a]] - z[p]);
                    gg += t * t;
                }
                uu += u[a * N + p] * u[a * N + p];
            }
            const double r = radius(gamma, al, p, kInf);
            const int8_t code = r > 0.0 && std::sqrt(gg) <= tol && std::sqrt(uu) >= r - tol;
            int64_t o = 0;
            for (int a = 0; a < k; ++a) o += g.coord(p, a) * ms[a];
            mask[o] = code;
            count += code;
        }
        return count;
    }

    // ---- derivatives ----------------------------------------------------------------

    // Threads (audit T2): the pixel passes run by rows, the region sums over
    // fixed chunks of the region-ordered pixel list, and the dot products in
    // fixed blocks of regions and of rows, so that the bits do not depend on
    // the thread count; an image of at most one chunk, block and row block
    // (64^2 and below) gives the bits of a single pass. jacobi stays serial
    // (one apply's worth, once per solve).
    struct Sys {
        const TVIso &op;
        const Geom &g;
        View s;
        double gamma;
        const double *alpha;  // pixel weights or null
        double wa[3];
        double *gz;   // k N: W D z at each entry
        double *pix;  // N
        double *hbuf; // k N scratch
        int64_t nc;
        int nt;        // threads
        double *part;  // rows() partial sums
        double *piece; // 2 per chunk of kSeg: the region pieces at its ends
        double *rpart; // 2 per block of kRegBlk regions

        // The radius gamma alpha_p of pixel p.
        double r(int64_t p) const noexcept { return radius(gamma, alpha, p, kInf); }
        // f(b) for the row at b = r n of every row r, in parallel.
        template <class F>
        void rows(F &&f) const {
            const int64_t n = g.L[op.k - 1];
            parallel_for(nt, g.rows(), std::max<int64_t>(1, kChunk / n), [&](int, int64_t r0, int64_t r1) {
                for (int64_t r = r0; r < r1; ++r) f(r * n);
            });
        }
        // The sum of f(b, t) (adding the row at b's terms to t, in order) over
        // the rows, in fixed blocks of about kChunk entries added in order.
        template <class F>
        double row_total(F &&f) const {
            const int64_t n = g.L[op.k - 1], R = g.rows(), rb = std::max<int64_t>(1, kChunk / n);
            const int64_t nb = (R + rb - 1) / rb;
            parallel_for(nt, nb, 1, [&](int, int64_t b0, int64_t b1) {
                for (int64_t blk = b0; blk < b1; ++blk) {
                    double t = 0.0;
                    for (int64_t r = blk * rb; r < std::min(R, (blk + 1) * rb); ++r) f(r * n, t);
                    part[blk] = t;
                }
            });
            double t = 0.0;
            for (int64_t blk = 0; blk < nb; ++blk) t += part[blk];
            return t;
        }
        // f(C0, C1) over the fixed blocks of regions, in parallel.
        template <class F>
        void regions(F &&f) const {
            const int64_t nb = (nc + kRegBlk - 1) / kRegBlk;
            parallel_for(nt, nb, std::max<int64_t>(1, kChunk / kRegBlk), [&](int, int64_t b0, int64_t b1) {
                f(b0 * kRegBlk, std::min(nc, b1 * kRegBlk));
            });
        }
        // The sums of f(C, t0, t1) (adding region C's terms to t0 and t1) over
        // the regions, in the fixed blocks added in order.
        template <class F>
        void region_totals(double &s0, double &s1, F &&f) const {
            const int64_t nb = (nc + kRegBlk - 1) / kRegBlk;
            parallel_for(nt, nb, std::max<int64_t>(1, kChunk / kRegBlk), [&](int, int64_t b0, int64_t b1) {
                for (int64_t blk = b0; blk < b1; ++blk) {
                    double t0 = 0.0, t1 = 0.0;
                    for (int64_t C = blk * kRegBlk; C < std::min(nc, (blk + 1) * kRegBlk); ++C) f(C, t0, t1);
                    rpart[2 * blk] = t0;
                    rpart[2 * blk + 1] = t1;
                }
            });
            s0 = s1 = 0.0;
            for (int64_t blk = 0; blk < nb; ++blk) {
                s0 += rpart[2 * blk];
                s1 += rpart[2 * blk + 1];
            }
        }
        // out[C] = the sum of v over the pixels of C (minus it with Neg), in
        // their order: a region inside one chunk of kSeg positions is summed
        // by that chunk's thread as one loop did; one cut by chunk boundaries
        // as a piece per chunk, the pieces then added in chunk order.
        template <bool Neg>
        void region_sum(const double *v, double *out) const {
            const int64_t N = g.N, nch = (N + kSeg - 1) / kSeg;
            const int32_t *rs = s.rstart, *mem = s.members, *lab = s.label;
            parallel_for(nt, nch, std::max<int64_t>(1, kChunk / kSeg), [&](int, int64_t j0, int64_t j1) {
                for (int64_t j = j0; j < j1; ++j) {
                    const int64_t lo = j * kSeg, hi = std::min(N, lo + kSeg);
                    for (int64_t C = lab[mem[lo]], m = lo; m < hi; ++C) {
                        const int64_t end = std::min<int64_t>(rs[C + 1], hi);
                        double t = 0.0;
                        for (; m < end; ++m) {
                            if constexpr (Neg) t -= v[mem[m]];
                            else t += v[mem[m]];
                        }
                        if (rs[C] >= lo && rs[C + 1] <= hi) out[C] = t;
                        else piece[2 * j + (rs[C] < lo ? 0 : 1)] = t;  // continued from before / on after
                    }
                }
            });
            double run = 0.0;  // the region open across chunks
            for (int64_t j = 0; j < nch; ++j) {
                const int64_t lo = j * kSeg, hi = std::min(N, lo + kSeg);
                const int64_t C = lab[mem[lo]];
                if (rs[C] < lo) {
                    run += piece[2 * j];
                    if (rs[C + 1] > hi) continue;  // the whole chunk, and on
                    out[C] = run;
                }
                const int64_t Ct = lab[mem[hi - 1]];
                if (rs[Ct] >= lo && rs[Ct + 1] > hi) run = piece[2 * j + 1];
            }
        }
        // y_p = (G x)_p, the scaled forward differences of the image x, on the
        // row at b (see add_diff_row).
        void grad_row(const double *x, double *y, int64_t b) const {
            const int64_t N = g.N, n = g.L[op.k - 1];
            for (int a = 0; a < op.k; ++a) {
                double *ya = y + a * N;
                const double wv = wa[a];
                if (wv == 0.0) {
                    std::fill(ya + b, ya + b + n, 0.0);
                    continue;
                }
                const int64_t Sa = g.S[a];
                if (a == op.k - 1) {
                    for (int64_t j = 0; j + 1 < n; ++j) ya[b + j] = wv * (x[b + j + 1] - x[b + j]);
                    ya[b + n - 1] = 0.0;
                } else if (g.has_next(b, a)) {
                    for (int64_t j = 0; j < n; ++j) ya[b + j] = wv * (x[b + j + Sa] - x[b + j]);
                } else {
                    std::fill(ya + b, ya + b + n, 0.0);
                }
            }
        }
        void grad(const double *x, double *y) const {
            rows([&](int64_t b) { grad_row(x, y, b); });
        }
        // x -= G^T h (h zero past the ends), i.e. x += sum_a w_a (h_a[p] - h_a[p - e_a]),
        // on the row at b.
        void sub_gradT_row(const double *h, double *x, int64_t b) const {
            for (int a = 0; a < op.k; ++a)
                if (wa[a] != 0.0) add_diff_row(g, a, wa[a], h + a * g.N, x, b);
        }
        // H_p y_p on non-flat entries, 0 on flat ones (in place), on the row at b.
        void hess_row(double *y, int64_t b) const {
            const int64_t N = g.N, n = g.L[op.k - 1];
            for (int64_t p = b; p < b + n; ++p) {
                if (s.flat[p]) {
                    for (int a = 0; a < op.k; ++a) y[a * N + p] = 0.0;
                    continue;
                }
                double nn = 0.0, dot = 0.0;
                for (int a = 0; a < op.k; ++a) nn += gz[a * N + p] * gz[a * N + p];
                const double nr = std::sqrt(nn);
                for (int a = 0; a < op.k; ++a) dot += gz[a * N + p] * y[a * N + p];
                const double h = r(p) / std::max(nr, *s.mu);
                for (int a = 0; a < op.k; ++a) {
                    const double na = gz[a * N + p] / nr;
                    y[a * N + p] = h * (y[a * N + p] - na * dot / nr);
                }
            }
        }
        void hess(double *y) const {
            rows([&](int64_t b) { hess_row(y, b); });
        }
        // 1 / diag(M^T (I + G^T H G) M): |C| plus, for every non-flat entry p,
        // h^T H_p h with h = (G 1_C)_p, which is nonzero only for the regions of
        // p and of its forward neighbours.
        void jacobi(double *dinv) const {
            const int64_t N = g.N, n = g.L[op.k - 1];
            for (int64_t C = 0; C < nc; ++C) dinv[C] = 1.0 / s.inv[C];
            bool nxt[3];
            for (int64_t p = 0, j = 0; p < N; ++p, j = j + 1 == n ? 0 : j + 1) {  // j: p's column
                if (j == 0) row_next(g, wa, p, nxt);
                if (s.flat[p]) continue;
                double nn = 0.0;
                for (int a = 0; a < op.k; ++a) nn += gz[a * N + p] * gz[a * N + p];
                const double nr = std::sqrt(nn), hs = r(p) / std::max(nr, *s.mu);
                int64_t lab[4];
                bool has[3];
                lab[0] = s.label[p];
                for (int a = 0; a < op.k; ++a) {
                    has[a] = nxt[a] && (a + 1 < op.k || j + 1 < n);
                    lab[a + 1] = has[a] ? s.label[p + g.S[a]] : -1;
                }
                for (int c = 0; c <= op.k; ++c) {
                    const int64_t C = lab[c];
                    if (C < 0) continue;
                    bool seen = false;
                    for (int e = 0; e < c; ++e) seen = seen || lab[e] == C;
                    if (seen) continue;
                    double hh = 0.0, nh = 0.0;
                    for (int a = 0; a < op.k; ++a) {
                        if (!has[a]) continue;
                        const double h = wa[a] * ((lab[a + 1] == C ? 1.0 : 0.0) - (lab[0] == C ? 1.0 : 0.0));
                        hh += h * h;
                        nh += h * gz[a * N + p] / nr;
                    }
                    dinv[C] += hs * (hh - nh * nh);
                }
            }
            for (int64_t C = 0; C < nc; ++C) dinv[C] = 1.0 / dinv[C];
        }
        // out = M^T (I + G^T H G) M c (component space).
        void apply(const double *cv, double *out) const {
            rows([&](int64_t b) {
                for (int64_t p = b; p < b + g.L[op.k - 1]; ++p) pix[p] = cv[s.label[p]];
            });
            rows([&](int64_t b) {
                grad_row(pix, hbuf, b);
                hess_row(hbuf, b);
            });
            // pix + G^T H G pix: G^T h = -sum_a w_a (h_a[p] - h_a[p - e_a]).
            rows([&](int64_t b) {
                for (int64_t p = b; p < b + g.L[op.k - 1]; ++p) pix[p] = -pix[p];
                sub_gradT_row(hbuf, pix, b);
            });
            region_sum<true>(pix, out);
        }
    };

    // Solves S c = b by CG with the Jacobi preconditioner (dinv, the inverse
    // diagonal of S); the residual is reported relative to b in the norm
    // weighted by 1/|C|, the pixel-space norm of a region residual. A
    // tolerance of 0 stops at the certified relative error of z.
    static void cg(const Sys &S, const double *b, double *x, double *r, double *pv, double *qv, double *dinv,
                   DerivCtl &dc) {
        const double *inv = S.s.inv;
        S.jacobi(dinv);
        double bn = 0.0, rz = 0.0, unused = 0.0;
        S.region_totals(bn, unused, [&](int64_t C, double &t, double &) { t += b[C] * b[C] * inv[C]; });
        S.regions([&](int64_t C0, int64_t C1) { std::fill(x + C0, x + C1, 0.0); });
        if (bn == 0.0) {
            dc.residual = 0.0;
            dc.iters = 0;
            return;
        }
        S.region_totals(rz, unused, [&](int64_t C, double &t, double &) {
            r[C] = b[C];
            pv[C] = r[C] * dinv[C];
            t += r[C] * pv[C];
        });
        const double tol = dc.tol > 0.0 ? dc.tol : std::max(kDerivFloor, *S.s.rel);
        int it = 0;
        double res = 1.0;
        while (res > tol && it < dc.max_iter) {
            S.apply(pv, qv);
            double pq = 0.0;
            S.region_totals(pq, unused, [&](int64_t C, double &t, double &) { t += pv[C] * qv[C]; });
            if (!(pq > 0.0)) break;
            const double alpha = rz / pq;
            double rzn = 0.0, rn = 0.0;
            S.region_totals(rzn, rn, [&](int64_t C, double &t0, double &t1) {
                x[C] += alpha * pv[C];
                r[C] -= alpha * qv[C];
                t0 += r[C] * r[C] * dinv[C];
                t1 += r[C] * r[C] * inv[C];
            });
            const double beta = rzn / rz;
            rz = rzn;
            S.regions([&](int64_t C0, int64_t C1) {
                for (int64_t C = C0; C < C1; ++C) pv[C] = r[C] * dinv[C] + beta * pv[C];
            });
            ++it;
            res = std::sqrt(rn / bn);
        }
        dc.residual = res;
        dc.iters = it;
    }

    Sys system(const char *save, const Geom &g, double gamma, char *work, int nt) const {
        Sys S{*this, g, view(const_cast<char *>(save), g.N), gamma, pixel_weights(), {0, 0, 0}, nullptr, nullptr,
              nullptr, 0, std::max(1, nt), nullptr, nullptr, nullptr};
        for (int a = 0; a < k; ++a) S.wa[a] = (gamma > 0.0 && w[a] > 0.0 && g.L[a] > 1) ? w[a] : 0.0;
        double *base = reinterpret_cast<double *>(work);
        S.gz = base;
        S.hbuf = base + k * g.N;
        S.pix = base + 2 * k * g.N;
        S.part = base + (2 * k + 8) * g.N;
        S.piece = S.part + g.rows();
        S.rpart = S.piece + 2 * ((g.N + kSeg - 1) / kSeg);
        S.nc = *S.s.ncomp;
        S.grad(S.s.z, S.gz);
        return S;
    }

    template <class T>
    void jvp_img(const char *save, const T *dx, T *dz, const int64_t *shape, double gamma, double dgamma,
                 const double *dparams, char *work, DerivCtl &dc) const noexcept {
        const Geom g(k, shape);
        const int64_t N = g.N, n = g.L[k - 1];
        // At the solved scale (prox_img): z = 2^e zh(2^-e x, 2^-e gamma, theta),
        // so the derivatives in x and gamma are zh's, and theta's carry 2^e.
        const int e = static_cast<int>(*view(const_cast<char *>(save), N).ex);
        const double sc = std::ldexp(1.0, e);
        gamma = std::ldexp(gamma, -e);
        Sys S = system(save, g, gamma, work, dc.nt);
        double *rhs = S.pix + N, *cb = rhs + N, *x = cb + N, *r = x + N, *pv = r + N, *q2 = pv + N, *dg = q2 + N;
        // rhs = dx + sum_a dw_a (u_a[p] - u_a[p - e_a]) - G_A^T c_A,
        // c_p = H_p (dW D z)_p + dr_p n_p on non-flat entries, with
        // dr_p = dgamma alpha_p + gamma dalpha_p (dalpha_p = dparams[k + p]).
        double *cA = S.hbuf;
        S.rows([&](int64_t b) {
            for (int64_t p = b; p < b + n; ++p) rhs[p] = static_cast<double>(dx[p]);
            if (dparams)
                for (int a = 0; a < k; ++a)
                    if (dparams[a] != 0.0 && g.L[a] > 1) add_diff_row(g, a, sc * dparams[a], S.s.u + a * N, rhs, b);
            for (int a = 0; a < k; ++a) {
                double *ca = cA + a * N;
                for (int64_t p = b; p < b + n; ++p) {
                    const double wd = S.wa[a] != 0.0 ? S.gz[a * N + p] / S.wa[a] : 0.0;  // (D_a z)_p
                    ca[p] = (dparams && S.wa[a] != 0.0) ? sc * dparams[a] * wd : 0.0;
                }
            }
            S.hess_row(cA, b);  // H (dW D z)
            for (int64_t p = b; p < b + n; ++p) {
                if (S.s.flat[p]) continue;
                double nn = 0.0;
                for (int a = 0; a < k; ++a) nn += S.gz[a * N + p] * S.gz[a * N + p];
                const double nr = std::sqrt(nn);
                double dr = dgamma;
                if (S.alpha) dr = dgamma * S.alpha[p] + (dparams ? gamma * (sc * dparams[k + p]) : 0.0);
                for (int a = 0; a < k; ++a) cA[a * N + p] += dr * S.gz[a * N + p] / nr;
            }
        });
        S.rows([&](int64_t b) { S.sub_gradT_row(cA, rhs, b); });  // rhs -= G^T c
        S.region_sum<false>(rhs, cb);
        cg(S, cb, x, r, pv, q2, dg, dc);
        S.rows([&](int64_t b) {
            for (int64_t p = b; p < b + n; ++p) dz[p] = static_cast<T>(x[S.s.label[p]]);
        });
    }

    template <class T>
    void vjp_img(const char *save, const T *gin, T *gx, const int64_t *shape, double gamma, double *acc,
                 char *work, DerivCtl &dc) const noexcept {
        const Geom g(k, shape);
        const int64_t N = g.N, n = g.L[k - 1];
        const int e = static_cast<int>(*view(const_cast<char *>(save), N).ex);  // as in jvp_img
        const double sc = std::ldexp(1.0, e);
        gamma = std::ldexp(gamma, -e);
        Sys S = system(save, g, gamma, work, dc.nt);
        double *cb = S.pix + N, *x = cb + N, *r = x + N, *pv = r + N, *q2 = pv + N, *dg = q2 + N;
        S.rows([&](int64_t b) {
            for (int64_t p = b; p < b + n; ++p) S.pix[p] = static_cast<double>(gin[p]);
        });
        S.region_sum<false>(S.pix, cb);
        cg(S, cb, x, r, pv, q2, dg, dc);
        double *vpix = S.pix;
        S.rows([&](int64_t b) {
            for (int64_t p = b; p < b + n; ++p) {
                vpix[p] = x[S.s.label[p]];
                gx[p] = static_cast<T>(vpix[p]);
            }
        });
        // Parameter gradients from v = J^T g:
        //   d/dw_a:     -<D_a v, u_a> - sum_A [H_p (G v)_p]_a (D_a z)_p
        //   d/dgamma:   -sum_A alpha_p <(G v)_p, n_p>
        //   d/dalpha_p: -gamma <(G v)_p, n_p> on A, 0 elsewhere
        const int64_t P = nparams();
        double *gv = S.hbuf;
        S.grad(vpix, gv);  // G v
        for (int a = 0; a < k; ++a) {
            if (g.L[a] == 1) continue;
            const double *ua = S.s.u + a * N;
            const int64_t Sa = g.S[a];
            const double t = S.row_total([&](int64_t b, double &sum) {
                if (a == k - 1) {
                    for (int64_t j = 0; j + 1 < n; ++j) sum += (vpix[b + j + 1] - vpix[b + j]) * ua[b + j];
                } else if (g.has_next(b, a)) {
                    for (int64_t j = 0; j < n; ++j) sum += (vpix[b + j + Sa] - vpix[b + j]) * ua[b + j];
                }
            });
            acc[a] -= sc * t;
        }
        const double ggam = S.row_total([&](int64_t b, double &sum) {
            for (int64_t p = b; p < b + n; ++p) {
                if (S.s.flat[p]) continue;
                double nn = 0.0;
                for (int a = 0; a < k; ++a) nn += S.gz[a * N + p] * S.gz[a * N + p];
                const double nr = std::sqrt(nn);
                double t = 0.0;
                for (int a = 0; a < k; ++a) t += gv[a * N + p] * S.gz[a * N + p] / nr;
                if (S.alpha) {
                    sum += S.alpha[p] * t;
                    acc[k + p] -= sc * (gamma * t);
                } else {
                    sum += t;
                }
            }
        });
        acc[P] -= ggam;
        S.hess(gv);  // H G v
        for (int a = 0; a < k; ++a) {
            if (S.wa[a] == 0.0) continue;
            const double t = S.row_total([&](int64_t b, double &sum) {
                for (int64_t p = b; p < b + n; ++p)
                    if (!S.s.flat[p]) sum += gv[a * N + p] * (S.gz[a * N + p] / S.wa[a]);
            });
            acc[a] -= sc * t;
        }
    }
};

}  // namespace

fp_status make_tv_iso(int ndim, Vec w, fp_op **out, Vec alpha, const int64_t *shape) {
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
    if (!alpha.empty()) {
        double n = 1.0;
        for (int a = 0; a < ndim; ++a) n *= static_cast<double>(shape[a]);
        if (!(n < 2147483647.0)) return FP_ESHAPE;  // 32-bit region labels
        for (double v : alpha)
            if (!finite_nonneg(v)) return FP_EPARAM;
    }
    fp_op *t = nullptr;
    if (fp_status s = make_tv1d(Vec{1.0}, FP_TV1D_AUTO, &t)) return s;
    std::unique_ptr<fp_op> tv(t);
    *out = new TVIso(ndim, ww, std::move(tv), std::move(alpha), shape);
    return FP_OK;
}

}  // namespace fp
