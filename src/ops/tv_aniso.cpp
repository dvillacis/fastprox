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
//    The point certified and returned is instead the second block step of
//    the iteration, whose gap needs no further 1D pass (see adbpg).
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
// Restarts. Both methods are FISTA on a dual problem, a convex quadratic
// over a polyhedron, which grows quadratically away from its solution set
// (Hoffman's bound); restarted accelerated gradient methods then converge
// linearly. The restart is the gradient scheme of B. O'Donoghue and
// E. Candes, "Adaptive restart for accelerated gradient schemes", Found.
// Comput. Math. 15 (2015) 715-732, as in tv_iso.cpp. In 3D (32^3) it takes
// 3x fewer iterations to a gap of 1e-10 and 6x fewer to 1e-12.
//
// Polish. The primal z(y) of a dual iterate converges much more slowly than
// the dual: small errors in y leave many tiny jumps in z, each costing
// lam_a |jump| in the gap. The flat regions can instead be read off the dual
// (an edge whose 1D dual is strictly inside its bound has no jump at the
// solution), and the solution on given regions has a closed form (see
// try_polish). Its gap is checked against the same dual, so the
// certificate is unchanged; once the regions are right it is the exact
// solution and the gap is the dual's suboptimality alone, often orders of
// magnitude below the gap of z(y). The polish runs at the gap checks on a
// schedule (PolishSchedule) and ends the solve when its gap meets the target
// and is at most kPolishRatio of the iterate's.
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

// Iterations between restart tests, and between gap checks (in 2D, at most).
constexpr int kCheck = 4;
// A polish is retried once the gap of the iterate has fallen by this factor.
constexpr double kPolishDrop = 0.1;
// Edges whose 1D dual is within this relative distance of its bound count as
// boundary edges in the polish.
constexpr double kPolishEta = 1e-9;
// A polished point is taken only when its gap is at most this fraction of the
// iterate's. A larger gap means wrong regions: at loose tolerances such
// points met the target with 2-4x the iterate's error (gap ratios 0.03-0.4),
// while right regions give ratios below 4e-3.
constexpr double kPolishRatio = 0.01;
constexpr int64_t kEdgeChunk = 64;  // fibers per polish chunk along a strided axis

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

    // Working set: d, z, u, y_0..2, w_0..2, the polish scratch (3N doubles),
    // and partial sums per row or fiber block.
    size_t image_bytes(const int64_t *shape) const noexcept override {
        const Geom g(k, shape);
        int64_t P = g.rows();
        for (int a = 0; a < k; ++a) P = std::max(P, fiber_blocks(g, a));
        return (12 * static_cast<size_t>(g.N) + static_cast<size_t>(P) + 8) * sizeof(double);
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
        return fiber_thread_bytes(*tv, g);
    }

    // ---- fiber passes (image_util.hpp) ----------------------------------------

    template <class In, class Out>
    void pass(const Geom &g, int a, double lam, const ImageCall &c, In &&in, Out &&out) const {
        fiber_pass(*tv, g, a, lam, c, in, out);
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

    // Scratch of the polish: the candidate, per-component sums, and labels,
    // component sizes and union-find parents (32-bit); parent has room for
    // 2N entries (it is u), the second N holding the roots.
    struct PolishBuf {
        double *zp, *sum;
        int32_t *label, *cnt, *parent;
    };

    template <class T>
    int prox_img(const T *x, T *z, ImageCall &c) const noexcept {
        const Geom g(k, c.shape);
        const int64_t N = g.N;
        double *base = reinterpret_cast<double *>(c.work);
        double *d = base, *zz = d + N, *u = zz + N, *y = u + N, *wv = y + 3 * N;
        PolishBuf pb;
        pb.zp = wv + 3 * N;
        pb.sum = pb.zp + N;
        pb.label = reinterpret_cast<int32_t *>(pb.sum + N);
        pb.cnt = pb.label + N;
        pb.parent = reinterpret_cast<int32_t *>(u);  // u is free whenever the polish runs
        double *part = pb.sum + 2 * N;
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
                structure(g, c, none, -1, d, d, u);
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
        // The certificate needs a feasible dual, so a given one is made so.
        if (c.warm && na >= 2)
            for (int a = 0; a < k; ++a)
                if (lam[a] > 0.0) feasible_dual(g, c, a, lam[a], q + a * N);

        if (na == 0) {
            for (int64_t i = 0; i < N; ++i) z[i] = x[i];
            if (c.save) structure(g, c, lam, -1, d, d, u);
            return kRowOk;
        }
        if (na == 1) {  // exact
            const int a = act[0];
            double *qa = q + a * N;
            pass(g, a, lam[a], c, [&](int64_t o) { return d[o]; },
                 [&](int64_t o, double zo, double v) { zz[o] = zo; qa[o] = v - zo; });
            if (c.save) structure(g, c, lam, a, d, zz, u);
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
        // The iterate z is a block step along exact_axis, piecewise constant
        // there (exactly in 3D, up to rounding in 2D); a polished z is not.
        int exact_axis;
        if (na == 2) {
            exact_axis = adbpg(g, c, act, lam, d, zz, u, y, wv, pb, part, target);
        } else {
            exact_axis = 2;
            if (fdpg3(g, c, lam, d, zz, u, y, wv, pb, part, target)) exact_axis = -1;
        }
        if (c.save) structure(g, c, lam, exact_axis, d, zz, u);
        for (int64_t i = 0; i < N; ++i) z[i] = static_cast<T>(zz[i]);
        return kRowOk;
    }

    // Gradient restarts (O'Donoghue and Candes): every kCheck-th iteration the
    // update also returns <w - y+, y+ - y>; when it is positive the momentum
    // points uphill and the method restarts from y+ with t = 1, but never
    // sooner after the last restart than that one came after its predecessor.
    struct Restart {
        int since = 0, last = 0;
        bool operator()(bool test, double up) {
            ++since;
            if (!(test && up > 0.0 && since >= last)) return false;
            last = since;
            since = 0;
            return true;
        }
    };

    // When to polish: at the first gap check, then after each failed attempt
    // (iterate gap gp, candidate gap pz) once the iterate's gap has fallen by
    // kPolishDrop, or sooner if the ratio pz / gp of that attempt predicts
    // that the candidate meets the target. Where a polish costs more than an
    // iteration (2D: about 1.2 on one thread, against a third in 3D), only
    // ratios of at most kPolishRatio predict: larger ones are not taken, and
    // at lambda = 0.5 they stayed at 0.1-0.4 for 50-90 iterations. In 3D the
    // eager retries reach 1e-4 error 20% sooner, and in 2D the restricted ones
    // 2-16% sooner at 1e-3 and 1e-4. A last attempt at the end keeps the
    // candidate if its gap is at most kPolishRatio of the iterate's.
    struct PolishSchedule {
        bool eager;
        double next = std::numeric_limits<double>::infinity();
        explicit PolishSchedule(bool e) : eager(e) {}
        bool due(double gp) const { return gp <= next; }
        void done(double gp, double pz, double target) {
            next = gp * kPolishDrop;
            if (pz > 0.0 && pz < (eager ? 1.0 : kPolishRatio) * gp) next = std::max(next, target * gp / pz);
        }
    };

    // When to check the gap (ADBPG, where a check is one sweep over z):
    // after the first two iterations, then when the gap's rate of fall
    // between the last two checks predicts that it meets the target, and at
    // least every kCheck iterations.
    struct CheckSchedule {
        int next = 1, prev_it = -1;
        double prev = 0.0;
        void done(int it, double gp, double target) {
            int step = 1;
            if (prev_it >= 0) {
                step = kCheck;
                if (gp < prev && gp > target) {
                    const double n = std::log(target / gp) / std::log(gp / prev) * (it - prev_it);
                    if (n < kCheck) step = std::max(1, static_cast<int>(n));
                }
            }
            prev_it = it;
            prev = gp;
            next = it + step;
        }
    };

    // Two active axes a1 (the one with the smaller stride) and a2. Returns
    // the axis along which the returned z is a block step (piecewise constant
    // up to rounding), or -1.
    //
    // The iterate checked is the second block step of an iteration,
    // r = prox_{g2}(u - w) with u = prox_{g1}(d + w), not z(y+) =
    // prox_{g1}(d + y+): with q1 = d + w - u (the residual of the first
    // step) and q2 = -y+ = u - w - r (of the second), both feasible and
    // d - r = q1 + q2, the gap of r is a certificate, and it costs one sweep
    // over r instead of another 1D pass. On the 2D benchmark images its gap
    // is 0.2-1.7 times that of z(y+), mostly below 1 at lambda = 0.5. In
    // terms of y+, q1 = d + y+ - r.
    //
    // The z returned unpolished is d - q1 - q2 for the dual returned, which
    // is r up to rounding, so that a warm start from that dual rebuilds it
    // bit for bit; the polish depends on (d, q) alone as well.
    int adbpg(const Geom &g, ImageCall &c, const int *act, const double *lam, const double *d, double *z,
              double *u, double *y, double *wv, const PolishBuf &pb, double *part, double target) const {
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

        // The dual of (y, z), and zo = d - q1 - q2 from it.
        auto dual = [&](double *zo) {
            for (int64_t i = 0; i < N; ++i) {
                q1[i] = d[i] + y[i] - z[i];
                q2[i] = -y[i];
                zo[i] = d[i] - q1[i] - q2[i];
            }
        };
        double t = 1.0;
        int it = 0;
        Restart restart;
        PolishSchedule sched(false);
        CheckSchedule checks;
        bool polished = false;
        // A polish from the dual of (y, z). Not at the first checks of a cold
        // start, whose candidates never had the right regions (a warm start's
        // often have them from the start), and always at max_iter.
        auto polish = [&](double gz) {
            const bool last = it == c.max_iter;
            if (gz <= target || !(last || ((c.warm || it >= kCheck) && sched.due(gz)))) return gz;
            if (it > 0) dual(z);  // at it = 0 it is already the dual of z
            double pz;
            if (try_polish(g, c, lam, d, z, pb, part, last ? gz : target, gz, pz)) polished = true;
            else sched.done(gz, pz, target);
            return gz;
        };
        // A warm start first tries d - q1 - q2 for the dual given and its
        // polish, which give the z of a solve of the same d from its dual. If
        // neither certifies, it continues as from z(y) = prox_{g1}(d + y) and
        // its dual (with y = 0 when max_iter is 0): for a nearby d, q1 is then
        // fitted to the new data, which certifies more often.
        double gp = std::numeric_limits<double>::infinity();
        if (c.warm) {
            for (int64_t i = 0; i < N; ++i) z[i] = d[i] - q1[i] - q2[i];
            gp = polish(gap(g, c, lam, d, z, part));
        }
        if (!(gp <= target) && (c.warm || c.max_iter == 0)) {
            pass(g, a1, lam[a1], c, [&](int64_t o) { return d[o] + y[o]; },
                 [&](int64_t o, double zo, double) { z[o] = zo; });
            dual(z);
            gp = polish(gap(g, c, lam, d, z, part));
        }
        while (!(gp <= target) && it < c.max_iter) {
            pass(g, a1, lam[a1], c, [&](int64_t o) { return d[o] + wv[o]; },
                 [&](int64_t o, double zo, double) { u[o] = zo; });
            const double tn = fista_t(t);
            const double beta = (t - 1.0) / tn;
            t = tn;
            auto upd = [&](int64_t o, double zo, double v) {
                const double yn = zo - v;
                const double r = (wv[o] - yn) * (yn - y[o]);
                wv[o] = yn + beta * (yn - y[o]);
                y[o] = yn;
                z[o] = zo;
                return r;
            };
            auto in = [&](int64_t o) { return u[o] - wv[o]; };
            const bool test = it % kCheck == 0;
            double up = 0.0;
            if (test) up = fiber_pass_sum(*tv, g, a2, lam[a2], c, part, in, upd);
            else pass(g, a2, lam[a2], c, in, [&](int64_t o, double zo, double v) { upd(o, zo, v); });
            if (restart(test, up)) {
                std::memcpy(wv, y, static_cast<size_t>(N) * sizeof(double));
                t = 1.0;
            }
            ++it;
            if (it == checks.next || it == c.max_iter) {
                gp = polish(gap(g, c, lam, d, z, part));
                checks.done(it, gp, target);
            }
        }
        int exact = polished || it == 0 ? -1 : a2;
        if (!polished && it > 0) {
            // d - q1 - q2 in place of r, unless rounding alone takes its gap
            // over the target.
            dual(pb.zp);
            const double gw = gap(g, c, lam, d, pb.zp, part);
            if (gw <= target || !(gp <= target)) {
                std::memcpy(z, pb.zp, static_cast<size_t>(N) * sizeof(double));
                gp = gw;
            }
        }
        c.gap = std::max(gp, 0.0);
        c.iters = it;
        c.converged = gp <= target;
        return exact;
    }

    // Three active axes (k == 3). y holds y_0..2 (y_i = -q_i), wv the
    // extrapolated w_0..2.
    bool fdpg3(const Geom &g, ImageCall &c, const double *lam, const double *d, double *z, double *u,
               double *y, double *wv, const PolishBuf &pb, double *part, double target) const {
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
        auto dual = [&]() {
            double *q = c.dual;
            for (int64_t i = 0; i < N; ++i) {
                q[i] = -y[i];
                q[N + i] = -y[N + i];
                q[2 * N + i] = d[i] + y[i] + y[N + i] - z[i];
            }
        };
        double t = 1.0;
        int it = 0;
        Restart restart;
        PolishSchedule sched(true);
        bool polished = false;
        // At a gap check: the gap of z(y), and a polish when one is due.
        auto check = [&]() {
            double gz = primal();
            if (!(gz <= target) && sched.due(gz)) {
                dual();
                double pz;
                if (try_polish(g, c, lam, d, z, pb, part, target, gz, pz)) polished = true;
                else sched.done(gz, pz, target);
            }
            return gz;
        };
        double gp = c.warm ? check() : std::numeric_limits<double>::infinity();
        while (!(gp <= target) && it < c.max_iter) {
            for (int64_t i = 0; i < N; ++i) u[i] = d[i] + wv[i] + wv[N + i] + wv[2 * N + i];
            const double tn = fista_t(t);
            const double beta = (t - 1.0) / tn;
            t = tn;
            const bool test = it % kCheck == 0;
            double up = 0.0;
            for (int a = 0; a < 3; ++a) {
                double *ya = y + a * N, *wa = wv + a * N;
                auto upd = [&](int64_t o, double zo, double v) {
                    // y+ = w - u/p + zo/p with v = u - p w, i.e. (zo - v) / p.
                    const double yn = (zo - v) / p;
                    const double r = (wa[o] - yn) * (yn - ya[o]);
                    wa[o] = yn + beta * (yn - ya[o]);
                    ya[o] = yn;
                    return r;
                };
                auto in = [&](int64_t o) { return u[o] - p * wa[o]; };
                if (test) up += fiber_pass_sum(*tv, g, a, p * lam[a], c, part, in, upd);
                else pass(g, a, p * lam[a], c, in, [&](int64_t o, double zo, double v) { upd(o, zo, v); });
            }
            if (restart(test, up)) {
                std::memcpy(wv, y, static_cast<size_t>(3 * N) * sizeof(double));
                t = 1.0;
            }
            ++it;
            if (it % kCheck == 0 || it == c.max_iter) gp = check();
        }
        if (!std::isfinite(gp)) gp = primal();
        if (!polished) {
            dual();
            double pz;
            if (!(gp <= target)) polished = try_polish(g, c, lam, d, z, pb, part, gp, gp, pz);
        }
        c.gap = std::max(gp, 0.0);
        c.iters = it;
        c.converged = gp <= target;
        return polished;
    }

    // ---- warm starts -----------------------------------------------------------

    // Makes q_a, a warm-start dual for axis a, feasible: q_a = D_a^T p_a with
    // |p_e| <= lam on every edge, that is, along each fiber the running sums
    // -p_e of q_a stay within lam and the full sum is 0. A fiber that already
    // satisfies this up to rounding (tolerance 1e-10 lam + 16 eps sum|q_a|)
    // is left as it is, so the dual of an earlier solve restarts it bit for
    // bit; any other is rebuilt from its running sums clipped to [-lam, lam]
    // (non-finite ones set to 0). Without this, an infeasible dual could
    // certify a wrong point: q_1 = 0, q_2 = x - mean(x) gives z = mean(x)
    // with gap 0 in 2D. Fibers are independent, so chunks run in parallel.
    static void feasible_dual(const Geom &g, const ImageCall &c, int a, double lam, double *__restrict qa) {
        constexpr int64_t W = kEdgeChunk;
        const int64_t s = g.S[a], L = g.L[a], outer = g.N / (L * s);
        const int64_t per = s == 1 ? 1 : (s + W - 1) / W;
        parallel_for(c.nt, outer * per, std::max<int64_t>(1, kChunk / (L * std::min(s, W))),
                     [&](int, int64_t u0, int64_t u1) {
            double acc[kEdgeChunk], mass[kEdgeChunk], peak[kEdgeChunk];
            for (int64_t u = u0; u < u1; ++u) {
                const int64_t oo = u / per, r0 = (u % per) * W;
                const int64_t w = std::min(W, s - r0);
                const int64_t b = oo * L * s + r0;
                for (int64_t r = 0; r < w; ++r) acc[r] = mass[r] = peak[r] = 0.0;
                for (int64_t l = 0; l < L; ++l) {
                    const int64_t o = b + l * s;
                    for (int64_t r = 0; r < w; ++r) {
                        acc[r] += qa[o + r];
                        mass[r] += std::fabs(qa[o + r]);
                        // NaN in a fiber makes peak NaN and Inf makes mass Inf,
                        // and either fails the test below.
                        const double e = l + 1 < L ? std::fabs(acc[r]) : 0.0;
                        peak[r] = e > peak[r] || e != e ? e : peak[r];
                    }
                }
                for (int64_t r = 0; r < w; ++r) {
                    const double tau = 1e-10 * lam + 16.0 * std::numeric_limits<double>::epsilon() * mass[r];
                    if (std::isfinite(mass[r]) && peak[r] <= lam + tau && std::fabs(acc[r]) <= tau) continue;
                    // q_i = p_{i-1} - p_i, p_e = clip(-(running sum)).
                    double run = 0.0, prev = 0.0;
                    for (int64_t l = 0; l < L; ++l) {
                        double &qi = qa[b + l * s + r];
                        run += qi;
                        double pe = 0.0;
                        if (l + 1 < L && std::isfinite(run)) pe = std::clamp(-run, -lam, lam);
                        qi = prev - pe;
                        prev = pe;
                    }
                }
            }
        });
    }

    // ---- polish -------------------------------------------------------------

    // For every edge e = (i, j = i + S[a]) along axis a: the 1D dual on it,
    // recovered from q_a = D_a^T p_a as p_e = -(sum of q_a along the fiber up
    // to i); bit a of flag[i] when |p_e| < lam (1 - eta), and otherwise
    // lam sign(p_e) added to v[i] and taken from v[j]. Each
    // entry is touched only by the edges of its own fiber, so the fixed chunks
    // of fibers run in parallel.
    static void mark_edges(const Geom &g, const ImageCall &c, int a, double lam, double eta, const double *qa,
                           uint8_t *flag, double *v) {
        constexpr int64_t W = kEdgeChunk;
        const int64_t s = g.S[a], L = g.L[a], outer = g.N / (L * s);
        const double in = lam * (1.0 - eta);
        const uint8_t bit = static_cast<uint8_t>(1u << a);
        const int64_t per = s == 1 ? 1 : (s + W - 1) / W;
        parallel_for(c.nt, outer * per, std::max<int64_t>(1, kChunk / (L * std::min(s, W))),
                     [&](int, int64_t u0, int64_t u1) {
            for (int64_t u = u0; u < u1; ++u) {
                const int64_t oo = u / per, r0 = (u % per) * W;
                const int64_t b = oo * L * s + r0;
                mark_chunk(qa + b, flag + b, v + b, std::min(W, s - r0), L, s, lam, in, bit);
            }
        });
    }
    // One chunk of w fibers of length L and stride s starting at q, flag, v.
    // A separate function, so that the pointers are restrict here: through a
    // lambda's captures they are not, and the loop did not vectorise (5.6x
    // slower).
    static void mark_chunk(const double *__restrict q, uint8_t *__restrict flag, double *__restrict v, int64_t w,
                           int64_t L, int64_t s, double lam, double in, uint8_t bit) {
        double acc[kEdgeChunk];
        for (int64_t r = 0; r < w; ++r) acc[r] = 0.0;
        for (int64_t l = 0; l + 1 < L; ++l) {
            const int64_t o = l * s;
            for (int64_t r = 0; r < w; ++r) {
                acc[r] += q[o + r];
                const double pe = -acc[r];
                const bool inside = std::fabs(pe) < in;
                const double ls = inside ? 0.0 : std::copysign(lam, pe);
                flag[o + r] |= inside ? bit : uint8_t{0};
                v[o + r] += ls;
                v[o + r + s] -= ls;
            }
        }
    }

    // A candidate from the dual q = c.dual. An edge whose 1D dual lies
    // strictly inside its bound, |p_e| < lam_a (1 - eta), is flat at the
    // solution (complementary slackness: a jump forces p_e = lam_a
    // sign(jump)), so the solution is constant on the connected components C
    // of those edges. With p_e replaced by lam_a sign(p_e) on the other edges,
    // the optimality condition z - d + sum_a D_a^T p_a = 0 summed over C
    // gives (the edges inside C cancel)
    //     zp_C = (sum_C d + sum_{e=(i,j): i in C} lam_e sign(p_e) - sum_{e=(i,j): j in C} lam_e sign(p_e)) / |C|,
    // which is the exact solution once the components are right. Its gap
    // with the same dual,
    //     P(zp) - D(q),   D(q) = <d, s> - |s|^2 / 2,  s = sum_a q_a,
    // still bounds |zp - z*|^2 / 2. pz is that gap; when it is at most accept
    // and at most kPolishRatio gp, z and gp take the candidate and it returns
    // true. The candidate and pz depend on d and q alone, so a warm start
    // from the dual of a polished solve polishes to the same bits. Uses u
    // (the union-find parents) and the polish scratch.
    bool try_polish(const Geom &g, const ImageCall &c, const double *lam, const double *d, double *z,
                    const PolishBuf &pb, double *part, double accept, double &gp, double &pz) const {
        const int64_t N = g.N;
        int32_t *parent = pb.parent, *label = pb.label, *cnt = pb.cnt;
        double *zp = pb.zp, *sum = pb.sum;
        // Inside edges as bit a of flag[i] for the edge (i, i + S[a]), and in
        // zp, d plus the lam_e sign(p_e) of the other edges at each entry (+ at
        // i, - at j).
        uint8_t *flag = reinterpret_cast<uint8_t *>(label);
        std::memset(flag, 0, static_cast<size_t>(N));
        std::memcpy(zp, d, static_cast<size_t>(N) * sizeof(double));
        for (int a = 0; a < g.k; ++a)
            if (lam[a] > 0.0) mark_edges(g, c, a, lam[a], kPolishEta, c.dual + a * N, flag, zp);
        // Components by union-find with union by the smaller root, so every
        // root is the smallest index of its component and every parent
        // pointer points backwards. Along the contiguous axis the components
        // are runs, linked directly; run[i] (in cnt) is the start of i's run,
        // and an edge along another axis is redundant when the edge before it
        // in the row joins the same two runs.
        //
        // In parallel: the image is cut into slabs of whole units of S[0]
        // (planes in 3D, rows in 2D), fixed by the shape, so only axis-0
        // edges cross slabs. Each slab's edges are joined by its own thread
        // (finds stay inside the slab, since pointers point backwards), then
        // the crossing edges in one serial pass. The components, and so the
        // labels and sums below, are those of the serial order.
        const int last = g.k - 1;
        const int64_t n = g.L[last];
        const uint8_t bl = static_cast<uint8_t>(1u << last);
        int32_t *run = cnt;
        int32_t *root = parent + N;  // second half of u
        row_for(g, c, [&](int64_t r) {
            const int64_t b = r * n;
            parent[b] = static_cast<int32_t>(b);
            for (int64_t j = 1; j < n; ++j)
                parent[b + j] = (flag[b + j - 1] & bl) ? parent[b + j - 1] : static_cast<int32_t>(b + j);
        });
        std::memcpy(run, parent, static_cast<size_t>(N) * sizeof(int32_t));
        auto find = [&](int64_t i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        // The edge (i, i + s) along axis a, unless the skip rule applies.
        auto join = [&](int64_t i, int64_t j, int64_t s, uint8_t bit) {
            if (j > 0 && (flag[i - 1] & bit) && run[i] == run[i - 1] && run[i + s] == run[i + s - 1]) return;
            const int64_t ri = find(i), rj = find(i + s);
            if (ri == rj) return;
            if (ri < rj) parent[rj] = static_cast<int32_t>(ri);
            else parent[ri] = static_cast<int32_t>(rj);
        };
        // At most 64 slabs of at least 4 units, so that at most a quarter
        // of the axis-0 edges cross.
        const int64_t units = g.L[0], unit = g.S[0];
        const int64_t per = std::max<int64_t>(4, (units + 63) / 64);
        const int64_t nslab = (units + per - 1) / per;
        auto slab_lo = [&](int64_t t) { return std::min(units, t * per) * unit; };
        parallel_for(c.nt, nslab, 1, [&](int, int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                const int64_t lo = slab_lo(t), hi = slab_lo(t + 1);
                for (int a = 0; a < last; ++a) {
                    if (lam[a] == 0.0) continue;
                    const uint8_t bit = static_cast<uint8_t>(1u << a);
                    const int64_t s = g.S[a];
                    for (int64_t i = lo; i < hi; ++i)
                        if ((flag[i] & bit) && i + s < hi) join(i, i % n, s, bit);
                }
            }
        });
        // Crossing edges: axis 0 from the last unit of each slab. Their
        // components are marked crossing (bit 7 of flag at the final root).
        constexpr uint8_t kCross = 0x80;
        if (lam[0] > 0.0 && nslab > 1) {
            for (int64_t t = 0; t + 1 < nslab; ++t)
                for (int64_t i = slab_lo(t + 1) - unit; i < slab_lo(t + 1); ++i)
                    if (flag[i] & 1u) join(i, i % n, unit, 1u);
            for (int64_t t = 0; t + 1 < nslab; ++t)
                for (int64_t i = slab_lo(t + 1) - unit; i < slab_lo(t + 1); ++i)
                    if (flag[i] & 1u) flag[find(i)] |= kCross;
        }
        // root[i], as ~r for crossing components (reads parent only), and
        // the number of roots before each slab (in part).
        int64_t *roots_in = reinterpret_cast<int64_t *>(part);
        parallel_for(c.nt, nslab, 1, [&](int, int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                int64_t m = 0;
                for (int64_t i = slab_lo(t); i < slab_lo(t + 1); ++i) {
                    int64_t r = i;
                    while (parent[r] != r) r = parent[r];
                    m += r == i;
                    root[i] = (flag[r] & kCross) ? ~static_cast<int32_t>(r) : static_cast<int32_t>(r);
                }
                roots_in[t + 1] = m;
            }
        });
        roots_in[0] = 0;
        for (int64_t t = 0; t < nslab; ++t) roots_in[t + 1] += roots_in[t];
        const int64_t nc = roots_in[nslab];
        // Labels in index order of the roots (flag is no longer needed, and
        // shares label's memory): roots first, then the others, which read
        // their root's label.
        parallel_for(c.nt, nslab, 1, [&](int, int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                int64_t m = roots_in[t];
                for (int64_t i = slab_lo(t); i < slab_lo(t + 1); ++i) {
                    const int32_t r = root[i] < 0 ? ~root[i] : root[i];
                    if (r == i) label[i] = static_cast<int32_t>(m++);
                }
            }
        });
        parallel_for(c.nt, nslab, 1, [&](int, int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t)
                for (int64_t i = slab_lo(t); i < slab_lo(t + 1); ++i) {
                    const int32_t r = root[i] < 0 ? ~root[i] : root[i];
                    if (r != i) label[i] = label[r];
                }
        });
        // Region sums in index order: a component that does not cross lies in
        // one slab and is summed by its thread; the crossing ones serially.
        std::fill(sum, sum + nc, 0.0);
        std::fill(cnt, cnt + nc, 0);
        // The crossing entries of slab t are listed from parent[slab_lo(t)]
        // on (parent is no longer needed), their number in xin[t].
        int64_t *xin = roots_in + nslab + 1;
        parallel_for(c.nt, nslab, 1, [&](int, int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                int32_t *list = parent + slab_lo(t);
                int64_t m = 0;
                for (int64_t i = slab_lo(t); i < slab_lo(t + 1); ++i) {
                    if (root[i] < 0) {
                        list[m++] = static_cast<int32_t>(i);
                        continue;
                    }
                    sum[label[i]] += zp[i];
                    ++cnt[label[i]];
                }
                xin[t] = m;
            }
        });
        for (int64_t t = 0; t < nslab; ++t)
            for (int64_t m = 0; m < xin[t]; ++m) {
                const int64_t i = parent[slab_lo(t) + m];
                sum[label[i]] += zp[i];
                ++cnt[label[i]];
            }
        row_for(g, c, [&](int64_t r) {
            for (int64_t i = r * n; i < (r + 1) * n; ++i) zp[i] = sum[label[i]] / cnt[label[i]];
        });

        const double *q = c.dual;
        pz = row_sum(g, c, part, [&](int64_t r) {
            const int64_t b = r * n;
            double acc = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                const double e = zp[b + j] - d[b + j];
                double sj = q[b + j];
                for (int a = 1; a < g.k; ++a) sj += q[a * N + b + j];
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
        if (!(pz <= accept && pz <= kPolishRatio * gp)) return false;
        std::memcpy(z, zp, static_cast<size_t>(N) * sizeof(double));
        gp = pz;
        return true;
    }

    // ---- structure ----------------------------------------------------------

    // Saved for jvp/vjp: the components of the graph of flat edges of z
    // (|z_j - z_i| <= tau along an active axis; equal up to rounding along
    // exact_axis, where z is a block step and so piecewise constant by
    // construction, or d - q1 - q2 for the q of one) and the boundary sign
    // sums S_{C,a} on every axis. Uses u as scratch.
    void structure(const Geom &g, ImageCall &c, const double *lam, int exact_axis, const double *d,
                   const double *z, double *u) const {
        const int64_t N = g.N;
        const View v = view(c.save, N);
        int32_t *label = v.label;
        double *inv = v.inv, *S = v.S;
        int32_t *parent = reinterpret_cast<int32_t *>(u);

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
        // The entries of d - q1 - q2 are within a few ulps of
        // max(|d|, |q1|, |q2|), and |q_a| <= 2 lam_a.
        const double ulps = 64.0 * 2.2e-16 * (scale + 2.0 * (lam[0] + lam[1] + lam[2]));
        for (int a = 0; a < g.k; ++a) {
            if (lam[a] == 0.0) continue;
            const double tau = a == exact_axis ? ulps : 8.0 * rms + 64.0 * 2.2e-16 * scale;
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
        for (int64_t i = 0; i < N; ++i) inv[label[i]] += 1.0;
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
        *v.ncomp = nc;
    }

    // ---- derivatives (structure saved by prox_img) --------------------------

    // dz_C = mean_C(dx) - (1/|C|) sum_a (dgamma w_a + gamma dw_a) S_{C,a}.
    template <class T>
    void jvp_img(const char *save, const T *dx, T *dz, const int64_t *shape, double gamma, double dgamma,
                 const double *dparams, char *work, DerivCtl &) const noexcept {
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
                 char *work, DerivCtl &) const noexcept {
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
