// Projection onto the 1D TV ball {z : sum_e w_e |z_{e+1} - z_e| <= alpha} along
// the axis, as a level set (A. Beck, First-Order Methods in Optimization, SIAM
// 2017, Theorem 6.30, p. 149): x itself when TV(x) <= alpha, else
// prox_{gamma* TV}(x), with gamma* a root of phi(gamma) = TV(prox_{gamma TV}(x)) - alpha.
// phi is nonincreasing (Beck), and for 1D TV piecewise linear: on a fixed
// segment structure every segment value is affine in gamma (tv1d.cpp,
// |B| z_B = sum_B x + gamma (s_l w_l - s_r w_r)), and so is TV(z) while the
// jump signs hold. Newton steps with the slope of the current structure are
// therefore exact once they reach the root's piece; they run inside a
// bracket [L, R] that shrinks with every evaluation, with bisection when a
// step leaves it. The first step is Newton from gamma = 0+, and while
// TV(z) > 2 alpha the steps are taken on log TV against log gamma. R starts at gamma_max, where every piece between zero
// weights is constant: max_e |U_e| / w_e with U_e the cumulative sum of x
// minus the piece mean. alpha = 0 gives that constant solution.
//
// Derivatives. On the selected structure z = prox_{gamma TV}(x; w) with
// gamma = gamma*(x, w, alpha) fixed by TV(z; w) = alpha:
//   dgamma = (dalpha - sum_e |Dz_e| dw_e - <q, dz0>) / <q, z_gamma>,
//   dz = dz0 + z_gamma dgamma,
// where dz0 is the 1D TV JVP at fixed gamma, z_gamma its derivative in gamma,
// and q = D^T (w sign(Dz)) the gradient of TV at z. The VJP is the adjoint,
// one 1D TV VJP of g - beta q with beta = <g, z_gamma> / <q, z_gamma>. With no
// jump (z constant per piece) <q, z_gamma> = 0 and dz = dz0. gamma* is read
// back from the residual at the first jump, sum_{i <= e} (x_i - z_i) = s_e
// gamma w_e (tv1d.cpp), so the derivatives need no second solve.
//
// Kinks: TV(x) = alpha keeps x (inside: identity), as for the other balls.
//
// Scale. The projection is positively homogeneous, P_alpha(c x) = c P_{alpha/c}(x),
// so a row with max |x_i| outside [2^-960, 2^960] is solved as x 2^-k with radius
// alpha 2^-k and the result scaled back by 2^k (as in hyperplane.cpp): the piece
// sums of gamma_max and the TV of z overflowed from max |x_i| ~ 5e305 (nightly
// fuzz, the conjugate at sigma = 1.8e-306).
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

// The 1D TV structure codes (tv1d.cpp): code[i], i >= 1, for the edge i - 1.
constexpr uint8_t kDown = 1, kUp = 2;
// code[0] is unused by 1D TV; here it marks x inside the ball.
constexpr uint8_t kInside = 8;

struct TVBall final : RowOp<TVBall> {
    double alpha;
    Vec w;  // as for 1D TV: length 1 or n - 1
    std::unique_ptr<fp_op> tv;
    static constexpr int kMaxEvals = 200;

    TVBall(double a, Vec ww, fp_op *t) : alpha(a), w(std::move(ww)), tv(t) {}

    const char *name() const noexcept override { return "tv1d_ball"; }
    double step_scale() const noexcept override { return 0.0; }
    int64_t nparams() const noexcept override { return 1 + static_cast<int64_t>(w.size()); }
    fp_status check_n(int64_t n) const noexcept override { return tv->check_n(n); }
    bool needs_distinct() const noexcept override { return true; }
    // The 1D TV scratch, then n code bytes (padded to 8), then 3 n + 2 doubles.
    static size_t pad(size_t b) noexcept { return (b + 7) & ~size_t(7); }
    size_t tv_bytes(int64_t n) const noexcept { return pad(tv->scratch_bytes(n)); }
    size_t scratch_bytes(int64_t n) const noexcept override {
        return tv_bytes(n) + pad(static_cast<size_t>(n)) + static_cast<size_t>(3 * n + 2) * sizeof(double);
    }
    double we(int64_t e) const noexcept { return w.size() == 1 ? w[0] : w[e]; }
    // The k-th n-entry array of T after the code bytes; the accumulators
    // (k = 2) take n + 2 doubles.
    template <class T>
    T *buffer(void *scr, int64_t n, int k) const noexcept {
        char *p = static_cast<char *>(scr) + tv_bytes(n) + pad(static_cast<size_t>(n));
        return reinterpret_cast<T *>(p + static_cast<size_t>(k * n) * sizeof(double));
    }

    template <class T>
    static int sgn_diff(const T *z, int64_t e) noexcept {
        return z[e + 1] > z[e] ? 1 : (z[e + 1] < z[e] ? -1 : 0);
    }
    template <class T>
    double tv_value(const T *z, int64_t n) const noexcept {
        double s = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e)
            s += we(e) * std::fabs(static_cast<double>(z[e + 1]) - static_cast<double>(z[e]));
        return s;
    }
    // <q, v> = sum_e w_e sign(Dz_e) (v_{e+1} - v_e).
    template <class T, class V>
    double tv_lin(const T *z, const V *v, int64_t n) const noexcept {
        double s = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const int sg = sgn_diff(z, e);
            if (sg) s += sg * we(e) * (static_cast<double>(v[e + 1]) - static_cast<double>(v[e]));
        }
        return s;
    }

    // gamma_max: the smallest step at which every piece is constant.
    template <class T>
    double gamma_max(const T *x, int64_t n) const noexcept {
        double gm = 0.0;
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e < n - 1 && we(e) != 0.0) continue;
            double m = 0.0;  // piece [s, e]
            for (int64_t i = s; i <= e; ++i) m += static_cast<double>(x[i]);
            m /= static_cast<double>(e - s + 1);
            double u = 0.0;
            for (int64_t i = s; i < e; ++i) {
                u += static_cast<double>(x[i]) - m;
                gm = std::max(gm, std::fabs(u) / we(i));
            }
            s = e + 1;
        }
        return gm;
    }

    bool has_value() const noexcept override { return true; }
    // The indicator of sum_e w_e |x_{e+1} - x_e| <= alpha.
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        double s = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) s += we(e) * std::fabs(x[e + 1] - x[e]);
        return indicator(s - alpha, value_atol(x, n, tol));
    }

    // The exponent k of the scale 2^-k for a row of magnitude M = max |x_i|, or
    // 0 when M is in [2^-960, 2^960], where no sum of n < 2^62 terms over- or
    // underflows. It is 0 for float, whose range lies inside.
    static int ball_scale(double M) noexcept {
        if (!(M > 0.0) || (M >= 0x1p-960 && M <= 0x1p960)) return 0;
        int k;
        std::frexp(M, &k);
        return k;
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double, void *scr, uint8_t *code) const noexcept {
        uint8_t *c = code ? code : reinterpret_cast<uint8_t *>(static_cast<char *>(scr) + tv_bytes(n));
        if (!all_finite(x, n)) {
            bool nan;
            if (scan_row(x, n, nan) != kRowOk) return kRowNonFinite;
            fill_nan(z, n);
            std::memset(c, 0, static_cast<size_t>(n));
            return kRowOk;
        }
        double M = 0.0;
        for (int64_t i = 0; i < n; ++i) M = std::max(M, std::fabs(static_cast<double>(x[i])));
        const int k = ball_scale(M);
        if (k == 0) return solve(x, z, n, alpha, scr, c);
        T *xs = buffer<T>(scr, n, 2);  // the vjp's accumulators, unused here
        for (int64_t i = 0; i < n; ++i) xs[i] = static_cast<T>(std::ldexp(static_cast<double>(x[i]), -k));
        const int r = solve(xs, z, n, std::ldexp(alpha, -k), scr, c);
        if (c[0] == kInside)  // x itself, not x 2^-k 2^k, which rounds where x 2^-k is subnormal
            std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
        else
            for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(std::ldexp(static_cast<double>(z[i]), k));
        return r;
    }

    // The projection onto the ball of radius alpha (shadowing the member: the
    // rescaled radius in prox), on a row of moderate scale.
    template <class T>
    int solve(const T *x, T *z, int64_t n, double alpha, void *scr, uint8_t *c) const noexcept {
        void *tscr = scr;
        T *zero = buffer<T>(scr, n, 0), *zg = buffer<T>(scr, n, 1);
        if (tv_value(x, n) <= alpha) {
            std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
            std::memset(c, 0, static_cast<size_t>(n));
            c[0] = kInside;
            return kRowOk;
        }
        double L = 0.0, R = gamma_max(x, n);
        // phi(R) = -alpha <= 0; the root lies in (0, R].
        auto eval = [&](double g) {
            tv->prox_row(x, z, n, g, tscr, c);
            return tv_value(z, n) - alpha;
        };
        if (alpha == 0.0) {
            eval(R);
            c[0] = 0;
            return kRowOk;
        }
        const double eps = 4.0 * std::numeric_limits<T>::epsilon();
        for (int64_t i = 0; i < n; ++i) zero[i] = T(0);
        // First step: Newton from gamma = 0+, on the structure of the identity
        // (1D TV at gamma = 0 returns x with that structure). phi is close to
        // convex in practice, so the steps then approach the root from the
        // left; a secant through gamma_max lands right of it, where the steps
        // fall back to bisection over many orders of magnitude.
        const double phi0 = eval(0.0);
        tv->jvp_row(x, z, c, zero, zg, n, 0.0, 1.0, nullptr, tscr);
        const double slope0 = tv_lin(z, zg, n);
        double g = slope0 < 0.0 ? -phi0 / slope0 : 0.5 * R;
        if (!(g > L && g < R)) g = 0.5 * (L + R);
        for (int it = 0; it < kMaxEvals; ++it) {
            const double ph = eval(g);
            if (ph == 0.0) break;
            if (ph > 0.0) L = g; else R = g;
            // Slope of phi on the current structure: <q, z_gamma>.
            tv->jvp_row(x, z, c, zero, zg, n, g, 1.0, nullptr, tscr);
            const double slope = tv_lin(z, zg, n);
            // Far above alpha, TV(z) behaves like a power of gamma: a Newton
            // step on log TV against log gamma (exact for a power law) takes
            // 7-8 solves where plain Newton took 15-17 at alpha = 1e-4 TV(x).
            const double f = ph + alpha;
            double next;
            if (!(slope < 0.0)) next = 0.5 * (L + R);
            else if (f > 2.0 * alpha) next = g * std::exp(std::log(alpha / f) * f / (g * slope));
            else next = g - ph / slope;
            if (!(next > L && next < R)) next = 0.5 * (L + R);
            if (std::fabs(next - g) <= eps * g || R - L <= eps * R) {
                if (next != g) eval(next);
                break;
            }
            g = next;
        }
        c[0] = 0;
        return kRowOk;
    }

    // gamma* from the residual at the first jump; 0 without jumps.
    template <class T>
    double gamma_star(const T *x, const T *z, const uint8_t *code, int64_t n) const noexcept {
        double u = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            u += static_cast<double>(x[e]) - static_cast<double>(z[e]);
            const uint8_t c = code[e + 1];
            if ((c == kDown || c == kUp) && we(e) > 0.0) return std::fabs(u) / we(e);
        }
        return 0.0;
    }

    template <class T>
    void jvp(const T *x, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n, double,
             double, const double *dp, void *scr) const noexcept {
        if (code[0] == kInside) {
            for (int64_t i = 0; i < n; ++i) dz[i] = dx[i];
            return;
        }
        void *tscr = scr;
        T *zero = buffer<T>(scr, n, 0), *zg = buffer<T>(scr, n, 1);
        for (int64_t i = 0; i < n; ++i) zero[i] = T(0);
        const double g = gamma_star(x, z, code, n);
        tv->jvp_row(x, z, code, dx, dz, n, g, 0.0, dp ? dp + 1 : nullptr, tscr);
        tv->jvp_row(x, z, code, zero, zg, n, g, 1.0, nullptr, tscr);  // z_gamma
        const double kappa = tv_lin(z, zg, n);
        if (kappa == 0.0) return;
        double rhs = dp ? dp[0] : 0.0;
        if (dp)
            for (int64_t e = 0; e + 1 < n; ++e)
                rhs -= std::fabs(static_cast<double>(z[e + 1]) - static_cast<double>(z[e])) *
                       dp[1 + (w.size() == 1 ? 0 : e)];
        rhs -= tv_lin(z, dz, n);
        const double dg = rhs / kappa;
        for (int64_t i = 0; i < n; ++i)
            dz[i] = static_cast<T>(static_cast<double>(dz[i]) + dg * static_cast<double>(zg[i]));
    }

    template <class T>
    void vjp(const T *x, const T *z, const uint8_t *code, const T *gv, T *gx, int64_t n, double,
             double *acc, void *scr) const noexcept {
        if (code[0] == kInside) {
            for (int64_t i = 0; i < n; ++i) gx[i] = gv[i];
            return;
        }
        const int64_t nw = static_cast<int64_t>(w.size());
        void *tscr = scr;
        T *h = buffer<T>(scr, n, 0), *zg = buffer<T>(scr, n, 1);
        double *ia = buffer<double>(scr, n, 2);  // nw + 1 <= n + 1 accumulators
        const double g = gamma_star(x, z, code, n);
        for (int64_t i = 0; i < n; ++i) h[i] = T(0);
        tv->jvp_row(x, z, code, h, zg, n, g, 1.0, nullptr, tscr);  // z_gamma
        const double kappa = tv_lin(z, zg, n);
        double beta = 0.0;
        if (kappa != 0.0) {
            double gz = 0.0;
            for (int64_t i = 0; i < n; ++i) gz += static_cast<double>(gv[i]) * static_cast<double>(zg[i]);
            beta = gz / kappa;
        }
        // h = g - beta q, q_i = v_{i-1} - v_i with v_e = w_e sign(Dz_e).
        for (int64_t i = 0; i < n; ++i) {
            double q = 0.0;
            if (i > 0) q += we(i - 1) * sgn_diff(z, i - 1);
            if (i + 1 < n) q -= we(i) * sgn_diff(z, i);
            h[i] = static_cast<T>(static_cast<double>(gv[i]) - beta * q);
        }
        for (int64_t k = 0; k < nw + 1; ++k) ia[k] = 0.0;
        tv->vjp_row(x, z, code, h, gx, n, g, ia, tscr);
        acc[0] += beta;
        for (int64_t k = 0; k < nw; ++k) acc[1 + k] += ia[k];
        if (beta != 0.0)
            for (int64_t e = 0; e + 1 < n; ++e)
                acc[1 + (nw == 1 ? 0 : e)] -=
                    beta * std::fabs(static_cast<double>(z[e + 1]) - static_cast<double>(z[e]));
    }
};

}  // namespace

fp_status make_tv1d_ball(double alpha, Vec w, fp_op **out) {
    if (!finite_nonneg(alpha)) return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    fp_op *tv = nullptr;
    if (fp_status s = make_tv1d(w, FP_TV1D_AUTO, &tv)) return s;
    *out = new TVBall(alpha, std::move(w), tv);
    return FP_OK;
}

}  // namespace fp
