// prox of gamma ||W D z||_2 along the axis, (W D z)_e = w_e (z_{e+1} - z_e): 1D
// TV with the l2 norm of the differences. Write K = D^T W^2 D (tridiagonal) and
// a_e = w_e^2. The dual is a trust-region subproblem: z = x - D^T W u with u
// minimising ||x - D^T W u||^2 / 2 over ||u||_2 <= gamma.
//
// Inside. The minimal-norm unconstrained minimiser u0 is a cumulative sum,
// u0_e = -U_e / w_e with U_e the running sum of x minus its piece mean (pieces
// between zero weights; a weight whose square underflows counts as zero). If
// ||u0|| <= gamma, z is the piecewise mean.
//
// Outside. u(mu) = (W D D^T W + mu I)^{-1} W D x with ||u(mu*)|| = gamma, mu* > 0,
// and z = x - D^T W u(mu) solves (mu I + K) z = mu x. mu* is the root of
// psi(mu) = 1/||u(mu)|| - 1/gamma, found by Newton's method as in J. Nocedal and
// S. J. Wright, Numerical Optimization, Springer 1999, Section 4.2, (4.26)-(4.27)
// and Algorithm 4.4, pp. 81-82, after J. J. More and D. C. Sorensen, SIAM J. Sci.
// Stat. Comput. 4(3) (1983) 553-572. psi is concave and increasing, so every
// Newton step lands at or below mu*, and the iterates from mu = 0 increase to
// it; a bracket [L, R] with R = ||W D x|| / gamma guards against rounding.
// The iteration runs on the differences: with T = D D^T and A = diag(a), D
// applied to (mu I + K) z = mu x gives (mu I + T A) Dz = mu D x, and the
// normalised dual uh = u / gamma = W Dz / (mu gamma) has ||uh|| = 1 at the root.
// Write dh = Dz / (mu gamma) = uh / w, nu = ||uh|| = (sum_e a_e dh_e^2)^(1/2) and
// yh for (mu I + T A) yh = mu dh. Then psi' gives the step
// mu (1 + (nu - 1) nu^2 / Sh) with Sh = sum_e a_e dh_e yh_e. Differences of z
// itself, computed from z, carry its rounding: 1 ulp of z across an edge of
// weight 0.5 is 1e-17 in ||W D z||, which hid an edge of weight 4e-149 whose
// part was 1e-150 (fuzzing). z itself is not taken as x - gamma D^T W uh, a
// difference of flows that grow along a long row, but solved for. At mu = 0
// the same quantities come from cumulative sums: ||u0||, and ||y0||^2 with y0
// the mean-free solution of w_e (y0_{e+1} - y0_e) = u0_e. Once a step is below
// 1e-7 mu it is taken, from either side of the root, and the iteration stops:
// Newton's quadratic rate puts the next error below rounding.
//
// Solves. B = sigma I + kappa T A (sigma = mu, kappa = 1 for mu < 1, else 1 and
// 1/mu) on the edges of each piece is an M-matrix, diagonally dominant by
// columns with excess sigma, or sigma + kappa a_e at the ends of a piece. Its
// pivots are kept in excess form, e_j = sigma + kappa a_j e_{j-1} / p_{j-1},
// p_j = e_j + kappa a_j (e_{-1} / p_{-1} = 1 at the start of a piece), sums of
// positive terms, so they are accurate for any mu, and the substitutions only
// add positive multiples. B dh = (sigma / (mu gamma)) D x has no factor mu for
// mu < 1, so no product of a tiny mu and a tiny difference underflows. After
// the piece means are removed, the inside test and the iteration run on x and
// gamma times a power of two that brings the largest entry to [0.5, 1). z then
// comes from one solve with sigma I + kappa K at the final mu.
//
// Derivatives. F(z) = z - x + gamma K z / s = 0. Its Jacobian in z is
// H = A - (c / s^2) p p^T with c = gamma / s, A = I + c K and p = K z, and
// (Sherman-Morrison) H^{-1} v = A^{-1} v + c t (t^T v) / (p^T y) with t = A^{-1} p
// and y = A^{-1} z; p^T y = s^2 (1 - (c / s^2) p^T t) is computed as
// sum_e a_e Dz_e Dy_e, without the cancellation. JVP: dz = H^{-1} r with
// r = dx - (dgamma / s) p - gamma sum_e dw_e d(K z / s)/dw_e, and
// d(K z / s)/dw_e = 2 w_e Dz_e D_e^T / s - p w_e Dz_e^2 / s^3. The VJP is the
// adjoint. Inside, the Jacobian is the piecewise averaging. All of it runs in
// scale-free form (see setup): delta = Dz / s, pi = p / s and c, so neither s
// nor s^3 is formed; they under- or overflowed for |z| beyond 1e+-103.
//
// Kinks: ||u0|| = gamma is inside.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <utility>

namespace fp {
namespace {

constexpr uint8_t kInside = 1;  // code[0]: z is the piecewise mean

struct TV1DL2 final : RowOp<TV1DL2> {
    Vec w, a;  // weights and their squares: length 1 or n - 1
    static constexpr int kMaxIter = 100;

    TV1DL2(Vec ww, Vec aa) : w(std::move(ww)), a(std::move(aa)) {}

    const char *name() const noexcept override { return "tv1d_l2"; }
    double step_scale() const noexcept override { return vmax(w); }
    bool conj_homogeneous() const noexcept override { return true; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n - 1) ? FP_OK : FP_ESHAPE;
    }
    // Four n-entry double arrays.
    size_t scratch_bytes(int64_t n) const noexcept override {
        return static_cast<size_t>(4 * n) * sizeof(double);
    }
    bool needs_distinct() const noexcept override { return false; }
    double we(int64_t e) const noexcept { return w.size() == 1 ? w[0] : w[e]; }
    double ae(int64_t e) const noexcept { return a.size() == 1 ? a[0] : a[e]; }

    // 1 / d_j of sigma I + kappa K. The ratio e / d <= 1 comes first: kappa a_j e
    // underflowed for a_j = 2e-297 and e = 7e-295 and dropped the edge.
    void factor(double sigma, double kappa, double *inv, int64_t n) const noexcept {
        double e = sigma;
        for (int64_t j = 0; j < n; ++j) {
            const double d = e + (j + 1 < n ? kappa * ae(j) : 0.0);
            inv[j] = 1.0 / d;
            if (j + 1 < n) e = sigma + kappa * ae(j) * (e * inv[j]);
        }
    }
    // v <- (sigma I + kappa K)^{-1} v, in place.
    void solve(double kappa, const double *inv, double *v, int64_t n) const noexcept {
        for (int64_t j = 1; j < n; ++j) v[j] += kappa * ae(j - 1) * inv[j - 1] * v[j - 1];
        v[n - 1] *= inv[n - 1];
        for (int64_t j = n - 2; j >= 0; --j) v[j] = (v[j] + kappa * ae(j) * v[j + 1]) * inv[j];
    }
    // Factor sigma I + kappa K and solve v = (sigma I + kappa K)^{-1} sigma b in one
    // pass each way, returning ||W D v||^2 (not finite if v is not).
    double factor_solve_z(double sigma, double kappa, double *inv, const double *b, double *v,
                          int64_t n) const noexcept {
        double e = sigma, f = 0.0, m = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            const double off = j + 1 < n ? kappa * ae(j) : 0.0;
            const double ij = 1.0 / (e + off);
            f = sigma * b[j] + m * f;
            inv[j] = ij;
            v[j] = f;
            m = off * ij;
            e = sigma + m * e;
        }
        double s = 0.0, next = v[n - 1] *= inv[n - 1];
        for (int64_t j = n - 2; j >= 0; --j) {
            const double vj = (v[j] + kappa * ae(j) * next) * inv[j];
            v[j] = vj;
            s += ae(j) * (next - vj) * (next - vj);
            next = vj;
        }
        return s;
    }
    // The Newton loop's passes on the edges, fused so that the dependent chains
    // of the factorisation and the substitutions overlap: factor B and solve
    // B dh = f D xc, returning sum_e a_e dh_e^2. Edges of zero weight store
    // inv = 0, so dh = 0 there and each piece is solved on its own; on the
    // chain, their e / p = sigma (1 / sigma) is 1 to rounding for sigma >= DBL_MIN,
    // which starts the next piece.
    double factor_solve(double sigma, double kappa, double f, const double *xc, double *inv,
                        double *dh, int64_t n) const noexcept {
        double r = 1.0, g = 0.0, m = 0.0;  // r = e_{j-1} / p_{j-1}, m = kappa a_{j-1} / p_{j-1}
        for (int64_t j = 0; j + 1 < n; ++j) {
            const double ka = kappa * ae(j), e = sigma + ka * r;
            const double ij = 1.0 / (e + ka);
            g = f * (xc[j + 1] - xc[j]) + m * g;
            inv[j] = ae(j) != 0.0 ? ij : 0.0;
            dh[j] = g;
            m = ka * ij;
            r = e * ij;
        }
        double s = 0.0, next = 0.0, kn = 0.0;  // kn = kappa a_{j+1}
        for (int64_t j = n - 2; j >= 0; --j) {
            const double dj = (dh[j] + kn * next) * inv[j];
            dh[j] = dj;
            s += ae(j) * dj * dj;
            next = dj;
            kn = kappa * ae(j);
        }
        return s;
    }
    // yh = B^{-1} sigma dh with the factors above, returning sum_e a_e dh_e yh_e.
    double solve_cross(double sigma, double kappa, const double *inv, const double *dh, double *yh,
                       int64_t n) const noexcept {
        double g = 0.0, m = 0.0;
        for (int64_t j = 0; j + 1 < n; ++j) {
            g = sigma * dh[j] + m * g;
            yh[j] = g;
            m = kappa * ae(j) * inv[j];
        }
        double S = 0.0, next = 0.0, kn = 0.0;
        for (int64_t j = n - 2; j >= 0; --j) {
            const double yj = (yh[j] + kn * next) * inv[j];
            yh[j] = yj;
            S += ae(j) * dh[j] * yj;
            next = yj;
            kn = kappa * ae(j);
        }
        return S;
    }
    // xc <- x minus its piece means. The mean is the piece's first entry plus
    // the mean of the offsets from it, so a constant piece gives exactly zero.
    // The inside test divides running sums of xc by w_e^2, and the rounding of
    // a plain running mean of equal entries, over a tiny weight, passed for a
    // huge dual (found by fuzzing: NaN at w_e = 2e-53). Each entry is still
    // rounded once (centring on the first entry itself rounded twice and left
    // 25x larger residuals on a long random walk without FMA contraction).
    template <class T>
    void center(const T *x, double *xc, int64_t n) const noexcept {
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e < n - 1 && ae(e) != 0.0) continue;
            const double x0 = static_cast<double>(x[s]);  // piece [s, e]
            double m = 0.0;
            for (int64_t i = s; i <= e; ++i) m += static_cast<double>(x[i]) - x0;
            m = x0 + m / static_cast<double>(e - s + 1);
            for (int64_t i = s; i <= e; ++i) xc[i] = static_cast<double>(x[i]) - m;
            s = e + 1;
        }
    }
    // Piecewise mean of v into out (may alias).
    template <class T, class U>
    void piece_mean(const T *v, U *out, int64_t n) const noexcept {
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e < n - 1 && ae(e) != 0.0) continue;
            double m = 0.0;
            for (int64_t i = s; i <= e; ++i) m += static_cast<double>(v[i]);
            m /= static_cast<double>(e - s + 1);
            for (int64_t i = s; i <= e; ++i) out[i] = static_cast<U>(m);
            s = e + 1;
        }
    }
    template <class V>
    double norm_wd(const V *z, int64_t n) const noexcept {
        double s = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double d = static_cast<double>(z[e + 1]) - static_cast<double>(z[e]);
            s += ae(e) * d * d;
        }
        return std::sqrt(s);
    }

    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double, double *) const noexcept override {
        double s = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) s += ae(e) * (x[e + 1] - x[e]) * (x[e + 1] - x[e]);
        return std::sqrt(s);
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *code) const noexcept {
        if (!all_finite(x, n)) {
            bool nan;
            if (scan_row(x, n, nan) != kRowOk) return kRowNonFinite;
            fill_nan(z, n);
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            return kRowOk;
        }
        if (code) std::memset(code, 0, static_cast<size_t>(n));
        if (n == 1 || gamma == 0.0) {
            if (z != x) std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
            return kRowOk;
        }
        double *xc = static_cast<double *>(scr), *inv = xc + n, *dh = inv + n, *y = dh + n;
        center(x, xc, n);
        // Everything below runs on xc and gamma times t = 2^k with max |t xc| in
        // [0.5, 1): z - mean is homogeneous in (xc, gamma), mu is unchanged, and
        // a power of two scales every rounded result exactly. Unscaled, ||u0||^2
        // underflowed to 0 for entries near 1e-177 (inside, wrongly), and sigma xc
        // underflowed in the solves for mu = 2e-295 at entries near 1e-147 (NaN;
        // both found by fuzzing).
        double t = 0.0;
        for (int64_t i = 0; i < n; ++i) t = std::fmax(t, std::fabs(xc[i]));
        int k;
        std::frexp(t, &k);
        t = std::ldexp(1.0, -k);
        const double gt = gamma * t;
        // ||u0||^2 and ||y0||^2 (y0 per piece, mean-free), and ||W D xc||^2.
        double rho0sq = 0.0, kap0 = 0.0, wdx = 0.0;
        {
            int64_t s = 0;
            double U = 0.0, yv = 0.0, ys = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                y[i] = yv;
                ys += yv;
                const bool end = i == n - 1 || ae(i) == 0.0;
                if (!end) {
                    U += t * xc[i];
                    rho0sq += U * U / ae(i);
                    yv -= U / ae(i);
                    const double d = t * (xc[i + 1] - xc[i]);
                    wdx += ae(i) * d * d;
                    continue;
                }
                const double m = ys / static_cast<double>(i - s + 1);
                for (int64_t k = s; k <= i; ++k) kap0 += (y[k] - m) * (y[k] - m);
                s = i + 1;
                U = yv = ys = 0.0;
            }
        }
        const double rho0 = std::sqrt(rho0sq);
        double L = 0.0, R = std::sqrt(wdx) / gt;
        // R = 0: no differences across nonzero weights, so x is its own
        // piecewise mean whatever rounding left in rho0.
        if (rho0 <= gt || !(R > 0.0)) {
            for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(static_cast<double>(x[i]) - xc[i]);
            if (code) code[0] = kInside;
            return kRowOk;
        }
        for (int64_t i = 0; i < n; ++i) xc[i] *= t;
        // Bisection in the exponent, a dive below R while no lower bound is
        // known: mu* can lie hundreds of orders of magnitude below R (w_e^2 of
        // 2e-297 against R = 1e-146, where ||y0||^2 overflows and Newton has no
        // start). From below, Newton steps then rise to the root.
        auto mid = [&] { return L > 0.0 ? std::sqrt(L) * std::sqrt(R) : R * 0x1p-64; };
        double mu = (rho0 / gt - 1.0) * rho0sq / kap0;  // Newton from mu = 0
        if (!(mu > L && mu < R)) mu = mid();
        // mu >= DBL_MIN keeps 1 / sigma finite; a root below it needs w_e^2 of
        // denormal size, and z is then taken at DBL_MIN.
        if (!(mu >= DBL_MIN)) mu = DBL_MIN;
        // dh at mu into dh, returning nu; yh likewise from dh, returning Sh.
        double sigma, kappa;
        auto solve_dh = [&](double m) {
            sigma = m < 1.0 ? m : 1.0;
            kappa = m < 1.0 ? 1.0 : 1.0 / m;
            const double f = m < 1.0 ? 1.0 / gt : 1.0 / (m * gt);
            return std::sqrt(factor_solve(sigma, kappa, f, xc, inv, dh, n));
        };
        // The loop sets mu for z: the root, or failing that the last mu whose
        // solve was finite (an overflow far from the root leaves inf or NaN,
        // taken as below it).
        double good = 0.0, muz = 0.0;
        bool finite = false;  // dh holds a finite solve
        for (int it = 0; it < kMaxIter; ++it) {
            const double nu = solve_dh(mu);
            finite = nu <= DBL_MAX;
            if (finite) good = mu;
            if (nu < 1.0) R = mu;
            else if (nu > 1.0 || !finite) L = mu;
            else {
                muz = mu;
                break;
            }
            const double S = solve_cross(sigma, kappa, inv, dh, y, n);
            double next = S > 0.0 ? mu + mu * ((nu - 1.0) * (nu * nu / S)) : mid();  // nu^2 / S >= 1
            // Take the last small step from either side: rounding can put an
            // iterate just right of the root, and a Newton step from there
            // also lands at or below it, with the same quadratic error.
            if (std::fabs(next - mu) <= 1e-7 * mu) {
                muz = next > L && next < R && next >= DBL_MIN ? next : mu;
                break;
            }
            if (!(next > L && next < R)) next = mid();
            if (next < DBL_MIN) next = DBL_MIN;  // a step from the right can land at mu 1e-16
            if (!(next > L && next < R)) break;  // the bracket is two adjacent doubles
            mu = next;
        }
        if (!(muz > 0.0) || (muz == mu && !finite)) muz = good;
        if (!(muz > 0.0)) {  // no finite solve: the piecewise mean
            for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(static_cast<double>(x[i]) - xc[i] / t);
            return kRowOk;
        }
        // z itself from (sigma I + kappa K) zc = sigma xc, accurate to rounding of
        // xc; its differences were not (above). Should that solve overflow,
        // z = x - gamma D^T W uh from dh takes over.
        const double sg = muz < 1.0 ? muz : 1.0, kp = muz < 1.0 ? 1.0 : 1.0 / muz;
        if (factor_solve_z(sg, kp, inv, xc, y, n) <= DBL_MAX) {
            for (int64_t i = 0; i < n; ++i)
                z[i] = static_cast<T>(static_cast<double>(x[i]) - (xc[i] - y[i]) / t);
            return kRowOk;
        }
        if (!(finite && muz == mu)) solve_dh(muz);
        double q = 0.0;  // gamma a_{i-1} dh_{i-1}, with W uh = A dh
        for (int64_t i = 0; i < n; ++i) {
            const double qi = i + 1 < n ? gamma * ae(i) * dh[i] : 0.0;
            z[i] = static_cast<T>(static_cast<double>(x[i]) - (q - qi));
            q = qi;
        }
        return kRowOk;
    }

    // The outside structure at the saved z, scale-free (audit K4): H is
    // unchanged when z and gamma are both multiplied by t, so the setup runs
    // on the differences t D z with t a power of two that brings max w_e |Dz_e|
    // to [1/2, 1) (h = 1/2 first when Dz itself overflows). With sh = ||W D(t z)||
    // = t s: delta_e = Dz_e / s (|w_e delta_e| <= 1), c = gamma / s =
    // gamma t / sh, pi = K z / s = D^T A delta, and y = A^{-1} (t z_c) with z_c
    // the centred z. kOutside sets c, pi (into pi), the factored A = I + c K
    // and y; kIdentity when z has no differences; kMean when c is beyond the
    // range (the piecewise mean, the limit of A^{-1} as c -> inf).
    struct Scale {
        double h = 1.0, t = 1.0, sh = 0.0, f = 0.0;  // f = t / sh: a multiply per delta, not a division
        template <class T>
        double delta(const T *z, int64_t e) const noexcept {
            return (h * static_cast<double>(z[e + 1]) - h * static_cast<double>(z[e])) * f;
        }
    };
    enum Kind { kIdentity, kMean, kOutside };
    template <class T>
    Kind setup(const T *z, int64_t n, double gamma, Scale &sc, double &c, double *inv, double *pi,
               double *y) const noexcept {
        double M = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e)
            M = std::max(M, we(e) * std::fabs(static_cast<double>(z[e + 1]) - static_cast<double>(z[e])));
        if (!(M <= DBL_MAX)) {  // the differences overflow: halves first
            sc.h = 0.5;
            M = 0.0;
            for (int64_t e = 0; e + 1 < n; ++e)
                M = std::max(M, we(e) * std::fabs(0.5 * static_cast<double>(z[e + 1]) - 0.5 * static_cast<double>(z[e])));
        }
        if (!(M > 0.0)) return kIdentity;
        int k;
        std::frexp(M, &k);
        sc.t = std::ldexp(1.0, -k);
        double ss = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double d = (sc.h * static_cast<double>(z[e + 1]) - sc.h * static_cast<double>(z[e])) * sc.t;
            ss += ae(e) * d * d;
        }
        sc.sh = std::sqrt(ss);
        sc.f = sc.t / sc.sh;
        c = gamma * sc.h * sc.t / sc.sh;  // gamma / s; with h, s is ||W D(h z)|| / h
        if (!(c <= DBL_MAX)) return kMean;
        for (int64_t i = 0; i < n; ++i) pi[i] = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double v = ae(e) * sc.delta(z, e);
            pi[e + 1] += v;
            pi[e] -= v;
        }
        factor(1.0, c, inv, n);
        piece_mean(z, y, n);
        const double tz = sc.h * sc.t;
        for (int64_t i = 0; i < n; ++i) y[i] = (static_cast<double>(z[i]) - y[i]) * tz;
        return kOutside;
    }

    // v <- H^{-1} v given t = A^{-1} pi and pty = pi^T A^{-1} (t z_c) / sh: the
    // Sherman-Morrison term c t (t^T v) / pty, scale-free (K4).
    void apply_hinv(double c, const double *inv, const double *t, double pty, double *v,
                    int64_t n) const noexcept {
        double tv = 0.0;
        for (int64_t i = 0; i < n; ++i) tv += t[i] * v[i];
        solve(c, inv, v, n);
        const double f = c * tv / pty;
        for (int64_t i = 0; i < n; ++i) v[i] += f * t[i];
    }

    // y <- A^{-1} y and pi^T y / sh = sum_e a_e delta_e (y_{e+1} - y_e) / sh,
    // without the cancellation of pi^T y.
    template <class T>
    double solve_pty(const T *z, const Scale &sc, double c, const double *inv, double *y, int64_t n) const noexcept {
        solve(c, inv, y, n);
        double pty = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) pty += ae(e) * sc.delta(z, e) * (y[e + 1] - y[e]);
        return pty / sc.sh;
    }

    template <class T>
    void jvp(const T *, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n,
             double gamma, double dgamma, const double *dp, void *scr) const noexcept {
        double *inv = static_cast<double *>(scr), *pi = inv + n, *y = pi + n, *r = y + n;
        Scale sc;
        double c = 0.0;
        const Kind kind = code[0] == kInside ? kMean : (n == 1 ? kIdentity : setup(z, n, gamma, sc, c, inv, pi, y));
        if (kind == kMean) {
            piece_mean(dx, dz, n);
            return;
        }
        if (kind == kIdentity) {
            for (int64_t i = 0; i < n; ++i) dz[i] = dx[i];
            return;
        }
        const double pty = solve_pty(z, sc, c, inv, y, n);
        // r = dx - dgamma pi + sum_e 2 gamma w_e delta_e dw_e (e_e - e_{e+1})
        //     + gamma (sum_e dw_e w_e delta_e^2) pi.
        for (int64_t i = 0; i < n; ++i) r[i] = static_cast<double>(dx[i]) - dgamma * pi[i];
        if (dp) {
            double wsum = 0.0;
            for (int64_t e = 0; e + 1 < n; ++e) {
                const double de = sc.delta(z, e), dwe = dp[w.size() == 1 ? 0 : e];
                const double q = 2.0 * gamma * we(e) * de * dwe;
                r[e] += q;
                r[e + 1] -= q;
                wsum += dwe * we(e) * de * de;
            }
            for (int64_t i = 0; i < n; ++i) r[i] += gamma * wsum * pi[i];
        }
        solve(c, inv, pi, n);  // tau = A^{-1} pi
        apply_hinv(c, inv, pi, pty, r, n);
        for (int64_t i = 0; i < n; ++i) dz[i] = static_cast<T>(r[i]);
    }

    template <class T>
    void vjp(const T *, const T *z, const uint8_t *code, const T *gv, T *gx, int64_t n,
             double gamma, double *acc, void *scr) const noexcept {
        double *inv = static_cast<double *>(scr), *pi = inv + n, *y = pi + n, *q = y + n;
        Scale sc;
        double c = 0.0;
        const Kind kind = code[0] == kInside ? kMean : (n == 1 ? kIdentity : setup(z, n, gamma, sc, c, inv, pi, y));
        if (kind == kMean) {
            piece_mean(gv, gx, n);
            return;
        }
        if (kind == kIdentity) {
            for (int64_t i = 0; i < n; ++i) gx[i] = gv[i];
            return;
        }
        const double pty = solve_pty(z, sc, c, inv, y, n);
        // y is free again: keep pi there, and tau = A^{-1} pi in pi's buffer.
        std::memcpy(y, pi, static_cast<size_t>(n) * sizeof(double));
        solve(c, inv, pi, n);
        for (int64_t i = 0; i < n; ++i) q[i] = static_cast<double>(gv[i]);
        apply_hinv(c, inv, pi, pty, q, n);
        for (int64_t i = 0; i < n; ++i) gx[i] = static_cast<T>(q[i]);
        double piq = 0.0;
        for (int64_t i = 0; i < n; ++i) piq += y[i] * q[i];
        const int64_t P = nparams();
        acc[P] -= piq;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double de = sc.delta(z, e);
            acc[w.size() == 1 ? 0 : e] -= gamma * we(e) * de * (2.0 * (q[e + 1] - q[e]) - piq * de);
        }
    }
};

}  // namespace

fp_status make_tv1d_l2(Vec w, fp_op **out) {
    if (w.empty()) w.assign(1, 1.0);
    Vec a(w.size());
    for (size_t e = 0; e < w.size(); ++e) {
        if (!finite_nonneg(w[e])) return FP_EPARAM;
        a[e] = w[e] * w[e];
        if (!finite_nonneg(a[e])) return FP_EPARAM;  // w_e^2 overflows
    }
    *out = new TV1DL2(std::move(w), std::move(a));
    return FP_OK;
}

}  // namespace fp
