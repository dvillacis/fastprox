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
// Derivatives (revised in 1.0.3, SCD review S2). Outside, F(z) = z - x +
// gamma D^T W n(z) = 0 with n = W D z / s, s = ||W D z|| = mu gamma, has
// H = I + (1 / mu) D^T W (I - n n^T) W D, symmetric with H >= I, so J = H^{-1}
// has its spectrum in (0, 1]. With T = W D D^T W and Q a basis of n's
// complement, J = I - D^T W Q (Q^T (mu I + T) Q)^{-1} Q^T W D (push-through),
// applied as the bordered system [mu I + T, n; n^T, 0] (apply_j). mu and n
// come from the root solve, recomputed from the saved x, step and weights
// (root): n = W dh / ||W dh|| from the solver's scaled differences. The 1.0.2
// form read n = W D z / s from differences of the saved z and inverted H by
// Sherman-Morrison; near the boundary (s -> 0) those differences carry a
// relative error of eps ||x|| / s and the denominator 1 - c pi^T A^-1 pi
// cancelled, which gave eigenvalues of J up to 7 on the boundary. Parameters:
// dz = J (dx + r_theta) with r_theta = -dgamma pi - gamma (2 D^T (n o dw) -
// pi sum_e n_e^2 dw_e / w_e), pi = D^T W n; the VJP is the adjoint. Inside,
// the Jacobian is the piecewise averaging. Where gamma t is below the normal
// range (z = x), mu is beyond it: J = I, with n from W D x.
//
// Kinks: ||u0|| = gamma is inside.
//
// Weights. They are stored over 2^k, k the exponent of the largest (as for
// isotonic regression), and every call runs at the step gamma 2^k, since
// ||W D z|| is homogeneous in w: a weight of 1e-180 at gamma = 1e180 squared
// to 0 and dropped its edge, a problem whose solution is 0 (math audit M5).
// Derivatives in gamma pick up 2^k and those in w 2^-k. A positive weight
// below 2^-511 times the largest is FP_EPARAM: its square would still not be
// normal.
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
    static constexpr int kStruct = FP_STRUCT_CURVED;
    Vec w, a;  // weights over 2^wk and their squares: length 1 or n - 1
    double wmax_ = vmax(w);  // the largest weight as given, for step_scale: once, not per call (it scanned every weight on each prox call)
    int wk = 0;
    static constexpr int kMaxIter = 100;

    explicit TV1DL2(Vec ww) : w(std::move(ww)), a(w.size()) {
        std::frexp(wmax_, &wk);
        for (size_t e = 0; e < w.size(); ++e) {
            w[e] = std::ldexp(w[e], -wk);
            a[e] = w[e] * w[e];
        }
    }

    const char *name() const noexcept override { return "tv1d_l2"; }
    double step_scale() const noexcept override { return wmax_; }
    bool conj_homogeneous() const noexcept override { return true; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    fp_status check_n(int64_t n) const noexcept override {
        return (w.size() == 1 || static_cast<int64_t>(w.size()) == n - 1) ? FP_OK : FP_ESHAPE;
    }
    // Five n-entry double arrays: the root solve's four (xc, inv, dh, y)
    // and the derivative's right-hand side.
    size_t scratch_bytes(int64_t n) const noexcept override {
        return static_cast<size_t>(5 * n) * sizeof(double);
    }
    bool needs_distinct() const noexcept override { return false; }
    double we(int64_t e) const noexcept { return w.size() == 1 ? w[0] : w[e]; }
    double ae(int64_t e) const noexcept { return a.size() == 1 ? a[0] : a[e]; }

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
    // Should the sum of offsets overflow (x0 = 9e307 with 18 small entries,
    // fuzz 2026-10-25: the row came back -inf), it is summed again over 2^k,
    // k from the piece's length, and only then.
    template <class T>
    void center(const T *x, double *xc, int64_t n) const noexcept {
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e < n - 1 && ae(e) != 0.0) continue;
            const double x0 = static_cast<double>(x[s]);  // piece [s, e]
            const double len = static_cast<double>(e - s + 1);
            double m = 0.0;
            for (int64_t i = s; i <= e; ++i) m += static_cast<double>(x[i]) - x0;
            if (std::isfinite(m)) {
                m = x0 + m / len;
            } else {
                int k;
                std::frexp(len, &k);
                m = 0.0;
                for (int64_t i = s; i <= e; ++i) m += std::ldexp(static_cast<double>(x[i]), -k - 1) - std::ldexp(x0, -k - 1);
                m = x0 + std::ldexp(m / len, k + 1);
            }
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
    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double, double *) const noexcept override {
        double s = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) s += ae(e) * (x[e + 1] - x[e]) * (x[e + 1] - x[e]);
        return std::ldexp(std::sqrt(s), wk);
    }

    // dh at mu into dh (B = sigma I + kappa T A factored into inv), returning
    // nu = ||W dh||; sigma and kappa as the solves take them.
    double solve_dh(double m, double gt, const double *xc, double *inv, double *dh, int64_t n, double &sigma,
                    double &kappa) const noexcept {
        sigma = m < 1.0 ? m : 1.0;
        kappa = m < 1.0 ? 1.0 : 1.0 / m;
        const double f = m < 1.0 ? 1.0 / gt : 1.0 / (m * gt);
        return std::sqrt(factor_solve(sigma, kappa, f, xc, inv, dh, n));
    }

    // The classification and root shared by the prox and the derivatives (SCD
    // review S2): the derivatives recompute them from the saved x, step and
    // weights rather than from differences of the saved z. Requires a finite
    // row with n >= 2; gamma is the stored weights' step.
    //   kTiny     gamma t below the normal range (gamma = 0 included): z = x
    //   kMean     inside, or no finite solve: the piecewise mean x - xc / xs
    //   kOutside  mu, the root, with sigma and kappa; with want_dh, inv holds
    //             the factors of B at mu and dh the solution there, nu = ||W dh||.
    // xc holds x minus its piece means, times xs.
    enum Kind { kTiny, kMean, kOutside };
    struct Root {
        Kind kind = kTiny;
        double t = 1.0, gt = 0.0, xs = 1.0, mu = 0.0, sigma = 1.0, kappa = 1.0, nu = 0.0;
    };
    template <class T>
    Root root(const T *x, int64_t n, double gamma, double *scr, bool want_dh) const noexcept {
        Root rt;
        double *xc = scr, *inv = xc + n, *dh = inv + n, *y = dh + n;
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
        rt.t = t;
        rt.gt = gt;
        // gamma t below the normal range: |z - x| <= 2 gamma, under 2^-1020
        // of the largest entry, so z = x to rounding. Taken as it, since the
        // bracket below would start from [0, inf] (fuzz 2026-10-25: the
        // piecewise mean came back, at gamma = 3.6e-97 with an entry of 1e308).
        if (gt < DBL_MIN) return rt;
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
            rt.kind = kMean;
            return rt;
        }
        for (int64_t i = 0; i < n; ++i) xc[i] *= t;
        rt.xs = t;
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
        double sigma, kappa;
        // The loop sets mu for z: the root, or failing that the last mu whose
        // solve was finite (an overflow far from the root leaves inf or NaN,
        // taken as below it).
        double good = 0.0, muz = 0.0, nu = 0.0;
        bool finite = false;  // dh holds a finite solve
        for (int it = 0; it < kMaxIter; ++it) {
            nu = solve_dh(mu, gt, xc, inv, dh, n, sigma, kappa);
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
            rt.kind = kMean;
            return rt;
        }
        rt.kind = kOutside;
        rt.mu = muz;
        rt.sigma = muz < 1.0 ? muz : 1.0;
        rt.kappa = muz < 1.0 ? 1.0 : 1.0 / muz;
        if (want_dh) rt.nu = finite && muz == mu ? nu : solve_dh(muz, gt, xc, inv, dh, n, sigma, kappa);
        return rt;
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *code) const noexcept {
        gamma = std::ldexp(gamma, wk);  // the stored weights' step
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
        const Root rt = root(x, n, gamma, xc, false);
        if (rt.kind == kTiny) {
            if (z != x) std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
            return kRowOk;
        }
        if (rt.kind == kMean) {
            for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(static_cast<double>(x[i]) - xc[i] / rt.xs);
            if (code) code[0] = kInside;
            return kRowOk;
        }
        const double t = rt.t;
        // z itself from (sigma I + kappa K) zc = sigma xc, accurate to rounding of
        // xc; its differences were not (above). Should that solve overflow,
        // z = x - gamma D^T W uh from dh takes over.
        if (factor_solve_z(rt.sigma, rt.kappa, inv, xc, y, n) <= DBL_MAX) {
            for (int64_t i = 0; i < n; ++i)
                z[i] = static_cast<T>(static_cast<double>(x[i]) - (xc[i] - y[i]) / t);
            return kRowOk;
        }
        double sigma, kappa;
        solve_dh(rt.mu, rt.gt, xc, inv, dh, n, sigma, kappa);
        double q = 0.0;  // gamma a_{i-1} dh_{i-1}, with W uh = A dh
        for (int64_t i = 0; i < n; ++i) {
            const double qi = i + 1 < n ? gamma * ae(i) * dh[i] : 0.0;
            z[i] = static_cast<T>(static_cast<double>(x[i]) - (q - qi));
            q = qi;
        }
        return kRowOk;
    }

    // The derivative's setup at the saved x (S2). Inside (code[0]) the
    // piecewise mean. Otherwise the root solve again, which leaves dh, with
    // delta = dh / nu = Dz / s and n = W delta the unit dual direction, and
    // B factored at mu in inv. kappa = 0 is J = I: mu beyond the range (kTiny,
    // where z = x, or a root that overflows, gamma near the bottom of the range
    // on a long row), where J and n differ from I and from W D x / ||W D x|| by
    // O(1 / mu), so delta is taken from W D x. No direction at all (W D x = 0,
    // or a solve without a finite one, which the prox's own success rules out)
    // is the identity with no parameter terms.
    enum DKind { kDMean, kDIdentity, kDLive };
    struct DSetup {
        DKind kind = kDIdentity;
        double kappa = 0.0, rnu = 0.0;  // rnu = 1 / nu
    };
    template <class T>
    DSetup dsetup(const T *x, const uint8_t *code, int64_t n, double gamma, double *scr) const noexcept {
        DSetup d;
        if (code[0] == kInside) {
            d.kind = kDMean;
            return d;
        }
        if (n == 1) return d;
        double *xc = scr, *dh = scr + 2 * n;
        const Root rt = root(x, n, gamma, xc, true);
        if (rt.kind == kMean) {  // as the prox's z, should it ever differ
            d.kind = kDMean;
            return d;
        }
        double nu = rt.nu;
        if (rt.kind == kOutside && rt.kappa > 0.0) {
            d.kappa = rt.kappa;
        } else {  // W D x itself, from x centred and times t (|t xc| < 1)
            const double f = rt.t / rt.xs;
            double ss = 0.0;
            for (int64_t e = 0; e + 1 < n; ++e) {
                dh[e] = ae(e) != 0.0 ? f * (xc[e + 1] - xc[e]) : 0.0;
                ss += ae(e) * dh[e] * dh[e];
            }
            nu = std::sqrt(ss);
        }
        if (!(nu > 0.0 && nu <= DBL_MAX)) return d;
        d.kind = kDLive;
        d.rnu = 1.0 / nu;
        return d;
    }

    // v <- J v on the outside piece: J = I - kappa D^T A B^{-1} (rho - eta dh)
    // with rho = D v and eta = sum_e a_e gd_e gr_e / p_e over sum_e a_e gd_e^2 / p_e,
    // gd and gr the forward substitutions of dh and rho with B's factors. In
    // the symmetric form, with T = W D D^T W and the unit direction n = W dh / nu,
    // this is J = I - D^T W Q (Q^T (mu I + T) Q)^{-1} Q^T W D, Q a basis of n's
    // complement: the bordered system [mu I + T, n; n^T, 0], solved through
    // B = W^-1 (sigma I + kappa T) W. The denominator is a sum of squares over
    // positive pivots, so it does not cancel (the Sherman-Morrison form's
    // 1 - c pi^T A^-1 pi did, near the boundary). gd and gr are scratch.
    void apply_j(double kappa, const double *inv, const double *dh, double *v, double *gd, double *gr,
                 int64_t n) const noexcept {
        if (kappa == 0.0) return;
        double num = 0.0, den = 0.0, m = 0.0, pr = 0.0, pd = 0.0;
        for (int64_t j = 0; j + 1 < n; ++j) {
            pr = (v[j + 1] - v[j]) + m * pr;
            pd = dh[j] + m * pd;
            gr[j] = pr;
            gd[j] = pd;
            const double ai = ae(j) * inv[j];
            num += ai * pd * pr;
            den += ai * pd * pd;
            m = kappa * ai;
        }
        const double eta = den > 0.0 ? num / den : 0.0;
        double next = 0.0, kn = 0.0, u = 0.0;  // u = kappa a_j vt_j, of the edge after i
        for (int64_t j = n - 2; j >= 0; --j) {
            next = (gr[j] - eta * gd[j] + kn * next) * inv[j];
            kn = kappa * ae(j);
            gr[j] = kn * next;  // kappa (A vt)_j
        }
        for (int64_t i = 0; i < n; ++i) {
            const double ui = i + 1 < n ? gr[i] : 0.0;
            v[i] -= u - ui;  // kappa (D^T A vt)_i
            u = ui;
        }
    }

    // Structure (since 1.0.2): inside, the pieces between zero weights as
    // regions; otherwise singletons, where J is positive definite (a finite
    // root) or the identity. From the prox's classification, not from the
    // saved z, whose differences can vanish outside (S2).
    template <class T>
    int64_t structure(const T *, const T *, const uint8_t *code, int64_t n, double, RowStruct &s,
                      void *) const noexcept {
        if (code[0] == kInside) return runs(s, n, [this](int64_t i) { return ae(i - 1) == 0.0; });
        for (int64_t i = 0; i < n; ++i) s.state[i] = FP_ENTRY_FREE;
        return singletons(s, n);
    }

    // JVP: dz = J (dx + r_theta), r_theta = -dgamma pi - gamma (2 D^T (n o dw)
    // - pi sum_e n_e^2 dw_e / w_e) with pi = D^T W n = D^T A delta; with
    // n_e / w_e = delta_e no weight is divided by, and a zero-weight edge,
    // where dh = 0, contributes nothing. The weights are stored over 2^wk.
    template <class T>
    void jvp(const T *x, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dp, void *scr) const noexcept {
        gamma = std::ldexp(gamma, wk);
        dgamma = std::ldexp(dgamma, wk);
        double *g0 = static_cast<double *>(scr), *inv = g0 + n, *dh = inv + n, *g1 = dh + n, *r = g1 + n;
        const DSetup d = dsetup(x, code, n, gamma, g0);
        if (d.kind == kDMean) {
            piece_mean(dx, dz, n);
            return;
        }
        if (d.kind == kDIdentity) {
            for (int64_t i = 0; i < n; ++i) dz[i] = dx[i];
            return;
        }
        for (int64_t i = 0; i < n; ++i) r[i] = static_cast<double>(dx[i]);
        double wsum = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double de = dh[e] * d.rnu, pe = ae(e) * de;  // pi = D^T A delta
            r[e] += dgamma * pe;
            r[e + 1] -= dgamma * pe;
            if (dp) {
                const double dwe = std::ldexp(dp[w.size() == 1 ? 0 : e], -wk);
                const double q = 2.0 * gamma * we(e) * de * dwe;
                r[e] += q;
                r[e + 1] -= q;
                wsum += dwe * we(e) * de * de;
            }
        }
        if (dp && wsum != 0.0)
            for (int64_t e = 0; e + 1 < n; ++e) {
                const double pe = gamma * wsum * ae(e) * dh[e] * d.rnu;
                r[e] -= pe;
                r[e + 1] += pe;
            }
        // J is linear: on r over a power of two, so that a large right-hand
        // side (dgamma 2^k for weights far from unit scale) cannot overflow
        // inside it where dz itself is in range (M5).
        double rm = 0.0;
        for (int64_t i = 0; i < n; ++i) rm = std::fmax(rm, std::fabs(r[i]));
        int re = 0;
        if (rm > 0.0 && std::isfinite(rm)) std::frexp(rm, &re);
        for (int64_t i = 0; i < n; ++i) r[i] = std::ldexp(r[i], -re);
        apply_j(d.kappa, inv, dh, r, g0, g1, n);
        for (int64_t i = 0; i < n; ++i) dz[i] = static_cast<T>(std::ldexp(r[i], re));
    }

    // VJP: q = J gv (J is symmetric); gamma's adjoint -pi^T q and w_e's
    // -gamma w_e delta_e (2 (D q)_e - (pi^T q) delta_e).
    template <class T>
    void vjp(const T *x, const T *, const uint8_t *code, const T *gv, T *gx, int64_t n, double gamma,
             double *acc, void *scr) const noexcept {
        gamma = std::ldexp(gamma, wk);
        double *g0 = static_cast<double *>(scr), *inv = g0 + n, *dh = inv + n, *g1 = dh + n, *q = g1 + n;
        const DSetup d = dsetup(x, code, n, gamma, g0);
        if (d.kind == kDMean) {
            piece_mean(gv, gx, n);
            return;
        }
        if (d.kind == kDIdentity) {
            for (int64_t i = 0; i < n; ++i) gx[i] = gv[i];
            return;
        }
        for (int64_t i = 0; i < n; ++i) q[i] = static_cast<double>(gv[i]);
        apply_j(d.kappa, inv, dh, q, g0, g1, n);
        for (int64_t i = 0; i < n; ++i) gx[i] = static_cast<T>(q[i]);
        double piq = 0.0;  // pi^T q = delta^T A D q
        for (int64_t e = 0; e + 1 < n; ++e) piq += ae(e) * dh[e] * d.rnu * (q[e + 1] - q[e]);
        const int64_t P = nparams();
        acc[P] -= std::ldexp(piq, wk);  // d/dgamma = 2^k d/d(gamma 2^k)
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double de = dh[e] * d.rnu;
            acc[w.size() == 1 ? 0 : e] -= std::ldexp(gamma * we(e) * de * (2.0 * (q[e + 1] - q[e]) - piq * de), -wk);
        }
    }
};

}  // namespace

fp_status make_tv1d_l2(Vec w, fp_op **out) {
    if (w.empty()) w.assign(1, 1.0);
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    const double m = vmax(w);
    for (double v : w)
        if (v > 0.0 && v < 0x1p-511 * m) return FP_EPARAM;  // its square, over 2^k, not normal (M5)
    *out = new TV1DL2(std::move(w));
    return FP_OK;
}

}  // namespace fp
