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
// psi' needs u^T (W D D^T W + mu I)^{-1} u = sum_e a_e Dz_e Dy_e / mu^3, where
// (mu I + K) y = mu z, and the step is mu + (s - mu gamma) s^2 / (gamma S) with
// s = ||W D z|| and S = sum_e a_e Dz_e Dy_e. At mu = 0 the same quantities come
// from cumulative sums: ||u0||, and ||y0||^2 with y0 the mean-free solution of
// w_e (y0_{e+1} - y0_e) = u0_e. Once a step is below 1e-7 mu it is taken and
// the iteration stops: Newton's quadratic rate puts the next error below
// rounding.
//
// Solves. sigma I + kappa K (sigma = mu, kappa = 1 for mu < 1, else 1 and 1/mu)
// is a diagonally dominant M-matrix. Its pivots are kept in excess form,
// e_j = sigma + kappa a_{j-1} e_{j-1} / d_{j-1}, d_j = e_j + kappa a_j, sums of
// positive terms, so they are accurate for any mu; the piece means are removed
// from x first.
//
// Derivatives. F(z) = z - x + gamma K z / s = 0. Its Jacobian in z is
// H = A - (c / s^2) p p^T with c = gamma / s, A = I + c K and p = K z, and
// (Sherman-Morrison) H^{-1} v = A^{-1} v + c t (t^T v) / (p^T y) with t = A^{-1} p
// and y = A^{-1} z; p^T y = s^2 (1 - (c / s^2) p^T t) is computed as
// sum_e a_e Dz_e Dy_e, without the cancellation. JVP: dz = H^{-1} r with
// r = dx - (dgamma / s) p - gamma sum_e dw_e d(K z / s)/dw_e, and
// d(K z / s)/dw_e = 2 w_e Dz_e D_e^T / s - p w_e Dz_e^2 / s^3. The VJP is the
// adjoint. Inside, the Jacobian is the piecewise averaging.
//
// Kinks: ||u0|| = gamma is inside.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

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

    // 1 / d_j of sigma I + kappa K.
    void factor(double sigma, double kappa, double *inv, int64_t n) const noexcept {
        double e = sigma;
        for (int64_t j = 0; j < n; ++j) {
            const double d = e + (j + 1 < n ? kappa * ae(j) : 0.0);
            inv[j] = 1.0 / d;
            if (j + 1 < n) e = sigma + kappa * ae(j) * e * inv[j];
        }
    }
    // v <- (sigma I + kappa K)^{-1} v, in place.
    void solve(double kappa, const double *inv, double *v, int64_t n) const noexcept {
        for (int64_t j = 1; j < n; ++j) v[j] += kappa * ae(j - 1) * inv[j - 1] * v[j - 1];
        v[n - 1] *= inv[n - 1];
        for (int64_t j = n - 2; j >= 0; --j) v[j] = (v[j] + kappa * ae(j) * v[j + 1]) * inv[j];
    }
    // The Newton loop's passes, fused so that the dependent chains of the
    // factorisation and the substitutions overlap: factor sigma I + kappa K and
    // solve for v = (sigma I + kappa K)^{-1} sigma b, returning ||W D v||^2.
    double factor_solve(double sigma, double kappa, double *inv, const double *b, double *v,
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
    // y = (sigma I + kappa K)^{-1} sigma z with the factors above, returning
    // sum_e a_e Dz_e Dy_e.
    double solve_cross(double sigma, double kappa, const double *inv, const double *z, double *y,
                       int64_t n) const noexcept {
        double f = sigma * z[0];
        y[0] = f;
        for (int64_t j = 1; j < n; ++j) {
            f = sigma * z[j] + kappa * ae(j - 1) * inv[j - 1] * f;
            y[j] = f;
        }
        double S = 0.0, next = y[n - 1] *= inv[n - 1];
        for (int64_t j = n - 2; j >= 0; --j) {
            const double yj = (y[j] + kappa * ae(j) * next) * inv[j];
            y[j] = yj;
            S += ae(j) * (z[j + 1] - z[j]) * (next - yj);
            next = yj;
        }
        return S;
    }
    // xc <- x minus its piece means, taken relative to the piece's first entry,
    // so that the rounding scales with the variation within the piece and a
    // constant piece gives exactly zero. The inside test divides running sums
    // of xc by w_e^2: a rounding residual of eps |x| over a tiny weight would
    // otherwise pass for a huge dual (found by fuzzing: NaN at w_e = 2e-53).
    template <class T>
    void center(const T *x, double *xc, int64_t n) const noexcept {
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e < n - 1 && ae(e) != 0.0) continue;
            const double x0 = static_cast<double>(x[s]);  // piece [s, e]
            double m = 0.0;
            for (int64_t i = s; i <= e; ++i) m += static_cast<double>(x[i]) - x0;
            m /= static_cast<double>(e - s + 1);
            for (int64_t i = s; i <= e; ++i) xc[i] = (static_cast<double>(x[i]) - x0) - m;
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
        double *xc = static_cast<double *>(scr), *inv = xc + n, *zc = inv + n, *y = zc + n;
        center(x, xc, n);
        // ||u0||^2 and ||y0||^2 (y0 per piece, mean-free).
        double rho0sq = 0.0, kap0 = 0.0;
        {
            int64_t s = 0;
            double U = 0.0, yv = 0.0, ys = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                y[i] = yv;
                ys += yv;
                const bool end = i == n - 1 || ae(i) == 0.0;
                if (!end) {
                    U += xc[i];
                    rho0sq += U * U / ae(i);
                    yv -= U / ae(i);
                    continue;
                }
                const double m = ys / static_cast<double>(i - s + 1);
                for (int64_t k = s; k <= i; ++k) kap0 += (y[k] - m) * (y[k] - m);
                s = i + 1;
                U = yv = ys = 0.0;
            }
        }
        const double rho0 = std::sqrt(rho0sq);
        double L = 0.0, R = norm_wd(xc, n) / gamma;
        // R = 0: no differences across nonzero weights, so x is its own
        // piecewise mean whatever rounding left in rho0.
        if (rho0 <= gamma || !(R > 0.0)) {
            for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(static_cast<double>(x[i]) - xc[i]);
            if (code) code[0] = kInside;
            return kRowOk;
        }
        double mu = (rho0 / gamma - 1.0) * rho0sq / kap0;  // Newton from mu = 0
        if (!(mu > L && mu < R)) mu = 0.5 * (L + R);
        // zc = (sigma I + kappa K)^{-1} sigma xc at mu, returning s; y likewise
        // from zc, returning S.
        auto solve_z = [&](double m, double &sigma, double &kappa) {
            sigma = m < 1.0 ? m : 1.0;
            kappa = m < 1.0 ? 1.0 : 1.0 / m;
            return std::sqrt(factor_solve(sigma, kappa, inv, xc, zc, n));
        };
        for (int it = 0; it < kMaxIter; ++it) {
            double sigma, kappa;
            const double s = solve_z(mu, sigma, kappa);
            if (s > mu * gamma) L = mu;
            else if (s < mu * gamma) R = mu;
            else break;
            const double S = solve_cross(sigma, kappa, inv, zc, y, n);
            double next = S > 0.0 ? mu + (s - mu * gamma) * s * s / (gamma * S) : 0.5 * (L + R);
            if (std::fabs(next - mu) <= 1e-7 * mu) {
                if (s > mu * gamma && next > L && next < R) solve_z(next, sigma, kappa);
                break;
            }
            if (!(next > L && next < R)) next = 0.5 * (L + R);
            mu = next;
        }
        for (int64_t i = 0; i < n; ++i)
            z[i] = static_cast<T>(static_cast<double>(x[i]) - (xc[i] - zc[i]));
        return kRowOk;
    }

    // The outside structure at the saved z: s, c, p (into p), and the factored
    // A = I + c K. Returns false when z has no differences (identity).
    template <class T>
    bool setup(const T *z, int64_t n, double gamma, double &s, double &c, double *inv, double *p,
               double *y) const noexcept {
        s = norm_wd(z, n);
        if (!(s > 0.0)) return false;
        c = gamma / s;
        for (int64_t i = 0; i < n; ++i) p[i] = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double v = ae(e) * (static_cast<double>(z[e + 1]) - static_cast<double>(z[e]));
            p[e + 1] += v;
            p[e] -= v;
        }
        factor(1.0, c, inv, n);
        piece_mean(z, y, n);
        for (int64_t i = 0; i < n; ++i) y[i] = static_cast<double>(z[i]) - y[i];
        return true;
    }

    // v <- H^{-1} v given t = A^{-1} p and pty = p^T A^{-1} z.
    void apply_hinv(double c, const double *inv, const double *t, double pty, double *v,
                    int64_t n) const noexcept {
        double tv = 0.0;
        for (int64_t i = 0; i < n; ++i) tv += t[i] * v[i];
        solve(c, inv, v, n);
        const double f = c * tv / pty;
        for (int64_t i = 0; i < n; ++i) v[i] += f * t[i];
    }

    template <class T>
    void jvp(const T *, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n,
             double gamma, double dgamma, const double *dp, void *scr) const noexcept {
        if (code[0] == kInside) {
            piece_mean(dx, dz, n);
            return;
        }
        double *inv = static_cast<double *>(scr), *p = inv + n, *y = p + n, *r = y + n;
        double s, c;
        if (n == 1 || !setup(z, n, gamma, s, c, inv, p, y)) {
            for (int64_t i = 0; i < n; ++i) dz[i] = dx[i];
            return;
        }
        // y = A^{-1} z (centred), p^T y; then t = A^{-1} p into p's buffer, keeping
        // p for r first.
        solve(c, inv, y, n);
        double pty = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double dze = static_cast<double>(z[e + 1]) - static_cast<double>(z[e]);
            pty += ae(e) * dze * (y[e + 1] - y[e]);
        }
        double wsum = 0.0;  // sum_e dw_e w_e Dz_e^2
        for (int64_t i = 0; i < n; ++i) r[i] = static_cast<double>(dx[i]) - dgamma / s * p[i];
        if (dp) {
            for (int64_t e = 0; e + 1 < n; ++e) {
                const double dze = static_cast<double>(z[e + 1]) - static_cast<double>(z[e]);
                const double dwe = dp[w.size() == 1 ? 0 : e];
                const double q = 2.0 * gamma * we(e) * dze * dwe / s;
                r[e] += q;
                r[e + 1] -= q;
                wsum += dwe * we(e) * dze * dze;
            }
            const double f = gamma * wsum / (s * s * s);
            for (int64_t i = 0; i < n; ++i) r[i] += f * p[i];
        }
        solve(c, inv, p, n);  // t
        apply_hinv(c, inv, p, pty, r, n);
        for (int64_t i = 0; i < n; ++i) dz[i] = static_cast<T>(r[i]);
    }

    template <class T>
    void vjp(const T *, const T *z, const uint8_t *code, const T *gv, T *gx, int64_t n,
             double gamma, double *acc, void *scr) const noexcept {
        if (code[0] == kInside) {
            piece_mean(gv, gx, n);
            return;
        }
        double *inv = static_cast<double *>(scr), *p = inv + n, *y = p + n, *q = y + n;
        double s, c;
        if (n == 1 || !setup(z, n, gamma, s, c, inv, p, y)) {
            for (int64_t i = 0; i < n; ++i) gx[i] = gv[i];
            return;
        }
        solve(c, inv, y, n);
        double pty = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double dze = static_cast<double>(z[e + 1]) - static_cast<double>(z[e]);
            pty += ae(e) * dze * (y[e + 1] - y[e]);
        }
        // y is free again: keep p there, and t = A^{-1} p in p's buffer.
        std::memcpy(y, p, static_cast<size_t>(n) * sizeof(double));
        solve(c, inv, p, n);
        for (int64_t i = 0; i < n; ++i) q[i] = static_cast<double>(gv[i]);
        apply_hinv(c, inv, p, pty, q, n);
        for (int64_t i = 0; i < n; ++i) gx[i] = static_cast<T>(q[i]);
        double pq = 0.0;
        for (int64_t i = 0; i < n; ++i) pq += y[i] * q[i];
        const int64_t P = nparams();
        acc[P] -= pq / s;
        for (int64_t e = 0; e + 1 < n; ++e) {
            const double dze = static_cast<double>(z[e + 1]) - static_cast<double>(z[e]);
            acc[w.size() == 1 ? 0 : e] -=
                gamma * we(e) * dze * (2.0 * (q[e + 1] - q[e]) / s - pq * dze / (s * s * s));
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
