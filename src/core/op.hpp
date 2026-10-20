// Operator base class behind the opaque fp_op handle.
//
// Two kinds of operator exist. Elementwise operators (soft thresholding, box,
// ...) map each entry independently and are driven over the whole array in
// fixed chunks. Row operators (projections, group norms, 1D TV, ...) couple
// the entries along the operator axis and are driven one contiguous row at a
// time. In both cases the kernels are templates on the scalar type; the CRTP
// helpers below turn them into the float and double virtual overloads.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "fastprox.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

// Target of min_nonan / max_nonan (see there).
#if defined(__aarch64__) || defined(_M_ARM64)
#define FP_MINMAX_ARM64 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define FP_MINMAX_SSE2 1
#include <emmintrin.h>
#endif

namespace fp {

// Derivative of one entry of an elementwise prox:
//   dz = j dx + dg dgamma + sum_k pv[k] dtheta[pi[k]].
struct ElemDeriv {
    double j = 0.0;
    double dg = 0.0;
    int np = 0;
    int64_t pi[3] = {0, 0, 0};
    double pv[3] = {0.0, 0.0, 0.0};
};

// A breakpoint of an elementwise map near an entry: its position b (in
// signed x) and the magnitude of the terms it is computed from, which sets
// the step used to land on either side of it.
struct Kink {
    double b = 0.0, scale = 0.0;
};
// A piece of an elementwise map: an operator-specific id and the sign that
// its formulas use (sgn(x) by default, the side's sign across a kink).
struct Piece {
    int id = 0;
    double s = 0.0;
};

// Status returned by row kernels.
enum RowStatus : int { kRowOk = 0, kRowNonFinite = 1 };

// One image handed to an image operator's prox (see fp_op::image_ndim). The
// image is contiguous in C order. The kernel may spread its own work over nt
// threads (thread scratch t at thread_base + t * thread_bytes), and must give
// the same result for any nt.
struct ImageCall {
    const int64_t *shape = nullptr;  // image_ndim() extents
    int64_t numel = 0;
    double gamma = 0.0;
    double tol = 0.0;  // stop when gap <= tol * ||x||^2 / 2
    int max_iter = 0;
    int nt = 1;
    char *work = nullptr;  // image_bytes(shape)
    char *thread_base = nullptr;
    size_t thread_bytes = 0;
    double *dual = nullptr;  // image_ndim() * numel: read when warm, always written
    bool warm = false;
    char *save = nullptr;  // structure for jvp/vjp (struct_bytes), null unless kept
    // Results.
    double gap = 0.0;
    int iters = 0;
    bool converged = true;

    char *thread_scratch(int t) const noexcept { return thread_base + static_cast<size_t>(t) * thread_bytes; }
};

// Iterative derivative solves of an image operator (conjugate gradients):
// tolerance on the relative residual and iteration cap in, the achieved
// residual and iterations out. Operators with closed-form derivatives leave
// the outputs at zero.
struct DerivCtl {
    double tol = 0.0;  // 0: the operator's default
    int max_iter = 10000;
    double residual = 0.0;
    int iters = 0;
};

}  // namespace fp

// Global namespace: this is the type the C header declares as opaque.
struct fp_op {
    virtual ~fp_op() = default;

    virtual const char *name() const noexcept = 0;
    virtual bool elementwise() const noexcept = 0;
    virtual int64_t nparams() const noexcept = 0;
    // Validates the operator against the length of the operator axis.
    virtual fp_status check_n(int64_t n) const noexcept = 0;
    // Validates the step the kernel will use (1/sigma after fp_prox_conj):
    // FP_EPARAM where the prox is not single-valued (SCAD, MCP).
    virtual fp_status check_step(double) const noexcept { return FP_OK; }
    // True when the prox can be set-valued at thresholds (nonconvex maps):
    // prox_run then applies the tie rule and counts the entries on a tie.

    // ---- elementwise interface -------------------------------------------
    // True when some parameter varies along the operator axis.
    virtual bool axis_params() const noexcept { return false; }
    virtual bool has_ties() const noexcept { return false; }

    // z[i*zs] = prox(x[i*xs]) for i < len. widx is the axis coordinate of the
    // first entry and wstep its increment along the run. save, if not null,
    // receives the (scaled) input contiguously. In conjugate mode the run
    // computes y - sigma prox_{g/sigma}(y/sigma) and gamma is 1/sigma. tie is
    // the fp_tie rule; the return value counts the entries that sat on a tie.
    virtual int64_t prox_run(const float *, int64_t, float *, int64_t, int64_t, double, int64_t,
                             int64_t, float *, bool, double, int) const noexcept { return 0; }
    virtual int64_t prox_run(const double *, int64_t, double *, int64_t, int64_t, double, int64_t,
                             int64_t, double *, bool, double, int) const noexcept { return 0; }
    // xsave is contiguous; dx may be null. tie: the rule of the prox call.
    // side, if not null, is the branch override along the run (contiguous,
    // -1, 0 or +1 per entry), applied to the entries within kink_tol of a
    // breakpoint (operators with has_kinks()).
    virtual void jvp_run(const float *, const float *, int64_t, float *, int64_t, int64_t, double,
                         double, const double *, int64_t, int64_t, bool, double, int,
                         const int8_t *, double) const noexcept {}
    virtual void jvp_run(const double *, const double *, int64_t, double *, int64_t, int64_t,
                         double, double, const double *, int64_t, int64_t, bool,
                         double, int, const int8_t *, double) const noexcept {}
    // acc has nparams() + 2 slots: theta..., gamma, conjugate extra term.
    virtual void vjp_run(const float *, const float *, int64_t, float *, int64_t, int64_t, double,
                         double *, int64_t, int64_t, bool, double, int, const int8_t *,
                         double) const noexcept {}
    virtual void vjp_run(const double *, const double *, int64_t, double *, int64_t, int64_t,
                         double, double *, int64_t, int64_t, bool, double, int, const int8_t *,
                         double) const noexcept {}
    // Kinks (elementwise operators with has_kinks()): for each saved entry of
    // the run, the code of the breakpoint within tol of it (0: none) into
    // codes (may be null); returns the number of flagged entries.
    virtual bool has_kinks() const noexcept { return false; }
    virtual int64_t kink_run(const float *, int64_t, double, int64_t, int64_t, double,
                             int8_t *) const noexcept { return 0; }
    virtual int64_t kink_run(const double *, int64_t, double, int64_t, int64_t, double,
                             int8_t *) const noexcept { return 0; }
    // Kinks of row and image operators with has_kinks() (since 0.3), from the
    // saved input x and solution z (in double) and, for images, the dual of
    // the call: codes into mask (one per entry, at the given strides, the
    // image's axes for kink_image; never null for kink_image); returns the
    // number of flagged entries. kink_image gets the image's saved structure
    // and image_bytes(shape) of work.
    virtual int64_t kink_row(const double *, const double *, int64_t, double, double, int8_t *, int64_t,
                             void *) const noexcept { return 0; }
    virtual int64_t kink_image(const char *, const double *, const double *, const double *, const int64_t *,
                               double, double, int8_t *, const int64_t *, char *) const noexcept { return 0; }
    // Kink sites per image when they are not the image's entries (since 0.4):
    // tree TV's m edges, so that its masks and sides have the shape of x with
    // the last axis of length m. 0: one site per entry.
    virtual int64_t kink_sites() const noexcept { return 0; }
    // Branch overrides (fp_workspace_set_branch): elementwise operators with
    // kinks, and since 0.3 the row and image operators that implement them.
    virtual bool has_branches() const noexcept { return elementwise() && has_kinks(); }
    // Rows: rewrite the saved codes of one row at its kinks within tol by side
    // (one per entry, -1 / 0 / +1), from the saved x and z (in double).
    // kink_row and branch_row get scratch_bytes(n) of scratch.
    virtual void branch_row(const double *, const double *, uint8_t *, int64_t, double, double,
                            const int8_t *, void *) const noexcept {}
    // Images: rebuild the saved structure of one image from x, z (double) and
    // the dual, with side (C order, never null) applied at its kinks within
    // tol; work: image_bytes(shape) of scratch. The API restores the prox
    // call's own structure by a copy.
    virtual void branch_image(char *, const double *, const double *, const double *, const int64_t *,
                              double, double, const int8_t *, char *) const noexcept {}

    // ---- row interface ---------------------------------------------------
    // Bytes of kernel scratch needed for one row of length n.
    virtual size_t scratch_bytes(int64_t) const noexcept { return 0; }

    // The value g(x) (since 0.5, fp_value), in double: of one row of length n
    // (elementwise and row operators) or one image of the
    // given shape (image operators). The indicator of a set is 0 when x lies
    // in it up to atol = tol * max(1, max |x_i|) and INFINITY otherwise. scr has
// value_doubles(n) doubles.
    // Spectral use (since 1.1, fp_op_spectral_check): kSpectralSvd when the
    // map is absolutely symmetric (permutation and sign invariant, so it can
    // act on singular values), kSpectralEig when it is permutation invariant
    // (eigenvalues); both bits for the former.
    virtual int spectral() const noexcept { return 0; }
    virtual bool has_value() const noexcept { return false; }
    virtual int64_t value_doubles(int64_t n) const noexcept { return n; }  // scr's length
    virtual double value_row(const double *, int64_t, double, double *) const noexcept { return 0.0; }
    virtual double value_image(const double *, const int64_t *, double) const noexcept { return 0.0; }
    // 1D TV only: the row prox with edge weights given per call (n of them,
    // the last 0), for image operators with weights per edge.
    virtual int prox_edges(const double *, double *, int64_t, double, const double *, void *) const noexcept {
        return fp::kRowNonFinite;
    }
    // True when the kernel cannot write its output over its input.
    virtual bool needs_distinct() const noexcept { return false; }
    // Rows are contiguous. code, if not null, receives n bytes of structure.
    virtual int prox_row(const float *, float *, int64_t, double, void *,
                         uint8_t *) const noexcept {
        return fp::kRowOk;
    }
    virtual int prox_row(const double *, double *, int64_t, double, void *,
                         uint8_t *) const noexcept {
        return fp::kRowOk;
    }
    // x, z, code: the saved row. dx is never null; dparams may be null.
    virtual void jvp_row(const float *, const float *, const uint8_t *, const float *, float *,
                         int64_t, double, double, const double *, void *) const noexcept {}
    virtual void jvp_row(const double *, const double *, const uint8_t *, const double *,
                         double *, int64_t, double, double, const double *,
                         void *) const noexcept {}
    // gx = J^T g; acc[0..P) += (dz/dtheta)^T g, acc[P] += <dz/dgamma, g>.
    virtual void vjp_row(const float *, const float *, const uint8_t *, const float *, float *,
                         int64_t, double, double *, void *) const noexcept {}
    virtual void vjp_row(const double *, const double *, const uint8_t *, const double *,
                         double *, int64_t, double, double *, void *) const noexcept {}

    // ---- image interface -------------------------------------------------
    // Image operators (multi-dimensional TV) act on the last image_ndim() axes
    // and are batched over the others; 0 for every other operator.
    virtual int image_ndim() const noexcept { return 0; }
    virtual fp_status check_image(const int64_t *) const noexcept { return FP_OK; }
    // Bytes of the working set of one image, of the per-thread scratch used
    // inside one image, and of the structure saved for derivatives.
    virtual size_t image_bytes(const int64_t *) const noexcept { return 0; }
    virtual size_t image_thread_bytes(const int64_t *) const noexcept { return 0; }
    virtual size_t struct_bytes(const int64_t *) const noexcept { return 0; }
    // Returns a RowStatus. Fills c.gap, c.iters, c.converged.
    virtual int prox_image(const float *, float *, fp::ImageCall &) const noexcept { return fp::kRowOk; }
    virtual int prox_image(const double *, double *, fp::ImageCall &) const noexcept { return fp::kRowOk; }
    // Derivatives at the structure saved by prox_image. dx is never null;
    // dparams may be null. work: image_bytes(shape) of scratch.
    virtual void jvp_image(const char *, const float *, float *, const int64_t *, double, double,
                           const double *, char *, fp::DerivCtl &) const noexcept {}
    virtual void jvp_image(const char *, const double *, double *, const int64_t *, double, double,
                           const double *, char *, fp::DerivCtl &) const noexcept {}
    // gx = J^T g; acc[0..P) += (dz/dtheta)^T g, acc[P] += <dz/dgamma, g>.
    virtual void vjp_image(const char *, const float *, float *, const int64_t *, double, double *,
                           char *, fp::DerivCtl &) const noexcept {}
    virtual void vjp_image(const char *, const double *, double *, const int64_t *, double, double *,
                           char *, fp::DerivCtl &) const noexcept {}
};

namespace fp {

// CRTP base for elementwise operators. D provides
//   template <class T> T prox1(T x, double gamma, int64_t widx) const noexcept;
//   ElemDeriv deriv1(double x, double gamma, int64_t widx) const noexcept;
// Per-entry evaluation of an elementwise operator D: the tie-aware form
// (D::kTies, nonconvex maps) takes the tie rule and reports a tie; the plain
// form ignores both.
// Bits of fp_op::spectral(): absolutely symmetric maps act on both kinds.
constexpr int kSpectralEig = 1, kSpectralSvd = 2, kSpectralBoth = kSpectralEig | kSpectralSvd;
// Spectral use needs the same map for every entry: no weights, or equal ones.
inline bool uniform(const std::vector<double> &w) noexcept {
    for (double v : w)
        if (v != w[0]) return false;
    return true;
}

// atol of a set's indicator for the row x (see fp_op::value_row).
inline double value_atol(const double *x, int64_t n, double tol) noexcept {
    double m = 1.0;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    return tol * m;
}
// 0 when the violation is within atol, else INFINITY.
inline double indicator(double violation, double atol) noexcept {
    return violation <= atol ? 0.0 : std::numeric_limits<double>::infinity();
}

template <class D>
struct ElemCall {
    template <class T>
    static T prox(const D &d, T x, double gamma, int64_t w, int tie, bool &tied) noexcept {
        if constexpr (D::kTies) return d.template prox1<T>(x, gamma, w, tie, tied);
        else return d.template prox1<T>(x, gamma, w);
    }
    static ElemDeriv deriv(const D &d, double x, double gamma, int64_t w, int tie) noexcept {
        if constexpr (D::kTies) return d.deriv1(x, gamma, w, tie);
        else return d.deriv1(x, gamma, w);
    }
};

// CRTP base for elementwise operators. D provides
//   template <class T> T prox1(T x, double gamma, int64_t widx) const noexcept;
//   ElemDeriv deriv1(double x, double gamma, int64_t widx) const noexcept;
// or, with static constexpr bool kTies = true, the same with an extra tie
// rule argument (and a bool &tied out-argument for prox1).
template <class D>
struct ElemOp : fp_op {
    static constexpr bool kTies = false;
    // Operators with kinks (kKinks = true) also provide
    //   int kink1(double x, double gamma, int64_t widx, double tol, Kink &k);
    //   Piece piece_of(double x, double gamma, int64_t widx);
    //   ElemDeriv deriv_piece(double x, double gamma, int64_t widx, Piece p);
    // kink1 returns the code of the breakpoint nearest to x within tol (0 if
    // none) and fills k; deriv1 must equal deriv_piece(x, piece_of(x)).
    static constexpr bool kKinks = false;
    bool elementwise() const noexcept override { return true; }
    bool has_ties() const noexcept override { return D::kTies; }
    bool has_kinks() const noexcept override { return D::kKinks; }

    // g(x) = sum_i of D's value1(x_i, i, atol), the term of entry i (atol for
    // the indicators of sets, see fp_op::value_row).
    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        const D &d = static_cast<const D &>(*this);
        const double atol = value_atol(x, n, tol);
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += d.value1(x[i], i, atol);
        return s;
    }

    // The derivative used at entry x: the default rule, or with a side
    // override at a flagged entry the piece on that side of its breakpoint
    // (the side of b + side * delta), whose formula is evaluated at x.
    static ElemDeriv deriv_at(const D &d, double x, double gamma, int64_t w, int tie, int side,
                              double tol) noexcept {
        if constexpr (D::kKinks) {
            if (side != 0) {
                Kink k;
                if (d.kink1(x, gamma, w, tol, k)) {
                    const double delta = 32.0 * 2.220446049250313e-16 * k.scale + 1e-300;
                    return d.deriv_piece(x, gamma, w, d.piece_of(k.b + side * delta, gamma, w));
                }
            }
        }
        return ElemCall<D>::deriv(d, x, gamma, w, tie);
    }

    template <class T>
    int64_t run_kinks(const T *xsave, int64_t len, double gamma, int64_t widx, int64_t wstep,
                      double tol, int8_t *codes) const noexcept {
        int64_t count = 0;
        if constexpr (D::kKinks) {
            const D &d = static_cast<const D &>(*this);
            for (int64_t i = 0; i < len; ++i) {
                Kink k;
                const int c = d.kink1(static_cast<double>(xsave[i]), gamma, widx + i * wstep, tol, k);
                if (codes) codes[i] = static_cast<int8_t>(c);
                count += c != 0;
            }
        }
        return count;
    }

    template <class T>
    int64_t run_prox(const T *x, int64_t xs, T *z, int64_t zs, int64_t len, double gamma,
                     int64_t widx, int64_t wstep, T *save, bool conj, double sigma,
                     int tie) const noexcept {
        const D &d = static_cast<const D &>(*this);
        using C = ElemCall<D>;
        int64_t ties = 0;
        bool tied = false;
        if (conj) {
            for (int64_t i = 0; i < len; ++i) {
                const double y = static_cast<double>(x[i * xs]);
                const T t = static_cast<T>(y / sigma);
                if (save) save[i] = t;
                const double p = static_cast<double>(C::template prox<T>(d, t, gamma, widx + i * wstep, tie, tied));
                ties += tied;
                z[i * zs] = static_cast<T>(y - sigma * p);
            }
        } else if (save) {
            for (int64_t i = 0; i < len; ++i) {
                const T xi = x[i * xs];
                save[i] = xi;
                z[i * zs] = C::template prox<T>(d, xi, gamma, widx + i * wstep, tie, tied);
                ties += tied;
            }
        } else if (xs == 1 && zs == 1) {
            for (int64_t i = 0; i < len; ++i) {
                z[i] = C::template prox<T>(d, x[i], gamma, widx + i * wstep, tie, tied);
                ties += tied;
            }
        } else {
            for (int64_t i = 0; i < len; ++i) {
                z[i * zs] = C::template prox<T>(d, x[i * xs], gamma, widx + i * wstep, tie, tied);
                ties += tied;
            }
        }
        return ties;
    }

    // Overrides are rare: the loops below are instantiated with and without
    // them, so the default loop is unchanged.
    template <class T>
    void run_jvp(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,
                 double gamma, double dgamma, const double *dparams, int64_t widx, int64_t wstep,
                 bool conj, double sigma, int tie, const int8_t *side, double tol) const noexcept {
        if (side && D::kKinks)
            run_jvp_<T, true>(xsave, dx, dxs, dz, dzs, len, gamma, dgamma, dparams, widx, wstep, conj,
                              sigma, tie, side, tol);
        else
            run_jvp_<T, false>(xsave, dx, dxs, dz, dzs, len, gamma, dgamma, dparams, widx, wstep, conj,
                               sigma, tie, nullptr, tol);
    }
    template <class T>
    void run_vjp(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,
                 double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,
                 double sigma, int tie, const int8_t *side, double tol) const noexcept {
        if (side && D::kKinks)
            run_vjp_<T, true>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma, tie,
                              side, tol);
        else
            run_vjp_<T, false>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma, tie,
                               nullptr, tol);
    }

    template <class T, bool Side>
    void run_jvp_(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,
                  double gamma, double dgamma, const double *dparams, int64_t widx, int64_t wstep,
                  bool conj, double sigma, int tie, const int8_t *side, double tol) const noexcept {
        const D &d = static_cast<const D &>(*this);
        using C = ElemCall<D>;
        bool tied = false;
        for (int64_t i = 0; i < len; ++i) {
            const int64_t w = widx + i * wstep;
            const double xi = static_cast<double>(xsave[i]);
            ElemDeriv e;
            if constexpr (Side) e = deriv_at(d, xi, gamma, w, tie, side[i], tol);
            else e = C::deriv(d, xi, gamma, w, tie);
            const double dxi = dx ? static_cast<double>(dx[i * dxs]) : 0.0;
            double ptheta = 0.0;
            if (dparams)
                for (int k = 0; k < e.np; ++k) ptheta += e.pv[k] * dparams[e.pi[k]];
            if (!conj) {
                dz[i * dzs] = static_cast<T>(e.j * dxi + e.dg * dgamma + ptheta);
            } else {
                // out = y - sigma p(gamma = 1/sigma, t = y/sigma), dgamma reads as dsigma.
                const double p = static_cast<double>(C::template prox<double>(d, xi, gamma, w, tie, tied));
                const double dt = dxi / sigma - xi * dgamma / sigma;
                const double dgam = -dgamma / (sigma * sigma);
                const double dp = e.j * dt + e.dg * dgam + ptheta;
                dz[i * dzs] = static_cast<T>(dxi - dgamma * p - sigma * dp);
            }
        }
    }

    template <class T, bool Side>
    void run_vjp_(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,
                  double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,
                  double, int tie, const int8_t *side, double tol) const noexcept {
        const D &d = static_cast<const D &>(*this);
        using C = ElemCall<D>;
        bool tied = false;
        const int64_t P = this->nparams();
        for (int64_t i = 0; i < len; ++i) {
            const int64_t w = widx + i * wstep;
            const double xi = static_cast<double>(xsave[i]);
            ElemDeriv e;
            if constexpr (Side) e = deriv_at(d, xi, gamma, w, tie, side[i], tol);
            else e = C::deriv(d, xi, gamma, w, tie);
            const double gi = static_cast<double>(g[i * gs]);
            for (int k = 0; k < e.np; ++k) acc[e.pi[k]] += e.pv[k] * gi;
            acc[P] += e.dg * gi;
            if (!conj) {
                gx[i * gxs] = static_cast<T>(e.j * gi);
            } else {
                const double p = static_cast<double>(C::template prox<double>(d, xi, gamma, w, tie, tied));
                acc[P + 1] += -gi * p + e.j * gi * xi;
                gx[i * gxs] = static_cast<T>(gi - e.j * gi);
            }
        }
    }

#define FP_ELEM_FORWARD(T)                                                                       \
    int64_t prox_run(const T *x, int64_t xs, T *z, int64_t zs, int64_t len, double gamma,        \
                     int64_t widx, int64_t wstep, T *save, bool conj, double sigma, int tie)     \
        const noexcept override {                                                                \
        return run_prox<T>(x, xs, z, zs, len, gamma, widx, wstep, save, conj, sigma, tie);       \
    }                                                                                            \
    void jvp_run(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,      \
                 double gamma, double dgamma, const double *dparams, int64_t widx,               \
                 int64_t wstep, bool conj, double sigma, int tie, const int8_t *side,            \
                 double tol) const noexcept override {                                           \
        run_jvp<T>(xsave, dx, dxs, dz, dzs, len, gamma, dgamma, dparams, widx, wstep, conj,      \
                   sigma, tie, side, tol);                                                       \
    }                                                                                            \
    void vjp_run(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,        \
                 double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,              \
                 double sigma, int tie, const int8_t *side, double tol) const noexcept override {\
        run_vjp<T>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma, tie, side,  \
                   tol);                                                                         \
    }                                                                                            \
    int64_t kink_run(const T *xsave, int64_t len, double gamma, int64_t widx, int64_t wstep,     \
                     double tol, int8_t *codes) const noexcept override {                        \
        return run_kinks<T>(xsave, len, gamma, widx, wstep, tol, codes);                         \
    }
    FP_ELEM_FORWARD(float)
    FP_ELEM_FORWARD(double)
#undef FP_ELEM_FORWARD
};

// CRTP base for row operators. D provides templates prox, jvp and vjp with
// the signatures of prox_row, jvp_row and vjp_row.
template <class D>
struct RowOp : fp_op {
    bool elementwise() const noexcept override { return false; }

#define FP_ROW_FORWARD(T)                                                                        \
    int prox_row(const T *x, T *z, int64_t n, double gamma, void *scratch, uint8_t *code)        \
        const noexcept override {                                                                \
        return static_cast<const D *>(this)->template prox<T>(x, z, n, gamma, scratch, code);    \
    }                                                                                            \
    void jvp_row(const T *x, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n,     \
                 double gamma, double dgamma, const double *dparams, void *scratch)              \
        const noexcept override {                                                                \
        static_cast<const D *>(this)->template jvp<T>(x, z, code, dx, dz, n, gamma, dgamma,      \
                                                      dparams, scratch);                         \
    }                                                                                            \
    void vjp_row(const T *x, const T *z, const uint8_t *code, const T *g, T *gx, int64_t n,      \
                 double gamma, double *acc, void *scratch) const noexcept override {             \
        static_cast<const D *>(this)->template vjp<T>(x, z, code, g, gx, n, gamma, acc,          \
                                                      scratch);                                  \
    }
    FP_ROW_FORWARD(float)
    FP_ROW_FORWARD(double)
#undef FP_ROW_FORWARD
};

// CRTP base for image operators. D provides templates prox_img, jvp_img and
// vjp_img with the signatures of prox_image, jvp_image and vjp_image.
template <class D>
struct ImageOp : fp_op {
    bool elementwise() const noexcept override { return false; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

#define FP_IMAGE_FORWARD(T)                                                                      \
    int prox_image(const T *x, T *z, ImageCall &c) const noexcept override {                     \
        return static_cast<const D *>(this)->template prox_img<T>(x, z, c);                      \
    }                                                                                            \
    void jvp_image(const char *save, const T *dx, T *dz, const int64_t *shape, double gamma,     \
                   double dgamma, const double *dparams, char *work, DerivCtl &dc)               \
        const noexcept override {                                                                \
        static_cast<const D *>(this)->template jvp_img<T>(save, dx, dz, shape, gamma, dgamma,    \
                                                          dparams, work, dc);                    \
    }                                                                                            \
    void vjp_image(const char *save, const T *g, T *gx, const int64_t *shape, double gamma,      \
                   double *acc, char *work, DerivCtl &dc) const noexcept override {              \
        static_cast<const D *>(this)->template vjp_img<T>(save, g, gx, shape, gamma, acc, work,  \
                                                          dc);                                   \
    }
    FP_IMAGE_FORWARD(float)
    FP_IMAGE_FORWARD(double)
#undef FP_IMAGE_FORWARD
};

// Bit layout of the exponent field, to test finiteness without branches.
template <class T>
struct FloatBits;
template <>
struct FloatBits<float> {
    using U = uint32_t;
    static constexpr U exp = 0x7f800000u;
};
template <>
struct FloatBits<double> {
    using U = uint64_t;
    static constexpr U exp = 0x7ff0000000000000ull;
};

// True when every entry is finite. A branch-free pass over the bit patterns,
// which the compiler vectorises; the common case costs a fraction of a
// nanosecond per entry.
template <class T>
inline bool all_finite(const T *x, int64_t n) noexcept {
    using U = typename FloatBits<T>::U;
    constexpr U e = FloatBits<T>::exp;
    U bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        U b;
        std::memcpy(&b, x + i, sizeof b);
        bad |= static_cast<U>((b & e) == e);
    }
    return bad == 0;
}

// Minimum and maximum of two values known to be NaN-free (or whose NaN
// inputs are tested separately), as one instruction and never a branch:
// fminnm/fmaxnm on arm64 and minsd/maxsd (minss/maxss) on x86-64, where
// SSE2 is part of the base architecture. Compilers do not reliably turn a
// compare-select into these: GCC emitted compare-and-branch for both the
// select and the ternary forms, which mispredict in Condat's bound updates.
// minsd(v, a) returns v < a ? v : a, so the result is that of the portable
// form, which other targets use.

template <class T>
inline T min_nonan(T a, T v) noexcept {
#if defined(FP_MINMAX_ARM64)
    return std::fmin(a, v);
#elif defined(FP_MINMAX_SSE2)
    if constexpr (std::is_same_v<T, double>)
        return _mm_cvtsd_f64(_mm_min_sd(_mm_set_sd(v), _mm_set_sd(a)));
    else
        return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(v), _mm_set_ss(a)));
#else
    return v < a ? v : a;
#endif
}
template <class T>
inline T max_nonan(T a, T v) noexcept {
#if defined(FP_MINMAX_ARM64)
    return std::fmax(a, v);
#elif defined(FP_MINMAX_SSE2)
    if constexpr (std::is_same_v<T, double>)
        return _mm_cvtsd_f64(_mm_max_sd(_mm_set_sd(v), _mm_set_sd(a)));
    else
        return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(v), _mm_set_ss(a)));
#else
    return v > a ? v : a;
#endif
}

// True when every entry is finite; then lo and hi are the smallest and
// largest entries. One pass in independent lanes: a single running minimum
// is a chain of dependent operations (2 to 4 cycles per entry), which the
// compiler may not reorder without fast-math. On finite input the order of
// the comparisons changes at most the sign of a zero extreme.
template <class T>
inline bool finite_range(const T *x, int64_t n, double &lo, double &hi) noexcept {
    using U = typename FloatBits<T>::U;
    constexpr U e = FloatBits<T>::exp;
    constexpr int64_t kLanes = 8;  // measured: 4 to 16 are close; 1 is twice as slow
    U bad[kLanes];
    T mn[kLanes], mx[kLanes];
    for (int64_t j = 0; j < kLanes; ++j) {
        bad[j] = 0;
        mn[j] = mx[j] = x[0];
    }
    auto step = [&](int64_t j, T v) {
        U b;
        std::memcpy(&b, &v, sizeof b);
        bad[j] |= static_cast<U>((b & e) == e);
        mn[j] = min_nonan(mn[j], v);
        mx[j] = max_nonan(mx[j], v);
    };
    int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes)
        for (int64_t j = 0; j < kLanes; ++j) step(j, x[i + j]);
    for (; i < n; ++i) step(0, x[i]);
    for (int64_t j = 1; j < kLanes; ++j) step(0, mn[j]), step(0, mx[j]), bad[0] |= bad[j];
    lo = static_cast<double>(mn[0]);
    hi = static_cast<double>(mx[0]);
    return bad[0] == 0;
}

// Scans a row: returns kRowNonFinite on any infinity and sets has_nan.
template <class T>
inline int scan_row(const T *x, int64_t n, bool &has_nan) noexcept {
    has_nan = false;
    if (all_finite(x, n)) return kRowOk;
    for (int64_t i = 0; i < n; ++i) {
        const T v = x[i];
        if (v != v) {
            has_nan = true;
        } else if (std::isinf(v)) {
            return kRowNonFinite;
        }
    }
    return kRowOk;
}

template <class T>
inline void fill_nan(T *z, int64_t n) noexcept {
    const T q = std::numeric_limits<T>::quiet_NaN();
    for (int64_t i = 0; i < n; ++i) z[i] = q;
}

}  // namespace fp
