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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

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
    int64_t pi[2] = {0, 0};
    double pv[2] = {0.0, 0.0};
};

// Status returned by row kernels.
enum RowStatus : int { kRowOk = 0, kRowNonFinite = 1 };

}  // namespace fp

// Global namespace: this is the type the C header declares as opaque.
struct fp_op {
    virtual ~fp_op() = default;

    virtual const char *name() const noexcept = 0;
    virtual bool elementwise() const noexcept = 0;
    virtual int64_t nparams() const noexcept = 0;
    // Validates the operator against the length of the operator axis.
    virtual fp_status check_n(int64_t n) const noexcept = 0;

    // ---- elementwise interface -------------------------------------------
    // True when some parameter varies along the operator axis.
    virtual bool axis_params() const noexcept { return false; }

    // z[i*zs] = prox(x[i*xs]) for i < len. widx is the axis coordinate of the
    // first entry and wstep its increment along the run. save, if not null,
    // receives the (scaled) input contiguously. In conjugate mode the run
    // computes y - sigma prox_{g/sigma}(y/sigma) and gamma is 1/sigma.
    virtual void prox_run(const float *, int64_t, float *, int64_t, int64_t, double, int64_t,
                          int64_t, float *, bool, double) const noexcept {}
    virtual void prox_run(const double *, int64_t, double *, int64_t, int64_t, double, int64_t,
                          int64_t, double *, bool, double) const noexcept {}
    // xsave is contiguous; dx may be null.
    virtual void jvp_run(const float *, const float *, int64_t, float *, int64_t, int64_t, double,
                         double, const double *, int64_t, int64_t, bool, double) const noexcept {}
    virtual void jvp_run(const double *, const double *, int64_t, double *, int64_t, int64_t,
                         double, double, const double *, int64_t, int64_t, bool,
                         double) const noexcept {}
    // acc has nparams() + 2 slots: theta..., gamma, conjugate extra term.
    virtual void vjp_run(const float *, const float *, int64_t, float *, int64_t, int64_t, double,
                         double *, int64_t, int64_t, bool, double) const noexcept {}
    virtual void vjp_run(const double *, const double *, int64_t, double *, int64_t, int64_t,
                         double, double *, int64_t, int64_t, bool, double) const noexcept {}

    // ---- row interface ---------------------------------------------------
    // Bytes of kernel scratch needed for one row of length n.
    virtual size_t scratch_bytes(int64_t) const noexcept { return 0; }
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
};

namespace fp {

// CRTP base for elementwise operators. D provides
//   template <class T> T prox1(T x, double gamma, int64_t widx) const noexcept;
//   ElemDeriv deriv1(double x, double gamma, int64_t widx) const noexcept;
template <class D>
struct ElemOp : fp_op {
    bool elementwise() const noexcept override { return true; }

    template <class T>
    void run_prox(const T *x, int64_t xs, T *z, int64_t zs, int64_t len, double gamma,
                  int64_t widx, int64_t wstep, T *save, bool conj, double sigma) const noexcept {
        const D &d = static_cast<const D &>(*this);
        if (conj) {
            for (int64_t i = 0; i < len; ++i) {
                const double y = static_cast<double>(x[i * xs]);
                const T t = static_cast<T>(y / sigma);
                if (save) save[i] = t;
                const double p = static_cast<double>(d.template prox1<T>(t, gamma, widx + i * wstep));
                z[i * zs] = static_cast<T>(y - sigma * p);
            }
        } else if (save) {
            for (int64_t i = 0; i < len; ++i) {
                const T xi = x[i * xs];
                save[i] = xi;
                z[i * zs] = d.template prox1<T>(xi, gamma, widx + i * wstep);
            }
        } else if (xs == 1 && zs == 1) {
            for (int64_t i = 0; i < len; ++i) z[i] = d.template prox1<T>(x[i], gamma, widx + i * wstep);
        } else {
            for (int64_t i = 0; i < len; ++i)
                z[i * zs] = d.template prox1<T>(x[i * xs], gamma, widx + i * wstep);
        }
    }

    template <class T>
    void run_jvp(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,
                 double gamma, double dgamma, const double *dparams, int64_t widx, int64_t wstep,
                 bool conj, double sigma) const noexcept {
        const D &d = static_cast<const D &>(*this);
        for (int64_t i = 0; i < len; ++i) {
            const int64_t w = widx + i * wstep;
            const double xi = static_cast<double>(xsave[i]);
            const ElemDeriv e = d.deriv1(xi, gamma, w);
            const double dxi = dx ? static_cast<double>(dx[i * dxs]) : 0.0;
            double ptheta = 0.0;
            if (dparams)
                for (int k = 0; k < e.np; ++k) ptheta += e.pv[k] * dparams[e.pi[k]];
            if (!conj) {
                dz[i * dzs] = static_cast<T>(e.j * dxi + e.dg * dgamma + ptheta);
            } else {
                // out = y - sigma p(gamma = 1/sigma, t = y/sigma), dgamma reads as dsigma.
                const double p = static_cast<double>(d.template prox1<double>(xi, gamma, w));
                const double dt = dxi / sigma - xi * dgamma / sigma;
                const double dgam = -dgamma / (sigma * sigma);
                const double dp = e.j * dt + e.dg * dgam + ptheta;
                dz[i * dzs] = static_cast<T>(dxi - dgamma * p - sigma * dp);
            }
        }
    }

    template <class T>
    void run_vjp(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,
                 double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,
                 double) const noexcept {
        const D &d = static_cast<const D &>(*this);
        const int64_t P = this->nparams();
        for (int64_t i = 0; i < len; ++i) {
            const int64_t w = widx + i * wstep;
            const double xi = static_cast<double>(xsave[i]);
            const ElemDeriv e = d.deriv1(xi, gamma, w);
            const double gi = static_cast<double>(g[i * gs]);
            for (int k = 0; k < e.np; ++k) acc[e.pi[k]] += e.pv[k] * gi;
            acc[P] += e.dg * gi;
            if (!conj) {
                gx[i * gxs] = static_cast<T>(e.j * gi);
            } else {
                const double p = static_cast<double>(d.template prox1<double>(xi, gamma, w));
                acc[P + 1] += -gi * p + e.j * gi * xi;
                gx[i * gxs] = static_cast<T>(gi - e.j * gi);
            }
        }
    }

#define FP_ELEM_FORWARD(T)                                                                       \
    void prox_run(const T *x, int64_t xs, T *z, int64_t zs, int64_t len, double gamma,           \
                  int64_t widx, int64_t wstep, T *save, bool conj, double sigma)                 \
        const noexcept override {                                                                \
        run_prox<T>(x, xs, z, zs, len, gamma, widx, wstep, save, conj, sigma);                   \
    }                                                                                            \
    void jvp_run(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,      \
                 double gamma, double dgamma, const double *dparams, int64_t widx,               \
                 int64_t wstep, bool conj, double sigma) const noexcept override {               \
        run_jvp<T>(xsave, dx, dxs, dz, dzs, len, gamma, dgamma, dparams, widx, wstep, conj,      \
                   sigma);                                                                       \
    }                                                                                            \
    void vjp_run(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,        \
                 double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,              \
                 double sigma) const noexcept override {                                         \
        run_vjp<T>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma);            \
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
