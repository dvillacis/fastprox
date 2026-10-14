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
#include <limits>

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

// Scans a row: returns kRowNonFinite on any infinity and sets has_nan.
template <class T>
inline int scan_row(const T *x, int64_t n, bool &has_nan) noexcept {
    has_nan = false;
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
