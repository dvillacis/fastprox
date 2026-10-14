// Batching layer: splits an n-d strided array into contiguous work for the
// kernels, runs it under OpenMP, and saves structure for derivatives.
//
// Determinism: every row or chunk is computed by the same code whatever the
// thread count, and the only cross-row sums (parameter gradients in the vjp)
// are formed in fixed blocks that are added in order at the end.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "core/layout.hpp"
#include "core/op.hpp"
#include "core/workspace.hpp"

#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fp {

inline int thread_id() noexcept {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline int max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline size_t round64(size_t b) noexcept { return (b + 63) / 64 * 64; }

// Rows handed to a thread at a time. Scheduling is dynamic so that fast and
// slow cores (and rows of uneven cost) balance; rows are independent, so the
// result does not depend on which thread computes which row.
inline int64_t row_chunk(int64_t n) noexcept { return std::max<int64_t>(1, kChunk / std::max<int64_t>(n, 1)); }

// Per-thread scratch for row operators: four rows of T, then kernel scratch.
inline size_t row_thread_bytes(const fp_op &op, int64_t n, size_t itemsize) noexcept {
    return round64(4 * static_cast<size_t>(n) * itemsize) + round64(op.scratch_bytes(n));
}

template <class T>
struct RowBuffers {
    T *a, *b, *c, *d;
    void *kernel;
    RowBuffers(const fp_workspace &ws, int64_t n) noexcept {
        char *base = static_cast<char *>(ws.thread_scratch(thread_id()));
        a = reinterpret_cast<T *>(base);
        b = a + n;
        c = b + n;
        d = c + n;
        kernel = base + round64(4 * static_cast<size_t>(n) * sizeof(T));
    }
};

template <class T>
inline void gather(const T *src, int64_t s, T *dst, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) dst[i] = src[i * s];
}
template <class T>
inline void scatter(const T *src, T *dst, int64_t s, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) dst[i * s] = src[i];
}

// ---------------------------------------------------------------------------
// prox and prox_conj
// ---------------------------------------------------------------------------

template <class T>
fp_status prox_driver(const fp_op &op, const Arr &x, const Arr &out, int axis, double gamma,
                      bool conj, double sigma, fp_workspace &ws, int nt, bool keep) {
    const T *xd = reinterpret_cast<const T *>(x.data);
    T *od = reinterpret_cast<T *>(out.data);
    const int64_t *st[2] = {x.strides, out.strides};

    if (op.elementwise()) {
        const ElemPlan p = make_elem_plan(x.ndim, x.shape, axis, op.axis_params(), 2, st);
        const int64_t nch = ceil_div(p.numel, kChunk);
        const int64_t xs = p.stride[0][p.ndim - 1], os = p.stride[1][p.ndim - 1];
        T *save = keep ? ws.xs<T>() : nullptr;
        if (p.numel < kParallelMin) nt = 1;
        (void)nt;  // unused without OpenMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
        for (int64_t c = 0; c < nch; ++c) {
            p.for_runs(c * kChunk, std::min(p.numel, (c + 1) * kChunk),
                       [&](const int64_t *off, int64_t len, int64_t pos, int64_t widx,
                           int64_t wstep) {
                           op.prox_run(xd + off[0], xs, od + off[1], os, len, gamma, widx, wstep,
                                       save ? save + pos : nullptr, conj, sigma);
                       });
        }
        return FP_OK;
    }

    const RowPlan p = make_row_plan(x.ndim, x.shape, axis, 2, st);
    const int64_t n = p.n;
    const int64_t sx = p.astride[0], so = p.astride[1];
    const bool alias = x.data == out.data;
    const bool distinct = op.needs_distinct();
    if (p.rows == 1 || p.rows * n < kParallelMin) nt = 1;
    (void)nt;  // unused without OpenMP
    int err = 0;
#pragma omp parallel for schedule(dynamic, row_chunk(n)) num_threads(nt) if (nt > 1) reduction(| : err)
    for (int64_t r = 0; r < p.rows; ++r) {
        RowBuffers<T> buf(ws, n);
        int64_t off[2];
        p.offsets(r, off);
        const T *xr = xd + off[0];
        T *orow = od + off[1];

        const T *in = xr;
        T *staged = nullptr;
        if (keep) staged = ws.xs<T>() + r * n;
        else if (conj || sx != 1 || (alias && distinct)) staged = buf.a;
        if (staged) {
            if (conj)
                for (int64_t i = 0; i < n; ++i)
                    staged[i] = static_cast<T>(static_cast<double>(xr[i * sx]) / sigma);
            else
                gather(xr, sx, staged, n);
            in = staged;
        }

        T *res;
        if (keep) res = ws.zs<T>() + r * n;
        else if (!conj && so == 1 && !(alias && distinct)) res = orow;
        else res = buf.b;

        uint8_t *code = keep ? ws.code.data() + r * n : nullptr;
        err |= op.prox_row(in, res, n, gamma, buf.kernel, code);

        if (conj) {
            for (int64_t i = 0; i < n; ++i)
                orow[i * so] = static_cast<T>(static_cast<double>(xr[i * sx]) -
                                              sigma * static_cast<double>(res[i]));
        } else if (res != orow) {
            scatter(res, orow, so, n);
        }
    }
    return err ? FP_EINVAL : FP_OK;
}

// ---------------------------------------------------------------------------
// jvp
// ---------------------------------------------------------------------------

template <class T>
fp_status jvp_driver(const fp_op &op, const fp_workspace &ws, const Arr *dx, double dgamma,
                     const double *dparams, const Arr &dout) {
    int nt = ws.nthreads;
    const T *dxd = dx ? reinterpret_cast<const T *>(dx->data) : nullptr;
    T *dd = reinterpret_cast<T *>(dout.data);
    const int64_t *st[2] = {dx ? dx->strides : dout.strides, dout.strides};
    const bool conj = ws.conj;
    const double gamma = ws.gamma, sigma = ws.sigma;

    if (op.elementwise()) {
        const ElemPlan p = make_elem_plan(dout.ndim, dout.shape, ws.axis, op.axis_params(), 2, st);
        const int64_t nch = ceil_div(p.numel, kChunk);
        const int64_t xs = p.stride[0][p.ndim - 1], os = p.stride[1][p.ndim - 1];
        const T *save = ws.xs<T>();
        if (p.numel < kParallelMin) nt = 1;
        (void)nt;  // unused without OpenMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
        for (int64_t c = 0; c < nch; ++c) {
            p.for_runs(c * kChunk, std::min(p.numel, (c + 1) * kChunk),
                       [&](const int64_t *off, int64_t len, int64_t pos, int64_t widx,
                           int64_t wstep) {
                           op.jvp_run(save + pos, dxd ? dxd + off[0] : nullptr, xs, dd + off[1],
                                      os, len, gamma, dgamma, dparams, widx, wstep, conj, sigma);
                       });
        }
        return FP_OK;
    }

    const RowPlan p = make_row_plan(dout.ndim, dout.shape, ws.axis, 2, st);
    const int64_t n = p.n;
    const int64_t sx = p.astride[0], so = p.astride[1];
    const bool alias = dx && dx->data == dout.data;
    if (p.rows == 1 || p.rows * n < kParallelMin) nt = 1;
    (void)nt;  // unused without OpenMP
#pragma omp parallel for schedule(dynamic, row_chunk(n)) num_threads(nt) if (nt > 1)
    for (int64_t r = 0; r < p.rows; ++r) {
        RowBuffers<T> buf(ws, n);
        int64_t off[2];
        p.offsets(r, off);
        T *orow = dd + off[1];
        const T *x = ws.xs<T>() + r * n;
        const T *z = ws.zs<T>() + r * n;
        const uint8_t *code = ws.code.data() + r * n;

        const T *dxr;
        if (!dxd) {
            std::memset(buf.a, 0, static_cast<size_t>(n) * sizeof(T));
            dxr = buf.a;
        } else if (sx != 1 || alias) {
            gather(dxd + off[0], sx, buf.a, n);
            dxr = buf.a;
        } else {
            dxr = dxd + off[0];
        }
        T *res = (so == 1 && !alias) ? orow : buf.b;

        if (!conj) {
            op.jvp_row(x, z, code, dxr, res, n, gamma, dgamma, dparams, buf.kernel);
        } else {
            // out = y - sigma p(1/sigma, y/sigma); dgamma reads as dsigma.
            for (int64_t i = 0; i < n; ++i)
                buf.c[i] = static_cast<T>(static_cast<double>(dxr[i]) / sigma -
                                          static_cast<double>(x[i]) * dgamma / sigma);
            op.jvp_row(x, z, code, buf.c, buf.d, n, gamma, -dgamma / (sigma * sigma), dparams,
                       buf.kernel);
            for (int64_t i = 0; i < n; ++i)
                res[i] = static_cast<T>(static_cast<double>(dxr[i]) -
                                        dgamma * static_cast<double>(z[i]) -
                                        sigma * static_cast<double>(buf.d[i]));
        }
        if (res != orow) scatter(res, orow, so, n);
    }
    return FP_OK;
}

// ---------------------------------------------------------------------------
// vjp
// ---------------------------------------------------------------------------

// Writes gx and returns the totals over the batch in tot[0..P+2):
// theta gradients, the gamma slot and the conjugate extra slot.
template <class T>
fp_status vjp_driver(const fp_op &op, const fp_workspace &ws, const Arr &g, const Arr &gx,
                     double *tot) {
    int nt = ws.nthreads;
    const int64_t P = op.nparams();
    const int64_t W = P + 2;
    const int64_t nb = ws.nblocks;
    double *acc = ws.acc.data();
    std::fill(acc, acc + nb * W, 0.0);
    const T *gd = reinterpret_cast<const T *>(g.data);
    T *xd = reinterpret_cast<T *>(gx.data);
    const int64_t *st[2] = {g.strides, gx.strides};
    const bool conj = ws.conj;
    const double gamma = ws.gamma, sigma = ws.sigma;

    if (op.elementwise()) {
        const ElemPlan p = make_elem_plan(g.ndim, g.shape, ws.axis, op.axis_params(), 2, st);
        const int64_t nch = ceil_div(p.numel, kChunk);
        const int64_t gs = p.stride[0][p.ndim - 1], os = p.stride[1][p.ndim - 1];
        const T *save = ws.xs<T>();
        if (p.numel < kParallelMin) nt = 1;
        (void)nt;  // unused without OpenMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
        for (int64_t b = 0; b < nb; ++b) {
            double *a = acc + b * W;
            const int64_t c0 = nch * b / nb, c1 = nch * (b + 1) / nb;
            p.for_runs(std::min(p.numel, c0 * kChunk), std::min(p.numel, c1 * kChunk),
                       [&](const int64_t *off, int64_t len, int64_t pos, int64_t widx,
                           int64_t wstep) {
                           op.vjp_run(save + pos, gd + off[0], gs, xd + off[1], os, len, gamma,
                                      a, widx, wstep, conj, sigma);
                       });
        }
    } else {
        const RowPlan p = make_row_plan(g.ndim, g.shape, ws.axis, 2, st);
        const int64_t n = p.n;
        const int64_t sg = p.astride[0], so = p.astride[1];
        const bool alias = g.data == gx.data;
        if (p.rows == 1 || p.rows * n < kParallelMin) nt = 1;
        (void)nt;  // unused without OpenMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1)
        for (int64_t b = 0; b < nb; ++b) {
            double *a = acc + b * W;
            const int64_t r0 = p.rows * b / nb, r1 = p.rows * (b + 1) / nb;
            RowBuffers<T> buf(ws, n);
            for (int64_t r = r0; r < r1; ++r) {
                int64_t off[2];
                p.offsets(r, off);
                T *orow = xd + off[1];
                const T *x = ws.xs<T>() + r * n;
                const T *z = ws.zs<T>() + r * n;
                const uint8_t *code = ws.code.data() + r * n;
                const T *gr = gd + off[0];
                if (sg != 1 || alias) {
                    gather(gr, sg, buf.a, n);
                    gr = buf.a;
                }
                T *res = (so == 1 && !alias) ? orow : buf.b;
                if (!conj) {
                    op.vjp_row(x, z, code, gr, res, n, gamma, a, buf.kernel);
                } else {
                    op.vjp_row(x, z, code, gr, buf.d, n, gamma, a, buf.kernel);
                    double e = 0.0;
                    for (int64_t i = 0; i < n; ++i) {
                        const double gi = static_cast<double>(gr[i]);
                        const double jg = static_cast<double>(buf.d[i]);
                        e += -gi * static_cast<double>(z[i]) + jg * static_cast<double>(x[i]);
                        res[i] = static_cast<T>(gi - jg);
                    }
                    a[P + 1] += e;
                }
                if (res != orow) scatter(res, orow, so, n);
            }
        }
    }

    for (int64_t k = 0; k < W; ++k) tot[k] = 0.0;
    for (int64_t b = 0; b < nb; ++b)
        for (int64_t k = 0; k < W; ++k) tot[k] += acc[b * W + k];
    return FP_OK;
}

}  // namespace fp
