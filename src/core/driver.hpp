// Batching layer: splits an n-d strided array into contiguous work for the
// kernels, runs it on the thread pool, and saves structure for derivatives.
//
// Determinism: every row or chunk is computed by the same code whatever the
// thread count, and the only cross-row sums (parameter gradients in the vjp)
// are formed in fixed blocks that are added in order at the end.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "core/layout.hpp"
#include "core/op.hpp"
#include "core/parallel.hpp"
#include "core/workspace.hpp"

#include <atomic>
#include <cstring>
#include <vector>

namespace fp {

inline size_t round64(size_t b) noexcept { return (b + 63) / 64 * 64; }
// a * b, or SIZE_MAX when it overflows: a buffer of that size fails to
// allocate (FP_ENOMEM) instead of coming out short.
inline size_t mul_size(size_t a, size_t b) noexcept { return a != 0 && b > SIZE_MAX / a ? SIZE_MAX : a * b; }

// Rows handed to a thread at a time. Scheduling is dynamic so that fast and
// slow cores (and rows of uneven cost) balance; rows are independent, so the
// result does not depend on which thread computes which row.
inline int64_t row_chunk(int64_t n) noexcept { return std::max<int64_t>(1, kChunk / std::max<int64_t>(n, 1)); }

// Pieces of prox and jvp scaled to the thread count (audit C7): with fixed
// pieces of kChunk entries, 2048 rows of 16 made 2 pieces whatever the
// threads. prox and jvp have no cross-entry floating-point sums, so their
// pieces may depend on nt; the vjp's blocks may not (vjp_driver).
constexpr int64_t kPiecesPerThread = 8;      // rows
constexpr int64_t kRowPieceMin = 512;        // entries per row piece, at least
constexpr int64_t kElemPiecesPerThread = 4;
constexpr int64_t kElemPieceMin = 8192;      // entries per elementwise piece, at least

// Rows per piece: about 8 pieces per thread, at least 512 entries and at
// most kChunk entries (the old grain) per piece.
inline int64_t row_grain(int64_t n, int64_t rows, int nt) noexcept {
    const int64_t want = ceil_div(rows, kPiecesPerThread * std::max(nt, 1));
    const int64_t floor_rows = ceil_div(kRowPieceMin, std::max<int64_t>(n, 1));
    return std::max<int64_t>(1, std::min(row_chunk(n), std::max(want, floor_rows)));
}
// Entries per elementwise piece: about 4 per thread, between 8192 and kChunk,
// a multiple of 64.
inline int64_t elem_piece(int64_t numel, int nt) noexcept {
    const int64_t want = ceil_div(numel, kElemPiecesPerThread * std::max(nt, 1));
    return std::min(kChunk, std::max(kElemPieceMin, (want + 63) / 64 * 64));
}

// Rows along a strided axis are gathered and scattered in tiles of up to 16
// rows (audit C8): rows r and r + 1 start one element apart when the last
// outer dimension has stride 1, so a tile reads b neighbouring elements per
// step instead of one element per cache line. kTileBytes bounds a tile.
constexpr size_t kTileBytes = 32 * 1024;

// Per-thread scratch for row operators: four rows of T, kernel scratch, then
// an input and an output tile. SIZE_MAX for rows beyond 2^48 entries, whose
// byte counts (a few hundred bytes per entry at most) could wrap; only
// stride-0 views reach them.
inline size_t row_thread_bytes(const fp_op &op, int64_t n, size_t itemsize) noexcept {
    if (n > (int64_t{1} << 48)) return SIZE_MAX;
    return round64(4 * static_cast<size_t>(n) * itemsize) + round64(op.scratch_bytes(n)) + 2 * round64(kTileBytes);
}

template <class T>
struct RowBuffers {
    T *a, *b, *c, *d;
    void *kernel;
    T *tin, *tout;  // tiles of kTileBytes
    RowBuffers(const fp_workspace &ws, const fp_op &op, int tid, int64_t n) noexcept {
        char *base = static_cast<char *>(ws.thread_scratch(tid));
        a = reinterpret_cast<T *>(base);
        b = a + n;
        c = b + n;
        d = c + n;
        char *k = base + round64(4 * static_cast<size_t>(n) * sizeof(T));
        kernel = k;
        tin = reinterpret_cast<T *>(k + round64(op.scratch_bytes(n)));
        tout = reinterpret_cast<T *>(reinterpret_cast<char *>(tin) + round64(kTileBytes));
    }
};

// Tiles of rows over the last outer dimension of a row plan, for array a:
// used when the axis is strided in a and the last outer dimension has stride
// 1 there, with at least 4 rows per tile. Tile t covers rows r0 .. r0 + b - 1
// of line t / tpl, all of one line, consecutive in canonical order.
struct Tiling {
    bool on = false;
    int64_t B = 1, L = 1, tpl = 1, count = 0;
    void tile(int64_t t, int64_t &r0, int64_t &b) const noexcept {
        const int64_t line = t / tpl, j0 = (t % tpl) * B;
        r0 = line * L + j0;
        b = std::min(B, L - j0);
    }
};
inline Tiling make_tiling(const RowPlan &p, int a, size_t itemsize) noexcept {
    Tiling t;
    if (p.nouter < 1 || p.astride[a] == 1 || p.ostride[a][p.nouter - 1] != 1 || p.n < 1) return t;
    t.B = std::min<int64_t>(16, static_cast<int64_t>(kTileBytes / (static_cast<size_t>(p.n) * itemsize)));
    if (t.B < 4) return t;
    t.L = p.oshape[p.nouter - 1];
    t.tpl = ceil_div(t.L, t.B);
    t.count = p.rows / t.L * t.tpl;
    t.on = true;
    return t;
}
// Whether rows of array a are written tile by tile (the same layout test).
inline bool tile_layout(const RowPlan &p, int a) noexcept {
    return p.nouter >= 1 && p.astride[a] != 1 && p.ostride[a][p.nouter - 1] == 1;
}
// b rows from src (row k at src + k, entries s apart) into dst (row k at
// dst + k n), through f.
template <class T, class F>
inline void gather_tile(const T *src, int64_t s, int64_t b, int64_t n, T *dst, F &&f) noexcept {
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < b; ++k) dst[k * n + i] = f(src[i * s + k]);
}

template <class T>
inline void gather(const T *src, int64_t s, T *dst, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) dst[i] = src[i * s];
}
template <class T>
inline void scatter(const T *src, T *dst, int64_t s, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) dst[i * s] = src[i];
}

// ---------------------------------------------------------------------------
// One step per row (fp_prox_steps, since 1.0)
// ---------------------------------------------------------------------------

// The steps of a batch, one per row in canonical order (C order over the
// dimensions other than the axis, or over the leading dimensions of an image
// operator), read through the caller's strides.
struct Steps {
    const char *data = nullptr;
    fp_dtype dtype = FP_F64;
    int ndim = 0;
    int64_t shape[FP_MAX_NDIM] = {};
    int64_t stride[FP_MAX_NDIM] = {};

    double at(int64_t r) const noexcept {
        int64_t off = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            off += (r % shape[d]) * stride[d];
            r /= shape[d];
        }
        return dtype == FP_F32 ? static_cast<double>(reinterpret_cast<const float *>(data)[off])
                               : reinterpret_cast<const double *>(data)[off];
    }
};

// The row of the entry at C-order position pos of an elementwise operator.
struct RowOfPos {
    int64_t n = 1, inner = 1;  // the axis length, and the extent after it
    int64_t operator()(int64_t pos) const noexcept { return pos / (n * inner) * inner + pos % inner; }
};

inline RowOfPos row_of_pos(int ndim, const int64_t *shape, int axis) noexcept {
    RowOfPos rp;
    rp.n = std::max<int64_t>(shape[axis], 1);
    for (int d = axis + 1; d < ndim; ++d) rp.inner *= shape[d];
    return rp;
}

// Splits an elementwise run into pieces that lie in one row, for one step per
// row: g(i0, len, row) for the entries [i0, i0 + len) of the run. With the
// axis kept as the plan's last dimension and nothing after it (inner 1), a
// run lies in one row; otherwise consecutive entries are in different rows.
// Without steps the run stays whole (row 0).
template <class G>
inline void split_rows(const RowOfPos &rp, bool on, int64_t pos, int64_t len, G &&g) {
    if (!on) {
        g(int64_t(0), len, int64_t(0));
    } else if (rp.inner == 1) {
        g(int64_t(0), len, rp(pos));
    } else {
        for (int64_t i = 0; i < len; ++i) g(i, int64_t(1), rp(pos + i));
    }
}

// ---------------------------------------------------------------------------
// prox and prox_conj
// ---------------------------------------------------------------------------

template <class T>
fp_status prox_driver(const fp_op &op, const Arr &x, const Arr &out, int axis, double gamma,
                      bool conj, double sigma, fp_workspace &ws, int nt, bool keep, int tie = 0,
                      int64_t *ties = nullptr, const Steps *steps = nullptr) {
    const T *xd = reinterpret_cast<const T *>(x.data);
    T *od = reinterpret_cast<T *>(out.data);
    const int64_t *st[2] = {x.strides, out.strides};

    if (op.elementwise()) {
        const ElemPlan p = make_elem_plan(x.ndim, x.shape, axis, op.axis_params() || steps != nullptr, 2, st);
        const RowOfPos rp = row_of_pos(x.ndim, x.shape, axis);
        const int64_t xs = p.stride[0][p.ndim - 1], os = p.stride[1][p.ndim - 1];
        T *save = keep ? ws.xs<T>() : nullptr;
        if (p.numel < kParallelMin) nt = 1;
        const int64_t E = elem_piece(p.numel, nt), nch = ceil_div(p.numel, E);
        std::atomic<int64_t> nties{0};  // integer sums: the same for any order
        std::atomic<bool> ovf{false};
        parallel_for(nt, nch, 1, [&](int, int64_t c0, int64_t c1) {
            int64_t local = 0;
            p.for_runs(c0 * E, std::min(p.numel, c1 * E),
                       [&](const int64_t *off, int64_t len, int64_t pos, int64_t widx,
                           int64_t wstep) {
                           split_rows(rp, steps != nullptr, pos, len, [&](int64_t i0, int64_t m, int64_t row) {
                               const int64_t r = op.prox_run(xd + off[0] + i0 * xs, xs, od + off[1] + i0 * os, os, m,
                                                             steps ? steps->at(row) : gamma, widx + i0 * wstep, wstep,
                                                             save ? save + pos + i0 : nullptr, conj, sigma, tie);
                               if (r < 0) ovf.store(true, std::memory_order_relaxed);  // y / sigma overflowed
                               else local += r;
                           });
                       });
            nties.fetch_add(local, std::memory_order_relaxed);
        });
        if (ovf.load()) return FP_EPARAM;
        if (ties) *ties = nties.load();
        return FP_OK;
    }

    const RowPlan p = make_row_plan(x.ndim, x.shape, axis, 2, st);
    const int64_t n = p.n;
    const int64_t sx = p.astride[0], so = p.astride[1];
    const bool alias = x.data == out.data;
    const bool distinct = op.needs_distinct();
    if (p.rows == 1 || p.rows * n < kParallelMin) nt = 1;
    std::atomic<int> err{0};
    const Tiling tl = make_tiling(p, 0, sizeof(T));
    if (tl.on) {
        // Strided rows in tiles (audit C8): the same kernel on the same
        // contiguous values, row by row, so the same bits. A whole tile is
        // read before any of it is written, so in place is safe.
        const bool tile_out = tile_layout(p, 1);
        parallel_for(nt, tl.count, std::max<int64_t>(1, row_grain(n, p.rows, nt) / tl.B),
                     [&](int tid, int64_t t0, int64_t t1) {
            RowBuffers<T> buf(ws, op, tid, n);
            for (int64_t t = t0; t < t1; ++t) {
                int64_t r0, b, off[2];
                tl.tile(t, r0, b);
                p.offsets(r0, off);
                const T *x0 = xd + off[0];
                T *dst = keep ? ws.xs<T>() + r0 * n : buf.tin;
                if (conj) {
                    bool ovf = false;
                    if (sigma < kStagingCheck<T>)
                        for (int64_t k = 0; k < b && !ovf; ++k) ovf = staging_overflows<T>(x0 + k, sx, n, sigma);
                    if (ovf) {
                        err.fetch_or(kRowStepOverflow, std::memory_order_relaxed);
                        continue;
                    }
                    gather_tile(x0, sx, b, n, dst, [&](T v) { return static_cast<T>(static_cast<double>(v) / sigma); });
                } else {
                    gather_tile(x0, sx, b, n, dst, [](T v) { return v; });
                }
                for (int64_t k = 0; k < b; ++k) {
                    const int64_t r = r0 + k;
                    T *res = keep ? ws.zs<T>() + r * n : buf.tout + k * n;
                    uint8_t *code = keep ? ws.code.data() + r * n : nullptr;
                    if (int e = op.prox_row(dst + k * n, res, n, steps ? steps->at(r) : gamma, buf.kernel, code))
                        err.fetch_or(e, std::memory_order_relaxed);
                }
                auto res_of = [&](int64_t k) -> const T * {
                    return keep ? ws.zs<T>() + (r0 + k) * n : buf.tout + k * n;
                };
                if (tile_out) {
                    T *o0 = od + off[1];
                    for (int64_t i = 0; i < n; ++i)
                        for (int64_t k = 0; k < b; ++k) {
                            const T v = res_of(k)[i];
                            o0[i * so + k] = conj ? static_cast<T>(static_cast<double>(x0[i * sx + k]) -
                                                                   sigma * static_cast<double>(v))
                                                  : v;
                        }
                } else {
                    for (int64_t k = 0; k < b; ++k) {
                        int64_t ok[2];
                        p.offsets(r0 + k, ok);
                        T *orow = od + ok[1];
                        const T *res = res_of(k);
                        if (conj)
                            for (int64_t i = 0; i < n; ++i)
                                orow[i * so] = static_cast<T>(static_cast<double>(x0[i * sx + k]) -
                                                              sigma * static_cast<double>(res[i]));
                        else
                            scatter(res, orow, so, n);
                    }
                }
            }
        });
        const int e = err.load();
        return (e & kRowStepOverflow) ? FP_EPARAM : (e ? FP_EINVAL : FP_OK);
    }
    parallel_for(nt, p.rows, row_grain(n, p.rows, nt), [&](int tid, int64_t r0, int64_t r1) {
        RowBuffers<T> buf(ws, op, tid, n);
        for (int64_t r = r0; r < r1; ++r) {
            int64_t off[2];
            p.offsets(r, off);
            const T *xr = xd + off[0];
            T *orow = od + off[1];

            const T *in = xr;
            T *staged = nullptr;
            if (keep) staged = ws.xs<T>() + r * n;
            else if (conj || sx != 1 || (alias && distinct)) staged = buf.a;
            if (staged) {
                if (conj) {  // checked where y / sigma can leave T's range (kStagingCheck)
                    if (sigma < kStagingCheck<T> && staging_overflows<T>(xr, sx, n, sigma)) {
                        err.fetch_or(kRowStepOverflow, std::memory_order_relaxed);
                        continue;
                    }
                    for (int64_t i = 0; i < n; ++i)
                        staged[i] = static_cast<T>(static_cast<double>(xr[i * sx]) / sigma);
                } else
                    gather(xr, sx, staged, n);
                in = staged;
            }

            T *res;
            if (keep) res = ws.zs<T>() + r * n;
            else if (!conj && so == 1 && !(alias && distinct)) res = orow;
            else res = buf.b;

            uint8_t *code = keep ? ws.code.data() + r * n : nullptr;
            if (int e = op.prox_row(in, res, n, steps ? steps->at(r) : gamma, buf.kernel, code))
                err.fetch_or(e, std::memory_order_relaxed);

            if (conj) {
                for (int64_t i = 0; i < n; ++i)
                    orow[i * so] = static_cast<T>(static_cast<double>(xr[i * sx]) -
                                                  sigma * static_cast<double>(res[i]));
            } else if (res != orow) {
                scatter(res, orow, so, n);
            }
        }
    });
    const int e = err.load();
    return (e & kRowStepOverflow) ? FP_EPARAM : (e ? FP_EINVAL : FP_OK);
}

// The kink tolerance in the units of the saved entries: the input, or y / sigma
// after fp_prox_conj.
inline double branch_tol(const fp_workspace &ws) noexcept {
    return ws.conj ? ws.branch_tol / ws.sigma : ws.branch_tol;
}

// Kink codes of the saved entries (elementwise operators with has_kinks()):
// written into mask (FP_I8, the shape of x, any strides; may be null), and
// the number of flagged entries returned. tol is in the units of the input.
template <class T>
int64_t kinks_driver(const fp_op &op, const fp_workspace &ws, double tol, const Arr *mask) {
    const int64_t cstr[FP_MAX_NDIM] = {};
    const int64_t *st[1] = {mask ? mask->strides : cstr};
    const ElemPlan p = make_elem_plan(ws.ndim, ws.shape, ws.axis, op.axis_params() || ws.steps_on, 1, st);
    const RowOfPos rp = row_of_pos(ws.ndim, ws.shape, ws.axis);
    const T *save = ws.xs<T>();
    const double t = ws.conj ? tol / ws.sigma : tol;
    int8_t *md = mask ? reinterpret_cast<int8_t *>(mask->data) : nullptr;
    const int64_t ms = p.stride[0][p.ndim - 1];
    int64_t count = 0;
    p.for_runs(0, p.numel, [&](const int64_t *off, int64_t len, int64_t pos, int64_t widx,
                               int64_t wstep) {
        split_rows(rp, ws.steps_on, pos, len, [&](int64_t j0, int64_t jlen, int64_t row) {
            int8_t buf[256];
            for (int64_t i0 = j0; i0 < j0 + jlen; i0 += 256) {
                const int64_t m = std::min<int64_t>(256, j0 + jlen - i0);
                count += op.kink_run(save + pos + i0, m, ws.step(row), widx + i0 * wstep, wstep, t,
                                     md ? buf : nullptr);
                if (md)
                    for (int64_t i = 0; i < m; ++i) md[off[0] + (i0 + i) * ms] = buf[i];
            }
        });
    });
    return count;
}

// Kink codes of row operators with has_kinks() (since 0.3): row by row from
// the saved input and solution, into mask (may be null) through its strides.
template <class T>
int64_t row_kinks_driver(const fp_op &op, const fp_workspace &ws, double tol, const Arr *mask) {
    const int64_t cstr[FP_MAX_NDIM] = {};
    const int64_t *st[1] = {mask ? mask->strides : cstr};
    const RowPlan p = make_row_plan(ws.ndim, ws.shape, ws.axis, 1, st);
    const int64_t n = p.n;
    const double t = ws.conj ? tol / ws.sigma : tol;
    int8_t *md = mask ? reinterpret_cast<int8_t *>(mask->data) : nullptr;
    std::vector<double> xb(static_cast<size_t>(n)), zb(static_cast<size_t>(n));
    std::vector<std::max_align_t> scr(op.scratch_bytes(n) / sizeof(std::max_align_t) + 1);
    int64_t count = 0;
    for (int64_t r = 0; r < p.rows; ++r) {
        const T *xs = ws.xs<T>() + r * n, *zs = ws.zs<T>() + r * n;
        for (int64_t i = 0; i < n; ++i) {
            xb[static_cast<size_t>(i)] = static_cast<double>(xs[i]);
            zb[static_cast<size_t>(i)] = static_cast<double>(zs[i]);
        }
        int64_t off[1];
        p.offsets(r, off);
        count += op.kink_row(xb.data(), zb.data(), n, ws.step(r), t, md ? md + off[0] : nullptr, p.astride[0],
                             scr.data());
    }
    return count;
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
    const double sigma = ws.sigma;

    if (op.elementwise()) {
        const ElemPlan p = make_elem_plan(dout.ndim, dout.shape, ws.axis, op.axis_params() || ws.steps_on, 2, st);
        const RowOfPos rp = row_of_pos(dout.ndim, dout.shape, ws.axis);
        const int64_t xs = p.stride[0][p.ndim - 1], os = p.stride[1][p.ndim - 1];
        const T *save = ws.xs<T>();
        const int8_t *side = ws.branch_on ? ws.branch.data() : nullptr;
        const double ktol = branch_tol(ws);
        if (p.numel < kParallelMin) nt = 1;
        const int64_t E = elem_piece(p.numel, nt), nch = ceil_div(p.numel, E);
        parallel_for(nt, nch, 1, [&](int, int64_t c0, int64_t c1) {
            p.for_runs(c0 * E, std::min(p.numel, c1 * E),
                       [&](const int64_t *off, int64_t len, int64_t pos, int64_t widx,
                           int64_t wstep) {
                           split_rows(rp, ws.steps_on, pos, len, [&](int64_t i0, int64_t m, int64_t row) {
                               op.jvp_run(save + pos + i0, dxd ? dxd + off[0] + i0 * xs : nullptr, xs,
                                          dd + off[1] + i0 * os, os, m, ws.step(row), dgamma, dparams,
                                          widx + i0 * wstep, wstep, conj, sigma, ws.tie,
                                          side ? side + pos + i0 : nullptr, ktol);
                           });
                       });
        });
        return FP_OK;
    }

    const RowPlan p = make_row_plan(dout.ndim, dout.shape, ws.axis, 2, st);
    const int64_t n = p.n;
    const int64_t sx = p.astride[0], so = p.astride[1];
    const bool alias = dx && dx->data == dout.data;
    if (p.rows == 1 || p.rows * n < kParallelMin) nt = 1;
    const Tiling tl = dxd ? make_tiling(p, 0, sizeof(T)) : Tiling{};
    if (tl.on) {
        // dx's rows gathered in tiles (audit C8); each row then as below.
        const bool tile_out = tile_layout(p, 1);
        parallel_for(nt, tl.count, std::max<int64_t>(1, row_grain(n, p.rows, nt) / tl.B),
                     [&](int tid, int64_t t0, int64_t t1) {
            RowBuffers<T> buf(ws, op, tid, n);
            for (int64_t t = t0; t < t1; ++t) {
                int64_t r0, b, off[2];
                tl.tile(t, r0, b);
                p.offsets(r0, off);
                gather_tile(dxd + off[0], sx, b, n, buf.tin, [](T v) { return v; });
                for (int64_t k = 0; k < b; ++k) {
                    const int64_t r = r0 + k;
                    const T *x = ws.xs<T>() + r * n;
                    const T *z = ws.zs<T>() + r * n;
                    const uint8_t *code = ws.code.data() + r * n;
                    const T *dxr = buf.tin + k * n;
                    T *res = buf.tout + k * n;
                    if (!conj) {
                        op.jvp_row(x, z, code, dxr, res, n, ws.step(r), dgamma, dparams, buf.kernel);
                    } else {
                        for (int64_t i = 0; i < n; ++i)
                            buf.c[i] = static_cast<T>(static_cast<double>(dxr[i]) / sigma -
                                                      static_cast<double>(x[i]) * dgamma / sigma);
                        op.jvp_row(x, z, code, buf.c, buf.d, n, ws.gamma, -dgamma / (sigma * sigma), dparams,
                                   buf.kernel);
                        for (int64_t i = 0; i < n; ++i)
                            res[i] = static_cast<T>(static_cast<double>(dxr[i]) -
                                                    dgamma * static_cast<double>(z[i]) -
                                                    sigma * static_cast<double>(buf.d[i]));
                    }
                }
                if (tile_out) {
                    T *o0 = dd + off[1];
                    for (int64_t i = 0; i < n; ++i)
                        for (int64_t k = 0; k < b; ++k) o0[i * so + k] = buf.tout[k * n + i];
                } else {
                    for (int64_t k = 0; k < b; ++k) {
                        int64_t ok[2];
                        p.offsets(r0 + k, ok);
                        scatter(buf.tout + k * n, dd + ok[1], so, n);
                    }
                }
            }
        });
        return FP_OK;
    }
    parallel_for(nt, p.rows, row_grain(n, p.rows, nt), [&](int tid, int64_t r0, int64_t r1) {
        RowBuffers<T> buf(ws, op, tid, n);
        for (int64_t r = r0; r < r1; ++r) {
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
                op.jvp_row(x, z, code, dxr, res, n, ws.step(r), dgamma, dparams, buf.kernel);
            } else {
                // out = y - sigma p(1/sigma, y/sigma); dgamma reads as dsigma.
                for (int64_t i = 0; i < n; ++i)
                    buf.c[i] = static_cast<T>(static_cast<double>(dxr[i]) / sigma -
                                              static_cast<double>(x[i]) * dgamma / sigma);
                op.jvp_row(x, z, code, buf.c, buf.d, n, ws.gamma, -dgamma / (sigma * sigma), dparams,
                           buf.kernel);
                for (int64_t i = 0; i < n; ++i)
                    res[i] = static_cast<T>(static_cast<double>(dxr[i]) -
                                            dgamma * static_cast<double>(z[i]) -
                                            sigma * static_cast<double>(buf.d[i]));
            }
            if (res != orow) scatter(res, orow, so, n);
        }
    });
    return FP_OK;
}

// ---------------------------------------------------------------------------
// vjp
// ---------------------------------------------------------------------------

// Writes gx and returns the totals over the batch in tot[0..P+2):
// theta gradients, the gamma slot and the conjugate extra slot. Its blocks
// (reduction_blocks over rows, or over chunks of kChunk entries) fix the
// order of the cross-row sums, so they are part of the result: they must not
// depend on the thread count (no row_grain or elem_piece here).
template <class T>
fp_status vjp_driver(const fp_op &op, const fp_workspace &ws, const Arr &g, const Arr &gx,
                     double *tot, bool want = true) {
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
    const double sigma = ws.sigma;

    if (op.elementwise()) {
        const ElemPlan p = make_elem_plan(g.ndim, g.shape, ws.axis, op.axis_params() || ws.steps_on, 2, st);
        const RowOfPos rp = row_of_pos(g.ndim, g.shape, ws.axis);
        const int64_t nch = ceil_div(p.numel, kChunk);
        const int64_t gs = p.stride[0][p.ndim - 1], os = p.stride[1][p.ndim - 1];
        const T *save = ws.xs<T>();
        const int8_t *side = ws.branch_on ? ws.branch.data() : nullptr;
        const double ktol = branch_tol(ws);
        if (p.numel < kParallelMin) nt = 1;
        parallel_for(nt, nb, 1, [&](int, int64_t b0, int64_t b1) {
            for (int64_t b = b0; b < b1; ++b) {
                double *a = acc + b * W;
                const int64_t c0 = nch * b / nb, c1 = nch * (b + 1) / nb;
                p.for_runs(std::min(p.numel, c0 * kChunk), std::min(p.numel, c1 * kChunk),
                           [&](const int64_t *off, int64_t len, int64_t pos, int64_t widx,
                               int64_t wstep) {
                               split_rows(rp, ws.steps_on, pos, len, [&](int64_t i0, int64_t m, int64_t row) {
                                   op.vjp_run(save + pos + i0, gd + off[0] + i0 * gs, gs, xd + off[1] + i0 * os,
                                              os, m, ws.step(row), want ? a : nullptr, widx + i0 * wstep, wstep, conj, sigma,
                                              ws.tie, side ? side + pos + i0 : nullptr, ktol);
                               });
                           });
            }
        });
    } else {
        const RowPlan p = make_row_plan(g.ndim, g.shape, ws.axis, 2, st);
        const int64_t n = p.n;
        const int64_t sg = p.astride[0], so = p.astride[1];
        const bool alias = g.data == gx.data;
        if (p.rows == 1 || p.rows * n < kParallelMin) nt = 1;
        const Tiling tl = make_tiling(p, 0, sizeof(T));
        if (tl.on) {
            // g's rows gathered in tiles (audit C8), cut at the blocks'
            // boundaries as well as the lines': every block still sums its
            // rows in ascending order, so the gradients are the same bits.
            const bool tile_out = tile_layout(p, 1);
            parallel_for(nt, nb, 1, [&](int tid, int64_t b0, int64_t b1) {
                RowBuffers<T> buf(ws, op, tid, n);
                for (int64_t blk = b0; blk < b1; ++blk) {
                    double *a = acc + blk * W;
                    const int64_t rb1 = p.rows * (blk + 1) / nb;
                    for (int64_t r0 = p.rows * blk / nb; r0 < rb1;) {
                        const int64_t b = std::min({tl.B, tl.L - r0 % tl.L, rb1 - r0});
                        int64_t off[2];
                        p.offsets(r0, off);
                        gather_tile(gd + off[0], sg, b, n, buf.tin, [](T v) { return v; });
                        for (int64_t k = 0; k < b; ++k) {
                            const int64_t r = r0 + k;
                            const T *x = ws.xs<T>() + r * n;
                            const T *z = ws.zs<T>() + r * n;
                            const uint8_t *code = ws.code.data() + r * n;
                            const T *gr = buf.tin + k * n;
                            T *res = buf.tout + k * n;
                            if (!conj) {
                                op.vjp_row(x, z, code, gr, res, n, ws.step(r), a, buf.kernel);
                            } else {
                                op.vjp_row(x, z, code, gr, buf.d, n, ws.gamma, a, buf.kernel);
                                double e = 0.0;
                                for (int64_t i = 0; i < n; ++i) {
                                    const double gi = static_cast<double>(gr[i]);
                                    const double jg = static_cast<double>(buf.d[i]);
                                    e += -gi * static_cast<double>(z[i]) + jg * static_cast<double>(x[i]);
                                    res[i] = static_cast<T>(gi - jg);
                                }
                                a[P + 1] += e;
                            }
                        }
                        if (tile_out) {
                            T *o0 = xd + off[1];
                            for (int64_t i = 0; i < n; ++i)
                                for (int64_t k = 0; k < b; ++k) o0[i * so + k] = buf.tout[k * n + i];
                        } else {
                            for (int64_t k = 0; k < b; ++k) {
                                int64_t ok[2];
                                p.offsets(r0 + k, ok);
                                scatter(buf.tout + k * n, xd + ok[1], so, n);
                            }
                        }
                        r0 += b;
                    }
                }
            });
            for (int64_t k = 0; k < W; ++k) tot[k] = 0.0;
            for (int64_t b = 0; b < nb; ++b)
                for (int64_t k = 0; k < W; ++k) tot[k] += acc[b * W + k];
            return FP_OK;
        }
        parallel_for(nt, nb, 1, [&](int tid, int64_t b0, int64_t b1) {
            RowBuffers<T> buf(ws, op, tid, n);
            for (int64_t b = b0; b < b1; ++b) {
                double *a = acc + b * W;
                const int64_t r0 = p.rows * b / nb, r1 = p.rows * (b + 1) / nb;
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
                        op.vjp_row(x, z, code, gr, res, n, ws.step(r), a, buf.kernel);
                    } else {
                        op.vjp_row(x, z, code, gr, buf.d, n, ws.gamma, a, buf.kernel);
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
        });
    }

    for (int64_t k = 0; k < W; ++k) tot[k] = 0.0;
    for (int64_t b = 0; b < nb; ++b)
        for (int64_t k = 0; k < W; ++k) tot[k] += acc[b * W + k];
    return FP_OK;
}

// ---------------------------------------------------------------------------
// Image operators: the last k axes form an image, batched over the others.
// ---------------------------------------------------------------------------

// Walks the images of an array: image r starts at offset(r) in each array,
// and its entries are visited in C order through the strides of the last k
// axes (a contiguous image is one run).
struct ImagePlan {
    int k = 0;
    int nlead = 0;
    int narr = 0;
    int64_t images = 1, numel = 1;
    int64_t lshape[FP_MAX_NDIM] = {};
    int64_t lstride[kMaxArr][FP_MAX_NDIM] = {};
    int64_t ishape[3] = {1, 1, 1};
    int64_t istride[kMaxArr][3] = {};
    bool contiguous[kMaxArr] = {};

    void offsets(int64_t r, int64_t *off) const noexcept {
        for (int a = 0; a < narr; ++a) off[a] = 0;
        for (int d = nlead - 1; d >= 0; --d) {
            const int64_t i = r % lshape[d];
            r /= lshape[d];
            for (int a = 0; a < narr; ++a) off[a] += i * lstride[a][d];
        }
    }
    // f(position in C order, element offset from the image start in array a).
    template <class F>
    void for_each(int a, F &&f) const {
        int64_t pos = 0;
        for (int64_t i = 0; i < ishape[0]; ++i)
            for (int64_t j = 0; j < ishape[1]; ++j)
                for (int64_t l = 0; l < ishape[2]; ++l)
                    f(pos++, i * istride[a][0] + j * istride[a][1] + l * istride[a][2]);
    }
};

inline ImagePlan make_image_plan(int ndim, const int64_t *shape, int k, int narr,
                                 const int64_t *const *strides) noexcept {
    ImagePlan p;
    p.k = k;
    p.narr = narr;
    p.nlead = ndim - k;
    for (int d = 0; d < p.nlead; ++d) {
        p.lshape[d] = shape[d];
        p.images *= shape[d];
        for (int a = 0; a < narr; ++a) p.lstride[a][d] = strides[a][d];
    }
    // Pad the image to three axes in front (extent 1, stride 0).
    for (int d = 0; d < k; ++d) {
        const int e = 3 - k + d;
        p.ishape[e] = shape[p.nlead + d];
        p.numel *= p.ishape[e];
        for (int a = 0; a < narr; ++a) p.istride[a][e] = strides[a][p.nlead + d];
    }
    for (int a = 0; a < narr; ++a) {
        int64_t s = 1;
        bool c = true;
        for (int e = 2; e >= 0; --e) {
            if (p.ishape[e] != 1 && p.istride[a][e] != s) c = false;
            s *= p.ishape[e];
        }
        p.contiguous[a] = c;
    }
    return p;
}

// Driver-owned staging at the start of each image working set: three images
// of T (input, output, and a third for the conjugate jvp), then the kernel's
// own working set.
inline size_t image_staging_bytes(int64_t numel) noexcept { return round64(3 * static_cast<size_t>(numel) * 8); }

// Kink codes of image operators with has_kinks() (since 0.3): image by image
// from the saved input, solution and dual.
template <class T>
int64_t image_kinks_driver(const fp_op &op, const fp_workspace &ws, double tol, const Arr *mask) {
    const int64_t cstr[FP_MAX_NDIM] = {};
    const int64_t *st[1] = {mask ? mask->strides : cstr};
    // The mask's shape: x's, or with kink_sites() along the last axis.
    int64_t mshape[FP_MAX_NDIM];
    for (int d = 0; d < ws.ndim; ++d) mshape[d] = ws.shape[d];
    if (op.kink_sites()) mshape[ws.ndim - 1] = op.kink_sites();
    const ImagePlan p = make_image_plan(ws.ndim, mshape, op.image_ndim(), 1, st);
    const int64_t N = ws.image_numel, M = p.numel;
    const int k = p.k;
    const double t = ws.conj ? tol / ws.sigma : tol;
    int8_t *md = mask ? reinterpret_cast<int8_t *>(mask->data) : nullptr;
    int64_t ms[3];
    for (int a = 0; a < k; ++a) ms[a] = p.istride[0][3 - k + a];
    // Without a mask, one image's codes in scratch (C order): the kernels
    // count flagged entries from them.
    std::vector<int8_t> scratch;
    if (!md) {
        scratch.resize(static_cast<size_t>(M));
        int64_t s = 1;
        for (int a = k - 1; a >= 0; --a) {
            ms[a] = s;
            s *= p.ishape[3 - k + a];
        }
    }
    std::vector<double> xb(static_cast<size_t>(N)), zb(static_cast<size_t>(N));
    std::vector<std::max_align_t> work(op.image_bytes(ws.shape + p.nlead) / sizeof(std::max_align_t) + 2);
    int64_t count = 0;
    for (int64_t r = 0; r < p.images; ++r) {
        const T *xs = ws.xs<T>() + r * N, *zs = ws.zs<T>() + r * N;
        for (int64_t i = 0; i < N; ++i) {
            xb[static_cast<size_t>(i)] = static_cast<double>(xs[i]);
            zb[static_cast<size_t>(i)] = static_cast<double>(zs[i]);
        }
        int64_t off[1];
        p.offsets(r, off);
        const double *dual = ws.dual.data() + static_cast<size_t>(r) * static_cast<size_t>(k) * static_cast<size_t>(N);
        count += op.kink_image(ws.image_struct(r), xb.data(), zb.data(), dual, ws.shape + p.nlead, ws.step(r), t,
                               md ? md + off[0] : scratch.data(), ms, reinterpret_cast<char *>(work.data()));
    }
    return count;
}

// sqrt(2 gap) / |x| for the image the kernel solved, x(i, o) its entry i
// (offset o in the input): the relative error of z that the gap certifies,
// since |z - z*|^2 <= 2 gap (feedback U1). 0 for an image of zeros, NaN for
// one holding a NaN (its gap is NaN). The norm is summed in double, in eight
// partial sums by position (a fixed order, not the adder's latency chain);
// when its square leaves the normal range it is summed again scaled by the
// largest entry.
template <class X>
double certified_error(double gap, const ImagePlan &p, X x) noexcept {
    if (std::isnan(gap)) return gap;
    const double e = std::sqrt(2.0 * std::max(gap, 0.0));
    double acc[8] = {};
    p.for_each(0, [&](int64_t i, int64_t o) {
        const double v = x(i, o);
        acc[i & 7] += v * v;
    });
    const double s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    if (s >= 1e-280 && s <= 1e280) return e / std::sqrt(s);
    double m = 0.0;
    p.for_each(0, [&](int64_t i, int64_t o) { m = std::max(m, std::fabs(x(i, o))); });
    if (!(m > 0.0)) return std::isnan(m) ? m : 0.0;
    double t = 0.0;
    p.for_each(0, [&](int64_t i, int64_t o) {
        const double v = x(i, o) / m;
        t += v * v;
    });
    return e / m / std::sqrt(t);
}

// An image's results in the workspace: its gap (of the conjugate's output
// after one), iterations, certified error, and the derivative's exactness.
inline void record_image(fp_workspace &ws, int64_t r, const ImageCall &c, bool conj, double sigma,
                         double zerr) noexcept {
    ws.image_gaps.data()[r] = conj ? c.gap * sigma * sigma : c.gap;
    reinterpret_cast<int *>(ws.image_iters.data())[r] = c.iters;
    ws.image_zerr.data()[r] = zerr;
    ws.image_exact.data()[r] = c.exact ? 1.0 : 0.0;
    ws.image_smooth.data()[r] = c.smoothing;
}

// The certified error of an image from the kernel's input in(i), entry i in
// C order, at the scale the image was solved at (c.scale_exp).
template <class In>
double image_zerr(const ImageCall &c, const ImagePlan &p, In in) noexcept {
    const int se = c.scale_exp;
    if (!se) return certified_error(c.gap, p, in);
    return certified_error(c.scaled_gap, p, [&](int64_t i, int64_t o) { return std::ldexp(in(i, o), -se); });
}

template <class T>
fp_status image_prox_driver(const fp_op &op, const Arr &x, const Arr &out, double gamma, bool conj,
                            double sigma, fp_workspace &ws, int nt, bool keep, double tol,
                            int max_iter, fp_info *info, const Steps *steps = nullptr) {
    const T *xd = reinterpret_cast<const T *>(x.data);
    T *od = reinterpret_cast<T *>(out.data);
    const int64_t *st[2] = {x.strides, out.strides};
    const ImagePlan p = make_image_plan(x.ndim, x.shape, op.image_ndim(), 2, st);
    const int64_t N = p.numel, B = p.images;
    const int k = p.k;
    const int64_t *ishape = x.shape + p.nlead;
    const bool warm = ws.dual_valid && (ws.warm || ws.seeded);
    const bool outer = B >= 2 && (B >= nt || !op.splits_image());
    if (outer) nt = std::min<int>(nt, ws.image_sets);
    double *gaps = ws.image_gaps.data();
    int *iters = reinterpret_cast<int *>(ws.image_iters.data());
    std::atomic<int> err{0}, notconv{0};

    auto solve = [&](int64_t r, char *set, char *threads, int inner) {
        int64_t off[2];
        p.offsets(r, off);
        const T *xr = xd + off[0];
        T *orow = od + off[1];
        T *sin = reinterpret_cast<T *>(set);
        T *sout = sin + N;
        const T *in = xr;
        if (keep || conj || !p.contiguous[0] || x.data == out.data) {
            T *dst = keep ? ws.xs<T>() + r * N : sin;
            if (conj) {
                // No image operator gets here with sigma != 1 (conj_homogeneous);
                // a check like the rows' guards future ones.
                bool ovf = false;
                if (sigma < kStagingCheck<T>) {
                    const StagingBound bound(sigma, std::numeric_limits<T>::max());
                    p.for_each(0, [&](int64_t, int64_t o) { ovf = ovf || bound.exceeded(static_cast<double>(xr[o])); });
                }
                if (ovf) {
                    err.fetch_or(kRowStepOverflow, std::memory_order_relaxed);
                    return;
                }
                p.for_each(0, [&](int64_t i, int64_t o) { dst[i] = static_cast<T>(static_cast<double>(xr[o]) / sigma); });
            } else
                p.for_each(0, [&](int64_t i, int64_t o) { dst[i] = xr[o]; });
            in = dst;
        }
        T *res = keep ? ws.zs<T>() + r * N : ((conj || !p.contiguous[1]) ? sout : orow);
        ImageCall c;
        c.shape = ishape;
        c.numel = N;
        c.gamma = steps ? steps->at(r) : gamma;
        c.tol = tol;
        c.max_iter = max_iter;
        c.nt = inner;
        c.work = set + image_staging_bytes(N);
        c.thread_base = threads;
        c.thread_bytes = ws.image_thread_bytes;
        c.dual = ws.dual.data() + static_cast<size_t>(r) * static_cast<size_t>(k) * static_cast<size_t>(N);
        c.warm = warm;
        c.save = keep ? ws.image_struct(r) : nullptr;
        if (int e = op.prox_image(in, res, c)) err.fetch_or(e, std::memory_order_relaxed);
        if (!c.converged) notconv.store(1, std::memory_order_relaxed);
        // The certified error from the kernel's input: out may be x itself,
        // and holds z by now. After a conjugate, c.gap and y / sigma give the
        // error relative to y.
        const double ze = in == xr ? image_zerr(c, p, [&](int64_t, int64_t o) { return static_cast<double>(xr[o]); })
                                   : image_zerr(c, p, [&](int64_t i, int64_t) { return static_cast<double>(in[i]); });
        record_image(ws, r, c, conj, sigma, ze);
        if (conj) {
            // out = x - sigma z, reading x through its own strides.
            int64_t pos = 0;
            for (int64_t a0 = 0; a0 < p.ishape[0]; ++a0)
                for (int64_t a1 = 0; a1 < p.ishape[1]; ++a1)
                    for (int64_t a2 = 0; a2 < p.ishape[2]; ++a2, ++pos) {
                        const int64_t ox = a0 * p.istride[0][0] + a1 * p.istride[0][1] + a2 * p.istride[0][2];
                        const int64_t oo = a0 * p.istride[1][0] + a1 * p.istride[1][1] + a2 * p.istride[1][2];
                        orow[oo] = static_cast<T>(static_cast<double>(xr[ox]) - sigma * static_cast<double>(res[pos]));
                    }
        } else if (res != orow) {
            p.for_each(1, [&](int64_t i, int64_t o) { orow[o] = res[i]; });
        }
    };

    if (outer) {
        parallel_for(nt, B, 1, [&](int tid, int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) solve(r, ws.image_set(tid), ws.image_threads(tid, 1), 1);
        });
    } else {
        for (int64_t r = 0; r < B; ++r) solve(r, ws.image_set(0), ws.image_threads(0, nt), nt);
    }

    ws.seeded = false;
    if (const int e = err.load()) return (e & kRowStepOverflow) ? FP_EPARAM : FP_EINVAL;
    ws.dual_valid = true;
    if (info) {
        double g = 0.0;
        int it = 0;
        for (int64_t r = 0; r < B; ++r) {
            g += gaps[r];
            it = std::max(it, iters[r]);
        }
        info->gap = g;
        info->iters = it;
    }
    return notconv.load() ? FP_ENOTCONV : FP_OK;
}

// fp_workspace_refine (feedback U4): the kept call solved again from its
// saved input and its dual, as a warm start, to a new tolerance. The saved
// z, structure, dual and per-image results are rewritten; out, when given,
// receives the result as the prox call wrote it (y - sigma z after a
// conjugate). The workspace's scratch only, as for a prox call.
template <class T>
fp_status image_refine_driver(const fp_op &op, fp_workspace &ws, int nt, double tol, int max_iter,
                              const Arr *out, fp_info *info) {
    int64_t cst[FP_MAX_NDIM];  // the saved arrays: contiguous, in C order
    for (int64_t d = ws.ndim - 1, m = 1; d >= 0; --d) {
        cst[d] = m;
        m *= ws.shape[d];
    }
    const int64_t *st[2] = {cst, out ? out->strides : cst};
    const ImagePlan p = make_image_plan(ws.ndim, ws.shape, op.image_ndim(), 2, st);
    const int64_t N = p.numel, B = p.images;
    const int k = p.k;
    const int64_t *ishape = ws.shape + p.nlead;
    T *od = out ? reinterpret_cast<T *>(out->data) : nullptr;
    const bool conj = ws.conj;
    const double sigma = ws.sigma;
    const bool outer = B >= 2 && (B >= nt || !op.splits_image());
    if (outer) nt = std::min<int>(nt, ws.image_sets);
    std::atomic<int> err{0}, notconv{0};

    auto solve = [&](int64_t r, char *set, char *threads, int inner) {
        const T *in = ws.xs<T>() + r * N;
        T *res = ws.zs<T>() + r * N;
        ImageCall c;
        c.shape = ishape;
        c.numel = N;
        c.gamma = ws.step(r);
        c.tol = tol;
        c.max_iter = max_iter;
        c.nt = inner;
        c.work = set + image_staging_bytes(N);
        c.thread_base = threads;
        c.thread_bytes = ws.image_thread_bytes;
        c.dual = ws.dual.data() + static_cast<size_t>(r) * static_cast<size_t>(k) * static_cast<size_t>(N);
        c.warm = true;
        c.save = ws.image_struct(r);
        if (int e = op.prox_image(in, res, c)) err.fetch_or(e, std::memory_order_relaxed);
        if (!c.converged) notconv.store(1, std::memory_order_relaxed);
        record_image(ws, r, c, conj, sigma,
                     image_zerr(c, p, [&](int64_t i, int64_t) { return static_cast<double>(in[i]); }));
        if (od) {
            int64_t off[2];
            p.offsets(r, off);
            T *orow = od + off[1];
            if (conj)  // y = sigma in, as the prox call staged it
                p.for_each(1, [&](int64_t i, int64_t o) {
                    orow[o] = static_cast<T>(sigma * static_cast<double>(in[i]) - sigma * static_cast<double>(res[i]));
                });
            else
                p.for_each(1, [&](int64_t i, int64_t o) { orow[o] = res[i]; });
        }
    };

    if (outer) {
        parallel_for(nt, B, 1, [&](int tid, int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) solve(r, ws.image_set(tid), ws.image_threads(tid, 1), 1);
        });
    } else {
        for (int64_t r = 0; r < B; ++r) solve(r, ws.image_set(0), ws.image_threads(0, nt), nt);
    }
    if (err.load()) return FP_EINVAL;
    if (info) {
        double g = 0.0;
        int it = 0;
        const int *iters = reinterpret_cast<const int *>(ws.image_iters.data());
        for (int64_t r = 0; r < B; ++r) {
            g += ws.image_gaps.data()[r];
            it = std::max(it, iters[r]);
        }
        info->gap = g;
        info->iters = it;
    }
    return notconv.load() ? FP_ENOTCONV : FP_OK;
}

// The worst residual and iteration count of the derivative solves of a batch.
inline void finish_deriv(const fp_workspace &ws, int64_t B) noexcept {
    double r = 0.0;
    double it = 0.0;
    for (int64_t i = 0; i < B; ++i) {
        r = std::max(r, ws.deriv_res.data()[i]);
        it = std::max(it, ws.deriv_its.data()[i]);
    }
    ws.deriv_residual = r;
    ws.deriv_iters = static_cast<int>(it);
}

// Whether the derivatives of B images of N entries, with `width` images at a
// time in parallel, run one image at a time on all the threads instead (an
// operator with threads_derivatives(), audit T2): for a single image, or when
// an image's chunks of about kChunk entries give twice the width. The bits do
// not depend on it.
inline bool derivs_inside(const fp_op &op, const fp_workspace &ws, int64_t B, int64_t N, int64_t width) noexcept {
    if (!op.threads_derivatives() || ws.nthreads < 2) return false;
    const int64_t inner = std::min<int64_t>(ws.nthreads, (N + kChunk - 1) / kChunk);
    return B == 1 || inner >= 2 * std::min<int64_t>(width, B);
}

template <class T>
fp_status image_jvp_driver(const fp_op &op, const fp_workspace &ws, const Arr *dx, double dgamma,
                           const double *dparams, const Arr &dout) {
    const T *dxd = dx ? reinterpret_cast<const T *>(dx->data) : nullptr;
    T *dd = reinterpret_cast<T *>(dout.data);
    const int64_t *st[2] = {dx ? dx->strides : dout.strides, dout.strides};
    const ImagePlan p = make_image_plan(dout.ndim, dout.shape, op.image_ndim(), 2, st);
    const int64_t N = p.numel, B = p.images;
    const int64_t *ishape = dout.shape + p.nlead;
    const bool conj = ws.conj;
    const double sigma = ws.sigma;
    const int nt = std::min<int>(ws.nthreads, std::max(1, ws.image_sets));
    // Images in parallel, or one at a time on all the threads.
    const bool inside = derivs_inside(op, ws, B, N, nt);
    parallel_for(inside ? 1 : nt, B, 1, [&](int tid, int64_t r0, int64_t r1) {
        char *set = ws.image_set(tid);
        T *a = reinterpret_cast<T *>(set);
        T *b = a + N;
        char *work = set + image_staging_bytes(N);
        for (int64_t r = r0; r < r1; ++r) {
            int64_t off[2];
            p.offsets(r, off);
            if (dxd) {
                const T *src = dxd + off[0];
                p.for_each(0, [&](int64_t i, int64_t o) { a[i] = src[o]; });
            } else {
                std::memset(a, 0, static_cast<size_t>(N) * sizeof(T));
            }
            const char *save = ws.image_struct(r);
            DerivCtl dc;
            dc.tol = ws.deriv_tol;
            dc.max_iter = ws.deriv_max_iter;
            dc.nt = inside ? ws.nthreads : 1;
            if (!conj) {
                op.jvp_image(save, a, b, ishape, ws.step(r), dgamma, dparams, work, dc);
            } else {
                // out = y - sigma z(1/sigma, y/sigma); dgamma reads as dsigma.
                const T *xs = ws.xs<T>() + r * N;
                const T *zs = ws.zs<T>() + r * N;
                for (int64_t i = 0; i < N; ++i)
                    b[i] = static_cast<T>(static_cast<double>(a[i]) / sigma -
                                          static_cast<double>(xs[i]) * dgamma / sigma);
                T *c = b + N;
                op.jvp_image(save, b, c, ishape, ws.gamma, -dgamma / (sigma * sigma), dparams, work, dc);
                for (int64_t i = 0; i < N; ++i)
                    b[i] = static_cast<T>(static_cast<double>(a[i]) - dgamma * static_cast<double>(zs[i]) -
                                          sigma * static_cast<double>(c[i]));
            }
            T *dst = dd + off[1];
            p.for_each(1, [&](int64_t i, int64_t o) { dst[o] = b[i]; });
            const_cast<double *>(ws.deriv_res.data())[r] = dc.residual;
            const_cast<double *>(ws.deriv_its.data())[r] = dc.iters;
        }
    });
    finish_deriv(ws, B);
    return FP_OK;
}

template <class T>
fp_status image_vjp_driver(const fp_op &op, const fp_workspace &ws, const Arr &g, const Arr &gx,
                           double *tot) {
    const int64_t P = op.nparams();
    const int64_t W = P + 2;
    const int64_t nb = ws.nblocks;
    double *acc = ws.acc.data();
    std::fill(acc, acc + nb * W, 0.0);
    const T *gd = reinterpret_cast<const T *>(g.data);
    T *xd = reinterpret_cast<T *>(gx.data);
    const int64_t *st[2] = {g.strides, gx.strides};
    const ImagePlan p = make_image_plan(g.ndim, g.shape, op.image_ndim(), 2, st);
    const int64_t N = p.numel, B = p.images;
    const int64_t *ishape = g.shape + p.nlead;
    const bool conj = ws.conj;
    const int nt = std::min<int>(ws.nthreads, std::max(1, ws.image_sets));
    // The fixed blocks in parallel, or in block order on this thread with all
    // the threads inside each image; the accumulators are the same.
    const bool inside = derivs_inside(op, ws, B, N, std::min<int64_t>(nt, nb));
    parallel_for(inside ? 1 : nt, nb, 1, [&](int tid, int64_t b0, int64_t b1) {
        char *set = ws.image_set(tid);
        T *a = reinterpret_cast<T *>(set);
        T *b = a + N;
        char *work = set + image_staging_bytes(N);
        for (int64_t blk = b0; blk < b1; ++blk) {
            double *ac = acc + blk * W;
            for (int64_t r = B * blk / nb; r < B * (blk + 1) / nb; ++r) {
                int64_t off[2];
                p.offsets(r, off);
                const T *src = gd + off[0];
                p.for_each(0, [&](int64_t i, int64_t o) { a[i] = src[o]; });
                const char *save = ws.image_struct(r);
                DerivCtl dc;
                dc.tol = ws.deriv_tol;
                dc.max_iter = ws.deriv_max_iter;
                dc.nt = inside ? ws.nthreads : 1;
                op.vjp_image(save, a, b, ishape, ws.step(r), ac, work, dc);
                const_cast<double *>(ws.deriv_res.data())[r] = dc.residual;
                const_cast<double *>(ws.deriv_its.data())[r] = dc.iters;
                if (conj) {
                    const T *xs = ws.xs<T>() + r * N;
                    const T *zs = ws.zs<T>() + r * N;
                    double e = 0.0;
                    for (int64_t i = 0; i < N; ++i) {
                        const double gi = static_cast<double>(a[i]);
                        const double jg = static_cast<double>(b[i]);
                        e += -gi * static_cast<double>(zs[i]) + jg * static_cast<double>(xs[i]);
                        b[i] = static_cast<T>(gi - jg);
                    }
                    ac[P + 1] += e;
                }
                T *dst = xd + off[1];
                p.for_each(1, [&](int64_t i, int64_t o) { dst[o] = b[i]; });
            }
        }
    });
    for (int64_t k = 0; k < W; ++k) tot[k] = 0.0;
    for (int64_t blk = 0; blk < nb; ++blk)
        for (int64_t k = 0; k < W; ++k) tot[k] += acc[blk * W + k];
    finish_deriv(ws, B);
    return FP_OK;
}

}  // namespace fp
