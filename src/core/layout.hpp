// Strided array descriptors and the two iteration plans used by the drivers.
//
// A RowPlan walks the rows along the operator axis, in C order over the other
// dimensions. An ElemPlan walks every entry in C order as runs along the
// innermost dimension. Both coalesce adjacent dimensions that every array
// involved stores contiguously, so the common case collapses to one dimension.
// Neither changes the canonical order in which saved structure is laid out.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "fastprox.h"

#include <algorithm>
#include <cstdint>

namespace fp {

constexpr int kMaxArr = 4;

// Internal array: ndim >= 1 (scalars become shape [1]).
struct Arr {
    char *data = nullptr;
    fp_dtype dtype = FP_F64;
    int ndim = 1;
    int64_t shape[FP_MAX_NDIM] = {1};
    int64_t strides[FP_MAX_NDIM] = {1};

    int64_t numel() const noexcept {
        int64_t m = 1;
        for (int d = 0; d < ndim; ++d) m *= shape[d];
        return m;
    }
    size_t itemsize() const noexcept { return dtype == FP_F32 ? 4 : 8; }
};

inline bool same_shape(const Arr &a, const Arr &b) noexcept {
    if (a.ndim != b.ndim) return false;
    for (int d = 0; d < a.ndim; ++d)
        if (a.shape[d] != b.shape[d]) return false;
    return true;
}

struct RowPlan {
    int64_t n = 1;     // row length
    int64_t rows = 1;  // number of rows
    int nouter = 0;
    int narr = 0;
    int64_t oshape[FP_MAX_NDIM] = {};
    int64_t ostride[kMaxArr][FP_MAX_NDIM] = {};
    int64_t astride[kMaxArr] = {};  // stride along the axis

    // Element offsets of row r in each array.
    void offsets(int64_t r, int64_t *off) const noexcept {
        for (int a = 0; a < narr; ++a) off[a] = 0;
        for (int d = nouter - 1; d >= 0; --d) {
            const int64_t i = r % oshape[d];
            r /= oshape[d];
            for (int a = 0; a < narr; ++a) off[a] += i * ostride[a][d];
        }
    }
};

inline RowPlan make_row_plan(int ndim, const int64_t *shape, int axis, int narr,
                             const int64_t *const *strides) noexcept {
    RowPlan p;
    p.narr = narr;
    p.n = shape[axis];
    for (int a = 0; a < narr; ++a) p.astride[a] = strides[a][axis];
    for (int d = 0; d < ndim; ++d) {
        if (d == axis || shape[d] == 1) continue;
        p.rows *= shape[d];
        const int k = p.nouter;
        bool merge = k > 0;
        for (int a = 0; a < narr && merge; ++a)
            merge = strides[a][d] * shape[d] == p.ostride[a][k - 1];
        if (merge) {
            p.oshape[k - 1] *= shape[d];
            for (int a = 0; a < narr; ++a) p.ostride[a][k - 1] = strides[a][d];
        } else {
            p.oshape[k] = shape[d];
            for (int a = 0; a < narr; ++a) p.ostride[a][k] = strides[a][d];
            ++p.nouter;
        }
    }
    return p;
}

struct ElemPlan {
    int ndim = 1;
    int axis = -1;  // coalesced index of the operator axis, -1 if irrelevant
    int narr = 0;
    int64_t numel = 1;
    int64_t shape[FP_MAX_NDIM] = {1};
    int64_t stride[kMaxArr][FP_MAX_NDIM] = {};

    // Calls f(off, len, pos, widx, wstep) for each run in [begin, end): off are
    // element offsets of the run start in each array, pos its C-order index.
    template <class F>
    void for_runs(int64_t begin, int64_t end, F &&f) const {
        int64_t idx[FP_MAX_NDIM];
        int64_t r = begin;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = r % shape[d];
            r /= shape[d];
        }
        const int last = ndim - 1;
        int64_t pos = begin;
        int64_t off[kMaxArr];
        while (pos < end) {
            const int64_t len = std::min(shape[last] - idx[last], end - pos);
            for (int a = 0; a < narr; ++a) {
                off[a] = 0;
                for (int d = 0; d < ndim; ++d) off[a] += idx[d] * stride[a][d];
            }
            const int64_t widx = axis >= 0 ? idx[axis] : 0;
            const int64_t wstep = axis == last ? 1 : 0;
            f(off, len, pos, widx, wstep);
            pos += len;
            idx[last] += len;
            for (int d = last; d > 0 && idx[d] == shape[d]; --d) {
                idx[d] = 0;
                ++idx[d - 1];
            }
        }
    }
};

// keep_axis: the axis carries parameters and must stay its own dimension.
inline ElemPlan make_elem_plan(int ndim, const int64_t *shape, int axis, bool keep_axis, int narr,
                               const int64_t *const *strides) noexcept {
    ElemPlan p;
    p.narr = narr;
    p.ndim = 0;
    p.numel = 1;
    int last_kept = -2;  // original dimension merged into the last plan dim
    for (int d = 0; d < ndim; ++d) {
        p.numel *= shape[d];
        const bool is_axis = keep_axis && d == axis;
        if (shape[d] == 1 && !is_axis) continue;
        const int k = p.ndim;
        bool merge = k > 0 && !is_axis && !(keep_axis && last_kept == axis);
        for (int a = 0; a < narr && merge; ++a) merge = strides[a][d] * shape[d] == p.stride[a][k - 1];
        if (merge) {
            p.shape[k - 1] *= shape[d];
            for (int a = 0; a < narr; ++a) p.stride[a][k - 1] = strides[a][d];
        } else {
            p.shape[k] = shape[d];
            for (int a = 0; a < narr; ++a) p.stride[a][k] = strides[a][d];
            if (is_axis) p.axis = k;
            ++p.ndim;
        }
        last_kept = d;
    }
    if (p.ndim == 0) {
        p.ndim = 1;
        p.shape[0] = 1;
        for (int a = 0; a < narr; ++a) p.stride[a][0] = 1;
    }
    return p;
}

}  // namespace fp
