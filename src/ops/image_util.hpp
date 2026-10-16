// Shared pieces of the image operators (tv_aniso.cpp, tv_iso.cpp): image
// geometry, exact 1D TV passes over every fiber along one axis, and row sums
// in a fixed order (so results do not depend on the thread count).
//
// SPDX-License-Identifier: MIT
#pragma once

#include "core/op.hpp"
#include "core/parallel.hpp"
#include "core/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace fp {

constexpr int64_t kFiberBlock = 8;  // strided fibers gathered together

// C-ordered image of k <= 3 axes.
struct Geom {
    int k = 0;
    int64_t L[3] = {1, 1, 1};  // extents of the image axes
    int64_t S[3] = {1, 1, 1};  // strides
    int64_t N = 1;
    int64_t Lmax = 1;
    explicit Geom(int kk, const int64_t *shape) : k(kk) {
        for (int a = 0; a < k; ++a) L[a] = shape[a];
        S[k - 1] = 1;
        for (int a = k - 2; a >= 0; --a) S[a] = S[a + 1] * L[a + 1];
        N = S[0] * L[0];
        for (int a = 0; a < k; ++a) Lmax = std::max(Lmax, L[a]);
    }
    int64_t rows() const noexcept { return N / L[k - 1]; }
    // Coordinate of entry o along axis a, and whether o has a forward
    // neighbour along a.
    int64_t coord(int64_t o, int a) const noexcept { return (o / S[a]) % L[a]; }
    bool has_next(int64_t o, int a) const noexcept { return coord(o, a) + 1 < L[a]; }
};

inline double fista_t(double t) noexcept { return 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t)); }

// Per-thread scratch of fiber_pass.
inline size_t fiber_thread_bytes(const fp_op &tv, const Geom &g) noexcept {
    return 2 * static_cast<size_t>(kFiberBlock * g.Lmax) * sizeof(double) + 64 + tv.scratch_bytes(g.Lmax);
}

// For every fiber along axis a: z = prox_{lam TV}(v) by the 1D TV operator tv
// (unit weight), with v(o) = in(o) at each entry o of the fiber, then
// out(o, z, v). lam == 0 is the identity.
template <class In, class Out>
void fiber_pass(const fp_op &tv, const Geom &g, int a, double lam, const ImageCall &c, In &&in,
                Out &&out) {
    const int64_t L = g.L[a], s = g.S[a];
    const int64_t nf = g.N / L;
    const int64_t outer = nf / s;
    const int64_t bpo = s == 1 ? 1 : (s + kFiberBlock - 1) / kFiberBlock;
    const int64_t nblk = outer * bpo;
    const bool ident = lam == 0.0 || L == 1;
    const int64_t width = s == 1 ? 1 : kFiberBlock;
    const int64_t grain = std::max<int64_t>(1, kChunk / std::max<int64_t>(1, L * width));
    parallel_for(c.nt, nblk, grain, [&](int tid, int64_t b0, int64_t b1) {
        double *bin = reinterpret_cast<double *>(c.thread_scratch(tid));
        double *bout = bin + kFiberBlock * g.Lmax;
        void *scr = bout + kFiberBlock * g.Lmax;
        for (int64_t blk = b0; blk < b1; ++blk) {
            const int64_t oo = blk / bpo;
            const int64_t q0 = (blk % bpo) * kFiberBlock;
            const int64_t W = s == 1 ? 1 : std::min<int64_t>(kFiberBlock, s - q0);
            const int64_t base = oo * L * s + q0;
            for (int64_t i = 0; i < L; ++i)
                for (int64_t q = 0; q < W; ++q) bin[q * L + i] = in(base + i * s + q);
            for (int64_t q = 0; q < W; ++q) {
                if (ident) std::memcpy(bout + q * L, bin + q * L, static_cast<size_t>(L) * sizeof(double));
                else tv.prox_row(bin + q * L, bout + q * L, L, lam, scr, nullptr);
            }
            for (int64_t i = 0; i < L; ++i)
                for (int64_t q = 0; q < W; ++q) out(base + i * s + q, bout[q * L + i], bin[q * L + i]);
        }
    });
}

// Sum over the rows (length L[k-1]) of f(r), formed in a fixed order.
template <class F>
double row_sum(const Geom &g, const ImageCall &c, double *part, F &&f) {
    const int64_t R = g.rows(), n = g.L[g.k - 1];
    parallel_for(c.nt, R, std::max<int64_t>(1, kChunk / n), [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) part[r] = f(r);
    });
    double s = 0.0;
    for (int64_t r = 0; r < R; ++r) s += part[r];
    return s;
}

// f(r) for every row, in parallel.
template <class F>
void row_for(const Geom &g, const ImageCall &c, F &&f) {
    const int64_t R = g.rows(), n = g.L[g.k - 1];
    parallel_for(c.nt, R, std::max<int64_t>(1, kChunk / n), [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) f(r);
    });
}

}  // namespace fp
