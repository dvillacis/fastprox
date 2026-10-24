// Shared pieces of the image operators (tv_aniso.cpp, tv_iso.cpp): image
// geometry, exact 1D TV passes over every fiber along one axis, row sums
// in a fixed order (so results do not depend on the thread count), and the
// power-of-two rescaling of images far from unit scale.
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

// Images whose largest entry is far from 1 are solved on 2^-e d with the
// step 2^-e gamma and scaled back, all exact: TV is positively homogeneous,
// so prox_{gamma g}(c d) = c prox_{(gamma/c) g}(d) for c > 0 (H. H. Bauschke
// and P. L. Combettes, Convex Analysis and Monotone Operator Theory, 2nd ed.,
// Springer 2017, Proposition 24.8(iii)), with the dual scaled by c and the
// gap by c^2. The squares in the stopping target and the gap overflow beyond
// |d| ~ 1e154 and underflow below ~1e-154. e is 0, and the image is solved
// as before bit for bit, when max |d| is in [2^-64, 2^65) and when d == 0.
// Scaling up, the scaled step 2^-e step and the scaled gamma 2^-e gamma stay
// in the step range (step: gamma times the operator's step_scale(), its
// largest threshold or capacity): a small image with a large step is scaled
// up only that far, and not at all when the step is already at the top of
// the range. With every weight below 1 the bound is gamma's own: a capacity
// of 5e-315 let 2^-e gamma overflow to inf (nightly fuzz, tree TV).
inline int rescale_exponent(const double *d, int64_t N, double gamma, double step) noexcept {
    double m = 0.0;
    for (int64_t i = 0; i < N; ++i) m = std::max(m, std::fabs(d[i]));
    if (!(m > 0.0)) return 0;
    int e = std::ilogb(m);
    if (e >= -64 && e <= 64) return 0;
    const double top = std::max(gamma, step);
    if (e < 0 && top > 0.0) e = std::min(0, std::max(e, std::ilogb(top) + 1 - std::ilogb(kMaxScaledStep)));
    return e;
}

// v *= 2^e, entry by entry.
inline void scale2(double *v, int64_t n, int e) noexcept {
    for (int64_t i = 0; i < n; ++i) v[i] = std::ldexp(v[i], e);
}

inline double fista_t(double t) noexcept { return 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t)); }

// Per-thread scratch of fiber_pass: inputs, outputs and edge weights of a
// block of fibers, and the 1D kernel's scratch.
inline size_t fiber_thread_bytes(const fp_op &tv, const Geom &g) noexcept {
    return 3 * static_cast<size_t>(kFiberBlock * g.Lmax) * sizeof(double) + 64 + tv.scratch_bytes(g.Lmax);
}

// Blocks of fibers along axis a that fiber_pass hands out: one per fiber for
// the contiguous axis, else kFiberBlock neighbouring fibers.
inline int64_t fiber_blocks(const Geom &g, int a) noexcept {
    const int64_t s = g.S[a];
    const int64_t outer = g.N / g.L[a] / s;
    return outer * (s == 1 ? 1 : (s + kFiberBlock - 1) / kFiberBlock);
}

template <bool Sum, class In, class Out>
double fiber_pass_impl(const fp_op &tv, const Geom &g, int a, double lam, const ImageCall &c, double *part,
                       In &&in, Out &&out, const double *wt) {
    const int64_t L = g.L[a], s = g.S[a];
    const int64_t bpo = s == 1 ? 1 : (s + kFiberBlock - 1) / kFiberBlock;
    const int64_t nblk = fiber_blocks(g, a);
    const bool ident = lam == 0.0 || L == 1;
    const int64_t width = s == 1 ? 1 : kFiberBlock;
    const int64_t grain = std::max<int64_t>(1, kChunk / std::max<int64_t>(1, L * width));
    parallel_for(c.nt, nblk, grain, [&](int tid, int64_t b0, int64_t b1) {
        double *bin = reinterpret_cast<double *>(c.thread_scratch(tid));
        double *bout = bin + kFiberBlock * g.Lmax;
        double *bw = bout + kFiberBlock * g.Lmax;
        void *scr = bw + kFiberBlock * g.Lmax;
        for (int64_t blk = b0; blk < b1; ++blk) {
            const int64_t oo = blk / bpo;
            const int64_t q0 = (blk % bpo) * kFiberBlock;
            const int64_t W = s == 1 ? 1 : std::min<int64_t>(kFiberBlock, s - q0);
            const int64_t base = oo * L * s + q0;
            for (int64_t i = 0; i < L; ++i)
                for (int64_t q = 0; q < W; ++q) bin[q * L + i] = in(base + i * s + q);
            if (wt && !ident)
                for (int64_t i = 0; i < L; ++i)
                    for (int64_t q = 0; q < W; ++q) bw[q * L + i] = i + 1 < L ? wt[base + i * s + q] : 0.0;
            for (int64_t q = 0; q < W; ++q) {
                if (ident) std::memcpy(bout + q * L, bin + q * L, static_cast<size_t>(L) * sizeof(double));
                else if (wt) tv.prox_edges(bin + q * L, bout + q * L, L, lam, bw + q * L, scr);
                else tv.prox_row(bin + q * L, bout + q * L, L, lam, scr, nullptr);
            }
            double acc = 0.0;
            for (int64_t i = 0; i < L; ++i)
                for (int64_t q = 0; q < W; ++q) {
                    if constexpr (Sum) acc += out(base + i * s + q, bout[q * L + i], bin[q * L + i]);
                    else out(base + i * s + q, bout[q * L + i], bin[q * L + i]);
                }
            if constexpr (Sum) part[blk] = acc;
        }
    });
    double s_ = 0.0;
    if constexpr (Sum)
        for (int64_t blk = 0; blk < nblk; ++blk) s_ += part[blk];
    return s_;
}

// For every fiber along axis a: z = prox_{lam TV}(v) by the 1D TV operator tv
// (unit weight), with v(o) = in(o) at each entry o of the fiber, then
// out(o, z, v). lam == 0 is the identity. With wt (N entries), the edge
// (o, o + e_a) has weight wt[o] (wt at the last entry of a fiber is unused).
template <class In, class Out>
void fiber_pass(const fp_op &tv, const Geom &g, int a, double lam, const ImageCall &c, In &&in,
                Out &&out, const double *wt = nullptr) {
    fiber_pass_impl<false>(tv, g, a, lam, c, nullptr, in, out, wt);
}

// fiber_pass whose out returns a double: their sum, per block into part
// (fiber_blocks(g, a) entries) and then in block order, so it does not depend
// on the thread count.
template <class In, class Out>
double fiber_pass_sum(const fp_op &tv, const Geom &g, int a, double lam, const ImageCall &c, double *part,
                      In &&in, Out &&out, const double *wt = nullptr) {
    return fiber_pass_impl<true>(tv, g, a, lam, c, part, in, out, wt);
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
