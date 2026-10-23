// Spectral operators (since 1.0): the k x k core of the derivative of
// X -> U diag(f(s)) V^T (singular values) or Q diag(f(s)) Q^T (eigenvalues),
// with f the prox of an operator on the spectrum s and J its Jacobian.
//
// With A = U^T H V and its symmetric and skew parts S, K, the derivative is
// U B V^T + (I - U U^T) H V diag(r) V^T (for m >= n), where
//   B = diag(J diag(A)) + G+ o S + G- o K,
//   G+_ij = (z_i - z_j) / (s_i - s_j),  G-_ij = (z_i + z_j) / (s_i + s_j),
//   r_i = z_i / s_i;
// for eigenvalues B = diag(J diag(A)) + G+ o A. This is Theorem 3.3 of
// C. Ding, D. Sun, J. Sun and K.-C. Toh, Spectral operators of matrices,
// Math. Program. 168 (2018), eq. (54) (arXiv 1401.2269v1); the eigenvalue case
// is A. S. Lewis and H. S. Sendov, SIAM J. Matrix Anal. Appl. 23 (2001).
//
// Where a quotient has no value, its limit comes from J: G+_ij -> J_ii - J_ij
// at s_i = s_j, and G-_ij, r_i -> J_ii at zero, since J_ij = 0 there (Ding et
// al., Lemma 3.2). Pairs closer than eps^(1/3) max |s| count as tied and take
// the average of their two limits: a quotient there would lose eps |z| / gap
// to rounding, while the average is off by O(gap^2) and exact at a tie. The
// rule is per pair: values whose adjacent gaps are within that band chain
// into clusters, which can be much wider than the band, and a pair of the
// same cluster farther apart takes its quotient (audit T7).
//
// The limits come from one JVP per row, in a direction d that is nonzero on
// every member of each cluster (the chained one) and sums to zero over it. On a
// cluster of equal values, permutation invariance gives J the form
// a I + b 1 1^T there and a constant c 1 1^T to each other cluster, so
// (J d)_i / d_i = a for every member: J_ii - J_ij for all its pairs. For a
// separable f it is J_ii exactly. A value 0 of the SVD kind is in d even when
// alone, since its row of J is J_ii e_i.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "core/driver.hpp"
#include "core/layout.hpp"
#include "core/op.hpp"
#include "core/parallel.hpp"
#include "core/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace fp {

// Scratch doubles per thread for rows of length k.
inline size_t spectral_doubles(int64_t k) noexcept { return 8 * static_cast<size_t>(k) + 8; }

// out = J dx at saved row r (no gamma or parameter term). ws.axis is the last.
template <class T>
void row_jvp(const fp_op &op, const fp_workspace &ws, int tid, int64_t r, const T *dx, T *out) {
    const int64_t n = ws.n;
    if (op.elementwise()) {
        const int8_t *side = ws.branch_on ? ws.branch.data() + r * n : nullptr;
        op.jvp_run(ws.xs<T>() + r * n, dx, 1, out, 1, n, ws.step(r), 0.0, nullptr, 0, 1, false, ws.sigma,
                   ws.tie, side, branch_tol(ws));
    } else {
        RowBuffers<T> buf(ws, tid, n);
        op.jvp_row(ws.xs<T>() + r * n, ws.zs<T>() + r * n, ws.code.data() + r * n, dx, out, n, ws.step(r), 0.0,
                   nullptr, buf.kernel);
    }
}

// The off-diagonal of B and r for every matrix; the diagonal of B is the
// caller's (a batched fp_jvp on the diagonals). s, z, r: shape (..., k);
// a, b: (..., k, k); all through their strides. Returns FP_EINVAL for a
// negative singular value.
template <class T>
fp_status spectral_driver(const fp_op &op, const fp_workspace &ws, bool svd, const Arr &s, const Arr &z,
                          const Arr &a, const Arr &b, const Arr *r, double *scratch) {
    const int nd = ws.ndim;
    const int64_t k = ws.n, rows = ws.rows;
    const T *sd = reinterpret_cast<const T *>(s.data), *zd = reinterpret_cast<const T *>(z.data);
    const T *ad = reinterpret_cast<const T *>(a.data);
    T *bd = reinterpret_cast<T *>(b.data);
    T *rd = r ? reinterpret_cast<T *>(r->data) : nullptr;
    const int64_t ss = s.strides[nd - 1], zs = z.strides[nd - 1], rs = r ? r->strides[nd - 1] : 0;
    const int64_t ai = a.strides[nd - 1], aj = a.strides[nd], bi = b.strides[nd - 1], bj = b.strides[nd];
    const double eps = std::numeric_limits<T>::epsilon();
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    int nt = static_cast<int>(std::min<int64_t>(ws.nthreads, rows));  // the scratch's regions (audit C2)
    if (rows == 1 || rows * k * k < kParallelMin) nt = 1;
    std::atomic<int> err{0};

    parallel_for(nt, rows, std::max<int64_t>(1, kChunk / std::max<int64_t>(k * k, 1)),
                 [&](int tid, int64_t r0, int64_t r1) {
        double *sv = scratch + static_cast<size_t>(tid) * spectral_doubles(k);
        double *zv = sv + k, *lam = zv + k, *d = lam + k;
        int64_t *perm = reinterpret_cast<int64_t *>(d + k), *cid = perm + k;
        T *tdx = reinterpret_cast<T *>(cid + k), *tout = tdx + k;
        for (int64_t row = r0; row < r1; ++row) {
            // Offsets of this matrix: the leading indices of row, in C order.
            int64_t os = 0, oz = 0, oa = 0, ob = 0, orr = 0, rem = row;
            for (int dim = nd - 2; dim >= 0; --dim) {
                const int64_t idx = rem % ws.shape[dim];
                rem /= ws.shape[dim];
                os += idx * s.strides[dim];
                oz += idx * z.strides[dim];
                oa += idx * a.strides[dim];
                ob += idx * b.strides[dim];
                if (r) orr += idx * r->strides[dim];
            }
            bool finite = true;
            double smax = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                sv[i] = static_cast<double>(sd[os + i * ss]);
                zv[i] = static_cast<double>(zd[oz + i * zs]);
                if (!std::isfinite(sv[i]) || !std::isfinite(zv[i])) finite = false;
                else if (svd && sv[i] < 0.0) err.store(1, std::memory_order_relaxed);
                smax = std::max(smax, std::fabs(sv[i]));
            }
            T *brow = bd + ob;
            if (!finite) {
                for (int64_t i = 0; i < k; ++i) {
                    for (int64_t j = 0; j < k; ++j)
                        if (i != j) brow[i * bi + j * bj] = static_cast<T>(qnan);
                    if (rd) rd[orr + i * rs] = static_cast<T>(qnan);
                }
                continue;
            }

            // Clusters of tied values, and the direction d for their limits.
            const double band = std::cbrt(eps) * smax;
            for (int64_t i = 0; i < k; ++i) perm[i] = i;
            std::sort(perm, perm + k, [&](int64_t p, int64_t q) {
                return sv[p] < sv[q] || (sv[p] == sv[q] && p < q);
            });
            bool need = false;
            for (int64_t t0 = 0; t0 < k;) {
                int64_t t1 = t0 + 1;
                while (t1 < k && sv[perm[t1]] - sv[perm[t1 - 1]] <= band) ++t1;
                const int64_t c = t1 - t0;
                for (int64_t t = t0; t < t1; ++t) {
                    cid[perm[t]] = t0;
                    double v = 0.0;
                    if (c >= 2) {
                        // Alternating signs; an odd cluster ends in (1, 1, -2).
                        const int64_t u = t - t0;
                        if (c % 2 == 1 && u >= c - 3) v = u == c - 1 ? -2.0 : 1.0;
                        else v = u % 2 == 0 ? 1.0 : -1.0;
                    } else if (svd && sv[perm[t]] == 0.0) {
                        v = 1.0;
                    }
                    d[perm[t]] = v;
                    need = need || v != 0.0;
                }
                t0 = t1;
            }
            if (need) {
                for (int64_t i = 0; i < k; ++i) tdx[i] = static_cast<T>(d[i]);
                row_jvp<T>(op, ws, tid, row, tdx, tout);
                for (int64_t i = 0; i < k; ++i) lam[i] = d[i] != 0.0 ? static_cast<double>(tout[i]) / d[i] : 0.0;
            }

            const T *arow = ad + oa;
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    if (i == j) continue;
                    // The limit only for a pair within the band: a run of close
                    // values chains into a cluster much wider (audit T7), and
                    // the quotient over a gap above the band loses only eps^(2/3).
                    const bool tied = cid[i] == cid[j] && std::fabs(sv[i] - sv[j]) <= band;
                    const double gp = tied ? 0.5 * (lam[i] + lam[j]) : (zv[i] - zv[j]) / (sv[i] - sv[j]);
                    const double aij = static_cast<double>(arow[i * ai + j * aj]);
                    double v;
                    if (svd) {
                        const double aji = static_cast<double>(arow[j * ai + i * aj]);
                        const double den = sv[i] + sv[j];
                        const double gm = den == 0.0 ? 0.5 * (lam[i] + lam[j]) : (zv[i] + zv[j]) / den;
                        v = 0.5 * (gp * (aij + aji) + gm * (aij - aji));
                    } else {
                        v = gp * aij;
                    }
                    brow[i * bi + j * bj] = static_cast<T>(v);
                }
                if (rd) rd[orr + i * rs] = static_cast<T>(sv[i] != 0.0 ? zv[i] / sv[i] : lam[i]);
            }
        }
    });
    return err.load() ? FP_EINVAL : FP_OK;
}

}  // namespace fp
