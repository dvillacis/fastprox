// Tree TV: the prox of gamma sum_e w_e |z_u - z_v| when the edges form a
// forest, by dynamic programming in one pass up and one pass down. The signal
// is the last axis (length n), batched over the others.
//
// Messages (V. Kolmogorov, T. Pock and M. Rolinek, "Total variation on a
// tree", SIAM J. Imaging Sci. 9(2), 2016, 605-636, Algorithm 2 and
// Section 4.2). Each tree is rooted at its smallest node (a path at its
// smallest end) and visited from the leaves up. At node i the derivative of the best cost of its subtree as a
// function of z_i is
//     m_i(z) = z - x_i + sum_{children c} clip_[-l_c, l_c](m_c(z)),
// l_c = gamma w_e for the edge from c to its parent. m_i is continuous,
// piecewise linear and increasing with slopes at least 1, so the clip points
// z-_i < z+_i (m_i = -l_i and m_i = l_i) are unique; going down,
// z_i = clip_[z-_i, z+_i](z_parent), and a root takes the zero of m_i.
//
// Representation (as in Section 4.2). m_i is held as its two outer linear
// pieces a z + b, which have slope 1 at a node (the clipped children are
// constant there), and a multiset of breakpoints (position, change of
// slope). The slopes are integers (counts of nodes), so they are exact.
// Clipping walks in from both ends, popping breakpoints until the crossing
// lies on the outer piece, then adds one breakpoint at each clip point; the
// clipped message is merged into the parent's. The paper keeps the
// breakpoints in two mergeable heaps, one for each end; here they are pairing
// heaps (min and max) over one pool of breakpoints, a breakpoint popped from
// one heap is marked and dropped when it reaches the top of the other. Each
// breakpoint is inserted and popped once: O(n log n) for a forest of n nodes.
// When every tree is a path (a chain), each path is rooted at an end and the
// breakpoints of the one child form a sorted deque, as in Section 4.1: the
// clip points are added at its two ends, O(n) in all.
// An edge with gamma w_e = 0 splits the tree: its child is solved as a root.
//
// Exactness. The values are then recomputed as means: on each region R (the
// subtrees of edges with gamma w_e > 0 whose child took its parent's value),
//     z_R = (sum_R x_p - sum_{e in dR} gamma w_e s_{e,R}) / |R|,
// the form of the derivatives. The dual on a tree is determined by z: the
// flow on the edge above c is the sum of x - z over the subtree of c,
// clipped to [-gamma w_e, gamma w_e], and the gap (src/ops/edge_tv.hpp) is
// zero up to rounding.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/edge_tv.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace fp {
namespace {

using edge_tv::round64;

struct TreeTV final : edge_tv::Base<TreeTV> {
    // The passes run over positions in a depth-first preorder, so a subtree
    // is contiguous and a parent comes before its children.
    std::vector<int32_t> order;  // node at each position
    std::vector<int32_t> up;     // position of the parent, -1 at a root
    std::vector<double> wup;     // weight of the edge to the parent
    std::vector<int32_t> below;  // position of the child end of each edge
    bool chains = true;          // every tree is a path, rooted at an end

    TreeTV(int64_t nn, std::vector<int32_t> u, std::vector<int32_t> v, Vec ww)
        : Base(nn, std::move(u), std::move(v), std::move(ww)) {
        const size_t N = static_cast<size_t>(n), M = static_cast<size_t>(m);
        std::vector<int32_t> first(N + 1, 0), adj(2 * M);
        for (size_t e = 0; e < M; ++e) {
            ++first[eu[e] + 1];
            ++first[ev[e] + 1];
        }
        for (size_t i = 0; i < N; ++i) first[i + 1] += first[i];
        std::vector<int32_t> at(first.begin(), first.end() - 1);
        for (size_t e = 0; e < M; ++e) {
            adj[at[eu[e]]++] = static_cast<int32_t>(e);
            adj[at[ev[e]]++] = static_cast<int32_t>(e);
        }
        for (size_t i = 0; i < N; ++i) chains = chains && first[i + 1] - first[i] <= 2;
        order.reserve(N);
        std::vector<int32_t> parent(N, -2), pedge(N, -1), stack;
        stack.reserve(N);
        for (size_t r = 0; r < N; ++r) {
            if (parent[r] != -2 || (chains && first[r + 1] - first[r] == 2)) continue;
            parent[r] = -1;
            stack.push_back(static_cast<int32_t>(r));
            while (!stack.empty()) {  // depth first: each subtree is contiguous in order
                const int32_t i = stack.back();
                stack.pop_back();
                order.push_back(i);
                for (int32_t k = first[i + 1] - 1; k >= first[i]; --k) {
                    const int32_t e = adj[k];
                    const int32_t j = eu[e] == i ? ev[e] : eu[e];
                    if (e == pedge[i]) continue;
                    parent[j] = i;
                    pedge[j] = e;
                    stack.push_back(j);
                }
            }
        }
        std::vector<int32_t> at_pos(N);
        for (size_t j = 0; j < N; ++j) at_pos[order[j]] = static_cast<int32_t>(j);
        up.resize(N);
        wup.resize(N);
        below.resize(M);
        for (size_t j = 0; j < N; ++j) {
            const int32_t i = order[j];
            up[j] = parent[i] < 0 ? -1 : at_pos[parent[i]];
            wup[j] = parent[i] < 0 ? 0.0 : we(pedge[i]);
            if (parent[i] >= 0) below[pedge[i]] = static_cast<int32_t>(j);
        }
    }

    const char *name() const noexcept override { return "tree_tv"; }

    // ---- working set --------------------------------------------------------

    struct Work {
        double *d, *zn;  // input and solution by node
        double *x, *zv, *lo, *hi, *s;  // by position: input, solution, clip points, sum of the children's l
        int32_t *hmin, *hmax;          // heap roots per node
        double *pos, *ds;              // breakpoints: position, change of slope
        int32_t *cmin, *smin, *cmax, *smax;  // heap links (child, sibling)
        uint8_t *dead;
    };
    size_t layout(Work *wk, char *base) const noexcept {
        const size_t N = static_cast<size_t>(n), C = 2 * N;
        size_t off = 0;
        auto take = [&](size_t bytes) {
            char *p = base ? base + off : nullptr;
            off += round64(bytes);
            return p;
        };
        char *d = take(7 * N * sizeof(double));
        char *h = take(2 * N * sizeof(int32_t));
        char *bp = take(2 * C * sizeof(double));
        char *hk = take(4 * C * sizeof(int32_t));
        char *dead = take(C);
        if (wk) {
            wk->d = reinterpret_cast<double *>(d);
            wk->zn = wk->d + N;
            wk->x = wk->zn + N;
            wk->zv = wk->x + N;
            wk->lo = wk->zv + N;
            wk->hi = wk->lo + N;
            wk->s = wk->hi + N;
            wk->hmin = reinterpret_cast<int32_t *>(h);
            wk->hmax = wk->hmin + N;
            wk->pos = reinterpret_cast<double *>(bp);
            wk->ds = wk->pos + C;
            wk->cmin = reinterpret_cast<int32_t *>(hk);
            wk->smin = wk->cmin + C;
            wk->cmax = wk->smin + C;
            wk->smax = wk->cmax + C;
            wk->dead = reinterpret_cast<uint8_t *>(dead);
        }
        return off;
    }
    size_t image_bytes(const int64_t *) const noexcept override { return layout(nullptr, nullptr) + 64; }
    static char *aligned(char *p) noexcept { return p + ((64 - reinterpret_cast<uintptr_t>(p) % 64) % 64); }

    // ---- kinks (since 0.4) ------------------------------------------------------

    // The flow on every edge, net from eu to ev: the flow from a child up to
    // its parent is the sum of x - z over the child's subtree (z = x - div f),
    // unique on a forest. acc: n doubles by position.
    void flows(const double *x, const double *z, double *acc, double *f) const noexcept {
        for (int64_t j = 0; j < n; ++j) acc[j] = x[order[j]] - z[order[j]];
        for (int64_t j = n - 1; j >= 0; --j)
            if (up[j] >= 0) acc[up[j]] += acc[j];
        for (int64_t e = 0; e < m; ++e) {
            const int32_t j = below[e];
            f[e] = eu[e] == order[j] ? acc[j] : -acc[j];
        }
    }
    // Kinks are edges (kink_sites): code 1 at an edge with no jump whose flow
    // is within tol of its capacity gamma w_e > 0; overrides join it (-1) or
    // keep it apart (+1), rebuilding the saved regions from x and z.
    int64_t kink_sites() const noexcept override { return m; }
    bool has_kinks() const noexcept override { return true; }
    bool has_branches() const noexcept override { return true; }
    int64_t kink_image(const double *x, const double *z, const double *, const int64_t *, double gamma, double tol,
                       int8_t *mask, const int64_t *ms, char *work) const noexcept override {
        Work wk;
        layout(&wk, aligned(work));
        const double *f = wk.s;
        flows(x, z, wk.zv, wk.s);  // by position, then one flow per edge (m < n)
        int64_t count = 0;
        for (int64_t e = 0; e < m; ++e) {
            const double cap = gamma * we(e);
            const int8_t c = cap > 0.0 && std::fabs(z[eu[e]] - z[ev[e]]) <= tol && std::fabs(f[e]) >= cap - tol;
            mask[e * ms[0]] = c;
            count += c;
        }
        return count;
    }
    void branch_image(char *save, const double *x, const double *z, const double *, const int64_t *shape,
                      double gamma, double tol, const int8_t *side, char *work) const noexcept override {
        Work wk;
        layout(&wk, aligned(work));
        ImageCall c;
        c.shape = shape;
        c.numel = n;
        c.gamma = gamma;
        c.save = save;
        flows(x, z, wk.zv, wk.s);  // positions buffer, then one flow per edge (m < n)
        structure(x, z, wk.lo, wk.hmax, c, side, tol, wk.s);
    }

    // ---- pairing heaps --------------------------------------------------------

    // One of the two heaps over the breakpoint pool: first child and next
    // sibling of each breakpoint. Insertion and merging link two roots, O(1);
    // a pop pairs the root's children left to right and merges the pairs
    // right to left, O(log n) amortised (Fredman, Sedgewick, Sleator and
    // Tarjan, Algorithmica 1, 1986).
    struct Heap {
        const double *pos;
        int32_t *child, *sib;
        bool max;

        int32_t link(int32_t a, int32_t b) const noexcept {
            if (a < 0) return b;
            if (b < 0) return a;
            if (max ? pos[b] > pos[a] : pos[b] < pos[a]) std::swap(a, b);
            sib[b] = child[a];
            child[a] = b;
            return a;
        }
        int32_t pairing(int32_t first) const noexcept {
            int32_t acc = -1;  // the pairs, last first, through sib
            while (first >= 0) {
                const int32_t a = first, b = sib[a];
                if (b < 0) {
                    sib[a] = acc;
                    acc = a;
                    break;
                }
                first = sib[b];
                sib[a] = sib[b] = -1;
                const int32_t t = link(a, b);
                sib[t] = acc;
                acc = t;
            }
            int32_t r = -1;
            while (acc >= 0) {
                const int32_t next = sib[acc];
                sib[acc] = -1;
                r = link(r, acc);
                acc = next;
            }
            return r;
        }
        // Top of the heap at root, after dropping the breakpoints popped from the other end.
        int32_t top(int32_t &root, const uint8_t *dead) const noexcept {
            while (root >= 0 && dead[root]) root = pairing(child[root]);
            return root;
        }
        void pop(int32_t &root, uint8_t *dead) const noexcept {
            dead[root] = 1;
            root = pairing(child[root]);
        }
    };

    // ---- prox ---------------------------------------------------------------

    template <class T>
    int prox_img(const T *x, T *z, ImageCall &c) const noexcept {
        Work wk;
        layout(&wk, aligned(c.work));
        c.gap = 0.0;
        c.iters = 1;
        c.converged = true;
        bool done;
        const int st = read_input(x, z, wk.d, c, done);
        if (st != kRowOk || done) return st;
        for (int64_t j = 0; j < n; ++j) wk.x[j] = wk.d[order[static_cast<size_t>(j)]];
        solve(wk, c.gamma);
        polish(wk, c.gamma);
        for (int64_t j = 0; j < n; ++j) wk.zn[order[static_cast<size_t>(j)]] = wk.zv[j];
        certificate(wk.d, wk.zn, wk.s, c);
        for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(wk.zn[i]);
        if (c.save) structure(wk.d, wk.zn, wk.lo, wk.hmax, c);
        return kRowOk;
    }

    // Breakpoint stores for the pass up: one heap pair per node, or one deque
    // (centred in the pool) for the path being climbed.
    struct Heaps {
        Heap lo, hi;
        int32_t *rlo, *rhi;
        uint8_t *dead;
        int32_t used = 0;

        int32_t top_lo(int32_t i) noexcept { return lo.top(rlo[i], dead); }
        int32_t top_hi(int32_t i) noexcept { return hi.top(rhi[i], dead); }
        void pop_lo(int32_t i) noexcept { lo.pop(rlo[i], dead); }
        void pop_hi(int32_t i) noexcept { hi.pop(rhi[i], dead); }
        int32_t add_lo(int32_t) noexcept { return used++; }
        int32_t add_hi(int32_t) noexcept { return used++; }
        void link(int32_t i, int32_t t) noexcept {
            dead[t] = 0;
            lo.child[t] = lo.sib[t] = hi.child[t] = hi.sib[t] = -1;
            rlo[i] = lo.link(rlo[i], t);
            rhi[i] = hi.link(rhi[i], t);
        }
        void up(int32_t i, int32_t p) noexcept {
            rlo[p] = lo.link(rlo[p], rlo[i]);
            rhi[p] = hi.link(rhi[p], rhi[i]);
        }
        void cut() noexcept {}
    };
    struct Deque {
        int32_t mid, f, b;  // [f, b) in the pool

        int32_t top_lo(int32_t) const noexcept { return f < b ? f : -1; }
        int32_t top_hi(int32_t) const noexcept { return f < b ? b - 1 : -1; }
        void pop_lo(int32_t) noexcept { ++f; }
        void pop_hi(int32_t) noexcept { --b; }
        int32_t add_lo(int32_t) noexcept { return --f; }
        int32_t add_hi(int32_t) noexcept { return b++; }
        void link(int32_t, int32_t) noexcept {}
        void up(int32_t, int32_t) noexcept {}
        void cut() noexcept { f = b = mid; }
    };

    void solve(Work &wk, double gamma) const noexcept {
        if (chains) {
            Deque st{static_cast<int32_t>(n), static_cast<int32_t>(n), static_cast<int32_t>(n)};
            climb(wk, gamma, st);
        } else {
            std::fill(wk.hmin, wk.hmin + n, -1);
            std::fill(wk.hmax, wk.hmax + n, -1);
            Heaps st{{wk.pos, wk.cmin, wk.smin, false}, {wk.pos, wk.cmax, wk.smax, true},
                     wk.hmin, wk.hmax, wk.dead};
            climb(wk, gamma, st);
        }
        // Down.
        const double *lo = wk.lo, *hi = wk.hi;
        double *zv = wk.zv;
        for (int64_t j = 0; j < n; ++j) {
            const int32_t p = up[static_cast<size_t>(j)];
            zv[j] = p < 0 ? lo[j] : std::min(std::max(zv[p], lo[j]), hi[j]);
        }
    }

    // Up: m_i from its children's clipped messages, then its own clip points.
    template <class S>
    void climb(Work &wk, double gamma, S &st) const noexcept {
        const double *x = wk.x;
        double *lo = wk.lo, *hi = wk.hi, *s = wk.s, *pos = wk.pos, *ds = wk.ds;
        std::fill(s, s + n, 0.0);
        for (int32_t i = static_cast<int32_t>(n) - 1; i >= 0; --i) {
            const int32_t p = up[static_cast<size_t>(i)];
            const double l = gamma * wup[static_cast<size_t>(i)];
            double aL = 1.0, bL = -x[i] - s[i], aR = 1.0, bR = -x[i] + s[i];
            for (int32_t t; (t = st.top_lo(i)) >= 0 && aL * pos[t] + bL < -l;) {
                aL += ds[t];
                bL -= ds[t] * pos[t];
                st.pop_lo(i);
            }
            const double zl = (-l - bL) / aL;
            lo[i] = hi[i] = zl;
            if (p < 0 || !(l > 0.0)) {  // a root: the zero of m_i
                st.cut();
                continue;
            }
            for (int32_t t; (t = st.top_hi(i)) >= 0 && aR * pos[t] + bR > l;) {
                aR -= ds[t];
                bR += ds[t] * pos[t];
                st.pop_hi(i);
            }
            const double zr = std::max((l - bR) / aR, zl);
            hi[i] = zr;
            const int32_t t0 = st.add_lo(i), t1 = st.add_hi(i);
            pos[t0] = zl;
            ds[t0] = aL;
            pos[t1] = zr;
            ds[t1] = -aR;
            st.link(i, t0);
            st.link(i, t1);
            st.up(i, p);
            s[p] += l;
        }
    }

    // Values as region means; then the subtree sums of x - z into s, the flow.
    void polish(Work &wk, double gamma) const noexcept {
        const double *x = wk.x;
        double *zv = wk.zv, *sum = wk.lo, *cnt = wk.hi, *s = wk.s;
        int32_t *lab = wk.hmin;
        int32_t nr = 0;
        for (int64_t j = 0; j < n; ++j) {
            const int32_t p = up[static_cast<size_t>(j)];
            const bool same = p >= 0 && gamma * wup[static_cast<size_t>(j)] > 0.0 && zv[j] == zv[p];
            lab[j] = same ? lab[p] : nr++;
        }
        std::fill(sum, sum + nr, 0.0);
        std::fill(cnt, cnt + nr, 0.0);
        for (int64_t j = 0; j < n; ++j) {
            sum[lab[j]] += x[j];
            cnt[lab[j]] += 1.0;
            const int32_t p = up[static_cast<size_t>(j)];
            const double l = gamma * wup[static_cast<size_t>(j)];
            if (p < 0 || lab[j] == lab[p] || !(l > 0.0)) continue;
            const double t = zv[j] > zv[p] ? l : -l;
            sum[lab[j]] -= t;
            sum[lab[p]] += t;
        }
        for (int32_t R = 0; R < nr; ++R) sum[R] /= cnt[R];
        for (int64_t j = 0; j < n; ++j) zv[j] = sum[lab[j]];
        std::fill(s, s + n, 0.0);
        for (int64_t j = n - 1; j >= 0; --j) {
            s[j] += x[j] - zv[j];
            if (up[static_cast<size_t>(j)] >= 0) s[up[static_cast<size_t>(j)]] += s[j];
        }
    }

    // The flow on edge e from the subtree sum of its child, clipped to its capacity.
    void certificate(const double *d, const double *zv, const double *s, ImageCall &c) const noexcept {
        const double gamma = c.gamma;
        Base::certificate(d, zv, [&](int64_t e) {
            const double cap = gamma * we(e);
            const int32_t k = below[static_cast<size_t>(e)];
            const double f = eu[e] == order[static_cast<size_t>(k)] ? s[k] : -s[k];
            return std::min(std::max(f, -cap), cap);
        }, c);
    }
};

}  // namespace

fp_status make_tree_tv(int64_t n, int64_t m, const int64_t *edges, Vec w, fp_op **out) {
    std::vector<int32_t> u, v;
    const fp_status st = edge_tv::parse(n, m, edges, w, u, v);
    if (st != FP_OK) return st;
    std::vector<int32_t> uf(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) uf[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    for (int64_t e = 0; e < m; ++e)
        if (!edge_tv::unite(uf.data(), u[static_cast<size_t>(e)], v[static_cast<size_t>(e)]))
            return FP_EPARAM;  // a cycle: not a forest
    *out = new TreeTV(n, std::move(u), std::move(v), std::move(w));
    return FP_OK;
}

}  // namespace fp
