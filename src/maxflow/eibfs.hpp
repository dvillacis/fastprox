// Excesses incremental breadth-first search (EIBFS) for maximum flow, graph
// TV's engine since 0.4 (MaxFlow in maxflow.hpp, the Boykov-Kolmogorov
// engine, has the same interface).
//
// Implemented from the papers: A. V. Goldberg, S. Hed, H. Kaplan, R. E. Tarjan
// and R. F. Werneck, "Maximum flows by incremental breadth-first search", ESA
// 2011 (IBFS), and A. V. Goldberg, S. Hed, H. Kaplan, P. Kohli, R. E. Tarjan
// and R. F. Werneck, "Faster and more dynamic maximum flow by incremental
// breadth-first search", ESA 2015 (EIBFS).
//
// The terminals are folded into the nodes (ESA 2011, Section 6.1): a node's
// terminal residual tr is an excess (tr > 0) or a deficit (tr < 0). Two
// vertex-disjoint forests grow from them, S from the excesses and T into the
// deficits, with distance labels d (from the S roots in S, to the T roots in
// T) for which every forest arc is admissible, so the forests are breadth-
// first and every augmentation runs along a shortest path between an excess
// and a deficit. A pass grows one forest by one level: the one whose orphans
// took less adoption work so far (ESA 2015, Section 3). A residual arc from S
// to T found while scanning starts an augmentation by the bottleneck of the
// path, roots included (case 4 of ESA 2015, Section 2); the saturated forest
// arcs and drained roots become orphans, repaired at once by round-robin
// adoption: the first admissible arc after the current arc, else an orphan
// relabel to the neighbour of least label within the forest's levels (ending
// at a neighbour of the least possible label, ESA 2011, Section 6.1), else
// the node leaves its forest and its children become orphans. The solve ends
// when a pass adds no level: no residual arc leaves S (or enters T), so the
// flow is maximum.
//
// Each solve starts the forests anew from the excesses and deficits of the
// current pseudoflow (the reset of ESA 2015, Section 2, "Dynamic setting"),
// which is how graph TV's rounds reuse the flow, and may be restricted to a
// set of nodes that no residual arc joins to the others (graph TV's active
// parts), so finished parts cost nothing.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "maxflow/maxflow.hpp"

#include <cstddef>
#include <cstdint>

namespace fp {
namespace maxflow {

template <class T>
class EIBFS {
  public:
    enum : uint8_t { kFree = 0, kSource = 1, kSink = 2 };

    // Bytes of working memory for graph g, at the alignment of max_align_t.
    static size_t bytes(const Graph &g) noexcept {
        const size_t n = static_cast<size_t>(g.n), m2 = 2 * static_cast<size_t>(g.m);
        return rnd(sizeof(T) * n) + rnd(sizeof(T) * m2) + 9 * rnd(sizeof(Index) * n) + 3 * rnd(n);
    }

    // mem: bytes(g) bytes aligned to alignof(std::max_align_t), kept by the
    // caller for the life of this object.
    EIBFS(const Graph &g, void *mem) noexcept : g_(g) {
        const size_t n = static_cast<size_t>(g.n), m2 = 2 * static_cast<size_t>(g.m);
        char *c = static_cast<char *>(mem);
        tr_ = reinterpret_cast<T *>(c);
        c += rnd(sizeof(T) * n);
        r_ = reinterpret_cast<T *>(c);
        c += rnd(sizeof(T) * m2);
        Index **arrs[9] = {&d_, &par_, &cur_, &list_[0], &list_[1], &list_[2], &list_[3], &oq_[0], &oq_[1]};
        for (Index **a : arrs) {
            *a = reinterpret_cast<Index *>(c);
            c += rnd(sizeof(Index) * n);
        }
        tree_ = reinterpret_cast<uint8_t *>(c);
        c += rnd(n);
        mark_ = reinterpret_cast<uint8_t *>(c);
        c += rnd(n);
        side_ = reinterpret_cast<uint8_t *>(c);
        for (Index p = 0; p < g.n; ++p) side_[p] = kFree;
    }

    // The problem: set every edge and terminal, then start().
    void init_edge(Index e, T cap_uv, T cap_vu) noexcept {
        const Index a = g_.arc[e];
        r_[a] = cap_uv;
        r_[g_.sister[a]] = cap_vu;
    }
    void init_terminal(Index p, T cap) noexcept { tr_[p] = cap; }
    void start() noexcept {}

    // Changes between solves (the forests are rebuilt by each solve).
    void add_terminal(Index p, T delta) noexcept { tr_[p] += delta; }
    void set_terminal(Index p, T value) noexcept { tr_[p] = value; }
    void set_edge(Index e, T r_uv, T r_vu) noexcept {
        const Index a = g_.arc[e];
        r_[a] = r_uv;
        r_[g_.sister[a]] = r_vu;
    }

    // After a solve: kSource for the nodes reachable from an excess in the
    // residual graph (the smallest minimum cut's source side), kFree for the
    // others (no kSink: graph TV does not need the sink side).
    uint8_t tree(Index p) const noexcept { return side_[p]; }
    T terminal(Index p) const noexcept { return tr_[p]; }
    T residual(Index a) const noexcept { return r_[a]; }

    // Routes excess to deficits until no residual path joins them; returns
    // the flow added. With nodes, only those count of them (no residual arc
    // may join them to the other nodes, whose state is left alone).
    T solve(const int32_t *nodes = nullptr, int64_t count = 0) noexcept {
        nodes_ = nodes;
        count_ = nodes ? count : g_.n;
        flow_ = 0;
        reset();
        while (pass(work_[kS] <= work_[kT] ? kS : kT)) {
        }
        find_side();
        return flow_;
    }

  private:
    static constexpr Index kRoot = -2;            // par_ of a root
    static constexpr int kS = 0, kT = 1;          // forest: tree_ kSource / kSink
    static constexpr int kFront = 0, kNext = 2;   // list_[kFront + x], list_[kNext + x]
    static constexpr uint8_t kOrphan = 16;        // mark_ bits: 1 << list, and this

    static size_t rnd(size_t b) noexcept {
        const size_t a = alignof(std::max_align_t);
        return (b + a - 1) / a * a;
    }
    static uint8_t tag(int x) noexcept { return x == kS ? kSource : kSink; }
    static uint8_t bit(int l) noexcept { return static_cast<uint8_t>(1u << l); }
    Index node(int64_t j) const noexcept { return nodes_ ? static_cast<Index>(nodes_[j]) : static_cast<Index>(j); }

    // Residual capacity of forest arc a (from a node to its parent) in the
    // direction of the flow: parent -> node in S, node -> parent in T.
    T up_cap(int x, Index a) const noexcept { return x == kS ? r_[g_.sister[a]] : r_[a]; }

    void push(int l, Index p) noexcept {
        if (mark_[p] & bit(l)) return;
        mark_[p] |= bit(l);
        list_[l][len_[l]++] = p;
    }
    // Lists p for scanning when its label is its forest's top level D (front)
    // or D + 1, during that forest's pass (next).
    void listed(int x, Index p) noexcept {
        if (d_[p] == D_[x]) push(kFront + x, p);
        else if (d_[p] == D_[x] + 1) push(kNext + x, p);
    }
    void join(int x, Index p, Index parent_arc, Index d) noexcept {
        tree_[p] = tag(x);
        par_[p] = parent_arc;
        cur_[p] = g_.first[p];
        d_[p] = d;
        listed(x, p);
    }

    void reset() noexcept {
        for (int l = 0; l < 4; ++l) len_[l] = 0;
        for (int x = 0; x < 2; ++x) D_[x] = oh_[x] = oc_[x] = 0, work_[x] = 0;
        for (int64_t j = 0; j < count_; ++j) {
            const Index p = node(j);
            mark_[p] = 0;
            tree_[p] = kFree;
            if (tr_[p] > 0) join(kS, p, kRoot, 0);
            else if (tr_[p] < 0) join(kT, p, kRoot, 0);
        }
    }

    // Grows forest x by one level; false when it cannot grow (maximum flow).
    bool pass(int x) noexcept {
        fwd_ = x == kS;
        const int f = kFront + x, nx = kNext + x;
        const Index D = D_[x];
        for (Index i = 0; i < len_[f]; ++i) {  // the list may grow during the pass
            const Index v = list_[f][i];
            mark_[v] &= static_cast<uint8_t>(~bit(f));
            if (tree_[v] == tag(x) && d_[v] == D) scan(x, v);
        }
        len_[f] = 0;
        if (len_[nx] == 0) return false;
        ++D_[x];
        Index *t = list_[f];
        list_[f] = list_[nx];
        list_[nx] = t;
        len_[f] = len_[nx];
        len_[nx] = 0;
        for (Index i = 0; i < len_[f]; ++i) {
            const Index v = list_[f][i];
            mark_[v] = static_cast<uint8_t>((mark_[v] & ~bit(nx)) | bit(f));
        }
        return true;
    }

    // Growth step on v of forest x at the top level D.
    void scan(int x, Index v) noexcept {
        const Index D = d_[v], end = g_.first[v + 1];
        for (Index a = g_.first[v]; a < end; ++a) {
            const Index w = g_.head[a], b = g_.sister[a];
            if (!((x == kS ? r_[a] : r_[b]) > 0)) continue;  // v -> w in S, w -> v in T
            if (tree_[w] == kFree) {
                join(x, w, b, D + 1);
            } else if (tree_[w] != tag(x)) {
                if (x == kS) augment(v, a, w);
                else augment(w, b, v);
                adopt();
                if (tree_[v] != tag(x) || d_[v] != D) return;  // no longer on the top level
                --a;  // the same arc again: it may still join the forests
            }
        }
    }

    void orphan(int x, Index p) noexcept {
        if (mark_[p] & kOrphan) return;
        mark_[p] |= kOrphan;
        Index i = oh_[x] + oc_[x];
        if (i >= g_.n) i -= g_.n;
        oq_[x][i] = p;
        ++oc_[x];
    }

    // Along root ... v -> w ... root, v in S and w in T, by the bottleneck.
    void augment(Index v, Index a, Index w) noexcept {
        T delta = r_[a];
        Index x = v;
        for (; par_[x] != kRoot; x = g_.head[par_[x]]) {
            const T c = r_[g_.sister[par_[x]]];
            if (c < delta) delta = c;
        }
        if (tr_[x] < delta) delta = tr_[x];
        for (x = w; par_[x] != kRoot; x = g_.head[par_[x]]) {
            const T c = r_[par_[x]];
            if (c < delta) delta = c;
        }
        if (-tr_[x] < delta) delta = -tr_[x];
        flow_ += delta;

        r_[a] -= delta;
        r_[g_.sister[a]] += delta;
        for (x = v; par_[x] != kRoot;) {
            const Index pa = par_[x], b = g_.sister[pa];
            r_[b] -= delta;
            r_[pa] += delta;
            if (!(r_[b] > 0)) orphan(kS, x);
            x = g_.head[pa];
        }
        tr_[x] -= delta;
        if (!(tr_[x] > 0)) orphan(kS, x);
        for (x = w; par_[x] != kRoot;) {
            const Index pa = par_[x];
            r_[pa] -= delta;
            r_[g_.sister[pa]] += delta;
            if (!(r_[pa] > 0)) orphan(kT, x);
            x = g_.head[pa];
        }
        tr_[x] += delta;
        if (!(tr_[x] < 0)) orphan(kT, x);
    }

    // Orphans in first-in first-out order (ESA 2011, Section 6.1).
    void adopt() noexcept {
        for (int x = 0; x < 2; ++x)
            while (oc_[x] > 0) {
                const Index v = oq_[x][oh_[x]];
                if (++oh_[x] == g_.n) oh_[x] = 0;
                --oc_[x];
                mark_[v] &= static_cast<uint8_t>(~kOrphan);
                ++work_[x];
                adopt_one(x, v);
            }
    }

    // Round-robin adoption of orphan v of forest x (ESA 2015, Section 2).
    void adopt_one(int x, Index v) noexcept {
        if (x == kS ? tr_[v] > 0 : tr_[v] < 0) {  // still an excess (deficit): a root again
            par_[v] = kRoot;
            return;
        }
        const Index first = g_.first[v], end = g_.first[v + 1], dv = d_[v];
        for (Index a = cur_[v]; a < end; ++a) {
            const Index u = g_.head[a];
            if (tree_[u] == tag(x) && d_[u] == dv - 1 && up_cap(x, a) > 0) {
                par_[v] = a;
                cur_[v] = a;
                return;
            }
        }
        // Orphan relabel: the first neighbour of least label within the
        // forest's levels (up to D while it grows, below D otherwise). Labels
        // never decrease, so a neighbour of label dv is the least possible.
        const Index lim = x == kS ? (fwd_ ? D_[kS] : D_[kS] - 1) : (fwd_ ? D_[kT] - 1 : D_[kT]);
        Index best = -1, ba = -1;
        for (Index a = first; a < end && !(ba >= 0 && best == dv); ++a) {
            const Index u = g_.head[a];
            if (tree_[u] == tag(x) && up_cap(x, a) > 0 && d_[u] <= lim && (ba < 0 || d_[u] < best)) {
                best = d_[u];
                ba = a;
            }
        }
        for (Index a = first; a < end; ++a) {  // the children become orphans
            const Index w = g_.head[a];
            if (tree_[w] == tag(x) && par_[w] == g_.sister[a]) orphan(x, w);
        }
        if (ba >= 0) {
            par_[v] = ba;
            cur_[v] = ba;
            d_[v] = best + 1;
            listed(x, v);
            return;
        }
        tree_[v] = kFree;
        if (x == kS ? tr_[v] < 0 : tr_[v] > 0) {  // a deficit (excess): a root of the other forest
            const int y = 1 - x;
            const bool growing = (y == kS) == fwd_;
            join(y, v, kRoot, growing ? D_[y] + 1 : D_[y]);
        }
    }

    // The nodes reachable from S in the residual graph.
    void find_side() noexcept {
        Index *q = oq_[0], qh = 0, qt = 0;
        for (int64_t j = 0; j < count_; ++j) {
            const Index p = node(j);
            side_[p] = tree_[p] == kSource ? kSource : kFree;
            if (side_[p]) q[qt++] = p;
        }
        while (qh < qt) {
            const Index p = q[qh++];
            for (Index a = g_.first[p]; a < g_.first[p + 1]; ++a) {
                const Index w = g_.head[a];
                if (!side_[w] && r_[a] > 0) {
                    side_[w] = kSource;
                    q[qt++] = w;
                }
            }
        }
    }

    const Graph &g_;
    T *tr_, *r_;
    Index *d_, *par_, *cur_;
    Index *list_[4];  // front and next lists of S and T
    Index *oq_[2];    // orphan queues of S and T (rings)
    uint8_t *tree_, *mark_, *side_;
    Index len_[4] = {0, 0, 0, 0};
    Index D_[2] = {0, 0};
    Index oh_[2] = {0, 0}, oc_[2] = {0, 0};
    int64_t work_[2] = {0, 0};  // orphans adopted per forest in this solve
    bool fwd_ = true;
    const int32_t *nodes_ = nullptr;
    int64_t count_ = 0;
    T flow_ = 0;
};

}  // namespace maxflow
}  // namespace fp
