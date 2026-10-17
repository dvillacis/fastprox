// Maximum flow and minimum cut by augmenting paths on two search trees,
// written from the description in Y. Boykov and V. Kolmogorov, "An
// experimental comparison of min-cut/max-flow algorithms for energy
// minimization in vision", IEEE TPAMI 26(9), 2004, 1124-1137, Section 3.
// No implementation was consulted (see the clean-room rule in docs/spec.md).
//
// Two trees S and T are grown from the source and the sink along
// nonsaturated arcs: in S every arc from a parent to its child has residual
// capacity, in T every arc from a child to its parent. Three stages repeat:
//  * growth: active nodes, taken first in first out, scan their arcs and
//    acquire free neighbours as active children, until an arc joins S to T;
//  * augmentation: the bottleneck of that s-t path is pushed through it, and
//    the nodes below its saturated arcs become orphans;
//  * adoption: each orphan looks for a parent in its own tree, joined by a
//    nonsaturated arc and with a root at a terminal; one without becomes
//    free, its children become orphans and its tree neighbours active.
// It ends when no node is active. S is then the set of nodes reachable from
// the source in the residual graph and T the set that reaches the sink:
// the source sides of the smallest and the largest minimum cut.
//
// Tuning, as Section 3.3 suggests: orphans also run first in first out, and
// an adoption stage marks the nodes it has confirmed to reach a terminal
// with the stage number and their distance, so other orphans stop their
// walk there and choose the candidate parent closest to the terminal.
//
// Reuse. The trees outlive a solve. After changing terminal or arc residuals
// through add_terminal and set_edge, the next solve starts from the trees
// left by the last one, repaired where the change broke them, the recycling
// of search trees of P. Kohli and P. H. S. Torr, "Dynamic graph cuts for
// efficient inference in Markov random fields", IEEE TPAMI 29(12), 2007,
// 2079-2088, Section 5. A tree arc left without residual capacity orphans its
// child; a root whose terminal residual drops to zero becomes an orphan; a
// node whose terminal residual changes sign leaves its tree for the other one
// as a root, and its children become orphans; a node next to new residual
// capacity becomes active. Graph TV (Chambolle and Darbon) solves a nested
// sequence of cut problems this way, one per level.
//
// Terminals. Each node carries one signed terminal residual: positive for
// capacity from the source, negative for capacity to the sink. A problem with
// both arcs s -> p and p -> t is first reduced by sending min(c_sp, c_pt)
// straight through p, which shifts the flow value and changes no cut.
//
// Memory. Graph holds the arc structure, built once; MaxFlow works in memory
// the caller provides (bytes() of it), so solves never allocate. Everything
// is sequential and deterministic: the result depends only on the input and
// the order of the edges.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fp::maxflow {

using Index = std::int32_t;

// Arcs grouped by tail: the arcs leaving node p are first[p] .. first[p+1]-1.
// Edge e gives the arc arc[e] from u[e] to v[e] and its reverse
// sister[arc[e]], each with its own residual capacity.
struct Graph {
    Index n = 0, m = 0;
    std::vector<Index> first, head, sister, arc;

    // Edges in the order given; parallel edges are allowed, self-loops are
    // not. Throws std::invalid_argument for a bad endpoint and
    // std::length_error when 2m arcs do not fit in Index.
    template <class I>
    void build(int64_t nodes, int64_t edges, const I *u, const I *v) {
        const int64_t lim = std::numeric_limits<Index>::max();
        if (nodes < 0 || edges < 0) throw std::invalid_argument("maxflow: negative size");
        if (nodes >= lim || edges > (lim - 1) / 2) throw std::length_error("maxflow: graph too large");
        for (int64_t e = 0; e < edges; ++e) {
            const int64_t a = static_cast<int64_t>(u[e]), b = static_cast<int64_t>(v[e]);
            if (a < 0 || a >= nodes || b < 0 || b >= nodes || a == b)
                throw std::invalid_argument("maxflow: bad edge");
        }
        n = static_cast<Index>(nodes);
        m = static_cast<Index>(edges);
        first.assign(static_cast<size_t>(n) + 1, 0);
        head.resize(2 * static_cast<size_t>(m));
        sister.resize(2 * static_cast<size_t>(m));
        arc.resize(static_cast<size_t>(m));
        for (Index e = 0; e < m; ++e) {
            ++first[static_cast<size_t>(u[e]) + 1];
            ++first[static_cast<size_t>(v[e]) + 1];
        }
        for (Index p = 0; p < n; ++p) first[p + 1] += first[p];
        std::vector<Index> pos(first.begin(), first.end() - 1);
        for (Index e = 0; e < m; ++e) {
            const Index a = static_cast<Index>(u[e]), b = static_cast<Index>(v[e]);
            const Index ab = pos[a]++, ba = pos[b]++;
            head[ab] = b;
            head[ba] = a;
            sister[ab] = ba;
            sister[ba] = ab;
            arc[e] = ab;
        }
    }
};

template <class T>
class MaxFlow {
  public:
    enum : uint8_t { kFree = 0, kSource = 1, kSink = 2 };

    // Bytes of working memory for graph g, at the alignment of max_align_t.
    static size_t bytes(const Graph &g) noexcept {
        return round(sizeof(Node) * static_cast<size_t>(g.n)) + round(sizeof(T) * 2 * static_cast<size_t>(g.m)) +
               round(sizeof(Index) * static_cast<size_t>(g.n));
    }

    // mem: bytes(g) bytes aligned to alignof(std::max_align_t), kept by the
    // caller for the life of this object. The state is undefined until the
    // residuals are set and start() is called.
    MaxFlow(const Graph &g, void *mem) noexcept : g_(g) {
        char *c = static_cast<char *>(mem);
        nd_ = reinterpret_cast<Node *>(c);
        c += round(sizeof(Node) * static_cast<size_t>(g.n));
        r_ = reinterpret_cast<T *>(c);
        c += round(sizeof(T) * 2 * static_cast<size_t>(g.m));
        oq_ = reinterpret_cast<Index *>(c);
    }

    // A fresh problem: set every edge and terminal, then start().
    void init_edge(Index e, T cap_uv, T cap_vu) noexcept {
        const Index a = g_.arc[e];
        r_[a] = cap_uv;
        r_[g_.sister[a]] = cap_vu;
    }
    void init_terminal(Index p, T cap) noexcept { nd_[p].tr = cap; }

    // Drops the trees and plants the roots: every node with terminal residual.
    void start() noexcept {
        time_ = 1;
        qhead_ = qtail_ = -1;
        ohead_ = ocount_ = 0;
        for (Index p = 0; p < g_.n; ++p) {
            Node &x = nd_[p];
            x.next = -1;
            x.ts = 0;
            x.dist = 0;
            x.parent = kNone;
            x.tree = kFree;
            x.queued = 0;
            if (x.tr > 0 || x.tr < 0) plant(p, x.tr > 0 ? kSource : kSink);
        }
    }

    // Augments until no s-t path is left; returns the flow added.
    T solve() noexcept {
        T flow = 0;
        adopt();
        while (qhead_ >= 0) {
            const Index p = qhead_;
            const Node &x = nd_[p];
            if (x.tree == kFree) {
                pop();
                continue;
            }
            const Index meet = grow(p);
            if (meet < 0) {
                pop();
                continue;
            }
            flow += augment(meet);  // p stays first: it may have more to explore
            adopt();
        }
        return flow;
    }

    // Changes between solves, keeping the trees.
    void add_terminal(Index p, T delta) noexcept {
        nd_[p].tr += delta;
        retree(p);
    }
    void set_terminal(Index p, T value) noexcept {
        nd_[p].tr = value;
        retree(p);
    }
    // Sets the residuals of edge e, u -> v and v -> u. Zeroing both removes
    // the edge from the problem.
    void set_edge(Index e, T r_uv, T r_vu) noexcept {
        const Index a = g_.arc[e], b = g_.sister[a];
        const Index u = g_.head[b], v = g_.head[a];
        r_[a] = r_uv;
        r_[b] = r_vu;
        check_parent(u, a);
        check_parent(v, b);
        if (r_uv > 0 || r_vu > 0) {
            if (nd_[u].tree != kFree) activate(u);
            if (nd_[v].tree != kFree) activate(v);
        }
    }

    // After a solve: kSource for the nodes reachable from the source in the
    // residual graph, kSink for those that reach the sink, kFree otherwise.
    uint8_t tree(Index p) const noexcept { return nd_[p].tree; }
    T terminal(Index p) const noexcept { return nd_[p].tr; }
    T residual(Index a) const noexcept { return r_[a]; }

  private:
    static constexpr Index kNone = -1;      // free: no parent
    static constexpr Index kTerminal = -2;  // a root, attached to its terminal
    static constexpr Index kOrphan = -3;    // waiting in the orphan queue

    struct Node {
        T tr;          // terminal residual: > 0 from the source, < 0 to the sink
        Index parent;  // arc from this node to its parent, or kNone, kTerminal, kOrphan
        Index next;    // next in the active queue; -1 when not queued, itself when last
        uint32_t ts;   // adoption stage that last confirmed the path to the terminal
        Index dist;    // arcs to the terminal at that stage, the root counting 1
        uint8_t tree;
        uint8_t queued;  // in the orphan queue
    };

    static size_t round(size_t b) noexcept {
        const size_t a = alignof(std::max_align_t);
        return (b + a - 1) / a * a;
    }

    // Residual of the arc that must be nonsaturated for a node of tree t to
    // be the parent of its neighbour across arc a (a leaving the parent).
    T tree_cap(uint8_t t, Index a) const noexcept { return t == kSource ? r_[a] : r_[g_.sister[a]]; }

    void activate(Index p) noexcept {
        Node &x = nd_[p];
        if (x.next >= 0) return;
        x.next = p;
        if (qtail_ >= 0)
            nd_[qtail_].next = p;
        else
            qhead_ = p;
        qtail_ = p;
    }
    void pop() noexcept {
        Node &x = nd_[qhead_];
        const Index nx = x.next;
        x.next = -1;
        if (nx == qhead_)
            qhead_ = qtail_ = -1;
        else
            qhead_ = nx;
    }

    void orphan(Index p) noexcept {
        Node &x = nd_[p];
        x.parent = kOrphan;
        if (x.queued) return;
        x.queued = 1;
        Index i = ohead_ + ocount_;
        if (i >= g_.n) i -= g_.n;
        oq_[i] = p;
        ++ocount_;
    }

    void plant(Index p, uint8_t t) noexcept {
        Node &x = nd_[p];
        x.tree = t;
        x.parent = kTerminal;
        x.ts = time_;
        x.dist = 1;
        activate(p);
    }

    // Growth from active node p: returns an arc from S to T, or -1 once p has
    // no free neighbour left to acquire.
    Index grow(Index p) noexcept {
        const Node &x = nd_[p];
        const uint8_t t = x.tree;
        for (Index a = g_.first[p], e = g_.first[p + 1]; a < e; ++a) {
            if (!(tree_cap(t, a) > 0)) continue;
            const Index q = g_.head[a];
            Node &y = nd_[q];
            if (y.tree == kFree) {
                y.tree = t;
                y.parent = g_.sister[a];
                y.ts = x.ts;
                y.dist = x.dist + 1;
                activate(q);
            } else if (y.tree != t) {
                return t == kSource ? a : g_.sister[a];
            }
        }
        return -1;
    }

    // Pushes the bottleneck through the path made of arc a (S to T) and the
    // tree paths of its ends; orphans the nodes below the saturated arcs.
    T augment(Index a) noexcept {
        const Index u = g_.head[g_.sister[a]], v = g_.head[a];
        T d = r_[a];
        for (Index p = u;;) {
            const Node &x = nd_[p];
            if (x.parent == kTerminal) {
                if (x.tr < d) d = x.tr;
                break;
            }
            const T c = r_[g_.sister[x.parent]];
            if (c < d) d = c;
            p = g_.head[x.parent];
        }
        for (Index p = v;;) {
            const Node &x = nd_[p];
            if (x.parent == kTerminal) {
                if (-x.tr < d) d = -x.tr;
                break;
            }
            const T c = r_[x.parent];
            if (c < d) d = c;
            p = g_.head[x.parent];
        }
        r_[a] -= d;
        r_[g_.sister[a]] += d;
        for (Index p = u;;) {
            Node &x = nd_[p];
            if (x.parent == kTerminal) {
                x.tr -= d;
                if (x.tr == 0) orphan(p);
                break;
            }
            const Index pa = x.parent, down = g_.sister[pa], up = g_.head[pa];
            r_[down] -= d;
            r_[pa] += d;
            if (r_[down] == 0) orphan(p);
            p = up;
        }
        for (Index p = v;;) {
            Node &x = nd_[p];
            if (x.parent == kTerminal) {
                x.tr += d;
                if (x.tr == 0) orphan(p);
                break;
            }
            const Index pa = x.parent, up = g_.head[pa];
            r_[pa] -= d;
            r_[g_.sister[pa]] += d;
            if (r_[pa] == 0) orphan(p);
            p = up;
        }
        return d;
    }

    void adopt() noexcept {
        if (ocount_ == 0) return;
        if (++time_ == 0) {  // wrapped: forget every mark
            for (Index p = 0; p < g_.n; ++p) nd_[p].ts = 0;
            time_ = 1;
        }
        while (ocount_ > 0) {
            const Index p = oq_[ohead_];
            if (++ohead_ == g_.n) ohead_ = 0;
            --ocount_;
            nd_[p].queued = 0;
            process(p);
        }
    }

    // Distance to the terminal of tree node q, or -1 when its root is an
    // orphan; marks the path walked with the current stage.
    Index origin(Index q) noexcept {
        Index d = 0;
        for (Index j = q;;) {
            Node &z = nd_[j];
            if (z.ts == time_) {
                d += z.dist;
                break;
            }
            if (z.parent == kTerminal) {
                z.ts = time_;
                z.dist = 1;
                d += 1;
                break;
            }
            if (z.parent < 0) return -1;  // an orphan
            ++d;
            j = g_.head[z.parent];
        }
        Index k = d;
        for (Index j = q; nd_[j].ts != time_; j = g_.head[nd_[j].parent]) {
            nd_[j].ts = time_;
            nd_[j].dist = k--;
        }
        return d;
    }

    void process(Index p) noexcept {
        Node &x = nd_[p];
        if (x.parent != kOrphan) return;  // rooted again by set_terminal while queued
        const uint8_t t = x.tree;
        if (t == kSource ? x.tr > 0 : x.tr < 0) {
            x.parent = kTerminal;
            x.ts = time_;
            x.dist = 1;
            return;
        }
        Index best = -1, bestd = std::numeric_limits<Index>::max();
        for (Index a = g_.first[p], e = g_.first[p + 1]; a < e; ++a) {
            const Index q = g_.head[a];
            if (nd_[q].tree != t || !(tree_cap(t, g_.sister[a]) > 0)) continue;
            const Index d = origin(q);
            if (d >= 0 && d < bestd) {
                bestd = d;
                best = a;
            }
        }
        if (best >= 0) {
            x.parent = best;
            x.ts = time_;
            x.dist = bestd + 1;
            return;
        }
        for (Index a = g_.first[p], e = g_.first[p + 1]; a < e; ++a) {
            const Index q = g_.head[a];
            Node &y = nd_[q];
            if (y.tree != t) continue;
            if (tree_cap(t, g_.sister[a]) > 0) activate(q);
            if (y.parent >= 0 && g_.head[y.parent] == p) orphan(q);
        }
        x.tree = kFree;
        x.parent = kNone;
    }

    // Repairs the trees at p after its terminal residual changed.
    void retree(Index p) noexcept {
        Node &x = nd_[p];
        const uint8_t want = x.tr > 0 ? kSource : x.tr < 0 ? kSink : kFree;
        if (x.tree == kFree) {
            if (want != kFree) plant(p, want);
        } else if (want == x.tree) {
            if (x.parent >= 0) {  // a shorter way to the terminal; the children stay valid
                x.parent = kTerminal;
                x.dist = 1;
            } else if (x.parent == kOrphan) {
                x.parent = kTerminal;  // left in the queue; process skips it
                x.ts = time_;
                x.dist = 1;
            }
        } else if (want == kFree) {
            if (x.parent == kTerminal) orphan(p);
        } else {  // the other terminal: move p to the other tree as a root
            // Its children become orphans. Its old tree neighbours with
            // residual into p become active: no passive node may face the
            // other tree across a nonsaturated arc, or a path through it
            // would be missed once p is freed.
            const uint8_t old = x.tree;
            for (Index a = g_.first[p], e = g_.first[p + 1]; a < e; ++a) {
                const Index q = g_.head[a];
                Node &y = nd_[q];
                if (y.tree != old) continue;
                if (tree_cap(old, g_.sister[a]) > 0) activate(q);
                if (y.parent >= 0 && g_.head[y.parent] == p) orphan(q);
            }
            if (x.parent == kOrphan) {
                x.tree = want;
                x.parent = kTerminal;  // still queued; process skips it
                x.ts = time_;
                x.dist = 1;
                activate(p);
            } else {
                plant(p, want);
            }
        }
    }

    // Orphans p if its parent arc is arc a (p -> parent) and a lost the
    // residual its tree needs.
    void check_parent(Index p, Index a) noexcept {
        Node &x = nd_[p];
        if (x.parent != a) return;
        if (!(x.tree == kSource ? r_[g_.sister[a]] > 0 : r_[a] > 0)) orphan(p);
    }

    const Graph &g_;
    Node *nd_ = nullptr;
    T *r_ = nullptr;
    Index *oq_ = nullptr;
    Index qhead_ = -1, qtail_ = -1, ohead_ = 0, ocount_ = 0;
    uint32_t time_ = 1;
};

}  // namespace fp::maxflow
