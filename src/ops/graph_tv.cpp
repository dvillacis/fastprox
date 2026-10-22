// Graph TV: the prox of gamma sum_e w_e |z_u - z_v| over the edges e = (u, v)
// of a given graph, by parametric max-flow on the EIBFS engine of
// src/maxflow/eibfs.hpp (since 0.4; the Boykov-Kolmogorov engine of
// src/maxflow/maxflow.hpp before, still built with -DFASTPROX_GRAPH_BK for
// comparisons). The signal is the last axis (length n), batched over the
// others.
//
// Levels (A. Chambolle and J. Darbon, "On total variation minimization and
// surface evolution using parametric maximum flows", IJCV 84(3), 2009,
// 288-307; D. S. Hochbaum, "An efficient algorithm for image segmentation,
// Markov random fields and related problems", J. ACM 48(4), 2001, 686-701).
// For a level t, the set {z > t} of the solution is the smallest minimiser of
//     E_t(S) = sum_{p in S} (t - x_p) + gamma sum_e w_e [e crosses S],
// a minimum cut with terminal x_p - t at node p (positive towards the
// source) and capacity gamma w_e both ways on edge e; the engine's source
// tree is that smallest source side. Once a part P is cut at t, its two sides
// are independent problems: an edge e from the upper side U to the lower
// side L is replaced by the linear terms gamma w_e z_u on U and
// -gamma w_e z_v on L (with b_p the sum of these, the side's problem is
// min 1/2 |z - x|^2 - <b, z> + gamma TV). The maximum flow saturates every
// such edge from U to L, so deleting the edges and moving each side's
// terminals by its change of level leaves a feasible flow of the new
// problems: the next solve continues from the flow, on the active parts only
// (EIBFS rebuilds its forests there; BK keeps its search trees).
//
// Mean levels (Hochbaum). Inside a part the dual's divergence sums to zero,
// so the solution has mean m_P = sum_P (x_p + b_p) / |P| there. A part is
// cut at m_P; if the cut is trivial, the solution is constant m_P on P (it is
// at most m_P everywhere and has mean m_P), which is exact and ends the part.
// Otherwise its sides split into their connected pieces, each with its own
// mean. The level is clamped into the middle half of the interval known to
// hold the part's values, so every round shrinks that interval to at most
// three quarters; a clamped trivial cut only shrinks it. A part whose
// interval is narrower than eps = tol * (max x - min x), or a few ulps, is set
// to its mean, within eps of the solution at every node. All parts are cut
// in the same max-flow solve, one solve per round.
//
// Certificate and derivatives (src/ops/edge_tv.hpp). The flow is the dual:
// f_e = gamma w_e - (residual u -> v) on live edges, +-gamma w_e on deleted
// ones; the gap is zero up to rounding when every part ended exactly.
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "maxflow/maxflow.hpp"
#include "maxflow/eibfs.hpp"
#include "ops/edge_tv.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace fp {
namespace {

using maxflow::Index;
#ifndef FASTPROX_GRAPH_BK
using MF = maxflow::EIBFS<double>;
#else
using MF = maxflow::MaxFlow<double>;
#endif

using edge_tv::find;
using edge_tv::kEps;
using edge_tv::round64;
using edge_tv::unite;

struct GraphTV final : edge_tv::Base<GraphTV> {
    maxflow::Graph g;
    std::vector<int32_t> edge_of;  // edge of each arc

    GraphTV(int64_t nn, std::vector<int32_t> u, std::vector<int32_t> v, Vec ww)
        : Base(nn, std::move(u), std::move(v), std::move(ww)) {
        g.build(n, m, eu.data(), ev.data());
        edge_of.resize(2 * static_cast<size_t>(m));
        for (int64_t e = 0; e < m; ++e) {
            edge_of[g.arc[e]] = static_cast<int32_t>(e);
            edge_of[g.sister[g.arc[e]]] = static_cast<int32_t>(e);
        }
    }

    const char *name() const noexcept override { return "graph_tv"; }

    // ---- working set --------------------------------------------------------

    // Per part of the current round.
    struct Part {
        double lo, hi;  // the part's values lie in [lo, hi]
        double t;       // level of this round's cut
        double mean;    // sum_P (x_p + b_p) / |P|
        double sum;
        int32_t cnt, up;  // nodes, and nodes on the source side after the cut
        uint8_t state;    // after the cut: kSplit, kMoved (clamped and trivial) or kDone
        uint8_t clamped;  // t != mean
    };
    enum : uint8_t { kSplit = 1, kMoved = 2, kDone = 3 };

    struct Work {
        void *engine;
        double *d, *b, *lev, *zv;  // input, boundary terms, terminal level, value of frozen nodes
        int32_t *part, *uf, *nid_part, *nid_root, *act;
        uint8_t *up;
        Part *pa, *pb;  // this round's parts and the next round's
        int8_t *esgn;   // 0: live edge; +-1: deleted, the sign of z_u - z_v
    };
    size_t layout(Work *wk, char *base) const noexcept {
        const size_t N = static_cast<size_t>(n), M = static_cast<size_t>(m);
        size_t off = 0;
        auto take = [&](size_t bytes) {
            char *p = base ? base + off : nullptr;
            off += round64(bytes);
            return p;
        };
        char *eng = take(MF::bytes(g));
        char *d = take(4 * N * sizeof(double));
        char *ints = take(5 * N * sizeof(int32_t));
        char *up = take(N);
        char *pa = take(2 * N * sizeof(Part));
        char *es = take(M);
        if (wk) {
            wk->engine = eng;
            wk->d = reinterpret_cast<double *>(d);
            wk->b = wk->d + N;
            wk->lev = wk->b + N;
            wk->zv = wk->lev + N;
            wk->part = reinterpret_cast<int32_t *>(ints);
            wk->uf = wk->part + N;
            wk->nid_part = wk->uf + N;
            wk->nid_root = wk->nid_part + N;
            wk->act = wk->nid_root + N;
            wk->up = reinterpret_cast<uint8_t *>(up);
            wk->pa = reinterpret_cast<Part *>(pa);
            wk->pb = wk->pa + N;
            wk->esgn = reinterpret_cast<int8_t *>(es);
        }
        return off;
    }
    size_t image_bytes(const int64_t *) const noexcept override { return layout(nullptr, nullptr) + 64; }
    static char *aligned(char *p) noexcept { return p + ((64 - reinterpret_cast<uintptr_t>(p) % 64) % 64); }

    // Saved structure: edge_tv::Base's, then the solver's flow on every edge
    // (since 0.4), which decides the kinks: on a graph with cycles the flow
    // is not unique, and this is the one the certificate used.
    size_t base_bytes() const noexcept { return round64(Base::struct_bytes(nullptr)); }
    size_t struct_bytes(const int64_t *) const noexcept override {
        return base_bytes() + 8 * static_cast<size_t>(m);
    }
    const double *saved_flow(const char *save) const noexcept {
        return reinterpret_cast<const double *>(save + base_bytes());
    }

    // Kinks at edges (edge_tv::Base) with the saved flow; overrides join an
    // edge's ends (-1) or keep them apart (+1), rebuilding the saved regions.
    // A region is split only where every edge across the split is kept apart.
    int64_t kink_image(const char *save, const double *, const double *z, const double *, const int64_t *,
                       double gamma, double tol, int8_t *mask, const int64_t *ms, char *) const noexcept override {
        return kink_edges(z, saved_flow(save), gamma, tol, mask, ms[0]);
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
        structure(x, z, wk.lev, wk.uf, c, side, tol, saved_flow(save));
    }

    // ---- prox ---------------------------------------------------------------

    template <class T>
    int prox_img(const T *x, T *z, ImageCall &c) const noexcept {
        Work wk;
        layout(&wk, aligned(c.work));
        c.gap = 0.0;
        c.iters = 0;
        c.converged = true;
        bool done;
        const int st = read_input(x, z, wk.d, c, done);
        if (st != kRowOk || done) return st;
        const double gamma0 = c.gamma;
        const int e = rescale(wk.d, c);
        solve(wk, c);
        if (c.save) structure(wk.d, wk.zv, wk.lev, wk.uf, c);
        if (e) {  // the regions are scale-free; the saved flow is compared with gamma w_e
            unscale(e, wk.zv, gamma0, c);
            if (c.save) scale2(reinterpret_cast<double *>(c.save + base_bytes()), m, e);
        }
        for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(wk.zv[i]);
        return kRowOk;
    }

    // Level of part P for its next cut, or false when P is done (set to its mean).
    static bool choose_level(Part &P, double eps) noexcept {
        P.mean = P.sum / P.cnt;
        const double width = P.hi - P.lo;
        if (P.cnt == 1 || !(width > eps)) return false;
        const double a = P.lo + 0.25 * width, b = P.hi - 0.25 * width;
        P.t = std::min(std::max(P.mean, a), b);
        P.clamped = P.t != P.mean;
        return P.t > P.lo && P.t < P.hi;
    }

    // The edge of each arc leaving node i, with its other end, for the arcs
    // that run u -> v (each edge once, from its first node).
    template <class F>
    void out_edges(int32_t i, F &&f) const noexcept {
        for (Index a = g.first[i], end = g.first[i + 1]; a < end; ++a) {
            const int32_t e = edge_of[a];
            if (g.arc[e] == a) f(e, g.head[a]);
        }
    }

    void solve(Work &wk, ImageCall &c) const noexcept {
        const double gamma = c.gamma;
        double *d = wk.d, *b = wk.b, *lev = wk.lev, *zv = wk.zv;
        int32_t *part = wk.part, *uf = wk.uf, *map = wk.nid_part, *root_id = wk.nid_root;
        int32_t *act = wk.act;
        int8_t *esgn = wk.esgn;
        uint8_t *up = wk.up;
        double xmin = d[0], xmax = d[0];
        for (int64_t i = 1; i < n; ++i) {
            xmin = std::min(xmin, d[i]);
            xmax = std::max(xmax, d[i]);
        }
        const double scale = std::max(std::fabs(xmin), std::fabs(xmax));
        const double eps = std::max(c.tol * (xmax - xmin), 32.0 * kEps * scale);

        MF mf(g, wk.engine);
        for (int64_t e = 0; e < m; ++e) {
            const double cap = gamma * we(e);
            mf.init_edge(static_cast<Index>(e), cap, cap);
            esgn[e] = 0;
        }
        std::fill(b, b + n, 0.0);

        // Round 0: the connected components of the edges with gamma w_e > 0.
        for (int64_t i = 0; i < n; ++i) uf[i] = static_cast<int32_t>(i);
        for (int64_t e = 0; e < m; ++e)
            if (gamma * we(e) > 0.0) unite(uf, eu[e], ev[e]);
        Part *pa = wk.pa, *pb = wk.pb;
        int32_t nn = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t r = find(uf, static_cast<int32_t>(i));
            if (r == i) {
                root_id[i] = nn;
                Part &P = pb[nn++];
                P.lo = P.hi = d[i];
                P.sum = 0.0;
                P.cnt = 0;
            }
            Part &P = pb[root_id[r]];
            part[i] = root_id[r];
            P.lo = std::min(P.lo, d[i]);
            P.hi = std::max(P.hi, d[i]);
            P.sum += d[i];
            ++P.cnt;
        }
        int32_t na = 0;
        for (int32_t k = 0; k < nn; ++k) {
            const bool on = choose_level(pb[k], eps);
            map[k] = on ? na : -1;
            if (on) pa[na++] = pb[k];
        }
        int64_t nact = 0;
        for (int64_t i = 0; i < n; ++i) {
            const Part &P = pb[part[i]];
            if (map[part[i]] >= 0) {
                lev[i] = P.t;
                mf.init_terminal(static_cast<Index>(i), d[i] - P.t);
                act[nact++] = static_cast<int32_t>(i);
            } else {
                zv[i] = P.mean;
                mf.init_terminal(static_cast<Index>(i), 0.0);
            }
            part[i] = map[part[i]];
        }
        int32_t np = na;
        mf.start();

        // Rounds: one solve cuts every active part at its level. Only the
        // active nodes (act, in increasing order) and their edges are visited.
        int rounds = 0;
        while (np > 0) {
            if (rounds >= c.max_iter) {
                for (int64_t j = 0; j < nact; ++j) zv[act[j]] = pa[part[act[j]]].mean;
                c.converged = false;
                break;
            }
            ++rounds;
#ifndef FASTPROX_GRAPH_BK
            mf.solve(act, nact);  // no edge is left between finished parts and active ones
#else
            mf.solve();
#endif
            for (int32_t k = 0; k < np; ++k) pa[k].up = 0;
            for (int64_t j = 0; j < nact; ++j) {
                const int32_t i = act[j];
                up[i] = mf.tree(i) == MF::kSource;
                pa[part[i]].up += up[i];
            }
            for (int32_t k = 0; k < np; ++k) {
                Part &P = pa[k];
                if (P.up > 0 && P.up < P.cnt) {
                    P.state = kSplit;
                } else if (P.clamped) {  // the values lie on one side of t
                    (P.up == 0 ? P.hi : P.lo) = P.t;
                    P.state = kMoved;
                } else {  // trivial at the mean: constant
                    P.state = kDone;
                }
            }
            // Delete the edges across each split part's cut, then find the
            // pieces: components of the remaining edges of split parts.
            for (int64_t j = 0; j < nact; ++j) {
                const int32_t u = act[j];
                if (pa[part[u]].state != kSplit) continue;
                uf[u] = u;
                out_edges(u, [&](int32_t e, int32_t v) {
                    const double cap = gamma * we(e);
                    if (esgn[e] || !(cap > 0.0) || up[u] == up[v]) return;
                    mf.set_edge(e, 0.0, 0.0);
                    esgn[e] = up[u] ? 1 : -1;
                    b[up[u] ? u : v] -= cap;
                    b[up[u] ? v : u] += cap;
                });
            }
            for (int64_t j = 0; j < nact; ++j) {
                const int32_t u = act[j];
                if (pa[part[u]].state != kSplit) continue;
                out_edges(u, [&](int32_t e, int32_t v) {
                    if (!esgn[e] && gamma * we(e) > 0.0) unite(uf, u, v);
                });
            }
            // Next round's parts, numbered by first node: moved parts keep
            // their nodes, split parts become their pieces.
            nn = 0;
            for (int32_t k = 0; k < np; ++k) map[k] = -1;
            for (int64_t j = 0; j < nact; ++j) {
                const int32_t i = act[j];
                const int32_t k = part[i];
                const Part &P = pa[k];
                if (P.state == kDone) {
                    zv[i] = P.mean;
                    part[i] = -1;
                } else if (P.state == kMoved) {
                    if (map[k] < 0) {
                        map[k] = nn;
                        pb[nn++] = P;
                    }
                    part[i] = map[k];
                } else {
                    const int32_t r = find(uf, i);
                    if (r == i) {
                        root_id[i] = nn;
                        Part &Q = pb[nn++];
                        Q.lo = up[i] ? P.t : P.lo;
                        Q.hi = up[i] ? P.hi : P.t;
                        Q.sum = 0.0;
                        Q.cnt = 0;
                    }
                    Part &Q = pb[root_id[r]];
                    Q.sum += d[i] + b[i];
                    ++Q.cnt;
                    part[i] = root_id[r];
                }
            }
            // Levels: the parts that go on move their terminals to the new
            // level; the others take their mean.
            na = 0;
            for (int32_t k = 0; k < nn; ++k) {
                const bool on = choose_level(pb[k], eps);
                map[k] = on ? na : -1;
                if (on) pa[na++] = pb[k];
            }
            int64_t keep = 0;
            for (int64_t j = 0; j < nact; ++j) {
                const int32_t i = act[j];
                const int32_t k = part[i];
                if (k < 0) continue;
                if (map[k] >= 0) {
                    const double t = pa[map[k]].t;
                    mf.add_terminal(i, lev[i] - t);
                    lev[i] = t;
                    act[keep++] = i;
                } else {
                    zv[i] = pb[k].mean;
                }
                part[i] = map[k];
            }
            nact = keep;
            np = na;
        }
        c.iters = rounds;
        double *fsave = c.save ? reinterpret_cast<double *>(c.save + base_bytes()) : nullptr;
        certificate(d, zv, [&](int64_t e) {
            const double cap = gamma * we(e);
            const double f = esgn[e] ? esgn[e] * cap : cap - mf.residual(g.arc[e]);
            if (fsave) fsave[e] = f;
            return f;
        }, c);
    }
};

}  // namespace

fp_status make_graph_tv(int64_t n, int64_t m, const int64_t *edges, Vec w, fp_op **out) {
    std::vector<int32_t> u, v;
    const fp_status st = edge_tv::parse(n, m, edges, w, u, v);
    if (st != FP_OK) return st;
    *out = new GraphTV(n, std::move(u), std::move(v), std::move(w));
    return FP_OK;
}

}  // namespace fp
