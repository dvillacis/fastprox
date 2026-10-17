// Graph TV: the prox of gamma sum_e w_e |z_u - z_v| over the edges e = (u, v)
// of a given graph, by parametric max-flow on the engine of
// src/maxflow/maxflow.hpp. The signal is the last axis (length n), batched
// over the others.
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
// problems: the next solve continues from the flow and the search trees.
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
// Certificate. The flow is a dual: f_e (net flow u -> v, |f_e| <= gamma w_e)
// with divergence q_p = sum of f_e out of p. The duality gap of (z, f) is
//     gap = 1/2 |z - x + q|^2 + sum_e (gamma w_e |z_u - z_v| - f_e (z_u - z_v)),
// both terms nonnegative; it is zero up to rounding when every part ended
// exactly. q is written as the workspace dual.
//
// Derivatives. The solution is constant on the regions R of the graph of
// edges with gamma w_e > 0 and equal ends (up to rounding), and there
//     z_R = (sum_R x_p - gamma sum_{e in dR} w_e s_{e,R}) / |R|,
// s_{e,R} the sign of z_R minus the value across e. So the Jacobian averages
// over regions and the parameter derivatives are O(n + m).
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "maxflow/maxflow.hpp"
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
using MF = maxflow::MaxFlow<double>;

constexpr double kEps = 2.220446049250313e-16;

inline size_t round64(size_t b) noexcept { return (b + 63) / 64 * 64; }

struct GraphTV final : ImageOp<GraphTV> {
    int64_t n, m;
    std::vector<int32_t> eu, ev;
    Vec w;  // one shared weight or one per edge
    maxflow::Graph g;
    std::vector<int32_t> edge_of;  // edge of each arc

    GraphTV(int64_t nn, std::vector<int32_t> u, std::vector<int32_t> v, Vec ww)
        : n(nn), m(static_cast<int64_t>(u.size())), eu(std::move(u)), ev(std::move(v)), w(std::move(ww)) {
        g.build(n, m, eu.data(), ev.data());
        edge_of.resize(2 * static_cast<size_t>(m));
        for (int64_t e = 0; e < m; ++e) {
            edge_of[g.arc[e]] = static_cast<int32_t>(e);
            edge_of[g.sister[g.arc[e]]] = static_cast<int32_t>(e);
        }
    }

    const char *name() const noexcept override { return "graph_tv"; }
    int64_t nparams() const noexcept override { return static_cast<int64_t>(w.size()); }
    int image_ndim() const noexcept override { return 1; }
    fp_status check_image(const int64_t *shape) const noexcept override {
        return shape[0] == n ? FP_OK : FP_ESHAPE;
    }
    double we(int64_t e) const noexcept { return w.size() == 1 ? w[0] : w[static_cast<size_t>(e)]; }

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

    // Saved structure: the number of regions, a region label per node, 1/|R|
    // per region, and the sign of z_u - z_v per edge (0 inside a region).
    size_t struct_bytes(const int64_t *) const noexcept override {
        return 8 + round64(4 * static_cast<size_t>(n)) + 8 * static_cast<size_t>(n) + static_cast<size_t>(m);
    }
    struct View {
        int64_t *nreg;
        int32_t *label;
        double *inv;
        int8_t *sgn;
    };
    View view(char *save) const noexcept {
        View v;
        v.nreg = reinterpret_cast<int64_t *>(save);
        v.label = reinterpret_cast<int32_t *>(save + 8);
        v.inv = reinterpret_cast<double *>(save + 8 + round64(4 * static_cast<size_t>(n)));
        v.sgn = reinterpret_cast<int8_t *>(v.inv + n);
        return v;
    }

    // Union-find with roots at the smallest index, so labels are deterministic.
    static int32_t find(int32_t *uf, int32_t i) noexcept {
        while (uf[i] != i) {
            uf[i] = uf[uf[i]];
            i = uf[i];
        }
        return i;
    }
    static void unite(int32_t *uf, int32_t a, int32_t b) noexcept {
        a = find(uf, a);
        b = find(uf, b);
        if (a == b) return;
        if (a < b) uf[b] = a;
        else uf[a] = b;
    }

    // ---- prox ---------------------------------------------------------------

    template <class T>
    int prox_img(const T *x, T *z, ImageCall &c) const noexcept {
        Work wk;
        char *base = c.work + ((64 - reinterpret_cast<uintptr_t>(c.work) % 64) % 64);
        layout(&wk, base);
        double *d = wk.d;
        c.gap = 0.0;
        c.iters = 0;
        c.converged = true;
        bool nan = false;
        for (int64_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            if (std::isinf(v)) return kRowNonFinite;
            nan = nan || std::isnan(v);
            d[i] = v;
        }
        double *q = c.dual;
        if (nan) {  // NaN anywhere: the whole signal is NaN, with singleton regions
            for (int64_t i = 0; i < n; ++i) z[i] = std::numeric_limits<T>::quiet_NaN();
            std::fill(q, q + n, std::numeric_limits<double>::quiet_NaN());
            if (c.save) singletons(c.save);
            return kRowOk;
        }
        if (n == 0) {
            if (c.save) *view(c.save).nreg = 0;
            return kRowOk;
        }
        solve(wk, c);
        for (int64_t i = 0; i < n; ++i) z[i] = static_cast<T>(wk.zv[i]);
        if (c.save) structure(wk, c);
        return kRowOk;
    }

    void singletons(char *save) const noexcept {
        const View v = view(save);
        for (int64_t i = 0; i < n; ++i) {
            v.label[i] = static_cast<int32_t>(i);
            v.inv[i] = 1.0;
        }
        std::fill(v.sgn, v.sgn + m, int8_t(0));
        *v.nreg = n;
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
            mf.solve();
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
        certificate(wk, mf, c);
    }

    // Gap of z with the flow's dual, and the dual q into c.dual.
    void certificate(const Work &wk, const MF &mf, ImageCall &c) const noexcept {
        const double gamma = c.gamma;
        double *q = c.dual;
        const double *zv = wk.zv, *d = wk.d;
        std::fill(q, q + n, 0.0);
        double edge = 0.0;
        for (int64_t e = 0; e < m; ++e) {
            const double cap = gamma * we(e);
            const double f = wk.esgn[e] ? wk.esgn[e] * cap : cap - mf.residual(g.arc[e]);
            const int32_t u = eu[e], v = ev[e];
            q[u] += f;
            q[v] -= f;
            const double dz = zv[u] - zv[v];
            edge += cap * std::fabs(dz) - f * dz;
        }
        double node = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double r = zv[i] - d[i] + q[i];
            node += r * r;
        }
        c.gap = 0.5 * node + std::max(edge, 0.0);
    }

    // ---- structure ----------------------------------------------------------

    void structure(Work &wk, ImageCall &c) const noexcept {
        const View v = view(c.save);
        const double gamma = c.gamma;
        const double *zv = wk.zv;
        int32_t *uf = wk.uf;
        // Values are means of x plus boundary terms, within a few ulps of
        // max |x| + gamma (largest weighted degree).
        double *deg = wk.lev;
        std::fill(deg, deg + n, 0.0);
        double scale = 0.0;
        for (int64_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(wk.d[i]));
        for (int64_t e = 0; e < m; ++e) {
            const double cap = gamma * we(e);
            deg[eu[e]] += cap;
            deg[ev[e]] += cap;
        }
        double dmax = 0.0;
        for (int64_t i = 0; i < n; ++i) dmax = std::max(dmax, deg[i]);
        const double tau = 64.0 * kEps * (scale + dmax);
        for (int64_t i = 0; i < n; ++i) uf[i] = static_cast<int32_t>(i);
        for (int64_t e = 0; e < m; ++e)
            if (gamma * we(e) > 0.0 && std::fabs(zv[eu[e]] - zv[ev[e]]) <= tau) unite(uf, eu[e], ev[e]);
        int64_t nr = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t r = find(uf, static_cast<int32_t>(i));
            v.label[i] = r == i ? static_cast<int32_t>(nr++) : v.label[r];
        }
        std::fill(v.inv, v.inv + nr, 0.0);
        for (int64_t i = 0; i < n; ++i) v.inv[v.label[i]] += 1.0;
        for (int64_t R = 0; R < nr; ++R) v.inv[R] = 1.0 / v.inv[R];
        for (int64_t e = 0; e < m; ++e) {
            const int32_t a = v.label[eu[e]], bb = v.label[ev[e]];
            const double dz = zv[eu[e]] - zv[ev[e]];
            v.sgn[e] = a == bb ? 0 : (dz > 0.0 ? 1 : (dz < 0.0 ? -1 : 0));
        }
        *v.nreg = nr;
    }

    // ---- derivatives --------------------------------------------------------

    // dz_R = mean_R(dx) - (1/|R|) sum_{e in dR} (dgamma w_e + gamma dw_e) s_{e,R}.
    template <class T>
    void jvp_img(const char *save, const T *dx, T *dz, const int64_t *, double gamma, double dgamma,
                 const double *dparams, char *work, DerivCtl &) const noexcept {
        const View v = view(const_cast<char *>(save));
        const int64_t nr = *v.nreg;
        double *acc = reinterpret_cast<double *>(work);
        std::fill(acc, acc + nr, 0.0);
        for (int64_t i = 0; i < n; ++i) acc[v.label[i]] += static_cast<double>(dx[i]);
        for (int64_t e = 0; e < m; ++e) {
            if (!v.sgn[e]) continue;
            const double dwe = dparams ? dparams[w.size() == 1 ? 0 : e] : 0.0;
            const double t = (dgamma * we(e) + gamma * dwe) * v.sgn[e];
            acc[v.label[eu[e]]] -= t;
            acc[v.label[ev[e]]] += t;
        }
        for (int64_t R = 0; R < nr; ++R) acc[R] *= v.inv[R];
        for (int64_t i = 0; i < n; ++i) dz[i] = static_cast<T>(acc[v.label[i]]);
    }

    template <class T>
    void vjp_img(const char *save, const T *gin, T *gx, const int64_t *, double gamma, double *acc, char *work,
                 DerivCtl &) const noexcept {
        const View v = view(const_cast<char *>(save));
        const int64_t nr = *v.nreg;
        double *G = reinterpret_cast<double *>(work);
        std::fill(G, G + nr, 0.0);
        for (int64_t i = 0; i < n; ++i) G[v.label[i]] += static_cast<double>(gin[i]);
        for (int64_t R = 0; R < nr; ++R) G[R] *= v.inv[R];
        const int64_t P = nparams();
        for (int64_t e = 0; e < m; ++e) {
            if (!v.sgn[e]) continue;
            const double h = v.sgn[e] * (G[v.label[eu[e]]] - G[v.label[ev[e]]]);
            acc[w.size() == 1 ? 0 : e] -= gamma * h;
            acc[P] -= we(e) * h;
        }
        for (int64_t i = 0; i < n; ++i) gx[i] = static_cast<T>(G[v.label[i]]);
    }
};

}  // namespace

fp_status make_graph_tv(int64_t n, int64_t m, const int64_t *edges, Vec w, fp_op **out) {
    if (n < 0 || m < 0 || (m > 0 && !edges)) return FP_EINVAL;
    if (n >= 2147483647 || m > 1073741823) return FP_ESHAPE;
    if (w.empty()) w.assign(1, 1.0);
    if (w.size() != 1 && static_cast<int64_t>(w.size()) != m) return FP_ESHAPE;
    for (double v : w)
        if (!finite_nonneg(v)) return FP_EPARAM;
    std::vector<int32_t> u(static_cast<size_t>(m)), v(static_cast<size_t>(m));
    for (int64_t e = 0; e < m; ++e) {
        const int64_t a = edges[2 * e], b = edges[2 * e + 1];
        if (a < 0 || a >= n || b < 0 || b >= n || a == b) return FP_EPARAM;
        u[static_cast<size_t>(e)] = static_cast<int32_t>(a);
        v[static_cast<size_t>(e)] = static_cast<int32_t>(b);
    }
    *out = new GraphTV(n, std::move(u), std::move(v), std::move(w));
    return FP_OK;
}

}  // namespace fp
