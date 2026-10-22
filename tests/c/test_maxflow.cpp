// Tests of the max-flow engines (src/maxflow/maxflow.hpp, and with
// FP_TEST_EIBFS src/maxflow/eibfs.hpp, built as test_eibfs), which are
// internal and tested here directly. Every solve is checked by a certificate: the
// residuals are a feasible flow with conservation at every node, the source
// tree equals the set reachable from the source in the residual graph and
// the sink tree the set that reaches the sink, and the cut of the source
// tree costs the flow value, so both are optimal. Flow values are also
// compared with Dinic's algorithm written here. Reuse is checked against a
// fresh solve of the same residual problem, whose smallest and largest
// minimum cuts are unique, so the trees must agree node for node.
#include "maxflow/eibfs.hpp"
#include "maxflow/maxflow.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <new>
#include <random>
#include <vector>

#if defined(__SANITIZE_THREAD__)
#define FP_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FP_TSAN 1
#endif
#endif

static std::atomic<long> g_allocs{0};
#ifndef FP_TSAN
void *operator new(std::size_t n) {
    ++g_allocs;
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void *operator new(std::size_t n, const std::nothrow_t &) noexcept {
    ++g_allocs;
    return std::malloc(n ? n : 1);
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
#endif

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                                          \
    do {                                                                                     \
        ++g_checks;                                                                          \
        if (!(cond)) {                                                                       \
            ++g_fail;                                                                        \
            if (g_fail < 50) std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                    \
    } while (0)

using fp::maxflow::Graph;
using fp::maxflow::Index;
#ifdef FP_TEST_EIBFS
template <class T>
using MF = fp::maxflow::EIBFS<T>;
constexpr bool kSinkTree = false;  // EIBFS reports the source side only
#else
template <class T>
using MF = fp::maxflow::MaxFlow<T>;
constexpr bool kSinkTree = true;
#endif

static std::mt19937_64 rng(2026);
static int irand(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
static double unif(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }

// A problem: edges u -> v with capacities c_uv and c_vu, signed terminals.
struct Problem {
    Index n = 0;
    std::vector<Index> u, v;
    std::vector<double> cuv, cvu, term;
};

// Working memory for an engine, aligned for any type.
struct Mem {
    std::vector<std::max_align_t> buf;
    explicit Mem(size_t bytes) : buf(bytes / sizeof(std::max_align_t) + 1) {}
    void *data() { return buf.data(); }
};

template <class T>
static void load(MF<T> &mf, const Problem &pb) {
    for (Index e = 0; e < Index(pb.u.size()); ++e) mf.init_edge(e, T(pb.cuv[e]), T(pb.cvu[e]));
    for (Index p = 0; p < pb.n; ++p) mf.init_terminal(p, T(pb.term[p]));
    mf.start();
}

// ---------------------------------------------------------------- reference
// Dinic's algorithm on the graph with explicit terminals.
static double dinic(const Problem &pb) {
    const int n = pb.n + 2, s = pb.n, t = pb.n + 1;
    struct A {
        int to;
        double c;
    };
    std::vector<A> arcs;
    std::vector<std::vector<int>> adj(n);
    auto add = [&](int a, int b, double c1, double c2) {
        adj[a].push_back(int(arcs.size()));
        arcs.push_back({b, c1});
        adj[b].push_back(int(arcs.size()));
        arcs.push_back({a, c2});
    };
    for (size_t e = 0; e < pb.u.size(); ++e) add(pb.u[e], pb.v[e], pb.cuv[e], pb.cvu[e]);
    for (Index p = 0; p < pb.n; ++p) {
        if (pb.term[p] > 0) add(s, p, pb.term[p], 0);
        if (pb.term[p] < 0) add(p, t, -pb.term[p], 0);
    }
    std::vector<int> level(n), it(n);
    double flow = 0;
    for (;;) {
        std::fill(level.begin(), level.end(), -1);
        std::deque<int> q{s};
        level[s] = 0;
        while (!q.empty()) {
            const int a = q.front();
            q.pop_front();
            for (int k : adj[a])
                if (arcs[k].c > 0 && level[arcs[k].to] < 0) {
                    level[arcs[k].to] = level[a] + 1;
                    q.push_back(arcs[k].to);
                }
        }
        if (level[t] < 0) return flow;
        std::fill(it.begin(), it.end(), 0);
        std::vector<int> path;  // arcs from s
        for (;;) {
            const int a = path.empty() ? s : arcs[path.back()].to;
            if (a == t) {
                double d = INFINITY;
                for (int k : path) d = std::min(d, arcs[k].c);
                for (int k : path) {
                    arcs[k].c -= d;
                    arcs[k ^ 1].c += d;
                }
                flow += d;
                path.clear();
                continue;
            }
            bool moved = false;
            for (; it[a] < int(adj[a].size()); ++it[a]) {
                const int k = adj[a][it[a]];
                if (arcs[k].c > 0 && level[arcs[k].to] == level[a] + 1) {
                    path.push_back(k);
                    moved = true;
                    break;
                }
            }
            if (moved) continue;
            if (a == s) break;
            level[a] = -1;  // dead end
            path.pop_back();
        }
    }
}

// ---------------------------------------------------------------- certificate
// Sets reachable from the source and reaching the sink in the residual graph.
template <class T>
static void reach(const Graph &g, const MF<T> &mf, std::vector<char> &src, std::vector<char> &snk) {
    src.assign(g.n, 0);
    snk.assign(g.n, 0);
    std::deque<Index> q;
    for (Index p = 0; p < g.n; ++p)
        if (mf.terminal(p) > 0) src[p] = 1, q.push_back(p);
    while (!q.empty()) {
        const Index p = q.front();
        q.pop_front();
        for (Index a = g.first[p]; a < g.first[p + 1]; ++a)
            if (mf.residual(a) > 0 && !src[g.head[a]]) src[g.head[a]] = 1, q.push_back(g.head[a]);
    }
    for (Index p = 0; p < g.n; ++p)
        if (mf.terminal(p) < 0) snk[p] = 1, q.push_back(p);
    while (!q.empty()) {
        const Index p = q.front();
        q.pop_front();
        for (Index a = g.first[p]; a < g.first[p + 1]; ++a)
            if (mf.residual(g.sister[a]) > 0 && !snk[g.head[a]]) snk[g.head[a]] = 1, q.push_back(g.head[a]);
    }
}

// Checks the state left by a solve of problem pb (capacities as given, flow
// value `flow`) to tolerance tol; returns false on the first failure.
template <class T>
static bool certify(const Graph &g, const MF<T> &mf, const Problem &pb, double flow, double tol) {
    bool ok = true;
    std::vector<double> out(g.n, 0.0);
    for (Index e = 0; e < g.m; ++e) {
        const Index a = g.arc[e], b = g.sister[a];
        const double ra = mf.residual(a), rb = mf.residual(b);
        ok &= ra >= 0 && rb >= 0;
        ok &= std::fabs(ra + rb - pb.cuv[e] - pb.cvu[e]) <= tol;
        const double f = pb.cuv[e] - ra;  // net flow u -> v
        out[pb.u[e]] += f;
        out[pb.v[e]] -= f;
    }
    double from_s = 0, to_t = 0;
    for (Index p = 0; p < g.n; ++p) {
        const double c = pb.term[p], r = mf.terminal(p);
        ok &= (c > 0 ? r >= 0 && r <= c : c < 0 ? r <= 0 && r >= c : r == 0);
        ok &= std::fabs((c - r) - out[p]) <= tol;  // conservation
        if (c > 0) from_s += c - r;
        if (c < 0) to_t += r - c;
    }
    ok &= std::fabs(from_s - flow) <= tol && std::fabs(to_t - flow) <= tol;
    std::vector<char> src, snk;
    reach(g, mf, src, snk);
    double cut = 0;
    for (Index p = 0; p < g.n; ++p) {
        ok &= (mf.tree(p) == MF<T>::kSource) == bool(src[p]);
        if (kSinkTree) ok &= (mf.tree(p) == MF<T>::kSink) == bool(snk[p]);
        ok &= !(src[p] && snk[p]);
        if (!src[p] && pb.term[p] > 0) cut += pb.term[p];
        if (src[p] && pb.term[p] < 0) cut -= pb.term[p];
    }
    for (Index e = 0; e < g.m; ++e) {
        if (src[pb.u[e]] && !src[pb.v[e]]) cut += pb.cuv[e];
        if (src[pb.v[e]] && !src[pb.u[e]]) cut += pb.cvu[e];
    }
    ok &= std::fabs(cut - flow) <= tol;
    return ok;
}

// The engine's net flow (audit T6), less f0 when given, against
// cap - residual on every edge.
template <class T>
static bool flows_match(const Graph &g, const MF<T> &mf, const std::vector<double> &cuv, const double *f0,
                        double tol) {
    bool ok = true;
    for (Index e = 0; e < g.m; ++e) {
        const Index a = g.arc[e];
        const double f = double(mf.flow(a)) - (f0 ? f0[e] : 0.0);
        ok &= std::fabs(f - (cuv[e] - double(mf.residual(a)))) <= tol;
        ok &= mf.flow(g.sister[a]) == -mf.flow(a);
    }
    return ok;
}

// The residual problem of the current state: capacities are the residuals.
template <class T>
static Problem residual_problem(const Graph &g, const MF<T> &mf, const Problem &pb) {
    Problem r = pb;
    for (Index e = 0; e < g.m; ++e) {
        r.cuv[e] = double(mf.residual(g.arc[e]));
        r.cvu[e] = double(mf.residual(g.sister[g.arc[e]]));
    }
    for (Index p = 0; p < g.n; ++p) r.term[p] = double(mf.terminal(p));
    return r;
}

// ---------------------------------------------------------------- problems
// Random graph with integer capacities in [0, k] (some zero), parallel edges
// allowed, terminals in [-k, k].
static Problem random_problem(Index n, Index m, int k, bool integer) {
    Problem pb;
    pb.n = n;
    for (Index e = 0; e < m && n > 1; ++e) {
        Index a = irand(0, n - 1), b = irand(0, n - 2);
        if (b >= a) ++b;
        pb.u.push_back(a);
        pb.v.push_back(b);
        auto cap = [&] { return irand(0, 3) == 0 ? 0.0 : integer ? double(irand(1, k)) : unif(0, k); };
        pb.cuv.push_back(cap());
        pb.cvu.push_back(irand(0, 2) == 0 ? 0.0 : cap());
    }
    for (Index p = 0; p < n; ++p) {
        const int r = irand(0, 2);
        const double c = integer ? double(irand(1, k)) : unif(0, k);
        pb.term.push_back(r == 0 ? 0.0 : r == 1 ? c : -c);
    }
    return pb;
}

// Hubs (since 0.3; MaxFlow scans nodes of degree above 32 differently): a
// random problem plus `hubs` nodes joined to a share of all the others, or
// with share 1 and no random edges, a star around node 0.
static Problem hub_problem(Index n, Index m, int hubs, double share, int k, bool integer) {
    Problem pb = random_problem(n, m, k, integer);
    auto cap = [&] { return irand(0, 3) == 0 ? 0.0 : integer ? double(irand(1, k)) : unif(0, k); };
    for (int h = 0; h < hubs && n > 1; ++h) {
        const Index c = irand(0, n - 1);
        for (Index q = 0; q < n; ++q) {
            if (q == c || unif(0, 1) >= share) continue;
            pb.u.push_back(irand(0, 1) ? c : q);
            pb.v.push_back(pb.u.back() == c ? q : c);
            pb.cuv.push_back(cap());
            pb.cvu.push_back(irand(0, 2) == 0 ? 0.0 : cap());
        }
    }
    return pb;
}

// Graph TV cut at level t on a grid of shape (h, w) or (d, h, w): terminal
// x_p - t, edges of weight lam both ways (4, 8 or 6 neighbours).
static Problem grid_problem(const std::vector<int> &shape, int nb, const std::vector<double> &x, double t,
                            double lam) {
    Problem pb;
    int n = 1;
    for (int s : shape) n *= s;
    pb.n = n;
    auto edge = [&](int a, int b, double c) {
        pb.u.push_back(a);
        pb.v.push_back(b);
        pb.cuv.push_back(c);
        pb.cvu.push_back(c);
    };
    if (shape.size() == 2) {
        const int h = shape[0], w = shape[1];
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j) {
                const int p = i * w + j;
                if (j + 1 < w) edge(p, p + 1, lam);
                if (i + 1 < h) edge(p, p + w, lam);
                if (nb == 8 && i + 1 < h && j + 1 < w) edge(p, p + w + 1, lam / std::sqrt(2.0));
                if (nb == 8 && i + 1 < h && j > 0) edge(p, p + w - 1, lam / std::sqrt(2.0));
            }
    } else {
        const int d = shape[0], h = shape[1], w = shape[2];
        for (int k = 0; k < d; ++k)
            for (int i = 0; i < h; ++i)
                for (int j = 0; j < w; ++j) {
                    const int p = (k * h + i) * w + j;
                    if (j + 1 < w) edge(p, p + 1, lam);
                    if (i + 1 < h) edge(p, p + w, lam);
                    if (k + 1 < d) edge(p, p + h * w, lam);
                }
    }
    for (int p = 0; p < n; ++p) pb.term.push_back(x[p] - t);
    return pb;
}

static std::vector<double> blocky(int n, int levels, bool integer) {
    std::vector<double> x(n);
    double v = 0;
    for (int p = 0; p < n; ++p) {
        if (irand(0, 9) == 0) v = double(irand(0, levels));
        x[p] = integer ? v + irand(-2, 2) : v + unif(-1, 1);
    }
    return x;
}

template <class T>
static void build(Graph &g, const Problem &pb) {
    g.build(pb.n, int64_t(pb.u.size()), pb.u.data(), pb.v.data());
}

// ---------------------------------------------------------------- tests
template <class T>
static void check_fresh(const Problem &pb, double tol) {
    Graph g;
    build<T>(g, pb);
    Mem mem(MF<T>::bytes(g));
    MF<T> mf(g, mem.data());
    mf.track_flow();
    load(mf, pb);
    const double flow = double(mf.solve());
    CHECK(certify(g, mf, pb, flow, tol));
    CHECK(std::fabs(flow - dinic(pb)) <= tol);
    CHECK(flows_match(g, mf, pb.cuv, nullptr, tol));
    // A second solve finds nothing more and leaves the state alone.
    CHECK(mf.solve() == 0);
    CHECK(certify(g, mf, pb, flow, tol));
}

static void test_random() {
    const int sizes[][2] = {{1, 0}, {2, 1}, {2, 4}, {3, 3}, {5, 8}, {10, 30}, {40, 100}, {200, 1000}, {1000, 3000}, {2000, 20000}};
    for (auto &s : sizes)
        for (int rep = 0; rep < 6; ++rep) {
            const int k = rep % 2 ? 3 : 1000;
            check_fresh<double>(random_problem(s[0], s[1], k, true), 0.0);
            check_fresh<float>(random_problem(s[0], s[1], 50, true), 0.0);
            const Problem r = random_problem(s[0], s[1], 10, false);
            check_fresh<double>(r, 1e-9 * (1 + s[1]));
        }
    // Edge cases: no terminals, all zero, one-sided terminals.
    Problem z = random_problem(30, 60, 5, true);
    std::fill(z.term.begin(), z.term.end(), 0.0);
    check_fresh<double>(z, 0.0);
    std::fill(z.cuv.begin(), z.cuv.end(), 0.0);
    std::fill(z.cvu.begin(), z.cvu.end(), 0.0);
    for (auto &c : z.term) c = double(irand(-3, 3));
    check_fresh<double>(z, 0.0);
    Problem o = random_problem(50, 200, 5, true);
    for (auto &c : o.term) c = std::fabs(c);
    check_fresh<double>(o, 0.0);
}

static void test_grids() {
    const std::vector<std::vector<int>> shapes = {{1, 1}, {1, 7}, {7, 1}, {16, 16}, {33, 47}, {96, 80}};
    for (const auto &sh : shapes)
        for (int nb : {4, 8})
            for (double lam : {0.0, 0.5, 2.0, 20.0}) {
                const int n = sh[0] * sh[1];
                const auto x = blocky(n, 6, nb == 4);
                const double t = 3.0;
                check_fresh<double>(grid_problem(sh, nb, x, t, nb == 4 ? std::round(lam) : lam),
                                    nb == 4 ? 0.0 : 1e-9 * n);
            }
    for (double lam : {0.5, 3.0}) {
        const std::vector<int> sh = {12, 20, 18};
        const auto x = blocky(sh[0] * sh[1] * sh[2], 6, false);
        check_fresh<double>(grid_problem(sh, 6, x, 2.5, lam), 1e-8);
    }
}

// Random changes between solves: every reused solve must agree with a fresh
// solve of the same residual problem.
template <class T>
static void check_reuse(Problem pb, int rounds, int k, bool integer) {
    Graph g;
    build<T>(g, pb);
    Mem mem(MF<T>::bytes(g)), mem2(MF<T>::bytes(g));
    MF<T> mf(g, mem.data());
    mf.track_flow();
    load(mf, pb);
    mf.solve();
    const double tol = integer ? 0.0 : 1e-9 * (1 + g.m);
    for (int round = 0; round < rounds; ++round) {
        const int changes = irand(1, std::max(1, int(g.n) / 4));
        for (int c = 0; c < changes; ++c) {
            const int kind = irand(0, 9);
            if (kind < 6 || g.m == 0) {
                const Index p = irand(0, g.n - 1);
                const double d = integer ? double(irand(-k, k)) : unif(-k, k);
                if (kind == 0)
                    mf.set_terminal(p, T(0));
                else if (kind == 1)
                    mf.set_terminal(p, T(d));
                else
                    mf.add_terminal(p, T(d));
            } else {
                const Index e = irand(0, g.m - 1);
                if (kind < 8)
                    mf.set_edge(e, T(0), T(0));
                else
                    mf.set_edge(e, T(integer ? irand(0, k) : unif(0, k)), T(integer ? irand(0, k) : unif(0, k)));
            }
        }
        // set_terminal and add_terminal keep integers exact; the residual
        // problem is read before the reused solve.
        const Problem rp = residual_problem(g, mf, pb);
        std::vector<double> f0(g.m);
        for (Index e = 0; e < g.m; ++e) f0[e] = double(mf.flow(g.arc[e]));
        const double flow = double(mf.solve());
        CHECK(certify(g, mf, rp, flow, tol));
        CHECK(flows_match(g, mf, rp.cuv, f0.data(), tol));
        MF<T> fresh(g, mem2.data());
        load(fresh, rp);
        const double flow2 = double(fresh.solve());
        CHECK(std::fabs(flow - flow2) <= tol);
        bool same = true;
        for (Index p = 0; p < g.n; ++p) same &= mf.tree(p) == fresh.tree(p);
        CHECK(same);
    }
}

// Capacities far above the flow (graph TV at a large step, audit T6):
// cap - residual loses the flow below an ulp of cap, the engine's net flow
// does not, so conservation holds at every node to rounding of the terminals.
static void test_large_caps() {
    for (int rep = 0; rep < 12; ++rep) {
        Problem pb = random_problem(rep < 6 ? 200 : 1000, rep < 6 ? 1000 : 5000, 10, false);
        const double big = rep % 3 == 0 ? 1e17 : rep % 3 == 1 ? 1e20 : 1e280;
        for (size_t e = 0; e < pb.u.size(); ++e)
            if (irand(0, 1)) pb.cuv[e] *= big, pb.cvu[e] *= big;
        Graph g;
        build<double>(g, pb);
        Mem mem(MF<double>::bytes(g));
        MF<double> mf(g, mem.data());
        mf.track_flow();
        load(mf, pb);
        mf.solve();
        std::vector<double> out(g.n, 0.0);
        double scale = 0.0;
        for (Index e = 0; e < g.m; ++e) {
            const double f = mf.flow(g.arc[e]);
            // within capacity to rounding of the summed pushes (terminals <= 10)
            CHECK(f - pb.cuv[e] <= 1e-12 * (pb.cuv[e] + 10.0) && -f - pb.cvu[e] <= 1e-12 * (pb.cvu[e] + 10.0));
            out[pb.u[e]] += f;
            out[pb.v[e]] -= f;
        }
        for (Index p = 0; p < g.n; ++p) scale = std::max(scale, std::fabs(pb.term[p]));
        double worst = 0.0;
        for (Index p = 0; p < g.n; ++p) worst = std::max(worst, std::fabs(pb.term[p] - mf.terminal(p) - out[p]));
        CHECK(worst <= 1e-12 * scale * double(g.m));
    }
}

static void test_reuse() {
    for (int rep = 0; rep < 8; ++rep) {
        check_reuse<double>(random_problem(30, 80, 5, true), 30, 5, true);
        check_reuse<double>(random_problem(400, 2000, 20, true), 20, 20, true);
        check_reuse<float>(random_problem(200, 800, 10, true), 20, 10, true);
        check_reuse<double>(random_problem(300, 1500, 5, false), 15, 5, false);
    }
    const std::vector<int> sh = {40, 50};
    check_reuse<double>(grid_problem(sh, 4, blocky(2000, 6, true), 3.0, 1.0), 30, 4, true);
}

static void test_hubs() {
    for (int rep = 0; rep < 6; ++rep) {
        const int k = rep % 2 ? 3 : 1000;
        check_fresh<double>(hub_problem(300, 0, 1, 1.0, k, true), 0.0);        // a star
        check_fresh<double>(hub_problem(400, 800, 3, 0.4, k, true), 0.0);
        check_fresh<double>(hub_problem(2000, 4000, 5, 0.1, k, true), 0.0);
        check_fresh<float>(hub_problem(500, 1000, 2, 0.5, 50, true), 0.0);
        check_fresh<double>(hub_problem(400, 800, 3, 0.4, 10, false), 1e-9 * 3000);
    }
    for (int rep = 0; rep < 6; ++rep) {
        check_reuse<double>(hub_problem(200, 0, 1, 1.0, 5, true), 25, 5, true);
        check_reuse<double>(hub_problem(400, 1000, 3, 0.3, 20, true), 20, 20, true);
        check_reuse<double>(hub_problem(300, 600, 2, 0.5, 5, false), 15, 5, false);
    }
}

// The graph TV pattern (Chambolle and Darbon; Hochbaum): after a cut at
// level t, the edges across it are removed and each side moves to its next
// level by shifting its terminal residuals, every side in the same solve.
// With integer data every level is exact, and the source side found inside
// each part must be the set {z > t'} of a fresh solve of the whole problem
// at that part's level t', restricted to the part.
static void test_levels() {
    const std::vector<std::vector<int>> shapes = {{48, 64}, {48, 64}, {10, 12, 14}};
    for (size_t c = 0; c < shapes.size(); ++c) {
        const std::vector<int> &sh = shapes[c];
        const int nb = sh.size() == 3 ? 6 : 4;
        int n = 1;
        for (int s : sh) n *= s;
        const auto x = blocky(n, 16, true);
        const double lam = c == 0 ? 2.0 : 1.0;
        const Problem base = grid_problem(sh, nb, x, 0.0, lam);
        Graph g;
        build<double>(g, base);
        Mem mem(MF<double>::bytes(g)), mem2(MF<double>::bytes(g));
        MF<double> mf(g, mem.data());
        // Levels in [lo, hi) per part; part of a node by its interval.
        std::vector<double> lo(n, -4.0), hi(n, 20.0), lev(n, 8.0);
        Problem start = base;
        for (int p = 0; p < n; ++p) start.term[p] = x[p] - lev[p];
        load(mf, start);
        long allocs = 0;
        for (int depth = 0; depth < 5; ++depth) {
            const long a0 = g_allocs.load();
            mf.solve();
            allocs += g_allocs.load() - a0;
            // Check every part against a fresh full solve at its level.
            bool ok = true;
            std::vector<double> levels(lev);
            std::sort(levels.begin(), levels.end());
            levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
            for (double t : levels) {
                MF<double> fresh(g, mem2.data());
                load(fresh, grid_problem(sh, nb, x, t, lam));
                fresh.solve();
                for (int p = 0; p < n; ++p)
                    if (lev[p] == t) ok &= (mf.tree(p) == MF<double>::kSource) == (fresh.tree(p) == MF<double>::kSource);
            }
            CHECK(ok);
            // Split: remove the edges across the cut, move each side to the
            // middle of its new interval.
            std::vector<char> up(n);
            for (int p = 0; p < n; ++p) up[p] = mf.tree(p) == MF<double>::kSource;
            const long a1 = g_allocs.load();
            for (Index e = 0; e < g.m; ++e)
                if (up[base.u[e]] != up[base.v[e]]) mf.set_edge(e, 0.0, 0.0);
            for (int p = 0; p < n; ++p) {
                if (up[p])
                    lo[p] = lev[p];
                else
                    hi[p] = lev[p];
                const double t = std::floor((lo[p] + hi[p]) / 2);
                mf.add_terminal(p, lev[p] - t);
                lev[p] = t;
            }
            allocs += g_allocs.load() - a1;
        }
        CHECK(allocs == 0);
    }
}

#ifdef FP_TEST_EIBFS
// EIBFS's solve on a set of nodes: two disjoint problems, the first solved
// alone must match a solve of both on its nodes, the second left untouched.
static void test_subset() {
    for (int rep = 0; rep < 8; ++rep) {
        const Problem a = random_problem(300, 1200, 10, true), b = random_problem(200, 800, 10, true);
        Problem pb = a;
        for (size_t e = 0; e < b.u.size(); ++e) {
            pb.u.push_back(b.u[e] + a.n);
            pb.v.push_back(b.v[e] + a.n);
            pb.cuv.push_back(b.cuv[e]);
            pb.cvu.push_back(b.cvu[e]);
        }
        pb.term.insert(pb.term.end(), b.term.begin(), b.term.end());
        pb.n = a.n + b.n;
        Graph g;
        build<double>(g, pb);
        Mem mem(MF<double>::bytes(g)), mem2(MF<double>::bytes(g));
        MF<double> part(g, mem.data()), full(g, mem2.data());
        load(part, pb);
        load(full, pb);
        std::vector<int32_t> nodes(a.n);
        for (Index p = 0; p < a.n; ++p) nodes[p] = rep % 2 ? a.n - 1 - p : p;  // any order
        part.solve(nodes.data(), a.n);
        full.solve();
        bool same = true, untouched = true;
        for (Index p = 0; p < a.n; ++p) same &= part.tree(p) == full.tree(p);
        for (Index p = a.n; p < pb.n; ++p) untouched &= part.terminal(p) == pb.term[p];
        for (Index e = Index(a.u.size()); e < g.m; ++e)
            untouched &= part.residual(g.arc[e]) == pb.cuv[e] && part.residual(g.sister[g.arc[e]]) == pb.cvu[e];
        CHECK(same);
        CHECK(untouched);
    }
    // After a full solve every node has forest state; cutting the edges
    // between two halves and solving one must leave the other alone (graph
    // TV's finished parts) and agree with a fresh solve of that half.
    for (int rep = 0; rep < 8; ++rep) {
        const std::vector<int> sh = {30, 40};
        Problem pb = grid_problem(sh, rep % 2 ? 8 : 4, blocky(1200, 6, true), 3.0, 1.0);
        Graph g;
        build<double>(g, pb);
        Mem mem(MF<double>::bytes(g)), mem2(MF<double>::bytes(g));
        MF<double> mf(g, mem.data());
        load(mf, pb);
        mf.solve();
        std::vector<char> in(pb.n);
        for (Index p = 0; p < pb.n; ++p) in[p] = (p % 40) < 20 + rep;  // left columns
        for (Index e = 0; e < g.m; ++e)
            if (in[pb.u[e]] != in[pb.v[e]]) mf.set_edge(e, 0.0, 0.0);
        for (Index p = 0; p < pb.n; ++p)
            if (in[p]) mf.add_terminal(p, double(irand(-3, 3)));
        const Problem rp = residual_problem(g, mf, pb);
        std::vector<int32_t> nodes;
        for (Index p = 0; p < pb.n; ++p)
            if (in[p]) nodes.push_back(p);
        mf.solve(nodes.data(), int64_t(nodes.size()));
        MF<double> fresh(g, mem2.data());
        load(fresh, rp);
        fresh.solve();
        bool same = true, untouched = true;
        for (Index p = 0; p < pb.n; ++p) {
            if (in[p]) same &= mf.tree(p) == fresh.tree(p);
            else untouched &= mf.terminal(p) == rp.term[p];
        }
        for (Index e = 0; e < g.m; ++e)
            if (!in[pb.u[e]] && !in[pb.v[e]])
                untouched &= mf.residual(g.arc[e]) == rp.cuv[e] && mf.residual(g.sister[g.arc[e]]) == rp.cvu[e];
        CHECK(same);
        CHECK(untouched);
    }
}
#endif

static void test_build_errors() {
    Graph g;
    const int64_t u[] = {0, 1}, v[] = {1, 1};
    bool threw = false;
    try {
        g.build(2, 2, u, v);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    const int64_t w[] = {0, 2};
    threw = false;
    try {
        g.build(2, 2, u, w);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        g.build(int64_t(1) << 40, 0, u, v);
    } catch (const std::length_error &) {
        threw = true;
    }
    CHECK(threw);
    g.build(0, 0, u, v);
    Mem mem(MF<double>::bytes(g));
    MF<double> mf(g, mem.data());
    mf.start();
    CHECK(mf.solve() == 0);
}

int main() {
    test_build_errors();
    test_random();
    test_hubs();
    test_grids();
    test_reuse();
    test_large_caps();
    test_levels();
#ifdef FP_TEST_EIBFS
    test_subset();
#endif
    std::printf("%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
