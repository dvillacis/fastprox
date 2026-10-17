// The max-flow engine (src/maxflow/maxflow.hpp) on the cut problems of
// graph TV: grids of 4 or 8 neighbours in 2D and 6 in 3D, terminal x_p - t,
// edges gamma w. Two parts:
//  * one cut at the median level, against Dinic's algorithm written here
//    (arrays by tail, BFS levels, DFS with current-arc pointers);
//  * the divide and conquer over levels of graph TV down to a precision of
//    2^-depth of the value range, where every part of the image is cut at
//    the middle of its interval in the same solve. Three ways: keeping the
//    flow and the trees between levels (the engine's reuse), keeping the
//    flow but replanting the trees (start() before every solve), and
//    building every level from scratch. They must find the same cuts.
//
//   c++ -std=c++17 -O3 -DNDEBUG -fno-fast-math -Isrc bench/bench_maxflow.cpp -o bench_maxflow
//   ./bench_maxflow [reps]
//
// The input is synthetic: a smooth ramp with disks and squares and Gaussian
// noise of 0.1, values about [0, 1].
//
// SPDX-License-Identifier: MIT
#include "maxflow/maxflow.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using fp::maxflow::Graph;
using fp::maxflow::Index;
using MF = fp::maxflow::MaxFlow<double>;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Grid {
    std::vector<int> shape;
    int nb = 4;
    Index n = 0;
    std::vector<Index> u, v;
    std::vector<double> w;
};

static Grid grid(const std::vector<int> &shape, int nb) {
    Grid g;
    g.shape = shape;
    g.nb = nb;
    auto edge = [&](Index a, Index b, double w) {
        g.u.push_back(a);
        g.v.push_back(b);
        g.w.push_back(w);
    };
    if (shape.size() == 2) {
        const int h = shape[0], w = shape[1];
        g.n = h * w;
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j) {
                const Index p = i * w + j;
                if (j + 1 < w) edge(p, p + 1, 1.0);
                if (i + 1 < h) edge(p, p + w, 1.0);
                if (nb == 8 && i + 1 < h && j + 1 < w) edge(p, p + w + 1, std::sqrt(0.5));
                if (nb == 8 && i + 1 < h && j > 0) edge(p, p + w - 1, std::sqrt(0.5));
            }
    } else {
        const int d = shape[0], h = shape[1], w = shape[2];
        g.n = d * h * w;
        for (int k = 0; k < d; ++k)
            for (int i = 0; i < h; ++i)
                for (int j = 0; j < w; ++j) {
                    const Index p = (k * h + i) * w + j;
                    if (j + 1 < w) edge(p, p + 1, 1.0);
                    if (i + 1 < h) edge(p, p + w, 1.0);
                    if (k + 1 < d) edge(p, p + h * w, 1.0);
                }
    }
    return g;
}

static std::vector<double> image(const std::vector<int> &shape) {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> noise(0.0, 0.1);
    const int d = shape.size() == 3 ? shape[0] : 1, h = shape[shape.size() - 2], w = shape.back();
    std::vector<double> x(size_t(d) * h * w);
    for (int k = 0; k < d; ++k)
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j) {
                const double a = double(i) / h, b = double(j) / w, c = double(k) / d;
                double v = 0.3 * b + 0.1 * c;
                if ((a - 0.35) * (a - 0.35) + (b - 0.3) * (b - 0.3) < 0.04) v = 0.8;
                if (std::fabs(a - 0.7) < 0.15 && std::fabs(b - 0.65) < 0.2) v = 0.15;
                if ((a - 0.6) * (a - 0.6) + (b - 0.25) * (b - 0.25) < 0.01) v = 0.95;
                x[(size_t(k) * h + i) * w + j] = v + noise(rng);
            }
    return x;
}

// ---------------------------------------------------------------- Dinic
struct Dinic {
    Index n;
    const Graph &g;
    std::vector<double> r;  // arcs of g, then source arcs and sink arcs per node
    std::vector<double> rs, rt;
    std::vector<Index> level, cur, queue;
    Dinic(const Graph &g) : n(g.n), g(g), r(2 * size_t(g.m)), rs(g.n), rt(g.n), level(g.n), cur(g.n), queue(g.n) {}

    // Reaches node p with at most `lim` from the source; returns the flow
    // pushed from p to the sink.
    double dfs(Index p, double lim) {
        double pushed = 0;
        if (rt[p] > 0) {
            const double d = std::min(lim, rt[p]);
            rt[p] -= d;
            pushed += d;
            if (pushed == lim) return pushed;
        }
        for (Index &a = cur[p]; a < g.first[p + 1]; ++a) {
            const Index q = g.head[a];
            if (r[a] > 0 && level[q] == level[p] + 1) {
                const double d = dfs(q, std::min(lim - pushed, r[a]));
                if (d > 0) {
                    r[a] -= d;
                    r[g.sister[a]] += d;
                    pushed += d;
                    if (pushed == lim) return pushed;
                }
            }
        }
        level[p] = -1;
        return pushed;
    }

    double solve() {
        double flow = 0;
        for (;;) {
            Index qh = 0, qt = 0;
            bool sink = false;
            for (Index p = 0; p < n; ++p) {
                level[p] = rs[p] > 0 ? 1 : -1;
                if (rs[p] > 0) queue[qt++] = p;
            }
            while (qh < qt) {
                const Index p = queue[qh++];
                if (rt[p] > 0) sink = true;
                for (Index a = g.first[p]; a < g.first[p + 1]; ++a)
                    if (r[a] > 0 && level[g.head[a]] < 0) {
                        level[g.head[a]] = level[p] + 1;
                        queue[qt++] = g.head[a];
                    }
            }
            if (!sink) return flow;
            for (Index p = 0; p < n; ++p) cur[p] = g.first[p];
            for (Index p = 0; p < n; ++p)
                if (rs[p] > 0 && level[p] == 1) {
                    const double d = dfs(p, rs[p]);
                    rs[p] -= d;
                    flow += d;
                }
        }
    }
};

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void load_cut(MF &mf, const Grid &gr, const std::vector<double> &x, double t, double gamma) {
    for (Index e = 0; e < Index(gr.w.size()); ++e) mf.init_edge(e, gamma * gr.w[e], gamma * gr.w[e]);
    for (Index p = 0; p < gr.n; ++p) mf.init_terminal(p, x[p] - t);
    mf.start();
}

static void one_cut(const Grid &gr, const Graph &g, const std::vector<double> &x, double gamma, int reps) {
    std::vector<double> xs(x);
    std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
    const double t = xs[xs.size() / 2];
    std::vector<std::max_align_t> mem(MF::bytes(g) / sizeof(std::max_align_t) + 1);
    std::vector<double> tb, td;
    double fb = 0, fd = 0;
    for (int rep = 0; rep < reps; ++rep) {
        MF mf(g, mem.data());
        double t0 = now();
        load_cut(mf, gr, x, t, gamma);
        fb = mf.solve();
        tb.push_back(now() - t0);
        Dinic dn(g);
        t0 = now();
        for (Index e = 0; e < g.m; ++e) {
            dn.r[g.arc[e]] = gamma * gr.w[e];
            dn.r[g.sister[g.arc[e]]] = gamma * gr.w[e];
        }
        for (Index p = 0; p < g.n; ++p) {
            dn.rs[p] = std::max(x[p] - t, 0.0);
            dn.rt[p] = std::max(t - x[p], 0.0);
        }
        fd = dn.solve();
        td.push_back(now() - t0);
    }
    std::printf("  gamma %-5g cut: engine %8.2f ms   Dinic %8.2f ms   (%.1fx)   flow rel. diff %.1e\n", gamma,
                1e3 * median(tb), 1e3 * median(td), median(td) / median(tb), std::fabs(fb - fd) / std::max(fb, 1e-300));
}

// Divide and conquer over levels. mode 0: reuse flow and trees, 1: reuse the
// flow and replant the trees, 2: every level from scratch. Returns the
// source sides of the last level (one bit per node per level, packed) and the
// time.
static double levels(const Grid &gr, const Graph &g, const std::vector<double> &x, double gamma, int depth, int mode,
                     std::vector<char> &up_all) {
    const Index n = gr.n;
    std::vector<std::max_align_t> mem(MF::bytes(g) / sizeof(std::max_align_t) + 1);
    MF mf(g, mem.data());
    const double xmin = *std::min_element(x.begin(), x.end()), xmax = *std::max_element(x.begin(), x.end());
    std::vector<double> lo(n, xmin), hi(n, xmax), lev(n, 0.5 * (xmin + xmax)), bnd(n, 0.0);
    std::vector<char> cut(g.m, 0), up(n);
    up_all.assign(size_t(n) * depth, 0);
    const double t0 = now();
    for (Index e = 0; e < g.m; ++e) mf.init_edge(e, gamma * gr.w[e], gamma * gr.w[e]);
    for (Index p = 0; p < n; ++p) mf.init_terminal(p, x[p] - lev[p]);
    mf.start();
    for (int k = 0; k < depth; ++k) {
        if (k > 0 && mode == 1) mf.start();
        if (k > 0 && mode == 2) {
            for (Index e = 0; e < g.m; ++e) {
                const double c = cut[e] ? 0.0 : gamma * gr.w[e];
                mf.init_edge(e, c, c);
            }
            for (Index p = 0; p < n; ++p) mf.init_terminal(p, x[p] - lev[p] + bnd[p]);
            mf.start();
        }
        mf.solve();
        for (Index p = 0; p < n; ++p) up[p] = up_all[size_t(k) * n + p] = mf.tree(p) == MF::kSource;
        for (Index e = 0; e < g.m; ++e) {
            const Index a = gr.u[e], b = gr.v[e];
            if (cut[e] || up[a] == up[b]) continue;
            cut[e] = 1;
            const double c = gamma * gr.w[e];
            bnd[up[a] ? a : b] -= c;
            bnd[up[a] ? b : a] += c;
            if (mode < 2) mf.set_edge(e, 0.0, 0.0);
        }
        for (Index p = 0; p < n; ++p) {
            (up[p] ? lo[p] : hi[p]) = lev[p];
            const double t = 0.5 * (lo[p] + hi[p]);
            if (mode < 2) mf.add_terminal(p, lev[p] - t);
            lev[p] = t;
        }
    }
    return now() - t0;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int depth = 16;
    struct Case {
        std::vector<int> shape;
        int nb;
    };
    const Case cases[] = {{{256, 256}, 4}, {{512, 512}, 4}, {{1024, 1024}, 4}, {{512, 512}, 8}, {{64, 64, 64}, 6}};
    for (const Case &c : cases) {
        const Grid gr = grid(c.shape, c.nb);
        Graph g;
        g.build(gr.n, Index(gr.u.size()), gr.u.data(), gr.v.data());
        const auto x = image(c.shape);
        std::printf("%s %d neighbours, %d nodes, %d edges\n",
                    c.shape.size() == 2 ? (std::to_string(c.shape[0]) + "x" + std::to_string(c.shape[1])).c_str()
                                        : (std::to_string(c.shape[0]) + "^3").c_str(),
                    c.nb, int(gr.n), int(g.m));
        for (double gamma : {0.02, 0.1, 0.5}) {
            one_cut(gr, g, x, gamma, reps);
            std::vector<double> t[3];
            std::vector<char> up[3];
            for (int rep = 0; rep < std::max(1, reps / 2); ++rep)
                for (int mode = 0; mode < 3; ++mode) t[mode].push_back(levels(gr, g, x, gamma, depth, mode, up[mode]));
            size_t diff1 = 0, diff2 = 0;
            for (size_t i = 0; i < up[0].size(); ++i) {
                diff1 += up[0][i] != up[1][i];
                diff2 += up[0][i] != up[2][i];
            }
            std::printf("          %d levels: reuse %8.1f ms   replant %8.1f ms (%.2fx)   scratch %8.1f ms (%.2fx)"
                        "   cuts differ: %zu, %zu\n",
                        depth, 1e3 * median(t[0]), 1e3 * median(t[1]), median(t[1]) / median(t[0]),
                        1e3 * median(t[2]), median(t[2]) / median(t[0]), diff1, diff2);
        }
    }
}
