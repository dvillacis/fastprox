// Core tests through the public C API: optimality certificates of the direct
// algorithms, path agreement for 1D TV, derivatives (finite differences and
// adjointness), determinism across thread counts, strided and in-place
// arrays, and the no-allocation guarantee.
#include "fastprox.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <new>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------- allocation counter
static std::atomic<long> g_allocs{0};
void *operator new(std::size_t n) {
    ++g_allocs;
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }

// ---------------------------------------------------------------- mini framework
static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_fail;                                                                 \
            if (g_fail < 50) std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                             \
    } while (0)
#define CHECK_OK(expr) CHECK((expr) == FP_OK)

using Vec = std::vector<double>;
static std::mt19937_64 rng(12345);

static Vec randn(int64_t n, double scale = 1.0) {
    std::normal_distribution<double> d(0.0, scale);
    Vec v(static_cast<size_t>(n));
    for (auto &x : v) x = d(rng);
    return v;
}
static double unif(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }

struct Arr {  // owning C-contiguous array with an fp_array view
    std::vector<int64_t> shape, strides;
    Vec data;
    fp_array a{};
    explicit Arr(std::vector<int64_t> s, Vec d = {}) : shape(std::move(s)) {
        int64_t m = 1;
        strides.resize(shape.size());
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = m;
            m *= shape[i];
        }
        data = d.empty() ? Vec(static_cast<size_t>(m), 0.0) : std::move(d);
        a.data = data.data();
        a.dtype = FP_F64;
        a.ndim = static_cast<int>(shape.size());
        a.shape = shape.data();
        a.strides = strides.data();
        a.device = FP_DEVICE_CPU;
    }
    Arr(const Arr &o) : Arr(o.shape, o.data) {}
};

static fp_options opts(int axis = -1, int threads = 0, int keep = 0) {
    fp_options o;
    fp_options_init(&o);
    o.axis = axis;
    o.threads = threads;
    o.keep_structure = keep;
    return o;
}

static Vec prox(fp_op *op, double gamma, const Vec &x) {
    Arr xa({static_cast<int64_t>(x.size())}, x), za({static_cast<int64_t>(x.size())});
    CHECK_OK(fp_prox(op, gamma, &xa.a, &za.a, nullptr, nullptr, nullptr));
    return za.data;
}

static double maxabs(const Vec &v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

// ---------------------------------------------------------------- 1D TV
// KKT for prox of gamma sum w_e |z_{e+1} - z_e|: u_e = sum_{i<=e}(x_i - z_i)
// satisfies |u_e| <= gamma w_e, u_{n-1} = 0, and u_e = -+gamma w_e at jumps.
static bool tv_kkt(const Vec &x, const Vec &z, double gamma, const Vec &w) {
    const size_t n = x.size();
    const double tol = 1e-9 * (1.0 + maxabs(x)) * static_cast<double>(n);
    double u = 0;
    for (size_t e = 0; e + 1 < n; ++e) {
        u += x[e] - z[e];
        const double lam = gamma * (w.size() == 1 ? w[0] : w[e]);
        if (std::fabs(u) > lam + tol) return false;
        const double jump = z[e + 1] - z[e];
        if (jump > tol && std::fabs(u + lam) > tol) return false;
        if (jump < -tol && std::fabs(u - lam) > tol) return false;
    }
    u += x[n - 1] - z[n - 1];
    return std::fabs(u) <= tol;
}

static Vec tv_signal(int kind, int64_t n) {
    Vec x(static_cast<size_t>(n));
    switch (kind) {
        case 0: return randn(n);
        case 1: {  // piecewise constant plus noise
            double level = 0;
            for (int64_t i = 0; i < n; ++i) {
                if (unif(0, 1) < 0.05) level = unif(-5, 5);
                x[i] = level + 0.3 * randn(1)[0];
            }
            return x;
        }
        case 2:  // alternating, heavy backtracking for small gamma
            for (int64_t i = 0; i < n; ++i) x[i] = (i % 2 ? 1.0 : -1.0) * (1.0 + 1e-3 * i);
            return x;
        case 3:  // ramp
            for (int64_t i = 0; i < n; ++i) x[i] = 0.01 * static_cast<double>(i);
            return x;
        default:  // random walk
            x[0] = 0;
            for (int64_t i = 1; i < n; ++i) x[i] = x[i - 1] + randn(1)[0];
            return x;
    }
}

static void test_tv1d() {
    fp_op *auto_op, *condat, *taut;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &auto_op));
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_CONDAT, &condat));
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_TAUT_STRING, &taut));
    for (int kind = 0; kind < 5; ++kind)
        for (int64_t n : {1, 2, 3, 5, 17, 100, 1000, 5000})
            for (double gamma : {1e-3, 0.1, 0.7, 3.0, 50.0}) {
                const Vec x = tv_signal(kind, n);
                const Vec za = prox(auto_op, gamma, x), zc = prox(condat, gamma, x),
                          zt = prox(taut, gamma, x);
                CHECK(tv_kkt(x, zc, gamma, {1.0}));
                CHECK(tv_kkt(x, zt, gamma, {1.0}));
                double d = 0;
                for (size_t i = 0; i < x.size(); ++i) d = std::max(d, std::fabs(zc[i] - zt[i]));
                CHECK(d <= 1e-9 * (1 + maxabs(x)));
                CHECK(za == zc || tv_kkt(x, za, gamma, {1.0}));
            }
    // Weighted, with zero weights splitting the row.
    for (int trial = 0; trial < 200; ++trial) {
        const int64_t n = 2 + static_cast<int64_t>(unif(0, 300));
        Vec w(static_cast<size_t>(n - 1));
        for (auto &v : w) v = unif(0, 1) < 0.1 ? 0.0 : unif(0.0, 2.0);
        Arr wa({n - 1}, w);
        fp_op *op;
        CHECK_OK(fp_op_tv1d(&wa.a, FP_TV1D_AUTO, &op));
        const Vec x = tv_signal(trial % 5, n);
        const double gamma = unif(0.01, 3.0);
        CHECK(tv_kkt(x, prox(op, gamma, x), gamma, w));
        fp_op_free(op);
    }
    // gamma = 0 returns the input.
    const Vec x = randn(50);
    CHECK(prox(auto_op, 0.0, x) == x);
    fp_op_free(auto_op);
    fp_op_free(condat);
    fp_op_free(taut);
}

// ---------------------------------------------------------------- projections
static void test_projections() {
    for (int trial = 0; trial < 300; ++trial) {
        const int64_t n = 1 + static_cast<int64_t>(unif(0, 400));
        const Vec x = randn(n, unif(0.1, 10));
        const double r = unif(0.01, 5);
        fp_op *op;

        // Simplex: z = max(x - tau, 0), sum z = r.
        CHECK_OK(fp_op_simplex(r, &op));
        Vec z = prox(op, 1.0, x);
        double s = 0, tau = NAN;
        for (int64_t i = 0; i < n; ++i) {
            CHECK(z[i] >= 0);
            s += z[i];
            if (z[i] > 0) tau = x[i] - z[i];
        }
        CHECK(std::fabs(s - r) <= 1e-12 * (r + maxabs(x)) * n);
        for (int64_t i = 0; i < n; ++i)
            CHECK(z[i] > 0 ? std::fabs(x[i] - z[i] - tau) <= 1e-12 * (1 + maxabs(x))
                           : x[i] <= tau + 1e-12 * (1 + maxabs(x)));
        fp_op_free(op);

        // Weighted l1 ball.
        Vec w(static_cast<size_t>(n));
        for (auto &v : w) v = unif(0.2, 3);
        Arr wa({n}, w);
        CHECK_OK(fp_op_l1_ball(r, trial % 2 ? &wa.a : nullptr, &op));
        z = prox(op, 1.0, x);
        double nrm = 0, nx = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double wi = trial % 2 ? w[i] : 1.0;
            nrm += wi * std::fabs(z[i]);
            nx += wi * std::fabs(x[i]);
        }
        if (nx <= r) {
            CHECK(z == x);
        } else {
            CHECK(std::fabs(nrm - r) <= 1e-12 * (r + nx));
            // x - z = tau * w * sign(z) on the support, |x_i| <= tau w_i off it.
            double t = NAN;
            for (int64_t i = 0; i < n; ++i)
                if (z[i] != 0) t = std::fabs(x[i] - z[i]) / (trial % 2 ? w[i] : 1.0);
            for (int64_t i = 0; i < n; ++i) {
                const double wi = trial % 2 ? w[i] : 1.0;
                if (z[i] != 0) {
                    CHECK(std::fabs(std::fabs(x[i] - z[i]) - t * wi) <= 1e-10 * (1 + maxabs(x)));
                    CHECK(z[i] * x[i] > 0);
                } else {
                    CHECK(std::fabs(x[i]) <= t * wi + 1e-10 * (1 + maxabs(x)));
                }
            }
        }
        fp_op_free(op);

        // Isotonic: monotone, and equal to a reference PAVA by repeated pooling.
        CHECK_OK(fp_op_isotonic(nullptr, 1, &op));
        z = prox(op, 1.0, x);
        std::vector<std::pair<double, double>> blocks;  // (sum, count)
        for (double v : x) {
            blocks.push_back({v, 1});
            while (blocks.size() > 1) {
                auto &a = blocks[blocks.size() - 2], &b = blocks.back();
                if (a.first / a.second <= b.first / b.second) break;
                a.first += b.first;
                a.second += b.second;
                blocks.pop_back();
            }
        }
        size_t k = 0;
        for (auto &b : blocks)
            for (int c = 0; c < b.second; ++c, ++k)
                CHECK(std::fabs(z[k] - b.first / b.second) <= 1e-12 * (1 + maxabs(x)));
        for (int64_t i = 1; i < n; ++i) CHECK(z[i] >= z[i - 1]);
        fp_op_free(op);
    }
    // Radius zero gives zero.
    fp_op *op;
    CHECK_OK(fp_op_simplex(0.0, &op));
    CHECK(maxabs(prox(op, 1.0, randn(10))) == 0.0);
    fp_op_free(op);
    CHECK_OK(fp_op_l1_ball(0.0, nullptr, &op));
    CHECK(maxabs(prox(op, 1.0, randn(10))) == 0.0);
    fp_op_free(op);
}

// ---------------------------------------------------------------- derivatives
struct OpCase {
    std::string name;
    std::function<fp_status(fp_op **)> make;
    double gamma;
    bool smooth_away_from_kinks = true;
};

static std::vector<OpCase> derivative_cases(int64_t n) {
    static std::deque<Arr> keep;  // stable addresses: parameter arrays outlive the constructors
    Vec w(static_cast<size_t>(n));
    for (auto &v : w) v = unif(0.5, 1.5);
    keep.emplace_back(std::vector<int64_t>{n}, w);
    fp_array *wn = &keep.back().a;
    Vec we(static_cast<size_t>(n - 1));
    for (auto &v : we) v = unif(0.2, 1.5);
    keep.emplace_back(std::vector<int64_t>{n - 1}, we);
    fp_array *wm = &keep.back().a;
    Vec lo(static_cast<size_t>(n)), hi(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        lo[i] = unif(-1, 0);
        hi[i] = unif(0, 1);
    }
    keep.emplace_back(std::vector<int64_t>{n}, lo);
    fp_array *la = &keep.back().a;
    keep.emplace_back(std::vector<int64_t>{n}, hi);
    fp_array *ha = &keep.back().a;
    static std::vector<int64_t> groups;
    groups.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) groups[i] = (i * 7) % 3;

    return {
        {"soft", [](fp_op **o) { return fp_op_soft(0.4, nullptr, o); }, 0.8},
        {"soft_w", [=](fp_op **o) { return fp_op_soft(0.4, wn, o); }, 0.8},
        {"elastic_net", [](fp_op **o) { return fp_op_elastic_net(0.3, 0.7, o); }, 0.9},
        {"huber", [](fp_op **o) { return fp_op_huber(0.6, 0.5, o); }, 0.7},
        {"box", [=](fp_op **o) { return fp_op_box(0, 0, la, ha, o); }, 1.0},
        {"nonneg", [](fp_op **o) { return fp_op_nonneg(o); }, 1.0},
        {"l2", [](fp_op **o) { return fp_op_group_l2(0.5, 0, nullptr, 1, nullptr, o); }, 0.9},
        {"group_l2",
         [=](fp_op **o) { return fp_op_group_l2(0.6, n, groups.data(), 3, nullptr, o); }, 0.9},
        {"l2_ball", [](fp_op **o) { return fp_op_l2_ball(0.7, o); }, 1.0},
        {"simplex", [](fp_op **o) { return fp_op_simplex(1.3, o); }, 1.0},
        {"l1_ball", [](fp_op **o) { return fp_op_l1_ball(1.1, nullptr, o); }, 1.0},
        {"l1_ball_w", [=](fp_op **o) { return fp_op_l1_ball(1.1, wn, o); }, 1.0},
        {"linf", [](fp_op **o) { return fp_op_linf(0.9, o); }, 1.0},
        {"isotonic", [](fp_op **o) { return fp_op_isotonic(nullptr, 1, o); }, 1.0},
        {"isotonic_w", [=](fp_op **o) { return fp_op_isotonic(wn, 0, o); }, 1.0},
        {"tv1d", [](fp_op **o) { return fp_op_tv1d(nullptr, FP_TV1D_AUTO, o); }, 0.6},
        {"tv1d_w", [=](fp_op **o) { return fp_op_tv1d(wm, FP_TV1D_AUTO, o); }, 0.6},
    };
}

static double dot(const Vec &a, const Vec &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Checks adjointness of (dx, dgamma, dtheta) -> dz, and the jvp against
// central differences, for one operator on a batch of rows.
static void check_derivatives(const OpCase &c, bool conj) {
    const int64_t rows = 3, n = 12;
    fp_op *op;
    CHECK_OK(c.make(&op));
    const int64_t P = fp_op_nparams(op);
    Arr x({rows, n}, randn(rows * n, 1.5)), z({rows, n});
    fp_options o = opts(-1, 0, 1);
    fp_workspace *ws;
    CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
    auto run = conj ? fp_prox_conj : fp_prox;
    CHECK_OK(run(op, c.gamma, &x.a, &z.a, &o, ws, nullptr));

    Arr dx({rows, n}, randn(rows * n)), dz({rows, n}), g({rows, n}, randn(rows * n)),
        gx({rows, n});
    Arr dp({P > 0 ? P : 1}, randn(P > 0 ? P : 1)), gp({P > 0 ? P : 1});
    const double dgamma = 0.37;
    CHECK_OK(fp_jvp(op, ws, &dx.a, dgamma, P ? &dp.a : nullptr, &dz.a));
    double gg = 0;
    CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, P ? &gp.a : nullptr));
    const double lhs = dot(g.data, dz.data);
    double rhs = dot(gx.data, dx.data) + gg * dgamma;
    if (P) rhs += dot(gp.data, dp.data);
    const bool adj = std::fabs(lhs - rhs) <= 1e-10 * (1 + std::fabs(lhs));
    CHECK(adj);
    if (!adj) std::fprintf(stderr, "  adjoint %s conj=%d: %.15g vs %.15g\n", c.name.c_str(), conj, lhs, rhs);

    // Central differences in x and gamma (theta through a rebuilt operator is
    // covered on the Python side).
    const double h = 1e-6;
    Arr xp(x), xm(x), zp({rows, n}), zm({rows, n});
    for (size_t i = 0; i < x.data.size(); ++i) {
        xp.data[i] += h * dx.data[i];
        xm.data[i] -= h * dx.data[i];
    }
    CHECK_OK(run(op, c.gamma + h * dgamma, &xp.a, &zp.a, nullptr, nullptr, nullptr));
    CHECK_OK(run(op, c.gamma - h * dgamma, &xm.a, &zm.a, nullptr, nullptr, nullptr));
    CHECK_OK(fp_jvp(op, ws, &dx.a, dgamma, nullptr, &dz.a));
    double err = 0, scale = 1;
    for (size_t i = 0; i < x.data.size(); ++i) {
        const double fd = (zp.data[i] - zm.data[i]) / (2 * h);
        err = std::max(err, std::fabs(fd - dz.data[i]));
        scale = std::max(scale, std::fabs(fd));
    }
    const bool fd_ok = err <= 1e-5 * scale;
    CHECK(fd_ok);
    if (!fd_ok) std::fprintf(stderr, "  fd %s conj=%d: err %.3g\n", c.name.c_str(), conj, err);
    fp_workspace_free(ws);
    fp_op_free(op);
}

static void test_derivatives() {
    for (int rep = 0; rep < 20; ++rep)
        for (const auto &c : derivative_cases(12)) {
            check_derivatives(c, false);
            check_derivatives(c, true);
        }
}

// ---------------------------------------------------------------- batching
static void test_batching() {
    for (const auto &c : derivative_cases(64)) {
        fp_op *op;
        CHECK_OK(c.make(&op));
        const int64_t rows = 300, n = 64;
        Arr x({rows, n}, randn(rows * n, 2.0));
        std::vector<Vec> outs;
        for (int t : {1, 2, 8}) {
            Arr z({rows, n});
            fp_options o = opts(-1, t);
            CHECK_OK(fp_prox(op, c.gamma, &x.a, &z.a, &o, nullptr, nullptr));
            outs.push_back(z.data);
        }
        CHECK(std::memcmp(outs[0].data(), outs[1].data(), outs[0].size() * 8) == 0);
        CHECK(std::memcmp(outs[0].data(), outs[2].data(), outs[0].size() * 8) == 0);

        // Same rows stored transposed and processed along axis 0.
        Arr xt({n, rows}), zt({n, rows});
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < n; ++j) xt.data[j * rows + i] = x.data[i * n + j];
        fp_options o0 = opts(0);
        CHECK_OK(fp_prox(op, c.gamma, &xt.a, &zt.a, &o0, nullptr, nullptr));
        bool same = true;
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < n; ++j) same = same && zt.data[j * rows + i] == outs[0][i * n + j];
        CHECK(same);
        if (!same) std::fprintf(stderr, "  transposed mismatch for %s\n", c.name.c_str());

        // In place.
        Arr y(x);
        CHECK_OK(fp_prox(op, c.gamma, &y.a, &y.a, nullptr, nullptr, nullptr));
        CHECK(y.data == outs[0]);

        // Parameter gradients are bit-identical across thread counts too.
        std::vector<Vec> grads;
        for (int t : {1, 3, 8}) {
            fp_options o = opts(-1, t, 1);
            fp_workspace *ws;
            Arr z({rows, n}), g({rows, n}), gx({rows, n});
            for (size_t i = 0; i < g.data.size(); ++i) g.data[i] = std::sin(0.1 * i);
            CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
            CHECK_OK(fp_prox(op, c.gamma, &x.a, &z.a, &o, ws, nullptr));
            const int64_t P = fp_op_nparams(op);
            Arr gp({std::max<int64_t>(P, 1)});
            double gg = 0;
            CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, P ? &gp.a : nullptr));
            gp.data.push_back(gg);
            grads.push_back(gp.data);
            fp_workspace_free(ws);
        }
        CHECK(grads[0] == grads[1] && grads[0] == grads[2]);
        fp_op_free(op);
    }
}

// ---------------------------------------------------------------- misc
static void test_no_allocation() {
    fp_op *op;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &op));
    Arr x({64, 1000}, randn(64000)), z({64, 1000}), g({64, 1000}, randn(64000)), gx({64, 1000});
    fp_options o = opts(-1, 4, 1);
    fp_workspace *ws;
    CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
    Arr gp({1});
    double gg;
    CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, nullptr));  // warm up the thread pool
    const long before = g_allocs.load();
    for (int k = 0; k < 5; ++k) {
        CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, nullptr));
        CHECK_OK(fp_jvp(op, ws, &x.a, 0.1, nullptr, &gx.a));
        CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, &gp.a));
    }
    CHECK(g_allocs.load() == before);
    fp_workspace_free(ws);
    fp_op_free(op);
}

static void test_errors_and_special_values() {
    fp_op *op;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &op));
    Vec x = randn(20);
    x[5] = INFINITY;
    Arr xa({20}, x), za({20});
    CHECK(fp_prox(op, 1.0, &xa.a, &za.a, nullptr, nullptr, nullptr) == FP_EINVAL);
    xa.data[5] = NAN;
    CHECK_OK(fp_prox(op, 1.0, &xa.a, &za.a, nullptr, nullptr, nullptr));
    CHECK(std::isnan(za.data[0]));
    // jvp without saved structure.
    fp_workspace *ws;
    fp_options o = opts();
    CHECK_OK(fp_workspace_create(op, &xa.a, &o, &ws));
    CHECK(fp_jvp(op, ws, &xa.a, 0.0, nullptr, &za.a) == FP_EINVAL);
    fp_workspace_free(ws);
    // Shape mismatch of edge weights.
    Arr w({7}, randn(7));
    for (auto &v : w.data) v = std::fabs(v);
    fp_op *bad;
    CHECK_OK(fp_op_tv1d(&w.a, FP_TV1D_AUTO, &bad));
    CHECK(fp_prox(bad, 1.0, &xa.a, &za.a, nullptr, nullptr, nullptr) == FP_ESHAPE);
    fp_op_free(bad);
    // Empty and length-one inputs.
    Arr e({0}), e2({0});
    CHECK_OK(fp_prox(op, 1.0, &e.a, &e2.a, nullptr, nullptr, nullptr));
    Arr one({1}, {3.5}), one2({1});
    CHECK_OK(fp_prox(op, 1.0, &one.a, &one2.a, nullptr, nullptr, nullptr));
    CHECK(one2.data[0] == 3.5);
    fp_op_free(op);
    // Soft thresholding propagates NaN entrywise.
    CHECK_OK(fp_op_soft(1.0, nullptr, &op));
    Vec s = prox(op, 1.0, {NAN, 3.0, INFINITY});
    CHECK(std::isnan(s[0]) && s[1] == 2.0 && std::isinf(s[2]));
    fp_op_free(op);
    // A non-CPU device is rejected.
    CHECK_OK(fp_op_soft(1.0, nullptr, &op));
    Arr d({3}, {1, 2, 3});
    d.a.device = static_cast<fp_device>(1);
    CHECK(fp_prox(op, 1.0, &d.a, &d.a, nullptr, nullptr, nullptr) == FP_EINVAL);
    fp_op_free(op);
}

static void test_float32() {
    fp_op *op;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &op));
    const int64_t n = 2000;
    Vec x = tv_signal(1, n);
    std::vector<float> xf(x.begin(), x.end()), zf(n);
    int64_t shape[1] = {n};
    fp_array a{xf.data(), FP_F32, 1, shape, nullptr, FP_DEVICE_CPU};
    fp_array b{zf.data(), FP_F32, 1, shape, nullptr, FP_DEVICE_CPU};
    CHECK_OK(fp_prox(op, 0.5, &a, &b, nullptr, nullptr, nullptr));
    const Vec zd = prox(op, 0.5, Vec(xf.begin(), xf.end()));
    double err = 0;
    for (int64_t i = 0; i < n; ++i) err = std::max(err, std::fabs(zf[i] - zd[i]));
    CHECK(err <= 1e-5 * (1 + maxabs(zd)));
    fp_op_free(op);
}

int main() {
    test_tv1d();
    test_projections();
    test_derivatives();
    test_batching();
    test_no_allocation();
    test_errors_and_special_values();
    test_float32();
    std::printf("%d checks, %d failures (openmp: %d)\n", g_checks, g_fail, fp_has_openmp());
    return g_fail != 0;
}
