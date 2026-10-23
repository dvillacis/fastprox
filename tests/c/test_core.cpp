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
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <dirent.h>
#endif
#if defined(__SANITIZE_THREAD__)
#define FP_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FP_TSAN 1
#endif
#endif

// ---------------------------------------------------------------- allocation counter
// Not under ThreadSanitizer, whose runtime defines these operators itself on
// Linux; the no-allocation test then checks nothing.
static std::atomic<long> g_allocs{0};
static std::atomic<long long> g_alloc_bytes{0};  // bytes requested (audit C2)
#ifndef FP_TSAN
void *operator new(std::size_t n) {
    ++g_allocs;
    g_alloc_bytes += static_cast<long long>(n);
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
// libstdc++'s std::get_temporary_buffer (std::stable_sort) uses this form.
void *operator new(std::size_t n, const std::nothrow_t &) noexcept {
    ++g_allocs;
    g_alloc_bytes += static_cast<long long>(n);
    return std::malloc(n ? n : 1);
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
#endif

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
    int64_t zero_entry = -1;  // set this entry of every row to exactly 0
};

static std::vector<OpCase> derivative_cases(int64_t n) {
    static std::deque<Arr> keep;  // stable addresses: parameter arrays outlive the constructors
    Vec w(static_cast<size_t>(n));
    for (auto &v : w) v = unif(0.5, 1.5);
    keep.emplace_back(std::vector<int64_t>{n}, w);
    fp_array *wn = &keep.back().a;
    Vec w0(w);
    w0[0] = 0.0;  // a zero weight: the identity at entry 0, which zero_entry puts at x = 0
    keep.emplace_back(std::vector<int64_t>{n}, w0);
    fp_array *wz = &keep.back().a;
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
    Vec av(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) av[i] = (i % 3 == 1 ? -1.0 : 1.0) * unif(0.5, 1.5);
    av[4] = 0.0;  // an entry outside the constraint
    keep.emplace_back(std::vector<int64_t>{n}, av);
    fp_array *aa = &keep.back().a;
    // A tree over the first 10 entries: leaves, a subtree, a singleton, the root.
    static const int64_t tree_ptr[] = {0, 2, 5, 10, 11, 15, 25};
    static const int64_t tree_idx[] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    static std::vector<int64_t> sg_ptr, sg_idx;  // the partition (i * 7) % 3 as a tree
    sg_ptr.assign({0});
    sg_idx.clear();
    for (int64_t g = 0; g < 3; ++g) {
        for (int64_t i = 0; i < n; ++i)
            if ((i * 7) % 3 == g) sg_idx.push_back(i);
        sg_ptr.push_back(static_cast<int64_t>(sg_idx.size()));
    }
    static std::vector<int64_t> groups;
    groups.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) groups[i] = (i * 7) % 3;

    return {
        {"soft", [](fp_op **o) { return fp_op_soft(0.4, nullptr, o); }, 0.8},
        {"soft_w", [=](fp_op **o) { return fp_op_soft(0.4, wn, o); }, 0.8},
        {"soft_w0_at_0", [=](fp_op **o) { return fp_op_soft(0.4, wz, o); }, 0.8, true, 0},
        {"soft_lam0_at_0", [](fp_op **o) { return fp_op_soft(0.0, nullptr, o); }, 0.8, true, 0},
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
        {"max", [](fp_op **o) { return fp_op_max(0.7, o); }, 0.9},
        {"sum_largest", [](fp_op **o) { return fp_op_sum_largest(0.6, 3, 0, o); }, 0.9},
        {"sum_largest_abs", [](fp_op **o) { return fp_op_sum_largest(2.5, 3, 1, o); }, 0.9},  // gamma lam > 1
        {"hyperplane",
         [=](fp_op **o) { return fp_op_hyperplane(aa, 0.4, -INFINITY, INFINITY, nullptr, nullptr, 0, o); },
         1.0},
        {"hyperplane_box", [=](fp_op **o) { return fp_op_hyperplane(aa, 0.2, 0, 0, la, ha, 0, o); }, 1.0},
        {"capped_simplex", [](fp_op **o) { return fp_op_hyperplane(nullptr, 3.0, 0, 1, nullptr, nullptr, 0, o); },
         1.0},
        {"halfspace_box", [=](fp_op **o) { return fp_op_hyperplane(aa, 0.2, 0, 0, la, ha, 1, o); }, 1.0},
        {"log_barrier", [](fp_op **o) { return fp_op_log_barrier(0.6, o); }, 0.8},
        {"l1_box", [=](fp_op **o) { return fp_op_l1_box(0.4, wn, 0, 0, la, ha, o); }, 0.8},
        {"l1_box_shifted", [](fp_op **o) { return fp_op_l1_box(0.4, nullptr, 0.2, 0.9, nullptr, nullptr, o); },
         0.8},
        {"soc", [](fp_op **o) { return fp_op_soc(o); }, 1.0},
        {"group_huber",
         [=](fp_op **o) { return fp_op_group_huber(0.6, 2.0, n, groups.data(), 3, nullptr, o); }, 0.9},
        {"huber_l2", [](fp_op **o) { return fp_op_group_huber(0.6, 2.0, 0, nullptr, 1, nullptr, o); }, 0.9},
        {"group_scad",
         [=](fp_op **o) { return fp_op_group_scad(1.0, 3.7, n, groups.data(), 3, nullptr, o); }, 0.8},
        {"group_mcp",
         [=](fp_op **o) { return fp_op_group_mcp(1.0, 3.0, n, groups.data(), 3, nullptr, o); }, 0.8},
        {"tree_l2", [](fp_op **o) { return fp_op_tree_l2(0.2, 6, tree_ptr, tree_idx, nullptr, o); }, 0.9},
        {"sparse_group",
         [](fp_op **o) { return fp_op_tree_l2(0.3, 3, sg_ptr.data(), sg_idx.data(), nullptr, o); }, 0.9},
        {"tv1d_ball", [](fp_op **o) { return fp_op_tv1d_ball(2.0, nullptr, o); }, 1.0},
        {"tv1d_ball_w", [=](fp_op **o) { return fp_op_tv1d_ball(1.5, wm, o); }, 1.0},
        {"tv1d_l2", [](fp_op **o) { return fp_op_tv1d_l2(nullptr, o); }, 0.6},
        {"tv1d_l2_w", [=](fp_op **o) { return fp_op_tv1d_l2(wm, o); }, 0.6},
        {"isotonic", [](fp_op **o) { return fp_op_isotonic(nullptr, 1, o); }, 1.0},
        {"isotonic_w", [=](fp_op **o) { return fp_op_isotonic(wn, 0, o); }, 1.0},
        {"tv1d", [](fp_op **o) { return fp_op_tv1d(nullptr, FP_TV1D_AUTO, o); }, 0.6},
        {"tv1d_w", [=](fp_op **o) { return fp_op_tv1d(wm, FP_TV1D_AUTO, o); }, 0.6},
        // Nonconvex maps (the conjugate's step 1/gamma keeps SCAD's a > 1 + 1/gamma
        // and MCP's b > 1/gamma).
        {"l0_w", [=](fp_op **o) { return fp_op_l0(0.3, wn, o); }, 0.8},
        {"lq_half", [=](fp_op **o) { return fp_op_lq(0.4, 0.5, wn, o); }, 0.8},
        {"lq_two_thirds", [](fp_op **o) { return fp_op_lq(0.4, 2.0 / 3.0, nullptr, o); }, 0.8},
        {"lq_0.3", [](fp_op **o) { return fp_op_lq(0.4, 0.3, nullptr, o); }, 0.8},
        {"scad", [](fp_op **o) { return fp_op_scad(0.5, 3.7, nullptr, o); }, 0.8},
        {"scad_w", [=](fp_op **o) { return fp_op_scad(0.5, 3.7, wn, o); }, 0.8},
        {"mcp", [](fp_op **o) { return fp_op_mcp(0.5, 3.0, nullptr, o); }, 0.8},
        {"mcp_w", [=](fp_op **o) { return fp_op_mcp(0.5, 3.0, wn, o); }, 0.8},
        {"log", [](fp_op **o) { return fp_op_log_penalty(0.8, 0.5, o); }, 0.8},
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
    if (c.zero_entry >= 0)
        for (int64_t r = 0; r < rows; ++r) x.data[r * n + c.zero_entry] = 0.0;
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

// ---------------------------------------------------------------- anisotropic TV
// Independent certificate for z = prox of gamma sum_a w_a TV_a at x, from the
// dual q of shape (k, *shape): every q_a must be feasible (its running sums
// along each fiber stay within gamma w_a and end at 0), and then
// P(z) - D(q) with D(q) = <x, s> - |s|^2 / 2, s = sum_a q_a, bounds the gap.
struct AnisoCheck {
    double gap = 0, feas = 0, dn2 = 0;
};
static AnisoCheck aniso_certificate(int k, const std::vector<int64_t> &shape, double gamma, const Vec &w,
                                    const Vec &x, const Vec &z, const Vec &q) {
    const int nd = static_cast<int>(shape.size());
    std::vector<int64_t> L(shape.end() - k, shape.end()), S(static_cast<size_t>(k));
    S[k - 1] = 1;
    for (int a = k - 2; a >= 0; --a) S[a] = S[a + 1] * L[a + 1];
    const int64_t Ni = S[0] * L[0], N = static_cast<int64_t>(x.size()), B = N / Ni;
    (void)nd;
    AnisoCheck r;
    for (int64_t b = 0; b < B; ++b) {
        const double *xb = x.data() + b * Ni, *zb = z.data() + b * Ni;
        double P = 0, D = 0;
        for (int a = 0; a < k; ++a) {
            const double lam = gamma * w[w.size() == 1 ? 0 : a];
            const double *qb = q.data() + a * N + b * Ni;
            for (int64_t st = 0; st < Ni; ++st) {
                if ((st / S[a]) % L[a] != 0) continue;
                double p = 0;
                for (int64_t i = 0; i < L[a]; ++i) {
                    p += qb[st + i * S[a]];
                    r.feas = std::max(r.feas, i + 1 < L[a] ? std::fabs(p) - lam : std::fabs(p));
                }
            }
            for (int64_t i = 0; i < Ni; ++i)
                if ((i / S[a]) % L[a] + 1 < L[a]) P += lam * std::fabs(zb[i + S[a]] - zb[i]);
        }
        for (int64_t i = 0; i < Ni; ++i) {
            double s = 0;
            for (int a = 0; a < k; ++a) s += q[a * N + b * Ni + i];
            P += 0.5 * (zb[i] - xb[i]) * (zb[i] - xb[i]);
            D += xb[i] * s - 0.5 * s * s;
            r.dn2 += xb[i] * xb[i];
        }
        r.gap += P - D;
    }
    return r;
}

static Vec image_signal(int64_t n) {
    Vec x = randn(n, 0.3);
    for (int64_t i = 0; i < n; ++i)
        if ((i / 5) % 3 == 0) x[i] += 1.0;
    return x;
}

static fp_op *make_aniso(int k, Vec w) {
    fp_op *op = nullptr;
    Arr wa({static_cast<int64_t>(w.size())}, w);
    CHECK_OK(fp_op_tv_aniso(k, &wa.a, &op));
    return op;
}

static fp_op *make_spatial(bool iso, int k, const std::vector<int64_t> &img, Vec w, const Vec &sw);

static void test_tv_aniso() {
    struct Case {
        int k;
        std::vector<int64_t> shape;
        double gamma;
        Vec w;
    };
    const std::vector<Case> cases = {
        {2, {24, 31}, 0.5, {1.0}},           {2, {3, 17, 20}, 0.3, {1.0, 2.0}},
        {3, {9, 10, 11}, 0.4, {1.0}},        {3, {2, 6, 7, 8}, 0.3, {0.5, 1.0, 1.5}},
        {3, {6, 7, 8}, 0.4, {1.0, 0.0, 2.0}}, {2, {1, 40}, 0.4, {1.0}},
        {3, {5, 1, 9}, 0.4, {1.0}},
        // Large enough for the polish to split into slabs over threads.
        {3, {21, 12, 30}, 0.3, {0.8}},       {2, {70, 45}, 0.3, {0.8}},
    };
    for (const auto &c : cases) {
        fp_op *op = make_aniso(c.k, c.w);
        int64_t N = 1;
        for (int64_t s : c.shape) N *= s;
        Arr x(c.shape, image_signal(N));
        std::vector<int64_t> qshape = {c.k};
        qshape.insert(qshape.end(), c.shape.begin(), c.shape.end());

        std::vector<Vec> outs;
        std::vector<int> its;
        for (int t : {1, 2, 8}) {
            fp_options o = opts(-1, t);
            o.tol = 1e-10;
            o.max_iter = 100000;
            fp_workspace *ws;
            CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
            Arr z(c.shape), q(qshape);
            fp_info info;
            CHECK_OK(fp_prox(op, c.gamma, &x.a, &z.a, &o, ws, &info));
            CHECK_OK(fp_workspace_get_dual(ws, &q.a));
            const AnisoCheck r = aniso_certificate(c.k, c.shape, c.gamma, c.w, x.data, z.data, q.data);
            const double target = o.tol * 0.5 * r.dn2;
            CHECK(r.feas <= 1e-12 * (1 + maxabs(x.data)));
            CHECK(r.gap <= target * (1 + 1e-6) + 1e-13 * r.dn2);
            CHECK(r.gap >= -1e-12 * r.dn2);
            CHECK(info.gap <= target);
            outs.push_back(z.data);
            its.push_back(info.iters);
            fp_workspace_free(ws);
        }
        CHECK(outs[0] == outs[1] && outs[0] == outs[2]);  // bit-identical for any thread count
        CHECK(its[0] == its[1] && its[0] == its[2]);

        // Each image of a batch is solved as if alone; in place gives the same.
        const int64_t Ni = N / (c.shape.size() > static_cast<size_t>(c.k) ? c.shape[0] : 1);
        if (c.shape.size() > static_cast<size_t>(c.k)) {
            std::vector<int64_t> ishape(c.shape.begin() + 1, c.shape.end());
            fp_options o = opts();
            o.tol = 1e-10;
            o.max_iter = 100000;
            Arr zb(c.shape);
            CHECK_OK(fp_prox(op, c.gamma, &x.a, &zb.a, &o, nullptr, nullptr));
            for (int64_t b = 0; b < c.shape[0]; ++b) {
                Arr xi(ishape, Vec(x.data.begin() + b * Ni, x.data.begin() + (b + 1) * Ni)), zi(ishape);
                CHECK_OK(fp_prox(op, c.gamma, &xi.a, &zi.a, &o, nullptr, nullptr));
                CHECK(std::equal(zi.data.begin(), zi.data.end(), zb.data.begin() + b * Ni));
            }
            Arr y(x);
            CHECK_OK(fp_prox(op, c.gamma, &y.a, &y.a, &o, nullptr, nullptr));
            CHECK(y.data == zb.data);
        }

        // Derivatives: finite differences in (x, gamma, w) and adjointness, for
        // the prox and its conjugate.
        for (bool conj : {false, true}) {
            const double step = conj ? 1.0 / c.gamma : c.gamma;
            fp_options o = opts(-1, 0, 1);
            o.tol = 1e-14;
            o.max_iter = 200000;
            fp_workspace *ws;
            CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
            Arr z(c.shape);
            CHECK_OK(conj ? fp_prox_conj(op, step, &x.a, &z.a, &o, ws, nullptr)
                          : fp_prox(op, step, &x.a, &z.a, &o, ws, nullptr));
            Arr dx(c.shape, randn(N)), jv(c.shape), g(c.shape, randn(N)), gx(c.shape);
            Vec wv(static_cast<size_t>(c.k));
            for (int a = 0; a < c.k; ++a) wv[a] = c.w[c.w.size() == 1 ? 0 : a];
            Arr dw({c.k}, randn(c.k)), gw({c.k});
            for (int a = 0; a < c.k; ++a)
                if (wv[a] == 0.0) dw.data[a] = 0.0;  // a zero weight cannot step below 0
            const double dg = 0.3;
            CHECK_OK(fp_jvp(op, ws, &dx.a, dg, &dw.a, &jv.a));
            double gg = 0;
            CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, &gw.a));
            double lhs = 0, rhs = gg * dg;
            for (int64_t i = 0; i < N; ++i) {
                lhs += g.data[i] * jv.data[i];
                rhs += gx.data[i] * dx.data[i];
            }
            for (int a = 0; a < c.k; ++a) rhs += gw.data[a] * dw.data[a];
            CHECK(std::fabs(lhs - rhs) <= 1e-10 * (1 + std::fabs(lhs)));

            const double h = 1e-6;
            Vec zs[2];
            for (int sgn = 0; sgn < 2; ++sgn) {
                const double e = sgn ? -h : h;
                Vec wp(wv), xp(x.data);
                for (int a = 0; a < c.k; ++a) wp[a] += e * dw.data[a];
                for (int64_t i = 0; i < N; ++i) xp[i] += e * dx.data[i];
                fp_op *op2 = make_aniso(c.k, wp);
                Arr xa(c.shape, xp), za(c.shape);
                fp_options o2 = o;
                o2.keep_structure = 0;
                CHECK_OK(conj ? fp_prox_conj(op2, step + e * dg, &xa.a, &za.a, &o2, nullptr, nullptr)
                              : fp_prox(op2, step + e * dg, &xa.a, &za.a, &o2, nullptr, nullptr));
                zs[sgn] = za.data;
                fp_op_free(op2);
            }
            double err = 0, scale = 1;
            for (int64_t i = 0; i < N; ++i) {
                const double fd = (zs[0][i] - zs[1][i]) / (2 * h);
                err = std::max(err, std::fabs(fd - jv.data[i]));
                scale = std::max(scale, std::fabs(fd));
            }
            CHECK(err <= 1e-6 * scale);
            if (err > 1e-6 * scale) std::fprintf(stderr, "  tv_aniso k=%d fd error %.2e\n", c.k, err);
            fp_workspace_free(ws);
        }
        fp_op_free(op);
    }

    // One active axis is exactly 1D TV along it; gamma = 0 and zero weights are
    // the identity.
    {
        fp_op *op = make_aniso(2, {0.0, 1.0}), *tv;
        CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv));
        Arr x({7, 30}, randn(210)), z({7, 30}), z1({7, 30});
        fp_info info;
        CHECK_OK(fp_prox(op, 0.6, &x.a, &z.a, nullptr, nullptr, &info));
        CHECK_OK(fp_prox(tv, 0.6, &x.a, &z1.a, nullptr, nullptr, nullptr));
        CHECK(z.data == z1.data && info.gap == 0.0 && info.iters == 0);
        CHECK_OK(fp_prox(op, 0.0, &x.a, &z.a, nullptr, nullptr, nullptr));
        CHECK(z.data == x.data);
        fp_op_free(op);
        fp_op_free(tv);
        op = make_aniso(3, {0.0});
        Arr x3({4, 5, 6}, randn(120)), z3({4, 5, 6});
        CHECK_OK(fp_prox(op, 1.0, &x3.a, &z3.a, nullptr, nullptr, nullptr));
        CHECK(z3.data == x3.data);
        fp_op_free(op);
    }

    // A conjugate prox at a tiny sigma scales x by 1/sigma, so |x|^2 and the
    // gap target overflow to inf (found by fuzzing): z is still computed, and
    // the result, the projection of x onto the dual set, sums to 0 per image
    // (the 2D method returned x, from a z it never wrote).
    for (int wtd = 0; wtd < 2; ++wtd) {
        Vec beta(50, 0.5);
        fp_op *op = wtd ? make_spatial(false, 2, {5, 5}, {1.0, 0.7}, beta) : make_aniso(2, {1.0, 0.7});
        Vec xv = image_signal(250);
        for (double &v : xv) v += 7.0;
        Arr x({10, 5, 5}, xv), z1({10, 5, 5}), z4({10, 5, 5});
        fp_options o = opts(-1, 1);
        o.max_iter = 30;
        const fp_status s1 = fp_prox_conj(op, 1e-250, &x.a, &z1.a, &o, nullptr, nullptr);
        o.threads = 4;
        const fp_status s4 = fp_prox_conj(op, 1e-250, &x.a, &z4.a, &o, nullptr, nullptr);
        CHECK((s1 == FP_OK || s1 == FP_ENOTCONV) && s1 == s4 && z1.data == z4.data);
        for (int b = 0; b < 10; ++b) {
            double sum = 0;
            for (int i = 0; i < 25; ++i) sum += z1.data[b * 25 + i];
            CHECK(std::fabs(sum) <= 1e-9 * 25 * 15);
        }
        fp_op_free(op);
    }

    // Warm starts: the dual of a solve restarts it at 0 iterations; seeding
    // with a nearby problem's dual takes fewer iterations than a cold start.
    {
        fp_op *op = make_aniso(2, {1.0});
        Arr x({40, 50}, image_signal(2000)), z({40, 50}), q({2, 40, 50});
        fp_options o = opts();
        o.tol = 1e-10;
        o.max_iter = 100000;
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        CHECK(fp_workspace_get_dual(ws, &q.a) == FP_EINVAL);  // no call yet
        fp_info cold, warm;
        CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, &cold));
        CHECK_OK(fp_workspace_set_warm(ws, 1));
        CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, &warm));
        CHECK(warm.iters == 0 && cold.iters > 0);
        CHECK_OK(fp_workspace_set_warm(ws, 0));
        CHECK_OK(fp_workspace_get_dual(ws, &q.a));
        // Seeded from a nearby problem's dual: fewer iterations in total over
        // a few such problems (not always for each one, since restarts and
        // the polish make the count uneven).
        int cold_sum = 0, warm_sum = 0;
        for (int j = 0; j < 4; ++j) {
            Arr x2(x);
            for (auto &v : x2.data) v += 0.01 * std::sin(v * 7 + j);
            fp_info c2, w2;
            CHECK_OK(fp_prox(op, 0.5, &x2.a, &z.a, &o, ws, &c2));  // cold (warm is off, nothing seeded)
            CHECK_OK(fp_workspace_set_dual(ws, &q.a));
            CHECK_OK(fp_prox(op, 0.5, &x2.a, &z.a, &o, ws, &w2));
            cold_sum += c2.iters;
            warm_sum += w2.iters;
        }
        CHECK(warm_sum < cold_sum);
        // An infeasible seed is made feasible before use: q_0 = x - mean(x),
        // q_1 = 0 would otherwise give z = mean(x) with gap 0 at once.
        {
            Arr qb({2, 40, 50}, Vec(4000, 0.0));
            double mean = 0.0;
            for (double v : x.data) mean += v / 2000.0;
            for (int i = 0; i < 2000; ++i) qb.data[i] = x.data[i] - mean;
            fp_info ib;
            CHECK_OK(fp_workspace_set_dual(ws, &qb.a));
            CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, &ib));
            CHECK_OK(fp_workspace_get_dual(ws, &qb.a));
            const AnisoCheck r = aniso_certificate(2, {40, 50}, 0.5, {1.0}, x.data, z.data, qb.data);
            CHECK(ib.iters > 0 && r.feas <= 1e-12 && r.gap <= 1e-10 * r.dn2 / 2 * (1 + 1e-6));
        }
        Arr bad({3, 40, 50});
        CHECK(fp_workspace_set_dual(ws, &bad.a) == FP_ESHAPE);
        q.data[5] = INFINITY;
        CHECK(fp_workspace_set_dual(ws, &q.a) == FP_EINVAL);
        // max_iter reached: FP_ENOTCONV with the last iterate written.
        fp_options o1 = o;
        o1.max_iter = 1;
        o1.tol = 1e-14;
        Arr z1({40, 50}, Vec(2000, NAN));
        fp_info i1;
        CHECK(fp_prox(op, 0.5, &x.a, &z1.a, &o1, ws, &i1) == FP_ENOTCONV);
        CHECK(i1.iters == 1 && i1.gap > 0 && std::isfinite(maxabs(z1.data)));
        fp_workspace_free(ws);

        // Errors and special values.
        fp_op *bad_op = nullptr;
        CHECK(fp_op_tv_aniso(4, nullptr, &bad_op) == FP_EPARAM);
        Arr wneg({2}, {1.0, -1.0}), w3({3}, {1.0, 1.0, 1.0});
        CHECK(fp_op_tv_aniso(2, &wneg.a, &bad_op) == FP_EPARAM);
        CHECK(fp_op_tv_aniso(2, &w3.a, &bad_op) == FP_ESHAPE);
        Arr x1({50}, randn(50)), z1d({50});
        CHECK(fp_prox(op, 0.5, &x1.a, &z1d.a, nullptr, nullptr, nullptr) == FP_ESHAPE);
        fp_options oneg = o;
        oneg.tol = -1;
        CHECK(fp_prox(op, 0.5, &x.a, &z.a, &oneg, nullptr, nullptr) == FP_EPARAM);
        Arr xb({3, 10, 12}, randn(360)), zb({3, 10, 12});
        xb.data[130] = NAN;  // in image 1
        CHECK_OK(fp_prox(op, 0.5, &xb.a, &zb.a, nullptr, nullptr, nullptr));
        CHECK(std::isnan(zb.data[120]) && std::isnan(zb.data[239]));
        CHECK(!std::isnan(zb.data[0]) && !std::isnan(zb.data[240]));
        xb.data[130] = INFINITY;
        CHECK(fp_prox(op, 0.5, &xb.a, &zb.a, nullptr, nullptr, nullptr) == FP_EINVAL);
        fp_op *tv;
        CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv));
        CHECK_OK(fp_workspace_create(tv, &x.a, nullptr, &ws));
        CHECK(fp_workspace_get_dual(ws, &q.a) == FP_EINVAL);
        CHECK(fp_workspace_set_warm(ws, 1) == FP_EINVAL);
        fp_workspace_free(ws);
        fp_op_free(tv);

        // A strided image (every other column of a wider array) gives the
        // same result as a contiguous copy.
        Arr wide({40, 100}, randn(4000)), wz({40, 100});
        Arr narrow({40, 50}), nz({40, 50});
        for (int64_t i = 0; i < 40; ++i)
            for (int64_t j = 0; j < 50; ++j) narrow.data[i * 50 + j] = wide.data[i * 100 + 2 * j];
        const int64_t sstr[2] = {100, 2};
        fp_array sv = wide.a, so = wz.a;
        sv.shape = narrow.shape.data();
        so.shape = narrow.shape.data();
        sv.strides = sstr;
        so.strides = sstr;
        CHECK_OK(fp_prox(op, 0.5, &sv, &so, &o, nullptr, nullptr));
        CHECK_OK(fp_prox(op, 0.5, &narrow.a, &nz.a, &o, nullptr, nullptr));
        bool same = true;
        for (int64_t i = 0; i < 40; ++i)
            for (int64_t j = 0; j < 50; ++j) same = same && wz.data[i * 100 + 2 * j] == nz.data[i * 50 + j];
        CHECK(same);

        // float32 agrees with float64 to float32 accuracy.
        std::vector<float> xf(x.data.begin(), x.data.end()), zf(2000);
        fp_array xfa = x.a, zfa = x.a;
        xfa.dtype = zfa.dtype = FP_F32;
        xfa.data = xf.data();
        zfa.data = zf.data();
        fp_options of = o;
        of.tol = 1e-8;
        CHECK_OK(fp_prox(op, 0.5, &xfa, &zfa, &of, nullptr, nullptr));
        CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, nullptr, nullptr));
        double df = 0;
        for (int64_t i = 0; i < 2000; ++i) df = std::max(df, std::fabs(zf[i] - z.data[i]));
        CHECK(df <= 1e-3 * (1 + maxabs(x.data)));
        fp_op_free(op);
    }

    // Tiny bounds (audit T1): at gamma = 1e-9 every jump of a randn image is far
    // above 2 gamma, so the Jacobian is the identity. The dual test alone read
    // most edges as flat there, its rounding being above lam_e.
    {
        fp_op *op = make_aniso(2, {1.0});
        Arr x({16, 16}, randn(256)), z({16, 16}), dx({16, 16}, randn(256)), dz({16, 16}), gx({16, 16});
        fp_options o = opts(-1, 0, 1);
        o.tol = 1e-15;
        o.max_iter = 200000;
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        CHECK_OK(fp_prox(op, 1e-9, &x.a, &z.a, &o, ws, nullptr));
        CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a));
        double gg = 0;
        CHECK_OK(fp_vjp(op, ws, &dx.a, &gx.a, &gg, nullptr));
        double ej = 0, ev = 0;
        for (int64_t i = 0; i < 256; ++i) {
            ej = std::max(ej, std::fabs(dz.data[i] - dx.data[i]));
            ev = std::max(ev, std::fabs(gx.data[i] - dx.data[i]));
        }
        CHECK(ej <= 1e-12 * maxabs(dx.data) && ev <= 1e-12 * maxabs(dx.data));
        if (ej > 1e-12 * maxabs(dx.data)) std::fprintf(stderr, "  aniso tiny gamma: jvp off identity by %.3g\n", ej);
        fp_workspace_free(ws);
        fp_op_free(op);
    }
}

// ---------------------------------------------------------------- nonconvex maps
// prox of op at one entry with a tie rule; ties reported through info.
static double prox1_tie(fp_op *op, double gamma, double x, int tie, int64_t *ties) {
    Arr xa({1}, {x}), za({1});
    fp_options o = opts();
    o.tie = tie;
    fp_info info;
    CHECK_OK(fp_prox(op, gamma, &xa.a, &za.a, &o, nullptr, &info));
    if (ties) *ties = info.ties;
    return za.data[0];
}

// The smallest double where the nonzero tie rule leaves a nonzero result:
// the tie itself when the threshold is representable, else the first double
// above it. Bisection on the representable values between lo (0) and hi.
static double tie_point(fp_op *op, double gamma, double lo, double hi) {
    for (int it = 0; it < 200 && std::nextafter(lo, hi) < hi; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (prox1_tie(op, gamma, mid, FP_TIE_NONZERO, nullptr) == 0.0) lo = mid;
        else hi = mid;
    }
    return hi;
}

static void test_nonconvex() {
    // l0 and lq: one double sits on the tie; the rules differ there only, and
    // it is counted.
    for (int kind = 0; kind < 3; ++kind) {
        fp_op *op;
        if (kind == 0) CHECK_OK(fp_op_l0(0.7, nullptr, &op));
        else CHECK_OK(fp_op_lq(0.6, kind == 1 ? 0.5 : 0.4, nullptr, &op));
        const double gamma = 0.9;
        const double t = tie_point(op, gamma, 0.0, 10.0);
        int64_t ties = -1, ties0 = -1;
        const double zn = prox1_tie(op, gamma, t, FP_TIE_NONZERO, &ties);
        const double zz = prox1_tie(op, gamma, t, FP_TIE_ZERO, &ties0);
        CHECK(zn != 0.0 && ties == ties0);
        // On a tie the rules differ and the entry is counted; otherwise both
        // keep the nonzero branch and nothing is counted.
        CHECK(ties == 1 ? zz == 0.0 : (ties == 0 && zz == zn));
        const double below = std::nextafter(t, 0.0);
        int64_t tb = -1;
        CHECK(prox1_tie(op, gamma, below, FP_TIE_NONZERO, &tb) == 0.0 && tb == 0);
        CHECK(prox1_tie(op, gamma, -t, FP_TIE_NONZERO, nullptr) == -zn);
        // l0's tie is exactly at sqrt(2 gamma lambda), always representable.
        if (kind == 0) CHECK(t == std::sqrt(2.0 * gamma * 0.7) && ties == 1);
        // Tie counts over a batch, for any thread count.
        Arr xb({4, 50000}), zb({4, 50000});
        for (size_t i = 0; i < xb.data.size(); ++i) xb.data[i] = (i % 7 == 0) ? t : unif(-3, 3);
        int64_t counts[3];
        int ti = 0;
        for (int nt : {1, 3, 8}) {
            fp_options o = opts(-1, nt);
            fp_info info;
            CHECK_OK(fp_prox(op, gamma, &xb.a, &zb.a, &o, nullptr, &info));
            counts[ti++] = info.ties;
        }
        CHECK(counts[0] == counts[1] && counts[0] == counts[2]);
        CHECK(counts[0] == (ties == 1 ? int64_t((xb.data.size() + 6) / 7) : 0));
        fp_op_free(op);
    }

    // SCAD and MCP need the concavity parameter large enough for the step.
    fp_op *scad, *mcp;
    CHECK_OK(fp_op_scad(0.5, 3.7, nullptr, &scad));
    CHECK_OK(fp_op_mcp(0.5, 3.0, nullptr, &mcp));
    Arr x({5}, {-2.0, -0.4, 0.0, 0.7, 3.0}), z({5});
    CHECK_OK(fp_prox(scad, 2.6, &x.a, &z.a, nullptr, nullptr, nullptr));
    CHECK(fp_prox(scad, 2.7, &x.a, &z.a, nullptr, nullptr, nullptr) == FP_EPARAM);
    CHECK(fp_prox_conj(scad, 1.0 / 2.7, &x.a, &z.a, nullptr, nullptr, nullptr) == FP_EPARAM);
    CHECK_OK(fp_prox(mcp, 2.9, &x.a, &z.a, nullptr, nullptr, nullptr));
    CHECK(fp_prox(mcp, 3.0, &x.a, &z.a, nullptr, nullptr, nullptr) == FP_EPARAM);
    fp_op_free(scad);
    fp_op_free(mcp);
    // With weights the conditions use max w: a > 1 + gamma max w, b > gamma max w.
    {
        Arr w2({5}, {1.0, 2.0, 0.0, 0.5, 1.0}), w3({3}, {1.0, 1.0, 1.0});
        CHECK_OK(fp_op_scad(0.5, 3.7, &w2.a, &scad));
        CHECK_OK(fp_op_mcp(0.5, 3.0, &w2.a, &mcp));
        CHECK_OK(fp_prox(scad, 1.3, &x.a, &z.a, nullptr, nullptr, nullptr));
        CHECK(z.data[2] == x.data[2]);  // zero weight: the identity
        CHECK(fp_prox(scad, 1.4, &x.a, &z.a, nullptr, nullptr, nullptr) == FP_EPARAM);
        CHECK_OK(fp_prox(mcp, 1.45, &x.a, &z.a, nullptr, nullptr, nullptr));
        CHECK(fp_prox(mcp, 1.5, &x.a, &z.a, nullptr, nullptr, nullptr) == FP_EPARAM);
        CHECK(fp_op_nparams(scad) == 7 && fp_op_nparams(mcp) == 7);
        fp_op_free(scad);
        fp_op_free(mcp);
        CHECK_OK(fp_op_scad(0.5, 3.7, &w3.a, &scad));
        CHECK(fp_prox(scad, 1.0, &x.a, &z.a, nullptr, nullptr, nullptr) == FP_ESHAPE);
        fp_op_free(scad);
        Arr wneg({1}, {-1.0});
        CHECK(fp_op_mcp(0.5, 3.0, &wneg.a, &mcp) == FP_EPARAM);
    }

    // Parameters, the tie option and special values.
    fp_op *bad = nullptr;
    CHECK(fp_op_lq(1.0, 0.0, nullptr, &bad) == FP_EPARAM);
    CHECK(fp_op_lq(1.0, 1.0, nullptr, &bad) == FP_EPARAM);
    CHECK(fp_op_scad(1.0, 1.0, nullptr, &bad) == FP_EPARAM);
    CHECK(fp_op_mcp(1.0, 0.0, nullptr, &bad) == FP_EPARAM);
    CHECK(fp_op_log_penalty(1.0, 0.0, &bad) == FP_EPARAM);
    CHECK(fp_op_l0(-1.0, nullptr, &bad) == FP_EPARAM);
    fp_op *op;
    CHECK_OK(fp_op_lq(0.5, 0.5, nullptr, &op));
    fp_options o = opts();
    o.tie = 7;
    CHECK(fp_prox(op, 1.0, &x.a, &z.a, &o, nullptr, nullptr) == FP_EINVAL);
    Arr xn({3}, {1.0, NAN, 4.0}), zn({3});
    CHECK_OK(fp_prox(op, 1.0, &xn.a, &zn.a, nullptr, nullptr, nullptr));
    CHECK(std::isnan(zn.data[1]) && !std::isnan(zn.data[0]) && !std::isnan(zn.data[2]));
    CHECK(prox1_tie(op, 0.0, 0.3, FP_TIE_NONZERO, nullptr) == 0.3);  // gamma = 0: identity
    fp_op_free(op);
}

// ---------------------------------------------------------------- isotropic TV
// Independent certificate from the dual u of shape (k, *shape): |u_p| <= gamma,
// u_a = 0 at the last entry along a, z = x - sum_a w_a D_a^T u_a, and
// gap = sum_p gamma |g_p| - <g_p, u_p> with g = W D z.
static void iso_certificate(int k, const std::vector<int64_t> &shape, double gamma, const Vec &w, const Vec &x,
                            const Vec &z, const Vec &u, double tol, int iters_checked) {
    std::vector<int64_t> L(shape.end() - k, shape.end()), S(static_cast<size_t>(k));
    S[k - 1] = 1;
    for (int a = k - 2; a >= 0; --a) S[a] = S[a + 1] * L[a + 1];
    const int64_t Ni = S[0] * L[0], N = static_cast<int64_t>(x.size());
    double feas = 0, resid = 0;
    for (int64_t b = 0; b < N / Ni; ++b) {
        double gap = 0, dn2 = 0;
        for (int64_t p = 0; p < Ni; ++p) {
            const int64_t P = b * Ni + p;
            double nu = 0, ng = 0, dot = 0, zz = x[P];
            for (int a = 0; a < k; ++a) {
                const double wa = w[w.size() == 1 ? 0 : a];
                const double ua = u[a * N + P];
                const bool nx = (p / S[a]) % L[a] + 1 < L[a], pv = (p / S[a]) % L[a] > 0;
                nu += ua * ua;
                zz += wa * (ua - (pv ? u[a * N + P - S[a]] : 0.0));
                const double ga = nx ? wa * (z[P + S[a]] - z[P]) : 0.0;
                ng += ga * ga;
                dot += ga * ua;
                if (!nx) feas = std::max(feas, std::fabs(ua));
            }
            feas = std::max(feas, std::sqrt(nu) - gamma);
            resid = std::max(resid, std::fabs(zz - z[P]));
            gap += gamma * std::sqrt(ng) - dot;
            dn2 += x[P] * x[P];
        }
        CHECK(gap <= tol * 0.5 * dn2 * (1 + 1e-6) + 1e-13 * dn2);
        CHECK(gap >= -1e-12 * dn2);
    }
    CHECK(feas <= 1e-12 * (1 + gamma));
    CHECK(resid <= 1e-12 * (1 + maxabs(x)));
    (void)iters_checked;
}

static fp_op *make_iso(int k, Vec w) {
    fp_op *op = nullptr;
    Arr wa({static_cast<int64_t>(w.size())}, w);
    CHECK_OK(fp_op_tv_iso(k, &wa.a, &op));
    return op;
}

static void test_tv_iso() {
    struct Case {
        int k;
        std::vector<int64_t> shape, dshape;  // dshape: smaller, for the finite differences
        double gamma;
        Vec w;
    };
    const std::vector<Case> cases = {
        {2, {18, 23}, {8, 9}, 0.4, {1.0}},
        {2, {3, 11, 13}, {2, 6, 7}, 0.3, {0.7, 1.3}},
        {3, {6, 7, 8}, {4, 4, 5}, 0.3, {1.0, 0.8, 1.2}},
        {3, {6, 7, 8}, {4, 4, 5}, 0.3, {1.0, 0.0, 1.0}},
        {2, {1, 30}, {1, 30}, 0.5, {1.0}},
    };
    for (const auto &c : cases) {
        fp_op *op = make_iso(c.k, c.w);
        int64_t N = 1;
        for (int64_t s : c.shape) N *= s;
        Arr x(c.shape, image_signal(N));
        std::vector<int64_t> qshape = {c.k};
        qshape.insert(qshape.end(), c.shape.begin(), c.shape.end());
        std::vector<Vec> outs;
        for (int t : {1, 2, 8}) {
            fp_options o = opts(-1, t);
            o.tol = 1e-9;
            o.max_iter = 200000;
            fp_workspace *ws;
            CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
            Arr z(c.shape), q(qshape);
            fp_info info;
            CHECK_OK(fp_prox(op, c.gamma, &x.a, &z.a, &o, ws, &info));
            CHECK_OK(fp_workspace_get_dual(ws, &q.a));
            iso_certificate(c.k, c.shape, c.gamma, c.w, x.data, z.data, q.data, o.tol, info.iters);
            outs.push_back(z.data);
            fp_workspace_free(ws);
        }
        CHECK(outs[0] == outs[1] && outs[0] == outs[2]);

        int64_t Nd = 1;
        for (int64_t s : c.dshape) Nd *= s;
        Arr xd(c.dshape, image_signal(Nd));
        for (bool conj : {false, true}) {
            const double step = conj ? 1.0 / c.gamma : c.gamma;
            fp_options o = opts(-1, 0, 1);
            o.tol = 1e-13;  // FISTA needs ~1/sqrt(tol) iterations
            o.max_iter = 3000000;
            fp_workspace *ws;
            CHECK_OK(fp_workspace_create(op, &xd.a, &o, &ws));
            Arr z(c.dshape);
            CHECK_OK(conj ? fp_prox_conj(op, step, &xd.a, &z.a, &o, ws, nullptr)
                          : fp_prox(op, step, &xd.a, &z.a, &o, ws, nullptr));
            CHECK_OK(fp_workspace_set_deriv_tol(ws, 1e-10, 10000));  // the default stops at sqrt(2 gap) / |x|
            Arr dx(c.dshape, randn(Nd)), jv(c.dshape), g(c.dshape, randn(Nd)), gx(c.dshape);
            Vec wv(static_cast<size_t>(c.k));
            for (int a = 0; a < c.k; ++a) wv[a] = c.w[c.w.size() == 1 ? 0 : a];
            Arr dw({c.k}, randn(c.k)), gw({c.k});
            for (int a = 0; a < c.k; ++a)
                if (wv[a] == 0.0) dw.data[a] = 0.0;
            const double dg = 0.2;
            CHECK_OK(fp_jvp(op, ws, &dx.a, dg, &dw.a, &jv.a));
            double res = -1;
            int its = -1;
            CHECK_OK(fp_workspace_deriv_info(ws, &res, &its));
            CHECK(res <= 1e-10 && its >= 0);
            double gg = 0;
            CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, &gw.a));
            double lhs = 0, rhs = gg * dg;
            for (int64_t i = 0; i < Nd; ++i) {
                lhs += g.data[i] * jv.data[i];
                rhs += gx.data[i] * dx.data[i];
            }
            for (int a = 0; a < c.k; ++a) rhs += gw.data[a] * dw.data[a];
            CHECK(std::fabs(lhs - rhs) <= 1e-8 * (1 + std::fabs(lhs)));
            // The solve error (gap 1e-13, times sigma after the conjugate)
            // divided by h bounds the finite-difference accuracy: h = 1e-5,
            // and the tolerance accordingly. Adjointness above is exact to CG.
            const double h = 1e-5;
            Vec zs[2];
            for (int sgn = 0; sgn < 2; ++sgn) {
                const double e = sgn ? -h : h;
                Vec wp(wv), xp(xd.data);
                for (int a = 0; a < c.k; ++a) wp[a] += e * dw.data[a];
                for (int64_t i = 0; i < Nd; ++i) xp[i] += e * dx.data[i];
                fp_op *op2 = make_iso(c.k, wp);
                Arr xa(c.dshape, xp), za(c.dshape);
                fp_options o2 = o;
                o2.keep_structure = 0;
                CHECK_OK(conj ? fp_prox_conj(op2, step + e * dg, &xa.a, &za.a, &o2, nullptr, nullptr)
                              : fp_prox(op2, step + e * dg, &xa.a, &za.a, &o2, nullptr, nullptr));
                zs[sgn] = za.data;
                fp_op_free(op2);
            }
            double err = 0, scale = 1;
            for (int64_t i = 0; i < Nd; ++i) {
                const double fd = (zs[0][i] - zs[1][i]) / (2 * h);
                err = std::max(err, std::fabs(fd - jv.data[i]));
                scale = std::max(scale, std::fabs(fd));
            }
            CHECK(err <= 3e-3 * scale);
            if (err > 3e-3 * scale) std::fprintf(stderr, "  tv_iso k=%d conj=%d fd error %.2e\n", c.k, conj, err);
            fp_workspace_free(ws);
        }
        fp_op_free(op);
    }

    // Warm starts, the derivative controls and errors.
    fp_op *op = make_iso(2, {1.0});
    Arr x({30, 40}, image_signal(1200)), z({30, 40}), q({2, 30, 40});
    fp_options o = opts(-1, 0, 1);
    o.tol = 1e-8;
    o.max_iter = 100000;
    fp_workspace *ws;
    CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
    fp_info cold, warm;
    CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, &cold));
    CHECK_OK(fp_workspace_set_warm(ws, 1));
    CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, &warm));
    CHECK(warm.iters == 0 && cold.iters > 0);
    // A dual for a larger gamma is projected onto the smaller ball.
    CHECK_OK(fp_prox(op, 0.9, &x.a, &z.a, &o, ws, nullptr));
    CHECK_OK(fp_prox(op, 0.3, &x.a, &z.a, &o, ws, &warm));
    CHECK_OK(fp_workspace_get_dual(ws, &q.a));
    iso_certificate(2, {30, 40}, 0.3, {1.0}, x.data, z.data, q.data, o.tol, warm.iters);
    CHECK(fp_workspace_set_deriv_tol(ws, -1.0, 10) == FP_EPARAM);
    CHECK(fp_workspace_set_deriv_tol(ws, 1e-3, 0) == FP_EPARAM);
    CHECK_OK(fp_workspace_set_deriv_tol(ws, 1e-3, 5));
    Arr g({30, 40}, randn(1200)), gx({30, 40});
    CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, nullptr, nullptr));
    double res;
    int its;
    CHECK_OK(fp_workspace_deriv_info(ws, &res, &its));
    CHECK(its <= 5);
    // The default (0) stops at the relative error of z that the gap certifies.
    CHECK_OK(fp_workspace_set_deriv_tol(ws, 0.0, 10000));
    CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, nullptr, nullptr));
    CHECK_OK(fp_workspace_deriv_info(ws, &res, &its));
    double xx = 0;
    for (double v : x.data) xx += v * v;
    CHECK(res <= std::sqrt(2 * warm.gap / xx) && res > 1e-10 && its > 0);
    fp_workspace_free(ws);
    fp_op *tv;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv));
    CHECK_OK(fp_workspace_create(tv, &x.a, nullptr, &ws));
    CHECK(fp_workspace_set_deriv_tol(ws, 1e-6, 10) == FP_EINVAL);
    CHECK(fp_workspace_deriv_info(ws, &res, &its) == FP_EINVAL);
    fp_workspace_free(ws);
    fp_op_free(tv);
    // Extreme weights and steps (found by fuzzing: 4 w^2 underflowed to 0):
    // a denormal weight is the identity to rounding, a huge radius the
    // constant mean, and every output is finite with a certified gap.
    for (double wv : {9.7728522813346931e-309, 1e-200, 1e200, 1.7e308}) {
        for (double gam : {0.2, 1e100}) {
            fp_op *eop = make_iso(2, {wv});
            Arr ex({6, 7}, randn(42)), ez({6, 7});
            fp_options eo = opts();
            eo.tol = 1e-10;
            eo.max_iter = 100000;
            fp_info ei;
            if (wv * gam > 0x1p960) {  // outside the step range (audit A1)
                CHECK(fp_prox(eop, gam, &ex.a, &ez.a, &eo, nullptr, &ei) == FP_EPARAM);
                fp_op_free(eop);
                continue;
            }
            CHECK_OK(fp_prox(eop, gam, &ex.a, &ez.a, &eo, nullptr, &ei));
            CHECK(std::isfinite(maxabs(ez.data)) && ei.gap <= 1e-10 * 0.5 * 42 * 9);
            if (wv * gam > 1e10) {  // constant mean
                double m = 0;
                for (double t : ex.data) m += t / 42;
                double dev = 0;
                for (double t : ez.data) dev = std::max(dev, std::fabs(t - m));
                CHECK(dev <= 1e-6 * (1 + std::fabs(m)));
            }
            if (wv < 1e-100) CHECK(std::fabs(maxabs(ez.data) - maxabs(ex.data)) <= 1e-12);
            fp_op_free(eop);
        }
    }

    fp_op *bad = nullptr;
    CHECK(fp_op_tv_iso(1, nullptr, &bad) == FP_EPARAM);
    Arr wneg({2}, {1.0, -1.0});
    CHECK(fp_op_tv_iso(2, &wneg.a, &bad) == FP_EPARAM);
    Arr xb({2, 10, 12}, randn(240)), zb({2, 10, 12});
    xb.data[3] = NAN;
    CHECK_OK(fp_prox(op, 0.5, &xb.a, &zb.a, nullptr, nullptr, nullptr));
    CHECK(std::isnan(zb.data[0]) && !std::isnan(zb.data[120]));
    xb.data[3] = INFINITY;
    CHECK(fp_prox(op, 0.5, &xb.a, &zb.a, nullptr, nullptr, nullptr) == FP_EINVAL);
    fp_op_free(op);
}

// ---------------------------------------------------------------- spatial weights

// Isotropic TV with pixel weights (iso) or anisotropic TV with edge weights,
// for images of shape img: weights of size m per image (pixel weights, or
// ndim blocks of edge weights).
static fp_op *make_spatial(bool iso, int k, const std::vector<int64_t> &img, Vec w, const Vec &sw) {
    fp_op *op = nullptr;
    Arr wa({static_cast<int64_t>(w.size())}, w);
    std::vector<int64_t> sh(img);
    if (!iso) sh.insert(sh.begin(), k);
    Arr sa(sh, sw);
    CHECK_OK(iso ? fp_op_tv_iso_weighted(k, &wa.a, &sa.a, &op) : fp_op_tv_aniso_weighted(k, &wa.a, &sa.a, &op));
    return op;
}

static void test_tv_spatial_weights() {
    struct Case {
        bool iso;
        int k;
        std::vector<int64_t> shape;  // batch axes first
        double gamma;
        Vec w;
    };
    const std::vector<Case> cases = {
        {true, 2, {2, 9, 11}, 0.4, {1.0, 0.7}}, {true, 3, {4, 5, 6}, 0.3, {1.0}},
        {false, 2, {2, 9, 11}, 0.4, {1.0, 0.7}}, {false, 3, {4, 5, 6}, 0.3, {1.0, 0.5, 1.2}},
        {false, 2, {1, 13}, 0.5, {1.0}},
    };
    for (const auto &c : cases) {
        const std::vector<int64_t> img(c.shape.end() - c.k, c.shape.end());
        int64_t N = 1, Ni = 1;
        for (int64_t s : c.shape) N *= s;
        for (int64_t s : img) Ni *= s;
        const int64_t m = c.iso ? Ni : c.k * Ni;
        Vec sw(static_cast<size_t>(m));
        for (int64_t i = 0; i < m; ++i) sw[i] = i % 7 == 3 ? 0.0 : 0.3 + 1.4 * std::fabs(std::sin(1.7 * i));
        fp_op *op = make_spatial(c.iso, c.k, img, c.w, sw);
        Arr x(c.shape, image_signal(N));

        // Weights of one equal the unweighted operator bit for bit; the
        // weighted solve is the same for any thread count.
        {
            fp_op *one = make_spatial(c.iso, c.k, img, c.w, Vec(static_cast<size_t>(m), 1.0));
            fp_op *plain = nullptr;
            Arr wa({static_cast<int64_t>(c.w.size())}, c.w);
            CHECK_OK(c.iso ? fp_op_tv_iso(c.k, &wa.a, &plain) : fp_op_tv_aniso(c.k, &wa.a, &plain));
            fp_options o = opts();
            o.tol = 1e-10;
            o.max_iter = 100000;
            Arr z1(c.shape), z2(c.shape);
            CHECK_OK(fp_prox(one, c.gamma, &x.a, &z1.a, &o, nullptr, nullptr));
            CHECK_OK(fp_prox(plain, c.gamma, &x.a, &z2.a, &o, nullptr, nullptr));
            CHECK(z1.data == z2.data);
            fp_op_free(one);
            fp_op_free(plain);
            std::vector<Vec> outs;
            for (int t : {1, 2, 8}) {
                fp_options ot = opts(-1, t);
                ot.tol = 1e-10;
                ot.max_iter = 100000;
                Arr z(c.shape);
                fp_info info;
                CHECK_OK(fp_prox(op, c.gamma, &x.a, &z.a, &ot, nullptr, &info));  // FP_ENOTCONV otherwise
                outs.push_back(z.data);
            }
            CHECK(outs[0] == outs[1] && outs[0] == outs[2]);
        }

        // Derivatives in (x, gamma, w, spatial weights): adjointness and
        // finite differences (only positive weights move).
        const int64_t P = c.k + m;
        fp_options o = opts(-1, 0, 1);
        o.tol = 1e-14;
        o.max_iter = 1000000;
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        CHECK_OK(fp_workspace_set_deriv_tol(ws, 1e-12, 100000));
        Arr z(c.shape);
        CHECK_OK(fp_prox(op, c.gamma, &x.a, &z.a, &o, ws, nullptr));
        Arr dx(c.shape, randn(N)), jv(c.shape), g(c.shape, randn(N)), gx(c.shape);
        Vec wv(static_cast<size_t>(c.k));
        for (int a = 0; a < c.k; ++a) wv[a] = c.w[c.w.size() == 1 ? 0 : a];
        Arr dp({P}, randn(P)), gp({P});
        for (int64_t i = 0; i < m; ++i)
            if (sw[i] == 0.0) dp.data[c.k + i] = 0.0;
        const double dg = 0.2;
        CHECK_OK(fp_jvp(op, ws, &dx.a, dg, &dp.a, &jv.a));
        double gg = 0;
        CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, &gp.a));
        double lhs = 0, rhs = gg * dg;
        for (int64_t i = 0; i < N; ++i) {
            lhs += g.data[i] * jv.data[i];
            rhs += gx.data[i] * dx.data[i];
        }
        for (int64_t i = 0; i < P; ++i) rhs += gp.data[i] * dp.data[i];
        CHECK(std::fabs(lhs - rhs) <= 1e-8 * (1 + std::fabs(lhs)));
        // Anisotropic TV is piecewise affine and solved exactly once polished: a
        // step of 1e-6 crossed a breakpoint here.
        const double h = c.iso ? 1e-6 : 1e-8;
        Vec zs[2];
        for (int sgn = 0; sgn < 2; ++sgn) {
            const double e = sgn ? -h : h;
            Vec wp(wv), sp(sw), xp(x.data);
            for (int a = 0; a < c.k; ++a) wp[a] += e * dp.data[a];
            for (int64_t i = 0; i < m; ++i) sp[i] += e * dp.data[c.k + i];
            for (int64_t i = 0; i < N; ++i) xp[i] += e * dx.data[i];
            fp_op *op2 = make_spatial(c.iso, c.k, img, wp, sp);
            Arr xa(c.shape, xp), za(c.shape);
            fp_options o2 = o;
            o2.keep_structure = 0;
            CHECK_OK(fp_prox(op2, c.gamma + e * dg, &xa.a, &za.a, &o2, nullptr, nullptr));
            zs[sgn] = za.data;
            fp_op_free(op2);
        }
        double err = 0, scale = 1;
        for (int64_t i = 0; i < N; ++i) {
            const double fd = (zs[0][i] - zs[1][i]) / (2 * h);
            err = std::max(err, std::fabs(fd - jv.data[i]));
            scale = std::max(scale, std::fabs(fd));
        }
        CHECK(err <= 1e-4 * scale);
        if (err > 1e-4 * scale) std::fprintf(stderr, "  spatial iso=%d k=%d fd error %.2e\n", c.iso, c.k, err);
        fp_workspace_free(ws);

        // Images of another shape are rejected.
        std::vector<int64_t> other(c.shape);
        other.back() += 1;
        Arr xo(other, randn(N / c.shape.back() * other.back())), zo(other);
        CHECK(fp_prox(op, c.gamma, &xo.a, &zo.a, nullptr, nullptr, nullptr) == FP_ESHAPE);
        fp_op_free(op);
    }

    fp_op *bad = nullptr;
    Arr pw({4, 5}, Vec(20, 1.0)), pwn({4, 5}, Vec(20, 1.0)), p1({20}, Vec(20, 1.0));
    pwn.data[7] = -1.0;
    CHECK(fp_op_tv_iso_weighted(2, nullptr, nullptr, &bad) == FP_EINVAL);
    CHECK(fp_op_tv_iso_weighted(2, nullptr, &pwn.a, &bad) == FP_EPARAM);
    CHECK(fp_op_tv_iso_weighted(2, nullptr, &p1.a, &bad) == FP_ESHAPE);
    CHECK(fp_op_tv_iso_weighted(3, nullptr, &pw.a, &bad) == FP_ESHAPE);
    Arr ew({3, 4, 5}, Vec(60, 1.0)), ewn({2, 4, 5}, Vec(40, 1.0));
    ewn.data[11] = NAN;
    CHECK(fp_op_tv_aniso_weighted(2, nullptr, nullptr, &bad) == FP_EINVAL);
    CHECK(fp_op_tv_aniso_weighted(2, nullptr, &ewn.a, &bad) == FP_EPARAM);
    CHECK(fp_op_tv_aniso_weighted(2, nullptr, &ew.a, &bad) == FP_ESHAPE);
    CHECK(fp_op_tv_aniso_weighted(2, nullptr, &pw.a, &bad) == FP_ESHAPE);
    CHECK(bad == nullptr);
}

// ---------------------------------------------------------------- misc
// ---------------------------------------------------------------- kinks and branch overrides

struct I8 {  // int8 array with an fp_array view; strides in elements
    std::vector<int64_t> shape, strides;
    std::vector<int8_t> data;
    fp_array a{};
    I8(std::vector<int64_t> s, std::vector<int64_t> st, size_t n, int64_t offset = 0)
        : shape(std::move(s)), strides(std::move(st)), data(n, 0) {
        a.data = data.data() + offset;
        a.dtype = FP_I8;
        a.ndim = static_cast<int>(shape.size());
        a.shape = shape.data();
        a.strides = strides.data();
        a.device = FP_DEVICE_CPU;
    }
};

// Tree TV's kinks are edges (since 0.4): masks and sides of shape (..., m).
// x = (2, -1, 5) on the chain 0-1-2, gamma = 1: z = (1, 1, 4) with the flow
// on edge (0, 1) at its bound.
// fp_value (since 0.5): one value per row or image, sets as indicators.
static void test_value() {
    fp_op *op;
    CHECK_OK(fp_op_soft(0.5, nullptr, &op));
    Arr x({2, 3}, {1.0, -2.0, 0.5, 0.0, 3.0, -1.0}), v({2}), v3({3}), bad({2});
    CHECK_OK(fp_value(op, &x.a, nullptr, &v.a));
    CHECK(v.data == Vec({1.75, 2.0}));
    fp_options o = opts(0);  // along axis 0: three columns
    CHECK_OK(fp_value(op, &x.a, &o, &v3.a));
    CHECK(v3.data == Vec({0.5, 2.5, 0.75}));
    CHECK(fp_value(op, &x.a, nullptr, &v3.a) == FP_ESHAPE);
    bad.a.dtype = FP_F32;
    CHECK(fp_value(op, &x.a, nullptr, &bad.a) == FP_EDTYPE);
    o = opts();
    o.tol = -1.0;
    CHECK(fp_value(op, &x.a, &o, &v.a) == FP_EPARAM);
    CHECK(fp_value(nullptr, &x.a, nullptr, &v.a) == FP_EINVAL);
    fp_op_free(op);
    // The simplex of radius 1: 0 on it (within tol), INFINITY off it.
    CHECK_OK(fp_op_simplex(1.0, &op));
    Arr s({2, 3}, {0.2, 0.3, 0.5, 0.2, 0.3, 0.6});
    CHECK_OK(fp_value(op, &s.a, nullptr, &v.a));
    CHECK(v.data[0] == 0.0 && std::isinf(v.data[1]));
    fp_op_free(op);
    // An image operator: one value per image, out of ndim 0 for one image.
    CHECK_OK(fp_op_tv_aniso(2, nullptr, &op));
    Arr img({2, 2}, {0.0, 1.0, 3.0, 1.0}), one({});
    one.a.ndim = 0;
    CHECK_OK(fp_value(op, &img.a, nullptr, &one.a));
    CHECK(one.data[0] == 6.0);  // down the columns |3 - 0| + |1 - 1|, along the rows |1 - 0| + |1 - 3|
    fp_op_free(op);
}

// ---------------------------------------------------------------- spectral operators
// fp_prox on the spectrum s (batch shape `lead`, k values each), then
// fp_spectral_jvp from A; z, B and r are returned.
struct Spec {
    Vec z, b, r;
    fp_status st = FP_OK;
};
static Spec spectral(fp_op *op, fp_spectral_kind kind, std::vector<int64_t> lead, int64_t k, const Vec &s,
                     const Vec &a, double gamma = 1.0, double dgamma = 0.0, int threads = 0) {
    std::vector<int64_t> sh = lead, ash = lead;
    sh.push_back(k);
    ash.push_back(k);
    ash.push_back(k);
    Arr sa(sh, s), za(sh), aa(ash, a), ba(ash), ra(sh);
    fp_options o = opts(-1, threads, 1);
    fp_workspace *ws;
    CHECK_OK(fp_workspace_create(op, &sa.a, &o, &ws));
    CHECK_OK(fp_prox(op, gamma, &sa.a, &za.a, &o, ws, nullptr));
    Spec out;
    out.st = fp_spectral_jvp(op, ws, kind, &sa.a, &za.a, &aa.a, dgamma, nullptr, &ba.a,
                             kind == FP_SPECTRAL_SVD ? &ra.a : nullptr);
    fp_workspace_free(ws);
    out.z = za.data;
    out.b = ba.data;
    out.r = ra.data;
    return out;
}

// B and r of one matrix for a separable f with J = diag(jd), from the
// formulas: exact ties and zeros only, which take the average of J.
static void spectral_ref(bool svd, const Vec &s, const Vec &z, const Vec &jd, const Vec &a, Vec &b, Vec &r) {
    const size_t k = s.size();
    b.assign(k * k, 0.0);
    r.assign(k, 0.0);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const double aij = a[i * k + j], aji = a[j * k + i];
            if (i == j) {
                b[i * k + i] = jd[i] * aij;
                continue;
            }
            const double gp = s[i] == s[j] ? 0.5 * (jd[i] + jd[j]) : (z[i] - z[j]) / (s[i] - s[j]);
            if (!svd) {
                b[i * k + j] = gp * aij;
                continue;
            }
            const double gm = s[i] + s[j] == 0.0 ? 0.5 * (jd[i] + jd[j]) : (z[i] + z[j]) / (s[i] + s[j]);
            b[i * k + j] = 0.5 * (gp * (aij + aji) + gm * (aij - aji));
        }
        r[i] = s[i] != 0.0 ? z[i] / s[i] : jd[i];
    }
}

static double maxdiff(const Vec &u, const Vec &v) {
    double m = 0.0;
    for (size_t i = 0; i < u.size(); ++i) m = std::max(m, std::fabs(u[i] - v[i]));
    return m;
}

static void test_spectral() {
    fp_op *soft, *nonneg, *iso, *wsoft, *sym_box, *box, *l1b, *simplex, *barrier;
    CHECK_OK(fp_op_soft(0.5, nullptr, &soft));
    CHECK_OK(fp_op_nonneg(&nonneg));
    CHECK_OK(fp_op_isotonic(nullptr, 1, &iso));
    Arr w({3}, {1.0, 2.0, 1.0});
    CHECK_OK(fp_op_soft(0.5, &w.a, &wsoft));
    CHECK_OK(fp_op_box(-2.0, 2.0, nullptr, nullptr, &sym_box));
    CHECK_OK(fp_op_box(0.0, 2.0, nullptr, nullptr, &box));
    CHECK_OK(fp_op_l1_ball(3.0, nullptr, &l1b));
    CHECK_OK(fp_op_simplex(1.0, &simplex));
    CHECK_OK(fp_op_log_barrier(0.4, &barrier));

    // Which operators qualify.
    CHECK_OK(fp_op_spectral_check(soft, FP_SPECTRAL_SVD));
    CHECK_OK(fp_op_spectral_check(soft, FP_SPECTRAL_EIG));
    CHECK(fp_op_spectral_check(nonneg, FP_SPECTRAL_SVD) == FP_EINVAL);
    CHECK_OK(fp_op_spectral_check(nonneg, FP_SPECTRAL_EIG));
    CHECK(fp_op_spectral_check(iso, FP_SPECTRAL_EIG) == FP_EINVAL);
    CHECK(fp_op_spectral_check(wsoft, FP_SPECTRAL_EIG) == FP_EINVAL);
    CHECK_OK(fp_op_spectral_check(sym_box, FP_SPECTRAL_SVD));
    CHECK(fp_op_spectral_check(box, FP_SPECTRAL_SVD) == FP_EINVAL);
    CHECK_OK(fp_op_spectral_check(box, FP_SPECTRAL_EIG));
    CHECK(fp_op_spectral_check(simplex, FP_SPECTRAL_SVD) == FP_EINVAL);
    CHECK(fp_op_spectral_check(soft, static_cast<fp_spectral_kind>(7)) == FP_EINVAL);
    CHECK(fp_op_spectral_check(nullptr, FP_SPECTRAL_SVD) == FP_EINVAL);

    // Nuclear norm: distinct values, a tie, a value under the threshold, a zero.
    {
        const Vec s = {3.0, 2.0, 2.0, 0.3, 0.0};
        const Vec a = randn(25);
        Spec out = spectral(soft, FP_SPECTRAL_SVD, {}, 5, s, a);
        CHECK_OK(out.st);
        Vec b, r;
        spectral_ref(true, s, out.z, {1, 1, 1, 0, 0}, a, b, r);
        CHECK(maxdiff(out.b, b) < 1e-14);
        CHECK(maxdiff(out.r, r) < 1e-14);
        // Eigenvalues of the same map: no G- and no r.
        out = spectral(soft, FP_SPECTRAL_EIG, {}, 5, s, a);
        spectral_ref(false, s, out.z, {1, 1, 1, 0, 0}, a, b, r);
        CHECK_OK(out.st);
        CHECK(maxdiff(out.b, b) < 1e-14);
        // dgamma reaches the diagonal only: dz/dgamma = -lambda on the support.
        out = spectral(soft, FP_SPECTRAL_SVD, {}, 5, s, Vec(25, 0.0), 1.0, 1.0);
        CHECK(out.b[0] == -0.5 && out.b[6] == -0.5 && out.b[18] == 0.0 && out.b[24] == 0.0);
        CHECK(out.b[1] == 0.0);
    }
    // Near ties of a nonlinear map (-log det): the average of the two
    // derivatives, against the divided difference at the midpoint.
    {
        const double g = 1.0, l = 0.4, x0 = 1.3;
        auto f = [&](double x) { return 0.5 * (x + std::sqrt(x * x + 4 * g * l)); };
        auto fd = [&](double x) { return 0.5 * (1 + x / std::sqrt(x * x + 4 * g * l)); };
        for (double gap : {1e-3, 1e-7, 1e-12, 0.0}) {
            const Vec s = {x0, x0 + gap, -0.7};
            Vec a(9, 0.0);
            a[1] = 1.0;  // A_01
            Spec out = spectral(barrier, FP_SPECTRAL_EIG, {}, 3, s, a, g);
            CHECK_OK(out.st);
            const double exact = gap > 1e-6 ? (f(x0) - f(x0 + gap)) / -gap : fd(x0 + gap / 2);
            CHECK(std::fabs(out.b[1] - exact) < 1e-11 * exact);
            CHECK(out.b[3] == 0.0);  // B_10 = G+_10 A_10
        }
    }
    // A row operator (the nuclear-norm ball): tied values on the support take
    // J_ii - J_ij = 1 from J = I - 1 1^T / |S|; tied values off it 0.
    {
        const Vec s = {2.0, 2.0, 1.0, 0.1, 0.1};  // tau = 2/3: z = (4/3, 4/3, 1/3, 0, 0)
        const Vec a = randn(25);
        Spec out = spectral(l1b, FP_SPECTRAL_SVD, {}, 5, s, a);
        CHECK_OK(out.st);
        CHECK(std::fabs(out.z[0] - 4.0 / 3) < 1e-15 && out.z[3] == 0.0);
        auto bij = [&](int i, int j, double gp, double gm) {
            return 0.5 * (gp * (a[i * 5 + j] + a[j * 5 + i]) + gm * (a[i * 5 + j] - a[j * 5 + i]));
        };
        CHECK(std::fabs(out.b[1] - bij(0, 1, 1.0, (8.0 / 3) / 4)) < 1e-14);
        CHECK(std::fabs(out.b[2] - bij(0, 2, 1.0, (5.0 / 3) / 3)) < 1e-14);
        CHECK(std::fabs(out.b[3 * 5 + 4] - bij(3, 4, 0.0, 0.0)) < 1e-14);
        CHECK(std::fabs(out.b[0] - (2.0 / 3 * a[0] - a[6] / 3 - a[12] / 3)) < 1e-14);  // (J diag A)_0
        CHECK(std::fabs(out.r[2] - 1.0 / 3) < 1e-15 && out.r[4] == 0.0);
    }
    // The spectraplex: eigenvalues onto the simplex.
    {
        const Vec s = {0.5, 0.5, -0.2};  // z = (0.5, 0.5, 0), support {0, 1}
        Vec a(9, 0.0);
        a[1] = 1.0;
        a[2] = 2.0;
        Spec out = spectral(simplex, FP_SPECTRAL_EIG, {}, 3, s, a);
        CHECK_OK(out.st);
        CHECK(std::fabs(out.b[1] - 1.0) < 1e-15);              // J_00 - J_01 = 1
        CHECK(std::fabs(out.b[2] - 2.0 * 0.5 / 0.7) < 1e-15);  // G+_02 A_02
    }
    // Batches and strides: each matrix as if alone; A given transposed.
    {
        const int64_t k = 4;
        const Vec s = {3.0, 1.0, 1.0, 0.2, 2.0, 2.0, 2.0, 2.0, 0.6, 0.6, 0.0, 0.0,
                       5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.9, 0.7};
        const Vec a = randn(6 * k * k);
        Spec all = spectral(soft, FP_SPECTRAL_SVD, {2, 3}, k, s, a);
        CHECK_OK(all.st);
        for (int m = 0; m < 6; ++m) {
            const Vec sm(s.begin() + m * k, s.begin() + (m + 1) * k);
            const Vec am(a.begin() + m * k * k, a.begin() + (m + 1) * k * k);
            Spec one = spectral(soft, FP_SPECTRAL_SVD, {}, k, sm, am);
            CHECK(maxdiff(one.b, Vec(all.b.begin() + m * k * k, all.b.begin() + (m + 1) * k * k)) == 0.0);
            CHECK(maxdiff(one.r, Vec(all.r.begin() + m * k, all.r.begin() + (m + 1) * k)) == 0.0);
        }
        // A through transposed strides equals A^T given contiguously.
        Vec at(k * k);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) at[i * k + j] = a[j * k + i];
        const Vec s0(s.begin(), s.begin() + k);
        Arr sa({k}, s0), za({k}), aa({k, k}, Vec(a.begin(), a.begin() + k * k)), b1({k, k}), b2({k, k});
        Arr att({k, k}, at);
        std::swap(aa.strides[0], aa.strides[1]);
        fp_options o = opts(-1, 0, 1);
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(soft, &sa.a, &o, &ws));
        CHECK_OK(fp_prox(soft, 1.0, &sa.a, &za.a, &o, ws, nullptr));
        CHECK_OK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_SVD, &sa.a, &za.a, &aa.a, 0.0, nullptr, &b1.a, nullptr));
        CHECK_OK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_SVD, &sa.a, &za.a, &att.a, 0.0, nullptr, &b2.a, nullptr));
        CHECK(b1.data == b2.data);
        // No allocation after the first call.
        const long before = g_allocs.load();
        CHECK_OK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_SVD, &sa.a, &za.a, &att.a, 0.0, nullptr, &b2.a, nullptr));
#ifndef FP_TSAN
        CHECK(g_allocs.load() == before);
#endif
        // Errors.
        Arr neg({k}, {1.0, -1.0, 0.5, 0.0}), bad({k + 1, k + 1}), r({k});
        CHECK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_SVD, &neg.a, &za.a, &att.a, 0.0, nullptr, &b2.a, nullptr) ==
              FP_EINVAL);
        CHECK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_SVD, &sa.a, &za.a, &bad.a, 0.0, nullptr, &b2.a, nullptr) ==
              FP_ESHAPE);
        CHECK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_EIG, &sa.a, &za.a, &att.a, 0.0, nullptr, &b2.a, &r.a) ==
              FP_EINVAL);
        CHECK(fp_spectral_jvp(nonneg, ws, FP_SPECTRAL_EIG, &sa.a, &za.a, &att.a, 0.0, nullptr, &b2.a, nullptr) ==
              FP_EINVAL);
        CHECK_OK(fp_prox_conj(soft, 1.0, &sa.a, &za.a, &o, ws, nullptr));
        CHECK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_SVD, &sa.a, &za.a, &att.a, 0.0, nullptr, &b2.a, nullptr) ==
              FP_EINVAL);
        fp_workspace_free(ws);
        // The workspace of a prox along another axis.
        Arr s2({k, 2}, Vec(2 * k, 1.0)), z2({k, 2});
        o = opts(0, 0, 1);
        CHECK_OK(fp_workspace_create(soft, &s2.a, &o, &ws));
        CHECK_OK(fp_prox(soft, 1.0, &s2.a, &z2.a, &o, ws, nullptr));
        Arr a2({k, 2, 2}), b3({k, 2, 2});
        CHECK(fp_spectral_jvp(soft, ws, FP_SPECTRAL_EIG, &s2.a, &z2.a, &a2.a, 0.0, nullptr, &b3.a, nullptr) ==
              FP_EINVAL);
        fp_workspace_free(ws);
    }
    // A matrix with NaN gives NaN off its diagonal; the others are unaffected.
    {
        const Vec s = {1.0, NAN, 0.5, 2.0, 1.0, 0.5};
        Spec out = spectral(soft, FP_SPECTRAL_SVD, {2}, 3, s, randn(18));
        CHECK_OK(out.st);
        CHECK(std::isnan(out.b[1]) && std::isnan(out.r[0]));
        CHECK(std::isfinite(out.b[9 + 1]) && std::isfinite(out.r[3]));
    }
    // Bit-identical for any thread count, float32 close to float64.
    {
        const int64_t k = 32, nb = 64;
        Vec s(static_cast<size_t>(nb * k));
        for (auto &v : s) v = std::floor(unif(0.0, 8.0)) / 4;  // many ties and zeros
        const Vec a = randn(nb * k * k);
        Spec one = spectral(l1b, FP_SPECTRAL_SVD, {nb}, k, s, a, 1.0, 0.0, 1);
        Spec many = spectral(l1b, FP_SPECTRAL_SVD, {nb}, k, s, a, 1.0, 0.0, 8);
        CHECK(one.b == many.b && one.r == many.r);

        std::vector<float> sf(s.begin(), s.end()), af(a.begin(), a.end()), zf(s.size()), bf(a.size());
        int64_t sh[2] = {nb, k}, ash[3] = {nb, k, k};
        fp_array sa{sf.data(), FP_F32, 2, sh, nullptr, FP_DEVICE_CPU}, za{zf.data(), FP_F32, 2, sh, nullptr, FP_DEVICE_CPU};
        fp_array aa{af.data(), FP_F32, 3, ash, nullptr, FP_DEVICE_CPU}, ba{bf.data(), FP_F32, 3, ash, nullptr, FP_DEVICE_CPU};
        fp_options o = opts(-1, 0, 1);
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(l1b, &sa, &o, &ws));
        CHECK_OK(fp_prox(l1b, 1.0, &sa, &za, &o, ws, nullptr));
        CHECK_OK(fp_spectral_jvp(l1b, ws, FP_SPECTRAL_SVD, &sa, &za, &aa, 0.0, nullptr, &ba, nullptr));
        double err = 0.0;
        for (size_t i = 0; i < bf.size(); ++i) err = std::max(err, std::fabs(bf[i] - one.b[i]));
        CHECK(err < 1e-4);
        fp_workspace_free(ws);
    }
    for (fp_op *op : {soft, nonneg, iso, wsoft, sym_box, box, l1b, simplex, barrier}) fp_op_free(op);
}

static void test_edge_kinks() {
    const int64_t edges[4] = {0, 1, 1, 2};
    fp_op *op;
    CHECK_OK(fp_op_tree_tv(3, 2, edges, nullptr, &op));
    Arr x({2, 3}, {2.0, -1.0, 5.0, 3.0, 0.0, -3.0}), z({2, 3}), d({2, 3});
    Arr dx({2, 3}, {1.0, -1.0, 0.0, 1.0, -1.0, 0.0});
    fp_options o = opts(-1, 0, 1);
    fp_workspace *ws;
    CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
    CHECK_OK(fp_prox(op, 1.0, &x.a, &z.a, &o, ws, nullptr));
    I8 m({2, 2}, {2, 1}, 4), wrong({2, 3}, {3, 1}, 6);
    int64_t count = -1;
    CHECK_OK(fp_workspace_kinks(ws, 1e-12, &m.a, &count));
    CHECK(count == 1 && m.data == std::vector<int8_t>({1, 0, 0, 0}));
    CHECK(fp_workspace_kinks(ws, 1e-12, &wrong.a, &count) == FP_ESHAPE);
    CHECK(fp_workspace_set_branch(ws, &wrong.a, 1e-12) == FP_ESHAPE);
    // Default: joined, so the split direction has no effect on nodes 0 and 1.
    CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &d.a));
    CHECK(std::fabs(d.data[0]) < 1e-15 && std::fabs(d.data[1]) < 1e-15);
    I8 up({2, 2}, {2, 1}, 4);
    up.data[0] = 1;
    CHECK_OK(fp_workspace_set_branch(ws, &up.a, 1e-12));
    CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &d.a));
    CHECK(std::fabs(d.data[0] - 1.0) < 1e-12 && std::fabs(d.data[1] + 1.0) < 1e-12);
    // The VJP follows the override (the Jacobian is symmetric here).
    double gg = 0.0;
    Arr gp({1}), gx({2, 3});
    CHECK_OK(fp_vjp(op, ws, &dx.a, &gx.a, &gg, &gp.a));
    CHECK(std::fabs(gx.data[0] - 1.0) < 1e-12 && std::fabs(gx.data[1] + 1.0) < 1e-12);
    CHECK_OK(fp_workspace_set_branch(ws, nullptr, 0.0));
    CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &d.a));
    CHECK(std::fabs(d.data[0]) < 1e-15 && std::fabs(d.data[1]) < 1e-15);
    fp_workspace_free(ws);
    fp_op_free(op);

    // Graph TV on the 4-cycle 0-1-2-3-0 at x = (1, 1, -1, -1): the solver's
    // flow saturates edges (1, 2) and (3, 0); both kept apart split the cycle.
    const int64_t cyc[8] = {0, 1, 1, 2, 2, 3, 3, 0};
    CHECK_OK(fp_op_graph_tv(4, 4, cyc, nullptr, &op));
    Arr xc({4}, {1.0, 1.0, -1.0, -1.0}), zc({4}), dc({4});
    CHECK_OK(fp_workspace_create(op, &xc.a, &o, &ws));
    CHECK_OK(fp_prox(op, 1.0, &xc.a, &zc.a, &o, ws, nullptr));
    I8 mc({4}, {1}, 4), both({4}, {1}, 4), one({4}, {1}, 4);
    CHECK_OK(fp_workspace_kinks(ws, 1e-10, &mc.a, &count));
    CHECK(count == 2 && mc.data == std::vector<int8_t>({0, 1, 0, 1}));
    both.data = {0, 1, 0, 1};
    one.data = {0, 1, 0, 0};
    CHECK_OK(fp_workspace_set_branch(ws, &one.a, 1e-10));
    CHECK_OK(fp_jvp(op, ws, &xc.a, 0.0, nullptr, &dc.a));
    CHECK(std::fabs(dc.data[0]) < 1e-12 && std::fabs(dc.data[3]) < 1e-12);
    CHECK_OK(fp_workspace_set_branch(ws, &both.a, 1e-10));
    CHECK_OK(fp_jvp(op, ws, &xc.a, 0.0, nullptr, &dc.a));
    for (int i = 0; i < 4; ++i) CHECK(std::fabs(dc.data[i] - xc.data[i]) < 1e-10);
    fp_workspace_free(ws);
    fp_op_free(op);
}

static void test_kinks() {
    // Soft thresholding at gamma lambda = 0.4: entries on +-0.4, near it, and away.
    fp_op *op;
    CHECK_OK(fp_op_soft(0.5, nullptr, &op));
    Arr x({2, 3}, {0.4, -0.4, 0.4 + 1e-9, 1.0, 0.0, -0.3}), z({2, 3}), d({2, 3}), ones({2, 3}, Vec(6, 1.0));
    fp_options o = opts(-1, 0, 1);
    fp_workspace *ws;
    CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
    int64_t count = -1;
    CHECK(fp_workspace_kinks(ws, 0.0, nullptr, &count) == FP_EINVAL);  // no kept call yet
    CHECK_OK(fp_prox(op, 0.8, &x.a, &z.a, &o, ws, nullptr));
    // The mask through negative strides: element (i, j) at offset 11 - 4 i - 2 j.
    I8 m({2, 3}, {-4, -2}, 12, 11);
    CHECK_OK(fp_workspace_kinks(ws, 0.0, &m.a, &count));
    const int8_t *mp = static_cast<const int8_t *>(m.a.data);
    CHECK(count == 2 && mp[0] == 1 && mp[-2] == 1 && mp[-4] == 0 && mp[-6] == 0);
    CHECK_OK(fp_workspace_kinks(ws, 1e-8, nullptr, &count));
    CHECK(count == 3);
    CHECK(fp_workspace_kinks(ws, -1.0, nullptr, &count) == FP_EPARAM);
    // An int8 array is not an input of the prox.
    I8 xi({2, 3}, {3, 1}, 6);
    CHECK(fp_prox(op, 0.8, &xi.a, &z.a, &o, ws, nullptr) == FP_EDTYPE);
    // Sides: +1 takes the shifted branch (1) at +0.4 and the thresholded one (0)
    // at -0.4, -1 the reverse; the near kink follows only with its tolerance.
    CHECK_OK(fp_prox(op, 0.8, &x.a, &z.a, &o, ws, nullptr));
    I8 up({2, 3}, {3, 1}, 6), dn({2, 3}, {3, 1}, 6), bad({2, 3}, {3, 1}, 6);
    for (auto &v : up.data) v = 1;
    for (auto &v : dn.data) v = -1;
    bad.data[4] = 2;
    CHECK_OK(fp_workspace_set_branch(ws, &up.a, 0.0));
    CHECK_OK(fp_jvp(op, ws, &ones.a, 0.0, nullptr, &d.a));
    CHECK(d.data == Vec({1.0, 0.0, 1.0, 1.0, 0.0, 0.0}));
    CHECK_OK(fp_workspace_set_branch(ws, &dn.a, 0.0));
    CHECK_OK(fp_jvp(op, ws, &ones.a, 0.0, nullptr, &d.a));
    CHECK(d.data == Vec({0.0, 1.0, 1.0, 1.0, 0.0, 0.0}));
    CHECK_OK(fp_workspace_set_branch(ws, &dn.a, 1e-8));
    CHECK_OK(fp_jvp(op, ws, &ones.a, 0.0, nullptr, &d.a));
    CHECK(d.data == Vec({0.0, 1.0, 0.0, 1.0, 0.0, 0.0}));
    double gg = 0.0;
    Arr gp({1}), gx({2, 3});
    CHECK_OK(fp_vjp(op, ws, &ones.a, &gx.a, &gg, &gp.a));
    CHECK(gx.data == d.data);
    CHECK(fp_workspace_set_branch(ws, &bad.a, 0.0) == FP_EPARAM);
    // A prox call clears the override, and so does NULL.
    CHECK_OK(fp_workspace_set_branch(ws, &up.a, 0.0));
    CHECK_OK(fp_prox(op, 0.8, &x.a, &z.a, &o, ws, nullptr));
    CHECK_OK(fp_jvp(op, ws, &ones.a, 0.0, nullptr, &d.a));
    CHECK(d.data == Vec({0.0, 0.0, 1.0, 1.0, 0.0, 0.0}));
    CHECK_OK(fp_workspace_set_branch(ws, &up.a, 0.0));
    CHECK_OK(fp_workspace_set_branch(ws, nullptr, 0.0));
    CHECK_OK(fp_jvp(op, ws, &ones.a, 0.0, nullptr, &d.a));
    CHECK(d.data == Vec({0.0, 0.0, 1.0, 1.0, 0.0, 0.0}));
    I8 wrong({3, 2}, {2, 1}, 6);
    CHECK(fp_workspace_set_branch(ws, &wrong.a, 0.0) == FP_ESHAPE);
    fp_workspace_free(ws);

    // A workspace without keep_structure, and an operator without kinks.
    fp_options o0 = opts();
    CHECK_OK(fp_workspace_create(op, &x.a, &o0, &ws));
    CHECK_OK(fp_prox(op, 0.8, &x.a, &z.a, &o0, ws, nullptr));
    CHECK(fp_workspace_kinks(ws, 0.0, nullptr, &count) == FP_EINVAL);
    fp_workspace_free(ws);
    fp_op_free(op);
    CHECK_OK(fp_op_huber(0.5, 0.3, &op));
    CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
    CHECK_OK(fp_prox(op, 0.8, &x.a, &z.a, &o, ws, nullptr));
    CHECK(fp_workspace_kinks(ws, 0.0, nullptr, &count) == FP_EINVAL);
    CHECK(fp_workspace_set_branch(ws, &up.a, 0.0) == FP_EINVAL);
    fp_workspace_free(ws);
    fp_op_free(op);

    // MCP with weights in a batch large enough for threads: the same mask on
    // 1 and 8 threads.
    Arr wv({50}, Vec(50, 1.0));
    for (int i = 0; i < 50; ++i) wv.data[static_cast<size_t>(i)] = 0.5 + 0.02 * i;
    CHECK_OK(fp_op_mcp(0.5, 3.0, &wv.a, &op));
    Arr xb({1000, 50}, randn(50000));
    for (int r = 0; r < 1000; r += 3)
        for (int i = 0; i < 50; i += 2) xb.data[static_cast<size_t>(r * 50 + i)] = 0.8 * wv.data[static_cast<size_t>(i)] * 0.5;
    Arr zb({1000, 50});
    std::vector<std::vector<int8_t>> masks;
    for (int t : {1, 8}) {
        fp_options ot = opts(-1, t, 1);
        CHECK_OK(fp_workspace_create(op, &xb.a, &ot, &ws));
        CHECK_OK(fp_prox(op, 0.8, &xb.a, &zb.a, &ot, ws, nullptr));
        I8 mb({1000, 50}, {50, 1}, 50000);
        CHECK_OK(fp_workspace_kinks(ws, 0.0, &mb.a, &count));
        CHECK(count >= 334 * 25);
        masks.push_back(mb.data);
        fp_workspace_free(ws);
    }
    CHECK(masks[0] == masks[1]);
    fp_op_free(op);

    // Since 0.3: the tie of l0 at |x| = sqrt(2 gamma lambda) = 1 is a kink, and
    // its sides are the zero and identity pieces.
    {
        fp_op *l0;
        CHECK_OK(fp_op_l0(0.625, nullptr, &l0));
        Arr xl({4}, {1.0, -1.0, 0.5, 2.0}), zl({4}), dl({4}), one4({4}, Vec(4, 1.0));
        fp_options ol = opts(-1, 0, 1);
        fp_workspace *wl;
        CHECK_OK(fp_workspace_create(l0, &xl.a, &ol, &wl));
        CHECK_OK(fp_prox(l0, 0.8, &xl.a, &zl.a, &ol, wl, nullptr));
        I8 ml({4}, {1}, 4), su({4}, {1}, 4);
        CHECK_OK(fp_workspace_kinks(wl, 0.0, &ml.a, &count));
        CHECK(count == 2 && ml.data == std::vector<int8_t>({1, 1, 0, 0}));
        for (auto &v : su.data) v = 1;
        CHECK_OK(fp_workspace_set_branch(wl, &su.a, 0.0));
        CHECK_OK(fp_jvp(l0, wl, &one4.a, 0.0, nullptr, &dl.a));
        CHECK(dl.data == Vec({1.0, 0.0, 0.0, 1.0}));  // x + eps: the identity at +1, zero at -1
        fp_workspace_free(wl);
        fp_op_free(l0);
    }
    // 1D, anisotropic and isotropic TV: rows (0, 1) at gamma = 1/2 have the
    // mean as their solution with the dual exactly on its bound, a kink at the
    // tail of the edge; (0, 3) does not. Branch overrides for 1D and
    // anisotropic TV (since 0.3).
    {
        Arr xt({2, 2}, {0.0, 1.0, 0.0, 3.0}), zt({2, 2});
        fp_op *tv, *an, *is;
        CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv));
        Arr w01({2}, {0.0, 1.0});
        CHECK_OK(fp_op_tv_aniso(2, &w01.a, &an));
        CHECK_OK(fp_op_tv_iso(2, &w01.a, &is));
        const int8_t bit[3] = {1, 2, 1};  // aniso: bit 1 for an edge along axis 1
        fp_op *ops3[3] = {tv, an, is};
        for (int j = 0; j < 3; ++j) {
            fp_options ot = opts(-1, 0, 1);
            fp_workspace *wt;
            CHECK_OK(fp_workspace_create(ops3[j], &xt.a, &ot, &wt));
            I8 mt({2, 2}, {2, 1}, 4), st({2, 2}, {2, 1}, 4);
            CHECK(fp_workspace_kinks(wt, 0.0, &mt.a, &count) == FP_EINVAL);  // no kept call yet
            CHECK_OK(fp_prox(ops3[j], 0.5, &xt.a, &zt.a, &ot, wt, nullptr));
            CHECK_OK(fp_workspace_kinks(wt, 0.0, &mt.a, &count));
            CHECK(count == 1 && mt.data == std::vector<int8_t>({bit[j], 0, 0, 0}));
            if (j == 2) {  // isotropic TV: no branch overrides yet
                CHECK(fp_workspace_set_branch(wt, &st.a, 0.0) == FP_EINVAL);
                fp_workspace_free(wt);
                continue;
            }
            // Side -1 joins the kink's edge (the mean of the first row), +1 keeps
            // it apart (the identity there); NULL and a new prox call restore the
            // solver's choice.
            Arr dxt({2, 2}, {1.0, -1.0, 0.5, 0.25}), jt({2, 2}), j0({2, 2});
            CHECK_OK(fp_jvp(ops3[j], wt, &dxt.a, 0.0, nullptr, &j0.a));
            for (auto &v : st.data) v = -1;
            CHECK_OK(fp_workspace_set_branch(wt, &st.a, 0.0));
            CHECK_OK(fp_jvp(ops3[j], wt, &dxt.a, 0.0, nullptr, &jt.a));
            CHECK(jt.data[0] == 0.0 && jt.data[1] == 0.0);
            for (auto &v : st.data) v = 1;
            CHECK_OK(fp_workspace_set_branch(wt, &st.a, 0.0));
            CHECK_OK(fp_jvp(ops3[j], wt, &dxt.a, 0.0, nullptr, &jt.a));
            CHECK(jt.data[0] == 1.0 && jt.data[1] == -1.0);
            CHECK(jt.data[2] == j0.data[2] && jt.data[3] == j0.data[3]);  // no kink in row 2
            CHECK_OK(fp_workspace_set_branch(wt, nullptr, 0.0));
            CHECK_OK(fp_jvp(ops3[j], wt, &dxt.a, 0.0, nullptr, &jt.a));
            CHECK(jt.data == j0.data);
            CHECK_OK(fp_workspace_set_branch(wt, &st.a, 0.0));
            CHECK_OK(fp_prox(ops3[j], 0.5, &xt.a, &zt.a, &ot, wt, nullptr));
            CHECK_OK(fp_jvp(ops3[j], wt, &dxt.a, 0.0, nullptr, &jt.a));
            CHECK(jt.data == j0.data);
            st.data[1] = 2;
            CHECK(fp_workspace_set_branch(wt, &st.a, 0.0) == FP_EPARAM);
            fp_workspace_free(wt);
        }
        for (fp_op *o3 : ops3) fp_op_free(o3);
    }
}

// ---------------------------------------------------------------- graph TV
// A ring over n nodes plus `chords` random chords, as 2m endpoints.
static std::vector<int64_t> graph_edges(int64_t n, int64_t chords, unsigned seed) {
    std::mt19937_64 g(seed);
    std::vector<int64_t> e;
    for (int64_t i = 0; i < n && n > 2; ++i) {
        e.push_back(i);
        e.push_back((i + 1) % n);
    }
    for (int64_t k = 0; k < chords && n > 1; ++k) {
        const int64_t a = static_cast<int64_t>(g() % static_cast<uint64_t>(n));
        const int64_t b = (a + 1 + static_cast<int64_t>(g() % static_cast<uint64_t>(n - 1))) % n;
        e.push_back(a);
        e.push_back(b);
    }
    return e;
}

static void test_graph_tv() {
    fp_info info;
    // Chains are weighted 1D TV.
    for (int64_t n : {2, 9, 200, 3000})
        for (double gamma : {0.01, 0.5, 4.0}) {
            Vec x = randn(n, 2.0), w(static_cast<size_t>(n - 1));
            for (int64_t i = 1; i < n; ++i) x[i] += x[i - 1];
            std::vector<int64_t> e;
            for (int64_t i = 0; i + 1 < n; ++i) {
                e.push_back(i);
                e.push_back(i + 1);
                w[i] = unif(0.0, 2.0);
            }
            Arr wa({n - 1}, w), xa({n}, x), za({n});
            fp_op *g, *t;
            CHECK_OK(fp_op_graph_tv(n, n - 1, e.data(), &wa.a, &g));
            CHECK_OK(fp_op_tv1d(&wa.a, FP_TV1D_AUTO, &t));
            fp_options o = opts();
            o.tol = 0.0;
            CHECK_OK(fp_prox(g, gamma, &xa.a, &za.a, &o, nullptr, &info));
            const Vec zt = prox(t, gamma, x);
            double err = 0;
            for (int64_t i = 0; i < n; ++i) err = std::max(err, std::fabs(za.data[i] - zt[i]));
            CHECK(err <= 1e-12 * (1 + maxabs(x)));
            CHECK(info.gap <= 1e-12 * (1 + maxabs(x)));
            CHECK(info.iters >= 1 || n == 2);
            fp_op_free(g);
            fp_op_free(t);
        }
    // 4-neighbour grids are anisotropic TV.
    for (std::vector<int64_t> sh : {std::vector<int64_t>{17, 23}, std::vector<int64_t>{40, 9}}) {
        const int64_t N = sh[0] * sh[1];
        std::vector<int64_t> e;
        for (int64_t i = 0; i < sh[0]; ++i)
            for (int64_t j = 0; j < sh[1]; ++j) {
                if (j + 1 < sh[1]) e.insert(e.end(), {i * sh[1] + j, i * sh[1] + j + 1});
                if (i + 1 < sh[0]) e.insert(e.end(), {i * sh[1] + j, (i + 1) * sh[1] + j});
            }
        Vec x = randn(N);
        for (int64_t i = 0; i < N; ++i) x[i] += (i / 37) % 2;
        Arr xa({N}, x), za({N}), xi(sh, x), zi(sh);
        fp_op *g, *a;
        CHECK_OK(fp_op_graph_tv(N, static_cast<int64_t>(e.size()) / 2, e.data(), nullptr, &g));
        CHECK_OK(fp_op_tv_aniso(2, nullptr, &a));
        fp_options o = opts();
        o.tol = 0.0;
        CHECK_OK(fp_prox(g, 0.3, &xa.a, &za.a, &o, nullptr, &info));
        fp_options oa = opts();
        oa.tol = 1e-15;
        oa.max_iter = 1000000;
        CHECK_OK(fp_prox(a, 0.3, &xi.a, &zi.a, &oa, nullptr, nullptr));
        double err = 0;
        for (int64_t i = 0; i < N; ++i) err = std::max(err, std::fabs(za.data[i] - zi.data[i]));
        CHECK(err <= 1e-7);
        // Precision: within tol * (max x - min x) of the exact result.
        double lo = x[0], hi = x[0];
        for (double v : x) lo = std::min(lo, v), hi = std::max(hi, v);
        for (double tol : {1e-9, 1e-4, 1e-1}) {
            Arr zt({N});
            fp_options ot = opts();
            ot.tol = tol;
            fp_info it;
            CHECK_OK(fp_prox(g, 0.3, &xa.a, &zt.a, &ot, nullptr, &it));
            double d = 0;
            for (int64_t i = 0; i < N; ++i) d = std::max(d, std::fabs(zt.data[i] - za.data[i]));
            CHECK(d <= tol * (hi - lo));
            CHECK(it.iters <= info.iters);
        }
        // max_iter stops early and says so; every node still has a value.
        fp_options o1 = opts();
        o1.tol = 0.0;
        o1.max_iter = 1;
        CHECK(fp_prox(g, 0.3, &xa.a, &za.a, &o1, nullptr, &info) == FP_ENOTCONV);
        bool finite_all = true;
        for (double v : za.data) finite_all = finite_all && std::isfinite(v);
        CHECK(finite_all && info.iters == 1);
        fp_op_free(g);
        fp_op_free(a);
    }
    // Derivatives: adjointness and finite differences, unit and per-edge weights.
    static const std::vector<int64_t> e12 = graph_edges(12, 8, 5);
    static Arr w12({static_cast<int64_t>(e12.size()) / 2},
                   Vec(e12.size() / 2, 0.8));
    for (size_t k = 0; k < w12.data.size(); ++k) w12.data[k] = 0.4 + 0.1 * static_cast<double>(k % 7);
    const std::vector<OpCase> cases = {
        {"graph_tv", [](fp_op **o) { return fp_op_graph_tv(12, static_cast<int64_t>(e12.size()) / 2, e12.data(), nullptr, o); }, 0.6},
        {"graph_tv_w", [](fp_op **o) { return fp_op_graph_tv(12, static_cast<int64_t>(e12.size()) / 2, e12.data(), &w12.a, o); }, 0.6},
    };
    for (int rep = 0; rep < 10; ++rep)
        for (const auto &c : cases) {
            check_derivatives(c, false);
            check_derivatives(c, true);
        }
    // Batches: each signal on its own, for any thread count.
    {
        const std::vector<int64_t> e = graph_edges(40, 70, 9);
        const int64_t m = static_cast<int64_t>(e.size()) / 2;
        fp_op *g;
        CHECK_OK(fp_op_graph_tv(40, m, e.data(), nullptr, &g));
        Arr x({50, 40}, randn(2000, 2.0));
        std::vector<Vec> outs;
        for (int t : {1, 3, 8}) {
            Arr z({50, 40});
            fp_options o = opts(-1, t);
            o.tol = 0.0;
            CHECK_OK(fp_prox(g, 0.8, &x.a, &z.a, &o, nullptr, nullptr));
            outs.push_back(z.data);
        }
        CHECK(outs[0] == outs[1] && outs[0] == outs[2]);
        Vec row(x.data.begin() + 40 * 7, x.data.begin() + 40 * 8);
        Arr ra({40}, row), rz({40});
        fp_options o = opts();
        o.tol = 0.0;
        CHECK_OK(fp_prox(g, 0.8, &ra.a, &rz.a, &o, nullptr, nullptr));
        CHECK(std::equal(rz.data.begin(), rz.data.end(), outs[0].begin() + 40 * 7));
        // The wrong length, and infinities.
        Arr bad({41});
        CHECK(fp_prox(g, 0.8, &bad.a, &bad.a, nullptr, nullptr, nullptr) == FP_ESHAPE);
        ra.data[3] = INFINITY;
        CHECK(fp_prox(g, 0.8, &ra.a, &rz.a, nullptr, nullptr, nullptr) == FP_EINVAL);
        ra.data[3] = NAN;
        CHECK_OK(fp_prox(g, 0.8, &ra.a, &rz.a, nullptr, nullptr, nullptr));
        CHECK(std::isnan(rz.data[0]) && std::isnan(rz.data[39]));
        fp_op_free(g);
    }
    // Constructor errors.
    {
        fp_op *g = nullptr;
        const int64_t loop[] = {0, 1, 1, 1}, out[] = {0, 3}, ok[] = {0, 1, 1, 2};
        CHECK(fp_op_graph_tv(3, 2, loop, nullptr, &g) == FP_EPARAM);
        CHECK(fp_op_graph_tv(3, 1, out, nullptr, &g) == FP_EPARAM);
        CHECK(fp_op_graph_tv(3, 1, nullptr, nullptr, &g) == FP_EINVAL);
        CHECK(fp_op_graph_tv(-1, 0, nullptr, nullptr, &g) == FP_EINVAL);
        Arr neg({2}, {1.0, -1.0}), three({3}, {1.0, 1.0, 1.0});
        CHECK(fp_op_graph_tv(3, 2, ok, &neg.a, &g) == FP_EPARAM);
        CHECK(fp_op_graph_tv(3, 2, ok, &three.a, &g) == FP_ESHAPE);
        CHECK(g == nullptr);
        CHECK_OK(fp_op_graph_tv(3, 0, nullptr, nullptr, &g));
        CHECK(fp_op_nparams(g) == 1 && std::string(fp_op_name(g)) == "graph_tv");
        fp_op_free(g);
    }
}

// ---------------------------------------------------------------- tree TV
// A random forest on n nodes as 2m endpoints: each node after the first joins
// an earlier one (kind 0), the one before it (a path, kind 1) or node 0 (a
// star, kind 2); then the labels are shuffled, the ends swapped at random,
// the edges shuffled, and `drop` of them removed.
static std::vector<int64_t> tree_edges(int64_t n, int kind, int64_t drop, unsigned seed) {
    std::mt19937_64 g(seed);
    std::vector<int64_t> label(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) label[i] = i;
    std::shuffle(label.begin(), label.end(), g);
    std::vector<std::pair<int64_t, int64_t>> es;
    for (int64_t i = 1; i < n; ++i) {
        const int64_t p = kind == 1 ? i - 1 : (kind == 2 ? 0 : static_cast<int64_t>(g() % static_cast<uint64_t>(i)));
        if (g() % 2) es.emplace_back(label[i], label[p]);
        else es.emplace_back(label[p], label[i]);
    }
    std::shuffle(es.begin(), es.end(), g);
    es.resize(es.size() - static_cast<size_t>(std::min<int64_t>(drop, static_cast<int64_t>(es.size()))));
    std::vector<int64_t> e;
    for (auto &pq : es) e.insert(e.end(), {pq.first, pq.second});
    return e;
}

static void test_tree_tv() {
    fp_info info, ig;
    // Against exact graph TV on random trees, paths, stars and forests, with
    // some zero weights; the gap is zero up to rounding.
    for (int kind : {0, 1, 2})
        for (int64_t n : {1, 2, 3, 10, 200, 3000})
            for (int64_t drop : {0, 3})
                for (double gamma : {0.01, 0.5, 4.0}) {
                    const std::vector<int64_t> e = tree_edges(n, kind, drop, static_cast<unsigned>(n + kind));
                    const int64_t m = static_cast<int64_t>(e.size()) / 2;
                    Vec x = randn(n, 2.0), w(static_cast<size_t>(std::max<int64_t>(m, 1)));
                    for (auto &v : w) v = unif(0.0, 1.0) < 0.1 ? 0.0 : unif(0.0, 2.0);
                    Arr wa({static_cast<int64_t>(w.size())}, w), xa({n}, x), za({n}), zg({n});
                    fp_op *t, *g;
                    CHECK_OK(fp_op_tree_tv(n, m, e.data(), m ? &wa.a : nullptr, &t));
                    CHECK_OK(fp_op_graph_tv(n, m, e.data(), m ? &wa.a : nullptr, &g));
                    fp_options o = opts();
                    o.tol = 0.0;
                    CHECK_OK(fp_prox(t, gamma, &xa.a, &za.a, &o, nullptr, &info));
                    CHECK_OK(fp_prox(g, gamma, &xa.a, &zg.a, &o, nullptr, &ig));
                    double err = 0, x2 = 0;
                    for (int64_t i = 0; i < n; ++i) err = std::max(err, std::fabs(za.data[i] - zg.data[i]));
                    for (double v : x) x2 += v * v;
                    CHECK(err <= 1e-12 * (1 + maxabs(x)));
                    CHECK(info.gap <= 1e-13 * (1 + x2));
                    CHECK(info.iters == 1);
                    fp_op_free(t);
                    fp_op_free(g);
                }
    // Paths are weighted 1D TV, in any node order.
    for (int64_t n : {2, 9, 200, 3000})
        for (double gamma : {0.01, 0.5, 4.0}) {
            Vec x = randn(n, 2.0), w(static_cast<size_t>(n - 1));
            for (int64_t i = 1; i < n; ++i) x[i] += x[i - 1];
            std::vector<int64_t> perm(static_cast<size_t>(n)), e;
            for (int64_t i = 0; i < n; ++i) perm[i] = i;
            std::shuffle(perm.begin(), perm.end(), rng);
            Vec xp(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) xp[perm[i]] = x[i];
            for (int64_t i = 0; i + 1 < n; ++i) {
                e.insert(e.end(), {perm[i], perm[i + 1]});
                w[i] = unif(0.0, 1.0) < 0.05 ? 0.0 : unif(0.0, 2.0);
            }
            Arr wa({n - 1}, w), xa({n}, xp), za({n});
            fp_op *t, *d;
            CHECK_OK(fp_op_tree_tv(n, n - 1, e.data(), &wa.a, &t));
            CHECK_OK(fp_op_tv1d(&wa.a, FP_TV1D_AUTO, &d));
            CHECK_OK(fp_prox(t, gamma, &xa.a, &za.a, nullptr, nullptr, &info));
            const Vec zd = prox(d, gamma, x);
            double err = 0;
            for (int64_t i = 0; i < n; ++i) err = std::max(err, std::fabs(za.data[perm[i]] - zd[i]));
            CHECK(err <= 1e-12 * (1 + maxabs(x)));
            fp_op_free(t);
            fp_op_free(d);
        }
    // Derivatives on a tree and a forest, unit and per-edge weights.
    static const std::vector<int64_t> t12 = tree_edges(12, 0, 0, 3), f12 = tree_edges(12, 0, 2, 4);
    static Arr w11({11}, Vec(11, 0.8));
    for (size_t k = 0; k < 11; ++k) w11.data[k] = 0.4 + 0.1 * static_cast<double>(k % 7);
    const std::vector<OpCase> cases = {
        {"tree_tv", [](fp_op **o) { return fp_op_tree_tv(12, 11, t12.data(), nullptr, o); }, 0.6},
        {"tree_tv_w", [](fp_op **o) { return fp_op_tree_tv(12, 11, t12.data(), &w11.a, o); }, 0.6},
        {"tree_tv_forest", [](fp_op **o) { return fp_op_tree_tv(12, 9, f12.data(), nullptr, o); }, 0.6},
    };
    for (int rep = 0; rep < 10; ++rep)
        for (const auto &c : cases) {
            check_derivatives(c, false);
            check_derivatives(c, true);
        }
    // Batches: each signal on its own, for any thread count.
    {
        const std::vector<int64_t> e = tree_edges(40, 0, 2, 9);
        const int64_t m = static_cast<int64_t>(e.size()) / 2;
        fp_op *t;
        CHECK_OK(fp_op_tree_tv(40, m, e.data(), nullptr, &t));
        Arr x({50, 40}, randn(2000, 2.0));
        std::vector<Vec> outs;
        for (int th : {1, 3, 8}) {
            Arr z({50, 40});
            fp_options o = opts(-1, th);
            CHECK_OK(fp_prox(t, 0.8, &x.a, &z.a, &o, nullptr, nullptr));
            outs.push_back(z.data);
        }
        CHECK(outs[0] == outs[1] && outs[0] == outs[2]);
        Vec row(x.data.begin() + 40 * 7, x.data.begin() + 40 * 8);
        Arr ra({40}, row), rz({40});
        CHECK_OK(fp_prox(t, 0.8, &ra.a, &rz.a, nullptr, nullptr, nullptr));
        CHECK(std::equal(rz.data.begin(), rz.data.end(), outs[0].begin() + 40 * 7));
        // The wrong length, and infinities.
        Arr bad({41});
        CHECK(fp_prox(t, 0.8, &bad.a, &bad.a, nullptr, nullptr, nullptr) == FP_ESHAPE);
        ra.data[3] = INFINITY;
        CHECK(fp_prox(t, 0.8, &ra.a, &rz.a, nullptr, nullptr, nullptr) == FP_EINVAL);
        ra.data[3] = NAN;
        CHECK_OK(fp_prox(t, 0.8, &ra.a, &rz.a, nullptr, nullptr, nullptr));
        CHECK(std::isnan(rz.data[0]) && std::isnan(rz.data[39]));
        fp_op_free(t);
    }
    // Constructor errors: cycles and repeated edges are not forests.
    {
        fp_op *t = nullptr;
        const int64_t tri[] = {0, 1, 1, 2, 2, 0}, twice[] = {0, 1, 1, 0}, loop[] = {0, 1, 1, 1}, ok[] = {0, 1, 1, 2};
        CHECK(fp_op_tree_tv(3, 3, tri, nullptr, &t) == FP_EPARAM);
        CHECK(fp_op_tree_tv(3, 2, twice, nullptr, &t) == FP_EPARAM);
        CHECK(fp_op_tree_tv(3, 2, loop, nullptr, &t) == FP_EPARAM);
        CHECK(fp_op_tree_tv(3, 1, nullptr, nullptr, &t) == FP_EINVAL);
        Arr neg({2}, {1.0, -1.0}), three({3}, {1.0, 1.0, 1.0});
        CHECK(fp_op_tree_tv(3, 2, ok, &neg.a, &t) == FP_EPARAM);
        CHECK(fp_op_tree_tv(3, 2, ok, &three.a, &t) == FP_ESHAPE);
        CHECK(t == nullptr);
        CHECK_OK(fp_op_tree_tv(3, 2, ok, nullptr, &t));
        CHECK(fp_op_nparams(t) == 1 && std::string(fp_op_name(t)) == "tree_tv");
        fp_op_free(t);
    }
}

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
    // One step per row (since 1.1), strided.
    Vec sv(128);
    for (size_t i = 0; i < sv.size(); ++i) sv[i] = 0.2 + 0.01 * static_cast<double>(i);
    const int64_t sshape[1] = {64}, sstride[1] = {2};
    fp_array sa{sv.data(), FP_F64, 1, sshape, sstride, FP_DEVICE_CPU};
    CHECK_OK(fp_prox_steps(op, &sa, &x.a, &z.a, &o, ws));
    const long sb = g_allocs.load();
    for (int k = 0; k < 3; ++k) {
        CHECK_OK(fp_prox_steps(op, &sa, &x.a, &z.a, &o, ws));
        CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, &gp.a));
    }
    CHECK(g_allocs.load() == sb);
    fp_workspace_free(ws);
    fp_op_free(op);

    // Kinks and a branch override of an elementwise map.
    {
        CHECK_OK(fp_op_mcp(0.5, 3.0, nullptr, &op));
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        I8 side({64, 1000}, {1000, 1}, 64000), mask({64, 1000}, {1000, 1}, 64000);
        for (size_t i = 0; i < side.data.size(); ++i) side.data[i] = static_cast<int8_t>(i % 3) - 1;
        Arr gp2({2});
        CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, nullptr));
        const long kb = g_allocs.load();
        for (int k = 0; k < 3; ++k) {
            CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, ws, nullptr));
            int64_t count;
            CHECK_OK(fp_workspace_kinks(ws, 1e-3, &mask.a, &count));
            CHECK_OK(fp_workspace_set_branch(ws, &side.a, 1e-3));
            CHECK_OK(fp_jvp(op, ws, &x.a, 0.1, nullptr, &gx.a));
            CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, &gp2.a));
        }
        CHECK(g_allocs.load() == kb);
        fp_workspace_free(ws);
        fp_op_free(op);
    }

    // Anisotropic TV: a single image (threads inside it) and a batch, with
    // and without spatial weights.
    for (int wtd = 0; wtd < 2; ++wtd)
    for (int iso = 0; iso < 2; ++iso)
    for (std::vector<int64_t> shape : {std::vector<int64_t>{120, 130}, std::vector<int64_t>{6, 30, 40}}) {
        fp_op *iop;
        const std::vector<int64_t> img(shape.end() - 2, shape.end());
        const int64_t Ni = img[0] * img[1];
        Arr pw(img, Vec(static_cast<size_t>(Ni), 0.8)), ew({2, img[0], img[1]}, Vec(static_cast<size_t>(2 * Ni), 0.8));
        if (wtd) CHECK_OK(iso ? fp_op_tv_iso_weighted(2, nullptr, &pw.a, &iop) : fp_op_tv_aniso_weighted(2, nullptr, &ew.a, &iop));
        else CHECK_OK(iso ? fp_op_tv_iso(2, nullptr, &iop) : fp_op_tv_aniso(2, nullptr, &iop));
        int64_t N = 1;
        for (int64_t s : shape) N *= s;
        Arr ix(shape, randn(N)), iz(shape), ig(shape, randn(N)), igx(shape),
            iq({2 + (wtd ? (iso ? Ni : 2 * Ni) : 0)});
        std::vector<int64_t> qs = {2};
        qs.insert(qs.end(), shape.begin(), shape.end());
        Arr dual(qs);
        fp_options io = opts(-1, 4, 1);
        io.tol = 1e-8;
        io.max_iter = 100000;
        fp_workspace *iws;
        CHECK_OK(fp_workspace_create(iop, &ix.a, &io, &iws));
        CHECK_OK(fp_prox(iop, 0.5, &ix.a, &iz.a, &io, iws, nullptr));
        CHECK_OK(fp_workspace_set_warm(iws, 1));
        const long ibefore = g_allocs.load();
        for (int k = 0; k < 3; ++k) {
            CHECK_OK(fp_prox(iop, 0.5, &ix.a, &iz.a, &io, iws, nullptr));
            CHECK_OK(fp_jvp(iop, iws, &ix.a, 0.1, nullptr, &igx.a));
            CHECK_OK(fp_vjp(iop, iws, &ig.a, &igx.a, &gg, &iq.a));
            CHECK_OK(fp_workspace_get_dual(iws, &dual.a));
            CHECK_OK(fp_workspace_set_dual(iws, &dual.a));
        }
        CHECK(g_allocs.load() == ibefore);
        fp_workspace_free(iws);
        fp_op_free(iop);
    }

    // Graph TV: a batch of signals on one graph, with weight gradients.
    {
        const std::vector<int64_t> edges = graph_edges(40, 60, 3);
        const int64_t m = static_cast<int64_t>(edges.size()) / 2;
        Arr wa({m}, Vec(static_cast<size_t>(m), 0.7));
        fp_op *gop;
        CHECK_OK(fp_op_graph_tv(40, m, edges.data(), &wa.a, &gop));
        Arr gxa({64, 40}, randn(2560)), gza({64, 40}), gga({64, 40}, randn(2560)), ggx({64, 40}), gpa({m});
        fp_options go = opts(-1, 4, 1);
        go.tol = 0.0;
        fp_workspace *gws;
        CHECK_OK(fp_workspace_create(gop, &gxa.a, &go, &gws));
        CHECK_OK(fp_prox(gop, 0.5, &gxa.a, &gza.a, &go, gws, nullptr));
        const long gbefore = g_allocs.load();
        for (int k = 0; k < 3; ++k) {
            CHECK_OK(fp_prox(gop, 0.5, &gxa.a, &gza.a, &go, gws, nullptr));
            CHECK_OK(fp_jvp(gop, gws, &gxa.a, 0.1, &gpa.a, &ggx.a));
            CHECK_OK(fp_vjp(gop, gws, &gga.a, &ggx.a, &gg, &gpa.a));
        }
        CHECK(g_allocs.load() == gbefore);
        fp_workspace_free(gws);
        fp_op_free(gop);
    }

    // Tree TV: on a tree (heaps) and on a path (deque), with weight gradients.
    for (int kind : {0, 1}) {
        const std::vector<int64_t> edges = tree_edges(40, kind, 0, 6);
        Arr wa({39}, Vec(39, 0.7));
        fp_op *top;
        CHECK_OK(fp_op_tree_tv(40, 39, edges.data(), &wa.a, &top));
        Arr txa({64, 40}, randn(2560)), tza({64, 40}), tga({64, 40}, randn(2560)), tgx({64, 40}), tpa({39});
        fp_options to = opts(-1, 4, 1);
        fp_workspace *tws;
        CHECK_OK(fp_workspace_create(top, &txa.a, &to, &tws));
        CHECK_OK(fp_prox(top, 0.5, &txa.a, &tza.a, &to, tws, nullptr));
        const long tbefore = g_allocs.load();
        for (int k = 0; k < 3; ++k) {
            CHECK_OK(fp_prox(top, 0.5, &txa.a, &tza.a, &to, tws, nullptr));
            CHECK_OK(fp_jvp(top, tws, &txa.a, 0.1, &tpa.a, &tgx.a));
            CHECK_OK(fp_vjp(top, tws, &tga.a, &tgx.a, &gg, &tpa.a));
        }
        CHECK(g_allocs.load() == tbefore);
        fp_workspace_free(tws);
        fp_op_free(top);
    }
}

// Several callers at once: one of them gets the thread pool and the others
// run serially; every result matches the serial one. After fork() the child
// has none of the parent's pool threads and must build its own.
static void test_concurrent_callers() {
    fp_op *op;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &op));
    const int64_t rows = 64, n = 2000;
    Arr x({rows, n}, randn(rows * n)), ref({rows, n});
    fp_options serial = opts(-1, 1);
    CHECK_OK(fp_prox(op, 0.5, &x.a, &ref.a, &serial, nullptr, nullptr));

    std::deque<Arr> outs;
    for (int k = 0; k < 6; ++k) outs.emplace_back(std::vector<int64_t>{rows, n});
    std::atomic<int> failures{0};
    std::vector<std::thread> callers;
    for (int k = 0; k < 6; ++k)
        callers.emplace_back([&, k] {
            fp_options o = opts(-1, 4);
            for (int rep = 0; rep < 20; ++rep)
                if (fp_prox(op, 0.5, &x.a, &outs[k].a, &o, nullptr, nullptr) != FP_OK ||
                    outs[k].data != ref.data)
                    ++failures;
        });
    for (auto &t : callers) t.join();
    CHECK(failures.load() == 0);

#if !defined(_WIN32) && !defined(FP_TSAN)  // TSan cannot start threads after fork
    Arr z({rows, n});
    fp_options o = opts(-1, 4);
    CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, nullptr, nullptr));  // the pool exists now
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        Arr zc({rows, n});
        const bool ok = fp_prox(op, 0.5, &x.a, &zc.a, &o, nullptr, nullptr) == FP_OK &&
                        zc.data == ref.data;
        _exit(ok ? 0 : 1);
    }
    int status = -1;
    CHECK(pid > 0 && waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#endif
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

// Kept calls that do no work (audit A3): empty batches and sigma = 0 in the
// conjugate save a point whose derivatives work; steps are checked even on
// an empty batch.
static void test_kept_empty_and_identity() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto bits_equal = [](const Vec &a, const Vec &b) {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
    };
    std::vector<std::pair<const char *, std::function<fp_status(fp_op **)>>> makers = {
        {"soft", [](fp_op **o) { return fp_op_soft(1.0, nullptr, o); }},
        {"tv1d", [](fp_op **o) { return fp_op_tv1d(nullptr, FP_TV1D_AUTO, o); }},
        {"tv_iso", [](fp_op **o) { return fp_op_tv_iso(2, nullptr, o); }},
    };
    for (auto &mk : makers) {
        fp_op *op;
        CHECK_OK(mk.second(&op));
        const int64_t P = fp_op_nparams(op);
        for (std::vector<int64_t> shape : {std::vector<int64_t>{0, 5}, {3, 0}, {0, 4, 4}}) {
            Arr x(shape), z(shape), dz(shape);
            fp_options o = opts(-1, 0, 1);
            fp_workspace *ws;
            CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
            CHECK_OK(fp_prox(op, 0.7, &x.a, &z.a, &o, ws, nullptr));
            CHECK_OK(fp_jvp(op, ws, &x.a, 1.0, nullptr, &dz.a));
            Arr gp({std::max<int64_t>(P, 1)}, Vec(static_cast<size_t>(std::max<int64_t>(P, 1)), 7.0));
            double gg = 5.0;
            CHECK_OK(fp_vjp(op, ws, &x.a, &dz.a, &gg, P > 0 ? &gp.a : nullptr));
            CHECK(gg == 0.0);
            if (P > 0)
                for (double v : gp.data) CHECK(v == 0.0);
            int64_t count = -1;
            if (std::strcmp(mk.first, "tv_iso") != 0) {
                CHECK_OK(fp_workspace_kinks(ws, 1e-12, nullptr, &count));
                CHECK(count == 0);
                I8 side(shape, std::vector<int64_t>(shape.size(), 1), 0);
                CHECK_OK(fp_workspace_set_branch(ws, &side.a, 0.0));
            }
            fp_workspace_free(ws);
        }
        fp_op_free(op);
    }
    // sigma = 0: y itself, and the identity's derivatives with a NaN sigma-derivative.
    std::vector<std::pair<std::vector<int64_t>, std::function<fp_status(fp_op **)>>> conj = {
        {{3}, [](fp_op **o) { return fp_op_soft(1.0, nullptr, o); }},
        {{3}, [](fp_op **o) { return fp_op_l2_ball(0.7, o); }},
        {{4, 4}, [](fp_op **o) { return fp_op_tv_iso(2, nullptr, o); }},
    };
    for (auto &c : conj) {
        fp_op *op;
        CHECK_OK(c.second(&op));
        const std::vector<int64_t> &shape = c.first;
        const int64_t n = shape.size() == 1 ? shape[0] : shape[0] * shape[1];
        Arr y(shape, n == 3 ? Vec{0.5, 2.0, -3.0} : randn(n)), z(shape), dx(shape, randn(n)), dz(shape),
            g(shape, randn(n)), gx(shape);
        fp_options o = opts(-1, 0, 1);
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &y.a, &o, &ws));
        CHECK_OK(fp_prox_conj(op, 0.0, &y.a, &z.a, &o, ws, nullptr));
        CHECK(bits_equal(z.data, y.data));
        CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a));
        CHECK(bits_equal(dz.data, dx.data));
        CHECK_OK(fp_jvp(op, ws, &dx.a, 1.0, nullptr, &dz.a));
        for (double v : dz.data) CHECK(std::isnan(v));
        double gg = 0.0;
        CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, nullptr));
        CHECK(bits_equal(gx.data, g.data) && std::isnan(gg));
        if (fp_op_nparams(op) == 0 && n == 3) {
            int64_t count = -1;
            I8 m({3}, {1}, 3);
            m.data.assign(3, 9);
            CHECK_OK(fp_workspace_kinks(ws, 1.0, &m.a, &count));
            CHECK(count == 0 && m.data == std::vector<int8_t>(3, 0));
            I8 side({3}, {1}, 3);
            CHECK_OK(fp_workspace_set_branch(ws, &side.a, 0.0));
        }
        fp_workspace_free(ws);
        fp_op_free(op);
    }
    (void)nan;
    // Steps are checked on an empty batch too.
    fp_op *scad;
    CHECK_OK(fp_op_scad(0.5, 3.7, nullptr, &scad));
    Arr e({0}), e2({0});
    CHECK(fp_prox(scad, 10.0, &e.a, &e2.a, nullptr, nullptr, nullptr) == FP_EPARAM);
    fp_op_free(scad);
    fp_op *tv;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv));
    Arr x30({3, 0}), z30({3, 0}), steps({3}, {1.0, -1.0, 2.0});
    CHECK(fp_prox_steps(tv, &steps.a, &x30.a, &z30.a, nullptr, nullptr) == FP_EPARAM);
    steps.data[1] = 1.0;
    CHECK_OK(fp_prox_steps(tv, &steps.a, &x30.a, &z30.a, nullptr, nullptr));
    fp_op_free(tv);
}

// Image operators ignore fp_options.axis (audit A5).
static void test_image_axis_ignored() {
    const int64_t chain[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
                             12, 13, 13, 14, 14, 15};
    std::vector<std::pair<std::vector<int64_t>, std::function<fp_status(fp_op **)>>> ops = {
        {{4, 4}, [](fp_op **o) { return fp_op_tv_aniso(2, nullptr, o); }},
        {{4, 4}, [](fp_op **o) { return fp_op_tv_iso(2, nullptr, o); }},
        {{16}, [&](fp_op **o) { return fp_op_graph_tv(16, 15, chain, nullptr, o); }},
    };
    for (auto &c : ops) {
        fp_op *op;
        CHECK_OK(c.second(&op));
        Arr x(c.first, randn(16)), z0(c.first), z(c.first), dz(c.first);
        fp_options o = opts(-1);
        CHECK_OK(fp_prox(op, 0.4, &x.a, &z0.a, &o, nullptr, nullptr));
        for (int axis : {5, 0}) {
            o.axis = axis;
            CHECK_OK(fp_prox(op, 0.4, &x.a, &z.a, &o, nullptr, nullptr));
            CHECK(std::memcmp(z.data.data(), z0.data.data(), 16 * sizeof(double)) == 0);
        }
        fp_workspace *ws;
        o.axis = 7;
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        o.axis = -1;
        CHECK_OK(fp_prox(op, 0.4, &x.a, &z.a, &o, ws, nullptr));
        fp_workspace_free(ws);
        o = opts(-1, 0, 1);
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        o.axis = 0;
        CHECK_OK(fp_prox(op, 0.4, &x.a, &z.a, &o, ws, nullptr));
        CHECK_OK(fp_jvp(op, ws, &x.a, 0.0, nullptr, &dz.a));
        fp_workspace_free(ws);
        fp_op_free(op);
    }
}

// Counts too large to allocate (audit A7).
static void test_huge_counts() {
    const int64_t groups[4] = {0, 0, 1, 1};
    fp_op *op = reinterpret_cast<fp_op *>(&op);
    CHECK(fp_op_group_l2(1.0, INT64_MAX / 2, groups, 2, nullptr, &op) == FP_EPARAM && op == nullptr);
    op = reinterpret_cast<fp_op *>(&op);
    CHECK(fp_op_group_huber(1.0, 1.0, INT64_MAX / 2, groups, 2, nullptr, &op) == FP_EPARAM && op == nullptr);
    const int64_t ptr[2] = {0, INT64_MAX / 2}, idx[1] = {0};
    CHECK(fp_op_tree_l2(1.0, 1, ptr, idx, nullptr, &op) == FP_EPARAM);
    CHECK(fp_op_tree_l2(1.0, INT64_MAX, ptr, idx, nullptr, &op) == FP_EPARAM);
    double one = 1.0;
    int64_t shape[1] = {int64_t{1} << 62}, stride[1] = {0};
    fp_array w{&one, FP_F64, 1, shape, stride, FP_DEVICE_CPU};
    CHECK(fp_op_soft(1.0, &w, &op) == FP_ESHAPE);
}

// A failed fp_workspace_set_branch leaves an earlier override (audit A9).
// Images holding a NaN (audit A8): not solved, so a NaN gap and no
// iterations, without making the call FP_ENOTCONV; their NaN dual can be set
// back (stored as 0).
static void test_nan_images() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<int64_t> chain;
    for (int64_t i = 0; i < 15; ++i) chain.insert(chain.end(), {i, i + 1});
    struct Case {
        const char *name;
        std::vector<int64_t> shape;  // two images
        int k;                       // dual blocks
        bool iterative, warm;
        std::function<fp_status(fp_op **)> make;
    };
    std::vector<Case> cases = {
        {"tv_aniso", {2, 4, 4}, 2, true, true, [](fp_op **o) { return fp_op_tv_aniso(2, nullptr, o); }},
        {"tv_iso", {2, 4, 4}, 2, true, true, [](fp_op **o) { return fp_op_tv_iso(2, nullptr, o); }},
        {"graph_tv", {2, 16}, 1, false, true, [&](fp_op **o) { return fp_op_graph_tv(16, 15, chain.data(), nullptr, o); }},
        {"tree_tv", {2, 16}, 1, false, false, [&](fp_op **o) { return fp_op_tree_tv(16, 15, chain.data(), nullptr, o); }},
    };
    for (auto &c : cases) {
        fp_op *op;
        CHECK_OK(c.make(&op));
        Vec x0(32);
        for (int i = 0; i < 32; ++i) x0[static_cast<size_t>(i)] = 3.0 * std::sin(1.7 * (i % 16));
        fp_options o = opts();
        o.tol = 1e-10;
        o.max_iter = 5000;
        auto run = [&](const Vec &x, fp_info &info, fp_options &oo) {
            Arr xa(c.shape, x), za(c.shape);
            fp_workspace *ws;
            CHECK_OK(fp_workspace_create(op, &xa.a, &oo, &ws));
            const fp_status st = fp_prox(op, 0.7, &xa.a, &za.a, &oo, ws, &info);
            fp_info wi;
            CHECK(fp_workspace_info(ws, &wi) == st);
            CHECK((std::isnan(wi.gap) && std::isnan(info.gap)) || wi.gap == info.gap);
            CHECK(wi.iters == info.iters);
            fp_workspace_free(ws);
            return st;
        };
        // Image 0 alone: its iterations.
        std::vector<int64_t> one(c.shape.begin() + 1, c.shape.end());
        Arr x1(one, Vec(x0.begin(), x0.begin() + 16)), z1(one);
        fp_info i0;
        CHECK_OK(fp_prox(op, 0.7, &x1.a, &z1.a, &o, nullptr, &i0));
        Vec x = x0;
        x[20] = nan;
        fp_info info;
        CHECK(run(x, info, o) == FP_OK);
        CHECK(std::isnan(info.gap) && info.iters == i0.iters);
        for (double &v : x) v = nan;
        CHECK(run(x, info, o) == FP_OK);
        CHECK(std::isnan(info.gap) && info.iters == 0);
        if (c.iterative) {  // a real non-convergence is still reported
            x = x0;
            x[20] = nan;
            fp_options o1 = o;
            o1.max_iter = 1;
            CHECK(run(x, info, o1) == FP_ENOTCONV);
            fp_info clean;
            Vec xc = x0;
            CHECK(run(xc, clean, o1) == FP_ENOTCONV);
            CHECK(std::isfinite(clean.gap));
        }
        if (c.warm) {  // the dual of a NaN image can be set back
            x = x0;
            x[20] = nan;
            Arr xa(c.shape, x), za(c.shape), xc(c.shape, x0), zA(c.shape), zB(c.shape);
            std::vector<int64_t> dshape = c.shape;
            dshape.insert(dshape.begin(), c.k);
            Arr dual(dshape);
            fp_workspace *ws;
            CHECK_OK(fp_workspace_create(op, &xa.a, &o, &ws));
            CHECK_OK(fp_workspace_set_warm(ws, 1));
            CHECK_OK(fp_prox(op, 0.7, &xa.a, &za.a, &o, ws, nullptr));
            CHECK_OK(fp_workspace_get_dual(ws, &dual.a));
            const size_t N = 16;
            bool img1_nan = true, img0_finite = true;
            for (int b = 0; b < c.k; ++b)
                for (size_t i = 0; i < N; ++i) {
                    img0_finite = img0_finite && std::isfinite(dual.data[(b * 2 + 0) * N + i]);
                    img1_nan = img1_nan && std::isnan(dual.data[(b * 2 + 1) * N + i]);
                }
            CHECK(img1_nan && img0_finite);
            CHECK_OK(fp_workspace_set_dual(ws, &dual.a));
            CHECK_OK(fp_prox(op, 0.7, &xc.a, &zA.a, &o, ws, nullptr));
            Arr zeroed(dual);
            for (double &v : zeroed.data) v = std::isnan(v) ? 0.0 : v;
            CHECK_OK(fp_workspace_set_dual(ws, &zeroed.a));
            CHECK_OK(fp_prox(op, 0.7, &xc.a, &zB.a, &o, ws, nullptr));
            CHECK(std::memcmp(zA.data.data(), zB.data.data(), 32 * sizeof(double)) == 0);
            zeroed.data[3] = INFINITY;
            CHECK(fp_workspace_set_dual(ws, &zeroed.a) == FP_EINVAL);
            fp_workspace_free(ws);
        }
        fp_op_free(op);
    }
}

// The saved point belongs to the last prox call on the workspace that got
// past its argument checks (audit A4): a call rejected for its arguments
// leaves it, a call without keep_structure discards it, and a kept call
// replaces it. A soft(1) on [0.5, 2, -3] and an anisotropic TV image.
static void test_kept_point_lifetime() {
    struct Case {
        std::vector<int64_t> shape;
        std::function<fp_status(fp_op **)> make;
    };
    std::vector<Case> cases = {
        {{3}, [](fp_op **o) { return fp_op_soft(1.0, nullptr, o); }},
        {{2, 3}, [](fp_op **o) { return fp_op_tv_aniso(2, nullptr, o); }},
    };
    for (auto &c : cases) {
        fp_op *op;
        CHECK_OK(c.make(&op));
        const bool image = c.shape.size() == 2;
        Arr x(c.shape, image ? Vec{0.5, 2.0, -3.0, 1.0, 0.0, 4.0} : Vec{0.5, 2.0, -3.0}), z(c.shape);
        const int64_t n = static_cast<int64_t>(x.data.size());
        Arr dx(c.shape, randn(n)), dz(c.shape), y(c.shape, randn(n)), small({1});
        fp_options keep = opts(-1, 1, 1), plain = opts(-1, 1, 0);
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &x.a, &keep, &ws));
        CHECK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a) == FP_EINVAL);
        CHECK_OK(fp_prox(op, 1.0, &x.a, &z.a, &keep, ws, nullptr));
        CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a));
        const Vec D = dz.data;
        auto same_point = [&] {
            dz.data.assign(dz.data.size(), 0.0);
            CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a));
            CHECK(dz.data == D);
        };
        // Rejected for their arguments: the point stays.
        CHECK(fp_prox(op, -1.0, &y.a, &z.a, &keep, ws, nullptr) == FP_EPARAM);
        same_point();
        CHECK(fp_prox(op, 1e300, &y.a, &z.a, &keep, ws, nullptr) == FP_EPARAM);  // the step range
        same_point();
        CHECK(fp_prox(op, 1.0, &y.a, &small.a, &keep, ws, nullptr) == FP_ESHAPE);
        same_point();
        if (image) {
            fp_options bad = keep;
            bad.tol = -1.0;
            CHECK(fp_prox(op, 1.0, &y.a, &z.a, &bad, ws, nullptr) == FP_EPARAM);
            same_point();
        }
        // A call without keep_structure discards it, at any step.
        CHECK_OK(fp_prox(op, 1.0, &y.a, &z.a, &plain, ws, nullptr));
        CHECK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a) == FP_EINVAL);
        CHECK(fp_vjp(op, ws, &dx.a, &dz.a, nullptr, nullptr) == FP_EINVAL);
        CHECK_OK(fp_prox_conj(op, 0.0, &y.a, &z.a, &plain, ws, nullptr));
        CHECK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a) == FP_EINVAL);
        // A kept call saves a new point.
        CHECK_OK(fp_prox(op, 1.0, &x.a, &z.a, &keep, ws, nullptr));
        same_point();
        fp_workspace_free(ws);
        fp_op_free(op);
    }
}

static void test_failed_branch_keeps_override() {
    const int64_t edge[2] = {0, 1};
    struct Case {
        std::vector<int64_t> shape, sides;
        Vec x;
        double gamma;
        std::function<fp_status(fp_op **)> make;
    };
    std::vector<Case> cases = {
        {{4}, {4}, {1.0, 0.3, -2.0, -1.0}, 1.0, [](fp_op **o) { return fp_op_soft(1.0, nullptr, o); }},
        {{2}, {2}, {0.0, 1.0}, 0.5, [](fp_op **o) { return fp_op_tv1d(nullptr, FP_TV1D_AUTO, o); }},
        {{2, 2}, {2, 2}, {0.0, 1.0, 0.0, 1.0}, 0.5, [](fp_op **o) { return fp_op_tv_aniso(2, nullptr, o); }},
        {{2}, {1}, {0.0, 1.0}, 0.5, [&](fp_op **o) { return fp_op_graph_tv(2, 1, edge, nullptr, o); }},
    };
    for (auto &c : cases) {
        fp_op *op;
        CHECK_OK(c.make(&op));
        const int64_t n = static_cast<int64_t>(c.x.size());
        Arr x(c.shape, c.x), z(c.shape), dx(c.shape, randn(n)), dz(c.shape);
        fp_options o = opts(-1, 1, 1);
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        CHECK_OK(fp_prox(op, c.gamma, &x.a, &z.a, &o, ws, nullptr));
        auto jvp = [&] {
            CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &dz.a));
            return dz.data;
        };
        const Vec D = jvp();
        size_t ns = 1;
        for (int64_t v : c.sides) ns *= static_cast<size_t>(v);
        std::vector<int64_t> st(c.sides.size(), 1);
        for (int i = static_cast<int>(c.sides.size()) - 2; i >= 0; --i) st[i] = st[i + 1] * c.sides[i + 1];
        I8 plus(c.sides, st, ns), minus(c.sides, st, ns), bad(c.sides, st, ns);
        plus.data.assign(ns, 1);
        minus.data.assign(ns, -1);
        CHECK_OK(fp_workspace_set_branch(ws, &plus.a, 1e-6));
        const Vec Ap = jvp();
        CHECK_OK(fp_workspace_set_branch(ws, &minus.a, 1e-6));
        const Vec Am = jvp();
        const bool use_plus = Ap != D;
        CHECK(use_plus || Am != D);  // the point is at a kink
        I8 &side = use_plus ? plus : minus;
        CHECK_OK(fp_workspace_set_branch(ws, &side.a, 1e-6));
        const Vec A = jvp();
        bad.data = side.data;
        bad.data[ns - 1] = 2;
        CHECK(fp_workspace_set_branch(ws, &bad.a, 1e-6) == FP_EPARAM);
        const Vec B = jvp();
        CHECK(std::memcmp(A.data(), B.data(), A.size() * sizeof(double)) == 0);
        fp_workspace_free(ws);
        fp_op_free(op);
    }
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

// ---------------------------------------------------------------- devices (since 1.1)
// The device API on the CPU; tests/c/test_device.cpp covers the plugin loader.
static void test_device_api() {
    CHECK(fp_device_available(FP_DEVICE_CPU) == 1);
    CHECK(fp_device_available(static_cast<fp_device>(10)) == 0);
    CHECK(std::strcmp(fp_strerror(FP_EDEVICE), "device error") == 0);

    // Row parallelism on the CPU: elementwise maps and anisotropic TV split a row.
    fp_op *soft, *tv1, *aniso, *iso;
    CHECK_OK(fp_op_soft(0.5, nullptr, &soft));
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv1));
    CHECK_OK(fp_op_tv_aniso(2, nullptr, &aniso));
    CHECK_OK(fp_op_tv_iso(2, nullptr, &iso));
    const std::pair<fp_op *, int> rows[] = {{soft, 0}, {tv1, 1}, {aniso, 0}, {iso, 1}};
    for (const auto &c : rows) {
        int rpo = -1;
        CHECK_OK(fp_op_device_check(c.first, FP_DEVICE_CPU, &rpo));
        CHECK(rpo == c.second);
        CHECK_OK(fp_op_device_check(c.first, FP_DEVICE_CPU, nullptr));
        CHECK(fp_op_device_check(c.first, static_cast<fp_device>(10), &rpo) == FP_EINVAL);
        if (!fp_device_available(FP_DEVICE_CUDA))
            CHECK(fp_op_device_check(c.first, FP_DEVICE_CUDA, &rpo) == FP_EINVAL);
    }
    CHECK(fp_op_device_check(nullptr, FP_DEVICE_CPU, nullptr) == FP_EINVAL);

    // fp_workspace_info: the last call's info and status, with or without info.
    const int64_t side = 24;
    Arr x({2, side, side}, randn(2 * side * side)), z({2, side, side}), z2({2, side, side});
    fp_options o = opts(-1, 1, 1);
    fp_workspace *ws;
    CHECK_OK(fp_workspace_create(iso, &x.a, &o, &ws));
    fp_info info{}, got{};
    CHECK(fp_workspace_info(ws, &got) == FP_EINVAL);
    void *stream = &got;
    CHECK(fp_workspace_set_stream(ws, nullptr) == FP_EINVAL);
    CHECK(fp_workspace_get_stream(ws, &stream) == FP_EINVAL && stream == &got);
    CHECK_OK(fp_prox(iso, 0.3, &x.a, &z.a, &o, ws, &info));
    CHECK_OK(fp_workspace_info(ws, &got));
    CHECK(got.gap == info.gap && got.iters == info.iters && got.ties == info.ties && info.iters > 0);
    CHECK(fp_workspace_info(ws, nullptr) == FP_EINVAL);
    CHECK_OK(fp_prox(iso, 0.3, &x.a, &z2.a, &o, ws, nullptr));
    CHECK_OK(fp_workspace_info(ws, &got));
    CHECK(got.gap == info.gap && got.iters == info.iters && z2.data == z.data);
    fp_options few = o;
    few.max_iter = 1;
    few.tol = 1e-14;
    CHECK(fp_prox(iso, 0.3, &x.a, &z2.a, &few, ws, nullptr) == FP_ENOTCONV);
    CHECK(fp_workspace_info(ws, &got) == FP_ENOTCONV && got.iters == 1);
    Arr wrong({2, side, side + 1});
    CHECK(fp_prox(iso, 0.3, &wrong.a, &wrong.a, &o, ws, nullptr) == FP_ESHAPE);
    CHECK(fp_workspace_info(ws, &got) == FP_EINVAL);
    fp_workspace_free(ws);

    // fp_vjp_device: the step's gradient in a 0-d array, as fp_vjp's.
    Arr r({5, 9}, randn(45)), rz({5, 9}), g({5, 9}, randn(45)), gx({5, 9}), gx2({5, 9});
    CHECK_OK(fp_workspace_create(tv1, &r.a, &o, &ws));
    CHECK_OK(fp_prox(tv1, 0.4, &r.a, &rz.a, &o, ws, nullptr));
    CHECK_OK(fp_workspace_info(ws, &got));
    CHECK(got.gap == 0.0 && got.iters == 0);
    double gg = 0.0;
    CHECK_OK(fp_vjp(tv1, ws, &g.a, &gx.a, &gg, nullptr));
    double d = -1.0;
    float f = -1.0f;
    fp_array da{&d, FP_F64, 0, nullptr, nullptr, FP_DEVICE_CPU};
    fp_array fa{&f, FP_F32, 0, nullptr, nullptr, FP_DEVICE_CPU};
    CHECK_OK(fp_vjp_device(tv1, ws, &g.a, &gx2.a, &da, nullptr));
    CHECK(d == gg && gx2.data == gx.data && gg != 0.0);
    CHECK_OK(fp_vjp_device(tv1, ws, &g.a, &gx2.a, &fa, nullptr));
    CHECK(f == static_cast<float>(gg));
    CHECK_OK(fp_vjp_device(tv1, ws, &g.a, &gx2.a, nullptr, nullptr));
    Arr one({1}, {7.0});
    CHECK(fp_vjp_device(tv1, ws, &g.a, &gx2.a, &one.a, nullptr) == FP_ESHAPE && one.data[0] == 7.0);
    fp_array cuda = da;
    cuda.device = FP_DEVICE_CUDA;
    CHECK(fp_vjp_device(tv1, ws, &g.a, &gx2.a, &cuda, nullptr) == FP_EINVAL);
    fp_workspace_free(ws);

    // A device array still gives FP_EINVAL on the CPU path.
    fp_array xc = r.a;
    xc.device = FP_DEVICE_CUDA;
    CHECK(fp_prox(tv1, 0.4, &xc, &rz.a, nullptr, nullptr, nullptr) == FP_EINVAL);
    CHECK(fp_workspace_create(tv1, &xc, nullptr, &ws) == FP_EINVAL && ws == nullptr);
    for (fp_op *op : {soft, tv1, aniso, iso}) fp_op_free(op);
}

// ---------------------------------------------------------------- one step per row (since 1.1)
// fp_prox_steps against fp_prox row by row; tests/python/test_steps.py runs
// every operator, this covers what Python does not reach.
static void test_prox_steps() {
    const int64_t R = 5, n = 40;
    Vec steps(R);
    for (int64_t r = 0; r < R; ++r) steps[r] = 0.2 + 0.3 * static_cast<double>(r);
    Arr x({R, n}, randn(R * n, 2.0)), z({R, n}), zr({n}), xr({n});
    Arr sa({R}, steps);
    fp_options o = opts(-1, 1, 1);
    fp_op *ops[3];
    CHECK_OK(fp_op_soft(0.5, nullptr, &ops[0]));
    CHECK_OK(fp_op_mcp(0.6, 3.0, nullptr, &ops[1]));
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &ops[2]));
    for (fp_op *op : ops) {
        fp_workspace *ws, *wr;
        CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
        CHECK_OK(fp_workspace_create(op, &xr.a, &o, &wr));
        CHECK_OK(fp_prox_steps(op, &sa.a, &x.a, &z.a, &o, ws));
        fp_info info;
        CHECK_OK(fp_workspace_info(ws, &info));
        // The kink masks: each row's at its own step.
        I8 mask({R, n}, {n, 1}, R * n), mr({n}, {1}, n);
        int64_t count = 0, total = 0;
        CHECK_OK(fp_workspace_kinks(ws, 0.05, &mask.a, &count));
        for (int64_t r = 0; r < R; ++r) {
            std::copy(x.data.begin() + r * n, x.data.begin() + (r + 1) * n, xr.data.begin());
            CHECK_OK(fp_prox(op, steps[r], &xr.a, &zr.a, &o, wr, nullptr));
            CHECK(std::equal(zr.data.begin(), zr.data.end(), z.data.begin() + r * n));
            int64_t c = 0;
            CHECK_OK(fp_workspace_kinks(wr, 0.05, &mr.a, &c));
            CHECK(std::equal(mr.data.begin(), mr.data.end(), mask.data.begin() + r * n));
            total += c;
        }
        CHECK(count == total && total > 0);
        // A scalar call afterwards uses its own step again.
        CHECK_OK(fp_prox(op, 0.7, &x.a, &z.a, &o, ws, nullptr));
        Arr d1({R, n}), d2({R, n}), ones({R}, Vec(R, 0.7));
        CHECK_OK(fp_jvp(op, ws, nullptr, 1.0, nullptr, &d1.a));
        CHECK_OK(fp_prox_steps(op, &ones.a, &x.a, &z.a, &o, ws));
        CHECK_OK(fp_jvp(op, ws, nullptr, 1.0, nullptr, &d2.a));
        CHECK(d1.data == d2.data);
        fp_workspace_free(ws);
        fp_workspace_free(wr);
    }

    // A single row takes a 0-d step; strides of 0 broadcast one step.
    {
        double g0 = 0.8;
        fp_array s0{&g0, FP_F64, 0, nullptr, nullptr, FP_DEVICE_CPU};
        CHECK_OK(fp_prox_steps(ops[2], &s0, &xr.a, &zr.a, nullptr, nullptr));
        CHECK(zr.data == prox(ops[2], 0.8, xr.data));
        const int64_t bshape[1] = {R}, bstride[1] = {0};
        fp_array sb{&g0, FP_F64, 1, bshape, bstride, FP_DEVICE_CPU};
        Arr z2({R, n});
        CHECK_OK(fp_prox_steps(ops[2], &sb, &x.a, &z.a, nullptr, nullptr));
        CHECK_OK(fp_prox(ops[2], 0.8, &x.a, &z2.a, nullptr, nullptr, nullptr));
        CHECK(z.data == z2.data);
        float gf[R];
        for (int64_t r = 0; r < R; ++r) gf[r] = static_cast<float>(steps[r]);
        fp_array sf{gf, FP_F32, 1, bshape, nullptr, FP_DEVICE_CPU};
        CHECK_OK(fp_prox_steps(ops[0], &sf, &x.a, &z.a, nullptr, nullptr));
    }

    // The spectral core after one step per matrix: each matrix as alone.
    {
        const int64_t k = 4;
        Arr s({R, k}), zs({R, k}), a({R, k, k}, randn(R * k * k)), b({R, k, k});
        for (int64_t r = 0; r < R; ++r)
            for (int64_t i = 0; i < k; ++i) s.data[r * k + i] = 0.3 * static_cast<double>((i + r) % k) + 0.1;
        s.data[k + 1] = s.data[k];  // a tie
        fp_workspace *ws, *wr;
        CHECK_OK(fp_workspace_create(ops[0], &s.a, &o, &ws));
        CHECK_OK(fp_prox_steps(ops[0], &sa.a, &s.a, &zs.a, &o, ws));
        CHECK_OK(fp_spectral_jvp(ops[0], ws, FP_SPECTRAL_SVD, &s.a, &zs.a, &a.a, 0.3, nullptr, &b.a, nullptr));
        Arr s1({k}), z1({k}), a1({k, k}), b1({k, k});
        CHECK_OK(fp_workspace_create(ops[0], &s1.a, &o, &wr));
        for (int64_t r = 0; r < R; ++r) {
            std::copy(s.data.begin() + r * k, s.data.begin() + (r + 1) * k, s1.data.begin());
            std::copy(a.data.begin() + r * k * k, a.data.begin() + (r + 1) * k * k, a1.data.begin());
            CHECK_OK(fp_prox(ops[0], steps[r], &s1.a, &z1.a, &o, wr, nullptr));
            CHECK_OK(fp_spectral_jvp(ops[0], wr, FP_SPECTRAL_SVD, &s1.a, &z1.a, &a1.a, 0.3, nullptr, &b1.a, nullptr));
            CHECK(std::equal(b1.data.begin(), b1.data.end(), b.data.begin() + r * k * k));
        }
        fp_workspace_free(ws);
        fp_workspace_free(wr);
    }

    // Errors: the batch shape, the steps, and the step conditions.
    {
        Arr bad({n}), neg({R}, steps), big({R}, steps);
        neg.data[2] = -0.1;
        big.data[3] = 3.5;  // MCP needs b > gamma
        CHECK(fp_prox_steps(ops[0], nullptr, &x.a, &z.a, nullptr, nullptr) == FP_EINVAL);
        CHECK(fp_prox_steps(ops[0], &bad.a, &x.a, &z.a, nullptr, nullptr) == FP_ESHAPE);
        CHECK(fp_prox_steps(ops[0], &x.a, &x.a, &z.a, nullptr, nullptr) == FP_ESHAPE);
        CHECK(fp_prox_steps(ops[0], &neg.a, &x.a, &z.a, nullptr, nullptr) == FP_EPARAM);
        z.data.assign(z.data.size(), 9.0);
        CHECK(fp_prox_steps(ops[1], &big.a, &x.a, &z.a, nullptr, nullptr) == FP_EPARAM);
        CHECK(z.data == Vec(z.data.size(), 9.0));  // checked before any work
        fp_options o0 = opts(0);
        Arr sx({n}, Vec(n, 0.5));
        CHECK_OK(fp_prox_steps(ops[0], &sx.a, &x.a, &z.a, &o0, nullptr));  // axis 0: n rows
    }
    for (fp_op *op : ops) fp_op_free(op);

    // Image operators take the leading shape.
    {
        fp_op *tv;
        CHECK_OK(fp_op_tv_iso(2, nullptr, &tv));
        Arr im({2, 3, 6, 7}, randn(2 * 3 * 42)), iz({2, 3, 6, 7}), one({6, 7}), oz({6, 7});
        Arr is({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
        CHECK_OK(fp_prox_steps(tv, &is.a, &im.a, &iz.a, nullptr, nullptr));
        for (int64_t r = 0; r < 6; ++r) {
            std::copy(im.data.begin() + r * 42, im.data.begin() + (r + 1) * 42, one.data.begin());
            CHECK_OK(fp_prox(tv, is.data[r], &one.a, &oz.a, nullptr, nullptr, nullptr));
            CHECK(std::equal(oz.data.begin(), oz.data.end(), iz.data.begin() + r * 42));
        }
        Arr is2({6});
        CHECK(fp_prox_steps(tv, &is2.a, &im.a, &iz.a, nullptr, nullptr) == FP_ESHAPE);
        fp_op_free(tv);
    }
}

// ---------------------------------------------------------------- conjugate and step range
// fp_prox_conj at extreme sigma, and the step range of every call (audit A1).
template <class T>
static fp_status conj_call(fp_op *op, double sigma, const std::vector<int64_t> &shape, const Vec &y, std::vector<T> &z,
                           fp_workspace *ws = nullptr) {
    std::vector<T> yt(y.begin(), y.end());
    z.assign(y.size(), T(0));
    const fp_dtype dt = sizeof(T) == 4 ? FP_F32 : FP_F64;
    fp_array ya{yt.data(), dt, static_cast<int>(shape.size()), shape.data(), nullptr, FP_DEVICE_CPU};
    fp_array za{z.data(), dt, static_cast<int>(shape.size()), shape.data(), nullptr, FP_DEVICE_CPU};
    fp_options o = opts(-1, 0, ws ? 1 : 0);
    o.tol = 1e-12;
    o.max_iter = 100000;
    return fp_prox_conj(op, sigma, &ya, &za, &o, ws, nullptr);
}

static void test_conj_extremes() {
    static const int64_t chain[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    const Vec row = {0.5, 2, -3, 1, 20, -7};
    const Vec img = {0.5, 2, -3, 1, 20, -7, 0.3, -0.2, 4, 1.5, -2.5, 0.1, 9, -1, 3, 0.7};
    struct C {
        std::function<fp_status(fp_op **)> make;
        bool image;
    };
    const std::vector<C> homog = {
        {[](fp_op **o) { return fp_op_soft(10.0, nullptr, o); }, false},
        {[](fp_op **o) { return fp_op_group_l2(10.0, 0, nullptr, 1, nullptr, o); }, false},
        {[](fp_op **o) {
             static const int64_t ptr[] = {0, 3}, idx[] = {0, 1, 2};
             return fp_op_tree_l2(0.5, 1, ptr, idx, nullptr, o);
         }, false},
        {[](fp_op **o) { return fp_op_linf(10.0, o); }, false},
        {[](fp_op **o) { return fp_op_max(10.0, o); }, false},
        {[](fp_op **o) { return fp_op_sum_largest(10.0, 2, 0, o); }, false},
        {[](fp_op **o) { return fp_op_sum_largest(10.0, 2, 1, o); }, false},
        {[](fp_op **o) { return fp_op_tv1d(nullptr, FP_TV1D_AUTO, o); }, false},
        {[](fp_op **o) { return fp_op_tv1d_l2(nullptr, o); }, false},
        {[](fp_op **o) { return fp_op_graph_tv(6, 5, chain, nullptr, o); }, false},
        {[](fp_op **o) { return fp_op_tree_tv(6, 5, chain, nullptr, o); }, false},
        {[](fp_op **o) { return fp_op_nonneg(o); }, false},
        {[](fp_op **o) { return fp_op_soc(o); }, false},
        {[](fp_op **o) { return fp_op_isotonic(nullptr, 1, o); }, false},
        {[](fp_op **o) { return fp_op_tv_aniso(2, nullptr, o); }, true},
        {[](fp_op **o) { return fp_op_tv_iso(2, nullptr, o); }, true},
    };
    // Homogeneous g: y - prox_g(y), the same bits for every sigma > 0, and
    // derivatives constant in sigma.
    for (const C &c : homog) {
        fp_op *op;
        CHECK_OK(c.make(&op));
        const std::vector<int64_t> shape = c.image ? std::vector<int64_t>{4, 4} : std::vector<int64_t>{6};
        const Vec &y = c.image ? img : row;
        std::vector<double> z1, z;
        std::vector<float> f1, f;
        CHECK_OK(conj_call(op, 1.0, shape, y, z1));
        CHECK_OK(conj_call(op, 1.0, shape, y, f1));
        for (double sg : {5e-324, 1e-310, 1e-300, 1e-100, 1e100, 1e300}) {
            CHECK(conj_call(op, sg, shape, y, z) == FP_OK && z == z1);
            CHECK(conj_call(op, sg, shape, y, f) == FP_OK && f == f1);
        }
        Arr ya(shape, y), za(shape), d0(shape), d1(shape), g(shape, randn(y.size())), gx(shape);
        fp_options o = opts(-1, 0, 1);
        o.tol = 1e-12;
        o.max_iter = 100000;
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &ya.a, &o, &ws));
        CHECK_OK(fp_prox_conj(op, 1e-200, &ya.a, &za.a, &o, ws, nullptr));
        CHECK_OK(fp_jvp(op, ws, &g.a, 0.0, nullptr, &d0.a));
        CHECK_OK(fp_jvp(op, ws, &g.a, 1.0, nullptr, &d1.a));
        CHECK(d0.data == d1.data);
        double gg = -1.0;
        CHECK_OK(fp_vjp(op, ws, &g.a, &gx.a, &gg, nullptr));
        CHECK(gg == 0.0);
        fp_workspace_free(ws);
        fp_op_free(op);
    }
    {  // soft (lambda 10): y clipped to [-10, 10] at any sigma
        fp_op *op;
        CHECK_OK(fp_op_soft(10.0, nullptr, &op));
        std::vector<double> z;
        for (double sg : {1e-320, 1e300}) {
            CHECK_OK(conj_call(op, sg, {3}, {20, -3, 0.5}, z));
            CHECK((z == std::vector<double>{10, -3, 0.5}));
        }
        fp_op_free(op);
    }
    // Other operators: FP_EPARAM where 1/sigma or y/sigma overflows, or 1/sigma
    // leaves the step range; projections (which ignore the step) stay exact.
    const std::vector<std::pair<std::function<fp_status(fp_op **)>, bool>> other = {
        {[](fp_op **o) { return fp_op_elastic_net(1.0, 10.0, o); }, true},
        {[](fp_op **o) { return fp_op_huber(10.0, 1.0, o); }, true},
        {[](fp_op **o) { return fp_op_log_barrier(10.0, o); }, true},
        {[](fp_op **o) { return fp_op_l0(1.0, nullptr, o); }, true},
        {[](fp_op **o) { return fp_op_box(-1.0, 2.0, nullptr, nullptr, o); }, false},
        {[](fp_op **o) { return fp_op_l2_ball(1.0, o); }, false},
        {[](fp_op **o) { return fp_op_simplex(1.0, o); }, false},
        {[](fp_op **o) { return fp_op_l1_ball(1.0, nullptr, o); }, false},
        {[](fp_op **o) { return fp_op_tv1d_ball(1.0, nullptr, o); }, false},
    };
    for (const auto &c : other) {
        fp_op *op;
        CHECK_OK(c.first(&op));
        std::vector<double> z;
        CHECK(conj_call(op, 1e-310, {6}, row, z) == FP_EPARAM);  // 1/sigma = inf
        const fp_status s = conj_call(op, 1e-300, {6}, row, z);
        if (c.second) {
            CHECK(s == FP_EPARAM);  // 1/sigma = 1e300 beyond the step range
        } else {
            CHECK(s == FP_OK);
            CHECK(std::isfinite(maxabs(z)));
        }
        fp_op_free(op);
    }
    {  // float32: y / sigma = 1e40 overflows the dtype, elementwise and row alike
        fp_op *en, *ball;
        CHECK_OK(fp_op_elastic_net(1.0, 1.0, &en));
        CHECK_OK(fp_op_l2_ball(1.0, &ball));
        std::vector<float> f;
        CHECK(conj_call(en, 1e-30, {3}, {1e10, 1.0, 2.0}, f) == FP_EPARAM);
        CHECK(conj_call(ball, 1e-30, {3}, {1e10, 1.0, 2.0}, f) == FP_EPARAM);
        fp_op_free(en);
        fp_op_free(ball);
    }
    // The step range: gamma lam > 2^960 is FP_EPARAM; inside it, finite.
    {
        const std::vector<std::function<fp_status(fp_op **)>> ranged = {
            [](fp_op **o) { return fp_op_log_barrier(10.0, o); },
            [](fp_op **o) { return fp_op_max(10.0, o); },
            [](fp_op **o) { return fp_op_sum_largest(10.0, 2, 0, o); },
            [](fp_op **o) { return fp_op_tree_tv(6, 5, chain, nullptr, o); },
        };
        for (const auto &make : ranged) {
            fp_op *op;
            CHECK_OK(make(&op));
            Arr x({6}, row), z({6});
            CHECK(fp_prox(op, 1e308, &x.a, &z.a, nullptr, nullptr, nullptr) == FP_EPARAM);
            CHECK_OK(fp_prox(op, 1e280, &x.a, &z.a, nullptr, nullptr, nullptr));
            CHECK(std::isfinite(maxabs(z.data)));
            fp_op_free(op);
        }
        fp_op *lb, *tt;
        CHECK_OK(fp_op_log_barrier(10.0, &lb));
        CHECK_OK(fp_op_tree_tv(6, 5, chain, nullptr, &tt));
        Arr x({6}, row), z({6});
        CHECK_OK(fp_prox(lb, 1e280, &x.a, &z.a, nullptr, nullptr, nullptr));
        const double expect = (-3.0 + std::hypot(-3.0, 2.0 * std::sqrt(1e280) * std::sqrt(10.0))) / 2;
        CHECK(std::fabs(z.data[2] - expect) <= 1e-15 * expect);
        CHECK_OK(fp_prox(tt, 1e280, &x.a, &z.a, nullptr, nullptr, nullptr));
        double mean = 0;
        for (double v : row) mean += v / 6;
        for (double v : z.data) CHECK(std::fabs(v - mean) <= 1e-15 * 20);
        fp_op_free(lb);
        fp_op_free(tt);
    }
}

// ---------------------------------------------------------------- outputs and descriptors
static fp_array view(double *data, int nd, const int64_t *shape, const int64_t *strides) {
    return fp_array{data, FP_F64, nd, shape, strides, FP_DEVICE_CPU};
}

// Outputs (audit C1): exactly the input, or disjoint from its byte range,
// and no stride 0 on a dimension of extent > 1; anything else is FP_EINVAL
// before any work.
static void test_aliasing() {
    fp_op *soft, *tv1d, *aniso;
    CHECK_OK(fp_op_soft(0.5, nullptr, &soft));
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv1d));
    aniso = make_aniso(2, {1.0});
    // Exact alias accepted, also through negative non-unit strides: (60, 57)
    // read as x[i, j] = buf[(59 - i) 114 + (56 - j) 2].
    for (fp_op *op : {soft, tv1d, aniso})
        for (int t : {1, 8}) {
            const int64_t shape[2] = {60, 57}, neg[2] = {-114, -2};
            Vec buf = randn(60 * 114, 1.5);
            Arr xc({60, 57}), ref({60, 57});
            for (int64_t i = 0; i < 60; ++i)
                for (int64_t j = 0; j < 57; ++j) xc.data[i * 57 + j] = buf[(59 - i) * 114 + (56 - j) * 2];
            fp_options o = opts(-1, t);
            o.tol = 1e-10;
            o.max_iter = 100000;
            CHECK_OK(fp_prox(op, 0.7, &xc.a, &ref.a, &o, nullptr, nullptr));
            fp_array v = view(buf.data() + 59 * 114 + 56 * 2, 2, shape, neg);
            CHECK_OK(fp_prox(op, 0.7, &v, &v, &o, nullptr, nullptr));
            bool same = true;
            for (int64_t i = 0; i < 60; ++i)
                for (int64_t j = 0; j < 57; ++j) same = same && buf[(59 - i) * 114 + (56 - j) * 2] == ref.data[i * 57 + j];
            CHECK(same);
            CHECK_OK(fp_prox(op, 0.7, &xc.a, &xc.a, &o, nullptr, nullptr));  // contiguous in place
            CHECK(xc.data == ref.data);
        }
    // Overlaps rejected, with the buffer untouched.
    auto rejected = [](fp_op *op, fp_array x, fp_array out, const Vec &buf, const Vec &before) {
        CHECK(fp_prox(op, 0.7, &x, &out, nullptr, nullptr, nullptr) == FP_EINVAL);
        CHECK(buf == before);
    };
    {
        Vec buf = randn(64 * 64);
        const Vec before = buf;
        const int64_t shape[2] = {64, 64}, c[2] = {64, 1}, tr[2] = {1, 64};
        rejected(soft, view(buf.data(), 2, shape, c), view(buf.data(), 2, shape, tr), buf, before);  // transpose
    }
    {
        Vec buf = randn(257 * 64);
        const Vec before = buf;
        const int64_t shape[2] = {256, 64};
        rejected(tv1d, view(buf.data(), 2, shape, nullptr), view(buf.data() + 64, 2, shape, nullptr), buf, before);
        rejected(tv1d, view(buf.data() + 64, 2, shape, nullptr), view(buf.data(), 2, shape, nullptr), buf, before);
    }
    {
        Vec buf = randn(200);
        const Vec before = buf;
        const int64_t n[1] = {100}, two[1] = {2}, rev[1] = {-1}, one[1] = {1};
        rejected(soft, view(buf.data(), 1, n, two), view(buf.data() + 1, 1, n, two), buf, before);  // a[0::2], a[1::2]
        rejected(soft, view(buf.data(), 1, n, one), view(buf.data() + 99, 1, n, rev), buf, before);  // reversed
    }
    // Stride 0 in an output: rejected on an extent > 1, accepted on extent 1.
    {
        Arr x({1000});
        Vec z(1, 0.0);
        const int64_t n[1] = {1000}, zero[1] = {0};
        fp_array out = view(z.data(), 1, n, zero);
        CHECK(fp_prox(soft, 0.7, &x.a, &out, nullptr, nullptr, nullptr) == FP_EINVAL);
        Arr x2({1, 1000}, randn(1000)), z2({1, 1000});
        const int64_t sh[2] = {1, 1000}, st[2] = {0, 1};
        fp_array o2 = view(z2.data.data(), 2, sh, st);
        CHECK_OK(fp_prox(soft, 0.7, &x2.a, &o2, nullptr, nullptr, nullptr));
    }
    // A rejected call leaves the workspace's structure usable; derivatives
    // follow the same rules (dx/dout, g/gx).
    for (fp_op *op : {soft, tv1d}) {
        Vec buf = randn(2 * 640);
        const int64_t shape[2] = {10, 64};
        fp_array x = view(buf.data(), 2, shape, nullptr);
        Arr z({10, 64}), dx({10, 64}, randn(640)), d1({10, 64}), d2({10, 64});
        fp_options o = opts(-1, 0, 1);
        fp_workspace *ws;
        CHECK_OK(fp_workspace_create(op, &x, &o, &ws));
        CHECK_OK(fp_prox(op, 0.7, &x, &z.a, &o, ws, nullptr));
        CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &d1.a));
        fp_array shifted = view(buf.data() + 7, 2, shape, nullptr);
        CHECK(fp_prox(op, 0.7, &x, &shifted, &o, ws, nullptr) == FP_EINVAL);
        CHECK_OK(fp_jvp(op, ws, &dx.a, 0.0, nullptr, &d2.a));
        CHECK(d1.data == d2.data);
        // jvp: dx and dout partially overlapping, then exactly the same.
        Vec tb(2 * 640);
        for (size_t i = 0; i < 640; ++i) tb[i] = dx.data[i];
        fp_array t0 = view(tb.data(), 2, shape, nullptr), t1 = view(tb.data() + 5, 2, shape, nullptr);
        CHECK(fp_jvp(op, ws, &t0, 0.0, nullptr, &t1) == FP_EINVAL);
        CHECK_OK(fp_jvp(op, ws, &t0, 0.0, nullptr, &t0));
        CHECK(std::equal(d1.data.begin(), d1.data.end(), tb.begin()));
        // vjp: the same pair.
        Arr gx({10, 64});
        double gg;
        CHECK_OK(fp_vjp(op, ws, &dx.a, &gx.a, &gg, nullptr));
        for (size_t i = 0; i < 640; ++i) tb[i] = dx.data[i];
        CHECK(fp_vjp(op, ws, &t0, &t1, &gg, nullptr) == FP_EINVAL);
        CHECK_OK(fp_vjp(op, ws, &t0, &t0, &gg, nullptr));
        CHECK(std::equal(gx.data.begin(), gx.data.end(), tb.begin()));
        fp_workspace_free(ws);
    }
    // fp_prox_steps: the steps inside out's byte range.
    {
        Arr x({8, 16}, randn(128));
        Vec obuf(128, 0.0);
        const int64_t shape[2] = {8, 16}, sshape[1] = {8};
        fp_array out = view(obuf.data(), 2, shape, nullptr), steps = view(obuf.data() + 3, 1, sshape, nullptr);
        for (int i = 0; i < 8; ++i) obuf[3 + i] = 0.5;
        CHECK(fp_prox_steps(soft, &steps, &x.a, &out, nullptr, nullptr) == FP_EINVAL);
    }
    // fp_value: out overlapping x.
    {
        Vec buf = randn(64);
        const int64_t shape[2] = {8, 8}, bshape[1] = {8};
        fp_array x = view(buf.data(), 2, shape, nullptr), out = view(buf.data() + 8, 1, bshape, nullptr);
        CHECK(fp_value(soft, &x, nullptr, &out) == FP_EINVAL);
    }
    fp_op_free(soft);
    fp_op_free(tv1d);
    fp_op_free(aniso);
}

// Descriptors whose entry count or offset span exceeds 2^60 - 1 (audit A2).
static void test_descriptor_bounds() {
    fp_op *op;
    CHECK_OK(fp_op_soft(0.5, nullptr, &op));
    Vec small(64, 1.0);
    const int64_t B32 = int64_t{1} << 32, B62 = int64_t{1} << 62, B40 = int64_t{1} << 40;
    struct Bad {
        std::vector<int64_t> shape, strides;  // empty strides: C order
        bool null_data;
    };
    const std::vector<Bad> bad = {
        {{B32, B32}, {}, true},
        {{B32 + 1, B32}, {}, false},
        {{3, B62}, {}, false},
        {{B62 + 1, 4}, {0, 1}, false},
        {{B40, B40, 0}, {}, false},
        {{2, 4}, {INT64_MIN, 1}, false},
        {{2, 2}, {int64_t{1} << 60, 1}, false},
    };
    for (const Bad &b : bad) {
        fp_array x = view(b.null_data ? nullptr : small.data(), static_cast<int>(b.shape.size()), b.shape.data(),
                          b.strides.empty() ? nullptr : b.strides.data());
        fp_options o = opts(-1, 0, 1);
        fp_workspace *ws = nullptr;
        CHECK(fp_workspace_create(op, &x, &o, &ws) == FP_ESHAPE && ws == nullptr);
        CHECK(fp_prox(op, 0.7, &x, &x, nullptr, nullptr, nullptr) == FP_ESHAPE);
        CHECK(fp_prox_conj(op, 0.7, &x, &x, nullptr, nullptr, nullptr) == FP_ESHAPE);
        double v = 0.0;
        const int64_t one[1] = {1};
        fp_array vo = view(&v, 1, one, nullptr);
        CHECK(fp_value(op, &x, nullptr, &vo) == FP_ESHAPE);
    }
    // Broadcast and empty inputs stay valid.
    {
        const int64_t shape[2] = {1000, 4}, st[2] = {0, 1};
        fp_array x = view(small.data(), 2, shape, st);
        Arr z({1000, 4});
        CHECK_OK(fp_prox(op, 0.7, &x, &z.a, nullptr, nullptr, nullptr));
        for (const std::vector<int64_t> &sh : {std::vector<int64_t>{0, 5}, std::vector<int64_t>{5, 0}}) {
            fp_array e = view(small.data(), 2, sh.data(), nullptr);
            CHECK_OK(fp_prox(op, 0.7, &e, &e, nullptr, nullptr, nullptr));
        }
    }
    fp_op_free(op);
}

// Row scratch is sized for the threads the rows can use (audit C2).
static void test_row_scratch_bytes() {
#ifndef FP_TSAN
    fp_op *op;
    CHECK_OK(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &op));
    for (int64_t rows : {1, 3}) {
        Arr x({rows, int64_t{1} << 16});
        auto bytes = [&](int threads) {
            fp_options o = opts(-1, threads, 0);
            fp_workspace *ws;
            const long long before = g_alloc_bytes.load();
            CHECK_OK(fp_workspace_create(op, &x.a, &o, &ws));
            const long long b = g_alloc_bytes.load() - before;
            fp_workspace_free(ws);
            return b;
        };
        const long long one = bytes(1), all = bytes(0), many = bytes(64);
        CHECK(all <= 1.2 * rows * one && many <= 1.2 * rows * one);
    }
    fp_op_free(op);
#endif
}

// Threads in this process, or -1 where they cannot be counted.
static int process_threads() {
#if defined(__APPLE__)
    thread_act_array_t list;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &list, &count) != KERN_SUCCESS) return -1;
    for (mach_msg_type_number_t i = 0; i < count; ++i) mach_port_deallocate(mach_task_self(), list[i]);
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(list), count * sizeof(thread_act_t));
    return static_cast<int>(count);
#elif defined(__linux__)
    int n = 0;
    if (DIR *d = opendir("/proc/self/task")) {
        while (dirent *e = readdir(d))
            if (e->d_name[0] != '.') ++n;
        closedir(d);
        return n;
    }
    return -1;
#else
    return -1;
#endif
}

// fp_options.threads is capped at 4 x the CPUs (audit C3). Run last: it
// grows the pool for the rest of the process.
static void test_thread_cap() {
    fp_op *op;
    CHECK_OK(fp_op_soft(1.0, nullptr, &op));
    const int64_t n = int64_t{1} << 22;  // 256 chunks
    Arr x({n}, randn(n)), z({n});
    fp_options o = opts(-1, 100000, 0);
    CHECK_OK(fp_prox(op, 0.5, &x.a, &z.a, &o, nullptr, nullptr));
    const int t = process_threads();
    const unsigned hw = std::thread::hardware_concurrency();
    if (t >= 0 && hw > 0) CHECK(t <= 4 * static_cast<int>(hw) + 2);
    fp_op_free(op);
}

int main() {
    test_tv1d();
    test_projections();
    test_derivatives();
    test_batching();
    test_tv_aniso();
    test_tv_iso();
    test_tv_spatial_weights();
    test_graph_tv();
    test_tree_tv();
    test_nonconvex();
    test_kinks();
    test_edge_kinks();
    test_value();
    test_spectral();
    test_device_api();
    test_prox_steps();
    test_no_allocation();
    test_concurrent_callers();
    test_errors_and_special_values();
    test_kept_empty_and_identity();
    test_image_axis_ignored();
    test_huge_counts();
    test_failed_branch_keeps_override();
    test_kept_point_lifetime();
    test_nan_images();
    test_aliasing();
    test_descriptor_bounds();
    test_conj_extremes();
    test_float32();
    test_row_scratch_bytes();
    test_thread_cap();  // last: it grows the pool
    std::printf("%d checks, %d failures (default threads: %d)\n", g_checks, g_fail, fp_default_threads());
    return g_fail != 0;
}
