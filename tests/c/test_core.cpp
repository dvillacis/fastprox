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
#ifndef FP_TSAN
void *operator new(std::size_t n) {
    ++g_allocs;
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
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
        Arr x2(x);
        for (auto &v : x2.data) v += 0.01 * std::sin(v * 7);
        fp_info c2, w2;
        CHECK_OK(fp_prox(op, 0.5, &x2.a, &z.a, &o, ws, &c2));  // cold (warm is off, nothing seeded)
        CHECK_OK(fp_workspace_set_dual(ws, &q.a));
        CHECK_OK(fp_prox(op, 0.5, &x2.a, &z.a, &o, ws, &w2));
        CHECK(w2.iters < c2.iters);
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

    // Anisotropic TV: a single image (threads inside it) and a batch.
    for (std::vector<int64_t> shape : {std::vector<int64_t>{120, 130}, std::vector<int64_t>{6, 30, 40}}) {
        fp_op *iop;
        CHECK_OK(fp_op_tv_aniso(2, nullptr, &iop));
        int64_t N = 1;
        for (int64_t s : shape) N *= s;
        Arr ix(shape, randn(N)), iz(shape), ig(shape, randn(N)), igx(shape), iq({2});
        std::vector<int64_t> qs = {2};
        qs.insert(qs.end(), shape.begin(), shape.end());
        Arr dual(qs);
        fp_options io = opts(-1, 4, 1);
        io.tol = 1e-8;
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
    test_tv_aniso();
    test_no_allocation();
    test_concurrent_callers();
    test_errors_and_special_values();
    test_float32();
    std::printf("%d checks, %d failures (default threads: %d)\n", g_checks, g_fail, fp_default_threads());
    return g_fail != 0;
}
