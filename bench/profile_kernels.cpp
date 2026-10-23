// Single-thread profile of 1D TV: fastprox against proxTV, with hardware
// counters where the machine exposes them; and (audit C11) a profile of the
// core: workspace bytes, elementwise throughput, derivatives, the pool's
// latency, the grain, strided rows and the spin's CPU time.
//
//   c++ -O2 -std=c++17 bench/profile_kernels.cpp src/core/parallel.cpp -Iinclude -Isrc \
//       -Lbuild/c -lfastprox -ldl -pthread -Wl,-rpath,build/c -o profile_kernels
//   ./profile_kernels [path/to/libproxtv_tv1.so] [n] [reps]
//   ./profile_kernels core [wsbytes|elem|deriv|cold|grain|strided|spin]
//
// The core cases time the library through its C API, except cold's empty
// job and spin, which time the pool (src/core/parallel.cpp, compiled into
// this binary) directly.
//
// proxTV is loaded at run time through the extern "C" shim that
// bench/compare_external.py builds (build/external/libproxtv_tv1.*); without
// it only fastprox is profiled. Clean-room rule as in that script: proxTV is
// only compiled and called, never read.
//
// Every repetition draws a fresh input and runs every solver on it in turn,
// so drift of the machine affects all solvers alike and the branch predictor
// cannot learn the input; medians are reported. On Linux, cycles,
// instructions, branches and branch misses are read with perf_event_open
// around each call (they need kernel.perf_event_paranoid <= 2 and a virtual
// PMU; when unavailable only times are reported).
//
// SPDX-License-Identifier: MIT
#include "fastprox.h"
#include "core/parallel.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

// ---- counters ---------------------------------------------------------------

constexpr int kEvents = 4;  // cycles, instructions, branches, branch misses
const char *const kEventNames[kEvents] = {"cycles", "instr", "branches", "br-miss"};

struct Counters {
    int fd[kEvents] = {-1, -1, -1, -1};
    bool ok = false;

    Counters() {
#if defined(__linux__)
        const uint64_t cfg[kEvents] = {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
                                       PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
                                       PERF_COUNT_HW_BRANCH_MISSES};
        for (int e = 0; e < kEvents; ++e) {
            perf_event_attr a;
            std::memset(&a, 0, sizeof a);
            a.size = sizeof a;
            a.type = PERF_TYPE_HARDWARE;
            a.config = cfg[e];
            a.disabled = 1;
            a.exclude_kernel = 1;
            a.exclude_hv = 1;
            fd[e] = static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, -1, 0));
            if (fd[e] < 0) {
                close_all();
                return;
            }
        }
        ok = true;
#endif
    }
    ~Counters() { close_all(); }
    void close_all() {
#if defined(__linux__)
        for (int &f : fd)
            if (f >= 0) close(f), f = -1;
#endif
        ok = false;
    }
    void start() {
#if defined(__linux__)
        for (int f : fd) ioctl(f, PERF_EVENT_IOC_RESET, 0), ioctl(f, PERF_EVENT_IOC_ENABLE, 0);
#endif
    }
    void stop(double *v) {
#if defined(__linux__)
        for (int e = 0; e < kEvents; ++e) {
            ioctl(fd[e], PERF_EVENT_IOC_DISABLE, 0);
            uint64_t c = 0;
            v[e] = read(fd[e], &c, sizeof c) == static_cast<ssize_t>(sizeof c)
                       ? static_cast<double>(c) : NAN;
        }
#else
        for (int e = 0; e < kEvents; ++e) v[e] = NAN;
#endif
    }
};

// ---- signals (the families of bench/compare_external.py) --------------------

void signal(const std::string &kind, std::vector<double> &x, std::mt19937_64 &rng) {
    std::normal_distribution<double> N;
    const size_t n = x.size();
    if (kind == "blocks") {
        double b = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (i % 50 == 0) b = 3.0 * N(rng);
            x[i] = b + 0.5 * N(rng);
        }
    } else if (kind == "random walk") {
        double s = 0.0;
        for (auto &v : x) v = s += N(rng);
    } else {
        for (auto &v : x) v = N(rng);
    }
}

// ---- solvers -----------------------------------------------------------------

using Solve = std::function<void(const double *, double *)>;

struct Solver {
    std::string name;
    Solve run;
};

fp_array vec(double *p, const int64_t *shape) {
    return fp_array{p, FP_F64, 1, shape, nullptr, FP_DEVICE_CPU};
}

void check(fp_status st, const char *what) {
    if (st != FP_OK) {
        std::fprintf(stderr, "%s failed with status %d\n", what, static_cast<int>(st));
        std::exit(1);
    }
}

// An operator with its own workspace (a workspace belongs to one operator).
struct FpSolver {
    fp_op *op = nullptr;
    fp_workspace *ws = nullptr;
    FpSolver(const fp_array *weights, fp_tv1d_method method, double *probe, const int64_t *shape,
             const fp_options *opt) {
        check(fp_op_tv1d(weights, method, &op), "fp_op_tv1d");
        fp_array like = vec(probe, shape);
        check(fp_workspace_create(op, &like, opt, &ws), "fp_workspace_create");
    }
    ~FpSolver() {
        fp_workspace_free(ws);
        fp_op_free(op);
    }
    FpSolver(const FpSolver &) = delete;
    FpSolver &operator=(const FpSolver &) = delete;
};

double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void profile(const char *title, std::vector<Solver> &solvers, const std::string &kind, int64_t n,
             int reps, Counters &cnt, std::mt19937_64 &rng) {
    const size_t S = solvers.size();
    std::vector<double> x(static_cast<size_t>(n));
    std::vector<std::vector<double>> out(S, std::vector<double>(x.size()));
    std::vector<std::vector<double>> t(S), c(S * kEvents);
    double maxdiff = 0.0;
    for (int r = 0; r <= reps; ++r) {  // round 0 warms up and is not recorded
        signal(kind, x, rng);
        for (size_t k = 0; k < S; ++k) {
            // Rotate the order: a solver run right after another on the same
            // input benefits from what the branch predictor has learned.
            const size_t s = (k + static_cast<size_t>(r)) % S;
            double v[kEvents];
            if (cnt.ok) cnt.start();
            const auto t0 = std::chrono::steady_clock::now();
            solvers[s].run(x.data(), out[s].data());
            const auto t1 = std::chrono::steady_clock::now();
            if (cnt.ok) cnt.stop(v);
            if (r == 0) continue;
            t[s].push_back(std::chrono::duration<double>(t1 - t0).count() * 1e9 / static_cast<double>(n));
            if (cnt.ok)
                for (int e = 0; e < kEvents; ++e) c[s * kEvents + e].push_back(v[e] / static_cast<double>(n));
        }
        for (size_t s = 1; s < S; ++s)
            for (size_t i = 0; i < x.size(); ++i)
                maxdiff = std::max(maxdiff, std::fabs(out[s][i] - out[0][i]));
    }
    std::printf("\n%s, %s (n = %lld; max |difference| from %s %.1e)\n", title, kind.c_str(),
                static_cast<long long>(n), solvers[0].name.c_str(), maxdiff);
    std::printf("  %-28s %9s", "", "ns");
    if (cnt.ok)
        for (int e = 0; e < kEvents; ++e) std::printf(" %9s", kEventNames[e]);
    if (cnt.ok) std::printf(" %9s %9s", "IPC", "miss %");
    std::printf("   (per sample)\n");
    for (size_t s = 0; s < S; ++s) {
        std::printf("  %-28s %9.2f", solvers[s].name.c_str(), median(t[s]));
        if (cnt.ok) {
            double m[kEvents];
            for (int e = 0; e < kEvents; ++e) m[e] = median(c[s * kEvents + e]);
            for (int e = 0; e < kEvents; ++e) std::printf(" %9.2f", m[e]);
            std::printf(" %9.2f %9.2f", m[1] / m[0], 100.0 * m[3] / m[2]);
        }
        std::printf("\n");
    }
}

}  // namespace

namespace {

// ---- core (audit C11) ----------------------------------------------------------

using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}
// Best of reps calls of f, in microseconds.
template <class F>
double best_us(int reps, F &&f) {
    double b = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        f();
        b = std::min(b, us_since(t0));
    }
    return b;
}
double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())))];
}
std::vector<double> normals(size_t n, uint64_t seed = 1) {
    std::mt19937_64 r(seed);
    std::normal_distribution<double> N;
    std::vector<double> v(n);
    for (auto &a : v) a = N(r);
    return v;
}
fp_array arr(double *d, int ndim, const int64_t *shape, const int64_t *strides) {
    fp_array a{};
    a.data = d;
    a.dtype = FP_F64;
    a.ndim = ndim;
    a.shape = shape;
    a.strides = strides;
    a.device = FP_DEVICE_CPU;
    return a;
}
// Bytes the allocator has handed out, for the deltas around a call.
size_t heap_in_use() {
#if defined(__APPLE__)
    malloc_statistics_t st;
    malloc_zone_statistics(nullptr, &st);
    return st.size_in_use;
#elif defined(__GLIBC__)
    return mallinfo2().uordblks;
#else
    return 0;
#endif
}
fp_options topts(int threads, int keep = 0) {
    fp_options o;
    fp_options_init(&o);
    o.threads = threads;
    o.keep_structure = keep;
    return o;
}

void core_wsbytes() {
    std::printf("[wsbytes] tv1d, one row: workspace bytes and best-of-5 prox\n");
    fp_op *op;
    check(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &op), "fp_op_tv1d");
    for (int64_t n : {int64_t(1) << 16, int64_t(1) << 20, int64_t(1) << 23}) {
        std::vector<double> x = normals(static_cast<size_t>(n)), z(x.size());
        const int64_t shape[1] = {n}, st[1] = {1};
        fp_array xa = arr(x.data(), 1, shape, st), za = arr(z.data(), 1, shape, st);
        for (int t : {1, 0}) {
            fp_options o = topts(t);
            fp_workspace *ws;
            const size_t before = heap_in_use();
            check(fp_workspace_create(op, &xa, &o, &ws), "fp_workspace_create");
            const size_t bytes = heap_in_use() - before;
            const double us = best_us(5, [&] { check(fp_prox(op, 1.0, &xa, &za, &o, ws, nullptr), "fp_prox"); });
            std::printf("  n=2^%-2d t=%d  %10zu bytes  prox %9.1f us\n", static_cast<int>(std::log2(n)), t, bytes, us);
            fp_workspace_free(ws);
        }
    }
    fp_op_free(op);
}

void core_elem() {
    std::printf("[elem] ns per entry, t=1, 2^16 entries, best of 3000\n");
    const int64_t n = int64_t(1) << 16;
    std::vector<double> x = normals(static_cast<size_t>(n)), z(x.size()), w(x.size(), 0.7);
    const int64_t shape[1] = {n}, st[1] = {1}, ws1[1] = {n};
    fp_array xa = arr(x.data(), 1, shape, st), za = arr(z.data(), 1, shape, st);
    fp_array wa = arr(w.data(), 1, ws1, st);
    struct Case {
        const char *name;
        fp_op *op;
    };
    std::vector<Case> cases;
    auto add = [&](const char *name, fp_status s, fp_op *op) {
        check(s, name);
        cases.push_back({name, op});
    };
    fp_op *o;
    add("soft", fp_op_soft(0.3, nullptr, &o), o);
    add("soft weighted", fp_op_soft(0.3, &wa, &o), o);
    add("elastic net", fp_op_elastic_net(0.3, 0.2, &o), o);
    add("huber", fp_op_huber(0.5, 0.5, &o), o);
    add("box", fp_op_box(-0.5, 0.5, nullptr, nullptr, &o), o);
    add("nonneg", fp_op_nonneg(&o), o);
    add("mcp", fp_op_mcp(0.3, 3.0, nullptr, &o), o);
    add("l0", fp_op_l0(0.3, nullptr, &o), o);
    fp_options opt = topts(1);
    for (const Case &c : cases) {
        const double out = best_us(3000, [&] { check(fp_prox(c.op, 1.0, &xa, &za, &opt, nullptr, nullptr), c.name); });
        std::vector<double> y(x);
        fp_array ya = arr(y.data(), 1, shape, st);
        const double in = best_us(3000, [&] { check(fp_prox(c.op, 1.0, &ya, &ya, &opt, nullptr, nullptr), c.name); });
        std::printf("  %-14s out of place %6.3f ns  in place %6.3f ns\n", c.name, 1e3 * out / n, 1e3 * in / n);
        fp_op_free(c.op);
    }
    const double floor = best_us(3000, [&] {
        for (int64_t i = 0; i < n; ++i) z[i] = std::copysign(std::max(std::fabs(x[i]) - 0.3, 0.0), x[i]);
    });
    std::printf("  %-14s %6.3f ns (a hand-written loop)\n", "soft floor", 1e3 * floor / n);
}

void core_deriv() {
    std::printf("[deriv] 256 x 1000, kept structure, best of 30, us\n");
    const int64_t shape[2] = {256, 1000}, st[2] = {1000, 1};
    const size_t N = 256 * 1000;
    std::vector<double> x = normals(N), z(N), g = normals(N, 2), out(N);
    fp_array xa = arr(x.data(), 2, shape, st), za = arr(z.data(), 2, shape, st);
    fp_array ga = arr(g.data(), 2, shape, st), oa = arr(out.data(), 2, shape, st);
    fp_op *ops[3];
    check(fp_op_soft(0.3, nullptr, &ops[0]), "soft");
    check(fp_op_nonneg(&ops[1]), "nonneg");
    check(fp_op_huber(0.5, 0.5, &ops[2]), "huber");
    const char *names[3] = {"soft", "nonneg", "huber"};
    for (int k = 0; k < 3; ++k)
        for (int t : {1, 15}) {
            fp_options o = topts(t, 1);
            fp_workspace *ws;
            check(fp_workspace_create(ops[k], &xa, &o, &ws), "ws");
            const double tp = best_us(30, [&] { check(fp_prox(ops[k], 1.0, &xa, &za, &o, ws, nullptr), "prox"); });
            const double tj = best_us(30, [&] { check(fp_jvp(ops[k], ws, &ga, 0.3, nullptr, &oa), "jvp"); });
            std::vector<double> gp(static_cast<size_t>(std::max<int64_t>(1, fp_op_nparams(ops[k]))));
            const int64_t pshape[1] = {fp_op_nparams(ops[k])}, pst[1] = {1};
            fp_array pa = arr(gp.data(), 1, pshape, pst);
            double gg = 0.0;
            const double tv = best_us(30, [&] {
                check(fp_vjp(ops[k], ws, &ga, &oa, &gg, pshape[0] ? &pa : nullptr), "vjp");
            });
            const double tv0 = best_us(30, [&] { check(fp_vjp(ops[k], ws, &ga, &oa, nullptr, nullptr), "vjp"); });
            std::printf("  %-7s t=%-2d prox %8.1f  jvp %8.1f  vjp %8.1f  vjp(no sums) %8.1f\n", names[k], t, tp,
                        tj, tv, tv0);
            fp_workspace_free(ws);
        }
    for (fp_op *op : ops) fp_op_free(op);
}

void core_cold() {
    std::printf("[cold] empty parallel_for: warm median of 20000, cold (300 us sleep) median and p99 of 300\n");
    for (int nt : {4, 8, 15}) {
        auto job = [&] { fp::parallel_for(nt, nt, 1, [](int, int64_t, int64_t) {}); };
        std::vector<double> warm, cold;
        for (int i = 0; i < 20000; ++i) {
            const auto t0 = Clock::now();
            job();
            warm.push_back(us_since(t0));
        }
        for (int i = 0; i < 300; ++i) {
            std::this_thread::sleep_for(std::chrono::microseconds(300));
            const auto t0 = Clock::now();
            job();
            cold.push_back(us_since(t0));
        }
        std::printf("  nt=%-2d warm %6.2f us  cold median %7.2f us  p99 %7.2f us\n", nt, median(warm),
                    median(cold), quantile(cold, 0.99));
    }
    std::printf("[cold] soft prox: warm and cold medians of 200, us\n");
    fp_op *op;
    check(fp_op_soft(0.3, nullptr, &op), "soft");
    for (int e : {18, 20, 22}) {
        const int64_t n = int64_t(1) << e;
        std::vector<double> x = normals(static_cast<size_t>(n)), z(x.size());
        const int64_t shape[1] = {n}, st[1] = {1};
        fp_array xa = arr(x.data(), 1, shape, st), za = arr(z.data(), 1, shape, st);
        for (int t : {1, 4, 8, 15}) {
            fp_options o = topts(t);
            std::vector<double> warm, cold;
            for (int i = 0; i < 200; ++i) {
                const auto t0 = Clock::now();
                check(fp_prox(op, 1.0, &xa, &za, &o, nullptr, nullptr), "prox");
                warm.push_back(us_since(t0));
            }
            for (int i = 0; i < 200; ++i) {
                std::this_thread::sleep_for(std::chrono::microseconds(300));
                const auto t0 = Clock::now();
                check(fp_prox(op, 1.0, &xa, &za, &o, nullptr, nullptr), "prox");
                cold.push_back(us_since(t0));
            }
            std::printf("  2^%d t=%-2d warm %8.1f  cold %8.1f\n", e, t, median(warm), median(cold));
        }
    }
    fp_op_free(op);
}

void core_grain() {
    std::printf("[grain] best of 7, warm, us\n");
    const int ts[] = {1, 2, 4, 5, 8, 15};
    fp_op *soft, *tv;
    check(fp_op_soft(0.3, nullptr, &soft), "soft");
    check(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv), "tv1d");
    for (int e : {15, 16, 18, 20, 23}) {
        const int64_t n = int64_t(1) << e;
        std::vector<double> x = normals(static_cast<size_t>(n)), z(x.size());
        const int64_t shape[1] = {n}, st[1] = {1};
        fp_array xa = arr(x.data(), 1, shape, st), za = arr(z.data(), 1, shape, st);
        std::printf("  soft 2^%-2d", e);
        for (int t : ts) {
            fp_options o = topts(t);
            const double us = best_us(7, [&] { check(fp_prox(soft, 1.0, &xa, &za, &o, nullptr, nullptr), "soft"); });
            std::printf("  t=%-2d %8.1f", t, us);
        }
        std::printf("\n");
    }
    for (int e : {15, 16, 17, 18}) {
        const int64_t N = int64_t(1) << e, rows = N / 16;
        std::vector<double> x = normals(static_cast<size_t>(N)), z(x.size());
        const int64_t shape[2] = {rows, 16}, st[2] = {16, 1};
        fp_array xa = arr(x.data(), 2, shape, st), za = arr(z.data(), 2, shape, st);
        std::printf("  tv1d 2^%-2d in rows of 16", e);
        for (int t : ts) {
            fp_options o = topts(t);
            fp_workspace *ws;
            check(fp_workspace_create(tv, &xa, &o, &ws), "ws");
            const double us = best_us(7, [&] { check(fp_prox(tv, 0.5, &xa, &za, &o, ws, nullptr), "tv1d"); });
            std::printf("  t=%-2d %8.1f", t, us);
            fp_workspace_free(ws);
        }
        std::printf("\n");
    }
    fp_op_free(soft);
    fp_op_free(tv);
}

void core_strided() {
    std::printf("[strided] tv1d, t=1, best of 30, us\n");
    fp_op *tv;
    check(fp_op_tv1d(nullptr, FP_TV1D_AUTO, &tv), "tv1d");
    std::vector<double> x = normals(4096 * 256), z(x.size());
    struct Case {
        const char *name;
        int64_t shape[2], st[2];
        int axis;
    } cases[] = {{"(4096, 256) axis -1", {4096, 256}, {256, 1}, -1}, {"(256, 4096) axis 0", {256, 4096}, {4096, 1}, 0}};
    for (auto &c : cases) {
        fp_array xa = arr(x.data(), 2, c.shape, c.st), za = arr(z.data(), 2, c.shape, c.st);
        fp_options o = topts(1);
        o.axis = c.axis;
        fp_workspace *ws;
        check(fp_workspace_create(tv, &xa, &o, &ws), "ws");
        const double us = best_us(30, [&] { check(fp_prox(tv, 0.5, &xa, &za, &o, ws, nullptr), "tv1d"); });
        std::printf("  %-22s %9.1f\n", c.name, us);
        fp_workspace_free(ws);
    }
    fp_op_free(tv);
}

void core_spin() {
    const char *env = std::getenv("FASTPROX_SPIN_US");
    std::printf("[spin] 1000 empty 15-thread jobs, 1 ms apart (FASTPROX_SPIN_US=%s)\n", env ? env : "unset");
    const std::clock_t c0 = std::clock();
    const auto t0 = Clock::now();
    for (int i = 0; i < 1000; ++i) {
        fp::parallel_for(15, 15, 1, [](int, int64_t, int64_t) {});
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const double cpu = static_cast<double>(std::clock() - c0) / CLOCKS_PER_SEC;
    std::printf("  process CPU %.3f s, wall %.3f s\n", cpu, us_since(t0) * 1e-6);
}

int core_main(int argc, char **argv) {
    const std::string which = argc > 2 ? argv[2] : "";
    struct Entry {
        const char *name;
        void (*run)();
    } entries[] = {{"wsbytes", core_wsbytes}, {"elem", core_elem},       {"deriv", core_deriv}, {"cold", core_cold},
                   {"grain", core_grain},     {"strided", core_strided}, {"spin", core_spin}};
    std::printf("fastprox %s, core profile\n", fp_version_string());
    bool any = false;
    for (const Entry &e : entries)
        if (which.empty() || which == e.name) {
            e.run();
            any = true;
        }
    if (!any) {
        std::fprintf(stderr, "unknown case %s\n", which.c_str());
        return 2;
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1 && std::strcmp(argv[1], "core") == 0) return core_main(argc, argv);
    const char *ptv_path = argc > 1 && argv[1][0] ? argv[1] : nullptr;
    const int64_t n = argc > 2 ? std::atoll(argv[2]) : 100000;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 21;
    const int64_t shape[1] = {n};

    using Taut = int (*)(double *, int, double, double *);
    using HybridTaut = void (*)(double *, int, double, double *);
    using WTaut = int (*)(double *, double *, double *, int);
    Taut ptv_lin = nullptr;
    HybridTaut ptv_hyb = nullptr;
    WTaut ptv_w = nullptr;
    if (ptv_path) {
        void *h = dlopen(ptv_path, RTLD_NOW);
        if (!h) {
            std::fprintf(stderr, "proxTV unavailable: %s\n", dlerror());
        } else {
            ptv_lin = reinterpret_cast<Taut>(dlsym(h, "shim_linearized_taut"));
            ptv_hyb = reinterpret_cast<HybridTaut>(dlsym(h, "shim_hybrid_taut"));
            ptv_w = reinterpret_cast<WTaut>(dlsym(h, "shim_weighted_taut"));
        }
    }

    Counters cnt;
    std::printf("fastprox %s, single thread; hardware counters: %s\n", fp_version_string(),
                cnt.ok ? "yes" : "unavailable");

    fp_options opt;
    fp_options_init(&opt);
    opt.threads = 1;
    std::mt19937_64 rng(20261015);

    for (const char *kind : {"blocks", "random walk", "white noise"}) {
        for (double lam : {0.1, 1.0, 10.0}) {
            // Uniform weights: fastprox (auto and Condat) against proxTV.
            std::vector<double> probe(static_cast<size_t>(n));
            FpSolver f_auto(nullptr, FP_TV1D_AUTO, probe.data(), shape, &opt);
            auto fpx = [&](FpSolver &f, double gamma) {
                return [&, gamma](const double *x, double *z) {
                    fp_array xa = vec(const_cast<double *>(x), shape), za = vec(z, shape);
                    check(fp_prox(f.op, gamma, &xa, &za, &opt, f.ws, nullptr), "fp_prox");
                };
            };
            std::vector<Solver> uni = {{"fastprox auto", fpx(f_auto, lam)}};
            if (ptv_lin)
                uni.push_back({"proxTV linearized taut", [&](const double *x, double *z) {
                                   ptv_lin(const_cast<double *>(x), static_cast<int>(n), lam, z);
                               }});
            if (ptv_hyb)
                uni.push_back({"proxTV hybrid taut", [&](const double *x, double *z) {
                                   ptv_hyb(const_cast<double *>(x), static_cast<int>(n), lam, z);
                               }});
            char title[96];
            std::snprintf(title, sizeof title, "1D TV, lambda = %g", lam);
            profile(title, uni, kind, n, reps, cnt, rng);

            // Edge weights uniform in [0.5, 1.5] times lambda, fixed per operator.
            std::vector<double> w(static_cast<size_t>(n - 1));
            std::mt19937_64 wr(7);
            std::uniform_real_distribution<double> U(0.5, 1.5);
            for (auto &v : w) v = lam * U(wr);
            const int64_t wshape[1] = {n - 1};
            fp_array wa = vec(w.data(), wshape);
            FpSolver f_w(&wa, FP_TV1D_AUTO, probe.data(), shape, &opt);
            std::vector<Solver> wtd = {{"fastprox weighted", fpx(f_w, 1.0)}};
            if (ptv_w)
                wtd.push_back({"proxTV weighted taut", [&](const double *x, double *z) {
                                   ptv_w(const_cast<double *>(x), w.data(), z, static_cast<int>(n));
                               }});
            std::snprintf(title, sizeof title, "Weighted 1D TV, lambda = %g times [0.5, 1.5]", lam);
            profile(title, wtd, kind, n, reps, cnt, rng);

        }
    }
    return 0;
}
