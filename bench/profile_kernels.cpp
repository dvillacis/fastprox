// Single-thread profile of 1D TV: fastprox against proxTV, with hardware
// counters where the machine exposes them.
//
//   c++ -O2 -std=c++17 bench/profile_kernels.cpp -Iinclude -Lbuild/c -lfastprox -ldl \
//       -Wl,-rpath,build/c -o profile_kernels
//   ./profile_kernels [path/to/libproxtv_tv1.so] [n] [reps]
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
#include <string>
#include <vector>

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

int main(int argc, char **argv) {
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
