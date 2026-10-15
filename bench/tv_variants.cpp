// Variants of the 1D TV kernel side by side, against proxTV, in one binary:
// the same fresh input for every solver in each repetition, the order
// rotated per repetition, medians reported. For comparing kernel changes on
// a machine at hand (the `variants` workflow runs it on GitHub's runners),
// where comparisons across separate runs would mix in the CPU each job
// landed on.
//
//   c++ -std=c++17 -O3 -DNDEBUG -Iinclude -Isrc bench/tv_variants.cpp \
//       src/api/api.cpp src/core/parallel.cpp src/ops/pointwise.cpp src/ops/group.cpp \
//       src/ops/projections.cpp src/ops/isotonic.cpp -DFASTPROX_BUILDING \
//       -DFASTPROX_VERSION_STRING='"variants"' -ldl -lpthread -o tv_variants
//   ./tv_variants [path/to/libproxtv_tv1.so] [n] [reps]
//
// The kernel is compiled into this binary from src/ops/tv1d.cpp, with the
// library's flags; the variants are its template parameters (see VARIANTS).
// proxTV is loaded through the shim that bench/compare_external.py builds
// and is only called, never read.
//
// SPDX-License-Identifier: MIT
#include "ops/tv1d.cpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using fp::condat;
using fp::condat_budget;

// Uniform row with radius lam, or weighted row with radii w[e] (w[n-1] = 0).
using Kernel = std::function<bool(const double *, double *, int64_t, double, const double *)>;

template <int64_t Short, int64_t Adapt = 0>
Kernel switch_at() {
    return [](const double *x, double *z, int64_t n, double lam, const double *w) {
        if (w) return condat<Short, Adapt>(x, z, n, [w](int64_t e) { return w[e]; }, nullptr, condat_budget(n));
        return condat<Short, Adapt>(x, z, n, [lam](int64_t) { return lam; }, nullptr, condat_budget(n));
    };
}

struct Variant {
    const char *name;
    Kernel run;
};

// The first entry is the baseline (the library's kernel as compiled).
const std::vector<Variant> VARIANTS = {
    {"kernel (kShort = 3)", switch_at<fp::kShort>()},
    {"switch at 1", switch_at<1>()},
    {"3, at 1 after > 3", switch_at<3, 3>()},
    {"3, at 1 after > 8", switch_at<3, 8>()},
    {"3, at 1 after > 16", switch_at<3, 16>()},
    {"2, at 1 after > 8", switch_at<2, 8>()},
};

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

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
    const char *ptv_path = argc > 1 && argv[1][0] ? argv[1] : nullptr;
    const int64_t n = argc > 2 ? std::atoll(argv[2]) : 100000;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 31;

    using Taut = int (*)(double *, int, double, double *);
    Taut ptv_classic = nullptr;
    using HybridTaut = void (*)(double *, int, double, double *);
    using WTaut = int (*)(double *, double *, double *, int);
    Taut ptv_lin = nullptr;
    HybridTaut ptv_hyb = nullptr;
    WTaut ptv_w = nullptr;
    if (ptv_path) {
        if (void *h = dlopen(ptv_path, RTLD_NOW)) {
            ptv_classic = reinterpret_cast<Taut>(dlsym(h, "shim_classic_taut"));
            ptv_lin = reinterpret_cast<Taut>(dlsym(h, "shim_linearized_taut"));
            ptv_hyb = reinterpret_cast<HybridTaut>(dlsym(h, "shim_hybrid_taut"));
            ptv_w = reinterpret_cast<WTaut>(dlsym(h, "shim_weighted_taut"));
        } else {
            std::fprintf(stderr, "proxTV unavailable: %s\n", dlerror());
        }
    }
    std::printf("1D TV kernel variants, single thread, n = %lld, %d reps; ns per sample, and time over\n"
                "the best proxTV solver (with proxTV) or over the baseline (without)\n",
                static_cast<long long>(n), reps);

    std::mt19937_64 rng(20261016);
    std::vector<double> x(static_cast<size_t>(n)), w(static_cast<size_t>(n));
    for (const char *kind : {"blocks", "random walk", "white noise"}) {
        for (double lam : {0.1, 1.0, 10.0}) {
            for (int weighted = 0; weighted < 2; ++weighted) {
                std::mt19937_64 wr(7);
                std::uniform_real_distribution<double> U(0.5, 1.5);
                for (int64_t e = 0; e + 1 < n; ++e) w[static_cast<size_t>(e)] = lam * U(wr);
                w[static_cast<size_t>(n - 1)] = 0.0;
                const double *wp = weighted ? w.data() : nullptr;

                struct Row {
                    std::string name;
                    std::function<void(double *)> run;
                    bool external;
                };
                std::vector<Row> rows;
                for (const auto &v : VARIANTS)
                    rows.push_back({v.name, [&, k = v.run](double *z) {
                                        if (!k(x.data(), z, n, lam, wp)) std::abort();  // budget: not expected here
                                    }, false});
                if (weighted && ptv_w)
                    rows.push_back({"proxTV weighted taut", [&](double *z) { ptv_w(x.data(), w.data(), z, static_cast<int>(n)); }, true});
                if (!weighted && ptv_classic)
                    rows.push_back({"proxTV classic taut", [&](double *z) { ptv_classic(x.data(), static_cast<int>(n), lam, z); }, true});
                if (!weighted && ptv_lin)
                    rows.push_back({"proxTV linearized taut", [&](double *z) { ptv_lin(x.data(), static_cast<int>(n), lam, z); }, true});
                if (!weighted && ptv_hyb)
                    rows.push_back({"proxTV hybrid taut", [&](double *z) { ptv_hyb(x.data(), static_cast<int>(n), lam, z); }, true});

                const size_t S = rows.size();
                std::vector<std::vector<double>> out(S, std::vector<double>(x.size())), t(S);
                double maxdiff = 0.0;
                for (int r = 0; r <= reps; ++r) {  // round 0 warms up
                    signal(kind, x, rng);
                    for (size_t j = 0; j < S; ++j) {
                        const size_t s = (j + static_cast<size_t>(r)) % S;
                        const auto t0 = std::chrono::steady_clock::now();
                        rows[s].run(out[s].data());
                        const auto t1 = std::chrono::steady_clock::now();
                        if (r) t[s].push_back(std::chrono::duration<double>(t1 - t0).count() * 1e9 / static_cast<double>(n));
                    }
                    for (size_t s = 1; s < S; ++s)
                        if (!rows[s].external)
                            for (size_t i = 0; i < x.size(); ++i)
                                maxdiff = std::max(maxdiff, std::fabs(out[s][i] - out[0][i]));
                }
                double best_ext = INFINITY;
                for (size_t s = 0; s < S; ++s)
                    if (rows[s].external) best_ext = std::min(best_ext, median(t[s]));
                const double ref = std::isfinite(best_ext) ? best_ext : median(t[0]);
                std::printf("\n%s%s, lambda = %g (max |variant - baseline| %.1e)\n", weighted ? "Weighted, " : "", kind, lam, maxdiff);
                for (size_t s = 0; s < S; ++s)
                    std::printf("  %-26s %7.2f %6.2f\n", rows[s].name.c_str(), median(t[s]), median(t[s]) / ref);
            }
        }
    }
    return 0;
}
