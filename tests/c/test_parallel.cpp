// The thread pool (src/core/parallel.cpp), compiled into this test: a
// parallel_run from inside a job runs serially as tid 0 of its own call
// instead of locking the pool again (audit C10), and thread counts are
// capped (audit C3).
//
// SPDX-License-Identifier: MIT
#include "core/parallel.hpp"

#include <atomic>
#include <cstdio>
#include <vector>

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_fail;                                                                 \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                             \
    } while (0)

int main() {
    // An outer job of 64 items whose every item runs an inner job of 16.
    std::vector<std::atomic<int>> seen(64 * 16);
    for (auto &v : seen) v = 0;
    std::atomic<int> bad_tid{0};
    fp::parallel_for(8, 64, 1, [&](int, int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i)
            fp::parallel_for(8, 16, 1, [&](int tid, int64_t ib, int64_t ie) {
                if (tid != 0) ++bad_tid;
                for (int64_t j = ib; j < ie; ++j) ++seen[static_cast<size_t>(i * 16 + j)];
            });
    });
    bool once = true;
    for (auto &v : seen) once = once && v.load() == 1;
    CHECK(once);
    CHECK(bad_tid.load() == 0);
    // After a job the pool is usable again with threads.
    std::atomic<int64_t> sum{0};
    fp::parallel_for(8, 1000, 10, [&](int, int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) sum += i;
    });
    CHECK(sum.load() == 999 * 1000 / 2);
    CHECK(fp::max_threads() >= 1 && fp::max_threads() <= 4096);
    CHECK(fp::default_threads() <= fp::max_threads());
    std::printf("%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
