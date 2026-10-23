// The thread pool (src/core/parallel.cpp), compiled into this test: a
// parallel_run from inside a job runs serially as tid 0 of its own call
// instead of locking the pool again (audit C10), thread counts are capped
// (audit C3), and a call that closes its job before late workers arrive
// still runs every piece exactly once (audit C6). CTest also runs it with
// FASTPROX_SPIN_US=0 (audit C9), where workers sleep at once.
//
// SPDX-License-Identifier: MIT
#include "core/parallel.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
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
    // Late workers (audit C6): 10 000 jobs of 15 one-item pieces, every other
    // one after a 300 us sleep, so that workers are asleep and wake after the
    // caller has claimed the pieces. Each item runs exactly once, on a tid
    // below 15.
    {
        int wrong = 0, bad = 0;
        for (int call = 0; call < 10000; ++call) {
            if (call % 2 == 0) std::this_thread::sleep_for(std::chrono::microseconds(300));
            std::atomic<int> hits[15];
            int tids[15];
            for (int i = 0; i < 15; ++i) {
                hits[i] = 0;
                tids[i] = -1;
            }
            fp::parallel_for(15, 15, 1, [&](int tid, int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    ++hits[i];
                    tids[i] = tid;
                }
            });
            for (int i = 0; i < 15; ++i) {
                wrong += hits[i].load() != 1;
                bad += tids[i] < 0 || tids[i] >= 15;
            }
        }
        CHECK(wrong == 0);
        CHECK(bad == 0);
    }
    // Several callers at once: one gets the pool, the others run serially;
    // every sum is right.
    {
        std::atomic<int> errors{0};
        std::vector<std::thread> callers;
        for (int k = 0; k < 8; ++k)
            callers.emplace_back([&] {
                for (int rep = 0; rep < 200; ++rep) {
                    std::atomic<int64_t> s{0};
                    fp::parallel_for(8, 4096, 16, [&](int, int64_t b, int64_t e) {
                        int64_t local = 0;
                        for (int64_t i = b; i < e; ++i) local += i;
                        s += local;
                    });
                    if (s.load() != 4095 * 4096 / 2) ++errors;
                }
            });
        for (auto &t : callers) t.join();
        CHECK(errors.load() == 0);
    }
    CHECK(fp::max_threads() >= 1 && fp::max_threads() <= 4096);
    CHECK(fp::default_threads() <= fp::max_threads());
    std::printf("%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
