// Thread pool for the batching layer; see parallel.hpp.
//
// One pool per process, created on first use and grown to the largest thread
// count requested. Between jobs workers spin briefly, then sleep on a
// condition variable. The
// pool is never destroyed: joining threads from static destructors can
// deadlock (under the Windows loader lock, for instance), and the threads end
// with the process. After fork() the child has none of the parent's threads,
// so a pool created by another process id is abandoned (never touched again)
// and a new one built.
//
// SPDX-License-Identifier: MIT
#include "core/parallel.hpp"

#include <algorithm>
#include <cstdlib>

#ifndef FASTPROX_NO_THREADS
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#endif

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(_M_ARM64)
#include <intrin.h>
#endif
#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace fp {

#ifdef FASTPROX_NO_THREADS

int default_threads() noexcept { return 1; }

void parallel_run(int, int64_t count, int64_t, ChunkFn fn, void *ctx) noexcept {
    if (count > 0) fn(ctx, 0, 0, count);
}

#else

namespace {

int env_threads(const char *name) noexcept {
    const char *s = std::getenv(name);
    if (!s) return 0;
    char *end = nullptr;
    const long v = std::strtol(s, &end, 10);  // OMP_NUM_THREADS may be a list: "8,2"
    return end != s && v > 0 && v <= 4096 ? static_cast<int>(v) : 0;
}

int available_cpus() noexcept {
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int c = CPU_COUNT(&set);
        if (c > 0) return c;
    }
#endif
    const unsigned h = std::thread::hardware_concurrency();
    return h ? static_cast<int>(h) : 1;
}

long process_id() noexcept {
#if defined(_WIN32)
    return 0;  // no fork
#else
    return static_cast<long>(getpid());
#endif
}

// Spin this long before sleeping, so that back-to-back calls (a solver loop)
// find their workers awake, as OpenMP runtimes do.
constexpr auto kSpin = std::chrono::microseconds(50);

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
    __yield();
#else
    __asm__ __volatile__("yield");
#endif
#endif
}

// Spins on pred() for up to kSpin; true when it became true.
template <class Pred>
bool spin_until(Pred pred) noexcept {
    const auto end = std::chrono::steady_clock::now() + kSpin;
    for (;;) {
        for (int i = 0; i < 64; ++i) {
            if (pred()) return true;
            cpu_relax();
        }
        if (std::chrono::steady_clock::now() > end) return pred();
    }
}

class Pool {
  public:
    Pool(long pid, Pool *abandoned) noexcept : pid_(pid), abandoned_(abandoned) {}
    long pid() const noexcept { return pid_; }

    // False when another caller holds the pool.
    bool run(int nt, int64_t count, int64_t grain, ChunkFn fn, void *ctx) noexcept {
        std::unique_lock<std::mutex> busy(run_m_, std::try_to_lock);
        if (!busy.owns_lock()) return false;
        grow(nt - 1);
        const int nw = std::min({nt - 1, nthreads_.load(std::memory_order_relaxed), 0xffff});
        fn_ = fn;
        ctx_ = ctx;
        count_ = count;
        grain_ = grain;
        next_.store(0, std::memory_order_relaxed);
        pending_.store(nw, std::memory_order_relaxed);
        {
            // Generation and participant count in one word, so that a worker
            // reads a consistent pair; the job fields above are published with it.
            std::lock_guard<std::mutex> lk(m_);
            const uint64_t gen = (state_.load(std::memory_order_relaxed) >> 16) + 1;
            state_.store(gen << 16 | static_cast<uint64_t>(nw), std::memory_order_release);
        }
        if (nw > 0) wake_.notify_all();
        work(0);
        const auto done = [&] { return pending_.load(std::memory_order_acquire) == 0; };
        if (!spin_until(done)) {
            std::unique_lock<std::mutex> lk(m_);
            done_.wait(lk, done);
        }
        return true;
    }

  private:
    void work(int tid) noexcept {
        for (;;) {
            const int64_t b = next_.fetch_add(grain_, std::memory_order_relaxed);
            if (b >= count_) return;
            fn_(ctx_, tid, b, std::min(count_, b + grain_));
        }
    }

    // Called with run_m_ held. Allocates only when the pool grows.
    void grow(int want) noexcept {
        int have = nthreads_.load(std::memory_order_relaxed);
        if (have >= want) return;
        try {
            threads_.reserve(static_cast<size_t>(want));
            // A new worker starts from the current generation, so it takes
            // part in the job about to be posted however late it starts.
            const uint64_t gen = state_.load(std::memory_order_relaxed) >> 16;
            for (; have < want; ++have) {
                threads_.emplace_back([this, have, gen] { worker(have, gen); });
                nthreads_.store(have + 1, std::memory_order_relaxed);
            }
        } catch (...) {
            // run with the threads we have
        }
    }

    void worker(int index, uint64_t seen) noexcept {
        for (;;) {
            const auto posted = [&] {
                return (state_.load(std::memory_order_acquire) >> 16) != seen;
            };
            if (!spin_until(posted)) {
                std::unique_lock<std::mutex> lk(m_);
                wake_.wait(lk, posted);
            }
            const uint64_t st = state_.load(std::memory_order_acquire);
            seen = st >> 16;
            if (index >= static_cast<int>(st & 0xffff)) continue;
            work(index + 1);
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(m_);
                done_.notify_one();
            }
        }
    }

    const long pid_;
    [[maybe_unused]] Pool *const abandoned_;  // a parent process's pool, kept reachable on purpose
    std::mutex run_m_;  // one job at a time; guards threads_ and the job fields
    std::mutex m_;      // for the condition variables
    std::condition_variable wake_, done_;
    std::vector<std::thread> threads_;
    std::atomic<int> nthreads_{0};
    std::atomic<uint64_t> state_{0};  // generation << 16 | participating workers
    std::atomic<int> pending_{0};
    std::atomic<int64_t> next_{0};
    ChunkFn fn_ = nullptr;
    void *ctx_ = nullptr;
    int64_t count_ = 0, grain_ = 1;
};

std::atomic<Pool *> g_pool{nullptr};

Pool *get_pool() noexcept {
    const long pid = process_id();
    Pool *p = g_pool.load(std::memory_order_acquire);
    if (p && p->pid() == pid) return p;
    Pool *fresh = new (std::nothrow) Pool(pid, p);
    if (!fresh) return nullptr;
    if (g_pool.compare_exchange_strong(p, fresh, std::memory_order_acq_rel)) return fresh;
    delete fresh;  // another thread installed one first
    return p->pid() == pid ? p : nullptr;
}

}  // namespace

int default_threads() noexcept {
    static const int n = [] {
        if (int t = env_threads("FASTPROX_NUM_THREADS")) return t;
        if (int t = env_threads("OMP_NUM_THREADS")) return t;
        return available_cpus();
    }();
    return n;
}

void parallel_run(int nt, int64_t count, int64_t grain, ChunkFn fn, void *ctx) noexcept {
    if (count <= 0) return;
    grain = std::max<int64_t>(grain, 1);
    nt = static_cast<int>(std::min<int64_t>(nt, (count + grain - 1) / grain));
    if (nt > 1) {
        Pool *p = get_pool();
        if (p && p->run(nt, count, grain, fn, ctx)) return;
    }
    fn(ctx, 0, 0, count);
}

#endif

}  // namespace fp
