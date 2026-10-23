// Thread pool for the batching layer; see parallel.hpp.
//
// One pool per process, created on first use and grown to the largest thread
// count requested. Between jobs workers spin briefly (FASTPROX_SPIN_US, 50
// µs by default), then sleep on a condition variable. A call returns once
// every piece is done and every worker that entered the job has left; it
// does not wait for workers still waking up (audit C6). The
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
int max_threads() noexcept { return 1; }

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
// find their workers awake, as OpenMP runtimes do: FASTPROX_SPIN_US
// microseconds, read once, from 0 (sleep at once) to 100 000, default 50
// (audit C9).
std::chrono::microseconds spin_time() noexcept {
    static const std::chrono::microseconds t = [] {
        long v = 50;
        if (const char *s = std::getenv("FASTPROX_SPIN_US")) {
            char *end = nullptr;
            const long u = std::strtol(s, &end, 10);
            if (end != s && *end == '\0' && u >= 0) v = std::min(u, 100000L);
        }
        return std::chrono::microseconds(v);
    }();
    return t;
}

// Set while this thread runs pieces of a pool job, so that a nested
// parallel_run runs serially instead of locking run_m_ again (audit C10:
// try_lock on a mutex the caller holds is undefined).
thread_local bool t_in_job = false;
struct InJob {
    bool prev = t_in_job;
    InJob() noexcept { t_in_job = true; }
    ~InJob() { t_in_job = prev; }
};

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

// Spins on pred() for up to spin_time(); true when it became true. With a
// spin time of 0, one check.
template <class Pred>
bool spin_until(Pred pred) noexcept {
    const auto spin = spin_time();
    if (spin.count() == 0) return pred();
    const auto end = std::chrono::steady_clock::now() + spin;
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
    //
    // The job is announced in ctl_ (generation, CLOSED, workers allowed in,
    // workers inside). The caller works too, closes the job once every piece
    // is claimed, and waits only for the workers inside it: a worker that
    // wakes late finds the job closed and goes back to wait. The job fields
    // are written only while no worker is inside a job.
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
        {
            // The job fields above are published with the new generation.
            std::lock_guard<std::mutex> lk(m_);
            const uint64_t gen = gen_of(ctl_.load(std::memory_order_relaxed)) + 1;
            ctl_.store(gen << kGenShift | static_cast<uint64_t>(nw) << kNwShift, std::memory_order_release);
        }
        if (nw > 0) wake_.notify_all();
        work(0);  // returns once every piece is claimed
        const uint64_t w = ctl_.fetch_or(kClosed, std::memory_order_acq_rel) | kClosed;
        if (active_of(w) != 0) {
            // The workers inside finish their pieces; their decrements release
            // their output to this acquire.
            const auto done = [&] { return active_of(ctl_.load(std::memory_order_acquire)) == 0; };
            if (!spin_until(done)) {
                std::unique_lock<std::mutex> lk(m_);
                done_.wait(lk, done);
            }
        }
        return true;
    }

  private:
    void work(int tid) noexcept {
        const InJob in_job;
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
            // A new worker starts from the current generation, so it enters
            // the job about to be posted if it starts before that job closes.
            const uint64_t gen = gen_of(ctl_.load(std::memory_order_relaxed));
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
            const auto posted = [&] { return gen_of(ctl_.load(std::memory_order_acquire)) != seen; };
            if (!spin_until(posted)) {
                std::unique_lock<std::mutex> lk(m_);
                wake_.wait(lk, posted);
            }
            // Enter the open job, if this worker is allowed in and it is not
            // closed yet; otherwise note its generation and wait for the next.
            // A worker that slept through several jobs enters whichever is
            // open: the fields it then reads are that job's.
            uint64_t w = ctl_.load(std::memory_order_acquire);
            bool entered = false;
            for (;;) {
                if ((w & kClosed) || index >= nw_of(w)) {
                    seen = gen_of(w);
                    break;
                }
                if (ctl_.compare_exchange_weak(w, w + 1, std::memory_order_acquire, std::memory_order_acquire)) {
                    seen = gen_of(w);
                    entered = true;
                    break;
                }
            }
            if (!entered) continue;
            work(index + 1);
            const uint64_t left = ctl_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if ((left & kClosed) && active_of(left) == 0) {
                std::lock_guard<std::mutex> lk(m_);
                done_.notify_one();
            }
        }
    }

    // ctl_: generation (31 bits) << 33 | CLOSED << 32 | workers allowed in
    // (16 bits) << 16 | workers inside (16 bits).
    static constexpr int kGenShift = 33, kNwShift = 16;
    static constexpr uint64_t kClosed = uint64_t(1) << 32;
    static uint64_t gen_of(uint64_t w) noexcept { return w >> kGenShift; }
    static int nw_of(uint64_t w) noexcept { return static_cast<int>((w >> kNwShift) & 0xffff); }
    static int active_of(uint64_t w) noexcept { return static_cast<int>(w & 0xffff); }

    const long pid_;
    [[maybe_unused]] Pool *const abandoned_;  // a parent process's pool, kept reachable on purpose
    std::mutex run_m_;  // one job at a time; guards threads_ and the job fields
    std::mutex m_;      // for the condition variables
    std::condition_variable wake_, done_;
    std::vector<std::thread> threads_;
    std::atomic<int> nthreads_{0};
    std::atomic<uint64_t> ctl_{0};
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

int max_threads() noexcept {
    static const int n = std::clamp(4 * available_cpus(), 1, 4096);
    return n;
}

int default_threads() noexcept {
    static const int n = [] {
        if (int t = env_threads("FASTPROX_NUM_THREADS")) return std::min(t, max_threads());
        if (int t = env_threads("OMP_NUM_THREADS")) return std::min(t, max_threads());
        return std::min(available_cpus(), max_threads());
    }();
    return n;
}

void parallel_run(int nt, int64_t count, int64_t grain, ChunkFn fn, void *ctx) noexcept {
    if (count <= 0) return;
    if (t_in_job) {  // nested inside a job: serially, on this thread (audit C10)
        fn(ctx, 0, 0, count);
        return;
    }
    grain = std::max<int64_t>(grain, 1);
    nt = static_cast<int>(std::min<int64_t>({nt, (count + grain - 1) / grain, max_threads()}));
    if (nt > 1) {
        Pool *p = get_pool();
        if (p && p->run(nt, count, grain, fn, ctx)) return;
    }
    fn(ctx, 0, 0, count);
}

#endif

}  // namespace fp
