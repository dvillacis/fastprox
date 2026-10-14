// Thread pool for the batching layer.
//
// fastprox runs its rows on its own threads instead of OpenMP, so that no
// OpenMP runtime is linked or shipped: two copies of LLVM's libomp in one
// process abort each other (as with PyTorch's on macOS), and a runtime
// bundled in each wheel is exactly such a copy.
//
// Work is handed out dynamically in pieces of `grain` items, so fast and slow
// cores balance. Which thread computes which piece does not change any
// result: the batching layer only splits independent rows or fixed blocks.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <type_traits>

namespace fp {

// Threads used when fp_options.threads is 0: FASTPROX_NUM_THREADS, else the
// first number in OMP_NUM_THREADS, else the CPUs this process may run on.
// Read once; 1 in builds without threads.
int default_threads() noexcept;

using ChunkFn = void (*)(void *ctx, int tid, int64_t begin, int64_t end);

// Calls fn(ctx, tid, b, e) on pieces [b, e) covering [0, count), with tid in
// [0, nt) naming the caller's per-thread scratch. The caller takes part as
// tid 0. Runs serially (one call with tid 0) when nt <= 1, when another
// caller is using the pool, or when no thread can be started. fn must not
// throw and must not call parallel_run.
void parallel_run(int nt, int64_t count, int64_t grain, ChunkFn fn, void *ctx) noexcept;

template <class F>
void parallel_for(int nt, int64_t count, int64_t grain, F &&body) noexcept {
    using B = std::remove_reference_t<F>;
    parallel_run(
        nt, count, grain,
        [](void *c, int tid, int64_t b, int64_t e) { (*static_cast<B *>(c))(tid, b, e); },
        const_cast<void *>(static_cast<const void *>(&body)));
}

}  // namespace fp
