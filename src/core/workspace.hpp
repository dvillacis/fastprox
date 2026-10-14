// Workspace behind the opaque fp_workspace handle: per-thread scratch and the
// structure saved by a prox call for later jvp/vjp calls. Everything is sized
// at creation, so the entry points never allocate when given a workspace.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "core/layout.hpp"
#include "core/op.hpp"

#include <cstdint>
#include <memory>

namespace fp {

constexpr int64_t kChunk = 16384;          // elementwise work unit
constexpr int64_t kParallelMin = 1 << 15;  // below this many entries, run serially
constexpr int64_t kMaxBlocks = 64;         // vjp reduction blocks

inline int64_t ceil_div(int64_t a, int64_t b) noexcept { return (a + b - 1) / b; }

// Owning array that is not zero-filled: workspace memory is scratch, and
// clearing megabytes on every one-shot call would cost more than the kernels.
template <class T>
class Buffer {
  public:
    void resize(size_t n) {
        p_.reset(n ? new T[n] : nullptr);  // default-initialised: no clearing
        n_ = n;
    }
    T *data() const noexcept { return p_.get(); }
    size_t size() const noexcept { return n_; }

  private:
    std::unique_ptr<T[]> p_;
    size_t n_ = 0;
};

// Number of fixed reduction blocks for a vjp over `units` work units (rows or
// chunks). It depends only on the problem, never on the thread count, so
// parameter gradients are bit-identical for any number of threads.
inline int64_t reduction_blocks(int64_t units, int64_t nparams) noexcept {
    int64_t nb = std::min<int64_t>(units, kMaxBlocks);
    const int64_t cap = std::max<int64_t>(1, (int64_t(1) << 22) / (nparams + 2));
    nb = std::min(nb, cap);
    return std::max<int64_t>(nb, 1);
}

}  // namespace fp

struct fp_workspace {
    const fp_op *op = nullptr;
    fp_dtype dtype = FP_F64;
    int ndim = 1;
    int64_t shape[FP_MAX_NDIM] = {1};
    int axis = 0;
    int64_t n = 1, rows = 1, numel = 1;
    int nthreads = 1;
    bool keep = false;

    // Per-thread scratch: four rows of T plus the kernel's own scratch.
    size_t thread_bytes = 0;
    fp::Buffer<double> scratch;

    // Saved structure, in canonical order: C order for elementwise operators,
    // rows (axis last) for row operators.
    fp::Buffer<double> xsave, zsave;  // raw storage for T
    fp::Buffer<uint8_t> code;
    fp::Buffer<double> acc;      // reduction blocks, nblocks * (P + 2)
    fp::Buffer<double> tot;      // reduced totals, P + 2
    fp::Buffer<double> dparams;  // dparams converted to double
    int64_t nblocks = 1;

    // State of the last prox call.
    bool valid = false;
    bool conj = false;
    double gamma = 0.0;  // step used by the kernel (1/sigma in conjugate mode)
    double sigma = 0.0;

    void *thread_scratch(int t) const noexcept {
        return reinterpret_cast<char *>(scratch.data()) + static_cast<size_t>(t) * thread_bytes;
    }
    template <class T>
    T *xs() noexcept { return reinterpret_cast<T *>(xsave.data()); }
    template <class T>
    const T *xs() const noexcept { return reinterpret_cast<const T *>(xsave.data()); }
    template <class T>
    T *zs() noexcept { return reinterpret_cast<T *>(zsave.data()); }
    template <class T>
    const T *zs() const noexcept { return reinterpret_cast<const T *>(zsave.data()); }
};
