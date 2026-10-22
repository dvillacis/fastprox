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

    // Image operators: images of image_numel entries (the last image_ndim axes),
    // `rows` of them. Working sets for up to image_sets images at once, thread
    // scratch inside an image, saved structure per image, and the dual of the
    // last call: per image, image_ndim blocks of image_numel doubles.
    int image_ndim = 0;
    int64_t image_numel = 0;
    int image_sets = 0;
    size_t image_bytes = 0, image_thread_bytes = 0, struct_bytes = 0;
    fp::Buffer<double> image_work, image_thread, structure, dual;
    fp::Buffer<double> image_gaps, image_iters;  // per image results of the last call
    // Derivative solves (conjugate gradients) of image operators: settings,
    // per image results, and the worst of the last jvp/vjp.
    double deriv_tol = 0.0;
    int deriv_max_iter = 10000;
    fp::Buffer<double> deriv_res, deriv_its;
    mutable double deriv_residual = 0.0;
    mutable int deriv_iters = 0;
    bool dual_valid = false;  // dual holds the result of a call or a seed
    bool warm = false;        // fp_workspace_set_warm: every call starts from dual
    bool seeded = false;      // fp_workspace_set_dual: the next call starts from dual

    // Branch override of elementwise operators with kinks
    // (fp_workspace_set_branch): one side per entry in C order, applied to
    // the entries within branch_tol of a breakpoint, until the next prox call.
    fp::Buffer<int8_t> branch;
    bool branch_on = false;
    double branch_tol = 0.0;
    // Since 0.3, row and image operators apply an override to their saved
    // structure instead: rows keep the codes of the prox call in code_default;
    // images rebuild their structure from the saved x and z, and keep the
    // prox call's in struct_default (since 0.4; sized by the first override).
    // override_on: the saved structure holds an override (cleared by the next
    // prox call).
    fp::Buffer<uint8_t> code_default;
    fp::Buffer<double> struct_default;
    bool override_on = false;

    // Scratch of fp_spectral_jvp (since 1.1), per thread: sized by its first
    // call, so that workspaces of operators never used on a spectrum do not
    // carry it.
    mutable fp::Buffer<double> spectral;

    // Device and stream (since 1.1): FP_DEVICE_CPU until the CUDA plugin
    // creates workspaces.
    fp_device device = FP_DEVICE_CPU;
    void *stream = nullptr;

    // fp_info and status of the last prox call (fp_workspace_info), whether or
    // not it kept its structure.
    fp_info info = {0.0, 0, 0};
    fp_status info_status = FP_OK;
    bool info_valid = false;

    // State of the last prox call.
    bool valid = false;
    bool conj = false;
    bool conj_homog = false;  // conj by y - prox_g(y): derivatives constant in sigma
    double gamma = 0.0;  // step used by the kernel (1/sigma in conjugate mode)
    double sigma = 0.0;
    int tie = 0;  // fp_tie rule of the last prox call
    // One step per row (fp_prox_steps, since 1.1): the steps of the last kept
    // call in canonical row order, sized with the saved structure.
    fp::Buffer<double> steps;
    bool steps_on = false;
    double step(int64_t r) const noexcept { return steps_on ? steps.data()[r] : gamma; }

    char *image_set(int k) const noexcept {
        return reinterpret_cast<char *>(image_work.data()) + static_cast<size_t>(k) * image_bytes;
    }
    char *image_threads(int k, int nt_inner) const noexcept {
        return reinterpret_cast<char *>(image_thread.data()) +
               static_cast<size_t>(k) * static_cast<size_t>(nt_inner) * image_thread_bytes;
    }
    char *image_struct(int64_t r) const noexcept {
        return reinterpret_cast<char *>(structure.data()) + static_cast<size_t>(r) * struct_bytes;
    }

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
