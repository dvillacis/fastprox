// Operator base class behind the opaque fp_op handle.
//
// Two kinds of operator exist. Elementwise operators (soft thresholding, box,
// ...) map each entry independently and are driven over the whole array in
// fixed chunks. Row operators (projections, group norms, 1D TV, ...) couple
// the entries along the operator axis and are driven one contiguous row at a
// time. In both cases the kernels are templates on the scalar type; the CRTP
// helpers below turn them into the float and double virtual overloads.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "fastprox.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

// Target of min_nonan / max_nonan (see there).
#if defined(__aarch64__) || defined(_M_ARM64)
#define FP_MINMAX_ARM64 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define FP_MINMAX_SSE2 1
#include <emmintrin.h>
#endif

// Out of line and cold: a rare path kept out of the hot code that calls it.
// FP_INLINE: inlined even past the compiler's size threshold, for a per-entry
// function whose reference arguments would otherwise go through memory.
#if defined(_MSC_VER)
#define FP_COLD __declspec(noinline)
#define FP_INLINE __forceinline
#else
#define FP_COLD __attribute__((noinline, cold))
#define FP_INLINE inline __attribute__((always_inline))
#endif

namespace fp {

// The step range (audit A1): gamma step_scale() at most 2^960, which leaves
// 2^64 of headroom for the kernels' sums of scaled thresholds and capacities.
// Kernels that rescale their input keep the scaled step within it too.
constexpr double kMaxScaledStep = 0x1p960;

// Derivative of one entry of an elementwise prox:
//   dz = j dx + dg dgamma + sum_k pv[k] dtheta[pi[k]].
struct ElemDeriv {
    double j = 0.0;
    double dg = 0.0;
    int np = 0;
    int64_t pi[3] = {0, 0, 0};
    double pv[3] = {0.0, 0.0, 0.0};
};

// A breakpoint of an elementwise map near an entry: its position b (in
// signed x) and the magnitude of the terms it is computed from, which sets
// the step used to land on either side of it.
struct Kink {
    double b = 0.0, scale = 0.0;
};
// A piece of an elementwise map: an operator-specific id and the sign that
// its formulas use (sgn(x) by default, the side's sign across a kink).
struct Piece {
    int id = 0;
    double s = 0.0;
};

// Status returned by row kernels.
// kRowStepOverflow: in the conjugate, y_i / sigma left the range of the
// array's dtype (FP_EPARAM).
enum RowStatus : int { kRowOk = 0, kRowNonFinite = 1, kRowStepOverflow = 2 };

// True when some finite y_i / sigma is beyond T's range (the conjugate's
// staging; FP_EPARAM). After a false, every finite quotient converts to T
// without the undefined conversion of an out-of-range double. NaN and +-inf
// in y are not flagged. One branch-free pass on the bit patterns: an integer
// OR keeps it vectorised, where a floating-point one would not.
// |y| against sigma * lim shrunk by 2^-50: below it y / sigma is strictly
// inside [-lim, lim], at the price of flagging the few quotients within
// 2^-50 of lim. No division; |y| and the bound compare as integers (the
// order of nonnegative doubles), so loops carrying the test still vectorise.
// For sigma >= 1 no finite y exceeds it. NaN and +-inf are not flagged.
struct StagingBound {
    uint64_t bt;
    StagingBound(double sigma, double lim) noexcept {
        const double thr = sigma < 1.0 ? sigma * lim * (1.0 - 0x1p-50) : std::numeric_limits<double>::max();
        std::memcpy(&bt, &thr, 8);
    }
    uint64_t exceeded(double y) const noexcept {
        constexpr uint64_t abs = 0x7fffffffffffffffull, inf = 0x7ff0000000000000ull;
        uint64_t by;
        std::memcpy(&by, &y, 8);
        by &= abs;
        return static_cast<uint64_t>((by > bt) & (by < inf));
    }
};

// Below this sigma the conjugate's staging checks y / sigma against T's
// range (FP_EPARAM). Always in float32 (sigma < 1), where converting an
// out-of-range quotient would be undefined. In float64 only below 2^-64 (a
// decision of 2026-10-22: the check cost 15-30% on cheap conjugates): above
// it only |y| > 2^960 can overflow, and gives +-inf or NaN in that output.
template <class T>
constexpr double kStagingCheck = std::is_same_v<T, double> ? 0x1p-64 : 1.0;

template <class T>
inline bool staging_overflows(const T *x, int64_t xs, int64_t len, double sigma) noexcept {
    const StagingBound bound(sigma, std::numeric_limits<T>::max());
    uint64_t bad = 0;
    for (int64_t i = 0; i < len; ++i) bad |= bound.exceeded(static_cast<double>(x[i * xs]));
    return bad != 0;
}

// One image handed to an image operator's prox (see fp_op::image_ndim). The
// image is contiguous in C order. The kernel may spread its own work over nt
// threads (thread scratch t at thread_base + t * thread_bytes), and must give
// the same result for any nt.
struct ImageCall {
    const int64_t *shape = nullptr;  // image_ndim() extents
    int64_t numel = 0;
    double gamma = 0.0;
    double tol = 0.0;  // stop when gap <= tol * ||x||^2 / 2
    int max_iter = 0;
    int nt = 1;
    char *work = nullptr;  // image_bytes(shape)
    char *thread_base = nullptr;
    size_t thread_bytes = 0;
    double *dual = nullptr;  // image_ndim() * numel: read when warm, always written
    bool warm = false;
    char *save = nullptr;  // structure for jvp/vjp (struct_bytes), null unless kept
    // Results.
    double gap = 0.0;
    int iters = 0;
    bool converged = true;
    // An image solved at a power-of-two scale, as 2^-scale_exp x, had the gap
    // scaled_gap there; gap = 2^(2 scale_exp) scaled_gap may overflow or
    // underflow, the certified relative error sqrt(2 gap) / |x| does not.
    int scale_exp = 0;
    double scaled_gap = 0.0;
    // Whether the saved structure gives the exact derivative at the returned
    // z (feedback U4): the regions certified (anisotropic TV's polish, graph
    // TV with no part stopped by tol), or no smoothing (isotropic TV), whose
    // curvature cap is `smoothing`, in the input's units.
    bool exact = true;
    double smoothing = 0.0;

    char *thread_scratch(int t) const noexcept { return thread_base + static_cast<size_t>(t) * thread_bytes; }
};

// The structure of the derivative of one row (since 1.0.2), contiguous: per
// entry its fp_entry_state, its region's label (0, 1, ... in the order of the
// regions' first entries; -1 where not FREE), its sign (0 where not FREE) and
// its coefficient in the row's linear constraint (0 where not FREE, and in a
// row without an active constraint); and whether the constraint is active.
// state is always given; label, sign and coef are null when the query does
// not ask for them, and constraint is false when it asks for neither coef nor
// active, so that a hook skips what nobody reads (the performance gate: a
// query costs at most one jvp).
struct RowStruct {
    int8_t *state = nullptr;
    int64_t *label = nullptr;
    int8_t *sign = nullptr;
    double *coef = nullptr;
    bool constraint = true;  // coef or active asked for
    bool active = false;
};

// Zero coefficients and no constraint.
inline void no_constraint(RowStruct &s, int64_t n) noexcept {
    if (s.coef) std::fill(s.coef, s.coef + n, 0.0);
    s.active = false;
}

// Labels, signs and coefficients of a row whose regions are singletons, from
// its states; no constraint. Returns the number of regions.
inline int64_t singletons(RowStruct &s, int64_t n) noexcept {
    int64_t k = 0;
    if (s.label) {
        for (int64_t i = 0; i < n; ++i) {
            const bool free = s.state[i] == FP_ENTRY_FREE;
            s.label[i] = free ? k : -1;
            k += free;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) k += s.state[i] == FP_ENTRY_FREE;
    }
    if (s.sign)
        for (int64_t i = 0; i < n; ++i) s.sign[i] = s.state[i] == FP_ENTRY_FREE;
    no_constraint(s, n);
    return k;
}

// The same with the FREE entries where in(i) is true forming one region,
// with sign sg(i); the other FREE entries singletons.
template <class In, class Sg>
inline int64_t one_region(RowStruct &s, int64_t n, In in, Sg sg) noexcept {
    int64_t k = 0, shared = -1;
    for (int64_t i = 0; i < n; ++i) {
        int64_t l = -1;
        int8_t g = 0;
        if (s.state[i] != FP_ENTRY_FREE) {
        } else if (in(i)) {
            if (shared < 0) shared = k++;
            l = shared;
            g = static_cast<int8_t>(sg(i));
        } else {
            l = k++;
            g = 1;
        }
        if (s.label) s.label[i] = l;
        if (s.sign) s.sign[i] = g;
    }
    no_constraint(s, n);
    return k;
}

// Every entry FREE with sign +1, in contiguous regions that start where
// starts(i) is true (entry 0 always starts one); no constraint.
template <class St>
inline int64_t runs(RowStruct &s, int64_t n, St starts) noexcept {
    int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) {
        k += i == 0 || starts(i);
        if (s.label) s.label[i] = k - 1;
    }
    std::fill(s.state, s.state + n, int8_t{FP_ENTRY_FREE});
    if (s.sign) std::fill(s.sign, s.sign + n, int8_t{1});
    no_constraint(s, n);
    return k;
}

// Iterative derivative solves of an image operator (conjugate gradients):
// tolerance on the relative residual and iteration cap in, the achieved
// residual and iterations out. Operators with closed-form derivatives leave
// the outputs at zero.
struct DerivCtl {
    double tol = 0.0;  // 0: the operator's default
    int max_iter = 10000;
    double residual = 0.0;
    int iters = 0;
    int nt = 1;  // threads for one image (operators with threads_derivatives())
};

}  // namespace fp

// Global namespace: this is the type the C header declares as opaque.
struct fp_op {
    virtual ~fp_op() = default;

    virtual const char *name() const noexcept = 0;
    virtual bool elementwise() const noexcept = 0;
    virtual int64_t nparams() const noexcept = 0;
    // Validates the operator against the length of the operator axis.
    virtual fp_status check_n(int64_t n) const noexcept = 0;
    // Validates the step the kernel will use (1/sigma after fp_prox_conj):
    // FP_EPARAM where the prox is not single-valued (SCAD, MCP).
    virtual fp_status check_step(double) const noexcept { return FP_OK; }
    // True when g is positively homogeneous, g(t z) = t g(z) for t > 0, or the
    // indicator of a cone: then prox_{gamma g}(t x) = t prox_{(gamma/t) g}(x),
    // and fp_prox_conj evaluates y - sigma prox_{g/sigma}(y/sigma) as
    // y - prox_g(y), the same for every sigma > 0 (audit A1).
    virtual bool conj_homogeneous() const noexcept { return false; }
    // False when fp_prox_conj has no Euclidean meaning for this operator:
    // a prox in another metric, whose Moreau identity holds in that metric
    // only (weighted isotonic regression, math audit M2). fp_prox_conj then
    // returns FP_EINVAL.
    virtual bool conj_supported() const noexcept { return true; }
    // An upper bound on the factors the kernel multiplies its step by (lambda,
    // a weight, 1/delta, ...); 0 when it ignores the step (projections). The
    // step range of every call is gamma step_scale() <= kMaxScaledStep (audit A1).
    virtual double step_scale() const noexcept { return 1.0; }
    // Whether one row or image can use several threads on the CPU
    // (fp_op_device_check): elementwise maps, and image operators that say so.
    virtual bool splits_rows() const noexcept { return elementwise(); }
    // True when the prox can be set-valued at thresholds (nonconvex maps):
    // prox_run then applies the tie rule and counts the entries on a tie.

    // ---- elementwise interface -------------------------------------------
    // True when some parameter varies along the operator axis.
    virtual bool axis_params() const noexcept { return false; }
    virtual bool has_ties() const noexcept { return false; }

    // z[i*zs] = prox(x[i*xs]) for i < len. widx is the axis coordinate of the
    // first entry and wstep its increment along the run. save, if not null,
    // receives the (scaled) input contiguously. In conjugate mode the run
    // computes y - sigma prox_{g/sigma}(y/sigma) and gamma is 1/sigma. tie is
    // the fp_tie rule; the return value counts the entries that sat on a tie.
    virtual int64_t prox_run(const float *, int64_t, float *, int64_t, int64_t, double, int64_t,
                             int64_t, float *, bool, double, int) const noexcept { return 0; }
    virtual int64_t prox_run(const double *, int64_t, double *, int64_t, int64_t, double, int64_t,
                             int64_t, double *, bool, double, int) const noexcept { return 0; }
    // xsave is contiguous; dx may be null. tie: the rule of the prox call.
    // side, if not null, is the branch override along the run (contiguous,
    // -1, 0 or +1 per entry), applied to the entries within kink_tol of a
    // breakpoint (operators with has_kinks()).
    virtual void jvp_run(const float *, const float *, int64_t, float *, int64_t, int64_t, double,
                         double, const double *, int64_t, int64_t, bool, double, int,
                         const int8_t *, double) const noexcept {}
    virtual void jvp_run(const double *, const double *, int64_t, double *, int64_t, int64_t,
                         double, double, const double *, int64_t, int64_t, bool,
                         double, int, const int8_t *, double) const noexcept {}
    // acc has nparams() + 2 slots: theta..., gamma, conjugate extra term.
    virtual void vjp_run(const float *, const float *, int64_t, float *, int64_t, int64_t, double,
                         double *, int64_t, int64_t, bool, double, int, const int8_t *,
                         double) const noexcept {}
    virtual void vjp_run(const double *, const double *, int64_t, double *, int64_t, int64_t,
                         double, double *, int64_t, int64_t, bool, double, int, const int8_t *,
                         double) const noexcept {}
    // Kinks (elementwise operators with has_kinks()): for each saved entry of
    // the run, the code of the breakpoint within tol of it (0: none) into
    // codes (may be null); returns the number of flagged entries.
    virtual bool has_kinks() const noexcept { return false; }
    virtual int64_t kink_run(const float *, int64_t, double, int64_t, int64_t, double,
                             int8_t *) const noexcept { return 0; }
    virtual int64_t kink_run(const double *, int64_t, double, int64_t, int64_t, double,
                             int8_t *) const noexcept { return 0; }
    // Kinks of row and image operators with has_kinks() (since 0.3), from the
    // saved input x and solution z (in double) and, for images, the dual of
    // the call: codes into mask (one per entry, at the given strides, the
    // image's axes for kink_image; never null for kink_image); returns the
    // number of flagged entries. kink_image gets the image's saved structure
    // and image_bytes(shape) of work.
    virtual int64_t kink_row(const double *, const double *, int64_t, double, double, int8_t *, int64_t,
                             void *) const noexcept { return 0; }
    virtual int64_t kink_image(const char *, const double *, const double *, const double *, const int64_t *,
                               double, double, int8_t *, const int64_t *, char *) const noexcept { return 0; }
    // Kink sites per image when they are not the image's entries (since 0.4):
    // tree TV's m edges, so that its masks and sides have the shape of x with
    // the last axis of length m. 0: one site per entry.
    virtual int64_t kink_sites() const noexcept { return 0; }
    // Branch overrides (fp_workspace_set_branch): elementwise operators with
    // kinks, and since 0.3 the row and image operators that implement them.
    virtual bool has_branches() const noexcept { return elementwise() && has_kinks(); }
    // Rows: rewrite the saved codes of one row at its kinks within tol by side
    // (one per entry, -1 / 0 / +1), from the saved x and z (in double).
    // kink_row and branch_row get scratch_bytes(n) of scratch.
    virtual void branch_row(const double *, const double *, uint8_t *, int64_t, double, double,
                            const int8_t *, void *) const noexcept {}
    // Images: rebuild the saved structure of one image from x, z (double) and
    // the dual, with side (C order, never null) applied at its kinks within
    // tol; work: image_bytes(shape) of scratch. The API restores the prox
    // call's own structure by a copy.
    virtual void branch_image(char *, const double *, const double *, const double *, const int64_t *,
                              double, double, const int8_t *, char *) const noexcept {}

    // ---- structure of the derivative (since 1.0.2) -------------------------
    // The fp_struct_kind of every point (FP_STRUCT_NONE: no queries).
    virtual int struct_kind() const noexcept { return FP_STRUCT_NONE; }
    // Elementwise operators: the fp_entry_state of each saved entry of the run
    // into state (contiguous), with the arguments of jvp_run (side and tol
    // the branch override, as there).
    virtual void state_run(const float *, int64_t, double, int64_t, int64_t, int, const int8_t *, double,
                           int8_t *) const noexcept {}
    virtual void state_run(const double *, int64_t, double, int64_t, int64_t, int, const int8_t *, double,
                           int8_t *) const noexcept {}

    // ---- row interface ---------------------------------------------------
    // Bytes of kernel scratch needed for one row of length n.
    virtual size_t scratch_bytes(int64_t) const noexcept { return 0; }

    // The value g(x) (since 0.5, fp_value), in double: of one row of length n
    // (elementwise and row operators) or one image of the
    // given shape (image operators). The indicator of a set is 0 when x lies
    // in it up to atol = tol * max(1, max |x_i|) and INFINITY otherwise. scr has
// value_doubles(n) doubles.
    // Spectral use (since 1.0, fp_op_spectral_check): kSpectralSvd when the
    // map is absolutely symmetric (permutation and sign invariant, so it can
    // act on singular values), kSpectralEig when it is permutation invariant
    // (eigenvalues); both bits for the former.
    virtual int spectral() const noexcept { return 0; }
    virtual bool has_value() const noexcept { return false; }
    virtual int64_t value_doubles(int64_t n) const noexcept { return n; }  // scr's length
    virtual double value_row(const double *, int64_t, double, double *) const noexcept { return 0.0; }
    virtual double value_image(const double *, const int64_t *, double) const noexcept { return 0.0; }
    // 1D TV only: the row prox with edge weights given per call (n of them,
    // the last 0), for image operators with weights per edge.
    virtual int prox_edges(const double *, double *, int64_t, double, const double *, void *) const noexcept {
        return fp::kRowNonFinite;
    }
    // True when the kernel cannot write its output over its input.
    virtual bool needs_distinct() const noexcept { return false; }
    // Rows are contiguous. code, if not null, receives n bytes of structure.
    virtual int prox_row(const float *, float *, int64_t, double, void *,
                         uint8_t *) const noexcept {
        return fp::kRowOk;
    }
    virtual int prox_row(const double *, double *, int64_t, double, void *,
                         uint8_t *) const noexcept {
        return fp::kRowOk;
    }
    // x, z, code: the saved row. dx is never null; dparams may be null.
    virtual void jvp_row(const float *, const float *, const uint8_t *, const float *, float *,
                         int64_t, double, double, const double *, void *) const noexcept {}
    virtual void jvp_row(const double *, const double *, const uint8_t *, const double *,
                         double *, int64_t, double, double, const double *,
                         void *) const noexcept {}
    // gx = J^T g; acc[0..P) += (dz/dtheta)^T g, acc[P] += <dz/dgamma, g>.
    virtual void vjp_row(const float *, const float *, const uint8_t *, const float *, float *,
                         int64_t, double, double *, void *) const noexcept {}
    virtual void vjp_row(const double *, const double *, const uint8_t *, const double *,
                         double *, int64_t, double, double *, void *) const noexcept {}
    // The structure of the derivative at a saved row (since 1.0.2), read
    // from what jvp_row reads (x, z, code after any override, the step) into
    // s; returns the row's number of regions. scratch: scratch_bytes(n).
    virtual int64_t struct_row(const float *, const float *, const uint8_t *, int64_t, double, fp::RowStruct &,
                               void *) const noexcept { return 0; }
    virtual int64_t struct_row(const double *, const double *, const uint8_t *, int64_t, double, fp::RowStruct &,
                               void *) const noexcept { return 0; }
    // Image operators (since 1.0.2): the region label of each entry (C order)
    // in the image's saved structure, after any override, and their number;
    // every entry FREE with sign +1, no constraint. The driver renumbers the
    // labels canonically.
    virtual const int32_t *image_labels(const char *, const int64_t *, int64_t *nreg) const noexcept {
        *nreg = 0;
        return nullptr;
    }

    // ---- image interface -------------------------------------------------
    // Image operators (multi-dimensional TV) act on the last image_ndim() axes
    // and are batched over the others; 0 for every other operator.
    virtual int image_ndim() const noexcept { return 0; }
    // Whether prox_image splits one image over ImageCall::nt threads (TV on
    // grids). The drivers give a batch smaller than the thread count all the
    // threads one image at a time only then; graph and tree TV, which run an
    // image on one thread, solve their images in parallel (audit T4).
    virtual bool splits_image() const noexcept { return false; }
    // Whether jvp_image and vjp_image split one image over DerivCtl::nt
    // threads (isotropic TV's conjugate gradients, audit T2); the drivers
    // then run a batch smaller than the thread count one image at a time.
    virtual bool threads_derivatives() const noexcept { return false; }
    virtual fp_status check_image(const int64_t *) const noexcept { return FP_OK; }
    // Bytes of the working set of one image, of the per-thread scratch used
    // inside one image, and of the structure saved for derivatives.
    virtual size_t image_bytes(const int64_t *) const noexcept { return 0; }
    virtual size_t image_thread_bytes(const int64_t *) const noexcept { return 0; }
    virtual size_t struct_bytes(const int64_t *) const noexcept { return 0; }
    // Returns a RowStatus. Fills c.gap, c.iters, c.converged.
    virtual int prox_image(const float *, float *, fp::ImageCall &) const noexcept { return fp::kRowOk; }
    virtual int prox_image(const double *, double *, fp::ImageCall &) const noexcept { return fp::kRowOk; }
    // Derivatives at the structure saved by prox_image. dx is never null;
    // dparams may be null. work: image_bytes(shape) of scratch.
    virtual void jvp_image(const char *, const float *, float *, const int64_t *, double, double,
                           const double *, char *, fp::DerivCtl &) const noexcept {}
    virtual void jvp_image(const char *, const double *, double *, const int64_t *, double, double,
                           const double *, char *, fp::DerivCtl &) const noexcept {}
    // gx = J^T g; acc[0..P) += (dz/dtheta)^T g, acc[P] += <dz/dgamma, g>.
    virtual void vjp_image(const char *, const float *, float *, const int64_t *, double, double *,
                           char *, fp::DerivCtl &) const noexcept {}
    virtual void vjp_image(const char *, const double *, double *, const int64_t *, double, double *,
                           char *, fp::DerivCtl &) const noexcept {}
};

namespace fp {

// CRTP base for elementwise operators. D provides
//   template <class T> T prox1(T x, double gamma, int64_t widx) const noexcept;
//   ElemDeriv deriv1(double x, double gamma, int64_t widx) const noexcept;
// Per-entry evaluation of an elementwise operator D: the tie-aware form
// (D::kTies, nonconvex maps) takes the tie rule and reports a tie; the plain
// form ignores both.
// Bits of fp_op::spectral(): absolutely symmetric maps act on both kinds.
constexpr int kSpectralEig = 1, kSpectralSvd = 2, kSpectralBoth = kSpectralEig | kSpectralSvd;
// Spectral use needs the same map for every entry: no weights, or equal ones.
// (a b) c for a, b, c >= 0 finite, rounded as written where a b is normal,
// and otherwise from the mantissas, so that a b cannot overflow or underflow
// when the product is in range: gamma = 1e308 with lambda w = 1e-308 made
// the l0 threshold inf, and zeroed an entry whose minimiser is itself (math
// audit M3). Scaling by powers of 2 commutes with rounding, so the two paths
// agree wherever both are in range.
// Since 1.0.1 also soft thresholding, l1 over a box and group l2, whose
// gamma lambda overflowed to inf where every weight is 0 (NaN; fuzz).
inline bool normal_product(double a, double b) noexcept {
    const double ab = a * b;
    return ab <= std::numeric_limits<double>::max() &&
           (ab >= std::numeric_limits<double>::min() || (ab == 0.0 && (a == 0.0 || b == 0.0)));
}
inline double prod3(double a, double b, double c) noexcept {
    if (normal_product(a, b)) return a * b * c;
    int ea, eb, ec;
    const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb), mc = std::frexp(c, &ec);
    return std::ldexp(ma * mb * mc, ea + eb + ec);
}

// The largest of v (0 when empty), for step_scale().
inline double vmax(const std::vector<double> &v) noexcept {
    double m = 0.0;
    for (double x : v) m = std::max(m, x);
    return m;
}
inline bool uniform(const std::vector<double> &w) noexcept {
    for (double v : w)
        if (v != w[0]) return false;
    return true;
}

// atol of a set's indicator for the row x (see fp_op::value_row).
inline double value_atol(const double *x, int64_t n, double tol) noexcept {
    double m = 1.0;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    return tol * m;
}
// 0 when the violation is within atol, else INFINITY.
inline double indicator(double violation, double atol) noexcept {
    return violation <= atol ? 0.0 : std::numeric_limits<double>::infinity();
}

template <class D>
struct ElemCall {
    template <class T>
    FP_INLINE static T prox(const D &d, T x, double gamma, int64_t w, int tie, bool &tied) noexcept {
        if constexpr (D::kTies) return d.template prox1<T>(x, gamma, w, tie, tied);
        else return d.template prox1<T>(x, gamma, w);
    }
    static ElemDeriv deriv(const D &d, double x, double gamma, int64_t w, int tie) noexcept {
        if constexpr (D::kTies) return d.deriv1(x, gamma, w, tie);
        else return d.deriv1(x, gamma, w);
    }
};

// CRTP base for elementwise operators. D provides
//   template <class T> T prox1(T x, double gamma, int64_t widx) const noexcept;
//   ElemDeriv deriv1(double x, double gamma, int64_t widx) const noexcept;
// or, with static constexpr bool kTies = true, the same with an extra tie
// rule argument (and a bool &tied out-argument for prox1).
// A parameter of an indexed kernel (audit C4): per entry (an array) or shared
// by every entry (a value), fixed at compile time so that the loop has no
// runtime stride. Par<true>{w.data()}[i] is w[i], Par<false>{v}[i] is v.
template <bool PerEntry>
struct Par {
    const double *p;
    double operator[](int64_t i) const noexcept { return p[i]; }
};
template <>
struct Par<false> {
    double v;
    double operator[](int64_t) const noexcept { return v; }
};
template <bool B>
inline Par<B> par(const std::vector<double> &v) noexcept {
    if constexpr (B) return {v.data()};
    else return {v[0]};
}

template <class D>
struct ElemOp : fp_op {
    static constexpr bool kTies = false;
    // Optional by-value kernels for the prox loop (audit C4). D may define
    //   static constexpr bool kUniformKernel = true;
    //   template <class T> K uniform_kernel(double gamma) const;  // !axis_params()
    //   static constexpr bool kIndexedKernel = true;
    //   template <class T, class F> void with_indexed_kernel(double gamma, F &&f) const;
    // (the latter for axis_params(): it calls f(k) with a kernel whose shape,
    // which parameters are per entry, is fixed at compile time, see Par),
    // each a struct of plain values and const double * arrays with
    // T operator()(T x) const, or T operator()(T x, int64_t i) const for the
    // indexed one. It must compute exactly prox1's expression, so results are
    // bit-identical; holding no reference to the operator, it lets the
    // compiler keep its constants in registers and vectorise, in place too.
    // Operators with ties keep prox1.
    static constexpr bool kUniformKernel = false;
    static constexpr bool kIndexedKernel = false;
    // false where the indexed kernel's hoisted product (gamma lambda) leaves
    // the normal range: prox1 then runs, with prod3 (since 1.0.1).
    bool kernel_exact(double) const noexcept { return true; }
    // dz/dgamma is identically zero (projections): the vjp skips the gamma
    // slot (audit C5).
    static constexpr bool kGammaFree = false;
    // Operators with kinks (kKinks = true) also provide
    //   int kink1(double x, double gamma, int64_t widx, double tol, Kink &k);
    //   Piece piece_of(double x, double gamma, int64_t widx);
    //   ElemDeriv deriv_piece(double x, double gamma, int64_t widx, Piece p);
    // kink1 returns the code of the breakpoint nearest to x within tol (0 if
    // none) and fills k; deriv1 must equal deriv_piece(x, piece_of(x)).
    static constexpr bool kKinks = false;
    // The fp_struct_kind of the map (spec Section 7, "Structure of the
    // derivative"). An entry is FREE where its derivative in x is not 0, and
    // held otherwise: at 0, or, for D with kBounds, at the bound D's
    //   int bound_state(Piece p) const;
    // names for a held piece p.
    static constexpr int kStruct = FP_STRUCT_NONE;
    static constexpr bool kBounds = false;
    bool elementwise() const noexcept override { return true; }
    bool has_ties() const noexcept override { return D::kTies; }
    bool has_kinks() const noexcept override { return D::kKinks; }
    int struct_kind() const noexcept override { return D::kStruct; }

    // g(x) = sum_i of D's value1(x_i, i, atol), the term of entry i (atol for
    // the indicators of sets, see fp_op::value_row).
    bool has_value() const noexcept override { return true; }
    double value_row(const double *x, int64_t n, double tol, double *) const noexcept override {
        const D &d = static_cast<const D &>(*this);
        const double atol = value_atol(x, n, tol);
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += d.value1(x[i], i, atol);
        return s;
    }

    // The derivative used at entry x: the default rule, or with a side
    // override at a flagged entry the piece on that side of its breakpoint
    // (the side of b + side * delta), whose formula is evaluated at x.
    static ElemDeriv deriv_at(const D &d, double x, double gamma, int64_t w, int tie, int side,
                              double tol) noexcept {
        if constexpr (D::kKinks) {
            Piece p;
            if (side_piece(d, x, gamma, w, side, tol, p)) return d.deriv_piece(x, gamma, w, p);
        }
        return ElemCall<D>::deriv(d, x, gamma, w, tie);
    }
    // The piece on the side's side of the breakpoint within tol of x, if side
    // is not 0 and there is one (operators with kinks).
    static bool side_piece(const D &d, double x, double gamma, int64_t w, int side, double tol, Piece &p) noexcept {
        if constexpr (D::kKinks) {
            Kink k;
            if (side != 0 && d.kink1(x, gamma, w, tol, k)) {
                const double delta = 32.0 * 2.220446049250313e-16 * k.scale + 1e-300;
                p = d.piece_of(k.b + side * delta, gamma, w);
                return true;
            }
        }
        return false;
    }

    // The fp_entry_state of entry x under the derivative deriv_at takes: from
    // its piece (a held piece names its bound), or for the maps with ties,
    // whose default derivative follows the tie rule, from that derivative.
    static int state_of_piece(const D &d, double x, double gamma, int64_t w, Piece p) noexcept {
        if constexpr (D::kKinks) {
            if (d.deriv_piece(x, gamma, w, p).j != 0.0) return FP_ENTRY_FREE;
            if constexpr (D::kBounds) return d.bound_state(p);
        }
        return FP_ENTRY_ZERO;
    }
    static int state_at(const D &d, double x, double gamma, int64_t w, int tie, int side, double tol) noexcept {
        if constexpr (D::kKinks) {
            Piece p;
            if (side_piece(d, x, gamma, w, side, tol, p)) return state_of_piece(d, x, gamma, w, p);
            if constexpr (!D::kTies) return state_of_piece(d, x, gamma, w, d.piece_of(x, gamma, w));
        }
        return ElemCall<D>::deriv(d, x, gamma, w, tie).j != 0.0 ? FP_ENTRY_FREE : FP_ENTRY_ZERO;
    }
    template <class T>
    void run_states(const T *xsave, int64_t len, double gamma, int64_t widx, int64_t wstep, int tie,
                    const int8_t *side, double tol, int8_t *state) const noexcept {
        const D &d = static_cast<const D &>(*this);
        for (int64_t i = 0; i < len; ++i)
            state[i] = static_cast<int8_t>(state_at(d, static_cast<double>(xsave[i]), gamma, widx + i * wstep, tie,
                                                    side ? side[i] : 0, tol));
    }

    template <class T>
    int64_t run_kinks(const T *xsave, int64_t len, double gamma, int64_t widx, int64_t wstep,
                      double tol, int8_t *codes) const noexcept {
        int64_t count = 0;
        if constexpr (D::kKinks) {
            const D &d = static_cast<const D &>(*this);
            for (int64_t i = 0; i < len; ++i) {
                Kink k;
                const int c = d.kink1(static_cast<double>(xsave[i]), gamma, widx + i * wstep, tol, k);
                if (codes) codes[i] = static_cast<int8_t>(c);
                count += c != 0;
            }
        }
        return count;
    }

    // The prox loops of a by-value kernel: k is a local copy, so the in-place
    // loop reads and writes one pointer at one index, and the compiler needs
    // neither alias nor overlap checks.
    template <class T, class K>
    static void kernel_loops(const K k, const T *x, int64_t xs, T *z, int64_t zs, int64_t len, T *save) noexcept {
        if (save) {
            for (int64_t i = 0; i < len; ++i) {
                const T xi = x[i * xs];
                save[i] = xi;
                z[i * zs] = k(xi);
            }
        } else if (xs == 1 && zs == 1) {
            if (static_cast<const void *>(x) == static_cast<const void *>(z)) {
                T *p = z;
                for (int64_t i = 0; i < len; ++i) p[i] = k(p[i]);
            } else {
                for (int64_t i = 0; i < len; ++i) z[i] = k(x[i]);
            }
        } else {
            for (int64_t i = 0; i < len; ++i) z[i * zs] = k(x[i * xs]);
        }
    }
    template <class T, class K>
    static void indexed_loops(const K k, const T *x, int64_t xs, T *z, int64_t zs, int64_t len, int64_t widx,
                              int64_t wstep, T *save) noexcept {
        if (save) {
            for (int64_t i = 0; i < len; ++i) {
                const T xi = x[i * xs];
                save[i] = xi;
                z[i * zs] = k(xi, widx + i * wstep);
            }
        } else if (xs == 1 && zs == 1 && wstep == 1) {
            if (static_cast<const void *>(x) == static_cast<const void *>(z)) {
                T *p = z;
                for (int64_t i = 0; i < len; ++i) p[i] = k(p[i], widx + i);
            } else {
                for (int64_t i = 0; i < len; ++i) z[i] = k(x[i], widx + i);
            }
        } else {
            for (int64_t i = 0; i < len; ++i) z[i * zs] = k(x[i * xs], widx + i * wstep);
        }
    }

    template <class T>
    int64_t run_prox(const T *x, int64_t xs, T *z, int64_t zs, int64_t len, double gamma,
                     int64_t widx, int64_t wstep, T *save, bool conj, double sigma,
                     int tie) const noexcept {
        const D &d = static_cast<const D &>(*this);
        using C = ElemCall<D>;
        if constexpr (D::kUniformKernel && !D::kTies) {
            if (!conj && !this->axis_params()) {
                kernel_loops<T>(d.template uniform_kernel<T>(gamma), x, xs, z, zs, len, save);
                return 0;
            }
        }
        if constexpr (D::kIndexedKernel && !D::kTies) {
            if (!conj && this->axis_params() && d.kernel_exact(gamma)) {
                d.template with_indexed_kernel<T>(gamma, [&](const auto k) {
                    indexed_loops<T>(k, x, xs, z, zs, len, widx, wstep, save);
                });
                return 0;
            }
        }
        int64_t ties = 0;
        bool tied = false;
        if (conj) {
            // y / sigma can leave T's range only for sigma < 1 (never for the
            // conj_homogeneous operators, at sigma = 1): see kStagingCheck.
            if (sigma < kStagingCheck<T> && staging_overflows<T>(x, xs, len, sigma))
                return -1;  // the driver's FP_EPARAM (ties are never negative)
            for (int64_t i = 0; i < len; ++i) {
                const double y = static_cast<double>(x[i * xs]);
                const T t = static_cast<T>(y / sigma);
                if (save) save[i] = t;
                const double p = static_cast<double>(C::template prox<T>(d, t, gamma, widx + i * wstep, tie, tied));
                ties += tied;
                z[i * zs] = static_cast<T>(y - sigma * p);
            }
        } else if (save) {
            for (int64_t i = 0; i < len; ++i) {
                const T xi = x[i * xs];
                save[i] = xi;
                z[i * zs] = C::template prox<T>(d, xi, gamma, widx + i * wstep, tie, tied);
                ties += tied;
            }
        } else if (xs == 1 && zs == 1) {
            for (int64_t i = 0; i < len; ++i) {
                z[i] = C::template prox<T>(d, x[i], gamma, widx + i * wstep, tie, tied);
                ties += tied;
            }
        } else {
            for (int64_t i = 0; i < len; ++i) {
                z[i * zs] = C::template prox<T>(d, x[i * xs], gamma, widx + i * wstep, tie, tied);
                ties += tied;
            }
        }
        return ties;
    }

    // Overrides are rare: the loops below are instantiated with and without
    // them, so the default loop is unchanged.
    template <class T>
    void run_jvp(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,
                 double gamma, double dgamma, const double *dparams, int64_t widx, int64_t wstep,
                 bool conj, double sigma, int tie, const int8_t *side, double tol) const noexcept {
        if (side && D::kKinks)
            run_jvp_<T, true>(xsave, dx, dxs, dz, dzs, len, gamma, dgamma, dparams, widx, wstep, conj,
                              sigma, tie, side, tol);
        else
            run_jvp_<T, false>(xsave, dx, dxs, dz, dzs, len, gamma, dgamma, dparams, widx, wstep, conj,
                               sigma, tie, nullptr, tol);
    }
    template <class T>
    void run_vjp(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,
                 double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,
                 double sigma, int tie, const int8_t *side, double tol) const noexcept {
        if (side && D::kKinks)
            dispatch_vjp<T, true>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma, tie,
                                  side, tol);
        else
            dispatch_vjp<T, false>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma, tie,
                                   nullptr, tol);
    }
    // acc null: the caller asked for no sum, only gx. Otherwise the gamma and
    // conjugate slots, and up to two parameters shared by every entry, are
    // summed in registers over the run (audit C5); see run_vjp_.
    template <class T, bool Side>
    void dispatch_vjp(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,
                      double gamma, double *acc, int64_t widx, int64_t wstep, bool conj, double sigma,
                      int tie, const int8_t *side, double tol) const noexcept {
#define FP_VJP_(ACC, NP) \
    run_vjp_<T, Side, ACC, NP>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma, tie, side, tol)
        if (!acc) return FP_VJP_(false, 0);
        const int64_t P = this->nparams();
        if (!this->axis_params()) {
            if (P == 0) return FP_VJP_(true, 0);
            if (P == 1) return FP_VJP_(true, 1);
            if (P == 2) return FP_VJP_(true, 2);
        }
        return FP_VJP_(true, -1);
#undef FP_VJP_
    }

    template <class T, bool Side>
    void run_jvp_(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,
                  double gamma, double dgamma, const double *dparams, int64_t widx, int64_t wstep,
                  bool conj, double sigma, int tie, const int8_t *side, double tol) const noexcept {
        const D &d = static_cast<const D &>(*this);
        using C = ElemCall<D>;
        bool tied = false;
        if constexpr (!Side) {
            // Contiguous, no parameter tangent, not conjugate: the same
            // expression in a loop the compiler can vectorise (audit C5).
            if (!conj && !dparams && dx && dxs == 1 && dzs == 1) {
                for (int64_t i = 0; i < len; ++i) {
                    const ElemDeriv e = C::deriv(d, static_cast<double>(xsave[i]), gamma, widx + i * wstep, tie);
                    dz[i] = static_cast<T>(e.j * static_cast<double>(dx[i]) + e.dg * dgamma + 0.0);
                }
                return;
            }
        }
        for (int64_t i = 0; i < len; ++i) {
            const int64_t w = widx + i * wstep;
            const double xi = static_cast<double>(xsave[i]);
            ElemDeriv e;
            if constexpr (Side) e = deriv_at(d, xi, gamma, w, tie, side[i], tol);
            else e = C::deriv(d, xi, gamma, w, tie);
            const double dxi = dx ? static_cast<double>(dx[i * dxs]) : 0.0;
            double ptheta = 0.0;
            if (dparams)
                for (int k = 0; k < e.np; ++k) ptheta += e.pv[k] * dparams[e.pi[k]];
            if (!conj) {
                dz[i * dzs] = static_cast<T>(e.j * dxi + e.dg * dgamma + ptheta);
            } else {
                // out = y - sigma p(gamma = 1/sigma, t = y/sigma), dgamma reads as dsigma.
                const double p = static_cast<double>(C::template prox<double>(d, xi, gamma, w, tie, tied));
                const double dt = dxi / sigma - xi * dgamma / sigma;
                const double dgam = -dgamma / (sigma * sigma);
                const double dp = e.j * dt + e.dg * dgam + ptheta;
                dz[i * dzs] = static_cast<T>(dxi - dgamma * p - sigma * dp);
            }
        }
    }

    // Acc: the sums are wanted. NP: the parameter slots held in registers (0,
    // 1 or 2 parameters shared by every entry), or -1 to add in memory (the
    // parameters along the axis, whose slots vary with the entry, so no chain
    // runs through memory). Each slot receives the same additions in the
    // same order as when it was added to in memory: a register holds its
    // running value over the run and writes it back at the end, and the next
    // run reloads it, so the sequence is the block's C order whatever the
    // runs or the thread count. An inactive slot adds +0.0, which changes no
    // sum (a slot starts at +0.0 and is never -0.0), selected after the
    // multiply, so a NaN g_i on it adds nothing, as before.
    template <class T, bool Side, bool Acc, int NP>
    void run_vjp_(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,
                  double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,
                  double, int tie, const int8_t *side, double tol) const noexcept {
        const D &d = static_cast<const D &>(*this);
        using C = ElemCall<D>;
        bool tied = false;
        const int64_t P = this->nparams();
        if constexpr (!Acc && !Side) {
            // gx only, contiguous, not conjugate: a loop the compiler can
            // vectorise (audit C5).
            if (!conj && gs == 1 && gxs == 1) {
                for (int64_t i = 0; i < len; ++i) {
                    const ElemDeriv e = C::deriv(d, static_cast<double>(xsave[i]), gamma, widx + i * wstep, tie);
                    gx[i] = static_cast<T>(e.j * static_cast<double>(g[i]));
                }
                return;
            }
        }
        double aP = 0.0, aX = 0.0, l0 = 0.0, l1 = 0.0;
        if constexpr (Acc) {
            aP = acc[P];
            aX = acc[P + 1];
            if constexpr (NP >= 1) l0 = acc[0];
            if constexpr (NP == 2) l1 = acc[1];
        }
        for (int64_t i = 0; i < len; ++i) {
            const int64_t w = widx + i * wstep;
            const double xi = static_cast<double>(xsave[i]);
            ElemDeriv e;
            if constexpr (Side) e = deriv_at(d, xi, gamma, w, tie, side[i], tol);
            else e = C::deriv(d, xi, gamma, w, tie);
            const double gi = static_cast<double>(g[i * gs]);
            if constexpr (Acc) {
                if constexpr (NP == -1) {
                    for (int k = 0; k < e.np; ++k) acc[e.pi[k]] += e.pv[k] * gi;
                } else if constexpr (NP == 1) {
                    for (int k = 0; k < 3; ++k) l0 += k < e.np ? e.pv[k] * gi : 0.0;
                } else if constexpr (NP == 2) {
                    for (int k = 0; k < 3; ++k) {
                        const double c = k < e.np ? e.pv[k] * gi : 0.0;
                        l0 += k < e.np && e.pi[k] == 0 ? c : 0.0;
                        l1 += k < e.np && e.pi[k] == 1 ? c : 0.0;
                    }
                }
                if constexpr (!D::kGammaFree) aP += e.dg * gi;
            }
            if (!conj) {
                gx[i * gxs] = static_cast<T>(e.j * gi);
            } else {
                if constexpr (Acc) {
                    const double p = static_cast<double>(C::template prox<double>(d, xi, gamma, w, tie, tied));
                    aX += -gi * p + e.j * gi * xi;
                }
                gx[i * gxs] = static_cast<T>(gi - e.j * gi);
            }
        }
        if constexpr (Acc) {
            acc[P] = aP;
            acc[P + 1] = aX;
            if constexpr (NP >= 1) acc[0] = l0;
            if constexpr (NP == 2) acc[1] = l1;
        }
    }

#define FP_ELEM_FORWARD(T)                                                                       \
    int64_t prox_run(const T *x, int64_t xs, T *z, int64_t zs, int64_t len, double gamma,        \
                     int64_t widx, int64_t wstep, T *save, bool conj, double sigma, int tie)     \
        const noexcept override {                                                                \
        return run_prox<T>(x, xs, z, zs, len, gamma, widx, wstep, save, conj, sigma, tie);       \
    }                                                                                            \
    void jvp_run(const T *xsave, const T *dx, int64_t dxs, T *dz, int64_t dzs, int64_t len,      \
                 double gamma, double dgamma, const double *dparams, int64_t widx,               \
                 int64_t wstep, bool conj, double sigma, int tie, const int8_t *side,            \
                 double tol) const noexcept override {                                           \
        run_jvp<T>(xsave, dx, dxs, dz, dzs, len, gamma, dgamma, dparams, widx, wstep, conj,      \
                   sigma, tie, side, tol);                                                       \
    }                                                                                            \
    void vjp_run(const T *xsave, const T *g, int64_t gs, T *gx, int64_t gxs, int64_t len,        \
                 double gamma, double *acc, int64_t widx, int64_t wstep, bool conj,              \
                 double sigma, int tie, const int8_t *side, double tol) const noexcept override {\
        run_vjp<T>(xsave, g, gs, gx, gxs, len, gamma, acc, widx, wstep, conj, sigma, tie, side,  \
                   tol);                                                                         \
    }                                                                                            \
    int64_t kink_run(const T *xsave, int64_t len, double gamma, int64_t widx, int64_t wstep,     \
                     double tol, int8_t *codes) const noexcept override {                        \
        return run_kinks<T>(xsave, len, gamma, widx, wstep, tol, codes);                         \
    }                                                                                            \
    void state_run(const T *xsave, int64_t len, double gamma, int64_t widx, int64_t wstep,       \
                   int tie, const int8_t *side, double tol, int8_t *state) const noexcept override { \
        run_states<T>(xsave, len, gamma, widx, wstep, tie, side, tol, state);                    \
    }
    FP_ELEM_FORWARD(float)
    FP_ELEM_FORWARD(double)
#undef FP_ELEM_FORWARD
};

// CRTP base for row operators. D provides templates prox, jvp and vjp with
// the signatures of prox_row, jvp_row and vjp_row, and, with a kStruct other
// than FP_STRUCT_NONE (since 1.0.2), structure with that of struct_row.
template <class D>
struct RowOp : fp_op {
    static constexpr int kStruct = FP_STRUCT_NONE;
    bool elementwise() const noexcept override { return false; }
    int struct_kind() const noexcept override { return D::kStruct; }

#define FP_ROW_FORWARD(T)                                                                        \
    int prox_row(const T *x, T *z, int64_t n, double gamma, void *scratch, uint8_t *code)        \
        const noexcept override {                                                                \
        return static_cast<const D *>(this)->template prox<T>(x, z, n, gamma, scratch, code);    \
    }                                                                                            \
    void jvp_row(const T *x, const T *z, const uint8_t *code, const T *dx, T *dz, int64_t n,     \
                 double gamma, double dgamma, const double *dparams, void *scratch)              \
        const noexcept override {                                                                \
        static_cast<const D *>(this)->template jvp<T>(x, z, code, dx, dz, n, gamma, dgamma,      \
                                                      dparams, scratch);                         \
    }                                                                                            \
    void vjp_row(const T *x, const T *z, const uint8_t *code, const T *g, T *gx, int64_t n,      \
                 double gamma, double *acc, void *scratch) const noexcept override {             \
        static_cast<const D *>(this)->template vjp<T>(x, z, code, g, gx, n, gamma, acc,          \
                                                      scratch);                                  \
    }                                                                                            \
    int64_t struct_row(const T *x, const T *z, const uint8_t *code, int64_t n, double gamma,     \
                       RowStruct &s, void *scratch) const noexcept override {                    \
        if constexpr (D::kStruct != FP_STRUCT_NONE)                                              \
            return static_cast<const D *>(this)->template structure<T>(x, z, code, n, gamma, s,  \
                                                                       scratch);                 \
        else                                                                                     \
            return 0;                                                                            \
    }
    FP_ROW_FORWARD(float)
    FP_ROW_FORWARD(double)
#undef FP_ROW_FORWARD
};

// CRTP base for image operators. D provides templates prox_img, jvp_img and
// vjp_img with the signatures of prox_image, jvp_image and vjp_image.
template <class D>
struct ImageOp : fp_op {
    bool elementwise() const noexcept override { return false; }
    fp_status check_n(int64_t) const noexcept override { return FP_OK; }

#define FP_IMAGE_FORWARD(T)                                                                      \
    int prox_image(const T *x, T *z, ImageCall &c) const noexcept override {                     \
        return static_cast<const D *>(this)->template prox_img<T>(x, z, c);                      \
    }                                                                                            \
    void jvp_image(const char *save, const T *dx, T *dz, const int64_t *shape, double gamma,     \
                   double dgamma, const double *dparams, char *work, DerivCtl &dc)               \
        const noexcept override {                                                                \
        static_cast<const D *>(this)->template jvp_img<T>(save, dx, dz, shape, gamma, dgamma,    \
                                                          dparams, work, dc);                    \
    }                                                                                            \
    void vjp_image(const char *save, const T *g, T *gx, const int64_t *shape, double gamma,      \
                   double *acc, char *work, DerivCtl &dc) const noexcept override {              \
        static_cast<const D *>(this)->template vjp_img<T>(save, g, gx, shape, gamma, acc, work,  \
                                                          dc);                                   \
    }
    FP_IMAGE_FORWARD(float)
    FP_IMAGE_FORWARD(double)
#undef FP_IMAGE_FORWARD
};

// Bit layout of the exponent field, to test finiteness without branches.
template <class T>
struct FloatBits;
template <>
struct FloatBits<float> {
    using U = uint32_t;
    static constexpr U exp = 0x7f800000u;
};
template <>
struct FloatBits<double> {
    using U = uint64_t;
    static constexpr U exp = 0x7ff0000000000000ull;
};

// True when every entry is finite. A branch-free pass over the bit patterns,
// which the compiler vectorises; the common case costs a fraction of a
// nanosecond per entry.
template <class T>
inline bool all_finite(const T *x, int64_t n) noexcept {
    using U = typename FloatBits<T>::U;
    constexpr U e = FloatBits<T>::exp;
    U bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        U b;
        std::memcpy(&b, x + i, sizeof b);
        bad |= static_cast<U>((b & e) == e);
    }
    return bad == 0;
}

// Minimum and maximum of two values known to be NaN-free (or whose NaN
// inputs are tested separately), as one instruction and never a branch:
// fminnm/fmaxnm on arm64 and minsd/maxsd (minss/maxss) on x86-64, where
// SSE2 is part of the base architecture. Compilers do not reliably turn a
// compare-select into these: GCC emitted compare-and-branch for both the
// select and the ternary forms, which mispredict in Condat's bound updates.
// minsd(v, a) returns v < a ? v : a, so the result is that of the portable
// form, which other targets use.

template <class T>
inline T min_nonan(T a, T v) noexcept {
#if defined(FP_MINMAX_ARM64)
    return std::fmin(a, v);
#elif defined(FP_MINMAX_SSE2)
    if constexpr (std::is_same_v<T, double>)
        return _mm_cvtsd_f64(_mm_min_sd(_mm_set_sd(v), _mm_set_sd(a)));
    else
        return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(v), _mm_set_ss(a)));
#else
    return v < a ? v : a;
#endif
}
template <class T>
inline T max_nonan(T a, T v) noexcept {
#if defined(FP_MINMAX_ARM64)
    return std::fmax(a, v);
#elif defined(FP_MINMAX_SSE2)
    if constexpr (std::is_same_v<T, double>)
        return _mm_cvtsd_f64(_mm_max_sd(_mm_set_sd(v), _mm_set_sd(a)));
    else
        return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(v), _mm_set_ss(a)));
#else
    return v > a ? v : a;
#endif
}

// True when every entry is finite; then lo and hi are the smallest and
// largest entries. One pass in independent lanes: a single running minimum
// is a chain of dependent operations (2 to 4 cycles per entry), which the
// compiler may not reorder without fast-math. On finite input the order of
// the comparisons changes at most the sign of a zero extreme.
template <class T>
inline bool finite_range(const T *x, int64_t n, double &lo, double &hi) noexcept {
    using U = typename FloatBits<T>::U;
    constexpr U e = FloatBits<T>::exp;
    constexpr int64_t kLanes = 8;  // measured: 4 to 16 are close; 1 is twice as slow
    U bad[kLanes];
    T mn[kLanes], mx[kLanes];
    for (int64_t j = 0; j < kLanes; ++j) {
        bad[j] = 0;
        mn[j] = mx[j] = x[0];
    }
    auto step = [&](int64_t j, T v) {
        U b;
        std::memcpy(&b, &v, sizeof b);
        bad[j] |= static_cast<U>((b & e) == e);
        mn[j] = min_nonan(mn[j], v);
        mx[j] = max_nonan(mx[j], v);
    };
    int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes)
        for (int64_t j = 0; j < kLanes; ++j) step(j, x[i + j]);
    for (; i < n; ++i) step(0, x[i]);
    for (int64_t j = 1; j < kLanes; ++j) step(0, mn[j]), step(0, mx[j]), bad[0] |= bad[j];
    lo = static_cast<double>(mn[0]);
    hi = static_cast<double>(mx[0]);
    return bad[0] == 0;
}

// Scans a row: returns kRowNonFinite on any infinity and sets has_nan.
template <class T>
inline int scan_row(const T *x, int64_t n, bool &has_nan) noexcept {
    has_nan = false;
    if (all_finite(x, n)) return kRowOk;
    for (int64_t i = 0; i < n; ++i) {
        const T v = x[i];
        if (v != v) {
            has_nan = true;
        } else if (std::isinf(v)) {
            return kRowNonFinite;
        }
    }
    return kRowOk;
}

template <class T>
inline void fill_nan(T *z, int64_t n) noexcept {
    const T q = std::numeric_limits<T>::quiet_NaN();
    for (int64_t i = 0; i < n; ++i) z[i] = q;
}

// True when S, a sum of squares (>= 0 or NaN), lies in [2^-968, DBL_MAX],
// where its square root is the norm to rounding: one unsigned compare of the
// bit patterns (NaN, +inf and anything below 2^-968 fall outside).
inline bool sumsq_in_range(double S) noexcept {
    constexpr uint64_t lo = 0x0370000000000000ull;  // 2^-968
    constexpr uint64_t hi = 0x7fefffffffffffffull;  // DBL_MAX
    uint64_t b;
    std::memcpy(&b, &S, sizeof b);
    return b - lo <= hi - lo;
}

// Group norms sqrt(sum_{i in g} x_i^2) from the plain sums of squares S_g,
// which nrm holds on entry; x(i), i < n, the entries and group(i) < G their
// groups. A sum safely in range, [2^-968, DBL_MAX], is used as it is, so
// those norms are those of the plain formula, bit for bit (below 2^-968 the
// squares lost to underflow are no longer below rounding). The other groups
// are summed again on entries scaled by a power of two that brings their
// largest to [1/2, 1), so that no square overflows or underflows: the idea
// of the reference BLAS dnrm2 (J. L. Blue, "A portable Fortran program to
// find the Euclidean norm of a vector", ACM TOMS 4(1), 1978, 15-23; E.
// Anderson, "Algorithm 978: Safe scaling in the Level 1 BLAS", ACM TOMS
// 44(1), 2017), with the scale chosen after the fact. On return the norm of
// group g is nrm[g] / M[g], with M[g] = 1 unless the norm exceeds the
// largest double or is below the smallest normal one (where 1 / norm
// overflows); then M[g] = t is the power of two of the rescaling and
// nrm[g] = ||t x_g||, in [2^-53, sqrt(len)]. A NaN entry gives a NaN norm.
// Every pass runs in index order.
template <class X, class Grp>
inline void safe_group_norms(int64_t n, int64_t G, X x, Grp group, double *nrm, double *M) noexcept {
    bool slow = false;
    for (int64_t g = 0; g < G; ++g) {
        const bool ok = sumsq_in_range(nrm[g]);
        M[g] = ok ? -1.0 : 0.0;  // -1: done; else the running max of |x| (NaN sticks)
        slow = slow || !ok;
    }
    if (!slow) {
        for (int64_t g = 0; g < G; ++g) {
            nrm[g] = std::sqrt(nrm[g]);
            M[g] = 1.0;
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        double &m = M[group(i)];
        const double v = std::fabs(static_cast<double>(x(i)));
        if (!(m < 0.0) && m == m && !(v <= m)) m = v;
    }
    for (int64_t g = 0; g < G; ++g) {
        if (M[g] < 0.0) {
            nrm[g] = std::sqrt(nrm[g]);
            M[g] = 0.0;  // done
        } else if (!(M[g] > 0.0)) {
            nrm[g] = M[g];  // an all-zero group, or NaN
            M[g] = 0.0;
        } else {
            // The scale t = 2^-e brings the largest to [1/2, 1). For a subnormal
            // largest 2^-e would overflow; 2^1022 brings it to [2^-53, 1), whose
            // squares are still far from underflow.
            int e;
            std::frexp(M[g], &e);
            M[g] = std::ldexp(1.0, std::min(-e, 1022));
            nrm[g] = 0.0;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        const int64_t g = group(i);
        if (M[g] > 0.0) {
            const double v = static_cast<double>(x(i)) * M[g];
            nrm[g] += v * v;
        }
    }
    for (int64_t g = 0; g < G; ++g) {
        if (!(M[g] > 0.0)) {
            M[g] = 1.0;
            continue;
        }
        const double m = std::sqrt(nrm[g]), s = m / M[g];
        if (std::numeric_limits<double>::min() <= s && s <= std::numeric_limits<double>::max()) {
            nrm[g] = s;
            M[g] = 1.0;
        } else {
            nrm[g] = m;  // keep the scale
        }
    }
}

template <class X>
FP_COLD double safe_norm_slow(double S, int64_t n, X x, double &t) noexcept {
    safe_group_norms(n, 1, x, [](int64_t) { return int64_t{0}; }, &S, &t);
    return S;
}

// The power of two 2^-e that brings a > 0 to [1/2, 1) (2^1022 for a
// subnormal a), as in safe_group_norms; 1 for 0, inf or NaN.
inline double pow2_scale(double a) noexcept {
    if (!(a > 0.0) || !(a <= std::numeric_limits<double>::max())) return 1.0;
    int e;
    std::frexp(a, &e);
    return std::ldexp(1.0, std::min(-e, 1022));
}

// Group norms as safe_group_norms (nrm[g] = m_g, M[g] = t_g) and the dots
// ud[g] = <u_g, v_g>, u_g = (t_g x_g) / m_g, of x(i) and v(i), i < n, in
// group(i) < G; vm, xm and q are scratch of G each. <x_g, v_g> / m_g is used
// where t_g = 1 and m_g max|v_i| is within 2^+-900, where no product x_i v_i
// underflows beyond rounding or overflows; elsewhere the dot is redone on
// x_g and v_g scaled by powers of two (found by fuzzing in 1.0.2: with x
// and v near 1e-300, or 1e154, <x, v> left the range while ||x|| did not).
template <class X, class V, class Gr>
void group_unit_dots(int64_t n, int64_t G, X x, V v, Gr group, double *nrm, double *ud, double *M, double *vm,
                     double *xm, double *q) noexcept {
    for (int64_t g = 0; g < G; ++g) nrm[g] = ud[g] = vm[g] = xm[g] = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t g = group(i);
        const double xi = static_cast<double>(x(i)), vi = static_cast<double>(v(i));
        nrm[g] += xi * xi;
        ud[g] += xi * vi;
        vm[g] = std::fmax(vm[g], std::fabs(vi));
        xm[g] = std::fmax(xm[g], std::fabs(xi));
    }
    safe_group_norms(n, G, x, group, nrm, M);
    bool slow = false;
    for (int64_t g = 0; g < G; ++g) {
        const double P = nrm[g] * vm[g];
        if (M[g] == 1.0 && (vm[g] == 0.0 || (0x1p-900 <= P && P <= 0x1p900))) {
            ud[g] /= nrm[g];
            vm[g] = 0.0;  // done
        } else {
            xm[g] = pow2_scale(xm[g]);
            vm[g] = pow2_scale(vm[g]);  // > 0: redo below
            ud[g] = q[g] = 0.0;
            slow = true;
        }
    }
    if (!slow) return;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t g = group(i);
        if (vm[g] == 0.0) continue;
        const double xi = static_cast<double>(x(i)) * xm[g];
        q[g] += xi * xi;
        ud[g] += xi * (static_cast<double>(v(i)) * vm[g]);
    }
    for (int64_t g = 0; g < G; ++g)
        if (vm[g] != 0.0) ud[g] = ud[g] / std::sqrt(q[g]) / vm[g];
}

// <u, v>, u = x / ||x||, on x(k) and v(k), k < n, scaled by powers of two
// (the slow path of unit_dot).
template <class X, class V>
FP_COLD double unit_dot_slow(int64_t n, X x, V v) noexcept {
    double xm = 0.0, vm = 0.0;
    for (int64_t k = 0; k < n; ++k) {
        xm = std::fmax(xm, std::fabs(static_cast<double>(x(k))));
        vm = std::fmax(vm, std::fabs(static_cast<double>(v(k))));
    }
    const double sx = pow2_scale(xm), sv = pow2_scale(vm);
    double q = 0.0, d = 0.0;
    for (int64_t k = 0; k < n; ++k) {
        const double xi = static_cast<double>(x(k)) * sx;
        q += xi * xi;
        d += xi * (static_cast<double>(v(k)) * sv);
    }
    return d / std::sqrt(q) / sv;
}

// <u, v>, u = (t x) / m, for the norm ||x|| = m / t of safe_norm: the plain
// <t x, v> / m unless m max|v_k| leaves 2^+-900, as for group_unit_dots.
// That holds for t != 1 too: t caps at 2^1022, so a deep subnormal x keeps
// m far below 1 (fuzz 2026-10-28: the cone at x = (0, 4e-315), dx near
// 4e-312, where <t x, dx> was a subnormal of 17 bits).
template <class X, class V>
double unit_dot(int64_t n, X x, V v, double m, double t) noexcept {
    double d = 0.0, vm = 0.0;
    for (int64_t k = 0; k < n; ++k) {
        d += static_cast<double>(x(k)) * t * static_cast<double>(v(k));
        vm = std::fmax(vm, std::fabs(static_cast<double>(v(k))));
    }
    const double P = m * vm;
    if (vm == 0.0 || (0x1p-900 <= P && P <= 0x1p900)) return d / m;
    return unit_dot_slow(n, x, v);
}

// The norm of x(k), k < n, as m / t from the plain sum of squares S, as
// safe_group_norms for one group: returns m and sets t.
template <class X>
inline double safe_norm(double S, int64_t n, X x, double &t) noexcept {
    if (sumsq_in_range(S)) {
        t = 1.0;
        return std::sqrt(S);
    }
    return safe_norm_slow(S, n, x, t);
}

// f(m, t) with the norm of x(k), k < n, equal to m / t (safe_norm). f is
// called with the constant t = 1 when S is in range, so that once inlined
// every product by t folds away and that path is the code of the plain norm;
// with a runtime t the short rows of the l2 ball and the cone took 5-10%
// longer. The other call is out of line.
template <class X, class F>
FP_COLD auto with_norm_slow(double S, int64_t n, X x, F f) noexcept {
    double t;
    const double m = safe_norm_slow(S, n, x, t);
    return f(m, t);
}
template <class X, class F>
inline auto with_norm(double S, int64_t n, X x, F f) noexcept {
    if (sumsq_in_range(S)) return f(std::sqrt(S), 1.0);
    return with_norm_slow(S, n, x, f);  // out of line, so the hot path needs no frame
}

}  // namespace fp
