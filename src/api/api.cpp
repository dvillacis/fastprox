// The extern "C" boundary: argument checks, dtype dispatch, and the try/catch
// wrappers that turn every C++ exception into an fp_status.
//
// SPDX-License-Identifier: MIT
#include "fastprox.h"

#include "core/driver.hpp"
#include "core/layout.hpp"
#include "core/op.hpp"
#include "core/spectral.hpp"
#include "core/workspace.hpp"
#include "device/plugin.hpp"
#include "ops/ops.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fp {
namespace {

template <class F>
fp_status guard(F &&f) noexcept {
    try {
        return f();
    } catch (const std::bad_alloc &) {
        return FP_ENOMEM;
    } catch (const std::length_error &) {  // a size beyond max_size() (audit A7)
        return FP_ENOMEM;
    } catch (...) {
        return FP_EINTERNAL;
    }
}

bool finite(double v) noexcept { return std::isfinite(v); }

// The int in an enum argument or field. A C caller may pass any int there (C
// enums are ints), while in C++ loading an enum object outside its
// enumeration's range is undefined, so the boundary reads the bits and
// validates them before converting.
template <class E>
int enum_bits(const E &e) noexcept {
    static_assert(sizeof(E) == sizeof(int), "the C enums of fastprox.h are int-sized");
    int v;
    std::memcpy(&v, &e, sizeof v);
    return v;
}

// The dtypes an array may have: float32 and float64 (data), int8 (the kink,
// branch, state and sign arrays), int32 and int64 (the labels of the
// structure queries, since 1.0.2).
enum class Dt { Float, Int8, Label };

// Converts and validates a public descriptor. Scalars become shape [1].
fp_status to_arr(const fp_array *a, Arr &out, Dt kind = Dt::Float) noexcept {
    if (!a) return FP_EINVAL;
    if (enum_bits(a->device) != FP_DEVICE_CPU) return FP_EINVAL;
    const int dt = enum_bits(a->dtype);
    const bool ok = kind == Dt::Int8    ? dt == FP_I8
                    : kind == Dt::Label ? dt == FP_I32 || dt == FP_I64
                                        : dt == FP_F32 || dt == FP_F64;
    if (!ok) return FP_EDTYPE;
    if (a->ndim < 0 || a->ndim > FP_MAX_NDIM) return FP_ESHAPE;
    out.data = static_cast<char *>(a->data);
    out.dtype = static_cast<fp_dtype>(dt);
    if (a->ndim == 0) {
        out.ndim = 1;
        out.shape[0] = 1;
        out.strides[0] = 1;
    } else {
        if (!a->shape) return FP_EINVAL;
        out.ndim = a->ndim;
        // The entry count, zero extents skipped (so that every product
        // Arr::numel forms is bounded too), at most kMaxElems.
        int64_t count = 1;
        for (int d = 0; d < a->ndim; ++d) {
            if (a->shape[d] < 0) return FP_ESHAPE;
            out.shape[d] = a->shape[d];
            out.strides[d] = a->strides ? a->strides[d] : 0;
            if (a->shape[d] > 1 && !mul_bounded(count, a->shape[d], count)) return FP_ESHAPE;
        }
        if (!a->strides) {  // NULL strides: C-contiguous
            int64_t s = 1;
            for (int d = a->ndim - 1; d >= 0; --d) {
                out.strides[d] = s;
                s *= out.shape[d];
            }
        }
        // The element offset span, sum_d (shape_d - 1) |stride_d|, at most
        // kMaxElems: offsets times the item size then fit in int64.
        int64_t span = 0;
        for (int d = 0; d < a->ndim; ++d) {
            if (out.shape[d] < 2) continue;
            const int64_t st = out.strides[d];
            if (st < -kMaxElems || st > kMaxElems) return FP_ESHAPE;
            int64_t e;
            if (!mul_bounded(out.shape[d] - 1, st < 0 ? -st : st, e) || e > kMaxElems - span) return FP_ESHAPE;
            span += e;
        }
    }
    if (!out.data && out.numel() > 0) return FP_EINVAL;
    return FP_OK;
}

// Reads a one-dimensional parameter array of any dtype and stride.
fp_status read_vec(const fp_array *a, Vec &v) {
    v.clear();
    if (!a) return FP_OK;
    Arr x;
    if (fp_status s = to_arr(a, x)) return s;
    if (a->ndim > 1) return FP_ESHAPE;
    const int64_t m = x.shape[0];
    if (m == 0) return FP_ESHAPE;
    v.resize(static_cast<size_t>(m));
    const int64_t st = x.strides[0];
    if (x.dtype == FP_F64 && st == 1) {
        std::memcpy(v.data(), x.data, static_cast<size_t>(m) * sizeof(double));
    } else if (x.dtype == FP_F64) {
        const double *d = reinterpret_cast<const double *>(x.data);
        for (int64_t i = 0; i < m; ++i) v[static_cast<size_t>(i)] = d[i * st];
    } else {
        const float *f = reinterpret_cast<const float *>(x.data);
        for (int64_t i = 0; i < m; ++i) v[static_cast<size_t>(i)] = f[i * st];
    }
    return FP_OK;
}

// Reads an array of exactly nd >= 1 axes, each of extent >= 1, in C order,
// with its shape.
fp_status read_nd(const fp_array *a, int nd, Vec &v, int64_t *shape) {
    Arr x;
    if (fp_status s = to_arr(a, x)) return s;
    if (a->ndim != nd) return FP_ESHAPE;
    for (int d = 0; d < nd; ++d) {
        if (x.shape[d] < 1) return FP_ESHAPE;
        shape[d] = x.shape[d];
    }
    const int64_t m = x.numel();
    v.resize(static_cast<size_t>(m));
    int64_t idx[FP_MAX_NDIM] = {0};
    for (int64_t i = 0; i < m; ++i) {
        int64_t o = 0;
        for (int d = 0; d < nd; ++d) o += idx[d] * x.strides[d];
        v[static_cast<size_t>(i)] = x.dtype == FP_F32 ? reinterpret_cast<const float *>(x.data)[o]
                                                      : reinterpret_cast<const double *>(x.data)[o];
        for (int d = nd - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
    return FP_OK;
}

// Reads exactly m values (a vector of length m, or a scalar when m == 1).
fp_status read_exact(const fp_array *a, int64_t m, double *dst) noexcept {
    Arr x;
    if (fp_status s = to_arr(a, x)) return s;
    if (a->ndim > 1 || x.shape[0] != m) return FP_ESHAPE;
    for (int64_t i = 0; i < m; ++i) {
        const int64_t o = i * x.strides[0];
        dst[i] = x.dtype == FP_F32 ? reinterpret_cast<const float *>(x.data)[o]
                                   : reinterpret_cast<const double *>(x.data)[o];
    }
    return FP_OK;
}

fp_status write_exact(fp_array *a, int64_t m, const double *src) noexcept {
    Arr x;
    if (fp_status s = to_arr(a, x)) return s;
    if (a->ndim > 1 || x.shape[0] != m) return FP_ESHAPE;
    for (int64_t i = 0; i < m; ++i) {
        const int64_t o = i * x.strides[0];
        if (x.dtype == FP_F32) reinterpret_cast<float *>(x.data)[o] = static_cast<float>(src[i]);
        else reinterpret_cast<double *>(x.data)[o] = src[i];
    }
    return FP_OK;
}

fp_options defaults() noexcept {
    fp_options o;
    fp_options_init(&o);
    return o;
}

fp_status norm_axis(int axis, int ndim, int &out) noexcept {
    const int a = axis < 0 ? axis + ndim : axis;
    if (a < 0 || a >= ndim) return FP_EINVAL;
    out = a;
    return FP_OK;
}

// Allocates a workspace for arrays shaped like `x`.
fp_status build_workspace(const fp_op &op, const Arr &x, int axis, int nthreads, bool keep,
                          std::unique_ptr<fp_workspace> &ws) {
    ws.reset(new fp_workspace());
    fp_workspace &w = *ws;
    w.op = &op;
    w.dtype = x.dtype;
    w.ndim = x.ndim;
    for (int d = 0; d < x.ndim; ++d) w.shape[d] = x.shape[d];
    w.axis = axis;
    w.numel = x.numel();
    w.n = x.shape[axis];
    w.rows = w.n > 0 ? w.numel / w.n : 0;
    w.nthreads = std::max(1, nthreads);
    w.keep = keep;
    const size_t item = x.itemsize();
    const bool elem = op.elementwise();
    if (const int k = op.image_ndim()) {
        // Images: the last k axes, `rows` of them.
        const int64_t *ishape = x.shape + (x.ndim - k);
        int64_t N = 1;
        for (int d = 0; d < k; ++d) N *= ishape[d];
        w.image_ndim = k;
        w.image_numel = N;
        w.n = N;
        w.rows = N > 0 ? w.numel / N : 0;
        if (w.numel == 0) {  // a kept empty point still reports its (zero) gradients
            if (keep) {
                w.tot.resize(static_cast<size_t>(op.nparams() + 2));
                w.dparams.resize(static_cast<size_t>(op.nparams()) + 1);
            }
            return FP_OK;
        }
        const int64_t B = w.rows;
        w.image_sets = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(B, w.nthreads)));
        w.image_bytes = image_staging_bytes(N) + round64(op.image_bytes(ishape));
        w.image_thread_bytes = round64(op.image_thread_bytes(ishape));
        w.image_work.resize(mul_size(w.image_bytes, static_cast<size_t>(w.image_sets)) / sizeof(double) + 1);
        w.image_thread.resize(mul_size(w.image_thread_bytes, static_cast<size_t>(w.nthreads)) / sizeof(double) + 1);
        w.dual.resize(static_cast<size_t>(B) * static_cast<size_t>(k) * static_cast<size_t>(N));
        w.image_gaps.resize(static_cast<size_t>(B));
        w.image_iters.resize(static_cast<size_t>(B));
        w.image_zerr.resize(static_cast<size_t>(B));
        w.image_exact.resize(static_cast<size_t>(B));
        w.image_smooth.resize(static_cast<size_t>(B));
        if (keep) {
            const size_t bytes = static_cast<size_t>(w.numel) * item;
            w.xsave.resize(bytes / sizeof(double) + 1);
            w.zsave.resize(bytes / sizeof(double) + 1);
            w.struct_bytes = round64(op.struct_bytes(ishape));
            w.structure.resize(mul_size(w.struct_bytes, static_cast<size_t>(B)) / sizeof(double) + 1);
            const int64_t P = op.nparams();
            w.nblocks = reduction_blocks(B, P);
            w.acc.resize(static_cast<size_t>(w.nblocks * (P + 2)));
            w.tot.resize(static_cast<size_t>(P + 2));
            w.dparams.resize(static_cast<size_t>(P) + 1);
            w.deriv_res.resize(static_cast<size_t>(B));
            w.deriv_its.resize(static_cast<size_t>(B));
            w.steps.resize(static_cast<size_t>(B));
            if (op.has_branches())
                w.branch.resize(static_cast<size_t>(std::max(w.numel, B * op.kink_sites())) + 1);
            if (op.struct_kind() != FP_STRUCT_NONE) {  // a label map per image set
                w.struct_thread_bytes = round64(4 * static_cast<size_t>(N));
                w.struct_scratch.resize(mul_size(w.struct_thread_bytes, static_cast<size_t>(w.image_sets)) /
                                            sizeof(double) +
                                        1);
            }
        }
        return FP_OK;
    }
    // A row runs on one thread: no more scratch regions than rows (audit C2).
    if (!elem) w.nthreads = static_cast<int>(std::min<int64_t>(w.nthreads, std::max<int64_t>(w.rows, 1)));
    if (!elem && w.numel > 0) {
        w.thread_bytes = row_thread_bytes(op, w.n, item);
        w.scratch.resize(mul_size(w.thread_bytes, static_cast<size_t>(w.nthreads)) / sizeof(double) + 1);
    }
    if (keep) {
        const size_t bytes = static_cast<size_t>(w.numel) * item;
        w.xsave.resize(bytes / sizeof(double) + 1);
        if (!elem) {
            w.zsave.resize(bytes / sizeof(double) + 1);
            w.code.resize(static_cast<size_t>(w.numel) + 1);
        }
        if (op.has_branches()) w.branch.resize(static_cast<size_t>(w.numel) + 1);
        if (!elem && op.has_branches()) w.code_default.resize(static_cast<size_t>(w.numel) + 1);
        const int64_t P = op.nparams();
        const int64_t units = elem ? ceil_div(w.numel, kChunk) : w.rows;
        w.nblocks = reduction_blocks(units, P);
        w.acc.resize(static_cast<size_t>(w.nblocks * (P + 2)));
        if (elem) w.block_counts.resize(static_cast<size_t>(3 * w.nblocks));
        if (!elem && op.struct_kind() != FP_STRUCT_NONE && w.numel > 0) {
            w.struct_thread_bytes = row_struct_bytes(w.n);
            w.struct_scratch.resize(mul_size(w.struct_thread_bytes, static_cast<size_t>(w.nthreads)) / sizeof(double) +
                                    1);
        }
        w.tot.resize(static_cast<size_t>(P + 2));
        w.dparams.resize(static_cast<size_t>(P) + 1);
        w.steps.resize(static_cast<size_t>(w.rows) + 1);
    }
    return FP_OK;
}

// Capped at max_threads() (audit C3): absurd requests created thousands of
// permanent pool threads.
int resolve_threads(const fp_options &o) noexcept {
    return std::min(o.threads > 0 ? o.threads : default_threads(), max_threads());
}

// The steps of fp_prox_steps: an array of the batch shape, x's shape without
// the axis (row and elementwise operators) or without the image axes.
fp_status read_steps(const fp_op &op, const fp_array *sa, const Arr &x, int axis, Steps &st) noexcept {
    Arr t;
    if (fp_status s = to_arr(sa, t)) return s;
    const int k = op.image_ndim();
    const int nb = k ? x.ndim - k : x.ndim - 1;
    if (sa->ndim != nb) return FP_ESHAPE;
    st.data = t.data;
    st.dtype = t.dtype;
    st.ndim = nb;
    for (int d = 0, e = 0; d < (k ? nb : x.ndim); ++d) {
        if (!k && d == axis) continue;
        if (t.shape[e] != x.shape[d]) return FP_ESHAPE;
        st.shape[e] = t.shape[e];
        st.stride[e] = t.strides[e];
        ++e;
    }
    return FP_OK;
}

// The step range (audit A1): gamma times the operator's largest parameter at
// most kMaxScaledStep (op.hpp); then the operator's own check (SCAD, MCP).
fp_status step_ok(const fp_op &op, double g) noexcept {
    const double s = op.step_scale();
    if (s > 0.0 && !(g * s <= kMaxScaledStep)) return FP_EPARAM;
    return op.check_step(g);
}

// dst = src entry by entry through both arrays' strides, or every entry of
// dst = fill when src is null (an integer for the integer dtypes). memmove:
// dst may alias src exactly.
void copy_array(const Arr *src, const Arr &dst, double fill) noexcept {
    if (dst.numel() == 0) return;  // for_runs divides by every extent
    const int64_t *st[2] = {dst.strides, src ? src->strides : dst.strides};
    const ElemPlan p = make_elem_plan(dst.ndim, dst.shape, 0, false, 2, st);
    // Byte offsets stay signed: strides may be negative.
    const int64_t item = static_cast<int64_t>(dst.itemsize());
    const float f32 = static_cast<float>(fill);
    const bool integer = dst.dtype == FP_I8 || dst.dtype == FP_I32 || dst.dtype == FP_I64;
    const int64_t i64 = integer ? static_cast<int64_t>(fill) : 0;  // fill may be NaN otherwise
    const int8_t i8 = static_cast<int8_t>(i64);
    const int32_t i32 = static_cast<int32_t>(i64);
    const void *fv = dst.dtype == FP_F32   ? static_cast<const void *>(&f32)
                     : dst.dtype == FP_I8  ? static_cast<const void *>(&i8)
                     : dst.dtype == FP_I32 ? static_cast<const void *>(&i32)
                     : dst.dtype == FP_I64 ? static_cast<const void *>(&i64)
                                           : static_cast<const void *>(&fill);
    p.for_runs(0, p.numel, [&](const int64_t *off, int64_t len, int64_t, int64_t, int64_t) {
        for (int64_t i = 0; i < len; ++i)
            std::memmove(dst.data + (off[0] + i * p.stride[0][p.ndim - 1]) * item,
                         src ? src->data + (off[1] + i * p.stride[1][p.ndim - 1]) * item : fv,
                         static_cast<size_t>(item));
    });
}

fp_status prox_impl(const fp_op *op, double step, const fp_array *xa, fp_array *oa,
                    const fp_options *opt, fp_workspace *ws, fp_info *info, bool conj,
                    const fp_array *sa = nullptr) {
    if (!op || !xa || !oa) return FP_EINVAL;
    if (conj && !op->conj_supported()) return FP_EINVAL;
    if (!finite(step) || step < 0) return FP_EPARAM;
    const fp_options o = opt ? *opt : defaults();
    if (o.tie != FP_TIE_NONZERO && o.tie != FP_TIE_ZERO) return FP_EINVAL;
    Arr x, out;
    if (fp_status s = to_arr(xa, x)) return s;
    if (fp_status s = to_arr(oa, out)) return s;
    if (x.dtype != out.dtype) return FP_EDTYPE;
    if (!same_shape(x, out)) return FP_ESHAPE;
    // Before any change to ws, so that a rejected call keeps its structure.
    if (fp_status s = check_output(&x, out)) return s;
    int axis;
    if (op->image_ndim()) axis = x.ndim - 1;  // fp_options.axis is ignored (audit A5)
    else if (norm_axis(o.axis, x.ndim, axis)) return FP_EINVAL;
    const bool keep = o.keep_structure != 0;
    if (keep && !ws) return FP_EINVAL;
    Steps steps;
    if (sa) {
        if (op->image_ndim() && x.ndim < op->image_ndim()) return FP_ESHAPE;
        if (fp_status s = read_steps(*op, sa, x, axis, steps)) return s;
        Arr st;  // the steps are read during the work, so out must not overlap them
        if (fp_status s = to_arr(sa, st)) return s;
        if (fp_status s = check_output(&st, out)) return s;
    }

    std::unique_ptr<fp_workspace> local;
    int nt = resolve_threads(o);
    if (ws) {
        if (ws->op != op || ws->dtype != x.dtype || ws->axis != axis || ws->ndim != x.ndim)
            return FP_EINVAL;
        for (int d = 0; d < x.ndim; ++d)
            if (ws->shape[d] != x.shape[d]) return FP_ESHAPE;
        if (keep && !ws->keep) return FP_EINVAL;
        nt = std::min(nt, ws->nthreads);
    }

    if (info) {
        info->gap = 0.0;
        info->iters = 0;
        info->ties = 0;
    }
    const int k = op->image_ndim();
    if (k) {
        if (x.ndim < k) return FP_ESHAPE;
        if (!(o.tol >= 0.0) || !finite(o.tol) || o.max_iter < 1) return FP_EPARAM;
    }
    // The steps, before any work and before the empty batch returns (audit A3).
    // A positively homogeneous g gives y - sigma prox_{g/sigma}(y/sigma) =
    // y - prox_g(y): computed at gamma = sigma = 1, exact for every sigma > 0.
    const bool homog = conj && op->conj_homogeneous();
    const bool identity = conj && step == 0.0;  // prox_{0 g*} is the identity
    const double gamma = !conj ? step : (homog ? 1.0 : 1.0 / step);
    const double sigma = !conj ? 0.0 : (homog ? 1.0 : step);
    const Steps *sp = sa ? &steps : nullptr;
    if (sp) {
        // Every step, counted over the steps array: ws->rows is 0 for x of
        // shape (3, 0).
        int64_t count = 1;
        for (int d = 0; d < steps.ndim; ++d) count *= steps.shape[d];
        for (int64_t r = 0; r < count; ++r) {
            const double v = steps.at(r);
            if (!finite(v) || v < 0) return FP_EPARAM;
            if (fp_status s = step_ok(*op, v)) return s;
        }
    } else if (identity) {
    } else if (conj && !homog && !finite(gamma)) {
        return FP_EPARAM;  // 1 / sigma overflows
    } else if (fp_status s = step_ok(*op, gamma)) {
        return s;
    }
    if (x.numel() != 0) {
        if (k) {
            if (fp_status s = op->check_image(x.shape + (x.ndim - k))) return s;
        } else if (fp_status s = op->check_n(x.shape[axis])) {
            return s;
        }
    }
    // Every argument is checked: a rejected call leaves ws as it was (audit
    // A4). From here the call replaces the saved point, or discards it.
    if (ws) {
        ws->valid = false;
        ws->identity = false;
        ws->steps_on = false;
        ws->branch_on = false;  // a branch override belongs to the point it was set for
        ws->override_on = false;
    }
    // A kept call saves its point for the derivatives.
    auto save_point = [&](bool steps_on) {
        ws->valid = true;
        ws->identity = identity;
        ws->conj = conj;
        ws->conj_homog = homog;
        ws->gamma = gamma;
        ws->sigma = sigma;
        ws->tie = o.tie;
        ws->steps_on = steps_on;
    };
    if (x.numel() == 0) {  // an empty point: its derivatives are empty
        if (keep) save_point(false);
        if (ws && k) ws->dual_valid = true;  // and its dual, which get copies (audit P8)
        return FP_OK;
    }

    // sigma = 0 in the conjugate: y itself, and with keep the identity's
    // derivatives (the sigma-derivative NaN).
    if (identity) {
        copy_array(&x, out, 0.0);
        if (keep) save_point(false);
        return FP_OK;
    }

    if (!ws) {
        if (fp_status s = build_workspace(*op, x, axis, nt, false, local)) return s;
        ws = local.get();
    }
    if (sp && keep)
        for (int64_t r = 0; r < ws->rows; ++r) ws->steps.data()[r] = steps.at(r);
    if (k) {
        fp_status s = x.dtype == FP_F32
                          ? image_prox_driver<float>(*op, x, out, gamma, conj, sigma, *ws, nt, keep,
                                                     o.tol, o.max_iter, info, sp)
                          : image_prox_driver<double>(*op, x, out, gamma, conj, sigma, *ws, nt, keep,
                                                      o.tol, o.max_iter, info, sp);
        if ((s == FP_OK || s == FP_ENOTCONV) && keep) save_point(sp != nullptr);
        return s;
    }
    int64_t ties = 0;
    fp_status s = x.dtype == FP_F32
                      ? prox_driver<float>(*op, x, out, axis, gamma, conj, sigma, *ws, nt, keep, o.tie, &ties, sp)
                      : prox_driver<double>(*op, x, out, axis, gamma, conj, sigma, *ws, nt, keep, o.tie, &ties, sp);
    if (info) info->ties = ties;
    if (s == FP_OK && keep) save_point(sp != nullptr);
    return s;
}

// prox_impl, keeping the call's fp_info and status in ws for fp_workspace_info.
fp_status prox_call(const fp_op *op, double step, const fp_array *xa, fp_array *oa, const fp_options *opt,
                    fp_workspace *ws, fp_info *info, bool conj, const fp_array *sa = nullptr) {
    if (!ws) return prox_impl(op, step, xa, oa, opt, ws, info, conj, sa);
    ws->info_valid = false;
    fp_info local = {0.0, 0, 0};
    const fp_status s = prox_impl(op, step, xa, oa, opt, ws, info ? info : &local, conj, sa);
    if (s == FP_OK || s == FP_ENOTCONV) {
        ws->info = info ? *info : local;
        ws->info_status = s;
        ws->info_valid = true;
    }
    return s;
}

fp_status check_like_ws(const fp_workspace &ws, const Arr &a) noexcept {
    if (a.dtype != ws.dtype) return FP_EDTYPE;
    if (a.ndim != ws.ndim) return FP_ESHAPE;
    for (int d = 0; d < a.ndim; ++d)
        if (a.shape[d] != ws.shape[d]) return FP_ESHAPE;
    return FP_OK;
}

fp_status jvp_impl(const fp_op *op, const fp_workspace *ws, const fp_array *dxa, double dgamma,
                   const fp_array *dpa, fp_array *oa) {
    if (!op || !ws || !oa) return FP_EINVAL;
    if (ws->op != op || !ws->valid) return FP_EINVAL;
    if (!finite(dgamma)) return FP_EPARAM;
    Arr dx, out;
    if (fp_status s = to_arr(oa, out)) return s;
    if (fp_status s = check_like_ws(*ws, out)) return s;
    if (dxa) {
        if (fp_status s = to_arr(dxa, dx)) return s;
        if (fp_status s = check_like_ws(*ws, dx)) return s;
    }
    if (fp_status s = check_output(dxa ? &dx : nullptr, out)) return s;
    const int64_t P = op->nparams();
    const double *dp = nullptr;
    if (dpa && P > 0) {
        if (fp_status s = read_exact(dpa, P, ws->dparams.data())) return s;
        dp = ws->dparams.data();
    }
    if (ws->identity) {  // J = I; the sigma-derivative is not defined for every g
        if (dgamma != 0.0) copy_array(nullptr, out, std::numeric_limits<double>::quiet_NaN());
        else copy_array(dxa ? &dx : nullptr, out, 0.0);
        return FP_OK;
    }
    if (ws->numel == 0) return FP_OK;
    const Arr *dxp = dxa ? &dx : nullptr;
    if (ws->conj_homog) dgamma = 0.0;  // constant in sigma (y - prox_g(y))
    if (op->image_ndim())
        return ws->dtype == FP_F32 ? image_jvp_driver<float>(*op, *ws, dxp, dgamma, dp, out)
                                   : image_jvp_driver<double>(*op, *ws, dxp, dgamma, dp, out);
    return ws->dtype == FP_F32 ? jvp_driver<float>(*op, *ws, dxp, dgamma, dp, out)
                               : jvp_driver<double>(*op, *ws, dxp, dgamma, dp, out);
}

int spectral_bit(const fp_spectral_kind &kind) noexcept {
    const int k = enum_bits(kind);
    return k == FP_SPECTRAL_SVD ? kSpectralSvd : k == FP_SPECTRAL_EIG ? kSpectralEig : 0;
}

// A (..., k, k) array over the batch of ws: its leading axes those of ws.
fp_status check_square(const fp_workspace &ws, const Arr &a) noexcept {
    if (a.dtype != ws.dtype) return FP_EDTYPE;
    if (a.ndim != ws.ndim + 1) return FP_ESHAPE;
    for (int d = 0; d < ws.ndim; ++d)
        if (a.shape[d] != ws.shape[d]) return FP_ESHAPE;
    return a.shape[ws.ndim] == ws.n ? FP_OK : FP_ESHAPE;
}

// The diagonal of a (..., k, k) array, as a (..., k) view.
Arr diagonal(const Arr &a) noexcept {
    Arr d = a;
    d.ndim = a.ndim - 1;
    d.strides[d.ndim - 1] = a.strides[d.ndim - 1] + a.strides[d.ndim];
    return d;
}

fp_status spectral_jvp_impl(const fp_op *op, const fp_workspace *ws, const fp_spectral_kind &kind,
                            const fp_array *sa, const fp_array *za, const fp_array *aa, double dgamma,
                            const fp_array *dpa, fp_array *ba, fp_array *ra) {
    if (!op || !ws || !sa || !za || !aa || !ba) return FP_EINVAL;
    const int bit = spectral_bit(kind);
    if (!bit || !(op->spectral() & bit)) return FP_EINVAL;
    if (ws->op != op || !ws->valid || ws->conj || ws->axis != ws->ndim - 1) return FP_EINVAL;
    const bool svd = bit == kSpectralSvd;
    if (ra && !svd) return FP_EINVAL;
    if (!finite(dgamma)) return FP_EPARAM;
    Arr s, z, a, b, r;
    if (fp_status e = to_arr(sa, s)) return e;
    if (fp_status e = to_arr(za, z)) return e;
    if (fp_status e = to_arr(aa, a)) return e;
    if (fp_status e = to_arr(ba, b)) return e;
    if (fp_status e = check_like_ws(*ws, s)) return e;
    if (fp_status e = check_like_ws(*ws, z)) return e;
    if (fp_status e = check_square(*ws, a)) return e;
    if (fp_status e = check_square(*ws, b)) return e;
    if (ra) {
        if (fp_status e = to_arr(ra, r)) return e;
        if (fp_status e = check_like_ws(*ws, r)) return e;
    }
    // b and r must not overlap a, s or z (nor each other).
    for (const Arr *in : {&a, &s, &z}) {
        if (fp_status e = check_output(in, b)) return e;
        if (ra)
            if (fp_status e = check_output(in, r)) return e;
    }
    if (ra)
        if (fp_status e = check_output(&b, r)) return e;
    const int64_t P = op->nparams();
    const double *dp = nullptr;
    if (dpa && P > 0) {
        if (fp_status e = read_exact(dpa, P, ws->dparams.data())) return e;
        dp = ws->dparams.data();
    }
    if (ws->numel == 0) return FP_OK;

    // diag(B) = J diag(A) + the gamma and parameter terms: fp_jvp on the diagonals.
    const Arr da = diagonal(a), db = diagonal(b);
    fp_status e = ws->dtype == FP_F32 ? jvp_driver<float>(*op, *ws, &da, dgamma, dp, db)
                                      : jvp_driver<double>(*op, *ws, &da, dgamma, dp, db);
    if (e) return e;
    // One scratch region per thread the matrices can use (audit C2).
    const size_t need =
        spectral_doubles(ws->n) * static_cast<size_t>(std::min<int64_t>(ws->nthreads, std::max<int64_t>(ws->rows, 1)));
    if (ws->spectral.size() < need) ws->spectral.resize(need);
    return ws->dtype == FP_F32
               ? spectral_driver<float>(*op, *ws, svd, s, z, a, b, ra ? &r : nullptr, ws->spectral.data())
               : spectral_driver<double>(*op, *ws, svd, s, z, a, b, ra ? &r : nullptr, ws->spectral.data());
}

fp_status vjp_impl(const fp_op *op, const fp_workspace *ws, const fp_array *ga, fp_array *gxa,
                   double *ggamma, fp_array *gpa) {
    if (!op || !ws || !ga || !gxa) return FP_EINVAL;
    if (ws->op != op || !ws->valid) return FP_EINVAL;
    Arr g, gx;
    if (fp_status s = to_arr(ga, g)) return s;
    if (fp_status s = to_arr(gxa, gx)) return s;
    if (fp_status s = check_like_ws(*ws, g)) return s;
    if (fp_status s = check_like_ws(*ws, gx)) return s;
    if (fp_status s = check_output(&g, gx)) return s;
    const int64_t P = op->nparams();
    if (gpa) {  // validate before any work
        Arr t;
        if (fp_status s = to_arr(gpa, t)) return s;
        if (gpa->ndim > 1 || t.numel() != P) return FP_ESHAPE;
    }
    double *tot = ws->tot.data();
    for (int64_t k = 0; k < P + 2; ++k) tot[k] = 0.0;
    if (ws->identity) {  // gx = g, no parameter gradient, the sigma-derivative NaN
        copy_array(&g, gx, 0.0);
        if (ggamma) *ggamma = std::numeric_limits<double>::quiet_NaN();
        if (gpa && P > 0) return write_exact(gpa, P, tot);
        return FP_OK;
    }
    if (ws->numel > 0) {
        fp_status s;
        if (op->image_ndim())
            s = ws->dtype == FP_F32 ? image_vjp_driver<float>(*op, *ws, g, gx, tot)
                                    : image_vjp_driver<double>(*op, *ws, g, gx, tot);
        else {
            // Elementwise operators skip the sums nobody asked for (audit C5).
            const bool want = ggamma != nullptr || (gpa && P > 0);
            s = ws->dtype == FP_F32 ? vjp_driver<float>(*op, *ws, g, gx, tot, want)
                                    : vjp_driver<double>(*op, *ws, g, gx, tot, want);
        }
        if (s) return s;
    }
    if (ws->conj) {
        // Constant in sigma when computed as y - prox_g(y).
        tot[P] = ws->conj_homog ? 0.0 : tot[P + 1] + tot[P] / ws->sigma;
        for (int64_t k = 0; k < P; ++k) tot[k] *= -ws->sigma;
    }
    if (ggamma) *ggamma = tot[P];
    if (gpa && P > 0) return write_exact(gpa, P, tot);
    return FP_OK;
}

// fp_vjp with ggamma written to a 0-d array on the workspace's device.
fp_status vjp_device_impl(const fp_op *op, const fp_workspace *ws, const fp_array *ga, fp_array *gxa,
                          fp_array *gga, fp_array *gpa) {
    Arr t;
    if (gga) {  // validate before any work
        if (fp_status s = to_arr(gga, t)) return s;
        if (gga->ndim != 0) return FP_ESHAPE;
    }
    double gg = 0.0;
    if (fp_status s = vjp_impl(op, ws, ga, gxa, gga ? &gg : nullptr, gpa)) return s;
    if (gga) {
        if (t.dtype == FP_F32) *reinterpret_cast<float *>(t.data) = static_cast<float>(gg);
        else *reinterpret_cast<double *>(t.data) = gg;
    }
    return FP_OK;
}

// The workspace must hold a kept call of an operator with kinks (for branch
// overrides, an elementwise one), and a must be an FP_I8 array of x's shape.
fp_status kink_arrays(const fp_workspace *ws, const fp_array *aa, Arr *a, bool branch) noexcept {
    if (!ws || !ws->op || !ws->keep || !ws->op->has_kinks()) return FP_EINVAL;
    if (branch && !ws->op->has_branches()) return FP_EINVAL;
    if (!ws->valid) return FP_EINVAL;
    if (!aa) return FP_OK;
    if (fp_status s = to_arr(aa, *a, Dt::Int8)) return s;
    if (a->ndim != ws->ndim) return FP_ESHAPE;
    const int64_t sites = ws->op->kink_sites();  // tree TV: (..., m) rather than x's shape
    for (int d = 0; d < a->ndim; ++d)
        if (a->shape[d] != (sites && d == a->ndim - 1 ? sites : ws->shape[d])) return FP_ESHAPE;
    return FP_OK;
}

// g(x) per row (along the axis) or per image (the last image_ndim axes), one
// value per batch entry in out (since 0.5). Rows run in C order over the
// batch axes, so row r writes out at the C-order index r of its shape.
template <class T>
void value_driver(const fp_op &op, const Arr &x, int axis, double tol, const Arr &out) {
    const int k = op.image_ndim();
    const int64_t *st[1] = {x.strides};
    double *od = reinterpret_cast<double *>(out.data);
    auto out_at = [&](int64_t r) {  // out's element r in C order
        int64_t o = 0;
        for (int d = out.ndim - 1; d >= 0; --d) {
            o += (r % out.shape[d]) * out.strides[d];
            r /= out.shape[d];
        }
        return o;
    };
    const T *xd = reinterpret_cast<const T *>(x.data);
    auto any_nan = [](const std::vector<double> &v) {
        for (double a : v)
            if (std::isnan(a)) return true;
        return false;
    };
    if (k) {
        const ImagePlan p = make_image_plan(x.ndim, x.shape, k, 1, st);
        std::vector<double> xb(static_cast<size_t>(p.numel));
        for (int64_t r = 0; r < p.images; ++r) {
            int64_t off[1];
            p.offsets(r, off);
            p.for_each(0, [&](int64_t i, int64_t o) { xb[static_cast<size_t>(i)] = static_cast<double>(xd[off[0] + o]); });
            od[out_at(r)] = any_nan(xb) ? std::numeric_limits<double>::quiet_NaN()
                                        : op.value_image(xb.data(), x.shape + p.nlead, tol);
        }
        return;
    }
    const RowPlan p = make_row_plan(x.ndim, x.shape, axis, 1, st);
    const int64_t n = p.n;
    std::vector<double> xb(static_cast<size_t>(n)), scr(static_cast<size_t>(std::max(n, op.value_doubles(n))) + 1);
    for (int64_t r = 0; r < p.rows; ++r) {
        int64_t off[1];
        p.offsets(r, off);
        for (int64_t i = 0; i < n; ++i) xb[static_cast<size_t>(i)] = static_cast<double>(xd[off[0] + i * p.astride[0]]);
        od[out_at(r)] = any_nan(xb) ? std::numeric_limits<double>::quiet_NaN()
                                    : op.value_row(xb.data(), n, tol, scr.data());
    }
}

fp_status value_impl(const fp_op *op, const fp_array *xa, const fp_options *opt, fp_array *oa) {
    if (!op || !xa || !oa) return FP_EINVAL;
    if (!op->has_value()) return FP_EINVAL;
    const fp_options o = opt ? *opt : defaults();
    if (!(o.tol >= 0.0) || !finite(o.tol)) return FP_EPARAM;
    Arr x, out;
    if (fp_status s = to_arr(xa, x)) return s;
    if (fp_status s = to_arr(oa, out)) return s;
    if (out.dtype != FP_F64) return FP_EDTYPE;
    if (fp_status s = check_output(&x, out)) return s;
    // The batch shape: x's without the axis (rows) or the image axes.
    const int k = op->image_ndim();
    int axis = 0;
    int64_t bshape[FP_MAX_NDIM];
    int nb = 0;
    if (k) {
        if (xa->ndim < k) return FP_ESHAPE;
        for (int d = 0; d < xa->ndim - k; ++d) bshape[nb++] = xa->shape[d];
    } else {
        if (norm_axis(o.axis, x.ndim, axis)) return FP_EINVAL;
        for (int d = 0; d < xa->ndim; ++d)
            if (d != axis) bshape[nb++] = xa->shape[d];
    }
    if (oa->ndim != nb) return FP_ESHAPE;
    for (int d = 0; d < nb; ++d)
        if (oa->shape[d] != bshape[d]) return FP_ESHAPE;
    if (k) {
        if (fp_status s = op->check_image(x.shape + (x.ndim - k))) return s;
    } else if (fp_status s = op->check_n(x.shape[axis])) {
        return s;
    }
    if (out.numel() == 0) return FP_OK;
    if (x.dtype == FP_F32) value_driver<float>(*op, x, axis, o.tol, out);
    else value_driver<double>(*op, x, axis, o.tol, out);
    return FP_OK;
}

fp_status kinks_impl(const fp_workspace *ws, double tol, fp_array *ma, int64_t *count) {
    if (!(tol >= 0.0) || !finite(tol)) return FP_EPARAM;
    Arr m;
    if (fp_status s = kink_arrays(ws, ma, &m, false)) return s;
    if (ma)
        if (fp_status s = check_output(nullptr, m)) return s;
    const Arr *mp = ma ? &m : nullptr;
    const bool f32 = ws->dtype == FP_F32;
    const fp_op &op = *ws->op;
    int64_t c = 0;
    if (ws->identity) {  // no kinks: a zero mask
        if (mp) copy_array(nullptr, *mp, 0.0);
    } else if (ws->numel == 0) c = 0;
    else if (op.elementwise()) c = f32 ? kinks_driver<float>(op, *ws, tol, mp) : kinks_driver<double>(op, *ws, tol, mp);
    else if (ws->image_ndim) c = f32 ? image_kinks_driver<float>(op, *ws, tol, mp) : image_kinks_driver<double>(op, *ws, tol, mp);
    else c = f32 ? row_kinks_driver<float>(op, *ws, tol, mp) : row_kinks_driver<double>(op, *ws, tol, mp);
    if (count) *count = c;
    return FP_OK;
}

// Row and image operators (since 0.3): the saved structure with the override
// applied, from the saved x, z (and dual), or restored to the prox call's.
template <class T>
void apply_structure(fp_workspace &ws, const int8_t *side, double tol) {
    const fp_op &op = *ws.op;
    const double t = ws.conj ? tol / ws.sigma : tol;
    if (ws.image_ndim) {
        const size_t bytes = ws.structure.size() * sizeof(double);
        if (!side) {  // restore the prox call's structure
            if (ws.override_on) std::memcpy(ws.structure.data(), ws.struct_default.data(), bytes);
            return;
        }
        if (!ws.override_on) {
            if (ws.struct_default.size() != ws.structure.size()) ws.struct_default.resize(ws.structure.size());
            std::memcpy(ws.struct_default.data(), ws.structure.data(), bytes);
        }
        const int64_t N = ws.image_numel, B = ws.rows;
        const int k = ws.image_ndim;
        std::vector<double> xb(static_cast<size_t>(N)), zb(static_cast<size_t>(N));
        const int64_t *ishape = ws.shape + (ws.ndim - k);
        for (int64_t r = 0; r < B; ++r) {
            const T *xs = ws.xs<T>() + r * N, *zs = ws.zs<T>() + r * N;
            for (int64_t i = 0; i < N; ++i) {
                xb[static_cast<size_t>(i)] = static_cast<double>(xs[i]);
                zb[static_cast<size_t>(i)] = static_cast<double>(zs[i]);
            }
            const double *dual = ws.dual.data() + static_cast<size_t>(r) * static_cast<size_t>(k) * static_cast<size_t>(N);
            const int64_t S = op.kink_sites() ? op.kink_sites() : N;  // sides per image
            op.branch_image(ws.image_struct(r), xb.data(), zb.data(), dual, ishape, ws.step(r), t,
                            side + r * S, ws.image_set(0) + image_staging_bytes(N));
        }
        return;
    }
    const int64_t n = ws.n;
    // Allocated before the codes change, so that bad_alloc leaves them (audit A9).
    std::vector<double> xb(side ? static_cast<size_t>(n) : 0), zb(side ? static_cast<size_t>(n) : 0);
    std::vector<std::max_align_t> scr(side ? op.scratch_bytes(n) / sizeof(std::max_align_t) + 1 : 0);
    if (!ws.override_on) std::memcpy(ws.code_default.data(), ws.code.data(), static_cast<size_t>(ws.numel));
    else std::memcpy(ws.code.data(), ws.code_default.data(), static_cast<size_t>(ws.numel));
    if (!side) return;
    for (int64_t r = 0; r < ws.rows; ++r) {
        const T *xs = ws.xs<T>() + r * n, *zs = ws.zs<T>() + r * n;
        for (int64_t i = 0; i < n; ++i) {
            xb[static_cast<size_t>(i)] = static_cast<double>(xs[i]);
            zb[static_cast<size_t>(i)] = static_cast<double>(zs[i]);
        }
        op.branch_row(xb.data(), zb.data(), ws.code.data() + r * n, n, ws.step(r), t, side + r * n, scr.data());
    }
}

void restore_structure(fp_workspace &ws) {
    if (ws.dtype == FP_F32) apply_structure<float>(ws, nullptr, 0.0);
    else apply_structure<double>(ws, nullptr, 0.0);
    ws.override_on = false;
}

// Copies side into the order of the saved structure (rows with the axis
// last, or images in C order) and applies it.
fp_status set_structure_branch(fp_workspace &ws, const Arr &a, double tol) {
    const int8_t *src = reinterpret_cast<const int8_t *>(a.data);
    int8_t *dst = ws.branch.data();
    bool ok = true;
    if (ws.image_ndim) {
        const int64_t *st[1] = {a.strides};
        const ImagePlan p = make_image_plan(a.ndim, a.shape, ws.image_ndim, 1, st);
        for (int64_t r = 0; r < p.images; ++r) {
            int64_t off[1];
            p.offsets(r, off);
            p.for_each(0, [&](int64_t i, int64_t o) {
                const int8_t v = src[off[0] + o];
                ok = ok && v >= -1 && v <= 1;
                dst[r * p.numel + i] = v;
            });
        }
    } else {
        const int64_t *st[1] = {a.strides};
        const RowPlan p = make_row_plan(a.ndim, a.shape, ws.axis, 1, st);
        for (int64_t r = 0; r < p.rows; ++r) {
            int64_t off[1];
            p.offsets(r, off);
            for (int64_t i = 0; i < p.n; ++i) {
                const int8_t v = src[off[0] + i * p.astride[0]];
                ok = ok && v >= -1 && v <= 1;
                dst[r * p.n + i] = v;
            }
        }
    }
    if (!ok) return FP_EPARAM;
    if (ws.dtype == FP_F32) apply_structure<float>(ws, dst, tol);
    else apply_structure<double>(ws, dst, tol);
    ws.override_on = true;
    return FP_OK;
}

fp_status set_branch_impl(fp_workspace *ws, const fp_array *sa, double tol) {
    if (!sa) {
        if (!ws || !ws->op) return FP_EINVAL;
        ws->branch_on = false;
        if (ws->override_on) restore_structure(*ws);
        return FP_OK;
    }
    if (!(tol >= 0.0) || !finite(tol)) return FP_EPARAM;
    Arr a;
    if (fp_status s = kink_arrays(ws, sa, &a, true)) return s;
    if (ws->identity || ws->numel == 0) return FP_OK;  // no kinks to override
    if (!ws->op->elementwise()) return set_structure_branch(*ws, a, tol);
    // Into the workspace in C order, the order of the saved entries: checked
    // first, so that a failed call leaves an earlier override (audit A9).
    const int64_t *st[1] = {a.strides};
    const ElemPlan p = make_elem_plan(a.ndim, a.shape, ws->axis, false, 1, st);
    const int8_t *src = reinterpret_cast<const int8_t *>(a.data);
    int8_t *dst = ws->branch.data();
    const int64_t s0 = p.stride[0][p.ndim - 1];
    bool ok = true;
    p.for_runs(0, p.numel, [&](const int64_t *off, int64_t len, int64_t, int64_t, int64_t) {
        for (int64_t i = 0; i < len; ++i) {
            const int8_t v = src[off[0] + i * s0];
            ok = ok && v >= -1 && v <= 1;
        }
    });
    if (!ok) return FP_EPARAM;
    p.for_runs(0, p.numel, [&](const int64_t *off, int64_t len, int64_t pos, int64_t, int64_t) {
        for (int64_t i = 0; i < len; ++i) dst[pos + i] = src[off[0] + i * s0];
    });
    ws->branch_on = true;
    ws->branch_tol = tol;
    return FP_OK;
}

// ---- structure of the derivative (since 1.0.2) ------------------------------

// The queries need a kept point of an operator with a structure kind, on the
// CPU.
fp_status struct_ws(const fp_workspace *ws) noexcept {
    if (!ws || !ws->op || !ws->keep || !ws->valid) return FP_EINVAL;
    if (ws->device != FP_DEVICE_CPU || ws->op->struct_kind() == FP_STRUCT_NONE) return FP_EINVAL;
    return FP_OK;
}

// The units of the kept point: the whole array for elementwise operators,
// each row or image otherwise; none for an empty array.
int64_t struct_units(const fp_workspace &ws) noexcept {
    if (ws.numel == 0) return 0;
    return ws.op->elementwise() ? 1 : ws.rows;
}
int64_t struct_size(const fp_workspace &ws) noexcept { return ws.op->elementwise() ? ws.numel : ws.n; }

// An output array of the queries: x's shape, of the dtype class kind (Float:
// x's dtype), writable.
fp_status struct_out(const fp_workspace &ws, const fp_array *a, Arr &out, Dt kind) noexcept {
    if (fp_status s = to_arr(a, out, kind)) return s;
    if (kind == Dt::Float && out.dtype != ws.dtype) return FP_EDTYPE;
    if (out.ndim != ws.ndim) return FP_ESHAPE;
    for (int d = 0; d < out.ndim; ++d)
        if (out.shape[d] != ws.shape[d]) return FP_ESHAPE;
    return check_output(nullptr, out);
}

fp_status units_impl(const fp_workspace *ws, int64_t *count, int64_t *size) {
    if (fp_status s = struct_ws(ws)) return s;
    if (count) *count = struct_units(*ws);
    if (size) *size = struct_size(*ws);
    return FP_OK;
}

// The structure of the kept point into the outputs given, each checked and
// each may be null: per entry its state, label, sign and coefficient, per
// unit its regions and constraint flag. Returns the entries not FREE. After
// fp_prox_conj at sigma = 0 (ws.identity) the conjugate's Jacobian is I, so
// the inner prox's, I - J, is 0: every entry ZERO, no regions.
int64_t structure(const fp_workspace &ws, const Arr *st, const Arr *lab, const Arr *sg, const Arr *coef,
                  int64_t *nregions, int8_t *active) {
    const int64_t units = struct_units(ws);
    if (ws.numel == 0) return 0;
    if (ws.identity) {
        if (st) copy_array(nullptr, *st, FP_ENTRY_ZERO);
        if (lab) copy_array(nullptr, *lab, -1.0);
        if (sg) copy_array(nullptr, *sg, 0.0);
        if (coef) copy_array(nullptr, *coef, 0.0);
        if (nregions) std::fill(nregions, nregions + units, int64_t{0});
        if (active) std::fill(active, active + units, int8_t{0});
        return ws.numel;
    }
    const fp_op &op = *ws.op;
    const bool f32 = ws.dtype == FP_F32;
    if (op.elementwise()) {  // one unit, no constraint
        int64_t nf = 0;
        const int64_t k = f32 ? elem_structure_driver<float>(op, ws, st, lab, sg, &nf)
                              : elem_structure_driver<double>(op, ws, st, lab, sg, &nf);
        if (coef) copy_array(nullptr, *coef, 0.0);
        if (nregions) nregions[0] = k;
        if (active) active[0] = 0;
        return nf;
    }
    if (op.image_ndim())
        return f32 ? image_structure_driver<float>(op, ws, st, lab, sg, coef, nregions, active)
                   : image_structure_driver<double>(op, ws, st, lab, sg, coef, nregions, active);
    return f32 ? row_structure_driver<float>(op, ws, st, lab, sg, coef, nregions, active)
               : row_structure_driver<double>(op, ws, st, lab, sg, coef, nregions, active);
}

fp_status states_impl(const fp_workspace *ws, fp_array *sa, int64_t *nfixed) {
    if (fp_status s = struct_ws(ws)) return s;
    Arr st;
    if (sa)
        if (fp_status s = struct_out(*ws, sa, st, Dt::Int8)) return s;
    const int64_t nf = structure(*ws, sa ? &st : nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (nfixed) *nfixed = nf;
    return FP_OK;
}

fp_status regions_impl(const fp_workspace *ws, fp_array *la, fp_array *ga, int64_t *nregions, int64_t count) {
    if (fp_status s = struct_ws(ws)) return s;
    if (nregions && count != struct_units(*ws)) return FP_ESHAPE;
    Arr lab, sg;
    if (la) {
        if (fp_status s = struct_out(*ws, la, lab, Dt::Label)) return s;
        if (lab.dtype == FP_I32 && struct_size(*ws) > std::numeric_limits<int32_t>::max()) return FP_ESHAPE;
    }
    if (ga)
        if (fp_status s = struct_out(*ws, ga, sg, Dt::Int8)) return s;
    structure(*ws, nullptr, la ? &lab : nullptr, ga ? &sg : nullptr, nullptr, nregions, nullptr);
    return FP_OK;
}

fp_status constraint_impl(const fp_workspace *ws, fp_array *ca, int8_t *active, int64_t count) {
    if (fp_status s = struct_ws(ws)) return s;
    if (active && count != struct_units(*ws)) return FP_ESHAPE;
    Arr c;
    if (ca)
        if (fp_status s = struct_out(*ws, ca, c, Dt::Float)) return s;
    structure(*ws, nullptr, nullptr, nullptr, ca ? &c : nullptr, nullptr, active);
    return FP_OK;
}

// Copies between the workspace dual (per image, k blocks of image_numel) and a
// public array of shape (k, *x.shape). to_ws: from the array into the dual.
template <class T>
void dual_copy(fp_workspace &ws, const Arr &a, bool to_ws) noexcept {
    const int k = ws.image_ndim;
    const int64_t N = ws.image_numel;
    for (int b = 0; b < k; ++b) {
        Arr v;
        v.dtype = a.dtype;
        v.ndim = ws.ndim;
        v.data = a.data + b * a.strides[0] * static_cast<int64_t>(sizeof(T));
        for (int d = 0; d < ws.ndim; ++d) {
            v.shape[d] = ws.shape[d];
            v.strides[d] = a.strides[d + 1];
        }
        const int64_t *st[1] = {v.strides};
        const ImagePlan p = make_image_plan(v.ndim, v.shape, k, 1, st);
        T *base = reinterpret_cast<T *>(v.data);
        for (int64_t r = 0; r < p.images; ++r) {
            int64_t off[1];
            p.offsets(r, off);
            double *dual = ws.dual.data() + (static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(b)) * static_cast<size_t>(N);
            T *img = base + off[0];
            if (to_ws)  // NaN, which get returns for an image that held one, as 0 (audit A8)
                p.for_each(0, [&](int64_t i, int64_t o) {
                    const double v = static_cast<double>(img[o]);
                    dual[i] = std::isnan(v) ? 0.0 : v;
                });
            else
                p.for_each(0, [&](int64_t i, int64_t o) { img[o] = static_cast<T>(dual[i]); });
        }
    }
}

// fp_workspace_refine (feedback U4): the kept call of an iterative image
// operator solved again, warm from its dual, to tol. The point the
// derivatives read, its dual and fp_workspace_info become the refined call's.
fp_status refine_impl(const fp_op *op, fp_workspace *ws, double tol, int max_iter, fp_array *oa,
                      fp_info *info) {
    if (!op || !ws || ws->op != op || !ws->valid || !ws->image_ndim) return FP_EINVAL;
    if (!(tol >= 0.0) || !finite(tol) || max_iter < 1) return FP_EPARAM;
    Arr out;
    if (oa) {
        if (fp_status s = to_arr(oa, out)) return s;
        if (fp_status s = check_like_ws(*ws, out)) return s;
        if (fp_status s = check_output(nullptr, out)) return s;
    }
    if (ws->identity) return oa ? FP_EINVAL : FP_OK;  // sigma = 0 saved no input to refine
    fp_info local = {0.0, 0, 0};
    fp_info *inf = info ? info : &local;
    *inf = local;
    ws->info_valid = false;
    fp_status s = FP_OK;
    if (ws->numel != 0) {
        // A branch override belongs to the point it was set for.
        ws->branch_on = false;
        ws->override_on = false;
        const Arr *po = oa ? &out : nullptr;
        s = ws->dtype == FP_F32 ? image_refine_driver<float>(*op, *ws, ws->nthreads, tol, max_iter, po, inf)
                                : image_refine_driver<double>(*op, *ws, ws->nthreads, tol, max_iter, po, inf);
        if (s != FP_OK && s != FP_ENOTCONV) {
            ws->valid = false;  // partly rewritten
            return s;
        }
    }
    ws->info = *inf;
    ws->info_status = s;
    ws->info_valid = true;
    return s;
}

// Whether every image's derivative at the kept point is exact, and the
// largest smoothing used where it is not (feedback U4).
fp_status deriv_exact_impl(const fp_workspace *ws, int *exact, double *smoothing) {
    if (!ws || !ws->valid || !ws->image_ndim) return FP_EINVAL;
    int all = 1;
    double mu = 0.0;
    if (!ws->identity && ws->numel != 0)
        for (int64_t r = 0; r < ws->rows; ++r) {
            all = all && ws->image_exact.data()[r] != 0.0;
            mu = std::max(mu, ws->image_smooth.data()[r]);
        }
    if (exact) *exact = all;
    if (smoothing) *smoothing = mu;
    return FP_OK;
}

fp_status dual_impl(fp_workspace *ws, const fp_array *aa, bool to_ws) {
    if (!ws || !aa || !ws->op || !ws->image_ndim) return FP_EINVAL;
    Arr a;
    if (fp_status s = to_arr(aa, a)) return s;
    if (aa->ndim != ws->ndim + 1 || a.shape[0] != ws->image_ndim) return FP_ESHAPE;
    for (int d = 0; d < ws->ndim; ++d)
        if (a.shape[d + 1] != ws->shape[d]) return FP_ESHAPE;
    if (!to_ws && !ws->dual_valid) return FP_EINVAL;
    if (!to_ws)
        if (fp_status s = check_output(nullptr, a)) return s;
    if (ws->numel == 0) return FP_OK;
    if (to_ws) {  // a seed must not be infinite; NaN entries are stored as 0
        bool ok = true;
        if (a.dtype == FP_F32) dual_copy<float>(*ws, a, true);
        else dual_copy<double>(*ws, a, true);
        const double *d = ws->dual.data();
        const size_t m = ws->dual.size();
        for (size_t i = 0; i < m; ++i) ok = ok && !std::isinf(d[i]);
        if (!ok) {
            ws->dual_valid = false;
            return FP_EINVAL;
        }
        ws->dual_valid = true;
        ws->seeded = true;
        return FP_OK;
    }
    if (a.dtype == FP_F32) dual_copy<float>(*ws, a, false);
    else dual_copy<double>(*ws, a, false);
    return FP_OK;
}

// Group penalty around a scalar map built by make_scalar.
template <class F>
fp_status group_radial(F make_scalar, const char *name, int64_t n, const int64_t *groups,
                       int64_t ngroups, const fp_array *group_weights, fp_op **out) {
    if (!out) return FP_EINVAL;
    *out = nullptr;
    std::vector<int64_t> g;
    if (groups) {
        // Bounded before groups + n is formed (audit A7).
        if (n < 1 || n > kMaxElems) return FP_EPARAM;
        g.assign(groups, groups + n);
    }
    Vec c;
    if (fp_status s = read_vec(group_weights, c)) return s;
    fp_op *h = nullptr;
    if (fp_status s = make_scalar(&h)) return s;
    return make_group_radial(h, name, std::move(g), ngroups, std::move(c), out);
}

}  // namespace
}  // namespace fp

using fp::guard;
using fp::Vec;

extern "C" {

FP_API void fp_options_init(fp_options *opt) {
    if (!opt) return;
    opt->axis = -1;
    opt->threads = 0;
    opt->tol = 1e-6;
    opt->max_iter = 1000;
    opt->tie = FP_TIE_NONZERO;
    opt->keep_structure = 0;
}

FP_API fp_status fp_op_soft(double lambda, const fp_array *weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_soft(lambda, std::move(w), out);
    });
}

FP_API fp_status fp_op_elastic_net(double l1, double l2, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_elastic_net(l1, l2, out);
    });
}

FP_API fp_status fp_op_huber(double lambda, double delta, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_huber(lambda, delta, out);
    });
}

FP_API fp_status fp_op_group_l2(double lambda, int64_t n, const int64_t *groups, int64_t ngroups,
                                const fp_array *group_weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        std::vector<int64_t> g;
        if (groups) {
            if (n < 1 || n > fp::kMaxElems) return FP_EPARAM;  // before groups + n (audit A7)
            g.assign(groups, groups + n);
        }
        Vec c;
        if (fp_status s = fp::read_vec(group_weights, c)) return s;
        return fp::make_group_l2(lambda, std::move(g), ngroups, std::move(c), out);
    });
}

FP_API fp_status fp_op_group_huber(double lambda, double delta, int64_t n, const int64_t *groups,
                                   int64_t ngroups, const fp_array *group_weights, fp_op **out) {
    return guard([&] {
        return fp::group_radial([&](fp_op **h) { return fp::make_huber(lambda, delta, h); },
                            "group_huber", n, groups, ngroups, group_weights, out);
    });
}

FP_API fp_status fp_op_group_scad(double lambda, double a, int64_t n, const int64_t *groups,
                                  int64_t ngroups, const fp_array *group_weights, fp_op **out) {
    return guard([&] {
        return fp::group_radial([&](fp_op **h) { return fp::make_scad(lambda, a, Vec{}, h); },
                            "group_scad", n, groups, ngroups, group_weights, out);
    });
}

FP_API fp_status fp_op_group_mcp(double lambda, double b, int64_t n, const int64_t *groups,
                                 int64_t ngroups, const fp_array *group_weights, fp_op **out) {
    return guard([&] {
        return fp::group_radial([&](fp_op **h) { return fp::make_mcp(lambda, b, Vec{}, h); },
                            "group_mcp", n, groups, ngroups, group_weights, out);
    });
}

FP_API fp_status fp_op_tree_l2(double l1, int64_t ngroups, const int64_t *group_ptr,
                               const int64_t *group_idx, const fp_array *group_weights,
                               fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        if (ngroups < 0 || (!group_ptr && ngroups > 0)) return FP_EPARAM;
        if (ngroups > fp::kMaxElems - 1) return FP_EPARAM;  // before group_ptr + ngroups + 1 (audit A7)
        std::vector<int64_t> ptr(1, 0), idx;
        if (ngroups > 0) {
            ptr.assign(group_ptr, group_ptr + ngroups + 1);
            if (ptr[0] != 0 || ptr.back() < 0 || ptr.back() > fp::kMaxElems) return FP_EPARAM;
            if (ptr.back() > 0 && !group_idx) return FP_EPARAM;
            for (int64_t g = 0; g < ngroups; ++g)
                if (ptr[g + 1] < ptr[g]) return FP_EPARAM;
            idx.assign(group_idx, group_idx + ptr.back());
        }
        Vec e;
        if (fp_status s = fp::read_vec(group_weights, e)) return s;
        return fp::make_tree_l2(l1, std::move(ptr), std::move(idx), std::move(e), out);
    });
}

FP_API fp_status fp_op_box(double lo, double hi, const fp_array *lower, const fp_array *upper,
                           fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec l, h;
        if (fp_status s = fp::read_vec(lower, l)) return s;
        if (fp_status s = fp::read_vec(upper, h)) return s;
        if (!lower) l.assign(1, lo);
        if (!upper) h.assign(1, hi);
        return fp::make_box(std::move(l), std::move(h), out);
    });
}

FP_API fp_status fp_op_nonneg(fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_nonneg(out);
    });
}

FP_API fp_status fp_op_log_barrier(double lambda, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_log_barrier(lambda, out);
    });
}

FP_API fp_status fp_op_l1_box(double lambda, const fp_array *weights, double lo, double hi,
                              const fp_array *lower, const fp_array *upper, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w, l, h;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        if (fp_status s = fp::read_vec(lower, l)) return s;
        if (fp_status s = fp::read_vec(upper, h)) return s;
        if (!lower) l.assign(1, lo);
        if (!upper) h.assign(1, hi);
        return fp::make_l1_box(lambda, std::move(w), std::move(l), std::move(h), out);
    });
}

FP_API fp_status fp_op_soc(fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_soc(out);
    });
}

FP_API fp_status fp_op_l2_ball(double radius, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_l2_ball(radius, out);
    });
}

FP_API fp_status fp_op_simplex(double radius, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_simplex(radius, out);
    });
}

FP_API fp_status fp_op_l1_ball(double radius, const fp_array *weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_l1_ball(radius, std::move(w), out);
    });
}

FP_API fp_status fp_op_linf(double lambda, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_linf(lambda, out);
    });
}

FP_API fp_status fp_op_max(double lambda, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_max(lambda, out);
    });
}

FP_API fp_status fp_op_sum_largest(double lambda, int64_t k, int absolute, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_sum_largest(lambda, k, absolute != 0, out);
    });
}

FP_API fp_status fp_op_hyperplane(const fp_array *a, double b, double lo, double hi,
                                  const fp_array *lower, const fp_array *upper, int inequality,
                                  fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec va, l, h;
        if (fp_status s = fp::read_vec(a, va)) return s;
        if (fp_status s = fp::read_vec(lower, l)) return s;
        if (fp_status s = fp::read_vec(upper, h)) return s;
        if (!lower) l.assign(1, lo);
        if (!upper) h.assign(1, hi);
        return fp::make_hyperplane(std::move(va), b, std::move(l), std::move(h), inequality != 0, out);
    });
}

FP_API fp_status fp_op_isotonic(const fp_array *weights, int increasing, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_isotonic(std::move(w), increasing != 0, out);
    });
}

FP_API fp_status fp_op_sorted_l1(const fp_array *weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_sorted_l1(std::move(w), out);
    });
}

FP_API fp_status fp_op_tv1d(const fp_array *edge_weights, fp_tv1d_method method, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(edge_weights, w)) return s;
        const int m = fp::enum_bits(method);
        if (m != FP_TV1D_AUTO && m != FP_TV1D_CONDAT && m != FP_TV1D_TAUT_STRING) return FP_EPARAM;
        return fp::make_tv1d(std::move(w), static_cast<fp_tv1d_method>(m), out);
    });
}

FP_API fp_status fp_op_tv1d_ball(double alpha, const fp_array *edge_weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(edge_weights, w)) return s;
        return fp::make_tv1d_ball(alpha, std::move(w), out);
    });
}

FP_API fp_status fp_op_tv1d_l2(const fp_array *edge_weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(edge_weights, w)) return s;
        return fp::make_tv1d_l2(std::move(w), out);
    });
}

FP_API fp_status fp_op_l0(double lambda, const fp_array *weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_l0(lambda, std::move(w), out);
    });
}

FP_API fp_status fp_op_lq(double lambda, double q, const fp_array *weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_lq(lambda, q, std::move(w), out);
    });
}

FP_API fp_status fp_op_scad(double lambda, double a, const fp_array *weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_scad(lambda, a, std::move(w), out);
    });
}

FP_API fp_status fp_op_mcp(double lambda, double b, const fp_array *weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(weights, w)) return s;
        return fp::make_mcp(lambda, b, std::move(w), out);
    });
}

FP_API fp_status fp_op_log_penalty(double lambda, double eps, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        return fp::make_log(lambda, eps, out);
    });
}

FP_API fp_status fp_op_tv_aniso(int ndim, const fp_array *axis_weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(axis_weights, w)) return s;
        return fp::make_tv_aniso(ndim, std::move(w), out);
    });
}

FP_API fp_status fp_op_tv_iso(int ndim, const fp_array *axis_weights, fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(axis_weights, w)) return s;
        return fp::make_tv_iso(ndim, std::move(w), out);
    });
}

FP_API fp_status fp_op_tv_aniso_weighted(int ndim, const fp_array *axis_weights, const fp_array *edge_weights,
                                         fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        if (!edge_weights) return FP_EINVAL;
        if (ndim != 2 && ndim != 3) return FP_EPARAM;
        Vec w, b;
        int64_t shape[4] = {1, 1, 1, 1};
        if (fp_status s = fp::read_vec(axis_weights, w)) return s;
        if (fp_status s = fp::read_nd(edge_weights, ndim + 1, b, shape)) return s;
        if (shape[0] != ndim) return FP_ESHAPE;
        return fp::make_tv_aniso(ndim, std::move(w), out, std::move(b), shape + 1);
    });
}

FP_API fp_status fp_op_tv_iso_weighted(int ndim, const fp_array *axis_weights, const fp_array *pixel_weights,
                                       fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        if (!pixel_weights) return FP_EINVAL;
        if (ndim != 2 && ndim != 3) return FP_EPARAM;
        Vec w, al;
        int64_t shape[3] = {1, 1, 1};
        if (fp_status s = fp::read_vec(axis_weights, w)) return s;
        if (fp_status s = fp::read_nd(pixel_weights, ndim, al, shape)) return s;
        return fp::make_tv_iso(ndim, std::move(w), out, std::move(al), shape);
    });
}

FP_API fp_status fp_op_graph_tv(int64_t n, int64_t m, const int64_t *edges, const fp_array *edge_weights,
                                fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(edge_weights, w)) return s;
        return fp::make_graph_tv(n, m, edges, std::move(w), out);
    });
}

FP_API fp_status fp_op_tree_tv(int64_t n, int64_t m, const int64_t *edges, const fp_array *edge_weights,
                               fp_op **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        Vec w;
        if (fp_status s = fp::read_vec(edge_weights, w)) return s;
        return fp::make_tree_tv(n, m, edges, std::move(w), out);
    });
}

FP_API void fp_op_free(fp_op *op) { delete op; }

FP_API int64_t fp_op_nparams(const fp_op *op) { return op ? op->nparams() : -1; }

FP_API const char *fp_op_name(const fp_op *op) { return op ? op->name() : ""; }

FP_API fp_status fp_workspace_create(const fp_op *op, const fp_array *like, const fp_options *opt,
                                     fp_workspace **out) {
    return guard([&] {
        if (!out) return FP_EINVAL;
        *out = nullptr;
        if (!op || !like) return FP_EINVAL;
        const fp_options o = opt ? *opt : fp::defaults();
        fp::Arr x;
        if (fp_status s = fp::to_arr(like, x)) return s;
        int axis;
        if (op->image_ndim()) axis = x.ndim - 1;  // fp_options.axis is ignored (audit A5)
        else if (fp::norm_axis(o.axis, x.ndim, axis)) return FP_EINVAL;
        if (const int k = op->image_ndim()) {
            if (x.ndim < k) return FP_ESHAPE;
            if (x.numel() > 0)
                if (fp_status s = op->check_image(x.shape + (x.ndim - k))) return s;
        } else if (x.numel() > 0) {
            if (fp_status s = op->check_n(x.shape[axis])) return s;
        }
        std::unique_ptr<fp_workspace> ws;
        if (fp_status s = fp::build_workspace(*op, x, axis, fp::resolve_threads(o),
                                              o.keep_structure != 0, ws))
            return s;
        *out = ws.release();
        return FP_OK;
    });
}

FP_API void fp_workspace_free(fp_workspace *ws) { delete ws; }

FP_API fp_status fp_workspace_get_dual(const fp_workspace *ws, fp_array *out) {
    return guard([&] { return fp::dual_impl(const_cast<fp_workspace *>(ws), out, false); });
}

FP_API fp_status fp_workspace_set_dual(fp_workspace *ws, const fp_array *dual) {
    return guard([&] { return fp::dual_impl(ws, dual, true); });
}

FP_API fp_status fp_workspace_kinks(const fp_workspace *ws, double tol, fp_array *mask,
                                    int64_t *count) {
    return guard([&] { return fp::kinks_impl(ws, tol, mask, count); });
}

FP_API fp_status fp_workspace_set_branch(fp_workspace *ws, const fp_array *side, double tol) {
    return guard([&] { return fp::set_branch_impl(ws, side, tol); });
}

FP_API fp_status fp_op_struct_kind(const fp_op *op, int *kind) {
    if (!op || !kind) return FP_EINVAL;
    *kind = op->struct_kind();
    return FP_OK;
}

FP_API fp_status fp_workspace_units(const fp_workspace *ws, int64_t *count, int64_t *size) {
    return guard([&] { return fp::units_impl(ws, count, size); });
}

FP_API fp_status fp_workspace_states(const fp_workspace *ws, fp_array *state, int64_t *nfixed) {
    return guard([&] { return fp::states_impl(ws, state, nfixed); });
}

FP_API fp_status fp_workspace_regions(const fp_workspace *ws, fp_array *labels, fp_array *sign,
                                      int64_t *nregions, int64_t count) {
    return guard([&] { return fp::regions_impl(ws, labels, sign, nregions, count); });
}

FP_API fp_status fp_workspace_constraint(const fp_workspace *ws, fp_array *coef, int8_t *active,
                                         int64_t count) {
    return guard([&] { return fp::constraint_impl(ws, coef, active, count); });
}

FP_API fp_status fp_workspace_set_deriv_tol(fp_workspace *ws, double tol, int max_iter) {
    if (!ws || !ws->image_ndim) return FP_EINVAL;
    if (!(tol >= 0.0) || !std::isfinite(tol) || max_iter < 1) return FP_EPARAM;
    ws->deriv_tol = tol;
    ws->deriv_max_iter = max_iter;
    return FP_OK;
}

FP_API fp_status fp_workspace_deriv_info(const fp_workspace *ws, double *residual, int *iters) {
    if (!ws || !ws->image_ndim) return FP_EINVAL;
    if (residual) *residual = ws->deriv_residual;
    if (iters) *iters = ws->deriv_iters;
    return FP_OK;
}

FP_API fp_status fp_workspace_set_warm(fp_workspace *ws, int on) {
    if (!ws || !ws->image_ndim) return FP_EINVAL;
    ws->warm = on != 0;
    return FP_OK;
}

FP_API fp_status fp_prox(const fp_op *op, double gamma, const fp_array *x, fp_array *out,
                         const fp_options *opt, fp_workspace *ws, fp_info *info) {
    return guard([&] { return fp::prox_call(op, gamma, x, out, opt, ws, info, false); });
}

FP_API fp_status fp_prox_steps(const fp_op *op, const fp_array *gamma, const fp_array *x, fp_array *out,
                               const fp_options *opt, fp_workspace *ws) {
    return guard([&] {
        if (!gamma) return FP_EINVAL;
        return fp::prox_call(op, 1.0, x, out, opt, ws, nullptr, false, gamma);
    });
}

FP_API fp_status fp_value(const fp_op *op, const fp_array *x, const fp_options *opt, fp_array *out) {
    return guard([&] { return fp::value_impl(op, x, opt, out); });
}

FP_API fp_status fp_prox_conj(const fp_op *op, double sigma, const fp_array *y, fp_array *out,
                              const fp_options *opt, fp_workspace *ws, fp_info *info) {
    return guard([&] { return fp::prox_call(op, sigma, y, out, opt, ws, info, true); });
}

FP_API fp_status fp_jvp(const fp_op *op, const fp_workspace *ws, const fp_array *dx,
                        double dgamma, const fp_array *dparams, fp_array *dout) {
    return guard([&] { return fp::jvp_impl(op, ws, dx, dgamma, dparams, dout); });
}

FP_API fp_status fp_vjp(const fp_op *op, const fp_workspace *ws, const fp_array *g, fp_array *gx,
                        double *ggamma, fp_array *gparams) {
    return guard([&] { return fp::vjp_impl(op, ws, g, gx, ggamma, gparams); });
}

FP_API fp_status fp_op_spectral_check(const fp_op *op, fp_spectral_kind kind) {
    if (!op) return FP_EINVAL;
    const int bit = fp::spectral_bit(kind);
    return bit && (op->spectral() & bit) ? FP_OK : FP_EINVAL;
}

FP_API fp_status fp_spectral_jvp(const fp_op *op, const fp_workspace *ws, fp_spectral_kind kind,
                                 const fp_array *s, const fp_array *z, const fp_array *a, double dgamma,
                                 const fp_array *dparams, fp_array *b, fp_array *r) {
    return guard([&] { return fp::spectral_jvp_impl(op, ws, kind, s, z, a, dgamma, dparams, b, r); });
}

FP_API int fp_device_available(fp_device dev) {
    const int d = fp::enum_bits(dev);
    if (d == FP_DEVICE_CPU) return 1;
    return d == FP_DEVICE_CUDA && fp::cuda_devices() > 0 ? 1 : 0;
}

FP_API fp_status fp_op_device_check(const fp_op *op, fp_device dev, int *row_parallel_only) {
    if (!op) return FP_EINVAL;
    int rpo = 0;
    const int d = fp::enum_bits(dev);
    if (d == FP_DEVICE_CPU) {
        rpo = op->splits_rows() ? 0 : 1;
    } else {
        if (d != FP_DEVICE_CUDA || fp::cuda_devices() == 0) return FP_EINVAL;
        if (fp_status s = fp::cuda_plugin()->op_check(op, &rpo)) return s;
    }
    if (row_parallel_only) *row_parallel_only = rpo;
    return FP_OK;
}

FP_API fp_status fp_workspace_set_stream(fp_workspace *ws, void *stream) {
    if (!ws || ws->device == FP_DEVICE_CPU) return FP_EINVAL;
    ws->stream = stream;
    return FP_OK;
}

FP_API fp_status fp_workspace_get_stream(const fp_workspace *ws, void **stream) {
    if (!ws || !stream || ws->device == FP_DEVICE_CPU) return FP_EINVAL;
    *stream = ws->stream;
    return FP_OK;
}

FP_API fp_status fp_workspace_info(const fp_workspace *ws, fp_info *info) {
    if (!ws || !info || !ws->info_valid) return FP_EINVAL;
    *info = ws->info;
    return ws->info_status;
}

FP_API fp_status fp_workspace_refine(const fp_op *op, fp_workspace *ws, double tol, int max_iter,
                                     fp_array *out, fp_info *info) {
    return guard([&] { return fp::refine_impl(op, ws, tol, max_iter, out, info); });
}

FP_API fp_status fp_workspace_deriv_exact(const fp_workspace *ws, int *exact, double *smoothing) {
    return guard([&] { return fp::deriv_exact_impl(ws, exact, smoothing); });
}

FP_API fp_status fp_workspace_image_info(const fp_workspace *ws, double *gaps, int *iters,
                                         double *zerrs, int64_t count) {
    if (!ws || !ws->info_valid || !ws->image_ndim) return FP_EINVAL;
    int64_t images = 1;  // the batch shape's size: x's shape without the image axes
    for (int d = 0; d < ws->ndim - ws->image_ndim; ++d) images *= ws->shape[d];
    if (count != images) return FP_ESHAPE;
    const bool held = ws->numel != 0;  // an empty batch keeps no per-image results
    const int *its = reinterpret_cast<const int *>(ws->image_iters.data());
    for (int64_t r = 0; r < count; ++r) {
        if (gaps) gaps[r] = held ? ws->image_gaps.data()[r] : 0.0;
        if (iters) iters[r] = held ? its[r] : 0;
        if (zerrs) zerrs[r] = held ? ws->image_zerr.data()[r] : 0.0;
    }
    return FP_OK;
}

FP_API fp_status fp_vjp_device(const fp_op *op, const fp_workspace *ws, const fp_array *g, fp_array *gx,
                               fp_array *ggamma, fp_array *gparams) {
    return guard([&] { return fp::vjp_device_impl(op, ws, g, gx, ggamma, gparams); });
}

FP_API const char *fp_strerror(fp_status s) {
    switch (s) {
        case FP_OK: return "success";
        case FP_EINVAL: return "invalid argument";
        case FP_EDTYPE: return "unsupported or mismatched dtype";
        case FP_ESHAPE: return "mismatched shape or parameter length";
        case FP_ENOMEM: return "out of memory";
        case FP_ENOTCONV: return "iteration stopped before reaching the tolerance";
        case FP_EPARAM: return "invalid operator parameter or step";
        case FP_EINTERNAL: return "internal error";
        case FP_EDEVICE: return "device error";
    }
    return "unknown status";
}

FP_API int fp_abi_version(void) { return FP_ABI_VERSION; }

FP_API const char *fp_version_string(void) { return FASTPROX_VERSION_STRING; }

FP_API int fp_default_threads(void) { return fp::default_threads(); }

}  // extern "C"
