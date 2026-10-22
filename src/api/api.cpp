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
#include <memory>
#include <new>
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

// Converts and validates a public descriptor. Scalars become shape [1].
// int8: true for the kink and branch arrays, the only ones of that dtype.
fp_status to_arr(const fp_array *a, Arr &out, bool int8 = false) noexcept {
    if (!a) return FP_EINVAL;
    if (enum_bits(a->device) != FP_DEVICE_CPU) return FP_EINVAL;
    const int dt = enum_bits(a->dtype);
    if (int8 ? dt != FP_I8 : (dt != FP_F32 && dt != FP_F64)) return FP_EDTYPE;
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
        if (w.numel == 0) return FP_OK;
        const int64_t B = w.rows;
        w.image_sets = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(B, w.nthreads)));
        w.image_bytes = image_staging_bytes(N) + round64(op.image_bytes(ishape));
        w.image_thread_bytes = round64(op.image_thread_bytes(ishape));
        w.image_work.resize(mul_size(w.image_bytes, static_cast<size_t>(w.image_sets)) / sizeof(double) + 1);
        w.image_thread.resize(mul_size(w.image_thread_bytes, static_cast<size_t>(w.nthreads)) / sizeof(double) + 1);
        w.dual.resize(static_cast<size_t>(B) * static_cast<size_t>(k) * static_cast<size_t>(N));
        w.image_gaps.resize(static_cast<size_t>(B));
        w.image_iters.resize(static_cast<size_t>(B));
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
        }
        return FP_OK;
    }
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
        w.tot.resize(static_cast<size_t>(P + 2));
        w.dparams.resize(static_cast<size_t>(P) + 1);
        w.steps.resize(static_cast<size_t>(w.rows) + 1);
    }
    return FP_OK;
}

int resolve_threads(const fp_options &o) noexcept {
    return o.threads > 0 ? o.threads : default_threads();
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

fp_status prox_impl(const fp_op *op, double step, const fp_array *xa, fp_array *oa,
                    const fp_options *opt, fp_workspace *ws, fp_info *info, bool conj,
                    const fp_array *sa = nullptr) {
    if (!op || !xa || !oa) return FP_EINVAL;
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
    if (norm_axis(o.axis, x.ndim, axis)) return FP_EINVAL;
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
        ws->valid = false;
        ws->steps_on = false;
        ws->branch_on = false;  // a branch override belongs to the point it was set for
        ws->override_on = false;
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
    if (x.numel() == 0) return FP_OK;
    if (k) {
        if (fp_status s = op->check_image(x.shape + (x.ndim - k))) return s;
    } else if (fp_status s = op->check_n(x.shape[axis])) {
        return s;
    }

    // sigma = 0 in the conjugate: prox_{0 g*} is the identity.
    if (conj && step == 0.0) {
        Arr src = x;
        const int64_t *st[2] = {x.strides, out.strides};
        const ElemPlan p = make_elem_plan(x.ndim, x.shape, axis, false, 2, st);
        // Byte offsets stay signed: strides may be negative.
        const int64_t item = static_cast<int64_t>(x.itemsize());
        p.for_runs(0, p.numel, [&](const int64_t *off, int64_t len, int64_t, int64_t, int64_t) {
            for (int64_t i = 0; i < len; ++i)
                std::memmove(out.data + (off[1] + i * p.stride[1][p.ndim - 1]) * item,
                             src.data + (off[0] + i * p.stride[0][p.ndim - 1]) * item,
                             static_cast<size_t>(item));
        });
        return FP_OK;
    }

    if (!ws) {
        if (fp_status s = build_workspace(*op, x, axis, nt, false, local)) return s;
        ws = local.get();
    }
    const double gamma = conj ? 1.0 / step : step;
    const double sigma = conj ? step : 0.0;
    const Steps *sp = sa ? &steps : nullptr;
    if (sp) {
        // Every step before any work; a kept call saves them for the derivatives.
        for (int64_t r = 0; r < ws->rows; ++r) {
            const double v = steps.at(r);
            if (!finite(v) || v < 0) return FP_EPARAM;
            if (fp_status s = op->check_step(v)) return s;
        }
        if (keep)
            for (int64_t r = 0; r < ws->rows; ++r) ws->steps.data()[r] = steps.at(r);
    } else if (fp_status s = op->check_step(gamma)) {
        return s;
    }
    if (k) {
        fp_status s = x.dtype == FP_F32
                          ? image_prox_driver<float>(*op, x, out, gamma, conj, sigma, *ws, nt, keep,
                                                     o.tol, o.max_iter, info, sp)
                          : image_prox_driver<double>(*op, x, out, gamma, conj, sigma, *ws, nt, keep,
                                                      o.tol, o.max_iter, info, sp);
        if ((s == FP_OK || s == FP_ENOTCONV) && keep) {
            ws->valid = true;
            ws->conj = conj;
            ws->gamma = gamma;
            ws->sigma = sigma;
            ws->steps_on = sp != nullptr;
        }
        return s;
    }
    int64_t ties = 0;
    fp_status s = x.dtype == FP_F32
                      ? prox_driver<float>(*op, x, out, axis, gamma, conj, sigma, *ws, nt, keep, o.tie, &ties, sp)
                      : prox_driver<double>(*op, x, out, axis, gamma, conj, sigma, *ws, nt, keep, o.tie, &ties, sp);
    if (info) info->ties = ties;
    if (s == FP_OK && keep) {
        ws->valid = true;
        ws->conj = conj;
        ws->gamma = gamma;
        ws->sigma = sigma;
        ws->tie = o.tie;
        ws->steps_on = sp != nullptr;
    }
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
    if (ws->numel == 0) return FP_OK;
    const Arr *dxp = dxa ? &dx : nullptr;
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
    const size_t need = spectral_doubles(ws->n) * static_cast<size_t>(ws->nthreads);
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
    if (ws->numel > 0) {
        fp_status s;
        if (op->image_ndim())
            s = ws->dtype == FP_F32 ? image_vjp_driver<float>(*op, *ws, g, gx, tot)
                                    : image_vjp_driver<double>(*op, *ws, g, gx, tot);
        else
            s = ws->dtype == FP_F32 ? vjp_driver<float>(*op, *ws, g, gx, tot)
                                    : vjp_driver<double>(*op, *ws, g, gx, tot);
        if (s) return s;
    }
    if (ws->conj) {
        tot[P] = tot[P + 1] + tot[P] / ws->sigma;
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
    if (fp_status s = to_arr(aa, *a, true)) return s;
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
    if (ws->numel == 0) c = 0;
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
    if (!ws.override_on) std::memcpy(ws.code_default.data(), ws.code.data(), static_cast<size_t>(ws.numel));
    else std::memcpy(ws.code.data(), ws.code_default.data(), static_cast<size_t>(ws.numel));
    if (!side) return;
    std::vector<double> xb(static_cast<size_t>(n)), zb(static_cast<size_t>(n));
    std::vector<std::max_align_t> scr(op.scratch_bytes(n) / sizeof(std::max_align_t) + 1);
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
    if (!ws->op->elementwise()) return set_structure_branch(*ws, a, tol);
    // Into the workspace in C order, the order of the saved entries.
    const int64_t *st[1] = {a.strides};
    const ElemPlan p = make_elem_plan(a.ndim, a.shape, ws->axis, false, 1, st);
    const int8_t *src = reinterpret_cast<const int8_t *>(a.data);
    int8_t *dst = ws->branch.data();
    const int64_t s0 = p.stride[0][p.ndim - 1];
    bool ok = true;
    p.for_runs(0, p.numel, [&](const int64_t *off, int64_t len, int64_t pos, int64_t, int64_t) {
        for (int64_t i = 0; i < len; ++i) {
            const int8_t v = src[off[0] + i * s0];
            ok = ok && v >= -1 && v <= 1;
            dst[pos + i] = v;
        }
    });
    if (!ok) {
        ws->branch_on = false;
        return FP_EPARAM;
    }
    ws->branch_on = true;
    ws->branch_tol = tol;
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
            if (to_ws)
                p.for_each(0, [&](int64_t i, int64_t o) { dual[i] = static_cast<double>(img[o]); });
            else
                p.for_each(0, [&](int64_t i, int64_t o) { img[o] = static_cast<T>(dual[i]); });
        }
    }
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
    if (to_ws) {  // a seed must be finite
        bool ok = true;
        if (a.dtype == FP_F32) dual_copy<float>(*ws, a, true);
        else dual_copy<double>(*ws, a, true);
        const double *d = ws->dual.data();
        const size_t m = ws->dual.size();
        for (size_t i = 0; i < m; ++i) ok = ok && std::isfinite(d[i]);
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
        if (n < 1) return FP_EPARAM;
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
            if (n < 1) return FP_EPARAM;
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
        std::vector<int64_t> ptr(1, 0), idx;
        if (ngroups > 0) {
            ptr.assign(group_ptr, group_ptr + ngroups + 1);
            if (ptr[0] != 0 || ptr.back() < 0) return FP_EPARAM;
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
        if (fp::norm_axis(o.axis, x.ndim, axis)) return FP_EINVAL;
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
