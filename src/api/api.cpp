// The extern "C" boundary: argument checks, dtype dispatch, and the try/catch
// wrappers that turn every C++ exception into an fp_status.
//
// SPDX-License-Identifier: MIT
#include "fastprox.h"

#include "core/driver.hpp"
#include "core/layout.hpp"
#include "core/op.hpp"
#include "core/workspace.hpp"
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

// Converts and validates a public descriptor. Scalars become shape [1].
fp_status to_arr(const fp_array *a, Arr &out) noexcept {
    if (!a) return FP_EINVAL;
    if (a->device != FP_DEVICE_CPU) return FP_EINVAL;
    if (a->dtype != FP_F32 && a->dtype != FP_F64) return FP_EDTYPE;
    if (a->ndim < 0 || a->ndim > FP_MAX_NDIM) return FP_ESHAPE;
    out.data = static_cast<char *>(a->data);
    out.dtype = a->dtype;
    if (a->ndim == 0) {
        out.ndim = 1;
        out.shape[0] = 1;
        out.strides[0] = 1;
    } else {
        if (!a->shape) return FP_EINVAL;
        out.ndim = a->ndim;
        for (int d = 0; d < a->ndim; ++d) {
            if (a->shape[d] < 0) return FP_ESHAPE;
            out.shape[d] = a->shape[d];
            out.strides[d] = a->strides ? a->strides[d] : 0;
        }
        if (!a->strides) {  // NULL strides: C-contiguous
            int64_t s = 1;
            for (int d = a->ndim - 1; d >= 0; --d) {
                out.strides[d] = s;
                s *= out.shape[d];
            }
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
    for (int64_t i = 0; i < m; ++i) {
        const int64_t o = i * x.strides[0];
        v[static_cast<size_t>(i)] = x.dtype == FP_F32 ? reinterpret_cast<const float *>(x.data)[o]
                                                      : reinterpret_cast<const double *>(x.data)[o];
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
    if (!elem && w.numel > 0) {
        w.thread_bytes = row_thread_bytes(op, w.n, item);
        w.scratch.resize(w.thread_bytes * static_cast<size_t>(w.nthreads) / sizeof(double) + 1);
    }
    if (keep) {
        const size_t bytes = static_cast<size_t>(w.numel) * item;
        w.xsave.resize(bytes / sizeof(double) + 1);
        if (!elem) {
            w.zsave.resize(bytes / sizeof(double) + 1);
            w.code.resize(static_cast<size_t>(w.numel) + 1);
        }
        const int64_t P = op.nparams();
        const int64_t units = elem ? ceil_div(w.numel, kChunk) : w.rows;
        w.nblocks = reduction_blocks(units, P);
        w.acc.resize(static_cast<size_t>(w.nblocks * (P + 2)));
        w.tot.resize(static_cast<size_t>(P + 2));
        w.dparams.resize(static_cast<size_t>(P) + 1);
    }
    return FP_OK;
}

int resolve_threads(const fp_options &o) noexcept {
    return o.threads > 0 ? o.threads : max_threads();
}

fp_status prox_impl(const fp_op *op, double step, const fp_array *xa, fp_array *oa,
                    const fp_options *opt, fp_workspace *ws, fp_info *info, bool conj) {
    if (!op || !xa || !oa) return FP_EINVAL;
    if (!finite(step) || step < 0) return FP_EPARAM;
    const fp_options o = opt ? *opt : defaults();
    Arr x, out;
    if (fp_status s = to_arr(xa, x)) return s;
    if (fp_status s = to_arr(oa, out)) return s;
    if (x.dtype != out.dtype) return FP_EDTYPE;
    if (!same_shape(x, out)) return FP_ESHAPE;
    int axis;
    if (norm_axis(o.axis, x.ndim, axis)) return FP_EINVAL;
    const bool keep = o.keep_structure != 0;
    if (keep && !ws) return FP_EINVAL;

    std::unique_ptr<fp_workspace> local;
    int nt = resolve_threads(o);
    if (ws) {
        if (ws->op != op || ws->dtype != x.dtype || ws->axis != axis || ws->ndim != x.ndim)
            return FP_EINVAL;
        for (int d = 0; d < x.ndim; ++d)
            if (ws->shape[d] != x.shape[d]) return FP_ESHAPE;
        if (keep && !ws->keep) return FP_EINVAL;
        ws->valid = false;
        nt = std::min(nt, ws->nthreads);
    }

    if (info) {
        info->gap = 0.0;
        info->iters = 0;
        info->ties = 0;
    }
    if (x.numel() == 0) return FP_OK;
    if (fp_status s = op->check_n(x.shape[axis])) return s;

    // sigma = 0 in the conjugate: prox_{0 g*} is the identity.
    if (conj && step == 0.0) {
        Arr src = x;
        const int64_t *st[2] = {x.strides, out.strides};
        const ElemPlan p = make_elem_plan(x.ndim, x.shape, axis, false, 2, st);
        const size_t item = x.itemsize();
        p.for_runs(0, p.numel, [&](const int64_t *off, int64_t len, int64_t, int64_t, int64_t) {
            for (int64_t i = 0; i < len; ++i)
                std::memmove(out.data + (off[1] + i * p.stride[1][p.ndim - 1]) * item,
                             src.data + (off[0] + i * p.stride[0][p.ndim - 1]) * item, item);
        });
        return FP_OK;
    }

    if (!ws) {
        if (fp_status s = build_workspace(*op, x, axis, nt, false, local)) return s;
        ws = local.get();
    }
    const double gamma = conj ? 1.0 / step : step;
    const double sigma = conj ? step : 0.0;
    fp_status s = x.dtype == FP_F32
                      ? prox_driver<float>(*op, x, out, axis, gamma, conj, sigma, *ws, nt, keep)
                      : prox_driver<double>(*op, x, out, axis, gamma, conj, sigma, *ws, nt, keep);
    if (s == FP_OK && keep) {
        ws->valid = true;
        ws->conj = conj;
        ws->gamma = gamma;
        ws->sigma = sigma;
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
    const int64_t P = op->nparams();
    const double *dp = nullptr;
    if (dpa && P > 0) {
        if (fp_status s = read_exact(dpa, P, ws->dparams.data())) return s;
        dp = ws->dparams.data();
    }
    if (ws->numel == 0) return FP_OK;
    const Arr *dxp = dxa ? &dx : nullptr;
    return ws->dtype == FP_F32 ? jvp_driver<float>(*op, *ws, dxp, dgamma, dp, out)
                               : jvp_driver<double>(*op, *ws, dxp, dgamma, dp, out);
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
    const int64_t P = op->nparams();
    if (gpa) {  // validate before any work
        Arr t;
        if (fp_status s = to_arr(gpa, t)) return s;
        if (gpa->ndim > 1 || t.numel() != P) return FP_ESHAPE;
    }
    double *tot = ws->tot.data();
    for (int64_t k = 0; k < P + 2; ++k) tot[k] = 0.0;
    if (ws->numel > 0) {
        fp_status s = ws->dtype == FP_F32 ? vjp_driver<float>(*op, *ws, g, gx, tot)
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
        return fp::make_tv1d(std::move(w), method, out);
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
        if (x.numel() > 0)
            if (fp_status s = op->check_n(x.shape[axis])) return s;
        std::unique_ptr<fp_workspace> ws;
        if (fp_status s = fp::build_workspace(*op, x, axis, fp::resolve_threads(o),
                                              o.keep_structure != 0, ws))
            return s;
        *out = ws.release();
        return FP_OK;
    });
}

FP_API void fp_workspace_free(fp_workspace *ws) { delete ws; }

FP_API fp_status fp_prox(const fp_op *op, double gamma, const fp_array *x, fp_array *out,
                         const fp_options *opt, fp_workspace *ws, fp_info *info) {
    return guard([&] { return fp::prox_impl(op, gamma, x, out, opt, ws, info, false); });
}

FP_API fp_status fp_prox_conj(const fp_op *op, double sigma, const fp_array *y, fp_array *out,
                              const fp_options *opt, fp_workspace *ws, fp_info *info) {
    return guard([&] { return fp::prox_impl(op, sigma, y, out, opt, ws, info, true); });
}

FP_API fp_status fp_jvp(const fp_op *op, const fp_workspace *ws, const fp_array *dx,
                        double dgamma, const fp_array *dparams, fp_array *dout) {
    return guard([&] { return fp::jvp_impl(op, ws, dx, dgamma, dparams, dout); });
}

FP_API fp_status fp_vjp(const fp_op *op, const fp_workspace *ws, const fp_array *g, fp_array *gx,
                        double *ggamma, fp_array *gparams) {
    return guard([&] { return fp::vjp_impl(op, ws, g, gx, ggamma, gparams); });
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
    }
    return "unknown status";
}

FP_API int fp_abi_version(void) { return FP_ABI_VERSION; }

FP_API const char *fp_version_string(void) { return FASTPROX_VERSION_STRING; }

FP_API int fp_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

}  // extern "C"
