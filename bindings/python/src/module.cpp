// nanobind extension: a thin wrapper over the public C API, with no numerical
// code. The Python package in ../fastprox builds the user-facing interface on
// top of these functions.
//
// SPDX-License-Identifier: MIT
#include "fastprox.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using InArr = nb::ndarray<nb::ro, nb::device::cpu>;
using OutArr = nb::ndarray<nb::device::cpu>;
using Params = nb::ndarray<nb::ro, const double, nb::ndim<1>, nb::device::cpu>;
using NdParams = nb::ndarray<nb::ro, const double, nb::device::cpu>;  // e.g. pixel weights
using IdArr = nb::ndarray<nb::ro, const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Raises for an error status; what names the C entry point (audit P7), as
// in "fp_prox: invalid argument".
void check(fp_status s, const char *what = nullptr) {
    if (s == FP_OK) return;
    const std::string text = what ? std::string(what) + ": " + fp_strerror(s) : std::string(fp_strerror(s));
    const char *msg = text.c_str();
    switch (s) {
        case FP_ENOMEM: throw std::bad_alloc();
        case FP_EDTYPE: throw nb::type_error(msg);
        case FP_EINVAL:
        case FP_EPARAM:
        case FP_ESHAPE: throw nb::value_error(msg);
        default: throw std::runtime_error(msg);
    }
}

template <class A>
fp_array view(const A &a) {
    fp_array v;
    v.data = const_cast<void *>(static_cast<const void *>(a.data()));
    const nb::dlpack::dtype dt = a.dtype();
    if (dt == nb::dtype<double>()) v.dtype = FP_F64;
    else if (dt == nb::dtype<float>()) v.dtype = FP_F32;
    else throw nb::type_error("fastprox operates on float32 and float64 arrays");
    v.ndim = static_cast<int>(a.ndim());
    v.shape = a.shape_ptr();
    v.strides = a.stride_ptr();
    v.device = FP_DEVICE_CPU;
    return v;
}

using I8In = nb::ndarray<nb::ro, const int8_t, nb::device::cpu>;
using I8Out = nb::ndarray<int8_t, nb::device::cpu>;
using F64Out = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I32Out = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64Vec = nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I8Vec = nb::ndarray<int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// An int32 or int64 label array as an fp_array (since 1.0.2).
template <class A>
fp_array view_label(const A &a) {
    fp_array v;
    v.data = const_cast<void *>(static_cast<const void *>(a.data()));
    const nb::dlpack::dtype dt = a.dtype();
    if (dt == nb::dtype<int32_t>()) v.dtype = FP_I32;
    else if (dt == nb::dtype<int64_t>()) v.dtype = FP_I64;
    else throw nb::type_error("labels are int32 or int64");
    v.ndim = static_cast<int>(a.ndim());
    v.shape = a.shape_ptr();
    v.strides = a.stride_ptr();
    v.device = FP_DEVICE_CPU;
    return v;
}

// An int8 kink or branch array as an fp_array.
template <class A>
fp_array view_i8(const A &a) {
    fp_array v;
    v.data = const_cast<void *>(static_cast<const void *>(a.data()));
    v.dtype = FP_I8;
    v.ndim = static_cast<int>(a.ndim());
    v.shape = a.shape_ptr();
    v.strides = a.stride_ptr();
    v.device = FP_DEVICE_CPU;
    return v;
}

struct Op {
    fp_op *p = nullptr;
    Op() = default;
    Op(const Op &) = delete;
    Op &operator=(const Op &) = delete;
    ~Op() { fp_op_free(p); }
};

struct Workspace {
    fp_workspace *p = nullptr;
    Workspace() = default;
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    ~Workspace() { fp_workspace_free(p); }
};

// Optional 1-D float64 parameter array as an fp_array (or nullptr).
struct ParamView {
    fp_array a{};
    bool set = false;
    explicit ParamView(const std::optional<Params> &p) {
        if (p) {
            a = view(*p);
            set = true;
        }
    }
    const fp_array *get() const { return set ? &a : nullptr; }
};

template <class F>
Op *make(F &&f) {
    auto op = std::make_unique<Op>();
    fp_status s;
    {  // constructors are data-sized (graph TV builds its graph): without the GIL (audit P4)
        nb::gil_scoped_release release;
        s = f(&op->p);
    }
    check(s, "operator constructor");
    return op.release();
}

fp_options options(int axis, int threads, bool keep, double tol = 1e-6, int max_iter = 1000,
                   int tie = FP_TIE_NONZERO) {
    fp_options o;
    fp_options_init(&o);
    o.axis = axis;
    o.threads = threads;
    o.keep_structure = keep ? 1 : 0;
    o.tol = tol;
    o.max_iter = max_iter;
    o.tie = tie;
    return o;
}

// (gap, iters, ties, converged). FP_ENOTCONV is not an error here: the last
// iterate is written and converged is false.
std::tuple<double, int, int64_t, bool> prox(const Op &op, double step, const InArr &x, OutArr &out,
                                            int axis, int threads, bool keep, Workspace *ws, bool conj,
                                            double tol, int max_iter, int tie,
                                            const std::optional<InArr> &steps) {
    const fp_array xa = view(x);
    fp_array oa = view(out);
    const fp_options o = options(axis, threads, keep, tol, max_iter, tie);
    fp_info info{0.0, 0, 0};
    fp_status s;
    if (steps) {  // one step per row or image: fp_prox_steps, info from the workspace
        const fp_array sa = view(*steps);
        {
            nb::gil_scoped_release release;
            s = fp_prox_steps(op.p, &sa, &xa, &oa, &o, ws ? ws->p : nullptr);
        }
        if (ws && (s == FP_OK || s == FP_ENOTCONV)) fp_workspace_info(ws->p, &info);
    } else {
        nb::gil_scoped_release release;
        s = conj ? fp_prox_conj(op.p, step, &xa, &oa, &o, ws ? ws->p : nullptr, &info)
                 : fp_prox(op.p, step, &xa, &oa, &o, ws ? ws->p : nullptr, &info);
    }
    if (s != FP_ENOTCONV) check(s, steps ? "fp_prox_steps" : conj ? "fp_prox_conj" : "fp_prox");
    return {info.gap, info.iters, info.ties, s == FP_OK};
}

Workspace *workspace(const Op &op, const InArr &like, int axis, int threads, bool keep) {
    // tol and max_iter are per call; the workspace does not depend on them.
    auto ws = std::make_unique<Workspace>();
    const fp_array la = view(like);
    const fp_options o = options(axis, threads, keep);
    fp_status s;
    {
        nb::gil_scoped_release release;
        s = fp_workspace_create(op.p, &la, &o, &ws->p);
    }
    check(s, "fp_workspace_create");
    return ws.release();
}

void jvp(const Op &op, const Workspace &ws, const std::optional<InArr> &dx, double dgamma,
         const std::optional<Params> &dparams, OutArr &dout) {
    fp_array da{};
    if (dx) da = view(*dx);
    const ParamView dp(dparams);
    fp_array oa = view(dout);
    fp_status s;
    {
        nb::gil_scoped_release release;
        s = fp_jvp(op.p, ws.p, dx ? &da : nullptr, dgamma, dp.get(), &oa);
    }
    check(s, "fp_jvp");
}

double vjp(const Op &op, const Workspace &ws, const InArr &g, OutArr &gx,
           std::optional<OutArr> gparams, bool want_gamma) {
    const fp_array ga = view(g);
    fp_array xa = view(gx);
    fp_array pa{};
    if (gparams) pa = view(*gparams);
    double gg = 0.0;
    fp_status s;
    {
        nb::gil_scoped_release release;
        s = fp_vjp(op.p, ws.p, &ga, &xa, want_gamma ? &gg : nullptr, gparams ? &pa : nullptr);
    }
    check(s, "fp_vjp");
    return gg;
}

// B (and r for the SVD kind, if given) from A at the spectrum saved in ws.
void spectral_jvp(const Op &op, const Workspace &ws, int kind, const InArr &s, const InArr &z,
                  const InArr &a, double dgamma, const std::optional<Params> &dparams, OutArr &b,
                  std::optional<OutArr> r) {
    const fp_array sa = view(s), za = view(z), aa = view(a);
    const ParamView dp(dparams);
    fp_array ba = view(b), ra{};
    if (r) ra = view(*r);
    fp_status st;
    {
        nb::gil_scoped_release release;
        st = fp_spectral_jvp(op.p, ws.p, static_cast<fp_spectral_kind>(kind), &sa, &za, &aa, dgamma, dp.get(),
                             &ba, r ? &ra : nullptr);
    }
    check(st, "fp_spectral_jvp");
}

}  // namespace

NB_MODULE(_fastprox, m) {
    m.doc() = "Low-level bindings to the fastprox C API.";

    nb::class_<Op>(m, "Op")
        .def_prop_ro("nparams", [](const Op &o) { return fp_op_nparams(o.p); })
        .def_prop_ro("name", [](const Op &o) { return fp_op_name(o.p); });
    nb::class_<Workspace>(m, "Workspace");

    m.def("soft", [](double lam, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_soft(lam, pw.get(), o); });
    }, "lam"_a, "weights"_a.none());
    m.def("elastic_net", [](double l1, double l2) {
        return make([&](fp_op **o) { return fp_op_elastic_net(l1, l2, o); });
    }, "l1"_a, "l2"_a);
    m.def("huber", [](double lam, double delta) {
        return make([&](fp_op **o) { return fp_op_huber(lam, delta, o); });
    }, "lam"_a, "delta"_a);
    m.def("group_l2", [](double lam, std::optional<IdArr> groups, int64_t ngroups,
                         std::optional<Params> c) {
        const ParamView pc(c);
        const int64_t n = groups ? static_cast<int64_t>(groups->shape(0)) : 0;
        const int64_t *g = groups ? groups->data() : nullptr;
        return make([&](fp_op **o) { return fp_op_group_l2(lam, n, g, ngroups, pc.get(), o); });
    }, "lam"_a, "groups"_a.none(), "ngroups"_a, "group_weights"_a.none());
    m.def("tree_l2", [](double l1, IdArr ptr, IdArr idx, std::optional<Params> eta) {
        const ParamView pe(eta);
        const int64_t G = static_cast<int64_t>(ptr.shape(0)) - 1;
        return make([&](fp_op **o) {
            return fp_op_tree_l2(l1, G, ptr.data(), idx.data(), pe.get(), o);
        });
    }, "l1"_a, "ptr"_a, "idx"_a, "group_weights"_a.none());
    m.def("box", [](double lo, double hi, std::optional<Params> lower,
                    std::optional<Params> upper) {
        const ParamView pl(lower), pu(upper);
        return make([&](fp_op **o) { return fp_op_box(lo, hi, pl.get(), pu.get(), o); });
    }, "lo"_a, "hi"_a, "lower"_a.none(), "upper"_a.none());
    m.def("nonneg", [] { return make([](fp_op **o) { return fp_op_nonneg(o); }); });
    m.def("log_barrier", [](double lam) {
        return make([&](fp_op **o) { return fp_op_log_barrier(lam, o); });
    }, "lam"_a);
    m.def("l1_box", [](double lam, std::optional<Params> w, std::optional<Params> lower,
                       std::optional<Params> upper) {
        const ParamView pw(w), pl(lower), pu(upper);
        return make([&](fp_op **o) {
            return fp_op_l1_box(lam, pw.get(), -INFINITY, INFINITY, pl.get(), pu.get(), o);
        });
    }, "lam"_a, "weights"_a.none(), "lower"_a.none(), "upper"_a.none());
    m.def("soc", [] { return make([](fp_op **o) { return fp_op_soc(o); }); });
    m.def("l2_ball", [](double r) {
        return make([&](fp_op **o) { return fp_op_l2_ball(r, o); });
    }, "radius"_a);
    m.def("simplex", [](double r) {
        return make([&](fp_op **o) { return fp_op_simplex(r, o); });
    }, "radius"_a);
    m.def("l1_ball", [](double r, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_l1_ball(r, pw.get(), o); });
    }, "radius"_a, "weights"_a.none());
    m.def("linf", [](double lam) {
        return make([&](fp_op **o) { return fp_op_linf(lam, o); });
    }, "lam"_a);
    m.def("max", [](double lam) {
        return make([&](fp_op **o) { return fp_op_max(lam, o); });
    }, "lam"_a);
    m.def("sum_largest", [](double lam, int64_t k, bool absolute) {
        return make([&](fp_op **o) { return fp_op_sum_largest(lam, k, absolute, o); });
    }, "lam"_a, "k"_a, "absolute"_a);
    m.def("hyperplane", [](std::optional<Params> a, double b, std::optional<Params> lower,
                           std::optional<Params> upper, bool inequality) {
        const ParamView pa(a), pl(lower), pu(upper);
        return make([&](fp_op **o) {
            return fp_op_hyperplane(pa.get(), b, -INFINITY, INFINITY, pl.get(), pu.get(),
                                    inequality, o);
        });
    }, "a"_a.none(), "b"_a, "lower"_a.none(), "upper"_a.none(), "inequality"_a);
    m.def("isotonic", [](std::optional<Params> w, bool increasing) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_isotonic(pw.get(), increasing, o); });
    }, "weights"_a.none(), "increasing"_a);
    m.def("sorted_l1", [](Params w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_sorted_l1(pw.get(), o); });
    }, "weights"_a);
    m.def("tv1d_ball", [](double alpha, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_tv1d_ball(alpha, pw.get(), o); });
    }, "alpha"_a, "weights"_a.none());
    m.def("tv1d_l2", [](std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_tv1d_l2(pw.get(), o); });
    }, "weights"_a.none());
    m.def("tv1d", [](std::optional<Params> w, int method) {
        const ParamView pw(w);
        return make([&](fp_op **o) {
            return fp_op_tv1d(pw.get(), static_cast<fp_tv1d_method>(method), o);
        });
    }, "weights"_a.none(), "method"_a);

    m.def("l0", [](double lam, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_l0(lam, pw.get(), o); });
    }, "lam"_a, "weights"_a.none());
    m.def("lq", [](double lam, double q, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_lq(lam, q, pw.get(), o); });
    }, "lam"_a, "q"_a, "weights"_a.none());
    m.def("scad", [](double lam, double a, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_scad(lam, a, pw.get(), o); });
    }, "lam"_a, "a"_a, "weights"_a.none());
    m.def("mcp", [](double lam, double b, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_mcp(lam, b, pw.get(), o); });
    }, "lam"_a, "b"_a, "weights"_a.none());
    m.def("log_penalty", [](double lam, double eps) {
        return make([&](fp_op **o) { return fp_op_log_penalty(lam, eps, o); });
    }, "lam"_a, "eps"_a);
    // Group penalties: kind 0 Huber (p = delta), 1 SCAD (p = a), 2 MCP (p = b).
    m.def("group_radial", [](int kind, double lam, double p, std::optional<IdArr> groups,
                             int64_t ngroups, std::optional<Params> c) {
        const ParamView pc(c);
        const int64_t n = groups ? static_cast<int64_t>(groups->shape(0)) : 0;
        const int64_t *g = groups ? groups->data() : nullptr;
        return make([&](fp_op **o) {
            if (kind == 0) return fp_op_group_huber(lam, p, n, g, ngroups, pc.get(), o);
            if (kind == 1) return fp_op_group_scad(lam, p, n, g, ngroups, pc.get(), o);
            return fp_op_group_mcp(lam, p, n, g, ngroups, pc.get(), o);
        });
    }, "kind"_a, "lam"_a, "p"_a, "groups"_a.none(), "ngroups"_a, "group_weights"_a.none());
    m.def("tv_aniso", [](int ndim, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_tv_aniso(ndim, pw.get(), o); });
    }, "ndim"_a, "weights"_a.none());

    m.def("tv_iso", [](int ndim, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_tv_iso(ndim, pw.get(), o); });
    }, "ndim"_a, "weights"_a.none());
    m.def("tv_aniso_weighted", [](int ndim, std::optional<Params> w, NdParams beta) {
        const ParamView pw(w);
        const fp_array pb = view(beta);
        return make([&](fp_op **o) { return fp_op_tv_aniso_weighted(ndim, pw.get(), &pb, o); });
    }, "ndim"_a, "weights"_a.none(), "edge_weights"_a);
    m.def("tv_iso_weighted", [](int ndim, std::optional<Params> w, NdParams alpha) {
        const ParamView pw(w);
        const fp_array pa = view(alpha);
        return make([&](fp_op **o) { return fp_op_tv_iso_weighted(ndim, pw.get(), &pa, o); });
    }, "ndim"_a, "weights"_a.none(), "pixel_weights"_a);
    m.def("graph_tv", [](int64_t n, IdArr edges, std::optional<Params> w) {
        const ParamView pw(w);
        const int64_t m = static_cast<int64_t>(edges.shape(0)) / 2;
        return make([&](fp_op **o) { return fp_op_graph_tv(n, m, edges.data(), pw.get(), o); });
    }, "n"_a, "edges"_a, "weights"_a.none());
    m.def("tree_tv", [](int64_t n, IdArr edges, std::optional<Params> w) {
        const ParamView pw(w);
        const int64_t m = static_cast<int64_t>(edges.shape(0)) / 2;
        return make([&](fp_op **o) { return fp_op_tree_tv(n, m, edges.data(), pw.get(), o); });
    }, "n"_a, "edges"_a, "weights"_a.none());
    m.def("set_deriv_tol", [](Workspace &ws, double tol, int max_iter) {
        check(fp_workspace_set_deriv_tol(ws.p, tol, max_iter), "fp_workspace_set_deriv_tol");
    }, "ws"_a, "tol"_a, "max_iter"_a);
    // (exact, smoothing) of the kept point's derivative (feedback U4).
    m.def("deriv_exact", [](const Workspace &ws) {
        int exact = 0;
        double mu = 0.0;
        check(fp_workspace_deriv_exact(ws.p, &exact, &mu), "fp_workspace_deriv_exact");
        return std::make_tuple(exact != 0, mu);
    }, "ws"_a);
    // The kept call refined to tol: (gap, iters, converged), as prox.
    m.def("refine", [](const Op &op, Workspace &ws, double tol, int max_iter, std::optional<OutArr> out) {
        fp_array oa{};
        if (out) oa = view(*out);
        fp_info info{0.0, 0, 0};
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_refine(op.p, ws.p, tol, max_iter, out ? &oa : nullptr, &info);
        }
        if (s != FP_ENOTCONV) check(s, "fp_workspace_refine");
        return std::make_tuple(info.gap, info.iters, s == FP_OK);
    }, "op"_a, "ws"_a, "tol"_a, "max_iter"_a, "out"_a.none());
    m.def("deriv_info", [](const Workspace &ws) {
        double r = 0.0;
        int it = 0;
        check(fp_workspace_deriv_info(ws.p, &r, &it), "fp_workspace_deriv_info");
        return std::make_tuple(r, it);
    }, "ws"_a);

    m.def("image_info", [](const Workspace &ws, std::optional<F64Out> gaps,
                           std::optional<I32Out> iters, std::optional<F64Out> zerrs, int64_t count) {
        for (size_t n : {gaps ? gaps->shape(0) : size_t(count), iters ? iters->shape(0) : size_t(count),
                         zerrs ? zerrs->shape(0) : size_t(count)})
            if (n != size_t(count)) throw nb::value_error("image_info: an array's size is not count");
        check(fp_workspace_image_info(ws.p, gaps ? gaps->data() : nullptr, iters ? iters->data() : nullptr,
                                      zerrs ? zerrs->data() : nullptr, count),
              "fp_workspace_image_info");
    }, "ws"_a, "gaps"_a.none(), "iters"_a.none(), "zerrs"_a.none(), "count"_a);

    m.def("prox", &prox, "op"_a, "step"_a, "x"_a, "out"_a, "axis"_a, "threads"_a, "keep"_a,
          "ws"_a.none(), "conj"_a, "tol"_a = 1e-6, "max_iter"_a = 1000, "tie"_a = 0,
          "steps"_a.none() = nb::none());
    // The copies and the kink scans are data-sized: without the GIL (audit P4).
    m.def("get_dual", [](const Workspace &ws, OutArr &out) {
        fp_array oa = view(out);
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_get_dual(ws.p, &oa);
        }
        check(s, "fp_workspace_get_dual");
    }, "ws"_a, "out"_a);
    m.def("set_dual", [](Workspace &ws, const InArr &dual) {
        const fp_array da = view(dual);
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_set_dual(ws.p, &da);
        }
        check(s, "fp_workspace_set_dual");
    }, "ws"_a, "dual"_a);
    m.def("set_warm", [](Workspace &ws, bool on) {
        check(fp_workspace_set_warm(ws.p, on ? 1 : 0), "fp_workspace_set_warm");
    }, "ws"_a, "on"_a);
    m.def("kinks", [](const Workspace &ws, double tol, std::optional<I8Out> mask) {
        int64_t count = 0;
        fp_array ma{};
        if (mask) ma = view_i8(*mask);
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_kinks(ws.p, tol, mask ? &ma : nullptr, &count);
        }
        check(s, "fp_workspace_kinks");
        return count;
    }, "ws"_a, "tol"_a, "mask"_a.none());
    m.def("set_branch", [](Workspace &ws, std::optional<I8In> side, double tol) {
        fp_array sa{};
        if (side) sa = view_i8(*side);
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_set_branch(ws.p, side ? &sa : nullptr, tol);
        }
        check(s, "fp_workspace_set_branch");
    }, "ws"_a, "side"_a.none(), "tol"_a);
    // Structure of the derivative (since 1.0.2).
    m.def("struct_kind", [](const Op &op) {
        int kind = 0;
        check(fp_op_struct_kind(op.p, &kind), "fp_op_struct_kind");
        return kind;
    }, "op"_a);
    m.def("ws_units", [](const Workspace &ws) {
        int64_t count = 0, size = 0;
        check(fp_workspace_units(ws.p, &count, &size), "fp_workspace_units");
        return nb::make_tuple(count, size);
    }, "ws"_a);
    m.def("ws_states", [](const Workspace &ws, I8Out state) {
        fp_array sa = view_i8(state);
        int64_t nfixed = 0;
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_states(ws.p, &sa, &nfixed);
        }
        check(s, "fp_workspace_states");
        return nfixed;
    }, "ws"_a, "state"_a);
    m.def("ws_regions", [](const Workspace &ws, OutArr labels, I8Out sign, I64Vec nregions) {
        fp_array la = view_label(labels), ga = view_i8(sign);
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_regions(ws.p, &la, &ga, nregions.data(), static_cast<int64_t>(nregions.shape(0)));
        }
        check(s, "fp_workspace_regions");
    }, "ws"_a, "labels"_a, "sign"_a, "nregions"_a);
    m.def("ws_constraint", [](const Workspace &ws, OutArr coef, I8Vec active) {
        fp_array ca = view(coef);
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_workspace_constraint(ws.p, &ca, active.data(), static_cast<int64_t>(active.shape(0)));
        }
        check(s, "fp_workspace_constraint");
    }, "ws"_a, "coef"_a, "active"_a);
    m.def("value", [](const Op &op, const InArr &x, OutArr &out, int axis, double tol) {
        const fp_array xa = view(x);
        fp_array oa = view(out);
        const fp_options o = options(axis, 1, false, tol);
        fp_status s;
        {
            nb::gil_scoped_release release;
            s = fp_value(op.p, &xa, &o, &oa);
        }
        check(s, "fp_value");
    }, "op"_a, "x"_a, "out"_a, "axis"_a, "tol"_a);
    m.def("workspace", &workspace, "op"_a, "like"_a, "axis"_a, "threads"_a, "keep"_a,
          nb::keep_alive<0, 1>());
    m.def("jvp", &jvp, "op"_a, "ws"_a, "dx"_a.none(), "dgamma"_a, "dparams"_a.none(), "dout"_a);
    m.def("vjp", &vjp, "op"_a, "ws"_a, "g"_a, "gx"_a, "gparams"_a.none(), "want_gamma"_a = true);
    m.def("spectral_check", [](const Op &op, int kind) {
        return fp_op_spectral_check(op.p, static_cast<fp_spectral_kind>(kind)) == FP_OK;
    });
    m.def("spectral_jvp", &spectral_jvp, "op"_a, "ws"_a, "kind"_a, "s"_a, "z"_a, "a"_a, "dgamma"_a,
          "dparams"_a.none(), "b"_a, "r"_a.none());

    m.def("abi_version", &fp_abi_version);
    m.def("version", &fp_version_string);
    m.def("default_threads", &fp_default_threads);
}
