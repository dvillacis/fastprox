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

#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using InArr = nb::ndarray<nb::ro, nb::device::cpu>;
using OutArr = nb::ndarray<nb::device::cpu>;
using Params = nb::ndarray<nb::ro, const double, nb::ndim<1>, nb::device::cpu>;
using IdArr = nb::ndarray<nb::ro, const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void check(fp_status s) {
    if (s == FP_OK) return;
    const char *msg = fp_strerror(s);
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
    check(f(&op->p));
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
                                            double tol, int max_iter, int tie) {
    const fp_array xa = view(x);
    fp_array oa = view(out);
    const fp_options o = options(axis, threads, keep, tol, max_iter, tie);
    fp_info info{0.0, 0, 0};
    fp_status s;
    {
        nb::gil_scoped_release release;
        s = conj ? fp_prox_conj(op.p, step, &xa, &oa, &o, ws ? ws->p : nullptr, &info)
                 : fp_prox(op.p, step, &xa, &oa, &o, ws ? ws->p : nullptr, &info);
    }
    if (s != FP_ENOTCONV) check(s);
    return {info.gap, info.iters, info.ties, s == FP_OK};
}

Workspace *workspace(const Op &op, const InArr &like, int axis, int threads, bool keep) {
    // tol and max_iter are per call; the workspace does not depend on them.
    auto ws = std::make_unique<Workspace>();
    const fp_array la = view(like);
    const fp_options o = options(axis, threads, keep);
    check(fp_workspace_create(op.p, &la, &o, &ws->p));
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
    check(s);
}

double vjp(const Op &op, const Workspace &ws, const InArr &g, OutArr &gx,
           std::optional<OutArr> gparams) {
    const fp_array ga = view(g);
    fp_array xa = view(gx);
    fp_array pa{};
    if (gparams) pa = view(*gparams);
    double gg = 0.0;
    fp_status s;
    {
        nb::gil_scoped_release release;
        s = fp_vjp(op.p, ws.p, &ga, &xa, &gg, gparams ? &pa : nullptr);
    }
    check(s);
    return gg;
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
    m.def("box", [](double lo, double hi, std::optional<Params> lower,
                    std::optional<Params> upper) {
        const ParamView pl(lower), pu(upper);
        return make([&](fp_op **o) { return fp_op_box(lo, hi, pl.get(), pu.get(), o); });
    }, "lo"_a, "hi"_a, "lower"_a.none(), "upper"_a.none());
    m.def("nonneg", [] { return make([](fp_op **o) { return fp_op_nonneg(o); }); });
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
    m.def("isotonic", [](std::optional<Params> w, bool increasing) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_isotonic(pw.get(), increasing, o); });
    }, "weights"_a.none(), "increasing"_a);
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
    m.def("scad", [](double lam, double a) {
        return make([&](fp_op **o) { return fp_op_scad(lam, a, o); });
    }, "lam"_a, "a"_a);
    m.def("mcp", [](double lam, double b) {
        return make([&](fp_op **o) { return fp_op_mcp(lam, b, o); });
    }, "lam"_a, "b"_a);
    m.def("log_penalty", [](double lam, double eps) {
        return make([&](fp_op **o) { return fp_op_log(lam, eps, o); });
    }, "lam"_a, "eps"_a);
    m.def("tv_aniso", [](int ndim, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_tv_aniso(ndim, pw.get(), o); });
    }, "ndim"_a, "weights"_a.none());

    m.def("tv_iso", [](int ndim, std::optional<Params> w) {
        const ParamView pw(w);
        return make([&](fp_op **o) { return fp_op_tv_iso(ndim, pw.get(), o); });
    }, "ndim"_a, "weights"_a.none());
    m.def("set_deriv_tol", [](Workspace &ws, double tol, int max_iter) {
        check(fp_workspace_set_deriv_tol(ws.p, tol, max_iter));
    }, "ws"_a, "tol"_a, "max_iter"_a);
    m.def("deriv_info", [](const Workspace &ws) {
        double r = 0.0;
        int it = 0;
        check(fp_workspace_deriv_info(ws.p, &r, &it));
        return std::make_tuple(r, it);
    }, "ws"_a);

    m.def("prox", &prox, "op"_a, "step"_a, "x"_a, "out"_a, "axis"_a, "threads"_a, "keep"_a,
          "ws"_a.none(), "conj"_a, "tol"_a = 1e-6, "max_iter"_a = 1000, "tie"_a = 0);
    m.def("get_dual", [](const Workspace &ws, OutArr &out) {
        fp_array oa = view(out);
        check(fp_workspace_get_dual(ws.p, &oa));
    }, "ws"_a, "out"_a);
    m.def("set_dual", [](Workspace &ws, const InArr &dual) {
        const fp_array da = view(dual);
        check(fp_workspace_set_dual(ws.p, &da));
    }, "ws"_a, "dual"_a);
    m.def("set_warm", [](Workspace &ws, bool on) { check(fp_workspace_set_warm(ws.p, on ? 1 : 0)); },
          "ws"_a, "on"_a);
    m.def("workspace", &workspace, "op"_a, "like"_a, "axis"_a, "threads"_a, "keep"_a,
          nb::keep_alive<0, 1>());
    m.def("jvp", &jvp, "op"_a, "ws"_a, "dx"_a.none(), "dgamma"_a, "dparams"_a.none(), "dout"_a);
    m.def("vjp", &vjp, "op"_a, "ws"_a, "g"_a, "gx"_a, "gparams"_a.none());

    m.def("abi_version", &fp_abi_version);
    m.def("version", &fp_version_string);
    m.def("default_threads", &fp_default_threads);
}
