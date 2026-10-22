// Fuzz harness for the public API: arbitrary bytes become an operator, its
// parameters, an array (shape, axis, strides, dtype, values including NaN,
// infinities and denormals), a step and options; the harness runs prox,
// prox_conj, jvp and vjp and checks the invariants that must hold for every
// input:
//   * every call returns a valid fp_status and never crashes or hangs;
//   * results are bit-identical for one and several threads;
//   * finite moderate inputs give finite outputs from a successful prox.
//
// Built as a libFuzzer target where the compiler supports -fsanitize=fuzzer,
// and otherwise as a standalone driver (fuzz/standalone.cpp) that replays
// files or generates random inputs.
#include "fastprox.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Bytes {
    const uint8_t *p;
    size_t n;
    uint8_t u8() {
        if (!n) return 0;
        --n;
        return *p++;
    }
    uint64_t u64() {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v = (v << 8) | u8();
        return v;
    }
    double f64() {  // raw bit patterns: NaN, infinities, denormals included
        const uint64_t b = u64();
        double d;
        std::memcpy(&d, &b, 8);
        return d;
    }
    // A double that is usually tame and sometimes raw.
    double value() {
        switch (u8() % 8) {
            case 0: return f64();
            case 1: return 0.0;
            case 2: return static_cast<double>(static_cast<int8_t>(u8()));
            default: return static_cast<double>(static_cast<int16_t>(u8() << 8 | u8())) / 256.0;
        }
    }
    double max_param = 0.0;  // largest magnitude handed out by param()
    // Raises max_param to m. NaN sticks: a plain !(m <= max_param) test let a
    // NaN replace an infinity and the next finite value replace the NaN, which
    // lost an a of 6e-322 for the plane and flagged its genuine overflow.
    void raise(double m) {
        if (!std::isnan(max_param) && !(m <= max_param)) max_param = m;
    }
    double param() {
        double v;
        switch (u8() % 6) {
            case 0: v = f64(); break;
            case 1: v = 0.0; break;
            default: v = (u8() + 1) / 64.0;
        }
        raise(std::fabs(v));
        return v;
    }
};

char g_case[256];  // description of the case under test, for reports

[[noreturn]] void fail(const char *what) {
    std::fprintf(stderr, "invariant violated: %s\n  case: %s\n", what, g_case);
    std::abort();
}

bool valid_status(fp_status s) { return s >= FP_OK && s <= FP_EINTERNAL; }

struct Case {
    fp_op *op = nullptr;
    std::vector<double> params[3];
    std::vector<int64_t> groups;
    std::vector<float> fparams;  // float32 copy of params[1] (spatial weights)
    int64_t sites = -1;          // kinks on m edges (tree TV): masks of shape (..., m)
};

fp_array vec(std::vector<double> &v) {
    static thread_local int64_t shape[2];
    shape[0] = static_cast<int64_t>(v.size());
    return fp_array{v.data(), FP_F64, 1, shape, nullptr, FP_DEVICE_CPU};
}

// Spatial weights for image TV of k axes on arrays of shape xs (xnd axes):
// usually the array's last k extents (iso), with a leading k for edge
// weights, sometimes a wrong shape or number of axes, float64 or float32,
// some zeros; the values from param() (raw bit patterns included).
fp_array spatial(Bytes &b, Case &c, int k, bool edges, int xnd, const int64_t *xs) {
    static thread_local int64_t sh[4];
    int nd = 0;
    if (edges) sh[nd++] = b.u8() % 8 ? k : 1 + b.u8() % 4;
    const int kk = b.u8() % 8 ? k : 1 + b.u8() % 3;
    for (int a = 0; a < kk; ++a) {
        const int d = xnd - kk + a;
        sh[nd++] = (b.u8() % 8 && d >= 0) ? xs[d] : b.u8() % 6;
    }
    int64_t m = 1;
    for (int d = 0; d < nd; ++d) m *= sh[d];
    std::vector<double> &v = c.params[1];
    v.resize(static_cast<size_t>(m > 0 ? m : 1));
    const bool zeros = b.u8() % 3 == 0;
    for (auto &x : v) x = zeros && b.u8() % 4 == 0 ? 0.0 : b.param();
    if (b.u8() % 4 == 0) {
        c.fparams.assign(v.begin(), v.end());
        return fp_array{c.fparams.data(), FP_F32, nd, sh, nullptr, FP_DEVICE_CPU};
    }
    return fp_array{v.data(), FP_F64, nd, sh, nullptr, FP_DEVICE_CPU};
}

// Builds an operator from the bytes; params are sized for axis length n of
// an array with xnd axes of extents xs.
fp_status make_op(Bytes &b, int64_t n, Case &c, int xnd, const int64_t *xs) {
    const int kind = b.u8() % 35;
    auto fill = [&](std::vector<double> &v, int64_t len) {
        v.resize(static_cast<size_t>(len > 0 ? len : 1));
        for (auto &x : v) x = b.param();
    };
    const int64_t pick = (b.u8() % 3 == 0) ? 1 : n;  // scalar or per-entry
    static int64_t s0[1], s1[1];
    fp_array a0, a1;
    switch (kind) {
        case 0:
            fill(c.params[0], pick);
            a0 = vec(c.params[0]);
            return fp_op_soft(b.param(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        case 1: return fp_op_elastic_net(b.param(), b.param(), &c.op);
        case 2: return fp_op_huber(b.param(), b.param(), &c.op);
        case 3: {
            const int64_t G = 1 + b.u8() % 4;
            c.groups.resize(static_cast<size_t>(n > 0 ? n : 1));
            for (auto &g : c.groups) g = b.u8() % (G + (b.u8() % 8 == 0));  // occasionally out of range
            fill(c.params[0], b.u8() % 2 ? G : 1);
            a0 = vec(c.params[0]);
            return fp_op_group_l2(b.param(), static_cast<int64_t>(c.groups.size()),
                                  b.u8() % 3 ? c.groups.data() : nullptr, G, &a0, &c.op);
        }
        case 4: {
            fill(c.params[0], pick);
            fill(c.params[1], pick);
            for (size_t i = 0; i < c.params[0].size(); ++i) c.params[0][i] = -c.params[0][i];
            s0[0] = static_cast<int64_t>(c.params[0].size());
            s1[0] = static_cast<int64_t>(c.params[1].size());
            a0 = fp_array{c.params[0].data(), FP_F64, 1, s0, nullptr, FP_DEVICE_CPU};
            a1 = fp_array{c.params[1].data(), FP_F64, 1, s1, nullptr, FP_DEVICE_CPU};
            const bool arrays = b.u8() % 2;
            return fp_op_box(-b.param(), b.param(), arrays ? &a0 : nullptr, arrays ? &a1 : nullptr,
                             &c.op);
        }
        case 5: return fp_op_nonneg(&c.op);
        case 6: return fp_op_l2_ball(b.param(), &c.op);
        case 7: return fp_op_simplex(b.param(), &c.op);
        case 8:
            fill(c.params[0], pick);
            a0 = vec(c.params[0]);
            return fp_op_l1_ball(b.param(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        case 9: return fp_op_linf(b.param(), &c.op);
        case 10:
            fill(c.params[0], n);
            a0 = vec(c.params[0]);
            return fp_op_isotonic(b.u8() % 2 ? &a0 : nullptr, b.u8() % 2, &c.op);
        case 13: {  // anisotropic TV on the last 2 or 3 axes
            const int k = 2 + b.u8() % 2;
            fill(c.params[0], b.u8() % 2 ? k : 1);
            if (b.u8() % 4 == 0) c.params[0][0] = 0.0;
            a0 = vec(c.params[0]);
            return fp_op_tv_aniso(k, b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 14: {  // isotropic TV on the last 2 or 3 axes
            const int k = 2 + b.u8() % 2;
            fill(c.params[0], b.u8() % 2 ? k : 1);
            if (b.u8() % 4 == 0) c.params[0][0] = 0.0;
            a0 = vec(c.params[0]);
            return fp_op_tv_iso(k, b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 31:    // isotropic TV with pixel weights (since 0.3)
        case 32: {  // anisotropic TV with edge weights
            const int k = 2 + b.u8() % 2;
            fill(c.params[0], b.u8() % 2 ? k : 1);
            if (b.u8() % 4 == 0) c.params[0][0] = 0.0;
            a0 = vec(c.params[0]);
            a1 = spatial(b, c, k, kind == 32, xnd, xs);
            const fp_array *sw = b.u8() % 16 ? &a1 : nullptr;  // NULL: FP_EINVAL
            return kind == 31 ? fp_op_tv_iso_weighted(k, b.u8() % 2 ? &a0 : nullptr, sw, &c.op)
                              : fp_op_tv_aniso_weighted(k, b.u8() % 2 ? &a0 : nullptr, sw, &c.op);
        }
        case 15:
            fill(c.params[0], pick);
            a0 = vec(c.params[0]);
            return fp_op_l0(b.param(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        case 16: {
            fill(c.params[0], pick);
            a0 = vec(c.params[0]);
            const double qs[5] = {0.5, 2.0 / 3.0, 0.1, 0.9, b.f64()};  // closed forms, Newton, junk
            return fp_op_lq(b.param(), qs[b.u8() % 5], b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 17: {
            fill(c.params[0], pick);
            a0 = vec(c.params[0]);
            return fp_op_scad(b.param(), 1.0 + b.param(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 18: {
            fill(c.params[0], pick);
            a0 = vec(c.params[0]);
            return fp_op_mcp(b.param(), b.param(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 19: return fp_op_log_penalty(b.param(), b.param(), &c.op);
        case 20: return fp_op_max(b.param(), &c.op);
        case 21: return fp_op_sum_largest(b.param(), static_cast<int64_t>(b.u8() % 40) - 2, b.u8() % 2, &c.op);
        case 22: {  // hyperplane or half-space with a box; b often infeasible
            fill(c.params[0], pick);
            for (auto &v : c.params[0]) {
                if (b.u8() % 4 == 0) v = -v;
                else if (b.u8() % 8 == 0) v = 0.0;
                // The projection scales like |b| / |a_i|: count 1 / |a_i| as a
                // parameter size for the finite-output check.
                if (v != 0.0) b.raise(1.0 / std::fabs(v));
            }
            fill(c.params[1], b.u8() % 2 ? n : 1);
            s0[0] = static_cast<int64_t>(c.params[0].size());
            s1[0] = static_cast<int64_t>(c.params[1].size());
            a0 = fp_array{c.params[0].data(), FP_F64, 1, s0, nullptr, FP_DEVICE_CPU};
            a1 = fp_array{c.params[1].data(), FP_F64, 1, s1, nullptr, FP_DEVICE_CPU};
            const double lo = b.u8() % 3 ? -b.param() : -INFINITY, hi = b.u8() % 3 ? b.param() : INFINITY;
            const bool upper = b.u8() % 2;  // per-entry upper bounds from params[1]
            const double rhs = (b.u8() % 2 ? -1.0 : 1.0) * b.param();
            return fp_op_hyperplane(b.u8() % 4 ? &a0 : nullptr, rhs, lo, hi, nullptr,
                                    upper ? &a1 : nullptr, b.u8() % 2, &c.op);
        }
        case 23: return fp_op_log_barrier(b.param(), &c.op);
        case 24: {  // l1 over a box; lo > hi sometimes
            fill(c.params[0], pick);
            fill(c.params[1], b.u8() % 2 ? n : 1);
            fill(c.params[2], b.u8() % 2 ? n : 1);
            for (auto &v : c.params[1]) v = -v;
            static int64_t s2[1];
            s0[0] = static_cast<int64_t>(c.params[0].size());
            s1[0] = static_cast<int64_t>(c.params[1].size());
            s2[0] = static_cast<int64_t>(c.params[2].size());
            a0 = fp_array{c.params[0].data(), FP_F64, 1, s0, nullptr, FP_DEVICE_CPU};
            a1 = fp_array{c.params[1].data(), FP_F64, 1, s1, nullptr, FP_DEVICE_CPU};
            fp_array a2{c.params[2].data(), FP_F64, 1, s2, nullptr, FP_DEVICE_CPU};
            const bool arrays = b.u8() % 2;
            const double lam = b.param();
            return fp_op_l1_box(lam, b.u8() % 2 ? &a0 : nullptr, -b.param(), b.param(),
                                arrays ? &a1 : nullptr, arrays ? &a2 : nullptr, &c.op);
        }
        case 25: return fp_op_soc(&c.op);
        case 26: {  // group Huber, SCAD or MCP, groups as in case 3
            const int64_t G = 1 + b.u8() % 4;
            c.groups.resize(static_cast<size_t>(n > 0 ? n : 1));
            for (auto &g : c.groups) g = b.u8() % (G + (b.u8() % 8 == 0));
            fill(c.params[0], b.u8() % 2 ? G : 1);
            a0 = vec(c.params[0]);
            const int64_t *g = b.u8() % 3 ? c.groups.data() : nullptr;
            const int64_t len = static_cast<int64_t>(c.groups.size());
            const double lam = b.param(), p = b.param();
            switch (b.u8() % 3) {
                case 0: return fp_op_group_huber(lam, p, len, g, G, &a0, &c.op);
                case 1: return fp_op_group_scad(lam, 1.0 + p, len, g, G, &a0, &c.op);
                default: return fp_op_group_mcp(lam, p, len, g, G, &a0, &c.op);
            }
        }
        case 27: {  // tree norm: random intervals, nested or not (then FP_EPARAM)
            static std::vector<int64_t> ptr, idx;
            ptr.assign(1, 0);
            idx.clear();
            const int64_t G = b.u8() % 6, len = n > 0 ? n : 1;
            for (int64_t g = 0; g < G; ++g) {
                const int64_t lo = b.u8() % len, hi = lo + 1 + b.u8() % (len - lo + (b.u8() % 8 == 0));
                for (int64_t i = lo; i < hi; ++i) idx.push_back(i);  // hi may pass the axis
                ptr.push_back(static_cast<int64_t>(idx.size()));
            }
            fill(c.params[0], b.u8() % 2 ? G : 1);
            a0 = vec(c.params[0]);
            return fp_op_tree_l2(b.param(), G, ptr.data(), idx.data(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 28: {
            fill(c.params[0], pick == 1 ? 1 : n - 1);
            if (b.u8() % 4 == 0)
                for (auto &w : c.params[0])
                    if (b.u8() % 3 == 0) w = 0.0;
            a0 = vec(c.params[0]);
            return fp_op_tv1d_ball(b.param(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 33: {  // tree TV: a random forest on the last axis, some weights zero
            const int64_t N = xnd ? xs[xnd - 1] : 1;
            c.groups.clear();
            for (int64_t i = 1; i < N; ++i)
                if (b.u8() % 4) {
                    const int64_t j = b.u8() % i;
                    const bool flip = b.u8() % 2;
                    c.groups.push_back(flip ? j : i);
                    c.groups.push_back(flip ? i : j);
                }
            const int64_t m = static_cast<int64_t>(c.groups.size() / 2);
            fill(c.params[0], b.u8() % 2 ? 1 : m);
            if (b.u8() % 4 == 0)
                for (auto &w : c.params[0])
                    if (b.u8() % 3 == 0) w = 0.0;
            a0 = vec(c.params[0]);
            c.sites = m;
            return fp_op_tree_tv(N, m, c.groups.data(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 34: {  // graph TV: random edges on the last axis, cycles and repeats allowed
            const int64_t N = xnd ? xs[xnd - 1] : 1;
            c.groups.clear();
            const int64_t m0 = N > 1 ? b.u8() % (3 * N) : 0;
            for (int64_t e = 0; e < m0; ++e) {
                const int64_t i = b.u8() % N, j = (i + 1 + b.u8() % (N - 1)) % N;
                c.groups.push_back(i);
                c.groups.push_back(j);
            }
            const int64_t m = static_cast<int64_t>(c.groups.size() / 2);
            fill(c.params[0], b.u8() % 2 ? 1 : m);
            if (b.u8() % 4 == 0)
                for (auto &w : c.params[0])
                    if (b.u8() % 3 == 0) w = 0.0;
            a0 = vec(c.params[0]);
            c.sites = m;
            return fp_op_graph_tv(N, m, c.groups.data(), b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        case 29: {
            fill(c.params[0], pick == 1 ? 1 : n - 1);
            if (b.u8() % 4 == 0)
                for (auto &w : c.params[0])
                    if (b.u8() % 3 == 0) w = 0.0;
            a0 = vec(c.params[0]);
            return fp_op_tv1d_l2(b.u8() % 2 ? &a0 : nullptr, &c.op);
        }
        default:
            fill(c.params[0], pick == 1 ? 1 : n - 1);
            if (b.u8() % 4 == 0)  // zero weights split the row
                for (auto &w : c.params[0])
                    if (b.u8() % 3 == 0) w = 0.0;
            a0 = vec(c.params[0]);
            return fp_op_tv1d(b.u8() % 2 ? &a0 : nullptr, static_cast<fp_tv1d_method>(b.u8() % 4),
                              &c.op);
    }
}

void run(const uint8_t *data, size_t size) {
    Bytes b{data, size};

    // Array: up to 3 dimensions of up to 40 entries, optionally a strided
    // (reversed or stepped) view of a larger buffer.
    const int ndim = b.u8() % 4;  // 0 = scalar
    int64_t shape[3] = {1, 1, 1}, base[3], strides[3];
    int64_t numel = 1;
    for (int d = 0; d < ndim; ++d) {
        shape[d] = b.u8() % 41;
        numel *= shape[d];
    }
    const int axis = ndim ? static_cast<int>(b.u8() % (2 * ndim)) - ndim : -1;
    const int64_t n = ndim ? shape[(axis + ndim) % ndim] : 1;
    const bool f32 = b.u8() % 3 == 0;
    const int step = 1 + b.u8() % 2;
    const bool reverse = b.u8() % 4 == 0;

    int64_t total = 1;
    for (int d = ndim - 1; d >= 0; --d) {
        base[d] = total;
        total *= (d == ndim - 1 ? step : 1) * std::max<int64_t>(shape[d], 1);
    }
    for (int d = 0; d < ndim; ++d) strides[d] = base[d] * (d == ndim - 1 ? step : 1);
    const size_t item = f32 ? 4 : 8;
    std::vector<unsigned char> xbuf(static_cast<size_t>(total) * item + 8);
    int64_t offset = 0;
    if (reverse && ndim && shape[0] > 0) {
        offset = (shape[0] - 1) * strides[0];
        strides[0] = -strides[0];
    }
    for (int64_t i = 0; i < total; ++i) {
        const double v = b.value();
        if (f32) {
            const float f = static_cast<float>(v);
            std::memcpy(&xbuf[static_cast<size_t>(i) * 4], &f, 4);
        } else {
            std::memcpy(&xbuf[static_cast<size_t>(i) * 8], &v, 8);
        }
    }
    void *x0 = xbuf.data() + offset * static_cast<int64_t>(item);

    Case c;
    const fp_status cs = make_op(b, n, c, ndim, shape);
    if (!valid_status(cs)) fail("constructor status");
    if (cs != FP_OK) {
        if (c.op) fail("constructor wrote a handle on failure");
        return;
    }

    double gamma = b.u8() % 4 ? b.param() : b.f64();
    // Image TV far from unit scale (since 1.0, audit T5): x and gamma times
    // 2^e, |e| in [600, 1000], which the solvers rescale back exactly.
    const char *opname = fp_op_name(c.op);
    if (!f32 && (std::strcmp(opname, "tv_aniso") == 0 || std::strcmp(opname, "tv_iso") == 0) && b.u8() % 4 == 0) {
        const int e = (b.u8() % 2 ? 1 : -1) * (600 + b.u8() * 400 / 255);
        for (int64_t i = 0; i < total; ++i) {
            double v;
            std::memcpy(&v, &xbuf[static_cast<size_t>(i) * 8], 8);
            v = std::ldexp(v, e);
            std::memcpy(&xbuf[static_cast<size_t>(i) * 8], &v, 8);
        }
        gamma = std::ldexp(gamma, e);
    }
    std::snprintf(g_case, sizeof g_case, "op=%s nparams=%lld dtype=%s ndim=%d shape=(%lld,%lld,%lld) axis=%d gamma=%.17g",
                  fp_op_name(c.op), static_cast<long long>(fp_op_nparams(c.op)), f32 ? "f32" : "f64", ndim,
                  static_cast<long long>(shape[0]), static_cast<long long>(shape[1]),
                  static_cast<long long>(shape[2]), axis, gamma);
    fp_options o;
    fp_options_init(&o);
    o.axis = axis;
    o.max_iter = 30;  // iterative operators: keep cases fast (FP_ENOTCONV is fine)
    o.tie = b.u8() % 2;  // nonconvex maps: either tie rule

    // Contiguous outputs for one and four threads.
    const fp_dtype dt = f32 ? FP_F32 : FP_F64;
    std::vector<unsigned char> z1(static_cast<size_t>(numel) * item + 8),
        z4(static_cast<size_t>(numel) * item + 8);
    const fp_array x{x0, dt, ndim, shape, strides, FP_DEVICE_CPU};
    fp_array o1{z1.data(), dt, ndim, shape, nullptr, FP_DEVICE_CPU};
    fp_array o4{z4.data(), dt, ndim, shape, nullptr, FP_DEVICE_CPU};

    const bool conj = b.u8() % 3 == 0;
    std::strncat(g_case, conj ? " conj" : "", sizeof g_case - std::strlen(g_case) - 1);
    auto prox = conj ? fp_prox_conj : fp_prox;
    o.threads = 1;
    const fp_status s1 = prox(c.op, gamma, &x, &o1, &o, nullptr, nullptr);
    o.threads = 4;
    const fp_status s4 = prox(c.op, gamma, &x, &o4, &o, nullptr, nullptr);
    if (!valid_status(s1) || s1 != s4) fail("prox status");
    const bool written = s1 == FP_OK || s1 == FP_ENOTCONV;  // ENOTCONV still writes the iterate

    // The value (since 0.5), into x's shape without the axis: a row
    // operator's batch shape (an image operator usually answers FP_ESHAPE).
    {
        const int ax = ndim ? (axis + ndim) % ndim : 0;
        int64_t vshape[3], vn = 1;
        int vnd = 0;
        for (int d = 0; d < ndim; ++d)
            if (d != ax) {
                vshape[vnd++] = shape[d];
                vn *= shape[d];
            }
        std::vector<double> vals(static_cast<size_t>(vn) + 1);
        fp_array va{vals.data(), FP_F64, vnd, vshape, nullptr, FP_DEVICE_CPU};
        o.tol = b.u8() % 2 ? 1e-6 : std::fabs(b.value());
        const fp_status vs = fp_value(c.op, &x, &o, &va);
        if (!valid_status(vs)) fail("value status");
        o.tol = 1e-6;
    }
    if (written && std::memcmp(z1.data(), z4.data(), static_cast<size_t>(numel) * item) != 0)
        fail("thread count changed the result");

    // Finite moderate input, parameters and step: the output of a successful
    // prox is finite. Moderate means far from overflow in the array's dtype.
    const double big = f32 ? 1e12 : 1e100;
    if (written && std::isfinite(gamma) && gamma < 1e6 && b.max_param < big &&
        !(conj && gamma < 1e-6)) {
        bool moderate = true;
        std::vector<double> xs;
        for (int64_t i = 0; i < total && moderate; ++i) {
            double v;
            if (f32) {
                float f;
                std::memcpy(&f, &xbuf[static_cast<size_t>(i) * 4], 4);
                v = f;
            } else {
                std::memcpy(&v, &xbuf[static_cast<size_t>(i) * 8], 8);
            }
            moderate = std::isfinite(v) && std::fabs(v) < big;
        }
        if (moderate)
            for (int64_t i = 0; i < numel; ++i) {
                const double v = f32 ? static_cast<double>(reinterpret_cast<float *>(z1.data())[i])
                                     : reinterpret_cast<double *>(z1.data())[i];
                if (!std::isfinite(v)) {
                    std::fprintf(stderr, "x (raw buffer):");
                    for (int64_t k = 0; k < total; ++k)
                        std::fprintf(stderr, " %.9g", f32 ? static_cast<double>(reinterpret_cast<float *>(xbuf.data())[k])
                                                          : reinterpret_cast<double *>(xbuf.data())[k]);
                    std::fprintf(stderr, "\nz:");
                    for (int64_t k = 0; k < numel; ++k)
                        std::fprintf(stderr, " %.9g", f32 ? static_cast<double>(reinterpret_cast<float *>(z1.data())[k])
                                                          : reinterpret_cast<double *>(z1.data())[k]);
                    std::fprintf(stderr, "\nparams:");
                    for (auto p : c.params[0]) std::fprintf(stderr, " %.17g", p);
                    std::fprintf(stderr, "\n");
                    fail("non-finite output from finite input");
                }
            }
    }

    // Derivatives at the saved structure.
    if (b.u8() % 2) {
        o.keep_structure = 1;
        o.threads = 1 + b.u8() % 3;
        fp_workspace *ws = nullptr;
        const fp_status ws_s = fp_workspace_create(c.op, &x, &o, &ws);
        if (!valid_status(ws_s)) fail("workspace status");
        if (ws_s == FP_OK) {
            const fp_status ps = prox(c.op, gamma, &x, &o1, &o, ws, nullptr);
            if (!valid_status(ps)) fail("prox with workspace status");
            const int64_t P = fp_op_nparams(c.op);
            std::vector<double> dp(static_cast<size_t>(P > 0 ? P : 1)), gp(dp.size());
            for (auto &v : dp) v = b.value();
            int64_t ps_shape[1] = {P};
            fp_array dpa{dp.data(), FP_F64, 1, ps_shape, nullptr, FP_DEVICE_CPU};
            fp_array gpa{gp.data(), FP_F64, 1, ps_shape, nullptr, FP_DEVICE_CPU};
            // Kinks and a branch override (elementwise maps with kinks; any
            // other operator, or a failed prox, answers FP_EINVAL).
            std::vector<int8_t> mask, side;
            if (b.u8() % 2) {
                int64_t ms_shape[3], numel = 1;
                for (int d = 0; d < x.ndim; ++d) {
                    ms_shape[d] = c.sites >= 0 && d == x.ndim - 1 ? c.sites : x.shape[d];
                    numel *= ms_shape[d];
                }
                mask.assign(static_cast<size_t>(numel), 0);
                side.resize(mask.size());
                for (auto &v : side) v = static_cast<int8_t>(b.u8() % 3) - 1;
                fp_array ma{mask.data(), FP_I8, x.ndim, ms_shape, nullptr, FP_DEVICE_CPU};
                fp_array sa{side.data(), FP_I8, x.ndim, ms_shape, nullptr, FP_DEVICE_CPU};
                const double tol = b.u8() % 2 ? 0.0 : std::fabs(b.value());
                int64_t count = 0;
                const fp_status ks = fp_workspace_kinks(ws, tol, &ma, &count);
                const fp_status bs = fp_workspace_set_branch(ws, &sa, tol);
                if (!valid_status(ks) || !valid_status(bs)) fail("kink status");
                if (ks == FP_OK && (count < 0 || count > numel)) fail("kink count");
            }
            const fp_status js = fp_jvp(c.op, ws, &x, b.value(), P ? &dpa : nullptr, &o4);
            double gg = 0;
            const fp_status vs = fp_vjp(c.op, ws, &x, &o4, &gg, P ? &gpa : nullptr);
            if (!valid_status(js) || !valid_status(vs)) fail("derivative status");
            if (ps != FP_OK && ps != FP_ENOTCONV && (js == FP_OK || vs == FP_OK))
                fail("derivative without structure");
        }
        fp_workspace_free(ws);
    }
    fp_op_free(c.op);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    run(data, size);
    return 0;
}
