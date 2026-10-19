// 1D total variation: prox of gamma sum_e w_e |z_{e+1} - z_e| along the axis.
//
// Two exact paths:
//  * L. Condat, "A direct algorithm for 1-D total variation denoising", IEEE
//    Signal Processing Letters 20(11), 2013, generalised here to per-edge
//    weights (see condat below). At the end
//    of the signal, a restart after a jump uses the plain post-jump bounds
//    instead of the paper's tightened ones (measured over 200k signals: same
//    work, simpler). A budget on rescanned steps bounds the backtracking, which
//    is quadratic on ramps: past it, the row is redone by the taut string.
//  * The taut string (P. L. Davies and A. Kovac, Ann. Statist. 29(1), 2001;
//    weighted form as in A. Barbero and S. Sra, JMLR 19, 2018): the cumulative
//    sum of z is the shortest path through the tube S_k -+ gamma w_k, found by
//    a funnel over the upper and lower convex chains in O(n). The segment
//    representation of the chains follows proxTV (see taut_string below).
// Zero weights split the row into independent pieces.
//
// Saved structure (code[i] for i >= 1, the edge between i-1 and i):
//   0  i continues the segment of i-1;
//   1  new segment, dual u_{i-1} = +gamma w_{i-1} (a downward jump);
//   2  new segment, dual u_{i-1} = -gamma w_{i-1} (an upward jump);
//   3  new segment with a free edge (zero weight), u_{i-1} = 0.
// With u_e = sum_{i <= e} (x_i - z_i), a segment B = [s, t] satisfies
//   |B| z_B = sum_B x + u_{s-1} - u_t,
// which gives every derivative in O(n).
//
// SPDX-License-Identifier: MIT
#include "core/op.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace fp {
namespace {

constexpr uint8_t kSame = 0, kDown = 1, kUp = 2, kFree = 3;

inline double sigma_of(uint8_t c) noexcept { return c == kDown ? 1.0 : (c == kUp ? -1.0 : 0.0); }

// Reciprocals 1/m for segment lengths m < kRecip, filled once on first use
// (thread-safe static initialisation). Longer segments divide.
constexpr int64_t kRecip = int64_t(1) << 15;

struct RecipTable {
    double r[kRecip];
    RecipTable() noexcept {
        r[0] = 0.0;
        for (int64_t m = 1; m < kRecip; ++m) r[m] = 1.0 / static_cast<double>(m);
    }
};

constexpr int64_t kShort = 3;  // measured: 2 to 4 are close; 8 is up to a third slower at gamma >= 1

inline const double *recip_table() noexcept {
    static const RecipTable t;
    return t.r;
}

// Condat's direct algorithm. radius(k) is the threshold of edge k (between
// samples k and k+1), positive for k < n-1; radius(n-1) may be anything
// nonnegative (the end condition u_{n-1} = 0 is enforced separately). With
// per-edge radii the bounds of a new segment after a jump at edge e start
// from y_k + sigma radius(e) -+ radius(k), sigma = +1 down, -1 up; with a
// constant radius this is the paper's algorithm. Returns false when more
// than `budget` steps had to be rescanned after jumps (the output is then
// incomplete).
//
// The bounds are updated in two equivalent forms. In the first kShort samples
// of a segment, as in the paper: u += y - v, then v moves by the excess of u
// over the radius divided by the length m, with min/max instead of branches
// (whether a bound is clamped there is close to random). From then on the
// segment is tracked by its running sum s = sum_{k0..k} (y_i - ref) + sigma
// radius(e), ref = y_{k0}: since u = s - m (v - ref), a clamp of vmin at k sets
// it to ref + (s_k - radius(k)) / m_k, so vmin is the running maximum of these
// candidates and vmax the running minimum of ref + (s_k + radius(k)) / m_k, and
// a jump down (up) is the upper (lower) candidate passing below vmin (above
// vmax). The only value carried from step to step is then the sum, instead
// of a chain through the clamps; this is 1.3 to 3 times faster in long
// segments, and the bounds no longer accumulate one rounding per clamp.
template <int64_t Short = kShort, class T, class R>
bool condat(const T *y, T *z, int64_t n, R radius, uint8_t *code, int64_t budget) noexcept {
    const double *recip = recip_table();
    auto rcp = [recip](int64_t m) { return m < kRecip ? recip[m] : 1.0 / static_cast<double>(m); };
    int64_t k = 0, k0 = 0, km = 0, kp = 0;
    double lk = radius(0);
    double vmin = static_cast<double>(y[0]) - lk, vmax = static_cast<double>(y[0]) + lk;
    double umin = lk, umax = -lk;  // first kShort samples
    double ref = 0.0, s = 0.0;     // afterwards
    static_assert(Short >= 1, "the first sample of a segment opens it");
    // With Short = 1 (bench/tv_variants.cpp) the running sum starts when a
    // segment opens, at s = c = sigma radius(e).
    auto open_form = [&](double yk, double c) {
        if constexpr (Short == 1) {
            ref = yk;
            s = c;
        }
    };
    int64_t rescanned = 0;

    auto jump_down = [&]() {  // u_{km} = +radius(km)
        const T v = static_cast<T>(vmin);
        for (int64_t i = k0; i <= km; ++i) z[i] = v;
        if (code) code[km + 1] = kDown;
        rescanned += k - km;
        const double le = radius(km);
        k = k0 = kp = km = km + 1;
        lk = radius(k);
        const double yk = static_cast<double>(y[k]);
        vmin = yk + (le - lk);
        vmax = yk + (le + lk);
        umin = lk;
        umax = -lk;
        open_form(yk, le);
    };
    auto jump_up = [&]() {  // u_{kp} = -radius(kp)
        const T v = static_cast<T>(vmax);
        for (int64_t i = k0; i <= kp; ++i) z[i] = v;
        if (code) code[kp + 1] = kUp;
        rescanned += k - kp;
        const double le = radius(kp);
        k = k0 = km = kp = kp + 1;
        lk = radius(k);
        const double yk = static_cast<double>(y[k]);
        vmin = yk - (le + lk);
        vmax = yk - (le - lk);
        umin = lk;
        umax = -lk;
        open_form(yk, -le);
    };
    open_form(static_cast<double>(y[0]), 0.0);

    for (;;) {
        while (k < n - 1) {
            const double yn = static_cast<double>(y[k + 1]);
            const double ln = radius(k + 1);
            const int64_t m = k - k0 + 2;  // length after this step
            if (m <= Short) {
                if (yn + umin < vmin - ln) {
                    jump_down();
                    if (rescanned > budget) return false;
                } else if (yn + umax > vmax + ln) {
                    jump_up();
                    if (rescanned > budget) return false;
                } else {
                    ++k;
                    umin += yn - vmin;
                    umax += yn - vmax;
                    // The clamps as min/max rather than selects on a
                    // comparison, which GCC turns back into branches (the
                    // input is finite here). The excesses dm and dp are
                    // exactly 0 when no clamp happens.
                    const double r = recip[m];
                    const bool cm = umin >= ln, cp = umax <= -ln;
                    const double dm = max_nonan(umin - ln, 0.0), dp = min_nonan(umax + ln, 0.0);
                    vmin += dm * r;
                    umin = min_nonan(umin, ln);
                    km = cm ? k : km;
                    vmax += dp * r;
                    umax = max_nonan(umax, -ln);
                    kp = cp ? k : kp;
                    if (m == Short) {  // switch to the running sum
                        ref = static_cast<double>(y[k0]);
                        s = umin + static_cast<double>(m) * (vmin - ref);
                    }
                }
            } else {
                const double s1 = s + (yn - ref);
                const double r = rcp(m);
                const double a = ref + (s1 - ln) * r, b = ref + (s1 + ln) * r;
                if (b < vmin) {
                    jump_down();
                    if (rescanned > budget) return false;
                } else if (a > vmax) {
                    jump_up();
                    if (rescanned > budget) return false;
                } else {
                    ++k;
                    s = s1;
                    km = a >= vmin ? k : km;
                    vmin = max_nonan(vmin, a);
                    kp = b <= vmax ? k : kp;
                    vmax = min_nonan(vmax, b);
                }
            }
        }
        // End of the signal: the last segment must close with u_{n-1} = 0,
        // tested in the form its bounds were last updated in.
        const int64_t m = k - k0 + 1;
        bool down, up;
        double v;
        if (m <= Short) {
            down = umin < 0.0;
            up = umax > 0.0;
            v = vmin + umin / static_cast<double>(m);
        } else {
            // With the reciprocal the candidates used, the mean is at least
            // the last lower candidate and at most the last upper one (rounding
            // is monotone), so neither jump can start past the last sample.
            v = ref + s * rcp(m);
            down = v < vmin;
            up = v > vmax;
        }
        if (down) {
            jump_down();
        } else if (up) {
            jump_up();
        } else {
            const T vt = static_cast<T>(v);
            for (int64_t i = k0; i < n; ++i) z[i] = vt;
            return true;
        }
        if (rescanned > budget) return false;
    }
}

// A radius that edge e can never reach. The prox stays within [min x, max x]
// (maximum principle), so u_e = sum_{i<=e} (x_i - z_i) = -sum_{i>e} (x_i - z_i)
// satisfies |u_e| <= j * range with j = min(e + 1, n - 1 - e). Radii above
// this bound are inactive, and clamping them to twice the bound changes
// neither the solution nor its structure; it keeps huge weights from adding
// rounding of order eps * radius to every bound and tube height.
inline double inactive_radius(int64_t j, double range) noexcept {
    return 2.0 * static_cast<double>(std::max<int64_t>(j, 1)) * range;
}

// When no edge can be active the prox is the constant mean. A compensated
// sum keeps u_{n-1} = sum x - n z at rounding level.
template <class T>
void constant_mean(const T *x, T *z, int64_t m) noexcept {
    double hi = 0.0, lo = 0.0;
    for (int64_t i = 0; i < m; ++i) {
        const double v = static_cast<double>(x[i]);
        const double s = hi + v;
        const double bb = s - hi;
        lo += (hi - (s - bb)) + (v - bb);
        hi = s;
    }
    const T mean = static_cast<T>((hi + lo) / static_cast<double>(m));
    for (int64_t i = 0; i < m; ++i) z[i] = mean;
}

// Typical signals rescan 0 to 3 steps per sample; ramps and noisy ramps with
// large gamma rescan 10 to 3000 (quadratic). Past about 3n rescanned steps
// the taut string is faster.
inline int64_t condat_budget(int64_t n) noexcept { return 3 * n + 1024; }

// The two chains of the funnel are stored as segments (length, rise, slope)
// rather than as points, following proxTV's classic taut string (A. Barbero
// and S. Sra, BSD licence): appending a tube step is a unit segment whose rise
// is the sample itself plus the change of the tube radius, convexity tests
// compare cached slopes, and the cumulative sum of the signal is never formed,
// so slopes come from local sums and stay accurate on long rows. Unlike
// proxTV, the last explored tube points are also kept relative to the current
// knot, and the radius may vary per edge.
struct Seg {
    int64_t dx;
    double dy;
    double slope;
};

inline size_t taut_scratch(int64_t n) noexcept { return 2 * static_cast<size_t>(n + 1) * sizeof(Seg); }

// Taut string on y[0..m) with edge radii edge_radius(e), e < m - 1, all
// positive. Writes z[0..m) and code[1..m).
//
// up is the convex minorant of the upper tube boundary (slopes increasing),
// lo the concave majorant of the lower one (slopes decreasing); both start at
// the current knot. When the first slope of up falls below that of lo the
// funnel has closed: the vertex nearest to the knot is the next knot, its
// segment is final, and the other chain restarts as one segment from it.
template <class T, class R>
void taut_string(const T *y, T *z, int64_t m, R edge_radius, uint8_t *code, void *scr) noexcept {
    if (m == 1) {
        z[0] = y[0];
        return;
    }
    Seg *up = static_cast<Seg *>(scr);
    Seg *lo = up + (m + 1);
    int64_t uhd = 0, utl = 0, lhd = 0, ltl = 0;  // [head, tail)
    auto radius = [&](int64_t k) {  // tube radius at cumulative index k (edge k - 1)
        return (k <= 0 || k >= m) ? 0.0 : edge_radius(k - 1);
    };

    int64_t pos = 0;           // first sample not yet written
    int64_t dx = 0;            // last explored index relative to the knot
    double du = 0.0, dl = 0.0;  // last upper and lower heights relative to the knot

    auto emit = [&](const Seg &sg, uint8_t c) {
        const T v = static_cast<T>(sg.slope);
        for (int64_t i = 0; i < sg.dx; ++i) z[pos + i] = v;
        pos += sg.dx;
        if (code && pos < m) code[pos] = c;
        dx -= sg.dx;
    };

    for (int64_t k = 1; k <= m; ++k) {
        const double yk = static_cast<double>(y[k - 1]);
        const double dr = radius(k) - radius(k - 1);
        ++dx;
        du += yk + dr;
        dl += yk - dr;

        // Append the unit step to the convex chain under the upper boundary.
        Seg s{1, yk + dr, yk + dr};
        if (utl > uhd && s.slope < up[utl - 1].slope) {
            do {
                --utl;
                s.dx += up[utl].dx;
                s.dy += up[utl].dy;
            } while (utl > uhd && s.dy < static_cast<double>(s.dx) * up[utl - 1].slope);
            s.slope = s.dy / static_cast<double>(s.dx);
        }
        up[utl++] = s;

        // And to the concave chain over the lower boundary.
        s = Seg{1, yk - dr, yk - dr};
        if (ltl > lhd && s.slope > lo[ltl - 1].slope) {
            do {
                --ltl;
                s.dx += lo[ltl].dx;
                s.dy += lo[ltl].dy;
            } while (ltl > lhd && s.dy > static_cast<double>(s.dx) * lo[ltl - 1].slope);
            s.slope = s.dy / static_cast<double>(s.dx);
        }
        lo[ltl++] = s;

        // Close the funnel while the chains cross at the knot. In exact
        // arithmetic a knot never reaches the last explored point (the tube
        // has positive radius inside); at the pinned end point both chains
        // meet and rounding can make them seem to cross, hence dx > 0.
        while (dx > 0 && up[uhd].slope < lo[lhd].slope) {
            if (up[uhd].dx < lo[lhd].dx) {
                // The knot moves to the upper boundary: z jumps up there.
                const Seg knot = up[uhd++];
                emit(knot, kUp);
                du -= knot.dy;
                dl -= knot.dy;
                lo[0] = Seg{dx, dl, dl / static_cast<double>(dx)};
                lhd = 0;
                ltl = 1;
            } else {
                const Seg knot = lo[lhd++];
                emit(knot, kDown);
                du -= knot.dy;
                dl -= knot.dy;
                up[0] = Seg{dx, du, du / static_cast<double>(dx)};
                uhd = 0;
                utl = 1;
            }
        }
    }

    // Both chains end at the pinned end point; the string follows the one
    // with interior vertices (the other is a single segment).
    if (dx == 0) return;
    if (utl - uhd > 1) {
        for (int64_t j = uhd; j < utl; ++j) emit(up[j], kUp);
    } else {
        for (int64_t j = lhd; j < ltl; ++j) emit(lo[j], kDown);
    }
}

struct TV1D final : RowOp<TV1D> {
    Vec wpad;  // the weights followed by a zero: radius(m - 1) of the last piece reads it
    int64_t nw;  // number of weights (1 or n - 1)
    double wmax;
    bool uniform, has_zero;
    fp_tv1d_method method;

    // wp holds the weights plus one trailing zero.
    TV1D(Vec wp, double mx, bool uni, bool zero, fp_tv1d_method m)
        : wpad(std::move(wp)), nw(static_cast<int64_t>(wpad.size()) - 1), wmax(mx), uniform(uni),
          has_zero(zero), method(m) {}

    const char *name() const noexcept override { return "tv1d"; }
    int64_t nparams() const noexcept override { return nw; }
    fp_status check_n(int64_t n) const noexcept override {
        return (nw == 1 || nw == n - 1) ? FP_OK : FP_ESHAPE;
    }
    size_t scratch_bytes(int64_t n) const noexcept override {
        return taut_scratch(n) + static_cast<size_t>(n + 1) * sizeof(double);  // chains, radii
    }
    bool needs_distinct() const noexcept override { return true; }

    double we(int64_t e) const noexcept { return nw == 1 ? wpad[0] : wpad[e]; }

    // z = x with the structure of the limit gamma -> 0+.
    template <class T>
    void identity(const T *x, T *z, int64_t n, uint8_t *code) const noexcept {
        std::memcpy(z, x, static_cast<size_t>(n) * sizeof(T));
        if (!code) return;
        for (int64_t i = 1; i < n; ++i) {
            if (we(i - 1) == 0.0) code[i] = kFree;
            else if (x[i] < x[i - 1]) code[i] = kDown;
            else if (x[i] > x[i - 1]) code[i] = kUp;
            else code[i] = kSame;
        }
    }

    // One piece of a weighted row (all edge weights positive): Condat's
    // algorithm with per-edge radii, falling back to the taut string. range
    // is max x - min x over the row (see inactive_radius). The clamped radii
    // are computed once into scratch: evaluating them inside the scan costs
    // as much as the scan itself.
    template <class T>
    void solve_weighted(const T *x, T *z, int64_t m, double gamma, const double *we, double range,
                        uint8_t *code, void *scr) const noexcept {
        solve_weighted(x, z, m, gamma, we, wmax, range, code, scr);
    }
    template <class T>
    void solve_weighted(const T *x, T *z, int64_t m, double gamma, const double *we, double wmax,
                        double range, uint8_t *code, void *scr) const noexcept {
        // Fast path: every radius is at most 2 range, below the smallest
        // clamping bound, and comparable to the data, so the scan reads
        // gamma w_e directly. we[m - 1] is the zero that ends the piece
        // (a zero-weight edge, or the padding after the last edge).
        if (gamma * wmax <= 2.0 * range) {
            auto edge = [gamma, we](int64_t e) { return gamma * we[e]; };
            if (method != FP_TV1D_TAUT_STRING) {
                const int64_t budget = method == FP_TV1D_CONDAT ? INT64_MAX : condat_budget(m);
                if (condat(x, z, m, edge, code, budget)) return;
                if (code) std::memset(code + 1, 0, static_cast<size_t>(m - 1));
            }
            taut_string(x, z, m, edge, code, scr);
            return;
        }
        double *rad = reinterpret_cast<double *>(static_cast<char *>(scr) + taut_scratch(m));
        bool all_inactive = true;
        for (int64_t e = 0; e + 1 < m; ++e) {
            const double j = static_cast<double>(std::min(e + 1, m - 1 - e));
            const double r = gamma * we[e];
            all_inactive = all_inactive && r >= j * range;
            rad[e] = std::min(r, 2.0 * j * range);  // inactive_radius
        }
        rad[m - 1] = 0.0;  // the end condition u_{m-1} = 0 is enforced separately
        if (all_inactive) {
            constant_mean(x, z, m);
            return;
        }
        const double *rp = rad;
        if (method != FP_TV1D_TAUT_STRING) {
            const int64_t budget = method == FP_TV1D_CONDAT ? INT64_MAX : condat_budget(m);
            if (condat(x, z, m, [rp](int64_t k) { return rp[k]; }, code, budget)) return;
            if (code) std::memset(code + 1, 0, static_cast<size_t>(m - 1));
        }
        taut_string(x, z, m, [rp](int64_t e) { return rp[e]; }, code, scr);
    }

    template <class T>
    int prox(const T *x, T *z, int64_t n, double gamma, void *scr, uint8_t *code) const noexcept {
        double lo, hi;
        if (!finite_range(x, n, lo, hi)) {
            bool nan;
            if (scan_row(x, n, nan) != kRowOk) return kRowNonFinite;
            if (code) std::memset(code, 0, static_cast<size_t>(n));
            fill_nan(z, n);
            return kRowOk;
        }
        if (code) std::memset(code, 0, static_cast<size_t>(n));
        const double range = hi - lo;
        if (gamma == 0.0 || range == 0.0) {  // z = x
            identity(x, z, n, code);
            return kRowOk;
        }
        if (uniform) {
            // No edge can be active: |u_e| <= min(e + 1, n - 1 - e) range.
            if (gamma * wpad[0] >= static_cast<double>(n / 2) * range) {
                constant_mean(x, z, n);
                return kRowOk;
            }
            const double lam = std::min(gamma * wpad[0], inactive_radius(n / 2, range));
            if (lam == 0.0) {
                identity(x, z, n, code);
                return kRowOk;
            }
            if (method != FP_TV1D_TAUT_STRING) {
                const int64_t budget = method == FP_TV1D_CONDAT ? INT64_MAX : condat_budget(n);
                if (condat(x, z, n, [lam](int64_t) { return lam; }, code, budget)) return kRowOk;
                if (code) std::memset(code, 0, static_cast<size_t>(n));
            }
            taut_string(x, z, n, [lam](int64_t) { return lam; }, code, scr);
            return kRowOk;
        }
        // Weighted: independent pieces between zero-weight edges.
        if (!has_zero) {
            solve_weighted(x, z, n, gamma, wpad.data(), range, code, scr);
            return kRowOk;
        }
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e == n - 1 || wpad[e] == 0.0) {
                solve_weighted(x + s, z + s, e - s + 1, gamma, wpad.data() + s, range,
                               code ? code + s : nullptr, scr);
                if (code) {
                    if (s > 0) code[s] = kFree;
                    if (e < n - 1) code[e + 1] = kFree;
                }
                s = e + 1;
            }
        }
        return kRowOk;
    }

    // Kinks (since 0.3): code 1 at entry e when the edge (e, e + 1) is
    // biactive, its jump within tol of 0 and its dual u_e = sum_{i <= e}
    // (x_i - z_i) within tol of its bound gamma w_e (edges of zero weight are
    // free, never kinks); 0 elsewhere, and at the last entry.
    bool has_kinks() const noexcept override { return true; }
    int64_t kink_row(const double *x, const double *z, int64_t n, double gamma, double tol, int8_t *mask,
                     int64_t ms, void *) const noexcept override {
        int64_t count = 0;
        double u = 0.0;
        for (int64_t e = 0; e < n; ++e) {
            int8_t code = 0;
            if (e + 1 < n) {
                u += x[e] - z[e];
                const double r = gamma * we(e);
                code = r > 0.0 && std::fabs(z[e + 1] - z[e]) <= tol && std::fabs(u) >= r - tol;
            }
            if (mask) mask[e * ms] = code;
            count += code;
        }
        return count;
    }

    // Branch overrides (since 0.3): at a kink (as kink_row), side -1 joins the
    // edge's two segments and +1 keeps them apart, with the jump's sign that
    // of its dual: the running sum s_e of x - z is -u_e, negative across a
    // step up (z = x - D^T u). These are the two pieces that meet there.
    bool has_branches() const noexcept override { return true; }
    void branch_row(const double *x, const double *z, uint8_t *code, int64_t n, double gamma, double tol,
                    const int8_t *side, void *) const noexcept override {
        double s = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) {
            s += x[e] - z[e];
            const double r = gamma * we(e);
            if (side[e] == 0 || !(r > 0.0) || !(std::fabs(z[e + 1] - z[e]) <= tol) || !(std::fabs(s) >= r - tol))
                continue;
            code[e + 1] = side[e] < 0 ? kSame : (s > 0.0 ? kDown : kUp);
        }
    }

    // The prox of gamma sum_e we[e] |z_{e+1} - z_e| for weights given per call
    // (we[n - 1] = 0 ends the row), as for this operator's own weights.
    int prox_edges(const double *x, double *z, int64_t n, double gamma, const double *we,
                   void *scr) const noexcept override {
        double lo, hi;
        if (!finite_range(x, n, lo, hi)) {
            bool nan;
            if (scan_row(x, n, nan) != kRowOk) return kRowNonFinite;
            fill_nan(z, n);
            return kRowOk;
        }
        const double range = hi - lo;
        double wm = 0.0;
        for (int64_t e = 0; e + 1 < n; ++e) wm = std::max(wm, we[e]);
        if (gamma == 0.0 || range == 0.0 || wm == 0.0) {
            std::memcpy(z, x, static_cast<size_t>(n) * sizeof(double));
            return kRowOk;
        }
        int64_t s = 0;
        for (int64_t e = 0; e < n; ++e) {
            if (e == n - 1 || we[e] == 0.0) {
                solve_weighted(x + s, z + s, e - s + 1, gamma, we + s, wm, range, nullptr, scr);
                s = e + 1;
            }
        }
        return kRowOk;
    }

    // Calls f(s, t, sl, sr) for every segment [s, t] with the dual signs at
    // its left edge (s-1) and right edge (t); 0 at the ends of the row.
    template <class F>
    static void segments(const uint8_t *code, int64_t n, F &&f) {
        int64_t s = 0;
        double sl = 0.0;
        for (int64_t i = 1; i <= n; ++i) {
            if (i == n || code[i] != kSame) {
                const double sr = i == n ? 0.0 : sigma_of(code[i]);
                f(s, i - 1, sl, sr);
                s = i;
                sl = sr;
            }
        }
    }

    template <class T>
    void jvp(const T *, const T *, const uint8_t *code, const T *dx, T *dz, int64_t n, double gamma,
             double dgamma, const double *dp, void *) const noexcept {
        segments(code, n, [&](int64_t s, int64_t t, double sl, double sr) {
            double num = 0.0;
            for (int64_t i = s; i <= t; ++i) num += static_cast<double>(dx[i]);
            if (s > 0) {
                num += sl * we(s - 1) * dgamma;
                if (dp) num += gamma * sl * dp[nw == 1 ? 0 : s - 1];
            }
            if (t < n - 1) {
                num -= sr * we(t) * dgamma;
                if (dp) num -= gamma * sr * dp[nw == 1 ? 0 : t];
            }
            const T v = static_cast<T>(num / static_cast<double>(t - s + 1));
            for (int64_t i = s; i <= t; ++i) dz[i] = v;
        });
    }

    template <class T>
    void vjp(const T *, const T *, const uint8_t *code, const T *g, T *gx, int64_t n, double gamma,
             double *acc, void *) const noexcept {
        const int64_t P = nparams();
        segments(code, n, [&](int64_t s, int64_t t, double sl, double sr) {
            double G = 0.0;
            for (int64_t i = s; i <= t; ++i) G += static_cast<double>(g[i]);
            const double m = G / static_cast<double>(t - s + 1);
            for (int64_t i = s; i <= t; ++i) gx[i] = static_cast<T>(m);
            if (s > 0 && sl != 0.0) {
                acc[P] += sl * we(s - 1) * m;
                acc[nw == 1 ? 0 : s - 1] += gamma * sl * m;
            }
            if (t < n - 1 && sr != 0.0) {
                acc[P] -= sr * we(t) * m;
                acc[nw == 1 ? 0 : t] -= gamma * sr * m;
            }
        });
    }
};

}  // namespace

fp_status make_tv1d(Vec w, fp_tv1d_method method, fp_op **out) {
    if (method != FP_TV1D_AUTO && method != FP_TV1D_CONDAT && method != FP_TV1D_TAUT_STRING)
        return FP_EPARAM;
    if (w.empty()) w.assign(1, 1.0);
    bool uniform = true, zero = false;
    double wmax = 0.0;
    for (double v : w) {  // one pass: validate, and find uniformity, zeros, max
        if (!finite_nonneg(v)) return FP_EPARAM;
        uniform = uniform && v == w[0];
        zero = zero || v == 0.0;
        wmax = v > wmax ? v : wmax;
    }
    w.push_back(0.0);
    *out = new TV1D(std::move(w), wmax, uniform, zero, method);
    return FP_OK;
}

}  // namespace fp
