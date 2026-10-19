/*
 * fastprox.h -- exact proximal operators and projections with derivatives.
 *
 * Plain C99 public interface; the library itself is C++17. No STL types,
 * classes, references or templates appear here, and no exception crosses
 * this boundary: every function reports failure through fp_status.
 *
 * Conventions (docs/spec.md, Section 4):
 *   prox_{gamma g}(x) = argmin_z g(z) + 1/(2 gamma) ||z - x||^2,
 * applied along one axis of an n-d strided array and batched over the
 * others. Projections ignore gamma. Arrays are float32 or float64, strides
 * are counted in elements, and `out` may alias the input.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef FASTPROX_H
#define FASTPROX_H

#include <stddef.h>
#include <stdint.h>

#define FP_VERSION_MAJOR 0
#define FP_VERSION_MINOR 3
#define FP_VERSION_PATCH 0
#define FP_ABI_VERSION 0 /* bumped only when the ABI breaks */

#define FP_MAX_NDIM 32

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(FASTPROX_STATIC)
#    define FP_API
#  elif defined(FASTPROX_BUILDING)
#    define FP_API __declspec(dllexport)
#  else
#    define FP_API __declspec(dllimport)
#  endif
#else
#  define FP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* FP_I8 (since 0.2) is accepted only by the kink and branch arrays of
 * fp_workspace_kinks and fp_workspace_set_branch. */
typedef enum { FP_F32 = 1, FP_F64 = 2, FP_I8 = 3 } fp_dtype;
typedef enum { FP_DEVICE_CPU = 0 } fp_device; /* GPU values reserved */

/* Strided n-d array descriptor. Strides are in elements, not bytes, and may
 * be negative. ndim may be 0 (a scalar). The library never frees `data`. */
typedef struct {
    void          *data;
    fp_dtype       dtype;
    int            ndim;
    const int64_t *shape;
    const int64_t *strides;
    fp_device      device; /* must be FP_DEVICE_CPU in 1.x */
} fp_array;

typedef enum {
    FP_OK = 0,
    FP_EINVAL,   /* invalid argument, non-finite input, missing structure */
    FP_EDTYPE,   /* unsupported or mismatched dtype */
    FP_ESHAPE,   /* mismatched shapes or parameter lengths */
    FP_ENOMEM,   /* std::bad_alloc caught at the API layer */
    FP_ENOTCONV, /* iterative operator stopped before tol */
    FP_EPARAM,   /* invalid operator parameter or step */
    FP_EINTERNAL /* any other C++ exception, caught */
} fp_status;

typedef enum { FP_TIE_NONZERO = 0, FP_TIE_ZERO = 1 } fp_tie;

typedef enum {
    FP_TV1D_AUTO = 0,       /* Condat, taut string when Condat backtracks too much */
    FP_TV1D_CONDAT = 1,     /* Condat only (quadratic on adversarial inputs) */
    FP_TV1D_TAUT_STRING = 2
} fp_tv1d_method;

typedef struct fp_op fp_op;               /* opaque, immutable, shareable */
typedef struct fp_workspace fp_workspace; /* opaque, one per concurrent caller */

typedef struct {
    int    axis;           /* operator axis; negative counts from the end */
    int    threads;        /* 0 = fp_default_threads(), 1 = serial */
    double tol;            /* relative duality-gap tolerance (iterative ops) */
    int    max_iter;       /* iteration cap (iterative ops) */
    int    tie;            /* fp_tie, for nonconvex ops */
    int    keep_structure; /* save structure in the workspace for jvp/vjp */
} fp_options;

typedef struct {
    double  gap;   /* duality gap; 0 for exact operators */
    int     iters; /* iterations; 0 for direct algorithms */
    int64_t ties;  /* entries that sat on a tie (nonconvex ops) */
} fp_info;

/* Defaults: axis -1, threads 0, tol 1e-6, max_iter 1000, tie nonzero,
 * keep_structure 0. Passing opt = NULL anywhere uses these. */
FP_API void fp_options_init(fp_options *opt);

/* ------------------------------------------------------------------------
 * Operator constructors. Parameter arrays are copied, any dtype, and must be
 * one-dimensional. Each operator has a parameter vector theta whose length is
 * fp_op_nparams(op); gradients with respect to theta are what fp_vjp returns
 * in gparams and what fp_jvp reads from dparams.
 * ---------------------------------------------------------------------- */

/* g(z) = lambda * sum_i w_i |z_i|. weights: NULL (w = 1), length 1, or the
 * axis length. theta = w. */
FP_API fp_status fp_op_soft(double lambda, const fp_array *weights, fp_op **out);

/* g(z) = l1 ||z||_1 + (l2 / 2) ||z||^2. theta = (l1, l2). */
FP_API fp_status fp_op_elastic_net(double l1, double l2, fp_op **out);

/* g(z) = lambda * sum_i h_delta(z_i), h_delta(t) = t^2 / (2 delta) for
 * |t| <= delta and |t| - delta / 2 otherwise. theta = (lambda, delta). */
FP_API fp_status fp_op_huber(double lambda, double delta, fp_op **out);

/* g(z) = lambda * sum_g c_g ||z_g||_2 along the axis. groups: NULL for one
 * group spanning the axis (the l2 norm), else n group ids in [0, ngroups).
 * group_weights: NULL (c = 1), length 1, or ngroups. theta = c. */
FP_API fp_status fp_op_group_l2(double lambda, int64_t n, const int64_t *groups,
                                int64_t ngroups, const fp_array *group_weights,
                                fp_op **out);

/* g(z) = l1 ||z||_1 + sum_g eta_g ||z_g||_2 along the axis for groups that are
 * pairwise nested or disjoint (since 0.2); the sparse group lasso is a
 * partition. Group g holds the entries group_idx[group_ptr[g] ..
 * group_ptr[g+1]), in any order of the groups; ngroups >= 0, group_ptr has
 * ngroups + 1 entries from 0, indices >= 0 and below the axis length, none
 * repeated in a group. FP_EPARAM when two groups overlap without nesting.
 * group_weights: NULL (eta = 1), length 1, or ngroups. theta = (l1, eta...). */
FP_API fp_status fp_op_tree_l2(double l1, int64_t ngroups, const int64_t *group_ptr,
                               const int64_t *group_idx, const fp_array *group_weights,
                               fp_op **out);

/* Projection onto lo <= z <= hi. lower/upper: NULL to use the scalar, or an
 * array of length 1 or the axis length; infinite bounds allowed.
 * theta = (lower..., upper...). */
FP_API fp_status fp_op_box(double lo, double hi, const fp_array *lower,
                           const fp_array *upper, fp_op **out);

/* Projection onto z >= 0. No parameters. */
FP_API fp_status fp_op_nonneg(fp_op **out);

/* g(z) = -lambda * sum_i log z_i, lambda > 0 (since 0.2): z = (x + sqrt(x^2 +
 * 4 gamma lambda)) / 2. theta = (lambda). */
FP_API fp_status fp_op_log_barrier(double lambda, fp_op **out);

/* g(z) = lambda * sum_i w_i |z_i| restricted to lo <= z <= hi (since 0.2):
 * z = clamp(soft(x, gamma lambda w), lo, hi). weights as fp_op_soft, the box
 * as fp_op_box. theta = (w..., lower..., upper...). */
FP_API fp_status fp_op_l1_box(double lambda, const fp_array *weights, double lo, double hi,
                              const fp_array *lower, const fp_array *upper, fp_op **out);

/* Projection onto ||z||_2 <= radius. theta = (radius). */
FP_API fp_status fp_op_l2_ball(double radius, fp_op **out);

/* Projection onto { z >= 0, sum z = radius } (Condat 2016). theta = (radius). */
FP_API fp_status fp_op_simplex(double radius, fp_op **out);

/* Projection onto sum_i w_i |z_i| <= radius. weights: NULL (w = 1), length 1
 * or the axis length, all > 0, with min/max >= 1e-100. theta = (radius). */
FP_API fp_status fp_op_l1_ball(double radius, const fp_array *weights, fp_op **out);

/* g(z) = lambda ||z||_inf, through the Moreau identity with the l1 ball.
 * theta = (lambda). */
FP_API fp_status fp_op_linf(double lambda, fp_op **out);

/* g(z) = lambda max_i z_i along the axis (since 0.2): z = min(x, tau), with tau
 * the threshold of the projection onto the simplex of radius gamma lambda
 * (Moreau). theta = (lambda). */
FP_API fp_status fp_op_max(double lambda, fp_op **out);

/* g(z) = lambda * (sum of the k largest z_i) along the axis (since 0.2), k >= 1
 * and at most the axis length (else FP_EPARAM). absolute != 0: the k largest
 * |z_i|, for any k >= 1 (k >= the axis length gives lambda ||z||_1).
 * theta = (lambda). */
FP_API fp_status fp_op_sum_largest(double lambda, int64_t k, int absolute, fp_op **out);

/* Projection onto { a^T z = b, lo <= z <= hi } along the axis (since 0.2);
 * inequality != 0: a^T z <= b. a: NULL (a = 1), length 1 or the axis length,
 * finite, not all zero, with nonzero |a_i| / max |a| >= 1e-100. lower/upper
 * as fp_op_box (-INFINITY and INFINITY for no box). FP_EPARAM when the set is
 * empty for the axis length. theta = (b, a..., lower..., upper...). */
FP_API fp_status fp_op_hyperplane(const fp_array *a, double b, double lo, double hi,
                                  const fp_array *lower, const fp_array *upper, int inequality,
                                  fp_op **out);

/* Projection onto the second-order cone { (t, y) : ||y||_2 <= t } along the
 * axis, t the first entry (since 0.2). No parameters. */
FP_API fp_status fp_op_soc(fp_op **out);

/* Projection onto the monotone cone (isotonic regression by PAVA), minimising
 * sum_i w_i (z_i - x_i)^2. weights: NULL or the axis length, all > 0.
 * increasing: nonzero for z_1 <= ... <= z_n. theta = w (empty if NULL). */
FP_API fp_status fp_op_isotonic(const fp_array *weights, int increasing, fp_op **out);

/* g(z) = sum_i w_i |z_{i+1} - z_i| along the axis. edge_weights: NULL (w = 1),
 * length 1, or axis length - 1, all >= 0. theta = w. */
FP_API fp_status fp_op_tv1d(const fp_array *edge_weights, fp_tv1d_method method,
                            fp_op **out);

/* Projection onto { z : sum_i w_i |z_{i+1} - z_i| <= alpha } along the axis
 * (since 0.2), alpha >= 0: x when inside, else the 1D TV prox at the step
 * where the TV equals alpha, found by safeguarded Newton steps (exact on the
 * final structure). edge_weights as fp_op_tv1d. theta = (alpha, w...). */
FP_API fp_status fp_op_tv1d_ball(double alpha, const fp_array *edge_weights, fp_op **out);

/* g(z) = ||W D z||_2 = sqrt(sum_i w_i^2 (z_{i+1} - z_i)^2) along the axis
 * (since 0.2): 1D TV with the l2 norm of the differences. The dual is a
 * trust-region problem, solved by Newton's method on its secular equation with
 * one tridiagonal solve per step, to rounding (fp_info.gap is 0).
 * edge_weights as fp_op_tv1d, with w_i^2 finite. theta = w. */
FP_API fp_status fp_op_tv1d_l2(const fp_array *edge_weights, fp_op **out);

/* Nonconvex pointwise maps (elementwise). The prox can be set-valued at a
 * threshold: fp_options.tie picks the nonzero branch (FP_TIE_NONZERO, the
 * default) or 0 (FP_TIE_ZERO), and fp_info.ties counts the entries that sat
 * on a tie. fp_prox_conj applies the Moreau formula y - sigma prox(y/sigma),
 * which for these functions is not the prox of the conjugate. */

/* g(z) = lambda * sum_i w_i [z_i != 0] (hard thresholding at sqrt(2 gamma
 * lambda w_i)). weights: NULL (w = 1), length 1, or the axis length.
 * theta = w (the map is piecewise constant in it). */
FP_API fp_status fp_op_l0(double lambda, const fp_array *weights, fp_op **out);

/* g(z) = lambda * sum_i w_i |z_i|^q, 0 < q < 1 (closed form for q = 1/2 and
 * 2/3, safeguarded Newton otherwise). weights as fp_op_l0. theta = w. */
FP_API fp_status fp_op_lq(double lambda, double q, const fp_array *weights, fp_op **out);

/* g(z) = sum_i w_i SCAD(z_i), SCAD (Fan and Li) with threshold lambda and
 * concavity a > 1; entry i is the unweighted map at the step gamma w_i. The
 * prox needs a > 1 + gamma max w, else FP_EPARAM. weights: NULL (w = 1,
 * theta = (lambda, a)), or length 1 or the axis length, all >= 0
 * (theta = (lambda, a, w...)). Weights since 0.2. */
FP_API fp_status fp_op_scad(double lambda, double a, const fp_array *weights, fp_op **out);

/* g(z) = sum_i w_i MCP(z_i), MCP (Zhang) with threshold lambda and
 * concavity b > 0 (w MCP_{lambda, b} = MCP_{w lambda, b / w}). The prox needs
 * b > gamma max w, else FP_EPARAM. weights and theta as fp_op_scad, with b
 * for a. */
FP_API fp_status fp_op_mcp(double lambda, double b, const fp_array *weights, fp_op **out);

/* g(z) = lambda * sum_i log(1 + |z_i| / eps), eps > 0. theta = (lambda, eps). */
FP_API fp_status fp_op_log(double lambda, double eps, fp_op **out);

/* Group penalties g(z) = sum_g c_g h(||z_g||_2) along the axis (since 0.2),
 * with h the scalar map of fp_op_huber, fp_op_scad or fp_op_mcp: the prox
 * scales each group by prox_{gamma c_g h}(||x_g||) / ||x_g|| (norm
 * composition). groups, ngroups and group_weights as fp_op_group_l2. SCAD
 * needs a > 1 + gamma max c, MCP b > gamma max c (else FP_EPARAM).
 * theta = (lambda, delta | a | b, c...). */
FP_API fp_status fp_op_group_huber(double lambda, double delta, int64_t n, const int64_t *groups,
                                   int64_t ngroups, const fp_array *group_weights, fp_op **out);
FP_API fp_status fp_op_group_scad(double lambda, double a, int64_t n, const int64_t *groups,
                                  int64_t ngroups, const fp_array *group_weights, fp_op **out);
FP_API fp_status fp_op_group_mcp(double lambda, double b, int64_t n, const int64_t *groups,
                                 int64_t ngroups, const fp_array *group_weights, fp_op **out);

/* Anisotropic TV of an image: g(z) = sum over the last ndim axes a of
 * w_a * (1D TV of z along a), for ndim = 2 or 3. It acts on the last ndim axes
 * and is batched over the others (fp_options.axis is ignored). axis_weights:
 * NULL (w = 1), length 1, or ndim, all >= 0. theta = (w_0, ..., w_{ndim-1}).
 * Iterative: fp_info reports the duality gap, and the call stops when the gap
 * of every image is at most tol * ||x||^2 / 2 (FP_ENOTCONV after max_iter,
 * with the last iterate written). Its dual is kept in the workspace. */
FP_API fp_status fp_op_tv_aniso(int ndim, const fp_array *axis_weights, fp_op **out);

/* Anisotropic TV with edge weights (since 0.3):
 *     g(z) = sum_a w_a sum_p beta_{a,p} |z_{p + e_a} - z_p|,
 * as fp_op_tv_aniso otherwise. edge_weights: an array of shape (ndim, image
 * shape), all finite and >= 0, beta_{a,p} weighting the edge from p to its
 * forward neighbour along a (the entries at the last position along a are
 * unused); the operator then takes only images of that shape (FP_ESHAPE
 * otherwise). A zero weight leaves an edge free. theta = (w_0, ...,
 * w_{ndim-1}, beta in C order), the unused entries with zero gradients. */
FP_API fp_status fp_op_tv_aniso_weighted(int ndim, const fp_array *axis_weights, const fp_array *edge_weights,
                                         fp_op **out);

/* Isotropic TV of an image: g(z) = sum_p |(w_0 D_0 z, ..., w_{ndim-1} D_{ndim-1} z)_p|_2
 * with forward differences D_a along the last ndim axes (2 or 3; zero past the
 * last entry), batched over the others; the weights scale each axis inside
 * the norm (e.g. voxel spacing). axis_weights: NULL (w = 1), length 1, or
 * ndim, all >= 0. theta = (w_0, ..., w_{ndim-1}). Iterative, as
 * fp_op_tv_aniso. The dual u in the workspace has |u_p|_2 <= gamma and
 * z = x - sum_a w_a D_a^T u_a. Derivatives solve a linear system by conjugate
 * gradients (fp_workspace_set_deriv_tol, fp_workspace_deriv_info). */
FP_API fp_status fp_op_tv_iso(int ndim, const fp_array *axis_weights, fp_op **out);

/* Isotropic TV with pixel weights (since 0.3):
 *     g(z) = sum_p alpha_p |(w_0 D_0 z, ..., w_{ndim-1} D_{ndim-1} z)_p|_2,
 * as fp_op_tv_iso otherwise. pixel_weights: an array of ndim axes, the image
 * shape, all finite and >= 0; the operator then takes only images of that
 * shape (FP_ESHAPE otherwise). The dual has |u_p|_2 <= gamma alpha_p; a zero
 * weight leaves a pixel's differences free. theta = (w_0, ..., w_{ndim-1},
 * alpha in C order). */
FP_API fp_status fp_op_tv_iso_weighted(int ndim, const fp_array *axis_weights, const fp_array *pixel_weights,
                                       fp_op **out);

/* Graph TV (since 0.3): g(z) = sum_e w_e |z_u - z_v| over the m edges
 * e = (u, v) of a graph on n nodes. It acts on the last axis, of length n,
 * and is batched over the others (fp_options.axis is ignored). edges: 2m
 * node indices (u_0, v_0, u_1, v_1, ...) with u != v; repeated edges add.
 * edge_weights: NULL (w = 1), length 1, or m, all >= 0. theta = w.
 * By parametric max-flow, each part of the graph cut at its mean value: the
 * result is exact up to rounding where a part ends at a trivial cut, else
 * within tol * (max x - min x) of the solution at every node. fp_info.gap is
 * the duality gap with the flow as the dual, iters the number of max-flow
 * solves (FP_ENOTCONV after max_iter, every unfinished part at its mean).
 * The workspace dual (shape (1, *x.shape)) is q = D^T f, with z = x - q at
 * the solution; warm starts have no effect. jvp and vjp are exact: averages
 * over the regions where z is constant. */
FP_API fp_status fp_op_graph_tv(int64_t n, int64_t m, const int64_t *edges,
                                const fp_array *edge_weights, fp_op **out);

/* TV on a tree or forest (since 0.3): fp_op_graph_tv's g, n, edges and
 * edge_weights, for edges that form a forest (no cycles, no repeated edges;
 * FP_EPARAM otherwise), in any order and orientation; a chain is a tree. By
 * dynamic programming (Kolmogorov, Pock and Rolinek), O(n log n): exact up
 * to rounding, tol and max_iter are ignored, fp_info.iters is 1 and gap the
 * duality gap with the dual flow, which on a tree is determined by z. The
 * workspace dual and the derivatives are those of fp_op_graph_tv. */
FP_API fp_status fp_op_tree_tv(int64_t n, int64_t m, const int64_t *edges,
                               const fp_array *edge_weights, fp_op **out);

FP_API void        fp_op_free(fp_op *op);
FP_API int64_t     fp_op_nparams(const fp_op *op);
FP_API const char *fp_op_name(const fp_op *op);

/* ------------------------------------------------------------------------
 * Workspaces: scratch memory and saved structure for one array shape, dtype
 * and axis. With a workspace, fp_prox, fp_jvp and fp_vjp do not allocate.
 * ---------------------------------------------------------------------- */
FP_API fp_status fp_workspace_create(const fp_op *op, const fp_array *like,
                                     const fp_options *opt, fp_workspace **out);
FP_API void      fp_workspace_free(fp_workspace *ws);

/* Iterative image operators keep their dual in the workspace: an array of
 * shape (ndim, *x.shape) whose blocks q_a are feasible for the conjugate of
 * the term along axis a, with z = x - sum_a q_a at the solution. get copies
 * out the dual of the last call (FP_EINVAL before any call); set seeds it
 * (FP_EINVAL if it is not finite), and the next call starts from it; with
 * set_warm(ws, 1) every call starts from the dual left by the previous one.
 * A starting dual that is not feasible is first made so (clipped to the
 * operator's bounds), so any seed gives a valid certificate. Other
 * operators: FP_EINVAL. */
FP_API fp_status fp_workspace_get_dual(const fp_workspace *ws, fp_array *out);
FP_API fp_status fp_workspace_set_dual(fp_workspace *ws, const fp_array *dual);
FP_API fp_status fp_workspace_set_warm(fp_workspace *ws, int on);

/* Derivatives of iterative image operators that solve a linear system
 * (isotropic TV, by conjugate gradients): the relative residual tolerance
 * and iteration cap of fp_jvp/fp_vjp (defaults 0 and 10000; a tolerance of 0,
 * since 0.3, stops each image at the relative error of z that its gap
 * certifies, sqrt(2 gap) / |x|, and at 1e-10 at the tightest), and the
 * worst residual and iteration count of the last fp_jvp/fp_vjp over the
 * batch (0 for derivatives in closed form). Other operators: FP_EINVAL. */
FP_API fp_status fp_workspace_set_deriv_tol(fp_workspace *ws, double tol, int max_iter);
FP_API fp_status fp_workspace_deriv_info(const fp_workspace *ws, double *residual, int *iters);

/* Kinks of the elementwise maps soft thresholding, l1 over a box, SCAD and
 * MCP (since 0.2), at the point saved by the last fp_prox or fp_prox_conj
 * call with keep_structure (FP_EINVAL before one, and for other operators).
 * An entry is at a kink when it lies within tol (absolute, in the units of
 * the input x or y) of a breakpoint of its scalar map; the code says which,
 * the nearest one:
 *   soft thresholding  1: |x| = gamma lambda w
 *   l1 over a box      1: |x| = gamma lambda w, 2: soft(x) = lower,
 *                      3: soft(x) = upper
 *   SCAD               1: |x| = gamma w lambda, 2: |x| = (1 + gamma w) lambda,
 *                      3: |x| = a lambda
 *   MCP                1: |x| = gamma w lambda, 2: |x| = b lambda
 *   l0 (since 0.3)     1: the tie |x| = sqrt(2 gamma lambda w)
 *   lq (since 0.3)     1: the tie |x| = tau
 *   log (since 0.3)    1: the tie, or |x| = gamma lambda / eps where the map
 *                      is continuous (gamma lambda <= eps^2)
 *   box (since 0.4)    1: x = lower, 2: x = upper; orthant 1: x = 0
 *   simplex (0.4)      1: x_i at the threshold tau
 *   l1 ball (0.4)      1: |x_i| at tau w_i; 2 at every entry of a row on the
 *                      sphere, sum w_i |x_i| = r (within tol max w)
 *   group l2 (0.4)     1 at every entry of a group with ||x_g|| at gamma
 *                      lambda c_g
 * (breakpoints of a map that is the identity, gamma w = 0 or lambda = 0, are
 * not kinks). Since 0.3 also, from the saved solution and dual:
 *   1D TV              1 at entry i when the edge (i, i + 1) has no jump
 *                      (|z_{i+1} - z_i| <= tol) and its dual sum_{j <= i}
 *                      (x_j - z_j) is within tol of its bound gamma w_i
 *   anisotropic TV     bit a (value 1 << a) at pixel p for such an edge
 *                      (p, p + e_a), its 1D dual within tol of gamma w_a beta
 *   isotropic TV       1 at a pixel with |(W D z)_p| <= tol whose dual is
 *                      within tol of its sphere, |u_p| >= gamma alpha_p - tol
 * (edges and pixels of zero bound are free, not kinks).
 * fp_workspace_kinks writes the codes into mask (FP_I8, the shape
 * of x, any strides; NULL for the count only) and the number of entries at a
 * kink into *count (may be NULL).
 * The mask is the same for any thread count.
 *
 * fp_workspace_set_branch (elementwise operators, and since 0.3 1D and
 * anisotropic TV; FP_EINVAL for isotropic TV) chooses, for the entries at a
 * kink (with its own
 * tol), which piece fp_jvp and fp_vjp differentiate: side (FP_I8, the shape
 * of x, values -1, 0, +1) picks the piece on the x - eps side (-1) or the
 * x + eps side (+1) of the breakpoint, with its formula taken at x; 0, and
 * every entry not at a kink, keeps the default rule (docs/api.md,
 * "Derivatives at kinks"). So side +1 gives the derivative of the
 * one-sided difference (prox(x + h e_i) - prox(x)) / h, -1 that from below:
 * the elements of the B-Jacobian. For 1D and anisotropic TV, side -1 at the
 * tail of a kink's edge joins its two sides and +1 keeps them apart, the jump
 * taking the sign of the edge's dual. For the simplex and the l1 ball (since
 * 0.4) +1 at a threshold entry puts it on the support and -1 off it; a row
 * on the l1 sphere, and a group of group l2 at its threshold, follow the
 * side at their first entry (-1: inside, or the group zeroed; +1: the
 * projection with tau = 0, or the shrink). It applies to the fp_jvp and fp_vjp calls
 * on this workspace until the next prox call or fp_workspace_set_branch(ws,
 * NULL, 0). */
FP_API fp_status fp_workspace_kinks(const fp_workspace *ws, double tol, fp_array *mask,
                                    int64_t *count);
FP_API fp_status fp_workspace_set_branch(fp_workspace *ws, const fp_array *side, double tol);

/* ------------------------------------------------------------------------
 * Entry points. ws may be NULL unless opt->keep_structure is set; out may
 * alias x. On error the contents of out are unspecified.
 * ---------------------------------------------------------------------- */
FP_API fp_status fp_prox(const fp_op *op, double gamma, const fp_array *x,
                         fp_array *out, const fp_options *opt,
                         fp_workspace *ws, fp_info *info);

/* prox_{sigma g*}(y) = y - sigma prox_{g/sigma}(y / sigma). */
FP_API fp_status fp_prox_conj(const fp_op *op, double sigma, const fp_array *y,
                              fp_array *out, const fp_options *opt,
                              fp_workspace *ws, fp_info *info);

/* dout = J_x dx + (dz/dgamma) dgamma + (dz/dtheta) dparams at the point saved
 * by the last fp_prox or fp_prox_conj call with keep_structure (gamma reads
 * as sigma after fp_prox_conj). dx and dparams may be NULL (zero). */
FP_API fp_status fp_jvp(const fp_op *op, const fp_workspace *ws,
                        const fp_array *dx, double dgamma,
                        const fp_array *dparams, fp_array *dout);

/* gx = J_x^T g, *ggamma = <g, dz/dgamma>, gparams = (dz/dtheta)^T g, summed
 * over the batch. ggamma and gparams may be NULL. */
FP_API fp_status fp_vjp(const fp_op *op, const fp_workspace *ws,
                        const fp_array *g, fp_array *gx, double *ggamma,
                        fp_array *gparams);

FP_API const char *fp_strerror(fp_status s);
FP_API int         fp_abi_version(void);
FP_API const char *fp_version_string(void);
/* Threads used when fp_options.threads is 0: the environment variable
 * FASTPROX_NUM_THREADS, else the first number in OMP_NUM_THREADS, else the
 * CPUs the process may run on (read once). 1 in builds without threads. */
FP_API int         fp_default_threads(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTPROX_H */
