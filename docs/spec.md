# Compiled prox library: design specification

Oct 14, 2026 · @David

## Summary

This spec defines a permissively licensed library of exact proximal operators and projections with derivatives. It fills gap 1 of the compiled-solvers survey: a maintained replacement for proxTV and Condat's C codes. Its core is C++17 behind a stable plain-C API, with Python bindings (nanobind) as the only binding for now. Julia, MATLAB and Rust bindings are deferred until after 1.0.

| Decision | Choice |
| --- | --- |
| Scope | 20 operators: separable convex maps, projections, 1D, multi-dimensional, graph and tree TV, nonconvex pointwise maps (Section 5) |
| Exactness | direct algorithms wherever they exist; a returned duality gap for the three iterative operators |
| Derivatives | JVP and VJP in the input and the parameters for every operator, mostly O(n) from saved structure (Section 7) |
| Core language | C++17: templates for float32 and float64, std::vector and RAII for max-flow and tree TV, no Eigen or Boost |
| Public boundary | one extern "C" header with plain C types; exceptions caught and returned as error codes |
| Threads | OpenMP over rows, bit-identical results for any thread count |
| Bindings | Python only for now (nanobind); Julia (ccall), MATLAB (mex) and Rust (thin crate) deferred until after 1.0 |
| Linux runtime | libstdc++ and libgcc linked statically, so the Python extension does not depend on the host's libstdc++ and future mex files can load inside MATLAB |
| License | MIT, with kernels written from the papers, not from GPL, CeCILL or unlicensed code |
| Releases | 0.1 core and 1D, 0.2 images, 0.3 graph TV, 1.0 stable, each behind a gate (Section 13) |

## Goals and non-goals

Version 1 is a small, exact and fast library of proximal operators and projections with derivatives. Its core is written in C++17 behind a stable plain-C public API (extern "C"). C and C++ programs call it directly, and Python reaches it through nanobind bindings over that API. Julia, MATLAB and Rust bindings are deferred; the plain-C API keeps them possible later without changes to the core. It is built to be the inner kernel of other people's solvers, not a solver framework.

**Goals:**

1. **Exact where exact algorithms exist.** Use direct algorithms (taut string, Condat, simplex projection, pool-adjacent-violators) instead of iterative ones whenever the literature has them, and certify the rest with a duality gap.
2. **Derivatives as first-class output.** Every operator provides Jacobian-vector and vector-Jacobian products of the prox with respect to its input and its parameters, so bilevel and implicit-differentiation codes can use it directly.
3. **Batched and parallel.** Apply an operator along any axis of an n-dimensional array, with rows processed in parallel and results identical for any number of threads.
4. **Permissive and maintained.** The MIT license, released as a C library with pkg-config and CMake files, and as PyPI wheels and conda-forge packages for Linux, macOS (Intel and ARM) and Windows. Bindings for other languages are deferred until after 1.0 (Section 13).
5. **A C++17 core behind a plain-C boundary.** Templates generate the float32 and float64 kernels from one source. RAII containers manage the max-flow and tree-TV data structures. The public header contains only C types, and no exception ever crosses it.
6. **Nonconvex penalties.** Pointwise prox maps for ℓq (0 < q < 1), ℓ0, SCAD, MCP and log penalties, with a documented choice when the prox is set-valued.
7. **No slower than the best existing code.** At least as fast as proxTV and Condat's C code on one thread, measured on the BenchOpt TV benchmarks.

**Non-goals for version 1:**

- **No solver framework.** No FISTA, PDHG or ADMM drivers beyond a reference loop for isotropic TV. Those belong in the user's code or in PyProximal and ProximalAlgorithms.jl.
- **No GPU kernels.** CPU only. The array interface is designed so that a GPU backend can be added later without changing the API.
- **No matrix-valued prox maps.** Nuclear norm and spectral functions need an SVD and are deferred to version 2.
- **No automatic modelling.** No parser or DSL for composite objectives, unlike ProxImaL.

## Use cases

Four kinds of caller drive the design. Each one needs something existing libraries lack: compiled speed, derivatives, or nonconvex penalties.

| Caller | What it calls | What it needs from the library |
| --- | --- | --- |
| Proximal splitting (FISTA, PDHG, ADMM, Condat–Vũ) | prox of g or of the conjugate g\* once per iteration, often on a whole image | speed on large arrays, in-place output, a conjugate prox through the Moreau identity |
| Bilevel learning of regularisation parameters | prox inside the lower-level solver, then derivatives of the solution with respect to the parameters | Jacobian-vector and vector-Jacobian products in the input and in the weights; for single-loop methods, one fused prox step per outer iteration |
| Sparse hyperparameter learning (Lasso-type models) | soft thresholding, group and weighted ℓ1, simplex and ℓ1-ball projection | the active set or support as a by-product, and Jacobians restricted to it |
| Nonconvex regularisation (DC methods, IRL1, ℓq penalties, quasinormed TV) | pointwise nonconvex prox maps, and weighted 1D or anisotropic TV as convex subproblems | correct thresholds, a defined tie-breaking rule, and weighted TV with per-edge weights |

- **Many small problems.** Bilevel and cross-validation loops call the operator thousands of times on small inputs. Per-call overhead must therefore be small: no Python-level loops over rows, and no array copies when the input is contiguous.
- **Batches of signals.** 1D TV is often applied to every row or column of an image (anisotropic TV by splitting), or to many independent signals. The batch axis is the natural unit of parallelism.

## Mathematical conventions

Every operator computes the same object: the proximal map of a function scaled by a step γ > 0, in the Euclidean norm. Projections are the case where the function is an indicator.

```latex
\operatorname{prox}_{\gamma g}(x) = \operatorname*{arg\,min}_{z \in \mathbb{R}^n} \; g(z) + \frac{1}{2\gamma}\|z - x\|_2^2
```

- **Step and weights.** The step γ and the regularisation weights are separate arguments. For TV, the weights are per edge (a vector w of length n − 1 in 1D), so `prox(x, gamma, w)` minimises γ ∑ w\_i |z\_{i+1} − z\_i| + ½‖z − x‖². A scalar weight is a broadcast vector.
- **Conjugates through the Moreau identity.** The prox of the conjugate is computed from the prox of g. This is what primal-dual methods call on the dual variable.

```latex
\operatorname{prox}_{\sigma g^*}(y) = y - \sigma \operatorname{prox}_{g/\sigma}(y/\sigma)
```

- **Nonconvex penalties.** For ℓq with q < 1, ℓ0, SCAD, MCP and the log penalty, the prox can be set-valued at the threshold. The library returns one element and documents which: the larger-magnitude minimiser by default (the nonzero branch), selectable by an argument. Every call reports whether the input sat on a tie.
- **Derivatives.** The prox is Lipschitz but in general only piecewise smooth. The library returns an element of a generalised Jacobian. Where the prox is differentiable, this is the Jacobian; at kinks it is a documented choice, the same element used by implicit differentiation codes such as sparse-ho (the support-restricted Jacobian). Section 7 gives the formula for each operator.
- **Arrays and axes.** Inputs are real arrays (float32 or float64). Separable operators act entrywise; 1D TV, projections and group norms act along one axis and are batched over the others; 2D and graph TV act on the last two axes or on a supplied graph.
- **Optimality certificate.** Iterative operators (isotropic TV, multi-dimensional TV by splitting) return the duality gap of the prox problem at the returned point. Exact operators return a gap of zero up to rounding.

## Operator catalogue

Version 1 ships 20 operators in five groups. Seventeen are exact, by closed form, a direct algorithm or max-flow to a set precision; the three iterative ones return a duality gap. The table lists each one with its algorithm, cost per call and the release that brings it (Section 13); n is the input length along the operator's axis.

| Operator | Function g | Algorithm | Cost | Exact | Release |
| --- | --- | --- | --- | --- | --- |
| **Separable convex** |  |  |  |  |  |
| Soft thresholding | weighted ℓ1 norm | closed form | O(n) | yes | 0.1 |
| Elastic net, Huber | ℓ1 + ½ℓ₂²; Huber function | closed form | O(n) | yes | 0.1 |
| Group soft thresholding | ℓ2,1 over given groups; ℓ2 norm | closed form | O(n) | yes | 0.1 |
| **Projections** |  |  |  |  |  |
| Box, nonnegative orthant, ℓ2 ball | indicator | closed form | O(n) | yes | 0.1 |
| Simplex | indicator of the probability simplex | Condat's method | linear in practice, quadratic worst case | yes | 0.1 |
| ℓ1 ball, weighted ℓ1 ball | indicator | simplex projection of the magnitudes | as simplex | yes | 0.1 |
| ℓ∞ norm | ‖x‖∞ | Moreau identity with the ℓ1-ball projection | as simplex | yes | 0.1 |
| Monotone cone | isotonic regression | pool-adjacent-violators | O(n) | yes | 0.1 |
| **1D total variation** |  |  |  |  |  |
| 1D TV | ∑ \|z\_{i+1} − z\_i\| | Condat's direct algorithm; taut string as a second path | O(n) for taut string; Condat linear in practice | yes | 0.1 |
| Weighted 1D TV | ∑ w\_i \|z\_{i+1} − z\_i\| | weighted taut string | linear in practice | yes | 0.1 |
| 1D TV with ℓ2 of differences | ‖Dz‖₂ | projected Newton on the dual | iterative | gap | 0.2 |
| **Multi-dimensional and graph TV** |  |  |  |  |  |
| Anisotropic TV in 2D and 3D | sum of 1D TV along each axis | parallel Douglas–Rachford or Dykstra over axes, 1D TV inside | iterative | gap | 0.2 |
| Isotropic TV in 2D and 3D | ∑ ‖(∇z)\_i‖₂ | fast gradient projection on the dual, warm-started | iterative | gap | 0.2 |
| Graph TV | ∑ w\_ij \|z\_i − z\_j\| on a given graph | parametric max-flow (divide and conquer over levels) | max-flow calls, about log of the value range divided by the precision | yes, to a set precision | 0.3 |
| TV on a tree or chain | weighted TV on tree edges | direct dynamic programming | near-linear | yes | 0.3 |
| **Nonconvex pointwise** |  |  |  |  |  |
| ℓ0 | number of nonzeros | hard thresholding | O(n) | yes | 0.2 |
| ℓ1/2 and ℓ2/3 | \|t\|^q | closed-form thresholding formulas | O(n) | yes | 0.2 |
| General ℓq, 0 < q < 1 | \|t\|^q | threshold test, then safeguarded Newton on the scalar stationarity equation | O(n) with a few Newton steps per entry | to machine precision | 0.2 |
| SCAD, MCP | Fan–Li and Zhang penalties | closed form (requires the concavity parameter large enough relative to γ) | O(n) | yes | 0.2 |
| Log penalty | log(1 + \|t\|/ε) | root of a quadratic, then comparison with zero | O(n) | yes | 0.2 |

- **Graph TV precision.** Parametric max-flow is exact for TV with finitely many levels. For real-valued data it returns a solution exact to a user-set precision, the approach of Chambolle and Darbon and of Hochbaum. The number of max-flow calls grows with the logarithm of the value range divided by that precision.
- **What is deliberately missing.** TGV, Hessian-Schatten norms and nonconvex TV (such as the TV quasinorm) have no exact prox. They are better handled by the user's splitting method on top of the operators above, using a weighted TV or a pointwise nonconvex prox as the inner step.

## Algorithms in detail

The implementation follows published direct algorithms and is written from the papers, not from existing code (Section 12 explains why). This section fixes the choices that matter for speed, exactness and derivatives.

**1D total variation**

- **Primary path: Condat's direct algorithm.** It scans the signal once, keeping running bounds on the current segment's value, and backtracks only when a bound is violated. It needs no auxiliary arrays beyond the output, and it is the fastest path on typical signals.
- **Second path: taut string.** It computes the taut string of the cumulative sum inside a tube of radius γw, and it is the path used for weighted TV. It has a linear worst case, so it is also the fallback when Condat's method backtracks too often on adversarial input. The choice is automatic, with a manual override.
- **Segments as output.** Both paths produce the solution as constant segments. The segment boundaries are returned on request, because the derivative in Section 7 is defined by them.
- **Weights.** Edge weights w\_i ≥ 0 are supported. A zero weight splits the problem into independent pieces, which are processed separately.

**Multi-dimensional anisotropic TV**

- **Splitting over axes.** Anisotropic TV in d dimensions is a sum of d separable 1D TV terms. The prox is computed by parallel Douglas–Rachford (or Dykstra's method) over the axes, following Barbero and Sra. Each inner step is a batch of exact 1D TV prox calls along one axis, run in parallel over rows.
- **Stopping.** Stop on the duality gap of the full prox problem, computed from the dual variables the splitting already carries. The default tolerance is relative to ‖x‖², and warm starts take the previous dual variables.

**Isotropic TV**

- **Dual method.** Isotropic TV has no direct algorithm. Use fast gradient projection on the dual (Beck and Teboulle), where each step is one divergence, one gradient and one projection onto pointwise ℓ2 balls, fused into a single pass over the image.
- **Warm starts and gap.** Accept and return the dual field so callers inside PDHG or bilevel loops can warm-start. Stop on the duality gap, which costs one extra gradient evaluation.

**Graph TV by parametric max-flow**

- **Exact anisotropic TV on any graph.** Thresholding the solution at a level t gives a minimum cut of a graph whose source and sink capacities depend on t. Solving the cut problem for a nested sequence of levels by divide and conquer recovers the solution to any set precision (Chambolle and Darbon, following Hochbaum).
- **Own max-flow.** A Boykov–Kolmogorov style augmenting-path max-flow, with tree reuse between levels, written from the published description. Kolmogorov's code is GPL and cannot be used.
- **Trees and chains.** Kolmogorov, Pock and Rolínek give direct algorithms for TV on trees, which generalise 1D TV and serve graphs with tree structure, such as superpixel hierarchies.

**Projections**

- **Simplex and ℓ1 ball.** Condat's method: a single pass with a running pivot and a filtered candidate list. Linear in practice, quadratic in the worst case, with no sort. The ℓ1 ball projects the magnitudes onto the simplex when the input lies outside the ball, then restores the signs. The weighted ℓ1 ball uses the same scheme with weighted thresholds.
- **Monotone cone.** Pool-adjacent-violators in one pass with a stack of blocks. The weighted version is supported.

**Nonconvex pointwise maps**

- **Threshold first.** For each penalty, compute the threshold below which the prox is zero. For ℓ0 it is √(2γλ); for ℓq there is a closed-form threshold depending on q, γ and λ.
- **Then the nonzero branch.** Above the threshold, solve the scalar stationarity equation. For q = 1/2 and q = 2/3, use the closed-form solutions. For general q, use Newton's method started from |x| and safeguarded by bisection. The equation is strictly monotone on the correct branch, so this converges in a few steps.
- **Ties.** At the threshold both 0 and the nonzero point are minimisers. Return the branch selected by the tie argument and set a flag.
- **SCAD and MCP.** Closed form on each piece, valid when the concavity parameter exceeds a bound depending on γ. Below that bound the prox stops being single-valued, and the call raises an error instead of returning a wrong answer.

## Derivatives for bilevel use

Every operator exposes Jacobian-vector products (JVP) and vector-Jacobian products (VJP), with respect to the input x and to the parameters (γ, weights, radius). Most of these Jacobians are structured: diagonal, block-averaging or a projection onto a subspace. A JVP or VJP then costs O(n), with no matrix formed. The forward call can save the small structure needed, such as the support, segments or active set, in a workspace object, so the derivative call does not recompute it.

| Operator | Jacobian in x at the returned point | Saved structure |
| --- | --- | --- |
| Soft thresholding, ℓ0, ℓq, SCAD, MCP, log | diagonal; zero on the thresholded set, otherwise 1 / (1 + γλ ψ''(z\_i)) from the scalar equation z + γλ ψ'(z) = x, for the penalty λψ | support mask |
| Group soft thresholding | block diagonal; zero on killed groups, otherwise (I − c(P\_⊥)) with P\_⊥ the projection orthogonal to the group's direction | live groups and their norms |
| Box and orthant | diagonal 0/1 | active bounds |
| ℓ2 ball | identity inside; outside, (r / ‖x‖)(I − x xᵀ / ‖x‖²) | inside flag, norm |
| Simplex | on the support S: I − 1 1ᵀ / \|S\|; zero elsewhere | support |
| ℓ1 ball | signed version of the simplex Jacobian on the magnitudes when active, else identity | support, signs |
| Monotone cone | block averaging over the pooled blocks | block boundaries |
| 1D TV, weighted 1D TV | averaging over the constant segments of the solution | segment boundaries |
| Anisotropic, isotropic and graph TV | no simple closed form | the dual variables; derivatives by implicit differentiation of the dual optimality conditions, solved with conjugate gradients |

- **Parameter derivatives.** For 1D TV, the derivative in γ or in w\_i is constant on each segment and depends only on the segment length, the weights at its two ends and the signs of the jumps there, so it is also O(n). For thresholding maps, the derivative in λ is −γ ψ'(z\_i) / (1 + γλ ψ''(z\_i)) on the support.
- **Kinks.** At points where the prox is not differentiable (an entry exactly at a threshold, a segment boundary with zero jump), the library returns the Jacobian of the active structure it selected, consistent with the tie rule in Section 4. Callers that need a full Clarke Jacobian are out of scope.
- **Iterative operators.** For isotropic and multi-dimensional TV, the VJP solves a linear system on the dual active set by conjugate gradients, warm-started and stopped at a user tolerance. The returned residual tells bilevel codes with inexactness control, such as MAID-type methods, how accurate the hypergradient is.
- **Verification.** Every derivative is tested against finite differences away from kinks, and the JVP and VJP are tested for adjointness: ⟨v, J u⟩ = ⟨Jᵀ v, u⟩ to rounding.

## Architecture

The library is a C++17 core behind one plain-C public header (extern "C"), with four internal layers. Every language binding is a thin wrapper over that C API. A fix or speed-up in a kernel therefore reaches every language in the same release, and the C ABI is the only contract to keep stable.

[embedded content: library layers · bindings, C API, batching, kernels, support]

Calls flow down: a binding converts its array type to a strided descriptor, the batching layer splits the work into rows, and each row goes to a kernel that works on contiguous memory.

**Why C++17 inside and plain C outside**

- **A plain-C boundary for reach.** A C ABI is callable from every language this community uses (Python, Julia, MATLAB, R, Fortran, Rust) without a runtime, and C++ name mangling and STL layouts differ between compilers. So the public header `fastprox.h` is C: no classes, no templates, no STL types, no references, wrapped in `extern "C"`.
- **C++17 inside for safety and genericity.** Function templates on the scalar type generate the float32 and float64 kernels from one source. `std::vector` and RAII own the variable-size structures (max-flow graphs, search trees, tree-TV heaps), so error paths cannot leak. `constexpr` and `if constexpr` replace preprocessor tricks.
- **No exceptions across the boundary.** Internal code may throw, including `std::bad_alloc` from containers. Every extern "C" entry point wraps its body in `try { … } catch (const std::bad_alloc&) { return FP_ENOMEM; } catch (...) { return FP_EINTERNAL; }`, so an exception never reaches a C, Julia, MATLAB or Rust caller. Hot kernels are `noexcept` and do not allocate.
- **No heavy dependencies.** No Eigen, no Boost and no BLAS. The operators need only loops over contiguous memory, and avoiding these libraries keeps builds simple and avoids version conflicts inside MATLAB and Python processes.
- **Static libstdc++ on Linux.** The shared library links libstdc++ and libgcc statically (`-static-libstdc++ -static-libgcc`) and hides all non-API symbols. MATLAB ships its own, often older, libstdc++, and a mex file that needs a newer one fails to load; static linking removes that conflict. macOS (libc++) and Windows (MSVC runtime) use the platform runtime.
- **Threads.** OpenMP over rows inside the C++ core; the C API exposes only a thread count.

**Source layout**

- `include/fastprox.h` — the only public header, plain C: types, error codes, operator constructors, the prox, JVP and VJP entry points, and version macros.
- `src/api/` — the extern "C" layer: argument checks, dtype dispatch to the template instantiations, and the try/catch wrappers that turn exceptions into error codes.
- `src/core/` — strided descriptors, row iteration, OpenMP dispatch, workspace arena (C++).
- `src/ops/` — one header and source per operator family (pointwise, projections, tv1d, tvtree, tvnd, graphtv), each a `template <class T>` kernel on contiguous data plus its derivative, explicitly instantiated for `float` and `double`.
- `src/maxflow/` — the max-flow engine used by graph TV, built on `std::vector` and RAII.
- `bindings/python` — the nanobind extension, a thin wrapper over the C API with no numerical code. Directories for Julia, MATLAB and Rust are added only when those bindings are scheduled.
- `tests/`, `bench/`, `fuzz/` — unit and property tests through the C API, benchmark drivers, fuzzing harnesses.

**Build**

- **CMake** (3.20 or later) as the only build, exporting a `fastprox::fastprox` target and a pkg-config file. OpenMP is optional; without it the library runs serially with identical results.
- **Compiler baseline:** C++17 for the core and C99 for the public header, tested with GCC, Clang and MSVC. Symbols are hidden by default (`-fvisibility=hidden`), and only the `fp_` API is exported.
- **Python:** nanobind, built with scikit-build-core over the same CMake project; wheels with cibuildwheel.
- **Header check in CI:** `fastprox.h` is compiled by a C99 compiler with `-Werror` and by a C++ compiler, so a C++-only construct in the public header fails the build.

## API

The C API has three generic entry points (prox, JVP, VJP) that work on an operator handle, so bindings wrap three functions instead of 60. Operators are created once with their fixed parameters (weights, radius, q, graph). The step γ is passed on every call because splitting methods change it between iterations.

**C header (sketch)**

```c
/* fastprox.h -- plain C99, implemented in C++17, no STL types */
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define FP_API __declspec(dllimport)   /* dllexport when building */
#else
#  define FP_API __attribute__((visibility("default")))
#endif

typedef enum { FP_F32 = 1, FP_F64 = 2 } fp_dtype;
typedef enum { FP_DEVICE_CPU = 0 } fp_device;  /* GPU values reserved */

typedef struct {            /* strided n-d array, strides in elements */
    void          *data;
    fp_dtype       dtype;
    int            ndim;
    const int64_t *shape;
    const int64_t *strides;
    fp_device      device;   /* must be FP_DEVICE_CPU in 1.x */
} fp_array;

typedef enum {
    FP_OK = 0, FP_EINVAL, FP_EDTYPE, FP_ESHAPE,
    FP_ENOMEM,              /* std::bad_alloc caught at the API layer */
    FP_ENOTCONV,            /* iterative op stopped before tol */
    FP_EPARAM,              /* e.g. SCAD parameter too small for gamma */
    FP_EINTERNAL            /* any other C++ exception, caught */
} fp_status;

typedef struct fp_op fp_op;               /* opaque; wraps a C++ object */
typedef struct fp_workspace fp_workspace; /* opaque; owns std::vector scratch */

typedef struct {
    int    axis;            /* operator axis for batched operators */
    int    threads;         /* 0 = OpenMP default, 1 = serial */
    double tol;             /* relative duality-gap tolerance */
    int    max_iter;
    int    tie;             /* FP_TIE_NONZERO (default) or FP_TIE_ZERO */
    int    keep_structure;  /* save support/segments for jvp and vjp */
} fp_options;

typedef struct { double gap; int iters; int64_t ties; } fp_info;

/* constructors return a status and write the handle through out */
FP_API fp_status fp_op_soft(double lambda, const fp_array *weights, fp_op **out);
FP_API fp_status fp_op_tv1d(const fp_array *edge_weights, fp_op **out);
FP_API fp_status fp_op_simplex(double radius, fp_op **out);
FP_API fp_status fp_op_lq(double lambda, double q, fp_op **out);
FP_API fp_status fp_op_graph_tv(int64_t n, int64_t m, const int64_t *edges,
                                const double *w, fp_op **out);
FP_API void      fp_op_free(fp_op *op);

FP_API fp_status fp_workspace_create(const fp_op *op, const fp_array *like,
                                     const fp_options *opt, fp_workspace **out);
FP_API void      fp_workspace_free(fp_workspace *ws);

/* the three entry points; out may alias x */
FP_API fp_status fp_prox(const fp_op *op, double gamma, const fp_array *x,
                         fp_array *out, const fp_options *opt,
                         fp_workspace *ws, fp_info *info);
FP_API fp_status fp_prox_conj(const fp_op *op, double sigma, const fp_array *y,
                              fp_array *out, const fp_options *opt,
                              fp_workspace *ws, fp_info *info);
FP_API fp_status fp_jvp(const fp_op *op, const fp_workspace *ws,
                        const fp_array *dx, double dgamma, fp_array *dout);
FP_API fp_status fp_vjp(const fp_op *op, const fp_workspace *ws,
                        const fp_array *g, fp_array *gx, double *ggamma,
                        fp_array *gparams);

FP_API const char *fp_strerror(fp_status s);
FP_API int         fp_abi_version(void);

#ifdef __cplusplus
}
#endif
```

**C API rules**

- **Boundary pattern.** Each extern "C" function checks its arguments, dispatches on `dtype` to `kernel<float>` or `kernel<double>`, and wraps the call in try/catch. `std::bad_alloc` becomes FP\_ENOMEM and any other exception becomes FP\_EINTERNAL. Constructors return a status and write the handle through an out-pointer, so allocation failures are reported, not hidden in a NULL.
- **Opaque handles.** `fp_op` and `fp_workspace` are incomplete types in C; inside, they are C++ classes holding `std::vector` members and released by their destructors in `fp_op_free` and `fp_workspace_free`. Callers never see an STL type, a reference or a template.
- **Ownership.** The library never frees caller memory. Operators copy their parameter arrays at creation. Workspaces own all scratch memory and are reused across calls, so a loop of prox calls allocates nothing after the first.
- **Threads.** `fp_op` is immutable and safe to share. `fp_workspace` is not shared between concurrent calls; it holds one scratch region per OpenMP thread. Results are bit-identical for any value of `threads`.
- **Errors.** Every function returns `fp_status`; there are no global error states, no printing and no aborts. Invalid parameters (negative γ, q outside (0, 1), a SCAD parameter too small for γ) return FP\_EPARAM before any work.
- **Derivatives.** `fp_jvp` and `fp_vjp` read the structure saved in the workspace by the last `fp_prox` call with `keep_structure` set. Calling them without it returns FP\_EINVAL.
- **ABI stability.** Structs passed by pointer carry no hidden fields; new options are added only through new functions or a versioned options struct. The major version changes only when the ABI breaks. The device field of fp\_array is reserved from 0.1 so a GPU backend can be added without an ABI break; 1.x accepts only FP\_DEVICE\_CPU and returns FP\_EINVAL for any other value.

**Python interface (sketch)**

```python
import fastprox as fp

# one-shot functions, NumPy in and out, float32 or float64, any axis
z = fp.tv1d(x, gamma=0.5, weights=w, axis=-1, out=None)
z = fp.project_simplex(x, radius=1.0, axis=-1)
z = fp.lq(x, lam=0.1, q=0.5, gamma=1.0, tie="nonzero")
z, info = fp.tv_iso(img, gamma=0.1, tol=1e-6, warm_start=p0)   # info.gap, info.dual

# reusable operator with derivatives, for bilevel code
op = fp.TV1D(weights=w)
z, ctx = op.prox(x, gamma, keep=True)
dz = op.jvp(ctx, dx)
gx, grads = op.vjp(ctx, g)          # grads.gamma, grads.weights

# autodiff integration (CPU), built on the same calls
from fastprox.torch import tv1d as tv1d_torch   # torch.autograd.Function
from fastprox.jax import tv1d as tv1d_jax       # jax.custom_vjp + pure_callback
```

- **nanobind module.** The Python extension is written with nanobind and calls only the C API, never the C++ internals, so Python sees exactly what other languages see. `nb::ndarray` accepts C-contiguous and strided NumPy arrays without copying. Other dtypes are converted only when the caller sets `copy=True`; otherwise a `TypeError` is raised.
- **GIL.** Released (`nb::gil_scoped_release`) for every call into the C API, so Python threads can run operators concurrently on separate workspaces.
- **Errors.** A non-OK `fp_status` becomes a Python exception (`ValueError` for parameter errors, `MemoryError` for FP\_ENOMEM, `RuntimeError` otherwise) with the text of `fp_strerror`.

**Other languages (deferred).** Julia (ccall through a JLL package), MATLAB (C mex gateways) and Rust (`fastprox-sys` plus a thin safe crate) are not built for now. Because Python calls only the public C API, those bindings can be added later without changing the core or the ABI.

## Numerical and performance requirements

Each requirement below has a measurable target and a named check, so a release can be accepted or rejected on evidence. Speed targets are relative to the best existing code because absolute times depend on the machine.

| Requirement | Target | How it is checked |
| --- | --- | --- |
| Accuracy of exact operators | optimality residual at most 10 n ε ‖x‖∞, with ε the machine epsilon of the dtype | optimality conditions checked on every test input (Section 11) |
| Accuracy of iterative operators | duality gap at most tol · ½‖x‖², returned in `fp_info.gap` | gap recomputed independently in tests |
| Determinism | bit-identical output for any thread count, on the same build and machine | tests run with 1, 2 and 8 threads and compare bytes |
| Dtypes | float32 and float64; float32 accumulates sums in float64 where a sum spans a row | float32 results compared with float64 within the dtype's tolerance |
| Special values | NaN in the input gives NaN in the affected output only, never a hang; ±Inf rejected with `FP_EINVAL` for operators where it is undefined | fuzzing with NaN, Inf, denormals, empty and length-1 inputs |
| Allocation | no allocation inside `fp_prox`, `fp_jvp` or `fp_vjp` after the workspace is created | allocator hook counts calls in tests |
| Single-thread speed, 1D TV | no slower than Condat's C code and proxTV on the BenchOpt TV-1D problems | `bench/` drivers against both, from n = 10³ to 10⁷ |
| Single-thread speed, simplex and ℓ1 ball | no slower than Condat's C code | same drivers, uniform and adversarial inputs |
| Parallel scaling | at least 0.7 × threads speed-up on batches of at least 4 × threads rows of length 10⁴ or more | scaling runs on 1 to 16 threads |
| Call overhead from Python | under 5 µs per call for n = 100 | `pytest-benchmark` on the one-shot functions |
| Worst case | no input makes an exact operator slower than O(n²); 1D TV falls back to taut string (O(n)) on detected backtracking | adversarial generators in `fuzz/` |

- **Floating-point model.** Build without `-ffast-math`. Reordering sums would break determinism and the exactness of the direct algorithms, which compare running bounds.
- **Edge cases with defined output.** Zero weights split 1D TV into independent pieces. γ = 0 returns the input. A radius of zero returns the zero vector for both the ℓ1 ball and the simplex; a negative radius is rejected with FP\_EPARAM.
- **Targets to confirm.** The overhead (5 µs) and scaling (0.7) figures are design targets, not measurements. Version 0.1 should measure them and either confirm or revise them.

## Testing and verification

Correctness is checked from the mathematics, not only against other code. Every operator has a certificate a test can verify without a reference solver: its optimality conditions, or a duality gap. Reference solvers are used as a second line on small inputs.

| Layer | What is tested | Tools |
| --- | --- | --- |
| Optimality | x − z ∈ γ ∂g(z), checked through the operator's dual certificate: the dual vector for 1D TV, the threshold for the simplex, sign and magnitude conditions for thresholding | C unit tests on random, structured and adversarial inputs |
| Reference agreement | output matches a generic solver to 10⁻⁸ on inputs up to n = 200 | CVXPY with Clarabel, run offline to generate fixtures stored in the repository |
| Cross-implementation | Condat and taut-string paths agree; the C library agrees with proxTV and Condat's C code where their scope overlaps | comparison scripts in `bench/`, not part of the build |
| Prox properties | firm nonexpansiveness for convex operators, ‖z₁ − z₂‖² ≤ ⟨z₁ − z₂, x₁ − x₂⟩; the Moreau identity; idempotence of projections | property-based tests (Hypothesis on the Python side) |
| Nonconvex maps | the returned point minimises the scalar objective, checked against a dense grid plus local refinement; tie flags set exactly at thresholds | grid search in tests |
| Derivatives | JVP against central finite differences away from kinks; ⟨v, Ju⟩ = ⟨Jᵀv, u⟩ to rounding; parameter derivatives against finite differences | C tests and `torch.autograd.gradcheck` through the PyTorch wrapper |
| Memory and undefined behaviour | no leaks, overflows or undefined behaviour | AddressSanitizer, UndefinedBehaviorSanitizer and Valgrind in CI |
| Robustness | no crash or hang on arbitrary bytes as input arrays and parameters | libFuzzer harnesses in `fuzz/`, run nightly |
| Determinism | identical bytes across thread counts | CI matrix with `threads` = 1, 2, 8 |
| Performance | no regression larger than 5% between releases | benchmark suite on a fixed runner, compared with the previous tag |

- **CI matrix.** Linux (GCC, Clang), macOS (Clang, ARM and Intel) and Windows (MSVC), each in float32 and float64, with OpenMP on and off.
- **BenchOpt.** Solvers using the library are contributed to the TV-1D and TV-2D benchmarks, so its speed is measured publicly against the Python solvers already there.

## Licensing, packaging and integration

The library is permissively licensed and written from the papers, because the best existing codes are GPL, CeCILL, research-only or unlicensed. Every language package is built from one CMake project and one C++ source tree.

**Licensing**

- **License:** MIT for the library and all bindings.
- **Clean-room rule.** GPL code (Kolmogorov's maxflow, PyMaxflow), CeCILL code (Condat's C files) and code with no license (Chambolle's graphTV, Kolmogorov's TVTree) is not read while writing the corresponding kernel, and is never copied or linked. Each kernel cites the paper it implements in its header comment. proxTV (BSD-style) may be consulted, with attribution where an idea comes from its code.
- **Research-only code is excluded.** gco-v3.0 is not used even for testing, because its license restricts use to research.
- **Benchmarks against copyleft code** run from scripts in `bench/` that fetch and build those codes on the benchmark machine. They are never vendored or shipped in a release.
- **Static libstdc++.** Linking libstdc++ and libgcc statically into an MIT-licensed library is permitted by the GCC Runtime Library Exception.

**Packages**

| Target | Package | Built with | Notes |
| --- | --- | --- | --- |
| C and C++ | `libfastprox` (shared and static), headers, CMake config, pkg-config | CMake | conda-forge feedstock; Linux build links libstdc++ statically |
| Python | `fastprox` on PyPI and conda-forge | nanobind and scikit-build-core, wheels by cibuildwheel | manylinux\_2\_28 (x86\_64, aarch64), macOS (arm64, x86\_64), Windows (amd64); stable-ABI (abi3) wheels only, requires-python 3.12 or later; no per-version wheels |

Julia (`fastprox_jll` through BinaryBuilder), MATLAB (a `.mltbx` toolbox with C mex gateways) and Rust (`fastprox-sys` on crates.io) packages are deferred with their bindings.

- **OpenMP in wheels.** Linux wheels bundle libgomp through auditwheel. macOS wheels bundle libomp through delocate, because Apple Clang does not ship it.
- **Symbols.** Only `fp_` symbols are exported on every platform (a linker version script on Linux, `__declspec` on Windows, visibility on macOS), so two copies of the library or of libstdc++ in one process cannot clash.

**Integration**

- **BenchOpt:** solvers using the library in the TV-1D and TV-2D benchmarks, and a new simplex-projection benchmark.
- **PyProximal:** an adapter subclassing its `ProxOperator`, so PyProximal's solvers can call the compiled operators.
- **PyTorch and JAX:** `torch.autograd.Function` wrappers and `jax.custom_vjp` with `pure_callback` (CPU), both using the library's VJP rather than differentiating through iterations.

## Roadmap

Four releases take the library from the 1D core to a stable 1.0. Each one ships only when its gate passes. No dates are set: the order is fixed, and the pace depends on who builds it.

[embedded content: roadmap · four releases, one gate each, not to scale]

Release 0.1 is highlighted because it already serves most callers: thresholding, projections and 1D TV with derivatives, from Python. The C++17 core, the extern "C" boundary and the static-libstdc++ Linux build are all in place from 0.1, so later releases add operators and bindings without changing the ABI.

- **0.1 — core and 1D.** The C API and its try/catch boundary, separable convex maps, projections and pool-adjacent-violators, 1D TV by Condat and taut string with weights, JVP and VJP for all of these, and Python wheels built with nanobind.
- **0.2 — images.** Anisotropic and isotropic TV in 2D and 3D with duality gaps and warm starts, 1D TV with ℓ2 of differences, and the nonconvex pointwise maps, all exposed in the Python package.
- **0.3 — graph TV.** The max-flow engine, graph TV and tree TV, VJPs for the iterative operators by conjugate gradients, and the PyTorch and JAX wrappers.
- **1.0 — stable.** ABI freeze, conda-forge packages, documentation, and the PyProximal adapter. After 1.0, only additions that keep the ABI are allowed until 2.0.
- **After 1.0 — other languages.** Julia (ccall), MATLAB (mex, with a gate that the mex files load on Linux next to MATLAB's own libstdc++) and the thin Rust crate, each over the frozen C API, plus a ProximalOperators.jl adapter.

## Decisions

The six questions left open in the first draft were settled on 14 October 2026, and the binding scope was narrowed to Python the same day; the sections above reflect both. Work on 0.1 can start; nothing in the design is still waiting on a decision.

| Question | Decision | Where it applies |
| --- | --- | --- |
| Device field for a later GPU backend | reserved now: `fp_array` carries an `fp_device` field, and 1.x accepts only `FP_DEVICE_CPU` | API header and ABI rules (Section 9) |
| License | MIT for the library and all bindings | Goals, Licensing (Section 12), Summary |
| Name | `fastprox`; not taken on PyPI | all packages |
| Default tie rule for nonconvex prox maps | nonzero branch, with ties always reported | Conventions (Section 4), `fp_options.tie` |
| OpenMP in macOS wheels | bundle libomp with delocate | Packages (Section 12) |
| Python versions | CPython 3.12 or later only, as stable-ABI (abi3) wheels | Packages (Section 12) |
| Bindings to build now | Python only (nanobind); Julia, MATLAB and Rust deferred until after 1.0 | Summary, Goals, Architecture, API (Section 9), Packages (Section 12), Roadmap (Section 13) |

- **Decided earlier:** a C++17 core behind an extern "C" header, templates for float32 and float64, RAII containers for max-flow and tree TV, no exceptions across the boundary, no STL types in the header, no Eigen or Boost, static libstdc++ on Linux, and nanobind for the Python bindings.
- **Still to check before 0.1:** that `fastprox` is free on conda-forge. crates.io and the Julia General registry matter only when those bindings start.

## References

The papers below are the sources each kernel is implemented from. The first group was checked against publisher or Crossref records during the survey that preceded this spec; the second is cited from standard knowledge and should be checked before the kernel headers cite it.

**Checked**

- L. Condat, [A direct algorithm for 1-D total variation denoising](https://doi.org/10.1109/LSP.2013.2278339), IEEE Signal Processing Letters 20(11), 2013 — 1D TV.
- L. Condat, [Fast projection onto the simplex and the ℓ1 ball](https://doi.org/10.1007/s10107-015-0946-6), Mathematical Programming 158, 2016 — projections.
- P. L. Davies and A. Kovac, [Local extremes, runs, strings and multiresolution](https://doi.org/10.1214/aos/996986501), Annals of Statistics 29(1), 2001 — taut string.
- Á. Barbero and S. Sra, [Modular proximal optimization for multidimensional total-variation regularization](http://www.jmlr.org/papers/v19/13-538.html), JMLR 19, 2018 — weighted 1D TV and multi-dimensional splitting.
- A. Chambolle and J. Darbon, [On total variation minimization and surface evolution using parametric maximum flows](https://doi.org/10.1007/s11263-009-0238-9), IJCV 84(3), 2009 — graph TV.
- D. S. Hochbaum, An efficient algorithm for image segmentation, Markov random fields and related problems, J. ACM 48(4), 2001 — parametric divide and conquer.
- Y. Boykov and V. Kolmogorov, [An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision](https://doi.org/10.1109/TPAMI.2004.60), IEEE TPAMI 26(9), 2004 — max-flow engine.
- V. Kolmogorov, T. Pock and M. Rolínek, [Total variation on a tree](https://doi.org/10.1137/15M1010257), SIAM J. Imaging Sci. 9(2), 2016 — tree TV.
- A. Chambolle, [An algorithm for total variation minimization and applications](https://doi.org/10.1023/B:JMIV.0000011325.36760.1e), JMIV 20, 2004 — dual method for isotropic TV.
- Q. Bertrand et al., [Implicit differentiation for fast hyperparameter selection in non-smooth convex learning](https://www.jmlr.org/papers/v23/21-0486.html), JMLR 23, 2022 — support-restricted Jacobians.

**To check before citing in code**

- A. Beck and M. Teboulle, fast gradient-based algorithms for constrained TV denoising and deblurring, IEEE Trans. Image Processing, 2009 — FGP for isotropic TV.
- Z. Xu et al., L1/2 regularization: a thresholding representation theory and a fast solver, IEEE Trans. Neural Networks and Learning Systems, 2012 — ℓ1/2 thresholding.
- W. Cao, J. Sun and Z. Xu, closed-form thresholding formulas for ℓq with q = 1/2 and 2/3, J. Visual Communication and Image Representation, 2013 — ℓ2/3 thresholding.
- J. Fan and R. Li, SCAD penalty, JASA, 2001; C.-H. Zhang, MCP penalty, Annals of Statistics, 2010 — SCAD and MCP prox maps.
- Tooling documentation (nanobind, scikit-build-core, cibuildwheel, BinaryBuilder, the GCC Runtime Library Exception) was not opened while writing this spec; confirm the details in Section 12 against it.

The survey this spec builds on: [Compiled solvers for variational imaging and nonsmooth optimization](https://claude.ai/code/artifact/51657fed-21f0-4ccc-b9ed-2d47aa2a97f3).
