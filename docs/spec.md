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
- **No GPU kernels in the C core for 1.0.** The C library is CPU only up to 1.0. Before that, the PyTorch and JAX wrappers run the iterative image operators on the input's device, written in framework tensor operations (0.3). The array interface reserves a device field, so a GPU backend in the C core can follow in 1.x without changing the API. Its design is in Section 8, "GPU backend".
- **No SVD or eigendecomposition in the core.** Matrix prox maps such as the nuclear norm need one, so they come in 1.x with the decomposition supplied by the caller (Section 5, "Spectral operators"). Planned at first for version 2; revised 20 October 2026.
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

### Spectral operators (1.x, design)

A spectral operator applies one of the operators above to the singular values or eigenvalues of a matrix. Added in 1.x as new functions, it keeps the 1.0 ABI. The caller computes the decomposition with the framework it already uses (NumPy, PyTorch or JAX, on the CPU or the GPU), and also forms the matrix products. The core does what autograd through a decomposition gets wrong: the prox on the spectrum, and its exact derivatives where singular values repeat or sit on a kink. The core stays free of LAPACK and BLAS (Section 8).

**Two kinds.**

- **Singular values** of X ∈ ℝ^{m×n}, with X = U diag(s) Vᵀ the thin SVD, k = min(m, n), and g(X) = h(s). The prox is prox\_{γg}(X) = U diag(z) Vᵀ with z = prox\_{γh}(s). For convex h this needs h to be absolutely symmetric, that is invariant under permutations and sign changes of its argument (Beck, Theorem 7.29).
- **Eigenvalues** of a symmetric X = Q diag(s) Qᵀ, with g(X) = h(s) and h invariant under permutations. Then prox\_{γg}(X) = Q diag(prox\_{γh}(s)) Qᵀ (Beck, Theorem 7.18). An input that is not symmetric is replaced by (X + Xᵀ)/2, and the derivatives are those of that map.
- **Nonconvex penalties.** The same formula holds for a separable h with any lower-bounded, even scalar penalty (Lu, Zhu, Xu, Yan and Lin 2015, Theorems 1 and 2). The proof uses von Neumann's trace inequality, and the scalar prox of such a penalty is monotone. Where it is set-valued, equal singular values must get equal values. fastprox's tie rule is a function of the value (Section 4), so it satisfies this.

**Which operators qualify.** The core checks each operator against the kind; unweighted forms and scalar parameters are needed for the symmetry:

| Kind | Operators | Gives |
| --- | --- | --- |
| Singular values | soft thresholding, elastic net, Huber, ℓ0, ℓq, SCAD, MCP, log | nuclear norm, nonconvex low-rank penalties |
| | ℓ1 ball; ℓ∞ prox; box with lower = −upper; ℓ2 ball | nuclear-norm ball; spectral norm; spectral-norm ball; Frobenius ball |
| Eigenvalues | the above, and the nonnegative orthant, box, simplex, log barrier | PSD cone; eigenvalue bounds; spectraplex (PSD with trace r); −log det |

Each operator reports whether it is permutation invariant and whether it is even. Weighted variants qualify only with equal weights. Others, such as isotonic regression and 1D TV, are rejected with FP\_EINVAL. Not covered are symmetric penalties with no vector prox in fastprox, such as the truncated nuclear norm.

**Derivatives.** Let z = f(s) be the prox on the spectrum and J its Jacobian, which fp\_jvp already provides. Take the SVD kind with A = Uᵀ H V (k × k), and let A\_s and A\_k be its symmetric and skew parts. The derivative of the spectral map in the direction H is Theorem 3.3 of Ding, Sun, Sun and Toh (2018), equation (54), written there for m ≤ n. The map is differentiable at X exactly when f is differentiable at s, and continuously so exactly when f is (their Theorems 3.3 and 3.4). The eigenvalue case is Lewis and Sendov (2001), with Daleckii and Krein (1965) behind both.

D(H) = U B Vᵀ + (I − U Uᵀ) H V diag(r) Vᵀ,  B = diag(J diag(A)) + Γ⁺ ∘ A\_s + Γ⁻ ∘ A\_k

- The off-diagonal coefficients are Γ⁺\_ij = (z\_i − z\_j) / (s\_i − s\_j) and Γ⁻\_ij = (z\_i + z\_j) / (s\_i + s\_j), and the diagonals of Γ± are zero.
- The complement ratio is r\_i = z\_i / s\_i. Its term is present only for m > n; for m < n the roles of U and V swap.
- The eigenvalue kind has A = Qᵀ H Q, B = diag(J diag(A)) + Γ⁺ ∘ A, D(H) = Q B Qᵀ, and no Γ⁻ or r.

**Ties and zeros, exactly.** The quotients above have no value where s\_i = s\_j (for Γ⁻, s\_i = s\_j = 0; for r, s\_i = 0). Their limits come from J, as in Ding et al.'s fallback values, equations (49)–(52):
- Γ⁺\_ij → J\_ii − J\_ij;
- Γ⁻\_ij → J\_ii and r\_i → J\_ii at zero, since J\_ij = 0 whenever s\_i = 0 and i ≠ j (their Lemma 3.2).

On a group of equal values, permutation invariance gives J the form a I + b 1 1ᵀ there, so one fp\_jvp with a unit vector per group yields every limit in it.

**Near ties, in floating point.** A quotient (z\_i − z\_j)/(s\_i − s\_j) loses about ε·|z| / |s\_i − s\_j| to rounding, while a limit used at a distance δ from a tie is off by O(δ). The average of the two one-sided limits, for instance (f′(s\_i) + f′(s\_j))/2 when f is separable, is off by only O(δ²), and is exact at a tie. So a pair counts as tied when |s\_i − s\_j| ≤ ε^{1/3} · max\_l |s\_l|, and then takes ((J\_ii − J\_ij) + (J\_jj − J\_ji))/2. This is the usual balance between rounding and truncation, as in Golub and Van Loan, Section 2.7.

Checked in rational arithmetic on a smooth nonlinear f, the worst relative error over all gaps is about 2·10⁻¹¹. With a √ε band and one limit it would be about 10⁻⁸. Inside a cluster the coefficients are then equal, so B does not depend on the singular vectors the cluster leaves undetermined; Golub and Van Loan's Section 9.1.5 gives the same advice for functions of matrices with near-multiple eigenvalues. Outside the band the quotients are the exact derivative, kinks between s\_i and s\_j included. At a kink of f itself (s\_i on a threshold), J is the one-sided Jacobian of the structure the prox selected, as for every operator (Section 7), and the limits follow from it.

**Adjoint and parameters.** Γ⁺ and Γ⁻ are symmetric, and J is symmetric for every qualifying operator, so the map H ↦ D(H) is self-adjoint in the Frobenius inner product. The VJP of a cotangent G is therefore D(G). The derivatives in γ and the operator's parameters touch only the diagonal: dγ adds the γ-derivative of f to diag(B), and the parameter gradients are fp\_vjp of the operator on diag(Uᵀ G V).

**C API.** The decomposition and the products U B Vᵀ stay with the caller. The core takes the spectrum, its prox and the k × k matrix A, and returns B and r. That costs O(k²) per matrix, plus one O(k) JVP per tie group. The prox on the spectrum is plain fp\_prox on s, shape (..., k), with keep\_structure set.

```c
typedef enum { FP_SPECTRAL_SVD = 0, FP_SPECTRAL_EIG = 1 } fp_spectral_kind;

/* FP_OK if op can act on a spectrum of this kind, else FP_EINVAL. */
FP_API fp_status fp_op_spectral_check(const fp_op *op, fp_spectral_kind kind);

/* After fp_prox(op, gamma, s -> z) with keep_structure: B (..., k, k) and,
 * for the SVD kind, r (..., k; may be NULL) from A (..., k, k); dgamma and
 * dparams as in fp_jvp. Called with a cotangent's A it gives the VJP's B,
 * since the map is self-adjoint. */
FP_API fp_status fp_spectral_jvp(const fp_op *op, const fp_workspace *ws,
                                 fp_spectral_kind kind, const fp_array *s,
                                 const fp_array *z, const fp_array *a,
                                 double dgamma, const fp_array *dparams,
                                 fp_array *b, fp_array *r);
```

The workspace built for the prox on s also holds the spectral scratch, about 8k doubles per thread. It is sized by the first fp\_spectral\_jvp call and kept, so later calls allocate nothing. Sizing it in fp\_workspace\_create instead would add 16 bytes per entry to the workspace of every qualifying operator, soft thresholding on a long vector included, whether or not it is ever used on a spectrum. Rows of B are independent, and the result is bit-identical for any thread count.

**Python.**
- **Constructors:** `fp.Spectral(op, kind="svd" | "eigh")` wraps a qualifying operator. `fp.NuclearNorm(lam)`, `fp.SpectralNormBall(r)` and `fp.PSD()` are shorthands.
- **`prox`, `jvp`, `vjp` and `value`** take matrices of shape (..., m, n), batched like every other operator. They call `numpy.linalg.svd` or `eigh` and the BLAS products themselves.
- **`prox_from(U, s, Vh, gamma)`** is for a caller that already has the decomposition.
  - **Truncated decompositions:** it may also take only the leading triplets, as large matrix-completion codes compute them (PROPACK in Stella, Themelis and Patrinos 2017). This is valid when the caller knows the omitted values map to zero, as in singular value thresholding below γλ.
  - **Derivatives** need the full thin decomposition, because Γ± couple every pair.
- **`fastprox.torch` and `fastprox.jax`** get the same operators, with the decomposition and products in the framework, on the input's device. B is computed on the CPU from A, which costs a k × k round trip until a device path follows, as image TV had in 0.3.
- **Batches of tiny matrices** are a case to serve well. Total nuclear variation, the nuclear norm of each pixel's Jacobian (3 × 2 for a colour image), is solved through its dual, which projects millions of such matrices onto the spectral-norm ball per iteration. One call covers the whole batch, so the per-call cost is shared. A closed-form 2 × 2 and 3 × 2 SVD in the core could later remove the framework round trip there.

**Tests.**
- **Prox:** against a reference built from NumPy's SVD, and optimality checked through `value`: g(Z) + ‖Z − X‖²/(2γ) at Z against trial points.
- **Derivatives:**
  - JVP against finite differences on random matrices away from ties.
  - The same on matrices built with repeated and zero singular values, where the map is differentiable whenever f is.
    - **The autograd contrast:** PyTorch documents that SVD gradients "will only be finite when A does not have repeated singular values". On 7 × 5 matrices with s = (2, 2, 2, 1.5, 1.5) and soft thresholding at 0.7, it returns NaN where a computed gap is exactly 0. Where rounding leaves the values a few ulps apart, the gradient is finite but 28–35% off (checked on six random draws). The test shows the contrast.
    - **Near ties:** pairs at gaps from 10⁻² down to 10⁻¹² of the scale, against the exact divided difference.
  - **The divergence:** the trace of D, against the closed form of Candès, Sing-Long and Trzasko (2013): |m − n| Σ z\_i/s\_i + Σ f′(s\_i) + 2 Σ\_{i≠j} s\_i z\_i / (s\_i² − s\_j²). Checked before review to 12 digits on 6 × 6 and 9 × 5.
  - Adjointness ⟨G, D(H)⟩ = ⟨D(G), H⟩.
  - Parameter gradients against finite differences.
- **Shapes and dtypes:** square and rectangular (m < n, m > n), batched, float32 and float64.
- **Determinism:** the threading test, with bit-identical output for any thread count.

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
- **No exceptions across the boundary.** Internal code may throw, including `std::bad_alloc` from containers. Every extern "C" entry point wraps its body in `try { … } catch (const std::bad_alloc&) { return FP_ENOMEM; } catch (const std::length_error&) { return FP_ENOMEM; } catch (...) { return FP_EINTERNAL; }`, so an exception never reaches a C, Julia, MATLAB or Rust caller. Hot kernels are `noexcept` and do not allocate.
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
### GPU backend (1.x, design)

CUDA kernels in the C core, behind the device field that `fp_array` has carried since 0.1. They come in 1.x as new enum values and new functions, so the frozen ABI only grows. A program that never passes a device array runs exactly as before, and the CPU library gains no CUDA dependency.

**What the 0.3 device path showed.** The PyTorch and JAX wrappers already run image TV on GPUs in framework tensor operations (Section 12). On a Tesla T4 that path is 6–8× faster than DeepInverse's denoiser at loose accuracy, but only 1.1–1.4× at 1e-3 to 1e-4 (docs/benchmarks.md, 2026-10-19). Two limits come from the framework, not the GPU:
- **Anisotropic TV without exact 1D steps.** The C kernel reaches a gap of 1e-6 in 59 iterations; the device path's plain dual method takes 624. The C kernel's steps are exact 1D TV proxes of every fibre, which tensor operations cannot express.
- **A dozen passes over the batch per iteration.** Each framework operation reads and writes the whole image. Fused kernels do an iteration in one or two passes.

The backend therefore starts with what unlocks the rest: a batched exact 1D TV kernel. The exact row operators and the elementwise maps come with it.

**Operators and methods.** In stage order. "Exact" keeps the CPU's accuracy target (Section 10); iterative operators keep its gap certificate.

| Stage | Operators | GPU method | Parallel over |
| --- | --- | --- | --- |
| 1 | elementwise maps (soft, elastic net, Huber, box, the nonconvex maps, log barrier) | the CPU's scalar kernels, with the prox, the structure codes and the kink mask fused into one pass | entries |
| 1 | group norms, ℓ2 ball, radial operators | one warp per row or group, with a fixed reduction tree | rows |
| 1 | simplex, ℓ1 ball (weighted too) | sort-based (Held, Wolfe and Crowder; Duchi et al.): a bitonic sort in shared memory, then a fixed-tree scan for the threshold. Rows too long for one block are first filtered in fixed chunks (Dai and Chen) | rows |
| 1 | 1D TV (weighted too), the step of anisotropic TV | taut string, one thread per row, with rows interleaved in the workspace so neighbouring threads read neighbouring addresses | rows |
| 2 | anisotropic TV, 2D and 3D | the CPU's method: accelerated dual block proximal gradient (Beck, Section 12.5.4; Chambolle and Pock 2015) with the stage-1 1D TV step on every fibre, its restarts, gap and polish | fibres × images |
| 2 | isotropic TV, 2D and 3D | the CPU's dual FISTA with restarts, an iteration fused into one kernel | pixels |
| 2 | the image TV derivatives | regions by hooking to the minimum index and pointer jumping, then sort and segmented scan (exact averages for anisotropic TV); conjugate gradients in region space with Jacobi for isotropic TV | pixels, regions |
| 3 | spectral operators | B and r of `fp_spectral_jvp` on the device, one block per matrix, so the k × k round trip goes away | matrices |
| 3 | isotonic regression, 1D TV with ℓ2 of differences, the rest | one thread per row; long rows by Kearsley's merge of block solutions | rows |

- **Taut string, not Condat, on the GPU.** Condat's method is the faster path on one CPU core, but its worst case is quadratic (Barbero and Sra 2018, Sections 2.2–2.3, who cap it with taut string as fastprox's CPU path does). A thread that backtracks holds up its warp. Taut string is linear in the worst case.
- **A few long rows.** There is no practical exact method that splits one 1D TV row across threads. The taut string is a shortest path in a polygon, which is in NC on a PRAM (Goodrich, Shauck and Guha 1992), but nothing practical has come of that. With few rows the kernel still runs, one thread each, but slowly. `fp_op_device_check` (below) reports which operators are row-parallel only, and the Python wrappers keep the C kernel through the host when the batch has fewer rows than the device has threads in flight. The threshold is measured in the stage-1 gate.
- **The simplex threshold.** τ = (Σ_I y − a) / |I| is summed over the support by a fixed tree in float64. The CPU's running mean sums in input order, so the two agree to rounding, not bit for bit.
- **Isotropic TV keeps the CPU's iteration.** The same iterates and iteration counts make the device and CPU results comparable within the gap. Other methods were considered (below) and left for measurement.
- **Images that finish early.** Each image stops on its own gap, as on the CPU. Kernels skip converged images through a per-image flag, so a batch costs launches for its slowest image, not work. Bai, Kolter and Koltun (deep equilibrium models, Appendix E) report the waste when converged samples keep iterating.

**Staying on the CPU.**
- **Graph TV.** Max-flow is P-complete. Jensen, Jeppesen, Dahl and Dahl (TPAMI 2023) found no generic parallel min-cut method that consistently beats serial ones, and recommend parallel methods only for graphs over 5 M nodes or 5 s of serial time. GPU push-relabel succeeds on grids (Vineet and Narayanan 2008), where image TV already serves, and for one cut rather than graph TV's parametric sequence.
- **Tree TV.** The message passing of Kolmogorov, Pock and Rolínek is sequential along each tree. Batches of many small trees could later take one thread per tree.
- Both return FP\_EINVAL for a device array.

**Methods considered and not used.**
- **Squares-based splitting for isotropic TV** (Chambolle and Pock 2015, Section 4.2). It needs 3–5× fewer iterations than FISTA (their Tables 3–4) and is pixel-parallel, but it minimises TV discretised on squares, which is a different operator from fastprox's forward differences.
- **Beck's three-block splitting of isotropic TV** (Example 12.23). It keeps the forward-difference TV with closed-form steps, but only the non-accelerated method applies to three blocks. It is a candidate to measure, not a default.
- **Primal–dual methods for isotropic TV.** Chambolle and Pock (2011, Table 1) report 1.6–2× fewer iterations than dual FISTA to 1e-6 for their adaptive variant, but measured to a reference, since gap stopping favours primal–dual methods. Also a candidate to measure.
- **Projected Newton for 1D TV** (Yeh, Hu, Ren and Schwing, CVPR 2022). It is the one published GPU 1D TV prox, 37× faster than proxTV on one batch, but iterative. In 2D it runs a few Dykstra steps, which is approximate.
- **Bisection or Halley's method for the simplex threshold** (as in AdaSplash, Gonçalves, Treviso and Martins 2025). These are iterative, and Condat (2016, Section II) shows bisection's claimed linear bound fails when many entries sit near τ.
- **Dykstra's method for isotonic regression** (Sander et al. 2023). It does not terminate finitely.
- **The 3 × 3 SVD of McAdams et al. (2011).** It is float32, with approximate rotations and a fixed four sweeps, so it is too inexact for the spectral derivatives. A one-sided Jacobi SVD per thread, with a fixed rotation order (Golub and Van Loan, Sections 8.5–8.6; accurate by Demmel and Veselić 1992), is the candidate for the closed-form small SVD in the 1.x candidates.
- **Shared-memory tiles with several inner iterations per global step** (Pock, Unger, Cremers and Bischof 2008). This changes the iterates, and borders lag. It is an optimisation to measure after stage 2.
- **CUB, Thrust, cuSOLVER and cuBLAS as dependencies.** The core stays free of libraries beyond the CUDA runtime, as it stays free of BLAS on the CPU. CUB's floating-point scans are not run-to-run deterministic by default, and its deterministic levels depend on the architecture and toolkit (CCCL determinism documentation).

**C API.** Additions only:

```c
typedef enum { FP_DEVICE_CPU = 0, FP_DEVICE_CUDA = 2 } fp_device; /* 2 is DLPack's kDLCUDA; 10 (kDLROCM) reserved */
/* fp_status gains FP_EDEVICE, appended: a CUDA error, now or from earlier work on the stream */

/* 1 if the CUDA plugin is loaded and a device is present, else 0. */
FP_API int fp_device_available(fp_device dev);
/* FP_OK if op runs on dev; FP_EINVAL if not. Sets *row_parallel_only when
 * one row cannot use more than one thread (1D TV, isotonic regression). */
FP_API fp_status fp_op_device_check(const fp_op *op, fp_device dev, int *row_parallel_only);

/* A workspace created with a CUDA `like` lives on the calling thread's
 * current device, as a cuBLAS handle does, and holds all device scratch.
 * The stream is a cudaStream_t (or hipStream_t) passed as void*; NULL is the
 * legacy default stream. One workspace per stream. */
FP_API fp_status fp_workspace_set_stream(fp_workspace *ws, void *stream);
FP_API fp_status fp_workspace_get_stream(const fp_workspace *ws, void **stream);

/* Results of the last call, valid once the stream has reached its end. */
FP_API fp_status fp_workspace_info(const fp_workspace *ws, fp_info *info);
/* fp_vjp with the step's gradient written to a device scalar (0-d array). */
FP_API fp_status fp_vjp_device(const fp_op *op, const fp_workspace *ws, const fp_array *g,
                               fp_array *gx, fp_array *ggamma, fp_array *gparams);
/* One step per matrix or image: gamma has the batch shape (the shape of x
 * without the operator's axes), on x's device. */
FP_API fp_status fp_prox_steps(const fp_op *op, const fp_array *gamma, const fp_array *x,
                               fp_array *out, const fp_options *opt, fp_workspace *ws);
```

- **Device memory.** All of it is allocated by `cudaMalloc` in `fp_workspace_create`, and freed by `fp_workspace_free`: per-row scratch such as taut string's O(n) arrays, region labels, and the operator's parameters copied to the device. `fp_prox`, `fp_jvp` and `fp_vjp` only launch kernels on the workspace's stream. They never allocate, copy to the host or synchronise, unless the caller asks for a host result (next point). The allocation counter in the tests also counts `cudaMalloc` and `cudaMallocAsync`. Every pointer is checked against the workspace's device with `cudaPointerGetAttributes` on each call; a mismatch returns FP\_EINVAL. Arrays need not be contiguous: strides work as on the CPU, at the cost of uncoalesced reads.
- **Host results wait; device results do not.** The interfaces that write host memory keep working and make the call wait for its stream: `fp_info` from `fp_prox`, and `ggamma` from `fp_vjp`. The asynchronous forms are `fp_workspace_info`, read after the caller has synchronised, and `fp_vjp_device`. The cuBLAS host pointer mode has the same rule. `gparams` may be a device array in either form.
- **Errors.** Arguments are checked before any launch, as on the CPU. Each launch is followed by `cudaGetLastError`. A kernel fault that surfaces later is returned as FP\_EDEVICE by the next call on that workspace, or by `fp_workspace_info`. The CUDA runtime reports asynchronous errors the same way (CUDA Programming Guide, Section 2.5.4).
- **Iterative operators and stopping.** The gap is evaluated on the device at the CPU kernels' period (every fourth iteration for isotropic TV). By default the host reads a pinned per-image flag through an event, which is a wait but a cheap one. On a stream that is being captured into a CUDA graph (`cudaStreamIsCapturing`), the call instead launches the whole `max_iter` schedule. Each kernel then returns at once for images whose flag is set, so the graph is static and the result is the same iterate. CUDA's conditional graph nodes could later replace the empty launches.
- **Steps per image.** `fp_prox_steps` gives the C API what the 0.3 device path and DeepInverse's denoisers already take: one γ per image (deepinv/deepinv#336). It works on the CPU too. Per-image spatial weights stay with the framework path until an operator can take its weights as an array at call time.
- **Unchanged.** `fp_op` is immutable and shared across devices and streams, because its device copies live in workspaces. `fp_options.threads` is ignored on a device.

**Determinism and agreement with the CPU.**

| Promise | Holds |
| --- | --- |
| Bit-identical run to run | on one device model, toolkit and fastprox build, for any batch size, launch configuration and stream |
| Not promised | bit-identical across GPU models, or against the CPU |
| Exact operators against the CPU | within the accuracy target of Section 10, and bit-identical for the elementwise maps built from correctly rounded operations (+, −, ×, ÷, √) only |
| Iterative operators against the CPU | within the gap tolerance, at the same iteration count for isotropic TV |

- **Fixed geometry.** Grid and block sizes depend only on the shape and the operator. They never depend on the SM count, occupancy queries or the batch's position in a larger call.
- **No floating-point atomics.** Rows stay independent. Cross-row sums (the gap, vjp parameter gradients, region sums) use fixed-shape trees over `reduction_blocks`, whose shape is set by n alone. Integer atomics are allowed, as in labelling, where only counts and minima use them.
- **Canonical region labels.** Each region is labelled with its minimum linear index and renumbered by a scan over root flags. That is independent of scheduling (Jaiganesh and Burtscher 2018, ECL-CC; Komura 2015). The flat test is the same expression as on the CPU, so the regions agree whenever the duals do.
- **Floating-point flags.** Device code builds with `--fmad=false -prec-div=true -prec-sqrt=true -ftz=false` and never with `--use_fast_math`, the device form of the no-`-ffast-math` rule. Contraction is also turned off on the CPU (`-ffp-contract=off` for GCC and Clang, whose defaults differ). Then an elementwise map is the same sequence of correctly rounded operations on both, and matches bit for bit (NVIDIA, Floating Point and IEEE 754 Compliance, Sections 4.4 and 5.3). Maps that call `log` or `pow` (the log barrier, ℓq's Newton steps, the log penalty) agree to a few ulps: CUDA's double `log` is within 1 ulp and `pow` within 2 (CUDA Programming Guide, Section 5.5). Changing the CPU flag moves some CPU results by an ulp, once, in the release that ships stage 1.
- **float32.** As on the CPU, sums that span a row or an image accumulate in float64. On GPUs with float64 at 1/32 (T4) or 1/64 (L4, L40S, the RTX series) of float32 throughput (CUDA C++ Best Practices Guide, Section 12.1.1), these few reductions cost little. The iterates themselves are float32. Lee and Wright (2008) ran TV denoising in float32 on a GPU with gap stopping to 1e-6 at under 10% more iterations than float64. Their other solver's float32 gap stalled near 1e-4, so the tests check the gap at tight tolerances in both dtypes.
- **float64.** Fully supported. It is slow on parts with 1/64 rate, and the docs give the rates.

**Build and packaging.**
- **CMake.** `FASTPROX_CUDA=ON` builds the CUDA sources (`src/cuda/*.cu`, C++17, the same headers and templates) into a separate shared library, `libfastprox_cuda`.
- **The plugin.** `libfastprox` loads it with `dlopen`/`LoadLibrary` the first time a device array or `fp_device_available` appears, through a private versioned table of entry points. If it is missing, `fp_device_available` returns 0 and device arrays return FP\_EINVAL, as now. The CUDA runtime is linked statically; `libcuda` is never bundled.
- **Architectures.** SASS for sm\_75 (T4), 80, 86, 89 and 90, plus 100 and 120 where the toolkit has them. PTX for the newest, for later GPUs (NVIDIA, CUDA minor-version compatibility).
- **PyPI.** The `fastprox` wheels stay CPU-only abi3 wheels. The plugin ships as `fastprox-cuda12` and `fastprox-cuda13` (tagged `py3-none-manylinux_2_28_{x86_64,aarch64}`, no Python code), installed by `pip install fastprox[cuda12]`, as CuPy and JAX split theirs. Wheel variants would let one name serve both (PEP 817, PEP 825), but as of October 2026 only the format PEP is provisional. They are revisited when installers support them. Windows follows Linux.
- **conda-forge.** A `libfastprox-cuda` output built when `cuda_compiler_version` is not None, depending on `libfastprox` and pinned through `cuda-version`.
- **Other GPUs.** Kernels use a thin layer for launches, shared memory, warp width and streams. HIP is then mostly a rebuild (`libfastprox_rocm`, FP\_DEVICE\_ROCM = 10), with lane masks for 64-wide wavefronts. Apple GPUs have no float64, and Metal compiles with fast math by default, so a Metal backend would be a separate float32 tier; none is planned. Apple GPUs keep the 0.3 framework path.

**Bindings.**
- **Python.** nanobind's `ndarray` accepts device arrays and gives their device type and id. Bindings never synchronise. Workspaces are cached per operator, device, stream, shape and dtype.
- **PyTorch.** The wrappers pass `torch.cuda.current_stream().cuda_stream` and move to `torch.library.custom_op` (`device_types="cuda"`, with `register_autograd`), which `torch.compile` treats as opaque. The tensor-operation path stays for MPS and the CPU.
- **JAX.**
  - A small FFI handler library calls only the C API on XLA's stream (`jax.ffi.register_ffi_target`, platform CUDA). It replaces `pure_callback`, which copies to the host.
  - XLA's FFI headers are header-only and Apache-2.0, so they can be vendored with their notice. The FFI is versioned but still marked experimental.
  - The backward recomputes the prox, as it does now, since `custom_vjp` residuals must be arrays.
- **CuPy** arrays work through the same nanobind path, with the stream from `cupy.cuda.get_current_stream()`.

**Tests and CI.**
- **Every pull request**, free and without a GPU: the plugin compiles with CUDA 12 and 13; its exports are checked; the CPU tests run; and a missing plugin and device arrays fail cleanly.
- **On a GPU** (`tests/gpu`):
  - every device operator against its C kernel: exact ones within the accuracy target, the elementwise ones bit for bit, iterative ones within the gap;
  - the derivatives against the CPU's;
  - bit-identical repeats at two batch sizes and two stream setups;
  - a graph-capture test;
  - the allocation counter;
  - compute-sanitizer's memcheck, racecheck, initcheck and synccheck, with `--error-exitcode`.
- **Where.** There is no CUDA device here. The suite runs from a scripted Colab notebook (T4, sm\_75) before each release, and on a paid GitHub GPU runner or a cloud runner when triggered by hand. A persistent self-hosted runner is never attached to the public repository, since pull requests from forks could run code on it.
- **Gates.**
  - **Stage 1:** each operator is faster than the C kernel through the host on batches of 64 × 512² entries or rows on a T4, and the row-count threshold for 1D TV is measured.
  - **Stage 2:** anisotropic TV reaches 1e-6 in the C kernel's iterations, about a tenth of the 0.3 path's. Both image TVs are faster than the 0.3 path at every tolerance from 1e-2 to 1e-6 on `bench/compare_deepinv.py`. Numbers are also taken on one datacenter GPU (A100 or H100, float64 at half rate).

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
    fp_device      device;   /* FP_DEVICE_CPU; GPU values from the 1.x backend */
} fp_array;

typedef enum {
    FP_OK = 0, FP_EINVAL, FP_EDTYPE, FP_ESHAPE,
    FP_ENOMEM,              /* out of memory, or a size too large to allocate */
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

- **Boundary pattern.** Each extern "C" function checks its arguments, dispatches on `dtype` to `kernel<float>` or `kernel<double>`, and wraps the call in try/catch. `std::bad_alloc` and `std::length_error` become FP\_ENOMEM and any other exception becomes FP\_EINTERNAL. Constructors return a status and write the handle through an out-pointer, so allocation failures are reported, not hidden in a NULL.
- **Opaque handles.** `fp_op` and `fp_workspace` are incomplete types in C; inside, they are C++ classes holding `std::vector` members and released by their destructors in `fp_op_free` and `fp_workspace_free`. Callers never see an STL type, a reference or a template.
- **Ownership.** The library never frees caller memory. Operators copy their parameter arrays at creation. Workspaces own all scratch memory and are reused across calls, so a loop of prox calls allocates nothing after the first.
- **Threads.** `fp_op` is immutable and safe to share. `fp_workspace` is not shared between concurrent calls; it holds one scratch region per OpenMP thread. Results are bit-identical for any value of `threads`.
- **Errors.** Every function returns `fp_status`; there are no global error states, no printing and no aborts. Invalid parameters (negative γ, q outside (0, 1), a SCAD parameter too small for γ) return FP\_EPARAM before any work.
- **Derivatives.** `fp_jvp` and `fp_vjp` read the structure saved in the workspace by the last `fp_prox` call with `keep_structure` set. Calling them without it returns FP\_EINVAL.
- **ABI stability.** Structs passed by pointer carry no hidden fields; new options are added only through new functions or a versioned options struct. The major version changes only when the ABI breaks. The device field of fp\_array is reserved from 0.1 so a GPU backend can be added without an ABI break. Up to 1.0 only FP\_DEVICE\_CPU is accepted and any other value returns FP\_EINVAL. The 1.x GPU backend adds new fp\_device values and a function that reports which operators support a device; an operator given a device it does not implement still returns FP\_EINVAL. The stream, the asynchronous results and the steps per image come as new functions (Section 8, "GPU backend"), since `fp_options` cannot grow.

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

# autodiff integration, built on the same calls
from fastprox.torch import tv1d as tv1d_torch   # torch.autograd.Function
from fastprox.jax import tv1d as tv1d_jax       # jax.custom_vjp + pure_callback
from fastprox.torch import tv_iso as tv_iso_torch  # image TV on the tensor's device, GPU included
```

- **nanobind module.** The Python extension is written with nanobind and calls only the C API, never the C++ internals, so Python sees exactly what other languages see. `nb::ndarray` accepts C-contiguous and strided NumPy arrays without copying. Other dtypes are converted only when the caller sets `copy=True`; otherwise a `TypeError` is raised.
- **GIL.** Released (`nb::gil_scoped_release`) for every call that does work proportional to its data (prox, derivatives, values, operator constructors, workspace creation, dual and kink copies), so Python threads can run operators concurrently on separate workspaces; constant-time getters keep it, since releasing costs as much as the call.
- **Errors.** A non-OK `fp_status` becomes a Python exception (`ValueError` for parameter errors, `MemoryError` for FP\_ENOMEM, `RuntimeError` otherwise) with the text of `fp_strerror`.

**Other languages (deferred).** Julia (ccall through a JLL package), MATLAB (C mex gateways) and Rust (`fastprox-sys` plus a thin safe crate) are not built for now. Because Python calls only the public C API, those bindings can be added later without changing the core or the ABI.

## Numerical and performance requirements

Each requirement below has a measurable target and a named check, so a release can be accepted or rejected on evidence. Speed targets are relative to the best existing code because absolute times depend on the machine.

| Requirement | Target | How it is checked |
| --- | --- | --- |
| Accuracy of exact operators | optimality residual at most 10 n ε ‖x‖∞, with ε the machine epsilon of the dtype | optimality conditions checked on every test input (Section 11) |
| Accuracy of iterative operators | duality gap at most tol · ½‖x‖², returned in `fp_info.gap` | gap recomputed independently in tests |
| Determinism | bit-identical output for any thread count, on the same build and machine | tests run with 1, 2 and 8 threads and compare bytes |
| Determinism on GPUs | the same output on repeated runs on one device model, toolkit and build, for any batch size and launch configuration; not bit-identical to the CPU, but within its accuracy target or gap tolerance, and bit-identical for the elementwise maps built from correctly rounded operations (Section 8, "GPU backend") | repeated runs compared byte for byte; GPU against C kernel within the target or the gap |
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

The 1.x CUDA backend ships as a separate plugin library and package (`fastprox-cuda12`, `fastprox-cuda13`, `libfastprox-cuda`), so the packages above stay CPU-only (Section 8, "GPU backend").

Julia (`fastprox_jll` through BinaryBuilder), MATLAB (a `.mltbx` toolbox with C mex gateways) and Rust (`fastprox-sys` on crates.io) packages are deferred with their bindings.

- **OpenMP in wheels.** Linux wheels bundle libgomp through auditwheel. macOS wheels bundle libomp through delocate, because Apple Clang does not ship it.
- **Symbols.** Only `fp_` symbols are exported on every platform (a linker version script on Linux, `__declspec` on Windows, visibility on macOS), so two copies of the library or of libstdc++ in one process cannot clash.

**Integration**

- **BenchOpt:** solvers using the library in the TV-1D and TV-2D benchmarks, and a new simplex-projection benchmark.
- **PyProximal:** an adapter subclassing its `ProxOperator`, so PyProximal's solvers can call the compiled operators.
- **PyTorch and JAX:** `torch.autograd.Function` wrappers and `jax.custom_vjp` with `pure_callback` on the CPU, both using the library's VJP rather than differentiating through iterations. Isotropic and anisotropic TV also have a device-native path in each wrapper, written in framework tensor operations: dual fast gradient projection with the same gap stopping and warm starts, and a backward by conjugate gradients on the flat regions, so memory does not grow with the iteration count. It runs wherever the input lives, GPU included, and needs no change to the C API.
- **DeepInverse:** the device-native image TV is benchmarked against DeepInverse's TV denoiser (time to accuracy, transfer cost, memory of a backward pass) before an optional backend is proposed upstream.

## Roadmap

Four releases take the library from the 1D core to a stable 1.0. Each one ships only when its gate passes. No dates are set: the order is fixed, and the pace depends on who builds it.

[embedded content: roadmap · four releases, one gate each, not to scale]

Release 0.1 is highlighted because it already serves most callers: thresholding, projections and 1D TV with derivatives, from Python. The C++17 core, the extern "C" boundary and the static-libstdc++ Linux build are all in place from 0.1, so later releases add operators and bindings without changing the ABI.

- **0.1 — core and 1D.** The C API and its try/catch boundary, separable convex maps, projections and pool-adjacent-violators, 1D TV by Condat and taut string with weights, JVP and VJP for all of these, and Python wheels built with nanobind.
- **0.2 — images.** Anisotropic and isotropic TV in 2D and 3D with duality gaps and warm starts, 1D TV with ℓ2 of differences, and the nonconvex pointwise maps, all exposed in the Python package.
- **0.3 — graph TV.** The max-flow engine, graph TV and tree TV, VJPs for the iterative operators by conjugate gradients, and the PyTorch and JAX wrappers, with a device-native path for image TV that runs on GPUs (Section 12).
- **1.0 — stable.** ABI freeze, conda-forge packages, documentation, and the PyProximal adapter. After 1.0, only additions that keep the ABI are allowed until 2.0.
- **1.x — GPU backend.** CUDA kernels in the C core behind the device field, an addition that keeps the frozen ABI (design in Section 8). The first operators are those the 0.3 device path shows people running on GPUs: iterative image TV, elementwise maps and batched simplex and ℓ1-ball projections, and exact 1D TV with one thread per row for large batches. Max-flow graph TV stays on the CPU. CUDA builds ship as a separate package and through conda-forge, not in the main wheel.
- **After 1.0 — other languages.** Julia (ccall), MATLAB (mex, with a gate that the mex files load on Linux next to MATLAB's own libstdc++) and the thin Rust crate, each over the frozen C API, plus a ProximalOperators.jl adapter.

## Decisions

The six questions left open in the first draft were settled on 14 October 2026, and the binding scope was narrowed to Python the same day; the sections above reflect both. Work on 0.1 can start; nothing in the design is still waiting on a decision.

| Question | Decision | Where it applies |
| --- | --- | --- |
| Device field for a later GPU backend | reserved now: `fp_array` carries an `fp_device` field, and up to 1.0 only `FP_DEVICE_CPU` is accepted. Revised 16 October 2026: a GPU backend in the C core comes in 1.x as an ABI-compatible addition, and a device-native image TV path comes earlier in the 0.3 PyTorch and JAX wrappers | API header and ABI rules (Section 9), Integration (Section 12), Roadmap (Section 13) |
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

**Checked during 0.2** (2026-10-15/16, against the bibliographies of A. Beck, *First-Order Methods in Optimization*, SIAM 2017, Appendix D, of F. Wen, L. Chu, P. Liu and R. Qiu, *A survey on nonconvex regularization based sparse and low-rank recovery*, IEEE Access 6, 2018, and of a third citing paper; the original papers were not opened, so the kernels derive their closed forms themselves and verify them against minimisation on a refined grid)

- A. Beck and M. Teboulle, Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems, IEEE Trans. Image Processing 18 (2009) 2419–2434 — FGP for isotropic TV.
- Z. Xu, X. Chang, F. Xu and H. Zhang, L1/2 regularization: a thresholding representation theory and a fast solver, IEEE Trans. Neural Networks and Learning Systems 23(7) (2012) 1013–1027 — ℓ1/2 thresholding.
- W. Cao, J. Sun and Z. Xu, Fast image deconvolution using closed-form thresholding formulas of Lq (q = 1/2, 2/3) regularization, J. Visual Communication and Image Representation 24(1) (2013) 31–41 — ℓ2/3 thresholding.
- J. Fan and R. Li, Variable selection via nonconcave penalized likelihood and its oracle properties, JASA 96(456) (2001) 1348–1360; C.-H. Zhang, Nearly unbiased variable selection under minimax concave penalty, Annals of Statistics 38(2) (2010) 894–942 — SCAD and MCP.

**Checked for the spectral operators** (2026-10-20; the theorems were read in the papers, or in Beck's book, unless marked)

- A. Beck, *First-Order Methods in Optimization*, SIAM 2017, Theorems 7.18 and 7.29 — the spectral prox formulas for convex symmetric and absolutely symmetric functions.
- C. Lu, C. Zhu, C. Xu, S. Yan and Z. Lin, [Generalized singular value thresholding](https://arxiv.org/abs/1412.2231), AAAI 2015, Theorems 1 and 2 — the nonconvex case.
- C. Ding, D. Sun, J. Sun and K.-C. Toh, [Spectral operators of matrices](https://doi.org/10.1007/s10107-017-1162-3), Mathematical Programming 168, 2018, Lemma 3.2, equations (49)–(54) and Theorems 3.3–3.4 as numbered in arXiv 1401.2269v1 — the derivative, its fallback values and when it exists.
- E. J. Candès, C. A. Sing-Long and J. D. Trzasko, [Unbiased risk estimates for singular value thresholding and spectral estimators](https://arxiv.org/abs/1210.4139), IEEE Trans. Signal Processing 61(19), 2013 — the divergence formula, checked numerically, not in the paper's text.
- A. S. Lewis and H. S. Sendov, Twice differentiable spectral functions, SIAM J. Matrix Anal. Appl. 23(2), 2001; Nonsmooth analysis of singular values I and II, Set-Valued Analysis 13(3), 2005 — the eigenvalue case. Not opened; cited from reference lists.
- Ju. L. Daleckiĭ and S. G. Kreĭn, Integration and differentiation of functions of Hermitian operators and applications to the theory of perturbations, AMS Translations (2) 47, 1965 — divided differences of matrix functions. Not opened.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., 2013, Sections 2.7 and 9.1.5 — rounding against truncation, and clusters of eigenvalues.
- L. Stella, A. Themelis and P. Patrinos, Forward–backward quasi-Newton methods for nonsmooth optimization problems, Comput. Optim. Appl. 67, 2017, Section 5.4 — nuclear-norm prox with a partial SVD.
- PyTorch documentation of `torch.linalg.svd` (2.14) — gradients only finite without repeated singular values.

**Checked for the GPU backend** (2026-10-20; read in full or in the cited sections unless marked; papers in the user's Notability library are marked N)

- A. Chambolle and T. Pock, [A remark on accelerated block coordinate descent for computing the proximity operators of a sum of convex functions](https://doi.org/10.5802/smai-jcm.3), SMAI J. Comput. Math. 1, 2015, Tables 1–4 — chain splitting with exact 1D steps (80 iterations to a gap of 1e-3, 170–190 to 1e-6), and squares splitting, a different discretisation.
- A. Chambolle and T. Pock, An introduction to continuous optimization for imaging, Acta Numerica 25, 2016, Section 4.8.3 and Example 4.15 (N) — block descent suited to GPUs; FISTA on chains needs 10–20× fewer iterations than plain FISTA.
- A. Beck, *First-Order Methods in Optimization*, SIAM 2017, Section 12.5 and Examples 12.22–12.23 (N) — the accelerated two-block method and the pair and triplet splittings.
- A. Chambolle and T. Pock, [A first-order primal-dual algorithm for convex problems with applications to imaging](https://doi.org/10.1007/s10851-010-0251-1), JMIV 40, 2011, Table 1 (N) — iterations of dual FISTA against primal–dual variants for ROF.
- T. Pock, M. Unger, D. Cremers and H. Bischof, Fast and exact solution of total variation models on the GPU, CVPR Workshops 2008 — dual ROF in CUDA, 10–1000× faster than parametric max-flow on grids; shared-memory tiles with inner iterations.
- S. Lee and S. J. Wright, Implementing algorithms for signal and image reconstruction on graphical processing units, Optimization Online 2008/10/2131 — float32 TV with gap stopping to 1e-6 at under 10% more iterations; a float32 gap stalling near 1e-4 in another solver.
- M. G. McGaffin and J. A. Fessler, [Edge-preserving image denoising via group coordinate descent on the GPU](https://doi.org/10.1109/TIP.2015.2400813), IEEE TIP 24(4), 2015 — memory traffic and inner products as the cost on GPUs.
- D. Villacís, First order methods for high resolution image denoising, Latin American Journal of Computing IV(3), 2017 — Chambolle–Pock in PyCUDA on 7680 × 4320 images.
- Á. Barbero and S. Sra, JMLR 19, 2018 (above), Sections 2.2–2.3 and 4.3 — Condat's quadratic worst case and the taut-string cap; parallelism across fibres only.
- R. A. Yeh, Y.-T. Hu, Z. Ren and A. G. Schwing, [Total variation optimization layers for computer vision](https://arxiv.org/abs/2204.03643), CVPR 2022 — a GPU projected Newton method for 1D TV, iterative.
- L. Condat, Mathematical Programming 158, 2016 (above), Table I and Section II — complexities of the simplex methods, and bisection's missing bound.
- Y. Dai and C. Chen, Sparsity-exploiting distributed projections onto a simplex, INFORMS J. Computing 36(3), 2024, Proposition 3.4 — exact filtering of a long row in chunks.
- A. J. Kearsley, Projections onto order simplexes and isotonic regression, J. Res. NIST 111(2), 2006, Theorem 1 — merging isotonic regressions of blocks.
- G. Gonçalves, M. Treviso and A. F. T. Martins, AdaSplash, ICML 2025; M. E. Sander, J. Puigcerver, J. Djolonga, G. Peyré and M. Blondel, Fast, differentiable and sparse top-k, ICML 2023, Section 5 — iterative thresholds and Dykstra's method for isotonic regression on accelerators.
- P. M. Jensen, N. Jeppesen, A. B. Dahl and V. A. Dahl, [Review of serial and parallel min-cut/max-flow algorithms for computer vision](https://arxiv.org/abs/2202.00418), IEEE TPAMI 45(2), 2023, Section 8 — no generic parallel method consistently beats serial ones.
- J. Jaiganesh and M. Burtscher, A high-performance connected components implementation for GPUs (ECL-CC), HPDC 2018, Section 3; S. Allegretti, F. Bolelli, M. Cancilla and C. Grana, IPAS 2018 — labelling by hooking to the minimum index.
- A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis, Computing the singular value decomposition of 3 × 3 matrices with minimal branching and elementary floating point operations, UW–Madison TR1690, 2011 — float32 and approximate.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Sections 8.5–8.6 (N) — Jacobi methods and their parallel orderings.
- S. Bai, J. Z. Kolter and V. Koltun, Deep equilibrium models, NeurIPS 2019, Appendix E (N) — converged samples waiting in a batch.
- NVIDIA documentation (CUDA 13): the CUDA Programming Guide (Sections 2.5.4, 4.2 and 5.5), the CUDA C++ Best Practices Guide (Section 12.1.1), Floating Point and IEEE 754 Compliance (Sections 4.4 and 5.3), the nvcc manual (`--fmad`, `--prec-div`, `--prec-sqrt`, `--ftz`), cuBLAS (Sections 2.1.3, 2.1.4, 2.1.12, 2.4.7), cuFFT (Sections 2.7, 2.13, 3.5–3.8), cuSOLVER (Section 2.1.7), CUDA minor-version compatibility, Compute Sanitizer, and the CCCL determinism pages.
- DLPack's C API and the array API's `__dlpack__`; the CUDA Array Interface v3; nanobind's `ndarray`; PyTorch's custom operators and CUDA semantics; JAX's FFI; PEPs 817 and 825; conda-forge's CUDA knowledge base; GitHub's runner pricing.

**Cited for the GPU backend from abstracts or other papers**

- M. Held, P. Wolfe and H. P. Crowder, 1974; J. Duchi, S. Shalev-Shwartz, Y. Singer and T. Chandra, ICML 2008 — sort-based simplex projection, through Condat (2016).
- M. T. Goodrich, S. B. Shauck and S. Guha, Parallel methods for visibility and shortest-path problems in simple polygons, Algorithmica 8, 1992 (abstract).
- V. Vineet and P. J. Narayanan, CUDA cuts, CVPR Workshops 2008 (abstract); Y. Komura, Comput. Phys. Commun. 194, 2015 (abstract, and its generalisation, arXiv 1603.08357, Section 2).
- J. Demmel and K. Veselić, Jacobi's method is more accurate than QR, SIAM J. Matrix Anal. Appl. 13(4), 1992 (abstract).
- P. Ahrens, J. Demmel and H. D. Nguyen, Algorithms for efficient reproducible floating point summation, ACM TOMS 46(3), 2020 — not used: it gives the same sums across GPU models, which is not promised.

**To check before citing in code**

- Tooling documentation (nanobind, scikit-build-core, cibuildwheel, BinaryBuilder, the GCC Runtime Library Exception) was not opened while writing this spec; confirm the details in Section 12 against it.

The survey this spec builds on: [Compiled solvers for variational imaging and nonsmooth optimization](https://claude.ai/code/artifact/51657fed-21f0-4ccc-b9ed-2d47aa2a97f3).
