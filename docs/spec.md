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

- **No solver framework.** No FISTA, PDHG or ADMM drivers beyond a reference loop for isotropic TV. Those belong in the user's code or in PyProximal and ProximalAlgorithms.jl. Bilevel algorithms (implicit, MAID-type, single-loop) also belong in a separate package on the public API. fastprox gives them derivatives, certificates and the structure they need (Section 7; decision of 2026-10-23).
- **No GPU kernels in the C core for 1.0.** The C library is CPU only up to 1.0. Before that, the PyTorch and JAX wrappers run the iterative image operators on the input's device, written in framework tensor operations (0.3). The array interface reserves a device field, so a GPU backend in the C core can follow in 1.x without changing the API. Its design is in Section 8, "GPU backend".
- **No SVD or eigendecomposition in the core.** Matrix prox maps such as the nuclear norm need one, so they come in 1.x with the decomposition supplied by the caller (Section 5, "Spectral operators"). Planned at first for version 2; revised 20 October 2026.
- **No automatic modelling.** No parser or DSL for composite objectives, unlike ProxImaL.

## Use cases

Four kinds of caller drive the design. Each one needs something existing libraries lack: compiled speed, derivatives, or nonconvex penalties.

| Caller | What it calls | What it needs from the library |
| --- | --- | --- |
| Proximal splitting (FISTA, PDHG, ADMM, Condat–Vũ) | prox of g or of the conjugate g\* once per iteration, often on a whole image | speed on large arrays, in-place output, a conjugate prox through the Moreau identity |
| Bilevel learning of regularisation parameters | prox inside the lower-level solver, then derivatives of the solution with respect to the parameters | Jacobian-vector and vector-Jacobian products in the input and in the weights; for single-loop methods, one fused prox step per outer iteration |
| Sparse hyperparameter learning (Lasso-type models) | soft thresholding, group and weighted ℓ1, simplex and ℓ1-ball projection | the active set or support as a by-product, and Jacobians restricted to it (Section 7, "Structure of the derivative", since 1.0.2) |
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
- **Derivatives.** The prox maps fall into three classes (revised 25 October 2026, math audit C1):
  - **Lipschitz.** Every convex operator, with constant 1: it is firmly nonexpansive, and where it is differentiable its Jacobian is symmetric with eigenvalues in [0, 1] (Bauschke and Combettes, Prop. 12.28 and Rem. 14.4; Rockafellar and Wets, 13.53). Also SCAD and MCP under their step conditions, and the log penalty at γλ < ε²: g + (ρ/2)‖·‖² is convex with ρ = w/(a − 1), w/b and λ/ε², and the prox is (1/(1 − γρ))-Lipschitz (Rockafellar and Wets, 12.19). These maps can expand: MCP's slope b/(b − γw) is correct and large near the step limit, which is conditioning, not error.
  - **Hölder-½.** The log penalty at γλ = ε²: unique and continuous, but z ~ √(εh) at |x| = ε + h.
  - **Discontinuous.** ℓ0, ℓq with q < 1 (a jump from 0 to β = (2c(1 − q))^{1/(2−q)} at τ = β + cqβ^{q−1}, c = γλw), and the log penalty at γλ > ε². No tie rule removes the jump.

  Away from kinks and jumps every map has an ordinary Jacobian. For the first class the library returns, at a kink, the Jacobian of a piece: an element of the B-subdifferential and of Clarke's generalised Jacobian, which presuppose local Lipschitz continuity. For the other two, at a jump or at the Hölder point, it returns the slope of the selected branch, a convention, with no Clarke or Lipschitz property claimed. What the derivatives guarantee once composed, inside an outer program or a bilevel problem, is in `docs/api.md`, "What the derivatives guarantee". Where the prox is differentiable, the derivative is the Jacobian; at kinks it is a documented choice, the same element used by implicit differentiation codes such as sparse-ho (the support-restricted Jacobian). Section 7 gives the formula for each operator. For the iterative image operators the derivatives are those of the structure saved at the returned point, exact only once that point has the solution's regions. Since 1.0 the library says so (`fp_workspace_deriv_exact`, `ctx.deriv.exact`) and can re-solve the kept call to a tighter tolerance from its dual (`fp_workspace_refine`, `ctx.refine`).
- **Arrays and axes.** Inputs are real arrays (float32 or float64). Separable operators act entrywise; 1D TV, projections and group norms act along one axis and are batched over the others; 2D and graph TV act on the last two axes or on a supplied graph.
- **Optimality certificate.** Iterative operators (isotropic TV, multi-dimensional TV by splitting) return the duality gap of the prox problem at the returned point, and per image the relative error of z it certifies, √(2·gap)/‖x‖ plus the rounding of the returned array to its dtype, u(1 + √(2·gap)/‖x‖) with u the dtype's unit roundoff (twice that after a conjugate; since 1.0.1, math audit M7). The gap is evaluated in floating point, so the bound holds to working precision, not as a directed-rounding enclosure. Exact operators return a gap of zero: no certificate is computed for them.

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
| Sorted ℓ1 (OWL, SLOPE) | $\sum_k w_k \lvert z \rvert_{(k)}$, $w$ nonincreasing | sort, then pool-adjacent-violators (Bogdan et al. 2015) | O(n log n) | yes | 1.0 |
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
- **What is deliberately missing.** TGV, ICTV, the Schatten-1 Hessian norm and nonconvex TV (such as the TV quasinorm) have no exact prox. They are better handled by the user's splitting method on top of the operators above, using a weighted TV or a pointwise nonconvex prox as the inner step. Revised 23 October 2026: Huber-smoothed TV, the Frobenius Hessian norm (second-order TV) and filter-bank priors with learned potentials come in 1.x, with a duality gap like isotropic TV ("Analysis priors" below).

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

Each operator reports whether it is permutation invariant and whether it is even. Weighted variants qualify only with equal weights; the sorted ℓ1 norm qualifies with any nonincreasing weights, which gives the weighted nuclear norm. Others, such as isotonic regression and 1D TV, are rejected with FP\_EINVAL. Not covered are symmetric penalties with no vector prox in fastprox, such as the truncated nuclear norm.

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

### Analysis priors: smoothed TV, second-order TV and learned potentials (1.x, design)

An analysis prior applies scalar potentials to the responses of a bank of small filters:

g(z) = Σ\_k Σ\_p α\_{k,p} φ\_k(‖(W\_k z)\_p‖₂)

- **Filters.** W\_k is a group of c\_k filters, so (W\_k z)\_p ∈ ℝ^{c\_k}. A group of one filter gives a scalar response and φ\_k(|·|).
- **Potentials.** φ\_k is a scalar potential: |t|, the Huber function, or a learned spline (stage 2 below).
- **Pixel weights.** α\_{k,p} ≥ 0, and 1 without them.

Isotropic and anisotropic TV are the case W = forward differences and φ = |t|. They keep their own kernels. The form gives four additions, in this order. Each keeps the 1.0 ABI, since each is a new constructor.

| Stage | Operator | W | φ | What it serves |
| --- | --- | --- | --- | --- |
| 1 | Huber-smoothed TV, isotropic and anisotropic, 2D and 3D | forward differences, axis weights | h\_δ | the smooth lower level of bilevel TV papers; a linearly convergent prox; derivatives without flat regions |
| 2 | spline potentials, pointwise and on groups | none | learned φ, convex or weakly convex | the potentials of learned ridge regularisers, with an exact prox |
| 3 | analysis prior with a filter bank; second-order TV as a fixed instance | given filters, up to 9 taps per axis | \|t\|, Huber, elastic net, convex spline | second-order TV, filter-bank priors (Kunisch and Pock), learned convex ridge regularisers (CRR) |
| 4 | weakly convex potentials in the analysis prior | given filters | weakly convex spline | weakly convex ridge regularisers (WCRR) |

The comparative study of Hertrich et al. (2025, Section 2.1, (6)) writes the learned ridge regularisers as R(x) = Σ\_j ⟨1, ψ\_j(W\_j x)⟩ with learnable spline potentials ψ\_j, which is this form. Trained bilevel on BSD68 at σ = 0.1, CRR reaches 28.01 dB, WCRR 28.60 and an ICNN 27.90 (their results, as collected in `research_notes/`). ICNNs have no structure for a prox kernel to use: their prox is a general minimisation through a network, which belongs in a framework. DeepInverse ships one (`deepinv/models/icnn.py`). They are not planned.

**Stage 1: Huber-smoothed TV.**

- **The function.**
  - Isotropic: g(z) = Σ\_p α\_p h\_δ(‖(W D z)\_p‖₂).
  - Anisotropic: g(z) = Σ\_a w\_a Σ\_p β\_{a,p} h\_δ(|(D\_a z)\_p|).
  - The Huber function is that of `fp_op_huber`: h\_δ(t) = t²/(2δ) for |t| ≤ δ, |t| − δ/2 otherwise, with δ > 0. It is Beck's H\_μ (6.43), with μ = δ.
  - The smoothing is that of Hintermüller and Stadler (2006) and of the bilevel TV papers. De los Reyes, Schönlieb and Valkonen (2017, Definition 2.1) write it with γ = 1/δ, and their experiments take γ = 100, that is δ = 0.01.
  - Since |t| − δ/2 ≤ h\_δ(t) ≤ |t|, the operator tends to TV as δ → 0. The solutions satisfy ‖z\_δ − z\_TV‖² ≤ γ δ Σ\_p α\_p (anisotropic: γ δ Σ\_a w\_a Σ\_p β\_{a,p}), by strong convexity of the prox objective.
- **Dual.**
  - h\_δ is the Moreau envelope of the norm (Beck, Example 6.54), so its conjugate is the ball indicator plus (δ/2)‖·‖² (Corollary 6.56). With c\_p = γα\_p, (c\_p h\_δ)\*(u) = ι\_{‖u‖ ≤ c\_p} + δ‖u‖²/(2c\_p).
  - With G = W D, the prox dual is min\_u ½‖x − Gᵀu‖² + Σ\_p δ‖u\_p‖²/(2γα\_p) subject to ‖u\_p‖ ≤ γα\_p, and z = x − Gᵀu.
  - Its separable part is σ-strongly convex with σ = δ/(γ α\_max). A pixel with α\_p = 0 is free (u\_p = 0), as now.
- **Method.**
  - The constant-momentum FISTA for strongly convex problems (V-FISTA; Beck, Section 10.7.7 and Theorem 10.42), on the split f̃(u) = ½‖x − Gᵀu‖² + (σ/2)‖u‖² and ψ̃\_p = ψ\_p − (σ/2)‖·‖², which stays convex.
    - Step τ = 1/(L + σ), with L = 4 Σ w\_a² as in the isotropic kernel.
    - The dual step is u⁺\_p = proj\_{‖·‖ ≤ γα\_p}(y\_p / (1 + τ(δ/(γα\_p) − σ))), with y = v + τ(G z(v) − σ v), in closed form.
    - The momentum is (√κ − 1)/(√κ + 1), with κ = 1 + γ α\_max L / δ.
  - The rate is linear: of order √κ log(1/tol) iterations. In 2D at γ = 0.1, δ = 0.01 and unit weights, κ = 81. The plain TV kernels converge sublinearly, because their dual is degenerate.
  - Restarts are off. Below δ of about the certified error of a TV solve, the TV kernels are the better choice, and the docs say so.
  - Anisotropic Huber TV uses the same dual method, with intervals in place of balls. The accelerated block method of `TVAniso` needs exact 1D steps, and 1D Huber TV has no direct algorithm. The linear rate should make the block method unnecessary; the gate checks this.
- **Certificate.** Unchanged in form.
  - gap = Σ\_p [γα\_p h\_δ(‖g\_p‖) + δ‖u\_p‖²/(2γα\_p) − ⟨g\_p, u\_p⟩] with g = G z(u). Every term is ≥ 0 by Fenchel–Young, and ‖z − z\*‖² ≤ 2·gap (Beck, Lemma 12.7, with σ = 1 for the primal).
  - Stopping, warm starts through the dual, NaN images and the power-of-two rescaling are as for `TVIso`. The rescaled solve uses δ/c, since prox(c x; c γ, c δ) = c prox(x; γ, δ).
- **Derivatives.**
  - The prox objective has a Lipschitz gradient, so the prox is differentiable except where ‖g\_p‖ = δ. The system is (I + Gᵀ H G) dz = r, with H\_p = (γα\_p/δ) I where ‖g\_p‖ < δ and H\_p = (γα\_p/‖g\_p‖)(I − n\_p n\_pᵀ) where ‖g\_p‖ > δ, n\_p = g\_p/‖g\_p‖.
  - Its eigenvalues lie in [1, 1 + γ α\_max L / δ]. Conjugate gradients therefore need no flat regions, no union-find and no curvature cap. They use the Jacobi preconditioner of the isotropic derivative.
  - The isotropic derivative's cap (decision of 2026-10-18) is this H with δ at the certified error level, so the H-apply kernel is shared.
  - At ‖g\_p‖ = δ the default is the quadratic piece. The kink mask gives code 1 where |‖g\_p‖ − δ| ≤ tol, and `fp_workspace_set_branch` picks the quadratic (−1) or the linear (+1) piece.
  - The conjugate-gradient tolerance defaults to the accuracy the gap certifies, as for `TVIso`.
- **Parameters.**
  - θ = (w\_0, …, w\_{ndim−1}, α or β in C order when given, δ).
  - With v = Jᵀḡ from conjugate gradients, the gradient in a parameter ϑ is −Σ\_p ⟨(G v)\_p, ∂\_ϑ(γα\_p ∇h\_δ(g\_p))⟩, plus the product rule through G for the axis weights, as now.
  - In δ: ∂\_δ ∇h\_δ(g) = −g/δ² on the quadratic piece and 0 on the linear one, so δ can be learned.

```c
/* Huber-smoothed TV (1.x): g(z) = sum_p alpha_p h_delta(|(w_0 D_0 z, ..., )_p|_2)
 * (isotropic) or sum_a w_a sum_p beta_{a,p} h_delta(|(D_a z)_p|) (anisotropic),
 * h_delta as fp_op_huber, delta > 0 and finite (else FP_EPARAM). pixel_weights,
 * edge_weights: NULL, or as fp_op_tv_iso_weighted and fp_op_tv_aniso_weighted.
 * theta = (w..., alpha or beta in C order when given, delta). Iterative with a
 * duality gap, dual in the workspace, NaN images and rescaling as fp_op_tv_iso. */
FP_API fp_status fp_op_tv_iso_huber(int ndim, const fp_array *axis_weights,
                                    const fp_array *pixel_weights, double delta, fp_op **out);
FP_API fp_status fp_op_tv_aniso_huber(int ndim, const fp_array *axis_weights,
                                      const fp_array *edge_weights, double delta, fp_op **out);
```

- **Python.** `fp.TVIsoHuber(delta, ndim=2, weights=None, pixel_weights=None)` and `fp.TVAnisoHuber(delta, ndim=2, weights=None, edge_weights=None)`, with `grads.delta` alongside the existing fields.
- **PyTorch and JAX.** The device-native TV path, with the dual step above and the conjugate-gradient backward without regions.
- **Tests.**
  - The gap is recomputed independently.
  - The bound ‖z\_δ − z\_TV‖² ≤ γδΣα\_p, against `TVIso` and `TVAniso` for δ from 1e-1 to 1e-4.
  - CVXPY fixtures through its `huber` atom, rescaled.
  - Finite differences in x, γ, the weights and δ.
  - The linear rate: iterations against √κ log(1/tol).
  - The scaling identity, the kink sides, and determinism.

**Stage 2: spline potentials.**

- **The potential.** φ: ℝ → ℝ with φ(0) = 0 and φ′ = s + λ₁ sign.
  - s is continuous and piecewise linear through (t\_j, c\_j), j = 0, …, K − 1, for K ≥ 2 strictly increasing knots. Past the ends it is extended linearly with the slopes of the end pieces.
  - λ₁ ≥ 0 adds an ℓ1 kink at 0, which gives sparsity.
  - The pointwise operator is g(z) = Σ\_i w\_i φ(z\_i).
  - Special cases are Huber (knots ±δ), the elastic net, Huber plus (ε/2)t² (De los Reyes, Schönlieb and Valkonen's H¹ term on a scalar response), and learned spline potentials.
- **Convexity and the step condition.**
  - Let m\_j be the slopes of s, the two extensions included. φ is convex when every m\_j ≥ 0. Otherwise it is weakly convex with modulus ρ = −min\_j m\_j.
  - The prox is single-valued and Lipschitz when 1 + γ w\_i m\_j > 0 on every piece, that is γ ρ max w < 1. Otherwise the call returns FP\_EPARAM, as MCP does for b ≤ γ max w.
- **Prox, exact.**
  - z solves z + γ w φ′(z) ∋ x. The map F(z) = z + γ w s(z) is continuous, strictly increasing and piecewise linear. λ₁ adds a jump of 2γwλ₁ at 0, and that interval of x maps to 0.
  - A binary search over F(t\_j) = t\_j + γ w c\_j, computed on the fly so that per-entry weights work, finds the piece. Then z = t\_j + (x − F(t\_j)) / (1 + γ w m\_j).
  - This is exact to rounding, at about log₂ K comparisons per entry. The value φ(z) comes from prefix integrals of s at the knots, so `fp_value` is exact too.
- **Derivatives.**
  - In x: ∂z/∂x = 1/(1 + γ w m\_j) on the piece, and 0 in the dead zone.
  - In the step and the weights: off the dead zone, ∂z/∂γ = −w φ′(z)/(1 + γ w m\_j), and the same with γ for w.
  - In the coefficients: s(z) = (1 − θ) c\_j + θ c\_{j+1} with θ = (z − t\_j)/(t\_{j+1} − t\_j). So ∂z/∂c\_j = −γ w (1 − θ)/(1 + γ w m\_j) and ∂z/∂c\_{j+1} = −γ w θ/(1 + γ w m\_j), two nonzeros per entry; the extensions use the end pair.
  - In λ₁: ∂z/∂λ₁ = −γ w sign(z)/(1 + γ w m\_j).
  - Gradients in the coefficients and λ₁ are sums across rows, in the fixed blocks of `reduction_blocks`. The knots are fixed and are not parameters.
- **Kinks.** Code 1 where z sits at a knot (the slope changes), code 2 where x is at an end of the dead zone. `fp_workspace_set_branch` picks the piece. There are no ties.
- **Group form.** g(z) = Σ\_g c\_g φ(‖z\_g‖), by the norm-composition kernel of group Huber, SCAD and MCP (Beck, Theorem 6.18, for φ + ι\_{[0,∞)}, which needs only convexity). Weakly convex potentials take the same step condition as group SCAD and MCP.

```c
/* Spline potential (1.x): g(z) = sum_i w_i phi(z_i), phi(0) = 0, phi' = s + l1 sign(z),
 * s continuous piecewise linear through (knots[j], coef[j]), K >= 2 strictly
 * increasing finite knots, extended linearly past the ends; l1 >= 0. weights
 * as fp_op_scad. The prox needs 1 + gamma w_i m > 0 for every slope m of s
 * (else FP_EPARAM). theta = (coef..., l1, w...) (w only when weights are given). */
FP_API fp_status fp_op_spline(const fp_array *knots, const fp_array *coef, double l1,
                              const fp_array *weights, fp_op **out);
/* The group form sum_g c_g phi(|z_g|_2), groups as fp_op_group_l2.
 * theta = (coef..., l1, c...). */
FP_API fp_status fp_op_group_spline(const fp_array *knots, const fp_array *coef, double l1,
                                    int64_t n, const int64_t *groups, int64_t ngroups,
                                    const fp_array *group_weights, fp_op **out);
```

- **Python.** `fp.Spline(knots, coef, l1=0.0, weights=None)` and `fp.GroupSpline(knots, coef, l1=0.0, groups=None, group_weights=None)`, with `grads.coef`, `grads.l1` and `grads.weights`. Both are elementwise or radial classes, so `jacobian_diag` and the kink API apply.
- **Tests.**
  - Against minimisation on a refined grid, as for the nonconvex maps.
  - Huber and the elastic net as special cases, to rounding.
  - Finite differences in the coefficients and in λ₁.
  - Inputs at the step-condition boundary, and one-sided differences at knots.
  - Membership of `CASES`.

**Stage 3: analysis prior with a filter bank.**

- **Filters.**
  - Filter f is a small array w\_f of shape s\_f (up to 9 per axis), anchored at its first entry: (w\_f ⋆ z)\_p = Σ\_q w\_f[q] z\_{p+q}.
  - The response is zero wherever p + q leaves the image for some q in the support (the valid response, extended by zero). The two-tap forward differences of the TV kernels are this case, so W = D gives the isotropic and anisotropic TV operators exactly. Those keep their kernels.
- **Groups and potentials.**
  - Each filter belongs to a group k. The group's response at p is the vector of its filters' responses.
  - Each group has one potential, given as an operator handle: `fp_op_soft` (λ|t|), `fp_op_elastic_net`, `fp_op_huber` or a convex `fp_op_spline`. For a group of more than one filter it is applied to the norm, as in the group forms.
  - The potentials' parameters are copied at construction, as for every other operator, so the handles may be freed afterwards. Other operators return FP\_EINVAL.
  - Internally a potential provides φ, φ′, φ″, ∂\_θφ′, the conjugate's value and the prox of its conjugate. All four of the above have these in closed form.
- **Method.**
  - The dual method of stage 1, with W in place of G: u\_{k,p} ∈ ℝ^{c\_k} and z(u) = x − Σ\_k W\_kᵀ u\_k. The dual step of group k is the prox of (γα φ\_k ∘ ‖·‖)\*, in closed form by the Moreau identity from the potential's own radial prox.
  - The step is τ = 1/L, with L = Σ\_f ‖w\_f‖₁². This holds by Young's inequality ‖w ⋆ z‖ ≤ ‖w‖₁‖z‖, and gives the TV kernel's 4 Σ w\_a² for differences.
  - When every potential has a Lipschitz derivative (the Huber function, the elastic net without its ℓ1 part, a spline with λ₁ = 0), its conjugate is strongly convex and V-FISTA applies as in stage 1. Otherwise the method is FISTA with the isotropic kernel's gradient restarts.
- **Certificate.** gap = Σ\_{k,p} [γα\_{k,p} φ\_k(‖g\_{k,p}‖) + (γα\_{k,p} φ\_k)\*(‖u\_{k,p}‖) − ⟨g\_{k,p}, u\_{k,p}⟩]. The conjugates are in closed form (an indicator, quadratics, piecewise quadratics for splines), so ‖z − z\*‖² ≤ 2·gap as for TV.
- **Derivatives.** At the solution, z − x + γ Σ\_k W\_kᵀ α\_k ∇φ\_k(W\_k z) ∋ 0.
  - **Smooth entries.** H\_{k,p} = γα\_{k,p}[φ″(r) n nᵀ + (φ′(r)/r)(I − n nᵀ)] with r = ‖g\_{k,p}‖, and γα φ″(0) I at r = 0 where φ′(0) = 0.
  - **Flat entries.** These are g\_{k,p} = 0 under |t| or λ₁ > 0, read from the dual strictly inside its bound as `TVIso` reads them. They give constraints (W\_k dz)\_p = 0.
  - **Without flat entries:** conjugate gradients on (I + Wᵀ H W) dz = r.
  - **With flat entries:** the constraints merge pixels into regions only when W is a difference operator, so there is no union-find here. The design is MINRES on the symmetric saddle-point system with blocks I + W\_Aᵀ H W\_A and W\_Fᵀ in the first row and W\_F and 0 in the second, for the unknowns (dz, μ) and the right-hand side (r, 0). It is consistent, μ need not be unique and dz is, and it uses a block-diagonal preconditioner.
  - **The alternative to measure first:** the isotropic kernel's cap, with flat entries treated as Huber at the certified error level. That is plain conjugate gradients, but stiffer as the gap shrinks. The gate decides.
  - **Parameters.**
    - θ = (the filters in order, α\_k in C order when given, then each potential's θ in group order).
    - With v = Jᵀḡ and s\_{k,p} = α\_{k,p} ∇φ\_k(g\_{k,p}) (the dual over γ), the gradient in filter entry w\_f[q] is −γ Σ\_p (s\_{f,p} v\_{p+q} + (H W v)\_{f,p} z\_{p+q}). That is two correlations per filter, O(N |w\_f|).
    - The potentials' gradients go through ∂\_θφ′ at r.
- **Second-order TV.** `fp_op_tv2` is a fixed instance.
  - One group holds the second differences D\_aa for each axis and √2 D\_ab for a < b: 3 filters in 2D, 6 in 3D. With |t| this is Σ\_p α\_p ‖∇²z\_p‖\_F, the Frobenius (Schatten-2) Hessian norm; with h\_δ it is its Huber smoothing.
  - L = 64 in 2D and 144 in 3D. Affine images have zero responses, so they are fixed points of the prox.
  - The Schatten-1 Hessian norm needs a 2 × 2 or 3 × 3 eigendecomposition per pixel, and waits for the closed-form small decompositions of the spectral operators.

```c
/* Analysis prior (1.x): g(z) = sum_k sum_p alpha_{k,p} phi_k(|(W_k z)_p|_2) on the
 * last ndim (2 or 3) axes, batched over the others. nfilt filters: fshapes holds
 * nfilt * ndim extents (each 1..9), filters their entries concatenated in C
 * order; group[f] in [0, ngroups), every group nonempty; potentials: ngroups
 * handles among fp_op_soft, fp_op_elastic_net, fp_op_huber and a convex
 * fp_op_spline (copied; else FP_EINVAL). pixel_weights: NULL or
 * (ngroups, image shape), finite and >= 0; the operator then takes only that
 * shape. theta = (filters, alpha, potentials' theta in group order).
 * Iterative with a duality gap, as fp_op_tv_iso. */
FP_API fp_status fp_op_analysis(int ndim, int64_t nfilt, const int64_t *fshapes,
                                const fp_array *filters, const int64_t *group, int64_t ngroups,
                                const fp_op *const *potentials, const fp_array *pixel_weights,
                                fp_op **out);
/* Second-order TV: sum_p alpha_p |Hessian of z at p|_F by forward second
 * differences, with h_delta in place of the norm when delta > 0. */
FP_API fp_status fp_op_tv2(int ndim, const fp_array *pixel_weights, double delta, fp_op **out);
```

- **Python.**
  - `fp.Analysis(filters, groups=None, potentials=fp.Soft(1.0), pixel_weights=None)`, where filters is a list of arrays and potentials one operator or one per group.
  - `fp.TV2(ndim=2, pixel_weights=None, delta=0.0)`.
  - `grads.filters` is a list, beside `grads.pixel_weights` and `grads.potentials`, one `Grads` per group.
  - The PyTorch and JAX device paths use the framework's convolutions for W and Wᵀ, with the pointwise dual steps fused.
- **Tests.**
  - Forward differences as filters, against `TVIso` and `TVAniso` within the gaps.
  - Second-order TV against CVXPY, and affine images as fixed points.
  - The gap recomputed independently.
  - Finite differences in x, the filters, the pixel weights and the spline coefficients.
  - MINRES against the capped system on images with large flat sets.
  - Determinism and the allocation counter.

**Stage 4: weakly convex potentials.**

- **When the prox exists.** A weakly convex spline (modulus ρ) inside the analysis prior has no conjugate dual. The prox objective P(z) = ½‖z − x‖² + γ Σ α φ(‖W z‖) is m-strongly convex with m = 1 − γ ρ α\_max L, which must be > 0 (else FP\_EPARAM).
- **The primal method.** With λ₁ = 0, ∇P is Lipschitz, and the method is the constant-momentum accelerated gradient for strongly convex problems (Beck, Theorem 10.42, with no nonsmooth part).
- **Certificate.** It stops on ‖∇P(z)‖² ≤ 2m·tol·½‖x‖², and reports gap = ‖∇P(z)‖²/(2m), a bound on P(z) − P\*. Then ‖z − z\*‖² ≤ 2·gap/m, and the docs state the factor 1/m.
- **Derivatives.** As in stage 3. I + Wᵀ H W ⪰ m I, so conjugate gradients apply.

**Not planned here.**
- **ICNNs** (above).
- **TGV and ICTV.** They couple a second variable inside the norm, so their prox is not the prox of potentials on one linear operator. It needs a primal–dual solver whose iterates are not dual-feasible, which breaks the gap certificate. The single-loop route needs only the dual projections they are built from (Section 7, "Coderivatives for nonsmooth bilevel methods").
- **The Schatten-1 Hessian norm** (above).

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
| Anisotropic, graph and tree TV | averaging over the regions of the solution, with boundary terms in γ and the weights from the jump signs (revised 27 October 2026: this row described the 0.1 plan) | region labels, region sizes, edge signs |
| Isotropic TV | M(Mᵀ(I + GᵀHG)M)⁻¹Mᵀ, with M the flat regions and H the curvature of the non-flat pixels, solved in region space by conjugate gradients | region labels, flat mask, the dual |

- **Parameter derivatives.** For 1D TV, the derivative in γ or in w\_i is constant on each segment and depends only on the segment length, the weights at its two ends and the signs of the jumps there, so it is also O(n). For thresholding maps, the derivative in λ is −γ ψ'(z\_i) / (1 + γλ ψ''(z\_i)) on the support.
- **Kinks.** At points where the prox is not differentiable (an entry exactly at a threshold, a segment boundary with zero jump), the library returns the Jacobian of the active structure it selected, consistent with the tie rule in Section 4. Callers that need a full Clarke Jacobian are out of scope. Revised 23 October 2026: the regular and limiting coderivatives at kinks, and mixtures of the one-sided pieces, come in 1.x ("Coderivatives for nonsmooth bilevel methods" below). Since 1.0.2 the selected piece itself is queryable: its fixed entries, regions and constraint ("Structure of the derivative" below).
- **Iterative operators.** For isotropic and multi-dimensional TV, the VJP solves a linear system on the dual active set by conjugate gradients, warm-started and stopped at a user tolerance. The returned residual tells bilevel codes with inexactness control, such as MAID-type methods, how accurate the hypergradient is.
- **Verification.** Every derivative is tested against finite differences away from kinks, and the JVP and VJP are tested for adjointness: ⟨v, J u⟩ = ⟨Jᵀ v, u⟩ to rounding.

### Coderivatives for nonsmooth bilevel methods (1.x, design)

**Why coderivatives.** Where an inner problem is nonsmooth, its solution map has no derivative at kinks. Its adjoint equation then becomes an inclusion over a coderivative, Fréchet (regular) or limiting. The default derivatives of Section 7 return one element: the Jacobian of the selected piece. Two kinds of method need more.
- **Single-loop methods.** These take one inner step, one adjoint step and one outer step per iteration, and never solve the inner problem. Suonperä and Valkonen (2026, Algorithm 5.1) do this for nonsmooth inner problems:
  - the inner step is a primal–dual proximal splitting (PDPS) step, whose dual update is the prox of a conjugate, a projection onto pointwise balls for TV;
  - the adjoint step is a projected gradient step on a quadratic (their (5.20)), over a set built from that projection's structure: (5.21) for the Fréchet coderivative, (5.22) for the limiting one.
- **Implicit or adjoint methods that want a chosen element.** Examples are the element of least norm, or one consistent with an optimistic or pessimistic selection (Suonperä and Valkonen, Section 3). They need the whole set at the kinks, not one of its elements.

fastprox does not ship these methods: bilevel algorithms belong in a separate package on the public API (decision of 2026-10-23). The sets, however, are known exactly and cheaply only inside the operator. fastprox provides them for three families, in this order.

**Relation to the structure of the derivative (1.0.2).** The default derivatives, and the structure queries of 1.0.2 ("Structure of the derivative" below), describe one piece: the one the tie rule selects, or the one a branch override forces. A coderivative at a kink is a set over all the pieces that meet there, and each of those pieces is a structure the 1.0.2 queries describe under one choice of sides. So the families build on those queries rather than computing structure of their own (decision of 2026-10-27).
- **Family 1** reads its codes after `fp_prox_conj` from the states and the kink mask of the inner prox; only the multipliers, the codes of the image duals and the projections are new.
- **Family 2**'s two pieces at a kink are the structures under side −1 and side +1; the sets, their projections and the mixtures are new.
- **Family 3** is a quadratic programme in the coordinates of the structure, over the pieces that the sides of the row's kink entries select.

**Family 1: balls of a dual or a conjugate (closed form).**
- **Which operators.**
  - After `fp_prox_conj` with `keep_structure`, operators whose conjugate prox projects onto a product of balls: group ℓ2, with the ℓ2 norm as one group (Euclidean balls), and soft thresholding (intervals, the one-dimensional case). In a PDPS method for TV with K = ∇, the dual step is `fp_prox_conj` of group ℓ2.
  - After `fp_prox` with `keep_structure`, the dual balls of isotropic TV, of the analysis prior with |t| potentials (Section 5) and, as intervals, of anisotropic TV. These serve adjoint inclusions written on the prox's own dual.
- **The multiplier.**
  - For a conjugate projection, ball j with radius r\_j maps y\_j to u\_j = y\_j / max(1, ‖y\_j‖/r\_j). So y\_j = (1 + ν\_j) u\_j with ν\_j = max(0, ‖y\_j‖/r\_j − 1). This is the quantity of Suonperä and Valkonen's Remark 6.2. It recovers the normal-cone element [Δw\_d + K u\_p]\_j = v\_j u\_{d,j} from the PDPS projection without recomputing it. (Their remark writes B(0, 1) for a ball of radius C.)
  - For an image dual, the optimality condition g\_p ∈ N\_{B(γα\_p)}(u\_p), with g = W D z, gives g\_p = ν\_p u\_p with ν\_p = ‖g\_p‖/(γα\_p) on the sphere, from the saved z.
- **The codes**, per ball, with a tolerance as in `fp_workspace_kinks`:
  - 0: inside, ‖u\_j‖ < r\_j, read as the isotropic kernel reads its flat entries;
  - 1: strictly active, ‖u\_j‖ = r\_j and ν\_j > tol;
  - 2: degenerate, ‖u\_j‖ = r\_j and ν\_j ≤ tol.
  - Code 2 is the existing kink code of group ℓ2 and of isotropic TV.
  - After `fp_prox_conj`, the codes are not a second source of structure. The conjugate projection keeps y\_j inside its ball exactly where the inner prox (Section 7, "Structure of the derivative", edge cases) holds group j at ZERO, and puts it on the sphere with ν\_j = 0 exactly at the inner prox's kink. So at tol = 0 the codes are a function of the 1.0.2 states and the kink mask: 0 for ZERO entries off the kink, 1 for FREE entries off the kink, 2 at the kink. The implementation derives them that way, and a test checks the agreement. For tol > 0, ν\_j ≤ tol widens code 2 in ν's relative units, where the kink mask's tolerance is absolute in y's.
  - After `fp_prox` of the image operators, the codes are new: they are per dual ball (pixel, or edge for anisotropic TV), while the 1.0.2 structure describes the entries of x, and its flat regions do not say which pixel's ball is active.
- **The sets.** Per ball, for the adjoint pair (w\_j ∈ ℝ^{c}, λ\_j ∈ ℝ), from Suonperä and Valkonen's Lemma 5.5 and Corollary 5.6:

| Code | Fréchet, (5.21) | Limiting, (5.22) |
| --- | --- | --- |
| 0 | λ\_j = 0 | λ\_j = 0 |
| 1 | ⟨w\_j, u\_j⟩ = 0, λ\_j free | the same |
| 2 | ⟨w\_j, u\_j⟩ ≥ 0 and λ\_j ≥ 0 | {λ\_j ≥ 0, ⟨w\_j, u\_j⟩ ≥ 0} ∪ {λ\_j ≤ 0, ⟨w\_j, u\_j⟩ = 0} |

- **The projections.** Each is in closed form, O(c) per ball. The limiting case projects onto every convex piece and keeps the nearest, with ties going to the first piece, so the result is deterministic.
- **A discrepancy to settle before implementing.** Lemma 5.5's (5.18) states the third case as λ\_j⟨w\_j, u\_j⟩ ≥ 0 and max(⟨w\_j, u\_j⟩, λ\_j) ≥ 0. That set also contains {λ\_j = 0, ⟨w\_j, u\_j⟩ < 0} (case c.6 of its proof), which the two branches of (5.22) leave out. If (5.18) is the intended set, the limiting projection gets that third piece.

**Family 2: elementwise prox maps at kinks (closed form).**
- **Which operators.** Elementwise operators whose prox is piecewise linear in x: soft thresholding, the elastic net, Huber, box and orthant, ℓ1 over a box, SCAD, MCP and spline potentials, and their group forms in the norm. The maps with jumps at ties (ℓ0, ℓq and the log penalty) have unbounded coderivatives there and return FP\_EINVAL.
- **The sets.** Let a and b be the slopes in x of the pieces on the x − ε and x + ε sides of a kink, the pieces `fp_workspace_set_branch` already selects (sides −1 and +1).
  - The regular coderivative is the upper adjoint of the directional derivative (Rockafellar and Wets, Proposition 8.37, 8(26)): D̂\*prox(x)(y) = {v : v h ≤ y prox′(x; h) for all h}. That is the segment {v : a y ≤ v ≤ b y}, empty when (b − a) y < 0.
  - The limiting coderivative adds the two one-sided values: D\*prox(x)(y) = D̂\*prox(x)(y) ∪ {a y, b y}.
  - Away from kinks both are {slope · y}.
- **Jointly in x and the parameters.** At a kink, the two pieces' rows J\_− and J\_+ (derivatives in x, γ and θ) agree on the kink's hyperplane x = κ(γ, θ). The same argument gives the regular coderivative {(1 − s) y J\_−ᵀ + s y J\_+ᵀ : s ∈ [0, 1]} when (b − a) y ≥ 0 and the empty set otherwise. The limiting coderivative adds the endpoints s = 0 and s = 1.
- **Group forms.** For a group in the norm (group soft thresholding at ‖x\_g‖ = γλc\_g, group SCAD, MCP and splines), the pieces meet on a sphere rather than a hyperplane. The directional derivative is still J\_− h or J\_+ h by the side of the normal n, with J\_+ − J\_− = c nᵀ by continuity. So the same holds with (b − a) y replaced by ⟨c, y⟩: v = J\_−ᵀ y + t n with 0 ≤ t ≤ ⟨c, y⟩. Its projection is a small quadratic programme with two linear constraints, solved by cases.
- **The two pieces are the 1.0.2 structures.** J\_− and J\_+ are the Jacobians of the structures under side −1 and side +1 (`ctx.structure(branch=-1)` and `branch=+1`): for the projector maps, ZERO or LOWER/UPPER on one side and FREE on the other. The coderivative needs no structure beyond them.
- **Fractional branches.** So every element is a mixture of the two pieces with a weight s\_i ∈ [0, 1] per kink entry. `fp_workspace_set_branch_weights` generalises `fp_workspace_set_branch` (s = 0 is side −1, s = 1 side +1), and `fp_jvp` and `fp_vjp` then differentiate the mixture. That includes the parameter gradients, which is what a hypergradient through the selected element needs.
- **The projections.** Per entry, onto the graph of the coderivative in the (y, v) plane:
  - a convex cone at a kink in the regular case;
  - the cone and the two lines v = a y and v = b y in the limiting case, the nearest kept;
  - the line v = slope · y elsewhere.
  - Each is closed form in O(1). The call also returns the s that represents the projected v, to pass to `fp_workspace_set_branch_weights`.

**Family 3: polyhedral rows (design only, later).**
- **Which operators.** The simplex, the ℓ1 ball, the hyperplane with a box, isotonic regression, and 1D, anisotropic, graph and tree TV. They also have piecewise-affine prox maps, so 8(26) gives the regular coderivative as a polyhedral cone. The pieces are indexed by the kink structure: which threshold entries join the support, which zero-jump edges join or split.
- **Why not closed form.** One isolated kink in a row is the segment of family 2, between the two pieces' VJPs. Several kinks in one row interact: one entry joining the support moves τ for the others, and one edge joining moves a segment's mean. The set is then a cone over 2^m pieces, and its projection is a small quadratic programme over the m kink entries of the row.
- **What 1.0.2 already gives.** The kink mask counts m per row, and each piece is queryable: setting the sides of the row's kink entries with `fp_workspace_set_branch` and reading the structure gives the piece's subspace L\_σ, the range of its Jacobian J\_σ, which is the projector onto L\_σ for the PROJECTOR rows. The pieces with a full-dimensional cone of directions are the B-Jacobians, so their L\_σ are the subspaces of the SCD derivative (Gfrerer and Outrata 2022); the regular coderivative is {v : ⟨v, h⟩ ≤ ⟨y, J\_σ h⟩ for every σ and every h in the cone C\_σ of σ}, by 8(26).
- **What is planned.** The cones C\_σ are sign conditions on the m kink coordinates, so the set is described by the structure at one side plus m variables, not by 2^m explicit pieces. The design is a QP in those coordinates (the regions and the constraint of L, and the m kink entries), solved exactly by an active-set method, scheduled when a user needs more than isolated kinks.

```c
/* Family 1 (1.x). After fp_prox_conj on fp_op_group_l2 (groups NULL: the l2
 * norm) or fp_op_soft, or after fp_prox on fp_op_tv_iso, fp_op_tv_aniso or
 * fp_op_analysis with |t| potentials, both with keep_structure: per ball, the
 * multiplier nu >= 0 and the code 0 (inside), 1 (active, nu > tol) or 2
 * (degenerate). codes FP_I8 and nu (either may be NULL) have one entry per ball. */
FP_API fp_status fp_workspace_multipliers(const fp_workspace *ws, double tol,
                                          fp_array *codes, fp_array *nu);
typedef enum { FP_CODERIV_FRECHET = 0, FP_CODERIV_LIMITING = 1 } fp_coderiv_kind;
/* Family 1: projects (w, lambda) onto the coderivative set of the saved
 * projection, per ball; w has the shape of the dual, lambda one entry per ball.
 * Family 2: projects (y, v) per entry onto the graph of the coderivative of
 * the prox, w = y and lambda = v of the shape of x, and writes the mixture
 * weight to s (may be NULL). Outputs may alias inputs. */
FP_API fp_status fp_project_coderiv(const fp_op *op, const fp_workspace *ws,
                                    fp_coderiv_kind kind, double tol,
                                    const fp_array *w, const fp_array *lambda,
                                    fp_array *w_out, fp_array *lambda_out, fp_array *s);
/* Family 2: mixture weights s in [0, 1] (the shape of x) for the entries at a
 * kink within tol; fp_jvp and fp_vjp differentiate (1 - s) J_- + s J_+ there.
 * NULL clears it, as fp_workspace_set_branch(ws, NULL, 0). */
FP_API fp_status fp_workspace_set_branch_weights(fp_workspace *ws, const fp_array *s,
                                                 double tol);
```

- **Python.**
  - `ctx.multipliers(tol)` returns (codes, nu).
  - `op.project_coderivative(ctx, w, lam, kind="limiting", tol=0.0)` returns (w, lam) for family 1, and (y, v, s) for family 2.
  - `jvp` and `vjp` take `branch_weights=s` beside `branch=`.
- **Tests.**
  - Family 1: the sets against the case analysis of Lemma 5.5; each projection's variational inequality on each piece; the codes at the boundary within tol.
  - Family 2: the sets against 8(26) evaluated on the one-sided differences. The mixture's VJP is checked between the two one-sided VJPs, and the empty case for (b − a) y < 0.
  - Determinism for both.
  - A PDPS denoising run against Suonperä and Valkonen's Section 6 setup, in the bilevel package's test suite rather than here.
- **Out of fastprox.** The PDPS loop, the adjoint objective, the step sizes and the choice of element belong to the bilevel package.

### Structure of the derivative (1.0.2, design)

**Why the structure itself.** The derivative calls apply the Jacobian J of the prox in x at the kept point, but they never say what that Jacobian is built from. The structure is in the workspace already: the support of a projection, the segments of 1D TV, the region labels of image and graph TV. Three uses need it explicitly, not only through products.
- **Reduced systems for implicit differentiation through an outer iteration.** When the prox sits inside a solver with a forward operator A (deblurring, tomography), the hypergradient's adjoint system restricted to the identified structure is Mᵀ AᵀA M c = Mᵀ g, with M the matrix whose columns are the regions. It is far smaller than the full system: in the check of 2026-10-26 (1D TV deblurring, n = 120), it had 17 unknowns, and it agreed with the fixed-point adjoint built from `fp_vjp` to 2·10⁻¹⁴ and with finite differences to 6·10⁻¹¹. Without the labels the caller has to rebuild them from z with a threshold, which is fragile for image TV at loose tolerances.
- **SCD semismooth\* Newton steps.** The Newton direction of Gfrerer, Hubmer and Ramlau (Inverse Problems 41 (2025) 075002, Algorithm 3.8) is a stationary point of a quadratic programme over rng P, with P the orthogonal projector of a subspace in the SC derivative of ∂g. For the polyhedral operators P is determined by the structure alone ("Relation to SCD subspaces" below). `ctx.jacobian_diag` already serves the elementwise maps (0.2); this extends it to every row, image and graph operator.
- **Support-restricted hypergradients and the polyhedral coderivatives.** Sparse hyperparameter learning (Section 3, Use cases) wants the support as a by-product, and Family 3 of the coderivatives (above) is a QP in the coordinates of exactly this structure.

**Scope.** Queries on the kept point. No new operator, no change to `fp_prox`, `fp_jvp` or `fp_vjp`, and no solver: the reduced systems, Newton steps and hypergradients stay in the caller's code (Section 2, non-goals).

**The object: the tangent subspace L of each unit.** A unit is the part of the batch the operator solves as one problem (table below). For a unit with `size` entries, the structure is four things.
- **A state per entry**, s\_i ∈ {FREE, ZERO, LOWER, UPPER}. FREE entries move with x; the others are held: at 0 (thresholded, a killed group, off the support), at the lower bound or at the upper bound.
- **A region per FREE entry**, a label ℓ\_i ∈ {0, …, K − 1}, with K the unit's number of regions. Entries with one label move together: a singleton region is an entry that moves on its own.
- **A sign per FREE entry**, σ\_i ∈ {−1, +1}: the entry follows its region's coordinate with that sign. It is +1 everywhere except where an operator ties entries by magnitude (the sorted ℓ1 norm, the ℓ∞ norm, the sum of the k largest absolute values).
- **At most one linear constraint**, a vector a of the unit's size and a flag saying whether it is active. Six operators have one (table below); the others never do.

These define

```latex
L = \{\, d \in \mathbb{R}^{\mathrm{size}} :\; d_i = 0 \text{ if } s_i \neq \mathrm{FREE},\;\; d_i = \sigma_i\, c_{\ell_i} \text{ for some } c \in \mathbb{R}^K,\;\; a^\top d = 0 \text{ if the constraint is active} \,\}
```

so that L = {M c : c ∈ ℝᴷ, (Mᵀa)ᵀc = 0 when active}, with M the size × K matrix M\_{i,ℓ\_i} = σ\_i, and dim L = K − 1 with an active constraint (when Mᵀa ≠ 0), K otherwise.

**What the library guarantees.** Let J be the Jacobian in x that `fp_jvp` and `fp_vjp` apply at the kept point now: the default piece, or the one chosen by `fp_workspace_set_branch`.
1. **Range.** range J = L, not only ⊆: every direction in L is reached. For J symmetric, as for every convex operator (Section 4), the kernel is then L^⊥.
2. **Form of J,** per operator, given by `fp_op_struct_kind`:
   - **PROJECTOR:** J = P\_L, the orthogonal projector onto L. All the polyhedral operators: on a region it averages σ\_i x\_i, and an active constraint removes the component along P\_M a.
   - **WEIGHTED:** J is the projector onto L orthogonal in the metric diag(w). Only weighted isotonic regression, whose fit is a projection in that metric, not a Euclidean prox (decision of 2026-10-25).
   - **CURVED:** range J = L, but J is not a projector: J restricted to L is symmetric positive definite (the convex operators with curvature) or a positive diagonal (SCAD, MCP, ℓq, the log penalty: their slopes). The caller applies it by `fp_jvp`.
   - **NONE:** no structure query (the spectral operators; "Later" below).
3. **Parameters.** The columns of the Jacobian in γ and in the parameters θ respect the states, the regions and the signs (fixed entries do not move, tied entries move together), but not necessarily the constraint: a change of radius, α or b moves the affine level of the face. The guarantee on L is about x only.
4. **Determinism.** The four outputs are the same for any thread count, and the same for any strides of x and of the outputs.

**Relation to SCD subspaces.** For v = (x − z)/γ ∈ ∂g(z) at z = prox\_{γg}(x):
- **Convex g.** Gfrerer and Outrata (arXiv 2106.00519, Proposition 3.26 and Corollary 3.28; restated as Theorem 3.6 of Gfrerer, Outrata and Valdman 2022) show that ∂g is an SCD mapping, and that its subspaces at (z, v) are rge(B, (I − B)/γ) with B a limiting Jacobian of prox\_{γg} at x, symmetric positive semidefinite with ‖B‖ ≤ 1. The J of the Lipschitz class is the Jacobian of a smooth piece, an element of the B-subdifferential (Section 4), so rge(J, (I − J)/γ) is such a subspace.
- **In the (P, W) form** of Gfrerer, Hubmer and Ramlau (Definition 3.6: L = rge(P, W) with P² = P and W(I − P) = I − P, both symmetric), that subspace is
  - P = P\_L, the projector this section describes;
  - W = (I − P) + P (J\_L⁻¹ − I) P / γ, with J\_L the restriction of J to L, invertible by guarantee 1.
  - For PROJECTOR operators J\_L = I, so W = I − P: the structure alone gives the subspace, with no curvature term. For CURVED operators W's block on L needs J, which the caller applies through `fp_jvp` (for example by CG on J\_L y = s).
- **Prox-regular g.** SCAD and MCP under their step conditions, and the log penalty at γλ < ε²: Proposition 3.26 of the same paper covers prox-regular, subdifferentially continuous functions, with γ in the role of λ, so the same statement holds within the step conditions.
- **The discontinuous class** (ℓ0, ℓq, the log penalty above ε²): the states give the per-entry subspaces that `jacobian_diag` already documents, as a convention; this section claims nothing more there (Section 4).
- **Weighted isotonic regression:** the statements hold after the change of variables z ↦ W^½ z.
- **After `fp_prox_conj`:** Moreau's identity gives J\_conj = I − J, so the roles of the two components swap: for a PROJECTOR operator, the conjugate's tangent subspace is L^⊥. The queries describe the inner prox at y/σ, and say so (Python: `s.conj`).

**Units and numbering.**

| Operator class | Unit | size | Per-unit arrays (count entries) |
| --- | --- | --- | --- |
| Elementwise maps | the whole array | numel | count = 1 |
| Row operators (groups, projections, isotonic, sorted ℓ1, 1D TV and its variants) | each row along the axis | the axis length n | one per row, in C order over the dimensions other than the axis |
| Image operators (anisotropic and isotropic TV) | each image (the last 2 or 3 axes) | the image's pixel count N | one per image, in C order over the leading dimensions |
| Graph and tree TV | each graph instance (the last axis, of length n) | n | one per instance, in C order over the leading dimensions |

- **Labels are local to the unit and canonical:** numbered 0, 1, 2, … in the order of each region's first entry, in the unit's own order (along the axis for rows, C order over the image axes for images, node order for graphs, C order for elementwise maps). Two calls with the same structure give the same labels, whatever the strides, the axis or the thread count. Fixed entries have label −1.
- **Labels are int32 or int64,** as the caller chooses (new dtypes FP\_I32 and FP\_I64, accepted only by these queries). int32 returns FP\_ESHAPE for a unit with more than 2³¹ − 1 entries; the stored image and graph labels are int32 already.
- **Global numbering** across the batch is the offsets of the per-unit counts, in Python (`s.offsets`, `s.global_labels()`), not in C: a global int32 numbering would overflow on large batches.

**Operator by operator.** "Stored" means the derivative already saves it (inventory of `src/ops/*.cpp`, 2026-10-27), so the query is a copy and a scatter. "Recomputed" means O(1) per entry from the saved x and the operator's parameters, by the same function the derivative uses (`piece_of`, the group norms), so the two cannot disagree.

| Operator | States | Regions and signs | Constraint a (when active) | Kind | Source |
| --- | --- | --- | --- | --- | --- |
| Soft thresholding (weighted) | ZERO thresholded, FREE | singletons | — | PROJECTOR | recomputed |
| Elastic net | ZERO, FREE | singletons | — | CURVED | recomputed |
| Huber, log barrier | FREE | singletons | — | CURVED | — |
| Box, nonnegative orthant | LOWER, UPPER (orthant: LOWER at 0), FREE | singletons | — | PROJECTOR | recomputed |
| ℓ1 over a box | ZERO, LOWER, UPPER, FREE | singletons | — | PROJECTOR | recomputed |
| ℓ0 | ZERO (the zero branch of the tie rule), FREE | singletons | — | PROJECTOR | recomputed with `ws.tie` |
| ℓq, log penalty | ZERO, FREE | singletons | — | CURVED | recomputed with `ws.tie` |
| SCAD, MCP | ZERO (the thresholded piece), FREE | singletons | — | CURVED | recomputed |
| Group ℓ2 | ZERO on killed groups, FREE | singletons | — | CURVED | stored (`code`: group live) |
| Group Huber, SCAD, MCP | ZERO on groups set to 0, FREE | singletons | — | CURVED | recomputed |
| Tree-structured group ℓ2, sparse group lasso | ZERO where the ℓ1 step or a nested shrink zeroes the entry, FREE | singletons | — | CURVED | recomputed |
| ℓ2 ball | FREE | singletons | outside the ball: a = x/‖x‖ | CURVED | recomputed |
| Second-order cone (t first) | inside: FREE; in the polar cone: ZERO; on the boundary: FREE | singletons | on the boundary: a = (−1, y/‖y‖) | CURVED | recomputed |
| Simplex | FREE on the support, ZERO elsewhere | singletons | a = 1 on the support (none when the support is empty, r = 0) | PROJECTOR | stored |
| ℓ1 ball (weighted) | inside: FREE; projected: FREE on the support, ZERO elsewhere | singletons | projected: a\_i = w\_i sign(x\_i) on the support | PROJECTOR | stored |
| ℓ∞ norm | z = 0: ZERO; else FREE | the clipped set is one region with σ\_i = sign(x\_i); the rest singletons | — | PROJECTOR | stored |
| Maximum | FREE | the clipped (top) set is one region, σ = +1; the rest singletons | — | PROJECTOR | stored |
| Sum of the k largest | FREE; for absolute values with the sum constraint inactive, ZERO at the level (z = 0 there) | the entries at the shared level are one region, σ = +1 (sign(x\_i) for absolute values); entries below the level and at the cap are singletons | — | PROJECTOR | stored |
| Hyperplane, half-space, each with a box | LOWER, UPPER on the bounds (lo\_i = hi\_i: LOWER), FREE inside | singletons | the hyperplane, and the half-space when active: a\_i on the FREE entries (none when they are all 0) | PROJECTOR | stored |
| Isotonic regression | FREE | the pooled blocks | — | PROJECTOR; WEIGHTED with unequal weights | stored (block starts) |
| Sorted ℓ1 | ZERO on the block at 0, FREE | the blocks with z ≠ 0, σ\_i = sign(x\_i) | — | PROJECTOR | stored; a re-sort, O(n log n) |
| 1D TV (weighted) | FREE | the segments (an edge with a jump or a zero weight separates) | — | PROJECTOR | stored (edge codes) |
| 1D TV ball | FREE | inside: singletons; on the sphere: the segments of the 1D TV prox at γ\* | on the sphere: a = Dᵀ(w ⊙ sign(Dz)), with sign 0 where there is no jump (none when its segment means are all 0, as at α = 0) | PROJECTOR | stored |
| 1D TV with ℓ2 of differences | FREE | dual strictly inside its ball: the pieces between zero weights; otherwise singletons | — | CURVED | stored (inside flag) |
| Anisotropic TV (2D, 3D) | FREE | the regions of the saved structure | — | PROJECTOR | stored (`label`, `ncomp`) |
| Isotropic TV (2D, 3D) | FREE | the flat regions of the derivative's region space | — | CURVED | stored (`label`, `ncomp`) |
| Graph TV, tree TV | FREE | the regions | — | PROJECTOR | stored (`label`, `nreg`) |
| Spectral operators | — | — | — | NONE | — |

- **Checks on the table.** Each row follows from the Jacobian documented in Section 7 and `docs/api.md`; for instance the simplex's I − 1 1ᵀ/|S| on S is P\_L with a = 1\_S, the ℓ1 ball's P\_S − a aᵀ/‖a‖² is P\_L with a = w ⊙ sign(x) on S, and 1D TV's segment averaging is P\_L with the segments as regions. Where a row and the code disagree, the code's JVP is the reference and the table is corrected: guarantee 1 and 2 are tested against `fp_jvp` for every operator (Tests below), so a wrong row cannot ship.
- **Isotropic TV is CURVED:** its Jacobian is M(Mᵀ(I + GᵀHG)M)⁻¹Mᵀ on the flat regions (Section 7; `tv_iso.cpp`), whose range is span M. The curvature H, and with it W's block on L, is not exported ("Later").

**Edge cases.**
- **Kinks and branch overrides.** The queries describe the piece the derivatives use now. After `fp_workspace_set_branch`, the elementwise states come from `piece_of(b + side·δ)` exactly as `deriv_at` takes them, and the row and image queries read the overridden code or structure (`apply_structure`). So side −1 at the tail of a 1D TV kink edge merges two segments into one region, and +1 at a simplex threshold entry makes it FREE. Clearing the override restores the default.
- **Mixtures.** While fractional branch weights are set (`fp_workspace_set_branch_weights`, Family 2 above), J is a convex combination of two pieces and has no single tangent subspace: the queries return FP\_EINVAL.
- **The identity case** (γ = 0, λ = 0, a zero weight everywhere): every entry FREE and its own region, no constraint.
- **`fp_prox_conj` at σ = 0,** where nothing is saved and J = I: the inner prox's Jacobian, I − J, is 0, so every entry is ZERO, with no regions and no constraint (settled with the implementation, 2026-10-27).
- **A unit whose input holds a NaN:** its count of regions is −1, its labels −1, its states FREE, its signs 0 and its constraint inactive. The derivatives of such a unit are already undefined (NaN). For elementwise maps, whose entries are independent, only the NaN entries are marked so (FREE, label −1, sign 0), and they count neither as regions nor as fixed entries; the other entries keep their structure (decision of 2026-10-27).
- **Iterative image operators at a loose tolerance.** The structure is that of the saved point, as the derivatives are: it is the solution's only when `fp_workspace_deriv_exact` says so. After `fp_workspace_refine` the queries describe the refined point.
- **fp\_prox\_steps:** one step per row or image changes nothing here.
- **Devices.** A workspace on FP\_DEVICE\_CUDA returns FP\_EINVAL until the 1.x kernels save their structure.
- **Empty arrays:** count 0, nothing written, FP\_OK.

**C API** (all since 1.0.2; additions only, the ABI rules of Section 9 hold).

```c
/* FP_I32 and FP_I64 (since 1.0.2) are accepted only by the labels of
 * fp_workspace_regions. */
typedef enum { FP_F32 = 1, FP_F64 = 2, FP_I8 = 3, FP_I32 = 4, FP_I64 = 5 } fp_dtype;

typedef enum {
    FP_ENTRY_FREE  = 0, /* moves with x: on its own, or with its region */
    FP_ENTRY_ZERO  = 1, /* held at 0: thresholded, a killed group, off the support */
    FP_ENTRY_LOWER = 2, /* held at its lower bound (lower = upper included) */
    FP_ENTRY_UPPER = 3  /* held at its upper bound */
} fp_entry_state;

typedef enum {
    FP_STRUCT_NONE      = 0, /* no structure queries (spectral operators) */
    FP_STRUCT_PROJECTOR = 1, /* J is the orthogonal projector onto L */
    FP_STRUCT_WEIGHTED  = 2, /* J is the projector onto L orthogonal in diag(w) */
    FP_STRUCT_CURVED    = 3  /* range J = L; J is not a projector */
} fp_struct_kind;

/* The kind of every point of op (a property of the operator, not of a call):
 * an fp_struct_kind in *kind. */
FP_API fp_status fp_op_struct_kind(const fp_op *op, int *kind);

/* The units of the kept point: their number and the entries in each
 * (docs/spec.md, Section 7, "Structure of the derivative"). Either may be
 * NULL. FP_EINVAL without a kept point. */
FP_API fp_status fp_workspace_units(const fp_workspace *ws, int64_t *count, int64_t *size);

/* Per entry, an fp_entry_state (FP_I8, the shape of x, any strides; NULL for
 * the count only) and the number of entries not FREE (may be NULL). */
FP_API fp_status fp_workspace_states(const fp_workspace *ws, fp_array *state, int64_t *nfixed);

/* Per entry, the label of its region within its unit (FP_I32 or FP_I64, the
 * shape of x; -1 where the state is not FREE) and its sign (FP_I8, the shape
 * of x; +1 or -1, 0 where not FREE); per unit, the number of regions
 * (nregions, a contiguous array of count entries in the order of
 * fp_workspace_units; -1 for a unit whose input holds a NaN). Any of the
 * three may be NULL; count must equal the number of units when nregions is
 * given (FP_ESHAPE otherwise). */
FP_API fp_status fp_workspace_regions(const fp_workspace *ws, fp_array *labels, fp_array *sign,
                                      int64_t *nregions, int64_t count);

/* Per unit, whether its linear constraint is active (active, a contiguous
 * int8 array of count entries) and, per entry, its coefficient (coef, x's
 * dtype and shape; 0 where not FREE and in units without an active
 * constraint). Either may be NULL. Operators that never have a constraint
 * write zeros. */
FP_API fp_status fp_workspace_constraint(const fp_workspace *ws, fp_array *coef,
                                         int8_t *active, int64_t count);
```

- **Errors.** FP\_EINVAL without a kept point (as `fp_jvp`), for an operator of kind NONE, while branch weights are set, and on a device workspace; FP\_EDTYPE for a state or sign array that is not FP\_I8, labels that are not FP\_I32 or FP\_I64, or a coefficient array not of x's dtype; FP\_ESHAPE for a shape that is not x's, a count that is not the number of units, or int32 labels too narrow. Every argument is checked before anything is written, so a failed query writes nothing and leaves the workspace as it was.
- **Cost and allocation.** O(size) per unit (O(n log n) per row for the sorted ℓ1 norm), on the workspace's threads, parallel over units. The elementwise labels are a prefix count over the whole array, in the fixed blocks of `reduction_blocks` (two passes), so they do not depend on the thread count. No query allocates when the workspace exists: its scratch, including the per-block counts, is sized in `build_workspace`, and the allocation counter of `tests/c/test_core.cpp` covers them. A query costs at most one `fp_jvp` (performance gate below).
- **Thread safety.** As for the other queries: no two calls on one workspace may overlap.
- **Layout.** The saved structure is in canonical order (rows axis-last, images in C order over the leading dimensions: `RowPlan` and `ImagePlan` in `src/core/layout.hpp` and `driver.hpp`). The outputs are scattered to x's layout through the plan's offsets, as `fp_workspace_set_branch` gathers in the other direction.

**Python** (since 1.0.2).

```python
op = fp.TV1D(weights=w)
z, ctx = op.prox(x, gamma, keep=True)
s = ctx.structure()            # an fp.Structure, read under the context's lock
s.kind                         # "projector", "weighted", "curved"
s.state                        # int8, x's shape: 0 free, 1 zero, 2 lower, 3 upper
s.labels                       # int32 (int64 when a unit needs it), x's shape, -1 where not free
s.sign                         # int8, x's shape
s.nregions                     # int64, the batch shape (() for elementwise operators)
s.coef, s.active               # x's dtype and x's shape, and bool per unit; None for
                               # operators that never have a constraint
s.conj                         # True after prox_conj: s describes the inner prox
s.offsets                      # int64, count + 1: where each unit's regions start
s.global_labels()              # int64, x's shape: offsets[unit] + label, -1 where not free
s.reduce(v)                    # M^T v: the signed sum of v over each region, length offsets[-1]
s.expand(c)                    # M c, x's shape (0 where not free)
s.project(v)                   # P_L v (in the metric diag(w) for "weighted")
s.dim                          # dim L per unit: K - 1 with an active constraint, K otherwise
s.matrix(unit)                 # M of one unit, dense (size x K), for small problems and tests
```

- **The helpers are NumPy**, with `np.bincount` on the global labels; no SciPy, so the wheels still depend only on NumPy. `s.project` is the reference implementation of guarantee 2, used by the tests. With an active constraint it is Q v − (⟨b, Q v⟩/⟨b, b⟩) b, with Q the projector onto span M and b = Q a.
- **`ctx.structure()` after a branch override** is not possible in one call, since `branch=` applies to a single `jvp` or `vjp`. `ctx.structure(branch=side, kink_tol=0.0)` takes the same arguments and describes that piece.
- **The reduced deblurring system** of the motivating check, matrix-free: `lambda c: s.reduce(AT(A(s.expand(c))))`, by CG on its offsets[-1] unknowns.

**Tests.**
- **Guarantee 1 (range), every operator in `CASES`.** For random dx: J dx is 0 on the fixed entries, σ\_i (J dx)\_i is equal within each region, and aᵀ(J dx) = 0 under an active constraint, to rounding. Conversely, on small units, the matrix of J (n JVPs) has rank dim L, and J c ∈ L reaches each basis vector of L.
- **Guarantee 2 (form).** PROJECTOR: `fp_jvp` equals `s.project` to rounding for random dx; WEIGHTED: the same in the metric diag(w); CURVED: J maps L into L, and its restriction is symmetric positive definite (the convex ones) or a positive diagonal (SCAD, MCP, ℓq, log).
- **Branches.** The same checks after random sides at the kinks of every operator that has `fp_workspace_set_branch`, and after clearing it.
- **Edge cases.** The identity case, NaN units, empty arrays, `fp_prox_conj` (`fp_jvp` equals I − J of the inner structure), `fp_prox_steps`, refine on the image operators, and FP\_EINVAL for spectral operators, branch weights and device workspaces.
- **Determinism and layout.** The same outputs for 1, 2 and 15 threads, for transposed and strided inputs and outputs, and for every axis; labels in first-appearance order.
- **No allocation.** The allocation counter with a workspace; the C header check with FP\_I32 and FP\_I64.
- **Fuzzing.** `fuzz/fuzz_ops.cpp` checks guarantee 1 on the fuzzed rows, at the extreme parameter ranges of the 1.0.1 audit.
- **Examples as tests.** `examples/bilevel_deblurring.py`: the fixed-point adjoint, the reduced system through `s.reduce`/`s.expand` and finite differences agree (to 10⁻⁹ against finite differences, 10⁻¹² between the first two). `examples/scd_newton_lasso.py`: Algorithm 3.8 of Gfrerer, Hubmer and Ramlau for a Lasso problem with P from `s.project` and W = I − P, converging in a few steps once the support is identified. The Newton loop lives in the example, not in the library.

**Later (not in 1.0.2).**
- **The device-native image TV and the framework wrappers.** `_tv_torch.py` and `_tv_jax.py` build regions by hooking and pointer jumping but keep them private, labelled by the smallest global index. Compacting them to the canonical labels is a cumulative sum over the sorted keys (O(N), no new sort). Exposing them, and `structure` in `fastprox.torch` and `fastprox.jax`, waits for a user.
- **The curvature of the CURVED image operators.** Isotropic TV's H (the region-space operator of its CG) would give W's block on L without JVPs; it needs a design for exporting a sparse operator through a plain-C API.
- **The spectral operators.** Their structure lives in the clusters of the spectrum, which the caller's decomposition determines; a query would need the decomposition passed back.
- **The coderivatives** ("Coderivatives for nonsmooth bilevel methods", above) build on this section: Family 1 reads its codes after `fp_prox_conj` from the states and the kink mask, Family 2's pieces are the structures under side −1 and +1, and Family 3 is a QP in the coordinates this section defines.

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
- **Floating-point flags.** Device code builds with `--fmad=false -prec-div=true -prec-sqrt=true -ftz=false` and never with `--use_fast_math`, the device form of the no-`-ffast-math` rule. Contraction is also turned off on the CPU (`-ffp-contract=off` for GCC and Clang, whose defaults differ; done in 1.0.0rc2, where GCC's contraction broke determinism on aarch64). Then an elementwise map is the same sequence of correctly rounded operations on both, and matches bit for bit (NVIDIA, Floating Point and IEEE 754 Compliance, Sections 4.4 and 5.3). Maps that call `log` or `pow` (the log barrier, ℓq's Newton steps, the log penalty) agree to a few ulps: CUDA's double `log` is within 1 ulp and `pow` within 2 (CUDA Programming Guide, Section 5.5). Changing the CPU flag moved some arm64 results by an ulp, once, in 1.0.0rc2.
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
- **Threads.** `fp_op` is immutable and safe to share, and must outlive its workspaces. `fp_workspace` is not shared between concurrent calls; it holds one scratch region per pool thread, and no two calls on one workspace may overlap, derivative calls included. Results are bit-identical for any value of `threads`.
- **Errors.** Every function returns `fp_status`; there are no global error states, no printing and no aborts. Invalid parameters (negative γ, q outside (0, 1), a SCAD parameter too small for γ) return FP\_EPARAM before any work.
- **Derivatives.** `fp_jvp` and `fp_vjp` read the structure saved in the workspace by the last prox call on it, which must have had `keep_structure` set; otherwise they return FP\_EINVAL. A call rejected for its arguments does not count: it leaves the workspace as it was.
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
- **1.0.2 — structure of the derivative.** Queries for the entry states, region labels and linear constraint that define the range of each operator's Jacobian at the kept point, in C and Python, with the SCD (P, W) form for semismooth\* Newton callers (Section 7, "Structure of the derivative"). Added in a patch release at the maintainer's request (decision of 2026-10-27).
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

**Checked for the analysis priors and the coderivatives** (2026-10-23, in the user's Notability library; printed pages)

- A. Beck, *First-Order Methods in Optimization*, SIAM 2017 (N):
  - (6.43) and Example 6.54, pp. 163–164: the Huber function as the Moreau envelope of the norm.
  - Corollary 6.56, p. 165: the conjugate of a Moreau envelope, f\* + (μ/2)‖·‖².
  - Example 6.62, p. 167: its gradient.
  - Theorem 6.18, p. 142: norm composition, for g proper, closed and convex with dom g ⊆ [0, ∞).
  - Section 10.7.7 and Theorem 10.42, pp. 301–302: V-FISTA, with momentum (√κ − 1)/(√κ + 1).
  - Lemma 12.7, p. 356: ‖x̄ − x\*‖² ≤ (2/σ)(q\_opt − q(ȳ)).
  - Section 12.4.4, p. 365: 2D TV denoising by the dual.
- J. C. De los Reyes, C.-B. Schönlieb and T. Valkonen, [Bilevel parameter learning for higher-order total variation regularisation models](https://doi.org/10.1007/s10851-016-0662-8), JMIV 57, 2017 (N): Definition 2.1, the Huber function with γ = 1/δ; Section 5.1, γ = 100 and an H¹ weight μ = 1e-10.
- K. Kunisch and T. Pock, A bilevel optimization approach for parameter learning in variational models, SIAM J. Imaging Sci. 6(2), 2013 (N): (2.1), filter-bank regularisers Σ\_k ϑ\_k ‖K\_k x‖\_p^p with fixed convolution filters and learned weights ϑ.
- E. Suonperä and T. Valkonen, Single-loop approaches to nonsmooth bilevel optimisation, arXiv 2606.19143, 2026 (N):
  - Algorithm 5.1, p. 39.
  - Lemma 5.5 with (5.17)–(5.18), pp. 36–38.
  - Corollary 5.6 with (5.20)–(5.22), p. 38: read from the PDF's text layer, since the transcription garbles the cases.
  - Remark 6.2, p. 44.
- R. T. Rockafellar and R. J.-B. Wets, *Variational Analysis*, Springer 1998 (N): Proposition 8.37 and 8(26), the regular coderivative as the upper adjoint of the graphical derivative, which for a single-valued semidifferentiable map is its directional derivative.
- C. Clason and T. Valkonen, Introduction to nonsmooth analysis and optimization, arXiv 2001.00216 (N): the upper and lower adjoints of set-valued maps and the relations between derivatives and coderivatives, p. 293.
- J. Hertrich et al., Learning regularization functionals for inverse problems: a comparative study, arXiv 2510.01755 (N): Section 2.1, (6)–(7), the ridge-regulariser form and its gradient.

**Cited for the analysis priors from bibliographic records only** (not opened; check before citing in kernel headers)

- M. Hintermüller and G. Stadler, SIAM J. Sci. Comput. 28, 2006, doi:10.1137/040613263 — Huber-regularised TV in semismooth Newton methods.
- A. Goujon, S. Neumayer, P. Bohra, S. Ducotterd and M. Unser, A neural-network-based convex regularizer for inverse problems, IEEE TCI, 2023, doi:10.1109/TCI.2023.3306100 (CRR-NN); A. Goujon, S. Neumayer and M. Unser, Learning weakly convex regularizers for convergent image-reconstruction algorithms, SIAM J. Imaging Sci., doi:10.1137/23M1565243 (WCRR). Their knot grids, constraints and inner solvers are not yet checked.
- P. Bohra, J. Campos, H. Gupta, S. Aziznejad and M. Unser, Learning activation functions in deep (spline) neural networks, IEEE OJSP, 2020, doi:10.1109/OJSP.2020.3039379 — linear-spline activations.
- S. Lefkimmiatis, J. P. Ward and M. Unser, Hessian Schatten-norm regularization for linear inverse problems, IEEE TIP 22, 2013, doi:10.1109/TIP.2013.2237919 — second-order TV and its dual prox.
- Y. Chen, R. Ranftl and T. Pock, Insights into analysis operator learning, IEEE TIP, 2014, doi:10.1109/TIP.2014.2299065 — bilevel learning of filters and potentials.
- K. Bredies, K. Kunisch and T. Pock, Total generalized variation, SIAM J. Imaging Sci. 3(3), 2010, doi:10.1137/090769521.
- H. Gfrerer and J. V. Outrata, On (local) analysis of multifunctions via subspaces contained in graphs of generalized derivatives, arXiv 2106.00519 (v2) — SCD mappings; Proposition 3.26 and Corollary 3.28: the subspaces of ∂q for prox-regular and convex q are rge(B, (I − B)/λ) with B a limiting Jacobian of the prox (Section 7, "Structure of the derivative").
- H. Gfrerer, J. V. Outrata and J. Valdman, On the application of the SCD semismooth\* Newton method to variational inequalities of the second kind, Springer, 2022, pp. 1453–1484 — Theorem 3.6 (p. 1459), the same statement for convex q.
- H. Gfrerer, S. Hubmer and R. Ramlau, On SCD semismooth\* Newton methods for the efficient minimization of Tikhonov functionals with non-smooth and non-convex penalties, Inverse Problems 41 (2025) 075002 — Definition 3.6 (the (P, W) form), Lemma 3.7 and Algorithm 3.8 (the Newton step as a QP over rng P), Section 5 (ℓp and TV).

**To check before citing in code**

- Tooling documentation (nanobind, scikit-build-core, cibuildwheel, BinaryBuilder, the GCC Runtime Library Exception) was not opened while writing this spec; confirm the details in Section 12 against it.
- The journal of Gfrerer, Outrata and Valdman (2022): the library copy shows Springer, pp. 1453–1484, received 15 December 2021, published 28 November 2022, but not the journal's name (Set-Valued and Variational Analysis 30 is likely); confirm before citing. The converse used in Section 7, "Structure of the derivative" (every B-Jacobian of the prox gives a subspace of the SC derivative, not only every subspace comes from one), rests on the exact formula for graphically Lipschitzian mappings in Section 3 of arXiv 2106.00519; check the proposition number before citing it in code.

The survey this spec builds on: [Compiled solvers for variational imaging and nonsmooth optimization](https://claude.ai/code/artifact/51657fed-21f0-4ccc-b9ed-2d47aa2a97f3).
