# Review of the SCD subspaces of 2026-10-28: specs for 1.0.3

1.0.2 documents that the structure of the derivative gives, at the kept
point, a subspace of the SC derivative of ∂g in the (P, W) form of Gfrerer,
Hubmer and Ramlau (Inverse Problems 41 (2025) 075002, Definition 3.6): P the
orthogonal projector onto L = range J, and W = I − P for the PROJECTOR
operators (`docs/api.md`, "Structure of the derivative"; spec Section 7). This
review checked that claim operator by operator on master at `4438c69`.

**Method.** For each case, `reports/scd_review_2026_10_28.py`:
- builds the dense Jacobian J from `jvp`, and P\_L from `ctx.structure().project`;
- compares range J with L (guarantee 1);
- checks that J is symmetric with its spectrum in [0, 1]. Every limiting Jacobian of the prox of a convex function has this form (Gfrerer and Outrata, arXiv:2106.00519, Proposition 3.26);
- for the image operators, investigates whether the selected piece is a B-Jacobian by sampling nearby points and, for anisotropic TV, by a linear programme over the polyhedron of the piece. Sampling is evidence, not a proof of differentiability or of membership in the B-subdifferential.

**Scope of the guarantees.** Distinguish the exact mathematical derivative B of a selected piece, its floating-point implementation J, and the descriptor of L = range B. Successful queries must describe the selected piece; an unsupported range must be rejected rather than replaced by a larger space. Products approximate B to the accuracy established by the numerical tests. A numerical rank threshold on J cannot establish the rank of B when positive eigenvalues are below that threshold. SCD membership additionally requires that B be a limiting Jacobian at the stated point: symmetry and the convex spectral bounds alone do not establish this. Overrides within a nonzero `kink_tol` and rounding-level active-set decisions can select a convention for a nearby boundary rather than an exact piece at the supplied input. These qualifications must accompany the guarantees in spec Section 7.

## What holds

- **The (P, W) formula.**
  - For B symmetric positive semidefinite with ‖B‖ ≤ 1 and range B = L, rge(B, (I − B)/γ) = rge(P\_L, W) with W = (I − P) + P(B\_L⁻¹ − I)P/γ.
  - P and W are symmetric and commute, so the subspace equals its adjoint, and 𝒮∂g = 𝒮\*∂g there.
  - With f = ½‖Ax − b‖², the sum rule (GHR Lemma 4.1) gives rge(P, W + PHP). That is the system `examples/scd_newton_lasso.py` solves.
- **One subspace suffices.** Step NS of Algorithm 3.8 selects one valid element of 𝒮\_{P,W}∂φ(x̂, ŷ). Theorem 3.9 also requires Assumption 3.1, SCD semismoothness*, regularity, a sufficiently close start and the algorithm's approximation step. A valid default piece suffices; this is not a convergence guarantee for arbitrary numerical derivatives or overrides. See [GHR, Algorithm 3.8 and Theorem 3.9](https://arxiv.org/html/2410.13730v1#S3.SS2). Branch arrays explore supported selections, not necessarily all subspaces.
- **Polyhedral elementwise and row operators.** Soft thresholding on both sides of a kink, the simplex and 1D TV: J = P\_L to rounding, and both sides of an elementwise kink are B-Jacobians.
- **Graph and tree TV.** In exact arithmetic, joining every zero-jump edge with positive penalty gives a B-Jacobian. Zero-penalty edges impose no equality and are excluded. The implementation (`src/ops/edge_tv.hpp`) uses a rounding allowance, subject to the numerical qualification in S3:
  - Let p be the dual at x, and p\_J its restriction to the joined edges.
  - The point x′ = x − tγDᵀp\_J (0 < t < 1) has the same solution z, since its subgradient condition holds with p\_J scaled by 1 − t. At x′ every joined edge is strictly inside its bound.
  - The regions are connected through the joined edges, so the dual stays strictly feasible for every small perturbation of x′. The prox is therefore affine near x′, with J the averaging over the regions.
  - As t → 0, x′ → x.

  Where the prox is differentiable, this piece is its Jacobian. On 6 × 6 grids, 98 of 98 pieces that differ from anisotropic TV's passed the linear programme.
- **Isotropic TV.**
  - On 40 images: J symmetric to 6·10⁻¹⁵, spectrum in [0, 1], range J = L to 3·10⁻¹⁴. Finite differences agree to 10⁻⁷–10⁻⁸ at the best step; the larger differences grow like 1/ε and come from the perturbed solves.
  - Exact zero-gradient pixels impose flat constraints. Extending the joined-piece argument to isotropic TV also requires the smooth curvature of the nonzero-gradient pixels; the polyhedral graph argument alone does not prove it. The numerical survey supports the selected derivative, but does not certify all boundary cases.
  - Zero differences at pixels with a nonzero gradient are left to the curvature, as they should be.
- **The second-order cone, group SCAD and MCP, tree ℓ2 and the sparse group lasso.** Their kinks take the zero (or identity) side and accept no override, and those sides are B-Jacobians.

## Summary

| Item | Kind | Effort |
| --- | --- | --- |
| S1. Group ℓ2 forced live at its kink | correctness | S |
| S2. 1D TV with ℓ2 of differences near its boundary | correctness | M |
| S3. Anisotropic TV splits flat regions by its dual | correctness | M |
| S4. Weighted isotonic regression: the Euclidean (P, W) pair | docs | S |
| S5. What the docs say about SCD subspaces | docs | S |
| S6. Tests at exact kinks and on boundaries | tests | M |
| R1. The 1.0.3 release | process | S |

S1–S3 give wrong derivatives or a wrong subspace at inputs that a caller can reach: an override, a point on a boundary, an image with tied values. S1 and S3 are reported as exact.

---

<a id="s1"></a>
### S1. Group ℓ2 forced live at its kink

**Problem.**
- **The convention.** For group ℓ2, with τ\_g = γλc\_g, the override +1 keeps a group live with "the shrink, its formula taken at x" (`docs/api.md`, kinks): J\_g = (1 − c)I + c u uᵀ, with c = τ\_g/‖x\_g‖ and u = x\_g/‖x\_g‖.
- **At ‖x\_g‖ = τ\_g exactly.** J\_g = u uᵀ, which has rank 1. This is the correct one-sided Jacobian, the limit along the live piece.
  - The structure reports the whole group FREE as singletons. In the report, rank J = 1 against dim L = 4, and |P\_{rge J} − P\_L| = 0.93.
  - Guarantee 1 fails, and W's block (J\_L⁻¹ − I)/γ does not exist.
  - The correct subspace on the group is span(u) × u^⊥: P\_g = u uᵀ and W\_g = I − u uᵀ. For a group of 3 or more entries, the descriptor of 1.0.2 (states, ±1 regions, at most one constraint per unit) cannot express span(u).
- **Within `kink_tol` below τ\_g.** c > 1, so J\_g has the eigenvalue 1 − c < 0. The report gives −10⁻⁶ at ‖x\_g‖ = τ(1 − 10⁻⁶) with `kink_tol=1e-5`. This is not a B-Jacobian, since those are positive semidefinite, and guarantee 2 (positive definite on L) fails.
- **The same shape elsewhere.** The live or outside limits of the second-order cone at its polar boundary (½(1, u)(1, u)ᵀ), of group SCAD and MCP at their thresholds, and of 1D TV with ℓ2 (S2) also have a range that is not of the 1.0.2 form. None of them takes an override, and their default sides are fine. Any future override must follow this item's rule.

**Design.**
1. **Derivative.** A group forced live with 0 < ‖x\_g‖ ≤ τ\_g takes the live piece's Jacobian at the radial boundary point x̂\_g = τ\_g u, c = 1:
   - J\_g = u uᵀ, and ∂z\_g/∂τ\_g = −u. The parameter columns are those at ‖x\_g‖ = τ\_g.
   - In `src/ops/group.cpp` (`jvp` and `vjp`): cc = min(τ t/m, 1) on live groups.
   - Groups with ‖x\_g‖ > τ\_g are live by default and unchanged.
   - At τ\_g = 0 retain the existing identity convention, including x\_g = 0; do not divide by a zero norm or reject its structure. At x\_g = 0 with τ\_g > 0 there is no radial direction: reject a forced-live override with `FP_EINVAL` during branch validation, before changing the workspace. Python raises `ValueError`. The default and side −1 still return the zero derivative.
   - Below τ\_g this is a boundary-evaluation convention, not a B-Jacobian at the original x\_g, where the prox is locally zero.
   - This matches the elementwise maps, which take their piece's slope. A group's Jacobian varies along its piece, and its value at the kink is the limit.
2. **Structure.** A unit holding a group in that state (live with 0 < ‖x\_g‖ ≤ τ\_g and τ\_g > 0, which only an override produces) has no descriptor:
   - `fp_workspace_states`, `fp_workspace_regions` and `fp_workspace_constraint` return `FP_EINVAL`, as they do under branch weights (spec Section 7, "Mixtures").
   - Nothing is written to any output, including earlier units in a batch. Preflight all units before the first output write, using a linear pass over the group norms.
   - Python's `ctx.structure` raises `ValueError`. The message says that the forced piece's range is span(x\_g) on the group. Side −1 or removing the override avoids it; a smaller `kink_tol` only helps strictly away from the kink.
   - `fp_workspace_units` is unchanged.
3. **Docs.**
   - `docs/api.md`: the kinks paragraph ("its formula taken at x" becomes "its value at the kink") and the structure's edge cases.
   - Spec Section 7: the unsupported-query case; a successful query still satisfies the range contract.
   - A row in the ROADMAP decisions table.
   - Real coefficients per region, which would describe span(u), are a 1.x candidate.

**Constraints.**
- No new function, and the ABI is unchanged.
- Results change only under a live override with ‖x\_g‖ ≤ τ\_g.
- Determinism and allocation are unchanged, since the pre-pass uses the query's scratch.

**Tests** (C and Python), with τ > 0, at ‖x\_g‖ = τ exactly (`kink_tol` 0) and at τ(1 ± 10⁻⁶) (`kink_tol` 10⁻⁵), with side +1:
- At and below τ: J\_g = u uᵀ to rounding. Structure queries return `FP_EINVAL`, or raise `ValueError`, with all outputs unwritten. Sentinel tests include a batch whose unsupported unit is last.
- Above τ: J\_g = (1 − τ/‖x\_g‖)I + (τ/‖x\_g‖)u uᵀ; the structure succeeds with full group range. All three cases have eigenvalues in [0, 1] to rounding.
- Parameter JVPs agree with live-side differences at x̂\_g = τu for the at/below cases, and at the original x\_g above τ. Below τ, finite differences at the original input are not a reference for this convention. Check JVP/VJP adjointness too.
- Cover τ = 0 (identity) and x\_g = 0, τ > 0 (forced-live rejection with the previous branch state preserved).
- Side −1 and the default are unchanged.
- After `prox_conj`, construct these cases at the inner input y/σ with inner step 1/σ. Check J\_conj = I − J\_inner; structure and rejection rules refer to J\_inner. Parameter checks differentiate the full Moreau identity, including its scaling.

**Acceptance.** Every accepted override of group ℓ2 has a positive-semidefinite selected derivative, implemented to rounding accuracy. Successful structure queries describe its mathematical range; unsupported queries and directionless live overrides fail atomically as specified above.

<a id="s2"></a>
### S2. 1D TV with ℓ2 of differences near its boundary

**Problem.**
- **The operator.** g(z) = ‖WDz‖₂.
  - Inside (‖u₀‖ ≤ γ), J is the piecewise mean.
  - Outside, J = H⁻¹ with H = I + cK − cππᵀ, where K = DᵀW²D, s = ‖WDz‖, c = γ/s and π = Kz/s.
  - The kernel applies H⁻¹ by Sherman–Morrison on A = I + cK. The direction δ = Dz/s comes from z, and the denominator is πᵀA⁻¹(tz\_c), also from z (`src/ops/tv1d_l2.cpp`, `setup`, `apply_hinv`).
- **As s → 0.** δ, read from differences of z at the rounding level, has a relative error of about ε‖x‖/s. The denominator 1 − cπᵀA⁻¹π cancels.
- **Measured (n = 5).**
  - Just outside the boundary, the largest eigenvalue of J exceeds 1 by 5·10⁻⁵, 10⁻⁷, 10⁻¹⁰ and 3·10⁻¹⁴ at relative distances 10⁻¹², 10⁻⁹, 10⁻⁶ and 10⁻³.
  - On the boundary to rounding, 54 of 200 random points give eigenvalues outside [0, 1], up to 5.83. Some also have numerical rank 2 while advertising L = ℝ⁵; numerical rank alone does not prove an incorrect mathematical range near this singular limit.
  - These are wrong derivatives, not only a wrong structure: the prox of a convex function has a Jacobian with spectrum in [0, 1].

**Design.**
1. **The identity.** With n = WDz/s, the unit dual direction, K − ππᵀ = DᵀW(I − nnᵀ)WD. So H = I + c DᵀW(I − nnᵀ)WD is symmetric with H ⪰ I, and J = H⁻¹ has its spectrum in (0, 1] for every c ≥ 0 and every unit n.
2. **n and μ from the proximal solve.** Use μ = s/γ = 1/c, the root already found by the outside trust-region solve, and its normalised dual direction n. Refactor that solve into a shared internal helper which the derivative and structure queries recompute from the saved x, γ and weights. This avoids adding persistent arrays to the workspace and makes all queries use the same classification and root convention.
   - Recover n from the solver's scaled edge variables (`uh = W dh`), evaluated at the final root μ, before rounding z back to the caller's dtype. Normalise once over all positive-weight edges together, not separately on each disconnected piece. Check for a finite nonzero norm before dividing.
   - Do not reconstruct n by subtracting the saved x and z: this can give zero when γ is small, and division by tiny weights amplifies residual errors. Do not recover μ from differences of the saved z either. Both failures occur even when a valid outside solve exists.
   - Retain power-of-two scaling. Represent extreme μ by a mantissa and exponent, or an equivalent scaled system, so a finite positive root is not replaced by zero or infinity merely because a scalar intermediate underflows or overflows. The implementation must cover the existing extreme-scale tests, including mixed and zero weights.
3. **Bordered derivative system.** With r = WD dx and T = WDDᵀW restricted to positive-weight edges (tridiagonal blocks, positive definite), solve

   ```
   [ μ I + T   n ] [ v ]   [ r ]
   [ nᵀ       0 ] [ η ] = [ 0 ]
   ```

   and set dz = dx − DᵀW v. In exact arithmetic this is J = I − Bᵀ(μ I + BBᵀ)⁻¹B with B = QᵀWD, where Q is an orthonormal basis of n^⊥. Q is used only in the derivation, not formed in the kernel.
   - For A = μI + T, solve a = A⁻¹r and b = A⁻¹n, then η = (nᵀa)/(nᵀb) and v = a − bη. Compute the positive denominator through an LDLᵀ factorisation as a sum of squares divided by positive pivots. All pieces share this one scalar η.
   - This removes the Sherman–Morrison denominator cancellation. It does not eliminate subtraction in v or dz, nor prove floating-point stability by itself. Validate both the bordered residual and the resulting derivative against independent high-precision references; do not clip eigenvalues to make the spectrum tests pass.
   - Two tridiagonal solves per application, O(n), plus the shared root setup. The VJP in x uses the same symmetric map.
   - For γ and positive weights, use π = DᵀWn and the right-hand side rθ = −dγ π − γ(2Dᵀ(n ⊙ dw) − π Σ\_e n\_e² dw\_e/w\_e), then dz = J(dx + rθ). For q = J gv, the adjoints are gγ = −πᵀq and gw\_e = −γ(2n\_e(Dq)\_e − (πᵀq)n\_e²/w\_e). These formulas use physical weights and step; the implementation must include the chain factors for its scaled weights. On the outside piece, zero-weight columns contribute zero by continuous extension; omit them from divisions. Inside, retain the existing parameter conventions and document separately any direction that changes the zero-weight support. Validate JVPs and VJPs independently.
4. **Error budget.** For fixed n let C = DᵀW(I − nnᵀ)WD. Its positive eigenvalues a give eigenvalues μ/(μ+a) of J; on ker C, J is the identity. Thus |∂J/∂μ| in each positive eigendirection is a/(μ+a)², at most 1/(4μ), or at most 1/a\_min⁺ when a\_min⁺ exists. The old bound ‖∂J/∂c‖ ≤ 1/c does not establish an O(ε‖x‖/γ) error from a rounded s. Budget root error, direction error and linear-solve error separately. No conditioning-independent accuracy claim is made for arbitrarily unequal weights.
5. **Boundary and structure.** The common inside test retains ‖u₀‖ ≤ γ as inside and returns the piecewise mean and its regions. Equality tests use exactly representable fixtures; randomly normalised inputs may lie on either side after rounding.
   - For a finite positive outside root, the mathematical J is positive definite and its structure has singleton regions. A saved z with zero differences does not change that classification: use the root and direction from the shared solve.
   - The formal μ → 0 outside limit is the projector onto {d : WDd ∈ span n}; it generally has neither full range nor just the piecewise-constant range. It is a reference limit, not a new runtime branch. Do not substitute this projector for a finite-root derivative while advertising full range, or substitute the inside mean solely because saved differences vanish.
   - γ = 0, a row of length one, and all-zero weights give the identity. Zero-weight edges are omitted from T, with uncoupled entries handled explicitly.

**Constraints.**
- No API change.
- The reformulation changes results at ordinary scales in the last bits; the CHANGELOG says so.
- It is a row kernel, so it does not depend on the thread count.
- Scratch remains O(n), with its exact size accounted for in `scratch_bytes`; the root and bordered solves may reuse buffers. Re-measure the structure-query performance gate after adding root recomputation.
- The scale handling of math audit M5 and audit K4 is kept: weights over 2ᵏ, and differences scaled by a power of two.

**Tests.**
- 200 random rows (n = 5, 17, 64, with and without zero weights), at relative distances 0 and 10⁻¹⁵ … 10⁻³ from the nominal boundary, plus exactly representable boundary fixtures:
  - For float64, J symmetric and with eigenvalues in [−10⁻¹², 1 + 10⁻¹²]. Test inside and outside expectations according to the classified piece.
  - Generate references from the original input, step and weights: solve the prox/root independently in at least 50 digits, then form its derivative. Increase precision until the reference stabilises, especially near the boundary or for extreme weight ratios. Do not feed the implementation's rounded n and μ into the reference.
  - On moderate, well-conditioned fixtures, operator-norm error is at most 10⁻¹². For inputs within the rounding uncertainty of the inside test, compare with the documented selected side and record this ambiguity separately from interior accuracy.
  - Check analytic rank and the descriptor for each side. Only infer rank by SVD when the reference's smallest positive singular value is safely above the test threshold. Near the outside limit, compare matrix actions and reference eigenvalues instead of asserting numerical full rank.
- Include γ small enough that saved x = z, outside cases whose saved z has zero differences, large common offsets, μ outside the ordinary double range, unequal positive weights, disconnected pieces and all-zero weights. Extreme-scale tolerances must be justified by conditioning and reference accuracy, not widened after a failure to hide a wrong piece.
- Keep existing finite-difference, adjointness and extreme-scale tests. Add parameter JVP/VJP references, float32 coverage with dtype-appropriate tolerances, and Moreau checks after `prox_conj` as in S6.

**Acceptance.** The exact-arithmetic selected derivatives have spectrum in [0, 1], and their descriptors follow the boundary rules above. The computed products meet the stated numerical tests, including the original failures and extreme-scale cases. Document the measured error and conditioning limits; the bordered identity alone is not a proof of a uniform floating-point error bound.

<a id="s3"></a>
### S3. Anisotropic TV splits flat regions by its dual

**Problem.**
- **The rule.** The regions are the components of the edges whose 1D dual is strictly inside its bound, |p\_e| < λ\_e(1 − 10⁻⁹) (`src/ops/tv_aniso.cpp`, `structure`). A flat edge with a saturated dual becomes a boundary edge, with the dual's sign.
- **Why it can split.** In 2D and 3D the dual is not unique: the axes' duals q\_a have only a fixed sum. The dual the solver returns can saturate every edge between two parts of one flat region, and the structure then splits the region.
- **The counterexample** (in the report): a 6 × 6 image with values in {−1, 0, 1, 2}, γ = 0.3, `tol=1e-20`.
  - z agrees with graph TV on the same grid to 2·10⁻¹⁶.
  - Anisotropic TV has 13 regions, graph TV 12.
  - One Jacobian was observed at 200 points within 10⁻⁷, evidence consistent with differentiability at x. This sampling alone does not prove differentiability.
  - The anisotropic `jvp` differs from central differences by 1.68; graph TV's agrees to 2·10⁻¹⁰.
  - `ctx.deriv.exact` is True.
- **Frequency.**
  - Observed only on tied data in this survey. On 6 × 6 images with three levels, the Jacobian differed from graph TV's on 27 of 50 images; most of those are valid pieces at kinks.
  - A linear programme over each differing piece's polyhedron found 2 of 98 that are not B-Jacobians.
  - None of 100 16 × 16 images (8 and 256 levels) or 40 at 32 × 32 hit a wrong derivative.

  So it is rare, but silent and reported as exact.
- **The docs.** `docs/api.md` (graph and tree TV kinks, revised in 1.0.1 for math audit C5) says anisotropic TV's dual of a region is unique. It is not. Its kink mask is therefore dual-dependent too, a superset of the prox's kinks, as graph TV's is.
- **The device backward.** `_tv_torch.py` joins anisotropic edges by the same dual rule. Its isotropic rule already joins pixels whose |g| is at the rounding level.

**Design.**
1. **Join every positive-penalty edge at the rounding level as well.** Such an edge is flat when its dual is strictly inside its bound, or when its jump in the saved z is within the rounding allowance τ₀ that the structure already uses for certified jumps (64ε(max|d| + 2 Σ\_a max\_e λ\_e)). Zero-penalty edges never impose equality, even when their endpoints have equal values; at γ = 0 the map is the identity.
   - This is graph TV's rule (`edge_tv.hpp`), and isotropic TV's for pixels.
   - Once the polish certifies the regions, z is the closed form on the dual's components. Components with mathematically equal values agree to rounding. The allowance may also join components with small nonzero jumps: this is a numerical selection convention, not evidence that their values are exactly equal. The derivative update leaves the saved z unchanged.
   - The boundary sign sums S\_{C,a} drop the joined edges.
   - At a loose tolerance, z(y) rarely has jumps at the rounding level, so the rule changes little there, and `deriv.exact` already reports those derivatives as inexact.
2. **Mathematical and numerical claims.** For an exact solution with exactly zero jumps, the joined piece is a B-Jacobian by the argument in "What holds" and equals the Jacobian wherever the prox is differentiable. A rounding-level merge is not guaranteed to be a B-Jacobian at the original input. `deriv.exact` retains its existing solver-certification meaning; it is not a certificate of exact SCD membership or of zero jump in real arithmetic. Document this limitation alongside the selection rule.
3. **Kinks and overrides.**
   - The default joins every zero-jump edge.
   - The mask flags zero-jump edges whose saved dual is saturated, a superset of the prox's kinks.
   - Side +1 keeps an edge apart with its dual's sign, and may select no piece of the prox.
   - The docs say so, as for graph TV. The 1.x candidate for graph TV kinks independent of the saved flow (math audit C5) extends to anisotropic TV.
4. **Device.** `_tv_torch.py` and `_tv_jax.py`: extend their isotropic rounding convention to anisotropic edges, |g\_e| ≤ w\_max·64ε·max|x|, retaining the positive-penalty guard. Here g is the weighted device gradient. This allowance differs from the C kernel's; compare backends away from ambiguous thresholds, and test each backend's threshold rule directly rather than promising identical selections there.

**Constraints.**
- No API change.
- Derivatives change only where two dual components share a value to rounding.
- Union-find keeps its roots at the smallest index, so the labels stay deterministic.
- The edge pass already exists.
- The 1.0.2 performance gate (a query costs at most one `fp_jvp`) is re-measured.

**Tests.**
- The counterexample as a fixture in C and Python: the `jvp` equals graph TV's on the grid, and central differences to 10⁻⁸.
- On 200 tied 6 × 6 and 50 tied 16 × 16 images, and tied 3D volumes, compare against graph TV with matching grid edges and penalties at `tol=1e-20`. Require agreement to 10⁻¹⁰ where both solves certify and their jump classifications are unambiguous under their allowances. Investigate every mismatch; explicitly record threshold-ambiguous cases and verify each selection against its documented rule.
- Test jumps below, at and above the allowance, zero penalties, γ = 0 and positive rescalings of the same problem. Use polyhedral feasibility or an analytic certificate for fixtures asserted to be B-Jacobians; finite differences alone do not certify membership at kinks.
- Exercise both PyTorch and JAX device implementations on the counterexample using CPU tensors/arrays, with dtype-appropriate tolerances and tight solver tolerances. Test their threshold rules separately.
- Loose tolerances: `deriv.exact` unchanged.

**Acceptance.** The counterexample is fixed. In exact arithmetic, both operators select the fully joined zero-jump piece on the same positive-penalty grid. Numerically, agreement is required under the matched, unambiguous conditions above; rounding-level merges and inexact solves carry the stated qualifications.

<a id="s4"></a>
### S4. Weighted isotonic regression: the Euclidean (P, W) pair

**Problem.**
- **The operator.** With weights, `Isotonic` is the projection onto the monotone cone C in the metric D\_w = diag(w). Its J = M(MᵀD\_wM)⁻¹MᵀD\_w is not symmetric, and `s.project` returns it (kind WEIGHTED).
- **What the SCD pair is.** The subgradient is v = D\_w(x − z) ∈ N\_C(z). Since MᵀD\_w(I − J) = 0, rge(J, D\_w(I − J)) = L × L^⊥ in the Euclidean sense. That is the SCD subspace of the polyhedral N\_C at (z, v), so the pair is P = the Euclidean projector onto L (block means without the weights) and W = I − P.
- **The trap.** Spec Section 7 states this only "after the change of variables z ↦ W^½ z". A caller who copies `examples/scd_newton_lasso.py` builds P = `s.project`, which is not symmetric (asymmetry 0.29 in the report). GHR Definition 3.6 excludes such a P, and ŷ also needs D\_w.

**Design.** Docs only:
- `docs/api.md` (the SCD paragraph and the isotonic row) and spec Section 7 give the pair explicitly: P\_L v = `s.expand(s.reduce(v) / s.reduce(s.sign))` (MᵀM is the diagonal of region sizes), W = I − P, and v = D\_w(x − z).
- They also note that a weighted fit inside a forward–backward step is a variable-metric step.

**Tests.** In `test_structure.py`, for weighted isotonic: the documented P is symmetric and idempotent, and its range is range J.

**Acceptance.** The docs give the Euclidean (P, W) construction for every supported structure kind under its stated SCD assumptions, including weighted isotonic regression, and identify unsupported cases. No change of variables is left implicit.

<a id="s5"></a>
### S5. What the docs say about SCD subspaces

**Design.** `docs/api.md` ("SCD subspaces"), spec Section 7 ("Relation to SCD subspaces") and the docstring of `examples/scd_newton_lasso.py` say:
1. **One element is enough.** A valid default selection supplies the one subspace required by Algorithm 3.8. State all hypotheses of Theorem 3.9, as in "What holds", including SCD semismoothness* and the approximation step. Do not infer convergence from the structure query alone.
2. **SCD membership is a separate claim.** Under the convex/prox-regular hypotheses of Proposition 3.26, a limiting Jacobian B gives the SCD subspace rge(B, (I − B)/γ). A matrix merely having the right range, symmetry and spectral bounds is insufficient. Do not claim that every numerical default or override is an exact B-Jacobian at the supplied input.
   - At an exact kink, identify which supported selections are limiting pieces. Within a nonzero `kink_tol`, a side override can instead evaluate a boundary convention. Coupled row selections require feasibility; arbitrary combinations of flagged coordinates do not establish membership.
   - On graph TV and anisotropic TV, side +1 on a flagged edge may select no piece (math audit C5, S3).
   - Group ℓ2 forced live at its positive threshold has no descriptor; below it the derivative is a convention evaluated at the radial boundary (S1). A zero-norm input with positive threshold cannot supply that direction and rejects a live override.
   - `ctx.kinks` and `branch=` explore the supported choices; they are not a complete enumeration of SCD subspaces. Operators without overrides, dual-dependent masks and unsupported descriptors limit what is reachable.
   - Preserve spec Section 7's exclusions for discontinuous nonconvex prox maps. The SCD construction requires γ > 0 even where the prox API accepts γ = 0.
3. **Numerical solves and selections.** The descriptor describes the selected mathematical piece, with products approximated numerically as stated above. Before a Newton step, inspect `ctx.deriv.exact` and refine inexact solves where supported. Neither that flag nor refinement proves B-Jacobian membership, resolves every rounding-level active-set ambiguity, or establishes the outer algorithm's approximation-step condition.
4. **CURVED operators.** W's block (J\_L⁻¹ − I)/γ needs J\_L⁻¹, by CG on `jvp`, which nests CG for isotropic TV. The curvature export is a 1.x candidate.
5. **SCAD and MCP** (prox-regular). Slopes above 1 make J\_L⁻¹ − I negative, so W is indefinite. The subspace is still an SCD subspace (Proposition 3.26 within the step conditions).

**Acceptance.** Each statement cites its source, and `docs/api.md` agrees with the spec.

<a id="s6"></a>
### S6. Tests at exact kinks and on boundaries

**Problem.** The 1.0.2 tests draw random sides at kinks, but random inputs never land on a group's kink or on a TV1DL2 boundary. They also check range J = L but not the spectrum of J. S1 and S2 passed for both reasons.

**Design.**
- **`tests/python/test_structure.py`.** Exact-kink fixtures for every operator with overrides: the elementwise maps, group ℓ2, the simplex, the ℓ1 ball, the hyperplane with a box, 1D TV, anisotropic TV, and graph and tree TV.
  - x is built exactly on the kink, with `kink_tol` 0, and within `kink_tol` on each side.
  - On supported sides with successful structure queries: guarantees 1 and 2 for the selected piece, with numerical rank checks only when its positive singular values are resolved. Unsupported S1 queries must instead fail atomically. Reject the zero-norm positive-threshold live override as specified in S1.
  - For the convex operators: J symmetric with spectrum in [−10⁻¹², 1 + 10⁻¹²]. WEIGHTED is checked after the change of metric, and SCAD and MCP against their slope bounds.
  - After `prox` and `prox_conj`. For the latter, J\_inner is evaluated at y/σ with step 1/σ and J\_conj = I − J\_inner. Structure describes range J\_inner, not range J\_conj. Check the Moreau relation and conjugate parameter chain rule explicitly; do not reuse primal parameter expectations unchanged.
  - Test B-Jacobian membership separately from range and spectrum. For feasible exact-kink pieces use analytic limits or a polyhedral certificate; for tolerance-based conventions and invalid graph/image overrides test the documented behavior without asserting SCD membership.
- **`derivative_cases` in `tests/c/test_core.cpp`.** The group ℓ2 kink and the TV1DL2 boundary.
- **`fuzz/fuzz_ops.cpp`.** For rows of up to 8 entries, one input in 1000 forms J densely and checks symmetry and the convex spectral bounds with dtype-appropriate tolerances. Weighted isotonic uses D\_w^½ J D\_w^−½, and SCAD/MCP use their documented slope bounds rather than [0, 1]. Exclude rejected selections. The seeds include exact kinks (‖x\_g‖ = τ, entries at thresholds), the TV1DL2 boundary and the scale-degenerate cases in S2.
- **The report's counterexamples** (S1–S3) as assertions in the suite, as fixtures.

**Acceptance.** Every finding of this review fails on 1.0.2 in the new tests and passes on 1.0.3.

<a id="r1"></a>
### R1. The 1.0.3 release

- The version in `project()`, `FASTPROX_VERSION_SUFFIX` and the header macros, and the CHANGELOG.
- Full suites, the sanitizer build, and an hour of the fuzz driver with the new seeds.
- The `abi` job: no change.
- A TestPyPI dry run, then the tag `v1.0.3`.
- 1.0.3 adds no C function and changes no signature. The calls that newly fail are unsupported structure queries and zero-norm positive-threshold live overrides (S1). Derivatives change in S1 (live overrides below the threshold), S2 (near the boundary and at extreme scales, and in the last bits elsewhere) and S3 (rounding-level joins, observed on tied images). Release notes state the distinction between an exact piece, its numerical implementation and SCD membership.

## Decisions (2026-10-28)

1. **S1: the kink value and `FP_EINVAL`,** rather than real coefficients per region. Coefficients would describe span(u), but they add a query to the ABI and change what a region means. They become a 1.x candidate, together with overrides for the second-order cone and group SCAD and MCP, whose live sides have the same shape.
2. **S2: shared root and dual setup with a bordered derivative solve.** Derive n and μ from scaled solver variables, not rounded output differences. Finite positive outside roots retain their full-range mathematical derivative; the singular outside limit is only an analytical reference. Root, direction and linear-solve errors are validated independently. A threshold classifying rounding-level s as inside would still leave errors just above the threshold.
3. **S3: rounding-level joins with a qualified guarantee.** Apply the positive-penalty join rule at every solver tolerance. Exact zero-jump pieces have the B-Jacobian justification; rounding-level merges are numerical conventions. Backend comparisons account for differing allowances, and `deriv.exact` is not an SCD membership certificate.
4. **S4: docs only.** The Euclidean projector is one line from `reduce` and `expand`, so no new Python helper.
5. **Target.** S1–S6 gate 1.0.3, a patch release.
