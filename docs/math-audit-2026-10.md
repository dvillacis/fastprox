# Mathematical audit of 2026-10-25: specs for 1.0.1

Two audits of 1.0.0 examined its mathematics rather than its robustness:

- [the mathematical audit](mathematical-audit-2026-10-25.md) (M1–M7, D1–D2): numerical counterexamples;
- [the theoretical audit](theoretical-audit-2026-10-25.md) (T1–T9): what each guarantee needs.

Every finding still reproduced on master at `8daf8fa` (2026-10-25, with
`reports/mathematical_audit_2026_10_25.py` and
`reports/theoretical_audit_2026_10_25.py`).

The claims were then checked against the literature, and the sources confirm
the audits' mathematics. Among them:

- Bauschke and Combettes, Prop. 12.28 and Thm 6.30;
- Rockafellar and Wets, 12.19 and 13.53;
- Clarke, Def. 2.6.1 and Thm 2.6.6;
- Scholtes, Cor. 4.1.1;
- Bolte and Pauwels, Lemma 5;
- Beck, *First-Order Methods*, Thms 7.25–7.29;
- Böhm and Wright (JOTA 2021) for the SCAD and MCP constants;
- Wen et al. (IEEE Access 2018) for the ℓq threshold;
- De los Reyes and Meyer, and Villacís's thesis (Remark 4.1, Thms 4.6–4.7), for the TV derivatives.

Five statements in the audit texts need correcting (item A1). No finding was
contradicted.

The defects fall into two kinds:

- **Six wrong results** (M1–M6). M1 and M2 occur at ordinary inputs. M3–M6 need extreme parameters.
- **Overstated guarantees** in the spec and the API docs (M7, C1–C6).

Each item gives the problem, the change, its constraints under the project's
rules, its tests and its acceptance criterion, as in
[the audit of 2026-10-22](audit-2026-10.md). Every item gates 1.0.1, a patch
release. The release adds no C function and changes no signature. The only
calls that newly fail are those that returned a wrong result (M2, M5).

## Summary

| Item | Kind | Effort |
| --- | --- | --- |
| M1. Spectral near ties across a breakpoint | correctness | M |
| M2. Weighted isotonic regression has no Euclidean conjugate | correctness, API | S |
| M3. ℓ0 threshold overflow | correctness, extreme range | S |
| M4. Log barrier leaves its domain | correctness, extreme range | S |
| M5. 1D TV with ℓ2 drops edges whose squared weight underflows | correctness, extreme range | S |
| M6. SCAD and MCP overflow | correctness, extreme range | M |
| M7. Certificates and output rounding | correctness, docs | S |
| C1. Three classes of prox maps, and what a derivative is in each | docs | M |
| C2. Nonconvex penalties: boundedness, existence and conjugates | docs | S |
| C3. Composed and bilevel gradients | docs | S |
| C4. A small gap does not certify a derivative | docs | S |
| C5. Graph TV kinks describe the saved flow | docs | S |
| C6. Outer-method statements | docs | S |
| A1. The audit texts: corrections and placement | docs | S |
| R1. Regression tests and the 1.0.1 release | process | S |

---

<a id="m1"></a>
### M1. Spectral near ties across a breakpoint

**Problem.**
- **Where.** `spectral_core` in `src/core/spectral.hpp` handles a pair of distinct values within the band `cbrt(eps)·max|s|`. It replaces the divided difference (zᵢ − zⱼ)/(sᵢ − sⱼ) by the mean of the two branch derivatives, ½(λᵢ + λⱼ).
- **Why the mean can be wrong.** The mean approximates the quotient only when the scalar map is smooth between the two values: the trapezoid error is O(δ²f‴).
- **The counterexample.** X = diag(1 + δ, 1 − 2δ) with the nuclear norm (f(s) = (s − 1)₊) has a threshold between its singular values. The exact coefficient is δ/(3δ) = 1/3, and the library returns ½.
  - This was verified by derivation and central differences.
  - The error stays ½ − ⅓ as δ → 0.
  - It occurs in float64 at δ = 1e-7 and in float32 at diag(1.001, 0.998).
  - `ctx.deriv.exact` is True.
- **Why the tests missed it.** `test_chained_near_ties` puts the threshold at the midpoint, where the exact quotient is also ½.

**Design.** Keep the mean only where it agrees with the quotient. For a pair with sᵢ ≠ sⱼ and |sᵢ − sⱼ| ≤ band:
1. **Compute both and bound the quotient's error.**
   - the quotient q = (zᵢ − zⱼ)/(sᵢ − sⱼ);
   - the mean m = ½(λᵢ + λⱼ);
   - the quotient's rounding bound e = κ·u·(|zᵢ| + |zⱼ| + |q|·(|sᵢ| + |sⱼ|)) / |sᵢ − sⱼ|. Here u is the unit roundoff of the input dtype (2⁻²⁴ or 2⁻⁵³). κ = 4 covers the inner prox's own rounding, which is a few ulps for every operator `Spectral` accepts (closed forms and row algorithms).
2. **Choose.** Use m if |q − m| ≤ e, and q otherwise.
   - Either way, the result is within 2e of the true divided difference of the computed spectrum, whether or not f is smooth between the values. If m is used, |q* − m| ≤ |q* − q| + |q − m| ≤ 2e.
   - Where f is smooth, m is the more accurate of the two, and it is still chosen whenever it agrees with q.
3. **The antisymmetric quotient needs no rule.** The SVD's (zᵢ + zⱼ)/(sᵢ + sⱼ) has sᵢ, sⱼ ≥ 0, and zᵢ, zⱼ ≥ 0 for an absolutely symmetric prox, so its sums do not cancel and it is accurate at any gap. (Found while implementing; the first draft gave it the same rule.)
4. **Exact ties are unchanged.** For sᵢ = sⱼ the code keeps the limit computed from the direction d (the existing cluster rule). It is the Jacobian of a piece wherever f is differentiable at the tie. A tie that sits on a breakpoint is a kink, and joins the kink rules (C1).
5. **Exactness and the docs.**
   - `ctx.deriv.exact` stays True: the result is now the divided difference to the stated bound.
   - The comment at the tie rule, and the `Spectral` docstring's claim "exact where singular values repeat or sit on a threshold", state the bound and its 1/|sᵢ − sⱼ| conditioning instead.

**Constraints.**
- **No allocation and no ABI change:** the rule uses two more doubles per pair.
- **Determinism:** unchanged, because rows are independent.
- **Spec:** no departure. Section 5 promises the divided difference.

**Tests** (`tests/python/test_spectral.py`, and a case in `derivative_cases` of `tests/c/test_core.cpp`):
- **Asymmetric crossings:** diag(1 + δ, 1 − 2δ) for δ ∈ {1e-7, 1e-5, 1e-3} in float64, and diag(1.001, 0.998) in float32.
  - JVP and VJP, for `kind="svd"` (Soft) and `kind="eigh"` (Soft on eigenvalues).
  - Each is checked against the exact quotient computed in mpmath from the scalar formula.
- **Smooth near ties:** ElasticNet, and LogBarrier with eigh, at gaps from 1e-12 to 1e-3. These must still match an mpmath divided difference to 1e-8 relative, which shows that the mean path is kept where it is valid.
- **`test_chained_near_ties`:** keep it, and add the asymmetric version.

**Acceptance.**
- The audit's M1 rows give 1/3 to 1e-7 (float64) and 1e-4 (float32).
- No existing spectral test changes beyond its tolerance.

<a id="m2"></a>
### M2. Weighted isotonic regression has no Euclidean conjugate

**Problem.**
- **What the weighted operator is.** With weights, `Isotonic` minimises ∑ wᵢ(zᵢ − xᵢ)². That is the projection onto the monotone cone C in the metric W (documented, `docs/api.md`, "Isotonic regression").
- **What `fp_prox_conj` computes.** Through `conj_homogeneous`, it returns y − P_C^W(y). Moreau's decomposition holds in the metric of the projection (Bauschke–Combettes Thm 6.30, in ⟨·, W·⟩): y = P_C^W(y) + W⁻¹P^{W⁻¹}_{C°}(Wy). It does not hold in the Euclidean one.
- **The counterexample.** For weights (1, 3) and y = (2, 0), the result is (1.5, −0.5). That point is not in the polar cone C° = {(a, −a) : a ≥ 0}. The Euclidean conjugate prox is (1, −1).

**Design.**
1. **Reject the call.** For weights that are not all equal:
   - `fp_prox_conj` returns `FP_EINVAL`, and so do the conjugate's derivatives (`fp_jvp` and `fp_vjp` on a workspace kept from a conjugate call can no longer exist).
   - Python's `prox_conj` raises `ValueError`, with a message that names the metric. The same holds for the `fastprox.torch` and `fastprox.jax` conjugate paths.
   - Equal weights give the Euclidean projection, so they keep their conjugate.
2. **Mechanism.** A virtual `fp_op::conj_supported()`, default true, which `prox_impl` checks before dispatch. Isotonic returns false when its scaled weights are not all equal (checked once in the constructor).
3. **Docs.** `docs/api.md` ("Conjugates" and "Isotonic regression") and the `fp_op_isotonic` comment in `fastprox.h` give the reason. They also give the metric decomposition for callers who want the W-dual: W(y − P_C^W(y)) lies in C°.
4. **Decision.** The spec says every operator has its conjugate prox, so this departure gets a row in the ROADMAP decisions table.

**Constraints.**
- **ABI:** no new function; the change is an `FP_EINVAL` on a call that was wrong.
- **Determinism and allocation:** unchanged.

**Tests.**
- The audit's example raises in Python and returns `FP_EINVAL` in C.
- Equal weights (2, 2, 2) agree with the unweighted conjugate.
- The forward weighted fit is unchanged against the CVXPY reference.
- `CASES` in `tests/python/test_operators.py` drops weighted isotonic from its Moreau-identity checks and asserts the error instead.

**Acceptance.** No call returns y − P_C^W(y) as a Euclidean conjugate.

<a id="m3"></a>
### M3. ℓ0 threshold overflow

**Problem.**
- **Where.** `L0::thr` evaluates √(2γλw) as `sqrt(2.0 * gamma * lambda * w)`.
- **The counterexample.** `L0(1e-308).prox([2.0], 1e308)` overflows `2γ`. The threshold becomes inf, so the call returns 0 where 2 is the global minimiser (γλ ≈ 1, F(0) = 2 > F(2) ≈ 1), and reports gap 0.

**Design.**
1. **The ℓ0 threshold.** thr = √2·√γ·√λ·√w. Each factor is finite and nonzero whenever γ, λ and w are, so thr overflows only when the threshold itself is beyond the range. The value, the tie flag and the derivative all read `thr`, so they stay consistent.
2. **Every scalar threshold and constant.** The same rule applies to every place where a scalar map forms c = γλw or a power of it: ℓq's `thresholds` (β = (2c(1 − q))^{1/(2−q)}, τ), the log penalty's breakpoint, and the derivative terms.
   - Products are grouped so that no intermediate overflows or underflows when the quantity itself is representable.
   - Powers go through logarithms only where the direct product is out of range: β = exp((log 2 + log c + log(1 − q))/(2 − q)), with log c = log γ + log λ + log w. Ordinary scales keep the direct formula and its accuracy.

**Constraints.**
- **Results at ordinary scales may change by an ulp.** That is within the tests' tolerances; the existing bit-identity tests compare thread counts, not releases.
- **Determinism:** unchanged.

**Tests.**
- **The audit's reproducer.**
- **Reciprocal scaling:** prox(x; γ, λ) = prox(x; γ·2^k, λ·2^−k) for k ∈ {±200, ±600, ±1000}, for L0, Lq (q ∈ {0.3, 0.5, 0.8}), LogPenalty, SCAD and MCP (with M6).
  - Both sides are checked against the scalar objective minimised in mpmath, not only against each other.
  - The tie flags must agree too.

**Acceptance.** Every scaling above returns the global minimiser within the existing nonconvex tolerance.

<a id="m4"></a>
### M4. Log barrier leaves its domain

**Problem.**
- **Where.** `LogBarrier::value` returns γλ / (½(r − x)) for x < 0.
- **The counterexample.** For x = −1e308, the difference r − x overflows before the halving. The call returns 0, where the answer 1e-28 is representable. Zero is outside the domain, where the objective is +∞, and the reported gap is 0.
- **Related overflows.** γλ itself can overflow (γ = 1e300, λ = 1e280), and 4γλ inside the root even more easily.

**Design.**
1. **One bounded scale.** With s = √γ·√λ:
   - r = hypot(x, 2s);
   - z = s·(s / (½r − ½x)) for x < 0;
   - z = ½x + ½r for x ≥ 0.
   No intermediate overflows unless z does.
2. **Underflow.** When the true z is below the smallest positive value of the output dtype, return that value (`denorm_min` of T), not 0. The result stays in the domain, and the absolute error is at most one subnormal ulp. The derivative, z/r in x, is evaluated from the same quantities.
3. **Spectral form.** The eigenvalue form (−λ log det) uses the same kernel and needs no change.

**Constraints.** The result can change by an ulp at ordinary scales. Determinism and allocation are unchanged.

**Tests.**
- The audit's reproducer.
- z > 0 for every finite input with γλ > 0, on a grid of x ∈ ±10^{±300} and γλ ∈ 10^{±300}, in both dtypes.
- Agreement with a 700-digit mpmath root to 4 ulps where z is normal.
- Derivatives against mpmath at the same points.

**Acceptance.** No finite call with γλ > 0 returns z ≤ 0.

<a id="m5"></a>
### M5. 1D TV with ℓ2 drops edges whose squared weight underflows

**Problem.**
- **Where.** `make_tv1d_l2` stores aₑ = wₑ². The solver treats aₑ = 0 as a missing edge.
- **The counterexample.** `TV1DL2(1e-180).prox([1, −1], 1e180)` returns the input, where 0 is the solution (γw = 1). The objective there is 2, against 1 at 0, and the reported gap is 0.

**Design.**
1. **Store the weights over a power of two.** Take k as the exponent of max w, as isotonic regression does since `8daf8fa`: w̃ = w·2⁻ᵏ, so max w̃ ∈ [½, 1), and ã = w̃².
   - The norm is 1-homogeneous in w, so the call solves with step γ·2ᵏ.
   - The step check applies to γ·2ᵏ. An overflow there is `FP_EPARAM`, as for any step whose scaled problem overflows.
   - Parameter derivatives in w pick up the factor 2⁻ᵏ.
2. **Reject what scaling cannot fix.** A positive weight below 2⁻⁵¹¹·max w has a square that is subnormal or zero even after scaling. The constructor returns `FP_EPARAM` for it, and Python raises `ValueError` naming the ratio.
   - Zero weights stay allowed: they are free edges.
   - The documented range, weights over 150 orders of magnitude, lies inside this limit (2⁵¹¹ ≈ 10¹⁵³·⁸).
3. **Docs.** `docs/api.md` states the ratio limit and removes "underflowed squares count as zero".

**Constraints.**
- **ABI:** unchanged; `FP_EPARAM` is already a constructor status.
- **Results:** bit-identical for weights whose max is in [½, 1); otherwise within rounding.
- **Determinism:** unchanged.

**Tests.**
- The audit's reproducer.
- prox(x; γ, w) = prox(x; γ/c, c·w) for c = 2^{±k}, k up to 1000, against the CVXPY reference at the unit scale.
- The parameter VJP in w scales as 1/c, checked against finite differences at three scales.
- A ratio below the limit raises; weights over 150 orders still pass the existing test.

**Acceptance.** No accepted weight vector solves a different problem from the stated one.

<a id="m6"></a>
### M6. SCAD and MCP overflow

**Problem.**
- **Where.** The middle pieces `((a − 1)x − c·a·λ)/(a − 1 − c)` (SCAD) and `b(x − cλ)/(b − c)` (MCP) multiply the input by a or b before dividing.
- **The counterexample.** At a = b = 1e308, x = 3 and γ = λ = 1, both return inf, where the answers (2a − 3)/(a − 2) and 2b/(b − 1) round to 2. This was confirmed against Wen et al., Table I(v)–(vi).
- **Same failure in the group forms.** GroupSCAD and GroupMCP delegate to the scalar kernels (`src/ops/radial.cpp`) and fail the same way.
- **Derivatives.** The derivative formulas use related products.

**Design.**
1. **Bounded-ratio forms**, with c = γw and the sign s of x:
   - **MCP:** u = x − cλs and z = u + u·(c/(b − c)). The Jacobian is 1/(1 − c/b).
   - **SCAD:** z = x + (x/a − λs)·(c / (1 − (1 + c)/a)). The Jacobian is 1/(1 − c/(a − 1)).
   - Every ratio is bounded on the accepted range, a > 1 + c and b > c.
2. **Parameter derivatives** (λ, a, b, γ, w) are rewritten the same way, and are checked to the same tolerance at a, b up to 1e300.
3. **Piece boundaries.** The tests |x| ≤ 2cλ, ≤ aλ and ≤ bλ stay as they are. An overflowing product is +inf, which still compares correctly, and a > 2 rules out underflow below λ.

**Constraints.**
- **Results** at ordinary scales may change by an ulp.
- **Determinism and allocation:** unchanged.
- **The weak-convexity checks are kept:** a > 1 + γ·max w, b > γ·max w (Böhm and Wright: ρ = w/(a − 1) and w/b).

**Tests.**
- **The audit's four reproducers.**
- **Wen et al.'s formulas:** agreement in mpmath on a grid of x, γ, λ, and a, b ∈ {2 + γ, 10, 1e10, 1e300}.
- **Finite differences** of every derivative at those parameters, with a relative step.
- **The M3 scaling test** for both penalties and their group forms.

**Acceptance.** Every finite accepted call returns a finite result within 4 ulps of the mpmath value.

<a id="m7"></a>
### M7. Certificates and output rounding

**Problem.**
- **The bound.** `zerr` = √(2·gap)/‖x‖ bounds the error of the kernel's double iterate. That is the strong-convexity bound ‖z − z*‖² ≤ 2·gap, which is sound in real arithmetic.
- **The float32 output.** The output is the iterate rounded to T (for instance `tv_iso.cpp`, the final cast). For x = (0, 1) in float32 at γ = 0.1, TVIso and TVAniso report gap 0 and zerr 0, and the returned array is 2.4e-8 from (0.1, 0.9).
- **The docs overstate it.** The spec's "the certified relative error of z" and the `ztol=` docs claim more than holds.
- **Exact operators** report gap 0 ("up to rounding"). That is a convention, not a measured certificate.

**Design.**
1. **Add the rounding of the output.**
   - **The bound.** zerr = √(2·gap)/‖x‖ + u_T. Here u_T is the unit roundoff of the output dtype: 2⁻²⁴ for float32, 2⁻⁵³ for float64.
   - **Why it holds.** The rounding of a double iterate z to T is at most u_T‖z‖. Every image operator's prox is nonexpansive and maps 0 to 0, so ‖z‖ ≤ ‖x‖. For a conjugate output y − σz the same holds with ‖y‖, because Id − prox is also nonexpansive with fixed point 0.
   - **Where.** `image_zerr` adds the term (`src/core/driver.hpp`). It is one addition per image, with no pass over the output.
2. **Unattainable targets.** A `ztol` below u_T cannot be certified in that dtype.
   - The stopping rule is unchanged: it still tests the gap of the iterate.
   - The final `zerr` now exceeds such a `ztol`, so the `ConvergenceWarning` of feedback U2 fires. Its message names the dtype's floor.
3. **Docs.**
   - `docs/spec.md` ("Optimality certificate"), `docs/api.md` (U1's section), the `ztol`/`zerr` docstrings and the C header's tolerance text state the bound with its u_T term.
   - They also state that the gap is evaluated in floating point, so the bound holds to working precision, not as a directed-rounding enclosure.
   - They state that an exact operator's gap 0 means that no certificate is computed.

**Constraints.**
- **ABI:** unchanged; `fp_workspace_image_info` returns the new value.
- **Determinism:** unchanged.
- **Spec:** the text changes with the code; no departure.

**Tests.**
- **The audit's reproducer:** zerr ≥ the measured error of the returned array, in float32 and float64.
- **Random images:** zerr bounds the distance to a reference solved to 1e-20 in float64, cast to the output dtype.
- **The warning:** `ztol=1e-12` on float32 warns, and on float64 does not.

**Acceptance.** On every test, `zerr` bounds the error of the array actually returned.

---

<a id="c1"></a>
### C1. Three classes of prox maps, and what a derivative is in each

**Problem.**
- **The claim.** `docs/spec.md` ("Derivatives") says "The prox is Lipschitz but in general only piecewise smooth", and that the library returns an element of a generalised Jacobian. `docs/api.md` ("Why these are valid derivatives") justifies every table entry as a B-subdifferential element, hence a Clarke Jacobian element, citing Scholtes.
- **Why it is too broad.** The justification holds only for locally Lipschitz maps:
  - Clarke's Jacobian (Def. 2.6.1) presupposes local Lipschitz continuity (§2.6);
  - Scholtes' piecewise differentiable maps are continuous selections and therefore locally Lipschitz (Cor. 4.1.1).
- **Two kinds of map in the library are not locally Lipschitz** (audits D1, T1, T2):
  - **Discontinuous selections.** ℓ0; ℓq with q < 1, where the jump is from 0 to β = (2c(1 − q))^{1/(2−q)} at τ = β + cqβ^{q−1}; and the log penalty with γλ > ε². At the jump no locally Lipschitz calculus applies, and no tie rule removes the jump.
  - **The critical log penalty.** At γλ = ε² the map is unique and continuous but only Hölder-½ at its threshold: P(ε + h) ~ √(εh). The library returns the zero branch's slope there, which is a one-sided convention.

**Design.** Docs only.
1. **`docs/spec.md`, "Derivatives".** Replace the sentence with three classes.
   - **Lipschitz:**
     - every convex operator, with constant 1, firmly nonexpansive, and a symmetric Jacobian with eigenvalues in [0, 1] where it exists (Bauschke–Combettes Prop. 12.28 and Rem. 14.4; Rockafellar–Wets 13.53);
     - SCAD and MCP under their step conditions, with constant 1/(1 − γρ) (Rockafellar–Wets 12.19): ρ = w/(a − 1) and w/b;
     - the log penalty at γλ < ε², with ρ = λ/ε².
   - **Hölder-½:** the log penalty at γλ = ε².
   - **Discontinuous:** ℓ0, ℓq, and the log penalty at γλ > ε².
2. **What each class returns.** In the first class the returned derivative is the Jacobian of a piece, so it is an element of the B-subdifferential and of Clarke's Jacobian. In the other two, at a jump or at the Hölder point, it is the slope of the selected branch, a convention named as such. Away from those points every class has an ordinary Jacobian.
3. **Weakly convex expansion.** Weakly convex maps can expand: the slope b/(b − γw) is correct, and large near the step limit. This is conditioning, not error.
4. **`docs/api.md`.**
   - "Why these are valid derivatives" gets the same split. The Scholtes citation is restricted to the first class.
   - The derivative table marks the branch-convention entries.
   - The kink section gives the jump points of ℓq and the log penalty.
5. **The ℓ0 and ℓq docstrings** state the jump.

**Constraints.** Text only; it is the spec's own text, so no decision row is needed.

**Tests.** None beyond the docs build (`mkdocs build --strict`).

**Acceptance.** No sentence in the spec or the API docs attributes Clarke or Lipschitz properties to a map outside the first class.

<a id="c2"></a>
### C2. Nonconvex penalties: boundedness, existence and conjugates

**Problem.**
- **Boundedness.** `docs/api.md` ("Nonconvex penalties") calls SCAD, MCP, ℓ0, ℓq and the log penalty bounded. Only SCAD, MCP and ℓ0 are. ℓq and log(1 + |z|/ε) are unbounded and sublinear, and with positive weights they are coercive (audit D2).
- **Existence.** The text infers nonexistence from noncoercivity and from one algorithm's divergence. Neither proves it.
- **Conjugates.** The README says every operator has its conjugate prox, with no caveat. For these penalties the Fenchel conjugate is the indicator of {0}: g finite, nonnegative, g(0) = 0 and g(tv) = o(t) along rays give g* = δ_{0}, so prox_{σg*} = 0 (audit T6).
  - What the library returns is the Moreau residual y − σ prox_{g/σ}(y/σ). The detailed API discloses this.
  - Moreau's identity needs convexity in every source (Bauschke–Combettes Thm 14.3; Rockafellar–Wets 12.19).

**Design.** Docs only.
1. **Boundedness.** The paragraph splits into:
   - **bounded:** SCAD, MCP and ℓ0;
   - **unbounded and sublinear:** ℓq and the log penalty.
   Only the bounded ones can leave an objective noncoercive in the observed-entries example.
2. **Existence.** The MovieLens case is stated as what was observed: the objective kept falling and the iterates grew. It is not called a proof that no minimiser exists, and noncoercivity is called a risk, not a proof.
3. **Conjugates.**
   - The README sentence gains the caveat.
   - `docs/api.md` ("Conjugates") states the theorem, g* = δ_{0}, and names the returned map the Moreau residual.
   - A PDHG step built from it is not a conjugate prox step of the nonconvex penalty, and needs its own analysis.

**Constraints, tests.** Text only. The docs build.

**Acceptance.** The boundedness list is correct, and every mention of a nonconvex conjugate names the residual.

<a id="c3"></a>
### C3. Composed and bilevel gradients

**Problem.**
- **What the docs imply.** They suggest that the returned JVPs and VJPs give valid hypergradients of any enclosing computation.
- **Composition.** At a switching point a chosen generalised Jacobian need not compose (audit T3).
  - The projection onto ℝ²₊ taken along (t, −t) with the loss z₁ − z₂ is the function t. The chain rule with the library's DP(0) = I gives 2.
  - Clarke's chain rule is an inclusion (Thm 2.6.6; Thm 2.3.10 with a map that is not onto).
  - Conservative fields compose exactly (Bolte–Pauwels Lemma 5) but may contain such spurious values on null sets.
- **Implicit differentiation of a lower problem** needs a nonsingularity hypothesis on the fixed-point system I − J(I − γ∇²f). An exact prox derivative does not supply it (audit T7). At f = g = 0 the matrix is zero.

**Design.** Docs only: a section of `docs/api.md`, "What the derivatives guarantee", linked from the spec's "Derivatives". It gives the audit's guarantee table, with these points:
1. **Local derivatives.** Off switching sets the returned derivatives are classical and compose by the chain rule.
2. **On switching sets.** A fixed JVP or VJP convention gives a Clarke subgradient of L∘P for a C¹ loss L applied directly. Clarke 2.6.6 gives equality there, for every element of ∂P. It does not do so after a parametrisation that is not onto, or after a further nonsmooth composition.
3. **Bilevel use.**
   - Implicit differentiation of a lower problem needs the caller's regularity: invertibility of the fixed-point system, and strict complementarity where a TV derivative is used (Villacís's thesis, Thm 4.6; De los Reyes and Meyer, Cor. 2.10).
   - At nonsmooth points the result is a conservative Jacobian, not a Clarke one (Bolte, Le, Pauwels and Silveti-Falls 2021).
4. **The TV system.** It is J = Q[Qᵀ(I + K_AᵀHK_A)Q]⁻¹Qᵀ on the stable manifold, the form the implicit TV derivative solves, and it is stated with its hypotheses.

**Constraints, tests.** Text only.

**Acceptance.** The guarantee table is in the docs, and the spec links it.

<a id="c4"></a>
### C4. A small gap does not certify a derivative

**Problem.**
- **The counterexample.** In two-node TV with x = (1 + δ, −1 − δ), the fused candidate has gap δ², while its derivative misses the true Jacobian I by 1 for every δ (audit T4). No bound ‖J − J̃‖ ≤ C√gap holds uniformly.
- **What identification needs.** Region identification needs a nondegeneracy margin: a strictly interior dual on the flat groups that exceeds the dual error (Bertrand et al., Assumption 4; Borwein–Lewis §9.4).
- **What the docs suggest.** The docs on `ctx.deriv.exact` and `ctx.refine` could be read as saying that refining to a small gap makes the derivative exact.

**Design.** Docs only.
1. The `deriv.exact` and `refine` docs say that exact means the saved regions are those of the refined point.
2. They say that it is not a bound in terms of the gap. Near a switch, refining identifies the regions only once the margin exceeds the error.
3. The empirical constants quoted for TV derivatives are labelled as measurements.

**Constraints, tests.** Text only.

**Acceptance.** No text bounds a derivative's error by the gap.

<a id="c5"></a>
### C5. Graph TV kinks describe the saved flow

**Problem.**
- **The counterexample.** On the triangle with x = (1, 0, −1) and γ = 1, two optimal flows exist: (0, 0, 1), which saturates an edge, and (⅓, ⅓, ⅔), which is strictly interior (audit T5).
  - The prox is smooth near x, with Jacobian 11ᵀ/3.
  - The library returns that correct Jacobian, but `ctx.kinks()` flags an edge.
- **What the docs say.** `docs/api.md` (graph and tree TV kinks) says "a kink depends on which one is used". Nondifferentiability of the primal map cannot depend on the dual.
- **The theory.**
  - Villacís's thesis, Remark 4.1: the critical cone depends only on the solution.
  - De los Reyes and Meyer, Algorithm 1 step 6, search for another slack before declaring a biactive point.

**Design.**
1. **Docs (1.0.1).**
   - The mask of graph TV is "edges biactive in the saved flow". It is a superset of the intrinsic kinks on graphs with cycles.
   - A branch override on such an edge may select no accessible piece. The sentence about one-sided differences is kept only for tree TV, 1D TV and anisotropic TV, where the dual of a region is unique.
2. **Later (1.x candidate, not 1.0.1).** A representation-independent test.
   - For the flagged edges of a region, search the region's circulation space for a flow strictly inside the bounds. This is one max-flow per region with flagged edges.
   - Unflag the edges where one exists.

**Constraints, tests.** The docs only, for 1.0.1. The candidate goes in the ROADMAP's 1.x candidates.

**Acceptance.** The docs no longer call a dual-dependent flag a kink of the prox.

<a id="c6"></a>
### C6. Outer-method statements

**Problem.** `docs/api.md` ("Inexact proxes inside outer solvers" and "Nonconvex penalties") phrases sufficient conditions as necessities in places, and leaves out hypotheses (audit T8):
- **Forward–backward:** constant steps converge for γ ∈ (0, 2/L), with the endpoint excluded and a minimiser assumed (Bauschke–Combettes Cor. 28.9).
- **FISTA:** the standard guarantee uses γ ≤ 1/L (Chambolle and Pock, Algorithm 5 and Thm 4.10). This is a sufficient condition, not a bound beyond which FISTA cannot converge.
- **Nonconvex descent:** (1/(2γ) − L/2)‖x⁺ − x‖² needs γ < 1/L and γ below the penalty's prox-bound (Themelis, Stella and Patrinos, Lemma 2.1). Whole-sequence convergence by PALM/KL also needs bounded iterates.
- **Error summability:** Schmidt, Le Roux and Bach's conditions (∑√εₖ, and ∑k√εₖ for acceleration) are sufficient.
  - A power-law `ztol` schedule meets them only if the prox inputs stay bounded.
  - The direct recipe is √(2Δₖ) ≤ ηₖ with ηₖ summable, that is ztolₖ = ηₖ / max(1, ‖vₖ‖).

**Design.**
1. Rewrite those paragraphs with the stated hypotheses and "sufficient" where it applies.
2. Check the Schmidt, Le Roux and Bach and the Bolte, Sabach and Teboulle statements against the papers before the text lands. Neither paper is in the reviewed library.

**Constraints, tests.** Text only.

**Acceptance.** Each statement names its hypotheses, and each cited one has been read in its source.

<a id="a1"></a>
### A1. The audit texts: corrections and placement

**Problem.** Both audits are untracked, and five statements in them need correcting:
- **T9:** the Ky Fan identity reads w_k‖X‖_* where it should read w_k‖X‖_(k) (the Ky Fan k-norm). The two agree only for k = min(m, n), so the conclusion holds.
- **T3:** the "narrower statement" is stronger than written. For a C¹ outer loss, Clarke 2.6.6 gives equality for every element of ∂P, not only for accessible B-Jacobians.
- **T5:** it omits the scaling. The prox is I − Π onto γ·Kᵀ(weighted box).
- **T7:** "flat constraints locally stable" should read "a dual with ‖q_j‖ < 1 on every flat group exists" (strict complementarity).
- **T8:** the nonconvex descent lemma also needs γ below the prox-bound.

**Design.**
1. Commit both audits under `docs/`, unchanged except for an "Errata (2026-10-25)" section at the end of each with the corrections above.
2. Commit the two scripts and their JSON under `reports/`.
3. Add the audits and this file to the "Project" section of `mkdocs.yml`.

**Acceptance.** The docs build strictly, with the three pages in the navigation.

<a id="r1"></a>
### R1. Regression tests and the 1.0.1 release

**Design.**
1. **Tests.** Every M item's tests land with its fix.
   - The audit scripts stay diagnostic: they print and record, and are not run by pytest.
   - Their counterexamples become ordinary assertions in the suite.
   - Nothing depends on CVXPY or mpmath at test time: the references are stored as fixtures, as the spec's testing section requires, or computed in closed form.
2. **Revalidation.** Before the tag:
   - the full Python and C suites;
   - the sanitizer build;
   - a 24-hour fuzz run (with the extreme-range targets of M3–M6 seeded);
   - the frameworks' conjugate paths for M2.
3. **Release.** As in `CLAUDE.md`: version 1.0.1 in `project()` and in the `fastprox.h` macros, the CHANGELOG section, a TestPyPI dry run, then the tag `v1.0.1`.

**Acceptance.** 1.0.1 is on PyPI, and the CHANGELOG lists each item.

---

## Decisions (2026-10-25)

1. **M2: reject.** The Euclidean conjugate of the monotone cone's indicator ignores the weights entirely, so returning it would silently drop an argument the caller gave. A caller who wants the Euclidean conjugate can call the unweighted operator.
2. **M5: the ratio limit.** Weights are limited to a ratio of 2⁻⁵¹¹, so their squares stay normal after scaling. The alternative, a scaled-sum norm in the solver, would remove the limit but touches every step of the trust-region solve, for weights spanning over 150 orders of magnitude, which no caller is known to use.
3. **M7: a bound.** `zerr` adds the term u_T rather than computing ‖fl(z) − z‖ per image. The bound needs no pass over the output, and costs nothing visible in float64 (2⁻⁵³). In float32 it can exceed the actual rounding: 6.0e-8 against the 2.4e-8 measured in the audit's case, and more for images whose z is small against x.
4. **C5's representation-independent test** goes to the 1.x candidates, not to 1.0.1.
5. **Target.** Every item gates 1.0.1.
