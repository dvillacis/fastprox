# User feedback of 2026-10-23

Feedback from using fastprox to reproduce paper experiments: 14 replications
and about 25 experiments, in an imaging demo, an ML project and a papers
project. The derivatives passed every finite-difference test (52 in all, over
TV, soft thresholding, elastic net, the group penalties, MCP, SCAD and 1D TV).
Implicit differentiation gave the Ochs et al. hypergradient exactly in 12 ms,
where unrolling needed 3 000 iterations and 844 MB to come within 3e-4.
Warm starts, saved contexts and the parameter gradients behaved consistently
throughout.

The rough edges, and the specs below that address them:

| Feedback | Items |
| --- | --- |
| The prox had to be solved to about tol² of the outer solver's tolerance, or FISTA stalled, and nothing warned | U1, U2 |
| `TVAniso.vjp` takes no `tol`, `TVIso.vjp` does | U3 |
| Isotropic TV derivatives are biased at loose gaps | U4 |
| No spectral operators on PyPI; unequal weights refused; spectral MCP diverged | U5, U6, U7 |
| Building from the workspace broke mid-edit | U7 |

The four problems the report traced to its own outer solvers and test design
(GMRES on the adjoint, Lasso FISTA at small λ, unrolled tangents, a trivially
zero MCP test) need no change here.

Each item follows the audit's format (`docs/audit-2026-10.md`): the problem, the
change, its constraints under the project's rules, its tests and its
acceptance criterion. The items are tracked in
[ROADMAP.md](https://github.com/dvillacis/fastprox/blob/master/ROADMAP.md)
under "1.0 — user feedback of 2026-10-23". All of them gate 1.0.0 (decided
2026-10-23). By A11, master ships as 1.0.0, so the C functions added here join
the 1.0 ABI before it freezes, and need no "since" note.

## Summary

| Item | Kind | Target | Effort |
| --- | --- | --- | --- |
| U1. Accuracy in the units of z | docs, API | 1.0.0 | S |
| U2. Unconverged and too-loose calls are silent | usability | 1.0.0 | M |
| U3. One set of derivative keywords on every operator | API | 1.0.0 | S |
| U4. Derivatives of iterative TV at loose gaps | correctness | 1.0.0 | M |
| U5. Weighted nuclear norm: the sorted ℓ1 norm | feature | 1.0.0 | M |
| U6. Spectral MCP divergence: reproduce first | investigation | — | S |
| U7. Releases, so that no one builds from a workspace | process | 1.0.0rc2, 1.0.0 | S |

---

<a id="u1"></a>
### U1. Accuracy in the units of z

**Problem.**
- **What `tol` bounds.** An iterative operator stops each image when its duality gap is at most tol·‖x‖²/2. The prox objective is 1-strongly convex, so ‖z − z*‖² ≤ 2·gap, and the certified relative error of z is ‖z − z*‖/‖x‖ ≤ √tol.
- **The consequence.** An outer solver that needs z to 1e-5 needs tol = 1e-10. This is the "tol²" the report found by trial, and no text states it. The default tol = 1e-6 certifies only 1e-3 relative error of z.
- **What the library reports.** Nothing gives the certified error directly. `IterInfo.gap` is the batch sum, so its per-image meaning has to be reconstructed by hand.

**Design.**
1. **Docs (1.0.0).**
   - Every `tol` docstring and the C header's tolerance text (`fp_op_tv_aniso`, `fp_op_tv_iso`, `fp_op_graph_tv`) state the bound: "the certified relative error of z is √tol: tol = 1e-10 gives ‖z − z*‖ ≤ 1e-5 ‖x‖".
   - A new section of `docs/api.md`, "Inexact proxes inside outer solvers":
     - the bound;
     - the known conditions on prox errors for proximal gradient and its accelerated form, with the gap as the error measure. Schmidt, Le Roux and Bach (NeurIPS 2011) give a rate-preserving condition on the summability of √εₖ, with a factor k for the accelerated method. The exact statements are to be checked against the paper before the section lands;
     - the practical recipe: an inner `ztol` (item 2) a factor of 10 below the outer tolerance on z, or a decreasing schedule, with warm starts from the previous dual (`warm=True`), which make tight inner solves cheap near convergence.
2. **`ztol=`.**
   - **Where:** every iterative `prox`, `prox_conj` and `solve` in Python (TVAniso, TVIso, GraphTV, the `fastprox.torch` and `fastprox.jax` image functions, and their generic `prox` paths). It is the relative error of z to certify, and is mutually exclusive with `tol` (`ValueError` if both are given).
   - **Implementation:** it passes tol = ztol², so the per-image stopping rule is unchanged. A ztol whose square underflows (below about 1e-154) raises `ValueError` rather than silently meaning "run to max_iter".
   - **No C change.** fp_options.tol keeps its meaning, and its comment gains the √tol sentence.
3. **`IterInfo.zerr`.**
   - **What:** maxᵢ √(2·gapᵢ)/‖xᵢ‖ over the images, the certified relative error of the worst image, in the units `ztol` takes.
   - **Source:** per-image gaps come from a new C call, `fp_workspace_image_info(ws, gaps, iters)`, which writes the per-image gaps and iterations of the last call (FP_F64 and FP_I32 arrays of the batch shape). The workspace already holds them in `image_gaps` and `image_iters`. The function joins the 1.0 ABI.
   - **Values:** NaN for a NaN image (A8), and 0 where ‖xᵢ‖ = 0.

**Constraints.**
- **Determinism:** unchanged.
- **Allocation:** `fp_workspace_image_info` copies existing buffers.
- **ABI:** one new function, in the 1.0 baseline.
- **Spec:** no departure.

**Tests.**
- `ztol=1e-5` and `tol=1e-10` give bitwise-equal results.
- `IterInfo.zerr` bounds the distance to a reference solved to tol 1e-20, for TVIso, TVAniso and GraphTV on batches.
- The per-image gaps sum to `IterInfo.gap`.
- `ztol` together with `tol`, and `ztol` = 1e-200, raise `ValueError`.

**As implemented.** As designed, with these changes:
- **`fp_workspace_image_info(ws, gaps, iters, zerrs, count)`.** It takes plain contiguous buffers and a count, since there is no FP_I32 dtype. It also returns each image's certified error. The driver computes that error from the kernel's own input, at the power-of-two scale the image was solved at (`ImageCall.scale_exp`, `scaled_gap`), so it holds where the gap overflows or underflows, and in place. The norm uses eight partial sums by position: a fixed order, with no latency chain.
  - **Speed.** The extra pass costs about 1% on graph and tree TV at 512², and is within noise elsewhere.
  - **Why in C.** In Python the norms took 2–6 µs per call, and `solve` on a 16² image went from 12.9 to 18.3 µs.
- **Graph TV's `tol`** bounds each node: ‖z − z*‖∞ ≤ tol·(max x − min x). So `ztol` maps to tol = ztol·minᵢ ‖xᵢ‖/(√n·(max xᵢ − min xᵢ)), which gives the same guarantee. The tests check it with offsets and scales.
- **When the work runs.** `prox` without `keep` or `ctx` now skips building `IterInfo`, the dual copy included.
- **Recipe.** The recipe in docs/api.md was measured. On a 96² inpainting problem with isotropic TV, 200 FISTA iterations took 10–14× fewer inner iterations warm than cold, at the same `ztol` (1e-3 to 1e-5).
- **The paper's statements.** They were checked against arXiv:1109.2415v2: Propositions 1 and 2, and the error measure (4). Since √(2ε/L) = √(2·gap), their conditions read directly in `ztol`'s units.

**Tests.**
- **C:** `test_image_info` (`tests/c/test_core.cpp`) covers:
  - the gaps' sum against `fp_info.gap`;
  - NaN images;
  - in-place calls;
  - scales 2^±600;
  - the errors.
- **Python:** `tests/python/test_ztol.py` covers bitwise equality with `tol=ztol²`, zerr against references carrying their own certificate, graph and tree TV, the special cases, the errors, and PyTorch and JAX (float64, eager and jit).

**Acceptance.** The tests pass, and the docs section is in `docs/api.md`.

---

<a id="u2"></a>
### U2. Unconverged and too-loose calls are silent

**Problem.**
- **Non-convergence is silent.** `prox()` of an image operator returns z alone, and the binding drops `FP_ENOTCONV` (`bindings/python/src/module.cpp:148`). A solve that stops at `max_iter` therefore returns an unconverged z with no sign, unless the caller uses `solve()` and reads `converged`.
- **A loose tolerance is silent too.** Even when every solve converges, nothing relates its certified error to the accuracy the outer solver needs. The report asked for "a check or warning when the prox's gap is too loose for the outer solver's tolerance".

**Design.**
1. **`fastprox.ConvergenceWarning`** (a `UserWarning` subclass), raised with `warnings.warn` when an iterative call returns `FP_ENOTCONV`:
   - **Where:** `prox`, `prox_conj`, `solve`, the framework image functions and the generic `prox` paths.
   - **Message:** the worst image's certified error, `zerr` from U1, and the `max_iter` reached.
   - **Silencing:** the standard warning filters. `solve(..., warn=False)` silences one call, since `solve` already returns `converged`.
   - **JAX:** the generic `fastprox.jax.prox` warns from the `pure_callback` that already runs the kernel on the host. The JAX-native image functions (`fastprox.jax.tv_iso`, `tv_aniso`) return `converged` as a traced value, so they warn through `jax.debug.callback` when it is False, which also works under `jit`.
2. **An accuracy context:**

   ```python
   with fp.inner_accuracy(1e-5, factor=0.1):   # the outer solver's tolerance on z
       z = op.prox(x, gamma)                    # ztol defaults to 1e-6 here
   ```

   Inside the block, every iterative call without its own `tol` or `ztol` uses ztol = factor·target. A call whose explicit `tol` or `ztol` certifies a looser error than that raises `fastprox.AccuracyWarning` with both numbers, as does a call that stops before it. The state is a `contextvars.ContextVar`, so threads and asyncio tasks keep their own, and nested blocks restore on exit. The outer code needs no other change, which is what the report asked for.
3. **The default tolerance stays 1e-6.** Changing it would change every caller's results. The docs (U1) say what it certifies.

**Constraints.** Python only: no C, determinism, allocation or ABI change. Spec: no departure.

**Tests.**
- **`ConvergenceWarning`:** `max_iter=2` on a 64² TVIso warns, through `prox`, `solve`, `fastprox.torch.tv_iso` and `fastprox.jax.tv_iso` (also under `jit`). A converged call does not warn. `warn=False` silences.
- **The context:**
  - the default ztol inside it is used (bitwise equal to an explicit `ztol`);
  - a looser explicit `tol` warns;
  - nested contexts restore;
  - two threads with different contexts do not interfere.

**As implemented.** As designed, with these details:
- **One warning per call.** `ConvergenceWarning` subclasses `AccuracyWarning`, so one filter covers both. A call that stops early inside a block raises a single `ConvergenceWarning`, which names the block's ztol.
- **`warn=False`** works on `solve`, the one-shot functions and the framework image functions. It silences both warnings for that call.
- **Graph TV.** Inside a block it maps the block's ztol through its per-node bound (U1). Tree TV is exact and never warns.
- **The PyTorch and JAX generic paths** capture the context when the call is set up (`Spec.accuracy`), because JAX's `pure_callback` may run on another thread, where the context variable is unset. The JAX image functions read the context when traced. The callback reduces `converged` on the host: `jnp.all` of the flag has no differentiation rule under `grad`.
- **Cost.** The context variable is read once per call, and `zerr` is fetched only when a call did not converge. Small calls are unchanged within noise.
- **What the warning found.** Three examples had stopped isotropic TV at the default 1000 iterations short of their tolerance; they now pass `max_iter=10**5`. Tests that stop early on purpose now assert the warning, and reference solves pass `warn=False`.

**Tests.** `tests/python/test_warnings.py` (9 tests) covers:
- the warning and its message (zerr, max_iter, the caller's line) on every path, JAX eager and jit included;
- silence for converged calls and `warn=False`;
- the context's default, bitwise equal to an explicit `ztol`, for TV and graph TV;
- looser `tol` and `ztol` warning, tighter ones silent;
- tree TV;
- nesting and restoring on exceptions;
- argument errors;
- two threads with different blocks.

**Acceptance.** The tests pass, and the docs show the context in the "Inexact proxes" section.

---

<a id="u3"></a>
### U3. One set of derivative keywords on every operator

**Problem.**
- **Classes:** `TVIso.jvp` and `TVIso.vjp` take `tol` and `max_iter`, the tolerance of their conjugate gradient solve. Every other operator's `jvp` and `vjp`, `TVAniso` included, rejects them. Generic code over operators must special-case the class.
- **Why the difference:** anisotropic TV, graph TV and tree TV differentiate in closed form from their saved regions, so the keywords have nothing to control there.
- **Framework functions:** `fastprox.torch.tv_aniso` and `fastprox.jax.tv_aniso` already take `deriv_tol` and `deriv_max_iter`, but the generic `fastprox.torch.prox(op, ...)` and `fastprox.jax.prox` paths do not pass them for `TVIso`.

**Design.**
- **One signature:** `Operator.jvp(ctx, dx=None, dgamma=0.0, dparams=None, *, out=None, tol=None, max_iter=10000, branch=None, kink_tol=0.0)`, and the same keywords on `vjp`, on every operator.
- **Closed-form operators** validate the keywords (`tol` > 0 or None, `max_iter` ≥ 1) and ignore them. They set `ctx.deriv` to `(0.0, 0)`, so `ctx.deriv` is always defined after a derivative.
- **Spectral `jvp` and `vjp`** take the same keywords and ignore them, since their inner operators are closed-form.
- **Generic framework paths:** `fastprox.torch.prox` and `fastprox.jax.prox` take `deriv_tol` and `deriv_max_iter`, used for `TVIso` and ignored elsewhere, matching the image functions.
- **Docs:** one sentence in the `Operator` docstring: "`tol` and `max_iter` control iterative derivative solves (isotropic TV) and are accepted and ignored by every other operator".

**Constraints.** Python only, and additive. Spec: no departure.

**Tests.**
- For every operator in `CASES`, `jvp` and `vjp` accept `tol=1e-8, max_iter=50` and give results bitwise equal to the call without them, except `TVIso`, whose solve they change.
- `ctx.deriv` is set after every derivative.
- `fastprox.torch.prox(fp.TVIso(2), ..., deriv_tol=1e-4)` reaches `ctx.deriv`'s residual ≤ 1e-4.

**As implemented.** As designed:
- **One code path.** The iterative solve's setup moved into `Operator.jvp` and `vjp`, behind a class flag (`_iterative_deriv`, set by `TVIso`), and TVIso's overrides went away.
- **Validation.** `tol` must be > 0 or None. `max_iter` must be an integer ≥ 1; booleans are refused.
- **Cost.** A fast path skips the checks when both keep their defaults, so a tiny call costs 0.03–0.05 µs more than before (measured at 0.12 µs without it).
- **`ctx.deriv`.** `SpectralContext` gained a `deriv` slot, so every context reports `ctx.deriv`.
- **Frameworks.** The PyTorch and JAX generic paths carry `deriv_tol` and `deriv_max_iter` in `Spec.deriv`, to every `jvp` and `vjp` they make, the per-row step gradients included.

**Tests.** `tests/python/test_deriv_keywords.py` (61 tests; 60 fail on the previous bindings) covers:
- bitwise equality with and without the keywords for every operator in `CASES`, anisotropic, graph and tree TV, and the nuclear norm;
- TVIso's residual and iterations following them;
- the errors;
- the PyTorch and JAX generic paths reaching `deriv_tol`, checked through `ctx.deriv`.

**Acceptance.** The tests pass.

---

<a id="u4"></a>
### U4. Derivatives of iterative TV at loose gaps

**Problem.**
- **Isotropic TV.** Its derivative caps the curvature γ/‖gₚ‖ at γ/μ, with μ = 0.1·w_max·√(2·gap/N): a Huber smoothing at the certified error, which vanishes with the gap. On 2 × 128² images the derivative's error against the exact one is about 4% at a gap tolerance of 1e-6 and 0.5–0.7% at 1e-8 (CHANGELOG, derivatives of isotropic and anisotropic TV).
- **Anisotropic TV.** Its C derivative is exact once the polish has found the regions, from about 1e-10. Before that its error is 2–5% at 1e-8.
- **Nothing tells the caller.** A loose prox tolerance therefore gives a biased hypergradient silently. The report avoided it only by using tight gaps.

**Design.**
1. **Report.**
   - `ctx.deriv` becomes a named tuple `DerivInfo(residual, iters, exact, smoothing)`.
   - **`exact`:** for anisotropic TV, whether the derivative came from regions the polish certified. A new flag is saved in the structure, one byte, next to `ncomp`. Graph and tree TV are always exact. Isotropic TV is exact only at gap 0.
   - **`smoothing`:** μ for isotropic TV, scaled back to the input's units, and 0 elsewhere.
   - **In C:** `fp_workspace_deriv_info` keeps its signature. A new `fp_workspace_deriv_exact(ws, int *exact, double *smoothing)` reports the two fields.
2. **Refine on demand.**
   - **New call:** `ctx.refine(ztol)` in Python and `fp_workspace_refine(op, ws, tol, max_iter)` in C. It re-solves the kept call from its saved x and its saved dual, as a warm start, to the tighter tolerance, then rewrites the saved z, structure, dual and `fp_info`. The saved x and the steps are already in the workspace, so no input is needed. It returns `FP_ENOTCONV` as a prox would.
   - **Keyword shortcut:** `jvp(..., refine=ztol)` and `vjp(..., refine=ztol)` refine first.
   - **The point changes.** The derivative is then that at the refined point. The z the caller already holds is not changed; `ctx.z` is. The docs state this.
   - **Cost:** that of a warm-started solve from the saved dual to the new tolerance. It is measured as part of the item (see Acceptance).
3. **Warning.** A derivative of isotropic TV with `exact = False`, at a certified error above 1e-4, raises `AccuracyWarning`, once per context. The message gives the error and suggests `refine=`. Inside `fp.inner_accuracy` (U2), the threshold is the context's ztol.

**Constraints.**
- **Determinism:** a refine is an ordinary warm-started solve, bit-identical for any thread count.
- **Allocation:** none with a workspace; the refine uses the workspace's scratch.
- **ABI:** `fp_workspace_refine` and `fp_workspace_deriv_exact`, in the 1.0 baseline.
- **Spec:** an addition to Section 5's derivative contract, recorded in the decisions table when it lands.

**Tests.**
- **Refine:** on 2 × 128² isotropic TV, refining a 1e-6 solve to ztol 1e-5 brings the derivative's error against a 1e-14 reference below 0.7%. It is also bitwise equal to a fresh warm-started solve at that tolerance followed by the same derivative.
- **The `exact` flag:** it is True on anisotropic TV once polished and False at 1e-4 on an image where the polish fails. Graph and tree TV are always exact.
- **The warning** is raised and suppressed as designed.
- **C:** `fp_workspace_refine` keeps the allocation counter at zero.

**As implemented.** The design, with these changes:
- **The flags.** Each kernel reports `exact` and the smoothing μ in `ImageCall`. The driver keeps them per image in the workspace (`image_exact`, `image_smooth`, sized in `build_workspace`), as it keeps the gaps. They are not a byte in anisotropic TV's saved structure, which leaves its layout alone and lets graph TV report too.
  - **Anisotropic TV:** exact when polished.
  - **Isotropic TV:** exact when μ = 0, at gap 0 or with fewer than two active axes.
  - **Graph TV:** exact when no part wider than the rounding floor was stopped by `tol`, and the solve converged.
  - **Tree TV:** always exact.
- **`fp_workspace_refine(op, ws, tol, max_iter, out, info)`** takes an optional `out`, which receives the refined result as the prox call wrote it, and an `fp_info`. Without `out`, the caller could not get the z that the derivatives now read. `ctx.refine` returns that z. Graph TV refines to `tol=0`, since its ztol mapping needs x, which the context does not hold.
- **The warning** covers every operator whose derivative is not exact, not isotropic TV alone. The anisotropic and graph TV derivatives are biased too at a loose tolerance (2–5% for anisotropic TV at 1e-8). The warning goes once per point: a new prox call on the context resets it.
- **Examples.** Two examples differentiated isotropic TV after solves certified to 1e-3 and 3e-4, which now warn. They solve to `tol=1e-8`, with unchanged results.
- **Acceptance: only half met.** On the CHANGELOG's case (2 × 128², blocks plus noise, γ = 0.2, from a 1e-6 solve):

  | | Fresh solve to the tight tolerance | Refine | Derivative error |
  | --- | --- | --- | --- |
  | Isotropic TV, to 1e-10 (ztol 1e-5) | 32 432 iterations, 4.21 s | 31 036 iterations, 4.23 s | 2.9% before, 1.2e-4 after, as the fresh solve's |
  | Anisotropic TV, to 1e-12 | 66 iterations, 0.031 s | 34 iterations, 0.017 s | exact (error 0) |

  - **Anisotropic TV** meets "under half the cost".
  - **Isotropic TV** does not. Its FISTA's gap falls like 1/k², so going from 1e-6 to 1e-10 takes about 100 times the iterations whatever the start, and the warm start saves only the loose solve's own iterations. A refine is therefore a convenience (no input needed, the derivative at a tighter point) and not a shortcut. The docs say so: solve to the tolerance the derivative needs in the first place, and refine when the point must be tighter than the one already solved.

**Tests.**
- **C:** `test_refine` (`tests/c/test_core.cpp`) checks, for isotropic, anisotropic, graph and tree TV, prox and conjugate:
  - the refine is bitwise equal to a fresh solve seeded with the kept dual: its output, `fp_info`, per-image gaps, the vjp and the exactness;
  - the same bits for 1 and 4 threads;
  - no allocation;
  - the errors, and `FP_ENOTCONV`.
- **Python:** `tests/python/test_refine.py` checks:
  - the isotropic derivative within 0.7% of the reference after `refine=1e-5`, bitwise equal to the fresh seeded solve;
  - the refined z, prox and conjugate, float32 included;
  - the flags per operator;
  - the warning: once per point, reset by a new prox call, silent when exact or tight, inside `inner_accuracy`;
  - the errors.

**Acceptance.** The tests pass. On the CHANGELOG's 2 × 128² case, `vjp(..., refine=1e-5)` from a 1e-6 solve gives the 1e-10 solve's derivative error at under half the cost of solving to 1e-10 from scratch.

---

<a id="u5"></a>
### U5. Weighted nuclear norm: the sorted ℓ1 norm

**Problem.**
- **The refusal is correct.** `fp.Spectral(fp.Soft(lam, w))` with unequal `w` raises "weights must be equal". A spectral function needs an absolutely symmetric h, and Σ wᵢ|sᵢ| with fixed positions is not one.
- **Which weighting is spectral.** The weighted nuclear norm Σ wᵢ σᵢ(X) that papers use applies the weights in sorted order, σ₁ ≥ σ₂ ≥ …. With nonincreasing w ≥ 0 it is a convex spectral function of h(s) = Σ wᵢ |s|₍ᵢ₎, the sorted ℓ1 norm (OWL, SLOPE), and fastprox has no such operator.

**Design.**
- **A new row operator, `fp_op_sorted_l1(weights)`, and `fp.SortedL1(w)`:**
  - **h:** h(z) = Σᵢ wᵢ |z|₍ᵢ₎, with w nonincreasing, ≥ 0, of the row's length, or length 1, where it reduces to Soft. Other weights give FP_EPARAM.
  - **Prox:** sort |x| in decreasing order, subtract γw, project onto the nonincreasing nonnegative cone by a stack of pooled blocks, then restore the signs and the order. This is the algorithm of Bogdan, van den Berg, Sabatti, Su and Candès (Ann. Appl. Stat. 9(3), 2015, Algorithm 3) and of Zeng and Figueiredo (2014). It is implemented from the papers, as the clean-room rule requires.
  - **Cost:** O(n log n) for the sort plus O(n) for the pooling. The sort is the row's scratch; ties are broken by index, so results are deterministic.
  - **Derivatives:** z is piecewise linear. On each pooled block B with a positive value, every entry's magnitude is the block mean of |x| − γw over B. Hence:
    - the Jacobian averages within blocks, carrying the signs and the permutation, and is 0 on the zero block;
    - ∂|zᵢ|/∂γ is minus the mean of w over i's block;
    - ∂|zᵢ|/∂wⱼ is −γ/|B| for each weight position j that i's block covers.
  - **Kinks:** a block boundary where two pooled values tie, and the boundary of the zero block. They are handled by the existing kink machinery, as for isotonic regression, whose pooling this shares.
- **Spectral:**
  - `fp_op_spectral_check` accepts `sorted_l1` for `svd` and for `eigh`. For `eigh` this is the sorted ℓ1 of the eigenvalues: still absolutely symmetric, so permutation invariant.
  - `fp.WeightedNuclearNorm(w)` is shorthand for `Spectral(SortedL1(w))`.
- **Error message:** the Spectral refusal for unequal weights now points to `SortedL1` and `WeightedNuclearNorm`.
- **Not included:** nondecreasing weights (WNNM, Gu et al. 2014) give a nonconvex penalty, whose prox is closed form only under ordering conditions. They are left for a later item if asked for.

**Constraints.**
- **Determinism:** rows are independent.
- **Allocation:** the sort buffer comes from `scratch_bytes`.
- **ABI:** a new constructor, in the 1.0 baseline.
- **Spec:** a new operator, so `docs/spec.md` gains its row.
- **CLAUDE.md "Adding an operator":** the kernel, the factory, the C and Python constructors, `CASES` and `derivative_cases`.

**Tests.**
- **Reference:** the prox against a reference solver (cvxpy on small rows, in `tests/python/reference.py`) and against Soft at equal weights, bitwise.
- **The CASES suite:** finite differences, adjointness, nonexpansiveness, threads and strides, through `CASES`.
- **Spectral:** `WeightedNuclearNorm` against a direct minimisation on 6 × 5 matrices, and its derivatives at tied singular values.
- **Errors:** increasing weights give FP_EPARAM.

**As implemented.** As designed, in `src/ops/sorted_l1.cpp`, with these details:
- **Sort keys.** The sort is of (|x|, index) keys, contiguous in the row's scratch, ties by index. Sorting indices compared through x cost 2.28× the acceptance baseline; the keys cost 1.90×, measured on 64 rows of 10⁴ against `Isotonic` plus `np.sort`.
- **Pooling.** It works on block means, so no sum can overflow. Blocks merge only on a strict violation, which makes equal weights reproduce soft thresholding bit for bit: the prox, the JVP and the input gradient. The step and weight gradients sum block by block in sorted order, so they agree with Soft's to rounding.
- **Derivatives.** They sort the saved x again in their scratch rather than store the permutation: the saved code is one byte per entry, marking block starts.
- **Kinks.** None are reported, as for isotonic regression, whose "kink machinery" the spec cited but which has none. Decisions row.
- **Spectral.** `fp_op_spectral_check` accepts the operator for SVD and EIGH. `WeightedNuclearNorm(w, kind="svd")` is `Spectral(SortedL1(w))`, and the refusal of unequal weights points to both.

**Tests.**
- The `CASES` suite (`sorted_l1`, `sorted_l1_scalar`), with the reference in `tests/python/reference.py`: sort, the reference PAVA, clip.
- `derivative_cases` in C, and the fuzzer (case 35).
- `tests/python/test_sorted_l1.py`:
  - cvxpy (the norm as Σₖ (wₖ − wₖ₊₁) S_k(|z|)) on rows where blocks pool;
  - Soft bit for bit at equal weights;
  - finite differences where blocks pool;
  - the weighted nuclear norm's optimality against 200 random perturbations, on 6 × 5 matrices;
  - its JVP and VJP at singular values tied three times;
  - EIGH;
  - the errors.

**Acceptance.** The tests pass. The cost is within 2× of `Isotonic` plus a sort on rows of 10⁴.

---

<a id="u6"></a>
### U6. Spectral MCP divergence: reproduce first

**Problem.** The report dropped spectral MCP because it "diverged", with no reproducer. A check on 2026-10-23 found the operator correct where tested:
- **Prox:** on a random 8 × 6 matrix (λ = 0.5, b = 3, γ = 1), `Spectral(MCP)` matches U·firm(s)·Vᵀ exactly.
- **Derivatives:** the jvp agrees with central differences to 7e-10, and adjointness holds to 2e-15.
- **An outer loop:** proximal gradient for 50% matrix completion of a rank-4 30 × 25 matrix stays finite and fits the observed entries to 1e-15.

**Plausible causes, to tell apart with the reproducer.**
- **Derivative gain:** MCP's derivative on its middle piece is 1/(1 − γ/b). An implicit hypergradient through it is amplified without bound as b → γ⁺, the edge of `check_step`'s range.
- **Outer step:** nonconvex proximal gradient with an outer step above 1/L, which converges for convex penalties in practice but need not for MCP.
- **Ties at thresholds:** singular values at γλ or bλ, where the derivative's branch is a choice (kinks), combined with ties between singular values.

**Plan.**
1. Ask for the failing script, or the operator parameters, γ, the matrix sizes and the outer method.
2. Turn it into a regression test before any change.
3. If the library is at fault, a fix item follows here. If not, the docs gain the MCP gain 1/(1 − γ/b) and the step condition for nonconvex proximal gradient, in the "Inexact proxes" section (U1).
4. Either way, add spectral MCP and SCAD cases with singular values on γλ and bλ, and tied there, to the spectral tests.

**Acceptance.** The reproducer is understood and is covered by a test, and the outcome is recorded here.

**Outcome (2026-10-24).** The library is not at fault. The model has no minimiser, as the report itself said.

- **The reproducer.** The report's project is `bilevel_ml_fastprox`, experiment 8, `experiments/matrix_completion.py`. Its docstring notes that spectral MCP "leaves the singular values above λb unpenalised, so nothing bounds the completed matrix on the unobserved entries and the lower level has no minimiser: FISTA diverged". The setup:
  - MovieLens 100k, 300 × 300, 26 805 training ratings, centred;
  - ½ Σ_train (W_ij − R_ij)² + λ g(W);
  - FISTA with gradient restart at τ = 1/L.

  `bench/repro_u6_spectral_mcp.py` rewrites it independently. With `Spectral(MCP(1, 30))` and λ = 10, objective and ‖W‖_F:

  | iteration | 250 | 1 000 | 4 000 | 8 000 | 16 000 |
  |---|---|---|---|---|---|
  | FISTA + restart, τ = 1 | 5418 / 810 | 5374 / 1 569 | 5349 / 4 702 | 5344 / 8 691 | 5342 / 12 340 |
  | proximal gradient, τ = 0.99 | 5613 / 399 | 5478 / 497 | 5411 / 612 | 5392 / 680 | — |

  The objective falls at every iteration while the iterates grow. Unobserved entries reached 2 170 after 16 000 iterations, on centred ratings between −2.7 and 1.3. The nuclear norm stays at 10 792 / 132.9 from iteration 250 on, for both methods.
- **Why.** MCP is constant above bλ and the data term is blind to the unobserved entries, so the objective is not coercive. The lowest values lie at ever larger W, so it is not a step-size problem: the safe proximal gradient drifts too, only more slowly.
- **Small synthetic cases** (20–60 square, 30–60% observed, with and without noise) converge instead. They have enough degrees of freedom to fit the observed entries exactly at finite W, ending at a whole number of saturated MCP penalties. So the suite's tests check the operator, while the bench script carries the reproducer, which needs the 5 MB MovieLens archive.

The checks below were made before the reproducer turned up. They found the operator correct on every plausible cause, and the docs now cover each of them.
- **Derivative gain.** On the middle piece the JVP is exactly b/(b − γ): 1.5 at b = 3, 11 at 1.1, 1001 at 1.001, with γ = 1. γ ≥ b is refused (`ValueError`). A hypergradient through it is amplified without bound as b falls to γ.
- **Outer step.** On 50% completion of a rank-4 30 × 25 matrix (L = 1, λ = 0.5, b = 3, 300 iterations):

  | step | proximal gradient, MCP | proximal gradient, nuclear norm | FISTA, MCP | FISTA, nuclear norm |
  |---|---|---|---|---|
  | 0.5/L | converges, monotone | converges, monotone | converges, 78 rises | converges, 86 rises |
  | 0.99/L | converges, monotone | converges, monotone | converges, 28 rises | converges, 100 rises |
  | 1.5/L | converges, monotone | converges, monotone | diverges (iteration 74) | diverges (75) |
  | 1.99/L | stalls: 150 rises, relative error 1.0 | converges, monotone | diverges (24) | diverges (24) |
  | 2.2/L | diverges (90) | diverges (93) | diverges (20) | diverges (20) |

  So FISTA diverges above 1/L whatever the penalty. Proximal gradient tolerates up to 2/L only with the convex penalty; the theory for a nonconvex one needs a step below 1/L (PALM with one block: Bolte, Sabach and Teboulle 2014; Chambolle and Pock 2016, Section 6.2). FISTA is not monotone with either penalty, since it is not a descent method.
- **Ties at thresholds.** Values on a kink, alone or tied, give one element of the B-subdifferential, finite and adjoint to the VJP to 1e-15. On the kink each value takes the branch below. At a tie that rounding splits across the kink, the JVP is the limit of the Jacobian at points with the pair split ±δ in the same basis (agreement O(δ): 1.2e-4 at δ = 1e-3, 1.2e-5 at 1e-4). It depends on the basis inside the tie: a rotation there moved it by 12–20%. That is the same kind of choice as a branch at a kink. It explains why spectral MCP had been left out of the existing ties test, which used a tie at b λ.
- **Tests** (`tests/python/test_spectral.py`):
  - `test_mcp_ties`: ties on each piece, against finite differences;
  - `test_kinks_and_ties`: MCP at γλ and bλ, SCAD at (1 + γ)λ and aλ, alone and tied, exact and split by rounding;
  - `test_mcp_gain`;
  - `test_mcp_proximal_gradient`: monotone at 0.99/L.
- **Docs** (docs/api.md):
  - "Inexact proxes inside outer solvers" gains the step conditions for nonconvex penalties, the prox's step range and the derivative gains of MCP and SCAD;
  - "Spectral operators" gains the behaviour at kinks.
- **The docs** also gain the reproducer's lesson under "Inexact proxes inside outer solvers", "A minimiser may not exist": a bounded penalty needs a coercive term when the data term sees only part of the entries.

---

<a id="u7"></a>
### U7. Releases, so that no one builds from a workspace

**Problem.**
- **The release on PyPI lacks the spectral operators.** PyPI has 1.0.0rc1 (2026-10-20), and the spectral operators landed on master after it. Using them meant building from the development workspace, which broke once on uncommitted edits.
- **The papers project now pins the PyPI release**, so the spectral experiments were dropped.

**Design.**
1. **1.0.0rc2 (A11) ships master**, the spectral operators included, once the audit's remaining 1.0.0 items land. It goes to PyPI, so the spectral operators can be used without a build. The U items may land after rc2. If they change the API, as U1–U5 do, a 1.0.0rc3 follows before the final release.
2. **1.0.0** follows after some use of the last candidate, with every U item in.
3. **README.** The install section gains `pip install --pre fastprox` for release candidates, and `pip install "fastprox @ git+https://github.com/dvillacis/fastprox@v1.0.0rc2"` for a tag. CI already builds every push to master. Users who need unreleased work install from a commit of master, never from a working tree, and the README says so.

**Constraints.** Process only. The release steps are those of `CLAUDE.md` ("Releasing").

**Acceptance.** 1.0.0rc2 and then 1.0.0 are on PyPI, and the README documents both install paths.

---

## Decisions (2026-10-23)

1. **U2, the context's factor.** It is 0.1: inside `fp.inner_accuracy(target)`, ztol = 0.1·target. The inner error is then a tenth of the outer tolerance and never decides the outer test, and warm starts keep the tighter solves cheap. For accelerated methods the theory asks for errors that decrease over the iterations, so the docs (U1) also give the decreasing schedule. `ConvergenceWarning` is Python only, since C callers receive the status.
2. **U2, the default tolerance.** `tol` stays 1e-6, and its meaning (a certified relative error of 1e-3 for z) is documented. Tightening it would change every caller's results and run times. U1 and U2 address the actual pitfall: not knowing what tol certifies. A default in the units of z, `ztol` replacing `tol`, is a candidate for 2.0.
3. **U4, refining.** It is opt-in (`refine=`, `ctx.refine`), with the warning. An automatic refine would make a derivative's cost unpredictable, and would differentiate at a point other than the z the caller holds, a mismatch that is hard to trace in an autograd graph. With `fp.inner_accuracy` setting tight tolerances, the warning rarely fires.
4. **U5, nondecreasing weights.** They are deferred until asked for. The penalty is nonconvex. Its prox is simple, soft thresholding in sorted order, which stays sorted. But at singular values tied with different weights the prox is set-valued and has no derivative, which needs a tie rule and the nonconvex kink machinery. Nonincreasing weights, the convex weighted nuclear norm, cover the common use.
5. **Target.** Every U item gates 1.0.0, and U6 also needs the reproducer.
