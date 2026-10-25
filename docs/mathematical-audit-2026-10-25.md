# Mathematical audit of fastprox

Date: 2026-10-25. Source revision: `0a1a0cc` (`v1.0.0` plus seven commits).

## Assessment

The ordinary-scale proximal formulas and first derivatives examined in this
audit have strong supporting evidence. However, the package does not currently
meet an unqualified claim that every returned derivative is exact or that every
`prox_conj` is a proximal operator of the stated Fenchel conjugate.

There are six confirmed implementation/contract defects, one certificate
qualification, and two documentation findings. M1 and M2 affect mathematical
correctness on moderate inputs. M3–M6 concern extreme but accepted finite
parameters. They should not be presented as evidence that ordinary denoising
problems generally fail.

No library implementation was changed during this audit. The accompanying
script and JSON snapshot make the numerical findings reproducible.

## Scope and evidence

The review examined the Euclidean prox convention, conjugation, scalar
nonconvex branches, radial/group composition, projections, isotonic regression,
sorted L1, TV algorithms/certificates, and spectral differentiation. This is a
targeted mathematical audit, not a formal proof of every execution path.

The executable used was the existing local fastprox 1.0.0 extension. Its path,
dependency versions, source HEAD, and individual results are recorded in the
JSON snapshot. The source files responsible for M1–M6 are unchanged between
`v1.0.0` and the reviewed HEAD (`git diff v1.0.0 HEAD --` those files is empty), so
these findings also apply to the release source. A clean installation of each
published wheel was not performed. The existing C build was used for C tests.
An unrelated working-tree edit to `src/ops/isotonic.cpp` appeared during the
audit; it was left untouched and is outside the reviewed revision. Its weight
rescaling changes do not change the metric counterexample in M2.

Validation completed:

| Check | Result | Meaning and limit |
| --- | --- | --- |
| Complete existing Python suite | 2,291 passed, 47 skipped, 7 warnings; 112.64 s | Covers the locally available frameworks and devices; skipped paths are not validated |
| Existing C/C++ CTest suite | 11/11 targets passed | Existing local build; no new sanitizer build |
| Independent convex optimization comparisons | 91 passed across 19 families | Small float64 problems; normalized L2 error below 2e-5, worst observed 3.83e-6 |
| Independent scalar nonconvex minimizations | 140 passed | Lq at four q values, log, SCAD and MCP; compares objective minima on scalar intervals |
| Independent first-derivative checks | 19 operators passed | Central differences, JVP/VJP adjointness, symmetry; PSD Jacobians and the convex contraction bound where applicable |
| Analytic adversarial cases | M1–M7 reproduced | These are new cases outside the passing baseline |

The convex oracle models the objective directly in CVXPY rather than calling
the repository's reference helpers or comparing one fastprox solver against
another. Two vector problems whose Clarabel status was `optimal_inaccurate`
were rerun with SCS and required to finish `optimal`; solver/status details are
in the JSON. All six image references used Clarabel. Conic solvers are numerical
references, not formal exact-arithmetic proofs. The strongest findings below
instead have elementary analytic counterexamples.

The Python process also emitted nanobind reference-leak diagnostics at
shutdown. These are an engineering observation outside this mathematical
audit; they do not invalidate the numerical comparisons.

## Findings at a glance

| ID | Priority | Finding |
| --- | --- | --- |
| M1 | High | Spectral near-tie rule returns the wrong derivative across a scalar breakpoint, with `exact=True` |
| M2 | High | Weighted isotonic `prox_conj` violates the Euclidean conjugate contract |
| M3 | Medium, extreme range | L0 threshold overflows before a small parameter cancels the large step |
| M4 | Medium, extreme range | Log-barrier prox returns an infeasible zero although the positive answer is representable |
| M5 | Medium, extreme range | TV1DL2 silently drops a positive weight whose square underflows, even when gamma makes its effect order one |
| M6 | Medium, extreme range | SCAD/MCP and their radial forms return infinity for a finite solution near 2 |
| M7 | Certificate qualification | `zerr=0` omits rounding of the returned float32 image |
| D1 | Mathematical documentation | Clarke/Lipschitz claims improperly include discontinuous nonconvex threshold selections |
| D2 | Mathematical documentation | Lq and log penalties are incorrectly included among bounded penalties |

## M1. Spectral near ties across a breakpoint

Source: `src/core/spectral.hpp:125`, `src/core/spectral.hpp:164`;
`Spectral.jvp`/`vjp` in `bindings/python/fastprox/__init__.py`.

For singular-value thresholding, let

\[
X=\operatorname{diag}(1+\delta,1-2\delta),\qquad
H=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
f(s)=(s-1)_+.
\]

Both singular values are distinct, positive, and strictly away from the
threshold. Thus the matrix map is differentiable at X. Its derivative on this
symmetric off-diagonal direction is

\[
D\operatorname{prox}(X)[H]
=\frac{f(1+\delta)-f(1-2\delta)}{3\delta}H
=\frac13 H.
\]

For delta = 1e-7 in float64, fastprox returns `0.5 * H`. The saved spectrum's
exact divided difference is `0.33333333345669147`. This is approximately 50%
relative error in the entire JVP, not a choice of generalized derivative at a
kink. After the call, `ctx.deriv.exact` is **True** and smoothing is zero.

The failure also occurs in float32 with X = diag(1.001, 0.998): the computed
coefficient is 0.5 rather than approximately 0.3333466. Thus its practical
importance is not confined to double-precision gaps of 1e-7.

**Cause.** Pairs within `cbrt(eps) * max(abs(s))` use the mean of branch
derivatives, here (1 + 0)/2. Averaging endpoint derivatives approximates a
divided difference only under appropriate smoothness between the endpoints.
That assumption fails when a threshold lies between them. The error stays
order one as delta decreases. The same bad coefficient enters the spectral
VJP. The generic core is also shared by eigenvalue operators, although the
minimal reproducer here uses the nuclear norm.

`tests/python/test_spectral.py::test_chained_near_ties` explicitly places its
threshold halfway between two values. The exact divided difference in that
special arrangement is 1/2, so it cannot detect this asymmetric crossing.

**Required correction.** Preserve the actual divided difference for distinct
values on different pieces, with stable operator-specific evaluation if
needed. Use a tie limit only under its mathematical hypotheses. Add asymmetric
crossings in float32/float64, for both JVP and VJP. A heuristic approximation
must not be reported as exact. Simply narrowing the band does not repair the
underlying argument.

## M2. Weighted isotonic conjugation mixes metrics

Source: `src/ops/isotonic.cpp:42`; API conjugate dispatch. The forward metric
exception is documented at `docs/api.md:135`, but the conjugate exception is
not accounted for by the implementation.

For the monotone cone C = {z: z1 <= z2}, the operator's value is its indicator,
and its Euclidean Fenchel conjugate is the indicator of

\[
C^\circ=\{(a,-a):a\geq0\}.
\]

With weights (1,3) and y = (2,0), weighted least-squares isotonic regression
correctly returns (0.5,0.5). But the homogeneous conjugate dispatch computes

\[
y-P_C^W(y)=(1.5,-0.5).
\]

That point does not even belong to C-polar: its entries sum to 1 instead of 0.
The actual Euclidean conjugate prox is P_C-polar(y) = (1,-1), for every
positive sigma.

The forward weighted fit is not itself defective. The error is applying the
Euclidean Moreau identity to a projection in a different metric. For a
weighted projection it is W(y - P_C^W(y)) that has the normal/polar-cone
interpretation; the corresponding decomposition uses the inverse dual metric.

**Required correction.** Either reject conjugate calls for nonuniform
isotonic weights, compute the Euclidean conjugate independently, or expose an
explicit metric-aware primal/dual contract. Preserve the weighted forward
fit. Apply matching rules to conjugate derivatives and framework wrappers.

## M3. L0 threshold evaluation changes the global minimizer

Source: `src/ops/nonconvex.cpp`, `L0::thr`.

```python
fp.L0(1e-308).prox(np.array([2.0]), 1e308)
# observed: [0.0]; expected: [2.0]
```

The scaled scalar objective is

\[
F(z)=\tfrac12(z-2)^2+(\gamma\lambda)[z\ne0],\qquad
\gamma\lambda\simeq1.
\]

F(0) = 2 and F(2) is approximately 1, so the nonzero branch is unambiguously
better. The threshold should be approximately sqrt(2). Instead,
`sqrt(2.0 * gamma * lambda * weight)` first overflows `2.0 * gamma`, producing
an infinite threshold. The accepted scaled-step bound does not prevent this
intermediate overflow. The returned `Info.gap` is zero.

**Required correction.** Evaluate the product/root without avoidable
intermediate overflow, or reject unsupported combinations. Test reciprocal
step/parameter scalings that preserve the mathematical objective.

## M4. Log-barrier result leaves its domain

Source: `src/ops/pointwise.cpp:409`.

```python
fp.LogBarrier(1e280).prox(np.array([-1e308]), 1.0)
# observed: [0.0]; expected: approximately [1e-28]
```

For g(z) = -lambda log(z), z > 0, stationarity gives

\[
z=\frac{x+\sqrt{x^2+4\gamma\lambda}}{2}
 =\frac{\gamma\lambda}{\tfrac12\sqrt{x^2+4\gamma\lambda}-\tfrac12x}.
\]

The positive answer is approximately 1e-28 and is representable. A separate
700-digit mpmath calculation verifies it. The implementation obtains the
square root safely through `hypot`, but then forms `r - x` **before** dividing
by two. That difference overflows; division by infinity returns zero. Zero
is outside the barrier domain and has infinite objective. `Info.gap` remains
zero.

**Required correction.** Form the denominator with safe scaling, for example
the two half terms in this case, and audit the root's underflow regime as
well. A positive-step, positive-lambda log barrier must not return zero when
its positive optimum is representable.

## M5. TV1DL2 loses a mathematically significant edge

Source: `src/ops/tv1d_l2.cpp:509` and the code paths that treat a squared
weight of zero as an absent edge.

```python
fp.TV1DL2(1e-180).prox(np.array([1.0, -1.0]), 1e180)
# observed: [1.0, -1.0]; expected: [0.0, 0.0]
```

For two entries, gamma ||W Dz||2 is simply gamma w |z2-z1|. Here gamma w is
approximately one, so the exact problem is an ordinary two-node TV problem.
Its optimum is zero. The objective at the returned input is approximately 2,
whereas the objective at zero is 1.

The constructor accepts w, stores w squared as zero after underflow, and the
solver removes the edge. The zero gap therefore does not describe the stated
objective. Documentation acknowledges that underflowed squares count as zero,
but that is not an innocuous floating-point approximation: multiplication by
gamma can make the discarded term order one.

**Required correction.** Normalize weights with a corresponding adjustment of
gamma before squaring, or reject positive weights whose squares cannot be
represented. If mixed weight ranges require a restriction, validate and
document it rather than silently solving a different problem. Add the
identity prox(gamma,w) = prox(gamma/c,cw) over representable extreme scales.

## M6. SCAD/MCP intermediate products overflow

Source: `src/ops/nonconvex.cpp`, `Scad::prox1`, `Mcp::prox1`, and their
derivative formulas; `src/ops/radial.cpp` delegates to these scalar kernels.

```python
fp.SCAD(1.0, 1e308).prox(np.array([3.0]), 1.0)
fp.MCP(1.0, 1e308).prox(np.array([3.0]), 1.0)
# both observed: [inf]; expected: approximately [2.0]
```

The uniqueness conditions a > 1 + gamma and b > gamma hold by a huge margin.
The exact middle-piece answers are

\[
z_{SCAD}=\frac{2a-3}{a-2},\qquad
z_{MCP}=\frac{2b}{b-1},
\]

both rounding to 2. Intermediate products with a or b overflow even though
the final ratios are well conditioned. GroupSCAD and GroupMCP reproduce the
same failure on a one-entry group. Every call reports gap zero. Derivatives
use related products and should be checked along with the forward repair;
this audit does not claim a separate reproduced derivative counterexample
for M6.

**Required correction.** Evaluate bounded ratios before multiplying by the
input, restructure the SCAD numerator to avoid large cancelling products,
and check scalar and radial parameter derivatives. Alternatively enforce an
explicit supported parameter range.

## M7. Certificates omit output rounding

Sources: image certificate calculation and casting of the final image to its
requested dtype; API claims about `ztol` and `info.zerr`.

For either TVIso(2) or TVAniso(2):

```python
x = np.array([[0.0, 1.0]], dtype=np.float32)
z, info = op.solve(x, 0.1, ztol=1e-12, max_iter=1000, warn=False)
# info.gap == 0.0; info.zerr == 0.0
```

The real-valued solution is (0.1,0.9), but the returned float32 array has
relative L2 error approximately 2.38884e-8; ||x|| is one. This is ordinary
output rounding, not a failure of the TV algorithm. Nevertheless, a literal
claim that `zerr` bounds the error of the returned array, or that a 1e-12
target was certified, is false in this case.

In exact arithmetic, for a feasible dual u and the scaled objective
F(z) = 0.5 ||z-x|| squared + gamma g(z), strong convexity correctly yields

\[
\|z-z^\star\|^2\leq2(F(z)-D(u)).
\]

That proof is sound. The computed gap is not a rigorous directed-rounding
enclosure, however, and it must correspond to the point actually returned.

**Required qualification.** State that certificates hold to working precision,
or include output-rounding error in `zerr`/gap and report unattainable targets
appropriately. Re-evaluating a valid primal/dual gap at the cast output can
account for casting, but rigorous bounds on its own floating-point evaluation
would still require additional error analysis. Closed-form `Info.gap=0` should
also be distinguished from an independently measured certificate.

## D1. Discontinuous threshold maps have no usual Clarke Jacobian at a jump

Sources: `docs/spec.md:73`, `docs/api.md:543`, the derivative table including
L0/Lq/log ties, and the following claim about Clarke subgradients of an outer
loss.

Clarke's usual generalized Jacobian requires local Lipschitz continuity. A
single-valued hard-threshold selection at a positive threshold t has a jump:
the limit from below is zero, while the limit from above is t. Neither tie
choice removes it. Lq and the nonconvex log regime likewise have discontinuous
selections when their two minimizing branches have different values.

Therefore the table's blanket justification does not cover those points. A
branch slope is useful and can be documented as a convention, but is not the
Clarke Jacobian of the discontinuous selected map. A smooth outer loss does
not automatically repair this: the smooth loss L(z)=z preserves the jump.

**Required correction.** Separate locally Lipschitz continuous prox maps from
discontinuous selections, qualify the chain-rule claim, and distinguish
branch derivatives from true directional derivatives/generalized Jacobians.
The branch derivatives away from the jumps passed the ordinary-point checks.

## D2. Boundedness and existence claims are overgeneralized

Source: `docs/api.md:748`–`docs/api.md:763`.

The section lists SCAD, MCP, L0, Lq and log, then says these penalties are
bounded. Lq with q > 0 and log(1+|z|/epsilon) are **unbounded** and, for positive
coefficients in the finite-dimensional scalar/separable cases, coercive.
Their spectral versions with positive uniform coefficients also grow without
bound as the Frobenius norm tends to infinity. SCAD/MCP/L0 behave differently.

Also, noncoercivity alone does not prove that a minimizer does not exist, and
divergence of one algorithm does not prove nonattainment. The particular MCP
completion example may indeed have a nonattained infimum, but that needs its
own argument; this audit does not decide that example's existence question.

**Required correction.** Restrict the bounded-penalty statement to the right
families and state the additional assumptions behind existence and convergence
claims. The README's claim that every operator has its conjugate prox should
likewise inherit the existing nonconvex caveat from the detailed API: for
those penalties the implemented Moreau residual is deliberately not the
Fenchel-conjugate prox. That caveat is already explicit in the detailed API
and C header and is not counted as a new hidden implementation error.

## Mathematical parts supported by this audit

The following constructions are mathematically consistent under their stated
hypotheses and passed the sampled checks where applicable:

- Convex scalar soft thresholding, elastic net, the chosen Huber normalization,
  log barrier at ordinary scales, and projections onto the tested convex sets.
- Group radial reduction by Cauchy–Schwarz and the radial Jacobian, and
  nested/disjoint group-L2 composition in inclusion order.
- Unweighted isotonic PAVA; weighted isotonic as a weighted least-squares fit,
  with its weighted block Jacobian and a distinct Euclidean adjoint.
- Sorted L1's sorting/PAVA/positive-part construction for nonincreasing,
  nonnegative weights.
- Two-node and sampled larger TV problems, the TV1DL2 trust-region
  formulation, and graph TV agreement with independently modeled objectives.
- The real-arithmetic TV gap-to-primal-distance proof above. A primal error
  bound does not by itself bound a hypergradient; region identification,
  curvature approximation, conditioning, and adjoint residual also matter.
- Scalar nonconvex thresholds/global branch selection at sampled ordinary
  scales, and MCP/SCAD's necessity of the stated weak-convexity step
  restriction for the continuously single-valued formulas used here.
- Spectral divided-difference formulas at distinct values and the general
  approach to limits at repeated values. M1 is an implementation/approximation
  issue in when a limit replaces a quotient, not a rejection of that theory.

These are sampled validations and algebraic reviews, not exhaustive proofs.
The audit did not rerun long fuzz campaigns, establish all graph max-flow
invariants, validate every GPU/device path, certify higher-order derivatives,
or establish uniform backward-error bounds over the accepted parameter space.

## Recommended order of work

1. Repair M1 and M2 and add their analytic counterexamples as regression tests.
2. Repair or explicitly reject M3–M6; retain objective-level tests, not just
   finiteness checks or comparisons with a rescaled version of the same code.
3. Clarify M7 and D1–D2 so users can distinguish mathematical guarantees,
   numerical estimates, branch conventions, and unsupported operations.
4. Revalidate on float32/float64 and framework wrappers, then publish a patch
   release with these cases and the precise scope of the guarantees.

## Reproduction

Artifacts:

- `reports/mathematical_audit_2026_10_25.py`: independent calculations and
  adversarial cases; no calls to repository reference implementations.
- `reports/mathematical_audit_2026_10_25.json`: captured results and environment.

From the existing reference environment:

```sh
uv run --no-sync python reports/mathematical_audit_2026_10_25.py
uv run --no-sync pytest -q
ctest --test-dir build/c --output-on-failure --timeout 60
```

The script is diagnostic: reproduced findings are printed/recorded as such,
rather than installed as deliberately failing tests in the normal suite.
It requires CVXPY, Clarabel, SCS, SciPy and mpmath in addition to fastprox/NumPy.
This run used `uv --cache-dir /private/tmp/fastprox-audit-uv` to keep cache
writes in an allowed location. It does not install or change dependencies.

## Mathematical references

- Parikh and Boyd, [Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.pdf),
  especially Moreau decomposition and convex proximal properties.
- Ding, Sun, Sun and Toh, [Spectral Operators of Matrices](https://arxiv.org/abs/1401.2269),
  for spectral differentiability and the divided-difference/tie-limit framework.
- [A Note on Clarke's Generalized Jacobian for the Inverse of Bi-Lipschitz Maps](https://link.springer.com/article/10.1007/s10957-023-02333-x),
  Definition 1.1, for the local-Lipschitz hypothesis of the usual Clarke Jacobian.
- Schmidt, Le Roux and Bach, [Convergence Rates of Inexact Proximal-Gradient Methods](https://arxiv.org/abs/1109.2415),
  for assumptions on decreasing inner errors; such conditions do not follow
  from a fixed loose tolerance or a small adjoint residual alone.

The counterexamples M1–M7 are derived and reproduced in this audit; they do
not depend on accepting an external source's assessment of fastprox.

## Errata (2026-10-25)

- **The JSON snapshot.** `reports/mathematical_audit_2026_10_25.json` was
  regenerated at `8daf8fa`, the master revision that the
  [specs for 1.0.1](math-audit-2026-10.md) start from. All of M1–M7 still
  reproduce there, with the same values.
- **D1.** On wording: the B-subdifferential (the set of limits of nearby
  Jacobians) is defined for any map. What a jump removes is Clarke's
  generalised Jacobian and the Lipschitz calculus built on it. The finding is
  unchanged.
