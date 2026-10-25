# Theoretical audit of fastprox

2026-10-25. Committed source baseline: `0a1a0cc`, version 1.0.0.
This supplements the [numerical/mathematical audit](mathematical-audit-2026-10-25.md).
Concurrent working-tree changes in isotonic regression and its tests were left
untouched. No library implementation was modified for this review.

## Conclusion and scope

The basic architecture rests on sound mathematics: Euclidean proximal maps,
convex dual certificates, scalar-to-radial and scalar-to-spectral reductions,
and differentiation on identified smooth pieces. Its strongest guarantees are
conditional, however. A correct prox, an exact branch derivative, a Clarke
Jacobian, and a valid bilevel hypergradient are different mathematical objects.
The API documentation sometimes treats them as interchangeable.

This audit derives the conditions separating those objects. Its main new
counterexamples concern composition of generalized derivatives, the critical
log-penalty step, and graph-TV kink detection. Other sections establish valid
formulas, correct overbroad claims, or explain what an additional theorem would
need. These are not all new implementation defects.

| Item | Theoretical result | Status for the package |
| --- | --- | --- |
| T1 | Convex and weakly convex proximal guarantees need different constants | Sound formulas under the specified restrictions |
| T2 | A unique continuous prox need not be locally Lipschitz | Allowed log-penalty boundary is a concrete exception to blanket Clarke claims |
| T3 | Valid ambient generalized Jacobians do not automatically compose/restrict correctly | Counterexample propagates 2 for a function whose derivative is 1 |
| T4 | Small primal gap does not uniformly control Jacobian error | A two-node TV construction has gap tending to zero and derivative error one |
| T5 | Saturation in one graph dual is not intrinsic primal nondifferentiability | A smooth triangle-TV example produces a nonzero kink mask |
| T6 | The listed whole-space nonconvex penalties have trivial Fenchel conjugates | Their exposed Moreau residuals need a distinct mathematical name/contract |
| T7 | Exact prox derivatives do not guarantee differentiability of a lower solution | Bilevel differentiation additionally needs an invertibility/regularity condition |
| T8 | Outer convergence and step ranges require precise hypotheses | Standard sufficient conditions are sometimes phrased as necessities or unconditional guarantees |
| T9 | Spectral near-tie approximations require smoothness across the interval | Existing limit heuristic has no such guarantee across thresholds |

The proofs below concern finite-dimensional real spaces and exact arithmetic.
They do not establish floating-point enclosures, correctness of every max-flow
implementation path, or correctness of all generalized-Jacobian selections in
every operator. Small executable checks accompany the explicit examples;
the conclusions depend on the derivations, not empirical convergence alone.

## T1. What the proximal contract actually guarantees

Write

\[
P_{\gamma,g}(x)\in\arg\min_z
\{\tfrac12\|z-x\|^2+\gamma g(z)\},\qquad\gamma>0.
\]

### Closed proper convex penalties

For a proper lower-semicontinuous convex g, the objective is strongly convex
and has exactly one minimizer. Optimality is

\[
(x-P(x))/\gamma\in\partial g(P(x)).
\]

Monotonicity of the subdifferential at two points gives

\[
\|P(x)-P(y)\|^2\leq\langle P(x)-P(y),x-y\rangle.
\]

Thus P is firmly nonexpansive and 1-Lipschitz. At differentiability points its
Jacobian is symmetric positive semidefinite with eigenvalues in [0,1]. One
way to see symmetry and positivity is to write

\[
P=\nabla\Psi,\qquad
\Psi(x)=(\tfrac12\|\cdot\|^2+\gamma g)^*(x).
\]

Here the conjugated function is strongly convex, so its conjugate is
differentiable with Lipschitz gradient. These facts justify useful structural
checks on Euclidean convex JVPs/VJPs. They apply to the actual prox, not to an
arbitrary approximate derivative computed from guessed regions.

Weighted isotonic regression is an intentional exception: its metric is W,
and its block Jacobian need not be symmetric in the Euclidean inner product.
It satisfies a weighted projection contract. The earlier audit's conjugate
counterexample results from forgetting this distinction.

### Weakly convex penalties

Suppose g + (rho/2)||z|| squared is convex. If gamma rho < 1, then the prox
objective is (1-gamma rho)-strongly convex. The same argument gives

\[
\langle P(x)-P(y),x-y\rangle
\geq(1-\gamma\rho)\|P(x)-P(y)\|^2,
\]

and therefore

\[
\operatorname{Lip}(P)\leq\frac1{1-\gamma\rho}.
\]

The map may be expansive. For weighted MCP, rho = max(w)/b; for weighted SCAD,
rho = max(w)/(a-1). The library's b > gamma max(w) and a > 1 + gamma max(w)
are exactly the corresponding sufficient strong-convexity restrictions.
On the middle pieces the Jacobian gains b/(b-gamma w) and
(a-1)/(a-1-gamma w) realize this amplification.

Radial group versions inherit the scalar weak-convexity bound: for r > 0,

\[
\nabla^2[h(\|z\|)]
=h''(r)uu^\top+\frac{h'(r)}r(I-uu^\top),\qquad u=z/r.
\]

For SCAD/MCP, h' is nonnegative and h'' is bounded below by -rho.
The tangential eigenvalues are nonnegative, so the same lower curvature bound
applies. Appropriate even scalar weakly convex penalties also lift to
spectral penalties after adding rho ||X||_F squared / 2.

**Consequence.** Single-valued does not mean nonexpansive; a small positive
denominator can produce a large, mathematically correct sensitivity. This is
conditioning rather than an implementation error.

### Discontinuous selections

For L0 and Lq with 0 < q < 1, the global prox can have a jump. The Lq jump can
be derived directly. Put c = gamma lambda w and consider x,z >= 0. At a
nonzero stationary point, x = z + c q z^(q-1). Equality of its objective with
the zero candidate then gives

\[
0=-\tfrac12z^2+c(1-q)z^q,
\quad\text{so}\quad
\beta=[2c(1-q)]^{1/(2-q)}>0.
\]

The transition input is tau = beta + c q beta^(q-1). The one-sided limits of
any tie-broken prox at tau are 0 and beta. Choosing either minimizer at the
tie cannot remove the discontinuity. Lq is not globally weakly convex: its
negative second derivative diverges near zero.

Away from jumps, branch derivatives are useful and ordinary. At the jump,
the usual locally-Lipschitz Clarke Jacobian is unavailable. A branch slope
must be described as a selection convention, not given the general convex
proximal calculus automatically.

## T2. An allowed continuous prox that is not locally Lipschitz

The log penalty needs a third category between Lipschitz maps and jump maps.
For g(z) = lambda log(1+|z|/epsilon), its weak-convexity constant is
rho = lambda/epsilon squared. Accordingly:

| Regime | Scalar prox near its transition |
| --- | --- |
| gamma lambda < epsilon squared | Unique and Lipschitz; strong-convexity argument applies |
| gamma lambda = epsilon squared | Unique and continuous, but generally only Hölder 1/2 at the threshold |
| gamma lambda > epsilon squared | Two competing minima at a transition; tie-broken selection jumps |

For an exact example, let gamma = lambda = epsilon = 1 and x = 1+h. For h > 0
the nonzero stationarity equation is

\[
z-(1+h)+\frac1{1+z}=0,
\quad z^2-hz-h=0,
\]

so

\[
P(1+h)=\frac{h+\sqrt{h^2+4h}}2\sim\sqrt h,
\qquad P(1)=0.
\]

The objective is convex and strictly convex, so this is not an ambiguity in
the minimizing branch. Nevertheless,

\[
\frac{P(1+h)-P(1)}h\longrightarrow+\infty.
\]

There is no finite local Lipschitz constant or usual Clarke Jacobian at x=1.
The implementation permits the parameters and returns the zero branch's JVP
at the transition. That is a one-sided branch convention, not the derivative
of a locally Lipschitz prox. The check script observes right difference
quotients about 10.5, 100.5 and 1000.5 as h goes from 1e-2 to 1e-6.

**Required theoretical qualification.** The statement that all continuous
nonconvex maps admit the same generalized-Jacobian calculus is too broad.
Either exclude the critical equality from a Lipschitz guarantee or explicitly
document its Hölder behavior and one-sided/infinite sensitivity.

## T3. Ambient generalized derivatives do not automatically give valid hypergradients

This issue remains even for a convex penalty with an exact forward solve.
Let P be projection onto the nonnegative orthant in R squared and define

\[
A t=(t,-t),\qquad L(z)=z_1-z_2,\qquad
F(t)=L(P(At)).
\]

The identity max(t,0)-max(-t,0)=t shows F(t)=t everywhere. Thus F is smooth and
its only Clarke derivative at zero is 1.

At the ambient point (0,0), the library chooses DP = I for the orthant. This
is a valid B-subdifferential selection for P: it is reached from the positive
quadrant. But applying that selection to the composite gives

\[
\nabla L^\top I A=(1,-1)(1,-1)^\top=2.
\]

The check script reproduces 2 through the VJP while the direct difference
quotient of F is 1. Both local operations are implemented as documented.
The problem is the inference from local validity to validity after restriction.
No nonzero point on the line (t,-t) accesses the positive quadrant used to
justify the ambient selection.

This does **not** refute the narrower statement that, for a locally Lipschitz
piecewise smooth P and a smooth scalar L, an accessible B-Jacobian of P gives
a Clarke subgradient of L composed with P as a function of the full ambient
input. That argument can use the same ambient sequence of smooth points.
The failure arises after an upstream parametrization or additional nonsmooth
composition makes that sequence inaccessible.

Nor does changing every local kink slope to 1/2 solve the problem generally:
the identically zero function ReLU(-ReLU(t)) would then propagate -1/4.

**Consequence for the wrappers.** Outside switching sets, classical chain
rules are appropriate. On switching sets, a fixed JVP/VJP convention need not
yield a Clarke derivative of an arbitrary program or bilevel objective.
The proper alternatives are a composition-compatible conservative derivative
framework with verified hypotheses, explicit directional differentiation, or
clear documentation that the output is a branch derivative. The results of
[Bolte and Pauwels](https://arxiv.org/abs/1909.10300) explain why nonsmooth AD
uses richer objects than the Clarke subdifferential alone. Their theory does
not automatically validate every branch policy in this package, particularly
for discontinuous maps.

## T4. The primal certificate is sound but does not certify a Jacobian

### The exact certificate theorem

For the normalized convex objective

\[
F_x(z)=\tfrac12\|z-x\|^2+\gamma g(z),
\]

strong convexity implies

\[
F_x(z)-F_x(z^\star)\geq\tfrac12\|z-z^\star\|^2.
\]

If D(u) is any feasible dual value, weak duality gives D(u) <= F_x(z-star).
Consequently gap = F_x(z)-D(u) satisfies

\[
\boxed{\|z-z^\star\|\leq\sqrt{2\,\mathrm{gap}}.}
\]

This proves the real-arithmetic basis of `ztol`/`zerr`. Output rounding and
evaluation error are separate issues addressed in the previous audit.

### Why no uniform sensitivity bound follows

Consider two-node TV, gamma = 1, and

\[
x_\delta=(1+\delta,-1-\delta),\qquad \delta>0.
\]

The true solution is z-star = (delta,-delta), in a locally unfused region, so
its Jacobian is I. The approximate candidate z-tilde = 0, together with the
feasible edge flow f = 1 and divergence q=(1,-1), has gap

\[
\tfrac12\|\widetilde z-x_\delta+q\|^2=\delta^2.
\]

A derivative assembled from the candidate's fused region is
J-tilde = (1/2) 11-transpose. Nevertheless,

\[
\|I-\widetilde J\|_2=1
\quad\text{for every }\delta>0,
\qquad\mathrm{gap}\to0.
\]

This is a counterexample to any unconditional, uniform bound of the form
Jacobian error <= C sqrt(gap), including for convex piecewise-affine proxes.
It does not assert that the package necessarily accepts this particular
candidate as an exact solve. It establishes what a primal gap alone cannot
prove. Reported empirical constants for TV derivatives are not theorems over
all inputs and parameters.

### Sufficient local conditions

For a fixed nondegenerate point, local identification/sensitivity results
usually require a stable active manifold, a strict dual margin on inactive
groups, and separation of active jumps or singular values from transitions.
Such assumptions can yield a neighborhood with a fixed smooth model. They
are problem-dependent and degenerate as the point approaches a switch.

In particular, approximate dual interiority alone does not identify a true
flat edge unless its margin dominates a bound on dual error. A primal gap
does not control motion in a nonunique dual's null directions. Partial
smoothness and nondegeneracy results provide a route to local identification;
they do not turn an arbitrary positive gap threshold into an exact region
certificate. See the identification discussion in
[Vaiter, Peyré and Fadili](https://samuelvaiter.com/publications/vaiter2015low.pdf).

## T5. A graph dual's saturated edge need not be a kink of the prox

Consider the triangle with edges (0,1), (1,2), (0,2), unit weights, gamma = 1,
and x=(1,0,-1). Orient the incidence rows as e-u minus e-v. The following
two flows have the same divergence x:

\[
f^{(1)}=(0,0,1),\qquad
f^{(2)}=(1/3,1/3,2/3).
\]

At z=0 both are optimal duals. The first has a saturated edge; the second is
strictly inside every edge bound. Their difference is a circulation.

Because K-transpose maps onto the zero-mean subspace and the second flow has
a positive margin to every bound, all sufficiently small perturbations of
the centered data also have a strictly feasible flow. Thus on a neighborhood
of x the primal solution is exactly

\[
P(x')=\operatorname{mean}(x')\,\mathbf1.
\]

The prox is smooth there, with its unique Jacobian 11-transpose / 3. There is
no intrinsic primal kink at x.

The package returns z=0 and the **correct** averaging Jacobian, but
`ctx.kinks()` returns `[0, 0, 1]`. The numerical check also verifies 20 nearby
points and the strict flow algebraically.

**The theoretical issue.** The documentation states that on cyclic graphs a
kink depends on which dual is used. Nondifferentiability of the primal map
cannot depend on a dual representation. What depends on that representation
is the diagnostic condition “zero jump and saturation in this saved dual.”

Geometrically, graph TV is the support function of the incidence image of a
box, and its prox is the identity minus projection onto that image, a
zonotope. Relevant changes are changes of faces/normal geometry in the image;
not every boundary coordinate of a preimage flow is significant.

The default Jacobian in this example is not defective, and overriding the
single flagged edge does not split the region because the other triangle
edges still join it. The finding concerns the meaning of the mask and any
claim that every flagged edge represents two accessible primal branches.
Such access requires a globally consistent perturbation, not independent
binary choices in a redundant dual representation.

**Required correction.** Describe these masks as diagnostics for the saved
dual, or perform a representation-independent test before declaring a true
primal kink. Any proof that graph branch overrides select B-Jacobians must
also establish accessibility of the requested partition/sign pattern.

## T6. What the nonconvex Fenchel conjugates really are

There is a stronger statement than merely “Moreau's identity may fail.”
Suppose g is finite, nonnegative, g(0)=0, and g(tv)=o(t) as t tends to
infinity for every fixed v. Then for y != 0,

\[
g^*(y)\geq\sup_{t>0}\{t\|y\|^2-g(ty)\}=+\infty,
\qquad g^*(0)=0.
\]

Therefore

\[
\boxed{g^*=\delta_{\{0\}},\quad g^{**}=0,\quad
\operatorname{prox}_{\sigma g^*}(y)=0\ \ (\sigma>0).}
\]

These assumptions hold for the library's finite-dimensional whole-space L0,
Lq (0<q<1), SCAD, MCP, and LogPenalty penalties with the stated nonnegative
coefficients. SCAD/MCP/L0 are bounded along rays; Lq and LogPenalty are
unbounded but sublinear. The same argument applies to their whole-space
radial/spectral versions where provided.

The Moreau residual exposed for those functions is usually not zero. For
example MCP(lambda=1,b=3), sigma=1 and y=2 returns 0.5; the true Fenchel
conjugate prox is zero. This behavior is disclosed in the detailed API and
was not counted as a hidden implementation error in the earlier audit.

**Consequence.** A PDHG-style dual update using that residual is not justified
as a Fenchel-conjugate proximal step for the original nonconvex penalty. Also,
Fenchel biconjugation loses the nonconvex penalty entirely in these cases.
The operation can still be useful under a different explicitly stated
algorithmic interpretation, but requires its own analysis.

Do not confuse LogPenalty with the convex positive-domain LogBarrier; the
theorem above is about the former.

## T7. A theorem for a prox derivative is not a theorem for a bilevel solution

### Correct local TV system under a stable manifold

Let g(z)=sum_p r_p ||(Kz)_p||, absorbing fixed weights into K or r. At a
solution, ignore zero-weight groups and divide the remaining groups into
flat F and active A. Assume the flat constraints
are locally stable and active group gradients are nonzero. Let Q have
orthonormal columns spanning

\[
T=\{d:K_Fd=0\}.
\]

On active groups define n-p = (Kz)-p / ||(Kz)-p|| and

\[
H_p=\frac{\gamma r_p}{\|(Kz)_p\|}(I-n_pn_p^\top).
\]

Differentiating the optimality conditions at fixed parameters gives

\[
\boxed{J=Q[Q^\top(I+K_A^\top H K_A)Q]^{-1}Q^\top.}
\]

This is a sound basis for the implicit TV derivative. Its middle matrix is
positive definite with minimum eigenvalue at least one. For anisotropic TV,
the active scalar absolute-value pieces have no curvature, so the formula
reduces to orthogonal projection onto the stable fused-region subspace.
For isotropic TV the tangential curvature remains. With a nonorthonormal
region-indicator basis R, the correct mass matrix is R-transpose R, not I.

### Distinguish residual error from model error

Let A p=b be the correct derivative/adjoint system and let p-tilde be computed
using A-tilde,b-tilde. With residual r=b-tilde-A-tilde p-tilde,

\[
p-\widetilde p=A^{-1}
[(b-\widetilde b)+(\widetilde A-A)\widetilde p+r].
\]

Thus even a zero reported linear residual controls only the last term. Wrong
regions, curvature caps, approximate primal values and parameter right-hand
sides affect the other terms. For a positive-definite A with smallest
eigenvalue m, the corresponding norm bound has the factor 1/m. Relative
residuals additionally need conversion using the right-hand-side scale.

### The separate lower-solution system

For a lower objective f(z,theta)+g(z,theta), fix gamma and define

\[
T(z,\theta)=P_{\gamma,g_\theta}
(z-\gamma\nabla_z f(z,\theta)).
\]

At a differentiable fixed point, implicit differentiation requires solving

\[
[I-J(I-\gamma\nabla^2_{zz}f)]\,D_\theta z^*
=-\gamma J\nabla^2_{z\theta}f+
\partial_\theta P_{\gamma,g_\theta},
\]

where the partial derivative on the right holds the prox input fixed.
Existence of this formula as a local solution derivative requires a suitable
invertibility/regularity hypothesis, not merely existence of a JVP for P.
At nonsmooth points an appropriate nonsmooth implicit theorem is needed;
arbitrarily substituting branch matrices is insufficient. See
[Bolte, Le, Pauwels and Silveti-Falls](https://arxiv.org/abs/2106.04350).

An elementary failure is f=g=0. The prox is exactly the identity and has a
perfect derivative I, but every z is a lower minimizer and the fixed-point
matrix I-J is zero. No unique solution derivative has been specified.

Strong convexity of a lower problem is one useful route to uniqueness and
stability, together with appropriate smooth/manifold assumptions for classical
differentiation. A merely convex lower problem, or a nonconvex problem with
multiple stationary branches, does not supply those conditions automatically.

**Recommended contract.** The package supplies local prox sensitivities and
diagnostics; it cannot certify the hypergradient of an arbitrary enclosing
optimization problem without information about that problem.

## T8. Outer methods: sufficient conditions, endpoints, and inner error units

### Exact proximal gradient

For convex f with L-Lipschitz gradient and convex g, the standard constant
forward-backward convergence range is 0 < gamma < 2/L, assuming a minimizer
exists. The endpoint must not be included without extra conditions. For
f(x)=L x squared /2, g=0, gamma=2/L, the iteration is x-next=-x: it does not
converge from a nonzero starting point.

For nonconvex g and a global prox minimizer, the elementary comparison with
the old iterate plus the smooth descent lemma gives

\[
(f+g)(x_{k+1})\leq(f+g)(x_k)
-\left(\frac1{2\gamma}-\frac L2\right)
\|x_{k+1}-x_k\|^2.
\]

Hence gamma < 1/L provides sufficient descent. With a lower objective bound it
also yields square-summable steps. It does not by itself prove boundedness of
iterates, existence of cluster points, or convergence of the whole sequence.
The PALM/Kurdyka–Lojasiewicz theory cited in the API uses additional hypotheses,
including boundedness for its whole-sequence conclusion. See
[Bolte, Sabach and Teboulle](https://bolte.perso.math.cnrs.fr/BST2013.pdf).

### FISTA's usual step is a sufficient theorem condition

The standard convex FISTA guarantee uses gamma <= 1/L. Saying no FISTA
instance can converge above 1/L is stronger and false. For g=0 and
f(x)=L x squared /2, take gamma L=1.2. Writing q=1-gamma L=-0.2, the limiting
momentum recurrence has characteristic equation

\[
r^2-2qr+q=0,
\]

with roots about 0.28990 and -0.68990, both of modulus less than one.
Standard FISTA momentum tends to one, so its companion matrices tend to this
stable matrix. A matrix norm in which the limit has norm less than one also
makes all sufficiently late matrices contractions; the finite prefix does
not affect convergence. This proves convergence for this example above 1/L.
It does not extend the universal FISTA guarantee to that step.

### Translate inner gaps before invoking a rate theorem

For gamma=1/L, the conventional proximal subproblem
L||z-v|| squared /2 + g(z) is L times the package's normalized objective.
If Delta-k is the normalized feasible duality gap, an objective error bound
for that conventional subproblem is epsilon-k=L Delta-k. Therefore the
distance quantity sqrt(2 epsilon-k/L) equals sqrt(2 Delta-k), as stated in
the API. That normalization is correct.

Standard sufficient worst-case conditions for retaining the usual convex
rates are absolute error summability conditions of the form

\[
\sum_k\sqrt{\Delta_k}<\infty\quad\text{(basic)},\qquad
\sum_k k\sqrt{\Delta_k}<\infty\quad\text{(accelerated)}.
\]

These are sufficient conditions in the cited analysis, not necessary for
every favorable individual instance. See
[Schmidt, Le Roux and Bach](https://arxiv.org/abs/1109.2415).

A relative target ztol-k controls sqrt(2 Delta-k) relative to the prox input
norm. Translating a power-law ztol schedule directly into absolute
summability needs a bound on those input norms, or a separate argument
establishing one. A direct recipe with an explicit proof obligation is to
enforce sqrt(2 Delta-k) <= eta-k for an absolutely summable eta sequence
(weighted summability for acceleration). If expressed through relative
targets, divide eta-k by max(1, ||v-k||).

Finite-precision floors prevent arbitrarily tight asymptotic certification in
a fixed dtype, so the implemented finite stopping rule and the idealized
asymptotic theorem should be distinguished.

## T9. Spectral reductions are valid, but tie limits have hypotheses

For a singular-value function g(X)=h(sigma(X)), the standard convex reduction
requires h to be absolutely symmetric. For an eigenvalue function on
symmetric matrices, permutation symmetry is the corresponding condition.
The vector prox must respect that symmetry, including consistent choices at
ties. For separable nonconvex even penalties, a global scalar prox permits a
global spectral construction using the trace inequality, but differentiability
still fails at scalar jumps or switches.

For the sorted-weight nuclear penalty, nonincreasing nonnegative weights are
the convex case. Indeed,

\[
\sum_{i=1}^k w_i\sigma_i
=w_k\|X\|_*+
\sum_{j=1}^{k-1}(w_j-w_{j+1})\sum_{i=1}^j\sigma_i,
\]

a nonnegative combination of convex Ky Fan/nuclear norms. Arbitrarily weighted
singular coordinates do not have the same convex/symmetry contract.

At distinct spectra the symmetric off-diagonal derivative coefficient is
(f(s-i)-f(s-j))/(s-i-s-j). At an exact tie its limit needs differentiability
or an explicitly justified generalized selection. Replacing that quotient by
an average of endpoint derivatives at merely nearby values requires suitable
smoothness between the endpoints.

For a smooth scalar f with bounded third derivative, the trapezoidal-rule
error for that average is order |s-i-s-j| squared. Balancing it against a
cancellation error proportional to machine epsilon divided by the separation
explains a cube-root separation heuristic. It is not valid across a breakpoint,
and its constants depend on f and its scale. The prior audit's 1/3 versus 1/2
counterexample is exactly this missing-hypothesis case.

The general spectral differentiation framework is developed by
[Ding, Sun, Sun and Toh](https://arxiv.org/abs/1401.2269). It supports the
formulas under their hypotheses, not an unconditional replacement of distinct
spectra by ties.

## Proposed guarantee taxonomy

The package could make its scientific contract substantially clearer by
separating the following guarantees in documentation and diagnostics:

| Object | A defensible statement |
| --- | --- |
| Convex forward prox | Unique global minimizer; numerical error subject to its algorithm/certificate and dtype |
| Restricted weakly convex prox | Unique global minimizer under gamma rho < 1; sensitivity may exceed one |
| Discontinuous or critical selection | A specified minimizing branch; no blanket Lipschitz/Clarke promise at the transition |
| Exact smooth sensitivity | Classical derivative on a stable smooth piece/manifold, under the necessary identification assumptions |
| Generalized sensitivity | A specified accessible generalized-Jacobian selection under stated local-Lipschitz and compatibility hypotheses |
| Approximate sensitivity | Derivative of a stated approximate model, with separate primal/model/linear-solve diagnostics |
| Kink mask | A precisely described diagnostic; dual-dependent biactivity need not be intrinsic primal nondifferentiability |
| Composed/bilevel gradient | Requires additional chain-rule and lower-solution regularity assumptions supplied by the caller/problem |

The main theoretical strengths survive these qualifications. What should be
withdrawn is the unconditional implication from “we return a local JVP/VJP” to
“the result is an exact or Clarke-valid hypergradient for any surrounding
computation.” The report provides explicit moderate-scale counterexamples to
that implication rather than relying on numerical edge cases.

## Reproduction and review limits

Run the small checks with the existing environment:

```sh
uv run --no-sync python reports/theoretical_audit_2026_10_25.py
```

Results are captured in `reports/theoretical_audit_2026_10_25.json`. They verify
the restricted-composition example, critical log asymptotics, triangle flow
and kink mask, two-node gap construction, MCP Moreau residual, and singular
bilevel system. The script uses only public fastprox operations and NumPy.
The run used `uv --cache-dir /private/tmp/fastprox-audit-uv` for cache access.

No unchanged full test suite was rerun for this document-only theoretical
follow-up; its evidence consists of the derivations and these targeted checks.
The previous audit records the complete suite results. Formal verification of
every branch override, a proof of uniform numerical stability, and a complete
theorem for every external bilevel algorithm remain outside this review.

## Errata (2026-10-25)

These corrections come from checking the claims above against their sources
([specs for 1.0.1](math-audit-2026-10.md), item A1). None of them changes a
conclusion.

- **T9.** The Ky Fan identity should read
  w_k‖X‖_(k) + Σ_{j<k}(w_j − w_{j+1})‖X‖_(j), with ‖X‖_(j) the Ky Fan
  j-norm. The text's w_k‖X‖_* equals it only for k = min(m, n), which is the
  case of the full penalty, so convexity still follows.
- **T3.** The "narrower statement" is stronger than written. For a C¹ outer
  loss L, Clarke's chain rule (*Optimization and Nonsmooth Analysis*,
  Thm 2.6.6) gives ∂(L∘P)(x) = ∇L(P(x))ᵀ∂P(x), with equality, for every
  element of Clarke's Jacobian of P, not only for accessible B-Jacobians.
- **T5.** The scaling was omitted. The prox is I − Π onto γ·Kᵀ(B), with B the
  box of the edge weights. A strictly interior flow is sufficient for
  differentiability, not necessary.
- **T7.** "Flat constraints locally stable" means strict complementarity: a
  dual exists with ‖q_j‖ < 1 on every flat group (De los Reyes and Meyer,
  Cor. 2.10; Villacís, PhD thesis, Thm 4.6).
- **T8.** The nonconvex descent inequality also needs γ below the penalty's
  prox-boundedness threshold (Themelis, Stella and Patrinos, *SIAM J. Optim.*
  2018, Lemma 2.1). Every bounded-below penalty in the library meets it.
