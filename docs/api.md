# API reference

Covers 0.1 and, marked *since 0.2*, what has landed for 0.2 so far.

fastprox computes proximal operators and projections,

    prox_{γg}(x) = argmin_z  g(z) + ‖z − x‖² / (2γ),

along one axis of a float32 or float64 array, batched over the other axes,
together with their Jacobian-vector and vector-Jacobian products. This page
covers the C API in [`include/fastprox.h`](../include/fastprox.h) and the
Python package. The design rationale is in [spec.md](spec.md).

- [Operators](#operators)
- [C API](#c-api)
- [Python](#python)
- [Conventions shared by both](#conventions)

## Operators

Every operator has a parameter vector θ. It is what `fp_vjp` returns
gradients for and what `fp_jvp` perturbs; its length is `fp_op_nparams(op)`.
"Per entry" means a vector with the length of the operator axis, n.

| Operator | g(z) or set | C constructor | Python | θ |
| --- | --- | --- | --- | --- |
| Soft thresholding | λ Σ wᵢ \|zᵢ\| | `fp_op_soft(λ, w)` | `Soft(lam, weights)`, `soft_threshold` | w (scalar or per entry) |
| Elastic net | l₁‖z‖₁ + (l₂/2)‖z‖² | `fp_op_elastic_net(l1, l2)` | `ElasticNet`, `elastic_net` | (l₁, l₂) |
| Huber | λ Σ h_δ(zᵢ), h_δ(t) = t²/(2δ) for \|t\| ≤ δ, \|t\| − δ/2 otherwise | `fp_op_huber(λ, δ)` | `Huber`, `huber` | (λ, δ) |
| Group ℓ2 | λ Σ_g c_g ‖z_g‖₂ | `fp_op_group_l2(λ, n, groups, G, c)` | `GroupL2(lam, groups, group_weights)`, `group_soft_threshold`, `prox_l2` | c (scalar or one per group) |
| Box | lo ≤ z ≤ hi | `fp_op_box(lo, hi, lower, upper)` | `Box(lower, upper)`, `project_box` | (lower…, upper…) |
| Nonnegative orthant | z ≥ 0 | `fp_op_nonneg()` | `NonNeg`, `project_nonneg` | none |
| ℓ2 ball | ‖z‖₂ ≤ r | `fp_op_l2_ball(r)` | `L2Ball`, `project_l2_ball` | (r) |
| Simplex | z ≥ 0, Σ z = r | `fp_op_simplex(r)` | `Simplex`, `project_simplex` | (r) |
| Weighted ℓ1 ball | Σ wᵢ\|zᵢ\| ≤ r | `fp_op_l1_ball(r, w)` | `L1Ball(radius, weights)`, `project_l1_ball` | (r) |
| ℓ∞ norm | λ‖z‖∞ | `fp_op_linf(λ)` | `LInf`, `prox_linf` | (λ) |
| Maximum (*since 0.2*) | λ maxᵢ zᵢ | `fp_op_max(λ)` | `Max(lam)`, `prox_max` | (λ) |
| Sum of the k largest (*since 0.2*) | λ Σ of the k largest zᵢ (or \|zᵢ\|) | `fp_op_sum_largest(λ, k, absolute)` | `SumLargest(lam, k, absolute)`, `prox_sum_largest` | (λ) |
| Hyperplane ∩ box (*since 0.2*) | aᵀz = b, lo ≤ z ≤ hi | `fp_op_hyperplane(a, b, lo, hi, lower, upper, 0)` | `Hyperplane(a, b, lower, upper)`, `project_hyperplane` | (b, a…, lower…, upper…) |
| Half-space ∩ box (*since 0.2*) | aᵀz ≤ b, lo ≤ z ≤ hi | `fp_op_hyperplane(a, b, lo, hi, lower, upper, 1)` | `HalfSpace(a, b, lower, upper)`, `project_halfspace` | (b, a…, lower…, upper…) |
| Second-order cone (*since 0.2*) | ‖(z₂, …, zₙ)‖₂ ≤ z₁ | `fp_op_soc()` | `SOC`, `project_soc` | none |
| Log barrier (*since 0.2*) | −λ Σ log zᵢ | `fp_op_log_barrier(λ)` | `LogBarrier(lam)`, `prox_log_barrier` | (λ) |
| ℓ1 over a box (*since 0.2*) | λ Σ wᵢ \|zᵢ\| on lo ≤ z ≤ hi | `fp_op_l1_box(λ, w, lo, hi, lower, upper)` | `L1Box(lam, lower, upper, weights)`, `prox_l1_box` | (w…, lower…, upper…) |
| Group Huber (*since 0.2*) | λ Σ_g c_g H_δ(‖z_g‖₂), H_δ as for Huber | `fp_op_group_huber(λ, δ, n, groups, G, c)` | `GroupHuber(lam, delta, groups, group_weights)`, `group_huber` | (λ, δ, c…) |
| Group SCAD, group MCP (*since 0.2*) | Σ_g c_g ρ(‖z_g‖₂), ρ the SCAD or MCP penalty | `fp_op_group_scad(λ, a, …)`, `fp_op_group_mcp(λ, b, …)` | `GroupSCAD`, `GroupMCP`, `group_scad`, `group_mcp` | (λ, a or b, c…) |
| Monotone cone | z₁ ≤ … ≤ zₙ (or ≥), weighted least squares | `fp_op_isotonic(w, increasing)` | `Isotonic(weights, increasing)`, `isotonic` | w (empty if none) |
| 1D total variation | Σ wᵢ \|zᵢ₊₁ − zᵢ\| | `fp_op_tv1d(w, method)` | `TV1D(weights, method)`, `tv1d` | w (scalar or n − 1 edges) |
| ℓ0 (*since 0.2*) | λ Σ wᵢ [zᵢ ≠ 0] | `fp_op_l0(λ, w)` | `L0(lam, weights)`, `hard_threshold` | w (scalar or per entry) |
| ℓq, 0 < q < 1 (*since 0.2*) | λ Σ wᵢ \|zᵢ\|^q | `fp_op_lq(λ, q, w)` | `Lq(lam, q, weights)`, `lq` | w (scalar or per entry) |
| SCAD (*since 0.2*) | Fan and Li's penalty, threshold λ, concavity a | `fp_op_scad(λ, a)` | `SCAD(lam, a)`, `scad` | (λ, a) |
| MCP (*since 0.2*) | Zhang's minimax concave penalty, threshold λ, concavity b | `fp_op_mcp(λ, b)` | `MCP(lam, b)`, `mcp` | (λ, b) |
| Log penalty (*since 0.2*) | λ Σ log(1 + \|zᵢ\|/ε) | `fp_op_log(λ, ε)` | `LogPenalty(lam, eps)`, `prox_log` | (λ, ε) |
| Anisotropic TV, 2D and 3D (*since 0.2*) | Σₐ wₐ Σ \|z_{p+eₐ} − z_p\| over the last 2 or 3 axes | `fp_op_tv_aniso(ndim, w)` | `TVAniso(ndim, weights)`, `tv_aniso` | (w₀, …, w_{ndim−1}) |
| Isotropic TV, 2D and 3D (*since 0.2*) | Σ_p ‖(w₀ D₀z, …, w_{ndim−1} D_{ndim−1}z)_p‖₂, forward differences | `fp_op_tv_iso(ndim, w)` | `TVIso(ndim, weights)`, `tv_iso` | (w₀, …, w_{ndim−1}) |

Notes:

- **Step versus weights.** γ multiplies the whole of g, so soft thresholding
  uses the threshold γλwᵢ and 1D TV the edge thresholds γwᵢ. Projections ignore γ.
- **Weighted ℓ1 ball:** weights must be positive with min/max ≥ 1e-100 (only
  their ratios matter; smaller ratios are rejected with `FP_EPARAM`).
- **Maximum and the sum of the k largest** are support functions (of the
  simplex and of {Σ y = k, 0 ≤ y ≤ 1}, or {‖y‖₁ ≤ k, ‖y‖∞ ≤ 1} for absolute
  values), so their prox is x − P(x) with P the projection onto the set
  scaled by γλ. The maximum clips x from above at the simplex threshold
  (min(x, τ), the cost of a simplex projection). The sum of the k largest
  entries needs k ≤ n; for absolute values any k ≥ 1 works, and k ≥ n gives
  λ‖z‖₁.
- **Hyperplane ∩ box.** a is a scalar or per entry (`NULL`/`None`: all ones),
  the box as for `Box`, unbounded by default; a = 1 with the box [0, 1] is the
  capped simplex. The set is checked at each call for the axis length (b must
  lie in the range of aᵀz over the box, or above its lower end for the
  half-space), and `FP_EPARAM` reports an empty one. Since only the ratios of
  (a, b) matter, nonzero |aᵢ| / max |a| must be at least 1e-100. The
  half-space keeps the box projection when it meets aᵀz ≤ b.
- **Second-order cone.** t is the first entry of the row, as in conic
  solvers (Beck's Example 6.37 puts it last). Rows of length 1 give the
  projection onto t ≥ 0.
- **Log barrier** needs λ > 0. The prox (x + √(x² + 4γλ))/2 is evaluated
  without cancellation for x < 0 and without overflow for huge |x| or γλ.
- **ℓ1 over a box** is clip(soft(x, γλw), lo, hi), for any interval (Beck,
  Example 6.23, states the box [−α, α]); weights as for soft thresholding and
  bounds as for `Box`. `grads.lam` is also provided in Python.
- **Group penalties by norm composition.** Group Huber, SCAD and MCP scale each
  group by φ(‖x_g‖)/‖x_g‖, with φ the scalar prox at the step γc_g (Beck,
  Theorem 6.18); groups and weights are as for group ℓ2, and a one-entry
  group is the scalar map. SCAD needs a > 1 + γ max c and MCP b > γ max c,
  checked at each call (`FP_EPARAM`). Group SCAD and MCP are nonconvex, but
  their prox is single-valued under these conditions, so there is no tie rule.
- **Isotonic regression** minimises Σ wᵢ(zᵢ − xᵢ)², which is the Euclidean
  projection only for unit weights.
- **1D TV methods:** `FP_TV1D_AUTO` (default) runs Condat's direct
  algorithm (generalised to per-edge weights) and falls back to the taut
  string when it backtracks too much (more than 3n + 1024 rescanned steps,
  as on ramps). `FP_TV1D_CONDAT` forces Condat's algorithm (quadratic on
  adversarial inputs), `FP_TV1D_TAUT_STRING` forces the taut string. Zero
  weights split a row into independent pieces; when no edge can be active
  the result is the constant mean.
- **Kinks.** At a point where the prox is not differentiable the derivative
  is that of the structure the forward pass selected (see "Derivatives at
  kinks" below).

### Nonconvex pointwise maps (*since 0.2*)

ℓ0, ℓq, SCAD, MCP and the log penalty act entry by entry, like soft
thresholding. Their prox is a global minimiser of the nonconvex scalar
problem ½(z − x)² + γψ(z):
- **ℓ0** is hard thresholding at √(2γλwᵢ).
- **ℓq.** Zero below the threshold τ = β + γλwᵢ q β^{q−1}, with β = (2γλwᵢ(1 − q))^{1/(2−q)}
  the size of the jump. Above it, the root of z + γλwᵢ q z^{q−1} = |x|: in
  closed form for q = 1/2 and 2/3 (a cubic and a quartic), by Newton's
  method from |x| otherwise (monotone on that branch).
- **SCAD** needs a > 1 + γ and **MCP** b > γ; below that the prox is not
  single-valued and the call returns `FP_EPARAM` (with `fp_prox_conj` the
  condition applies to its step 1/σ). Both are continuous and piecewise
  linear; with γ = 1 they are Fan and Li's and Zhang's thresholding rules.
- **Log penalty.** The positive root of a quadratic, kept when it beats 0;
  for γλ ≤ ε² the problem is convex.
- **Ties.** Where two minimisers exist (ℓ0 and ℓq at the threshold, the log
  penalty when both objective values are equal), `fp_options.tie` chooses:
  `FP_TIE_NONZERO` (default) or `FP_TIE_ZERO`, in Python `tie="nonzero"` or
  `"zero"`. `fp_info.ties` (Python: `ctx.info.ties`) counts the entries that
  sat on a tie, the same for any thread count. A tie means the input is
  exactly the threshold computed in double precision; a threshold that
  falls between two doubles has no tie.
- **Derivatives** are those of the selected branch: 0 off the support, and
  1/(1 + γλψ''(z)) on it, from the scalar equation z + γλψ'(z) = x, with the
  γ and parameter derivatives from the same equation. ℓ0 is piecewise
  constant in γ and w.
- **Conjugates.** `fp_prox_conj` returns y − σ prox_{g/σ}(y/σ). For these
  nonconvex g that is not the prox of g*: the Moreau identity holds only for
  convex g.
- Every decision is made in double precision, also for float32 inputs, so
  float32 and float64 pick the same branch.

### Image operators (*since 0.2*)

Anisotropic and isotropic TV act on the **last** 2 or 3 axes of the array, an
image or a volume, and are batched over the leading axes; `fp_options.axis`
is ignored. What follows describes anisotropic TV; isotropic TV, which shares
the gap stopping, `FP_ENOTCONV`, warm starts and special values, is
described after it.
Each image is solved independently, so a batch gives the same result as
image-by-image calls, for any thread count.

- **Algorithms.** Dual methods whose steps are exact 1D TV proxes along one
  axis, applied to every fiber of the image:
  - with two active axes, the accelerated dual block proximal gradient
    method (Beck, *First-Order Methods in Optimization*, 2017, §12.5.4,
    Thm 12.19; Chambolle and Pock, SMAI J. Comput. Math. 1, 2015). It
    alternates rows and columns at an O(1/k²) rate;
  - with three, the fast dual proximal gradient method for three blocks
    (Beck, Algorithm 9).
  An axis of extent 1 or weight 0 drops out. With a single active axis the
  prox is the exact 1D TV, with gap 0.
- **Gap and stopping.** Every iterate carries duals q_a, one per axis and
  each feasible for its term, with x − z = Σₐ q_a. The duality gap of the prox
  problem at (z, q) bounds the error: ‖z − z*‖² ≤ 2·gap. Each image stops
  when its gap is at most `tol`·‖x‖²/2 (default 1e-6), or after `max_iter`
  iterations (default 1000), in which case the call returns `FP_ENOTCONV`
  with the last iterate written. `fp_info.gap` is the sum over the batch,
  `fp_info.iters` the maximum.
- **Polish.** At the end the solution's constant regions (edges with a jump
  below a threshold set by the certified error) are identified. Their
  closed-form values are then recomputed from the input and the signs of the
  jumps on their boundary. The polished point is kept when its gap, with
  the same dual, is no larger, which makes the output exact to rounding
  when the regions are identified correctly.
- **Warm starts.** The dual of the last call stays in the workspace, shaped
  `(ndim, *x.shape)`. `fp_workspace_get_dual` copies it out, and
  `fp_workspace_set_dual` seeds the next call with a dual, for example one
  from a nearby problem. After `fp_workspace_set_warm(ws, 1)`, every call
  starts from the dual of the previous call, as in a PDHG or bilevel loop.
  Without either, each call starts from zero.
- **Derivatives** are exact for the identified regions: averaging over each
  region, with γ and weight terms from the jump signs on its boundary. They
  cost O(n), with no linear solve.
- **Special values.** A NaN anywhere in an image makes that whole image NaN
  (the rest of the batch is unaffected); ±Inf is rejected with `FP_EINVAL`.
  Images are limited to 2³¹ − 1 entries.

**Isotropic TV.** The weights scale each axis inside the norm (for example
voxel spacing); differences past the last entry are zero, as in Beck's
TV_I.
- **Algorithm.** The fast dual proximal gradient method, that is FISTA on the
  dual (Beck and Teboulle, IEEE TIP 18, 2009; Beck 2017, §12.4.4): each
  iteration is a divergence, a gradient and a pixelwise projection onto the
  ball of radius γ, with step 1/L, L = 4 Σₐ wₐ² (8 in 2D, 12 in 3D), and the
  primal taken from the last dual iterate. A single active axis is the exact
  1D TV. It is rescaled internally by the largest weight, so tiny or huge
  weights and steps stay finite, and an axis whose weight is below 1e-100 of
  the largest drops out.
- **Dual.** u of shape `(ndim, *x.shape)` with ‖u_p‖₂ ≤ γ, zero past the last
  entry along each axis, and z = x − Σₐ wₐ Dₐᵀuₐ. A warm start projects the
  given dual onto the current ball, so a dual from another γ can be reused.
  The gap is Σ_p (γ‖g_p‖ − ⟨g_p, u_p⟩) with g = W D z, each term ≥ 0.
- **Derivatives** come from implicit differentiation of the optimality
  conditions. Entries with a vanishing gradient (below a threshold set by
  the certified error) keep z constant across their edges, which reduces the
  unknowns to one per flat region; elsewhere the dual turns on the sphere
  (H_p = γ/‖g_p‖ (I − n nᵀ)). The resulting symmetric positive definite system
  is solved by conjugate gradients, to a relative residual of 1e-10 by
  default: `fp_workspace_set_deriv_tol(ws, tol, max_iter)` changes it and
  `fp_workspace_deriv_info(ws, &residual, &iters)` reports the last solve
  (in Python, `jvp`/`vjp` take `tol` and `max_iter`, and `ctx.deriv` holds
  the residual and iterations). No polish: isotropic regions have no
  closed-form values.
- FISTA converges like 1/k² in the dual, so very tight tolerances take
  many iterations (about 1/√tol); the default `tol` of 1e-6 is cheap.

### Derivatives at kinks

Every operator is piecewise smooth. At a point where two pieces meet,
`fp_jvp` and `fp_vjp` use the Jacobian of one of them, chosen as follows:

| Operator | At the kink | Jacobian used |
| --- | --- | --- |
| soft thresholding, elastic net | \|xᵢ\| equal to the threshold | thresholded: 0 |
| Huber | \|xᵢ\| = δ + γλ | inner (quadratic) branch: δ/(δ + γλ) |
| box, nonnegative orthant | xᵢ on a bound | interior: 1 |
| group ℓ2, ℓ2 norm | ‖x_g‖ equal to the threshold | group set to zero: 0 |
| ℓ2 ball | ‖x‖ = r | inside: identity |
| simplex, ℓ1 ball | xᵢ (or \|xᵢ\|/wᵢ) equal to the threshold τ | entry off the support |
| ℓ1 ball | weighted ℓ1 norm of x equal to r | inside: identity |
| ℓ∞ norm | \|xᵢ\| = τ | entry not clipped: 1 |
| ℓ∞ norm | ‖x‖₁ = γλ | z = 0: 0 |
| maximum | xᵢ = τ | entry not clipped: 1 |
| sum of the k largest | xᵢ − μ on 0 or γλ (an end of the scaled cap) | entry at the shared level μ |
| hyperplane, half-space ∩ box | xᵢ − μaᵢ on a bound | interior (free) |
| half-space ∩ box | aᵀP_box(x) = b | box projection (constraint inactive) |
| second-order cone | ‖y‖ = t (on the cone), or the origin | inside: identity |
| second-order cone | ‖y‖ = −t > 0 (on the polar cone) | z = 0: 0 |
| log barrier | none for γλ > 0; at γ = 0, xᵢ = 0 | 1 |
| ℓ1 over a box | \|xᵢ\| equal to the threshold; soft(xᵢ) on a bound | thresholded: 0; interior (soft's Jacobian) |
| group Huber, SCAD, MCP | ‖x_g‖ at a breakpoint of the scalar map | the scalar map's rule (Huber: inner branch; SCAD, MCP: the branch below) |
| group Huber, SCAD, MCP | x_g = 0 | φ′(0) I, the limit of φ(s)/s |
| isotonic regression | adjacent blocks with equal means | blocks not pooled (pooling happens only on a strict violation) |
| 1D TV | adjacent segments with equal values | the segments the solver returned, each averaged on its own |
| anisotropic TV | adjacent regions with equal values | the regions identified at the end (jumps below the threshold merged), each averaged on its own |
| ℓ0, ℓq, log penalty | \|xᵢ\| on the threshold (two minimisers) | the branch of the tie rule: nonzero by default, 0 with `FP_TIE_ZERO` |
| SCAD, MCP | \|xᵢ\| at a breakpoint | the branch below it (at γλ: thresholded) |
| isotropic TV | an entry with a vanishing gradient | flat: z held constant across its edges (the gradient below the threshold counts as zero) |

The ℓ∞ rule is the reverse of the soft-thresholding rule. The ℓ∞ prox is
x − γλ P(x/(γλ)), with P the projection onto the ℓ1 ball, so an entry kept
off the support of P is left unclipped. `fp_prox_conj` flips its operator's
rule in the same way.

**Why these are valid derivatives.** Every choice in the table is the
Jacobian of one of the smooth pieces that meet at the point. So it belongs
to the B-subdifferential, the set of limits of nearby Jacobians, and hence
to Clarke's generalised Jacobian:
- S. Scholtes, *Introduction to Piecewise Differentiable Equations*,
  Springer 2012, Prop. 4.3.1, and Prop. 2.4.3–2.4.5 for projections onto
  polyhedra (simplex, ℓ1 ball, monotone cone, and 1D TV through the Moreau
  identity).
- C. Clason and T. Valkonen, *Introduction to Nonsmooth Analysis and
  Optimization*, Ex. 14.10. For soft thresholding at |x| = τ, both 0 and 1
  are such limits, and neither is canonical.

The derivatives in x, γ and the parameters θ all come from the same saved
structure. Together they are therefore the Jacobian of one piece in the
joint variables (x, γ, θ). As a result, a VJP through a smooth outer loss
is a Clarke subgradient of the composition.

At a kink, the JVP is not the one-sided directional derivative for
directions that leave the chosen piece. Finite-difference tests therefore
avoid kinks.

## C API

### Arrays

```c
typedef struct {
    void          *data;
    fp_dtype       dtype;    /* FP_F32 or FP_F64 */
    int            ndim;     /* 0 (scalar) to FP_MAX_NDIM */
    const int64_t *shape;
    const int64_t *strides;  /* in elements, may be negative; NULL = C order */
    fp_device      device;   /* FP_DEVICE_CPU */
} fp_array;
```

The library never frees `data`. `out` may be the input array itself; other
overlaps between input and output are undefined.

### Lifecycle

```c
fp_op *op;
fp_status s = fp_op_tv1d(NULL, FP_TV1D_AUTO, &op);   /* unit weights */
if (s != FP_OK) fprintf(stderr, "%s\n", fp_strerror(s));

fp_options opt;
fp_options_init(&opt);     /* axis -1, threads 0 (default count), ... */
opt.axis = 1;

s = fp_prox(op, 0.5, &x, &z, &opt, NULL, NULL);       /* one-shot */
fp_op_free(op);
```

Constructors copy their parameter arrays (any dtype, one-dimensional, any
stride), validate them (`FP_EPARAM` for negative weights, a negative radius,
and so on) and write the handle through the last argument; on failure the
handle is set to NULL. Operators are immutable and can be shared between
threads.

### Workspaces and derivatives

```c
fp_workspace *ws;
opt.keep_structure = 1;
fp_workspace_create(op, &x, &opt, &ws);    /* sized for x's shape, dtype, axis */

fp_prox(op, gamma, &x, &z, &opt, ws, NULL);            /* saves structure */
fp_jvp(op, ws, &dx, dgamma, &dtheta, &dz);             /* dx, dtheta may be NULL */
double ggamma;
fp_vjp(op, ws, &g, &gx, &ggamma, &gtheta);             /* ggamma, gtheta may be NULL */
fp_workspace_free(ws);
```

- With a workspace, `fp_prox`, `fp_prox_conj`, `fp_jvp` and `fp_vjp` do not
  allocate. Without one, `fp_prox` allocates scratch for the call.
- `fp_jvp` computes J_x dx + (∂z/∂γ) dγ + (∂z/∂θ) dθ; `fp_vjp` computes
  J_xᵀ g, ⟨g, ∂z/∂γ⟩ and (∂z/∂θ)ᵀ g, summed over the batch. `dtheta` and
  `gtheta` are vectors of length `fp_op_nparams(op)`.
- Derivatives use the structure saved by the last prox call that had
  `keep_structure` set; without it they return `FP_EINVAL`.
- After `fp_prox_conj`, derivatives are those of the conjugate prox and γ
  reads as σ.
- A workspace is used by one call at a time; use one per thread.

### Warm starts and the dual (*since 0.2*)

```c
fp_workspace_get_dual(ws, &q);       /* q: shape (ndim, *x.shape), any float dtype */
fp_workspace_set_dual(ws, &q);       /* the next call starts from q */
fp_workspace_set_warm(ws, 1);        /* every call starts from the previous dual */
```

Only image operators have a dual; for other operators these return
`FP_EINVAL`, as does `get` before any call. `set` rejects non-finite values.

```c
fp_workspace_set_deriv_tol(ws, 1e-8, 500);   /* CG of isotropic TV's fp_jvp/fp_vjp */
double residual; int iters;
fp_workspace_deriv_info(ws, &residual, &iters);  /* worst over the batch, last call */
```

### Conjugates

`fp_prox_conj(op, σ, y, out, ...)` computes prox_{σg*}(y) =
y − σ prox_{g/σ}(y/σ). σ = 0 returns y.

### Errors

| Status | Meaning |
| --- | --- |
| `FP_OK` | success |
| `FP_EINVAL` | invalid argument, ±Inf in the input of an operator where it is undefined, missing saved structure, mismatched workspace |
| `FP_EDTYPE` | dtype other than float32/float64, or mismatched dtypes |
| `FP_ESHAPE` | mismatched shapes or parameter lengths |
| `FP_ENOMEM` | out of memory |
| `FP_EPARAM` | invalid parameter or step (negative γ, negative radius, …), or a constraint set that is empty for the axis length (hyperplane ∩ box, the sum of k > n largest entries) |
| `FP_EINTERNAL` | any other internal error |

No function prints, aborts or keeps global error state. NaN in the input
gives NaN in the affected output: the entry for elementwise operators, the
row for the others.

### Threads and determinism

`opt.threads` sets the number of threads (0 = `fp_default_threads()`, 1 = serial).
The default is the environment variable `FASTPROX_NUM_THREADS`, else the first
number in `OMP_NUM_THREADS`, else the CPUs the process may run on; it is read
once.
Results, including parameter gradients, are bit-identical for any thread
count. Rows run on fastprox's own thread pool, not OpenMP, so no OpenMP
runtime is linked. The pool's threads spin for about 50 µs after a call before
sleeping. When several threads call fastprox at once, one call uses the pool
and the others run serially, which avoids oversubscription; a child process
after `fork()` builds its own pool. Builds with `FASTPROX_THREADS=OFF` are
serial and `fp_default_threads()` returns 1.

### Building and linking

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build && cmake --install build --prefix /usr/local
```

Then `find_package(fastprox)` and link `fastprox::fastprox`, or use
`pkg-config --cflags --libs fastprox`. Options: `FASTPROX_THREADS`,
`BUILD_SHARED_LIBS`, `FASTPROX_BUILD_TESTS`, `FASTPROX_SANITIZE`,
`FASTPROX_FUZZ`.

## Python

```python
import numpy as np
import fastprox as fp

x = np.random.default_rng(0).normal(size=(64, 1000))

# One-shot functions: NumPy in, NumPy out, along an axis.
z = fp.tv1d(x, gamma=0.5)                       # weights=..., method=..., axis=-1
p = fp.project_simplex(x, radius=1.0, axis=-1)
s = fp.soft_threshold(x, lam=0.1, gamma=1.0)

# Operators with derivatives.
op = fp.TV1D(weights=0.8)
z, ctx = op.prox(x, 0.5, keep=True)             # ctx: saved structure
dz = op.jvp(ctx, dx=np.ones_like(x), dgamma=0.0, dparams={"weights": 1.0})
gx, grads = op.vjp(ctx, g=np.ones_like(x))      # grads.gamma, grads.weights, grads.params
y = op.prox_conj(x, sigma=2.0)                  # Moreau identity

# Nonconvex maps (since 0.2): ties at thresholds follow tie= and are counted.
zq = fp.lq(x, lam=0.1, q=0.5, gamma=1.0, tie="nonzero")
zs = fp.scad(x, lam=0.2, a=3.7, gamma=1.0)          # needs a > 1 + gamma
z0, ctx = fp.L0(0.3).prox(x, 1.0, keep=True, tie="zero")
ctx.info.ties                                       # entries exactly on the threshold

# Support functions and planes (since 0.2), along the axis.
zm = fp.prox_max(x, lam=0.5)                         # min(x, tau)
zk = fp.prox_sum_largest(x, lam=0.5, k=3)            # absolute=True for |x|
w = np.linspace(0.5, 1.5, 1000)                     # one a_i per entry of the axis
zp = fp.project_hyperplane(x, a=w, b=100.0, lower=0.0, upper=1.0)
zh = fp.project_halfspace(x, a=w, b=1.0)

# Cones, barriers and group penalties (since 0.2).
zc = fp.project_soc(x)                               # t = x[..., 0]
zb = fp.prox_log_barrier(x, lam=0.1)                 # > 0
zl = fp.prox_l1_box(x, lam=0.2, lower=-1.0, upper=1.0)
groups = np.arange(1000) // 10                       # one group id per entry of the axis
zg = fp.group_scad(x, lam=1.0, a=3.7, groups=groups)

# Image operators (since 0.2): the last 2 or 3 axes, with a certificate.
img = np.random.default_rng(1).normal(size=(256, 256))
z, info = fp.tv_aniso(img, gamma=0.2, tol=1e-6)   # info.gap, info.iters, info.converged, info.dual
z, info = fp.tv_aniso(img, gamma=0.25, warm_start=info.dual)
iso = fp.TVIso(ndim=2)
v, ctx = iso.prox(img, 0.2, keep=True, tol=1e-8, max_iter=100000)
gx, grads = iso.vjp(ctx, np.ones_like(img), tol=1e-10)   # conjugate gradients
residual, cg_iters = ctx.deriv
tv = fp.TVAniso(ndim=3, weights=[1.0, 1.0, 0.5])
v, ctx = tv.prox(np.zeros((32, 32, 16)), 0.3, keep=True)   # ctx.info is an IterInfo
v, ctx = tv.prox(np.zeros((32, 32, 16)), 0.3, keep=True, ctx=ctx, warm=True)
```

- **Inputs** must be float32 or float64 NumPy arrays; other dtypes raise
  `TypeError` unless `copy=True` is passed. Lists are converted to float64.
  Strided and reversed views are used without copying.
- **Common keywords:** `axis` (default −1), `out` (preallocated output, may
  be the input), `threads` (0 = `fp.default_threads()`).
- **`keep=True`** returns `(z, ctx)`. Pass `ctx=ctx` to a later `prox` call
  with the same shape to reuse its memory. `ctx.info` holds the certificate
  (`gap`, `iters`, `ties`).
- **`dparams`** is a flat vector of length `op.nparams` or a dict by
  parameter name; **`grads`** exposes the same names as attributes.
  `Soft` also reports `grads.lam`.
- **Iterative operators** (`TVAniso`): `solve` and the one-shot function
  return `(z, info)`, where `info` is an `IterInfo`: `gap`, `iters`, `ties`,
  `converged` (False when `max_iter` was reached; this is not an error) and
  `dual`. `prox` and `prox_conj` take `tol`, `max_iter`, `warm_start=dual` and
  `warm=True` (start from the dual of the previous call on `ctx`).
- **Errors** map to `ValueError` (invalid arguments, parameters, shapes),
  `TypeError` (dtype), `MemoryError` and `RuntimeError`. The GIL is released
  during every computation.

## Conventions

- Derivatives are an element of the generalized Jacobian, chosen by the
  structure of the forward pass (see "Kinks" above).
- Exact operators report `gap = 0`; `iters` and `ties` are zero for every
  0.1 operator. Iterative operators report their duality gap and iterations,
  and the nonconvex maps the number of entries on a tie.
- float32 inputs are computed with double-precision accumulators wherever a
  sum spans a row.
