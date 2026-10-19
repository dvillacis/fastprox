# API reference

Covers the current release. Features added after 0.1 are marked with the
version that introduced them (*since 0.3*).

fastprox computes proximal operators and projections,

$$\operatorname{prox}_{\gamma g}(x) = \operatorname*{argmin}_z \; g(z) + \frac{1}{2\gamma} \lVert z - x \rVert_2^2,$$

along one axis of a float32 or float64 array, batched over the other axes,
together with their Jacobian-vector and vector-Jacobian products. This page
covers the C API in [`include/fastprox.h`](https://github.com/dvillacis/fastprox/blob/master/include/fastprox.h) and the
Python package. The design rationale is in [spec.md](spec.md).

- [Operators](#operators)
- [C API](#c-api)
- [Python](#python)
- [Conventions shared by both](#conventions)

## Operators

Every operator has a parameter vector $\theta$. It is what `fp_vjp` returns
gradients for and what `fp_jvp` perturbs; its length is `fp_op_nparams(op)`.
"Per entry" means a vector with the length of the operator axis, $n$.

| Operator | g(z) or set | C constructor | Python | θ |
| --- | --- | --- | --- | --- |
| Soft thresholding | $\lambda \sum_i w_i \lvert z_i \rvert$ | `fp_op_soft(λ, w)` | `Soft(lam, weights)`, `soft_threshold` | w (scalar or per entry) |
| Elastic net | $\ell_1 \lVert z \rVert_1 + \tfrac{\ell_2}{2} \lVert z \rVert_2^2$ | `fp_op_elastic_net(l1, l2)` | `ElasticNet`, `elastic_net` | (l₁, l₂) |
| Huber | $\lambda \sum_i h_\delta(z_i)$, $h_\delta(t) = t^2 / (2\delta)$ for $\lvert t \rvert \le \delta$, $\lvert t \rvert - \delta/2$ otherwise | `fp_op_huber(λ, δ)` | `Huber`, `huber` | (λ, δ) |
| Group ℓ2 | $\lambda \sum_g c_g \lVert z_g \rVert_2$ | `fp_op_group_l2(λ, n, groups, G, c)` | `GroupL2(lam, groups, group_weights)`, `group_soft_threshold`, `prox_l2` | c (scalar or one per group) |
| Tree-structured group ℓ2 (*since 0.2*) | $\ell_1 \lVert z \rVert_1 + \sum_g \eta_g \lVert z_g \rVert_2$, groups nested or disjoint | `fp_op_tree_l2(l1, G, ptr, idx, η)` | `TreeL2(groups, group_weights, l1)`, `TreeL2.sparse_group`, `tree_soft_threshold`, `sparse_group_soft_threshold` | (l₁, η…) |
| Box | $\mathrm{lo} \le z \le \mathrm{hi}$ | `fp_op_box(lo, hi, lower, upper)` | `Box(lower, upper)`, `project_box` | (lower…, upper…) |
| Nonnegative orthant | $z \ge 0$ | `fp_op_nonneg()` | `NonNeg`, `project_nonneg` | none |
| ℓ2 ball | $\lVert z \rVert_2 \le r$ | `fp_op_l2_ball(r)` | `L2Ball`, `project_l2_ball` | (r) |
| Simplex | $z \ge 0$, $\sum_i z_i = r$ | `fp_op_simplex(r)` | `Simplex`, `project_simplex` | (r) |
| Weighted ℓ1 ball | $\sum_i w_i \lvert z_i \rvert \le r$ | `fp_op_l1_ball(r, w)` | `L1Ball(radius, weights)`, `project_l1_ball` | (r) |
| ℓ∞ norm | $\lambda \lVert z \rVert_\infty$ | `fp_op_linf(λ)` | `LInf`, `prox_linf` | (λ) |
| Maximum (*since 0.2*) | $\lambda \max_i z_i$ | `fp_op_max(λ)` | `Max(lam)`, `prox_max` | (λ) |
| Sum of the k largest (*since 0.2*) | $\lambda$ times the sum of the $k$ largest $z_i$ (or $\lvert z_i \rvert$) | `fp_op_sum_largest(λ, k, absolute)` | `SumLargest(lam, k, absolute)`, `prox_sum_largest` | (λ) |
| Hyperplane ∩ box (*since 0.2*) | $a^\top z = b$, $\mathrm{lo} \le z \le \mathrm{hi}$ | `fp_op_hyperplane(a, b, lo, hi, lower, upper, 0)` | `Hyperplane(a, b, lower, upper)`, `project_hyperplane` | (b, a…, lower…, upper…) |
| Half-space ∩ box (*since 0.2*) | $a^\top z \le b$, $\mathrm{lo} \le z \le \mathrm{hi}$ | `fp_op_hyperplane(a, b, lo, hi, lower, upper, 1)` | `HalfSpace(a, b, lower, upper)`, `project_halfspace` | (b, a…, lower…, upper…) |
| Second-order cone (*since 0.2*) | $\lVert (z_2, \dots, z_n) \rVert_2 \le z_1$ | `fp_op_soc()` | `SOC`, `project_soc` | none |
| Log barrier (*since 0.2*) | $-\lambda \sum_i \log z_i$ | `fp_op_log_barrier(λ)` | `LogBarrier(lam)`, `prox_log_barrier` | (λ) |
| ℓ1 over a box (*since 0.2*) | $\lambda \sum_i w_i \lvert z_i \rvert$ on $\mathrm{lo} \le z \le \mathrm{hi}$ | `fp_op_l1_box(λ, w, lo, hi, lower, upper)` | `L1Box(lam, lower, upper, weights)`, `prox_l1_box` | (w…, lower…, upper…) |
| Group Huber (*since 0.2*) | $\lambda \sum_g c_g h_\delta(\lVert z_g \rVert_2)$, $h_\delta$ as for Huber | `fp_op_group_huber(λ, δ, n, groups, G, c)` | `GroupHuber(lam, delta, groups, group_weights)`, `group_huber` | (λ, δ, c…) |
| Group SCAD, group MCP (*since 0.2*) | $\sum_g c_g \rho(\lVert z_g \rVert_2)$, $\rho$ the SCAD or MCP penalty | `fp_op_group_scad(λ, a, …)`, `fp_op_group_mcp(λ, b, …)` | `GroupSCAD`, `GroupMCP`, `group_scad`, `group_mcp` | (λ, a or b, c…) |
| Monotone cone | $z_1 \le \dots \le z_n$ (or $\ge$), weighted least squares | `fp_op_isotonic(w, increasing)` | `Isotonic(weights, increasing)`, `isotonic` | w (empty if none) |
| 1D total variation | $\sum_i w_i \lvert z_{i+1} - z_i \rvert$ | `fp_op_tv1d(w, method)` | `TV1D(weights, method)`, `tv1d` | w (scalar or n − 1 edges) |
| 1D TV ball (*since 0.2*) | $\sum_i w_i \lvert z_{i+1} - z_i \rvert \le \alpha$ | `fp_op_tv1d_ball(α, w)` | `TV1DBall(alpha, weights)`, `project_tv1d_ball` | (α, w…) |
| 1D TV, ℓ2 of differences (*since 0.2*) | $\bigl(\sum_i w_i^2 (z_{i+1} - z_i)^2\bigr)^{1/2}$ | `fp_op_tv1d_l2(w)` | `TV1DL2(weights)`, `tv1d_l2` | w (scalar or n − 1 edges) |
| ℓ0 (*since 0.2*) | $\lambda \sum_i w_i [z_i \ne 0]$ | `fp_op_l0(λ, w)` | `L0(lam, weights)`, `hard_threshold` | w (scalar or per entry) |
| ℓq, 0 < q < 1 (*since 0.2*) | $\lambda \sum_i w_i \lvert z_i \rvert^q$ | `fp_op_lq(λ, q, w)` | `Lq(lam, q, weights)`, `lq` | w (scalar or per entry) |
| SCAD (*since 0.2*) | $\sum_i w_i \rho(z_i)$, Fan and Li's penalty $\rho$, threshold $\lambda$, concavity $a$ | `fp_op_scad(λ, a, w)` | `SCAD(lam, a, weights)`, `scad` | (λ, a), or (λ, a, w…) with weights |
| MCP (*since 0.2*) | $\sum_i w_i \rho(z_i)$, Zhang's minimax concave penalty $\rho$, threshold $\lambda$, concavity $b$ | `fp_op_mcp(λ, b, w)` | `MCP(lam, b, weights)`, `mcp` | (λ, b), or (λ, b, w…) with weights |
| Log penalty (*since 0.2*) | $\lambda \sum_i \log(1 + \lvert z_i \rvert / \varepsilon)$ | `fp_op_log_penalty(λ, ε)` | `LogPenalty(lam, eps)`, `prox_log` | (λ, ε) |
| Anisotropic TV, 2D and 3D (*since 0.2*) | $\sum_a w_a \sum_p \lvert z_{p+e_a} - z_p \rvert$ over the last 2 or 3 axes | `fp_op_tv_aniso(ndim, w)` | `TVAniso(ndim, weights)`, `tv_aniso` | (w₀, …, w_{ndim−1}) |
| Isotropic TV, 2D and 3D (*since 0.2*) | $\sum_p \lVert (w_0 D_0 z, \dots, w_{d-1} D_{d-1} z)_p \rVert_2$, forward differences $D_a$, $d$ = `ndim` | `fp_op_tv_iso(ndim, w)` | `TVIso(ndim, weights)`, `tv_iso` | (w₀, …, w_{ndim−1}) |
| Anisotropic TV with edge weights (*since 0.3*) | $\sum_a w_a \sum_p \beta_{a,p} \lvert z_{p+e_a} - z_p \rvert$ | `fp_op_tv_aniso_weighted(ndim, w, beta)` | `TVAniso(ndim, weights, edge_weights=beta)` | (w, β in C order) |
| Isotropic TV with pixel weights (*since 0.3*) | $\sum_p \alpha_p \lVert (w_0 D_0 z, \dots)_p \rVert_2$ | `fp_op_tv_iso_weighted(ndim, w, alpha)` | `TVIso(ndim, weights, pixel_weights=alpha)` | (w, α in C order) |
| Graph TV (*since 0.3*) | $\sum_e w_e \lvert z_u - z_v \rvert$ over the edges $e = (u, v)$ of a graph, on the last axis | `fp_op_graph_tv(n, m, edges, w)` | `GraphTV(n, edges, weights)`, `graph_tv` | w (scalar or one per edge) |
| Tree TV (*since 0.3*) | as graph TV, for edges that form a forest (a tree, a path) | `fp_op_tree_tv(n, m, edges, w)` | `TreeTV(n, edges, weights)`, `tree_tv` | w (scalar or one per edge) |

Notes:

- **Step versus weights.** $\gamma$ multiplies the whole of $g$, so soft
  thresholding uses the threshold $\gamma\lambda w_i$ and 1D TV the edge
  thresholds $\gamma w_i$. Projections ignore $\gamma$.
- **Weighted ℓ1 ball:** weights must be positive with
  $\min w / \max w \ge 10^{-100}$ (only their ratios matter; smaller ratios are
  rejected with `FP_EPARAM`).
- **Maximum and the sum of the k largest** are support functions (of the
  simplex and of $\lbrace \sum_i y_i = k,\ 0 \le y \le 1 \rbrace$, or
  $\lbrace \lVert y \rVert_1 \le k,\ \lVert y \rVert_\infty \le 1 \rbrace$ for
  absolute values), so their prox is $x - P(x)$ with $P$ the projection onto
  the set scaled by $\gamma\lambda$. The maximum clips $x$ from above at the
  simplex threshold ($\min(x, \tau)$, the cost of a simplex projection). The
  sum of the $k$ largest entries needs $k \le n$; for absolute values any
  $k \ge 1$ works, and $k \ge n$ gives $\lambda \lVert z \rVert_1$.
- **Hyperplane ∩ box.** $a$ is a scalar or per entry (`NULL`/`None`: all ones),
  the box as for `Box`, unbounded by default; $a = 1$ with the box $[0, 1]$ is
  the capped simplex. The set is checked at each call for the axis length ($b$
  must lie in the range of $a^\top z$ over the box, or above its lower end for
  the half-space), and `FP_EPARAM` reports an empty one. Since only the ratios
  of $(a, b)$ matter, nonzero $\lvert a_i \rvert / \max \lvert a \rvert$ must be
  at least 1e-100. The half-space keeps the box projection when it meets
  $a^\top z \le b$.
- **Second-order cone.** $t$ is the first entry of the row, as in conic
  solvers (Beck's Example 6.37 puts it last). Rows of length 1 give the
  projection onto $t \ge 0$.
- **Log barrier** needs $\lambda > 0$. The prox
  $\bigl(x + \sqrt{x^2 + 4\gamma\lambda}\bigr)/2$ is evaluated without
  cancellation for $x < 0$ and without overflow for huge $\lvert x \rvert$ or
  $\gamma\lambda$.
- **ℓ1 over a box** is
  $\operatorname{clip}(\operatorname{soft}(x, \gamma\lambda w), \mathrm{lo}, \mathrm{hi})$,
  for any interval (Beck, Example 6.23, states the box $[-\alpha, \alpha]$);
  weights as for soft thresholding and bounds as for `Box`. `grads.lam` is also
  provided in Python.
- **Group penalties by norm composition.** Group Huber, SCAD and MCP scale each
  group by $\varphi(\lVert x_g \rVert)/\lVert x_g \rVert$, with $\varphi$ the
  scalar prox at the step $\gamma c_g$ (Beck, Theorem 6.18); groups and weights
  are as for group ℓ2, and a one-entry group is the scalar map. SCAD needs
  $a > 1 + \gamma \max c$ and MCP $b > \gamma \max c$, checked at each call
  (`FP_EPARAM`). Group SCAD and MCP are nonconvex, but their prox is
  single-valued under these conditions, so there is no tie rule.
- **Tree-structured group norms.** Any two groups must be nested or disjoint
  (else `FP_EPARAM`), given in any order as CSR arrays (`ptr` of length $G + 1$
  from 0, indices into the axis); entries may belong to no group. The prox
  composes soft thresholding and the group shrinks from the smallest group up
  (Bach, Jenatton, Mairal and Obozinski 2012, (3.8)), so it costs the total size
  of the groups. A partition is the sparse group lasso (`TreeL2.sparse_group`
  in Python, $\eta = \lambda c$), and a repeated group adds its weights.
- **1D TV ball.** $\alpha \ge 0$, weights as for 1D TV. Outside the ball the
  result is the 1D TV prox at the step $\gamma^\star$ where the TV equals
  $\alpha$ (Beck, Theorem 6.30); $\gamma^\star$ comes from safeguarded Newton
  steps on that equation, each a 1D TV solve. Rows took 4 to 15 solves in tests,
  so the projection costs 6–20 times a 1D TV prox. $\alpha = 0$ gives the mean
  of each piece between zero weights.
- **1D TV with ℓ2 of differences.** The dual is a trust-region problem in the
  differences. When its unconstrained minimiser, a cumulative sum of $x$, lies
  in the ball of radius $\gamma$, $z$ is the mean of each piece between zero
  weights (a weight whose square underflows counts as zero). Otherwise $z$
  solves the tridiagonal system $(\mu I + D^\top W^2 D) z = \mu x$, with $\mu$
  the root of Moré and Sorensen's secular equation, found by Newton's method
  from $\mu = 0$ (Nocedal and Wright, Algorithm 4.4). It converges monotonically, in 2–8 steps of one
  tridiagonal factorisation and two solves, to rounding, so `fp_info.gap` is 0.
  For $n = 2$ it is 1D TV. Derivatives are in closed form: the Hessian of the
  prox problem is that tridiagonal matrix minus a rank-one term (Sherman–
  Morrison), so a JVP or VJP costs three $O(n)$ solves.
- **Isotonic regression** minimises $\sum_i w_i (z_i - x_i)^2$, which is the
  Euclidean projection only for unit weights.
- **1D TV methods:** `FP_TV1D_AUTO` (default) runs Condat's direct
  algorithm (generalised to per-edge weights) and falls back to the taut
  string when it backtracks too much (more than $3n + 1024$ rescanned steps,
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
problem $\tfrac12 (z - x)^2 + \gamma\psi(z)$:
- **ℓ0** is hard thresholding at $\sqrt{2\gamma\lambda w_i}$.
- **ℓq.** Zero below the threshold
  $\tau = \beta + \gamma\lambda w_i q \beta^{q-1}$, with
  $\beta = \bigl(2\gamma\lambda w_i (1 - q)\bigr)^{1/(2-q)}$ the size of the
  jump. Above it, the root of
  $z + \gamma\lambda w_i q z^{q-1} = \lvert x \rvert$: in closed form for
  $q = 1/2$ and $2/3$ (a cubic and a quartic), by Newton's method from
  $\lvert x \rvert$ otherwise (monotone on that branch).
- **SCAD** needs $a > 1 + \gamma \max w$ and **MCP** $b > \gamma \max w$;
  below that the prox is not single-valued and the call returns `FP_EPARAM`
  (with `fp_prox_conj` the condition applies to its step $1/\sigma$). Both are
  continuous and piecewise linear; with $\gamma = 1$ and unit weights they are
  Fan and Li's and Zhang's thresholding rules. With weights, entry $i$ is the
  unweighted map at the step $\gamma w_i$ (for MCP,
  $w \rho_{\lambda,b} = \rho_{w\lambda,b/w}$), and a zero weight leaves the
  entry unchanged. The weights are optional: without them $\theta$ is
  $(\lambda, a)$ or $(\lambda, b)$; with them, a scalar or one per element
  along the axis, $\theta$ ends with $w$ and the VJP gives their gradient.
- **Log penalty.** The positive root of a quadratic, kept when it beats 0;
  for $\gamma\lambda \le \varepsilon^2$ the problem is convex.
- **Ties.** Where two minimisers exist (ℓ0 and ℓq at the threshold, the log
  penalty when both objective values are equal), `fp_options.tie` chooses:
  `FP_TIE_NONZERO` (default) or `FP_TIE_ZERO`, in Python `tie="nonzero"` or
  `"zero"`. `fp_info.ties` (Python: `ctx.info.ties`) counts the entries that
  sat on a tie, the same for any thread count. A tie means the input is
  exactly the threshold computed in double precision; a threshold that
  falls between two doubles has no tie.
- **Derivatives** are those of the selected branch: 0 off the support, and
  $1/(1 + \gamma\lambda\psi''(z))$ on it, from the scalar equation
  $z + \gamma\lambda\psi'(z) = x$, with the $\gamma$ and parameter derivatives
  from the same equation. ℓ0 is piecewise constant in $\gamma$ and $w$.
- **Conjugates.** `fp_prox_conj` returns
  $y - \sigma \operatorname{prox}_{g/\sigma}(y/\sigma)$. For these nonconvex
  $g$ that is not the prox of $g^\ast$: the Moreau identity holds only for
  convex $g$.
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
    alternates rows and columns at an $O(1/k^2)$ rate;
  - with three, the fast dual proximal gradient method for three blocks
    (Beck, Algorithm 9).
  Both restart their momentum when it points uphill (the gradient test of
  O'Donoghue and Candès, Found. Comput. Math. 15, 2015). The dual problem is
  a quadratic over a polyhedron, on which restarted methods converge
  linearly, so tight tolerances are not much dearer than loose ones.
  An axis of extent 1 or weight 0 drops out. With a single active axis the
  prox is the exact 1D TV, with gap 0.
- **Gap and stopping.** Every iterate carries duals $q_a$, one per axis and
  each feasible for its term, with $x - z = \sum_a q_a$. The duality gap of the
  prox problem at $(z, q)$ bounds the error:
  $\lVert z - z^\star \rVert^2 \le 2 \cdot \mathrm{gap}$. Each image stops when
  its gap is at most $\mathrm{tol} \cdot \lVert x \rVert^2 / 2$ (default
  1e-6), or after `max_iter`
  iterations (default 1000), in which case the call returns `FP_ENOTCONV`
  with the last iterate written. `fp_info.gap` is the sum over the batch,
  `fp_info.iters` the maximum. In 2D the iterate checked is the second
  block step of an iteration, so a check is one sweep over the image; checks
  come after iterations 1 and 2, then when the gap's rate of fall predicts
  the tolerance, and at least every 4 iterations, so a loose solve can stop
  after one or two iterations. In 3D the gap is
  checked every 4 iterations. An unpolished $z$ returned is $x - \sum_a q_a$ for
  the dual returned.
- **Polish.** The constant regions of the solution are read off the dual:
  an edge whose 1D dual lies strictly inside its bound has no jump at the
  solution. The values on those regions have a closed form, from the input
  and the dual's signs on their boundary. At the gap checks (the first, in
  2D from iteration 4 of a cold start; then once the iterate's gap has
  fallen tenfold, or sooner when the last attempt predicts success; and at
  the end) this polished
  point is formed and its gap, with the same dual, computed; the solve stops
  as soon as that gap meets the tolerance and is at most 1/100 of the
  iterate's (a larger gap means the regions are not yet right, and such a
  point can be further from the solution than the iterate). Once the regions are right the
  polished point is the exact solution to rounding, and its gap is often
  orders of magnitude below that of the iterate, so the polish both ends
  tight solves early and makes their output exact. The dual returned is the
  iterate's: $x - z = \sum_a q_a$ holds for the iterate, and the gap with that
  dual certifies the polished output.
- **Warm starts.** The dual of the last call stays in the workspace, shaped
  `(ndim, *x.shape)`. `fp_workspace_get_dual` copies it out, and
  `fp_workspace_set_dual` seeds the next call with a dual, for example one from
  a nearby problem. After `fp_workspace_set_warm(ws, 1)`, every call starts from
  the dual of the previous call, as in a PDHG or bilevel loop. Without either,
  each call starts from zero. A starting dual is made feasible first: a fiber
  whose running sums leave $[-\gamma w_a, \gamma w_a]$ or do not end at 0
  (beyond rounding) is rebuilt from its running sums clipped to that interval,
  and non-finite ones set to 0. Duals from earlier calls are feasible and are
  used unchanged, and the output and the polish are functions of the input and
  the dual, so a warm start from the dual of a solve of the same input returns
  its output bit for bit after 0 iterations.
- **Derivatives** are exact for the identified regions: averaging over each
  region, with $\gamma$ and weight terms from the jump signs on its boundary.
  They cost $O(n)$, with no linear solve. The regions are those of the polish,
  read off the dual: an edge whose 1D dual is strictly inside its bound,
  $\lvert p_e \rvert < \gamma w_a (1 - 10^{-9})$, has no jump at the solution,
  and a boundary edge takes the sign of $p_e$. Once the dual identifies the
  regions (here from a gap tolerance of 1e-10, or sooner where the polish
  succeeds) the derivative is the exact one; at 1e-8 on 128² images its error
  was 2–5%, against 7–19% from a threshold on the jumps of $z$.
- **Special values.** A NaN anywhere in an image makes that whole image NaN
  (the rest of the batch is unaffected); ±Inf is rejected with `FP_EINVAL`.
  Images are limited to 2³¹ − 1 entries.

**Isotropic TV.** The weights scale each axis inside the norm (for example
voxel spacing); differences past the last entry are zero, as in Beck's
$\mathrm{TV}_\mathrm{I}$.
- **Algorithm.** The fast dual proximal gradient method, that is FISTA on the
  dual (Beck and Teboulle, IEEE TIP 18, 2009; Beck 2017, §12.4.4): each
  iteration is a divergence, a gradient and a pixelwise projection onto the
  ball of radius $\gamma$, with step $1/L$, $L = 4 \sum_a w_a^2$ (8 in 2D,
  12 in 3D), and the
  primal taken from the last dual iterate. A single active axis is the exact
  1D TV. It is rescaled internally by the largest weight, so tiny or huge
  weights and steps stay finite, and an axis whose weight is below 1e-100 of
  the largest drops out.
- **Dual.** $u$ of shape `(ndim, *x.shape)` with
  $\lVert u_p \rVert_2 \le \gamma$, zero past the last entry along each axis,
  and $z = x - \sum_a w_a D_a^\top u_a$. A warm start projects the given dual
  onto the current ball, so a dual from another $\gamma$ can be reused. The
  gap is $\sum_p (\gamma \lVert g_p \rVert - \langle g_p, u_p \rangle)$ with
  $g = W D z$, each term $\ge 0$.
- **Derivatives** come from implicit differentiation of the optimality
  conditions. Flat entries keep $z$ constant across their edges, which reduces
  the unknowns to one per flat region; elsewhere the dual turns on the sphere
  ($H_p = \gamma / \lVert g_p \rVert \, (I - n n^\top)$). An entry is flat when
  its dual is strictly inside the ball,
  $\lVert u_p \rVert < \gamma (1 - 10^{-9})$ (complementary slackness), or its
  gradient is at the rounding level of $x$. Pixels on the sphere with a small
  gradient make the system stiff, more so as the gap shrinks, though such
  gradients are below the error that the gap certifies; the curvature
  $\gamma / \lVert g_p \rVert$ is capped at $\gamma/\mu$ with
  $\mu = 0.1\, w_{\max} \mathrm{rms}(\mathrm{gap})$,
  $\mathrm{rms}(\mathrm{gap}) = \sqrt{2\, \mathrm{gap} / N}$ (a Huber smoothing
  at that level, which vanishes with the gap). On 128² images the derivative's
  error against the exact one is 4% at a gap tolerance of 1e-6 and 0.5–0.7% at
  1e-8, against 18–57% and 5–24% from the earlier threshold on
  $\lVert g_p \rVert$ set by the certified error, at 3–8× its conjugate gradient
  iterations. Below about 1% the comparison reaches pixels that sit on the
  sphere with a gradient near zero even at a gap of 5e-16, where the derivative
  is set-valued (two valid choices differ by 0.8% at $\gamma = 0.1$). The
  resulting symmetric positive definite system is solved by conjugate gradients.
  By default each image's solve stops at the relative error of $z$ that its gap
  certifies, $\sqrt{2\, \mathrm{gap}} / \lVert x \rVert$ (at most
  $\sqrt{\mathrm{tol}}$ for a solve to `tol`, and no tighter than 1e-10): the
  derivative's own error after a solve to `tol` is 50–100 $\sqrt{\mathrm{tol}}$
  on 128² images, and residuals below $\sqrt{\mathrm{tol}}$ did not change it,
  so this takes 2–5× fewer iterations than a fixed 1e-10 at the same accuracy.
  `fp_workspace_set_deriv_tol(ws, tol, max_iter)` sets a fixed tolerance (0
  restores the default) and `fp_workspace_deriv_info(ws, &residual, &iters)`
  reports the last solve (in Python, `jvp`/`vjp` take `tol` and `max_iter`, and
  `ctx.deriv` holds the residual and iterations). No polish: isotropic regions
  have no closed-form values.
- FISTA converges like $1/k^2$ in the dual, so very tight tolerances take many
  iterations (about $1/\sqrt{\mathrm{tol}}$); the default `tol` of 1e-6 is
  cheap.

**Spatially varying weights** (*since 0.3*), for spatially dependent
regularisation (e.g. learned in a bilevel problem) and for reweighted ℓ1
methods, whose convex majorisers of nonconvex TV are weighted TV problems
(Ochs, Dosovitskiy, Brox and Pock, SIAM J. Imaging Sci. 8, 2015; see
[examples/reweighted_tv.py](https://github.com/dvillacis/fastprox/blob/master/examples/reweighted_tv.py)):

- `fp_op_tv_iso_weighted(ndim, w, alpha)` minimises
  $\gamma \sum_p \alpha_p \lVert (w_0 D_0 z, \dots)_p \rVert_2$ with pixel
  weights $\alpha \ge 0$, an array of the image shape;
  `fp_op_tv_aniso_weighted(ndim, w, beta)` minimises
  $\gamma \sum_a w_a \sum_p \beta_{a,p} \lvert z_{p+e_a} - z_p \rvert$ with edge
  weights $\beta \ge 0$ of shape (ndim, image shape), $\beta_{a,p}$ weighting
  the edge from $p$ to its forward neighbour along $a$ (the entries at the last
  position along $a$ belong to no edge and are ignored). The operator then takes
  only images of that shape (`FP_ESHAPE` otherwise), batched over leading axes
  as before. In Python: `TVIso(ndim, weights, pixel_weights=alpha)` and
  `TVAniso(ndim, weights, edge_weights=beta)`.
- **Methods.** The weights enter only the dual bounds:
  $\lVert u_p \rVert \le \gamma\alpha_p$, and
  $\lvert p_e \rvert \le \gamma w_a \beta_e$ for the 1D duals, so the
  iterations, the gap and warm starts are those above. Anisotropic TV's 1D steps
  are weighted 1D TV proxes (the same kernels as `TV1D` with edge weights) and
  its polish uses each edge's bound. Isotropic TV with a single active axis runs
  the 2D method rather than the exact 1D pass. A zero weight frees its pixel or
  edge: no dual, no coupling. All-ones weights give the unweighted operator bit
  for bit.
- **Derivatives** in $x$, $\gamma$, $w$ and the spatial weights, with one
  gradient per pixel or edge (zero at the ignored entries):
  `grads.pixel_weights` has the image shape and `grads.edge_weights` the shape
  of $\beta$. Isotropic TV's linear system takes $\gamma\alpha_p$ in its
  curvature and
  $\mathrm{d}r_p = \mathrm{d}\gamma\, \alpha_p + \gamma\, \mathrm{d}\alpha_p$ in
  its right-hand side; the gradient in $\alpha_p$ is
  $-\gamma \langle (G v)_p, n_p \rangle$ on non-flat pixels and 0 on flat ones.
  Anisotropic TV's region averages take the weight of each boundary edge; the
  gradient in $\beta_e$ is
  $-\gamma w_a \sigma_e (\bar g_{C(i)} - \bar g_{C(j)})$ for $e = (i, j)$, with
  $\sigma_e$ the sign of the jump and $\bar g$ the cotangent averaged over each
  region. At a zero weight the derivative is one-sided (weights cannot go below
  0).
- **Cost.** The same iterations as the unweighted operator for comparable
  weights, and 14% (isotropic) to 20–33% (anisotropic) more time per solve
  for the per-pixel or per-edge bounds (512² image, $\gamma = 0.1$, tol 1e-6, 15
  threads: 101 against 89 ms, 36–40 against 30 ms). Isotropic TV with
  weights near zero can need many more iterations to very tight tolerances
  (a 20 × 9 test image: a gap of 2e-11 after 10⁶ iterations, against 1e-14
  unweighted).
- The device-native `fastprox.torch.tv_iso`/`tv_aniso` and
  `fastprox.jax.tv_iso`/`tv_aniso` take them too, as `pixel_weights` and
  `edge_weights`, with one map per image if wanted (see Device-native image
  TV below); `fastprox.torch.prox(fp.TVIso, x, gamma, pixel_weights=a)` runs
  the C kernel with gradients in `a`.

### Graph TV (*since 0.3*)

Graph TV acts on the **last** axis, whose length must be the number of nodes
$n$, and is batched over the leading axes; `fp_options.axis` is ignored. The
$m$ edges are given as $2m$ node indices $(u_0, v_0, u_1, v_1, \dots)$, in
Python an array of shape $(m, 2)$; $u \ne v$, and a repeated edge adds its
weight. Each signal is solved on its own, sequentially, so a batch gives the
same result as one call per signal, for any thread count; threads go to the
batch.

- **Algorithm.** Parametric max-flow (Chambolle and Darbon, IJCV 84, 2009;
  Hochbaum, J. ACM 48, 2001). The set $\lbrace z > t \rbrace$ of the solution is
  the smallest minimum cut of a graph with terminal $x_p - t$ at node $p$ and
  capacity $\gamma w_e$ both ways on each edge. Each part of the graph is cut at
  the mean of its solution, which is known in closed form: if that cut is
  trivial the part is constant at its mean, exactly; otherwise it splits, the
  edges across the cut become fixed terms of the two sides, and each side goes
  on as its connected pieces. All parts are cut in one max-flow solve per round,
  which starts from the previous round's flow on the parts still active (since
  0.4 the excesses incremental breadth-first search of Goldberg, Hed, Kaplan,
  Kohli, Tarjan and Werneck, ESA 2015, in `src/maxflow/eibfs.hpp`: augmenting
  along shortest paths between the excesses and deficits of the flow; before,
  the Boykov–Kolmogorov engine of `src/maxflow/maxflow.hpp`, 1.9–2.7× slower at
  $\gamma = 0.5$ on 512² grids). A level is clamped into the middle half of the
  interval known to hold the part's values, so the interval shrinks by at least
  a quarter per round.
- **Precision.** A part whose values are known within
  $\mathrm{tol} \cdot (\max x - \min x)$ is set to its mean, which is then
  within that distance of the solution at every node; `tol` = 0 asks for the
  solution to rounding. The Python methods default to `tol=0`; in C,
  `fp_options.tol` defaults to 1e-6, so set it to 0 for the exact result. On
  512² images the exact solve takes about 20 rounds; a loose `tol` saves the
  last ones (1e-4: 10–50% of the time, 1e-2: 2–5×). `max_iter` caps the rounds:
  the call then returns `FP_ENOTCONV`, with every unfinished part at its mean.
- **Certificate.** The final flow is a dual: $f_e$ with
  $\lvert f_e \rvert \le \gamma w_e$ and divergence $q = D^\top f$.
  `fp_info.gap` is the duality gap of $(z, f)$, summed over the batch, and
  `fp_info.iters` the largest number of max-flow solves. The workspace dual, of
  shape `(1, *x.shape)`, is $q$, with $z = x - q$ at the solution. Warm starts
  are accepted and have no effect.
- **Derivatives** are exact: $z$ is constant on the regions joined by edges
  with $\gamma w_e > 0$ and equal values (to rounding), and each region's
  value is its mean of $x$ minus $\gamma \sum_e w_e s_e / \lvert R \rvert$ over
  its boundary edges, $s_e$ the sign of the jump. The Jacobian averages over
  the regions, and the $\gamma$ and weight derivatives come from the boundary
  signs, in $O(n + m)$.
- **Special values** as for the image operators: a NaN makes that signal NaN,
  ±Inf is rejected with `FP_EINVAL`. Graphs are limited to 2³¹ − 1 nodes and
  2³⁰ − 1 edges.
- **Cost.** Augmenting paths are fast on small and medium graphs and at small
  $\gamma$; on large grids at large $\gamma$ each round re-routes flow through
  the flat regions. On 4-neighbour grids, where it is the same problem,
  `TVAniso` is faster at moderate accuracy, and exact graph TV is comparable
  with it at a gap tolerance of 1e-10 (see docs/benchmarks.md). For a forest,
  `TreeTV` is exact in one pass and faster.

### Tree TV (*since 0.3*)

`TreeTV` takes graph TV's arguments and conventions (last axis, edges as
pairs, weights NULL, one or one per edge) for edges that form a forest: no
cycles and no repeated edges, otherwise the constructor fails with
`FP_EPARAM`. Edges may come in any order and orientation, and node labels in
any order; a chain is a tree, so a signal along a curve or a path through a
graph can be given as it is. In Python `TreeTV` is a subclass of `GraphTV`.

- **Algorithm.** Dynamic programming (Kolmogorov, Pock and Rolínek, SIAM J.
  Imaging Sci. 9, 2016, Algorithm 2 and Section 4.2). Each tree is rooted and
  visited from the leaves up; at each node the derivative of the best cost of
  its subtree is a piecewise linear function, held as its outer pieces and a set
  of breakpoints, which is clipped to $[-\gamma w_e, \gamma w_e]$ and passed to
  the parent. The breakpoints sit in two pairing heaps, one for each end:
  $O(n \log n)$. When every tree is a path the breakpoints form a sorted deque,
  $O(n)$ (Section 4.1). Then each node takes its parent's value clipped to its
  own interval, on the way down.
- **Exactness.** Exact up to rounding: the values are recomputed as the means
  of the regions found, as in the derivative formula. `tol`, `max_iter` and
  warm starts have no effect; `fp_info.iters` is 1.
- **Certificate and derivatives** are graph TV's: on a tree the dual flow is
  determined by $z$ (the flow above a node is the sum of $x - z$ over its
  subtree), `fp_info.gap` is the duality gap with it, the workspace dual is
  $q = D^\top f$, and the JVP and VJP average over the regions where $z$ is
  constant.
- **Cost.** On 10⁶ nodes 70–190 ms for random trees and 35–75 ms for a comb
  spanning tree of a 1000² image, 4–8× faster than graph TV on the same
  trees; 22–27 ms on a path, 3.5–4× the time of `TV1D` on the ordered signal
  (see docs/benchmarks.md).


Every operator is piecewise smooth. At a point where two pieces meet,
`fp_jvp` and `fp_vjp` use the Jacobian of one of them, chosen as follows:

| Operator | At the kink | Jacobian used |
| --- | --- | --- |
| soft thresholding, elastic net | $\lvert x_i \rvert$ equal to the threshold | thresholded: 0 |
| Huber | $\lvert x_i \rvert = \delta + \gamma\lambda$ | inner (quadratic) branch: $\delta/(\delta + \gamma\lambda)$ |
| box, nonnegative orthant | $x_i$ on a bound | interior: 1 |
| group ℓ2, ℓ2 norm | $\lVert x_g \rVert$ equal to the threshold | group set to zero: 0 |
| ℓ2 ball | $\lVert x \rVert = r$ | inside: identity |
| simplex, ℓ1 ball | $x_i$ (or $\lvert x_i \rvert / w_i$) equal to the threshold $\tau$ | entry off the support |
| ℓ1 ball | weighted ℓ1 norm of $x$ equal to $r$ | inside: identity |
| ℓ∞ norm | $\lvert x_i \rvert = \tau$ | entry not clipped: 1 |
| ℓ∞ norm | $\lVert x \rVert_1 = \gamma\lambda$ | $z = 0$: 0 |
| maximum | $x_i = \tau$ | entry not clipped: 1 |
| sum of the k largest | $x_i - \mu$ on 0 or $\gamma\lambda$ (an end of the scaled cap) | entry at the shared level $\mu$ |
| hyperplane, half-space ∩ box | $x_i - \mu a_i$ on a bound | interior (free) |
| half-space ∩ box | $a^\top P_{\mathrm{box}}(x) = b$ | box projection (constraint inactive) |
| second-order cone | $\lVert y \rVert = t$ (on the cone), or the origin | inside: identity |
| second-order cone | $\lVert y \rVert = -t > 0$ (on the polar cone) | $z = 0$: 0 |
| log barrier | none for $\gamma\lambda > 0$; at $\gamma = 0$, $x_i = 0$ | 1 |
| ℓ1 over a box | $\lvert x_i \rvert$ equal to the threshold; $\operatorname{soft}(x_i)$ on a bound | thresholded: 0; interior (soft's Jacobian) |
| group Huber, SCAD, MCP | $\lVert x_g \rVert$ at a breakpoint of the scalar map | the scalar map's rule (Huber: inner branch; SCAD, MCP: the branch below) |
| group Huber, SCAD, MCP | $x_g = 0$ | $\varphi'(0) I$, the limit of $\varphi(s)/s$ |
| tree-structured group ℓ2 | $\lVert z_g \rVert$ equal to the threshold at a shrink; $\lvert x_i \rvert$ at the ℓ1 threshold | group set to zero; thresholded |
| 1D TV ball | $\mathrm{TV}(x) = \alpha$ | inside: identity |
| 1D TV ball | adjacent segments with equal values | as for 1D TV, at the step $\gamma^\star$ |
| 1D TV, ℓ2 of differences | the dual minimiser on the sphere, $\lVert u_0 \rVert = \gamma$ | inside: piecewise averaging |
| isotonic regression | adjacent blocks with equal means | blocks not pooled (pooling happens only on a strict violation) |
| 1D TV | adjacent segments with equal values | the segments the solver returned, each averaged on its own |
| anisotropic TV | an edge with no jump whose dual is on its bound | not joined: its two sides averaged apart, with the sign of the dual |
| ℓ0, ℓq, log penalty | $\lvert x_i \rvert$ on the threshold (two minimisers) | the branch of the tie rule: nonzero by default, 0 with `FP_TIE_ZERO` |
| SCAD, MCP | $\lvert x_i \rvert$ at a breakpoint | the branch below it (at $\gamma\lambda$: thresholded) |
| isotropic TV | an entry with a vanishing gradient whose dual is on the sphere | not flat: $H_p$ from its gradient (flat only when the gradient is at the rounding level) |

The ℓ∞ rule is the reverse of the soft-thresholding rule. The ℓ∞ prox is
$x - \gamma\lambda P(x/(\gamma\lambda))$, with $P$ the projection onto the ℓ1
ball, so an entry kept off the support of $P$ is left unclipped. `fp_prox_conj`
flips its operator's rule in the same way.

**Why these are valid derivatives.** Every choice in the table is the
Jacobian of one of the smooth pieces that meet at the point. So it belongs
to the B-subdifferential, the set of limits of nearby Jacobians, and hence
to Clarke's generalised Jacobian:
- S. Scholtes, *Introduction to Piecewise Differentiable Equations*,
  Springer 2012, Prop. 4.3.1, and Prop. 2.4.3–2.4.5 for projections onto
  polyhedra (simplex, ℓ1 ball, monotone cone, and 1D TV through the Moreau
  identity).
- C. Clason and T. Valkonen, *Introduction to Nonsmooth Analysis and
  Optimization*, Ex. 14.10. For soft thresholding at $\lvert x \rvert = \tau$,
  both 0 and 1 are such limits, and neither is canonical.

The derivatives in $x$, $\gamma$ and the parameters $\theta$ all come from the
same saved structure. Together they are therefore the Jacobian of one piece in
the joint variables $(x, \gamma, \theta)$. As a result, a VJP through a smooth
outer loss is a Clarke subgradient of the composition.

At a kink, the JVP is not the one-sided directional derivative for
directions that leave the chosen piece. Finite-difference tests therefore
avoid kinks.

**Kink masks and branch overrides (*since 0.2*).** For soft thresholding,
ℓ1 over a box, SCAD and MCP (since 0.3 also ℓ0, ℓq and the log penalty, 1D
and anisotropic TV, and masks for isotropic TV; since 0.4 the box, the
orthant, the simplex, the ℓ1 ball, group ℓ2, graph and tree TV), the kinks of a kept prox call can be read and
the piece differentiated there chosen, which gives the other elements of the
B-subdifferential, for example both elements at a biactive entry, or the
SCD subspaces of each entry:
- `fp_workspace_kinks(ws, tol, mask, &count)` writes an int8 code per entry
  (dtype `FP_I8`, the shape of $x$): 0 away from a kink, and $k \ge 1$ for the
  breakpoint, the nearest one, within `tol` of the entry. `tol` is absolute,
  in the units of the input ($x$, or $y$ after `fp_prox_conj`); exact equality
  rarely occurs in a computed solution, so `tol` is usually set from its
  accuracy. Breakpoints: soft thresholding 1: $\lvert x \rvert = \gamma\lambda w_i$;
  ℓ1 over a box 1: $\lvert x \rvert = \gamma\lambda w_i$, 2:
  $\operatorname{soft}(x) = \mathrm{lower}$, 3:
  $\operatorname{soft}(x) = \mathrm{upper}$; SCAD 1:
  $\lvert x \rvert = \gamma w_i \lambda$, 2: $(1 + \gamma w_i)\lambda$, 3:
  $a\lambda$; MCP 1: $\lvert x \rvert = \gamma w_i \lambda$, 2: $b\lambda$.
  Since 0.3 the ties of the nonconvex maps too, each with code 1: ℓ0 at
  $\lvert x \rvert = \sqrt{2\gamma\lambda w_i}$, ℓq at its threshold $\tau$,
  the log penalty at its tie (found by bisection) or, where its map is
  continuous ($\gamma\lambda \le \varepsilon^2$), at
  $\lvert x \rvert = \gamma\lambda/\varepsilon$; their two sides are the zero
  piece and the nonzero branch, taken at $\max(\lvert x \rvert, \tau)$ where
  the branch starts at $\tau$. A map that is the identity ($\gamma w_i = 0$ or
  $\lambda = 0$) has none. The mask is
  the same for any thread count.
- **Projections and groups** (*since 0.4*): the box flags entries on its
  bounds (1: lower, 2: upper) and the orthant those at 0, with the inside
  piece on one side and the clipped one on the other; the simplex flags
  entries at its threshold $\tau$ (code 1) and the ℓ1 ball those with
  $\lvert x_i \rvert$ at $\tau w_i$ (1), with the support including them (+1)
  or not (−1); the ℓ1 ball also flags every entry of a row on its sphere,
  $\sum_i w_i \lvert x_i \rvert = r$ within $\mathrm{tol} \cdot \max w$ (2),
  and group ℓ2 every entry of a group whose norm is at its threshold
  $\gamma\lambda c_g$ (1). These row-level kinks take the side given at the
  row's or group's first entry: −1 the inside piece (the identity) or the
  zeroed group, +1 the projection with $\tau = 0$ or the live shrink, its
  formula taken at $x$. Each side is the derivative of the one-sided
  difference in a direction that keeps its piece.
- **Graph and tree TV** (*since 0.4*): their kinks are edges, so
  `ctx.kinks(tol)` and `branch=` have the shape of $x$ with the last axis
  over the edges, in the order of `edges`. An edge is at a kink when its
  ends are equal (within `tol`) and its flow is within `tol` of $\gamma w_e$; −1
  joins its ends (the default) and +1 keeps them apart, with the jump
  taking the flow's sign. A region splits only where every edge across the
  split is kept apart (on a cycle, at least two edges). Tree TV's flow is
  the sum of $x - z$ over the subtree below the edge, unique on a forest;
  graph TV uses the flow its solver found, saved with the structure (one
  double per edge and image), since on a graph with cycles the flow is not
  unique and a kink depends on which one is used.
- **Isotropic TV** has masks but no branch overrides (`fp_workspace_set_branch`
  returns `FP_EINVAL`): keeping a pixel with zero gradient apart from its
  neighbours has infinite curvature $\gamma / \lVert g_p \rVert$, which its
  derivative system holds only through the curvature cap.
- **TV kinks** (*since 0.3*), from the saved solution $z$ and dual: 1D TV flags
  (code 1) the entry $i$ at the tail of an edge with no jump,
  $\lvert z_{i+1} - z_i \rvert \le \mathrm{tol}$, whose dual
  $\sum_{j \le i} (x_j - z_j)$ is within `tol` of its bound $\gamma w_i$;
  anisotropic TV sets bit $a$ (value $2^a$) at pixel $p$ for such an edge along
  axis $a$ (its 1D dual against $\gamma w_a \beta_e$); isotropic TV flags a
  pixel with $\lVert (W D z)_p \rVert \le \mathrm{tol}$ whose dual is within
  `tol` of its sphere, $\lVert u_p \rVert \ge \gamma\alpha_p - \mathrm{tol}$.
  These are the biactive edges and pixels of the table above, where the
  derivative is set-valued (joined or apart). Zero bounds (free edges, zero
  weights) are never kinks.
- **TV branch overrides** (*since 0.3*, 1D and anisotropic TV):
  `fp_workspace_set_branch(ws, side, tol)` (`branch=` and `kink_tol=` in
  Python) chooses at each kink within `tol` whether the derivative treats
  the edge as joined (side −1 at its tail: its two sides averaged together)
  or apart (+1: averaged separately, the jump taking the sign of the edge's
  dual). These are the two pieces of the piecewise-affine prox that meet
  there, so each is the derivative of the one-sided difference in a
  direction that keeps it: closing the gap keeps the edge joined, opening it
  or lowering $\gamma$ keeps it apart. 1D TV rewrites its saved segment codes;
  anisotropic TV rebuilds its saved regions from the kept $z$ and dual (one
  pass over the image, as in the prox call). Isotropic TV has none yet:
  apart, a pixel with zero gradient is the limit of curvatures
  $\gamma / \lVert g_p \rVert \to \infty$,
  a constraint the conjugate gradient system holds only through its cap
  (`fp_workspace_set_branch` returns `FP_EINVAL`).
- `fp_workspace_set_branch(ws, side, tol)` sets the piece that `fp_jvp` and
  `fp_vjp` differentiate at the entries within `tol` of a breakpoint: side
  −1 or +1 (int8, the shape of $x$) takes the piece on the $x - \varepsilon$
  or $x + \varepsilon$ side of the breakpoint, with that piece's formula
  evaluated at $x$; 0, and every entry away from a kink, keep the rule in the
  table. Side +1 is the derivative of the forward difference
  $(\operatorname{prox}(x + h e_i) - \operatorname{prox}(x))/h$ and −1 of the
  backward one, in $x$ and, along the same piece, in $\gamma$ and $\theta$. The
  override lasts until the next prox call or `fp_workspace_set_branch(ws,
  NULL, 0)`.
- In Python: `ctx.kinks(tol)` returns the mask, and `jvp` and `vjp` take
  `branch=` (an array or scalar of −1, 0, 1) and `kink_tol=`, for that call
  only.
- `ctx.jacobian_diag(branch=None, kink_tol=0.0)` (Python) is the diagonal of
  $\mathrm{d}z/\mathrm{d}x$ for every elementwise operator, `jvp(ctx, dx=ones)`
  under its own name and honouring the override. For the nonconvex maps it reads
  off each entry's SCD subspace for a semismooth* Newton step: 0 where the entry
  is fixed at zero, 1 on a piece without curvature, and otherwise the slope of
  the concave branch ($b/(b - \gamma w_i)$ for MCP). Other operators raise
  `ValueError`, since their Jacobian is not diagonal.

```python
op = fp.MCP(0.5, 3.0)
z, ctx = op.prox(x, 0.8, keep=True)
k = ctx.kinks(1e-8)                                   # 1: |x| = 0.4, 2: |x| = 1.5
dx = np.ones_like(x)
up = op.jvp(ctx, dx, branch=1, kink_tol=1e-8)         # pieces on the x + eps side
dn = op.jvp(ctx, dx, branch=-1, kink_tol=1e-8)        # and on the x - eps side
d = ctx.jacobian_diag(branch=1, kink_tol=1e-8)        # the same as up here, dx being ones
```

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
- `fp_jvp` computes
  $J_x\, \mathrm{d}x + (\partial z/\partial\gamma)\, \mathrm{d}\gamma + (\partial z/\partial\theta)\, \mathrm{d}\theta$;
  `fp_vjp` computes $J_x^\top g$, $\langle g, \partial z/\partial\gamma \rangle$
  and $(\partial z/\partial\theta)^\top g$, summed over the batch. `dtheta` and
  `gtheta` are vectors of length `fp_op_nparams(op)`.
- Derivatives use the structure saved by the last prox call that had
  `keep_structure` set; without it they return `FP_EINVAL`.
- After `fp_prox_conj`, derivatives are those of the conjugate prox and $\gamma$
  reads as $\sigma$.
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

`fp_prox_conj(op, σ, y, out, ...)` computes
$\operatorname{prox}_{\sigma g^\ast}(y) = y - \sigma \operatorname{prox}_{g/\sigma}(y/\sigma)$.
$\sigma = 0$ returns $y$.

### Errors

| Status | Meaning |
| --- | --- |
| `FP_OK` | success |
| `FP_EINVAL` | invalid argument, ±Inf in the input of an operator where it is undefined, missing saved structure, mismatched workspace |
| `FP_EDTYPE` | dtype other than float32/float64, or mismatched dtypes |
| `FP_ESHAPE` | mismatched shapes or parameter lengths |
| `FP_ENOMEM` | out of memory |
| `FP_EPARAM` | invalid parameter or step (negative $\gamma$, negative radius, …), or a constraint set that is empty for the axis length (hyperplane ∩ box, the sum of $k > n$ largest entries) |
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
w = np.linspace(0.5, 1.5, x.shape[-1])
zm = fp.mcp(x, lam=0.2, b=3.0, weights=w)           # sum_i w_i MCP(z_i); b > gamma max(w)
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

# Tree-structured groups and level sets (since 0.2).
tree = [range(0, 10), range(10, 20), range(0, 20), range(20, 1000)]
zt = fp.tree_soft_threshold(x, tree, group_weights=[0.5, 0.5, 1.0, 2.0], l1=0.1)
zs = fp.sparse_group_soft_threshold(x, l1=0.1, lam=0.5, groups=groups)
zv = fp.project_tv1d_ball(x, alpha=5.0)              # sum |z_{i+1} - z_i| <= 5 per row
zq = fp.tv1d_l2(x, gamma=2.0)                        # prox of 2 ||D z||_2 per row

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
alpha = np.ones((256, 256)); alpha[:, :128] = 0.5          # since 0.3: one weight per pixel
wiso = fp.TVIso(2, pixel_weights=alpha)
v, ctx = wiso.prox(img, 0.2, keep=True)
gx, grads = wiso.vjp(ctx, np.ones_like(img))                 # grads.pixel_weights: (256, 256)
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

### PyTorch and JAX (*since 0.3*)

`fastprox.torch` and `fastprox.jax` make any operator a differentiable
function of its input, its step and its parameters. The gradients come from
the library's own VJP (and JVP), not from differentiating through
iterations, so the iterative operators cost one solve and one derivative
pass.

```python
import torch, jax, jax.numpy as jnp
import fastprox as fp
import fastprox.torch as fpt
import fastprox.jax as fpj

x = torch.randn(64, 1000, dtype=torch.float64, requires_grad=True)
w = torch.full((999,), 0.8, dtype=torch.float64, requires_grad=True)
z = fpt.prox(fp.TV1D, x, 0.5, weights=w)       # a class and its parameters
z.pow(2).sum().backward()                      # x.grad, w.grad
v = fpt.prox(fp.TVIso(), torch.randn(8, 64, 64), 0.1, tol=1e-6)   # an operator
c = fpt.prox_conj(fp.Simplex(), x, 2.0, axis=-1)

xj = jnp.asarray(x.detach().numpy(), jnp.float32)
loss = lambda w: jnp.sum(fpj.prox(fp.TV1D, xj, 0.5, weights=w) ** 2)
g = jax.jit(jax.grad(loss))(jnp.full(999, 0.8))
```

- **`prox(op, x, gamma=1.0, *, axis=None, threads=0, tol=None, max_iter=None, **params)`**
  and `prox_conj(op, y, sigma, ...)`. `op` is an operator, whose
  parameters are then constants, or an operator class with its constructor
  arguments as keywords. `axis` (row operators), `tol` and `max_iter`
  (iterative operators) are passed on when given.
- **Parameter gradients.** A parameter gets the gradient of the `Grads` field
  of the same name (see `Operator.vjp`): in PyTorch, float tensors that
  require grad; in JAX, JAX float arrays (Python and NumPy values are
  constants). PyTorch raises `ValueError` for a tensor that requires grad
  and has no gradient field (`L0`'s `lam`, `Isotonic`'s weights); JAX, which
  computes every cotangent, returns NaN for it. Integer arguments (groups,
  edges) are constants and must be concrete under `jax.jit`.
- **PyTorch**: a `torch.autograd.Function` with `backward` (the VJP) and
  `jvp` (the JVP, for `torch.autograd.forward_ad`; only for parameters that
  are part of $\theta$, so not `Soft`'s `lam`). Not twice differentiable, and not
  under the `torch.func` transforms (`grad`, `jvp`, `vmap`), which pass the
  derivatives wrapped tensors that cannot be read as NumPy arrays. Checked
  with `torch.autograd.gradcheck` in both modes. `torch.compile` breaks the
  graph at the call.
- **JAX**: `jax.custom_vjp` over `jax.pure_callback`, so it works under
  `jit`, `grad` and `vmap` (one call per batch element, as
  `vmap_method="sequential"`); reverse mode only. The backward pass
  recomputes the prox to rebuild its structure, so a gradient costs two
  prox calls and one VJP. Float64 needs `jax_enable_x64`.
- **Devices and dtypes.** The computation runs on the CPU in float32 or
  float64. PyTorch tensors on another device are copied to the CPU and the
  result back; other dtypes raise `TypeError`. For image TV on a GPU, use the
  device-native functions below.
- **Install** PyTorch or JAX yourself; fastprox does not depend on them. In
  this repository, `uv sync --group interop` installs both (PyTorch's CPU
  wheels on Linux).

### Device-native image TV (*since 0.3*)

`fastprox.torch.tv_iso`, `tv_aniso` and `fastprox.jax.tv_iso`, `tv_aniso`
solve the problems of `TVIso` and `TVAniso` in tensor operations on the
input's device (CUDA, MPS or the CPU), with no copy to the host and no
change to the C API.

```python
z = fpt.tv_iso(x_cuda, 0.1, tol=1e-5)                 # last 2 axes, batched
z, info = fpt.tv_aniso(x, gamma, ndim=3, weights=w, return_info=True)
z2 = fpt.tv_aniso(x, gamma, ndim=3, weights=w, warm_start=info.dual)
zj = jax.jit(lambda x: fpj.tv_iso(x, 0.1))(xj)          # jit, grad, vmap
```

- **Arguments**: `x`, `gamma`, `ndim` (2 or 3), `weights`
  (one per axis or one for all), `tol` and `max_iter` as for the C kernels,
  `warm_start` (a dual in the C kernels' form, so the duals of `TVIso` and
  `TVAniso` work too), `return_info`, `deriv_tol` and `deriv_max_iter`.
  `info` is an `IterInfo` whose `gap` and `dual` live on the device.
- **Steps per image**: `gamma` is a number or a tensor (array) that
  broadcasts to the batch shape `x.shape[:-ndim]`, one step per image, as
  DeepInverse's denoisers take one $\sigma$ per sample: for `x` of shape
  (B, C, H, W), a `gamma` of shape (B, 1) gives each sample its own step
  for all its channels. Each image is the one solved with its own scalar
  step, and the gradient in `gamma` has `gamma`'s shape (summed over what
  it broadcast over). PyTorch rejects negative or non-finite steps. When
  every image has the same step the projections are those of one scalar;
  distinct steps cost 6% (isotropic) to 24% (anisotropic) more per
  iteration on MPS.
- **Spatial weights** (*since 0.3*): `tv_iso(..., pixel_weights=alpha)`
  and `tv_aniso(..., edge_weights=beta)`, the problems of
  `TVIso(pixel_weights=)` and `TVAniso(edge_weights=)`. `alpha` broadcasts
  to `x.shape` and `beta` has shape (ndim, ...) with the rest broadcasting
  to `x.shape`, so one map serves every image, or each image has its own
  (a regularisation map predicted by a network, say); both are
  differentiable, their gradients summed back to their shapes, and each
  image matches the C kernel with its own map. They multiply the radii of
  the projection, the gap and the flat test, as in the C kernels; an edge
  of zero weight takes the sign of its jump, as there. PyTorch rejects
  negative or non-finite weights (JAX cannot check traced values). On MPS
  (8 × 512², float32, tol 1e-6) a forward costs 7% (isotropic) and 12%
  (anisotropic) more, at the same iterations; the backward is unchanged.
- **Forward**: FISTA on the dual with the C kernels' step, restart test and
  stopping rule (each image stops when its gap is at most
  $\mathrm{tol} \cdot \lVert x \rVert^2 / 2$), so
  isotropic TV takes the C kernel's iterations. The anisotropic C kernel's
  exact 1D steps and polish stay on the CPU, so the device path is a plain
  dual method there: same certificate, more iterations to tight tolerances.
  Agreement with the C kernels:
  $\lVert z - z_C \rVert^2 \le 2 (\mathrm{gap} + \mathrm{gap}_C)$.
- **Backward** (PyTorch: reverse and forward mode; JAX: reverse mode, under
  `vmap` too) in `x`, `gamma` and `weights`: the C kernels' systems on the flat
  regions of the solution, with the flat pixels and edges read off the dual and
  the isotropic curvature capped as there (the bound's margin is
  $\max(10^{-9}, 64\varepsilon)$). Anisotropic TV averages over the regions
  exactly; isotropic TV solves one unknown per region by conjugate gradients
  with the C kernel's Jacobi preconditioner, stopped as there at the relative
  error of $z$ that each image's gap certifies (`deriv_tol` sets a fixed
  residual instead; the floor is $\max(10^{-10}, 100\varepsilon)$). The regions
  are found by label propagation and summed by a sort and a two-level segmented
  scan, so memory is a few images and does not grow with the iterations.
- **Reproducible** run to run on one device: every step is a stencil, a
  gather, a sort or a fixed-order reduction (no atomic scatter-adds). Not
  bit-identical to the C kernels or across devices.
- **Cost** (docs/benchmarks.md): on an Apple M5 Pro GPU (MPS), isotropic TV
  of eight 512² images is about 4× faster than the C kernel on one thread
  and level with it on 15, with a backward of 0.14–0.55 s. Anisotropic TV is
  slower than the C kernel, whose exact 1D steps need about 10× fewer
  iterations; its backward is 60 ms.

## Conventions

- Derivatives are an element of the generalized Jacobian, chosen by the
  structure of the forward pass (see "Kinks" above).
- Exact operators report `gap = 0`; `iters` and `ties` are zero for every
  0.1 operator. Iterative operators report their duality gap and iterations,
  and the nonconvex maps the number of entries on a tie.
- float32 inputs are computed with double-precision accumulators wherever a
  sum spans a row.
