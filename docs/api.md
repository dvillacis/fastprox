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
| Sorted ℓ1 (OWL, SLOPE; *since 1.0*) | $\sum_k w_k \lvert z \rvert_{(k)}$, magnitudes in decreasing order, $w_1 \ge w_2 \ge \dots \ge 0$ | `fp_op_sorted_l1(w)` | `SortedL1(weights)` | w (one per entry, or 1) |
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
  The steps solve for the differences $Dz$ rather than $z$, so the result stays
  exact to rounding when weights and entries span many orders of magnitude
  (tested with weights over 150 orders and rows rescaled by $2^{\pm 600}$).
  For $n = 2$ it is 1D TV. Derivatives are in closed form: the Hessian of the
  prox problem is that tridiagonal matrix minus a rank-one term (Sherman–
  Morrison), so a JVP or VJP costs three $O(n)$ solves.
- **Isotonic regression** minimises $\sum_i w_i (z_i - x_i)^2$, which is the
  Euclidean projection only for unit weights.
- **The sorted ℓ1 norm** (*since 1.0*) sorts $\lvert x \rvert$ in decreasing
  order (ties by index), subtracts $\gamma w$, fits a nonincreasing sequence
  by pool-adjacent-violators, clips at 0 and puts the signs and the order
  back (Bogdan, van den Berg, Sabatti, Su and Candès, *Ann. Appl. Stat.* 9,
  2015, Algorithm 3), in $O(n \log n)$: 1.9× isotonic regression plus a
  NumPy sort on rows of $10^4$. With equal weights it is soft thresholding,
  bit for bit. The derivatives average within the pooled blocks: on a block
  $B$ of positive value, $\partial \lvert z_i \rvert / \partial \gamma$ is minus the
  mean of $w$ over $B$ and $\partial \lvert z_i \rvert / \partial w_k = -\gamma /
  \lvert B \rvert$ for each weight position $k$ in $B$. It is absolutely
  symmetric, so on singular values it gives the weighted nuclear norm
  $\sum_k w_k \sigma_k$ (`WeightedNuclearNorm`). Kinks (two blocks with equal
  values, a block at 0) are not reported, as for isotonic regression.
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
  1e-6), which certifies
  $\lVert z - z^\star \rVert \le \sqrt{\mathrm{tol}}\,\lVert x \rVert$ (so
  `tol=1e-10` gives $z$ to 1e-5 relative; see "Inexact proxes inside outer
  solvers"), or after `max_iter`
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
- **Special values.** A NaN anywhere in an image makes that whole image NaN,
  and its dual; ±Inf is rejected with `FP_EINVAL`. Such an image is not
  solved (*since 1.0*): it reports a NaN gap and 0 iterations and does not
  count as unconverged. The rest of the batch is unaffected, and
  `fp_info.gap`, a sum, is NaN.
  Images are limited to 2³¹ − 1 entries.
- **Scale** (*since 1.0*). An image whose largest entry is outside
  $[2^{-64}, 2^{65})$ is solved rescaled by a power of two and scaled back,
  all exact. The result does not depend on the scale of $x$:
  $\operatorname{prox}(c x, c\gamma) = c \operatorname{prox}(x, \gamma)$ for
  $c = 2^k$, up to rounding. Convergence is decided on the rescaled problem.
  The reported gap scales by $c^2$, so it may be $+\infty$ when that exceeds
  the largest double. Before, images beyond about $10^{154}$ (or below
  $10^{-154}$) were returned unsolved or ran to `max_iter`.

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
  are accepted and have no effect. A signal containing a NaN is NaN, with a
  NaN dual; it reports a NaN gap and 0 iterations and does not count as
  unconverged (*since 1.0*).
- **Derivatives** are exact with `tol` = 0: $z$ is constant on the regions joined by edges
  with $\gamma w_e > 0$ and equal values (to rounding), and each region's
  value is its mean of $x$ minus $\gamma \sum_e w_e s_e / \lvert R \rvert$ over
  its boundary edges, $s_e$ the sign of the jump. The Jacobian averages over
  the regions, and the $\gamma$ and weight derivatives come from the boundary
  signs, in $O(n + m)$. With `tol` > 0 the solver may stop a part at its
  mean before it is split, and the derivatives are those of the regions it
  stopped at, accurate only as far as those regions are the solution's; use
  `tol` = 0 when derivatives matter.
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
| soft thresholding, elastic net | $\lvert x_i \rvert$ equal to a positive threshold | thresholded: 0 |
| Huber | $\lvert x_i \rvert = \delta + \gamma\lambda$ | inner (quadratic) branch: $\delta/(\delta + \gamma\lambda)$ |
| box, nonnegative orthant | $x_i$ on a bound | interior: 1 (an entry with $\mathrm{lo}_i = \mathrm{hi}_i$ is fixed: 0, *since 1.0*) |
| group ℓ2, ℓ2 norm | $\lVert x_g \rVert$ equal to the threshold | group set to zero: 0 |
| ℓ2 ball | $\lVert x \rVert = r$ | inside: identity |
| simplex, ℓ1 ball | $x_i$ (or $\lvert x_i \rvert / w_i$) equal to the threshold $\tau$ | entry off the support |
| ℓ1 ball | weighted ℓ1 norm of $x$ equal to $r$ | inside: identity |
| ℓ∞ norm | $\lvert x_i \rvert = \tau$ | entry not clipped: 1 |
| ℓ∞ norm | $\lVert x \rVert_1 = \gamma\lambda$ | $z = 0$: 0 |
| maximum | $x_i = \tau$ | entry not clipped: 1 |
| sum of the k largest | $x_i - \mu$ on 0 or $\gamma\lambda$ (an end of the scaled cap) | entry at the shared level $\mu$ |
| hyperplane, half-space ∩ box | $x_i - \mu a_i$ on a bound | interior (free), unless $\mathrm{lo}_i = \mathrm{hi}_i$ (fixed, *since 1.0*) |
| half-space ∩ box | $a^\top P_{\mathrm{box}}(x) = b$ | box projection (constraint inactive) |
| second-order cone | $\lVert y \rVert = t$ (on the cone), or the origin | inside: identity |
| second-order cone | $\lVert y \rVert = -t > 0$ (on the polar cone) | $z = 0$: 0 |
| log barrier | none for $\gamma\lambda > 0$; at $\gamma = 0$, $x_i = 0$ | 1 |
| ℓ1 over a box | $\lvert x_i \rvert$ equal to the threshold; $\operatorname{soft}(x_i)$ on a bound | thresholded: 0; interior (soft's Jacobian), unless $\mathrm{lo}_i = \mathrm{hi}_i$ (fixed: 0, *since 1.0*) |
| group Huber, SCAD, MCP | $\lVert x_g \rVert$ at a breakpoint of the scalar map | the scalar map's rule (Huber: inner branch; SCAD, MCP: the branch below) |
| group Huber, SCAD, MCP | $x_g = 0$ | $\varphi'(0) I$, the limit of $\varphi(s)/s$ |
| tree-structured group ℓ2 | $\lVert z_g \rVert$ equal to the threshold at a shrink; $\lvert x_i \rvert$ at the ℓ1 threshold | group set to zero; thresholded |
| 1D TV ball | $\mathrm{TV}(x) = \alpha$ | inside: identity |
| 1D TV ball | adjacent segments with equal values | as for 1D TV, at the step $\gamma^\star$ |
| 1D TV, ℓ2 of differences | the dual minimiser on the sphere, $\lVert u_0 \rVert = \gamma$ | inside: piecewise averaging |
| isotonic regression | adjacent blocks with equal means | blocks not pooled (pooling happens only on a strict violation) |
| sorted ℓ1 | adjacent blocks with equal values; a block at 0 | blocks not pooled; the block at 0 thresholded |
| 1D TV | adjacent segments with equal values | the segments the solver returned, each averaged on its own |
| anisotropic TV | an edge with no jump whose dual is on its bound | not joined: its two sides averaged apart, with the sign of the dual |
| ℓ0, ℓq, log penalty | $\lvert x_i \rvert$ on the threshold (two minimisers) | the branch of the tie rule: nonzero by default, 0 with `FP_TIE_ZERO` |
| SCAD, MCP | $\lvert x_i \rvert$ at a breakpoint | the branch below it (at $\gamma\lambda$: thresholded) |
| isotropic TV | an entry with a vanishing gradient whose dual is on the sphere | not flat: $H_p$ from its gradient (flat only when the gradient is at the rounding level) |

A zero threshold ($\lambda = 0$, $w_i = 0$, $c_g = 0$ or $\gamma = 0$) is not a
kink: the map is the identity there and its Jacobian is 1 ($1/(1+\gamma\ell_2)$
for the elastic net), also at $x_i = 0$, where $z$ does not move with $\gamma$
or the parameters.

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
  NULL, 0)`. On failure (`FP_EPARAM` for a value outside −1..1 or a bad
  `tol`, `FP_ESHAPE`, `FP_EINVAL`) the workspace is unchanged: an override
  set earlier still applies (since 1.0).
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

### Inexact proxes inside outer solvers (*since 1.0*)

The iterative operators (`TVAniso`, `TVIso`, `GraphTV` and the device-native
image TV) return an approximate prox with a certificate. This section says
what the certificate bounds, and how accurate the inner solves must be for
an outer method such as proximal gradient or FISTA to keep its rate.

- **What `tol` certifies.** The prox objective
  $\tfrac12 \lVert z - x \rVert^2 + \gamma g(z)$ is 1-strongly convex, so
  $\lVert z - z^\star \rVert^2 \le 2 \cdot \mathrm{gap}$. An image stops when
  its gap is at most $\mathrm{tol} \cdot \lVert x \rVert^2 / 2$, which
  certifies a relative error of $z$ of $\sqrt{\mathrm{tol}}$:

  | `tol` | $\lVert z - z^\star \rVert / \lVert x \rVert$ at most |
  | --- | --- |
  | 1e-6 (the default) | 1e-3 |
  | 1e-8 | 1e-4 |
  | 1e-10 | 1e-5 |
  | 1e-12 | 1e-6 |

  An outer solver that needs $z$ to 1e-5 therefore needs `tol=1e-10`. Graph
  TV's `tol` bounds each node instead, within
  $\mathrm{tol} \cdot (\max x - \min x)$, but its gap certifies $z$ the same
  way.
- **`ztol=`** (Python: every `prox`, `prox_conj` and `solve` of these
  operators, the one-shot functions, and `fastprox.torch`/`fastprox.jax`)
  asks for the relative error of $z$ directly:
  $\lVert z_i - z_i^\star \rVert \le \mathrm{ztol}\,\lVert x_i \rVert$ for
  every image $i$. It passes `tol = ztol**2`; for graph TV it passes the `tol`
  whose per-node bound gives the same guarantee,
  $\mathrm{ztol} \min_i \lVert x_i \rVert / (\sqrt{n}\,(\max x_i - \min x_i))$.
  `ztol` and `tol` are exclusive, and a positive `ztol` below about 1e-154,
  whose square underflows, raises `ValueError`.
- **`IterInfo.zerr`** is the error certified for the worst image,
  $\max_i \sqrt{2\,\mathrm{gap}_i} / \lVert x_i \rVert$, in `ztol`'s units:
  below `ztol` when the call converged, above it when an image stopped at
  `max_iter`. It is NaN when an image holds a NaN. In float32 (JAX's
  default) the gap is a difference of two sums of the size of
  $\lVert x \rVert^2$, so its rounding, and `zerr`'s, can reach a fraction of
  a percent at tight tolerances. In C the per-image gaps and
  iterations, and each image's certified error, come from
  `fp_workspace_image_info(ws, gaps, iters, zerrs, count)`; `fp_info.gap` is
  the gaps' sum. The errors are computed at the scale the image was solved
  at, so they hold where the gap itself overflows or underflows.

**What outer methods need.** Schmidt, Le Roux and Bach (*Convergence rates of
inexact proximal-gradient methods for convex optimization*, NeurIPS 2011,
arXiv:1109.2415) analyse $x_k = \operatorname{prox}_{h/L}(y_{k-1} - \nabla
g(y_{k-1})/L)$ with the prox solved to $\varepsilon_k$ in the objective
$\tfrac{L}{2}\lVert x - y \rVert^2 + h(x)$ (their (4)). With $\gamma = 1/L$
that objective is $L$ times fastprox's, so $\varepsilon_k = L \cdot
\mathrm{gap}_k$, and the quantity their bounds use, $\sqrt{2\varepsilon_k/L}$, is
$\sqrt{2\,\mathrm{gap}_k}$: the certified distance $\lVert z_k - z_k^\star
\rVert$, at most $\mathrm{ztol}_k \lVert x_k \rVert$ for the prox's input
$x_k$. Without gradient errors:

- **Proximal gradient** (their Proposition 1): $f(\bar x_k) - f^\star \le
  \frac{L}{2k}\bigl(\lVert x_0 - x^\star \rVert + 2A_k + \sqrt{2B_k}\bigr)^2$
  for the average $\bar x_k$ of the iterates, with $A_k = \sum_{i \le k}
  \sqrt{2\varepsilon_i/L}$ and $B_k = \sum_{i \le k} \varepsilon_i/L$. The
  $O(1/k)$ rate holds when $\sum_k \sqrt{\varepsilon_k}$ converges, for
  instance when $\mathrm{ztol}_k$ falls as $O(1/k^{1+\delta})$.
- **Accelerated proximal gradient** with $\beta_k = (k-1)/(k+2)$ (their
  Proposition 2): $f(x_k) - f^\star \le
  \frac{2L}{(k+1)^2}\bigl(\lVert x_0 - x^\star \rVert + 2\tilde A_k +
  \sqrt{2\tilde B_k}\bigr)^2$ with $\tilde A_k = \sum_{i \le k} i
  \sqrt{2\varepsilon_i/L}$ and $\tilde B_k = \sum_{i \le k} i^2
  \varepsilon_i/L$. The $O(1/k^2)$ rate needs $\sum_k k\sqrt{\varepsilon_k}$
  to converge, for instance $\mathrm{ztol}_k = O(1/k^{2+\delta})$: a stronger
  condition, which is why FISTA stalls at a fixed loose tolerance.

**Nonconvex penalties** (SCAD, MCP, $\ell_0$, $\ell_q$, the log penalty, and
their group and spectral forms). The bounds above are for convex penalties.
For these:

- **A minimiser may not exist.** These penalties are bounded, so with a data
  term that sees only part of the entries the objective is not coercive. On
  matrix completion of MovieLens ratings (300 × 300, 30% observed) with
  spectral MCP, the objective kept falling while the iterates grew without
  bound, for FISTA and for proximal gradient at $0.99/L$ alike: after 16 000
  FISTA iterations $\lVert W \rVert_F$ was 12 000 and unobserved entries
  exceeded 2 000, on centred ratings between −2.7 and 1.3. The nuclear norm settled within
  250 iterations. `bench/repro_u6_spectral_mcp.py` reproduces it. Such a
  model needs a coercive term as well, for instance a small nuclear or
  Frobenius norm.
- **The outer step.** Proximal gradient reaches critical points with a step
  below $1/L$ for a nonconvex penalty: this is PALM (Bolte, Sabach and Teboulle,
  *Math. Program.* 146, 2014) with one block, as presented by Chambolle and
  Pock (*Acta Numerica* 25, 2016, Section 6.2). A convex penalty allows steps
  up to $2/L$. On 50% matrix completion of a rank-4 30 × 25 matrix ($L = 1$)
  with spectral MCP ($\lambda = 0.5$, $b = 3$), steps of $0.5/L$ to $1.5/L$
  converged. At $1.99/L$ the objective rose at half the iterations and the
  iterates did not approach the matrix, while the nuclear norm still
  converged there. FISTA needs a step of at most $1/L$ even with a convex
  penalty: at $1.5/L$ it diverged on that problem with MCP and with the
  nuclear norm. FISTA is not a descent method: its objective rose at up to a
  third of the iterations at smaller steps, with either penalty. For a
  nonconvex penalty it also has no convergence guarantee.
- **The prox's step range.** The outer step is the prox's $\gamma$, so it must
  also satisfy SCAD's $a > 1 + \gamma \max w$ and MCP's $b > \gamma \max w$
  (`FP_EPARAM`, `ValueError` in Python).
- **Derivatives.** On MCP's middle piece, $\gamma w_i \lambda < \lvert x_i
  \rvert < b \lambda$, the derivative is $b / (b - \gamma w_i)$. SCAD's
  concave branch has $(a - 1)/(a - 1 - \gamma w_i)$. Both are unbounded as
  the parameter falls to the edge of its range, and a hypergradient through
  the prox is amplified as much: $b \ge 2 \gamma \max w$ keeps MCP's gain
  at most 2.

**Warnings** (*since 1.0*). A call that stops at `max_iter` before its
tolerance raises `fastprox.ConvergenceWarning`, with the error it did
certify (`zerr`). This holds for `prox`, `prox_conj` and `solve`, the
one-shot functions, `fastprox.torch` and `fastprox.jax`; JAX warns from the
host through `jax.debug.callback`, under `jit` too. `solve(..., warn=False)`
and the framework functions' `warn=False` silence one call, which still
reports `converged` and `zerr`; the standard warning filters silence the
rest. `ConvergenceWarning` is a subclass of `fastprox.AccuracyWarning`.

**`fp.inner_accuracy(target, factor=0.1)`** sets the accuracy once for a
block of code. `target` is the relative accuracy the outer solver needs on
$z$. Every iterative call in the block that sets neither `tol` nor `ztol`
uses `ztol = factor * target`, so the outer code needs no other change:

```python
with fp.inner_accuracy(1e-5):                 # ztol = 1e-6 inside
    for k in range(iters):
        z, ctx = op.prox(y - step * grad(y), step * lam, keep=True, ctx=ctx, warm=True,
                         max_iter=10 ** 5)
        ...
```

A call whose own `tol` or `ztol` certifies less raises `AccuracyWarning`
with both numbers, and one that stops at `max_iter` raises
`ConvergenceWarning`, naming the block's ztol. Tight inner tolerances need
more iterations than the default `max_iter` of 1000, as above; the
warning says when. Blocks nest. The state is a context variable, so each
thread and asyncio task has its own; the PyTorch and JAX wrappers capture
it when a call is set up, and under `jax.jit` the block in force when the
function is traced applies.

**In practice.** Either take `ztol` a factor of 10 below the accuracy the
outer method needs on $z$ (what `inner_accuracy` does), so that the inner
error never decides the outer stopping test, or decrease it with the
iterations as above. In both cases
start each inner solve from the previous one's dual: keep a context and pass
it back with `warm=True`, which makes tight inner solves cheap near
convergence. On a 96² inpainting problem with isotropic TV, 200 FISTA
iterations took 10–14× fewer inner iterations warm than cold at the same
`ztol` (1e-3 to 1e-5).

```python
op = fp.TVIso()
ctx = None
for k in range(1, iters + 1):
    v = y - step * grad(y)
    z, ctx = op.prox(v, step * lam, keep=True, ctx=ctx, warm=True, ztol=0.1 * target)
    ...                                   # ctx.info.zerr: the error certified for this z
```

**Derivatives at a loose tolerance** (*since 1.0*). The derivatives read the
structure saved at the returned point, which is exact only when that point
has the solution's regions: anisotropic TV once its polish certified them
(from about 1e-8 to 1e-10 on 128² images), graph TV when no part was stopped
by `tol`, tree TV always. Isotropic TV caps its curvature at a smoothing that
vanishes with the gap, so its derivative carries an error of the order of the
certified error of $z$: on 2 × 128² images 3% after a 1e-6 solve, 1e-4 after
a 1e-10 one.

- **`ctx.deriv`** is a `DerivInfo(residual, iters, exact, smoothing)` after
  every derivative; `fp_workspace_deriv_exact(ws, &exact, &smoothing)` in C.
- **A warning.** A derivative that is not exact, at a certified error of $z$
  above 1e-4 (inside `inner_accuracy`, above its ztol), raises
  `AccuracyWarning` once per point.
- **`ctx.refine(ztol)`** solves the kept call again, from its saved input and
  its dual as a warm start, to `ztol`, and returns the refined $z$; the
  derivatives then read the refined point. `jvp(..., refine=ztol)` and
  `vjp(..., refine=ztol)` refine first. In C,
  `fp_workspace_refine(op, ws, tol, max_iter, out, info)`, bit for bit a
  fresh solve seeded with the kept dual. Graph TV refines to `tol=0`.
- **Cost.** A refine costs about what solving to the tighter tolerance from
  the start would have: the warm start saves the iterations the loose solve
  already made. On 2 × 128² images, from a 1e-6 solve, anisotropic TV refined
  to 1e-12 in 34 iterations (a fresh solve took 66) and its derivative became
  exact; isotropic TV, whose FISTA converges like $1/k^2$, took 31 036
  iterations to 1e-10 (a fresh solve 32 432). So refine when the point the
  derivatives read must be tighter than the one already solved, and otherwise
  solve to the tolerance the derivative needs in the first place.

## C API

### Arrays

```c
typedef struct {
    void          *data;
    fp_dtype       dtype;    /* FP_F32 or FP_F64 */
    int            ndim;     /* 0 (scalar) to FP_MAX_NDIM */
    const int64_t *shape;
    const int64_t *strides;  /* in elements, may be negative; NULL = C order */
    fp_device      device;   /* FP_DEVICE_CPU; FP_DEVICE_CUDA needs the plugin */
} fp_array;
```

The library never frees `data`. The product of the extents (zero extents
aside) and the largest element offset, $\sum_d (\text{shape}_d - 1)
\lvert\text{strides}_d\rvert$, must not exceed $2^{60} - 1$; larger
descriptors return `FP_ESHAPE` (*since 1.0*).

`out` may be the input array itself: the same data pointer, shape and
strides, as in `fp_prox(op, g, &x, &x, ...)`. Any other overlap between the
byte ranges of an input and an output returns `FP_EINVAL` before any work
(*since 1.0*; before, it was undefined). That includes views that interleave
without sharing an element, such as `a[0::2]` and `a[1::2]`. An output
dimension of extent > 1 with stride 0 also returns `FP_EINVAL`. The same
rules apply to `dx` and `dout` in `fp_jvp`, to `g` and `gx` in `fp_vjp` and to
`x` and `out` in `fp_value`. The Python package copies its input when it may
share memory with `out` in another layout, so Python callers never see this
error.

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
threads. An operator must stay alive until every workspace created for it has
been freed: workspaces refer to their operator.

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
- Derivatives use the point saved by the last prox call on the workspace
  (`fp_prox`, `fp_prox_steps` or `fp_prox_conj`), which must have had
  `keep_structure` set. A later prox call on the workspace replaces that
  point, or discards it when the call has no `keep_structure` or fails during
  its work; derivatives then return `FP_EINVAL` until the next kept call. A
  call rejected for its arguments (a negative step, a wrong shape, ...) leaves
  the point as it was. To keep a point while computing other proxes, use a
  second workspace. An empty batch
  with `keep_structure` saves an empty point: derivatives return `FP_OK` and
  write nothing, gradients 0 (since 1.0). Steps are checked on an empty
  batch too.
- After `fp_prox_conj`, derivatives are those of the conjugate prox and $\gamma$
  reads as $\sigma$.
- A workspace is used by one call at a time, derivative calls included; use
  one per thread (see "Threads and determinism").

### One step per row (*since 1.0*)

```c
/* gamma: the batch shape, x's shape without opt.axis (without the image axes
 * for the image operators); float32 or float64, any strides */
fp_prox_steps(op, &gamma, &x, &z, &opt, ws);
fp_workspace_info(ws, &info);                /* fp_info of the call */
```

`fp_prox_steps` is `fp_prox` with one step per row, image or matrix, as
DeepInverse's denoisers take one noise level per image. A stride of 0
broadcasts a step over a dimension, and a single row takes a 0-d array. Each
row gives what `fp_prox` gives on that row alone with its step, bit for bit,
for any thread count. Every step is checked before any work: `FP_EPARAM` for a
negative or non-finite step, or one that breaks the operator's step condition
(SCAD, MCP).

With `keep_structure`, the steps are saved for the derivatives:

- `fp_jvp`'s `dgamma` moves every row's step by `dgamma`.
- `fp_vjp`'s `ggamma` is the sum over rows of each row's gradient in its own
  step.
- The gradient in one row's step is the sum over that row of $g$ times the
  output of `fp_jvp(op, ws, NULL, 1.0, NULL, &t)`, which is
  $\partial z / \partial \gamma$ of every row at its step.

There is no conjugate form. In Python, `gamma` may be an array that broadcasts
to the batch shape, in `prox` (and `prox_from` for spectral operators), and
so may `gamma` in `fastprox.torch.prox` and `fastprox.jax.prox` (below).

### Warm starts and the dual (*since 0.2*)

```c
fp_workspace_get_dual(ws, &q);       /* q: shape (ndim, *x.shape), any float dtype */
fp_workspace_set_dual(ws, &q);       /* the next call starts from q */
fp_workspace_set_warm(ws, 1);        /* every call starts from the previous dual */
```

Only image operators have a dual; for other operators these return
`FP_EINVAL`, as does `get` before any call. `set` rejects ±Inf and stores NaN entries as 0 (*since 1.0*), so a dual read by `get` can always be set back.

```c
fp_workspace_set_deriv_tol(ws, 1e-8, 500);   /* CG of isotropic TV's fp_jvp/fp_vjp */
double residual; int iters;
fp_workspace_deriv_info(ws, &residual, &iters);  /* worst over the batch, last call */
```

### Values (*since 0.5*)

`fp_value(op, x, opt, out)` writes $g(x)$ into `out` (`FP_F64`), one
value per row along `opt->axis`, or one per image for the image operators
(anisotropic and isotropic TV, graph and tree TV). The shape of `out` is the
shape of $x$ without the axis, or without the image axes; it has `ndim` 0
for a single row or image. A row or image containing a NaN gives NaN.

The projections give the indicator of their set. The value is 0 when every
constraint of the set is violated by at most
$\mathrm{tol} \cdot \max(1, \max_i \lvert x_i \rvert)$, with
$\mathrm{tol}$ = `opt->tol`, and `INFINITY` otherwise. This tolerance
absorbs the rounding of the projections' own outputs. $\ell_1$ over a box and
the log barrier are `INFINITY` outside their domain. Only `opt->axis` and
`opt->tol` are read. In Python, `op.value(x, axis=-1, tol=1e-6)` returns a
float for one row or image, else an array of the batch shape.

With the step $\gamma$, $z = \operatorname{prox}_{\gamma g}(x)$ minimises
$g(y) + \lVert y - x \rVert_2^2 / (2\gamma)$. `tests/python/test_value.py`
checks every operator's value against its prox this way.

### Spectral operators (*since 1.0*)

An operator on the singular values or eigenvalues of each matrix of a batch:
$g(X) = h(s)$ for $X = U \operatorname{diag}(s) V^\top$ (thin SVD,
$k = \min(m, n)$) or $X = Q \operatorname{diag}(s) Q^\top$, with
$\operatorname{prox}_{\gamma g}(X) = U \operatorname{diag}(z) V^\top$ and
$z = \operatorname{prox}_{\gamma h}(s)$. The caller computes the
decomposition and the matrix products, so the library needs no LAPACK or
BLAS; the prox is `fp_prox` on $s$, of shape `(..., k)` along the last axis.
`fp_op_spectral_check(op, kind)` tells whether `op` qualifies:

| `kind` | Needs | Operators |
| --- | --- | --- |
| `FP_SPECTRAL_SVD` | absolutely symmetric | soft thresholding (the nuclear norm), elastic net, Huber, $\ell_0$, $\ell_q$, SCAD, MCP, log penalty; $\ell_1$ ball (nuclear-norm ball), $\ell_\infty$ prox (spectral norm), box $[-c, c]$ (spectral-norm ball), $\ell_2$ ball (Frobenius ball) |
| `FP_SPECTRAL_EIG` | permutation invariant | all of the above, and the nonnegative orthant (PSD cone), any box (eigenvalue bounds), simplex (spectraplex), log barrier ($-\lambda \log\det$) |

Weighted operators qualify only with equal weights.

`fp_spectral_jvp(op, ws, kind, s, z, a, dgamma, dparams, b, r)` gives the
derivative, after `fp_prox(op, gamma, s -> z)` with `keep_structure`. From
$A = U^\top H V$ (or $Q^\top H Q$), shape `(..., k, k)`, it writes $B$ and,
for singular values, $r$ (`(..., k)`, may be NULL), so that the derivative in
the direction $H$ is

$$U B V^\top + (I - U U^\top) H V \operatorname{diag}(r) V^\top \quad (m \ge n),
\qquad U \operatorname{diag}(r) U^\top H (I - V V^\top) \quad (m < n),$$

or $Q B Q^\top$ for eigenvalues. $B$ holds the divided differences of the
prox on the spectrum (Ding, Sun, Sun and Toh, *Spectral operators of
matrices*, 2018, Theorem 3.3). Where two values coincide or one is zero, it
holds their limits, taken from the Jacobian of the prox on $s$: exact where
singular values repeat, where autograd through an SVD returns NaN or a wrong
gradient. Values within $\varepsilon^{1/3} \max_l \lvert s_l \rvert$ of each
other count as tied (docs/spec.md, "Spectral operators").

- **The VJP.** The map is self-adjoint, so a cotangent $G$ goes through the
  same call with $A = U^\top G V$. The gradients in $\gamma$ and the
  parameters are `fp_vjp` on $\operatorname{diag}(A)$.
- **Kinks.** At a singular value on a kink of the scalar map, the map is not
  differentiable. Examples are MCP at $\gamma\lambda$ and $b\lambda$, SCAD at
  $(1 + \gamma)\lambda$ and $a\lambda$, and soft thresholding at
  $\gamma\lambda$. There the JVP is one element of the B-subdifferential, and
  stays adjoint to the VJP:
  - at values exactly on the kink, the scalar map's branch below, for each
    value;
  - at a tie that rounding splits across the kink, as an SVD of such a
    matrix does, the limit of the Jacobian at points with the pair split
    in the same basis. It depends on the singular vectors chosen inside the
    tie: rotating them within the tie moved the JVP by 12–20% on a 7 × 5 example.
- **Cost.** $O(k^2)$ per matrix and one JVP on the spectrum per row; the
  first call on a workspace sizes about $8k$ doubles of scratch per thread,
  and later calls allocate nothing.

### Devices (*since 1.0*)

The GPU backend (docs/spec.md, "GPU backend") comes in stages. Its CUDA kernels
will live in a separate library, the plugin `libfastprox_cuda`
(`fastprox_cuda.dll` on Windows). The C library loads it the first time a
device is queried, from the first of:

1. the path in the environment variable `FASTPROX_CUDA_PLUGIN`, when it is set
   (an empty value turns the plugin off);
2. the directory of `libfastprox`, or of the Python extension that links it;
3. the system's library search path.

A plugin is used only when it comes from the same fastprox version. No plugin
has kernels yet, so arrays on a device still return `FP_EINVAL`. The functions
that answer device questions already work on the CPU:

```c
fp_device_available(FP_DEVICE_CUDA);        /* 1 with the plugin and a device */
int rpo;
fp_op_device_check(op, FP_DEVICE_CPU, &rpo); /* FP_OK; rpo = 1: a row uses one thread */
fp_prox(op, gamma, &x, &z, &opt, ws, NULL);
fp_info info;
fp_status s = fp_workspace_info(ws, &info);  /* that call's info; FP_ENOTCONV if it stopped early */
fp_vjp_device(op, ws, &g, &gx, &ggamma, &gtheta);  /* ggamma: a 0-d array */
```

- **`fp_op_device_check`** says whether an operator runs on a device. Its
  `row_parallel_only` flag is 1 when one row never uses more than one thread,
  so a batch of a few long rows gains nothing from more threads. On the CPU
  that is every operator with rows except the elementwise maps and
  anisotropic TV.
- **`fp_workspace_info`** returns the `fp_info` and status of the last
  `fp_prox` or `fp_prox_conj` call on the workspace. On a device, it lets a
  call pass `info = NULL` and not wait for its stream. It returns `FP_EINVAL`
  before a call and after one that failed.
- **`fp_vjp_device`** is `fp_vjp` with the step's gradient written to a 0-d
  float32 or float64 array on the workspace's device.
- **`fp_workspace_set_stream` and `fp_workspace_get_stream`** take a
  `cudaStream_t` as `void *`. They are for device workspaces only and return
  `FP_EINVAL` on the CPU.
- **`FP_EDEVICE`** is a new status: a device error, from the call itself or
  from earlier work on the stream.

### Conjugates

`fp_prox_conj(op, σ, y, out, ...)` computes
$\operatorname{prox}_{\sigma g^\ast}(y) = y - \sigma \operatorname{prox}_{g/\sigma}(y/\sigma)$.
$\sigma = 0$ returns $y$. With `keep_structure`, the derivatives there are those of the identity: $J = I$ and zero parameter gradients. The $\sigma$-derivative, which is not defined there for every operator, is NaN (`fp_vjp`'s `ggamma`; `fp_jvp`'s output when `dgamma` ≠ 0). For a smooth $g^\ast$ it would be $-\nabla g^\ast(y)$; for a homogeneous $g$ the conjugate jumps from $y$ to $P_C(y)$ (since 1.0, audit A3).

Since 1.0 the computation depends on the operator (audit A1).

- **Positively homogeneous $g$ and cone indicators.** This covers soft thresholding, ℓ2 and group ℓ2, the tree-structured norms, ℓ∞, the maximum, the sums of the largest entries, every TV operator, the nonnegative orthant, the second-order cone and isotonic regression. There $g^\ast$ is the indicator of a set $C$, and the result, $y - \operatorname{prox}_g(y)$, is the projection onto $C$. It is the same for every $\sigma > 0$, computed without scaling, so it is exact even at $\sigma = 10^{-320}$. Its derivative in $\sigma$ is 0. After the call, an image operator's dual is that of $\operatorname{prox}_g(y)$ at unit step, and $\sum_a q_a$ is the output; this is the same dual for every $\sigma$.
- **The other operators** evaluate the formula as written. $1/\sigma$ must lie in `fp_prox`'s step range, and every $y_i/\sigma$ must be finite in the array's dtype; otherwise the call returns `FP_EPARAM`. The quotients are checked for every $\sigma < 1$ in float32, and in float64 for $\sigma < 2^{-64}$. Above that, a float64 quotient can overflow only for $\lvert y_i \rvert$ beyond about $10^{289}$, and its output is then $\pm\infty$ or NaN. A check for every $\sigma < 1$ cost 15–30% on the cheapest conjugates.

### Errors

| Status | Meaning |
| --- | --- |
| `FP_OK` | success |
| `FP_EINVAL` | invalid argument, ±Inf in the input of an operator where it is undefined, missing saved structure, mismatched workspace |
| `FP_EDTYPE` | dtype other than float32/float64, or mismatched dtypes |
| `FP_ESHAPE` | mismatched shapes or parameter lengths |
| `FP_ENOMEM` | out of memory, or a requested size too large to allocate |
| `FP_EPARAM` | invalid parameter or step (negative $\gamma$, negative radius, …), a step beyond the operator's range ($\gamma$ times its largest parameter above $2^{960}$, since 1.0), a $\sigma$ too small for the conjugate's formula, or a constraint set that is empty for the axis length (hyperplane ∩ box, the sum of $k > n$ largest entries) |
| `FP_EINTERNAL` | any other internal error |
| `FP_EDEVICE` | (*since 1.0*) a device error, from the call or from earlier work on its stream |

No function prints, aborts or keeps global error state. NaN in the input
gives NaN in the affected output: the entry for elementwise operators, the
row for the others.

### Threads and determinism

`opt.threads` sets the number of threads (0 = `fp_default_threads()`, 1 = serial).
Thread counts above four times the CPUs the process may run on (and above 4096)
are reduced to that cap, for `opt.threads` and for the environment variables
alike (since 1.0). A row runs on one thread, so a row operator's workspace
holds scratch for at most as many threads as it has rows.
The default is the environment variable `FASTPROX_NUM_THREADS`, else the first
number in `OMP_NUM_THREADS`, else the CPUs the process may run on; it is read
once.
Results, including parameter gradients, are bit-identical for any thread
count. Rows run on fastprox's own thread pool, not OpenMP, so no OpenMP
runtime is linked. The pool's threads spin for 50 µs after a call before
sleeping, so that back-to-back calls find them awake. The environment variable
`FASTPROX_SPIN_US`, read once, sets this time in microseconds, from 0 (sleep
at once: the least CPU, but about 10–100 µs more latency on the next call) to
100 000 (*since 1.0*). A call waits only for the workers that joined it, not
for those still waking up, and its pieces scale with the thread count (*since
1.0*). When several threads call fastprox at once, one call uses the pool
and the others run serially, which avoids oversubscription; a child process
after `fork()` builds its own pool. Builds with `FASTPROX_THREADS=OFF` are
serial and `fp_default_threads()` returns 1.

No two calls on one workspace may run at the same time, whatever its
constness: `fp_jvp`, `fp_vjp`, `fp_vjp_device` and `fp_spectral_jvp` take it
`const` because they keep its saved point, but they use its scratch and write
their results' bookkeeping into it. Give each thread its own workspace; an
`fp_op` can be shared by any number of threads.

### Building and linking

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build && cmake --install build --prefix /usr/local
```

Then `find_package(fastprox)` and link `fastprox::fastprox`, or use
`pkg-config --cflags --libs fastprox`. Options: `FASTPROX_THREADS`,
`BUILD_SHARED_LIBS`, `FASTPROX_BUILD_TESTS`, `FASTPROX_SANITIZE`,
`FASTPROX_FUZZ`, `FASTPROX_STATIC_LIBSTDCXX` (on by default: libstdc++ and
libgcc linked statically on Linux with GCC).

### ABI stability (*from 1.0*)

From 1.0 to 2.0 the C ABI only grows. A program built against 1.x runs
with any later 1.y library without being rebuilt:

- No function changes its signature or goes away.
- The layouts of `fp_array`, `fp_options` and `fp_info` stay as they are,
  and no enumerator changes its value. New options come as new functions,
  as the workspace setters did.
- New functions, operators and enumerators may be added, such as new
  `fp_device` values for the GPU backend. A program that uses one needs
  the release that added it: check `FP_VERSION_MINOR` when building, or
  `fp_version_string()` at run time.

`FP_ABI_VERSION` and `fp_abi_version()` give the major version, which is
also the soname's (`libfastprox.so.1`). Only a release that breaks the ABI
bumps it. Before 1.0 a minor version could break the ABI (0.5 renamed
`fp_op_log` to `fp_op_log_penalty`).

CI holds this: the `abi` job builds the library with debug information
and compares it with the last release using `abidiff` from libabigail.
It fails on any change but an addition, judged on the types of
`fastprox.h`.

## Python

The wheels are abi3 for CPython 3.12 and later. Free-threaded CPython builds
(3.13t, 3.14t) are not supported yet: they cannot load abi3 wheels, and built
from source the extension re-enables the GIL at import, with a RuntimeWarning.

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
g = op.value(z)                                 # g(z) per row (since 0.5)

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
residual, cg_iters = ctx.deriv.residual, ctx.deriv.iters
tv = fp.TVAniso(ndim=3, weights=[1.0, 1.0, 0.5])
v, ctx = tv.prox(np.zeros((32, 32, 16)), 0.3, keep=True)   # ctx.info is an IterInfo
v, ctx = tv.prox(np.zeros((32, 32, 16)), 0.3, keep=True, ctx=ctx, warm=True)
alpha = np.ones((256, 256)); alpha[:, :128] = 0.5          # since 0.3: one weight per pixel
wiso = fp.TVIso(2, pixel_weights=alpha)
v, ctx = wiso.prox(img, 0.2, keep=True)
gx, grads = wiso.vjp(ctx, np.ones_like(img))                 # grads.pixel_weights: (256, 256)
```

- **Inputs** must be float32 or float64 NumPy arrays; other dtypes raise
  `TypeError` unless `copy=True` is passed. Non-NumPy inputs (lists, PyTorch
  and JAX arrays, memoryviews) are converted with `numpy.asarray`: float32 and
  float64 are kept, other real types become float64, and complex inputs raise
  `TypeError` (since 1.0). Strided and reversed views are used without copying.
- **Threads.** Calls on one `Context` are serialised (since 1.0); for parallel
  derivatives use one context per thread.
- **Common keywords:** `axis` (default −1), `out` (preallocated output, may
  be the input), `threads` (0 = `fp.default_threads()`).
- **`keep=True`** returns `(z, ctx)`. Pass `ctx=ctx` to a later `prox` call
  with the same shape to reuse its memory. `ctx.info` holds the certificate
  (`gap`, `iters`, `ties`).
- **`dparams`** is a flat vector of length `op.nparams` or a dict by
  parameter name; **`grads`** exposes the same names as attributes.
- **Derivative keywords** (*since 1.0* on every operator): `jvp` and `vjp`
  take `tol` and `max_iter`, which control iterative derivative solves
  (isotropic TV's conjugate gradients) and are accepted and ignored by every
  other operator, whose derivatives are in closed form. `ctx.deriv` is then
  the solve's `(residual, iterations)`, or `(0.0, 0)`.
  `Soft` also reports `grads.lam`, and `dparams` accepts `lam` too (since 1.0;
  as for `L1Box` and `Lq`, and `radius` for `SpectralNormBall`). An unknown
  name, or a size that is neither the parameter's nor 1, raises `ValueError`
  (since 1.0).
- **Iterative operators** (`TVAniso`): `solve` and the one-shot function
  return `(z, info)`, where `info` is an `IterInfo`: `gap`, `iters`, `ties`,
  `converged` (False when `max_iter` was reached; this is not an error) and
  `dual`. `prox` and `prox_conj` take `tol`, `max_iter`, `warm_start=dual` and
  `warm=True` (start from the dual of the previous call on `ctx`).
- **Errors** map to `ValueError` (invalid arguments, parameters, shapes),
  `TypeError` (dtype), `MemoryError` and `RuntimeError`; an error from the C
  library names its entry point, as in `fp_prox: invalid operator parameter or
  step` (since 1.0). The GIL is released in every call whose cost grows with
  its data: computations, operator constructors, workspace creation, and the
  dual and kink copies.

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

- **`prox(op, x, gamma=1.0, *, axis=None, threads=0, tol=None, ztol=None, max_iter=None, deriv_tol=None, deriv_max_iter=10000, **params)`**
  and `prox_conj(op, y, sigma, ...)`. `op` is an operator, whose
  parameters are then constants, or an operator class with its constructor
  arguments as keywords. `axis` (row operators), `tol`, `ztol` and `max_iter`
  (iterative operators) are passed on when given; `deriv_tol` and
  `deriv_max_iter` (*since 1.0*) go to the derivatives' `tol` and `max_iter`,
  as in `tv_iso`.
- **One step per row** (*since 1.0*). `gamma` may be a tensor or array that
  broadcasts to the batch shape: `x`'s shape without `axis`, the image axes
  or the matrix axes. Its gradient is each row's
  $\langle g, \partial z / \partial \gamma \rangle$, from one more JVP in $\gamma$, summed back
  to `gamma`'s own shape; in PyTorch's forward mode a tangent of that shape
  moves each row's step. `prox_conj` takes a scalar `sigma`.
- **Parameter gradients.** A parameter gets the gradient of the `Grads` field
  of the same name (see `Operator.vjp`): in PyTorch, float tensors that
  require grad; in JAX, JAX float arrays (Python and NumPy values are
  constants). PyTorch raises `ValueError` for a tensor that requires grad
  and has no gradient field (`L0`'s `lam`, `Isotonic`'s weights); JAX, which
  computes every cotangent, returns NaN for it. Integer arguments (groups,
  edges) are constants and must be concrete under `jax.jit`.
- **PyTorch**: a `torch.autograd.Function` with `backward` (the VJP) and
  `jvp` (the JVP, for `torch.autograd.forward_ad`; only for parameters that
  are part of $\theta$, so not `Soft`'s `lam`). Not twice differentiable:
  second-order reverse mode (`create_graph=True`, as in
  `torch.autograd.functional.hvp` and `hessian`) raises `RuntimeError` until
  1.0.x, for `prox`, `prox_conj` and the device-native TV functions. Not
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

### PyProximal (*since 0.5*)

`fastprox.pyproximal.ProxOp(op, dims=None, *, warm=False, value_tol=1e-6,
**prox_kwargs)` turns any operator into a `pyproximal.ProxOperator`, so it
can be used in PyProximal's solvers (proximal gradient, ADMM, primal-dual,
and others).

```python
import pyproximal
from fastprox.pyproximal import ProxOp

g = ProxOp(fp.TVIso(), dims=(64, 64), tol=1e-6, warm=True)
x = pyproximal.optimization.primal.ProximalGradient(f, g, x0, tau=0.9, epsg=0.02)
```

- **Shapes.** PyProximal passes flat vectors, and `dims` is the array
  shape the operator acts on: `(B, n)` is a batch of rows and `(B, H, W)` a
  batch of images. The value is summed over the batch.
- **Methods.** `g(x)` is $g(x)$, from `Operator.value`, with a set's
  indicator taken to `value_tol`. `prox(x, tau)` is
  $\operatorname{prox}_{\tau g}(x)$. `proxdual(x, tau)` is
  $\operatorname{prox}_{\tau g^\ast}(x)$, fastprox's `prox_conj`.
  `tau` must be a positive scalar; PyProximal's arrays of steps, for
  several right-hand sides, are rejected.
- **Warm starts.** With `warm=True`, the iterative image operators start
  each solve from the previous dual.
- **Keywords.** The other keywords (`axis`, `tol`, `max_iter`, `threads`)
  go to `prox` and `prox_conj`; `axis` also goes to `value`.
- **Installation.** PyProximal (LGPL-3.0) is not a dependency: install it
  yourself, or in this repository use `uv sync --group interop`. The adapter
  is written against PyProximal's documented interface.
- **Checks.** `tests/python/test_pyproximal.py` checks the adapter against
  PyProximal's `L1` and with its `moreau` test, and runs a lasso through
  `ProximalGradient`. `examples/pyproximal_tv.py` deblurs an image.

### Spectral operators in Python (*since 1.0*)

`fp.Spectral(op, kind="svd")` applies an operator to the singular values of
each matrix in the last two axes; `kind="eigh"` applies it to the
eigenvalues of symmetric matrices, an input that is not symmetric being
replaced by $(X + X^\top)/2$. `fp.NuclearNorm(lam)`, `fp.SpectralNormBall(r)`,
`fp.PSD()` and `fp.WeightedNuclearNorm(w)` (*since 1.0*: $\sum_k w_k \sigma_k$
with $w$ nonincreasing, `Spectral(SortedL1(w))`) are shorthands. Unequal
weights in a fixed order are refused, since $\sum_i w_i \lvert s_i \rvert$ is
not a function of the singular values; the error points to `SortedL1`.

```python
op = fp.NuclearNorm(0.5)                 # = fp.Spectral(fp.Soft(0.5))
Z, ctx = op.prox(X, gamma, keep=True)    # X: (..., m, n)
dZ = op.jvp(ctx, dX)
gX, grads = op.vjp(ctx, G)               # grads.gamma, grads.weights
op.value(X)                              # lam * sum of singular values
```

- **Decomposition.** `prox` calls `numpy.linalg.svd` (thin) or `eigh` and
  NumPy's matrix products. `prox_from(U, s, Vh, gamma)` takes a decomposition
  you already have. It may hold only the leading triplets when the others
  map to zero, as for singular value thresholding below $\gamma\lambda$, but
  then `jvp` and `vjp` raise: derivatives need the full thin decomposition.
- **Batches.** Any leading axes, float32 or float64, with `threads` as for
  the other operators. The results do not depend on the thread count.
- **Checks.** `tests/python/test_spectral.py` compares the prox with
  singular value thresholding, PSD projection and clipping, and checks
  optimality through `value`. The derivatives are checked against finite
  differences, including matrices with repeated and zero singular values,
  and against the closed-form divergence of Candès, Sing-Long and Trzasko
  (2013). Adjointness and the $\gamma$ and parameter gradients are checked
  too, as is the contrast with autograd through `torch.linalg.svd`.
- **PyTorch and JAX.** `fastprox.torch.prox` and `fastprox.jax.prox` take a
  spectral operator like any other, or `NuclearNorm` and `SpectralNormBall`
  as a class with `lam` or `radius` as a tensor that receives its gradient:

  ```python
  Z = fpt.prox(fp.NuclearNorm(0.5), X_cuda, gamma)    # differentiable in X and gamma
  Z = fpt.prox(fp.NuclearNorm, X, gamma, lam=lam)     # and in lam
  Zj = jax.jit(lambda X: fpj.prox(fp.PSD(), X))(Xj)   # jit, grad, vmap
  ```

  The decomposition (`torch.linalg.svd` or `eigh`, `jax.numpy.linalg.svd`
  or `eigh`) and the matrix products run in the framework on the input's
  device. The prox of the spectrum and the k × k matrix B of the derivative
  are computed on the CPU, so each call copies s, and each backward A and B,
  between the device and the host. PyTorch supports reverse and forward
  mode, forward mode in `x` and `gamma` only; JAX supports reverse mode, and
  its backward recomputes the prox of the spectrum. There is no `prox_conj`,
  and `axis`, `tol` and `max_iter` are rejected. The tests in
  `test_torch.py` and `test_jax.py` check the gradients against finite
  differences and NumPy's operator, and at tied singular values, where
  autograd through the SVD fails.

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
- **Special values** (*since 1.0*, the same on the NumPy, PyTorch and JAX
  paths). An empty batch (any axis of extent 0) returns an empty result with
  gap 0, 0 iterations, converged true and an empty dual, and its gradients
  are empty or 0. An image containing a NaN gives a NaN image: its gap is
  NaN, it takes no iterations and it counts as finished, so it does not by
  itself make `converged` false. The rest of the batch is unaffected. ±Inf
  is rejected (`ValueError`), except under `jax.jit`, where values cannot be
  checked and an image with an Inf is treated as a NaN image.
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
  $\mathrm{tol} \cdot \lVert x \rVert^2 / 2$; `ztol` and `info.zerr` as for the
  NumPy operators), so
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
