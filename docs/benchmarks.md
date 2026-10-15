# Benchmarks

Single-thread time per sample from `bench/compare_external.py`, which fetches
and builds proxTV and Condat's C code at run time (they are not vendored).
Every external result is checked against fastprox before it is timed, and
every call gets a fresh input: repeating one input lets the branch predictor
learn it and makes the direct algorithms look about three times faster.

## 2026-10-14, Apple M5 Pro, n = 10⁶

5 performance + 10 efficiency cores. Separate runs on this laptop differ by
up to 1.6× (macOS moves processes between core types), so the script runs
every solver on the same fresh input in turn and reports medians; only
comparisons within one run are meaningful. A homogeneous-core benchmark
runner is still to be set up (see ROADMAP.md).

**1D TV, float64, ns per sample** (fastest within 2% in bold)

| Signal, λ | fastprox auto | fastprox taut string | proxTV linearized | proxTV hybrid | proxTV classic |
| --- | --- | --- | --- | --- | --- |
| blocks, 0.1 | **7.0** | 10.7 | 7.2 | 7.3 | 9.4 |
| blocks, 1 | **10.1** | 16.8 | 13.3 | 13.0 | 14.7 |
| blocks, 10 | **7.2** | 10.4 | 9.4 | 9.5 | 9.0 |
| random walk, 0.1 | **7.9** | 11.9 | **7.9** | **8.0** | 10.4 |
| random walk, 1 | **9.7** | 15.5 | 10.8 | 11.1 | 13.4 |
| random walk, 10 | **14.2** | 16.7 | **13.9** | 14.4 | 14.6 |
| white noise, 0.1 | **7.0** | 11.3 | **7.0** | 7.2 | 10.1 |
| white noise, 1 | **10.8** | 19.7 | 14.7 | 14.9 | 17.5 |
| white noise, 10 | **6.1** | 12.8 | 6.3 | 6.4 | 11.1 |

**1D TV, float32, ns per sample** (Condat's code is single precision)

| Signal, λ | fastprox auto | Condat `TV1D_denoise` |
| --- | --- | --- |
| blocks, 0.1 | 7.4 | **7.0** |
| blocks, 1 | **10.8** | 11.0 |
| blocks, 10 | **7.5** | 8.8 |
| random walk, 0.1 | 8.4 | **7.8** |
| random walk, 1 | **10.1** | 10.6 |
| random walk, 10 | **14.5** | 16.9 |
| white noise, 0.1 | 7.5 | **6.7** |
| white noise, 1 | **11.1** | 12.5 |
| white noise, 10 | **6.3** | 9.7 |

**Weighted 1D TV, float64, ns per sample** (edge weights uniform in [0.5, 1.5], times λ; the fastprox operator is built once per weight vector)

| Signal, λ | fastprox | proxTV `tautString_TV1_Weighted` |
| --- | --- | --- |
| blocks, 0.1 | **10.0** | **10.0** |
| blocks, 1 | **11.6** | 15.0 |
| blocks, 10 | **13.2** | 14.2 |
| random walk, 0.1 | 11.5 | **11.3** |
| random walk, 1 | **12.9** | 15.1 |
| random walk, 10 | **16.4** | 19.3 |
| white noise, 0.1 | **9.4** | **9.3** |
| white noise, 1 | **11.4** | 15.3 |
| white noise, 10 | 5.7 | **5.1** |

**Simplex projection, radius 1, ns per sample**

| Input | fastprox | Condat `simplexproj_Condat` |
| --- | --- | --- |
| Gaussian | 0.69 | **0.57** |
| uniform | 0.84 | **0.82** |
| one large entry | 0.67 | **0.61** |

At n = 10³ fastprox is faster on all three inputs (1.6–4.2 against 2.8–5.7).

**Reading.** fastprox is the fastest code, or within 2% of it, in every uniform
double-precision setting (0.73–0.90× proxTV's time at λ = 1), and in eight
of nine weighted settings (0.75–0.93× proxTV's time at λ ≥ 1; 13% slower on
white noise at λ = 10, where the kernel itself ties and the rest is the
per-call finiteness and range pass). In single precision it beats Condat's
code at λ ≥ 1 and trails by 6–12% at λ = 0.1: fastprox converts every sample
to double, which the spec requires for running sums. The simplex projection
is 10–20% slower than Condat's code at n = 10⁶ in this run and 8–11% slower
when both run in one program (tied on uniform inputs); it is faster at
small n.

Kernel changes, each measured in one program on the same inputs (n = 10⁶):
branch-free Condat bound updates in the first 8 samples of a segment and
branches after (results bit-identical; 0.65–0.95× the time of the branchy
kernel at λ ≤ 1, and up to 21% faster than always branch-free at λ = 10);
the segment-based taut string, 0.74–0.83× the time of the double-double
version; weighted Condat within 10% of uniform Condat; the simplex with its
finiteness test folded into the pivot pass, 0.81× the previous time.

## 2026-10-15, Apple M5 Pro, n = 10⁶ (range pass and GCC selects)

Same script and machine after two changes to 1D TV: the per-call finiteness
and range pass runs in 8 independent lanes (it was one dependent chain of
compare-selects, about 2 cycles per sample), and the branch-free bound
updates are written as min/max so that GCC keeps them branch-free. Clang's
kernel code is unchanged, so on this machine only the range pass moved.

| Setting, ns per sample | fastprox | best external code |
| --- | --- | --- |
| weighted, white noise, λ = 10 | 5.33 | 4.91 (proxTV) |
| float32, blocks, λ = 0.1 | 7.14 | 7.10 (Condat) |
| float32, random walk, λ = 0.1 | 6.80 | 6.61 (Condat) |
| float32, white noise, λ = 0.1 | 5.63 | 5.36 (Condat) |

All 9 uniform float64 settings and 8 of 9 weighted ones stay fastest or
within 2%. Weighted white noise at λ = 10 is now 8% behind (was 13%); the
range pass costs 0.26 ns per sample of it, about what reading the row costs,
and the kernel alone takes 5.03 ns in a C++ harness. float32 at λ = 0.1 is
1–5% behind (was 6–12%).

The range pass alone, float64, n = 10⁶, fresh input: 0.48 → 0.25 ns per
sample with Clang 21 and 0.89 → 0.26 with GCC 16. Whole 1D TV rows with
GCC 16 on this machine, before → after (float64): uniform white noise at
λ = 1, 9.54 → 7.71; weighted white noise at λ = 1, 11.35 → 8.33; uniform
random walk at λ = 10, 8.93 → 10.64. GCC had compiled the short-segment
selects as branches, which lose at λ ≤ 1 and happened to win at λ = 10.
After the change the GCC build is within 3% of the Clang build at λ ≤ 1.
See the 2026-10-15 runner results below for Neoverse-N2.

## 2026-10-14, GitHub Actions runners (`bench` workflow, quick mode, n = 10⁵)

Started by hand from the Actions tab (`.github/workflows/bench.yml`); each
runner records its CPU and compiler in `machine.txt` in the artifact. On a
private repository the Linux runners have 2 vCPUs, and they are shared VMs:
two identical runs on x86_64 differed between 25 and 17 of 30 comparisons
won, so read one run at a time and prefer `full` mode for anything finer.

| Runner | CPU | Compiler | fastprox fastest or within 2% |
| --- | --- | --- | --- |
| `ubuntu-latest` | AMD EPYC 9V45, 1 core with 2 hyperthreads | GCC 13.3 | 25 and 17 of 30 (two runs) |
| `ubuntu-24.04-arm` | Neoverse-N2, 2 cores | GCC 13.3 | 12 and 11 of 30 (two runs) |

On Neoverse-N2 fastprox trails consistently: 1D TV at λ = 0.1 by 9–17%
(float64 and float32), weighted 1D TV at λ = 0.1 by 10–16% and on white
noise at λ = 10 by about 20%, and the simplex projection by 24–31%. The kernels
were tuned on Apple silicon with Clang (branch-free updates for short
segments, the finiteness test folded into the pivot pass); the same choices
do not carry over to this core and compiler.

Python call overhead was 2.0–3.4 µs at n = 100 on both runners (target:
under 5 µs). Parallel scaling on the arm64 runner's two physical cores: 1.96×
at 2 threads (efficiency 0.98; target 0.7). The x86_64 runner's two vCPUs are
hyperthreads of one core, so its 2-thread figure says nothing about scaling.

## 2026-10-15, GitHub Actions runners, before and after the GCC fix (quick mode, n = 10⁵)

Two `bench` runs started together, one on `master` and one on the branch
with the range pass and GCC changes. The repository is public now, so the
runners have 4 vCPUs. Both arm64 runs had Neoverse-N2 and GCC 13.3; the two
x86_64 runs landed on different CPUs (EPYC 9V74 and EPYC 7763), so only
their ratios to the external codes compare.

| Runner | fastprox fastest or within 2% of 30, master → branch |
| --- | --- |
| `ubuntu-24.04-arm`, Neoverse-N2, 4 cores | 10 → 20 |
| `ubuntu-latest`, EPYC, 2 cores with 2 hyperthreads | 24 → 24 |

On Neoverse-N2 (fastprox time over the best external code, master → branch):
1D TV at λ = 1 went from 0.92–1.06 to 0.76–0.92 and at λ = 0.1 from
1.07–1.20 to 0.96–1.08; weighted TV at λ = 1 from 1.00–1.04 to 0.79–0.90.
Still behind: weighted white noise at λ = 10 (1.25), uniform random walk at
λ = 10 (1.20, worse than 1.13 on master: the branches GCC emitted predicted
well there), float32 white noise at λ = 10 (1.11), white noise and random
walk at λ = 0.1 (1.06–1.09), and the simplex on Gaussian input and with one
large entry (1.22–1.23; tied on uniform input).

Parallel scaling, 64 rows of 10⁴, 1D TV: 3.90× at 4 threads on the four
physical Neoverse-N2 cores (efficiency 0.97–0.98, target 0.7). The x86_64
runner's 4 vCPUs are 2 cores with 2 hyperthreads each (3.48–3.78×). Python
call overhead stayed at 1.6–2.8 µs at n = 100.

## 2026-10-15, Apple M5 Pro: simplex and ℓ1 ball

The first pass of Condat's method now tests a block of 8 entries with one
comparison of its maximum against the running pivot, and visits the entries
only when the block passes; the ℓ1-ball norm is summed in 8 lanes. Results
are unchanged (the skipped entries could not have moved the pivot).

**Simplex projection, radius 1, ns per sample** (`bench/compare_external.py`,
both codes on the same fresh input)

| Input | n | fastprox | Condat `simplexproj_Condat` |
| --- | --- | --- | --- |
| Gaussian | 10⁵ | **0.32** | 0.42 |
| Gaussian | 10⁶ | **0.50** | 0.56 |
| Gaussian | 10⁷ | **0.48** | 0.52 |
| uniform | 10⁵ | 1.09 | **1.02** |
| uniform | 10⁶ | **0.77** | **0.77** |
| uniform | 10⁷ | **0.59** | 0.65 |
| one large entry | 10⁵ | **0.32** | 0.47 |
| one large entry | 10⁶ | **0.48** | 0.60 |
| one large entry | 10⁷ | **0.47** | 0.55 |

At n = 10³ fastprox is faster on all three inputs (1.2–3.8 against 2.3–5.0).
In a C++ harness (n = 10⁶, Gaussian input, before → after): simplex
0.50 → 0.33 with Clang 21 and GCC 16; ℓ1 ball with unit weights
1.08 → 0.64, with weights in [0.5, 1.5] 1.37 → 1.08 (Clang) and
1.68 → 1.05 (GCC). Block sizes 4 and 16 were slower with one of the two
compilers, and testing the block with an OR of the comparisons instead of
its maximum was 25% slower; in the weighted variant, where a maximum of
a_i/w_i would not reproduce the test a_i > ρ w_i exactly, the OR test
brought no gain and is not used.

## 2026-10-15, GitHub Actions runners: simplex block skip (quick mode, n = 10⁵)

`bench` runs on `master` and on the branch with the block skip, GCC 13.3.
Simplex projection, fastprox time over Condat's (master → branch):

| Runner | Gaussian | uniform | one large entry |
| --- | --- | --- | --- |
| `ubuntu-24.04-arm`, Neoverse-N2 | 1.27 → 0.71 | 1.33 → 1.01 | 1.27 → 0.85 |
| `ubuntu-latest`, x86_64 | 1.08 → 0.72 | 0.87 → 0.73 | 0.90 → 0.70 |

A first version kept the block in a scratch array and was twice as slow
as `master` on x86_64 (EPYC 7763: 1.96, 1.77 and 1.96): GCC reloaded the
array with wider loads than it stored, which the core cannot forward.
Holding the block in locals fixed it with no change on arm64.

The x86_64 branch run landed on an Intel Xeon Platinum 8573C, the `master`
run on an AMD EPYC 7763, so only the simplex rows above compare across them
(1D TV is unchanged on the branch). On that Intel core fastprox's 1D TV
times were close to those on the EPYC, but proxTV's were lower: fastprox
was within 2% of the best code in 8 of 30 settings, trailing by 7–24% in
most uniform and weighted settings at λ ≤ 1. On Neoverse-N2, 23 of 30
settings were fastest or within 2% (19 on `master`).


## 2026-10-15, GitHub Actions x86_64: 1D TV with SSE2 min/max (`profile` workflow, n = 10⁵)

Two `profile` runs, one on `master` (6 jobs) and one on the branch whose
`min_nonan`/`max_nonan` use `minsd`/`maxsd` on x86-64 (8 jobs). Each job
lands on a random CPU; none of the VMs exposes hardware counters. GCC 13.3,
single thread, 31 fresh inputs per setting, solver order rotated. Jobs on the
same CPU model agree to 0.01 in their ratios, but absolute times differ by up
to 18% between VMs, so only the ratios compare across runs.

On `master`, GCC compiled `max(u − l, 0)` to `comisd; jbe; pxor` and
`min(u, l)` to `comisd; ja`: the weighted Condat kernel had 56 conditional
jumps and no min/max instruction. On the branch it has 4 `minsd`/`maxsd`
and 52 jumps, and the inlined uniform kernel 65 min/max instructions.

**Settings (of 18: uniform and weighted, three signals, λ = 0.1, 1, 10)
where fastprox is fastest or within 2% of the best proxTV solver**

| CPU | `master` | branch |
| --- | --- | --- |
| Intel Xeon Platinum 8370C | 6 | — |
| Intel Xeon Platinum 8573C | 3 | 12 |
| Intel Xeon 6973P-C | — | 13 |
| AMD EPYC 7763 | 17 | 17 |
| AMD EPYC 9V74 | 14 | 18 |
| AMD EPYC 9V45 | — | 17–18 |

Fastprox time over the best proxTV solver on the Xeon 8573C (`master` →
branch): uniform rows at λ = 1 from 1.01–1.16 to 0.83–1.03 and at λ = 0.1
from 1.12–1.16 to 1.01–1.07; weighted rows at λ = 1 from 1.08–1.12 to
0.82–0.93 and at λ = 0.1 from 1.10–1.15 to 1.01–1.08. On the AMD cores the
same rows went from 0.89–1.04 to 0.72–0.98. Still behind on Intel: random
walk and white noise at λ = 0.1 (1.02–1.09), uniform random walk at λ = 10
(1.05–1.09; it got worse, from 1.04 on the 8573C and from 0.90 to 0.95–1.00
on AMD, as on Neoverse-N2 the branches predicted well there), and weighted
white noise at λ = 10 (1.12–1.14, 1.00–1.08 on AMD).
