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

## 2026-10-15, Apple M5 Pro: 1D TV, long segments by their running sum

Profiling weighted white noise at λ = 10, the one setting where fastprox
trailed proxTV on every machine. No hardware counters are available here
(Command Line Tools only, no Instruments), so the profile combines timings
with counts from an instrumented copy of the kernel:

- At λ = 10 Condat's kernel visits each sample 1.9–2.0 times (0.91–0.99
  rescanned steps per sample, rescans of 16–1000 steps) and never falls back
  to the taut string; weighted rows do no more work than uniform ones.
- proxTV's weighted taut string is the same kind of scan (it restarts after
  the last break point too) with branches for the clamps, and the bare
  kernels tied: 4.90 against 4.89 ns per sample. fastprox's call added
  0.2 ns of range pass (`finite_range`) and 0.05 ns of dispatch.
- The time per step, 11–12 cycles for about 10 flops, came from the bound
  updates: every clamp moved `vmin` by the excess of `umin`, and the next
  step's `umin` depended on that `vmin`.

Since u = s − m (v − ref) with s the running sum of the segment, a clamp of
`vmin` sets it to the candidate ref + (s − λₖ)/m, so `vmin` is the running
maximum of these candidates (and `vmax` the running minimum of
ref + (s + λₖ)/m). Only the sum is carried from step to step. In the first
3 samples of a segment, where jumps are frequent, the paper's update stays
(it resolves the jump test one add after the load).

**Kernel alone, ns per sample** (n = 10⁵, 41 fresh inputs, solver order
rotated; weights λ·[0.5, 1.5])

| Signal, λ | uniform, before | uniform, after | weighted, before | weighted, after | proxTV weighted |
| --- | --- | --- | --- | --- | --- |
| white noise, 0.1 | 4.98 | 5.03 | 5.56 | 5.60 | 6.23 |
| white noise, 1 | 7.38 | 5.84 | 7.73 | 6.44 | 11.24 |
| white noise, 10 | 5.05 | 1.73 | 4.90 | 1.89 | 4.85 |
| random walk, 0.1 | 6.16 | 6.25 | 6.83 | 6.78 | 7.58 |
| random walk, 1 | 8.70 | 7.91 | 8.85 | 8.26 | 11.18 |
| random walk, 10 | 9.97 | 6.34 | 10.28 | 4.66 | 13.00 |
| blocks, 1 | 7.79 | 4.14 | 7.92 | 4.59 | 11.09 |
| blocks, 10 | 6.76 | 2.65 | 7.99 | 2.82 | 9.39 |

The running sum alone, from the first sample of a segment, was faster still
at λ ≥ 1 (1.67 at white noise λ = 10) but 3–12% slower at λ = 0.1; switching
after 2 to 4 samples was close, after 8 up to a third slower at λ ≥ 1.

**Through the library, fastprox time over the best proxTV solver**
(`bench/profile_kernels.cpp`, n = 10⁵, before → after)

| Build | uniform, λ = 1 | uniform, λ = 10 | weighted, λ = 1 | weighted, λ = 10 | λ = 0.1 |
| --- | --- | --- | --- | --- | --- |
| Clang 21 | 0.71–0.88 → 0.41–0.80 | 0.74–0.99 → 0.30–0.64 | 0.73–0.84 → 0.45–0.79 | 0.84–1.08 → 0.34–0.43 | 0.92–0.99 → 0.94–1.00 |
| GCC 16 | 0.72–0.87 → 0.46–0.83 | 0.77–1.04 → 0.39–0.73 | 0.75–0.86 → 0.50–0.83 | 0.86–1.14 → 0.40–0.60 | 0.93–0.98 → 0.94–1.02 |

`bench/compare_external.py` (n = 10³ to 10⁷) agrees: weighted white noise at
λ = 10 went from 1.03–1.09 to 0.43–0.45 of proxTV's time, uniform random walk
at λ = 10 from 0.99–1.22 to 0.64–0.86, float32 rows at λ ≥ 1 from 0.60–0.97
to 0.23–0.86 of Condat's single-precision code. At λ = 0.1 the ratios moved
by −0.13 to +0.05, mostly within the ±4% that two runs of the same build
differ by at n = 10⁶; float32 at λ = 0.1 stays within 7% of Condat's code
(8% before).

Accuracy: on 3000 random rows (n up to 3000, offsets up to 10⁶, λ from 10⁻²
to 10², uniform and weighted) the structure is identical to the previous
kernel, and the error against segment values recomputed from that
structure with compensated sums is at most 52 ε ‖x‖∞, as before.

## 2026-10-15, GitHub Actions runners: running-sum Condat (GCC 13.3, n = 10⁵)

`profile` on the branch (8 jobs: 3 Intel, 5 AMD), compared with the run on
`master` (`profile` run 38805817290), and `bench` (quick mode) on the branch
and on `master`, started together. Fastprox time over the best proxTV solver:

| Setting | Intel Xeon (8573C, 6973P-C) | AMD EPYC (7763, 9V74, 9V45) | Neoverse-N2 (`bench`) |
| --- | --- | --- | --- |
| weighted white noise, λ = 10 | 1.12–1.14 → 0.73–0.78 | 1.00–1.08 → 0.55–0.62 | 1.22 → 0.92 |
| uniform random walk, λ = 10 | 1.05–1.09 → 0.83–0.84 | 0.95–1.00 → 0.69–0.70 | 1.19 → 1.06 |
| uniform, λ = 1 | 0.81–1.03 → 0.55–0.97 | 0.75–0.92 → 0.48–0.91 | 0.77–0.93 → 0.62–0.93 |
| weighted, λ = 1 | 0.81–0.93 → 0.58–0.91 | 0.72–0.83 → 0.48–0.84 | 0.79–0.89 → 0.65–0.90 |
| uniform, λ = 0.1 | 1.00–1.09 → 1.01–1.09 | 0.92–0.98 → 0.95–1.00 | 1.01–1.12 → 1.04–1.12 |
| weighted, λ = 0.1 | 1.00–1.08 → 1.04–1.12 | 0.89–0.98 → 0.92–0.99 | 1.03–1.09 → 1.03–1.11 |

Settings (of 18) fastest or within 2% in `profile`: Intel 12–13 → 13, AMD
17–18 → 18. The weighted Condat kernel compiles with 6 min/max instructions
(was 4). What remains behind proxTV: λ = 0.1 on Intel and Neoverse-N2 (it
was before, and the weighted rows lost about 3% on Intel), and uniform random
walk at λ = 10 on Neoverse-N2.

In the same `bench` runs, the simplex with the SSE2 block maximum on x86_64
took 0.82–0.99 of Condat's time on an EPYC 7763 and 0.90–1.06 on an EPYC
9V45 (Gaussian, uniform, one large entry; uniform input is the slowest), and
0.87–0.92 on Neoverse-N2.

## 2026-10-16, 1D TV at λ = 0.1: two attempts that did not hold

At λ = 0.1 most segments are 1 to 3 samples long and there are about 0.9
jumps per sample, so the time goes to the jumps. Neither of these is in the
code; they are recorded so that they are not retried as they were.

- **One branch for both jump directions** (`down | up`, the direction then
  chosen with selects in a single routine): on Apple M5 1.65–2.1× proxTV's
  time at λ = 0.1 with Clang (was 0.93–1.00) and slower with GCC 16 too.
  The restart of the next segment then waits for the direction.
- **Short segments written with 4 unconditional stores** (the slots past the
  segment belong to later segments), instead of the fill loop whose
  vectorised prologue and remainder branch on the length. On Apple M5 it
  helped: λ = 0.1 from 0.93–1.00 to 0.90–0.95 of proxTV's time (Clang) and
  from 0.94–1.02 to 0.90–1.00 (GCC 16). On GitHub's runners (GCC 13.3,
  n = 10⁵) it made λ = 0.1 slower, `master` → branch:

  | Runner | uniform | weighted |
  | --- | --- | --- |
  | Neoverse-N2 (`bench`) | 1.03–1.09 → 1.08–1.16 | 1.04–1.13 → 1.05–1.14 |
  | AMD EPYC 9V74 (`bench`, same CPU both runs) | 0.96–1.00 → 1.00–1.04 | 0.94–0.97 → 0.98–1.04 |
  | AMD EPYC 7763 (`profile`) | 0.95–0.99 → 0.99–1.10 | 0.92–0.97 → 0.98–1.04 |
  | Intel Xeon 8573C (`profile`) | 1.01–1.09 → 1.04–1.07 | 1.05–1.12 → 1.04–1.13 |

  λ = 1 gained a few percent and λ = 10 was unchanged. The stores per jump
  roughly triple, often as overlapping unaligned 16-byte writes, which the
  M5 absorbs and Neoverse-N2 and the x86 cores do not.

Apple M5 is a poor guide in this regime: the gap is on Neoverse-N2 and Intel,
and variants are best compared there, several in one binary on the same
runner.

## 2026-10-16, uniform random walk at λ = 10 on Neoverse-N2 (`variants` workflow)

`bench` had fastprox at 1.04–1.06 of proxTV's time here. The best proxTV
solver in this setting on Neoverse-N2 is its **classic** taut string
(12.19 ns per sample at n = 10⁵, against 15.14–17.05 for the linearized
and hybrid ones and 12.65 for fastprox): a single pass, where Condat's
algorithm visits each sample about 3 times (2.2 rescanned steps per sample).
Rotating the solver order in `compare_external.py` did not change the
ratio (1.04 on the same runner type).

Kernel alone (`bench/tv_variants.cpp`, n = 10⁵, 31 fresh inputs), time over
the best proxTV solver (classic, linearized or hybrid taut string; weighted
taut string for weighted rows). The kernel uses the paper's update for the
first 3 samples of a segment; the variant starts the running sum at the
first sample, opened at s = ±λₑ:

| Setting | Neoverse-N2, kernel | Neoverse-N2, running sum at once | AMD EPYC 7763, kernel | AMD EPYC 7763, running sum at once |
| --- | --- | --- | --- | --- |
| uniform random walk, λ = 10 | 0.95–0.96 | 0.82 | 0.68–0.69 | 0.61–0.62 |
| uniform, λ = 1 | 0.54–0.86 | 0.46–0.74 | 0.44–0.82 | 0.39–0.70 |
| uniform, λ = 0.1 | 0.93–0.99 | 1.06–1.17 | 0.87–0.90 | 1.05–1.16 |
| weighted, λ = 0.1 | 0.87–0.91 | 1.03–1.11 | 0.83–0.85 | 0.97–1.05 |

Starting the running sum at once is up to 15% faster at λ ≥ 1 and 13–29%
slower at λ = 0.1. Switching after 2 samples lies in between, after 4 is
close to 3 (a first run, against the linearized and hybrid solvers only).
No variant tried kept both:

- the running sum at once only after a segment longer than 3, 8 or 16
  samples: 0.92–0.96 here, no gain (and with GCC on x86 the weighted
  instantiations came out 1.2–1.6× proxTV's time);
- the radius moved off the path from the sum to the jump tests
  (s + (d ± λ) instead of (s + d) ± λ): slightly slower everywhere;
- opening the running sum at s = ±λₑ instead of u + (v − ref): with the
  running sum at once, λ = 0.1 went from 1.10–1.22 to 1.06–1.17 of proxTV's
  time on Neoverse-N2, not enough.

The kernel stays at 3. With the library's range pass (about 2%) the setting
is about even with proxTV's classic taut string on Neoverse-N2 (0.97 from
the kernel timings, 1.04–1.06 in `bench`), and ahead elsewhere: 0.64 on
Apple M5 and 0.73–0.74 on AMD EPYC 9V74 against all proxTV solvers, 0.80–0.84
on Intel against the linearized and hybrid ones.

## 2026-10-16, Google Cloud c4a-standard-16: parallel scaling at 16 cores

Neoverse-V2, 16 physical cores (1 thread per core), Ubuntu 24.04, `master`
at f2bc650 built by `uv sync` (GCC 13). `bench/bench_python.py` plus two
larger batches; 1D TV at λ = 1, best of 7.

| Batch | 2 threads | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- |
| 64 rows of 10⁴ (spec target ≥ 0.7) | 2.00 (1.00) | 3.99 (1.00) | 7.98 (1.00) | — | 15.86 (0.99) |
| 256 rows of 10⁴ | 2.04 (1.02) | 4.08 (1.02) | 8.19 (1.02) | 11.97 (1.00) | 16.27 (1.02) |
| 64 rows of 10⁵ | 1.99 (1.00) | 3.96 (0.99) | 7.81 (0.98) | 10.43 (0.87) | 15.57 (0.97) |

Speed-up over one thread, efficiency in parentheses. The dip at 12 threads
on 64 rows is the split: 64 rows over 12 threads leave the slowest with 6,
which bounds efficiency at 0.89. Python call overhead at n = 100 was
1.4–1.8 µs (target 5 µs).

## 2026-10-16, Apple M5 Pro: anisotropic TV in 2D and 3D

`bench/compare_tv_aniso.py`, single thread, piecewise-constant blocks plus
Gaussian noise (σ = 0.3), equal weights on every axis. The solvers stop on
different criteria, so each is run with a range of settings (fastprox: the
gap tolerance from 1e-2 to 1e-12; proxTV: iteration caps from 1 to 1024) and
the table gives the fastest run that reached each error |z − z*| / |z*|
against a reference solved to a gap of 1e-15. "–": no run reached it.

**ms to reach the error, 2D**

| Image, λ | Solver | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
| --- | --- | --- | --- | --- | --- |
| 256², 0.1 | fastprox (ADBPG) | 6.13 | **11.14** | 40.71 | **88.14** |
|  | proxTV DR2 (alternating reflections) | **4.13** | 11.98 | **38.35** | 492.29 |
|  | proxTV Kolmogorov2 / Yang2 / parallel Dykstra | 12.41 | 47.38 | 240.21 | 272.47 |
| 256², 0.5 | fastprox (ADBPG) | **4.01** | **17.54** | **42.80** | **193.01** |
|  | proxTV DR2 | 8.31 | 28.96 | 82.48 | 934.73 |
|  | best other proxTV | 13.59 | 50.24 | 152.43 | 587.45 |
| 512², 0.1 | fastprox (ADBPG) | 25.92 | **46.96** | **87.15** | **429.61** |
|  | proxTV DR2 | **17.97** | 52.92 | 182.91 | 2633.00 |
|  | best other proxTV | 53.60 | 208.98 | 1234.82 | 1307.63 |
| 512², 0.5 | fastprox (ADBPG) | **16.00** | **90.94** | **255.08** | **1239.12** |
|  | proxTV DR2 | 20.13 | 132.49 | 419.19 | 4581.20 |
|  | best other proxTV | 57.29 | 447.99 | 748.70 | 5793.85 |

**ms to reach the error, 3D (64³)**

| λ | Solver | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
| --- | --- | --- | --- | --- | --- |
| 0.1 | fastprox (FDPG, 3 blocks) | **71.91** | **392.98** | **744.34** | 10461.08 |
|  | proxTV Yang3 (ADMM) | 1023.79 | 2107.65 | 2107.65 | **4243.72** |
|  | proxTV parallel Dykstra | 171.13 | 1360.30 | – | – |
| 0.5 | fastprox (FDPG, 3 blocks) | **96.49** | **209.86** | **608.86** | 4332.43 |
|  | proxTV Yang3 (ADMM) | 1077.48 | 2005.35 | 2005.35 | **3880.73** |
|  | proxTV parallel Dykstra | 151.43 | 2536.11 | – | – |

proxTV's parallel Douglas–Rachford (`PDR_TV`) did not reach 1e-2 within
1024 iterations in any setting. In 2D, fastprox is fastest from 1e-3 on
(tied at 1e-4 on 256², λ = 0.1) and at 1e-6 3.0–3.7× faster than the best proxTV solver (3.7–6.1× faster than DR2);
DR2, which converges fast to medium accuracy as Barbero and Sra report, is
quicker to 1e-2 at λ = 0.1. In 3D the three-block method, with its step of
1/3, is 1.6–9.6× faster than the best proxTV solver up to 1e-4 but slower
than Yang's ADMM at 1e-6 (2.5× at λ = 0.1, 1.1× at λ = 0.5); parallel proximal Dykstra, the other candidate for three
blocks, is slower than both at every accuracy.

## 2026-10-16, Apple M5 Pro: isotropic TV in 2D and 3D

`bench/compare_tv_iso.py`, single thread, the same piecewise-constant images
as for anisotropic TV. Each solver is run with a range of settings (fastprox:
the gap tolerance from 1e-2 to 1e-12; scikit-image 0.26's
`denoise_tv_chambolle`, Chambolle's 2004 projection algorithm in NumPy, with
its own stopping off: iteration caps from 1 to 4096), and the table gives the
fastest run that reached each error |z − z*| / |z*| against a reference
solved to a gap of 1e-15. scikit-image solves the same problem (its `weight`
is γ): at 100 000 iterations it agrees with fastprox to 3.5e-5. It is
vectorised NumPy, not compiled code, so this is a baseline rather than a
like-for-like comparison. "–": no run reached it.

**ms to reach the error**

| Image, γ | Solver | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
| --- | --- | --- | --- | --- | --- |
| 256², 0.1 | fastprox (FDPG) | 3.43 | **7.61** | **16.05** | **141.42** |
|  | scikit-image | **2.36** | 19.54 | 156.61 | – |
| 256², 0.3 | fastprox (FDPG) | **6.62** | **54.99** | **145.65** | **1080.47** |
|  | scikit-image | 38.95 | 626.44 | – | – |
| 64³, 0.1 | fastprox (FDPG) | 15.51 | **37.69** | **80.45** | **824.43** |
|  | scikit-image | **12.45** | 101.98 | 820.66 | – |

fastprox is 2.6–11× faster from 1e-3 on and alone reaches 1e-6;
scikit-image's cheap first iterations are 25–45% faster to 1e-2 at
γ = 0.1. The run was stopped before 64³ at γ = 0.3 (scikit-image's sweep to
4096 iterations on 262 144 voxels, three repetitions each, made the full run
take over 50 minutes); the 512² images of the anisotropic benchmark were
left out for the same reason. At 128² and 32³ (`--quick`), fastprox was 2.7–18×
faster from 1e-3 on, and scikit-image 1.3× faster to 1e-2 at 128², γ = 0.1.

## 2026-10-16, Apple M5 Pro: support functions and hyperplane ∩ box

Single thread, operators built once, one row of normal entries, best of 25
calls through Python (ns per entry; the call overhead is included, so the
n = 10⁴ column carries about 0.15 ns of it). NumPy's sort of the same row is
there for scale.

| Operator | n = 10⁴ | n = 10⁶ |
| --- | --- | --- |
| simplex (Condat, for scale) | 0.34 | 0.32 |
| max | 0.34 | 0.32 |
| sum of k = 1 largest | 4.52 | 5.19 |
| sum of k = n/10 largest | 5.10 | 5.08 |
| sum of k = n/10 largest \|x\| | 5.21 | 4.46 |
| capped simplex k = n/10, random walk | 4.85 | 4.71 |
| hyperplane, no box (closed form) | 3.52 | 4.60 |
| hyperplane ∩ box [−0.5, 0.5] | 5.66 | 7.85 |
| hyperplane ∩ per-entry box | 4.48 | 7.99 |
| half-space ∩ box, active | 4.21 | 8.52 |
| `np.sort` (for scale) | 16.95 | 25.18 |

- **Max** is min(x, τ) with τ from Condat's simplex threshold, so it costs a
  simplex projection.
- **The plane solver** went through four versions. It started by sorting the
  2n breakpoints and bisecting with full evaluations of φ: 40–60 ns per entry,
  more than sorting the row. An exact median search over the breakpoints
  (`nth_element`, folding decided entries into running sums) came to 42–48 ns:
  one exact median of 2n doubles already costs about 14 ns per entry. Newton
  steps on φ inside a shrinking bracket, with the median as the fallback, took
  3–6 passes on most rows, but 15 passes for k = 1 (the root deep in the upper
  tail, approached from one side with short steps) and 7 on random walks
  (12–16 ns). The final version adds branch-free passes over the contiguous
  row, with no folding, which mostly kept entries anyway because one end of the
  bracket stays infinite. For rows of 8192 entries or more it also starts from
  the root of the problem on every 16th entry. That brings every case to
  3–5 passes.
- **The per-entry box** costs more at n = 10⁶ because each pass reads three
  arrays (a, lo, hi) besides x.

## 2026-10-16, Apple M5 Pro: cones, barriers and group penalties

Single thread, as in the previous section (best of 25 calls through Python,
ns per entry, one row of normal entries; groups of 16 consecutive entries).

| Operator | n = 10⁴ | n = 10⁶ |
| --- | --- | --- |
| soft thresholding (for scale) | 0.52 | 0.46 |
| log barrier | 0.82 | 0.76 |
| ℓ1 over the box [−0.5, 0.8] | 0.70 | 0.65 |
| ℓ1 over a per-entry box | 0.70 | 0.64 |
| second-order cone, one row | 0.93 | 0.90 |
| Huber of the ℓ2 norm, one row | 0.97 | 0.96 |
| group ℓ2 with group ids (for scale) | 3.34 | 3.28 |
| group Huber with group ids | 3.45 | 3.35 |
| group SCAD with group ids | 3.65 | 3.62 |
| group MCP with group ids | 3.50 | 3.48 |
| group Huber, rows of 16 | 1.08 | 1.09 |
| group ℓ2, rows of 16 (for scale) | 1.00 | 0.94 |

- **Group penalties** cost what group ℓ2 costs: the scalar map runs once per
  group (a virtual call into the scalar operator), and the time goes to the
  norm accumulation, which with group ids scatters into per-group sums. Rows
  of 16 along a batch axis avoid the scatter and run 3× faster.
- **Single-row norms** (the cone, Huber of the norm) are bound by the
  sequential sum of squares: one dependent addition per entry, which the
  compiler may not reorder without fast-math (as for the ℓ2 ball).

## 2026-10-16, Apple M5 Pro: tree-structured groups and the 1D TV ball

Single thread, as in the previous sections (best of 15 calls through Python,
ns per entry, one row; normal entries, or a Gaussian random walk where
marked).

| Operator | n = 10⁴ | n = 10⁶ |
| --- | --- | --- |
| sparse group lasso, groups of 16 | 2.44 | 1.04 |
| tree, 4-ary from groups of 16 (depth 5 and 8) | 11.6 | 8.08 |
| 1D TV prox, γ = 0.5 (for scale) | 6.70 | 7.01 |
| TV ball, noise, α = 0.01 TV(x) | 71.5 | 69.0 |
| TV ball, noise, α = 0.3 TV(x) | 64.0 | 111 |
| TV ball, noise, α = 0.9 TV(x) | 42.1 | 55.1 |
| TV ball, random walk, α = 0.01 TV(x) | 92.2 | 124 |
| TV ball, random walk, α = 0.3 TV(x) | 90.9 | 134 |
| TV ball, random walk, α = 0.9 TV(x) | 45.8 | 96.0 |

- **The tree norm** reads each group once, so its cost is the total size of
  the groups: depth times n for a full tree, through index lists.
- **The TV ball** solves 1D TV problems at trial steps. Counts at n = 10⁵:
  - plain Newton from γ = 0⁺ took 4–17 solves (17 at α = 10⁻⁴ TV(x) on a random
    walk), approaching the root from the left at about 2.5× per step while far;
  - a Newton step on log TV against log γ took 6–18;
  - the hybrid kept (log–log while TV(z) > 2α, Newton after) took 4–14.

  The first version started from the secant through γ_max. It landed right of
  the root and bisected down over several orders of magnitude, taking 13–27
  solves (80–460 ns per entry).

## 2026-10-17, Apple M5 Pro: 1D TV with ℓ2 of differences

`bench/compare_external.py` (new section), single thread, ns per entry, median
over fresh inputs, solvers alternating. Each input is scaled so that the prox
of λ‖Dz‖₂ becomes constant exactly at λ = 1, so λ is the fraction of that
threshold. proxTV's `more_TV2` is Moré–Sorensen on the same dual problem; it
stops at an internal tolerance and an iteration cap.

| Signal, λ | fastprox 10³ | proxTV 10³ | fastprox 10⁵ | proxTV 10⁵ | fastprox 10⁶ | proxTV 10⁶ |
| --- | --- | --- | --- | --- | --- | --- |
| blocks, 0.1 | 51.5 | 67.1 | 68.4 | 85.8 | 68.9 | 86.8 |
| blocks, 0.5 | 52.0 | 64.3 | 52.6 | 64.8 | 51.2 | 65.2 |
| blocks, 0.9 | 37.2 | 54.0 | 34.6 | 44.4 | 35.2 | 1069 (cap) |
| random walk, 0.001 | 57.7 | 59.5 | 51.9 | 54.3 | 51.2 | 54.7 |
| random walk, 0.1 | 52.0 | 54.9 | 50.6 | 54.3 | 44.8 | 54.8 |
| random walk, 0.5 | 45.9 | 58.8 | 42.7 | 45.1 | 34.6 | 1071 (cap) |
| random walk, 0.9 | 35.0 | 44.0 | 34.9 | 34.6 | 29.4 | 1076 (cap) |
| white noise, 0.001 | 43.1 | 54.5 | 67.6 | 75.8 | 77.6 | 88.3 |
| white noise, 0.1 | 72.5 | 92.5 | 77.0 | 86.1 | 77.3 | 87.0 |
| white noise, 0.5 | 63.0 | 70.0 | 53.5 | 66.4 | 60.1 | 75.8 |
| white noise, 0.9 | 44.2 | 54.3 | 43.0 | 44.4 | 43.2 | 55.8 |

- **Speed.** fastprox is 1.0–1.3× faster where proxTV converges. On blocks at
  λ = 0.001 proxTV's result differs from fastprox's beyond 10⁻⁹ on the check
  input, so the script excludes it.
- **proxTV's failures.** The rows marked "cap" ran to proxTV's iteration cap,
  about 1070 ns per entry, and returned a wrong result. On unscaled inputs, where λ
  is 10⁵–10¹² for random walks, this happened from n = 10⁴ on in most
  settings, and once at n = 10³ (random walk, λ = 0.9 of the threshold). There
  proxTV differed from fastprox by 0.49. The dual point built from each
  answer settles it. fastprox's is feasible to 7·10⁻¹⁶ with a relative gap of 2·10⁻¹⁷;
  proxTV's violates the ball constraint by a factor of 2.6 (relative gap
  0.13). fastprox's cost does not depend on the scale.
- **Where the time goes.** Newton from μ = 0 takes 2–8 steps (5–17 tridiagonal
  solves per row in tests). Each step is one factorisation and two
  substitutions, about 7 ns per entry. Most of that is the dependent division in the pivot
  recurrence, which is a continued fraction. Fusing the factorisation with the forward substitution, and
  the norms with the backward substitution, saved 3–8% on the rows run with
  the same settings before and after.
- **Stopping.** The first version stopped when the Newton step fell below
  4ε·μ. Rounding noise in ‖u(μ)‖ then kept the steps above that bound, and
  bisection took over for up to 40 iterations after convergence. The version kept stops once a step is below 10⁻⁷·μ, after taking it: the next error is
  below rounding at Newton's quadratic rate. Against a 50-digit mpmath
  bisection, the error stayed below 4.3·10⁻¹⁶ relative to max |x|.

## 2026-10-17, Apple M5 Pro: isotropic TV with restarts and a Jacobi-preconditioned derivative

**FISTA iterations to the gap tolerance**, plain against restarted, with the
O'Donoghue–Candès gradient test, run every 4th iteration, and an interval
between restarts that may not shrink. The images are 128² (blocks as in
`bench/compare_tv_iso.py`, pure noise, a noisy ramp) and 32³ blocks; "w"
marks axis weights (1, 0.3) or (1, 1, 0.3). The cap is 20 000 iterations.

| Image, γ | 1e-6 | 1e-8 | 1e-10 |
| --- | --- | --- | --- |
| blocks, 0.3 | 448 → 448 | 3012 → 1548 | > 20 000 → 11 476 |
| blocks, 1.0 | 1684 → 1536 | 10 952 → 10 572 | > 20 000 → > 20 000 |
| blocks, 1.0, w | 1560 → 1180 | 11 872 → 9060 | > 20 000 → > 20 000 |
| blocks, 3.0 | 5244 → 2236 | > 20 000 → > 20 000 | – |
| noise, 1.0 | 2356 → 1604 | 17 664 → 9436 | > 20 000 → > 20 000 |
| noise, 3.0 | 6212 → 1156 | > 20 000 → 1952 | > 20 000 → 3120 |
| ramp, 0.3, w | 220 → 220 | 1428 → 732 | 7248 → 4348 |
| ramp, 1.0 | 552 → 552 | 3216 → 1508 | 17 464 → 12 508 |
| ramp, 1.0, w | 664 → 664 | 3736 → 2232 | 17 940 → 17 748 |
| blocks 32³, 0.3 | 244 → 244 | 1620 → 692 | 8948 → 2600 |
| blocks 32³, 1.0 | 1384 → 1044 | 10 160 → 6812 | > 20 000 → > 20 000 |
| blocks 32³, 3.0, w | 2140 → 600 | 7780 → 1024 | > 20 000 → 1412 |

- **Where restarts help.** At γ ≤ 0.3 they first fire near a gap of 1e-6,
  so looser solves are unchanged; at γ = 3 they already take 1.4–1.8× fewer
  iterations to 1e-4. From 1e-8 on, they take 1.04× to over 10× fewer
  iterations, and 15× fewer on 64² noise at γ = 3 to 1e-10 (about 26 000 →
  1 500, now a test).
- **Interval rule.** Restarting at every positive test was slower in places:
  blocks at γ = 1 took 16 096 iterations to 1e-8 against 10 952 without
  restarts. Requiring the interval since the last restart not to shrink
  removed that loss and saved 6% overall. A minimum interval of 10 or 50
  iterations changed nothing.
- **Cost.** The test adds a dot product to the update pass. Run every
  iteration it cost 5% per iteration in 2D and 13–15% in 3D; every 4th
  iteration, compiled as a separate pass, 1–4%.
- **What is still slow.** The dual is degenerate, so the rate does not
  become linear. At γ ≥ 1 most images still need over 20 000 iterations to 1e-10.
- **A polish that did not work.** At every iteration count, 94% or more of
  the gap was the primal part P(z(u)) − P*, and the dual part D* − D(u) was
  14–160× smaller. So I tried to replace z(u) by a Newton polish: the
  minimiser over images constant on the flat regions found in z(u)
  (prototype with sparse direct solves, 256², γ = 0.3 and 1). It was never
  better than z(u). The best threshold gave P(z_pol) − P* = 1.8·10⁻⁸ after 1600 iterations, where z(u) already had a gap
  of 8.5·10⁻⁸, and larger thresholds were far worse (up to 10⁻⁴). Unlike
  anisotropic TV, isotropic solutions have many pixels with small nonzero
  gradients, which any threshold merges wrongly.

**Derivative CG iterations to a relative residual of 1e-10** (VJP, the same
128² and 32³ images), with the previous preconditioner |C| against the
diagonal of the reduced system (|C| plus the H_p terms on each region's
boundary, one O(Nk²) pass). The gradients agree to all printed digits.

| Image, γ, prox tol | \|C\| | Jacobi |
| --- | --- | --- |
| blocks, 0.05, 1e-10 | 147 | 204 |
| blocks, 0.3, 1e-10 | 954 | 512 |
| blocks, 1.0, 1e-10 | 712 | 366 |
| noise, 0.3, 1e-10 | 315 | 359 |
| noise, 1.0, 1e-10 | 777 | 367 |
| ramp, 0.3, 1e-10 | 694 | 411 |
| ramp, 1.0, 1e-10 | 1103 | 553 |
| blocks 32³, 0.3, 1e-10 | 995 (343 ms) | 498 (172 ms) |
| blocks 32³, 1.0, 1e-10 | 1261 (420 ms) | 426 (146 ms) |

The Jacobi preconditioner takes 1.4–3× fewer iterations in the expensive
solves (γ ≥ 0.3 at a tight prox tolerance), except noise at γ = 0.3 (14%
more). At γ = 0.05, where CG is cheap (tens of milliseconds at most), it
takes 8–40% more.

**Against scikit-image** (`bench/compare_tv_iso.py`, as in the 2026-10-16
section, ms to reach the error). Up to 1e-4 the times match that run, since at
these γ restarts first fire near a gap of 1e-6. At 1e-6 they now save 12–32%.

| Image, γ | fastprox 1e-6, before | fastprox 1e-6, with restarts | scikit-image |
| --- | --- | --- | --- |
| 256², 0.1 | 141.42 | 124.38 | – |
| 256², 0.3 | 1080.47 | 868.04 | – |
| 64³, 0.1 | 824.43 | 562.33 | – |

## 2026-10-17, Apple M5 Pro: anisotropic TV with restarts and a polish from the dual

`bench/compare_tv_aniso.py` as in the 2026-10-16 section (single thread, the
same images, sweeps and reference). Two changes to both solvers (ADBPG in 2D,
the three-block FDPG in 3D):

- **Gradient restarts** (O'Donoghue and Candès), as in isotropic TV. The dual
  is a quadratic over a polyhedron, so restarted FISTA converges linearly.
  On 32³ at λ = 0.5 the iterations to a gap of 1e-10 fell from 644 to 208, and
  to 1e-12 from 2256 to 360.
- **A polish from the dual's regions**, tried at the gap checks. An edge whose
  1D dual is strictly inside its bound is flat at the solution, and the
  solution on given regions has a closed form. The old polish ran only at the
  end and took its regions from jumps of z below a threshold; it merged small
  true jumps and, on 64³ at λ = 0.1, made the output worse (3.2e-6 → 7.8e-5
  relative error at tol 1e-8). The new polished point is the exact solution
  (error 4e-16) after 256 iterations at λ = 0.1. Its gap is then the dual's
  suboptimality only, 1e-4 or less of the iterate's gap. A polished point is
  kept only when its gap is at most 1/100 of the iterate's: at loose
  tolerances, polished points with gap ratios of 0.03–0.4 met the target with
  2–4× the iterate's error.

With restarts in place, the polish takes the iterations to a gap of 1e-12 on
64³ from 516 to 256 at λ = 0.1 and from 448 to 208 at λ = 0.5. A polish costs about as much as one iteration on one thread (6.5 ms on
64³, mostly a serial union-find) and about five on eight threads. Attempts are
spaced by a tenfold fall of the gap, so they add a few percent per solve.

**ms to reach the error.** Before: the 2026-10-16 run.

| Image, λ | Solver | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
| --- | --- | --- | --- | --- | --- |
| 256², 0.1 | fastprox, before | 6.13 | 11.14 | 40.71 | 88.14 |
|  | fastprox, now | 5.00 | **11.13** | **21.34** | **37.84** |
|  | best proxTV | **4.13** (DR2) | 12.16 (DR2) | 39.53 (DR2) | 274.82 (Yang2) |
| 256², 0.5 | fastprox, before | 4.01 | 17.54 | 42.80 | 193.01 |
|  | fastprox, now | **3.25** | **20.79** | **48.15** | **78.65** |
|  | best proxTV | 8.47 (DR2) | 29.55 (DR2) | 87.88 (DR2) | 582.91 (Yang2) |
| 512², 0.1 | fastprox, before | 25.92 | 46.96 | 87.15 | 429.61 |
|  | fastprox, now | 21.43 | **47.55** | **93.27** | **165.81** |
|  | best proxTV | **18.05** (DR2) | 52.45 (DR2) | 182.76 (DR2) | 1291.54 (Yang2) |
| 512², 0.5 | fastprox, before | 16.00 | 90.94 | 255.08 | 1239.12 |
|  | fastprox, now | **12.73** | **100.79** | **278.46** | **440.73** |
|  | best proxTV | 19.89 (DR2) | 129.80 (DR2) | 420.39 (DR2) | 4452.59 (DR2) |
| 64³, 0.1 | fastprox, before | 71.91 | 392.98 | 744.34 | 10461.08 |
|  | fastprox, now | **73.81** | **400.14** | **617.82** | **2245.32** |
|  | best proxTV | 167.15 (Dykstra) | 1315.82 (Dykstra) | 2046.40 (Yang3) | 4133.84 (Yang3) |
| 64³, 0.5 | fastprox, before | 96.49 | 209.86 | 608.86 | 4332.43 |
|  | fastprox, now | **103.75** | **215.18** | **477.75** | **777.42** |
|  | best proxTV | 146.47 (Dykstra) | 1920.85 (Yang3) | 1920.85 (Yang3) | 3703.42 (Yang3) |

Bold marks the fastest of the current runs (fastprox now and proxTV).

In 3D fastprox is now fastest at every accuracy. At 1e-6 it is 1.8× faster
than Yang's ADMM at λ = 0.1 and 4.8× at λ = 0.5 (it was 2.5× and 1.1× slower),
and 4.7× and 5.6× faster than before. In 2D, 1e-6 is 2.3–2.8× faster than
before and 7.3–10× faster than the best proxTV solver. Some medium accuracies
are 2–19% slower than before: 1e-4 except on 256² at λ = 0.1 (7–12%), 1e-3 at
λ = 0.5 (11–19%), and 1e-2 and 1e-3 in 3D (2–8%). There the old end-of-solve polish, which merged jumps below
a threshold, happened to land closer to the solution; fastprox is still ahead
of proxTV at all of them. proxTV's DR2 stays quicker to 1e-2 at λ = 0.1.

## 2026-10-17, Apple M5 Pro: parallel polish for anisotropic TV

Cost of one polish on 64³ (λ = 0.1, 200 iterations with a polish at every
gap check against none, divided by the extra polishes), and whole solves to a
gap of 1e-12 (best of 2). The polish's output is bit-identical to the serial
version: all 72 outputs (z, dual and iteration count) of 8 shapes × 3
tolerances match on 1 and 8 threads.

| | serial polish | parallel polish |
| --- | --- | --- |
| one polish, 1 thread | 7.1 ms | 6.5 ms |
| one polish, 8 threads | 3.9 ms | 1.8 ms |

Per phase, on 8 threads (ms): edge marks 0.16, runs 0.08, unions within
slabs 0.28, crossing unions (serial) 0.40, roots 0.14, labels 0.11, region
sums 0.08 plus the crossing components (serial, over a list of their
entries) 0.20, gap 0.06. The two serial steps are now 40% of the cost. The edge pass
had not vectorised: inside the `parallel_for` lambda its pointers were
captures, not `restrict` parameters. As a separate function it went from
3.56 to 0.64 ms on one thread.

| Image, threads | λ = 0.1, serial → parallel | λ = 0.5, serial → parallel |
| --- | --- | --- |
| 64³, 1 | 2369 → 2227 | 1179 → 1145 |
| 64³, 8 | 394 → 379 | 223 → 217 |
| 128³, 1 | 34933 → 33687 | 14557 → 14188 |
| 128³, 8 | 5406 → 5173 | 2461 → 2380 |
| 1024², 1 | 1262 → 1246 | 5132 → 5080 |
| 1024², 8 | 215 → 183 | 799 → 770 |

Solves run only a few polishes (one per tenfold fall of the gap), so the
whole-solve gain is 1–6% on one thread and 3–15% on eight. Polishing more
often (every threefold fall) was within ±5% either way and was not adopted.

## 2026-10-17, Apple M5 Pro: anisotropic TV in 2D, gap checks without a 1D pass

`bench/compare_tv_aniso.py` on the same images and reference, with finer
sweeps for every solver: fastprox's tolerance in quarter decades from 1e-1
to 1e-12 (decades before) and proxTV's iteration cap in half powers of two
from 1 to 1024 (powers of two before). A solver that stops exactly when its
gap meets the tolerance lands, on a decade grid, at an error up to 10×
below the target it was run for, so the coarse grid misjudged the change
below. "Before" is master run with the same script, so both rows use the
same sweeps. Its numbers differ from the earlier tables by that alone (64³,
λ = 0.1 to 1e-6: 2245 → 1343 ms, an early polish that the decade grid did
not sample).

Three changes, in 2D (ADBPG) unless noted:

- **The gap is checked on the second block step** r = prox_{g2}(u − w) of
  the iteration, whose duals (d + w − u for the first block, −y⁺ for the
  second) are feasible and sum to d − r. A check is then one sweep over r;
  before, it formed z(y) = prox_{g1}(d + y) with another 1D pass, about 60%
  of an iteration. The gap of r is 0.2–1.7 times that of z(y) on these
  images, mostly below 1 at λ = 0.5.
- **Checks are scheduled**: after iterations 1 and 2, then when the rate at
  which the gap fell between the last two checks predicts that it meets the
  target, and at least every 4 iterations. Before, the first check was at
  iteration 4, so no solve stopped sooner: at λ = 0.1 even tol = 1e-1 ran 4
  iterations, while 2 reach 1e-2 error.
- **Polish retries** (2D and 3D share the schedule). After a failed attempt,
  the next one waited for a tenfold fall of the gap, or less when the
  attempt's ratio pz/gp predicted success. At λ = 0.5 that ratio stayed at
  0.1–0.4 for 50–90 iterations, which would never be accepted (the cap is
  0.01), so the prediction retried at nearly every check, each attempt
  costing about 1.2 iterations. In 2D only ratios up to 0.01 now predict.
  In 3D, where a polish costs a third of an iteration, the eager retries
  stay: restricting them there made 1e-4 error 20% slower.

The z returned unpolished is d − q₁ − q₂ for the dual returned (r up to
rounding), and the polish is computed from d and the dual alone. So a warm
start from the dual of a solve still returns its z bit for bit after 0
iterations. A warm start also polishes at its first checks, which a cold
start skips. Warm starts from the dual for d on d + ε sin 7d (λ = 0.5;
40×50, 96×120 and 256²), in iterations:

| tol, ε | before | now |
| --- | --- | --- |
| 1e-6, 10⁻⁴ and 10⁻³ | 0 | 0 |
| 1e-6, 10⁻² | 4–12 | 1–2 |
| 1e-10, 10⁻⁴ | 4 | 1 |
| 1e-10, 10⁻³ | 4, 16, 4 | 6, 18, 6 |
| 1e-10, 10⁻² | 20, 20, 36 (cold 24, 36, 44) | 22, 22, 38 (cold 22, 38, 38) |

At ε = 10⁻² and tol 1e-10 the dual stalls for a while and a warm start
gains little; on 256² nothing.

**ms to reach the error**, single thread

| Image, λ | Solver | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
| --- | --- | --- | --- | --- | --- |
| 256², 0.1 | fastprox, before | 4.60 | 10.52 | **14.77** | 32.21 |
|  | fastprox, now | **2.55** | **8.33** | 16.48 | **22.75** |
|  | best proxTV | 4.02 (DR2) | 12.19 (DR2) | 37.35 (DR2) | 271.90 (Yang2) |
| 256², 0.5 | fastprox, before | **3.10** | 19.58 | 41.55 | 77.07 |
|  | fastprox, now | 5.27 | **19.51** | **39.41** | **72.91** |
|  | best proxTV | 6.42 (DR2) | 28.83 (DR2) | 83.63 (DR2) | 426.98 (Yang2) |
| 512², 0.1 | fastprox, before | 20.50 | 45.61 | **65.86** | 161.56 |
|  | fastprox, now | **10.51** | **35.43** | 72.85 | **135.65** |
|  | best proxTV | 17.68 (DR2) | 52.08 (DR2) | 178.99 (DR2) | 1296.29 (Yang2) |
| 512², 0.5 | fastprox, before | 12.54 | 101.83 | 224.61 | 430.58 |
|  | fastprox, now | **9.76** | **92.65** | **203.07** | **383.16** |
|  | best proxTV | 20.51 (DR2) | 134.16 (DR2) | 419.57 (DR2) | 4549.14 (DR2) |

proxTV from the master run; the branch run gave 1–5% different times. In
3D the three-block method takes the same iterations as before at every
accuracy, and the times are within 1–3% (64³, λ = 0.1: 70.7, 248, 564 and
1314 ms; λ = 0.5: 49.1, 212, 448 and 772 ms).

fastprox is now fastest at every accuracy on every image. At 1e-2 and
λ = 0.1 it is 1.6–1.7× faster than DR2, which was 14–16% faster before
(on the decade grid, 17–20%). Against master, 1e-2 at λ = 0.1 is 1.8–2.0×
faster, 1e-3 0–22% and 1e-6 5–29%. Two cells are slower:

- **1e-4 at λ = 0.1** (11–12%): r reaches it at iteration 13 and z(y) at 12.
  At small λ the gap and the error of r are a little larger than those of
  z(y).
- **1e-2 on 256² at λ = 0.5** (3.10 → 5.27 ms): the grid samples it badly.
  r has error 1.08e-2 at iteration 3 and 8.8e-3 at iteration 4, but no
  quarter-decade tolerance stops at 4: the closest stop at 3 or at 5. On an
  eighth-decade grid the times are 3.19 and 3.09 ms.

## 2026-10-17, Apple M5 Pro: the max-flow engine on graph TV cuts

`bench/bench_maxflow.cpp` (single thread, `-O3`; build line in the file), on
a synthetic image (a ramp with disks and a square, Gaussian noise 0.1, values
about [0, 1]). Each cut problem is graph TV at a level t: terminal x_p − t,
edges γ both ways (diagonals γ/√2 with 8 neighbours). The Dinic baseline is
written in the benchmark (arcs by tail, BFS levels, DFS with current-arc
pointers). Medians of 3 runs, one run for the level sequences.

**One cut at the median level, ms.** Flows agree to 1e-11 or better.

| Grid | γ = 0.02 | | γ = 0.1 | | γ = 0.5 | |
| --- | --- | --- | --- | --- | --- | --- |
| | engine | Dinic | engine | Dinic | engine | Dinic |
| 256², 4 | 2.8 | 10.8 | 2.7 | 11.9 | 18.2 | 43.5 |
| 512², 4 | 6.6 | 26.6 | 12.3 | 90.2 | 87.7 | 324.6 |
| 1024², 4 | 27.3 | 144.8 | 47.2 | 516.0 | 320.3 | 1933.8 |
| 512², 8 | 22.2 | 72.3 | 57.5 | 300.2 | 802.0 | 1310.5 |
| 64³, 6 | 13.5 | 61.0 | 63.0 | 293.1 | 539.9 | 353.3 |

**16 levels of divide and conquer, ms.** Every part of the image is cut at
the middle of its interval in the same solve, down to 2⁻¹⁶ of the value
range. *Reuse* keeps the flow and the trees between levels (the edges across
each cut removed, the terminals shifted), *replant* keeps the flow and
rebuilds the trees, *scratch* builds every level anew with the boundary
terms in the terminals. All three find the same cuts at every level.

| Grid | γ | reuse | replant | scratch |
| --- | --- | --- | --- | --- |
| 256², 4 | 0.02 | 52 | 49 | 43 |
| | 0.1 | 134 | 136 | 215 |
| | 0.5 | 643 | 629 | 749 |
| 512², 4 | 0.02 | 192 | 199 | 184 |
| | 0.1 | 799 | 805 | 1078 |
| | 0.5 | 4623 | 4754 | 5772 |
| 1024², 4 | 0.02 | 790 | 813 | 750 |
| | 0.1 | 3928 | 3964 | 5141 |
| | 0.5 | 33215 | 33650 | 39228 |
| 512², 8 | 0.02 | 462 | 476 | 697 |
| | 0.1 | 2928 | 2957 | 5543 |
| | 0.5 | 9966 | 10579 | 20177 |
| 64³, 6 | 0.02 | 339 | 353 | 418 |
| | 0.1 | 1951 | 1941 | 4336 |
| | 0.5 | 8974 | 9267 | 15406 |

Keeping the flow saves up to 2.2× at γ ≥ 0.1; at γ = 0.02 starting over is
as fast or faster (0.83–0.95× on the 4-neighbour grids). Keeping the trees saves at most 6% over replanting them. A level costs more
than the first cut: on 512² at γ = 0.5, levels 1–15 take 1–2 × 10⁵
augmentations each, over paths of 35–480 arcs that lengthen with depth, and
each augmentation empties one terminal. The level shift changes the sign of
small terminal residuals all over the flat regions of the solution, and their
flow has to cross the region again. Growth is about 5% of the time and
augmentation most of the rest, so tree reuse cannot help here. Graph TV
should cut each part at its own mean level and freeze the parts that come
out constant (see ROADMAP).

## 2026-10-17, Apple M5 Pro: graph TV

`bench/compare_graph_tv.py` (single thread, medians of 3), on the synthetic
images of the max-flow section and on k-nearest-neighbour graphs (k = 8, each
pair once, weights exp(−d²/median d²)) of random points in the unit square
carrying the same pattern plus noise 0.1. Graph TV is timed exact (`tol=0`)
and at `tol=1e-4`; the number of max-flow solves is in parentheses. On
4-neighbour grids graph TV is anisotropic TV, so `TVAniso` solves the same
problem; its errors are against the exact graph TV result. Repeated runs of
the 512² γ = 0.5 cases varied by up to 40%.

**4-neighbour grids, ms.**

| Image, γ | graph TV exact | graph TV 1e-4 (error) | TVAniso 1e-6 (error) | TVAniso 1e-10 (error) |
| --- | --- | --- | --- | --- |
| 256², 0.02 | 31.5 (19) | 32.2 (7e-5) | 11.7 (1e-3) | 20.4 (1e-5) |
| 256², 0.1 | 171.0 (23) | 99.0 (7e-5) | 47.7 (6e-4) | 159.3 (2e-16) |
| 256², 0.5 | 486.6 (18) | 375.3 (4e-5) | 117.9 (8e-4) | 138.0 (1e-5) |
| 512², 0.02 | 128.6 (21) | 125.7 (9e-5) | 48.6 (9e-4) | 87.7 (1e-5) |
| 512², 0.1 | 787.8 (20) | 525.2 (7e-5) | 208.7 (8e-4) | 1077.6 (1e-6) |
| 512², 0.5 | 4264.6 (17) | 2977.1 (8e-5) | 940.2 (4e-4) | 1746.5 (3e-16) |

**8-neighbour grids (diagonals 1/√2) and kNN graphs, ms.**

| Graph, γ | 0.02 exact | 0.02 1e-4 | 0.1 exact | 0.1 1e-4 | 0.5 exact | 0.5 1e-4 |
| --- | --- | --- | --- | --- | --- | --- |
| 256², 8 neighbours | 85.0 (25) | 80.1 | 425.1 (17) | 291.3 | 1106.8 (18) | 746.2 |
| 512², 8 neighbours | 363.1 (23) | 341.5 | 3873.3 (21) | 2076.0 | 8753.4 (18) | 6477.2 |
| kNN, 10⁴ points, 4.6 × 10⁴ edges | 12.4 (17) | 12.1 | 22.4 (18) | 21.1 | 29.9 (13) | 29.2 |
| kNN, 10⁵ points, 4.6 × 10⁵ edges | 145.3 (21) | 142.4 | 300.9 (22) | 274.1 | 645.0 (18) | 518.4 |

Every part is cut at the mean of its solution, so a part whose cut comes out
trivial is finished exactly, and the exact solve takes 13–25 max-flow solves
in every case. The time is in the solves: in the rounds where the large flat
regions are still unfinished, each solve re-routes flow through them (on 512²
at γ = 0.5, rounds 4–14 take 0.3–1.2 s each with nearly every node active).
Restarting each round from zero flow or replanting the trees changes little.
Starting every round from the exact dual (taken from a first solve) takes the
max-flow part from 5.5 s to 0.12 s, but approximate duals from up to 400
iterations of restarted FISTA on the edge dual gave no gain: the dual is far
from unique, and a flow close in value is not close in shape. The
bookkeeping between solves touches only the active nodes and their edges.

**Graphs with hubs** (2026-10-18). The engine rescanned every arc of a node
when it grew and when it was adopted, so a node of high degree was
quadratic: on a star with 10⁵ leaves graph TV took 24.6 s (tree TV: 10 ms),
80% in adoption, as the hub lost its parent once per leaf, the rest in its
growth after every augmentation. Nodes of degree above 32 now keep two scan
positions (src/maxflow/maxflow.hpp): growth resumes where it stopped, and
adoption takes the first valid parent from where it last found one; a node
freed next to such a hub is attached back to its old tree after the
adoption stage instead of activating the hub. Other nodes run the method as
before: applying the same changes to every node changed the trees grown and
cost the grids 25–47%. Single thread, ms (median of 3):

| Graph, γ | before | after |
| --- | --- | --- |
| star, 10⁴ leaves, 0.3 | 124 | 2.7 (tree TV 0.6) |
| star, 10⁵ leaves, 0.3 | 24 590 | 49–53 (tree TV 10) |
| star, 10⁶ leaves, 0.3 | — | 423 (tree TV 316) |
| 256² 4-neighbour grid + 5 hubs of 6553 edges, 0.02 / 0.1 / 0.5 | 264 / 451 / 1239 | 52 / 164 / 717 |

The grids and kNN graphs above are unchanged (within ±2% run to run, one
case 6%), with no node above degree 32.

## 2026-10-18, Apple M5 Pro: tree TV

`bench/compare_tree_tv.py` (single thread, medians of 5; graph TV once).
Random recursive trees carry the pattern of the graph TV section at random
points; combs are an h × h image's rows joined by its first column, the
spanning tree of Kolmogorov, Pock and Rolínek's image experiments; paths
carry a random walk plus noise. Both operators are exact, so only time is
compared.

| Tree, γ | 0.02 | 0.1 | 0.5 |
| --- | --- | --- | --- |
| random, 10⁴ nodes: tree / graph TV | 0.5 / 2.5 | 0.8 / 4.1 | 0.5 / 6.5 |
| random, 10⁵ nodes | 6.6 / 28.3 | 9.0 / 61.0 | 14.3 / 74.2 |
| random, 10⁶ nodes, tree TV only | 69–92 | 77–188 | 69–135 |
| comb 316² | 3.4 / 17.6 | 5.9 / 24.0 | 6.0 / 38.6 |
| comb 1000² | 33.9 / 193.1 | 60.7 / 267.9 | 73.0 / 546.0 |
| star, 3000 nodes | 0.2 / 6.8 | 0.2 / 12.5 | 0.2 / 9.2 |
| star, 10⁵ nodes, tree TV only | 9.7 | 10.2 | 9.6 |
| path, 10⁵ nodes: tree / TV1D | 2.3 / 0.6 | 2.6 / 0.7 | 2.4 / 0.6 |
| path, 10⁶ nodes | 22.4 / 6.4 | 26.8 / 7.6 | 24.3 / 6.0 |
| path, 10⁶ nodes, shuffled labels, tree TV | 26.3 | 31.0 | 27.0 |

The 10⁶-node random trees vary between instances (ranges over two trees and
five runs each). On trees the dynamic program is 4–8× faster than graph TV
and 30–60× on stars; graph TV took 30–60 s on a star with 10⁵ leaves
(measured while writing the tests), because its max-flow rescanned the hub's
arcs each time the hub lost its parent in the search tree (since fixed: 53 ms,
see the graph TV section). On paths the
deque version of Section 4.1 runs at 22–27 ns per node, 3.5–4× the time of
Condat's algorithm in `TV1D`, which needs the signal in order; labels in
random order cost 10–15% more.

Choices measured on the way, on 10⁶ nodes: pairing heaps instead of leftist
heaps (1.3–1.6× faster, since an insertion at the far end of a leftist heap
walks its right spine), a depth-first instead of a breadth-first order (comb
1000²: 54–122 ms to 35–61 ms, as each subtree is contiguous), and the passes
run over positions in that order with the input gathered once (random trees:
110 ms to 57 ms). The deque for paths took them from 35–150 ms with heaps to
21–26 ms. Recomputing the values as region means and the certificate cost
about 1 ms of the 21.

## 2026-10-18, Apple M5 Pro: device-native image TV

`bench/device_tv.py 5` (medians of 5): eight 512² images (a disk and a band
plus noise 0.1, float32), γ = 0.1, solved by `fastprox.torch.tv_iso` /
`tv_aniso` on CPU tensors and on the M5 Pro's GPU (MPS), by
`fastprox.jax.tv_iso` / `tv_aniso` under `jit` on the CPU, and by the C kernels
on one thread and on 15. The backward is the VJP of a random cotangent
(a constant one is trivial). Iterations in parentheses. Times in ms.

| | gap tol | torch CPU | torch MPS | JAX CPU | C, 1 thread | C, 15 threads |
| --- | --- | --- | --- | --- | --- | --- |
| isotropic | 1e-4 | 208 (64) | 127 (64) | 173 | 544 (64) | 98–114 |
| backward | | 269 | 137 | 635 | | |
| isotropic | 1e-6 | 1607 (492) | 960 (492) | 1351 | 4119 (492) | 794–1322 |
| backward | | 1062 | 546 | 3111 | | |
| anisotropic | 1e-4 | 230 (92) | 171 (92) | 184 | 288 (9) | 58 |
| backward | | 96 | 37 | 106 | | |
| anisotropic | 1e-6 | 1573 (624) | 1140 (624) | 1279 | 1545 (59) | 257 |
| backward | | 74 | 31 | 93 | | |

Isotropic TV on the device is the C kernel's method and takes its
iterations; on this integrated GPU, which shares the memory bandwidth of the
CPU, it is 4.3× faster than one C thread and level with fifteen. Anisotropic
TV on the device is a plain dual method: the C kernel's exact 1D steps reach
the same gap in about a tenth of the iterations, so the device path loses to
the C kernel here and pays off only where the data already lives on a GPU.
Each iteration is about a dozen passes over the batch; `torch.linalg.vector_norm`
over the component axis was 100× slower than `torch.hypot` on MPS (9 ms
against 0.07 ms per call) and is not used.

The backward first solved the linearised dual problem by conjugate gradients
without region labels (stencils only): 1500–2900 iterations on a 256² image,
3–21 s on these batches. Regions found on the device bring it to the C
kernels' systems: exact averages for anisotropic TV (no iterations) and
16–104 region-space iterations for isotropic TV (the C kernel: 26–197) with
the earlier flat test.
Finding them by min-label propagation took 225 rounds (670 ms on MPS);
hooking with a scatter of minima and pointer jumping takes a few rounds
(45–160 ms). JAX on the CPU runs the region sums more slowly than PyTorch.

The backward was rerun (2026-10-18) after the flat pixels and edges came to
be read off the dual and the isotropic curvature was capped (ROADMAP.md,
CHANGELOG). Before, with a threshold on |G z|, the isotropic backward took
245 ms (1e-4) and 516 ms (1e-6) on MPS and 343–697 ms on the CPU, with
gradient errors of 0.89 and 0.66 on the DeepInverse comparison's images
(below). Its system is now stiffer (pixels on the sphere with small
gradients), so it takes 3.6–4.8× longer on the GPU even with the C kernel's
Jacobi preconditioner (uncapped: 6–7×), and JAX on the CPU 5–6×. The
anisotropic backward is unchanged in cost (exact region averages); its
gradient is now exact at a gap tolerance of 1e-8 on 128² images.

The table above is from a third run (2026-10-18), after two changes to the
isotropic backward (ROADMAP.md, "Cheaper isotropic TV backward"); before
them it took 948 and 2494 ms on MPS, 1227 and 3282 ms with PyTorch on the
CPU and 5724 and 20 285 ms with JAX:

- **Stopping.** Conjugate gradients stop at the relative error of z that the
  gap certifies, √(2 gap)/‖x‖, instead of 100 ε (float32) or 1e-10
  (float64). On the DeepInverse comparison's images (float32, MPS) the
  gradient error after a solve to 1e-6 was 0.0648 with a residual of 1.2e-5
  (312 iterations), 0.0650 at 3e-3 (144) and 0.066 at 1e-2 (104); in
  float64 on 2 × 128² the error stopped changing once the residual reached
  √tol at every gap tolerance from 1e-4 to 1e-12, where it is 50–100 √tol.
  The C kernel's solves took 2–5× fewer iterations at the same error (34,
  172, 545 and about 1900 at tolerances of 1e-4 to 1e-10, against 182, 502,
  1346 and 3369).
- **Region sums.** The segmented scan by doubling took 18 passes over the
  batch on 512² images, 5.0 of the 6.8 ms of one product with the system on
  MPS. It now scans within blocks of 32 sorted entries (5 passes) and then
  over the blocks' carries, with masks stored as numbers and `addcmul`:
  0.83 ms on MPS, 1.5–1.8 ms against 5.4 on the CPU, with the same
  summation-tree accuracy. With the Hessian product and the vector updates
  fused, an iteration takes 2.9 ms on MPS against 8.

Better preconditioners were measured on the explicit region-space system of
one 128² image (SciPy and PyAMG; iterations to a relative residual of 1e-3
at gap tolerances of 1e-6 and 1e-8): Jacobi 154 and 398; two-level
aggregation over strong couplings with an exact coarse solve, symmetric
multiplicative, 49 and 71; smoothed-aggregation multigrid 26 and 45; ILU with
a drop tolerance of 1e-3 and fill, 3 and 7. A cycle costs about 3–4 products
with the system, so multigrid would save at most about 2×, and the
triangular solves of ILU are sequential; IC(0) broke down (the system is not
an M-matrix). Neither is used.

## 2026-10-18, Apple M5 Pro: isotropic TV against DeepInverse

`bench/compare_deepinv.py 256 8` against DeepInverse 0.4.2's `TVDenoiser`,
which solves the same problem as `TVIso` (both agree to 1e-12 when run to
convergence): an over-relaxed Chambolle-Pock method that stops on the
relative change of x and is differentiated by autograd through its
iterations. Eight 256² images (a disk and a band plus noise 0.1, float32),
γ = 0.1. The reference is the C kernel in float64 to a gap of 1e-13 (410 604
iterations); errors are ‖z − z*‖ / ‖z*‖ over the batch.

**Time to reach an error (ms).** DeepInverse at fixed iteration counts
(`crit = 0`); fastprox's device path (`fastprox.torch.tv_iso`) at gap
tolerances; the C kernel through `fastprox.torch.prox`, which copies to the
host and back.

| Error | DeepInverse, CPU | fastprox device, CPU | DeepInverse, MPS | fastprox device, MPS | C via host (MPS tensors) |
| --- | --- | --- | --- | --- | --- |
| 1e-2 | 145 (100 it) | 20 (tol 1e-3) | 191 | 11 | 52 (tol 1e-4) |
| 1e-3 | 575 (400 it) | 172 (tol 1e-5) | 762 | 92 | 358 (tol 1e-6) |
| 1e-4 | 575 (400 it) | 447 (tol 1e-6) | 762 | 232 | 358 (tol 1e-6) |

DeepInverse's defaults (`crit = 1e-5`, at most 1000 iterations) reach 6.9e-6
in 868 ms on the CPU and 1153 ms on MPS. In float32 a gap tolerance of 1e-8
is below rounding and is never met (100 000 iterations, 98 s): use 1e-7 or
looser.

**Host transfers.** The round trip of the batch (4 MiB) to the CPU and back
takes 0.9 ms on this machine's unified memory, against 50–2100 ms for the
solves: transfers do not dominate, so the C path through the host is a fair
option here, and moving the CUDA kernels of 1.x forward is not called for by
these numbers (a discrete GPU over PCIe was not measured).

**Backward** (gradient of ⟨z, c⟩ for a random c; memory on MPS is the
allocation alive after the forward; rerun after the flat pixels came to be
read off the dual, see below):

| Method | backward, MPS | memory kept | gradient error |
| --- | --- | --- | --- |
| DeepInverse, 50 iterations | 96 ms | 492 MiB | 2.4 |
| DeepInverse, 100 iterations | 182 ms | 992 MiB | 1.1 |
| DeepInverse, 200 iterations | 366 ms | 1992 MiB | 0.32 |
| DeepInverse, 400 iterations | 729 ms | 3992 MiB | 0.062 |
| fastprox, tol 1e-4 | 95 ms | 6 MiB | 0.47 |
| fastprox, tol 1e-6 | 166 ms | 6 MiB | 0.065 |
| fastprox, tol 1e-7 | 237 ms | 6 MiB | 0.066 |

The unrolled backward keeps 10 MiB per iteration of this batch, so its
memory grows with the iterations (4 GiB at 400; 1600 would not fit); the
implicit one keeps the solution and the dual. The gradient errors are
against the exact gradient at the reference solution, the C kernel's VJP
solved to a residual of 1e-10 (18 548 conjugate gradient iterations; the
default cap of 10 000 stops at 3e-5). At a gap tolerance of 1e-6 fastprox
matches DeepInverse's 400 iterations in gradient error with 1/650 of the
memory, in 397 ms for the forward and the backward against 1473 ms (231 + 166
against 761 + 712); in float32 its error levels off near 0.065 (1e-7 is no
better). The backward took 237, 612 and 816 ms at the three tolerances
before its solves stopped at the certified error and its region sums were
split into two levels (above), with the same gradient errors.

Before, fastprox's errors were 0.89 at 1e-4 and 0.66 at 1e-6 (backward 195
and 131 ms): its test for flat pixels, |g_p| ≤ 8·rms(gap), grouped pixels
that are nearly flat at a loose tolerance. Flat pixels are now read off the
dual (‖u_p‖ < γ(1 − δ)), with the curvature capped at the level of the
certified error (CHANGELOG): 1.9–10× more accurate, at 1.2–4.7× the
backward time.

## 2026-10-18, Apple M5 Pro: anisotropic TV against DeepInverse

`bench/compare_deepinv.py 256 8 aniso` against DeepInverse 0.4.2's
`TVL1Denoiser`, `TVDenoiser`'s method with the dual clamped per component,
which solves the problem of `TVAniso` (they agree to 4e-14 run to
convergence). Same images, γ and reference (the C kernel in float64 to a gap
of 1e-13); fastprox's device path is `fastprox.torch.tv_aniso`, a plain dual
method (the C kernel's exact 1D steps and polish stay on the CPU).

**Time to reach an error (ms):**

| Error | DeepInverse, CPU | fastprox device, CPU | DeepInverse, MPS | fastprox device, MPS | C via host (MPS tensors) |
| --- | --- | --- | --- | --- | --- |
| 1e-2 | 138 (100 it) | 17 (tol 1e-3) | 150 | 13 | 42 (tol 1e-4) |
| 1e-3 | 260 (200 it) | 183 (tol 1e-5) | 299 | 122 | 138 (tol 1e-6) |
| 1e-4 | 542 (400 it) | 394 (tol 1e-6) | 598 | 267 | 138 (tol 1e-6) |

DeepInverse's defaults reach 5.9e-6 in 825 ms (CPU) and 893 ms (MPS).

**Backward** (MPS):

| Method | backward | memory kept | gradient error |
| --- | --- | --- | --- |
| DeepInverse, 100 iterations | 160 ms | 398 MiB | 3.7 |
| DeepInverse, 200 iterations | 316 ms | 798 MiB | 1.2 |
| DeepInverse, 400 iterations | 630 ms | 1598 MiB | 0.16 |
| fastprox, tol 1e-4 | 67 ms | 10 MiB | 0.56 |
| fastprox, tol 1e-6 | 45 ms | 10 MiB | 0.099 |
| fastprox, tol 1e-7 | 52 ms | 10 MiB | 0.039 |

The device path is 2.2–12× faster than DeepInverse on MPS and 1.4–8× on
the CPU to the same accuracy, so the faster anisotropic device solver of
ROADMAP.md is not needed to beat it; the C kernel through the host is 2×
faster than the device path at 1e-4. The anisotropic gradient is exact in
float64 once the dual identifies the regions (from a gap tolerance of 1e-8
on 128² images); in float32 its error is 0.04 here at 1e-7, still 4× below
that of 400 unrolled iterations, with 1/160 of the memory.

## 2026-10-19, Google Colab, NVIDIA Tesla T4: image TV against DeepInverse on CUDA

`bench/compare_deepinv.py 256 8 iso` and `... aniso` with fastprox 0.3.0rc1
from PyPI and DeepInverse from pip, on a Colab T4 (15 GiB), the CPU section
skipped. The float64 references were computed on the M5 Pro and read from
the script's cache (on Colab's two CPU cores the isotropic one takes over
half an hour). Same images, γ and error measures as the two sections above.

**Time to reach an error (ms):**

| Error | `TVDenoiser` | `tv_iso` | C via host | `TVL1Denoiser` | `tv_aniso` | C via host |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-2 | 98 (100 it) | 13 (tol 1e-3) | 684 | 89 (100 it) | 16 (tol 1e-3) | 243 |
| 1e-3 | 379 (400 it) | 109 (tol 1e-5) | 4977 | 177 (200 it) | 155 (tol 1e-5) | 1067 |
| 1e-4 | 379 (400 it) | 279 (tol 1e-6) | 4977 | 353 (400 it) | 326 (tol 1e-6) | 1067 |

DeepInverse's defaults reach 6.8e-6 in 573 ms (isotropic) and 5.8e-6 in
528 ms (anisotropic).

**Backward** (memory: allocated after the forward minus before):

| Method | isotropic | memory | error | anisotropic | memory | error |
| --- | --- | --- | --- | --- | --- | --- |
| DeepInverse, 100 it | 250 ms | 992 MiB | 1.1 | 210 ms | 398 MiB | 3.7 |
| DeepInverse, 200 it | 499 ms | 1992 MiB | 0.32 | 415 ms | 798 MiB | 1.2 |
| DeepInverse, 400 it | 1002 ms | 3992 MiB | 0.061 | 843 ms | 1598 MiB | 0.16 |
| fastprox, tol 1e-6 | 281 ms | 6 MiB | 0.065 | 9.3 ms | 10 MiB | 0.099 |
| fastprox, tol 1e-7 | 458 ms | 6 MiB | 0.066 | 9.2 ms | 10 MiB | 0.040 |

The first backward of each method in a run includes one-time costs on the
device (fastprox at tol 1e-4: 316 and 142 ms; DeepInverse at 50 iterations:
249 and 144 ms), so those rows are left out.

On this GPU the forward gains are smaller than on MPS: 6–8× at loose
accuracy, but 1.1–1.4× at 1e-3 to 1e-4, as DeepInverse's iteration is a few
fused operations that a discrete GPU runs well. The backward keeps its
lead: at matched gradient accuracy isotropic TV's forward plus backward
takes 560 ms against 1381 ms (279 + 281 against 379 + 1002) with 1/650 of
the memory, and anisotropic TV's backward is about 90× faster with 1/160 of
the memory and a smaller error. The host round trip of the batch is 1.1 ms,
but the C kernel through the host is slow here, on two CPU cores.

