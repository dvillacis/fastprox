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
