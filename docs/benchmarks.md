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
Whether this closes the Neoverse-N2 gap needs a run of the `bench` workflow.

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
