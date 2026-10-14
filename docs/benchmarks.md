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
