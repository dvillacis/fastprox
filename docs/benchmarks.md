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
| blocks, 0.1 | **8.2** | 12.6 | 8.5 | 8.6 | 11.1 |
| blocks, 1 | **7.7** | 14.3 | 11.3 | 11.2 | 12.8 |
| blocks, 10 | **9.1** | 12.5 | 11.2 | 11.3 | 11.3 |
| random walk, 0.1 | **6.7** | 10.2 | **6.7** | **6.8** | 8.9 |
| random walk, 1 | **12.2** | 19.3 | 14.1 | 14.2 | 17.3 |
| random walk, 10 | 14.3 | 14.8 | **12.3** | 12.6 | 12.9 |
| white noise, 0.1 | **9.2** | 15.0 | **9.2** | **9.3** | 13.4 |
| white noise, 1 | **9.1** | 17.4 | 12.9 | 12.8 | 15.3 |
| white noise, 10 | 10.8 | 19.5 | **9.7** | **9.7** | 16.8 |

**1D TV, float32, ns per sample** (Condat's code is single precision)

| Signal, λ | fastprox auto | Condat `TV1D_denoise` |
| --- | --- | --- |
| blocks, 0.1 | 9.2 | **8.5** |
| blocks, 1 | **7.8** | 9.2 |
| blocks, 10 | **8.8** | 10.5 |
| random walk, 0.1 | 7.2 | **6.6** |
| random walk, 1 | **12.8** | 13.7 |
| random walk, 10 | **14.5** | 14.9 |
| white noise, 0.1 | 9.9 | **8.5** |
| white noise, 1 | **9.4** | 10.9 |
| white noise, 10 | **10.9** | 14.9 |

**Weighted 1D TV, float64, ns per sample** (edge weights uniform in [0.5, 1.5], times λ)

| Signal, λ | fastprox | proxTV `tautString_TV1_Weighted` |
| --- | --- | --- |
| blocks, 0.1 | 10.1 | **9.0** |
| blocks, 1 | **11.4** | 14.0 |
| blocks, 10 | **13.2** | 14.1 |
| random walk, 0.1 | 14.3 | **12.6** |
| random walk, 1 | **17.6** | 18.5 |
| random walk, 10 | **17.5** | 21.6 |
| white noise, 0.1 | 13.9 | **11.6** |
| white noise, 1 | **15.6** | 18.6 |
| white noise, 10 | 8.1 | **4.9** |

**Simplex projection, radius 1, ns per sample**

| Input | fastprox | Condat `simplexproj_Condat` |
| --- | --- | --- |
| Gaussian | 0.55 | **0.43** |
| uniform | 0.76 | **0.67** |
| one large entry | 0.52 | **0.41** |

At n = 10³ fastprox is faster on all three inputs (1.5–3.8 against 2.3–4.7).

**Reading.** In double precision fastprox's automatic path is the fastest
code or within 2% of it in seven of nine uniform settings (up to 40% faster
at λ = 1) and trails proxTV's linearized taut string at λ = 10 on random
walks (16%) and white noise (12%). In single precision it beats Condat's
code at λ ≥ 1 and trails by 8–16% at λ = 0.1. Weighted problems now use
Condat's algorithm with per-edge radii: fastprox is faster than proxTV's
weighted taut string in five of nine settings and slower at λ = 0.1
(12–20%) and on white noise at λ = 10 (about 1.6×). The simplex projection
is 13–28% slower than Condat's code at large n and faster at small n.

Kernel changes measured in one program on the same inputs (n = 10⁶): the
branch-free Condat updates take 0.65–0.95× the time of the previous kernel
in seven of nine settings and 1.01–1.10× at λ = 10 on smooth signals; the
segment-based taut string takes 0.74–0.83× the time of the previous
double-double version in all nine; weighted Condat with precomputed radii
runs within 10% of uniform Condat.
