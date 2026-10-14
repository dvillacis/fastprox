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

**1D TV, float64, ns per sample** (fastest in bold)

| Signal, λ | fastprox auto | fastprox taut string | proxTV linearized | proxTV hybrid | proxTV classic |
| --- | --- | --- | --- | --- | --- |
| blocks, 0.1 | **12.3** | 22.4 | **12.3** | **12.3** | 16.0 |
| blocks, 1 | **10.4** | 25.1 | 16.0 | 16.0 | 18.2 |
| blocks, 10 | **8.8** | 16.7 | 11.9 | 12.0 | 11.4 |
| random walk, 0.1 | **8.6** | 16.5 | 9.1 | 9.2 | 12.1 |
| random walk, 1 | **12.9** | 24.9 | 15.3 | 15.7 | 18.9 |
| random walk, 10 | 11.9 | 15.4 | **10.6** | 10.8 | 11.2 |
| white noise, 0.1 | **7.7** | 15.8 | 8.3 | 8.3 | 11.9 |
| white noise, 1 | **11.0** | 25.3 | 16.2 | 16.4 | 19.4 |
| white noise, 10 | 6.3 | 15.9 | **5.9** | **5.9** | 10.3 |

**1D TV, float32, ns per sample** (Condat's code is single precision)

| Signal, λ | fastprox auto | Condat `TV1D_denoise` |
| --- | --- | --- |
| blocks, 0.1 | 12.0 | 12.1 |
| blocks, 1 | **10.6** | 13.2 |
| blocks, 10 | **9.0** | 11.1 |
| random walk, 0.1 | 9.1 | 9.0 |
| random walk, 1 | **13.5** | 15.0 |
| random walk, 10 | **12.2** | 12.8 |
| white noise, 0.1 | 8.2 | 8.0 |
| white noise, 1 | **11.4** | 13.6 |
| white noise, 10 | **6.2** | 9.1 |

**Simplex projection, radius 1, ns per sample**

| Input | fastprox | Condat `simplexproj_Condat` |
| --- | --- | --- |
| Gaussian | 0.55 | **0.44** |
| uniform | 0.96 | **0.78** |
| one large entry | 0.79 | **0.59** |

At n = 10³ fastprox is faster on all three inputs (1.7–4.3 against 2.6–5.3).

**Reading.** In double precision fastprox's automatic path is the fastest in
seven of nine settings, by up to 35% at λ = 1, ties at λ = 0.1 on blocks, and
trails proxTV at λ = 10 by 12% (random walk) and 6% (white noise). In single
precision it matches or beats Condat's own code everywhere. fastprox's taut
string, used for weighted problems and as the fallback, is about 1.5× slower
than proxTV's linearized taut string. The simplex projection is 20–35% slower
than Condat's code at large n.

The branch-free Condat kernel (this release) against the previous one, both
compiled into one program and interleaved on the same inputs (n = 10⁶,
float64): 0.65–0.95× the time in seven settings, 1.01× on white noise at
λ = 10 and 1.10× on the random walk at λ = 10.
