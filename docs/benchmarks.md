# Benchmarks

Single-thread time per sample from `bench/compare_external.py`, which fetches
and builds proxTV and Condat's C code at run time (they are not vendored).
Every external result is checked against fastprox before it is timed, and
every call gets a fresh input: repeating one input lets the branch predictor
learn it and makes the direct algorithms look about three times faster.

## 2026-10-14, Apple M5 Pro (5 performance + 10 efficiency cores), n = 10⁶

Run-to-run variation on this laptop is about ±20%; treat differences smaller
than that as ties. A homogeneous-core benchmark runner is still to be set up
(see ROADMAP.md).

**1D TV, float64, ns per sample**

| Signal, λ | fastprox Condat | fastprox taut string | proxTV linearized | proxTV hybrid | proxTV classic |
| --- | --- | --- | --- | --- | --- |
| blocks, 0.1 | 8.1 | 15.4 | 10.1 | 10.0 | 13.0 |
| blocks, 1 | 12.8 | 19.0 | 11.1 | 10.6 | 12.9 |
| blocks, 10 | 11.6 | 15.1 | 10.2 | 9.6 | 10.3 |
| random walk, 0.1 | 8.5 | 16.5 | 9.2 | 9.3 | 12.3 |
| random walk, 1 | 14.5 | 26.8 | 12.2 | 11.1 | 18.8 |
| white noise, 0.1 | 10.2 | 20.2 | 9.2 | 9.3 | 15.3 |
| white noise, 1 | 17.9 | 28.6 | 18.7 | 16.1 | 21.2 |
| white noise, 10 | 7.0 | 19.7 | 7.9 | 8.0 | 15.1 |

**1D TV, float32, ns per sample** (Condat's code is single precision)

| Signal, λ | fastprox auto | Condat `TV1D_denoise` | Condat `TV1D_denoise_tautstring` |
| --- | --- | --- | --- |
| blocks, 0.1 | 10.5 | 9.0 | 13.7 |
| blocks, 1 | 11.5 | 9.1 | 17.5 |
| blocks, 10 | 10.2 | 9.8 | 12.6 |
| random walk, 0.1 | 9.8 | 8.4 | 15.3 |
| white noise, 0.1 | 9.6 | 9.0 | 16.9 |
| white noise, 1 | 16.9 | 10.3 | 16.5 |
| white noise, 10 | 8.4 | 12.2 | 18.4 |

**Simplex projection, radius 1, ns per sample**

| Input | fastprox | Condat `simplexproj_Condat` |
| --- | --- | --- |
| Gaussian | 0.93 | 0.71 |
| uniform | 1.26 | 1.10 |
| one large entry | 0.97 | 0.76 |

At n = 10³ fastprox is faster than Condat's simplex code on all three inputs
(2.5–5.2 against 3.9–11.8 ns per sample).

**Reading.** fastprox's Condat path is within noise of proxTV's best methods
at small λ and 10–30% slower at λ = 1; in single precision Condat's own code
is up to 40% faster at λ = 1. The taut string is about 1.5× slower than
proxTV's classic taut string, mostly from the compensated (double-double)
sums that keep it within the spec's accuracy bound on long rows. The
simplex projection is within 10–30% of Condat's code at large n. Adversarial
inputs (ramps) are covered separately by `bench/bench_python.py` and the
tests: forced Condat is quadratic there, and the automatic path switches to
the taut string after 4n + 1024 steps.
