# barrier_mc — the error bar a barrier option's Monte Carlo prints measures the wrong error

Measuring what daily monitoring costs a barrier option, in C++20 with no
dependencies: a near-exact benchmark built from Gaussian convolutions, the eight
Reiner–Rubinstein closed forms to check it against, a coupled Brownian-bridge
Monte Carlo, and the Broadie–Glasserman–Kou continuity correction.

It exists to measure one thing:

> **A barrier Monte Carlo converges correctly to its own answer while sitting a
> long way from the answer it is usually taken to approximate, and its standard
> error cannot see the difference.** At daily monitoring the gap between the
> discretely-monitored contract and the continuous closed form is **11.83% of the
> price and 80 Monte Carlo standard errors** at two million paths. It is bias, not
> variance: it shrinks like `sqrt(dt)` in the monitoring interval and not at all
> in the path count, so buying more paths tightens the interval around a number
> that was never the one being asked for.
>
> Moving the barrier to `H·exp(-beta·sigma·sqrt(dt))`, with
> `beta = -zeta(1/2)/sqrt(2*pi) = 0.5825971579`, removes **1854×** of that gap at
> daily monitoring and costs nothing.

The second number is the interesting one, because `zeta(1/2)` has no business
being in an option price. It is there for a reason that this repo derives and
checks for itself; see [§3](#3-where-05826-comes-from).

Measured on an Apple M1 (8 cores, 8 GB, macOS 15.6), Apple clang 17, `-O2`, no
`-ffast-math`. Every table below is reproduced by `./scripts/run_all.sh`, and the
transcripts are committed under `results/`.

---

## Contents

1. [How big the gap is](#1-how-big-the-gap-is)
2. [What the correction buys](#2-what-the-correction-buys)
3. [Where 0.5826 comes from](#3-where-05826-comes-from)
4. [Why the estimator is built this way](#4-why-the-estimator-is-built-this-way)
5. [Where this stops working](#5-where-this-stops-working)
6. [Predictions, written before measuring](#6-predictions-written-before-measuring)
7. [What is checked against what](#7-what-is-checked-against-what)
8. [Eight wrong numbers that looked right](#8-eight-wrong-numbers-that-looked-right-and-what-caught-each)
9. [What this does not cover](#9-what-this-does-not-cover)
10. [Build and run](#10-build-and-run)

---

## 1. How big the gap is

Headline contract: a down-and-out call, `S=100 K=100 H=95 r=0.05 q=0 sigma=0.30
T=0.2`. Continuous closed form: **4.004229446969**.

`m` is the number of monitoring dates; `m=50` over 0.2 years is daily monitoring,
which is what a real contract specifies.

| m | exact discrete | gap | gap % | plain MC s.e. | **gap / s.e.** |
|---|---|---|---|---|---|
| 25 | 4.640824619 | +0.63659517 | +15.90% | 5.94e-03 | **107.3** |
| 50 | 4.477770915 | +0.47354147 | +11.83% | 5.89e-03 | **80.4** |
| 100 | 4.350637301 | +0.34640785 | +8.65% | 5.85e-03 | **59.2** |
| 200 | 4.255101572 | +0.25087212 | +6.27% | 5.82e-03 | **43.1** |
| 400 | 4.184630815 | +0.18040137 | +4.51% | 5.80e-03 | **31.1** |
| 800 | 4.133312540 | +0.12908309 | +3.22% | 5.78e-03 | **22.3** |
| 1600 | 4.096270661 | +0.09204121 | +2.30% | 5.76e-03 | **16.0** |

The "exact discrete" column is not a simulation. It is `m` Gaussian convolutions
on a log-grid (`exact.cpp`), with no sampling error of any kind. The Monte Carlo
appears only as an independent check, and agrees with it to within ±1.8 standard
errors at every row.

A caution about the benchmark's own error bar, because it is easy to quote the
wrong quantity. `exp_convergence` prints a Richardson column running from 2.3e-06
at `m=25` to 2.4e-05 at `m=1600`. That is the error of the FINE GRID, not of the
extrapolated value actually reported, the extrapolation is the whole point of
computing it. Setting the 2.4e-05 next to the 1.6e-06 residual of §2 and
concluding the residual is unresolvable is a category error; the extrapolated
price's true error at that row is nearer 1e-07, which the grid ladder in §2
demonstrates directly instead of asserting it.

**The exponent, and why the obvious fit reads wrong.** Regressing `log|gap|` on
`log(dt)` gives slope **0.4665 ± 0.0046** with R² = 0.99952, a confident-looking
number that disagrees with the theoretical 1/2 by 7%. The successive ratios show
what is happening: 0.7439, 0.7315, 0.7242, 0.7191, 0.7155, 0.7130 against
`1/sqrt(2) = 0.7071`. They are converging, from above, and have not arrived by
`m=1600`.

The reason is that the gap is not a single power. Fit the two-term expansion

```
gap = c1 * sqrt(dt) + c2 * dt
```

and it describes the whole table to a residual of ~1e-04 on gaps of order 0.5,
with `c1 = 8.38088`, `c2 = -14.130`. A single-power fit through a two-term
expansion returns an average of the two exponents; the `c2·dt` term is 15.1% of
the leading term at `m=25` and still 1.9% at `m=1600`.

**Testing the value of beta, not just the exponent.** The obvious test is to
compare `c1` against its closed-form prediction `-beta·sigma·H·dV/dH` = 8.39948
(with `dV/dH = -0.50587030`); the fit gives 8.38088, agreeing to 0.221%.

That test is weaker than it looks, and the repo says so instead of banking it:
0.221% is the truncation error of a two-term basis, and it moves to 0.06% if the
fit window starts at `m=200`, to 0.14% with a `dt^1.5` term added, and to 0.77% on
a different contract. The headline case happens to land on the flattering side.

The test that does survive has no basis and no window. For each `m`, ask what
barrier shift the *price* implies — solve

```
barrier_cont(H·exp(-b·sigma·sqrt(dt))) = exact_discrete(m)
```

for `b` by bisection. If the correction is right, `b -> beta`; and because the
residual is `O(dt^1.5)` while the price's sensitivity to `b` is `O(sqrt(dt))`, the
approach must be *linear* in `dt`, so one Richardson step finishes it.

| m | b implied | b − beta | ratio to previous |
|---|---|---|---|
| 25 | 0.58027903 | −2.318e-03 |, |
| 50 | 0.58224220 | −3.550e-04 | 6.53 |
| 100 | 0.58240742 | −1.897e-04 | 1.87 |
| 200 | 0.58250707 | −9.009e-05 | 2.11 |
| 400 | 0.58255373 | −4.343e-05 | 2.07 |
| 800 | 0.58257602 | −2.114e-05 | 2.05 |
| 1600 | 0.58258673 | −1.043e-05 | 2.03 |

The ratios settle on 2, confirming the predicted linear approach. One Richardson
step on the last pair gives **0.5825974361** against
`-zeta(1/2)/sqrt(2*pi) = 0.5825971579`, **agreement 2.8e-07**, seven significant
figures of a zeta constant recovered from option prices alone.

One thing not to claim: this and the residual test in §2 are **not** independent
checks. Both are the first-order expansion of the same shifted closed form. The
genuinely independent evidence is (a) the residual *exponent* — a beta wrong by a
relative `eps` would leave an `O(sqrt(dt))` residual, not the measured
`dt^1.5`, and (b) the Spitzer route in §3, which never looks at an option price.

## 2. What the correction buys

`residual` is the exact discrete price minus the continuous price at the shifted
barrier. The validity ratio is `log(S/H) / (sigma·sqrt(dt))`, the quantity
Broadie–Glasserman–Kou require to be large.

| m | validity ratio | gap | residual | \|gap\|/\|residual\| | residual · m^1.5 |
|---|---|---|---|---|---|
| 25 | 1.912 | +0.6365952 | −2.127e-03 | 299 | −0.2658 |
| 50 | 2.703 | +0.4735415 | −2.554e-04 | **1854** | −0.0903 |
| 100 | 3.823 | +0.3464079 | −1.037e-04 | 3341 | **−0.1037** |
| 200 | 5.407 | +0.2508721 | −3.658e-05 | 6858 | **−0.1035** |
| 400 | 7.646 | +0.1804014 | −1.290e-05 | 13979 | **−0.1032** |
| 800 | 10.814 | +0.1290831 | −4.551e-06 | 28365 | **−0.1030** |
| 1600 | 15.293 | +0.0920412 | −1.615e-06 | 56997 | **−0.1033** |

**The residual is `O(dt^1.5)`, which is better than advertised.** The theorem
guarantees only `o(sqrt(dt))`. The `residual · m^1.5` column is flat to 0.7%
across `m = 100…1600` — a sixteenfold range, and the fitted slope over that
window is **1.5016**. The flatness is the stronger statement; the slope over all
rows reads 1.6327 because the two coarsest rows are not yet in the asymptotic
regime.

**Where that regime begins is itself a measurement.** `residual · m^1.5` is 13%
off the asymptote at `m=50` and 2.6× off at `m=25`. So the expansion becomes
usable at a validity ratio of about **3.8**, not at the ratio of 2 that
`log(S/H) >> sigma·sqrt(dt)` might suggest, which is the operationally useful
form of the theorem's hypothesis, and it comes out of the data, not out of
the paper.

**And the last two rows only became trustworthy after a bug was fixed.** They used
to drift (−0.1049, −0.1113), which looked like the power law failing at fine
monitoring and was in fact this repo's own Richardson step being wrong — see §8.
`exp_correction` still prints a grid self-check, because a power law confirmed
only where the measuring instrument runs out of resolution is not confirmed:

| m | benchmark grid | residual · m^1.5 |
|---|---|---|
| 400 | 2^12 / 2^13 | −0.10598 |
| 400 | 2^14 / 2^15 (default) | −0.10324 |
| 400 | 2^15 / 2^16 | −0.10323 |
| 1600 | 2^12 / 2^13 | −0.29263 |
| 1600 | 2^14 / 2^15 (default) | −0.10335 |
| 1600 | 2^15 / 2^16 | −0.10265 |

**And it preserves the hedge, which is the part that decides anything.** A price
that is close is worth little if the delta is not: the risk on a barrier book is
carried in the hedge ratio. `discrete_conv` already has the whole value function
on its grid, so delta costs two array reads.

| contract | m | validity ratio | exact δ | shifted δ | uncorrected δ |
|---|---|---|---|---|---|
| headline, daily | 50 | 2.70 | 0.746772 | 0.746657 (**−0.02%**) | 0.797614 (+6.81%) |
| headline, 252 fixings | 252 | 6.07 | 0.773972 | 0.773973 (**+0.00%**) | 0.797614 (+3.05%) |
| barrier 50bp from spot | 252 | 0.40 | 1.154489 | 1.333306 (**+15.49%**) | 1.390217 (+20.42%) |

The first two rows are the production argument for the whole technique: shift the
barrier and the closed form keeps *both* its price and its analytic greeks, so
discretely-fixed barriers can stay in the pricer that gets called on every quote.
Measured on this machine, that pricer costs **~100 ns** against a few hundred
milliseconds for the convolution benchmark at m=50 (both vary run to run by tens
of percent; `results/exp_correction.txt` has the figures from the committed run). That ratio is not a lattice-versus-formula
benchmark and should not be quoted as one — this benchmark deliberately solves the
whole value function on two grids and extrapolates, for six digits it does not
need to price with. The defensible statement is the order of magnitude: a closed
form is tens of nanoseconds, any grid method is milliseconds.

The third row is where that licence expires, and it is the case the rest of this
README does not otherwise cover. With the barrier 50bp from spot the shifted price
is still 4.60% wrong (1.661330 against 1.741485) and the delta is 15% wrong — the
correction buys almost nothing exactly where the hedge is most dangerous. A
correction with no measured failure mode is a paper; the operational form of the
result is the threshold, and on these contracts it sits near a validity ratio of
about 3.

## 3. Where 0.5826 comes from

`beta = -zeta(1/2)/sqrt(2*pi)` is quoted in every paper that uses the correction
and derived in almost none of them. It falls out of three steps, each checked
separately in `exp_overshoot`:

**Step 1, Spitzer's identity.** For any random walk with `S_0 = 0`,

```
E[max_{k<=n} S_k] = sum_{k=1..n} E[S_k^+] / k
```

Checked against brute-force Monte Carlo at 4×10⁶ paths: agreement within ±1.6
standard errors at `n = 1, 2, 5, 20, 100`.

**Step 2, Gaussian steps.** `S_k ~ N(0,k)`, so `E[S_k^+] = sqrt(k)/sqrt(2*pi)`
and every trace of probability leaves the identity:

```
E[M_n] = (1/sqrt(2*pi)) * sum_{k=1..n} k^{-1/2}
```

**Step 3 — Euler–Maclaurin.**

```
sum_{k=1..n} k^{-1/2} = 2*sqrt(n) + zeta(1/2) + (1/2)*n^{-1/2} + O(n^{-3/2})
```

The constant term is the analytic continuation of the zeta function to `s = 1/2`
— **the regularised value of a divergent sum**. That is the whole answer to why a
zeta value appears in an option price: the expected maximum of a Gaussian random
walk is a partial sum of `k^{-1/2}`, and that sum's Euler–Maclaurin constant is
`zeta(1/2)`.

| n | `sum k^{-1/2} − 2·sqrt(n)` | next-term check | recovered beta |
|---|---|---|---|
| 100 | −1.410396175216 | 0.999167 | 0.5826137804 |
| 10⁴ | −1.455354550476 | 0.999992 | 0.5825971746 |
| 10⁶ | −1.459854508851 | 1.000000 | 0.5825971580 |
| 10⁸ | −1.460304508808 | 1.000000 | **0.5825971579** |
| limit | −1.460354508810 = `zeta(1/2)` | 1 | 0.5825971579 |

The "next-term check" is `(sum − 2·sqrt(n) − zeta(1/2)) · 2·sqrt(n)`, which
confirms the `(1/2)·n^{-1/2}` term to six decimals. Recovered beta agrees with
`-zeta(1/2)/sqrt(2*pi)` to **ten significant figures**.

Beta is also recoverable from simulation alone, using no identity at all —
`beta_hat = sqrt(2n/pi) − E[M_n]_MC + (1/2)/sqrt(2*pi*n)` gives 0.580272,
0.577521, 0.578074 at `n = 25, 100, 400`, each within 1.7 standard errors. The
standard error grows like `sqrt(n)` while the finite-`n` bias shrinks like
`n^{-3/2}`, so this route has a floor; it is a confirmation, not the source.

**The route not taken.** The natural way to measure an overshoot is to simulate a
walk until it crosses a level and average the excess. That program never
finishes: the first-passage time of a *driftless* random walk has infinite
expectation, so a batch always has stragglers, however long you wait. The Spitzer
route does bounded work for a bounded answer.

**One numerical detail that matters at these sizes.** `sum k^{-1/2}` at `n = 10⁸`
accumulates 10⁸ terms of order 1e-04 into a total of 2e+04. Naive summation loses
1.85e-09 against the Kahan-compensated value, small, but the quantity being
extracted from that sum sits at the 1e-05 level, so the compensation is not
optional. `exp_overshoot` prints both.

## 4. Why the estimator is built this way

Measuring the gap by pricing the discrete contract and subtracting the closed
form is hopeless: the gap falls like `sqrt(dt)`, so resolving it to fixed
relative accuracy needs paths growing like `1/gap²`.

The estimator used instead computes, **on each path**, both the discrete payoff
(a 0/1 indicator) and the Brownian-bridge payoff (the probability the path stayed
above the barrier at every instant, given the sampled points — a closed form from
the reflection principle). The second is an unbiased estimator of the
*continuous* price at any `m`, being a conditional expectation of the indicator
the first uses; their per-path difference is an unbiased estimator of exactly the
gap, and it is identically zero on every path that never came near the barrier.

| m | var(coupled difference) | var if run independently | ratio |
|---|---|---|---|
| 25 | 6.80 | 139.11 | 20.4× |
| 50 | 5.42 | 138.96 | 25.6× |
| 200 | 3.04 | 138.34 | 45.5× |
| 1600 | 1.18 | 137.70 | **116.5×** |

In paths needed to pin the gap to 1%: **2.4×10⁵ coupled against 6.2×10⁶
independent** at `m=50`, and **1.4×10⁶ against 1.6×10⁸** at `m=1600`.

The bridge estimator's unbiasedness is a testable structural claim, not a
decoration: across `m = 25…1600` the discrete price moves by **0.5474** while the
bridge price moves by **0.0021**. `test_mc` asserts both, and also asserts that a
barrier placed out of reach makes the two estimators agree path-by-path and
reproduce the vanilla Black–Scholes price — a control that would catch a bias in
the path construction itself, which every test that looks only at their
*difference* would miss.

Levels are **nested**: a path is generated once on the finest grid and every
coarser `m` is obtained by taking every 2nd, 4th, 8th point, so all seven
estimates are positively correlated and share a terminal value. That is also why
the generator is counter-based (Philox4x32-10, normals by inversion): a path's
`d`-th normal must be a pure function of `(path, d)` regardless of thread count or
consumption order. `test_mc` asserts the answer is identical across thread counts.

Antithetic sampling, for completeness, is worth **1.40×** here (correlation
`rho = -0.288` between a path and its reflection; efficiency is `1/(1+rho)`)
well short of the factor it earns on a vanilla, the terminal payoff is
monotone in the driving normals but the knock-out indicator is not, so a path and
its reflection are rarely both near the barrier.

## 5. Where this stops working

Everything above is a Black–Scholes result, and two of the claims lean on the
diffusion being the only thing that moves the price. Under a Merton
jump-diffusion (`exp_jump`), they fail differently.

Controls first: the jump compensator is checked by `E[S_T]·e^{-(r-q)T}/S_0`, which
lands within one standard error of 1 at `lambda = 0, 1, 5` (that standard error
is ~1e-04, so the digits the transcript prints past the fourth are noise, not
precision); realised jump counts match `lambda·T`; and at
`lambda = 0` the jump code reproduces the pure-diffusion benchmark and closed form
to within ±0.83 standard errors.

**The Brownian bridge stops being exact, at `O(dt)`.** Between monitoring dates
the path is no longer a Brownian bridge but a bridge interrupted by jumps, so
applying the endpoint formula misses every crossing a jump caused and returned
from. Against a jump-aware estimator that conditions on the jump times and sizes:

| m | naive bridge | jump-aware | bias (s.e.) | bias · m |
|---|---|---|---|---|
| 25 | 4.609262 | 4.538370 | +0.070892 (0.000780) | +1.7723 |
| 50 | 4.601434 | 4.564544 | +0.036890 (0.000567) | +1.8445 |
| 100 | 4.594314 | 4.575779 | +0.018535 (0.000401) | +1.8535 |
| 200 | 4.593437 | 4.584780 | +0.008658 (0.000265) | +1.7315 |
| 400 | 4.596019 | 4.591687 | +0.004333 (0.000187) | +1.7331 |

The standard error is the *coupled* one; both estimators run on the same paths,
so the difference is accumulated per path, not formed from two separately
averaged means. That matters: the individual estimators have standard errors near
0.006, so an uncoupled bound would call the last two rows insignificant, while the
coupled one puts every row between 23 and 91 standard errors from zero.

`bias · m` is roughly constant, so the error is `O(dt)`: the chance of a jump
inside an interval is `lambda·dt`, and a jump is precisely what the endpoint
formula cannot see. Under pure diffusion this column is identically zero.

**The correction survives, but stops being asymptotically exact.** Stated as
rates:

```
pure diffusion    gap = O(dt^0.5),   residual = O(dt^1.5)
with jumps        gap = O(dt^0.5),   residual = O(dt^1.0)
```

The gap keeps its `sqrt(dt)` scaling in both cases — a fourfold refinement halves
it, measured at `lambda=5` as ratios 0.5035 and 0.5069. The residual does not: at
`lambda=5` it falls by 4.08× and 4.34× per fourfold refinement, i.e. like `dt`,
not like `dt^1.5`. The barrier shift compensates for a *Gaussian* overshoot, and
the part of the gap caused by a jump crossing and returning inside an interval is
untouched by it. So the improvement factor grows like `sqrt(m)` with jumps
instead of like `m` without them, still worth applying, since it is free, but no
longer converging away.

A caveat the experiment prints instead of hiding: at `lambda = 0` the residual is
~4e-05, an order of magnitude *below* what two million paths can resolve, so
those rows of `exp_jump` are noise and their improvement figures mean nothing.
That is not a defect of the experiment; it is the reason the rest of the repo
measures the residual against the convolution benchmark instead of by simulation.

## 6. Predictions, written before measuring

| Quantity | Predicted | Measured |
|---|---|---|
| exponent of the gap in `dt` | 1/2 | 0.4665 by a one-term fit — **the prediction looked wrong and was not**; the two-term expansion recovers it, see §1 |
| residual after the correction | `o(dt^0.5)`, expected `O(dt)` | **`O(dt^1.5)`**, better than the theorem promises |
| leading coefficient `c1` | `-beta·sigma·H·dV/dH` = 8.39948 | 8.38088, **but the agreement is basis-dependent**; replaced by an inversion giving beta to 2.8e-07 |
| `beta` | `-zeta(1/2)/sqrt(2*pi)` = 0.5825971579 | 0.5825971579 via Spitzer (10 digits); 0.5825974361 from implied barrier shift (2.8e-07) |
| validity condition | degrades once `log(S/H) ≲ sigma·sqrt(dt)` | asymptote reached only at ratio ≈ **3.8**; 13% off at ratio 2.7, 2.6× off at 1.9 |
| correction under jumps | survives to leading order, since jumps are `O(dt)` and the overshoot is `O(sqrt(dt))` | confirmed — but the **residual degrades from `dt^1.5` to `dt`** |
| benchmark quadrature at the barrier | `O(h^2)` with a half-weight node | confirmed: 4.26e-06 → 1.06e-06 → 2.66e-07 |

## 7. What is checked against what

Nothing in this repo is checked against itself.

| Claim | Independent reference |
|---|---|
| all 8 barrier closed forms | a separate scipy implementation, agreement to 1e-13, six parameter regimes |
| barrier formulas, again | `in + out = vanilla` parity to 1e-11 over a 63-case sweep |
| convolution benchmark, `m=1` | the barrier is inert, so the answer must be vanilla Black–Scholes, 1e-08 |
| convolution benchmark, `m=2` | an independent 1-D adaptive quadrature, 1.6e-09 |
| benchmark convergence order | error ratio under grid doubling must be 3–5× |
| Monte Carlo, all `m` | the convolution benchmark, ±1.8 s.e. across 7 levels |
| bridge estimator | the continuous closed form, at every `m` |
| path construction | far-barrier control reproduces vanilla Black–Scholes |
| Spitzer's identity | brute-force Monte Carlo, ±1.6 s.e. |
| Philox4x32-10 | the published digits-of-π known-answer vector |
| FFT | a direct O(n²) DFT, and a naive linear convolution |
| Welford | sample variance of 1..n, known in closed form |
| thread-independence | 1 thread vs 5 threads, to 1e-10 |
| jump simulator | the martingale identity `E[S_T]e^{-(r-q)T} = S_0`, and `lambda=0` collapsing onto the diffusion benchmark |

`ctest` runs 10 suites including ASan/UBSan variants; all pass, with zero warnings
at `-Wall -Wextra -Wpedantic`. `-UNDEBUG` is forced per test target, because
Release sets `-DNDEBUG` and silently no-ops every `assert`.

**Mutation testing** (`tools/mutants.sh`) injects fourteen defects of the kind this
project is actually about — each produces plausible output, not a crash —
and checks the suite notices. **It kills thirteen of fourteen**
(`results/mutants.txt`). Three of the original survivors were real holes and are
now closed; the fourth is genuine and is described below instead of quietly dropped. All three closed ones had the same shape: *the code was exercised only
by an experiment, and `ctest` does not run experiments.*

- `beta-sign`, flipping the barrier shift toward spot instead of away, survived
  because **the headline formula of the project had no unit test at all**.
  `test_bgk.cpp` now asserts the structural claim, not a hardcoded price:
  the shifted continuous price must be closer to the benchmark than the unshifted
  one, by a factor that grows as monitoring gets finer.
- `welford-naive`, accumulating M2 against the post-update mean, changed no
  assertion anywhere. `test_bgk.cpp` now checks the variance of `1..n` against its
  closed form.
- `richardson-fixed-3`, restoring the wrong extrapolation denominator from §8,
  survived every price check in `test_exact.cpp`, the price is still correct to
  six digits, which is why the bug lived long enough to bend a published exponent.
  What it cannot survive is grid refinement: the extrapolated value settles by
  ~16× per refinement when the ratio is right and far more slowly when it is not,
  measured at 3.4e-08 against 1.99e-06. That is now an assertion.
The fourth exposed that **`jump.cpp` had no unit tests at all**, so `test_jump.cpp`
now exists: it checks that `lambda=0` reproduces the pure-diffusion benchmark and
closed form, that the discounted asset is a martingale (which is what validates
the jump compensator), and that the estimators stay finite and correctly ordered
at an intensity where intervals routinely hold several jumps.

**`jump-unsorted` still survives, and this is the honest limit of the suite.**
Reversing the sort of the jump times within an interval is a real defect, the
sub-interval bridges are then built in the wrong order. Two things stop the tests
from seeing it. First, it does not produce a NaN: the guards `u > t_prev` and
`dt > t_prev` cause the mis-ordered sub-intervals to be skipped instead of given
a negative time span, so the result is *wrong but finite* — precisely the failure
mode this repo is about. Second, and more fundamentally, **the jump-aware bridge
has no independent reference to be checked against.** Every other estimator here
is anchored to something computed a different way; this one is the reference.
Measured directly: at `lambda=50, m=25` the mis-sorted version returns 7.9428
against the correct 7.8930, a difference of 0.05 against a Monte Carlo standard
error of 0.062, under the noise at any path count a laptop will run.

The obvious structural test, "the jump-aware bridge estimates the continuous
price, so its mean must not drift with `m`", the analogue of the check that works
for the Gaussian bridge in §4 — was tried and does not hold: at `lambda=50` the
correct estimator itself drifts by 0.54 between `m=25` and `m=200`, because
`lambda*dt = 2` there and the Poisson sampler truncates at six jumps per interval.
That truncation is documented in `jump.cpp` as valid for `lambda*dt < 0.1`, which
every row of `exp_jump` respects; the test's `lambda=50` is a deliberate stress
outside that range and asserts nothing about accuracy.

The harness itself has been wrong twice, both times in the direction that
flattered it. An early run reported 10 of 10 killed "by build", which was
worthless: it copied `include src tests` but not `experiments/`, so every mutant
died at CMake configure time for a reason having nothing to do with its defect.
Build failures are now a separate outcome and are never counted as kills.

The second recurred three times. Each mutant is a literal string substitution, so
any refactor touching a matched line silently disarms it, and the matrix runs for
twenty minutes before reporting a mutant it never applied. `tools/mutants.sh
--check` now verifies every pattern matches exactly once and takes about a second;
the full run does that check first. The summary counts *not applied* separately
from *survived*, because they mean opposite things: a survivor is a gap in the
tests, an unapplied pattern is a gap in the harness.

## 8. Eight wrong numbers that looked right, and what caught each

This is the project's own subject applied to itself. Not one of these was a
crash, a NaN or an obviously silly value; every one produced a plausible,
monotone, converging number, which is exactly why the list is organised by *what
detected it*, not by what it was.

| what was wrong | how it looked | what caught it |
|---|---|---|
| kernel orientation | 2% low, converged under refinement | `m=1`, where the barrier is inert and the answer must be the vanilla price |
| full weight at the barrier node | `O(h)` instead of `O(h^2)` | an error-ratio test asserting 3–5× per grid doubling |
| Richardson denominator | bent the residual exponent at fine `m` | `residual · m^1.5` failing to be flat |
| 53-bit uniform reaching 1.0 | never fires in 200k draws | asserting on the all-ones input directly |
| FFT twiddle drift | 1.4e-11 instead of 1e-15 | comparison against a direct DFT at large `n` |
| sign of the finite-`n` beta correction | converged smoothly to 0.5825573 | the exact value, known to 10 digits |
| `(H/S)^{2mu}` overflowing at small sigma | a clean 0.0 for a contract worth 4.877 | in + out = vanilla, once the sweep included small sigma |
| extrapolating two grids of equal spacing | `inf`, from a division by zero | walking the grid size down until the node count stopped changing |
| extrapolating across a change of stencil | 1.1437 against a converged 1.1545 | comparing the extrapolated delta to the finer grid's own |

**1. Kernel orientation in the convolution benchmark.** Backward induction needs
the transition density as a function of the *destination* given the source.
Writing it the other way round computes the adjoint operator. The resulting
scheme conserves mass, converges under grid refinement, and returns a positive,
monotone price about 2% low, a constant −0.1106 offset that no amount of
refinement removes, because it is a model error wearing a discretisation error's
clothes. It was caught only by pricing `m=1`, where the barrier cannot bite and
the answer must equal the vanilla price. That check is now the first assertion in
`test_exact.cpp`, before anything that the benchmark alone vouches for.

**2. First-order quadrature at the barrier.** Knocking out leaves the value
function discontinuous, and a uniform-weight sum across a jump is `O(h)` — the
same order as the effect being measured. Putting a node exactly on the barrier
and giving it *half* weight makes the sum a trapezoid rule on the live side and
restores `O(h^2)`. Measured against an independent quadrature at `m=2`: without,
1.23e-03 → 6.15e-04 → 3.07e-04; with, 4.26e-06 → 1.06e-06 → 2.66e-07.

**3. A sign error in the finite-`n` correction for beta.** `beta = -d +
(1/2)/sqrt(2*pi*n)`; subtracting the term instead of adding it moves the answer by
twice the term and produced a table that converged smoothly to 0.5825573, right
to four digits, wrong in the fifth, and entirely plausible unless compared
against the exact value.

**4. The Richardson step assumed a grid ratio it did not have.** `discrete_exact`
extrapolates with `(fine - coarse)/3`, which is correct when doubling the node
count halves the spacing. It does not, quite: the spacing is solved for from an
*integer* node count between barrier and spot so both land on nodes, and that
count goes 259 → 517, a ratio of 1.99614. The correct denominator is
`ratio^2 - 1 = 2.98457`. Using 3 put a 0.5% error into the correction term, and
since the correction is the only thing between this benchmark and its own
discretisation error, that 0.5% was the *dominant* error in the whole result. It
was visible as the `residual · m^1.5` column drifting at `m = 800, 1600`, which
read as the mathematics failing. Two lines, worth about a factor of 16, and it is
what moved the fitted residual exponent from 1.4555 to 1.5016.

**5. The uniform generator could return exactly 1.0.** `u01` packed 53 bits and
added half a ulp, with a comment asserting the endpoints were unreachable *by
construction*. They were not: for the all-ones input, `(double)(2^53 - 1) + 0.5`
is exactly halfway between two representable doubles and round-half-to-even takes
it **up** to `2^53`, giving `u = 1.0` and `ninv(u) = +inf`. Dropping to 52 bits
makes the largest value `2^52 - 0.5`, which is exactly representable, so
`u_max = 1 - 2^-53` closes the interval for real. A 200,000-draw sweep never fires
this; the test that catches it asserts on the extreme input directly.

**6. The FFT's twiddle factors drifted.** Advancing the twiddle purely by
recurrence is the textbook form and its relative error grows *linearly* in the
transform length — measured against a long-double direct DFT at 3.9e-13 for
`L=2^10` rising to 1.4e-11 at `L=2^17`, three orders worse than it should be, in a
routine `exact.cpp` runs thousands of times in sequence. Re-seeding from
`cos`/`sin` every 64 butterflies bounds it.

**7. The closed form overflowed to zero at small volatility.** `mu` grows like
`1/sigma^2`, so the factor `(H/S)^{2(mu+1)}` in the C and D terms of
Reiner-Rubinstein is the first thing to blow up. At `sigma = 0.002` on an up
barrier that exponent reaches 2383, `exp()` returns infinity, and the function
handed back **0.0 for an up-and-out call worth 4.877** — not a NaN, not an
infinity, a clean and entirely plausible zero. `in + out = vanilla` was off by the
whole premium, and the parity sweep had not caught it because it holds sigma at
0.25 throughout. The fix is to never build either factor alone: whenever the power
overflows, the normal CDF multiplying it is underflowing by as much, so the
products are formed as `exp(exponent + log Phi)`, with a `log_ncdf` that switches
to the asymptotic expansion below the point where `Phi` underflows.

**8. The two-grid extrapolation, asked twice to do something it cannot.** Both
failures live in `discrete_exact`, both need the barrier close enough to spot that
the node count between them is small, and neither is reachable from any experiment
here — which is why they survived until the grid size was walked down deliberately.

The first is a division by zero. Doubling `n` usually doubles the integer node
count between barrier and spot, and the Richardson denominator `ratio^2 - 1` is
built on that. When the count is already 1, doubling `n` leaves it at 1, both
grids come out with *identical* spacing, `ratio` is 1, and the extrapolation
returns `inf` for a price that was otherwise fine.

The second is subtler and returns a plausible number instead of an obvious one.
Near the barrier the coarse grid falls back to the one-sided delta stencil while
the fine grid still manages a central difference. Those have different error
constants, so the `h^2` ladder does not apply across the change: extrapolating
anyway gave a delta of 1.1437 against a converged 1.1545, wrong by 0.9% and
looking entirely settled. `GridInfo` now reports which stencil ran, and when they
disagree the finer grid's delta is taken and nothing is extrapolated.

Two more came from the design, not the code: simulating a driftless random walk
to first passage never terminates in expectation (§3), and a mutation harness that
cannot build its own mutants reports a perfect score (§7).

And one from the tests, worth recording because it happened twice running. A test
asserted that an up-and-out call's delta is strictly negative with the barrier
0.001% above spot. It is not: there the option is worth zero to machine precision,
so its delta is zero to machine precision, and the measured +1.8e-13 is rounding
noise whose sign is arbitrary. The first version passed by luck and broke the
moment the formulas above were cleaned up. Asserting a sign on a quantity that is
zero within rounding tests the rounding.

## 9. What this does not cover

Stated explicitly, because a scope that is never named reads as a scope that was
never noticed. Each of these has a definite answer, and two of them are more
interesting than the things the repo does do.

- **Rebates.** `Params` has no rebate field. A rebate paid *at maturity* on
  knock-out is a function of the survival indicator and `S_T` alone, so the
  shifted barrier corrects it unchanged. A rebate paid *at the hit* is different:
  the discrete hit time is later than the continuous one by `O(dt)` in
  expectation, so the discount factor is wrong at `O(dt)`, subleading to the
  `O(sqrt(dt))` barrier effect, but the same order as the residual measured in §2.
- **Discrete dividends.** These kill the correction outright, and for the same
  reason jumps degrade it (§5): a scheduled drop is an `O(1)` jump with no small
  parameter to expand in. Applying the shift per-step here would be wrong.
- **Double barriers.** Each barrier shifts outward independently, because hitting
  both inside one monitoring interval is exponentially unlikely.
- **Non-uniform fixing schedules.** Both `exact.cpp` and `mc.cpp` hard-code
  `dt = T/m`. No real schedule is uniform, a daily fixing over a weekend is a
  three-day gap. The correction still applies, but per fixing:
  `H_i = H*exp(-beta*sigma*sqrt(dt_i))`, at which point there is no single shifted
  barrier and no single closed-form evaluation. That is the commercially
  interesting statement, and this repo does not implement it.
- **Term structure of volatility**, local or stochastic vol: same conclusion, the
  shift becomes per-fixing.
- **Greeks beyond delta.** Delta is computed (§2) and is the one that matters for
  the hedge. Gamma is available from the same three grid nodes and is not done;
  vega and theta would each need a re-solve, which is a different kind of cost.

## 10. Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cd build && ctest --output-on-failure    # 10 suites, ~25 s

./scripts/run_all.sh                     # tests + all 5 experiments -> results/
./tools/mutants.sh                       # 14 mutants, ~20 min
```

No dependencies. The FFT, the quantile function, the generator and the test
harness are all in-repo — partly so a clone reproduces the tables without an
argument about which vendor library was installed, and partly because a benchmark
whose accuracy depends on an unexamined third-party transform is not a benchmark.

`results/` is committed, unlike in my other repos, because the numbers are
properties of the mathematics: the benchmark is deterministic and the Monte Carlo
is addressed by path index, so a clone on any box should reproduce them. Two
things will differ on a diff and neither is a bug: the wall-clock lines, and the
last digit or two of a threaded mean, since the reduction sums in a different
order on a different core count.

```
include/bm/   normal.hpp  erfc-based CDF, Wichura AS241 quantile
              rng.hpp     Philox4x32-10, normals by inversion
              fft.hpp     radix-2 + real-input wrapper
              bs.hpp      vanilla + 8 Reiner-Rubinstein barrier forms
              exact.hpp   the convolution benchmark
              bgk.hpp     the continuity correction
              mc.hpp      discrete / bridge / coupled estimators
              jump.hpp    Merton jump-diffusion, jump-aware bridge
              stats.hpp   Welford, least squares
experiments/  convergence  correction  overshoot  estimator  jump
tests/        numerics  bs  exact  mc  bgk  jump   (+ ASan/UBSan variants)
tools/        mutants.sh
```
