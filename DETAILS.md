# barrier_mc: the parts worth probing

The README has the result. This has the reasoning behind it: where the
correction's constant comes from, what was predicted before anything was
measured, what each number is checked against, and the wrong answers found on
the way.

## Where 0.5826 comes from

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

**Step 3, Euler–Maclaurin.**

```
sum_{k=1..n} k^{-1/2} = 2*sqrt(n) + zeta(1/2) + (1/2)*n^{-1/2} + O(n^{-3/2})
```

The constant term is the analytic continuation of the zeta function to `s = 1/2`
,  **the regularised value of a divergent sum**. That is the whole answer to why a
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

Beta is also recoverable from simulation alone, using no identity at all , 
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

## Why the estimator is built this way

Measuring the gap by pricing the discrete contract and subtracting the closed
form is hopeless: the gap falls like `sqrt(dt)`, so resolving it to fixed
relative accuracy needs paths growing like `1/gap²`.

The estimator used instead computes, **on each path**, both the discrete payoff
(a 0/1 indicator) and the Brownian-bridge payoff (the probability the path stayed
above the barrier at every instant, given the sampled points, a closed form from
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
reproduce the vanilla Black–Scholes price, a control that would catch a bias in
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

## Predictions, written before measuring

| Quantity | Predicted | Measured |
|---|---|---|
| exponent of the gap in `dt` | 1/2 | 0.4665 by a one-term fit, **the prediction looked wrong and was not**; the two-term expansion recovers it |
| residual after the correction | `o(dt^0.5)`, expected `O(dt)` | **`O(dt^1.5)`**, better than the theorem promises |
| leading coefficient `c1` | `-beta·sigma·H·dV/dH` = 8.39948 | 8.38088, **but the agreement is basis-dependent**; replaced by an inversion giving beta to 2.8e-07 |
| `beta` | `-zeta(1/2)/sqrt(2*pi)` = 0.5825971579 | 0.5825971579 via Spitzer (10 digits); 0.5825974361 from implied barrier shift (2.8e-07) |
| validity condition | degrades once `log(S/H) ≲ sigma·sqrt(dt)` | asymptote reached only at ratio ≈ **3.8**; 13% off at ratio 2.7, 2.6× off at 1.9 |
| correction under jumps | survives to leading order, since jumps are `O(dt)` and the overshoot is `O(sqrt(dt))` | confirmed, but the **residual degrades from `dt^1.5` to `dt`** |
| benchmark quadrature at the barrier | `O(h^2)` with a half-weight node | confirmed: 4.26e-06 → 1.06e-06 → 2.66e-07 |

## What is checked against what

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
project is actually about; each produces plausible output, not a crash , 
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
- `richardson-fixed-3`, restoring the wrong extrapolation denominator described below,
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
a negative time span, so the result is *wrong but finite*, precisely the failure
mode this repo is about. Second, and more fundamentally, **the jump-aware bridge
has no independent reference to be checked against.** Every other estimator here
is anchored to something computed a different way; this one is the reference.
Measured directly: at `lambda=50, m=25` the mis-sorted version returns 7.9428
against the correct 7.8930, a difference of 0.05 against a Monte Carlo standard
error of 0.062, under the noise at any path count a laptop will run.

The obvious structural test, "the jump-aware bridge estimates the continuous
price, so its mean must not drift with `m`", the analogue of the check that works
for the Gaussian bridge above, was tried and does not hold: at `lambda=50` the
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

## Eight wrong numbers that looked right, and what caught each

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
function discontinuous, and a uniform-weight sum across a jump is `O(h)`, the
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
transform length, measured against a long-double direct DFT at 3.9e-13 for
`L=2^10` rising to 1.4e-11 at `L=2^17`, three orders worse than it should be, in a
routine `exact.cpp` runs thousands of times in sequence. Re-seeding from
`cos`/`sin` every 64 butterflies bounds it.

**7. The closed form overflowed to zero at small volatility.** `mu` grows like
`1/sigma^2`, so the factor `(H/S)^{2(mu+1)}` in the C and D terms of
Reiner-Rubinstein is the first thing to blow up. At `sigma = 0.002` on an up
barrier that exponent reaches 2383, `exp()` returns infinity, and the function
handed back **0.0 for an up-and-out call worth 4.877**, not a NaN, not an
infinity, a clean and entirely plausible zero. `in + out = vanilla` was off by the
whole premium, and the parity sweep had not caught it because it holds sigma at
0.25 throughout. The fix is to never build either factor alone: whenever the power
overflows, the normal CDF multiplying it is underflowing by as much, so the
products are formed as `exp(exponent + log Phi)`, with a `log_ncdf` that switches
to the asymptotic expansion below the point where `Phi` underflows.

**8. The two-grid extrapolation, asked twice to do something it cannot.** Both
failures live in `discrete_exact`, both need the barrier close enough to spot that
the node count between them is small, and neither is reachable from any experiment
here, which is why they survived until the grid size was walked down deliberately.

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
to first passage never terminates in expectation, and a mutation harness that
cannot build its own mutants reports a perfect score.

And one from the tests, worth recording because it happened twice running. A test
asserted that an up-and-out call's delta is strictly negative with the barrier
0.001% above spot. It is not: there the option is worth zero to machine precision,
so its delta is zero to machine precision, and the measured +1.8e-13 is rounding
noise whose sign is arbitrary. The first version passed by luck and broke the
moment the formulas above were cleaned up. Asserting a sign on a quantity that is
zero within rounding tests the rounding.

## What this does not cover

Stated explicitly, because a scope that is never named reads as a scope that was
never noticed. Each of these has a definite answer, and two of them are more
interesting than the things the repo does do.

- **Rebates.** `Params` has no rebate field. A rebate paid *at maturity* on
  knock-out is a function of the survival indicator and `S_T` alone, so the
  shifted barrier corrects it unchanged. A rebate paid *at the hit* is different:
  the discrete hit time is later than the continuous one by `O(dt)` in
  expectation, so the discount factor is wrong at `O(dt)`, subleading to the
  `O(sqrt(dt))` barrier effect, but the same order as the residual in the README.
- **Discrete dividends.** These kill the correction outright, and for the same
  reason jumps degrade it: a scheduled drop is an `O(1)` jump with no small
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
- **Greeks beyond delta.** Delta is computed and is the one that matters for
  the hedge. Gamma is available from the same three grid nodes and is not done;
  vega and theta would each need a re-solve, which is a different kind of cost.

