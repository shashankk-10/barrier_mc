# barrier_mc: details

The README has the result. This has the working.

## Where 0.5826 comes from

Most papers quote `beta = -zeta(1/2)/sqrt(2*pi)` and leave it there. It comes out
of three steps. Each is checked separately in `exp_overshoot`.

1. Spitzer's identity. For any random walk with `S_0 = 0`:
   `E[max_{k<=n} S_k] = sum_{k=1..n} E[S_k^+] / k`
2. Gaussian steps. `S_k ~ N(0,k)`, so `E[S_k^+] = sqrt(k)/sqrt(2*pi)`. The
   identity becomes `E[M_n] = (1/sqrt(2*pi)) * sum k^{-1/2}`. No probability
   left in it.
3. Euler-Maclaurin.
   `sum_{k=1..n} k^{-1/2} = 2*sqrt(n) + zeta(1/2) + (1/2)*n^{-1/2} + ...`

So `zeta(1/2)` is the constant term of a divergent sum. That is the whole reason
a zeta value shows up in an option price.

| n | `sum k^{-1/2} - 2*sqrt(n)` | recovered beta |
|---|---|---|
| 100 | -1.410396 | 0.5826138 |
| 1e4 | -1.455355 | 0.5825972 |
| 1e6 | -1.459855 | 0.5825972 |
| 1e8 | -1.460305 | **0.5825972** |
| limit | -1.460355 = `zeta(1/2)` | 0.5825972 |

Checks:

- Spitzer's identity against brute-force Monte Carlo at n = 1, 2, 5, 20, 100.
  Agrees within 1.6 standard errors.
- Same constant recovered a second way, from prices alone. Solve
  `barrier_cont(H*exp(-b*sigma*sqrt(dt))) = exact_discrete(m)` for `b` at each m,
  then one Richardson step. Gives 0.5825974, off by **2.8e-07**. No fit, no
  basis, nothing to tune.
- Naive summation loses 1.85e-9 at n = 1e8. Kahan is not optional here.

Dead end worth recording: the natural way to measure an overshoot is to simulate
a walk until it crosses a level. That never finishes. A driftless random walk has
infinite expected first-passage time.

## Why the estimator is built this way

The obvious approach does not scale.

- Price the discrete contract, subtract the closed form.
- Bias falls like `sqrt(dt)`, so paths needed grow like `1/bias^2`.
- At bias 1e-5 that is about 1e12 paths.

Instead, on each path compute both payoffs:

- **discrete**: did it stay above the barrier at every monitoring date? 0 or 1.
- **bridge**: given those points, what is the chance it stayed above at every
  instant? Closed form, from the reflection principle.

The bridge one is unbiased for the continuous price at any m. Their per-path
difference is unbiased for the gap, and is zero on any path that never came near
the barrier.

| m | var(difference) | var if run separately | ratio |
|---|---|---|---|
| 25 | 6.80 | 139.11 | 20x |
| 1600 | 1.18 | 137.70 | **117x** |

Paths needed to pin the gap to 1% at m = 50: **2.4e5** coupled, **6.2e6**
separate.

Two more choices:

- **Nested grids.** One path is drawn on the finest grid. Coarser m come from
  taking every 2nd, 4th, 8th point. Subsampling exact GBM is still exact GBM, so
  all levels share paths and a fitted slope is not mostly noise.
- **Counter-based RNG (Philox).** A path's d-th normal is a pure function of
  (path, d). Thread count and draw order cannot change it.

Evidence the bridge really is unbiased: across m = 25 to 1600 the discrete price
moves by 0.5474, the bridge price by 0.0021.

Antithetic sampling is worth 1.40x here, not more. The payoff is monotone in the
driving normals but the knock-out indicator is not.

## Predictions, written before measuring

| quantity | predicted | measured |
|---|---|---|
| exponent of the gap | 1/2 | 0.4665 one-term. Looked wrong, was not: the two-term fit recovers it |
| residual after the fix | `o(dt^0.5)` | **`dt^1.5`**, better than guaranteed |
| beta | 0.5825971579 | 0.5825972 via Spitzer; 0.5825974 from prices |
| validity condition | degrades near the barrier | asymptote only from ratio **3.8**, not 2 |
| under jumps | survives, jumps are `O(dt)` | survives, but residual drops to `O(dt)` |
| benchmark at the barrier | `O(h^2)` with a half-weight node | confirmed, 4.26e-06 to 2.66e-07 |

## What is checked against what

Nothing is checked against itself.

| claim | checked against |
|---|---|
| 8 barrier closed forms | independent implementation, 1e-13 |
| same, again | in + out = vanilla, 1e-11, 63 cases |
| benchmark at m=1 | barrier is inert, must equal vanilla |
| benchmark at m=2 | independent 1-D quadrature, 1.6e-09 |
| Monte Carlo | the benchmark, within 1.8 s.e. at every m |
| bridge estimator | the continuous closed form, at every m |
| path construction | far barrier reproduces vanilla |
| Spitzer's identity | brute-force Monte Carlo |
| Philox | published digits-of-pi test vectors |
| FFT | direct O(n^2) DFT |
| jump simulator | `E[S_T]e^{-(r-q)T} = S_0`, and lambda=0 matching the diffusion |

Mutation testing injects 14 defects. Each produces a plausible wrong price, not a
crash. 13 are caught.

The one survivor is `jump-unsorted`: shuffling jump times inside an interval.
It does not NaN, and the jump-aware bridge is the only estimator with no
independent reference. It is the reference. The error is 0.05 against a standard
error of 0.062, so no test here can see it.

The harness itself was wrong twice:

- It once reported 10 of 10 killed "by build". It was not copying
  `experiments/`, so every mutant died at configure time. Build failures are now
  a separate outcome.
- Patterns are literal strings, so refactors silently disarm them. Happened three
  times. `tools/mutants.sh --check` now verifies all patterns in a second, and
  "not applied" is counted apart from "survived".

## Eight wrong numbers that looked right

None of these crashed. All were positive, monotone and plausible.

| wrong | how it looked | what caught it |
|---|---|---|
| kernel orientation | 2% low, converged fine | m=1, where the barrier cannot bite |
| full weight at the barrier | `O(h)` not `O(h^2)` | error ratio test per grid doubling |
| Richardson denominator | bent the exponent at fine m | `residual * m^1.5` not flat |
| 53-bit uniform hitting 1.0 | never fires in 200k draws | asserting on the all-ones input |
| FFT twiddle drift | 1.4e-11 instead of 1e-15 | direct DFT at large n |
| sign of a beta correction | smooth convergence to 0.5825573 | the exact value, known to 10 digits |
| `(H/S)^{2mu}` overflow | clean 0.0 for a contract worth 4.877 | in + out = vanilla at small sigma |
| extrapolating two equal grids | `inf`, division by zero | shrinking the grid until it broke |

Four worth more detail.

**Kernel orientation.** Backward induction needs the density as a function of
the destination. Written the other way it computes the adjoint. Mass is
conserved, it converges under refinement, and it is 2% low. A constant offset no
refinement removes.

**Half weight at the barrier.** Knocking out leaves the value function
discontinuous. A uniform sum across a jump is only `O(h)`. Put a node on the
barrier, give it half weight, and it is `O(h^2)`. Measured at m=2: 1.23e-3 to
3.07e-4 without, 4.26e-6 to 2.66e-7 with.

**Richardson denominator.** The textbook form divides by 3, assuming doubling n
halves h. It does not. Node counts go 259 to 517, a ratio of 1.99614. Using 3
put 0.5% error into the correction term. That was the dominant error in the whole
benchmark, and it bent the measured exponent from 1.50 to 1.46.

**Small-sigma overflow.** `mu` grows like `1/sigma^2`. At sigma = 0.002 the
exponent `2(mu+1)log(H/S)` hits 2383 and `exp` returns infinity. The function
returned 0.0 for a contract worth 4.877. Parity was off by the whole premium. The
parity sweep had not caught it because it held sigma at 0.25. Fix: whenever the
power overflows the normal CDF beside it underflows, so form the product as
`exp(exponent + log Phi)`.

Two from the design, not the code:

- A driftless walk never finishes a first-passage simulation.
- A mutation harness that cannot build its mutants reports a perfect score.

One from the tests, which happened twice:

- A test asserted an up-and-out delta is strictly negative with the barrier
  0.001% above spot. It is not. There the option is worth zero to machine
  precision, so the delta is too, and the measured +1.8e-13 is rounding noise.
  Asserting a sign on something that is zero tests the rounding.

## What this does not cover

- **Rebates.** Paid at maturity: the shift applies unchanged. Paid at the hit:
  the discrete hit time is later by `O(dt)`, same order as the residual.
- **Discrete dividends.** These kill the correction. A scheduled drop is an
  `O(1)` jump with no small parameter to expand in.
- **Double barriers.** Each shifts outward on its own. Hitting both in one
  interval is exponentially unlikely.
- **Non-uniform fixing schedules.** Both solvers hard-code `dt = T/m`. Real
  schedules are not uniform. A weekend is a three-day gap. The shift becomes
  per-fixing, and then there is no single shifted barrier and no closed form
  left. That is the interesting case, and it is not here.
- **Vol term structure, local or stochastic vol.** Same conclusion.
- **Greeks past delta.** Gamma is three grid nodes away. Vega and theta need a
  re-solve.
