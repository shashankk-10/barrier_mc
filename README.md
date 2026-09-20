# barrier_mc

Pricing a barrier option when the barrier is checked daily rather than
continuously, and measuring what the difference costs.

## The result

A down-and-out call, S = K = 100, H = 95, sigma = 0.30, T = 0.2. The continuous
Black-Scholes closed form gives **4.0042**. The same contract monitored daily is
worth **4.4778**, which is **11.8% more**.

That gap is bias, not sampling error. It shrinks like sqrt(dt) in the monitoring
interval and not at all in the path count, so at 2.1M paths it sits **80 Monte
Carlo standard errors** away from the closed form and buying more paths never
closes it.

Moving the barrier to `H * exp(-beta * sigma * sqrt(dt))`, with
`beta = 0.5826`, removes **1854x** of the gap and costs nothing at runtime.

| monitoring dates | discrete price | gap | gap / s.e. | residual after the shift |
|---|---|---|---|---|
| 25 | 4.6408 | +15.90% | 107 | -2.13e-03 |
| 50 (daily) | 4.4778 | +11.83% | 80 | -2.55e-04 |
| 400 | 4.1846 | +4.51% | 31 | -1.29e-05 |
| 1600 | 4.0963 | +2.30% | 16 | -1.62e-06 |

The residual falls like dt^1.5, better than the o(sqrt(dt)) the correction
guarantees. It also preserves the hedge: delta comes out within **0.02%** of the
benchmark, against **6.8%** uncorrected.

## Where it stops working

The shift compensates for a Gaussian overshoot, so it needs the barrier to sit
several diffusive standard deviations from spot. The asymptote is reached around
`log(S/H) / (sigma * sqrt(dt)) = 3.8`. Below that it degrades, and with the
barrier 50bp from spot it still leaves **4.6%** on the price and **15.5%** on
delta, at which point the contract belongs on a lattice instead.

Under a Merton jump diffusion the residual degrades from dt^1.5 to dt. The shift
corrects a diffusive overshoot and does nothing about a jump that crosses the
barrier and returns inside one monitoring interval.

## What is here

Comparing against a simulation would not work, since the thing being measured is
smaller than the sampling error. So the reference is deterministic: m Gaussian
convolutions on a log grid, over an FFT written for this repo. No dependencies.

- the eight single-barrier closed forms, agreeing with an independent
  implementation to 1e-13 and with in/out parity to 1e-11
- the convolution benchmark, Richardson extrapolated, checked at m = 1 and m = 2
  where the answer is known another way
- Monte Carlo with discrete and Brownian-bridge estimators on the same paths,
  which makes their difference a low-variance estimator of the gap
- Merton jump diffusion, with a jump-aware bridge

10 test suites, ASan and UBSan clean, zero warnings. Mutation testing injects 14
defects that each produce a plausible wrong price rather than a crash, and 13 are
caught.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cd build && ctest
```

`./scripts/run_all.sh` reproduces every number above into `results/`.

`DETAILS.md` has the derivations, the predictions written down before measuring,
and the wrong answers found along the way.
