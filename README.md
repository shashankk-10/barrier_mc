# barrier_mc

Pricing a barrier option when the barrier is checked daily rather than
continuously, and measuring what the difference costs.

## The result

A down-and-out call, S = K = 100, H = 95, sigma = 0.30, T = 0.2. The continuous
Black-Scholes price is **4.0042**. Monitored daily, the same contract is worth
**4.4778**, which is **11.8% more**.

That gap is bias, not sampling error. It shrinks like sqrt(dt) in the monitoring
interval and not at all in the path count, so at 2.1M paths it sits **80 Monte
Carlo standard errors** from the closed form, and more paths never close it.

Shifting the barrier to `H * exp(-beta * sigma * sqrt(dt))` with `beta = 0.5826`
removes **1854x** of the gap for free, and leaves delta within **0.02%** of the
benchmark against **6.8%** uncorrected.

| monitoring dates | price | gap | gap / s.e. | after the shift |
|---|---|---|---|---|
| 25 | 4.6408 | +15.90% | 107 | -2.13e-03 |
| 50 (daily) | 4.4778 | +11.83% | 80 | -2.55e-04 |
| 400 | 4.1846 | +4.51% | 31 | -1.29e-05 |
| 1600 | 4.0963 | +2.30% | 16 | -1.62e-06 |

## Where it stops working

The shift corrects a Gaussian overshoot, so it needs the barrier several
diffusive standard deviations from spot. The asymptote arrives around
`log(S/H) / (sigma * sqrt(dt)) = 3.8`. With the barrier 50bp from spot it still
leaves **4.6%** on the price and **15.5%** on delta, and the contract belongs on
a lattice instead. Under a Merton jump diffusion the residual degrades from
dt^1.5 to dt.

## The reference price

The effect being measured is smaller than Monte Carlo error, so the benchmark is
deterministic: m Gaussian convolutions on a log grid, over an FFT written for
this repo. No dependencies.

The eight barrier closed forms agree with an independent implementation to 1e-13.
10 test suites run ASan and UBSan clean with zero warnings, and mutation testing
catches 13 of 14 injected defects, each written to produce a plausible wrong
price rather than a crash.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
cd build && ctest
```

`./scripts/run_all.sh` regenerates every number above into `results/`.

`DETAILS.md` has the derivation of the constant, the predictions written down
before measuring, and the wrong answers found on the way.
