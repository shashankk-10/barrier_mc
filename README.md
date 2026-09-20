# barrier_mc

Barrier options get checked once a day. The standard formula assumes they get
checked continuously. This measures what that costs.

## Result

Down-and-out call. S = K = 100, H = 95, sigma = 0.30, T = 0.2.

- Continuous formula: **4.0042**
- Checked daily: **4.4778**
- Gap: **11.8%**

The gap is bias, not noise. More paths do not shrink it. At 2.1M paths it is 80
standard errors wide.

Fix: move the barrier to `H * exp(-beta * sigma * sqrt(dt))`, with
`beta = 0.5826`.

- Error drops **1854x**
- Delta lands within **0.02%**, against 6.8% off without it
- Costs nothing at runtime

| dates | price | gap | gap / s.e. | after the fix |
|---|---|---|---|---|
| 25 | 4.6408 | +15.90% | 107 | -2.13e-03 |
| 50 | 4.4778 | +11.83% | 80 | -2.55e-04 |
| 400 | 4.1846 | +4.51% | 31 | -1.29e-05 |
| 1600 | 4.0963 | +2.30% | 16 | -1.62e-06 |

## When the fix stops working

- It needs the barrier well away from spot. Good above
  `log(S/H) / (sigma * sqrt(dt)) = 3.8`.
- Barrier 50bp from spot: still 4.6% off on price, 15.5% off on delta. Use a
  lattice there.
- With jumps: the leftover error goes from dt^1.5 to dt.

## Reference price

The gap is smaller than Monte Carlo error. So the benchmark is not a simulation.
It is m Gaussian convolutions on a log grid, over an FFT written here. No
dependencies.

- 8 closed forms match an independent implementation to 1e-13
- 10 test suites, ASan and UBSan clean, no warnings
- Mutation testing catches 13 of 14 injected bugs

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
cd build && ctest
```

`./scripts/run_all.sh` regenerates every number into `results/`.

`DETAILS.md` has where beta comes from, what was predicted before measuring, and
the bugs found on the way.
