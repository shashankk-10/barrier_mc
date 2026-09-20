#pragma once

// Discrete and Brownian-bridge estimators, and the coupling between them.
//
// Measuring the gap the obvious way does not scale: it falls like sqrt(dt), so
// paths grow like 1/gap^2, which is ~1e12 paths once the gap is 1e-5.
//
// So both payoffs are computed on the same path. The discrete one is a 0/1
// indicator. The bridge one is the probability the path stayed above the barrier
// at every instant given the sampled points, which is a conditional expectation
// of that indicator and so unbiased for the continuous price at any m. Their
// per-path difference is zero on any path that never came near the barrier.
//
// Levels are nested: one path on the finest grid, coarser m by subsampling.
// Subsampling exact GBM is still exact GBM. See DETAILS.md.

#include <cstdint>
#include <vector>

#include "bm/bs.hpp"
#include "bm/stats.hpp"

namespace bm {

struct McConfig {
  uint64_t paths = 1u << 20;
  uint64_t seed = 0x5EEDULL;
  int m0 = 25;        // coarsest monitoring count
  int levels = 7;     // m_l = m0 << l, so the finest is m0 << (levels-1)
  int threads = 0;    // 0 = hardware_concurrency
  bool antithetic = false;
};

struct LevelStats {
  int m = 0;
  Welford discrete;  // discretely-monitored price at this m
  Welford bridge;    // continuously-monitored price, from this m's grid
  Welford bias;      // discrete - bridge, on the same path
};

struct ConvergenceRun {
  std::vector<LevelStats> levels;
  uint64_t paths = 0;
  int threads = 0;
};

// One pass over `paths` paths producing, at every nested level, the discrete
// price, the bridge price and their coupled difference.
ConvergenceRun run_convergence(Side side, Dir dir, const Params& p,
                               const McConfig& cfg);

// The naive estimator, for the comparison the README makes: price the discrete
// contract on its own grid with no coupling and no bridge, exactly as a
// straightforward implementation would.
struct PlainRun {
  Welford price;
  int m = 0;
  uint64_t paths = 0;
};
PlainRun run_plain(Side side, Dir dir, const Params& p, int m,
                   const McConfig& cfg);

}  // namespace bm
