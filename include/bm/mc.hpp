#pragma once

// The Monte Carlo engine: a discrete-monitoring estimator, a Brownian-bridge
// estimator, and the coupling between them that makes the bias measurable.
//
// Why a coupled estimator.
//
// The obvious way to measure the discretisation bias is to price the discretely
// monitored contract by Monte Carlo and subtract the continuous closed form.
// That works, but it does not scale: the bias falls like sqrt(dt), so pinning it
// to a fixed relative accuracy needs paths growing like 1/bias^2. Once the bias
// is 1e-5 -- where this project spends most of its time, after the continuity
// correction -- that is of order 1e12 paths. No amount of hardware rescues an
// estimator with that scaling; the estimator has to change.
//
// So: on each path, compute both payoffs.
//
//   discrete  -- did the path finish above the barrier at every monitoring
//                date? A 0/1 indicator.
//   bridge    -- what is the probability the path stayed above the barrier at
//                every instant, given where it was at the monitoring dates?
//                Between two dates the log-price is a Brownian bridge, and the
//                reflection principle gives that probability in closed form.
//
// The second is an unbiased estimator of the continuously monitored price, at
// any m, because it is the conditional expectation of the continuous indicator
// given the sampled points -- a Rao-Blackwellisation, so it also has strictly
// lower variance than the indicator it replaces. The first is an unbiased
// estimator of the discretely monitored price. Their difference, on the same
// path, estimates exactly the quantity this project is about, without bias.
//
// And it is nearly zero on most paths: the two estimators disagree only where a
// path dipped below the barrier between two monitoring dates and came back. So
// the difference has a variance far smaller than either term, and the bias
// becomes measurable at path counts a laptop can reach.
//
// Why the levels are nested.
//
// Fitting an exponent through prices at m = 25, 50, 100, ... needs those
// estimates to be positively correlated; otherwise the slope is mostly sampling
// noise, and it still comes with a confident-looking standard error. So a path is
// generated once on the finest grid and every coarser level is obtained by taking
// every 2nd, 4th, 8th point of it. Subsampling an exact GBM path gives an exact GBM path on the
// coarser dates, so no level is approximated, and every level shares the same
// terminal value -- which means they also share the same payoff, and the
// differences between levels are due to monitoring alone.
//
// A consequence worth testing instead of asserting: because the bridge
// estimator is unbiased for the continuous price at every m, its mean must be
// the same at every level. exp_convergence prints that row, and it is the
// sharpest evidence in the repo that the bridge really does remove the
// discretisation error rather than merely shrink it.

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
