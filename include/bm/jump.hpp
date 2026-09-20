#pragma once

// Merton jump-diffusion, and where the Gaussian machinery stops being exact.
//
// Two things change. The Brownian-bridge estimator is no longer exact, because
// between monitoring dates the path is a bridge interrupted by jumps; both the
// naive and a jump-aware version are here so the bias can be measured.
//
// The correction itself mostly survives: jumps inside an interval are O(dt)
// while the diffusive overshoot it corrects is O(sqrt(dt)). exp_jump sweeps
// lambda to find where that stops being true.

#include "bm/bs.hpp"
#include "bm/mc.hpp"
#include "bm/stats.hpp"

namespace bm {

struct JumpParams {
  double lambda = 1.0;    // jump intensity, per year
  double mu_j = -0.05;    // mean of the log jump size
  double sigma_j = 0.10;  // s.d. of the log jump size
};

struct JumpRun {
  int m = 0;
  Welford discrete;      // barrier tested at the monitoring dates only
  Welford bridge_naive;  // Gaussian bridge on the endpoints: ignores jumps
  Welford bridge_jump;   // conditions on the jumps: exact for continuous
  Welford coupled;       // discrete - bridge_jump, on the same path
  Welford bridge_gap;    // bridge_naive - bridge_jump, on the same path
  Welford spot;          // discounted S_T, for the martingale identity
  double mean_jumps = 0.0;  // realised jumps per path, a check on the sampler
};

// One monitoring frequency, three estimators, all on the same paths.
//
// `h_bridge` is the barrier used by the two continuous estimators; the discrete
// one always uses p.H. Passing the Broadie-Glasserman-Kou shifted barrier here
// therefore makes `coupled` an estimate of the residual the correction leaves
// behind, measured as a per-path difference instead of as the difference of
// two separately-averaged numbers. That matters: the residual is two orders of
// magnitude below either price, so uncoupled error bars would swallow it whole.
// Pass 0 to use p.H, i.e. to measure the uncorrected gap.
JumpRun run_jump(Side side, Dir dir, const Params& p, const JumpParams& jp,
                 int m, const McConfig& cfg, double h_bridge = 0.0);

}  // namespace bm
