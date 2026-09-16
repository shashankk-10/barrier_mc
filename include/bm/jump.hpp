#pragma once

// Merton jump-diffusion, and where the Gaussian machinery stops being exact.
//
// Everything else in this repo lives under Black-Scholes, where two facts hold
// that the rest of the project leans on very hard:
//
//   * between monitoring dates the log-price is a Brownian bridge, so the
//     probability it crossed the barrier has a closed form and mc.cpp's bridge
//     estimator is exactly unbiased for the continuously-monitored price; and
//   * the continuity correction's constant is beta = -zeta(1/2)/sqrt(2*pi),
//     which came from the ladder structure of a gaussian random walk.
//
// Add jumps and both statements need re-examining, in opposite directions:
//
//   * The bridge estimator stops being exact. Between two monitoring dates the
//     path is no longer a Brownian bridge -- it is a Brownian bridge interrupted
//     by jumps -- so applying the Gaussian formula to the endpoints ignores
//     every crossing that a jump caused and came back from. This file implements
//     both the naive estimator (Gaussian bridge on the endpoints, which is what
//     habit produces) and a jump-aware one that conditions on the jump times and
//     sizes and is exact again. The difference between them is the bias.
//
//   * The correction, though, should largely survive. For a finite-activity
//     jump process the probability of a jump inside an interval is lambda*dt =
//     O(dt), while the diffusive overshoot the correction compensates for is
//     O(sqrt(dt)). So to leading order in dt the gap is still beta*sigma*sqrt(dt)
//     with the same beta, and the correction should still work -- until lambda
//     is large enough, or sigma small enough, that the O(dt) jump term is
//     comparable to the O(sqrt(dt)) diffusive one at the dt actually used.
//     exp_jump.cpp sweeps lambda to find where that happens.


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
// behind -- measured as a per-path difference instead of as the difference of
// two separately-averaged numbers. That matters: the residual is two orders of
// magnitude below either price, so uncoupled error bars would swallow it whole.
// Pass 0 to use p.H, i.e. to measure the uncorrected gap.
JumpRun run_jump(Side side, Dir dir, const Params& p, const JumpParams& jp,
                 int m, const McConfig& cfg, double h_bridge = 0.0);

}  // namespace bm
