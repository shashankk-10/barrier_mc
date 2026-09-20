#pragma once

// Broadie-Glasserman-Kou (1997) continuity correction.
//
// A discretely monitored barrier behaves like a continuous one with the barrier
// moved away from spot by exp(+/- beta*sigma*sqrt(dt)), beta = -zeta(1/2)/sqrt(2pi).
//
// Where beta comes from, and the two ways it is checked: DETAILS.md.
// It is asymptotic in dt with the barrier a fixed distance from spot, so it
// degrades close to the barrier. exp_correction prints that ratio per row.

#include <cmath>

#include "bm/bs.hpp"

namespace bm {

// -zeta(1/2)/sqrt(2*pi). Checked against scipy.special.zeta to 17 digits and
// recovered independently from Spitzer's identity in exp_overshoot.cpp.
inline constexpr double kBeta = 0.58259715793901068;

inline double bgk_barrier(double H, Dir dir, double sigma, double dt) {
  const double s = (dir == Dir::Up) ? +1.0 : -1.0;
  return H * std::exp(s * kBeta * sigma * std::sqrt(dt));
}

// The corrected price: the continuous formula, evaluated at the shifted barrier.
inline double barrier_bgk(Side side, Dir dir, Knock knock, const Params& p,
                          int m) {
  Params q = p;
  q.H = bgk_barrier(p.H, dir, p.sigma, p.T / double(m));
  return barrier_cont(side, dir, knock, q);
}

// The delta a desk would actually quote if it priced this discretely-monitored
// contract with the analytic formula and the shifted barrier. This is the number
// that decides whether barriers can live in the fast pricer: a price that is
// close is worth little if the hedge ratio is not.
inline double barrier_bgk_delta(Side side, Dir dir, Knock knock,
                                const Params& p, int m) {
  Params q = p;
  q.H = bgk_barrier(p.H, dir, p.sigma, p.T / double(m));
  return barrier_cont_delta(side, dir, knock, q);
}

// dV/dH of the continuous price, by central difference in log-space.
//
// This exists so the leading term of the discretisation gap can be predicted,
// not just fitted. Expanding the shifted price,
//     V_disc ~ V_cont(H * exp(-beta*sigma*sqrt(dt)))
//            ~ V_cont(H) - beta*sigma*sqrt(dt) * H * dV/dH,
// so the coefficient of sqrt(dt) in the gap must be -beta*sigma*H*dV/dH. That is
// a number, computable from the closed form alone, and exp_correction.cpp checks
// the measured coefficient against it, which tests beta quantitatively instead
// of only testing the exponent 1/2.
inline double dV_dH(Side side, Dir dir, Knock knock, const Params& p) {
  const double eps = 1e-5;
  Params hi = p, lo = p;
  hi.H = p.H * (1.0 + eps);
  lo.H = p.H * (1.0 - eps);
  return (barrier_cont(side, dir, knock, hi) -
          barrier_cont(side, dir, knock, lo)) /
         (2.0 * eps * p.H);
}

inline double leading_gap_coeff(Side side, Dir dir, Knock knock,
                                const Params& p) {
  return -kBeta * p.sigma * p.H * dV_dH(side, dir, knock, p);
}

}  // namespace bm
