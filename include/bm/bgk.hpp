#pragma once

// The Broadie-Glasserman-Kou (1997) continuity correction.
//
// A discretely-monitored barrier option is worth approximately what a
// continuously monitored one would be worth if its barrier were moved away from
// spot by a factor exp(+/- beta * sigma * sqrt(dt)), with
//
//     beta = -zeta(1/2) / sqrt(2*pi) = 0.5825971579...
//
// The sign is + for an up-barrier and - for a down-barrier: in both cases the
// barrier moves away from spot, because discrete monitoring misses crossings and
// therefore behaves like a more distant barrier.
//
// Where the constant comes from. exp_overshoot.cpp checks each step separately:
//
//   1. Spitzer's identity, for any random walk S_k with S_0 = 0:
//          E[max_{k<=n} S_k]  =  sum_{k=1..n} E[S_k^+] / k.
//   2. For standard normal steps, S_k ~ N(0,k), so E[S_k^+] = sqrt(k)/sqrt(2*pi)
//      and the identity collapses to
//          E[M_n] = (1/sqrt(2*pi)) * sum_{k=1..n} k^{-1/2}.
//   3. Euler-Maclaurin on that partial sum:
//          sum_{k=1..n} k^{-1/2} = 2*sqrt(n) + zeta(1/2) + (1/2)*n^{-1/2} + ...
//      The constant term is the analytic continuation of the zeta function to
//      s = 1/2 -- literally the regularised value of a divergent sum.
//   4. Therefore E[M_n] = sqrt(2n/pi) - beta + o(1), and the discrete maximum
//      of the walk falls short of the continuous maximum of the Brownian motion
//      by beta standard deviations of one step. That shortfall is what the
//      barrier shift compensates for.
//
// So the Riemann zeta function turns up in an option price because the expected
// maximum of a Gaussian random walk is a partial sum of k^{-1/2}, and that sum's
// Euler-Maclaurin constant is zeta(1/2).
//
// validity. The result is asymptotic in dt with the barrier held a fixed
// distance from spot. It degrades when log(S/H) stops being large compared with
// sigma*sqrt(dt) -- there is no room for the limit to be a limit -- and
// exp_correction.cpp prints that ratio next to every row so the degradation is
// visible instead of asserted.

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
// the measured coefficient against it -- which tests beta quantitatively instead
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
