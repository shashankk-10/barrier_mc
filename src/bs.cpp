#include "bm/bs.hpp"

#include <cmath>

#include "bm/normal.hpp"

namespace bm {

double bs_vanilla(Side side, double S, double K, double r, double q,
                  double sigma, double T) {
  const double phi = (side == Side::Call) ? 1.0 : -1.0;
  if (T <= 0.0 || sigma <= 0.0) return std::fmax(phi * (S - K), 0.0);
  const double sT = sigma * std::sqrt(T);
  const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sT;
  const double d2 = d1 - sT;
  return phi * (S * std::exp(-q * T) * ncdf(phi * d1) -
                K * std::exp(-r * T) * ncdf(phi * d2));
}

double barrier_cont(Side side, Dir dir, Knock knock, const Params& p) {
  const double S = p.S, K = p.K, H = p.H, r = p.r, q = p.q;
  const double sigma = p.sigma, T = p.T;
  const double phi = (side == Side::Call) ? 1.0 : -1.0;
  const double eta = (dir == Dir::Down) ? 1.0 : -1.0;

  // Already knocked out (or in) at inception. Reiner-Rubinstein assumes the
  // barrier has not been touched; if spot is on the wrong side of it at t=0 the
  // formulas are simply about a different contract, so answer directly.
  const bool touched = (dir == Dir::Down) ? (S <= H) : (S >= H);
  if (touched) {
    return (knock == Knock::In) ? bs_vanilla(side, S, K, r, q, sigma, T) : 0.0;
  }
  if (T <= 0.0 || sigma <= 0.0) {
    const double intrinsic = std::fmax(phi * (S - K), 0.0);
    return (knock == Knock::Out) ? intrinsic : 0.0;
  }

  const double sT = sigma * std::sqrt(T);
  const double v2 = sigma * sigma;
  const double mu = (r - q - 0.5 * v2) / v2;
  const double dfq = std::exp(-q * T), dfr = std::exp(-r * T);

  // (H/S)^{2mu} and (H/S)^{2(mu+1)}. These are the first thing to overflow: mu
  // grows like 1/sigma^2, so at sigma = 0.002 with an up barrier the exponent
  // 2(mu+1)log(H/S) reaches 2383 and exp() returns infinity. What rescues the
  // answer is that the normal CDF each power multiplies is underflowing just as
  // fast, so the products below are formed as exp(exponent + log Phi) and neither
  // factor is ever built on its own. Before that, this function returned 0.0 for
  // a contract worth 4.877 and in + out = vanilla failed by the entire premium,
  // which is how it came to light.
  const double lhs = std::log(H / S);
  const double e_mu = 2.0 * mu * lhs;
  const double e_mu1 = 2.0 * (mu + 1.0) * lhs;

  const double x1 = std::log(S / K) / sT + (1.0 + mu) * sT;
  const double x2 = std::log(S / H) / sT + (1.0 + mu) * sT;
  const double y1 = std::log(H * H / (S * K)) / sT + (1.0 + mu) * sT;
  const double y2 = std::log(H / S) / sT + (1.0 + mu) * sT;

  const double A = phi * (S * dfq * ncdf(phi * x1) -
                          K * dfr * ncdf(phi * (x1 - sT)));
  const double B = phi * (S * dfq * ncdf(phi * x2) -
                          K * dfr * ncdf(phi * (x2 - sT)));
  const double C = phi * (S * dfq * std::exp(e_mu1 + log_ncdf(eta * y1)) -
                          K * dfr * std::exp(e_mu + log_ncdf(eta * (y1 - sT))));
  const double D = phi * (S * dfq * std::exp(e_mu1 + log_ncdf(eta * y2)) -
                          K * dfr * std::exp(e_mu + log_ncdf(eta * (y2 - sT))));

  const bool k_above_h = (K > H);

  // The eight cases. Each row is "which of A,B,C,D survive", and the split on
  // K vs H is not cosmetic: it decides whether the barrier sits inside or
  // outside the region where the payoff is non-zero, which changes which terms
  // are double-counted. Getting this table wrong produces prices that are
  // positive, monotone and plausible -- which is why test_bs.cpp checks every
  // row against in/out parity and against Monte Carlo rather than eyeballing.
  double v = 0.0;
  if (side == Side::Call && dir == Dir::Down) {         // down, call
    if (knock == Knock::In) v = k_above_h ? C : (A - B + D);
    else                    v = k_above_h ? (A - C) : (B - D);
  } else if (side == Side::Call && dir == Dir::Up) {    // up, call
    if (knock == Knock::In) v = k_above_h ? A : (B - C + D);
    else                    v = k_above_h ? 0.0 : (A - B + C - D);
  } else if (side == Side::Put && dir == Dir::Down) {   // down, put
    if (knock == Knock::In) v = k_above_h ? (B - C + D) : A;
    else                    v = k_above_h ? (A - B + C - D) : 0.0;
  } else {                                             // up, put
    if (knock == Knock::In) v = k_above_h ? (A - B + D) : C;
    else                    v = k_above_h ? (B - D) : (A - C);
  }
  // A knock-out can be worth zero but never less. Negative values here are
  // always a table error, not a market.
  return std::fmax(v, 0.0);
}

double barrier_cont_delta(Side side, Dir dir, Knock knock, const Params& p) {
  const double bump = 1e-4 * p.S;
  const bool down = (dir == Dir::Down);

  // Already on the dead side of the barrier: a knock-out is worth nothing and
  // goes on being worth nothing, so its delta is zero, and a knock-in has become
  // the vanilla.
  const double dist = down ? (p.S - p.H) : (p.H - p.S);
  if (dist <= 0.0) {
    if (knock == Knock::Out) return 0.0;
    Params hi = p, lo = p;
    hi.S = p.S + bump;
    lo.S = p.S - bump;
    return (bs_vanilla(side, hi.S, p.K, p.r, p.q, p.sigma, p.T) -
            bs_vanilla(side, lo.S, p.K, p.r, p.q, p.sigma, p.T)) /
           (2.0 * bump);
  }

  // A symmetric bump is wrong when it straddles the barrier. The far leg lands
  // in the knocked-out region and comes back as zero, so the quotient divides a
  // whole option price by a bump of 1e-4 -- which produces a large, smooth,
  // entirely wrong delta. Measured on a down-and-out call with spot 5e-5 above
  // the barrier: 1.07 against a true value near 1.43, and it degrades further
  // the closer spot gets. Step away from the barrier instead.
  if (dist > 2.0 * bump) {
    Params hi = p, lo = p;
    hi.S = p.S + bump;
    lo.S = p.S - bump;
    return (barrier_cont(side, dir, knock, hi) -
            barrier_cont(side, dir, knock, lo)) /
           (2.0 * bump);
  }
  const double step = down ? bump : -bump;
  Params a = p, b = p;
  a.S = p.S + step;
  b.S = p.S + 2.0 * step;
  const double f0 = barrier_cont(side, dir, knock, p);
  const double f1 = barrier_cont(side, dir, knock, a);
  const double f2 = barrier_cont(side, dir, knock, b);
  return (down ? 1.0 : -1.0) * (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * bump);
}

}  // namespace bm
