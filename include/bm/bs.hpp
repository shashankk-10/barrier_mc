#pragma once

// Black-Scholes vanillas and the eight single-barrier closed forms
// (Merton 1973, Reiner-Rubinstein 1991), zero rebate.
//
// Everything here assumes the barrier is watched at every instant. Whether that
// matches a real contract depends on the market: FX barriers are monitored
// continuously by convention, equity and index barriers fix on an official
// close. The reason is verifiability, not modelling. A discrete trigger is a
// fact both sides can look up.

#include <cstdint>

namespace bm {

enum class Side : uint8_t { Call, Put };
enum class Dir : uint8_t { Down, Up };    // barrier below / above spot
enum class Knock : uint8_t { Out, In };

struct Params {
  double S = 100.0;      // spot
  double K = 100.0;      // strike
  double H = 95.0;       // barrier
  double r = 0.05;       // risk-free rate
  double q = 0.0;        // continuous dividend yield
  double sigma = 0.30;   // volatility
  double T = 0.2;        // years to maturity
};

double bs_vanilla(Side side, double S, double K, double r, double q,
                  double sigma, double T);

// Continuously-monitored barrier price, zero rebate.
double barrier_cont(Side side, Dir dir, Knock knock, const Params& p);

// dV/dS of that price, by central difference in spot. A closed form for the
// barrier delta exists, but differencing the price keeps the delta and the price
// consistent with each other by construction, if the price table above were
// wrong, a hand-derived delta would disagree with it and the disagreement would
// be the thing under test instead of the quantity of interest.
double barrier_cont_delta(Side side, Dir dir, Knock knock, const Params& p);

// Convenience: the case the repo headlines.
inline double doc_cont(const Params& p) {
  return barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
}

}  // namespace bm
