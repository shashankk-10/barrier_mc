#pragma once

// Black-Scholes vanillas and the eight continuously-monitored single-barrier
// options (Merton 1973 for the down-and-out call; Reiner & Rubinstein 1991 for
// the full set), with zero rebate.
//
// "Continuously monitored" is the whole point of this header: every price here
// assumes the barrier is watched at every instant of [0,T]. Whether that matches
// a traded contract depends on the market, and the distinction is a convention
// instead of a modelling choice.
//
// Interbank FX barriers, one-touches and no-touches ARE monitored continuously
// by convention -- barrier defence is a whole desk activity because of it.
// Equity, index and structured-product barriers almost always fix on an official
// observable instead: a closing price, an index official close, a published fix.
// The reason is verifiability, not mathematics. A discrete trigger is a fact both
// counterparties can look up and neither can manufacture; a continuous trigger on
// a listed name would make the knock-out a function of the worst tick of the day,
// which is contestable and, in a thin book, purchasable.
//
// So the gap between this file and a term sheet is real for equities and absent
// for FX, and it is worth being precise about which of the two any number is:
// anything out of bs.hpp is the continuous idealisation, anything out of
// exact.hpp is the discretely-monitored contract.

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
// consistent with each other by construction -- if the price table above were
// wrong, a hand-derived delta would disagree with it and the disagreement would
// be the thing under test instead of the quantity of interest.
double barrier_cont_delta(Side side, Dir dir, Knock knock, const Params& p);

// Convenience: the case the repo headlines.
inline double doc_cont(const Params& p) {
  return barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
}

}  // namespace bm
