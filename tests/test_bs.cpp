// The continuously-monitored closed forms.
//
// The Reiner-Rubinstein table is eight rows, each splitting again on whether the
// strike is inside or outside the barrier, and a wrong row does not look wrong:
// it returns a positive number, monotone in every input, that sits between the
// vanilla and zero. So this file never eyeballs a price. Every value is checked
// three ways -- against an independent implementation, against the in/out parity
// identity, and against the limits where the barrier stops mattering.

#include <cmath>
#include <initializer_list>

#include "bm/bs.hpp"
#include "bm/normal.hpp"
#include "check.hpp"

using namespace bm;

static void test_vanilla() {
  CHECK_REL(bs_vanilla(Side::Call, 100, 100, 0.05, 0.0, 0.30, 0.2),
            5.834014051476267, 1e-14);
  CHECK_REL(bs_vanilla(Side::Put, 100, 100, 0.05, 0.0, 0.30, 0.2),
            4.838997426393071, 1e-14);
  // Put-call parity: C - P = S e^{-qT} - K e^{-rT}.
  for (double K : {80.0, 100.0, 130.0}) {
    for (double T : {0.05, 1.0, 3.0}) {
      const double c = bs_vanilla(Side::Call, 100, K, 0.05, 0.02, 0.3, T);
      const double p = bs_vanilla(Side::Put, 100, K, 0.05, 0.02, 0.3, T);
      CHECK_NEAR(c - p, 100 * std::exp(-0.02 * T) - K * std::exp(-0.05 * T),
                 1e-12);
    }
  }
  CHECK_NEAR(bs_vanilla(Side::Call, 100, 90, 0.05, 0.0, 0.3, 0.0), 10.0, 0.0);
  CHECK_NEAR(bs_vanilla(Side::Put, 100, 90, 0.05, 0.0, 0.3, 0.0), 0.0, 0.0);
}

// Reference values from an independently written implementation (scipy-based),
// printed at 17 significant digits. Six parameter sets so that every branch of
// the K-vs-H split is exercised in both directions, including a barrier half a
// percent from spot and a deep-out-of-the-money case with a dividend yield
// larger than the rate.
static void test_table() {
  // down, K>H -- the headline parameter set
  { Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::Out, p), 4.004229446968722, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::In, p), 1.8297846045075445, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Up, Knock::Out, p), 0.0, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Up, Knock::In, p), 5.834014051476267, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::Out, p), 0.03387015967674145, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::In, p), 4.80512726671633, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Up, Knock::Out, p), 0.0, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Up, Knock::In, p), 4.838997426393071, 1e-13);
  }
  // down, K<H
  { Params p{100.0, 90.0, 95.0, 0.05, 0.0, 0.3, 0.2};
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::Out, p), 6.938305678666929, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::In, p), 5.3236031725519055, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::Out, p), 0.0, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::In, p), 1.3663938886439801, 1e-13);
  }
  // up, K<H
  { Params p{100.0, 105.0, 110.0, 0.05, 0.0, 0.3, 0.2};
    CHECK_REL(barrier_cont(Side::Call, Dir::Up, Knock::Out, p), 0.04271052717287738, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Up, Knock::In, p), 3.659818421858441, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Up, Knock::Out, p), 6.679409321192606, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Up, Knock::In, p), 0.9783521715013652, 1e-13);
  }
  // up, K>H -- the up-and-out call is identically zero here, and that is a
  // statement about the contract, not a degenerate formula: the payoff needs
  // S_T > 120 and the barrier at 110 has already killed every such path.
  { Params p{100.0, 120.0, 110.0, 0.05, 0.0, 0.3, 0.2};
    CHECK_REL(barrier_cont(Side::Call, Dir::Up, Knock::Out, p), 0.0, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Up, Knock::In, p), 0.6870957745708637, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Up, Knock::Out, p), 14.359305679228793, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Up, Knock::In, p), 5.133770145242227, 1e-13);
  }
  // down, barrier half a percent from spot. This is where the powers
  // (H/S)^{2mu} and the near-cancelling A-C difference are worst conditioned.
  { Params p{100.0, 100.0, 99.5, 0.04, 0.02, 0.15, 1.0};
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::Out, p), 0.5811515792572379, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::In, p), 6.242836364506466, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::Out, p), 2.3787782765793963e-06, 1e-10);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::In, p), 4.883062149542219, 1e-13);
  }
  // deep OTM, high vol, dividend yield above the rate (so mu < 0)
  { Params p{80.0, 100.0, 60.0, 0.01, 0.03, 0.55, 2.5};
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::Out, p), 12.221132190903226, 1e-13);
    CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::In, p), 6.3190535393136, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::Out, p), 0.5432409195049459, 1e-13);
    CHECK_REL(barrier_cont(Side::Put, Dir::Down, Knock::In, p), 41.308457107260914, 1e-13);
  }
}

// in + out = vanilla, for every type, over a sweep. The knocked-in and
// knocked-out events partition the sample space, so this is an identity, not an
// approximation -- any discrepancy is a bug in the table.
static void test_parity() {
  for (double H : {60.0, 85.0, 95.0, 99.0, 101.0, 110.0, 140.0}) {
    for (double K : {70.0, 100.0, 125.0}) {
      for (double T : {0.1, 0.75, 2.0}) {
        Params p{100.0, K, H, 0.03, 0.01, 0.25, T};
        for (Side s : {Side::Call, Side::Put}) {
          for (Dir d : {Dir::Down, Dir::Up}) {
            const double out = barrier_cont(s, d, Knock::Out, p);
            const double in = barrier_cont(s, d, Knock::In, p);
            const double van = bs_vanilla(s, p.S, p.K, p.r, p.q, p.sigma, p.T);
            CHECK_NEAR(out + in, van, 1e-11);
            CHECK(out >= -1e-14 && in >= -1e-14);
            CHECK(out <= van + 1e-11 && in <= van + 1e-11);
          }
        }
      }
    }
  }
}

static void test_limits() {
  Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};
  const double van = bs_vanilla(Side::Call, p.S, p.K, p.r, p.q, p.sigma, p.T);

  // A barrier far enough away stops mattering.
  p.H = 1e-8;
  CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::Out, p), van, 1e-13);
  CHECK_NEAR(barrier_cont(Side::Call, Dir::Down, Knock::In, p), 0.0, 1e-12);
  p.H = 1e8;
  CHECK_REL(barrier_cont(Side::Call, Dir::Up, Knock::Out, p), van, 1e-13);

  // A barrier at spot kills the knock-out outright.
  p.H = 100.0;
  CHECK_NEAR(barrier_cont(Side::Call, Dir::Down, Knock::Out, p), 0.0, 0.0);
  CHECK_REL(barrier_cont(Side::Call, Dir::Down, Knock::In, p), van, 1e-14);

  // Monotone: raising a down-barrier can only destroy value.
  double prev = 1e18;
  for (double H = 10.0; H < 99.9; H += 2.5) {
    Params q2{100.0, 100.0, H, 0.05, 0.0, 0.3, 0.2};
    const double v = barrier_cont(Side::Call, Dir::Down, Knock::Out, q2);
    CHECK(v <= prev + 1e-12);
    prev = v;
  }
}

// Delta near the barrier, where a symmetric bump stops being usable.
//
// barrier_cont_delta differences the price in spot, and a +/-1bp bump straddles
// the barrier once spot is within a bp of it. The far leg then lands in the
// knocked-out region, comes back as zero, and the quotient divides a whole
// option price by the bump: a large, smooth, completely wrong delta. Before this
// was guarded, a down-and-out call 5e-5 above its barrier reported 1.07 against a
// true value near 1.43, falling to 0.72 as spot reached the barrier -- monotone
// and plausible the whole way down.
static void test_delta_near_barrier() {
  double prev = 0.0;
  for (double S : {101.0, 100.1, 100.01, 100.005, 100.001}) {
    Params p{S, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
    const double d = barrier_cont_delta(Side::Call, Dir::Down, Knock::Out, p);
    // It should approach a limit near 1.43, not collapse toward zero.
    CHECK(d > 1.35 && d < 1.50);
    if (prev > 0.0) CHECK(std::fabs(d - prev) < 0.05);
    prev = d;
  }
  // Sitting exactly on the barrier, a knock-out is dead: worth nothing now and
  // nothing later, so its delta is zero rather than the one-sided limit above.
  Params at{100.0, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
  CHECK_NEAR(barrier_cont(Side::Call, Dir::Down, Knock::Out, at), 0.0, 0.0);
  CHECK_NEAR(barrier_cont_delta(Side::Call, Dir::Down, Knock::Out, at), 0.0, 0.0);
  // The knock-in has become the vanilla, and hedges like one.
  const double sT = at.sigma * std::sqrt(at.T);
  const double d1 =
      (std::log(at.S / at.K) + (at.r - at.q + 0.5 * at.sigma * at.sigma) * at.T) / sT;
  CHECK_NEAR(barrier_cont_delta(Side::Call, Dir::Down, Knock::In, at), ncdf(d1),
             1e-6);
  // Upward, where the one-sided stencil points the other way.
  //
  // With the barrier a thousandth of a percent above spot the contract knocks out
  // almost surely: it is worth zero to machine precision, and differencing three
  // zeros gives a delta that is also zero to machine precision -- measured at
  // +1.8e-13, with a sign that is whichever way the rounding fell. So the thing
  // to assert is the magnitude. Two earlier versions of this test asserted the
  // SIGN here, which is a statement about rounding noise and nothing else; the
  // first passed only by luck and broke the moment the formulas were cleaned up.
  Params up{100.0, 100.0, 100.001, 0.05, 0.0, 0.20, 1.0};
  const double du = barrier_cont_delta(Side::Call, Dir::Up, Knock::Out, up);
  CHECK(std::isfinite(du));
  CHECK(std::fabs(du) < 1e-10);
  CHECK_NEAR(barrier_cont(Side::Call, Dir::Up, Knock::Out, up), 0.0, 1e-12);

  // Far enough out that the option is worth something, the sign is meaningful
  // and strict: an up-and-out call loses value as spot rises toward the barrier.
  Params up2{100.0, 100.0, 105.0, 0.05, 0.0, 0.20, 1.0};
  CHECK(barrier_cont(Side::Call, Dir::Up, Knock::Out, up2) > 1e-4);
  CHECK(barrier_cont_delta(Side::Call, Dir::Up, Knock::Out, up2) < 0.0);
}

// Parity across a volatility sweep, down to volatilities where the closed form
// nearly falls over.
//
// The existing parity sweep holds sigma at 0.25, which hides this: mu grows like
// 1/sigma^2, so the factor (H/S)^{2(mu+1)} in the C and D terms overflows for
// small sigma on an up barrier. At sigma = 0.002 the exponent is 2383, exp()
// returns infinity, and the function used to hand back 0.0 for an up-and-out call
// worth 4.877 -- a clean, plausible zero, with in + out = vanilla off by the whole
// premium. The terms are now formed as exp(exponent + log Phi), because whenever
// the power overflows the normal CDF beside it is underflowing by as much.
static void test_parity_small_sigma() {
  for (double sg : {0.30, 0.10, 0.02, 0.005, 0.002, 0.001}) {
    for (double H : {90.0, 110.0}) {
      Params p{100.0, 100.0, H, 0.05, 0.0, sg, 1.0};
      const Dir d = (H < p.S) ? Dir::Down : Dir::Up;
      for (Side side : {Side::Call, Side::Put}) {
        const double out = barrier_cont(side, d, Knock::Out, p);
        const double in = barrier_cont(side, d, Knock::In, p);
        const double van = bs_vanilla(side, p.S, p.K, p.r, p.q, sg, p.T);
        CHECK(std::isfinite(out) && std::isfinite(in));
        CHECK_NEAR(out + in, van, 1e-10);
      }
      // The knock-out cannot be worth zero here: at these volatilities spot
      // barely moves, so a barrier 10% away is essentially never touched and the
      // knock-out is worth almost exactly the vanilla.
      if (sg <= 0.01) {
        const double out = barrier_cont(Side::Call, d, Knock::Out, p);
        const double van = bs_vanilla(Side::Call, p.S, p.K, p.r, p.q, sg, p.T);
        if (d == Dir::Up) CHECK(out > 0.9 * van);
      }
    }
  }
}

int main() {
  test_vanilla();
  test_delta_near_barrier();
  test_parity_small_sigma();
  test_table();
  test_parity();
  test_limits();
  return check::finish("bs");
}
