// exp_jump: where this stops working.
//
// Every result in this repo so far is a Black-Scholes result. Two of them depend
// on the diffusion being the only thing that moves the price:
//
//   * the Brownian-bridge estimator is exactly unbiased for the continuously
//     monitored price, because between monitoring dates the path is a Brownian
//     bridge and nothing else; and
//   * the correction's residual falls like dt^1.5, well inside the o(sqrt(dt))
//     the theorem promises, because the only mechanism the discrete monitor
//     misses is a diffusive excursion.
//
// Add a compound-Poisson jump component and ask what survives. The answers are
// not the same for the two claims, which is the point of running it.

#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "bm/bgk.hpp"
#include "bm/bs.hpp"
#include "bm/exact.hpp"
#include "bm/jump.hpp"
#include "bm/mc.hpp"
#include "harness.hpp"

using namespace bm;

int main() {
  const Params p = bmx::headline();
  bmx::rule("exp_jump: what survives when the price can jump");
  bmx::describe(p, "down-and-out call, Merton jump-diffusion");

  McConfig cfg;
  cfg.paths = 1u << 21;

  // --- [0] does the simulator price the right asset? -----------------------
  printf(
      "\n[0] controls on the simulator\n"
      "    E[S_T] e^{-(r-q)T} / S_0 must be 1 exactly (the jump compensator),\n"
      "    and the realised jump count must match lambda*T.\n\n"
      "  lambda | mean jumps | expected |  martingale ratio (s.e.)\n"
      "---------+------------+----------+--------------------------\n");
  for (double lam : {0.0, 1.0, 5.0}) {
    JumpParams jp;
    jp.lambda = lam;
    const auto r = run_jump(Side::Call, Dir::Down, p, jp, 100, cfg);
    printf("%8.1f | %10.5f | %8.5f | %+.8f (%.2e)\n", lam, r.mean_jumps,
           lam * p.T, r.spot.mean(), r.spot.stderr_());
  }

  // --- [1] lambda = 0 must reproduce the Black-Scholes results -------------
  // If the jump code does not collapse onto the diffusion code when the jump
  // intensity is zero, nothing after this line means anything.
  printf("\n[1] lambda = 0 must reproduce the pure-diffusion results\n\n");
  {
    JumpParams jp;
    jp.lambda = 0.0;
    const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
    printf("      m |  discrete (s.e.)    |  exact benchmark |"
           "  bridge (s.e.)      | closed form\n"
           "--------+---------------------+------------------+"
           "---------------------+-------------\n");
    for (int m : {50, 200}) {
      const auto r = run_jump(Side::Call, Dir::Down, p, jp, m, cfg);
      const double ex =
          discrete_exact(Side::Call, Dir::Down, Knock::Out, p, m).price;
      printf("%7d | %9.6f (%.6f) | %16.6f | %9.6f (%.6f) | %11.6f\n", m,
             r.discrete.mean(), r.discrete.stderr_(), ex, r.bridge_jump.mean(),
             r.bridge_jump.stderr_(), cont);
      printf("        |  %+.2f s.e. vs benchmark                   |"
             "  %+.2f s.e. vs closed form\n",
             (r.discrete.mean() - ex) / r.discrete.stderr_(),
             (r.bridge_jump.mean() - cont) / r.bridge_jump.stderr_());
    }
  }

  // --- [2] the bridge estimator stops being exact ---------------------------
  printf(
      "\n[2] the Gaussian bridge, carried over unchanged, is now biased\n"
      "    Applying the endpoint formula ignores every crossing caused by a jump\n"
      "    inside an interval. The jump-aware estimator conditions on the jump\n"
      "    times and sizes and is exact again; the difference is the bias.\n\n"
      "      m |  naive bridge  |  jump-aware   |   bias (s.e.)      | bias * m\n"
      "--------+----------------+---------------+--------------------+----------\n");
  {
    JumpParams jp;
    jp.lambda = 3.0;
    for (int m : {25, 50, 100, 200, 400}) {
      const auto r = run_jump(Side::Call, Dir::Down, p, jp, m, cfg);
      // Coupled: both estimators run on the same paths, so the difference has
      // its own, far tighter, standard error.
      const double bias = r.bridge_gap.mean();
      printf("%7d | %14.6f | %13.6f | %+9.6f (%.6f) | %+8.4f\n", m,
             r.bridge_naive.mean(), r.bridge_jump.mean(), bias,
             r.bridge_gap.stderr_(), bias * double(m));
    }
    printf(
        "\n    bias * m roughly constant means the naive estimator's error is\n"
        "    O(dt): the chance of a jump inside an interval is lambda*dt, and a\n"
        "    jump is the only thing the endpoint formula cannot see. Under pure\n"
        "    diffusion the same column would be identically zero.\n");
  }

  // --- [3] the correction: what it still buys, and what it stops buying -----
  printf(
      "\n[3] the continuity correction under jumps\n"
      "    'gap' is discrete minus continuous, both from the jump-aware\n"
      "    estimator on the same paths. 'residual' is the same difference with\n"
      "    the continuous leg evaluated at the shifted barrier.\n"
      "\n"
      "    Read the |res|/s.e. column first. At lambda = 0 the true residual is\n"
      "    about 4e-5 (exp_correction measures it against the convolution\n"
      "    benchmark, which has no sampling error); that is an order of magnitude\n"
      "    below what two million paths can resolve, so those rows are noise and\n"
      "    their 'improvement' figures mean nothing. That is not a failure of the\n"
      "    experiment -- it is the reason the rest of this repo does not use Monte\n"
      "    Carlo to measure the residual. With jumps the residual rises above the\n"
      "    noise floor, and then the scaling can be read off.\n\n"
      " lambda |     m |    gap (s.e.)      |  residual (s.e.)   | |res|/s.e. |"
      "  res * m  | improvement\n"
      "--------+-------+--------------------+--------------------+------------+"
      "-----------+------------\n");
  for (double lam : {0.0, 1.0, 5.0}) {
    JumpParams jp;
    jp.lambda = lam;
    for (int m : {50, 200, 800}) {
      const double hs = bgk_barrier(p.H, Dir::Down, p.sigma, p.T / double(m));
      const auto g = run_jump(Side::Call, Dir::Down, p, jp, m, cfg);
      const auto s = run_jump(Side::Call, Dir::Down, p, jp, m, cfg, hs);
      const double gap = g.coupled.mean(), res = s.coupled.mean();
      const double rse = s.coupled.stderr_();
      printf("%7.1f | %5d | %+9.6f (%.6f) | %+9.6f (%.6f) | %10.1f | %+9.4f |",
             lam, m, gap, g.coupled.stderr_(), res, rse, std::fabs(res) / rse,
             res * double(m));
      if (std::fabs(res) / rse < 3.0) printf("   (noise)\n");
      else printf(" %10.1fx\n", std::fabs(gap / res));
    }
    printf("--------+-------+--------------------+--------------------+------------+"
           "-----------+------------\n");
  }
  printf(
      "\n    THE BOUNDARY, stated as a rate instead of as an adjective:\n"
      "\n"
      "      pure diffusion   gap = O(dt^0.5),  residual = O(dt^1.5)\n"
      "      with jumps       gap = O(dt^0.5),  residual = O(dt^1.0)\n"
      "\n"
      "    The gap column confirms the first half in both cases: a fourfold\n"
      "    refinement halves it, which is dt^0.5. The res*m column confirms the\n"
      "    second: it is roughly constant once jumps dominate, so the residual is\n"
      "    O(dt), whereas exp_correction's res*m^1.5 column is the constant one\n"
      "    under pure diffusion.\n"
      "\n"
      "    So the correction keeps working -- it still removes most of the gap,\n"
      "    and it is free -- but it stops being asymptotically exact. The barrier\n"
      "    shift compensates for a GAUSSIAN overshoot, and the part of the gap\n"
      "    caused by a jump crossing the barrier and returning inside a monitoring\n"
      "    interval is untouched by it. The improvement factor therefore grows\n"
      "    like sqrt(m) with jumps instead of like m without them.\n");
  return 0;
}
