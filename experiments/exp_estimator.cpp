// Why the estimator is built the way it is, in numbers.

#include <cmath>
#include <cstdio>

#include "bm/bs.hpp"
#include "bm/exact.hpp"
#include "bm/mc.hpp"
#include "harness.hpp"

using namespace bm;

int main() {
  const Params p = bmx::headline();
  bmx::rule("exp_estimator: why the coupled estimator, and what it buys");
  bmx::describe(p, "down-and-out call, zero rebate");

  McConfig cfg;
  cfg.paths = 1u << 20;
  cfg.m0 = 25;
  cfg.levels = 7;
  bmx::Timer tm;
  const auto run = run_convergence(Side::Call, Dir::Down, p, cfg);
  printf("%llu paths, %d threads, %.1f s\n\n",
         (unsigned long long)run.paths, run.threads, tm.s());

  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, p);

  // --- [1] Rao-Blackwell, and unbiasedness at every m ----------------------
  printf(
      "[1] the bridge estimator: lower variance, and flat in m\n\n"
      "    m | var(discrete) | var(bridge) | ratio |   bridge mean   |"
      " (mean - closed form)/s.e.\n"
      "------+---------------+-------------+-------+-----------------+"
      "--------------------------\n");
  for (const auto& L : run.levels) {
    printf("%5d | %13.4f | %11.4f | %5.3f | %15.9f | %+24.2f\n", L.m,
           L.discrete.var(), L.bridge.var(),
           L.bridge.var() / L.discrete.var(), L.bridge.mean(),
           (L.bridge.mean() - cont) / L.bridge.stderr_());
  }
  printf(
      "\n    The discrete price moves by %.4f across this range of m; the bridge\n"
      "    price moves by %.4f. That is the difference between an estimator that\n"
      "    has a discretisation bias and one that does not.\n",
      std::fabs(run.levels.back().discrete.mean() -
                run.levels.front().discrete.mean()),
      std::fabs(run.levels.back().bridge.mean() -
                run.levels.front().bridge.mean()));

  // --- [2] what the coupling buys ------------------------------------------
  printf(
      "\n[2] coupling the two estimators on one path\n\n"
      "    m |  var(coupled diff) | var if independent |  variance ratio\n"
      "------+--------------------+--------------------+----------------\n");
  for (const auto& L : run.levels) {
    const double indep = L.discrete.var() + L.bridge.var();
    printf("%5d | %18.6f | %18.4f | %14.1fx\n", L.m, L.bias.var(), indep,
           indep / L.bias.var());
  }
  printf(
      "\n    The two payoffs differ only on paths that dipped below the barrier\n"
      "    between two monitoring dates and came back, so the difference is\n"
      "    identically zero on most paths. Independent runs would have to\n"
      "    subtract two large noisy numbers to see the same small quantity.\n");

  // --- [3] the cost, in paths ----------------------------------------------
  printf(
      "\n[3] paths needed to measure the gap to 1%% relative\n\n"
      "    m |    gap     |  target s.e. |  coupled paths |  independent paths\n"
      "------+------------+--------------+----------------+-------------------\n");
  for (const auto& L : run.levels) {
    const double gap =
        discrete_exact(Side::Call, Dir::Down, Knock::Out, p, L.m).price - cont;
    const double target = 0.01 * std::fabs(gap);
    const double n_coupled = L.bias.var() / (target * target);
    const double n_indep =
        (L.discrete.var() + L.bridge.var()) / (target * target);
    printf("%5d | %+10.7f | %12.2e | %14.3e | %18.3e\n", L.m, gap, target,
           n_coupled, n_indep);
  }

  // --- [4] antithetic on a discontinuous payoff ----------------------------
  printf("\n[4] antithetic variates, on a payoff with a jump discontinuity\n\n");
  McConfig anti = cfg;
  anti.antithetic = true;
  anti.paths = cfg.paths / 2;  // same number of payoff evaluations
  anti.levels = 3;
  McConfig plain_cfg = cfg;
  plain_cfg.levels = 3;
  const auto ra = run_convergence(Side::Call, Dir::Down, p, anti);
  const auto rp = run_convergence(Side::Call, Dir::Down, p, plain_cfg);
  printf(
      "    matched on payoff evaluations: %llu antithetic pairs vs %llu plain\n\n"
      "    m | var/eval plain | var/eval antithetic | efficiency\n"
      "------+----------------+---------------------+-----------\n",
      (unsigned long long)anti.paths, (unsigned long long)plain_cfg.paths);
  for (size_t i = 0; i < ra.levels.size(); ++i) {
    // Cost normalisation, and it is easy to get backwards. An antithetic
    // "path" here reports the mean of two evaluations, so its variance is
    // sigma^2 (1+rho)/2 but it cost two evaluations. Variance per unit of work
    // is therefore that variance times two, not divided by it. Dividing gives
    // 4/(1+rho) instead of 1/(1+rho), a factor of four, which turns a modest
    // real gain into a spectacular fake one.
    const double v_anti = ra.levels[i].discrete.var() * 2.0;
    const double v_plain = rp.levels[i].discrete.var();
    const double eff = v_plain / v_anti;
    printf("%5d | %14.4f | %19.4f | %9.2fx   (rho = %+.3f)\n", ra.levels[i].m,
           v_plain, v_anti, eff, 1.0 / eff - 1.0);
  }
  printf(
      "\n    Efficiency is 1/(1+rho) for correlation rho between a path and its\n"
      "    reflection. The gain is real but modest: the terminal payoff is\n"
      "    monotone in the driving normals, which is the case antithetic\n"
      "    sampling suits, but the knock-out indicator is not. A path and its\n"
      "    reflection are not both near the barrier, so the negative correlation\n"
      "    that pays for the second evaluation is much weaker than it would be\n"
      "    for a vanilla.\n");
  return 0;
}
