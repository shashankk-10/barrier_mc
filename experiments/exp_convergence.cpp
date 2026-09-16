// exp_convergence: how big the discretisation gap is, and how slowly it closes.
//
// Prints, for a ladder of monitoring frequencies:
//   * the discretely-monitored price, from the convolution benchmark, with the
//     Richardson residual as its error bar (no sampling error at all);
//   * the continuously-monitored price from the closed form;
//   * the gap between them, and the same gap measured a completely different
//     way, by the coupled Monte Carlo estimator;
//   * the standard error a plain Monte Carlo run reports at the same m, and the
//     ratio of the gap to it.
//
// The last column is the point. It is the number of standard errors by which a
// simulation that is converging correctly to its own answer is away from the
// answer its user probably wanted.

#include <cmath>
#include <cstdio>
#include <vector>

#include "bm/bs.hpp"
#include "bm/exact.hpp"
#include "bm/mc.hpp"
#include "bm/stats.hpp"
#include "harness.hpp"

using namespace bm;

int main() {
  const Params p = bmx::headline();
  bmx::rule("exp_convergence: the size of the discretisation gap");
  bmx::describe(p, "down-and-out call, zero rebate");

  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
  printf("continuous closed form (bs.hpp)  : %.12f\n", cont);

  McConfig cfg;
  cfg.paths = 1u << 21;  // 2,097,152
  cfg.m0 = 25;
  cfg.levels = 7;  // 25 .. 1600
  bmx::Timer tm;
  const auto run = run_convergence(Side::Call, Dir::Down, p, cfg);
  const double mc_secs = tm.s();

  printf("monte carlo                      : %llu paths, %d threads, %.1f s\n",
         (unsigned long long)run.paths, run.threads, mc_secs);
  printf(
      "\n"
      "    m |   exact discrete  | richardson |  gap (exact) |   gap %%  |"
      "  gap (coupled MC)    | plain MC s.e. | gap/s.e.\n"
      "------+-------------------+------------+--------------+----------+"
      "----------------------+---------------+---------\n");

  std::vector<double> logdt, logbias, exact_gap;
  for (const auto& L : run.levels) {
    const auto ex = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, L.m);
    const double gap = ex.price - cont;
    exact_gap.push_back(gap);
    // The standard error a plain run would report at this m and this path
    // count: the coupled discrete estimator has the same variance as a plain
    // one, because it IS a plain one -- the coupling only affects the
    // difference, not the level.
    const double se = L.discrete.stderr_();
    printf("%5d | %.12f |  %8.1e | %+11.8f | %+6.2f%% | %+11.8f (%.2e) | %13.2e | %7.1f\n",
           L.m, ex.price, std::fabs(ex.richardson), gap, 100.0 * gap / cont,
           L.bias.mean(), L.bias.stderr_(), se, gap / se);
    logdt.push_back(std::log(p.T / double(L.m)));
    logbias.push_back(std::log(std::fabs(gap)));
  }

  const LineFit f = fit_line(logdt, logbias);
  printf(
      "\nfitted slope of log|gap| on log(dt) : %.4f   R^2 = %.5f\n"
      "   (residual scatter %.4f -- NOT a confidence interval: the points share\n"
      "    paths by subsampling, so their errors are correlated. See stats.hpp.)\n",
      f.slope, f.r2, f.slope_se);
  printf("theory (Broadie-Glasserman-Kou)     : 1/2\n");

  // Successive ratios are the honest evidence: a slope fitted across a range
  // that includes points which have not reached the asymptotic regime is an
  // average of two different behaviours. If the gap really scales as sqrt(dt),
  // halving dt must multiply the gap by 1/sqrt(2) = 0.7071, and the ratios
  // should approach that from one side instead of scatter around it.
  printf("\nsuccessive ratios gap(2m)/gap(m)    : ");
  for (size_t i = 1; i < run.levels.size(); ++i) {
    const double a = std::exp(logbias[i - 1]), b = std::exp(logbias[i]);
    printf("%.4f ", b / a);
  }
  printf("\n                          (theory)    : %.4f\n", 1.0 / std::sqrt(2.0));

  // The two independent measurements of the same quantity must agree. One comes
  // from a deterministic transform, the other from random paths; they share no
  // code beyond the parameter struct.
  printf("\nagreement, exact gap vs coupled MC gap (in MC standard errors):\n   ");
  for (size_t i = 0; i < run.levels.size(); ++i) {
    const auto& L = run.levels[i];
    printf("%+.2f ", (L.bias.mean() - exact_gap[i]) / L.bias.stderr_());
  }
  printf("\n");
  return 0;
}
