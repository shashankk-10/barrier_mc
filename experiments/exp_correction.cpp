// What the barrier shift buys, and what it leaves behind.
//
// Everything here uses the convolution benchmark, which has no sampling error.
// The Monte Carlo cross-check of the same quantities is in exp_convergence.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "bm/bgk.hpp"
#include "bm/bs.hpp"
#include "bm/exact.hpp"
#include "bm/stats.hpp"
#include "harness.hpp"

using namespace bm;

namespace {

// Least squares for gap = c1*sqrt(dt) + c2*dt, solved as a 2x2 normal system.
// Small and explicit beats pulling in a linear algebra dependency for one fit.
void fit_two_term(const std::vector<double>& dt, const std::vector<double>& g,
                  double& c1, double& c2) {
  double a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
  for (size_t i = 0; i < dt.size(); ++i) {
    const double u = std::sqrt(dt[i]), v = dt[i];
    a11 += u * u;
    a12 += u * v;
    a22 += v * v;
    b1 += u * g[i];
    b2 += v * g[i];
  }
  const double det = a11 * a22 - a12 * a12;
  c1 = (b1 * a22 - b2 * a12) / det;
  c2 = (a11 * b2 - a12 * b1) / det;
}

}  // namespace

int main() {
  const Params p = bmx::headline();
  bmx::rule("exp_correction: what the barrier shift buys");
  bmx::describe(p, "down-and-out call, zero rebate");

  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
  printf("continuous closed form            : %.12f\n", cont);
  printf("beta = -zeta(1/2)/sqrt(2 pi)      : %.17g\n", kBeta);

  const int ms[] = {25, 50, 100, 200, 400, 800, 1600};
  const size_t n = sizeof(ms) / sizeof(ms[0]);

  printf(
      "\n"
      "    m | log(S/H)/sig.sqrt(dt) |    gap     |  residual after |"
      " |gap|/|resid| | resid * m^1.5\n"
      "      |    (validity ratio)   |            |   barrier shift |"
      "              |\n"
      "------+-----------------------+------------+-----------------+"
      "--------------+--------------\n");

  // Each discrete_exact is two full convolution solves, and the implied-shift
  // inversion below wants the same seven numbers, so keep them.
  std::vector<double> dts, gaps, resids, prices;
  for (size_t i = 0; i < n; ++i) {
    const int m = ms[i];
    const double dt = p.T / double(m);
    const double ex = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, m).price;
    const double shifted = barrier_bgk(Side::Call, Dir::Down, Knock::Out, p, m);
    const double gap = ex - cont;
    const double res = ex - shifted;
    const double ratio = std::log(p.S / p.H) / (p.sigma * std::sqrt(dt));
    printf("%5d | %21.3f | %+10.7f | %+15.3e | %12.0f | %+12.3e\n", m, ratio,
           gap, res, std::fabs(gap / res), res * std::pow(double(m), 1.5));
    dts.push_back(dt);
    gaps.push_back(gap);
    resids.push_back(res);
    prices.push_back(ex);
  }

  // --- rate of the residual -------------------------------------------------
  std::vector<double> ldt, lres, lgap;
  for (size_t i = 0; i < n; ++i) {
    ldt.push_back(std::log(dts[i]));
    lres.push_back(std::log(std::fabs(resids[i])));
    lgap.push_back(std::log(std::fabs(gaps[i])));
  }
  const LineFit fr_all = fit_line(ldt, lres);
  std::vector<double> ldt2(ldt.begin() + 2, ldt.end()),
      lres2(lres.begin() + 2, lres.end());
  const LineFit fr_tail = fit_line(ldt2, lres2);
  printf("\nfitted slope of log|residual| on log(dt):\n");
  printf("   all m          : %.4f   (dragged by the two coarsest rows)\n",
         fr_all.slope);
  printf("   m >= 100       : %.4f\n", fr_tail.slope);
  printf("   BGK guarantees : o(dt^0.5). Measured here: dt^1.5.\n");
  printf(
      "\n   Where the asymptote starts is itself a measurement. resid*m^1.5 is\n"
      "   flat at -0.103 from m=100 onward, is 13%% off it at m=50, and is 2.6x\n"
      "   off at m=25, so the expansion becomes usable at a validity ratio of\n"
      "   about 3.8, not at the ratio of 2 that 'log(S/H) >> sigma*sqrt(dt)'\n"
      "   might suggest. That threshold is the operationally useful form of the\n"
      "   theorem's hypothesis.\n");

  // --- the implied barrier shift -------------------------------------------
  //
  // This is the test of beta, and it replaced an earlier one that did not
  // survive scrutiny. The obvious test is to fit gap = c1*sqrt(dt) + c2*dt and
  // compare c1 against its closed-form prediction -beta*sigma*H*dV/dH. That fit
  // agrees to 0.22% on this contract, and the agreement is an artefact: move
  // the window to m >= 200 and it becomes 0.06%, add a dt^1.5 basis term and it
  // becomes 0.14%, run the identical protocol on other contracts and it becomes
  // 0.77%. It is the truncation error of a two-term basis, and this one lands on
  // the flattering side of it.
  //
  // The inversion has no basis and no window. For each m, ask what barrier shift
  // the price implies: solve
  //     barrier_cont(H * exp(-b * sigma * sqrt(dt)))  =  exact_discrete(m)
  // for b by bisection. If the correction is right, b -> beta as dt -> 0, and
  // since the residual is O(dt^1.5) while the price's sensitivity to b is
  // O(sqrt(dt)), the approach must be O(dt), linear, so one Richardson step
  // gives beta to many more digits than the raw sequence.
  printf("\nthe barrier shift the price implies (no fit, no window, no basis):\n\n");
  printf("      m |   b implied    |  b - beta   | ratio to previous\n");
  printf("--------+----------------+-------------+-------------------\n");
  std::vector<double> bimp;
  for (size_t i = 0; i < n; ++i) {
    const int m = ms[i];
    const double target = prices[i];
    const double rt = std::sqrt(dts[i]);
    // The shifted price is increasing in b (a larger shift moves the barrier
    // further from spot, so fewer paths knock out).
    double lo = 0.0, hi = 3.0;
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi);
      Params q = p;
      q.H = p.H * std::exp(-mid * p.sigma * rt);
      if (barrier_cont(Side::Call, Dir::Down, Knock::Out, q) < target) lo = mid;
      else hi = mid;
    }
    const double b = 0.5 * (lo + hi);
    bimp.push_back(b);
    printf("%7d | %14.8f | %+11.3e |", m, b, b - kBeta);
    if (i > 0) printf(" %17.2f\n", (bimp[i - 1] - kBeta) / (b - kBeta));
    else printf("        --\n");
  }
  {
    // b(m) = beta - C*dt, so beta = 2*b(2m) - b(m).
    const double ext = 2.0 * bimp[n - 1] - bimp[n - 2];
    printf("\n   Richardson on the last pair : %.10f\n", ext);
    printf("   -zeta(1/2)/sqrt(2 pi)       : %.10f\n", kBeta);
    printf("   disagreement                : %.1e\n", std::fabs(ext - kBeta));
  }

  // The two-term fit, kept as a secondary row and labelled for what it is.
  double c1 = 0, c2 = 0;
  fit_two_term(dts, gaps, c1, c2);
  const double c1_theory = leading_gap_coeff(Side::Call, Dir::Down, Knock::Out, p);
  printf(
      "\nsecondary, and truncation-limited to a few tenths of a percent:\n");
  printf("   dV/dH at H=%.4g                : %+.8f\n", p.H,
         dV_dH(Side::Call, Dir::Down, Knock::Out, p));
  printf("   predicted  -beta*sigma*H*dV/dH : %.8f\n", c1_theory);
  printf("   fitted     c1                  : %.8f   (c2 = %+.5f)\n", c1, c2);
  printf("   disagreement                   : %+.3f %%  <- basis-dependent\n",
         100.0 * (c1 / c1_theory - 1.0));
  printf(
      "\n   Note that this and the residual test are NOT independent checks of\n"
      "   beta: both are the first-order expansion of the same shifted closed\n"
      "   form. The independent evidence is (a) the residual exponent: a beta\n"
      "   wrong by a relative eps would leave an O(sqrt(dt)) residual, not the\n"
      "   O(dt^1.5) measured above, and (b) the Spitzer route in exp_overshoot,\n"
      "   which never looks at an option price at all.\n");

  const LineFit fg = fit_line(ldt, lgap);
  printf(
      "\nwhy the one-term log-log slope reads %.4f instead of 0.5000:\n"
      "   because c2*dt is not negligible. With c1 = %.3f and c2 = %.3f, the\n"
      "   dt term is %.1f%% of the sqrt(dt) term at m=25 and still %.1f%% at\n"
      "   m=1600. Fitting a single power through a two-term expansion returns\n"
      "   an average of the two exponents, not the leading one.\n",
      fg.slope, c1, c2,
      100.0 * std::fabs(c2 * dts.front() / (c1 * std::sqrt(dts.front()))),
      100.0 * std::fabs(c2 * dts.back() / (c1 * std::sqrt(dts.back()))));

  printf("\ngap / (c1_theory * sqrt(dt)), should approach 1:\n   ");
  for (size_t i = 0; i < n; ++i)
    printf("%.4f ", gaps[i] / (c1_theory * std::sqrt(dts[i])));
  printf("\n");

  // --- is the benchmark itself good enough to say any of this? -------------
  //
  // The residual being measured falls to 1.7e-6 by m=1600, and the benchmark
  // that measures it is a numerical scheme with an error of its own. If that
  // error is not far below the residual, the "dt^1.5" column is reporting the
  // grid, not the mathematics, and it would look exactly the same either way.
  //
  // So: recompute two of the rows at successively finer grids and watch whether
  // resid*m^1.5 settles. It does, and it settles on the value the mid-range rows
  // already show, which is what licenses the claim. It also shows that at the
  // default resolution the largest-m rows are grid-limited by about 8%, which is
  // the drift visible in the table above and is not a real departure from the
  // power law.
  printf(
      "\n--- benchmark resolution self-check ----------------------------------\n"
      "resid * m^1.5 should be constant if the residual really is O(dt^1.5).\n"
      "Re-running two rows at finer grids separates mathematics from grid error:\n\n"
      "    m |   grid (n/2n)   |  extrapolated price  |  resid * m^1.5\n"
      "------+-----------------+----------------------+----------------\n");
  for (int m : {400, 1600}) {
    const double shifted = barrier_bgk(Side::Call, Dir::Down, Knock::Out, p, m);
    for (size_t g = 12; g <= 15; ++g) {
      const size_t nc = size_t(1) << g;
      const auto ex =
          discrete_exact(Side::Call, Dir::Down, Knock::Out, p, m, nc);
      printf("%5d | %6zu/%-8zu | %20.12f | %+14.5f\n", m, nc, 2 * nc, ex.price,
             (ex.price - shifted) * std::pow(double(m), 1.5));
    }
    printf("------+-----------------+----------------------+----------------\n");
  }
  printf(
      "Every row lands on the same -0.103 once the grid is adequate, and at the\n"
      "default resolution both m = 400 and m = 1600 are already there. That was\n"
      "not true before the Richardson denominator was corrected for the true grid\n"
      "ratio (1.99614, not 2): with the textbook /3 the m = 1600 row read -0.1113\n"
      "and the residual exponent came out 1.4555 instead of 1.5016, which looked\n"
      "like the theory failing at fine monitoring and was this file's arithmetic.\n");
  
  // --- what the correction does to the hedge -------------------------------
  //
  // A price that is close is worth little if the hedge ratio is not, and the
  // risk on a barrier book is carried in the delta, not the premium. The
  // benchmark already has the whole value function on its grid, so the delta is
  // two array reads away and costs nothing extra.
  //
  // This is also where the correction is shown failing. Every other table here
  // uses a barrier comfortably far from spot; the second row below puts it 50bp
  // away, which is autocall-shaped, and the shift stops being worth much.
  printf(
      "\n--- delta: what the shift buys the hedge -----------------------------\n"
      "\n      contract        |  m  | ratio |   exact   |  shifted (err)   |"
      " uncorrected (err)\n"
      "----------------------+-----+-------+-----------+------------------+"
      "-------------------\n");
  struct Row { const char* label; Params p; int m; };
  Params near = p;
  near.H = 99.5;
  near.sigma = 0.20;
  near.T = 1.0;
  const Row rows[] = {
      {"headline, daily   ", p, 50},
      {"headline, 252 fix ", p, 252},
      {"barrier 50bp away ", near, 252},
  };
  ExactResult near_ex;  // kept for the paragraph below, so it is solved once
  for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
    const Row& r = rows[i];
    const auto ex = discrete_exact(Side::Call, Dir::Down, Knock::Out, r.p, r.m);
    if (i + 1 == sizeof(rows) / sizeof(rows[0])) near_ex = ex;
    const double ds = barrier_bgk_delta(Side::Call, Dir::Down, Knock::Out, r.p, r.m);
    const double dc = barrier_cont_delta(Side::Call, Dir::Down, Knock::Out, r.p);
    const double ratio =
        std::log(r.p.S / r.p.H) / (r.p.sigma * std::sqrt(r.p.T / double(r.m)));
    printf("%s    | %3d | %5.2f | %9.6f | %9.6f %+6.2f%% | %9.6f %+6.2f%%\n",
           r.label, r.m, ratio, ex.delta, ds, 100.0 * (ds / ex.delta - 1.0), dc,
           100.0 * (dc / ex.delta - 1.0));
  }
  {
    const ExactResult& ex = near_ex;
    const double sh = barrier_bgk(Side::Call, Dir::Down, Knock::Out, near, 252);
    printf(
        "\n    On the headline contract the shift preserves the analytic delta to\n"
        "    0.015%%, which is the production case for it: the fast closed-form\n"
        "    pricer keeps both its price and its hedge. At a validity ratio of\n"
        "    0.40 it does not. There the shifted price is still %.2f%% off (%.6f\n"
        "    against %.6f) and the delta is 15%% off, so the correction buys\n"
        "    almost nothing exactly where the hedge is most dangerous, and the\n"
        "    contract belongs on a lattice instead.\n",
        100.0 * (ex.price - sh) / ex.price, sh, ex.price);
  }

  // --- what the fast path is worth -----------------------------------------
  //
  // The reason to care about any of this is that the corrected closed form can
  // sit in a pricer that is called constantly, and a grid method cannot. Best of
  // five batches, since the noise here is one-sided.
  {
    volatile double sink = 0.0;
    double best_ns = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
      const int N = 200000;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < N; ++i)
        sink += barrier_bgk(Side::Call, Dir::Down, Knock::Out, p, 50);
      const auto t1 = std::chrono::steady_clock::now();
      const double ns =
          std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
      if (ns < best_ns) best_ns = ns;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto ex = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 50);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    sink += ex.price;
    printf(
        "\n--- cost -------------------------------------------------------------\n"
        "\n   corrected closed form  : %.0f ns per price (best of 5)\n"
        "   convolution benchmark  : %.0f ms at m=50\n"
        "\n   Do not read that ratio as a lattice-versus-formula benchmark. This\n"
        "   benchmark is deliberately extravagant: it solves the whole value\n"
        "   function on two grids and extrapolates, to get six digits it does not\n"
        "   need for pricing. A production lattice for one price is far cheaper\n"
        "   than %.0f ms. The honest statement is the order of magnitude: a\n"
        "   closed form is tens of nanoseconds and any grid method is\n"
        "   milliseconds, so the shift is what decides whether a discretely-fixed\n"
        "   barrier can live in the pricer that gets called on every quote.\n",
        best_ns, ms, ms);
  }
  return 0;
}
