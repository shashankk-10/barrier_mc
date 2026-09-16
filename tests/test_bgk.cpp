// The continuity correction, and the accumulators underneath it.
//
// This file exists because of a mutation-testing result. Flipping the sign in
// bgk_barrier -- shifting the barrier toward spot instead of away from it, which
// is the single most likely thing to get wrong about the correction -- survived
// the entire suite. Everything that exercised the correction lived in the
// experiments, which ctest does not run, so the headline formula of the project
// was the one piece of it with no test at all.
//
// The assertion that kills it is not a hardcoded price. It is the structural
// statement the correction makes: the shifted continuous price must be closer to
// the discretely-monitored benchmark than the unshifted one, by a wide and
// widening margin. A correction applied backwards makes it further away.

#include <cmath>
#include <initializer_list>

#include "bm/bgk.hpp"
#include "bm/bs.hpp"
#include "bm/exact.hpp"
#include "bm/stats.hpp"
#include "check.hpp"

using namespace bm;

static void test_beta_constant() {
  // -zeta(1/2)/sqrt(2*pi), from scipy.special.zeta at 17 digits.
  CHECK_REL(kBeta, 0.58259715793901068, 1e-15);
  CHECK(kBeta > 0.0);
}

static void test_shift_direction() {
  const Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.30, 0.2};
  for (int m : {10, 50, 1000}) {
    const double dt = p.T / double(m);
    const double down = bgk_barrier(p.H, Dir::Down, p.sigma, dt);
    const double up = bgk_barrier(110.0, Dir::Up, p.sigma, dt);
    // Both move away from spot: discrete monitoring misses crossings, so it
    // behaves like a more distant barrier.
    CHECK(down < p.H);
    CHECK(up > 110.0);
    // and by the stated amount
    CHECK_REL(down, p.H * std::exp(-kBeta * p.sigma * std::sqrt(dt)), 1e-14);
    // The shift vanishes as monitoring becomes continuous.
    CHECK(p.H - down > 0.0);
  }
  // Finer monitoring means a smaller shift.
  const double s10 = bgk_barrier(95.0, Dir::Down, 0.3, 0.02);
  const double s1000 = bgk_barrier(95.0, Dir::Down, 0.3, 0.0002);
  CHECK(s1000 > s10);
}

static void test_correction_actually_corrects() {
  const Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.30, 0.2};
  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
  double prev_factor = 0.0;
  for (int m : {50, 100, 200, 400}) {
    const double ex =
        discrete_exact(Side::Call, Dir::Down, Knock::Out, p, m, 1u << 12).price;
    const double shifted = barrier_bgk(Side::Call, Dir::Down, Knock::Out, p, m);
    const double before = std::fabs(ex - cont);
    const double after = std::fabs(ex - shifted);
    CHECK(after < before);
    const double factor = before / after;
    CHECK(factor > 100.0);        // it is a large improvement, not a nudge
    CHECK(factor > prev_factor);  // and it improves as monitoring gets finer
    prev_factor = factor;
  }
}

static void test_leading_coefficient() {
  const Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.30, 0.2};
  // Raising a down-and-out barrier destroys value.
  const double d = dV_dH(Side::Call, Dir::Down, Knock::Out, p);
  CHECK(d < 0.0);
  // So the predicted leading coefficient of the gap is positive: discrete
  // monitoring makes a knock-out worth MORE.
  const double c1 = leading_gap_coeff(Side::Call, Dir::Down, Knock::Out, p);
  CHECK(c1 > 0.0);
  CHECK_REL(c1, -kBeta * p.sigma * p.H * d, 1e-14);
  // And it predicts the gap to within the second-order term at moderate m.
  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
  const double ex =
      discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 400, 1u << 12).price;
  const double predicted = c1 * std::sqrt(p.T / 400.0);
  CHECK_REL(ex - cont, predicted, 0.05);  // within 5% at m=400
}

// Welford against a variance that is known in closed form. The mutation run also
// found that accumulating M2 against the post-update mean -- a plausible
// misreading of the algorithm -- changed no assertion anywhere in the suite.
static void test_welford() {
  const int n = 1000;
  Welford w;
  for (int i = 1; i <= n; ++i) w.add(double(i));
  CHECK_EQ((long long)w.count(), (long long)n);
  CHECK_REL(w.mean(), 0.5 * double(n + 1), 1e-13);
  // Sample variance of 1..n is n(n+1)/12 exactly.
  CHECK_REL(w.var(), double(n) * double(n + 1) / 12.0, 1e-12);

  // Merging two halves must reproduce the single-pass result exactly enough
  // that the thread count cannot change a reported standard error.
  Welford a, b;
  for (int i = 1; i <= n / 2; ++i) a.add(double(i));
  for (int i = n / 2 + 1; i <= n; ++i) b.add(double(i));
  a.merge(b);
  CHECK_REL(a.mean(), w.mean(), 1e-13);
  CHECK_REL(a.var(), w.var(), 1e-11);

  // A constant stream has exactly zero variance, and must not come out
  // negative -- which is what the cancelling textbook formula does when the
  // mean is large relative to the spread.
  Welford c;
  for (int i = 0; i < 500; ++i) c.add(1e8);
  CHECK(c.var() >= 0.0);
  CHECK_NEAR(c.var(), 0.0, 1e-6);
}

static void test_fit_line() {
  // An exact line must come back exactly.
  std::vector<double> x, y;
  for (int i = 0; i < 10; ++i) {
    x.push_back(0.5 * double(i));
    y.push_back(-1.25 + 2.5 * (0.5 * double(i)));
  }
  const LineFit f = fit_line(x, y);
  CHECK_REL(f.slope, 2.5, 1e-12);
  CHECK_REL(f.intercept, -1.25, 1e-12);
  CHECK_NEAR(f.r2, 1.0, 1e-12);
}

int main() {
  test_beta_constant();
  test_shift_direction();
  test_correction_actually_corrects();
  test_leading_coefficient();
  test_welford();
  test_fit_line();
  return check::finish("bgk");
}
