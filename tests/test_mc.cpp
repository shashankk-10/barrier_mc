// The Monte Carlo estimators, checked against the two things that already have
// independent references: the continuous closed form and the convolution
// benchmark.
//
// The assertions that matter here are not "the price is about right". They are
// the structural claims the README makes:
//   - the bridge estimator is unbiased for the continuous price at every m, so
//     its mean must not drift with m;
//   - the discrete estimator converges to the convolution benchmark at each m;
//   - the coupled difference has a variance far below either term, which is the
//     entire reason the experiment is feasible;
//   - the answer does not depend on how many threads ran it.

#include <cmath>


#include "bm/bs.hpp"
#include "bm/exact.hpp"
#include "bm/mc.hpp"
#include "check.hpp"

using namespace bm;

static const Params kP{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};

// A barrier so far below spot that no path can reach it must leave both
// estimators equal to the vanilla payoff. That makes this a direct test of the
// path construction and the generator -- terminal drift, terminal variance and
// the independence of the 1600 normals a single path consumes -- with the
// barrier machinery switched off. A bias in the paths themselves would move the
// discrete and bridge prices together and so would be invisible to every test
// that looks at their difference.
static void test_far_barrier_is_vanilla() {
  Params p = kP;
  p.H = 1e-8;
  McConfig cfg;
  cfg.paths = 400000;
  cfg.m0 = 25;
  cfg.levels = 3;
  const auto run = run_convergence(Side::Call, Dir::Down, p, cfg);
  const double van = bs_vanilla(Side::Call, p.S, p.K, p.r, p.q, p.sigma, p.T);
  for (const auto& L : run.levels) {
    CHECK(std::fabs((L.discrete.mean() - van) / L.discrete.stderr_()) < 4.0);
    CHECK(std::fabs((L.bridge.mean() - van) / L.bridge.stderr_()) < 4.0);
    // With no reachable barrier the two estimators must agree path by path.
    CHECK_NEAR(L.bias.mean(), 0.0, 1e-12);
    CHECK_NEAR(L.bias.var(), 0.0, 1e-12);
  }
}

static void test_bridge_is_continuous() {
  McConfig cfg;
  cfg.paths = 200000;
  cfg.m0 = 25;
  cfg.levels = 5;  // 25, 50, 100, 200, 400
  const auto run = run_convergence(Side::Call, Dir::Down, kP, cfg);
  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, kP);

  for (const auto& L : run.levels) {
    // Unbiased for the continuous price, whatever m is.
    const double z = (L.bridge.mean() - cont) / L.bridge.stderr_();
    CHECK(std::fabs(z) < 4.0);
  }
  // And therefore flat in m: the coarsest and finest levels must agree with
  // each other far more tightly than either agrees with the discrete price.
  const auto& lo = run.levels.front();
  const auto& hi = run.levels.back();
  const double spread = std::fabs(hi.bridge.mean() - lo.bridge.mean());
  const double disc_spread =
      std::fabs(hi.discrete.mean() - lo.discrete.mean());
  CHECK(spread < 0.02);
  CHECK(disc_spread > 0.3);
  CHECK(disc_spread > 15.0 * spread);
}

static void test_discrete_matches_benchmark() {
  McConfig cfg;
  cfg.paths = 200000;
  cfg.m0 = 25;
  cfg.levels = 4;  // 25, 50, 100, 200
  const auto run = run_convergence(Side::Call, Dir::Down, kP, cfg);
  for (const auto& L : run.levels) {
    const double ref =
        discrete_conv(Side::Call, Dir::Down, Knock::Out, kP, L.m, 1u << 12);
    const double z = (L.discrete.mean() - ref) / L.discrete.stderr_();
    CHECK(std::fabs(z) < 4.0);
    // The coupled bias estimator must agree with (benchmark - closed form).
    const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, kP);
    const double zb = (L.bias.mean() - (ref - cont)) / L.bias.stderr_();
    CHECK(std::fabs(zb) < 4.0);
  }
}

static void test_coupling_reduces_variance() {
  McConfig cfg;
  cfg.paths = 100000;
  cfg.m0 = 50;
  cfg.levels = 1;
  const auto run = run_convergence(Side::Call, Dir::Down, kP, cfg);
  const auto& L = run.levels.front();

  // If the two estimators were run on independent paths, the variance of their
  // difference would be the sum of their variances. Coupling them on one path
  // must beat that by a wide margin, or the whole estimator design is pointless.
  const double indep = L.discrete.var() + L.bridge.var();
  CHECK(L.bias.var() > 0.0);
  CHECK(L.bias.var() * 20.0 < indep);
}

static void test_thread_invariance() {
  // A path's randomness is a pure function of its index, so every path gets the
  // same value whatever the thread count. What that does NOT give you is a
  // bit-identical mean: the reduction sums the same numbers in a different
  // order, and floating point addition is not associative. So the assertion
  // below is a tight tolerance, not equality -- see the note at the loop.
  // This is the test that would catch a generator being advanced per-thread
  // instead of per-path.
  McConfig a;
  a.paths = 20000;
  a.m0 = 25;
  a.levels = 3;
  a.threads = 1;
  McConfig b = a;
  b.threads = 5;
  const auto ra = run_convergence(Side::Call, Dir::Down, kP, a);
  const auto rb = run_convergence(Side::Call, Dir::Down, kP, b);
  CHECK_EQ((long long)ra.levels.size(), (long long)rb.levels.size());
  for (size_t i = 0; i < ra.levels.size(); ++i) {
    CHECK_EQ((long long)ra.levels[i].discrete.count(),
             (long long)rb.levels[i].discrete.count());
    // Floating-point summation order still differs between thread counts, so
    // this is "the same to rounding", not bit-identical -- and the tolerance is
    // tight enough that a genuinely different path set would fail it by orders
    // of magnitude.
    CHECK_NEAR(ra.levels[i].discrete.mean(), rb.levels[i].discrete.mean(), 1e-10);
    CHECK_NEAR(ra.levels[i].bridge.mean(), rb.levels[i].bridge.mean(), 1e-10);
    CHECK_NEAR(ra.levels[i].bias.mean(), rb.levels[i].bias.mean(), 1e-10);
  }
}

static void test_plain_agrees() {
  // The naive estimator and the coupled one must price the same contract.
  McConfig cfg;
  cfg.paths = 200000;
  cfg.m0 = 50;
  cfg.levels = 1;
  const auto coupled = run_convergence(Side::Call, Dir::Down, kP, cfg);
  const auto plain = run_plain(Side::Call, Dir::Down, kP, 50, cfg);
  const double se = std::sqrt(
      plain.price.stderr_() * plain.price.stderr_() +
      coupled.levels[0].discrete.stderr_() * coupled.levels[0].discrete.stderr_());
  CHECK_NEAR(plain.price.mean(), coupled.levels[0].discrete.mean(), 4.0 * se);
}

static void test_up_direction() {
  Params p{100.0, 100.0, 110.0, 0.05, 0.0, 0.3, 0.2};
  McConfig cfg;
  cfg.paths = 200000;
  cfg.m0 = 50;
  cfg.levels = 1;
  const auto run = run_convergence(Side::Call, Dir::Up, p, cfg);
  const double cont = barrier_cont(Side::Call, Dir::Up, Knock::Out, p);
  const double z = (run.levels[0].bridge.mean() - cont) /
                   run.levels[0].bridge.stderr_();
  CHECK(std::fabs(z) < 4.0);
  const double ref =
      discrete_conv(Side::Call, Dir::Up, Knock::Out, p, 50, 1u << 12);
  const double zd = (run.levels[0].discrete.mean() - ref) /
                    run.levels[0].discrete.stderr_();
  CHECK(std::fabs(zd) < 4.0);
}

static void test_antithetic_is_unbiased() {
  // Antithetic variates on a discontinuous payoff are still unbiased; whether
  // they reduce variance is a separate question the benchmark answers.
  McConfig cfg;
  cfg.paths = 100000;
  cfg.m0 = 50;
  cfg.levels = 1;
  cfg.antithetic = true;
  const auto run = run_convergence(Side::Call, Dir::Down, kP, cfg);
  const double ref =
      discrete_conv(Side::Call, Dir::Down, Knock::Out, kP, 50, 1u << 12);
  const double z =
      (run.levels[0].discrete.mean() - ref) / run.levels[0].discrete.stderr_();
  CHECK(std::fabs(z) < 4.0);
}

int main() {
  test_far_barrier_is_vanilla();
  test_bridge_is_continuous();
  test_discrete_matches_benchmark();
  test_coupling_reduces_variance();
  test_thread_invariance();
  test_plain_agrees();
  test_up_direction();
  test_antithetic_is_unbiased();
  return check::finish("mc");
}
