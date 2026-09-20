// The jump-diffusion module.
//
// Same story as test_bgk: mutation testing found jump.cpp had no assertions in
// ctest. The controls below are what make the rest of exp_jump believable.

#include <cmath>
#include <initializer_list>

#include "bm/bs.hpp"
#include "bm/exact.hpp"
#include "bm/jump.hpp"
#include "check.hpp"

using namespace bm;

static const Params kP{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};

// lambda = 0 must reproduce the pure-diffusion answers exactly, because the jump
// code path then differs from mc.cpp only in how it spends dimensions.
static void test_zero_intensity_matches_diffusion() {
  JumpParams jp;
  jp.lambda = 0.0;
  McConfig cfg;
  cfg.paths = 200000;

  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, kP);
  for (int m : {50, 200}) {
    const auto r = run_jump(Side::Call, Dir::Down, kP, jp, m, cfg);
    CHECK_NEAR(r.mean_jumps, 0.0, 0.0);
    const double ref =
        discrete_conv(Side::Call, Dir::Down, Knock::Out, kP, m, 1u << 12);
    CHECK(std::fabs((r.discrete.mean() - ref) / r.discrete.stderr_()) < 4.0);
    CHECK(std::fabs((r.bridge_jump.mean() - cont) / r.bridge_jump.stderr_()) <
          4.0);
    // With no jumps the two bridge estimators are the same estimator.
    CHECK_NEAR(r.bridge_naive.mean(), r.bridge_jump.mean(), 1e-12);
  }
}

// E[S_T] e^{-(r-q)T} / S_0 = 1 exactly. This is what checks the jump
// compensator: get -lambda*kappa wrong and the model prices a different asset,
// with every barrier number downstream quietly about that other asset.
static void test_martingale_and_jump_count() {
  McConfig cfg;
  cfg.paths = 200000;
  for (double lam : {0.0, 1.0, 5.0}) {
    JumpParams jp;
    jp.lambda = lam;
    const auto r = run_jump(Side::Call, Dir::Down, kP, jp, 100, cfg);
    CHECK(std::fabs(r.spot.mean() - 1.0) < 4.0 * r.spot.stderr_());
    // Poisson(lambda*T) per path, and the sampler truncates at 6 per interval --
    // so a realised mean far from lambda*T would mean the truncation is biting.
    CHECK_NEAR(r.mean_jumps, lam * kP.T, 0.02);
  }
}

// Every price must be finite and non-negative, at an intensity high enough that
// intervals routinely contain more than one jump. This is the assertion that
// kills a mis-ordered jump-time list: an out-of-order pair makes a sub-interval
// span negative time, log_survival is handed a negative argument, and the
// survival weight comes back NaN.
static void test_multi_jump_intervals_are_sane() {
  McConfig cfg;
  cfg.paths = 60000;
  JumpParams jp;
  // lambda*dt = 2 at m=25, so intervals routinely hold several jumps. This is
  // deliberately outside the range the Poisson sampler is calibrated for
  // (it truncates at six per interval, valid for lambda*dt < 0.1, which every
  // row of exp_jump respects). Nothing here asserts accuracy at this intensity
  //, only that the multi-jump code path stays finite and correctly ordered.
  jp.lambda = 50.0;
  for (int m : {5, 25}) {
    const auto r = run_jump(Side::Call, Dir::Down, kP, jp, m, cfg);
    CHECK(std::isfinite(r.discrete.mean()));
    CHECK(std::isfinite(r.bridge_naive.mean()));
    CHECK(std::isfinite(r.bridge_jump.mean()));
    CHECK(std::isfinite(r.bridge_jump.var()));
    CHECK(r.discrete.mean() >= 0.0);
    CHECK(r.bridge_jump.mean() >= 0.0);
    // A survival probability is at most one, so conditioning on the jumps can
    // only ever remove value relative to the discrete monitor.
    CHECK(r.bridge_jump.mean() <= r.discrete.mean() + 1e-9);
    // And the naive bridge misses precisely the crossings the jump-aware one
    // catches, so it can only be the larger of the two.
    CHECK(r.bridge_naive.mean() >= r.bridge_jump.mean() - 1e-9);
  }
}

static void test_ordering_of_estimators() {
  // At a realistic intensity the same ordering must hold, and the naive bridge
  // must be strictly biased high, that gap is the finding in exp_jump.
  McConfig cfg;
  cfg.paths = 400000;
  JumpParams jp;
  jp.lambda = 3.0;
  const auto r = run_jump(Side::Call, Dir::Down, kP, jp, 50, cfg);
  CHECK(r.bridge_naive.mean() > r.bridge_jump.mean());
  CHECK(r.discrete.mean() > r.bridge_jump.mean());
  // The two estimators run on the same paths, so the standard error of their
  // difference is the coupled one, not the sum of theirs, which is looser by
  // more than an order of magnitude here and would make this assertion vacuous.
  CHECK(r.bridge_gap.mean() > 6.0 * r.bridge_gap.stderr_());
  CHECK_NEAR(r.bridge_gap.mean(),
             r.bridge_naive.mean() - r.bridge_jump.mean(), 1e-9);
}

int main() {
  test_zero_intensity_matches_diffusion();
  test_martingale_and_jump_count();
  test_multi_jump_intervals_are_sane();
  test_ordering_of_estimators();
  return check::finish("jump");
}
