// The convolution benchmark.
//
// Order matters here. First a case where the barrier cannot bite, so the answer
// must be the vanilla price. Then a case small enough to integrate independently.
// Only after those does anything assert a number the benchmark alone produces.

#include <cmath>
#include <cstdlib>
#include <initializer_list>

#include "bm/bs.hpp"
#include "bm/bgk.hpp"
#include "bm/exact.hpp"
#include "bm/normal.hpp"
#include "check.hpp"

using namespace bm;

static void test_inert_barrier() {
  // A down-barrier at 95 with a strike at 100, monitored only at maturity. The
  // payoff is zero unless S_T > 100, and any such path is above 95, so the
  // barrier cannot knock out a path that would have paid. The price must be the
  // vanilla one. This is the orientation test.
  Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};
  const double van = bs_vanilla(Side::Call, p.S, p.K, p.r, p.q, p.sigma, p.T);
  CHECK_REL(discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 1).price, van,
            1e-8);
  // m=0 means no monitoring at all: the same statement, by a different code
  // path. Delta has to come back too, the knock-in parity subtracts it, so a
  // zero-valued knock-in would otherwise report the whole vanilla delta.
  GridInfo g0;
  CHECK_REL(discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 0, 1u << 12, &g0),
            van, 1e-14);
  const double d0 = g0.delta;
  const double sT = p.sigma * std::sqrt(p.T);
  const double d1 =
      (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T) / sT;
  CHECK_REL(d0, std::exp(-p.q * p.T) * ncdf(d1), 1e-14);
  // An unmonitored knock-in is worth nothing, and hedges like nothing.
  GridInfo gin;
  CHECK_NEAR(discrete_conv(Side::Call, Dir::Down, Knock::In, p, 0, 1u << 12, &gin),
             0.0, 1e-13);
  CHECK_NEAR(gin.delta, 0.0, 1e-13);
}

static void test_against_quadrature() {
  // Two monitoring dates, T/2 and T. Conditioning on S_{T/2} reduces the price
  // to a one-dimensional integral of a Black-Scholes call against a lognormal
  // density, which scipy's adaptive quadrature evaluates to 1e-14. Nothing in
  // that calculation shares a line of code with the convolution.
  Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};
  const double ref = 5.597592802950301;
  const auto r = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 2);
  CHECK_REL(r.price, ref, 1e-8);
  CHECK(std::fabs(r.richardson) < 1e-6);

  // Second-order convergence at the barrier, which is what the half-weight node
  // buys. Doubling the grid must cut the error by about four; with a plain
  // full-weight sum across the discontinuity it only halves, and the residual
  // that the continuity correction is supposed to leave behind would be buried
  // under it.
  double err_prev = 0.0;
  for (size_t n = 1u << 10; n <= 1u << 13; n <<= 1) {
    const double e =
        std::fabs(discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 2, n) -
                  ref);
    if (err_prev > 0.0) {
      const double ratio = err_prev / e;
      CHECK(ratio > 3.0 && ratio < 5.0);
    }
    err_prev = e;
  }
}

// The extrapolated price must stop moving when the grid is refined.
//
// This is the assertion that was missing when the Richardson step was wrong.
// discrete_exact extrapolates with (fine - coarse) / (ratio^2 - 1), and the
// textbook form uses 3 because doubling the node count is assumed to halve the
// spacing. It does not: the spacing is solved for from an integer node count so
// that barrier and spot both land on nodes, so the count goes 259 -> 517 and the
// ratio is 1.99614. Using 3 leaves a 0.5% error in the correction term, which is
// invisible to every tolerance-based price check in this file, the price is
// still right to six digits, and showed up only as the residual exponent in
// exp_correction bending away from 1.5.
//
// What it is not invisible to is grid refinement. With the true ratio the
// extrapolated value settles by a factor of ~16 per refinement; with 3 the
// remaining error is proportional to the correction term itself and settles far
// more slowly. Measured at m=100: 3.4e-08 between the 2^12 and 2^13 grids when
// correct, 1.99e-06 when not, a factor of 58, with the threshold below sitting
// comfortably between them.
static void test_extrapolation_is_grid_stable() {
  Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};
  const double a =
      discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 100, 1u << 12).price;
  const double b =
      discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 100, 1u << 13).price;
  CHECK_NEAR(b, a, 2e-7);
}

// Delta is read off the same grid as the price, so it needs its own control.
// With the barrier out of reach the answer must be the vanilla Black-Scholes
// delta, and that is a number this repo can produce without touching the grid.
static void test_delta_far_barrier_is_vanilla() {
  Params p{100.0, 100.0, 1e-8, 0.05, 0.0, 0.3, 0.2};
  const auto r = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 50);
  const double sT = p.sigma * std::sqrt(p.T);
  const double d1 =
      (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T) / sT;
  CHECK_NEAR(r.delta, std::exp(-p.q * p.T) * ncdf(d1), 1e-7);
  // and the closed-form delta helper must agree with the same limit
  CHECK_NEAR(barrier_cont_delta(Side::Call, Dir::Down, Knock::Out, p),
             std::exp(-p.q * p.T) * ncdf(d1), 1e-6);
  // A down-and-out call is longer delta than the vanilla near the barrier,
  // because knocking out removes exactly the paths that finish worthless.
  Params q{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};
  const auto rq = discrete_exact(Side::Call, Dir::Down, Knock::Out, q, 50);
  CHECK(rq.delta > std::exp(-q.q * q.T) * ncdf(d1));
}

// The one-sided stencil. When spot sits one node from the barrier, a central
// difference would read the half-weighted barrier node, so delta falls back to a
// one-sided second-order stencil pointing into the live side. None of the grids
// the experiments use get anywhere near that, which is precisely why the branch
// needs a test of its own: it is live code that nothing else executes.
//
// A barrier 50bp from spot on a 2^10 grid lands there. The check is that the
// fallback returns a delta, not that it returns the same delta, a one-sided
// stencil on a grid that coarse has real error, and 2% is the honest bound.
static void test_delta_one_sided_stencil() {
  Params p{100.0, 100.0, 99.5, 0.05, 0.0, 0.20, 1.0};
  GridInfo gc, gf;
  discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 10, &gc);
  CHECK_EQ((long long)std::labs(gc.k0), 1LL);
  CHECK(gc.one_sided);
  discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 16, &gf);
  CHECK(!gf.one_sided);
  const double d_coarse = gc.delta, d_fine = gf.delta;
  CHECK(d_coarse > 0.0);
  CHECK(std::fabs(d_coarse / d_fine - 1.0) < 0.02);
}

// The up direction of the one-sided stencil, which is the only place the
// direction factor in it does anything.
//
// This test exists because a mutant that deleted that factor survived the whole
// suite. For a down barrier the live side is upward, step is +1, and dropping
// the factor changes nothing, the mutant is a no-op on every case the rest of
// this file covers. It only bites upward, where it flips the sign of the delta.
// An up-and-out call must lose value as spot rises toward the barrier, so the
// sign is the assertion that matters here.
static void test_delta_one_sided_up_barrier() {
  Params p{100.0, 100.0, 100.5, 0.05, 0.0, 0.20, 1.0};
  GridInfo gc, gf;
  discrete_conv(Side::Call, Dir::Up, Knock::Out, p, 252, 1u << 10, &gc);
  CHECK_EQ((long long)gc.k0, -1LL);
  CHECK(gc.one_sided);
  discrete_conv(Side::Call, Dir::Up, Knock::Out, p, 252, 1u << 16, &gf);
  const double d_coarse = gc.delta, d_fine = gf.delta;
  CHECK(d_coarse < 0.0);
  CHECK(d_fine < 0.0);
  CHECK(std::fabs(d_coarse / d_fine - 1.0) < 0.20);
}

// Two ways the two-grid extrapolation can be asked to do something it cannot,
// both reachable only when the barrier is close enough to spot that the node
// count between them is tiny. Neither is reached by any experiment here, which
// is why both survived until someone walked the grid size down on purpose.
static void test_extrapolation_degenerate_grids() {
  Params p{100.0, 100.0, 99.5, 0.05, 0.0, 0.20, 1.0};

  // Same node count on both grids, so identical spacing, so ratio == 1. The
  // denominator ratio^2 - 1 is then zero and the extrapolation used to return
  // inf for a price that was otherwise perfectly good.
  GridInfo a, b;
  discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 9, &a);
  discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 10, &b);
  CHECK_EQ((long long)a.k0, (long long)b.k0);  // the degenerate case really is set up
  const auto deg = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 9);
  CHECK(std::isfinite(deg.price));
  CHECK(std::isfinite(deg.delta));
  CHECK(std::isfinite(deg.richardson));
  CHECK(deg.price > 1.0 && deg.price < 2.0);

  // Stencils differ between the grids: the coarse one falls back to the one-sided
  // difference, the fine one manages a central one. They have different error
  // constants, so extrapolating across the change is not valid, delta must come
  // back as the finer grid's value untouched.
  GridInfo c, d;
  discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 10, &c);
  discrete_conv(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 11, &d);
  CHECK(c.one_sided && !d.one_sided);
  const auto mixed = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 252, 1u << 10);
  CHECK_NEAR(mixed.delta, d.delta, 1e-14);
}

static void test_headline() {
  // The parameter set the README leads with: 50 monitoring dates over 0.2
  // years, which is daily monitoring of a ten-week option.
  Params p{100.0, 100.0, 95.0, 0.05, 0.0, 0.3, 0.2};
  const auto r = discrete_exact(Side::Call, Dir::Down, Knock::Out, p, 50, 1u << 13);
  CHECK_NEAR(r.price, 4.4777709, 1e-5);
  CHECK(std::fabs(r.richardson) < 1e-4);

  // Discrete monitoring can only ever miss a crossing that continuous
  // monitoring would have caught, so a discretely-monitored knock-out is worth
  // at least as much as the continuous one, at every m.
  const double cont = barrier_cont(Side::Call, Dir::Down, Knock::Out, p);
  double prev = 1e18;
  for (int m : {25, 50, 100, 200}) {
    const double d =
        discrete_conv(Side::Call, Dir::Down, Knock::Out, p, m, 1u << 12);
    CHECK(d > cont);
    CHECK(d < prev);  // and monotonically approaches it from above
    prev = d;
  }
}

static void test_up_barrier() {
  // The up direction is a separate branch: a different side of the grid is
  // zeroed and the spot index sits below the barrier index rather than above.
  Params p{100.0, 100.0, 110.0, 0.05, 0.0, 0.3, 0.2};
  CHECK_NEAR(discrete_exact(Side::Call, Dir::Up, Knock::Out, p, 1).price,
             1.223435231127975, 1e-7);
  const double cont = barrier_cont(Side::Call, Dir::Up, Knock::Out, p);
  const double d50 =
      discrete_conv(Side::Call, Dir::Up, Knock::Out, p, 50, 1u << 13);
  CHECK(d50 > cont);
  CHECK_NEAR(d50, 0.4761689, 1e-4);
}

static void test_discrete_parity() {
  // in + out = vanilla holds for the discrete contract too, at every m, because
  // the two events partition the same sample space under the same monitoring
  // schedule.
  for (double H : {90.0, 95.0, 99.0}) {
    Params p{100.0, 100.0, H, 0.04, 0.01, 0.28, 0.5};
    const double van = bs_vanilla(Side::Call, p.S, p.K, p.r, p.q, p.sigma, p.T);
    for (int m : {1, 5, 40}) {
      const double out =
          discrete_conv(Side::Call, Dir::Down, Knock::Out, p, m, 1u << 12);
      const double in =
          discrete_conv(Side::Call, Dir::Down, Knock::In, p, m, 1u << 12);
      CHECK_NEAR(out + in, van, 1e-10);
    }
  }
}

int main() {
  test_inert_barrier();
  test_against_quadrature();
  test_extrapolation_is_grid_stable();
  test_delta_far_barrier_is_vanilla();
  test_delta_one_sided_stencil();
  test_delta_one_sided_up_barrier();
  test_extrapolation_degenerate_grids();
  test_headline();
  test_up_barrier();
  test_discrete_parity();
  return check::finish("exact");
}
