// Where 0.5826 comes from.
//
// Three steps, each checked separately: Spitzer's identity, Gaussian steps, and
// Euler-Maclaurin. Derivation in DETAILS.md.
//
// Note the route not taken. Simulating a walk until it crosses a level never
// finishes: a driftless random walk has infinite expected first-passage time.

#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "bm/bgk.hpp"
#include "bm/rng.hpp"
#include "bm/stats.hpp"
#include "harness.hpp"

using namespace bm;

namespace {

constexpr double kZetaHalf = -1.4603545088095868;  // zeta(1/2)

// sum_{k=1..n} k^{-1/2}, Kahan-compensated.
//
// This matters at the sizes below. At n = 1e8 the terms have fallen to 1e-4
// while the running total is 2e4, so naive accumulation loses about
// n * eps * total ~ 4e-4, which is larger than the entire quantity being
// extracted (the constant sits at the 1e-5 level against 2*sqrt(n)). The naive
// sum is printed alongside so the difference is visible instead of asserted.
double sum_inv_sqrt_kahan(long n) {
  double s = 0.0, c = 0.0;
  for (long k = 1; k <= n; ++k) {
    const double y = 1.0 / std::sqrt(double(k)) - c;
    const double t = s + y;
    c = (t - s) - y;
    s = t;
  }
  return s;
}

double sum_inv_sqrt_naive(long n) {
  double s = 0.0;
  for (long k = 1; k <= n; ++k) s += 1.0 / std::sqrt(double(k));
  return s;
}

// E[max_{k<=n} S_k] by simulation, standard normal steps.
Welford mc_expected_max(int n, uint64_t paths, uint64_t seed, int threads) {
  std::vector<Welford> per(static_cast<size_t>(threads));
  auto worker = [&](int tid) {
    Welford w;
    for (uint64_t i = uint64_t(tid); i < paths; i += uint64_t(threads)) {
      Stream st(seed, i);
      double cur = 0.0, mx = 0.0;
      for (int k = 0; k < n; ++k) {
        cur += st.gauss(uint64_t(k));
        if (cur > mx) mx = cur;
      }
      w.add(mx);
    }
    per[static_cast<size_t>(tid)] = w;
  };
  std::vector<std::thread> ts;
  ts.reserve(static_cast<size_t>(threads));
  for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t);
  for (auto& t : ts) t.join();
  Welford all;
  for (auto& w : per) all.merge(w);
  return all;
}

}  // namespace

int main() {
  bmx::rule("exp_overshoot: recovering beta from Spitzer's identity");
  const int threads = int(std::thread::hardware_concurrency());
  printf("beta as used by bgk.hpp : %.17g\n", kBeta);
  printf("zeta(1/2)               : %.17g\n", kZetaHalf);
  printf("-zeta(1/2)/sqrt(2 pi)   : %.17g\n",
         -kZetaHalf / std::sqrt(2.0 * M_PI));

  // --- step 3: the Euler-Maclaurin constant IS zeta(1/2) --------------------
  printf(
      "\n[1] sum_{k=1..n} k^{-1/2} - 2 sqrt(n)  ->  zeta(1/2)\n"
      "    and the next term, (sum - 2 sqrt(n) - zeta(1/2)) * 2 sqrt(n) -> 1\n\n"
      "          n |  kahan sum - 2sqrt(n) |   naive - kahan |  next-term check\n"
      "------------+-----------------------+-----------------+------------------\n");
  for (long n : {100L, 10000L, 1000000L, 100000000L}) {
    const double sk = sum_inv_sqrt_kahan(n);
    const double sn = sum_inv_sqrt_naive(n);
    const double c = sk - 2.0 * std::sqrt(double(n));
    printf("%11ld | %+21.12f | %+15.2e | %16.6f\n", n, c, sn - sk,
           (c - kZetaHalf) * 2.0 * std::sqrt(double(n)));
  }
  printf("%11s | %+21.12f |\n", "limit", kZetaHalf);

  // --- steps 2+3: therefore E[M_n] - sqrt(2n/pi) -> -beta -------------------
  printf(
      "\n[2] E[M_n] = (1/sqrt(2 pi)) sum k^{-1/2}   (Spitzer + Gaussian steps)\n"
      "    E[M_n] - sqrt(2n/pi)  ->  -beta\n\n"
      "          n |     E[M_n] exact  |  E[M_n] - sqrt(2n/pi) |  recovered beta\n"
      "------------+-------------------+-----------------------+-----------------\n");
  for (long n : {100L, 10000L, 1000000L, 100000000L}) {
    const double em = sum_inv_sqrt_kahan(n) / std::sqrt(2.0 * M_PI);
    const double d = em - std::sqrt(2.0 * double(n) / M_PI);
    // Undo the known (1/2) n^{-1/2} term so what is left is beta itself.
    // d = -beta + (1/2) n^{-1/2} / sqrt(2 pi), so beta = -d + that term: the
    // correction is added. Subtracting it instead moves the answer by twice the
    // term, which at n = 1e8 is the difference between 0.5825573 and the exact
    // 0.5825972, still a plausible-looking table, wrong in the 5th digit.
    const double recovered =
        -d + 0.5 / (std::sqrt(2.0 * M_PI) * std::sqrt(double(n)));
    printf("%11ld | %17.9f | %+21.12f | %15.10f\n", n, em, d, recovered);
  }
  printf("%11s |                   | %+21.12f | %15.10f\n", "limit", -kBeta,
         kBeta);

  // --- step 1: is Spitzer's identity actually true? -------------------------
  printf(
      "\n[3] Spitzer's identity vs brute-force Monte Carlo.\n"
      "    Everything above rests on this, so it is checked rather than cited.\n\n"
      "     n |   MC E[M_n]  (s.e.)   |  Spitzer exact  | difference\n"
      "-------+-----------------------+-----------------+------------\n");
  const uint64_t paths = 4000000;
  for (int n : {1, 2, 5, 20, 100}) {
    const Welford w = mc_expected_max(n, paths, 0xBE7A, threads);
    const double exact = sum_inv_sqrt_kahan(n) / std::sqrt(2.0 * M_PI);
    printf("%6d | %10.6f (%.6f) | %15.9f | %+6.2f s.e.\n", n, w.mean(),
           w.stderr_(), exact, (w.mean() - exact) / w.stderr_());
  }
  printf("   (%llu paths per row)\n", (unsigned long long)paths);

  // --- beta from simulation alone, with no Spitzer -------------------------
  printf(
      "\n[4] beta from simulation alone, using no identity at all:\n"
      "    beta_hat = sqrt(2n/pi) - E[M_n]_MC + (1/2)/sqrt(2 pi n)\n\n"
      "     n |   beta_hat  |    s.e.   | deviation from beta\n"
      "-------+-------------+-----------+--------------------\n");
  for (int n : {25, 100, 400}) {
    const Welford w = mc_expected_max(n, paths, 0xB37A, threads);
    const double bh = std::sqrt(2.0 * double(n) / M_PI) - w.mean() +
                      0.5 / (std::sqrt(2.0 * M_PI) * std::sqrt(double(n)));
    printf("%6d | %11.6f | %9.6f | %+.2f s.e.\n", n, bh, w.stderr_(),
           (bh - kBeta) / w.stderr_());
  }
  printf(
      "\n    The standard error grows like sqrt(n) while the finite-n bias\n"
      "    shrinks like n^{-3/2}, so there is a floor on how well this route can\n"
      "    do on a laptop. That is precisely why bgk.hpp takes beta from the\n"
      "    zeta value and this experiment only has to confirm it.\n");
  return 0;
}
