#pragma once

// Shared plumbing for the experiments: a wall clock, a rule, and the parameter
// set the README leads with.

#include <chrono>
#include <cstdio>
#include <string>

#include "bm/bs.hpp"

namespace bmx {

// The headline contract. A ten-week down-and-out call struck at the money with
// the barrier 5% below spot -- chosen because 50 monitoring dates over 0.2
// years is daily monitoring, which is what an actual contract specifies, and
// because a 5% barrier is close enough to matter and far enough that the
// asymptotic regime is reachable within a laptop's patience.
inline bm::Params headline() {
  return bm::Params{100.0, 100.0, 95.0, 0.05, 0.0, 0.30, 0.2};
}

inline void rule(const char* title) {
  printf("\n=== %s ", title);
  for (size_t i = std::string(title).size(); i < 66; ++i) putchar('=');
  putchar('\n');
}

inline void describe(const bm::Params& p, const char* contract) {
  printf("contract : %s\n", contract);
  printf("params   : S=%g K=%g H=%g r=%g q=%g sigma=%g T=%g\n", p.S, p.K, p.H,
         p.r, p.q, p.sigma, p.T);
}

class Timer {
 public:
  Timer() : t0_(std::chrono::steady_clock::now()) {}
  double s() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace bmx
