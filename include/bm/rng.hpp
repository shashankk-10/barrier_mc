#pragma once

// Philox4x32-10, a counter-based generator (Salmon et al., SC'11 -- the
// Random123 paper), plus normals by inversion.
//
// Why counter-based, when std::mt19937_64 is right there.
//
// A conventional generator has state that advances as you draw from it, so the
// normal a path receives depends on how many draws happened before it. That is
// fine until the experiment needs any of the three things this project needs:
//
//   1. Nested grids. exp_convergence prices the same option at m = 25, 50, 100,
//      ... and fits a slope through the results. Those estimates must be
//      positively correlated across m or the fitted exponent is mostly noise.
//      The way to correlate them is to build the m-grid path and the 2m-grid
//      path from the same underlying Brownian motion, refining by bisection --
//      which consumes dimensions out of order. A sequential generator cannot do
//      that; a counter-based one indexes (path, dimension) directly.
//
//   2. Threading. A path's randomness is a pure function of its index, so worker
//      t taking paths t, t+T, t+2T... gives every path exactly the value it
//      would have had single-threaded. The limit of that claim is worth stating:
//      the per-path values are bit-identical, the accumulated mean is not,
//      because floating point addition is not associative and a different thread
//      count sums them in a different order. test_mc asserts agreement to 1e-10,
//      which is what is actually true; bit-equality is not.
//
//   3. Auditability. A disagreement between two runs is a disagreement about one
//      named path, which can be replayed on its own without reproducing the
//      whole run that contained it.
//
// The cost is real: ~10 rounds of integer multiply-xor per counter, against a
// handful of operations for a Mersenne twister. It buys reproducibility that
// the experiments would otherwise have to assume rather than guarantee.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bm/normal.hpp"

namespace bm {

// The 4x32-10 bijection: 10 rounds, 4 words of counter, 2 words of key.
inline std::array<uint32_t, 4> philox4x32_10(std::array<uint32_t, 4> c,
                                             std::array<uint32_t, 2> k) {
  constexpr uint32_t kM0 = 0xD2511F53u;
  constexpr uint32_t kM1 = 0xCD9E8D57u;
  constexpr uint32_t kW0 = 0x9E3779B9u;  // round keys are bumped by the golden
  constexpr uint32_t kW1 = 0xBB67AE85u;  // ratio and sqrt(3)-1 fractions
  for (int r = 0; r < 10; ++r) {
    if (r != 0) {
      k[0] += kW0;
      k[1] += kW1;
    }
    const uint64_t p0 = uint64_t(kM0) * uint64_t(c[0]);
    const uint64_t p1 = uint64_t(kM1) * uint64_t(c[2]);
    const uint32_t hi0 = uint32_t(p0 >> 32), lo0 = uint32_t(p0);
    const uint32_t hi1 = uint32_t(p1 >> 32), lo1 = uint32_t(p1);
    c = {uint32_t(hi1 ^ c[1] ^ k[0]), lo1, uint32_t(hi0 ^ c[3] ^ k[1]), lo0};
  }
  return c;
}

// Two 32-bit words to a double in the open interval (0,1).
//
// ninv(0) is -inf and ninv(1) is +inf, so a generator that can reach an endpoint
// produces an infinite Brownian increment, and that surfaces as a NaN price long
// after the draw that caused it.
//
// 52 bits, not 53, and the reason is a rounding trap. With 53 bits the largest
// value is bits = 2^53 - 1, and (double)(2^53 - 1) + 0.5 = 2^53 - 0.5 is exactly
// halfway between two representable doubles at that binade; round-half-to-even
// takes it UP to 2^53, so the product is exactly 1.0 and the "half a ulp of
// headroom" argument is wrong. At 52 bits the largest value is 2^52 - 0.5, which
// IS representable (the spacing below 2^52 is 0.5), giving u_max = 1 - 2^-53 and
// u_min = 2^-53. Both ends are then closed by construction instead of by an
// argument that does not survive the rounding mode. The cost is one bit of
// resolution, which is nothing next to a NaN.
inline double u01(uint32_t hi, uint32_t lo) {
  const uint64_t bits = (uint64_t(hi >> 6) << 26) | uint64_t(lo >> 6);  // 52
  return (double(bits) + 0.5) * (1.0 / 4503599627370496.0);             // 2^-52
}

// A path's randomness, addressed by (path, dimension). Stateless in the sense
// that matters: gauss(p, d) is the same number in every run, on every thread,
// in any order, regardless of what else has been drawn.
class Stream {
 public:
  Stream(uint64_t seed, uint64_t path)
      : key_{uint32_t(seed), uint32_t(seed >> 32)},
        ctr_hi_{uint32_t(path), uint32_t(path >> 32)} {}

  double uniform(uint64_t dim) {
    refill(dim >> 1);
    const size_t o = (dim & 1u) ? 2 : 0;
    return u01(buf_[o], buf_[o + 1]);
  }

  double gauss(uint64_t dim) { return ninv(uniform(dim)); }

  // Antithetic partner: reflecting the uniform about 1/2 negates the normal
  // exactly, because ninv is odd. Doing it on the uniform rather than negating
  // the normal keeps the two branches on one code path.
  double gauss_anti(uint64_t dim) { return ninv(1.0 - uniform(dim)); }

  void fill_gauss(double* out, uint64_t dim0, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = gauss(dim0 + i);
  }

 private:
  void refill(uint64_t block) {
    if (block == have_ && primed_) return;
    buf_ = philox4x32_10(
        {uint32_t(block), uint32_t(block >> 32), ctr_hi_[0], ctr_hi_[1]}, key_);
    have_ = block;
    primed_ = true;
  }

  std::array<uint32_t, 2> key_;
  std::array<uint32_t, 2> ctr_hi_;
  std::array<uint32_t, 4> buf_{};
  uint64_t have_ = 0;
  bool primed_ = false;
};

}  // namespace bm
