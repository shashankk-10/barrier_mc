// Foundations: the normal functions, the counter-based generator, the FFT.
//
// Every assertion here compares against something produced a different way:
// scipy for the normal values, an independently written implementation of the
// Philox round function for the generator, a direct O(n^2) DFT for the transform.
// Checking a function against itself would pass whatever the function did.

#include <cmath>
#include <complex>
#include <set>
#include <vector>

#include "bm/fft.hpp"
#include "bm/normal.hpp"
#include "bm/rng.hpp"
#include "check.hpp"

using namespace bm;

static void test_normal() {
  // Reference values from scipy.stats.norm, printed at 17 significant digits.
  CHECK_NEAR(ncdf(0.0), 0.5, 0.0);
  CHECK_REL(ncdf(1.0), 0.84134474606854293, 1e-15);
  CHECK_REL(ncdf(-1.0), 0.15865525393145707, 1e-15);
  CHECK_REL(ncdf(2.5), 0.99379033467422384, 1e-15);

  // The far left tail is the reason ncdf is written with erfc. A 1+erf form
  // would have lost every significant digit to cancellation by here, while
  // still returning a small positive number that looks fine.
  CHECK_REL(ncdf(-5.0) / 2.8665157187919344e-07, 1.0, 1e-13);
  CHECK_REL(ncdf(-10.0) / 7.619853024160474e-24, 1.0, 1e-12);

  CHECK_NEAR(ninv(0.5), 0.0, 0.0);
  CHECK_REL(ninv(0.975), 1.959963984540054, 1e-14);
  CHECK_REL(ninv(0.025), -1.9599639845400545, 1e-14);
  CHECK_REL(ninv(0.9), 1.2815515655446004, 1e-14);
  CHECK_REL(ninv(1e-10), -6.3613409024040557, 1e-13);
  CHECK_REL(ninv(1e-300), -37.047096299361201, 1e-12);

  // Round trip across the range the sampler can actually reach.
  for (int i = 1; i < 2000; ++i) {
    const double u = double(i) / 2000.0;
    CHECK_REL(ncdf(ninv(u)), u, 1e-12);
  }
  CHECK(std::isinf(ninv(0.0)) && ninv(0.0) < 0);
  CHECK(std::isinf(ninv(1.0)) && ninv(1.0) > 0);
}

static void test_philox() {
  // Known-answer vectors. These were cross-checked against a separate
  // implementation written from the round function in the Random123 paper; the
  // third is the published vector, whose counter and key are the hex digits of
  // pi and e -- a wrong round function does not reproduce those sixteen hex
  // digits by accident.
  auto out1 = philox4x32_10({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
                            {0xffffffffu, 0xffffffffu});
  CHECK_EQ(out1[0], 0x408f276du);
  CHECK_EQ(out1[1], 0x41c83b0eu);
  CHECK_EQ(out1[2], 0xa20bc7c6u);
  CHECK_EQ(out1[3], 0x6d5451fdu);

  auto out2 = philox4x32_10({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u},
                            {0xa4093822u, 0x299f31d0u});
  CHECK_EQ(out2[0], 0xd16cfe09u);
  CHECK_EQ(out2[1], 0x94fdccebu);
  CHECK_EQ(out2[2], 0x5001e420u);
  CHECK_EQ(out2[3], 0x24126ea1u);

  // The uniforms must never reach an endpoint: ninv(0) is -inf, and a single
  // infinite Brownian increment turns a whole run into NaN long after the draw
  // that caused it.
  //
  // The extremes are checked directly instead of hoped for. An earlier version
  // packed 53 bits, and the all-ones input then produced (2^53 - 0.5), which is
  // exactly halfway between two doubles and rounds UP to 2^53 -- returning
  // exactly 1.0, and an infinite normal, from a line whose comment asserted that
  // could not happen. These assertions catch it; the 200000-draw sweep below
  // does not, because it will essentially never hit an all-ones word.
  CHECK(u01(0xFFFFFFFFu, 0xFFFFFFFFu) < 1.0);
  CHECK(u01(0xFFFFFFFFu, 0xFFFFFFFFu) > 0.0);
  CHECK(u01(0u, 0u) > 0.0);
  CHECK(u01(0u, 0u) < 1.0);
  CHECK(std::isfinite(ninv(u01(0xFFFFFFFFu, 0xFFFFFFFFu))));
  CHECK(std::isfinite(ninv(u01(0u, 0u))));

  double lo = 1.0, hi = 0.0, sum = 0.0;
  const int kN = 200000;
  for (int i = 0; i < kN; ++i) {
    Stream s(0xC0FFEEu, uint64_t(i));
    for (int d = 0; d < 4; ++d) {
      const double u = s.uniform(uint64_t(d));
      CHECK(u > 0.0 && u < 1.0);
      lo = std::fmin(lo, u);
      hi = std::fmax(hi, u);
      sum += u;
    }
  }
  CHECK_NEAR(sum / (4.0 * kN), 0.5, 0.005);
  CHECK(lo < 1e-4 && hi > 1.0 - 1e-4);

  // The property the nested-grid coupling depends on: a path's d-th normal is a
  // pure function of (path, d). Draw them out of order, interleaved with other
  // paths, and they must be identical.
  std::vector<double> fwd(64), rev(64);
  {
    Stream s(7, 12345);
    for (int d = 0; d < 64; ++d) fwd[size_t(d)] = s.gauss(uint64_t(d));
  }
  for (int d = 63; d >= 0; --d) {
    Stream s(7, 12345);
    Stream other(7, 999);
    (void)other.gauss(3);
    rev[size_t(d)] = s.gauss(uint64_t(d));
  }
  for (int d = 0; d < 64; ++d) CHECK_NEAR(fwd[size_t(d)], rev[size_t(d)], 0.0);

  // Antithetic partner is the exact negation, because ninv is odd.
  Stream s(11, 4);
  for (int d = 0; d < 32; ++d)
    CHECK_NEAR(s.gauss_anti(uint64_t(d)), -s.gauss(uint64_t(d)), 1e-13);

  // Different paths must not collide. 20000 paths x 1 dimension, all distinct.
  std::set<double> seen;
  for (int i = 0; i < 20000; ++i) {
    Stream t(42, uint64_t(i));
    seen.insert(t.uniform(0));
  }
  CHECK_EQ((long long)seen.size(), 20000LL);

  // Moments of the normals, as a coarse guard against an inversion that is
  // subtly wrong in the tails.
  double m1 = 0, m2 = 0, m4 = 0;
  const int kM = 400000;
  for (int i = 0; i < kM; ++i) {
    Stream t(99, uint64_t(i));
    const double z = t.gauss(0);
    m1 += z;
    m2 += z * z;
    m4 += z * z * z * z;
  }
  CHECK_NEAR(m1 / kM, 0.0, 0.01);
  CHECK_NEAR(m2 / kM, 1.0, 0.01);
  CHECK_NEAR(m4 / kM, 3.0, 0.08);
}

static void test_fft() {
  // Direct DFT, O(n^2), as the reference the fast one has to match.
  auto direct = [](const std::vector<double>& x, size_t k) {
    cpx acc(0.0, 0.0);
    const size_t n = x.size();
    for (size_t j = 0; j < n; ++j) {
      const double a = -2.0 * M_PI * double(k) * double(j) / double(n);
      acc += x[j] * cpx(std::cos(a), std::sin(a));
    }
    return acc;
  };

  for (size_t n : {8u, 16u, 64u, 256u}) {
    std::vector<double> x(n);
    for (size_t j = 0; j < n; ++j)
      x[j] = std::sin(0.7 * double(j)) + 0.3 * double(j % 5) - 1.1;
    RealFFT f(n);
    std::vector<cpx> X(f.spectrum_size());
    f.forward(x.data(), n, X.data());
    for (size_t k = 0; k < X.size(); ++k) {
      const cpx d = direct(x, k);
      CHECK_NEAR(X[k].real(), d.real(), 1e-10);
      CHECK_NEAR(X[k].imag(), d.imag(), 1e-10);
    }
    std::vector<double> back(n);
    f.inverse(X.data(), back.data());
    for (size_t j = 0; j < n; ++j) CHECK_NEAR(back[j], x[j], 1e-12);
  }

  // Zero padding: a length-`len` signal transformed at length n must agree with
  // the same signal explicitly zero-extended. exact.cpp relies on this to get a
  // LINEAR convolution out of a cyclic transform, and getting it wrong wraps the
  // top of the grid onto the bottom -- which in a barrier lattice means the
  // deep-in-the-money payoff leaks into the knocked-out region.
  {
    const size_t n = 64, len = 23;
    std::vector<double> shortv(len), padded(n, 0.0);
    for (size_t j = 0; j < len; ++j) {
      shortv[j] = std::cos(1.3 * double(j)) + 0.2;
      padded[j] = shortv[j];
    }
    RealFFT f(n);
    std::vector<cpx> A(f.spectrum_size()), B(f.spectrum_size());
    f.forward(shortv.data(), len, A.data());
    f.forward(padded.data(), n, B.data());
    for (size_t k = 0; k < A.size(); ++k) {
      CHECK_NEAR(A[k].real(), B[k].real(), 1e-12);
      CHECK_NEAR(A[k].imag(), B[k].imag(), 1e-12);
    }
  }

  // A full linear convolution against a naive O(n*m) reference.
  {
    const size_t na = 40, nb = 27, full = na + nb - 1;
    size_t n = 1;
    while (n < full) n <<= 1;
    std::vector<double> a(na), b(nb), want(full, 0.0);
    for (size_t i = 0; i < na; ++i) a[i] = std::sin(0.4 * double(i)) + 1.0;
    for (size_t i = 0; i < nb; ++i) b[i] = std::exp(-0.1 * double(i));
    for (size_t i = 0; i < na; ++i)
      for (size_t j = 0; j < nb; ++j) want[i + j] += a[i] * b[j];
    RealFFT f(n);
    std::vector<cpx> A(f.spectrum_size()), B(f.spectrum_size());
    f.forward(a.data(), na, A.data());
    f.forward(b.data(), nb, B.data());
    for (size_t k = 0; k < A.size(); ++k) A[k] *= B[k];
    std::vector<double> got(n);
    f.inverse(A.data(), got.data());
    for (size_t i = 0; i < full; ++i) CHECK_NEAR(got[i], want[i], 1e-10);
  }
}

int main() {
  test_normal();
  test_philox();
  test_fft();
  return check::finish("numerics");
}
