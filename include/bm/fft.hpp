#pragma once

// A radix-2 FFT, and a real-input wrapper built on a half-length complex
// transform.
//
// This is here instead of pulled from FFTW or Accelerate because the repo is
// dependency-free on purpose: the point of exact.hpp is that a reader can build
// it and reproduce the benchmark table, and a table that depends on which
// vendor library happened to be installed is worth less than one that does not.
// It is also ~150 lines, which is cheaper than the build-system argument.
//
// The real wrapper earns its keep: exact.hpp runs one forward and one inverse
// transform per monitoring date, and the largest table entry has 1600 of them on
// two grids. Packing a length-N real signal into a length-N/2 complex transform
// halves the time and, more to the point on an 8 GB machine, the working set.

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace bm {

using cpx = std::complex<double>;

inline bool is_pow2(size_t n) { return n && (n & (n - 1)) == 0; }

// Iterative Cooley-Tukey, decimation in time, in place.
// `inverse` applies the conjugate twiddles and does not divide by n; the
// callers that need the 1/n do it themselves, so that the real wrapper can fold
// it into a scaling it was doing anyway.
inline void fft_inplace(cpx* a, size_t n, bool inverse) {
  assert(is_pow2(n));
  for (size_t i = 1, j = 0; i < n; ++i) {  // bit-reversal permutation
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = 2.0 * M_PI / double(len) * (inverse ? 1.0 : -1.0);
    const cpx wl(std::cos(ang), std::sin(ang));
    for (size_t i = 0; i < n; i += len) {
      cpx w(1.0, 0.0);
      for (size_t k = 0; k < len / 2; ++k) {
        // Re-seed the twiddle from cos/sin every 64 steps. Advancing it purely
        // by recurrence (w *= wl) is the textbook form and it drifts: the
        // relative error grows linearly in the transform length, measured
        // against a long-double direct DFT at 3.9e-13 for L=2^10 rising to
        // 1.4e-11 at L=2^17, roughly doubling per doubling. That is three orders
        // worse than the ~1e-15 this should be, and exact.cpp runs thousands of
        // these transforms in sequence. Re-seeding bounds the accumulated drift
        // to 64 multiplies for one cos/sin per 64 butterflies.
        if ((k & 63u) == 0) {
          const double a2 = ang * double(k);
          w = cpx(std::cos(a2), std::sin(a2));
        }
        const cpx u = a[i + k];
        const cpx v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}

// Real-to-complex and back, for a fixed transform length.
//
// Forward: x[0..n-1] real  ->  X[0..n/2] (the non-redundant half of a Hermitian
// spectrum). Inverse: X[0..n/2] -> x[0..n-1] real, already divided by n.
class RealFFT {
 public:
  explicit RealFFT(size_t n) : n_(n), h_(n / 2), z_(n / 2), tw_(n / 2 + 1) {
    assert(is_pow2(n) && n >= 4);
    for (size_t k = 0; k <= h_; ++k) {
      const double ang = -2.0 * M_PI * double(k) / double(n_);
      tw_[k] = cpx(std::cos(ang), std::sin(ang));
    }
  }

  size_t size() const { return n_; }
  size_t spectrum_size() const { return h_ + 1; }

  // Zero-pads if `len` < n, which is how the linear convolution in exact.hpp
  // avoids the circular wrap-around that a same-length transform would give it.
  void forward(const double* x, size_t len, cpx* X) {
    assert(len <= n_);
    for (size_t j = 0; j < h_; ++j) {
      const double re = (2 * j < len) ? x[2 * j] : 0.0;
      const double im = (2 * j + 1 < len) ? x[2 * j + 1] : 0.0;
      z_[j] = cpx(re, im);
    }
    fft_inplace(z_.data(), h_, false);
    // Split the interleaved transform back into the even/odd sub-transforms and
    // recombine. Z[h] is Z[0] by periodicity, which is why the k==0 and k==h
    // ends work without a special case beyond the index wrap.
    for (size_t k = 0; k <= h_; ++k) {
      const cpx zk = z_[k % h_];
      const cpx zc = std::conj(z_[(h_ - k) % h_]);
      const cpx fe = 0.5 * (zk + zc);
      const cpx fo = cpx(0.0, -0.5) * (zk - zc);
      X[k] = fe + tw_[k] * fo;
    }
  }

  void inverse(const cpx* X, double* x) {
    for (size_t k = 0; k < h_; ++k) {
      const cpx xk = X[k];
      const cpx xc = std::conj(X[h_ - k]);
      const cpx fe = 0.5 * (xk + xc);
      const cpx fo = 0.5 * std::conj(tw_[k]) * (xk - xc);
      z_[k] = fe + cpx(0.0, 1.0) * fo;
    }
    fft_inplace(z_.data(), h_, true);
    const double s = 1.0 / double(h_);
    for (size_t j = 0; j < h_; ++j) {
      x[2 * j] = z_[j].real() * s;
      x[2 * j + 1] = z_[j].imag() * s;
    }
  }

 private:
  size_t n_, h_;
  std::vector<cpx> z_;
  std::vector<cpx> tw_;
};

}  // namespace bm
