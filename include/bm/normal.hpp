#pragma once

// Standard normal density, distribution and quantile.
//
// Two choices here are load-bearing for the rest of the project and are worth
// stating instead of burying:
//
// 1. The CDF goes through std::erfc, not through a polynomial approximation.
//    The barrier formulas in bs.hpp evaluate Phi at arguments that run to the
//    far left tail when the barrier is close to spot, and a rational
//    approximation good to 1e-7 absolutely is good to nothing at all
//    relatively once Phi(x) itself is 1e-9. erfc is the function that keeps its
//    relative accuracy out there, and it is the reason cdf() is written as
//    0.5 * erfc(-x / sqrt(2)) rather than the more familiar 0.5 * (1 + erf(...)):
//    the erf form computes 1 + (something that is nearly -1) and loses every
//    significant digit in the left tail to cancellation.
//
// 2. The quantile is Wichura's AS241 (PPND16), accurate to about 1e-16 relative
//    across the whole range, instead of Box-Muller or a Ziggurat. Normals in
//    this project are produced by inverting a uniform, never by transforming a
//    pair of them, because inversion maps one uniform to one normal. Box-Muller
//    consumes two uniforms and returns two normals through a rotation, which
//    destroys the correspondence between a stream position and a Brownian
//    increment -- and rng.hpp's whole design is that a path's d-th normal is a
//    pure function of (path, d), whatever the thread count or the order of
//    consumption. The nested-grid coupling in mc.cpp depends on that.

#include <cmath>

namespace bm {

inline constexpr double kSqrt2 = 1.41421356237309504880168872420969808;
inline constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934381868;

inline double npdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

inline double ncdf(double x) { return 0.5 * std::erfc(-x / kSqrt2); }

// log(Phi(x)), which stays finite where Phi(x) itself has underflowed to zero.
//
// bs.cpp needs this because the barrier formulas multiply (H/S)^{2mu} by a normal
// CDF, and for small sigma those two factors go to infinity and to zero at the
// same time. Forming each separately overflows one and underflows the other and
// the product comes back 0 or NaN; forming the sum of their logarithms does not.
// Below the point where Phi underflows (around x = -37) fall back to the standard
// asymptotic expansion Phi(x) ~ phi(x)/(-x) * (1 - 1/x^2 + 3/x^4 - ...).
inline double log_ncdf(double x) {
  if (x > -36.0) return std::log(ncdf(x));
  const double ix2 = 1.0 / (x * x);
  return -0.5 * x * x - 0.5 * std::log(2.0 * M_PI) - std::log(-x) +
         std::log1p(ix2 * (-1.0 + ix2 * (3.0 - 15.0 * ix2)));
}

// Wichura (1988), Algorithm AS 241, PPND16. Relative accuracy ~1e-16.
inline double ninv(double p) {
  if (!(p > 0.0) || !(p < 1.0)) {
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return INFINITY;
    return NAN;  // p is NaN
  }
  const double q = p - 0.5;
  double r;
  if (std::fabs(q) <= 0.425) {
    r = 0.180625 - q * q;
    return q *
           (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r +
                 6.7265770927008700853e+4) *
                    r +
                4.5921953931549871457e+4) *
                   r +
               1.3731693765509461125e+4) *
                  r +
              1.9715909503065514427e+3) *
                 r +
             1.3314166789178437745e+2) *
                r +
            3.3871328727963666080e+0) /
           (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r +
                 3.9307895800092710610e+4) *
                    r +
                2.1213794301586595867e+4) *
                   r +
               5.3941960214247511077e+3) *
                  r +
              6.8718700749205790830e+2) *
                 r +
             4.2313330701600911252e+1) *
                r +
            1.0);
  }
  r = (q < 0.0) ? p : 1.0 - p;
  r = std::sqrt(-std::log(r));
  double val;
  if (r <= 5.0) {
    r -= 1.6;
    val =
        (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r +
              2.41780725177450611770e-1) *
                 r +
             1.27045825245236838258e+0) *
                r +
            3.64784832476320460504e+0) *
               r +
           5.76949722146069140550e+0) *
              r +
          4.63033784615654529590e+0) *
             r +
         1.42343711074968357734e+0) /
        (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r +
              1.51986665636164571966e-2) *
                 r +
             1.48103976427480074590e-1) *
                r +
            6.89767334985100004550e-1) *
               r +
           1.67638483018380384940e+0) *
              r +
          2.05319162663775882187e+0) *
             r +
         1.0);
  } else {
    r -= 5.0;
    val =
        (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
              1.24266094738807843860e-3) *
                 r +
             2.65321895265761230930e-2) *
                r +
            2.96560571828504891230e-1) *
               r +
           1.78482653991729133580e+0) *
              r +
          5.46378491116411436990e+0) *
             r +
         6.65790464350110377720e+0) /
        (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r +
              1.84631831751005468180e-5) *
                 r +
             7.86869131145613259100e-4) *
                r +
            1.48753612908506148525e-2) *
               r +
           1.36929880922735805310e-1) *
              r +
          5.99832206555887937690e-1) *
             r +
         1.0);
  }
  return (q < 0.0) ? -val : val;
}

}  // namespace bm
