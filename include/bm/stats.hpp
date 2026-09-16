#pragma once

// Accumulators and the least-squares fit used to report convergence exponents.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bm {

// Welford, with Chan's parallel merge so that a run split across threads gives
// the same variance to the last reported digit as a single-threaded one. (Not
// bit-identical: the merge order differs and floating point addition is not
// associative.)
//
// An honest note on why, because the obvious justification is one I checked and
// could not support. The textbook `sum(x^2) - n*mean^2` is usually condemned for
// cancelling catastrophically, and I expected to be able to show that here. I
// could not: measured at 2e6 paths on the coupled bias estimator, the naive
// formula agrees with Welford to 2e-13 relative, because these samples have a
// variance comparable to their mean square and so there is nothing much to
// cancel.
//
// Welford is used anyway, for a reason that survives measurement: its accuracy
// does not depend on that ratio. The bridge level estimator has a mean near 4
// and a variance near 67, but a deep out-of-the-money or near-barrier contract
// can easily have a mean-square thousands of times its variance, and at that
// point the naive formula loses digits exactly where a standard error is being
// quoted. Paying one extra multiply per sample to make the accuracy independent
// of the contract is worth it; claiming the naive form is already broken here
// would not have been true.
class Welford {
 public:
  void add(double x) {
    ++n_;
    const double d = x - mean_;
    mean_ += d / double(n_);
    m2_ += d * (x - mean_);
  }
  void merge(const Welford& o) {
    if (o.n_ == 0) return;
    if (n_ == 0) { *this = o; return; }
    const double na = double(n_), nb = double(o.n_), nt = na + nb;
    const double d = o.mean_ - mean_;
    mean_ += d * nb / nt;
    m2_ += o.m2_ + d * d * na * nb / nt;
    n_ += o.n_;
  }
  uint64_t count() const { return n_; }
  double mean() const { return mean_; }
  double var() const { return n_ > 1 ? m2_ / double(n_ - 1) : 0.0; }
  double stderr_() const { return n_ > 1 ? std::sqrt(var() / double(n_)) : 0.0; }

 private:
  uint64_t n_ = 0;
  double mean_ = 0.0;
  double m2_ = 0.0;
};

struct LineFit {
  double slope = 0.0;
  double intercept = 0.0;
  double slope_se = 0.0;
  double r2 = 0.0;
  size_t n = 0;
};

// Ordinary least squares of y on x, with the textbook standard error on the
// slope.
//
// Every convergence exponent this repo reports comes out of here, so two
// cautions belong next to it:
//
//   - The exponent is only meaningful over a range where the quantity being
//     fitted is dominated by the effect and not by the noise. Fitting log|bias|
//     against log(dt) down to a point where the bias has fallen below its own
//     standard error drags the slope toward zero and reports it with a
//     confident-looking standard error, because OLS has no way to know that the
//     smallest points are noise.
//   - The standard error below assumes independent residuals. The points in the
//     convergence table are not independent: they come from the same paths by
//     subsampling, which is deliberate and is what makes the slope stable. Use
//     it as a scale, not as a confidence interval, and prefer the agreement of
//     successive ratios (which the experiments also print) as the real evidence.
inline LineFit fit_line(const std::vector<double>& x,
                        const std::vector<double>& y) {
  LineFit f;
  const size_t n = x.size();
  if (n < 2 || y.size() != n) return f;
  f.n = n;
  double sx = 0, sy = 0;
  for (size_t i = 0; i < n; ++i) { sx += x[i]; sy += y[i]; }
  const double mx = sx / double(n), my = sy / double(n);
  double sxx = 0, sxy = 0, syy = 0;
  for (size_t i = 0; i < n; ++i) {
    sxx += (x[i] - mx) * (x[i] - mx);
    sxy += (x[i] - mx) * (y[i] - my);
    syy += (y[i] - my) * (y[i] - my);
  }
  if (sxx <= 0.0) return f;
  f.slope = sxy / sxx;
  f.intercept = my - f.slope * mx;
  double sse = 0;
  for (size_t i = 0; i < n; ++i) {
    const double e = y[i] - (f.intercept + f.slope * x[i]);
    sse += e * e;
  }
  f.r2 = (syy > 0.0) ? 1.0 - sse / syy : 1.0;
  if (n > 2) f.slope_se = std::sqrt(sse / double(n - 2) / sxx);
  return f;
}

}  // namespace bm
