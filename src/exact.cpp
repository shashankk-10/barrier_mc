#include "bm/exact.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "bm/fft.hpp"
#include "bm/normal.hpp"

namespace bm {
namespace {

// Vanilla Black-Scholes delta. Needed in two places below: the no-monitoring
// case, and the knock-in parity (in + out = vanilla differentiates term by term).
double vanilla_delta(Side side, const Params& p) {
  if (p.T <= 0.0 || p.sigma <= 0.0) return 0.0;
  const double sT = p.sigma * std::sqrt(p.T);
  const double d1 =
      (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T) / sT;
  const double dq = std::exp(-p.q * p.T);
  return (side == Side::Call) ? dq * ncdf(d1) : -dq * ncdf(-d1);
}

size_t next_pow2(size_t v) {
  size_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

}  // namespace

double discrete_conv(Side side, Dir dir, Knock knock, const Params& p, int m,
                     size_t n, GridInfo* grid) {
  assert((n & (n - 1)) == 0 && "grid must be a power of two");
  assert(m >= 0);

  // Knock-in is priced by parity instead of by a second lattice. The knocked-in
  // and knocked-out events partition the sample space for the same monitoring
  // schedule, so in + out = vanilla holds for the discrete contract exactly, not
  // just in the continuous limit. Doing it this way also makes the parity test
  // in test_exact.cpp check the out-price twice over rather than checking a
  // tautology -- if this line were a separate convolution the test would pass
  // even when both were wrong in the same way.
  if (knock == Knock::In) {
    const double van = bs_vanilla(side, p.S, p.K, p.r, p.q, p.sigma, p.T);
    // in + out = vanilla differentiates term by term, so the knock-in delta is
    // the vanilla delta minus the knock-out one.
    const double out = discrete_conv(side, dir, Knock::Out, p, m, n, grid);
    if (grid) grid->delta = vanilla_delta(side, p) - grid->delta;
    return van - out;
  }
  if (m == 0) {
    // No monitoring at all, so the barrier is irrelevant and this is a vanilla.
    // delta_out has to be filled here: the knock-in branch above subtracts what
    // this returns, and leaving it untouched makes a zero-valued knock-in report
    // the whole vanilla delta.
    if (grid) grid->delta = vanilla_delta(side, p);
    return bs_vanilla(side, p.S, p.K, p.r, p.q, p.sigma, p.T);
  }

  const double x0 = std::log(p.S), b = std::log(p.H);
  const double dt = p.T / double(m);
  const double drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
  const double sd = p.sigma * std::sqrt(dt);

  // Grid extent: cover the spot-to-barrier gap plus enough standard deviations
  // of the total diffusion that the payoff beyond the edge is unreachable. At
  // 8 sigma*sqrt(T) the terminal density is ~1e-15 and the truncated payoff
  // contributes below the Richardson residual. Widening further only costs
  // resolution, which is what actually limits accuracy here.
  const double gap = std::fabs(x0 - b);
  const double half = gap + 8.0 * p.sigma * std::sqrt(p.T) + 0.5;

  // Both the barrier and the spot have to land exactly on grid nodes: the
  // barrier because the half-weight trick needs the discontinuity at a node, the
  // spot because otherwise the answer is an interpolation between two nodes and
  // that interpolation error sits on top of the effect being measured. Solving
  // for h from an integer node count buys both.
  const double h_target = 2.0 * half / double(n - 1);
  long k0 = std::lround((x0 - b) / h_target);
  if (k0 == 0) k0 = (x0 >= b) ? 1 : -1;
  const double h = (x0 - b) / double(k0);


  const long ib = long(n / 2) - k0 / 2;  // barrier index, spot-barrier centred
  const long is = ib + k0;               // spot index
  assert(ib > 0 && ib < long(n) && is > 0 && is < long(n));

  const bool down = (dir == Dir::Down);
  const double phi_s = (side == Side::Call) ? 1.0 : -1.0;

  std::vector<double> v(n);
  for (size_t j = 0; j < n; ++j) {
    const double x = b + double(long(j) - ib) * h;
    v[j] = std::fmax(phi_s * (std::exp(x) - p.K), 0.0);
  }
  auto apply_barrier = [&](std::vector<double>& a) {
    if (down) {
      for (long j = 0; j < ib; ++j) a[size_t(j)] = 0.0;
    } else {
      for (long j = ib + 1; j < long(n); ++j) a[size_t(j)] = 0.0;
    }
    a[size_t(ib)] *= 0.5;  // trapezoid endpoint on the live side
  };
  apply_barrier(v);

  // out[i] = sum_j v[j] * g((j - i) h).  A convolution gives
  // c[k] = sum_j v[j] * ker[k - j]; putting k = i + (n-1) and
  // ker[t] = g((n-1-t) h) turns one into the other. This indexing IS the kernel
  // orientation warned about in the header -- ker[t] = g((t-(n-1))h) compiles,
  // runs, and is wrong.
  const size_t klen = 2 * n - 1;
  const size_t L = next_pow2(n + klen - 1);
  RealFFT fft(L);

  std::vector<double> ker(klen);
  for (size_t t = 0; t < klen; ++t) {
    const double u = double(long(n) - 1 - long(t)) * h;
    ker[t] = npdf((u - drift) / sd) / sd * h;
  }

  std::vector<cpx> KF(fft.spectrum_size()), VF(fft.spectrum_size());
  fft.forward(ker.data(), klen, KF.data());

  std::vector<double> work(L);
  const double disc = std::exp(-p.r * dt);
  for (int step = 0; step < m; ++step) {
    fft.forward(v.data(), n, VF.data());
    for (size_t i = 0; i < VF.size(); ++i) VF[i] *= KF[i];
    fft.inverse(VF.data(), work.data());
    for (size_t j = 0; j < n; ++j) v[j] = work[n - 1 + j] * disc;
    apply_barrier(v);
  }

  // k0 != 0 is enforced above, so the spot node is never the barrier node and
  // never carries the half weight. Asserting that is better than branching on
  // it: a runtime branch the comment says is unreachable invites the reader to
  // wonder which of the two is wrong.
  assert(is != ib);

  // Delta, from the grid this function already built and would otherwise throw
  // away. v is a function of log-spot, so dV/dS = (dV/dx) / S.
  //
  // The neighbour that has to be watched is the barrier node: it carries half
  // weight (it is a trapezoid endpoint, not a value), so a central difference
  // that reaches it returns something that is not a derivative of anything.
  // That happens exactly when |k0| == 1, i.e. when spot is one node from the
  // barrier -- rare, but reachable: k0 ~ gap*n / (2*(gap + 8*sigma*sqrt(T))),
  // so a barrier a hundredth of a percent from spot on a 2^14 grid lands there.
  // In that case fall back to a one-sided second-order stencil pointing into
  // the live side, which never touches the barrier node.
  if (grid) {
    grid->h = h;
    grid->k0 = k0;
    const long step = (k0 > 0) ? +1 : -1;  // direction of the live side
    const double inv = 1.0 / (2.0 * h * p.S);
    grid->one_sided = !(std::labs(k0) >= 2 && is - 1 >= 0 && is + 1 < long(n));
    if (!grid->one_sided) {
      grid->delta = (v[size_t(is + 1)] - v[size_t(is - 1)]) * inv;
    } else {
      const long a = is + step, b2 = is + 2 * step;
      assert(a >= 0 && a < long(n) && b2 >= 0 && b2 < long(n));
      const double fwd = -3.0 * v[size_t(is)] + 4.0 * v[size_t(a)] - v[size_t(b2)];
      grid->delta = double(step) * fwd * inv;
    }
  }
  return v[size_t(is)];
}

ExactResult discrete_exact(Side side, Dir dir, Knock knock, const Params& p,
                           int m, size_t n) {
  ExactResult r;
  r.n = n;

  // No monitoring means no grid and nothing to extrapolate. Handled here rather
  // than left to fall through, because discrete_conv fills only `delta` in that
  // case and the ratio arithmetic below would be reading zeros.
  if (m == 0) {
    GridInfo g;
    r.price = r.coarse = r.fine = discrete_conv(side, dir, knock, p, 0, n, &g);
    r.delta = g.delta;
    r.richardson = 0.0;
    return r;
  }

  GridInfo gc, gf;
  r.coarse = discrete_conv(side, dir, knock, p, m, n, &gc);
  r.fine = discrete_conv(side, dir, knock, p, m, 2 * n, &gf);
  // Richardson on a second-order scheme: with e(h) = C h^2 + o(h^2),
  //     price = fine + (fine - coarse) / (ratio^2 - 1),   ratio = h_coarse/h_fine.
  //
  // The textbook denominator is 3, because doubling the node count is assumed to
  // halve the spacing. Here it does not, quite. The spacing is solved for from an
  // integer node count between barrier and spot so that both land on nodes, and
  // that count goes 259 -> 517, not 259 -> 518: a ratio of 1.99614.
  // Using 3 anyway puts a 0.5% error into the correction term, and since the
  // correction is the only thing standing between this benchmark and its own
  // discretisation error, that 0.5% was the dominant error in the whole result --
  // large enough to bend the measured residual exponent away from 1.5 at the
  // finest monitoring levels. Two lines, and it is worth about a factor of 16.
  const double ratio = (gf.h > 0.0) ? gc.h / gf.h : 2.0;

  // Doubling n usually doubles the node count between barrier and spot, but not
  // always: it is a rounded integer, and when the barrier is close enough to spot
  // that the count is already 1, doubling n leaves it at 1 and both grids come
  // out with the SAME spacing. Then ratio is 1, ratio^2 - 1 is 0, and the
  // extrapolation divides by zero -- it returned inf, silently, for a price that
  // was otherwise fine. Nothing in the experiments reaches this (they run node
  // counts in the hundreds), which is exactly why it went unnoticed.
  if (ratio < 1.0 + 1e-9) {
    r.price = r.fine;
    r.richardson = r.fine - r.coarse;  // the honest disagreement, unextrapolated
    r.delta = gf.delta;
    return r;
  }

  const double den = ratio * ratio - 1.0;
  r.richardson = (r.fine - r.coarse) / den;
  r.price = r.fine + r.richardson;

  // Delta extrapolates on the same h^2 ladder as the price, but only when both
  // grids computed it the same way. Near the barrier the coarse grid can fall
  // back to the one-sided stencil while the fine one still manages a central
  // difference, and those two have different error constants -- extrapolating
  // across the change gave 1.1437 against a converged 1.1545, a 0.9% error that
  // looks like a converged answer. When the stencils disagree, take the finer
  // grid's value and extrapolate nothing.
  r.delta = (gc.one_sided == gf.one_sided)
                ? gf.delta + (gf.delta - gc.delta) / den
                : gf.delta;
  return r;
}

}  // namespace bm
