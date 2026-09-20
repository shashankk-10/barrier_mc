#pragma once

// A near-exact price for the discretely monitored barrier option, computed
// without Monte Carlo.
//
// Between two monitoring dates the log-price moves by an exactly Gaussian
// increment, so the value function propagates backwards by convolution against
// a Gaussian kernel; at each monitoring date the barrier is applied by zeroing
// the dead side. m monitoring dates is m convolutions. On a uniform log-grid
// each one is an FFT, and the whole thing costs O(m * n log n) with no sampling
// error of any kind.
//
// This matters more than it looks. The claim the repo is built on is a
// statement about the size of an error, and you cannot measure an error against
// a reference that has an error of its own of the same order. A Monte Carlo
// reference at a few million paths pins the discrete price to about 1e-2; the
// post-correction residual this project needs to resolve is 1e-6. The
// convolution benchmark is what closes that four-order-of-magnitude gap.
//
// Two things here are easy to get wrong, and both produce plausible numbers.
//
// 1. Kernel orientation. Backward induction needs
//        V_i(x) = e^{-r dt} * integral V_{i+1}(y) * p(y | x) dy,
//    i.e. the density as a function of the destination y given the source x.
//    Writing the convolution with the kernel the other way round computes the
//    adjoint operator instead. It still conserves mass, still converges as the
//    grid is refined, and still returns a positive, monotone, plausible price.
//    It is simply the answer to a different question, off by a constant that no
//    amount of refinement removes. The only thing that catches it is pricing
//    a case whose answer is known independently, which is why test_exact.cpp
//    starts with m=1 (where the barrier is inert and the price must equal the
//    vanilla Black-Scholes value to 1e-7) before it trusts anything else.
//
// 2. Quadrature order at the barrier. Knocking out leaves the value function
//    discontinuous at the barrier, and a uniform-weight sum across a jump is
//    only first-order accurate, which is exactly the O(h) convergence this
//    file is trying to measure the O(sqrt(dt)) version of. Putting a grid node
//    exactly on the barrier and giving that node half weight makes the sum a
//    trapezoid rule on the live side, restoring O(h^2). Measured on the m=2
//    case against an independent quadrature: without the half weight the error
//    goes 1.23e-3 -> 6.15e-4 -> 3.07e-4 as the grid doubles (first order); with
//    it, 4.26e-6 -> 1.06e-6 -> 2.66e-7 (second order).

#include <cstddef>

#include "bm/bs.hpp"

namespace bm {

struct ExactResult {
  double price = 0.0;
  double delta = 0.0;     // dV/dS, read off the same grid
  double coarse = 0.0;    // value on the n-point grid
  double fine = 0.0;      // value on the 2n-point grid
  // (fine - coarse) / (ratio^2 - 1), where ratio is the true spacing ratio
  // between the two grids. Simultaneously the correction and an estimate of the
  // error left in `fine`. Not the error of `price`, which is smaller, see the
  // grid ladder printed by exp_correction.
  double richardson = 0.0;
  size_t n = 0;           // coarse grid size actually used
};

// What a single solve actually used, for callers that have to reason about it.
// discrete_exact does: the spacing is solved for from an integer node count so
// that barrier and spot both land on nodes, which means doubling n neither
// exactly halves h nor necessarily changes which delta stencil was used.
struct GridInfo {
  double h = 0.0;          // node spacing in log-space, always positive
  long k0 = 0;             // signed node count from barrier to spot
  double delta = 0.0;      // dV/dS at spot, off this grid
  bool one_sided = false;  // delta came from the one-sided stencil
};
// With m == 0 there is no grid, the price is the vanilla one and the function
// returns before building anything, so only `delta` is filled and the rest stay
// zero. discrete_exact handles that case separately instead of reading them.

// Single-grid price. `n` must be a power of two.
double discrete_conv(Side side, Dir dir, Knock knock, const Params& p, int m,
                     size_t n, GridInfo* grid = nullptr);

// Richardson-extrapolated in h^2 from grids of n and 2n points. This is the
// function every experiment should call; `richardson` is an honest error bar on
// the returned price and the tests assert it is small.
ExactResult discrete_exact(Side side, Dir dir, Knock knock, const Params& p,
                           int m, size_t n = (1u << 14));

}  // namespace bm
