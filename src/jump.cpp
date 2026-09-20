#include "bm/jump.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "bm/rng.hpp"

namespace bm {
namespace {

constexpr int kMaxJumps = 6;                     // per monitoring interval
constexpr int kDimsPerStep = 2 + 3 * kMaxJumps;  // dW, N, then (size,time,bridge)

// Poisson by inverse CDF. The means here are lambda*dt, at most a few
// hundredths, so the loop almost always exits at k = 0 and a fancier sampler
// would be slower. Truncating at kMaxJumps is safe at these means, P(N > 6) is
// below 1e-14 for lambda*dt < 0.1, but it IS a truncation, so run_jump reports
// the realised mean jump count for comparison against lambda*T.
inline int poisson_inv(double u, double mean) {
  double p = std::exp(-mean), c = p;
  int k = 0;
  while (u > c && k < kMaxJumps) {
    ++k;
    p *= mean / double(k);
    c += p;
  }
  return k;
}

inline double log_survival(double u) {
  if (u > 36.0) return 0.0;
  return std::log(-std::expm1(-u));
}

}  // namespace

JumpRun run_jump(Side side, Dir dir, const Params& p, const JumpParams& jp,
                 int m, const McConfig& cfg, double h_bridge) {
  JumpRun out;
  out.m = m;
  const int threads =
      cfg.threads > 0 ? cfg.threads : int(std::thread::hardware_concurrency());

  const double dt = p.T / double(m);
  const double sd = p.sigma * std::sqrt(dt);
  const double kappa = std::exp(jp.mu_j + 0.5 * jp.sigma_j * jp.sigma_j) - 1.0;
  // Compensated drift, so that E[S_T] = S_0 exp((r-q)T) exactly. If the
  // compensator is wrong the model prices a different asset and every number
  // downstream is quietly about that other asset; out.spot checks it.
  const double mu = p.r - p.q - jp.lambda * kappa - 0.5 * p.sigma * p.sigma;
  const double logb_d = std::log(p.H);
  const double logb_b = std::log(h_bridge > 0.0 ? h_bridge : p.H);
  const double phi = (side == Side::Call) ? 1.0 : -1.0;
  const double down = (dir == Dir::Down) ? 1.0 : -1.0;
  const double disc = std::exp(-p.r * p.T);
  const double inv_s2 = 1.0 / (p.sigma * p.sigma);
  const double lam_dt = jp.lambda * dt;
  const double s0 = p.S;
  const double fwd_disc = std::exp(-(p.r - p.q) * p.T);

  struct Acc {
    Welford d, bn, bj, cp, bg, sp;
    double jumps = 0.0;
    uint64_t n = 0;
  };
  std::vector<Acc> per(static_cast<size_t>(threads));

  auto worker = [&](int tid) {
    Acc a;
    double ut[kMaxJumps], yj[kMaxJumps];
    for (uint64_t i = uint64_t(tid); i < cfg.paths; i += uint64_t(threads)) {
      Stream st(cfg.seed, i);
      double x = std::log(p.S);
      bool alive_d = true;   // survives the grid checks at the discrete barrier
      bool alive_b = true;   // survives the grid checks at the bridge barrier
      bool alive_j = true;   // and no jump landed across the bridge barrier
      double logw_naive = 0.0, logw_jump = 0.0;
      int total_jumps = 0;

      for (int step = 0; step < m; ++step) {
        const uint64_t base = uint64_t(step) * kDimsPerStep;
        const double dW = sd * st.gauss(base);
        const int nj = poisson_inv(st.uniform(base + 1), lam_dt);
        total_jumps += nj;

        double jsum = 0.0;
        for (int k = 0; k < nj; ++k) {
          yj[k] = jp.mu_j + jp.sigma_j * st.gauss(base + 2 + 3 * uint64_t(k));
          ut[k] = dt * st.uniform(base + 3 + 3 * uint64_t(k));
          jsum += yj[k];
        }
        // Jump times must be sorted before the sub-interval bridges are built:
        // an unsorted pair produces a negative time gap, and log_survival of a
        // negative argument quietly returns a survival probability above one.
        for (int k = 1; k < nj; ++k)
          for (int l = k; l > 0 && ut[l] < ut[l - 1]; --l) {
            std::swap(ut[l], ut[l - 1]);
            std::swap(yj[l], yj[l - 1]);
          }

        const double x_start = x;
        const double x_end = x + mu * dt + dW + jsum;

        // --- naive: the Gaussian bridge applied to the endpoints ------------
        // Exactly what mc.cpp's formula gives if it is carried over unchanged.
        // It is correct when nj == 0 and wrong otherwise, because the endpoints
        // it conditions on contain the jumps while the bridge law it assumes
        // does not.
        if (alive_b) {
          const double da = down * (x_start - logb_b);
          const double db = down * (x_end - logb_b);
          if (da > 0.0 && db > 0.0)
            logw_naive += log_survival(2.0 * da * db * inv_s2 / dt);
        }

        // --- jump-aware: condition on the jump times and sizes --------------
        // Between consecutive jumps the path IS a Brownian bridge, so the closed
        // form applies piecewise; the jumps are checked as point events, since a
        // downward jump through the barrier is a crossing however briefly it
        // lasts.
        if (alive_j) {
          double t_prev = 0.0, b_prev = 0.0, x_prev = x_start, jcum = 0.0;
          for (int k = 0; k < nj && alive_j; ++k) {
            const double u = ut[k];
            const double rem = dt - t_prev;
            double b_u = b_prev;
            if (rem > 0.0 && u > t_prev) {
              const double frac = (u - t_prev) / rem;
              const double mean = b_prev + frac * (dW - b_prev);
              const double var = (u - t_prev) * (dt - u) / rem;
              b_u = mean + std::sqrt(std::fmax(var, 0.0)) *
                               st.gauss(base + 4 + 3 * uint64_t(k));
            }
            const double x_pre = x_start + mu * u + b_u + jcum;
            const double dp = down * (x_prev - logb_b);
            const double dc = down * (x_pre - logb_b);
            if (dc <= 0.0) { alive_j = false; break; }
            if (dp > 0.0 && u > t_prev)
              logw_jump += log_survival(2.0 * dp * dc * inv_s2 / (u - t_prev));
            jcum += yj[k];
            const double x_post = x_pre + yj[k];
            if (down * (x_post - logb_b) <= 0.0) { alive_j = false; break; }
            t_prev = u;
            b_prev = b_u;
            x_prev = x_post;
          }
          if (alive_j) {
            const double dp = down * (x_prev - logb_b);
            const double dc = down * (x_end - logb_b);
            if (dc <= 0.0) alive_j = false;
            else if (dp > 0.0 && dt > t_prev)
              logw_jump += log_survival(2.0 * dp * dc * inv_s2 / (dt - t_prev));
          }
        }

        x = x_end;
        if (down * (x - logb_d) <= 0.0) alive_d = false;
        if (down * (x - logb_b) <= 0.0) { alive_b = false; alive_j = false; }
      }

      const double payoff = std::fmax(phi * (std::exp(x) - p.K), 0.0) * disc;
      const double v_d = alive_d ? payoff : 0.0;
      const double v_bn = alive_b ? payoff * std::exp(logw_naive) : 0.0;
      const double v_bj = alive_j ? payoff * std::exp(logw_jump) : 0.0;
      a.d.add(v_d);
      a.bn.add(v_bn);
      a.bj.add(v_bj);
      a.cp.add(v_d - v_bj);
      a.bg.add(v_bn - v_bj);
      a.sp.add(std::exp(x) * fwd_disc / s0);
      a.jumps += double(total_jumps);
      ++a.n;
    }
    per[static_cast<size_t>(tid)] = a;
  };

  std::vector<std::thread> ts;
  ts.reserve(static_cast<size_t>(threads));
  for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t);
  for (auto& t : ts) t.join();

  double jumps = 0.0;
  uint64_t n = 0;
  for (auto& a : per) {
    out.discrete.merge(a.d);
    out.bridge_naive.merge(a.bn);
    out.bridge_jump.merge(a.bj);
    out.coupled.merge(a.cp);
    out.bridge_gap.merge(a.bg);
    out.spot.merge(a.sp);
    jumps += a.jumps;
    n += a.n;
  }
  out.mean_jumps = (n > 0) ? jumps / double(n) : 0.0;
  return out;
}

}  // namespace bm
