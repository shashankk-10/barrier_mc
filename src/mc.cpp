#include "bm/mc.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "bm/rng.hpp"

namespace bm {
namespace {

// log(1 - exp(-u)), the log of a one-step Brownian-bridge survival probability.
//
// Two things are deliberate here.
//
// -expm1(-u) instead of 1 - exp(-u): for a path far from the barrier u is
// large and the answer is nearly 1, which is fine either way; but for a path
// hugging the barrier u is small, 1 - exp(-u) cancels to a few bits, and expm1
// is the function that keeps them. Those are precisely the paths that carry the
// entire signal, since they are the ones where the two estimators disagree.
//
// The early exit is exact, not approximate: at u = 36 the survival probability
// differs from 1 by 2e-16, below the resolution of the double accumulating it.
// It fires on the overwhelming majority of steps, since most paths spend most of
// their life nowhere near the barrier. Without it the bridge weight costs a log
// and an expm1 per step per level, which dominates the run.
inline double log_survival(double u) {
  if (u > 36.0) return 0.0;
  return std::log(-std::expm1(-u));
}

struct PathScratch {
  std::vector<double> x;  // log-price on the finest grid, x[0] = log(S0)
};

// Accumulate one path (or one antithetic pair) into `out`.
template <bool kAnti>
void one_path(uint64_t path, Side side, Dir dir, const Params& p,
              const McConfig& cfg, int levels, int m_max, double drift,
              double sd, double logb, const std::vector<double>& inv2s2dt,
              PathScratch& sc, std::vector<LevelStats>& out) {
  const double phi = (side == Side::Call) ? 1.0 : -1.0;
  const double down = (dir == Dir::Down) ? 1.0 : -1.0;
  const double disc = std::exp(-p.r * p.T);

  const int reps = kAnti ? 2 : 1;
  // Per-level running totals across the antithetic pair.
  double acc_d[32] = {0}, acc_b[32] = {0};

  for (int rep = 0; rep < reps; ++rep) {
    Stream st(cfg.seed, path);
    double x = std::log(p.S);
    sc.x[0] = x;
    for (int j = 1; j <= m_max; ++j) {
      const double z = (rep == 0) ? st.gauss(uint64_t(j - 1))
                                  : st.gauss_anti(uint64_t(j - 1));
      x += drift + sd * z;
      sc.x[size_t(j)] = x;
    }
    const double payoff = std::fmax(phi * (std::exp(x) - p.K), 0.0) * disc;

    for (int l = 0; l < levels; ++l) {
      const int stride = 1 << (levels - 1 - l);
      double d_prev = down * (sc.x[0] - logb);
      bool alive = true;
      double logw = 0.0;
      for (int j = stride; j <= m_max; j += stride) {
        const double d_cur = down * (sc.x[size_t(j)] - logb);
        if (d_cur <= 0.0) { alive = false; break; }
        logw += log_survival(2.0 * d_prev * d_cur * inv2s2dt[size_t(l)]);
        d_prev = d_cur;
      }
      if (!alive) continue;  // both estimators are zero; nothing to add
      acc_d[l] += payoff;
      acc_b[l] += payoff * std::exp(logw);
    }
  }

  const double inv = 1.0 / double(reps);
  for (int l = 0; l < levels; ++l) {
    const double d = acc_d[l] * inv, b = acc_b[l] * inv;
    out[size_t(l)].discrete.add(d);
    out[size_t(l)].bridge.add(b);
    out[size_t(l)].bias.add(d - b);
  }
}

int resolve_threads(int want) {
  if (want > 0) return want;
  const unsigned hc = std::thread::hardware_concurrency();
  return int(hc == 0 ? 4u : hc);
}

}  // namespace

ConvergenceRun run_convergence(Side side, Dir dir, const Params& p,
                               const McConfig& cfg) {
  ConvergenceRun run;
  run.paths = cfg.paths;
  run.threads = resolve_threads(cfg.threads);
  const int levels = std::min(cfg.levels, 31);
  const int m_max = cfg.m0 << (levels - 1);

  const double dt = p.T / double(m_max);
  const double drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
  const double sd = p.sigma * std::sqrt(dt);
  const double logb = std::log(p.H);

  // 1 / (sigma^2 * dt_l) for each level, so the inner loop does one multiply.
  std::vector<double> inv2s2dt(static_cast<size_t>(levels));
  for (int l = 0; l < levels; ++l) {
    const double dtl = p.T / double(cfg.m0 << l);
    inv2s2dt[size_t(l)] = 1.0 / (p.sigma * p.sigma * dtl);
  }

  std::vector<std::vector<LevelStats>> per_thread(
      static_cast<size_t>(run.threads));
  for (auto& v : per_thread) {
    v.resize(size_t(levels));
    for (int l = 0; l < levels; ++l) v[size_t(l)].m = cfg.m0 << l;
  }

  auto worker = [&](int tid) {
    PathScratch sc;
    sc.x.resize(size_t(m_max) + 1);
    auto& out = per_thread[size_t(tid)];
    // Strided assignment, so the set of paths a thread owns depends on the
    // thread count but the value each path produces does not. That is what
    // makes the run reproducible across -j settings.
    for (uint64_t i = uint64_t(tid); i < cfg.paths;
         i += uint64_t(run.threads)) {
      if (cfg.antithetic)
        one_path<true>(i, side, dir, p, cfg, levels, m_max, drift, sd, logb,
                       inv2s2dt, sc, out);
      else
        one_path<false>(i, side, dir, p, cfg, levels, m_max, drift, sd, logb,
                        inv2s2dt, sc, out);
    }
  };

  std::vector<std::thread> ts;
  ts.reserve(size_t(run.threads));
  for (int t = 0; t < run.threads; ++t) ts.emplace_back(worker, t);
  for (auto& t : ts) t.join();

  run.levels.resize(size_t(levels));
  for (int l = 0; l < levels; ++l) {
    run.levels[size_t(l)].m = cfg.m0 << l;
    for (int t = 0; t < run.threads; ++t) {
      run.levels[size_t(l)].discrete.merge(per_thread[size_t(t)][size_t(l)].discrete);
      run.levels[size_t(l)].bridge.merge(per_thread[size_t(t)][size_t(l)].bridge);
      run.levels[size_t(l)].bias.merge(per_thread[size_t(t)][size_t(l)].bias);
    }
  }
  return run;
}

PlainRun run_plain(Side side, Dir dir, const Params& p, int m,
                   const McConfig& cfg) {
  PlainRun out;
  out.m = m;
  out.paths = cfg.paths;
  const int threads = resolve_threads(cfg.threads);
  const double dt = p.T / double(m);
  const double drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
  const double sd = p.sigma * std::sqrt(dt);
  const double logb = std::log(p.H);
  const double phi = (side == Side::Call) ? 1.0 : -1.0;
  const double down = (dir == Dir::Down) ? 1.0 : -1.0;
  const double disc = std::exp(-p.r * p.T);

  std::vector<Welford> per_thread(static_cast<size_t>(threads));
  auto worker = [&](int tid) {
    Welford w;
    for (uint64_t i = uint64_t(tid); i < cfg.paths; i += uint64_t(threads)) {
      Stream st(cfg.seed, i);
      double x = std::log(p.S);
      bool alive = true;
      for (int j = 0; j < m; ++j) {
        x += drift + sd * st.gauss(uint64_t(j));
        if (down * (x - logb) <= 0.0) { alive = false; break; }
      }
      // A knocked-out path must still be counted, at zero. Skipping it is the
      // classic way to price a barrier option far too high.
      w.add(alive ? std::fmax(phi * (std::exp(x) - p.K), 0.0) * disc : 0.0);
    }
    per_thread[size_t(tid)] = w;
  };
  std::vector<std::thread> ts;
  ts.reserve(size_t(threads));
  for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t);
  for (auto& t : ts) t.join();
  for (auto& w : per_thread) out.price.merge(w);
  return out;
}

}  // namespace bm
