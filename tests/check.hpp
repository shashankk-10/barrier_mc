#pragma once

// A three-line test harness. A dependency-free repo that anyone can build with
// cmake && ctest is worth more here than the features a framework would add.
//
// This project's assertions are almost all "this double equals that double to
// within a stated tolerance", which is why CHECK_NEAR and CHECK_REL exist and
// why they print %.17g on failure. A numerical test that fails and tells you
// only "assertion failed" has cost you the one piece of information you needed:
// whether you are off by a sign, by a factor, or in the last two bits.

#include <cmath>
#include <cstdio>

namespace check {

inline int g_failed = 0;
inline int g_total = 0;

inline void report(const char* file, int line, const char* expr, bool ok) {
  ++g_total;
  if (ok) return;
  ++g_failed;
  fprintf(stderr, "  FAIL %s:%d  %s\n", file, line, expr);
}

inline int finish(const char* suite) {
  printf("%s: %d/%d passed\n", suite, g_total - g_failed, g_total);
  return g_failed == 0 ? 0 : 1;
}

}  // namespace check

#define CHECK(expr) ::check::report(__FILE__, __LINE__, #expr, (expr))

#define CHECK_EQ(a, b)                                                         \
  do {                                                                         \
    auto _a = (a);                                                             \
    auto _b = (b);                                                             \
    bool _ok = (_a == _b);                                                     \
    ::check::report(__FILE__, __LINE__, #a " == " #b, _ok);                    \
    if (!_ok)                                                                  \
      fprintf(stderr, "       got %lld, want %lld\n", (long long)_a,           \
              (long long)_b);                                                  \
  } while (0)

// Absolute tolerance. Use when the quantity has a natural scale -- a price in
// currency units, a probability, a fitted exponent.
#define CHECK_NEAR(a, b, tol)                                                  \
  do {                                                                         \
    double _a = (a), _b = (b), _t = (tol);                                     \
    double _d = std::fabs(_a - _b);                                            \
    bool _ok = (_d <= _t);                                                     \
    ::check::report(__FILE__, __LINE__, #a " ~= " #b, _ok);                    \
    if (!_ok)                                                                  \
      fprintf(stderr, "       got %.17g, want %.17g   (|diff| %.3g > %.3g)\n", \
              _a, _b, _d, _t);                                                 \
  } while (0)

// Relative tolerance, floored at 1 so that values near zero do not demand
// impossible precision. Use when comparing against a reference value whose
// magnitude varies across the test table.
#define CHECK_REL(a, b, tol)                                                   \
  do {                                                                         \
    double _a = (a), _b = (b), _t = (tol);                                     \
    double _s = std::fmax(1.0, std::fabs(_b));                                 \
    double _d = std::fabs(_a - _b) / _s;                                       \
    bool _ok = (_d <= _t);                                                     \
    ::check::report(__FILE__, __LINE__, #a " ~= " #b, _ok);                    \
    if (!_ok)                                                                  \
      fprintf(stderr, "       got %.17g, want %.17g   (rel %.3g > %.3g)\n",    \
              _a, _b, _d, _t);                                                 \
  } while (0)
