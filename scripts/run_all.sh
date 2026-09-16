#!/usr/bin/env bash
# Builds, runs the test suite, then runs every experiment and saves the
# transcripts under results/.
#
# The order matters. The experiments print confident-looking tables whether or
# not the closed forms behind them are right, so nothing here believes an
# experiment until ctest is green: that is where the barrier formulas are
# checked against an independent implementation, the benchmark against an
# independent quadrature, and the generator against its published vectors.
#
# Unlike my tcp_probe repo, results/ is committed. Every number below is a
# property of the mathematics rather than of this machine -- the benchmark is
# deterministic and the Monte Carlo is seeded through a counter-based generator
# whose output depends only on the path index -- so a clone on any box should
# reproduce these files. If it does not, that is a bug worth hearing about.

set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=${BUILD:-build}
RESULTS=results
mkdir -p "$RESULTS"

echo "=== configure + build ==="
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo
echo "=== tests ==="
( cd "$BUILD" && ctest --output-on-failure )

{
  echo "host      : $(uname -srm)"
  echo "cpu       : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "cores     : $(sysctl -n hw.ncpu 2>/dev/null || nproc)"
  echo "compiler  : $(${CXX:-c++} --version | head -1)"
  echo "cmake     : $(cmake --version | head -1)"
  echo "build     : Release, -Wall -Wextra -Wpedantic, no -ffast-math"
} > "$RESULTS/run-metadata.txt"

echo
for e in convergence correction overshoot estimator jump; do
  bin="$BUILD/exp_$e"
  [ -x "$bin" ] || continue
  echo "=== exp_$e ==="
  "$bin" | tee "$RESULTS/exp_$e.txt"
  echo
done

echo "All transcripts written to $RESULTS/."
