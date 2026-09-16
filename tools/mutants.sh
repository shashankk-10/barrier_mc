#!/usr/bin/env bash
# Mutation testing: inject a defect, rebuild, and check that the suite notices.
#
# A green suite proves the tests pass, not that they would fail if the code were
# wrong. That distinction is the whole subject of this repo: every bug it
# actually hit produced plausible output -- a positive, monotone,
# correctly-converging price that was the right answer to a different question.
# So each mutant here is a defect of that kind, not a syntax error or a crash.
#
# Usage:  tools/mutants.sh [name ...]      (default: all)
# Exit status is the number of survivors.
#
# Survivors are printed, not hidden. Some are expected; see the README.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Fields are separated by @@ : name @@ file @@ sed-find @@ sed-replace @@ description
# sed runs with '#' as its delimiter, so no field may contain '#'.
MUTANTS=(
'kernel-orientation@@src/exact.cpp@@const double u = double(long(n) - 1 - long(t)) * h;@@const double u = double(long(t) - long(n) + 1) * h;@@density as a function of the source, not the destination: the adjoint operator'
'no-half-weight@@src/exact.cpp@@a[size_t(ib)] *= 0.5;@@a[size_t(ib)] *= 1.0;@@full weight at the barrier node: first-order quadrature across the jump'
'beta-sign@@include/bm/bgk.hpp@@const double s = (dir == Dir::Up) ? +1.0 : -1.0;@@const double s = (dir == Dir::Up) ? -1.0 : +1.0;@@barrier shifted toward spot instead of away'
'bridge-no-factor-2@@src/mc.cpp@@logw += log_survival(2.0 * d_prev * d_cur * inv2s2dt[size_t(l)]);@@logw += log_survival(1.0 * d_prev * d_cur * inv2s2dt[size_t(l)]);@@reflection-principle exponent missing its factor of two'
'bridge-early-exit@@src/mc.cpp@@if (u > 36.0) return 0.0;@@if (u > 4.0) return 0.0;@@survival approximation truncated far too early'
'plain-skip-knockout@@src/mc.cpp@@w.add(alive ? std::fmax(phi * (std::exp(x) - p.K), 0.0) * disc : 0.0);@@if (alive) w.add(std::fmax(phi * (std::exp(x) - p.K), 0.0) * disc);@@knocked-out paths dropped instead of counted at zero'
'ncdf-cancelling-form@@include/bm/normal.hpp@@return 0.5 * std::erfc(-x / kSqrt2);@@return 0.5 * (1.0 + std::erf(x / kSqrt2));@@left tail computed by cancellation'
'philox-9-rounds@@include/bm/rng.hpp@@for (int r = 0; r < 10; ++r) {@@for (int r = 0; r < 9; ++r) {@@one round short of the specified generator'
'inout-parity@@src/exact.cpp@@    return van - out;@@    return van + out;@@knock-in parity with the wrong sign'
'richardson-fixed-3@@src/exact.cpp@@  r.richardson = (r.fine - r.coarse) / den;@@  r.richardson = (r.fine - r.coarse) / 3.0;@@Richardson assuming an exact grid ratio of 2 that quantisation does not give'
'jump-unsorted@@src/jump.cpp@@ut[l] < ut[l - 1]@@ut[l] > ut[l - 1]@@jump times sorted the wrong way, so a sub-interval bridge gets a negative time gap'
'delta-no-spot-divide@@src/exact.cpp@@const double inv = 1.0 / (2.0 * h * p.S);@@const double inv = 1.0 / (2.0 * h);@@delta left as dV/dlogS instead of dV/dS'
'delta-onesided-sign@@src/exact.cpp@@grid->delta = double(step) * fwd * inv;@@grid->delta = fwd * inv;@@one-sided stencil missing the direction factor'
'welford-naive@@include/bm/stats.hpp@@m2_ += d * (x - mean_);@@m2_ += (x - mean_) * (x - mean_);@@variance accumulated against the post-update mean'
)

# Every mutant is a literal string substitution into a source file, so a refactor
# that touches the matched line silently disarms it. That has happened three times
# in this repo, and each time the whole matrix ran for twenty minutes before
# saying so. Verify every pattern matches before building anything: it takes a
# second, and `--check` does only this.
check_only=0
if [ "${1:-}" = "--check" ]; then check_only=1; shift; fi

broken=0
for entry in "${MUTANTS[@]}"; do
  IFS=$'\n' read -r -d '' cname cfile cfind crepl cdesc < <(printf '%s@@' "$entry" | sed 's/@@/\n/g'; printf '\0')
  hits=$(/usr/bin/env python3 -c "
import sys
print(open(sys.argv[1]).read().count(sys.argv[2]))" "$cfile" "$cfind")
  if [ "$hits" != "1" ]; then
    printf '%-22s  \033[31mPATTERN MATCHES %s TIMES\033[0m  in %s\n' "$cname" "$hits" "$cfile"
    broken=$((broken+1))
  fi
done
if [ $broken -gt 0 ]; then
  echo
  echo "$broken mutant pattern(s) no longer match their source. Fix them before"
  echo "trusting this matrix -- an unmatched pattern tests nothing."
fi
if [ $check_only -eq 1 ]; then
  [ $broken -eq 0 ] && echo "all ${#MUTANTS[@]} mutant patterns match"
  exit $broken
fi

want=("$@")
killed=0; survived=0; notapplied=0; survivors=()

for entry in "${MUTANTS[@]}"; do
  IFS=$'\n' read -r -d '' name file find repl desc < <(printf '%s@@' "$entry" | sed 's/@@/\n/g'; printf '\0')

  if [ ${#want[@]} -gt 0 ]; then
    hit=0
    for w in "${want[@]}"; do [ "$w" = "$name" ] && hit=1; done
    [ $hit -eq 1 ] || continue
  fi

  dir="$WORK/$name"
  mkdir -p "$dir"
  (cd "$ROOT" && tar cf - CMakeLists.txt include src tests experiments) | (cd "$dir" && tar xf -)

  # Literal, single-occurrence substitution done in python so that C++ operators
  # in the patterns need no escaping. A mutant whose pattern fails to match is a
  # broken mutant, and it is reported rather than silently counted as killed.
  if ! /usr/bin/env python3 - "$dir/$file" "$find" "$repl" <<'PY'
import sys
path, find, repl = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if s.count(find) != 1:
    sys.stderr.write("pattern matched %d times\n" % s.count(find)); sys.exit(1)
open(path, 'w').write(s.replace(find, repl))
PY
  then
    printf '%-22s  \033[33mNOT APPLIED\033[0m  %s\n' "$name" "$desc"
    notapplied=$((notapplied+1))
    continue
  fi

  if ! cmake -S "$dir" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
     || ! cmake --build "$dir/build" -j4 >/dev/null 2>&1; then
    printf '%-22s  \033[33mBUILD FAILED\033[0m        %s\n' "$name" "$desc"
    survived=$((survived+1)); survivors+=("$name(build-failed)")
    continue
  fi

  out=$( (cd "$dir/build" && ctest 2>&1) )
  if echo "$out" | grep -q "100% tests passed"; then
    printf '%-22s  \033[31mSURVIVED\033[0m            %s\n' "$name" "$desc"
    survived=$((survived+1)); survivors+=("$name")
  else
    by=$(echo "$out" | grep -E '\*\*\*Failed|\*\*\*Exception' | sed 's/.*: //;s/ .*//' | tr '\n' ',' | sed 's/,$//')
    printf '%-22s  \033[32mkilled\033[0m by %-12s %s\n' "$name" "${by:-ctest}" "$desc"
    killed=$((killed+1))
  fi
done

echo
echo "killed $killed, survived $survived, not applied $notapplied"
[ ${#survivors[@]} -gt 0 ] && echo "survivors: ${survivors[*]}"
[ $notapplied -gt 0 ] && echo "NOT APPLIED means a broken pattern, not a passing test."
exit $((survived + notapplied))
