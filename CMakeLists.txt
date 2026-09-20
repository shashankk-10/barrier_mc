cmake_minimum_required(VERSION 3.20)
project(barrier_mc CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# No fast-math, anywhere, and this is not a default I am leaving unexamined.
# -ffast-math lets the compiler assume no NaN/Inf and reassociate floating point
# freely. This project's entire claim is about the size of numerical errors, and
# two of its checks, the in/out parity identity in bs.hpp and the Richardson
# extrapolation in exact.hpp, work by subtracting nearly equal quantities.
# Reassociation would change the answers by more than the effects being
# measured, and would do it silently.
find_package(Threads REQUIRED)

add_library(bm STATIC
  src/bs.cpp
  src/exact.cpp
  src/mc.cpp
  src/jump.cpp
)
target_include_directories(bm PUBLIC include)
target_link_libraries(bm PUBLIC Threads::Threads)

enable_testing()

# ---------------------------------------------------------------- unit tests
# Every one of these compares against something computed a different way: a
# closed form against an independent quadrature, a transform against a direct
# DFT, a generator against its published known-answer vectors. A numerical test
# that only checks a function against itself will pass whatever the function does.
#
# Release sets -DNDEBUG, which turns every assert() into nothing. -UNDEBUG per
# target is not optional, fft.hpp asserts its length is a power of two, and
# without it a mis-sized transform silently reads out of bounds instead of
# stopping.
foreach(t numerics bs exact mc bgk jump)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE bm)
  target_compile_options(test_${t} PRIVATE -UNDEBUG)
  add_test(NAME ${t} COMMAND test_${t})
endforeach()

# Sanitiser variants. exact.cpp does index arithmetic into a grid from a barrier
# position that is computed, not given, ASan is the difference between "the
# benchmark is accurate" and "the benchmark is accurate and did not read one
# cell past the end of the lattice to get there".
foreach(t numerics exact mc jump)
  add_executable(test_${t}_asan tests/test_${t}.cpp)
  target_link_libraries(test_${t}_asan PRIVATE bm)
  target_compile_options(test_${t}_asan PRIVATE -UNDEBUG -fsanitize=address,undefined -g -O1)
  target_link_options(test_${t}_asan PRIVATE -fsanitize=address,undefined)
  add_test(NAME ${t}_asan COMMAND test_${t}_asan)
endforeach()

# ---------------------------------------------------------------- experiments
# These produce the tables in the README. They are not registered with ctest:
# they take minutes instead of seconds, and a suite that takes five minutes is
# a suite people stop running. scripts/run_all.sh runs them and saves the
# transcripts under results/.
foreach(e convergence correction overshoot estimator jump)
  add_executable(exp_${e} experiments/exp_${e}.cpp)
  target_link_libraries(exp_${e} PRIVATE bm)
  target_include_directories(exp_${e} PRIVATE experiments)
endforeach()
