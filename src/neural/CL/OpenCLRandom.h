/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto
    Copyright (C) 2018 Folkert Huizinga

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include <limits>
#include <random>

class Random {
 public:
  Random() = delete;
  Random(std::uint64_t seed = 0);

  static Random& GetRng(void);

  template <typename T = std::uint64_t>
  T RandInt(T max = std::numeric_limits<T>::max()) {
    static_assert(std::is_integral<T>::value, "Integral type required");
    std::uniform_int_distribution<T> uni_dist(T(0), max - T(1));
    return uni_dist(rand_engine_);
  }

  template <typename T = float>
  T RandFlt(T max = std::numeric_limits<T>::max()) {
    static_assert(std::is_floating_point<T>::value,
                  "Floating point type required");
    std::uniform_real_distribution<T> uni_dist(T(0), max);
    return uni_dist(rand_engine_);
  }

  template <typename T, int N = 3>
  T SparseRand() {
    std::uniform_int_distribution<T> uni_dist;
    T v = uni_dist(rand_engine_);
    for (int i = 1; i < N; i++) {
      v &= uni_dist(rand_engine_);
    }
    return v;
  }

  // UniformRandomBitGenerator interface
  constexpr static std::uint64_t min() {
    return std::numeric_limits<std::uint64_t>::min();
  }
  constexpr static std::uint64_t max() {
    return std::numeric_limits<std::uint64_t>::max();
  }
  std::uint64_t operator()();

 private:
  std::mt19937_64 rand_engine_;
};

