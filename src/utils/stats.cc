/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "stats.h"
#include <boost/math/distributions/students_t.hpp>

namespace lczero {

namespace {
auto constexpr kzEntries = 1000;
std::array<float, kzEntries> kzLookup;
}  // namespace

void CreatezTable(float ci_alpha) {
  for (auto i = 1; i < kzEntries + 1; i++) {
    boost::math::students_t dist(i);
    auto z = boost::math::quantile(boost::math::complement(dist, ci_alpha));
    kzLookup[i - 1] = z;
  }
}

float CachedtQuantile(int v) {
  if (v < 1) {
    return kzLookup[0];
  }
  if (v < kzEntries) {
    return kzLookup[v - 1];
  }
  // z approaches constant when v is high enough.
  // With default lookup table size the function is flat enough that we
  // can just return the last entry for all v bigger than it.
  return kzLookup[kzEntries - 1];
}

}  // namespace lczero
