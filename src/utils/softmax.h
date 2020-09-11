/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#pragma once

#include <cmath>
#include <tuple>

#include "utils/fastmath.h"

namespace lczero {

inline std::tuple<float, std::array<float, 256>> RelativeEntropySoftmax(
    std::array<float, 256>& q, std::array<float, 256>& p, int length,
    float temperature = 1.0) {
  float sum = 0.0;
  std::array<float, 256> new_policy;
  float max_p = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < length; i++) {
    new_policy[i] = q[i] / temperature;
    max_p = std::max(max_p, new_policy[i]);
  }
  for (int i = 0; i < length; i++) {
    new_policy[i] = p[i] * FastExp(new_policy[i] - max_p);
    sum += new_policy[i];
  }
  for (int i = 0; i < length; i++) {
    new_policy[i] /= sum;
  }
  return {FastLog(sum) + max_p, new_policy};
}
}  // namespace lczero