/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors

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
 */

#include "neural/shared/activation.h"

#include <algorithm>
#include <cmath>

namespace lczero {

void SoftmaxActivation(const size_t size, const float* input, float* output) {
  auto alpha = *std::max_element(input, input + size);

  auto denom = 0.0f;
  for (size_t i = 0; i < size; i++) {
    auto val = std::exp(input[i] - alpha);
    output[i] = val;
    denom += val;
  }
  for (size_t i = 0; i < size; i++) {
    output[i] = output[i] / denom;
  }
}
}  // namespace lczero
