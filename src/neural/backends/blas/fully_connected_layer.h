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

#pragma once

#include "neural/shared/activation.h"

#include <cstddef>
#include <vector>

namespace lczero {

template <bool use_eigen>
class FullyConnectedLayer {
 public:
  FullyConnectedLayer() = delete;

  // Forward inference, batched, from input_size to output_size
  static void Forward1D(const size_t batch_size, const size_t input_size,
                        const size_t output_size, const float* input,
                        const float* weights, const float* biases,
                        const ActivationFunction activation, float* output);

  // Forward inference, no batched, from input_size to scalar
  static float Forward0D(const size_t input_size, const float* input,
                         const float* weights);

};

}  // namespace lczero
