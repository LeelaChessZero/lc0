/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2022 The LCZero Authors

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

#include <cstddef>
#include <vector>

namespace lczero {
// The following list matches the one in net.proto. Ideally this would be done
// by including proto/net.pb.h, but this is incompatible with nvcc.
enum ActivationFunction {
  ACTIVATION_DEFAULT = 0,
  ACTIVATION_MISH = 1,
  ACTIVATION_RELU = 2,
  ACTIVATION_NONE = 3,
  ACTIVATION_TANH = 4,
  ACTIVATION_SIGMOID = 5,
  ACTIVATION_SELU = 6,
  ACTIVATION_SWISH = 7,
  ACTIVATION_RELU_2 = 8,
  ACTIVATION_SOFTMAX = 9,
};

struct Activations {
    ActivationFunction default_activation = ACTIVATION_RELU;
    ActivationFunction smolgen_activation = ACTIVATION_SWISH;
    ActivationFunction ffn_activation = ACTIVATION_RELU_2;
};

// Softmax activation
void SoftmaxActivation(const size_t size, const float* input, float* output);

void BiasResidual(const size_t batch_size, const size_t channels, float* data,
                  const float* biases, const float* eltwise,
                  const ActivationFunction activation);

void BiasActivate(const size_t batch_size, const size_t channels, float* data,
                  const float* biases,
                  const ActivationFunction activation);

float Activate(const float val, const ActivationFunction activation);

void Activate(const size_t len, const float* data, const float* bias,
              float* output, const ActivationFunction activation);

void Activate(const size_t len, float gamma, const float* data,
              const float* bias, float beta, float* out,
              const ActivationFunction activation);

}  // namespace lczero
