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
namespace {
constexpr int kWidth = 8;
constexpr int kHeight = 8;
constexpr int kSquares = kWidth * kHeight;
}  // namespace

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

float Activate(const float val, const ActivationFunction activation) {
  switch (activation) {
    case RELU:
      return val > 0 ? val : 0;
    case MISH: {
      auto e = expf(val);
      auto n = e * e + 2.0f * e;
      auto d = val / (n + 2.0f);
      if (val <= -0.6f) {
        return n * d;
      } else {
        return val - 2.0f * d;
      }
    }
    case TANH:
      return tanhf(val);
    case SIGMOID:
      return 1.0f / (1.0f + expf(-val));
    case SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (val > 0) {
        return scale * val;
      } else {
        return scale * alpha * (expf(val) - 1.0f);
      }
    }
  }
  return val;
}

void Activate(const size_t len, float* data,
              const ActivationFunction activation) {
  for (size_t i = 0; i < len; i++) {
    data[i] = Activate(data[i], activation);
  }
}

void BiasResidualRelu(const size_t batch_size, const size_t channels,
                 float* data, const float* biases,
                 const float* eltwise,
                 const ActivationFunction activation) {
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t c = 0; c < channels; ++c) {
      auto bias = biases[c];

      if (eltwise == nullptr) {
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = arr[b] + bias;
          if (activation != NONE) {
            val = Activate(val, activation);
          }
          arr[b] = val;
        }
      } else {
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + arr[b] + bias;
          if (activation != NONE) {
            val = Activate(val, activation);
          }
          arr[b] = val;
        }
      }
    }
    data += channels * kSquares;
    if (eltwise != nullptr) eltwise += channels * kSquares;
  }
}
}  // namespace lczero
