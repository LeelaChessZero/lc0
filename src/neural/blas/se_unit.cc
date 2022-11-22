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

#include "neural/blas/se_unit.h"
#include "neural/blas/fully_connected_layer.h"

#include <cmath>

namespace lczero {
namespace {
constexpr int kWidth = 8;
constexpr int kHeight = 8;
constexpr int kSquares = kWidth * kHeight;
}  // namespace

static void global_avg_pooling(size_t batches, size_t channels,
                               const float* input, const float* bias,
                               float* output) {
  for (auto b = size_t{0}; b < batches; b++) {
    for (auto ch = size_t{0}; ch < channels; ch++) {
      auto c = b * channels + ch;
      auto acc = 0.0f;
      for (auto i = size_t{0}; i < kSquares; i++) {
        acc += input[c * kSquares + i];
      }
      output[c] = acc / kSquares + bias[ch];
    }
  }
}

static void apply_se(const size_t channels, const size_t batch_size,
                     const float* input, const float* bias, const float* res,
                     const float* scale, float* output,
                     const ActivationFunction activation) {
  const auto lambda_sigmoid = [](const auto val) {
    return 1.0f / (1.0f + std::exp(-val));
  };

  for (auto batch = size_t{0}; batch < batch_size; batch++) {
    for (auto ch = size_t{0}; ch < channels; ch++) {
      auto c = channels * batch + ch;
      auto gamma = lambda_sigmoid(scale[c + batch * channels]);
      auto beta = scale[c + batch * channels + channels] + gamma * bias[ch];
      Activate(kSquares, gamma, &input[c * kSquares], &res[c * kSquares], beta,
               &output[c * kSquares], activation);
    }
  }
}

template <bool use_eigen>
void ApplySEUnit(const size_t batch_size, const size_t channels,
                 const size_t se_fc_outputs, const float* input,
                 const float* ch_bias, const float* residual,
                 const float* weights_w1, const float* weights_b1,
                 const float* weights_w2, const float* weights_b2,
                 float* output, const ActivationFunction activation) {
  std::vector<float> pool(2 * channels * batch_size);
  std::vector<float> fc_out1(batch_size * se_fc_outputs);

  global_avg_pooling(batch_size, channels, input, ch_bias, pool.data());

  FullyConnectedLayer<use_eigen>::Forward1D(batch_size, channels, se_fc_outputs,
                                            pool.data(), weights_w1, weights_b1,
                                            activation,  // Activation On
                                            fc_out1.data());

  FullyConnectedLayer<use_eigen>::Forward1D(batch_size, se_fc_outputs,
                                            2 * channels, fc_out1.data(),
                                            weights_w2, weights_b2,
                                            NONE,  // Activation Off
                                            pool.data());

  // Sigmoid, scale and add residual
  apply_se(channels, batch_size, input, ch_bias, residual, pool.data(), output,
           activation);
}

template void ApplySEUnit<true>(const size_t batch_size, const size_t channels,
                                const size_t se_fc_outputs, const float* input,
                                const float* bias, const float* residual,
                                const float* weights_w1,
                                const float* weights_b1,
                                const float* weights_w2,
                                const float* weights_b2, float* output,
                                const ActivationFunction activation);
#ifdef USE_BLAS
template void ApplySEUnit<false>(const size_t batch_size, const size_t channels,
                                 const size_t se_fc_outputs, const float* input,
                                 const float* bias, const float* residual,
                                 const float* weights_w1,
                                 const float* weights_b1,
                                 const float* weights_w2,
                                 const float* weights_b2, float* output,
                                 const ActivationFunction activation);
#endif
}  // namespace lczero
