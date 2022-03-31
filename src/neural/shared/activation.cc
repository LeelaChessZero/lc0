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

static inline float mish(float val) {
  auto e = expf(val);
  auto n = e * e + 2.0f * e;
  auto d = val / (n + 2.0f);
  if (val <= -0.125f) {
    return n * d;
  } else {
    return val - 2.0f * d;
  }
}

static inline float selu(float val) {
  float alpha = 1.67326324f, scale = 1.05070098f;
  if (val > 0) {
    return scale * val;
  } else {
    return scale * alpha * (expf(val) - 1.0f);
  }
}

float Activate(const float val, const ActivationFunction activation) {
  switch (activation) {
    case RELU:
      return val > 0 ? val : 0;
    case MISH:
      return mish(val);
    case TANH:
      return tanhf(val);
    case SIGMOID:
      return 1.0f / (1.0f + expf(-val));
    case SELU:
      return selu(val);
    case NONE:
      // Nothing to do.
      break;
  }
  return val;
}

void Activate(const size_t len, const float* data, const float* bias,
              float* output, const ActivationFunction activation) {
  if (activation == NONE) {
    for (size_t b = 0; b < len; b++) {
      output[b] = data[b] + bias[b];
    }
  } else if (activation == RELU) {
    for (size_t b = 0; b < len; b++) {
      float val = data[b] + bias[b];
      output[b] = val > 0 ? val : 0;
    }
  } else if (activation == MISH) {
    for (size_t b = 0; b < len; b++) {
      float val = data[b] + bias[b];
      output[b] = mish(val);
    }
  } else {
    for (size_t b = 0; b < len; b++) {
      float val = data[b] + bias[b];
      output[b] = Activate(val, activation);
    }
  }
}

void Activate(const size_t len, float gamma, const float* data,
              const float* bias, float beta, float* output,
              const ActivationFunction activation) {
  if (activation == NONE) {
    for (size_t b = 0; b < len; b++) {
      float val = gamma * data[b] + bias[b] + beta;
      output[b] = val;
    }
  } else if (activation == RELU) {
    for (size_t b = 0; b < len; b++) {
      float val = gamma * data[b] + bias[b] + beta;
      output[b] = val > 0 ? val : 0;
    }
  } else if (activation == MISH) {
    for (size_t b = 0; b < len; b++) {
      float val = gamma * data[b] + bias[b] + beta;
      output[b] = mish(val);
    }
  } else {
    for (size_t b = 0; b < len; b++) {
      float val = gamma * data[b] + bias[b] + beta;
      output[b] = Activate(val, activation);
    }
  }
}

void BiasResidual(const size_t batch_size, const size_t channels, float* data,
                  const float* biases, const float* eltwise,
                  const ActivationFunction activation) {
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t c = 0; c < channels; ++c) {
      auto bias = biases[c];
      if (activation == NONE) {
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + arr[b] + bias;
          arr[b] = val;
        }
      } else if (activation == RELU) {
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + arr[b] + bias;
          arr[b] = val > 0 ? val : 0;
        }
      } else if (activation == MISH) {
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + arr[b] + bias;
          arr[b] = mish(val);
        }
      } else {
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + arr[b] + bias;
          arr[b] = Activate(val, activation);
        }
      }
    }
    data += channels * kSquares;
    eltwise += channels * kSquares;
  }
}

void BiasActivate(const size_t batch_size, const size_t channels, float* data,
                  const float* biases, const ActivationFunction activation) {
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t c = 0; c < channels; ++c) {
      auto bias = biases[c];
      if (activation == NONE) {
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = arr[b] + bias;
          arr[b] = val;
        }
      } else if (activation == RELU) {
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = arr[b] + bias;
          arr[b] = val > 0 ? val : 0;
        }
      } else if (activation == MISH) {
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = arr[b] + bias;
          arr[b] = mish(val);
        }
      } else {
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = arr[b] + bias;
          arr[b] = Activate(val, activation);
        }
      }
    }
    data += channels * kSquares;
  }
}

}  // namespace lczero
