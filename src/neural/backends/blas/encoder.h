/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2022-2023 The LCZero Authors

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

#include <cmath>

#include "neural/shared/activation.h"

#ifdef USE_ISPC
#include "layer_norm_ispc.h"
#endif

namespace lczero {

void LayerNorm2DWithSkipConnection(const size_t batch_size,
                                   const size_t channels, float* data,
                                   const float alpha, const float* skip,
                                   const float* gammas, const float* betas,
                                   float epsilon) {
  for (size_t i = 0; i < batch_size; i++) {
#ifndef USE_ISPC
    // Mean taken in dimension C.
    float mean = 0;
    if (skip != nullptr) {
      for (size_t c = 0; c < channels; ++c) {
        data[i * channels + c] =
            data[i * channels + c] * alpha + skip[i * channels + c];
        mean += data[i * channels + c];
      }
    } else {
      for (size_t c = 0; c < channels; ++c) {
        data[i * channels + c] *= alpha;
        mean += data[i * channels + c];
      }
    }
    mean /= channels;

    // Variance.
    float var = 0;
    for (size_t c = 0; c < channels; ++c) {
      auto diff = data[i * channels + c] - mean;
      var += diff * diff;
    }
    var /= channels;

    // Norm.
    float den = 1.0f / std::sqrt(var + epsilon);
    for (size_t c = 0; c < channels; ++c) {
      data[i * channels + c] =
          betas[c] + gammas[c] * (data[i * channels + c] - mean) * den;
    }
#else
    if (skip != nullptr) {
      ispc::LayerNorm2DWithSkipConnection(channels, data + i * channels, alpha,
                                          skip + i * channels, gammas, betas,
                                          epsilon);
    } else {
      ispc::LayerNorm2DWithSkipConnection(channels, data + i * channels, alpha,
                                          nullptr, gammas, betas, epsilon);
    }

#endif
  }
}

}  // namespace lczero
