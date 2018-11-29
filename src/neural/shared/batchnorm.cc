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

#include "neural/shared/batchnorm.h"

namespace lczero {
namespace {
constexpr int kWidth = 8;
constexpr int kHeight = 8;
constexpr int kSquares = kWidth * kHeight;
}  // namespace

void ApplyBatchNormalization(const size_t batch_size, const size_t channels,
                             float* data, const float* means,
                             const float* stddivs, const float* eltwise,
                             const bool relu) {
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t c = 0; c < channels; ++c) {
      auto mean = means[c];
      auto scale_stddiv = stddivs[c];

      if (eltwise == nullptr) {
        // Classical BN
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = scale_stddiv * (arr[b] - mean);
          if (relu) {
            val = val > 0 ? val : 0;
          }
          arr[b] = val;
        }
      } else {
        // BN + residual add
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + (scale_stddiv * (arr[b] - mean));
          if (relu) {
            val = val > 0 ? val : 0;
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
