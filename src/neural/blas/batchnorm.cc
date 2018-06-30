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

#include "neural/blas/batchnorm.h"

#include <cmath>

namespace lczero {

void Batchnorm::Apply(const size_t batch_size, const size_t channels,
                      float* data, const float* means, const float* stddivs,
                      const float* eltwise) {
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t c = 0; c < channels; ++c) {
      auto mean = means[c];
      auto scale_stddiv = stddivs[c];

      if (eltwise == nullptr) {
        // Classical BN
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = scale_stddiv * (arr[b] - mean);
          arr[b] = val > 0 ? val : 0;
        }
      } else {
        // BN + residual add
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + (scale_stddiv * (arr[b] - mean));
          arr[b] = val > 0 ? val : 0;
        }
      }
    }
    data += channels * kSquares;
    if (eltwise != nullptr) eltwise += channels * kSquares;
  }
}

void Batchnorm::InvertStddev(Weights::ConvBlock* conv) {
  std::vector<float>& stddivs = conv->bn_stddivs;
  InvertStddev(stddivs.size(), stddivs.data());
}

void Batchnorm::OffsetMeans(Weights::ConvBlock* conv) {
  std::vector<float>& means = conv->bn_means;
  const std::vector<float>& biases = conv->biases;
  OffsetMeans(means.size(), means.data(), biases.data());
}

std::vector<float> Batchnorm::InvertStddev(const Weights::ConvBlock& conv) {
  std::vector<float> stddivs = conv.bn_stddivs;  // copy
  InvertStddev(stddivs.size(), stddivs.data());
  return stddivs;
}

std::vector<float> Batchnorm::OffsetMeans(const Weights::ConvBlock& conv) {
  std::vector<float> means = conv.bn_means;  // copy
  const std::vector<float>& biases = conv.biases;
  OffsetMeans(means.size(), means.data(), biases.data());
  return means;
}

void Batchnorm::InvertStddev(const size_t size, float* array) {
  for (size_t i = 0; i < size; i++) {
    auto safe_value=std::max(array[i], (float) kEpsilon);
    array[i] = 1.0f / std::sqrt(safe_value);
  }
}

void Batchnorm::OffsetMeans(const size_t size, float* means,
                            const float* biases) {
  for (size_t i = 0; i < size; i++) means[i] -= biases[i];
}

}  // namespace lczero
