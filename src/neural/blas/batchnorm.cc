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

#include "batchnorm.h"

#include "cmath"

namespace lczero {

  void Batchnorm::Apply(const int batch_size, const int channels,
                             float* data, const float* means,
                             const float* stddivs, const float* eltwise) {
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < channels; ++c) {
        auto mean = means[c];
        auto scale_stddiv = stddivs[c];
        
        if (eltwise == nullptr) {
          // Classical BN
          auto arr = &data[c * kSquares];
          for (int b = 0; b < kSquares; b++) {
            float val = scale_stddiv * (arr[b] - mean);
            arr[b] = val > 0 ? val : 0;
          }
        } else {
          // BN + residual add
          auto arr = &data[c * kSquares];
          auto res = &eltwise[c * kSquares];
          for (int b = 0; b < kSquares; b++) {
            float val = res[b] + (scale_stddiv * (arr[b] - mean));
            arr[b] = val > 0 ? val : 0;
          }
        }
      }
      data += channels * kSquares;
      if (eltwise != nullptr) eltwise += channels * kSquares;
    }
  }

  
  
  void Batchnorm::OffsetMeans(std::vector<float>& bn_means,
                                        const std::vector<float>& biases) {
    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    for (size_t i = 0; i < bn_means.size(); i++) bn_means[i] -= biases[i];
  }
  
  void Batchnorm::InvertStddev(std::vector<float>& weights) {
    for (auto& w : weights) w = 1.0f / std::sqrt(w + kEpsilon);
  }
  
} // namespace lczero

