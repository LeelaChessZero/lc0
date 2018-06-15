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

#include <array>
#include <vector>

namespace lczero {

class Transforms {
 public:
  
   template <unsigned int filter_size>
  static void Convolve(const int batch_size, const int input_channels,
                       const int output_channels, const float* input,
                       const float* weights, const float* biases,
                       float* output);

  static void Innerproduct(const int batch_size, const int input_size,
                           const int output_size, const float* input,
                           const float* weights, const float* biases,
                           bool apply_relu, float* output);

  static void Batchnorm(const int batch_size, const int channels, float* data,
                        const float* means, const float* stddivs,
                        const float* eltwise = nullptr);

  template <unsigned long filter_size>
  static void Im2Col(const int channels, const float* input, float* output);

  static void Softmax(const int size, const float* input, float* output);

  static float DotProduct(const int size, const float* x, const float* y);

  static void OffsetBatchNormMeans(std::vector<float>& bn_means,
                                   const std::vector<float>& biases);

  static void InvertBatchNormStddev(std::vector<float>& weights);
  
  
private:
  
  
  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kSquares = kWidth * kHeight;

  
};

}  // lczero
