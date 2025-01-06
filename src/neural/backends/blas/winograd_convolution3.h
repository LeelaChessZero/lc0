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

#include <cstddef>
#include <vector>

namespace lczero {

// Convolution 3x3 on a 8x8 board using the Winograd algorithm.
//
// Ref:
//
// Fast Algorithms for Convolutional Neural Networks
// https://arxiv.org/abs/1509.09308
//
// https://ai.intel.com/winograd/
// https://ai.intel.com/winograd-2/

// Convolution 3x3 using the Winograd algorithm
template <bool use_eigen>
class WinogradConvolution3 {
 public:
  // The instance will allocate memory resources for the
  // largest batch size, and the largest input and output
  // layers.
  WinogradConvolution3(const size_t max_batch_size,
                       const size_t max_input_layers,
                       const size_t max_output_layers);

  // Forward inference, batched.
  void Forward(const size_t batch_size, const size_t input_channels,
               const size_t output_channels, const float* input,
               const float* weights, float* output);

 private:
  void TransformIn(const size_t batch_size, const float* input,
                   const size_t channels);

  void Sgemm(const size_t batch_size, const float* weights,
             const size_t input_channels, const size_t output_channels);

  void TransformOut(const size_t batch_size, float* output,
                    const size_t channels);

  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kSquares = kWidth * kHeight;

  static constexpr auto kWtiles = (kWidth + 1) / 2;  // 4
  static constexpr auto kTiles = kWtiles * kWtiles;  // 16

  static constexpr auto kWinogradAlpha = 4;
  static constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;

  std::vector<float> V_;
  std::vector<float> M_;
};
}  // namespace lczero
