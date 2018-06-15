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

class WinogradConvolution3 {
 public:
  static std::vector<float> ZeropadU(const std::vector<float>& U,
                                     const int outputs, const int channels,
                                     const int outputs_pad,
                                     const int channels_pad);

  // Transform the weights
  static std::vector<float> TransformF(const std::vector<float>& f,
                                       const int outputs, const int channels);

  // Allocate for the largest
  WinogradConvolution3(const int max_batch_size, const int max_input_layers,
                       const int max_output_layers);

  void Forward(const int batch_size, const int input_channels,
               const int output_channels, const float* input,
               const float* weights, float* output);

 private:
  void TransformIn(const int batch_size, const float* input,
                   const int channels);

  void Sgemm(const int batch_size, const float* weights,
             const int input_channels, const int output_channels);

  void TransformOut(const int batch_size, float* output, const int channels);

  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kWtiles = (kWidth + 1) / 2;  // 4
  static constexpr auto kTiles = kWtiles * kWtiles;  // 16

  static constexpr auto kWinogradAlpha = 4;
  static constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;

  static constexpr auto kSquares = kWidth * kHeight;

  std::vector<float> V_;
  std::vector<float> M_;
};
}
