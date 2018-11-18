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

// Here are BLAS-free methods to setup the filter
// for the 3x3 winograd convolution algorithm.
//
// Ref:
//
// Fast Algorithms for Convolutional Neural Networks
// https://arxiv.org/abs/1509.09308
//
// https://ai.intel.com/winograd/
// https://ai.intel.com/winograd-2/

// Convolution filter for 3x3 Winograd algorithm
class WinogradFilter {
 public:
  WinogradFilter() = delete;

  // Create the zero-padded U matrix.
  static std::vector<float> ZeropadU(const std::vector<float>& U,
                                     const size_t outputs,
                                     const size_t channels,
                                     const size_t outputs_pad,
                                     const size_t channels_pad);

  // Create the filter transform matrix.
  static std::vector<float> TransformF(const std::vector<float>& f,
                                       const size_t outputs,
                                       const size_t channels);

 private:
  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kSquares = kWidth * kHeight;

  static constexpr auto kWtiles = (kWidth + 1) / 2;  // 4
  static constexpr auto kTiles = kWtiles * kWtiles;  // 16

  static constexpr auto kWinogradAlpha = 4;
  static constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;
};
}  // namespace lczero
