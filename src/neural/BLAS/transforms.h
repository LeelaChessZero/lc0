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
  static constexpr auto kWinogradAlpha = 4;
  static constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;

  static std::vector<float> ZeropadU(const std::vector<float>& U,
                                     const int outputs, const int channels,
                                     const int outputs_pad,
                                     const int channels_pad);

  static std::vector<float> WinogradTransformF(const std::vector<float>& f,
                                               const int outputs,
                                               const int channels);

  static void WinogradTransformIn(const std::vector<float>& in,
                                  std::vector<float>& V, const int C);

  static void WinogradSgemm(const std::vector<float>& U, std::vector<float>& V,
                            std::vector<float>& M, const int C, const int K);

  static void WinogradTransformOut(const std::vector<float>& M,
                                   std::vector<float>& Y, const int K);

  static void WinogradConvolve3(const int outputs,
                                const std::vector<float>& input,
                                const std::vector<float>& U,
                                std::vector<float>& V, std::vector<float>& M,
                                std::vector<float>& output);

  template <unsigned int filter_size>
  static void Convolve(size_t outputs, const std::vector<float>& input,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases,
                       std::vector<float>& output);

  static void Innerproduct(const std::vector<float>& input,
                           const std::vector<float>& weights,
                           const std::vector<float>& biases,
                           std::vector<float>& output, bool apply_relu = false);

  template <size_t spatial_size>
  static void Batchnorm(size_t channels, std::vector<float>& data,
                        const float* means, const float* stddivs,
                        const float* eltwise = nullptr);

  template <unsigned long filter_size>
  static void Im2Col(const int channels, const std::vector<float>& input,
                     std::vector<float>& output);

  static void Softmax(const std::vector<float>& input,
                      std::vector<float>& output);

  static float Innerproduct(const std::vector<float>& x,
                            const std::vector<float>& y);

  static void OffsetBatchNormMeans(std::vector<float>& bn_means,
                                   const std::vector<float>& biases);

  static void InvertBatchNormStddev(std::vector<float>& weights);
};

}  // lczero
