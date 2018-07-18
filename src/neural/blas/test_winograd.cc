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

#include <gtest/gtest.h>

#include "src/neural/blas/winograd_convolution3.h"
#include <iostream>
#include <random>

namespace lczero {

TEST(Winograd3, Winograd3) {
  {
    // check same output of
    // TransformInOrig and TransformInCo
    // when initialized with random input

    static constexpr auto kWidth = WinogradConvolution3::kWidth;
    static constexpr auto kHeight = WinogradConvolution3::kHeight;
    static constexpr auto kSquares = WinogradConvolution3::kSquares;

    static constexpr auto kWtiles = WinogradConvolution3::kWtiles;
    static constexpr auto kTiles = WinogradConvolution3::kTiles;

    static constexpr auto kWinogradAlpha = WinogradConvolution3::kWinogradAlpha;
    static constexpr auto kWinogradTile = WinogradConvolution3::kWinogradTile;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distF(0.0, 1.0);
    std::uniform_real_distribution<> distI(30, 40);

    const uint32_t bs_h = distI(gen);
    const uint32_t channels = distI(gen);
    std::vector<float> in(bs_h * kWidth * kHeight * kTiles * channels, 1.0);

    // init in vector with random in range (0,1.0)
    for (size_t i = 0; i < in.size(); ++i) {
      in[i] = distF(gen);
    }

    const size_t P = channels * kTiles * kTiles * bs_h;
    const float testK1 =  distF(gen);
    std::vector<float> out1(P + 1, 0);
    out1[P] = testK1;
    WinogradConvolution3::TransformInOrig(bs_h, &in[0], channels, &out1[0]);
    EXPECT_EQ(out1[P], testK1); // last value over range is not written

    const float testK2 =  distF(gen);
    std::vector<float> out2(out1.size(), 0);
    out2[P] = testK2;
    WinogradConvolution3::TransformInCo(bs_h, &in[0], channels, &out2[0]);
    EXPECT_EQ(out2[P], testK2);

    size_t diff = 0;
    for (size_t i = 0; i < P; ++i) {
      if (out1[i] != out2[i]) {
        ++diff;
      }
    }
    EXPECT_EQ(diff, 0); // all values in both vectors same
  }
}
} // namespace lczero

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
