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

#include "neural/shared/winograd_filter.h"

#include <array>

namespace lczero {
namespace {

static constexpr auto kWinogradAlpha = 4;
static constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;

}  // namespace

std::vector<float> WinogradFilterZeropadU(const std::vector<float>& U,
                                          const size_t outputs,
                                          const size_t channels,
                                          const size_t outputs_pad,
                                          const size_t channels_pad) {
  // Fill with zeroes.
  auto Upad = std::vector<float>(kWinogradTile * outputs_pad * channels_pad);

  for (size_t o = 0; o < outputs; o++) {
    for (size_t c = 0; c < channels; c++) {
      for (size_t xi = 0; xi < kWinogradAlpha; xi++) {
        for (size_t nu = 0; nu < kWinogradAlpha; nu++) {
          Upad[xi * (kWinogradAlpha * outputs_pad * channels_pad) +
               nu * (outputs_pad * channels_pad) + c * outputs_pad + o] =
              U[xi * (kWinogradAlpha * outputs * channels) +
                nu * (outputs * channels) + c * outputs + o];
        }
      }
    }
  }
  return Upad;
}

std::vector<float> WinogradFilterTransformF(const std::vector<float>& f,
                                            const size_t outputs,
                                            const size_t channels) {
  // F(2x2, 3x3) Winograd filter transformation
  // transpose(G.dot(f).dot(G.transpose()))
  // U matrix is transposed for better memory layout in SGEMM
  auto U = std::vector<float>(kWinogradTile * outputs * channels);
  auto G = std::array<float, kWinogradTile>{1.0, 0.0,  0.0, 0.5, 0.5, 0.5,
                                            0.5, -0.5, 0.5, 0.0, 0.0, 1.0};
  auto temp = std::array<float, 12>{};

  for (size_t o = 0; o < outputs; o++) {
    for (size_t c = 0; c < channels; c++) {
      for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 3; j++) {
          auto acc = 0.0f;
          for (size_t k = 0; k < 3; k++) {
            acc += G[i * 3 + k] * f[o * channels * 9 + c * 9 + k * 3 + j];
          }
          temp[i * 3 + j] = acc;
        }
      }

      for (size_t xi = 0; xi < 4; xi++) {
        for (size_t nu = 0; nu < 4; nu++) {
          auto acc = 0.0f;
          for (size_t k = 0; k < 3; k++) {
            acc += temp[xi * 3 + k] * G[nu * 3 + k];
          }
          U[xi * (4 * outputs * channels) + nu * (outputs * channels) +
            c * outputs + o] = acc;
        }
      }
    }
  }
  return U;
}

}  // namespace lczero
