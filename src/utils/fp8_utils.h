/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once

#include <cstdint>
#include <cstring>

namespace lczero {

inline uint8_t FP32toFP8E5M2(float f32) {
  unsigned int x;
  unsigned int sign = 0;
  memcpy(&x, &f32, sizeof(float));
  if (x & 0x80000000) sign = 0x80;
  x &= 0x7fffffff;
  if (x >= 0x47700000) {
    if ((x & 0x7f800000) == 0x7f800000 && (x & 0x7fffff)) {
      x = ((x >> 21) - 0x380) | 0x2;
    } else {
      x = 0x7c;
    }
  } else if (x < 0x37000000) {
    x = 0;
  } else if (x < 0x38700000) {
    int shift = 134 - ((x >> 23) & 0xff);
    x = (x & 0x7fffff) | 0x800000;
    if (x & (0x17fffff >> (24 - shift))) x += 0x800000 >> (24 - shift);
    x >>= shift;
  } else {
    // Adjust exponent and round to nearest even.
    if (x & 0x2fffff) {
      x -= 0x37f00000;
    } else {
      x -= 0x38000000;
    }
    x >>= 21;
  }
  return x | sign;
}

inline float FP8E5M2toFP32(uint8_t f8) {
  unsigned int x;
  float f;
  x = f8 & 0x7f;
  if ((x & 0x7c) == 0) {
    f = 1.5258789e-5f * x;
    memcpy(&x, &f, sizeof(float));
  } else if (x >= 0x7c) {
    x = (x + 0x380) << 21;
  } else {
    x = (x + 0x1c0) << 21;
  }
  if (f8 & 0x80) x |= 0x80000000;
  memcpy(&f, &x, sizeof(float));
  return f;
}

}  // namespace lczero
