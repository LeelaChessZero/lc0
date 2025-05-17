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
namespace lczero {

static inline uint16_t FP32toBF16(float f32) {
  uint32_t x;
  memcpy(&x, &f32, sizeof(float));
#ifndef BF16_TRUNC
  // Make sure NaN stays a NaN (the same way intel does fp16 conversion).
  if ((x & 0x7f800000) == 0x7f800000 && (x & 0x7fffff)) return (x >> 16) | 0x40;
  // Round to nearest even number.
  if (x & 0x17fff) x += 0x8000;
#endif
  return x >> 16;
}

static inline float BF16toFP32(uint16_t bf16) {
  uint32_t x = static_cast<uint32_t>(bf16) << 16;
  float f32;
  memcpy(&f32, &x, sizeof(float));
  return f32;
}

}  // namespace lczero
