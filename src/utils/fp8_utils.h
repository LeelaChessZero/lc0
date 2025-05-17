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

#include <cmath>
#include <cstdint>
#include <cstring>

// Define NO_F16C to avoid the F16C intrinsics. Also disabled with NO_POPCNT
// since it catches most processors without F16C instructions.
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || \
    defined(__x86_64__)
#include <immintrin.h>
#else
#define NO_F16C
#endif

namespace lczero {

inline uint8_t FP32toFP8E5M2(float f32, bool saturate = true) {
  unsigned int x;
  unsigned int sign = 0;
  memcpy(&x, &f32, sizeof(float));
  if (x & 0x80000000) sign = 0x80;
  x &= 0x7fffffff;
  if (x >= 0x47700000) {
    if (x > 0x7f800000) {
      x = 0x7f;
    } else if (saturate) {
      x = 0x7b;
    } else {
      x = 0x7c;
    }
  } else if (x < 0x37000000) {
    x = 0;
  } else if (x < 0x38700000) {
    float t = 128 + std::abs(f32);
    memcpy(&x, &t, sizeof(float));
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

inline uint8_t FP32toFP8E5M2_Saturate(float f32) {
  return FP32toFP8E5M2(f32, true);
}

#if defined(NO_POPCNT) || defined(NO_F16C) || \
    (defined(__GNUC__) && !defined(__F16C__))

inline float FP8E5M2toFP32(uint8_t f8) {
  unsigned int x;
  float f;
  x = f8 & 0x7f;
  if ((x & 0x7c) == 0) {
    f = 1.5258789e-5f * x;
    memcpy(&x, &f, sizeof(float));
  } else if (x >= 0x7c) {
    if (x & 0x1) x |= 0x2;
    x = (x + 0x380) << 21;
  } else {
    x = (x + 0x1c0) << 21;
  }
  if (f8 & 0x80) x |= 0x80000000;
  memcpy(&f, &x, sizeof(float));
  return f;
}

#else

inline float FP8E5M2toFP32(uint8_t f8) {
  __m128i H = _mm_setzero_si128();
  H = _mm_insert_epi8(H, f8, 1);
  __m128 A = _mm_cvtph_ps(H);
  return _mm_cvtss_f32(A);
}

#endif

inline uint8_t FP32toFP8E4M3FN(float f32, bool saturate = true) {
  unsigned int x;
  unsigned int sign = 0;
  memcpy(&x, &f32, sizeof(float));
  if (x & 0x80000000) sign = 0x80;
  x &= 0x7fffffff;
  if (x >= 0x43e80000) {
    if (saturate && x <= 0x7f800000) {
      x = 0x7e;
    } else {
      x = 0x7f;
    }
  } else if (x < 0x3a800000) {
    x = 0;
  } else if (x < 0x3c800000) {
    float t = 16384 + std::abs(f32);
    memcpy(&x, &t, sizeof(float));
  } else {
    // Adjust exponent and round to nearest even.
    if (x & 0x17ffff) {
      x -= 0x3bf80000;
    } else {
      x -= 0x3c000000;
    }
    x >>= 20;
  }
  return x | sign;
}

inline float FP8E4M3FNtoFP32(uint8_t f8) {
  unsigned int x;
  float f;
  x = f8 & 0x7f;
  if ((x & 0x78) == 0) {
    f = .001953125f * x;
    memcpy(&x, &f, sizeof(float));
  } else if (x == 0x7f) {
    x = 0x7ff00000;
  } else {
    x = (x + 0x3c0) << 20;
  }
  if (f8 & 0x80) x |= 0x80000000;
  memcpy(&f, &x, sizeof(float));
  return f;
}

}  // namespace lczero
