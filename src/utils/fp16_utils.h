/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

// Define NO_F16C to avoid the F16C intrinsics. Also disabled with NO_POPCNT
// since it catches most processors without F16C instructions.
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || \
    defined(__x86_64__)
#include <immintrin.h>
#else
#define NO_F16C
#endif

namespace lczero {

#if defined(HAS_FLOAT16) && (defined(__F16C__) || defined(__aarch64__))

inline uint16_t FP32toFP16(float f32) {
  _Float16 f16 = static_cast<_Float16>(f32);
  uint16_t x;
  std::memcpy(&x, &f16, sizeof(uint16_t));
  return x;
}

inline float FP16toFP32(uint16_t f16) {
  _Float16 x;
  std::memcpy(&x, &f16, sizeof(uint16_t));
  return static_cast<float>(x);
}

#elif !defined(NO_POPCNT) && !defined(NO_F16C) && \
    (!defined(__GNUC__) || defined(__F16C__))

inline uint16_t FP32toFP16(float f32) {
  __m128 A = _mm_set_ss(f32);
  __m128i H = _mm_cvtps_ph(A, 0);
  return _mm_extract_epi16(H, 0);
}

inline float FP16toFP32(uint16_t f16) {
  __m128i H = _mm_setzero_si128();
  H = _mm_insert_epi16(H, f16, 0);
  __m128 A = _mm_cvtph_ps(H);
  return _mm_cvtss_f32(A);
}

#else

inline uint16_t FP32toFP16(float f32) {
  uint32_t x;
  uint32_t sign = 0;
  memcpy(&x, &f32, sizeof(float));
  if (x & 0x80000000) sign = 0x8000;
  x &= 0x7fffffff;
  if (x < 0x477ff000) {
    if (x >= 0x387ff000) {
      // Normal fp16 result. Adjust exponent and round to nearest even.
      x -= 0x38000000;
      if (x & 0x2fff) {
        x += 0x1000;
      }
      x >>= 13;
    } else {
      // Subnormal or zero. The result is the last bits of fabs(f32) + 0.5f.
      float f;
      memcpy(&f, &x, sizeof(float));
      f += 0.5f;
      memcpy(&x, &f, sizeof(float));
    }
  } else {
    if (x > 0x7f800000) {
      // NaN
      x = ((x >> 13) - 0x38000) | 0x200;
    } else {
      // Inf
      x = 0x7c00;
    }
  }
  return x | sign;
}

inline float FP16toFP32(uint16_t f16) {
  int32_t s = static_cast<int16_t>(f16);
  uint32_t x;
  float f;
  if ((s & 0x7c00) == 0) {
    // Subnormal or zero. Scale to float.
    x = s & 0x7fff;
    f = 5.9604645e-8f * x;
    memcpy(&x, &f, sizeof(float));
    if (s & 0x8000) x |= 0x80000000;
  } else if ((s & 0x7c00) == 0x7c00) {
    // Inf or NaN. Adjust exponent and shift.
    if (s & 0x1ff) s |= 0x200; // Change sNaN to qNaN as intel does.
    x = ((s & 0x47fff) + 0x38000) << 13;
  } else {
    // Normal. Adjust exponent and shift.
    x = ((s & 0x47fff) + 0x1c000U) << 13;
  }
  memcpy(&f, &x, sizeof(float));
  return f;
}

#endif

}  // namespace lczero
