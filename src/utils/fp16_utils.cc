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

uint16_t FP32toFP16(float f32) {
#if defined(NO_POPCNT) || defined(NO_F16C) || \
    (defined(__GNUC__) && !defined(__F16C__))
  unsigned int x;
  unsigned int sign = 0;
  memcpy(&x, &f32, sizeof(float));
  if (x & 0x80000000) sign = 0x8000;
  x &= 0x7fffffff;
  if (x >= 0x477ff000) {
    if ((x & 0x7f800000) == 0x7f800000 && (x & 0x7fffff)) {
      x = ((x >> 13) - 0x38000) | 0x200;
    } else {
      x = 0x7c00;
    }
  } else if (x <= 0x33000000)
    x = 0;
  else if (x <= 0x387fefff) {
    int shift = 126 - ((x >> 23) & 0xff);
    x = (x & 0x7fffff) | 0x800000;
    if (x & (0x17fffff >> (24 - shift))) x += 0x800000 >> (24 - shift);
    x >>= shift;
  } else {
    // Adjust exponent and round to nearest even.
    if (x & 0x2fff) {
      x -= 0x37fff000;
    } else {
      x -= 0x38000000;
    }
    x >>= 13;
  }
  return x | sign;
#else
  __m128 A = _mm_set_ss(f32);
  __m128i H = _mm_cvtps_ph(A, 0);
  return _mm_extract_epi16(H, 0);
#endif
}

float FP16toFP32(uint16_t f16) {
#if defined(NO_POPCNT) || defined(NO_F16C) || \
    (defined(__GNUC__) && !defined(__F16C__))
  unsigned int x;
  float f;
  x = f16 & 0x7fff;
  if ((x & 0x7c00) == 0) {
    f = 5.9604645e-8f * x;
    memcpy(&x, &f, sizeof(float));
  } else if (x >= 0x7c00) {
    if (x & 0x1ff) x |= 0x200;
    x = (x + 0x38000) << 13;
  } else {
    x = (x + 0x1c000) << 13;
  }
  if (f16 & 0x8000) x |= 0x80000000;
  memcpy(&f, &x, sizeof(float));
  return f;
#else
  __m128i H = _mm_setzero_si128();
  H = _mm_insert_epi16(H, f16, 0);
  __m128 A = _mm_cvtph_ps(H);
  return _mm_cvtss_f32(A);
#endif
}

}  // namespace lczero
