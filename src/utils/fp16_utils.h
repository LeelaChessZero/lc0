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

// Define NO_F16C to avoid the F16C intrinsics. Also disabled with NO_POPCNT
// since it catches most processors without F16C instructions.
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || \
    defined(__x86_64__)
#include <immintrin.h>
#else
#define NO_F16C
#endif

namespace lczero {

#if defined(NO_POPCNT) || defined(NO_F16C) || \
    (defined(__GNUC__) && !defined(__F16C__))

uint16_t FP32toFP16(float f32);
float FP16toFP32(uint16_t f16);

#else

static inline uint16_t FP32toFP16(float f32) {
  __m128 A = _mm_set_ss(f32);
  __m128i H = _mm_cvtps_ph(A, 0);
  return _mm_extract_epi16(H, 0);
}

static inline float FP16toFP32(uint16_t f16) {
  __m128i H = _mm_setzero_si128();
  H = _mm_insert_epi16(H, f16, 0);
  __m128 A = _mm_cvtph_ps(H);
  return _mm_cvtss_f32(A);
}

#endif

}  // namespace lczero
