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
*/

#include <cstdint>
#include <immintrin.h>

namespace lczero {

uint16_t FP32toFP16(float f32) {
#if 0
  return _cvtss_sh(f32, 0);
#else
  __m128 A = _mm_set_ss(f32);
  __m128i H = _mm_cvtps_ph(A, 0);
  return _mm_extract_epi16(H, 0);
#endif
}

float FP16toFP32(uint16_t f16) {
#if 0
  return _cvtsh_ss(f16);
#else
  __m128i H  = _mm_setzero_si128();
  H = _mm_insert_epi16(H, f16, 0);
  __m128 A = _mm_cvtph_ps(H);
  return _mm_cvtss_f32(A);
#endif
}

};  // namespace lczero
