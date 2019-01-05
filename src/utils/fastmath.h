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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <cstring>

namespace lczero {
// These stunts are performed by trained professionals, do not try this at home.

// Fast approximate log2(x). Does no range checking.
// The approximation used here is log2(2^N*(1+f)) ~ N+f*(1.342671-0.342671*f)
// where N is the integer and f the fractional part, f>=0.
inline float FastLog2(const float a) {
  uint32_t tmp;
  std::memcpy(&tmp, &a, sizeof(float));
  uint32_t expb = tmp >> 23;
  tmp = (tmp & 0x7fffff) | (0x7f << 23);
  float out;
  std::memcpy(&out, &tmp, sizeof(float));
  return out * (2.028011f - 0.342671f * out) - 128.68534f + expb;
}

// Fast approximate 2^x. Does only limited range checking.
// The approximation used here is 2^(N+f) ~ 2^N*(1+f*(0.656366+0.343634*f))
// where N is the integer and f the fractional part, f>=0.
inline float FastPow2(const float a) {
  if (a < -126) return 0.0;
  int32_t exp = floor(a);
  float out = a - exp;
  out = 1.0f + out * (0.656366f + 0.343634f * out);
  int32_t tmp;
  std::memcpy(&tmp, &out, sizeof(float));
  tmp += static_cast<int32_t>(static_cast<uint32_t>(exp) << 23);
  std::memcpy(&out, &tmp, sizeof(float));
  return out;
}

// Fast approximate ln(x). Does no range checking.
inline float FastLog(const float a) {
  return 0.6931471805599453f * FastLog2(a);
}

}  // namespace lczero
