/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

namespace lczero {
// These stunts are performed by trained professionals, do not try this at home.

// Fast approximate log2(x). Does no range checking.
// The approximation used here is log2(2^N*(1+f)) ~ N+f*(1+k-k*f) where N is the
// exponent and f the fraction (mantissa), f>=0. The constant k is used to tune
// the approximation accuracy. In the final version some constants were slightly
// modified for better accuracy with 32 bit floating point math.
inline float FastLog2(const float a) {
  uint32_t tmp;
  std::memcpy(&tmp, &a, sizeof(float));
  uint32_t expb = tmp >> 23;
  tmp = (tmp & 0x7fffff) | (0x7f << 23);
  float out;
  std::memcpy(&out, &tmp, sizeof(float));
  out -= 1.0f;
  // Minimize max absolute error.
  return out * (1.3465552f - 0.34655523f * out) - 127 + expb;
}

// Fast approximate 2^x. Does only limited range checking.
// The approximation used here is 2^(N+f) ~ 2^N*(1+f*(1-k+k*f)) where N is the
// integer and f the fractional part, f>=0. The constant k is used to tune the
// approximation accuracy. In the final version some constants were slightly
// modified for better accuracy with 32 bit floating point math.
inline float FastExp2(const float a) {
  int32_t exp;
  if (a < 0) {
    if (a < -126) return 0.0;
    // Not all compilers optimize floor, so we use (a-1) here to round down.
    // This is obviously off-by-one for integer a, but fortunately the error
    // correction term gives the exact value for 1 (by design, for continuity).
    exp = static_cast<int32_t>(a - 1);
  } else {
    exp = static_cast<int32_t>(a);
  }
  float out = a - exp;
  // Minimize max relative error.
  out = 1.0f + out * (0.6602339f + 0.33976606f * out);
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

// Fast approximate exp(x). Does only limited range checking.
inline float FastExp(const float a) { return FastExp2(1.442695040f * a); }

// Safeguarded fast logistic function, based on FastExp().
inline float FastLogistic(const float a) {
  if (a > 20.0f) {return 1.0f;}
  if (a < -20.0f) {return 0.0f;}
  return 1.0f / (1.0f + FastExp(-a));
}

inline float FastSign(const float a) {
  // Microsoft compiler does not have a builtin for copysign and emits a
  // library call which is too expensive for hot paths.
#if defined(_MSC_VER)
  // This doesn't treat signed 0's the same way that copysign does, but it
  // should be good enough, for our use case.
  return a < 0 ? -1.0f : 1.0f;
#else
  return std::copysign(1.0f, a);
#endif
}

}  // namespace lczero
