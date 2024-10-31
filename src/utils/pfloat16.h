/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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
// Compressed 16-bit floating point format for probability values.
// Optimised for representing numbers in the [0,1] range.
//
// Source values are 32-bit floats:
// * bit 31 is sign (zero means positive)
// * bit 30 is sign of exponent (zero means nonpositive)
// * bits 29..23 are value bits of exponent
// * bits 22..0 are significand bits (plus a "virtual" always-on bit: s ∈ [1,2))
// The number is then sign * 2^exponent * significand, usually.
// See https://www.h-schmidt.net/FloatConverter/IEEE754.html for details.
//
// In compressed 16-bit value we store bits 27..12:
// * bit 31 is always off as values are always >= 0
// * bit 30 is always off as values are always < 2
// * bits 29..28 are only off for values < 4.6566e-10, assume they are always on
// * bits 11..0 are for higher precision, they are dropped leaving only 11 bits
//     of precision
//
// Out of 65556 possible values, 2047 are outside of [0,1] interval (they are in
// interval (1,2)).

// When converting to compressed format, bit 11 is added to in order to make it
// a rounding rather than truncation.
// If the two assumed-on exponent bits (3<<28) are in fact off, the input is
// rounded up to the smallest value with them on. We accomplish this by
// subtracting the two bits from the input and checking for a negative result
// (the subtraction works despite crossing from exponent to significand). This
// is combined with the round-to-nearest addition (1<<11) into one op.

class pfloat16 {
 public:
  pfloat16() { value = 0; }

  pfloat16(const float &p) {
    assert(0.0f <= p && p <= 1.0f);
    constexpr int32_t roundings = (1 << 11) - (3 << 28);
    int32_t tmp;
    std::memcpy(&tmp, &p, sizeof(float));
    tmp += roundings;
    value = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
  }

  operator float() const {
    // Reshift into place and set the assumed-set exponent bits.
    uint32_t tmp = (static_cast<uint32_t>(value) << 12) | (3 << 28);
    float ret;
    std::memcpy(&ret, &tmp, sizeof(uint32_t));
    return ret;
  }

 private:
  uint16_t value = 0;
};
}  // namespace lczero
