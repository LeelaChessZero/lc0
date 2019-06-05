/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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

namespace lczero {

uint16_t FP32toFP16(float f32) {
  uint32_t f = *(uint32_t*)&f32;

  uint16_t f16 = 0;

  f16 |= (f >> 16) & 0x8000;  // copy sign bit

  uint32_t e = (f >> 23) & 0xff;  // extract exponent
  uint32_t m = f & 0x7fffff;      // extract mantissa

  if (e == 255) {
    // dealing with a special here
    if (m == 0) {
      // infinity
      return (f16 | 0x7c00);  // e=31, m=0, preserve sign
    } else {
      // NaN
      return 0x7e00;  // e=31, m=0x200, s=0
    }
  } else if ((e >= 143) || ((e == 142) && (m > 0x7fe000))) {
    // not representable in FP16, so return infinity
    return (f16 | 0x7c00);  // e=31, m=0, preserve sign
  } else if ((e <= 101) || ((e == 102) && (m < 0x2000))) {
    // underflow to 0
    return f16;
  } else if (e <= 112) {
    // denorm situation
    m |= 0x800000;  // add leading 1

    // the 24-bit mantissa needs to shift 14 bits over to
    // fit into 10 bits, and then as many bits as the exponent
    // is below our denorm exponent
    //  127 (fp32 bias)
    // -  e (actual fp32 exponent)
    // + 24 (fp32 mantissa bits including leading 1)
    // - 10 (fp16 mantissa bits not including leading 1)
    // - 15 (fp16 denorm exponent)
    // = 126 - e
    m >>= (126 - e);

    return (uint16_t)(f16 | m);  // e=0, preserve sign
  } else {
    // can convert directly to fp16
    e -= 112;  // 127 - 15 exponent bias
    m >>= 13;  // 23 - 10 mantissa bits
    return (uint16_t)(f16 | (e << 10) | m);
  }
}

float FP16toFP32(uint16_t f16) {
  uint32_t f = f16;

  uint32_t f32 = 0;

  f32 |= (f << 16) & 0x80000000;  // copy sign bit

  uint32_t e = (f >> 10) & 0x1f;  // extract exponent
  uint32_t m = f & 0x3ff;         // extract mantissa

  if (e == 0) {
    if (m == 0) {
      // nothing to do; it's already +/- 0
    } else {
      // denorm
      e = 113;
      m <<= 13;
      // shift mantissa until the top bit is 1<<23
      // note that we've alrady guaranteed that the
      // mantissa is non-zero and that the top bit is
      // at or below 1<<23
      while (!(m & 0x800000)) {
        e--;
        m <<= 1;
      }
      m &= 0x7fffff;

      f32 |= (e << 23) | m;
    }
  } else if (e == 31) {
    // FP special
    if (m == 0) {
      // Inf
      f32 |= 0x7f800000;  // e=255, m=0, preserve sign
    } else {
      // NaN
      f32 = 0x7fc00000;  // e=255, m=0x800000, s=0
    }
  } else {
    e += 112;  // 127-15 exponent bias
    m <<= 13;  // 23-10 mantissa bits
    f32 |= (e << 23) | m;
  }

  return *(float*)&f32;
}

};  // namespace lczero
