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
#include <cstdint>
#include <iterator>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace lczero {

inline unsigned long GetLowestBit(std::uint64_t value) {
#if defined(_MSC_VER) && defined(_WIN64)
  unsigned long result;
  _BitScanForward64(&result, value);
  return result;
#elif defined(_MSC_VER)
  unsigned long result;
  if (value & 0xFFFFFFFF) {
    _BitScanForward(&result, value);
  } else {
    _BitScanForward(&result, value >> 32);
    result += 32;
  }
  return result;
#else
  return __builtin_ctzll(value);
#endif
}

enum BoardTransform {
  NoTransform = 0,
  // Horizontal mirror, ReverseBitsInBytes
  FlipTransform = 1,
  // Vertical mirror, ReverseBytesInBytes
  MirrorTransform = 2,
  // Diagonal transpose A1 to H8, TransposeBitsInBytes.
  TransposeTransform = 4,
};

inline uint64_t ReverseBitsInBytes(uint64_t v) {
  v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
  v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
  v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
  return v;
}

inline uint64_t ReverseBytesInBytes(uint64_t v) {
  v = (v & 0x00000000FFFFFFFF) << 32 | (v & 0xFFFFFFFF00000000) >> 32;
  v = (v & 0x0000FFFF0000FFFF) << 16 | (v & 0xFFFF0000FFFF0000) >> 16;
  v = (v & 0x00FF00FF00FF00FF) << 8 | (v & 0xFF00FF00FF00FF00) >> 8;
  return v;
}

// Transpose across the diagonal connecting bit 7 to bit 56.
inline uint64_t TransposeBitsInBytes(uint64_t v) {
  v = (v & 0xAA00AA00AA00AA00ULL) >> 9 | (v & 0x0055005500550055ULL) << 9 |
      (v & 0x55AA55AA55AA55AAULL);
  v = (v & 0xCCCC0000CCCC0000ULL) >> 18 | (v & 0x0000333300003333ULL) << 18 |
      (v & 0x3333CCCC3333CCCCULL);
  v = (v & 0xF0F0F0F000000000ULL) >> 36 | (v & 0x000000000F0F0F0FULL) << 36 |
      (v & 0x0F0F0F0FF0F0F0F0ULL);
  return v;
}

// Iterates over all set bits of the value, lower to upper. The value of
// dereferenced iterator is bit number (lower to upper, 0 bazed)
template <typename T>
class BitIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = T;
  using value_type = T;
  using pointer = T*;
  using reference = T&;

  BitIterator(std::uint64_t value) : value_(value){};
  bool operator!=(const BitIterator& other) { return value_ != other.value_; }

  void operator++() { value_ &= (value_ - 1); }
  T operator*() const { return GetLowestBit(value_); }

 private:
  std::uint64_t value_;
};

class IterateBits {
 public:
  IterateBits(std::uint64_t value) : value_(value) {}
  BitIterator<int> begin() { return value_; }
  BitIterator<int> end() { return 0; }

 private:
  std::uint64_t value_;
};

}  // namespace lczero
