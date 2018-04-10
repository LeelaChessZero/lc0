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
*/

#pragma once
#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace lczero {

// Iterates over all set bits of the value, lower to upper. The value of
// dereferenced iterator is bit number (lower to upper, 0 bazed)
template <typename T>
class BitIterator {
 public:
  BitIterator(std::uint64_t value) : value_(value){};
  bool operator!=(const BitIterator& other) { return value_ != other.value_; }

  void operator++() { value_ &= (value_ - 1); }
  T operator*() const {
#ifdef _MSC_VER
    unsigned long result;
    _BitScanForward64(&result, value_);
    return result;
#else
    return __builtin_ctzll(value_);
#endif
  }

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