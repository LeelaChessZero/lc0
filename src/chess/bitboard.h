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

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "chess/types.h"
#include "utils/bititer.h"

namespace lczero {

// Represents a board as an array of 64 bits.
// Bit enumeration goes from bottom to top, from left to right:
// Square a1 is bit 0, square h1 is bit 7, square a2 is bit 8.
class BitBoard {
 public:
  constexpr BitBoard(std::uint64_t board) : board_(board) {}
  BitBoard() = default;
  constexpr static BitBoard FromSquare(Square square) {
    return BitBoard(1ULL << square.as_idx());
  }

  std::uint64_t as_int() const { return board_; }
  void clear() { board_ = 0; }

  // Counts the number of set bits in the BitBoard.
  int count() const {
#if defined(NO_POPCNT)
    std::uint64_t x = board_;
    x -= (x >> 1) & 0x5555555555555555;
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
    return (x * 0x0101010101010101) >> 56;
#elif defined(_MSC_VER) && defined(_WIN64)
    return _mm_popcnt_u64(board_);
#elif defined(_MSC_VER)
    return __popcnt(board_) + __popcnt(board_ >> 32);
#else
    return __builtin_popcountll(board_);
#endif
  }

  // Like count() but using algorithm faster on a very sparse BitBoard.
  // May be slower for more than 4 set bits, but still correct.
  // Useful when counting bits in a Q, R, N or B BitBoard.
  int count_few() const {
#if defined(NO_POPCNT)
    std::uint64_t x = board_;
    int count;
    for (count = 0; x != 0; ++count) {
      // Clear the rightmost set bit.
      x &= x - 1;
    }
    return count;
#else
    return count();
#endif
  }

  // Sets the value for given square to 1 if cond is true.
  // Otherwise does nothing (doesn't reset!).
  void set_if(Square square, bool cond) {
    board_ |= (static_cast<uint64_t>(cond) << square.as_idx());
  }
  // Sets value of given square to 1.
  void set(Square square) { board_ |= (1ULL << square.as_idx()); }
  // Sets value of given square to 0.
  void reset(Square square) { board_ &= ~(1ULL << square.as_idx()); }
  // Gets value of a square.
  bool get(Square square) const { return board_ & (1ULL << square.as_idx()); }

  // Returns whether all bits of a board are set to 0.
  bool empty() const { return board_ == 0; }

  // Checks whether two bitboards have common bits set.
  bool intersects(const BitBoard& other) const { return board_ & other.board_; }

  // Flips black and white side of a board.
  void Mirror() { board_ = ReverseBytesInBytes(board_); }

  bool operator==(const BitBoard& other) const = default;
  bool operator!=(const BitBoard& other) const = default;

  struct Uin64ToSquare {
    constexpr Square operator()(uint64_t x) { return Square::FromIdx(x); }
  };
  using Iterator = BitIterator<Square, Uin64ToSquare>;
  Iterator begin() const { return board_; }
  Iterator end() const { return 0; }

  std::string DebugString() const {
    std::string res;
    for (int i = 7; i >= 0; --i) {
      for (int j = 0; j < 8; ++j)
        res += get({File::FromIdx(i), Rank::FromIdx(j)}) ? '#' : '.';
      res += '\n';
    }
    return res;
  }

  // Applies a mask to the bitboard (intersects).
  BitBoard& operator&=(const BitBoard& a) {
    board_ &= a.board_;
    return *this;
  }

  friend void swap(BitBoard& a, BitBoard& b) {
    using std::swap;
    swap(a.board_, b.board_);
  }

  // Returns union (bitwise OR) of two boards.
  friend BitBoard operator|(const BitBoard& a, const BitBoard& b) {
    return {a.board_ | b.board_};
  }

  // Returns intersection (bitwise AND) of two boards.
  friend BitBoard operator&(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & b.board_};
  }

  // Returns bitboard with one bit reset.
  friend BitBoard operator-(const BitBoard& a, const Square& b) {
    return {a.board_ & ~(1ULL << b.as_idx())};
  }

  // Returns difference (bitwise AND-NOT) of two boards.
  friend BitBoard operator-(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & ~b.board_};
  }

 private:
  std::uint64_t board_ = 0;
};

}  // namespace lczero
