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
#include <string>
#include <vector>

#include "utils/bititer.h"

namespace lczero {

// Stores a coordinates of a single square.
class BoardSquare {
 public:
  constexpr BoardSquare() {}
  // As a single number, 0 to 63, bottom to top, left to right.
  // 0 is a1, 8 is b1, 63 is h7.
  constexpr BoardSquare(std::uint8_t num) : square_(num) {}
  // From row(bottom to top), and col(left to right), 0-based.
  constexpr BoardSquare(int row, int col) : BoardSquare(row * 8 + col) {}
  // From Square name, e.g e4. Only lowercase.
  BoardSquare(const std::string& str, bool black = false)
      : BoardSquare(black ? '8' - str[1] : str[1] - '1', str[0] - 'a') {}
  constexpr std::uint8_t as_int() const { return square_; }
  void set(int row, int col) { square_ = row * 8 + col; }

  // 0-based, bottom to top.
  int row() const { return square_ / 8; }
  // 0-based, left to right.
  int col() const { return square_ % 8; }

  // Row := 7 - row.  Col remains the same.
  void Mirror() { square_ = square_ ^ 0b111000; }

  // Checks whether coordinate is within 0..7.
  static bool IsValidCoord(int x) { return x >= 0 && x < 8; }

  // Checks whether coordinates are within 0..7.
  static bool IsValid(int row, int col) {
    return row >= 0 && col >= 0 && row < 8 && col < 8;
  }

  constexpr bool operator==(const BoardSquare& other) const {
    return square_ == other.square_;
  }

  constexpr bool operator!=(const BoardSquare& other) const {
    return square_ != other.square_;
  }

  // Returns the square in algebraic notation (e.g. "e4").
  std::string as_string() const {
    return std::string(1, 'a' + col()) + std::string(1, '1' + row());
  }

 private:
  std::uint8_t square_ = 0;
};

// Represents a board as an array of 64 bits.
// Bit enumeration goes from bottom to top, from left to right:
// Square a1 is bit 0, square a8 is bit 7, square b1 is bit 8.
class BitBoard {
 public:
  constexpr BitBoard(std::uint64_t board) : board_(board) {}
  BitBoard() = default;
  BitBoard(const BitBoard&) = default;

  std::uint64_t as_int() const { return board_; }
  void clear() { board_ = 0; }

  // Sets the value for given square to 1 if cond is true.
  // Otherwise does nothing (doesn't reset!).
  void set_if(BoardSquare square, bool cond) { set_if(square.as_int(), cond); }
  void set_if(std::uint8_t pos, bool cond) {
    board_ |= (std::uint64_t(cond) << pos);
  }
  void set_if(int row, int col, bool cond) {
    set_if(BoardSquare(row, col), cond);
  }

  // Sets value of given square to 1.
  void set(BoardSquare square) { set(square.as_int()); }
  void set(std::uint8_t pos) { board_ |= (std::uint64_t(1) << pos); }
  void set(int row, int col) { set(BoardSquare(row, col)); }

  // Sets value of given square to 0.
  void reset(BoardSquare square) { reset(square.as_int()); }
  void reset(std ::uint8_t pos) { board_ &= ~(std::uint64_t(1) << pos); }
  void reset(int row, int col) { reset(BoardSquare(row, col)); }

  // Gets value of a square.
  bool get(BoardSquare square) const { return get(square.as_int()); }
  bool get(std::uint8_t pos) const {
    return board_ & (std::uint64_t(1) << pos);
  }
  bool get(int row, int col) const { return get(BoardSquare(row, col)); }

  // Returns whether all bits of a board are set to 0.
  bool empty() const { return board_ == 0; }

  // Checks whether two bitboards have common bits set.
  bool intersects(const BitBoard& other) const { return board_ & other.board_; }

  // Flips black and white side of a board.
  void Mirror() {
    board_ = (board_ & 0x00000000FFFFFFFF) << 32 |
             (board_ & 0xFFFFFFFF00000000) >> 32;
    board_ = (board_ & 0x0000FFFF0000FFFF) << 16 |
             (board_ & 0xFFFF0000FFFF0000) >> 16;
    board_ =
        (board_ & 0x00FF00FF00FF00FF) << 8 | (board_ & 0xFF00FF00FF00FF00) >> 8;
  }

  bool operator==(const BitBoard& other) const {
    return board_ == other.board_;
  }

  BitIterator<BoardSquare> begin() const { return board_; }
  BitIterator<BoardSquare> end() const { return 0; }

  std::string DebugString() const {
    std::string res;
    for (int i = 7; i >= 0; --i) {
      for (int j = 0; j < 8; ++j) {
        if (get(i, j))
          res += '#';
        else
          res += '.';
      }
      res += '\n';
    }
    return res;
  }

  // Applies a mask to the bitboard (intersects).
  BitBoard& operator*=(const BitBoard& a) {
    board_ &= a.board_;
    return *this;
  }

  friend void swap(BitBoard& a, BitBoard& b) {
    using std::swap;
    swap(a.board_, b.board_);
  }

  // Returns union (bitwise OR) of two boards.
  friend BitBoard operator+(const BitBoard& a, const BitBoard& b) {
    return {a.board_ | b.board_};
  }

  // Returns bitboard with one bit reset.
  friend BitBoard operator-(const BitBoard& a, const BoardSquare& b) {
    return {a.board_ & ~(1ULL << b.as_int())};
  }

  // Returns difference (bitwise AND-NOT) of two boards.
  friend BitBoard operator-(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & ~b.board_};
  }

  // Returns intersection (bitwise AND) of two boards.
  friend BitBoard operator*(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & b.board_};
  }

 private:
  std::uint64_t board_ = 0;
};

class Move {
 public:
  enum class Promotion : std::uint8_t { None, Queen, Rook, Bishop, Knight };
  Move() = default;
  Move(BoardSquare from, BoardSquare to) : from_(from), to_(to) {}
  Move(BoardSquare from, BoardSquare to, Promotion promotion)
      : from_(from), to_(to), promotion_(promotion) {}
  Move(const std::string& str, bool black = false);
  Move(const char* str, bool black = false) : Move(std::string(str), black) {}

  BoardSquare from() const { return from_; }
  BoardSquare to() const { return to_; }
  Promotion promotion() const { return promotion_; }
  bool IsCastling() const { return castling_; }
  void SetCastling() { castling_ = true; }

  // 0 .. 16384, knight promotion and no promotion is the same.
  uint16_t as_packed_int() const;

  // 0 .. 1857, to use in neural networks.
  uint16_t as_nn_index() const;

  bool operator==(const Move& other) const {
    return from_ == other.from_ && to_ == other.to_ &&
           promotion_ == other.promotion_;
  }

  bool operator!=(const Move& other) const { return !operator==(other); }
  operator bool() const { return from_.as_int() != 0 || to_.as_int() != 0; }

  void Mirror() {
    from_.Mirror();
    to_.Mirror();
  }

  std::string as_string() const {
    BoardSquare to = to_;
    if (castling_) {
      to = BoardSquare(to.row(), (to.col() == 7) ? 6 : 2);
    }
    std::string res = from_.as_string() + to.as_string();
    switch (promotion_) {
      case Promotion::None:
        return res;
      case Promotion::Queen:
        return res + 'q';
      case Promotion::Rook:
        return res + 'r';
      case Promotion::Bishop:
        return res + 'b';
      case Promotion::Knight:
        return res + 'n';
    }
  }

 private:
  BoardSquare from_;
  BoardSquare to_;
  Promotion promotion_ = Promotion::None;
  bool castling_ = false;
};

using MoveList = std::vector<Move>;

}  // namespace lczero