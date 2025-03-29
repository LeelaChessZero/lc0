/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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
#include <string>
#include <vector>

namespace lczero {

struct PieceType {
  uint8_t idx;
  static constexpr PieceType FromIdx(uint8_t idx) { return PieceType{idx}; }
  static PieceType Parse(char c);
  std::string ToString(bool uppercase = false) const {
    return std::string(1, "nqrbpk"[idx] + (uppercase ? 'A' - 'a' : 0));
  }
  bool CanPromoteInto() const { return idx < 4; }
  bool IsValid() const { return idx < 6; }
  bool operator==(const PieceType& other) const = default;
  bool operator!=(const PieceType& other) const = default;

 private:
  constexpr explicit PieceType(uint8_t idx) : idx(idx) {}
};

constexpr PieceType kKnight = PieceType::FromIdx(0),
                    kQueen = PieceType::FromIdx(1),
                    kRook = PieceType::FromIdx(2),
                    kBishop = PieceType::FromIdx(3),
                    kPawn = PieceType::FromIdx(4),
                    kKing = PieceType::FromIdx(5);

struct File {
  uint8_t idx;
  File() : idx(0x80) {}  // Not on board.
  constexpr bool IsValid() const { return idx < 8; }
  static constexpr File FromIdx(uint8_t idx) { return File{idx}; }
  static constexpr File Parse(char c) { return File(std::tolower(c) - 'a'); }
  std::string ToString(bool uppercase = false) const {
    return std::string(1, (uppercase ? 'A' : 'a') + idx);
  }
  void Flop() { idx ^= 0b111; }
  auto operator<=>(const File& other) const = default;
  void operator++() { ++idx; }
  void operator--() { --idx; }
  void operator+=(int delta) { idx += delta; }
  File operator+(int delta) const { return File(idx + delta); }
  File operator-(int delta) const { return File(idx - delta); }

 private:
  constexpr explicit File(uint8_t idx) : idx(idx) {}
};

constexpr File kFileA = File::FromIdx(0), kFileB = File::FromIdx(1),
               kFileC = File::FromIdx(2), kFileD = File::FromIdx(3),
               kFileE = File::FromIdx(4), kFileF = File::FromIdx(5),
               kFileG = File::FromIdx(6), kFileH = File::FromIdx(7);

struct Rank {
  uint8_t idx;
  constexpr bool IsValid() const { return idx < 8; }
  static constexpr Rank FromIdx(uint8_t idx) { return Rank{idx}; }
  static constexpr Rank Parse(char c) { return Rank(c - '1'); }
  void Flip() { idx ^= 0b111; }
  std::string ToString() const { return std::string(1, '1' + idx); }
  auto operator<=>(const Rank& other) const = default;
  void operator--() { --idx; }
  void operator++() { ++idx; }
  void operator+=(int delta) { idx += delta; }
  Rank operator+(int delta) const { return Rank(idx + delta); }
  Rank operator-(int delta) const { return Rank(idx - delta); }

 private:
  constexpr explicit Rank(uint8_t idx) : idx(idx) {}
};

constexpr Rank kRank1 = Rank::FromIdx(0), kRank2 = Rank::FromIdx(1),
               kRank3 = Rank::FromIdx(2), kRank4 = Rank::FromIdx(3),
               kRank5 = Rank::FromIdx(4), kRank6 = Rank::FromIdx(5),
               kRank7 = Rank::FromIdx(6), kRank8 = Rank::FromIdx(7);

// Stores a coordinates of a single square.
class Square {
 public:
  constexpr Square() = default;
  constexpr Square(File file, Rank rank) : idx_(rank.idx * 8 + file.idx) {}
  static constexpr Square FromIdx(uint8_t idx) { return Square{idx}; }
  static constexpr Square Parse(std::string_view);
  constexpr File file() const { return File::FromIdx(idx_ % 8); }
  constexpr Rank rank() const { return Rank::FromIdx(idx_ / 8); }
  // Flips the ranks. 1 becomes 8, 2 becomes 7, etc. Files remain the same.
  void Flip() { idx_ ^= 0b111000; }
  std::string ToString(bool uppercase = false) const {
    return file().ToString(uppercase) + rank().ToString();
  }
  constexpr bool operator==(const Square& other) const = default;
  constexpr bool operator!=(const Square& other) const = default;
  constexpr uint8_t as_idx() const { return idx_; }

 private:
  explicit constexpr Square(uint8_t idx) : idx_(idx) {}

  // 0 is a1, 1 is b1, 8 is a2, 63 is h8.
  uint8_t idx_;
};

constexpr Square kSquareA1 = Square(kFileA, kRank1),
                 kSquareC1 = Square(kFileC, kRank1),
                 kSquareE1 = Square(kFileE, kRank1),
                 kSquareG1 = Square(kFileG, kRank1),
                 kSquareH1 = Square(kFileH, kRank1);

class Move {
 public:
  Move() = default;
  static constexpr Move White(Square from, Square to) {
    return Move((from.as_idx() << 6) | to.as_idx());
  }
  static constexpr Move WhitePromotion(Square from, Square to,
                                       PieceType promotion_piece) {
    return Move((from.as_idx() << 6) | to.as_idx() | kPromotion |
                (promotion_piece.idx << 12));
  }
  static constexpr Move WhiteCastling(File king, File rook) {
    return Move((king.idx << 6) | rook.idx | kCastling);
  }
  static constexpr Move WhiteEnPassant(Square from, Square to) {
    return Move((from.as_idx() << 6) | to.as_idx() | kEnPassant);
  }

  bool operator==(const Move& other) const = default;
  bool operator!=(const Move& other) const = default;

  // Mirrors the ranks of the move.
  void Flip() { data_ ^= kFlipMask; }
  std::string ToString(bool is_chess960) const;

  Square from() const { return Square::FromIdx((data_ & kFromMask) >> 6); }
  Square to() const { return Square::FromIdx(data_ & kToMask); }
  bool is_promotion() const { return data_ & kPromotion; }
  PieceType promotion() const {
    return PieceType::FromIdx((data_ & kPieceMask) >> 12);
  }
  bool is_castling() const { return (data_ & kSpecialMask) == kCastling; }
  bool is_en_passant() const { return (data_ & kSpecialMask) == kEnPassant; }
  // TODO remove this once UciReponder starts using std::optional for ponder.
  bool is_null() const { return data_ == 0; }

  uint16_t raw_data() const { return data_; }

 private:
  explicit constexpr Move(uint16_t data) : data_(data) {}

  // Move encoding using 16 bits:
  // - bits  0-5:  "to" square (6 bits)
  // - bits  6-11: "from" square (6 bits)
  // - bits  12-13: if is_promotion:  promotion piece type
  //                if !is_promotion: SpecialMove
  // - bit   14:   is_promotion flag
  // - bit   15:   reserved (potentially for side-to-move)
  // Castling is always encoded as a "king takes rook" move.
  uint16_t data_ = 0;

  enum Masks : uint16_t {
    // clang-format off
    kToMask      = 0b0000000000111111,
    kFromMask    = 0b0000111111000000,
    kSpecialMask = 0b0111000000000000,
    kCastling    = 0b0001000000000000,
    kEnPassant   = 0b0010000000000000,
    kPromotion   = 0b0100000000000000,
    kPieceMask   = 0b0011000000000000,
    // If/when we have side-to-move bit, also flip it here.
    kFlipMask    = 0b0000111000111000,
    // clang-format on
  };
};

inline int operator-(File a, File b) { return static_cast<int>(a.idx) - b.idx; }
inline int operator-(Rank a, Rank b) { return static_cast<int>(a.idx) - b.idx; }

inline constexpr Square Square::Parse(std::string_view str) {
  return Square(File::Parse(str[0]), Rank::Parse(str[1]));
}

inline PieceType PieceType::Parse(char c) {
  switch (tolower(c)) {
    case 'n':
      return kKnight;
    case 'q':
      return kQueen;
    case 'r':
      return kRook;
    case 'b':
      return kBishop;
    case 'p':
      return kPawn;
    case 'k':
      return kKing;
    default:
      return PieceType{6};
  }
}

inline std::string Move::ToString(bool is_chess960) const {
  if (is_castling() && !is_chess960) {
    return from().ToString() + (to().file() > from().file() ? "g" : "c") +
           to().rank().ToString();
  }
  return from().ToString() + to().ToString() +
         (is_promotion() ? promotion().ToString(false) : "");
}

using MoveList = std::vector<Move>;

}  // namespace lczero