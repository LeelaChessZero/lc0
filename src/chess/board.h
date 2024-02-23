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
#include <string>

#include "chess/bitboard.h"
#include "utils/hashcat.h"

namespace lczero {

// Initializes internal magic bitboard structures.
void InitializeMagicBitboards();

// Represents king attack info used during legal move detection.
class KingAttackInfo {
 public:
  bool in_check() const { return attack_lines_.as_int(); }
  bool in_double_check() const { return double_check_; }
  bool is_pinned(const BoardSquare square) const {
    return pinned_pieces_.get(square);
  }
  bool is_on_attack_line(const BoardSquare square) const {
    return attack_lines_.get(square);
  }

  bool double_check_ = 0;
  BitBoard pinned_pieces_ = {0};
  BitBoard attack_lines_ = {0};
};

// Represents a board position.
// Unlike most chess engines, the board is mirrored for black.
class ChessBoard {
 public:
  ChessBoard() = default;
  ChessBoard(const std::string& fen) { SetFromFen(fen); }

  static const char* kStartposFen;
  static const ChessBoard kStartposBoard;
  static const BitBoard kPawnMask;

  // Sets position from FEN string.
  // If @rule50_ply and @moves are not nullptr, they are filled with number
  // of moves without capture and number of full moves since the beginning of
  // the game.
  void SetFromFen(std::string fen, int* rule50_ply = nullptr,
                  int* moves = nullptr);
  // Nullifies the whole structure.
  void Clear();
  // Swaps black and white pieces and mirrors them relative to the
  // middle of the board. (what was on rank 1 appears on rank 8, what was
  // on file b remains on file b).
  void Mirror();

  // Generates list of possible moves for "ours" (white), but may leave king
  // under check.
  MoveList GeneratePseudolegalMoves() const;
  // Applies the move. (Only for "ours" (white)). Returns true if 50 moves
  // counter should be removed.
  bool ApplyMove(Move move);
  // Checks if the square is under attack from "theirs" (black).
  bool IsUnderAttack(BoardSquare square) const;
  // Generates the king attack info used for legal move detection.
  KingAttackInfo GenerateKingAttackInfo() const;
  // Checks if "our" (white) king is under check.
  bool IsUnderCheck() const { return IsUnderAttack(our_king_); }

  // Checks whether at least one of the sides has mating material.
  bool HasMatingMaterial() const;
  // Generates legal moves.
  MoveList GenerateLegalMoves() const;
  // Check whether pseudolegal move is legal.
  bool IsLegalMove(Move move, const KingAttackInfo& king_attack_info) const;
  // Returns whether two moves are actually the same move in the position.
  bool IsSameMove(Move move1, Move move2) const;
  // Returns the same move but with castling encoded in legacy way.
  Move GetLegacyMove(Move move) const;
  // Returns the same move but with castling encoded in modern way.
  Move GetModernMove(Move move) const;

  uint64_t Hash() const {
    return HashCat({our_pieces_.as_int(), their_pieces_.as_int(),
                    rooks_.as_int(), bishops_.as_int(), pawns_.as_int(),
                    (static_cast<uint32_t>(our_king_.as_int()) << 24) |
                        (static_cast<uint32_t>(their_king_.as_int()) << 16) |
                        (static_cast<uint32_t>(castlings_.as_int()) << 8) |
                        static_cast<uint32_t>(flipped_)});
  }

  class Castlings {
   public:
    Castlings()
        : our_queenside_rook_(FILE_A),
          their_queenside_rook_(FILE_A),
          our_kingside_rook_(FILE_H),
          their_kingside_rook_(FILE_H),
          data_(0) {}

    void set_we_can_00() { data_ |= 1; }
    void set_we_can_000() { data_ |= 2; }
    void set_they_can_00() { data_ |= 4; }
    void set_they_can_000() { data_ |= 8; }

    void reset_we_can_00() { data_ &= ~1; }
    void reset_we_can_000() { data_ &= ~2; }
    void reset_they_can_00() { data_ &= ~4; }
    void reset_they_can_000() { data_ &= ~8; }

    bool we_can_00() const { return data_ & 1; }
    bool we_can_000() const { return data_ & 2; }
    bool they_can_00() const { return data_ & 4; }
    bool they_can_000() const { return data_ & 8; }
    bool no_legal_castle() const { return data_ == 0; }

    void Mirror() {
      std::swap(our_queenside_rook_, their_queenside_rook_);
      std::swap(our_kingside_rook_, their_kingside_rook_);
      data_ = ((data_ & 0b11) << 2) + ((data_ & 0b1100) >> 2);
    }

    // Note: this is not a strict xfen compatible output. Without access to the
    // board its not possible to know whether there is ambiguity so all cases
    // with any non-standard rook positions are encoded in the x-fen format
    std::string as_string() const {
      if (data_ == 0) return "-";
      std::string result;
      if (our_queenside_rook() == FILE_A && our_kingside_rook() == FILE_H &&
          their_queenside_rook() == FILE_A && their_kingside_rook() == FILE_H) {
        if (we_can_00()) result += 'K';
        if (we_can_000()) result += 'Q';
        if (they_can_00()) result += 'k';
        if (they_can_000()) result += 'q';
      } else {
        if (we_can_00()) result += 'A' + our_kingside_rook();
        if (we_can_000()) result += 'A' + our_queenside_rook();
        if (they_can_00()) result += 'a' + their_kingside_rook();
        if (they_can_000()) result += 'a' + their_queenside_rook();
      }
      return result;
    }

    std::string DebugString() const {
      std::string result;
      if (data_ == 0) result = "-";
      if (we_can_00()) result += 'K';
      if (we_can_000()) result += 'Q';
      if (they_can_00()) result += 'k';
      if (they_can_000()) result += 'q';
      result += '[';
      result += 'A' + our_queenside_rook();
      result += 'A' + our_kingside_rook();
      result += 'a' + their_queenside_rook();
      result += 'a' + their_kingside_rook();
      result += ']';
      return result;
    }

    uint8_t as_int() const { return data_; }

    bool operator==(const Castlings& other) const {
      assert(our_queenside_rook_ == other.our_queenside_rook_ &&
             our_kingside_rook_ == other.our_kingside_rook_ &&
             their_queenside_rook_ == other.their_queenside_rook_ &&
             their_kingside_rook_ == other.their_kingside_rook_);
      return data_ == other.data_;
    }

    uint8_t our_queenside_rook() const { return our_queenside_rook_; }
    uint8_t our_kingside_rook() const { return our_kingside_rook_; }
    uint8_t their_queenside_rook() const { return their_queenside_rook_; }
    uint8_t their_kingside_rook() const { return their_kingside_rook_; }
    void SetRookPositions(uint8_t our_left, uint8_t our_right,
                          uint8_t their_left, uint8_t their_right) {
      our_queenside_rook_ = our_left;
      our_kingside_rook_ = our_right;
      their_queenside_rook_ = their_left;
      their_kingside_rook_ = their_right;
    }

   private:
    // Position of "left" (queenside) rook in starting game position.
    uint8_t our_queenside_rook_;
    uint8_t their_queenside_rook_;
    // Position of "right" (kingside) rook in starting position.
    uint8_t our_kingside_rook_;
    uint8_t their_kingside_rook_;

    // - Bit 0 -- "our" side's kingside castle.
    // - Bit 1 -- "our" side's queenside castle.
    // - Bit 2 -- opponent's side's kingside castle.
    // - Bit 3 -- opponent's side's queenside castle.
    uint8_t data_;
  };

  std::string DebugString() const;

  BitBoard ours() const { return our_pieces_; }
  BitBoard theirs() const { return their_pieces_; }
  BitBoard pawns() const { return pawns_ & kPawnMask; }
  BitBoard en_passant() const { return pawns_ - kPawnMask; }
  BitBoard bishops() const { return bishops_ - rooks_; }
  BitBoard rooks() const { return rooks_ - bishops_; }
  BitBoard queens() const { return rooks_ & bishops_; }
  BitBoard knights() const {
    return (our_pieces_ | their_pieces_) - pawns() - our_king_ - their_king_ -
           rooks_ - bishops_;
  }
  BitBoard kings() const {
    return our_king_.as_board() | their_king_.as_board();
  }
  const Castlings& castlings() const { return castlings_; }
  bool flipped() const { return flipped_; }

  bool operator==(const ChessBoard& other) const {
    return (our_pieces_ == other.our_pieces_) &&
           (their_pieces_ == other.their_pieces_) && (rooks_ == other.rooks_) &&
           (bishops_ == other.bishops_) && (pawns_ == other.pawns_) &&
           (our_king_ == other.our_king_) &&
           (their_king_ == other.their_king_) &&
           (castlings_ == other.castlings_) && (flipped_ == other.flipped_);
  }

  bool operator!=(const ChessBoard& other) const { return !operator==(other); }

  enum Square : uint8_t {
    // clang-format off
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    // clang-format on
  };

  enum File : uint8_t {
    // clang-format off
    FILE_A = 0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H
    // clang-format on
  };

  enum Rank : uint8_t {
    // clang-format off
    RANK_1 = 0, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8
    // clang-format on
  };

 private:
  // All white pieces.
  BitBoard our_pieces_;
  // All black pieces.
  BitBoard their_pieces_;
  // Rooks and queens.
  BitBoard rooks_;
  // Bishops and queens;
  BitBoard bishops_;
  // Pawns.
  // Ranks 1 and 8 have special meaning. Pawn at rank 1 means that
  // corresponding white pawn on rank 4 can be taken en passant. Rank 8 is the
  // same for black pawns. Those "fake" pawns are not present in our_pieces_ and
  // their_pieces_ bitboards.
  BitBoard pawns_;
  BoardSquare our_king_;
  BoardSquare their_king_;
  Castlings castlings_;
  bool flipped_ = false;  // aka "Black to move".
};

}  // namespace lczero
