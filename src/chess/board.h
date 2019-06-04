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

  // Sets position from FEN string.
  // If @no_capture_ply and @moves are not nullptr, they are filled with number
  // of moves without capture and number of full moves since the beginning of
  // the game.
  void SetFromFen(const std::string& fen, int* no_capture_ply = nullptr,
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

    void Mirror() { data_ = ((data_ & 0b11) << 2) + ((data_ & 0b1100) >> 2); }

    std::string as_string() const {
      if (data_ == 0) return "-";
      std::string result;
      if (we_can_00()) result += 'K';
      if (we_can_000()) result += 'Q';
      if (they_can_00()) result += 'k';
      if (they_can_000()) result += 'q';
      return result;
    }

    uint8_t as_int() const { return data_; }

    bool operator==(const Castlings& other) const {
      return data_ == other.data_;
    }

   private:
    std::uint8_t data_ = 0;
  };

  std::string DebugString() const;

  BitBoard ours() const { return our_pieces_; }
  BitBoard theirs() const { return their_pieces_; }
  BitBoard pawns() const;
  BitBoard en_passant() const;
  BitBoard bishops() const { return bishops_ - rooks_; }
  BitBoard rooks() const { return rooks_ - bishops_; }
  BitBoard queens() const { return rooks_ & bishops_; }
  BitBoard our_knights() const {
    return our_pieces_ - pawns() - our_king_ - rooks_ - bishops_;
  }
  BitBoard their_knights() const {
    return their_pieces_ - pawns() - their_king_ - rooks_ - bishops_;
  }
  BitBoard our_king() const { return 1ull << our_king_.as_int(); }
  BitBoard their_king() const { return 1ull << their_king_.as_int(); }
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
