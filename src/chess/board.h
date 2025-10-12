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
#include "chess/types.h"
#include "utils/hashcat.h"

namespace lczero {

// Initializes internal magic bitboard structures.
void InitializeMagicBitboards();

// Represents king attack info used during legal move detection.
class KingAttackInfo {
 public:
  bool in_check() const { return attack_lines_.as_int(); }
  bool in_double_check() const { return double_check_; }
  bool is_pinned(const Square square) const {
    return pinned_pieces_.get(square);
  }
  bool is_on_attack_line(const Square square) const {
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
  ChessBoard(const ChessBoard&) = default;
  ChessBoard(const std::string& fen) { SetFromFen(fen); }

  ChessBoard& operator=(const ChessBoard&) = default;

  static const char* kStartposFen;
  static const ChessBoard kStartposBoard;
  static const BitBoard kPawnMask;

  // Sets position from FEN string.
  // If @rule50_ply and @moves are not nullptr, they are filled with number
  // of moves without capture and number of full moves since the beginning of
  // the game.
  void SetFromFen(std::string_view fen, int* rule50_ply = nullptr,
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
  bool IsUnderAttack(Square square) const;
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

  // Parses a move from move_str.
  // The input string should be in the "normal" notation rather than from the
  // player to move, i.e. "e7e5" for the black pawn move.
  // Output is currently "from the player to move" perspective (i.e. from=E2,
  // to=E4 for the same black move). This is temporary, plan is to change it
  // soon.
  Move ParseMove(std::string_view move_str) const;

  uint64_t Hash() const {
    return HashCat({our_pieces_.as_int(), their_pieces_.as_int(),
                    rooks_.as_int(), bishops_.as_int(), pawns_.as_int(),
                    (static_cast<uint32_t>(our_king_.as_idx()) << 24) |
                        (static_cast<uint32_t>(their_king_.as_idx()) << 16) |
                        (static_cast<uint32_t>(castlings_.as_int()) << 8) |
                        static_cast<uint32_t>(flipped_)});
  }

  class Castlings {
   public:
    Castlings()
        : our_queenside_rook(kFileA),
          their_queenside_rook(kFileA),
          our_kingside_rook(kFileH),
          their_kingside_rook(kFileH),
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
      std::swap(our_queenside_rook, their_queenside_rook);
      std::swap(our_kingside_rook, their_kingside_rook);
      data_ = ((data_ & 0b11) << 2) + ((data_ & 0b1100) >> 2);
    }

    // Note: this is not a strict xfen compatible output. Without access to the
    // board its not possible to know whether there is ambiguity so all cases
    // with any non-standard rook positions are encoded in the x-fen format
    std::string as_string() const {
      if (data_ == 0) return "-";
      std::string result;
      if (our_queenside_rook == kFileA && our_kingside_rook == kFileH &&
          their_queenside_rook == kFileA && their_kingside_rook == kFileH) {
        if (we_can_00()) result += 'K';
        if (we_can_000()) result += 'Q';
        if (they_can_00()) result += 'k';
        if (they_can_000()) result += 'q';
      } else {
        if (we_can_00()) result += our_kingside_rook.ToString(true);
        if (we_can_000()) result += our_queenside_rook.ToString(true);
        if (they_can_00()) result += their_kingside_rook.ToString(false);
        if (they_can_000()) result += their_queenside_rook.ToString(false);
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
      result += our_queenside_rook.ToString(true);
      result += our_kingside_rook.ToString(true);
      result += their_queenside_rook.ToString(false);
      result += their_kingside_rook.ToString(false);
      result += ']';
      return result;
    }

    uint8_t as_int() const { return data_; }
    bool operator==(const Castlings& other) const = default;

    // Position of "left" (queenside) rook in starting game position.
    File our_queenside_rook;
    File their_queenside_rook;
    // Position of "right" (kingside) rook in starting position.
    File our_kingside_rook;
    File their_kingside_rook;

   private:
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
    return BitBoard::FromSquare(our_king_) | BitBoard::FromSquare(their_king_);
  }
  const Castlings& castlings() const { return castlings_; }
  bool flipped() const { return flipped_; }

  bool operator==(const ChessBoard& other) const = default;
  bool operator!=(const ChessBoard& other) const = default;

 private:
  // Sets the piece on the square.
  void PutPiece(Square square, PieceType piece, bool is_theirs);
  // Check internal state is consistent after state transformations.
  bool IsValid() const;

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
  Square our_king_;
  Square their_king_;
  Castlings castlings_;
  bool flipped_ = false;  // aka "Black to move".
};

// Converts the board to FEN string.
std::string BoardToFen(const ChessBoard& board);

}  // namespace lczero
