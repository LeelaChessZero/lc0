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

#include <string>
#include "chess/board.h"

namespace lczero {

class Position {
 public:
  // From parent position and move.
  Position(const Position& parent, Move m);
  // From particular position.
  Position(const ChessBoard& board, int no_capture_ply, int game_ply);

  enum Castling { WE_CAN_OOO, WE_CAN_OO, THEY_CAN_OOO, THEY_CAN_OO };

  uint64_t Hash() const;
  bool IsBlackToMove() const { return us_board_.flipped(); }

  // Number of half-moves since beginning of the game.
  int GetGamePly() const { return ply_count_; }

  // How many time the same position appeared in the game before.
  int GetRepetitions() const { return repetitions_; }

  // Someone outside that class knows better about repetitions, so they can
  // set it.
  void SetRepetitions(int repetitions) { repetitions_ = repetitions; }

  // Number of ply with no captures and pawn moves.
  int GetNoCapturePly() const { return no_capture_ply_; }

  // Returns whether castle is still allowed in given direction.
  bool CanCastle(Castling) const;

  // Gets board from the point of view of player to move.
  const ChessBoard& GetBoard() const { return us_board_; }
  // Gets board from the point of view of opponent.
  const ChessBoard& GetThemBoard() const { return them_board_; }

  std::string DebugString() const;

 private:
  // The board from the point of view of the player to move.
  ChessBoard us_board_;
  // The board from the point of view of opponent.
  ChessBoard them_board_;

  // How many half-moves without capture or pawn move was there.
  int no_capture_ply_ = 0;
  // How many repetitions this position had before. For new positions it's 0.
  int repetitions_;
  // number of half-moves since beginning of the game.
  int ply_count_ = 0;
};

enum class GameResult { UNDECIDED, WHITE_WON, DRAW, BLACK_WON };

class PositionHistory {
 public:
  PositionHistory() = default;
  PositionHistory(const PositionHistory& other) = default;

  // Returns first position of the game (or fen from which it was initialized).
  const Position& Starting() const { return positions_.front(); }

  // Returns the latest position of the game.
  const Position& Last() const { return positions_.back(); }

  // N-th position of the game, 0-based.
  const Position& GetPositionAt(int idx) const { return positions_[idx]; }

  // Trims position to a given size.
  void Trim(int size) {
    positions_.erase(positions_.begin() + size, positions_.end());
  }

  // Number of positions in history.
  int GetLength() const { return positions_.size(); }

  // Resets the position to a given state.
  void Reset(const ChessBoard& board, int no_capture_ply, int game_ply);

  // Appends a position to history.
  void Append(Move m);

  // Pops last move from history.
  void Pop() { positions_.pop_back(); }

  // Finds the endgame state (win/lose/draw/nothing) for the last position.
  GameResult ComputeGameResult() const;

  // Returns whether next move is history should be black's.
  bool IsBlackToMove() const { return Last().IsBlackToMove(); }

  // Builds a hash from last X positions.
  uint64_t HashLast(int positions) const;

 private:
  int ComputeLastMoveRepetitions() const;

  std::vector<Position> positions_;
};

}  // namespace lczero