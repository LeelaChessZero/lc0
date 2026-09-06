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

#include <span>
#include <string>
#include <string_view>

#include "chess/board.h"

namespace lczero {

class Position {
 public:
  Position() = default;
  // From parent position and move.
  Position(const Position& parent, Move m);
  // From particular position.
  Position(const ChessBoard& board, int rule50_ply, int game_ply);
  // From fen.
  static Position FromFen(std::string_view fen);

  uint64_t Hash() const;
  bool IsBlackToMove() const { return us_board_.flipped(); }

  // Number of half-moves since beginning of the game.
  int GetGamePly() const { return ply_count_; }

  // How many time the same position appeared in the game before.
  int GetRepetitions() const { return repetitions_; }

  // How many half-moves since the same position appeared in the game before.
  int GetPliesSincePrevRepetition() const { return cycle_length_; }

  // Someone outside that class knows better about repetitions, so they can
  // set it.
  void SetRepetitions(int repetitions, int cycle_length) {
    repetitions_ = repetitions;
    cycle_length_ = cycle_length;
  }

  // Number of ply with no captures and pawn moves.
  int GetRule50Ply() const { return rule50_ply_; }

  // Gets board from the point of view of player to move.
  const ChessBoard& GetBoard() const { return us_board_; }

  bool operator==(const Position&) const = default;
  bool operator!=(const Position&) const = default;

  std::string DebugString() const;

 private:
  // The board from the point of view of the player to move.
  ChessBoard us_board_;

  // How many half-moves without capture or pawn move was there.
  int rule50_ply_ = 0;
  // How many repetitions this position had before. For new positions it's 0.
  int repetitions_ = 0;
  // How many half-moves since the position was repeated or 0.
  int cycle_length_ = 0;
  // number of half-moves since beginning of the game.
  int ply_count_ = 0;
};

// GetFen returns a FEN notation for the position.
std::string PositionToFen(const Position& pos);

// These are ordered so max() prefers the best result.
enum class GameResult : uint8_t { UNDECIDED, BLACK_WON, DRAW, WHITE_WON };
GameResult operator-(const GameResult& res);

class PositionHistory {
 public:
  PositionHistory() = default;
  PositionHistory(const PositionHistory& other) = default;
  PositionHistory(PositionHistory&& other) = default;
  PositionHistory(std::span<const Position> positions)
      : positions_(positions.begin(), positions.end()) {}

  PositionHistory& operator=(const PositionHistory& other) = default;
  PositionHistory& operator=(PositionHistory&& other) = default;

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

  // Can be used to reduce allocation cost while performing a sequence of moves
  // in succession.
  void Reserve(int size) { positions_.reserve(size); }

  // Number of positions in history.
  int GetLength() const { return positions_.size(); }

  // Resets the position to a given state.
  void Reset(const ChessBoard& board, int rule50_ply, int game_ply);
  void Reset(const Position& pos);

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

  // Checks for any repetitions since the last time 50 move rule was reset.
  bool DidRepeatSinceLastZeroingMove() const;

  std::span<const Position> GetPositions() const { return positions_; }

 private:
  int ComputeLastMoveRepetitions(int* cycle_length) const;

  std::vector<Position> positions_;
};

}  // namespace lczero
