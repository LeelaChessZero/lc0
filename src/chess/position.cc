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

#include "chess/position.h"

namespace lczero {

Position::Position(const Position& parent, Move m)
    : no_capture_ply_(parent.no_capture_ply_ + 1),
      ply_count_(parent.ply_count_ + 1) {
  them_board_ = parent.us_board_;
  bool capture = them_board_.ApplyMove(m);
  us_board_ = them_board_;
  us_board_.Mirror();
  if (capture) no_capture_ply_ = 0;
}

Position::Position(const ChessBoard& board, int no_capture_ply, int game_ply)
    : no_capture_ply_(no_capture_ply), repetitions_(0), ply_count_(0) {
  us_board_ = board;
  them_board_ = board;
  them_board_.Mirror();
}

uint64_t Position::Hash() const {
  return HashCat({us_board_.Hash(), static_cast<unsigned long>(repetitions_)});
}

bool Position::CanCastle(Castling castling) const {
  auto cast = us_board_.castlings();
  switch (castling) {
    case WE_CAN_OOO:
      return cast.we_can_000();
    case WE_CAN_OO:
      return cast.we_can_00();
    case THEY_CAN_OOO:
      return cast.they_can_000();
    case THEY_CAN_OO:
      return cast.they_can_00();
  }
}

std::string Position::DebugString() const { return us_board_.DebugString(); }

GameResult PositionHistory::ComputeGameResult() const {
  const auto& board = Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  if (legal_moves.empty()) {
    if (board.IsUnderCheck()) {
      // Checkmate.
      return IsBlackToMove() ? GameResult::WHITE_WON : GameResult::BLACK_WON;
    }
    // Stalemate.
    return GameResult::DRAW;
  }

  if (!board.HasMatingMaterial()) return GameResult::DRAW;
  if (Last().GetNoCapturePly() >= 100) return GameResult::DRAW;
  if (Last().GetGamePly() >= 450) return GameResult::DRAW;
  if (Last().GetRepetitions() >= 2) return GameResult::DRAW;

  return GameResult::UNDECIDED;
}

void PositionHistory::Reset(const ChessBoard& board, int no_capture_ply,
                            int game_ply) {
  positions_.clear();
  positions_.emplace_back(board, no_capture_ply, game_ply);
}

void PositionHistory::Append(Move m) {
  // TODO(mooskagh) That should be emplace_back(Last(), m), but MSVS STL
  //                has a bug in implementation of emplace_back, when
  //                reallocation happens. (it also reallocates Last())
  positions_.push_back(Position(Last(), m));
  positions_.back().SetRepetitions(ComputeLastMoveRepetitions());
}

int PositionHistory::ComputeLastMoveRepetitions() const {
  const auto& last = positions_.back();
  // TODO(crem) implement hash/cache based solution.
  if (last.GetNoCapturePly() < 4) return 0;

  for (int idx = positions_.size() - 3; idx >= 0; idx -= 2) {
    const auto& pos = positions_[idx];
    if (pos.GetBoard() == last.GetBoard()) {
      return 1 + pos.GetRepetitions();
    }
    if (pos.GetNoCapturePly() < 2) return 0;
  }
  return 0;
}

uint64_t PositionHistory::HashLast(int positions) const {
  uint64_t hash = positions;
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (!positions--) break;
    hash = HashCat(hash, iter->Hash());
  }
  return HashCat(hash, Last().GetNoCapturePly());
}

}  // namespace lczero