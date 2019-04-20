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

#include "chess/position.h"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cctype>

namespace lczero {

using std::string;

Position::Position(const Position& parent, Move m)
    : no_capture_ply_(parent.no_capture_ply_ + 1),
      ply_count_(parent.ply_count_ + 1) {
  them_board_ = parent.us_board_;
  const bool capture = them_board_.ApplyMove(m);
  us_board_ = them_board_;
  us_board_.Mirror();
  if (capture) no_capture_ply_ = 0;
}

Position::Position(const ChessBoard& board, int no_capture_ply, int game_ply)
    : no_capture_ply_(no_capture_ply), repetitions_(0), ply_count_(game_ply) {
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
  assert(false);
  return false;
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
  if (Last().GetNoCaptureNoPawnPly() >= 100) return GameResult::DRAW;
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
  if (last.GetNoCaptureNoPawnPly() < 4) return 0;

  for (int idx = positions_.size() - 3; idx >= 0; idx -= 2) {
    const auto& pos = positions_[idx];
    if (pos.GetBoard() == last.GetBoard()) {
      return 1 + pos.GetRepetitions();
    }
    if (pos.GetNoCaptureNoPawnPly() < 2) return 0;
  }
  return 0;
}

bool PositionHistory::DidRepeatSinceLastZeroingMove() const {
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (iter->GetRepetitions() > 0) return true;
    if (iter->GetNoCaptureNoPawnPly() == 0) return false;
  }
  return false;
}

uint64_t PositionHistory::HashLast(int positions) const {
  uint64_t hash = positions;
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (!positions--) break;
    hash = HashCat(hash, iter->Hash());
  }
  return HashCat(hash, Last().GetNoCaptureNoPawnPly());
}

// PrintFen outputs a FEN notation of the board. 
// based on https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation#Examples
string Position::GetFen()  {
	string result;
	string enpassant;
	ChessBoard board = GetBoard();
	bool fenflipped = false;
	if (board.flipped()) {
		board.Mirror();
		fenflipped = true;
	}
	int emptycounter = 0;
	for (int i = 7; i >= 0; --i) {
		for (int j = 0; j < 8; ++j) {
			if (emptycounter > 0 &&
				(
				    board.our_king().get(i,j) ||
					board.their_king().get(i,j) ||
					board.pawns().get(i,j) ||
					board.ours().get(i, j) ||
					board.theirs().get(i, j))) {
				result += std::to_string(emptycounter);
				emptycounter = 0;
			}
			if (board.our_king().get(i,j)) {
				result += 'K';
				continue;
			}
			if (board.their_king().get(i, j)) {
				result += 'k';
				continue;
			}
			if (board.ours().get(i, j) || board.theirs().get(i, j)) {
				char c = '?';
				if (board.pawns().get(i, j)) {
					c = 'p';
				}
				else if (board.bishops().get(i, j)) {
					c = 'b';
				}
				else if (board.queens().get(i, j)) {
					c = 'q';
				}
				else if (board.rooks().get(i, j)) {
					c = 'r';
				}
				else {
					c = 'n';
				}
				if (board.ours().get(i, j)) c = std::toupper(c); // capitals are for Black
				result += c;
			}
			else {
				emptycounter++;
			}			
		}
		if (emptycounter > 0) result += std::to_string(emptycounter);
		if (i > 0) result += "/";
		emptycounter = 0;
	}
	string castlings_no_fenflip = board.castlings().as_string();
	if (fenflipped) {
		board.Mirror();
	}
	enpassant = "-";
	for (auto sq : board.en_passant()) {
		// Our internal representation stores en_passant 2 rows away
		// from the actual sq.
		if (sq.row() == 0) {
			enpassant = ((BoardSquare)(sq.as_int() + 16)).as_string();
		}
		else {
			enpassant = ((BoardSquare)(sq.as_int() - 16)).as_string();
		}
	}
	result += IsBlackToMove() ? " b" : " w";
	result += " " + castlings_no_fenflip;
	result += " " + enpassant;
	result += " " + std::to_string(GetNoCaptureNoPawnPly());
	result += " " + std::to_string(ply_count_); 
	return result;
}

}  // namespace lczero
