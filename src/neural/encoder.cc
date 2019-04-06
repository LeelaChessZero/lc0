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

#include "neural/encoder.h"
#include <algorithm>
#include "utils/optional.h"

namespace lczero {

namespace {
const int kMoveHistory = 8;
const int kPlanesPerBoard = 13;
const int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
}  // namespace

InputPlanes EncodePositionForNN(const PositionHistory& history,
                                int history_planes,
                                FillEmptyHistory fill_empty_history) {
  InputPlanes result(kAuxPlaneBase + 8);

  {
    const ChessBoard& board = history.Last().GetBoard();
    const bool we_are_black = board.flipped();
    if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
    if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
    if (board.castlings().they_can_000()) result[kAuxPlaneBase + 2].SetAll();
    if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
    if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
    result[kAuxPlaneBase + 5].Fill(history.Last().GetNoCaptureNoPawnPly());
    // Plane kAuxPlaneBase + 6 used to be movecount plane, now it's all zeros.
    // Plane kAuxPlaneBase + 7 is all ones to help NN find board edges.
    result[kAuxPlaneBase + 7].SetAll();
  }

  bool flip = false;
  int history_idx = history.GetLength() - 1;
  for (int i = 0; i < std::min(history_planes, kMoveHistory);
       ++i, --history_idx) {
    const Position& position =
        history.GetPositionAt(history_idx < 0 ? 0 : history_idx);
    const ChessBoard& board =
        flip ? position.GetThemBoard() : position.GetBoard();
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::NO) break;
    // Board may be flipped so compare with position.GetBoard().
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
        position.GetBoard() == ChessBoard::kStartposBoard) {
      break;
    }

    const int base = i * kPlanesPerBoard;
    result[base + 0].mask = (board.ours() & board.pawns()).as_int();
    result[base + 1].mask = (board.our_knights()).as_int();
    result[base + 2].mask = (board.ours() & board.bishops()).as_int();
    result[base + 3].mask = (board.ours() & board.rooks()).as_int();
    result[base + 4].mask = (board.ours() & board.queens()).as_int();
    result[base + 5].mask = (board.our_king()).as_int();

    result[base + 6].mask = (board.theirs() & board.pawns()).as_int();
    result[base + 7].mask = (board.their_knights()).as_int();
    result[base + 8].mask = (board.theirs() & board.bishops()).as_int();
    result[base + 9].mask = (board.theirs() & board.rooks()).as_int();
    result[base + 10].mask = (board.theirs() & board.queens()).as_int();
    result[base + 11].mask = (board.their_king()).as_int();

    const int repetitions = position.GetRepetitions();
    if (repetitions >= 1) result[base + 12].SetAll();

    // If en passant flag is set, undo last pawn move by removing the pawn from
    // the new square and putting into pre-move square.
    if (history_idx < 0 && !board.en_passant().empty()) {
      const auto idx = GetLowestBit(board.en_passant().as_int());
      if (idx < 8) {  // "Us" board
        result[base + 0].mask +=
            ((0x0000000000000100ULL - 0x0000000001000000ULL) << idx);
      } else {
        result[base + 6].mask +=
            ((0x0001000000000000ULL - 0x0000000100000000ULL) << (idx - 56));
      }
    }
    if (history_idx > 0) flip = !flip;
  }

  return result;
}

namespace {
BoardSquare SingleSquare(BitBoard input) { 
  for (auto sq : input) {
    return sq;
  }
  assert(false);
  return BoardSquare();
}
}

Move DecodeMoveFromInput(const InputPlanes& planes) {
  auto pawndiff = BitBoard(planes[6].mask ^ planes[kPlanesPerBoard + 6].mask);
  auto knightdiff = BitBoard(planes[7].mask ^ planes[kPlanesPerBoard + 7].mask);
  auto bishopdiff = BitBoard(planes[8].mask ^ planes[kPlanesPerBoard + 8].mask);
  auto rookdiff = BitBoard(planes[9].mask ^ planes[kPlanesPerBoard + 9].mask);
  auto queendiff = 
    BitBoard(planes[10].mask ^ planes[kPlanesPerBoard + 10].mask);
  // Handle Promotion.
  if (pawndiff.count() == 1) {
    auto from = SingleSquare(pawndiff);
    if (knightdiff.count() == 1) {
      auto to = SingleSquare(knightdiff);
      return Move(from, to, Move::Promotion::Knight);
    }
    if (bishopdiff.count() == 1) {
      auto to = SingleSquare(bishopdiff);
      return Move(from, to, Move::Promotion::Bishop);
    }
    if (rookdiff.count() == 1) {
      auto to = SingleSquare(rookdiff);
      return Move(from, to, Move::Promotion::Rook);
    }
    if (queendiff.count() == 1) {
      auto to = SingleSquare(queendiff);
      return Move(from, to, Move::Promotion::Queen);
    }
    assert(false);
    return Move();
  }
  // check king first as castling moves both king and rook.
  auto kingdiff = BitBoard(planes[11].mask ^ planes[kPlanesPerBoard + 11].mask);
  if (kingdiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 11].mask & kingdiff.as_int());
    auto to = SingleSquare(planes[11].mask & kingdiff.as_int());
    return Move(from, to);
  }
  if (queendiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 10].mask & queendiff.as_int());
    auto to = SingleSquare(planes[10].mask & queendiff.as_int());
    return Move(from, to);
  }
  if (rookdiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 9].mask & rookdiff.as_int());
    auto to = SingleSquare(planes[9].mask & rookdiff.as_int());
    return Move(from, to);
  }
  if (bishopdiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 8].mask & bishopdiff.as_int());
    auto to = SingleSquare(planes[8].mask & bishopdiff.as_int());
    return Move(from, to);
  }
  if (knightdiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 7].mask & knightdiff.as_int());
    auto to = SingleSquare(planes[7].mask & knightdiff.as_int());
    return Move(from, to);
  }
  if (pawndiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 6].mask & pawndiff.as_int());
    auto to = SingleSquare(planes[6].mask & pawndiff.as_int());
    return Move(from, to);
  }
  assert(false);
  return Move();
}

}  // namespace lczero
