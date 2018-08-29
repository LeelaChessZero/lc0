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

#include "neural/encoder.h"
#include <algorithm>

namespace lczero {

namespace {
const int kMoveHistory = 8;
const int kPlanesPerBoard = 13;
const int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
}  // namespace

void SetInputPlanesFromBoard(InputPlanes& planes, int board_index,
                             const ChessBoard& board, int repetitions) {
  const int base = board_index * kPlanesPerBoard;
  planes[base + 0].mask = (board.ours() * board.pawns()).as_int();
  planes[base + 1].mask = (board.our_knights()).as_int();
  planes[base + 2].mask = (board.ours() * board.bishops()).as_int();
  planes[base + 3].mask = (board.ours() * board.rooks()).as_int();
  planes[base + 4].mask = (board.ours() * board.queens()).as_int();
  planes[base + 5].mask = (board.our_king()).as_int();

  planes[base + 6].mask = (board.theirs() * board.pawns()).as_int();
  planes[base + 7].mask = (board.their_knights()).as_int();
  planes[base + 8].mask = (board.theirs() * board.bishops()).as_int();
  planes[base + 9].mask = (board.theirs() * board.rooks()).as_int();
  planes[base + 10].mask = (board.theirs() * board.queens()).as_int();
  planes[base + 11].mask = (board.their_king()).as_int();

  if (repetitions >= 1) planes[base + 12].SetAll();
}

InputPlanes EncodePositionForNN(const PositionHistory& history,
                                int history_planes) {
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

  // Always fill all planes with valid chess positions.
  // Otherwise empty history positions yields nonsensical evaluations.
  const int history_positions = std::min(history.GetLength(), history_planes);
  const int fill_in_positions = kMoveHistory - history_positions;

  // First, fill in as many actual history positions as available.
  ChessBoard last_board_used;
  for (int i = 0; i < history_positions; i++) {
    const Position& position =
        history.GetPositionAt(history.GetLength() - 1 - i);
    const bool flip = (i % 2) == 1;
    const ChessBoard& board =
        flip ? position.GetThemBoard() : position.GetBoard();
    SetInputPlanesFromBoard(result, i, board, position.GetRepetitions());

    last_board_used = board;
  }

  // Second, fill in any remaining positions repeat of last available position.
  if (fill_in_positions > 0) {
    ChessBoard fill_in_board = ChessBoard(last_board_used);
    fill_in_board.UndoMoveToPriorBoardIfPossible();

    for (int i = 0; i < fill_in_positions; i++) {
      SetInputPlanesFromBoard(result, history_positions + i, fill_in_board, 0);
    }
  }

  return result;
}

}  // namespace lczero
