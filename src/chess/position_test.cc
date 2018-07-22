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

#include <gtest/gtest.h>

#include <iostream>
#include "src/chess/position.h"

namespace lczero {

// https://github.com/LeelaChessZero/lc0/issues/209
TEST(PositionHistory, ComputeLastMoveRepetitions) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b1k2/rp1r4/8/1RP2p1p/p1KP1P2/P3P2P/8/1R2B3 b - - 0 31");
  history.Reset(board, 0, 0);
  history.Append(Move("d7h7", true));
  history.Append(Move("c4d3", false));
  history.Append(Move("h7d7", true));
  history.Append(Move("d3c4", false));
  int history_idx = history.GetLength() - 1;
  const Position& repeated_position = history.GetPositionAt(history_idx);
  EXPECT_EQ(repeated_position.GetRepetitions(), 1);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
