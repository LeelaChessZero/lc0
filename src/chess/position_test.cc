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
*/

#include <gtest/gtest.h>

#include <iostream>
#include "src/chess/position.h"

namespace lczero {

// https://github.com/LeelaChessZero/lc0/issues/209
TEST(PositionHistory, ComputeLastMoveRepetitionsWithoutLegalEnPassant) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(Move("f7f8", true));
  history.Append(Move("f2f4", false));
  history.Append(Move("d7h7", true));
  history.Append(Move("c4d3", false));
  history.Append(Move("h7d7", true));
  history.Append(Move("d3c4", false));
  int history_idx = history.GetLength() - 1;
  const Position& repeated_position = history.GetPositionAt(history_idx);
  EXPECT_EQ(repeated_position.GetRepetitions(), 1);
}

TEST(PositionHistory, ComputeLastMoveRepetitionsWithLegalEnPassant) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP2p1/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(Move("f7f8", true));
  history.Append(Move("f2f4", false));
  history.Append(Move("d7h7", true));
  history.Append(Move("c4d3", false));
  history.Append(Move("h7d7", true));
  history.Append(Move("d3c4", false));
  int history_idx = history.GetLength() - 1;
  const Position& repeated_position = history.GetPositionAt(history_idx);
  EXPECT_EQ(repeated_position.GetRepetitions(), 0);
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveCurent) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(Move("f7f8", true));
  history.Append(Move("f2f4", false));
  history.Append(Move("d7h7", true));
  history.Append(Move("c4d3", false));
  history.Append(Move("h7d7", true));
  history.Append(Move("d3c4", false));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveBefore) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(Move("f7f8", true));
  history.Append(Move("f2f4", false));
  history.Append(Move("d7h7", true));
  history.Append(Move("c4d3", false));
  history.Append(Move("h7d7", true));
  history.Append(Move("d3c4", false));
  history.Append(Move("d7e7", true));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveOlder) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(Move("f7f8", true));
  history.Append(Move("f2f4", false));
  history.Append(Move("d7h7", true));
  history.Append(Move("c4d3", false));
  history.Append(Move("h7d7", true));
  history.Append(Move("d3c4", false));
  history.Append(Move("d7e7", true));
  history.Append(Move("c4b4", false));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveBeforeZero) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(Move("f7f8", true));
  history.Append(Move("f2f4", false));
  history.Append(Move("d7h7", true));
  history.Append(Move("c4d3", false));
  history.Append(Move("h7d7", true));
  history.Append(Move("d3c4", false));
  history.Append(Move("d7e7", true));
  history.Append(Move("c4b4", false));
  history.Append(Move("h5h4", true));
  EXPECT_FALSE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveNeverRepeated) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(Move("f7f8", true));
  history.Append(Move("f2f4", false));
  EXPECT_FALSE(history.DidRepeatSinceLastZeroingMove());
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
