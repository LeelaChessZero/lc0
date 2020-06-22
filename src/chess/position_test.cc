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

#include "chess/position.h"

#include <gtest/gtest.h>

#include <iostream>

#include "utils/string.h"

namespace lczero {

TEST(Position, SetFenGetFen) {
  std::vector<Position> positions;
  ChessBoard board;
  std::vector<std::string> fen_splitted;
  std::vector<std::string> source_fens = {
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 1 1",
      // has en_passant space e3 - black to move
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq e3 1 1",
	  // has en_passant space c6 - white to move
	  "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 1 1",
      "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
      "3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
      "8/8/8/4k3/8/8/2K5/8 w - - 0 1", "8/8/8/4k3/1N6/8/2K5/8 w - - 0 1"};
  for (int i = 0; i < source_fens.size(); i++) {
    board.Clear();
    fen_splitted = StrSplitAtWhitespace(source_fens[i]);
    PositionHistory history;
    board.SetFromFen(source_fens[i]);
    history.Reset(board, std::stoi(fen_splitted[4]),
                  std::stoi(fen_splitted[5]));
    Position pos = history.Last();
    std::string target_fen = pos.GetFen();
    EXPECT_EQ(source_fens[i], target_fen);
  }
}

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
