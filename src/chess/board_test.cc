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
#include "src/chess/bitboard.h"
#include "src/chess/board.h"

namespace lczero {

TEST(BoardSquare, BoardSquare) {
  {
    auto x = BoardSquare(10);  // Should be c2
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare("c2");
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare(1, 2);
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare(1, 2);
    x.Mirror();
    EXPECT_EQ(x.row(), 6);
    EXPECT_EQ(x.col(), 2);
  }
}

TEST(ChessBoard, PseudolegalMovesStartingPos) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);
  board.Mirror();
  auto moves = board.GeneratePseudolegalMoves();

  EXPECT_EQ(moves.size(), 20);
}

namespace {
int Perft(const ChessBoard& board, int max_depth, bool dump = false,
          int depth = 0) {
  if (depth == max_depth) return 1;
  int total_count = 0;
  auto moves = board.GeneratePseudolegalMoves();

  auto legal_moves = board.GenerateLegalMoves();
  auto iter = legal_moves.begin();

  for (const auto& move : moves) {
    auto new_board = board;
    new_board.ApplyMove(move);
    if (new_board.IsUnderCheck()) continue;
    new_board.Mirror();

    EXPECT_EQ(iter->as_packed_int(), move.as_packed_int())
        << board.DebugString() << iter->as_string() << move.as_string();
    ++iter;

    int count = Perft(new_board, max_depth, dump, depth + 1);
    if (dump && depth == 0) {
      Move m = move;
      std::cerr << m.as_string() << ' ' << count << '\n'
                << new_board.DebugString();
    }
    total_count += count;
  }

  EXPECT_EQ(iter, legal_moves.end());
  return total_count;
}
}  // namespace

/* TEST(ChessBoard, MoveGenStartingPos) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);

  EXPECT_EQ(Perft(board, 0), 1);
  EXPECT_EQ(Perft(board, 1), 20);
  EXPECT_EQ(Perft(board, 2), 400);
  EXPECT_EQ(Perft(board, 3), 8902);
  EXPECT_EQ(Perft(board, 4), 197281);
  EXPECT_EQ(Perft(board, 5), 4865609);
  EXPECT_EQ(Perft(board, 6), 119060324);
} */

TEST(ChessBoard, MoveGenKiwipete) {
  ChessBoard board;
  board.SetFromFen(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 1 1");

  EXPECT_EQ(Perft(board, 1), 48);
  EXPECT_EQ(Perft(board, 2), 2039);
  EXPECT_EQ(Perft(board, 3), 97862);
  EXPECT_EQ(Perft(board, 4), 4085603);
  //  EXPECT_EQ(Perft(board, 5), 193690690);
}

TEST(ChessBoard, MoveGenPosition3) {
  ChessBoard board;
  board.SetFromFen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 1 1");

  EXPECT_EQ(Perft(board, 1), 14);
  EXPECT_EQ(Perft(board, 2), 191);
  EXPECT_EQ(Perft(board, 3), 2812);
  EXPECT_EQ(Perft(board, 4), 43238);
  EXPECT_EQ(Perft(board, 5), 674624);
  EXPECT_EQ(Perft(board, 6), 11030083);
}

TEST(ChessBoard, MoveGenPosition4) {
  ChessBoard board;
  board.SetFromFen(
      "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1");

  EXPECT_EQ(Perft(board, 1), 6);
  EXPECT_EQ(Perft(board, 2), 264);
  EXPECT_EQ(Perft(board, 3), 9467);
  EXPECT_EQ(Perft(board, 4), 422333);
  EXPECT_EQ(Perft(board, 5), 15833292);
}

TEST(ChessBoard, MoveGenPosition5) {
  ChessBoard board;
  board.SetFromFen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");

  EXPECT_EQ(Perft(board, 1), 44);
  EXPECT_EQ(Perft(board, 2), 1486);
  EXPECT_EQ(Perft(board, 3), 62379);
  EXPECT_EQ(Perft(board, 4), 2103487);
  //  EXPECT_EQ(Perft(board, 5), 89941194);
}

TEST(ChessBoard, MoveGenPosition6) {
  ChessBoard board;
  board.SetFromFen(
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 "
      "10");

  EXPECT_EQ(Perft(board, 1), 46);
  EXPECT_EQ(Perft(board, 2), 2079);
  EXPECT_EQ(Perft(board, 3), 89890);
  EXPECT_EQ(Perft(board, 4), 3894594);
}

TEST(ChessBoard, HasMatingMaterialStartPosition) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialBareKings) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/8/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialSingleMinorPiece) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/1N6/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/7b/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialSingleMajorPieceOrPawn) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/8/5R2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/5q2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/5P2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialTwoKnights) {
  ChessBoard board;
  board.SetFromFen("8/8/8/3nk3/8/5N2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/3nk3/8/5n2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialBishopAndKnight) {
  ChessBoard board;
  board.SetFromFen("8/8/8/3bk3/8/5N2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/3Bk3/8/5N2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialMultipleBishopsSameColor) {
  ChessBoard board;
  board.SetFromFen("8/8/8/3Bk3/8/5B2/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4kb2/8/2K2B2/8/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("B7/1B3b2/2B3b1/4k2b/8/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("B7/1B6/2B5/4k3/8/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialMultipleBishopsNotSameColor) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/8/2K1bb2/8/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/2K1Bb2/8/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/2K2b2/5B2/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("B7/1B3b2/2B3b1/4k2b/7B/8/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
