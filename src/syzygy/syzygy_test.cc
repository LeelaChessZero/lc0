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

#include <gtest/gtest.h>

#include <iostream>
#include "src/syzygy/syzygy.h"

namespace lczero {

// Try to find syzygy relative to current working directory.
constexpr auto kPaths = "syzygy";

void TestValidRootExpectation(SyzygyTablebase* tablebase,
                              const std::string& fen,
                              const MoveList& valid_moves,
                              const MoveList& invalid_moves,
                              const MoveList& invalid_dtz_only = {},
                              bool has_repeated = false) {
  ChessBoard board;
  PositionHistory history;
  int rule50ply;
  int gameply;
  board.SetFromFen(fen, &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  MoveList allowed_moves_dtz;
  tablebase->root_probe(history.Last(), has_repeated, &allowed_moves_dtz);
  MoveList allowed_moves_wdl;
  tablebase->root_probe_wdl(history.Last(), &allowed_moves_wdl);
  for (auto move : valid_moves) {
    EXPECT_TRUE(std::find(allowed_moves_dtz.begin(), allowed_moves_dtz.end(),
                          move) != allowed_moves_dtz.end());
    EXPECT_TRUE(std::find(allowed_moves_wdl.begin(), allowed_moves_wdl.end(),
                          move) != allowed_moves_wdl.end());
  }
  for (auto move : invalid_moves) {
    EXPECT_FALSE(std::find(allowed_moves_dtz.begin(), allowed_moves_dtz.end(),
                           move) != allowed_moves_dtz.end());
    EXPECT_FALSE(std::find(allowed_moves_wdl.begin(), allowed_moves_wdl.end(),
                           move) != allowed_moves_wdl.end());
  }
  for (auto move : invalid_dtz_only) {
    EXPECT_FALSE(std::find(allowed_moves_dtz.begin(), allowed_moves_dtz.end(),
                           move) != allowed_moves_dtz.end());
    EXPECT_TRUE(std::find(allowed_moves_wdl.begin(), allowed_moves_wdl.end(),
                          move) != allowed_moves_wdl.end());
  }
}

void TestValidExpectation(SyzygyTablebase* tablebase, const std::string& fen,
                          WDLScore expected, int expected_dtz) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(fen);
  history.Reset(board, 0, 1);
  ProbeState result;
  WDLScore score = tablebase->probe_wdl(history.Last(), &result);
  EXPECT_NE(result, FAIL);
  EXPECT_EQ(score, expected);
  int moves = tablebase->probe_dtz(history.Last(), &result);
  EXPECT_NE(result, FAIL);
  EXPECT_EQ(moves, expected_dtz);
}

TEST(Syzygy, Simple3PieceProbes) {
  SyzygyTablebase tablebase;
  tablebase.init(kPaths);
  if (tablebase.max_cardinality() < 3) {
    // These probes require 3 piece tablebase.
    return;
  }
  // Longest 3 piece position.
  TestValidExpectation(&tablebase, "8/8/8/8/8/8/2Rk4/1K6 b - - 0 1", WDL_LOSS,
                       -32);
  // Invert color of above, no change.
  TestValidExpectation(&tablebase, "8/8/8/8/8/8/2rK4/1k6 w - - 0 1", WDL_LOSS,
                       -32);
  // Horizontal mirror.
  TestValidExpectation(&tablebase, "8/8/8/8/8/8/4kR2/6K1 b - - 0 1", WDL_LOSS,
                       -32);
  // Vertical mirror.
  TestValidExpectation(&tablebase, "6K1/4kR2/8/8/8/8/8/8 b - - 0 1", WDL_LOSS,
                       -32);
  // Horizontal mirror again.
  TestValidExpectation(&tablebase, "1K6/2Rk4/8/8/8/8/8/8 b - - 0 1", WDL_LOSS,
                       -32);

  // A draw by capture position, leaving KvK.
  TestValidExpectation(&tablebase, "5Qk1/8/8/8/8/8/8/4K3 b - - 0 1", WDL_DRAW,
                       0);

  // A position with a pawn which is to move and win from there.
  TestValidExpectation(&tablebase, "6k1/8/8/8/8/5p2/8/2K5 b - - 0 1", WDL_WIN,
                       1);

  // A position with a pawn that needs a king move first to win.
  TestValidExpectation(&tablebase, "8/8/8/8/8/k1p5/8/3K4 b - - 0 1", WDL_WIN,
                       3);

  // A position with a pawn that needs a few king moves before its a loss.
  TestValidExpectation(&tablebase, "8/2p5/8/8/8/5k2/8/2K5 w - - 0 1", WDL_LOSS,
                       -8);
}

TEST(Syzygy, Root3PieceProbes) {
  SyzygyTablebase tablebase;
  tablebase.init(kPaths);
  if (tablebase.max_cardinality() < 3) {
    // These probes require 3 piece tablebase.
    return;
  }
  TestValidRootExpectation(&tablebase, "5Qk1/8/8/8/8/8/8/4K3 b - - 0 1",
                           {Move("g8f8", true)}, {Move("g8h7", true)});
  TestValidRootExpectation(&tablebase, "6k1/8/8/8/8/5p2/8/2K5 b - - 0 1",
                           {Move("f3f2", true)}, {Move("g8h7", true)});
  TestValidRootExpectation(&tablebase, "8/8/8/8/8/k1p5/8/3K4 b - - 0 1",
                           {Move("a3b3", true)}, {Move("c3c2", true)});
  // WDL doesn't know that with such a high 50 ply count this position has
  // become a blessed loss (draw) for black.
  TestValidRootExpectation(&tablebase, "8/8/8/8/8/8/2Rk4/1K6 b - - 69 71",
                           {Move("d2d3", true)}, {}, {Move("d2e3", true)});
}

TEST(Syzygy, Simple4PieceProbes) {
  SyzygyTablebase tablebase;
  tablebase.init(kPaths);
  if (tablebase.max_cardinality() < 4) {
    // These probes require 4 piece tablebase.
    return;
  }

  // Longest 4 piece position.
  TestValidExpectation(&tablebase, "8/8/8/6B1/8/8/4k3/1K5N b - - 0 1", WDL_LOSS,
                       -65);

  // Some random checkmate position.
  TestValidExpectation(&tablebase, "8/8/8/8/8/2p5/3q2k1/4K3 w - - 0 1",
                       WDL_LOSS, -1);

  // Enpassant capture victory vs loss without rights.
  TestValidExpectation(&tablebase, "7k/8/8/8/Pp2K3/8/8/8 b - a3 0 1", WDL_WIN,
                       1);
  TestValidExpectation(&tablebase, "7k/8/8/8/Pp2K3/8/8/8 b - - 0 1", WDL_LOSS,
                       -1);
}

TEST(Syzygy, Simple5PieceProbes) {
  SyzygyTablebase tablebase;
  tablebase.init(kPaths);
  if (tablebase.max_cardinality() < 5) {
    // These probes require 5 piece tablebase.
    return;
  }

  // Longest 5 piece position.
  TestValidExpectation(&tablebase, "8/8/8/8/1p2P3/4P3/1k6/3K4 w - - 0 1",
                       WDL_CURSED_WIN, 101);

  // A blessed loss position.
  TestValidExpectation(&tablebase, "8/6B1/8/8/B7/8/K1pk4/8 b - - 0 1",
                       WDL_BLESSED_LOSS, -101);

  // A mate to be played on the board that is a capture.
  TestValidExpectation(&tablebase, "k7/p7/8/8/3Q4/8/5B2/7K w - - 0 1", WDL_WIN,
                       1);

  // Philidor draw position.
  TestValidExpectation(&tablebase, "8/8/8/8/4pk2/R7/7r/4K3 b - - 0 1", WDL_DRAW,
                       0);
  // Double mirrored and color swapped.
  TestValidExpectation(&tablebase, "3k4/R7/7r/2KP4/8/8/8/8 w - - 0 1", WDL_DRAW,
                       0);

  // En passant is a loss, without is draw by stalemate.
  TestValidExpectation(&tablebase, "8/8/8/8/6Pp/7K/5Q2/7k b - g3 0 1", WDL_LOSS,
                       -1);
  TestValidExpectation(&tablebase, "8/8/8/8/6Pp/7K/5Q2/7k b - - 0 1", WDL_DRAW,
                       0);

  // Some suggestions.
  TestValidExpectation(&tablebase, "kqqQK3/8/8/8/8/8/8/8 b - - 0 1", WDL_WIN,
                       1);
  TestValidExpectation(&tablebase, "kqqQK3/8/8/8/8/8/8/8 w - - 0 1", WDL_LOSS,
                       -2);
  TestValidExpectation(&tablebase, "KNNNk3/8/8/8/8/8/8/8 w - - 0 1", WDL_WIN,
                       30);
  TestValidExpectation(&tablebase, "8/1k6/1p1r4/5K2/8/8/8/2R5 w - - 0 1",
                       WDL_DRAW, 0);
  TestValidExpectation(&tablebase, "8/7p/5k2/8/5PK1/7P/8/8 b - - 0 1", WDL_DRAW,
                       0);
  TestValidExpectation(&tablebase, "1k1n4/8/p7/5KP1/8/8/8/8 b - - 0 1", WDL_WIN,
                       5);
  TestValidExpectation(&tablebase, "8/k7/8/2R5/8/4q3/8/4B2K w - - 0 1",
                       WDL_DRAW, 0);
}

TEST(Syzygy, Root5PieceProbes) {
  SyzygyTablebase tablebase;
  tablebase.init(kPaths);
  if (tablebase.max_cardinality() < 5) {
    // These probes require 5 piece tablebase.
    return;
  }
  TestValidRootExpectation(&tablebase, "8/8/8/Q7/8/1k1K4/1r6/8 w - - 79 44",
                           {Move("a5a1", false)}, {}, {Move("a5d5", false)});
  TestValidRootExpectation(&tablebase, "8/8/8/3Q4/k7/3K4/1r6/8 w - - 81 45",
                           {Move("d5a8", false)}, {}, {Move("d3c3", false)});

  // Variant of first test but with plenty of moves left.
  TestValidRootExpectation(&tablebase, "8/8/8/Q7/8/1k1K4/1r6/8 w - - 60 44",
                           {Move("a5a1", false), Move("a5d5", false)}, {}, {});
  // Same, but this time there is a repetition in history, so dtz will enforce
  // choice of equal lowest dtz.
  TestValidRootExpectation(&tablebase, "8/8/8/Q7/8/1k1K4/1r6/8 w - - 60 44",
                           {Move("a5a1", false)}, {}, {Move("a5d5", false)},
                           true);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
