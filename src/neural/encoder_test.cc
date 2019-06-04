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

#include "src/neural/encoder.h"

namespace lczero {

auto kAllSquaresMask = std::numeric_limits<std::uint64_t>::max();

TEST(EncodePositionForNN, EncodeStartPosition) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  InputPlanes encoded_planes =
      EncodePositionForNN(history, 8, FillEmptyHistory::NO);

  InputPlane our_pawns_plane = encoded_planes[0];
  auto our_pawns_mask = 0ull;
  for (auto i = 0; i < 8; i++) {
    // First pawn is at square a2 (position 8)
    // Last pawn is at square h2 (position 8 + 7 = 15)
    our_pawns_mask |= 1ull << (8 + i);
  }
  EXPECT_EQ(our_pawns_plane.mask, our_pawns_mask);
  EXPECT_EQ(our_pawns_plane.value, 1.0f);

  InputPlane our_knights_plane = encoded_planes[1];
  EXPECT_EQ(our_knights_plane.mask, (1ull << 1) | (1ull << 6));
  EXPECT_EQ(our_knights_plane.value, 1.0f);

  InputPlane our_bishops_plane = encoded_planes[2];
  EXPECT_EQ(our_bishops_plane.mask, (1ull << 2) | (1ull << 5));
  EXPECT_EQ(our_bishops_plane.value, 1.0f);

  InputPlane our_rooks_plane = encoded_planes[3];
  EXPECT_EQ(our_rooks_plane.mask, 1ull | (1ull << 7));
  EXPECT_EQ(our_rooks_plane.value, 1.0f);

  InputPlane our_queens_plane = encoded_planes[4];
  EXPECT_EQ(our_queens_plane.mask, 1ull << 3);
  EXPECT_EQ(our_queens_plane.value, 1.0f);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 4);
  EXPECT_EQ(our_king_plane.value, 1.0f);

  // Sanity check opponent's pieces
  InputPlane their_king_plane = encoded_planes[11];
  auto their_king_row = 7;
  auto their_king_col = 4;
  EXPECT_EQ(their_king_plane.mask,
            1ull << (8 * their_king_row + their_king_col));
  EXPECT_EQ(their_king_plane.value, 1.0f);

  // Auxiliary planes

  // It's the start of the game, so all castlings should be allowed.
  for (auto i = 0; i < 4; i++) {
    InputPlane can_castle_plane = encoded_planes[13 * 8 + i];
    EXPECT_EQ(can_castle_plane.mask, kAllSquaresMask);
    EXPECT_EQ(can_castle_plane.value, 1.0f);
  }

  InputPlane we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 0.0f);

  // We no longer encode the move count, so that plane should be all zeros
  InputPlane zeroed_move_count_plane = encoded_planes[13 * 8 + 6];
  EXPECT_EQ(zeroed_move_count_plane.mask, 0ull);

  InputPlane all_ones_plane = encoded_planes[13 * 8 + 7];
  EXPECT_EQ(all_ones_plane.mask, kAllSquaresMask);
  EXPECT_EQ(all_ones_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeFiftyMoveCounter) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  // 1. Nf3
  history.Append(Move("g1f3", false));

  InputPlanes encoded_planes =
      EncodePositionForNN(history, 8, FillEmptyHistory::NO);

  InputPlane we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, kAllSquaresMask);
  EXPECT_EQ(we_are_black_plane.value, 1.0f);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 1.0f);

  // 1. Nf3 Nf6
  history.Append(Move("g8f6", true));

  encoded_planes = EncodePositionForNN(history, 8, FillEmptyHistory::NO);

  we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 2.0f);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
