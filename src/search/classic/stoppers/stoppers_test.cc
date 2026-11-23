/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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
  combining with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "search/classic/stoppers/stoppers.h"

#include "gtest/gtest.h"
#include "search/classic/stoppers/timemgr.h"

namespace lczero {
namespace classic {
namespace {

TEST(StoppersHintsTest, VisitsStopperUpdatesRemainingNodesOnlyWhenFinite) {
  StoppersHints hints;
  IterationStats stats;
  stats.total_nodes = 250;
  stats.nodes_since_movestart = 150;

  VisitsStopper sentinel(4000000000, /*populate_remaining_playouts=*/false);
  EXPECT_FALSE(sentinel.ShouldStop(stats, &hints));
  EXPECT_FALSE(hints.HasEstimatedRemainingNodes());
  EXPECT_FALSE(hints.HasEstimatedRemainingTimeMs());

  VisitsStopper visits(1000, /*populate_remaining_playouts=*/false);
  EXPECT_FALSE(visits.ShouldStop(stats, &hints));
  EXPECT_TRUE(hints.HasEstimatedRemainingNodes());
  EXPECT_EQ(hints.GetEstimatedRemainingNodes(), 750);
  // populate_remaining_playouts=false must leave smart-pruning playouts unset.
  EXPECT_EQ(hints.GetEstimatedRemainingPlayouts(), 4000000000);
}

TEST(StoppersHintsTest, PlayoutsAndTimeStoppersPopulateBudgetsIndependently) {
  StoppersHints hints;
  IterationStats stats;
  stats.time_since_movestart = 120;
  stats.total_nodes = 500;
  stats.nodes_since_movestart = 200;

  PlayoutsStopper playouts(800, /*populate_remaining_playouts=*/false);
  EXPECT_FALSE(playouts.ShouldStop(stats, &hints));
  EXPECT_TRUE(hints.HasEstimatedRemainingNodes());
  EXPECT_EQ(hints.GetEstimatedRemainingNodes(), 600);
  EXPECT_FALSE(hints.HasEstimatedRemainingTimeMs());

  TimeLimitStopper timer(500);
  EXPECT_FALSE(timer.ShouldStop(stats, &hints));
  EXPECT_TRUE(hints.HasEstimatedRemainingTimeMs());
  EXPECT_EQ(hints.GetEstimatedRemainingTimeMs(), 380);

  hints.Reset();
  EXPECT_FALSE(hints.HasEstimatedRemainingNodes());
  EXPECT_FALSE(hints.HasEstimatedRemainingTimeMs());
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
