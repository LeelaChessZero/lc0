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
*/

#include "search/classic/stoppers/stoppers.h"

#include <gtest/gtest.h>

namespace lczero {
namespace classic {
namespace {

IterationStats MakeBaseStats() {
  IterationStats stats;
  stats.time_since_movestart = 100;
  stats.nodes_since_movestart = 500;
  stats.total_nodes = 500;
  stats.batches_since_movestart = 10;
  stats.edge_n = {300, 200};
  return stats;
}

StoppersHints MakeHints(int64_t remaining_ms = 10000,
                        int64_t remaining_playouts = 100000) {
  StoppersHints hints;
  hints.UpdateEstimatedRemainingTimeMs(remaining_ms);
  hints.UpdateEstimatedRemainingPlayouts(remaining_playouts);
  return hints;
}

TEST(SmartPruningStopper, DisabledWhenFactorZero) {
  SmartPruningStopper stopper(0.0f, 0);
  auto stats = MakeBaseStats();
  stats.edge_n = {100};
  stats.mate_depth = 1;
  stats.win_found = true;
  stats.win_is_most_visited = true;
  auto hints = MakeHints();
  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));
}

TEST(SmartPruningStopper, SingleMoveStopsImmediately) {
  SmartPruningStopper stopper(1.33f, 32);
  auto stats = MakeBaseStats();
  stats.edge_n = {10};
  stats.batches_since_movestart = 1;
  auto hints = MakeHints();
  EXPECT_TRUE(stopper.ShouldStop(stats, &hints));
}

TEST(SmartPruningStopper, MateInOneStopsImmediately) {
  SmartPruningStopper stopper(1.33f, 32);
  auto stats = MakeBaseStats();
  stats.mate_depth = 1;
  stats.win_found = true;
  stats.win_is_most_visited = false;
  stats.batches_since_movestart = 1;
  auto hints = MakeHints();
  EXPECT_TRUE(stopper.ShouldStop(stats, &hints));
}

TEST(SmartPruningStopper, SlowNonTbWinDoesNotStopWhileRivalLeads) {
  SmartPruningStopper stopper(1.33f, 0);
  auto stats = MakeBaseStats();
  stats.win_found = true;
  stats.win_is_most_visited = false;
  stats.mate_depth = 10;
  auto hints = MakeHints();

  // First call records first_eval_time_.
  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  // Past tolerance window (200ms), rival still leads in visits -> keep going.
  stats.time_since_movestart = 500;
  stats.nodes_since_movestart = 2000;
  stats.edge_n = {1500, 500};
  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  // Once the proven non-TB win strictly leads visits -> stop.
  stats.win_is_most_visited = true;
  EXPECT_TRUE(stopper.ShouldStop(stats, &hints));
}

TEST(SmartPruningStopper, TbOnlyWinFallsThroughToVisitGap) {
  SmartPruningStopper stopper(1.33f, 0);
  auto stats = MakeBaseStats();
  stats.win_found = true;
  stats.win_is_most_visited = false;  // TB-only win never sets this.
  stats.mate_depth = std::numeric_limits<int>::max();
  auto hints = MakeHints(10000, 100000);

  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  stats.time_since_movestart = 500;
  stats.nodes_since_movestart = 2000;
  stats.edge_n = {1100, 900};
  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  // Visit gap larger than remaining playouts still stops cleanly.
  auto tight_hints = MakeHints(10000, 100);
  stats.edge_n = {1800, 200};
  EXPECT_TRUE(stopper.ShouldStop(stats, &tight_hints));
}

TEST(SmartPruningStopper, MateInTwoWaitsForToleranceAndMinBatches) {
  SmartPruningStopper stopper(1.33f, 5);
  auto stats = MakeBaseStats();
  stats.mate_depth = 2;
  stats.win_found = true;
  stats.win_is_most_visited = false;
  stats.batches_since_movestart = 2;
  auto hints = MakeHints();

  // First eval records first_eval_time_.
  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  // Still inside tolerance window.
  stats.time_since_movestart = 250;
  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  // Past tolerance window, but below minimum_batches_.
  stats.time_since_movestart = 400;
  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  // Past tolerance window and at minimum_batches_ -> stops cleanly.
  stats.batches_since_movestart = 5;
  EXPECT_TRUE(stopper.ShouldStop(stats, &hints));
}

TEST(SmartPruningStopper, AtMostOneNonLosingRespectsMinimumBatches) {
  SmartPruningStopper stopper(1.33f, 5);
  auto stats = MakeBaseStats();
  stats.edge_n = {100, 100};
  stats.num_losing_edges = 1;
  stats.may_resign = false;
  stats.batches_since_movestart = 3;
  auto hints = MakeHints();

  EXPECT_FALSE(stopper.ShouldStop(stats, &hints));

  stats.batches_since_movestart = 5;
  EXPECT_TRUE(stopper.ShouldStop(stats, &hints));
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
