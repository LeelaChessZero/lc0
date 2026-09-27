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
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

// Tests for the dag_classic picking cache types.

#include "search/dag_classic/search.h"

#include "gtest/gtest.h"

namespace lczero {
namespace dag_classic {

// Friend of SearchWorker; re-exports its private picking-cache types.
class SearchWorkerTest {
 public:
  using CurrentPath = SearchWorker::CurrentPath;
  using TaskWorkspace = SearchWorker::TaskWorkspace;
};

namespace {

using CurrentPath = SearchWorkerTest::CurrentPath;
using TaskWorkspace = SearchWorkerTest::TaskWorkspace;

// The cache is reused across levels; a narrow level's own fill must win.
TEST(CachedNodeData, WorkspaceReuseDoesNotLeakBetweenCalls) {
  TaskWorkspace workspace;
  auto& cache = workspace.cache;

  // Round 1: a wide level fills 200 entries.
  for (int i = 0; i < 200; i++) {
    cache.children[i].utility = 111.0f;
    cache.children[i].uct_score = 222.0f;
    cache.children[i].n_started = 7;
  }
  ASSERT_FLOAT_EQ(cache.children[199].utility, 111.0f);

  // Round 2: a narrow level reuses the same cache.
  for (int i = 0; i < 5; i++) {
    cache.children[i].utility = 9.0f + i;
    cache.children[i].uct_score = 99.0f + i;
    cache.children[i].n_started = i;
  }
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(cache.children[i].utility, 9.0f + i)
        << "i=" << i << " -- round 2's own fill must win, not round 1's";
    EXPECT_EQ(cache.children[i].n_started, i);
  }
  // Entries past the narrow range are stale but never read.
  EXPECT_FLOAT_EQ(cache.children[199].utility, 111.0f)
      << "unread-this-round slot should still hold round 1's value";
}

// Re-initializing a reused visits_to_perform entry must clear its count.
TEST(CachedNodeData, VisitsToPerformResetPerIndexSurvivesReuse) {
  TaskWorkspace workspace;
  auto& vtp = workspace.cache.visits_to_perform;

  // Round 1: index 3 accumulates visits.
  vtp[3] = CurrentPath(0, false, false, false, 3);
  vtp[3] += 42u;
  EXPECT_EQ(vtp[3].visits_, 42u);

  // Round 2: a later level re-initializes the entry.
  vtp[3] = CurrentPath(0, false, false, false, 3);
  EXPECT_EQ(vtp[3].visits_, 0u)
      << "re-init must clear a stale visit count left by a previous level";
}

}  // namespace
}  // namespace dag_classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
