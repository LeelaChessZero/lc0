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

// Tests for the classic picking cache types.

#include "search/classic/search.h"
#include "gtest/gtest.h"

namespace lczero {
namespace classic {

// Friend of SearchWorker; re-exports its private picking-cache types.
class SearchWorkerTest {
 public:
  using InlineDepthStack = SearchWorker::InlineDepthStack;
  using TaskWorkspace = SearchWorker::TaskWorkspace;
};

namespace {

using InlineDepthStack = SearchWorkerTest::InlineDepthStack;
using TaskWorkspace = SearchWorkerTest::TaskWorkspace;

TEST(InlineDepthStack, InlinePathGatesExactlyByCount) {
  InlineDepthStack s;
  EXPECT_EQ(s.count, 0);
  for (int i = 0; i < InlineDepthStack::kInlineCapacity; ++i) s.push_back(i);
  EXPECT_EQ(s.count, InlineDepthStack::kInlineCapacity);
  EXPECT_EQ(s.back(), InlineDepthStack::kInlineCapacity - 1);
  for (int i = 0; i < InlineDepthStack::kInlineCapacity; ++i) {
    ASSERT_EQ(s.back(), InlineDepthStack::kInlineCapacity - 1 - i);
    s.pop_back();
  }
  EXPECT_EQ(s.count, 0);
  s.pop_back();  // popping empty is a no-op, must not go negative
  EXPECT_EQ(s.count, 0);
}

TEST(InlineDepthStack, SpillsPastInlineCapacity) {
  InlineDepthStack s;
  for (int i = 0; i < 257; ++i) s.push_back(i);  // one past inline capacity
  EXPECT_EQ(s.count, 257);
  EXPECT_EQ(s.back(), 256);
  s.back() = 12345;  // back() is writable
  EXPECT_EQ(s.back(), 12345);
  s.pop_back();  // drops the spilled entry
  EXPECT_EQ(s.count, InlineDepthStack::kInlineCapacity);
  EXPECT_EQ(s.back(), 255);  // last inline slot
  s.pop_back();
  EXPECT_EQ(s.count, 255);
  EXPECT_EQ(s.back(), 254);
}

TEST(InlineDepthStack, DeepDescentAndReuse) {
  InlineDepthStack s;
  for (int i = 0; i < 4000; ++i) s.push_back(-1);
  for (int i = 0; i < 4000; ++i) s.pop_back();
  EXPECT_EQ(s.count, 0);
  // Reuse after clear().
  for (int i = 0; i < 600; ++i) s.push_back(i);
  s.clear();
  EXPECT_EQ(s.count, 0);
  s.push_back(7);
  s.push_back(8);
  EXPECT_EQ(s.back(), 8);
}

// The cache is reused across levels; a narrow level's own fill must win.
TEST(CachedNodeData, WorkspaceReuseDoesNotLeakBetweenCalls) {
  TaskWorkspace workspace;
  Node wide(nullptr, 0);
  wide.CreateEdges(MoveList(200));
  {
    int idx = 0;
    for (auto& edge : wide.Edges()) edge.edge()->SetP(1.0f / (++idx + 1));
  }
  Node narrow(nullptr, 0);
  narrow.CreateEdges(MoveList(5));
  {
    int idx = 0;
    for (auto& edge : narrow.Edges()) edge.edge()->SetP(9.0f + idx++);
  }

  // Round 1: a wide level fills 200 entries.
  auto& cache = workspace.cache;
  cache.cache_filled_idx = -1;
  cache.max_policy_entries_needed = 200;
  for (int i = 0; i < cache.max_policy_entries_needed; i++) {
    cache.children[i].policy = wide.GetEdgeP(i);
    cache.children[i].utility = 111.0f;
  }
  ASSERT_FLOAT_EQ(cache.children[199].policy, wide.GetEdgeP(199));

  // Round 2: a narrow level reuses the same cache.
  cache.cache_filled_idx = -1;
  cache.max_policy_entries_needed = 5;
  for (int i = 0; i < cache.max_policy_entries_needed; i++) {
    cache.children[i].policy = narrow.GetEdgeP(i);
  }
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(cache.children[i].policy, narrow.GetEdgeP(i))
        << "i=" << i << " -- round 2's own fill must win, not round 1's";
  }
  // Entries past the narrow range are stale but never read.
  EXPECT_FLOAT_EQ(cache.children[199].policy, wide.GetEdgeP(199))
      << "unread-this-round slot should still hold round 1's value";
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
