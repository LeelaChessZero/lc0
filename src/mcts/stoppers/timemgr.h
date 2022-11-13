/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include "chess/uciloop.h"
#include "mcts/node.h"
#include "utils/optionsdict.h"

namespace lczero {

// Various statistics that search sends to stoppers for their stopping decision.
// It is expected that this structure will grow.
struct IterationStats {
  int64_t time_since_movestart = 0;
  int64_t time_since_first_batch = 0;
  int64_t total_nodes = 0;
  int64_t nodes_since_movestart = 0;
  int64_t batches_since_movestart = 0;
  int average_depth = 0;
  float move_selection_visits_scaling_power = 0.0f;
  float override_PUCT_node_budget_threshold = 0.0f;  
  std::vector<uint32_t> edge_n;
  std::vector<float> q;

  // TODO: remove this in favor of time_usage_hint_=kImmediateMove when
  // smooth time manager is the default.
  bool win_found = false;
  int num_losing_edges = 0;

  enum class TimeUsageHint { kNormal, kNeedMoreTime, kImmediateMove };
  TimeUsageHint time_usage_hint_ = TimeUsageHint::kNormal;
};

// Hints from stoppers back to the search engine. Currently include:
// 1. EstimatedRemainingTime -- for search watchdog thread to know when to
// expect running out of time.
// 2. EstimatedPlayouts -- for smart pruning at root (not pick root nodes that
// cannot potentially become good).
class StoppersHints {
 public:
  StoppersHints();
  void Reset();
  void UpdateIndexOfBestEdge(int64_t v);
  int64_t GetIndexOfBestEdge() const;
  void UpdateEstimatedRemainingTimeMs(int64_t v);
  int64_t GetEstimatedRemainingTimeMs() const;
  void UpdateEstimatedRemainingPlayouts(int64_t v);
  int64_t GetEstimatedRemainingPlayouts() const;
  void UpdateEstimatedNps(float v);
  std::optional<float> GetEstimatedNps() const;

 private:
  int64_t index_of_best_edge_;
  int64_t remaining_time_ms_;
  int64_t remaining_playouts_;
  std::optional<float> estimated_nps_;
};

// Interface for search stopper.
// Note that:
// 1. Stoppers are shared between all search threads, so if stopper has mutable
// varibles, it has to think about concurrency (mutex/atomics)
// (maybe in future it will be changed).
// 2. IterationStats and StoppersHints are per search thread, so access to
// them is fine without synchronization.
// 3. OnSearchDone is guaranteed to be called once (i.e. from only one thread).
class SearchStopper {
 public:
  virtual ~SearchStopper() = default;
  // Question to a stopper whether search should stop.
  // Search statistics is sent via IterationStats, the stopper can optionally
  // send hints to the search through StoppersHints.
  virtual bool ShouldStop(const IterationStats&, StoppersHints*) = 0;
  // Is called when search is done.
  virtual void OnSearchDone(const IterationStats&) {}
};

class TimeManager {
 public:
  virtual ~TimeManager() = default;
  virtual std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                                    const NodeTree& tree) = 0;
};

}  // namespace lczero
