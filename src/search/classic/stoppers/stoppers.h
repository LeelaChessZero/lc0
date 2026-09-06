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

#include <optional>
#include <vector>

#include "search/classic/stoppers/timemgr.h"

namespace lczero {
namespace classic {

// Combines multiple stoppers into one.
class ChainedSearchStopper : public SearchStopper {
 public:
  ChainedSearchStopper() = default;
  // Calls stoppers one by one until one of them returns true. If one of
  // stoppers modifies hints, next stoppers in the chain see that.
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
  // Can be nullptr, in that canse stopper is not added.
  void AddStopper(std::unique_ptr<SearchStopper> stopper);
  void OnSearchDone(const IterationStats&) override;

 private:
  std::vector<std::unique_ptr<SearchStopper>> stoppers_;
};

// Watches visits (total tree nodes) and predicts remaining visits.
class VisitsStopper : public SearchStopper {
 public:
  VisitsStopper(int64_t limit, bool populate_remaining_playouts)
      : nodes_limit_(limit),
        populate_remaining_playouts_(populate_remaining_playouts) {}
  int64_t GetVisitsLimit() const { return nodes_limit_; }
  bool ShouldStop(const IterationStats&, StoppersHints*) override;

 private:
  const int64_t nodes_limit_;
  const bool populate_remaining_playouts_;
};

// Watches playouts (new tree nodes) and predicts remaining visits.
class PlayoutsStopper : public SearchStopper {
 public:
  PlayoutsStopper(int64_t limit, bool populate_remaining_playouts)
      : nodes_limit_(limit),
        populate_remaining_playouts_(populate_remaining_playouts) {}
  int64_t GetVisitsLimit() const { return nodes_limit_; }
  bool ShouldStop(const IterationStats&, StoppersHints*) override;

 private:
  const int64_t nodes_limit_;
  const bool populate_remaining_playouts_;
};

// Computes tree size which may fit into the memory and limits by that tree
// size.
class MemoryWatchingStopper : public VisitsStopper {
 public:
  // Must be in sync with description at kRamLimitMbId.
  static constexpr size_t kAvgMovesPerPosition = 30;
  MemoryWatchingStopper(int ram_limit_mb, size_t total_memory,
                        size_t avg_node_size, uint32_t nodes,
                        bool populate_remaining_playouts);
};

// Stops after time budget is gone.
class TimeLimitStopper : public SearchStopper {
 public:
  TimeLimitStopper(int64_t time_limit_ms);
  bool ShouldStop(const IterationStats&, StoppersHints*) override;

 protected:
  int64_t GetTimeLimitMs() const;

 private:
  const int64_t time_limit_ms_;
};

// Stops when certain average depth is reached (who needs that?).
class DepthStopper : public SearchStopper {
 public:
  DepthStopper(int depth) : depth_(depth) {}
  bool ShouldStop(const IterationStats&, StoppersHints*) override;

 private:
  const int depth_;
};

// Stops when a mate at specified depth (or less) is found.
class MateStopper : public SearchStopper {
 public:
  MateStopper(int mate) : mate_(mate) {}
  bool ShouldStop(const IterationStats&, StoppersHints*) override;

 private:
  const int mate_;
};

// Stops when search doesn't bring required KLD gain.
class KldGainStopper : public SearchStopper {
 public:
  KldGainStopper(float min_gain, int average_interval);
  bool ShouldStop(const IterationStats&, StoppersHints*) override;

 private:
  const double min_gain_;
  const int average_interval_;
  Mutex mutex_;
  std::vector<uint32_t> prev_visits_ GUARDED_BY(mutex_);
  double prev_child_nodes_ GUARDED_BY(mutex_) = 0.0;
};

// Does many things:
// Computes how many nodes are remaining (from remaining time/nodes, scaled by
// smart pruning factor). When this amount of nodes is not enough for second
// best move to potentially become the best one, stop the search.
class SmartPruningStopper : public SearchStopper {
 public:
  SmartPruningStopper(float smart_pruning_factor, int64_t minimum_batches);
  bool ShouldStop(const IterationStats&, StoppersHints*) override;

 private:
  const double smart_pruning_factor_;
  const int64_t minimum_batches_;
  Mutex mutex_;
  std::optional<int64_t> first_eval_time_ GUARDED_BY(mutex_);
};

}  // namespace classic
}  // namespace lczero
