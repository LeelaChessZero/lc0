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

#include <vector>
#include "mcts/node.h"
#include "mcts/timemgr/timemgr.h"

namespace lczero {

class ChainedSearchStopper : public SearchStopper {
 public:
  ChainedSearchStopper() = default;
  bool ShouldStop(const IterationStats&, TimeManagerHints*) override;
  // Can be nullptr.
  void AddStopper(std::unique_ptr<SearchStopper> stopper);
  void OnSearchDone(const IterationStats&) override;

 private:
  std::vector<std::unique_ptr<SearchStopper>> stoppers_;
};

class VisitsStopper : public SearchStopper {
 public:
  VisitsStopper(int64_t limit) : nodes_limit_(limit) {}
  int64_t GetVisitsLimit() const { return nodes_limit_; }
  bool ShouldStop(const IterationStats&, TimeManagerHints*) override;

 private:
  const int64_t nodes_limit_;
};

class MemoryWatchingStopper : public VisitsStopper {
 public:
  // Must be in sync with description at kRamLimitMbId.
  static constexpr size_t kAvgMovesPerPosition = 30;
  MemoryWatchingStopper(int cache_size, int ram_limit_mb);
};

class TimeLimitStopper : public SearchStopper {
 public:
  TimeLimitStopper(int64_t time_limit_ms);
  bool ShouldStop(const IterationStats&, TimeManagerHints*) override;

 protected:
  int64_t GetTimeLimitMs() const;

 private:
  const int64_t time_limit_ms_;
};

class DepthStopper : public SearchStopper {
 public:
  DepthStopper(int depth) : depth_(depth) {}
  bool ShouldStop(const IterationStats&, TimeManagerHints*) override;

 private:
  const int depth_;
};

class KldGainStopper : public SearchStopper {
 public:
  KldGainStopper(float min_gain, int average_interval);
  bool ShouldStop(const IterationStats&, TimeManagerHints*) override;

 private:
  const int min_gain_;
  const int average_interval_;
  Mutex mutex_;
  std::vector<uint32_t> prev_visits_ GUARDED_BY(mutex_);
  int64_t prev_child_nodes_ GUARDED_BY(mutex_);
};

class SmartPruningStopper : public SearchStopper {
 public:
  SmartPruningStopper(float smart_pruning_factor);
  bool ShouldStop(const IterationStats&, TimeManagerHints*) override;

 private:
  const double smart_pruning_factor_;
  Mutex mutex_;
  optional<int64_t> first_eval_time_ GUARDED_BY(mutex_);
};

}  // namespace lczero