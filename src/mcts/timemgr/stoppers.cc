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

#include "mcts/timemgr/stoppers.h"
#include "mcts/node.h"
#include "neural/cache.h"

namespace lczero {

///////////////////////////
// ChainedSearchStopper
///////////////////////////

bool ChainedSearchStopper::ShouldStop(const IterationStats& stats,
                                      TimeManagerHints* hints) {
  for (const auto& x : stoppers_) {
    if (x->ShouldStop(stats, hints)) return true;
  }
  return false;
}

void ChainedSearchStopper::AddStopper(std::unique_ptr<SearchStopper> stopper) {
  if (stopper) stoppers_.push_back(std::move(stopper));
}

void ChainedSearchStopper::OnSearchDone() {
  for (const auto& x : stoppers_) x->OnSearchDone();
}

///////////////////////////
// VisitsStopper
///////////////////////////

bool VisitsStopper::ShouldStop(const IterationStats& stats,
                               TimeManagerHints* hints) {
  hints->UpdateEstimatedRemainingRemainingPlayouts(stats.total_nodes -
                                                   nodes_limit_);
  if (stats.total_nodes >= nodes_limit_) {
    LOGFILE << "Stopped search: Reached visits limit: " << stats.total_nodes
            << ">=" << nodes_limit_;
    return true;
  }
  return false;
}

///////////////////////////
// MemoryWatchingStopper
///////////////////////////

namespace {
const size_t kAvgNodeSize =
    sizeof(Node) + MemoryWatchingStopper::kAvgMovesPerPosition * sizeof(Edge);
const size_t kAvgCacheItemSize =
    NNCache::GetItemStructSize() + sizeof(CachedNNRequest) +
    sizeof(CachedNNRequest::IdxAndProb) *
        MemoryWatchingStopper::kAvgMovesPerPosition;
}  // namespace

MemoryWatchingStopper::MemoryWatchingStopper(int cache_size, int ram_limit_mb)
    : VisitsStopper(
          (ram_limit_mb * 1000000LL - cache_size * kAvgCacheItemSize) /
          kAvgNodeSize) {
  LOGFILE << "RAM limit " << ram_limit_mb << "MB. Cache takes "
          << cache_size * kAvgCacheItemSize / 1000000
          << "MB. Remaining memory is enough for " << GetVisitsLimit()
          << " nodes.";
}

///////////////////////////
// MovetimeStopper
///////////////////////////

DeadlineStopper::DeadlineStopper(std::chrono::steady_clock::time_point deadline)
    : deadline_(deadline) {}

bool DeadlineStopper::ShouldStop(const IterationStats&,
                                 TimeManagerHints* hints) {
  const auto now = std::chrono::steady_clock::now();
  hints->UpdateEstimatedRemainingTime(
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline_ - now));
  if (now >= deadline_) {
    LOGFILE << "Stopping search: Ran out of time.";
    return true;
  }
  return false;
}

std::chrono::steady_clock::time_point DeadlineStopper::GetDeadline() const {
  return deadline_;
}

///////////////////////////
// DepthStopper
///////////////////////////
bool DepthStopper::ShouldStop(const IterationStats& stats, TimeManagerHints*) {
  if (stats.average_depth >= depth_) {
    LOGFILE << "Stopped search: Reached depth.";
  }
  return false;
}

///////////////////////////
// KldGainStopper
///////////////////////////

KldGainStopper::KldGainStopper(float min_gain, int average_interval)
    : min_gain_(min_gain), average_interval_(average_interval) {}

bool KldGainStopper::ShouldStop(const IterationStats& stats,
                                TimeManagerHints*) {
  Mutex::Lock lock(mutex_);

  const auto new_child_nodes = stats.total_nodes - 1;
  if (new_child_nodes < prev_child_nodes_ + average_interval_) return false;

  const auto new_visits = stats.edge_n;
  if (!prev_visits_.empty()) {
    float kldgain = 0.0f;
    for (decltype(new_visits)::size_type i = 0; i < new_visits.size(); i++) {
      float o_p = prev_visits_[i] / prev_child_nodes_;
      float n_p = new_visits[i] / new_child_nodes;
      if (prev_visits_[i] != 0) kldgain += o_p * log(o_p / n_p);
    }
    if (kldgain / (new_child_nodes - prev_child_nodes_) < min_gain_) {
      LOGFILE << "Stopping search: KLDGain per node too small.";
      return true;
    }
  }
  prev_visits_ = new_visits;
  prev_child_nodes_ = new_child_nodes;
  return false;
}

///////////////////////////
// SmartPruningStopper
///////////////////////////

namespace {
const int kSmartPruningToleranceNodes = 300;
const int kSmartPruningToleranceMs = 200;
}  // namespace

bool SmartPruningStopper::ShouldStop(const IterationStats& stats,
                                     TimeManagerHints* hints) {
  auto now = std::chrono::steady_clock::now();
  if (stats.total_playouts > 0 && !time_since_first_eval_) {
    time_since_first_eval_ = now;
    return false;
  }
  if (std::chrono::duration_cast<std::chrono::milliseconds>(
          now - time_since_first_eval_) < kSmartPruningToleranceMs) {
    return false;
  }

  // I"M WRITING HERE!
}

}  // namespace lczero