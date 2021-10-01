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

#include "mcts/stoppers/stoppers.h"

#include "mcts/node.h"
#include "neural/cache.h"

namespace lczero {

///////////////////////////
// ChainedSearchStopper
///////////////////////////

bool ChainedSearchStopper::ShouldStop(const IterationStats& stats,
                                      StoppersHints* hints) {
  for (const auto& x : stoppers_) {
    if (x->ShouldStop(stats, hints)) return true;
  }
  return false;
}

void ChainedSearchStopper::AddStopper(std::unique_ptr<SearchStopper> stopper) {
  if (stopper) stoppers_.push_back(std::move(stopper));
}

void ChainedSearchStopper::OnSearchDone(const IterationStats& stats) {
  for (const auto& x : stoppers_) x->OnSearchDone(stats);
}

///////////////////////////
// VisitsStopper
///////////////////////////

bool VisitsStopper::ShouldStop(const IterationStats& stats,
                               StoppersHints* hints) {
  if (populate_remaining_playouts_) {
    hints->UpdateEstimatedRemainingPlayouts(nodes_limit_ - stats.total_nodes);
  }
  if (stats.total_nodes >= nodes_limit_) {
    LOGFILE << "Stopped search: Reached visits limit: " << stats.total_nodes
            << ">=" << nodes_limit_;
    return true;
  }
  return false;
}

///////////////////////////
// PlayoutsStopper
///////////////////////////

bool PlayoutsStopper::ShouldStop(const IterationStats& stats,
                                 StoppersHints* hints) {
  if (populate_remaining_playouts_) {
    hints->UpdateEstimatedRemainingPlayouts(nodes_limit_ -
                                            stats.nodes_since_movestart);
  }
  if (stats.nodes_since_movestart >= nodes_limit_) {
    LOGFILE << "Stopped search: Reached playouts limit: "
            << stats.nodes_since_movestart << ">=" << nodes_limit_;
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

MemoryWatchingStopper::MemoryWatchingStopper(int cache_size, int ram_limit_mb,
                                             bool populate_remaining_playouts)
    : VisitsStopper(
          (ram_limit_mb * 1000000LL - cache_size * kAvgCacheItemSize) /
              kAvgNodeSize,
          populate_remaining_playouts) {
  LOGFILE << "RAM limit " << ram_limit_mb << "MB. Cache takes "
          << cache_size * kAvgCacheItemSize / 1000000
          << "MB. Remaining memory is enough for " << GetVisitsLimit()
          << " nodes.";
}

///////////////////////////
// TimelimitStopper
///////////////////////////

TimeLimitStopper::TimeLimitStopper(int64_t time_limit_ms)
    : time_limit_ms_(time_limit_ms) {}

bool TimeLimitStopper::ShouldStop(const IterationStats& stats,
                                  StoppersHints* hints) {
  hints->UpdateEstimatedRemainingTimeMs(time_limit_ms_ -
                                        stats.time_since_movestart);
  if (stats.time_since_movestart >= time_limit_ms_) {
    LOGFILE << "Stopping search: Ran out of time.";
    return true;
  }
  return false;
}

int64_t TimeLimitStopper::GetTimeLimitMs() const { return time_limit_ms_; }

///////////////////////////
// DepthStopper
///////////////////////////
bool DepthStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  if (stats.average_depth >= depth_) {
    LOGFILE << "Stopped search: Reached depth.";
    return true;
  }
  return false;
}

///////////////////////////
// KldGainStopper
///////////////////////////

KldGainStopper::KldGainStopper(float min_gain, int average_interval)
    : min_gain_(min_gain), average_interval_(average_interval) {}

bool KldGainStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  Mutex::Lock lock(mutex_);
  const auto new_child_nodes = stats.total_nodes - 1.0;
  if (new_child_nodes < prev_child_nodes_ + average_interval_) return false;

  const auto new_visits = stats.edge_n;
  if (!prev_visits_.empty()) {
    double kldgain = 0.0;
    for (decltype(new_visits)::size_type i = 0; i < new_visits.size(); i++) {
      double o_p = prev_visits_[i] / prev_child_nodes_;
      double n_p = new_visits[i] / new_child_nodes;
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
const int kSmartPruningToleranceMs = 200;
const int kSmartPruningToleranceNodes = 300;
}  // namespace

SmartPruningStopper::SmartPruningStopper(float smart_pruning_factor,
                                         int64_t minimum_batches)
    : smart_pruning_factor_(smart_pruning_factor),
      minimum_batches_(minimum_batches) {}

bool SmartPruningStopper::ShouldStop(const IterationStats& stats,
                                     StoppersHints* hints) {
  if (smart_pruning_factor_ <= 0.0) return false;
  Mutex::Lock lock(mutex_);
  if (stats.edge_n.size() == 1) {
    LOGFILE << "Only one possible move. Moving immediately.";
    return true;
  }
  if (stats.edge_n.size() <= static_cast<size_t>(stats.num_losing_edges + 1)) {
    LOGFILE << "At most one non losing move, stopping search.";
    return true;
  }
  if (stats.win_found) {
    LOGFILE << "Terminal win found, stopping search.";
    return true;
  }
  if (stats.nodes_since_movestart > 0 && !first_eval_time_) {
    first_eval_time_ = stats.time_since_movestart;
    return false;
  }
  if (!first_eval_time_) return false;
  if (stats.edge_n.size() == 0) return false;
  if (stats.time_since_movestart <
      *first_eval_time_ + kSmartPruningToleranceMs) {
    return false;
  }

  const auto nodes = stats.nodes_since_movestart + kSmartPruningToleranceNodes;
  const auto time = stats.time_since_movestart - *first_eval_time_;
  // If nps is populated by someone who knows better, use it. Otherwise use the
  // value calculated here.
  const auto nps = hints->GetEstimatedNps().value_or(1000LL * nodes / time + 1);

  const double remaining_time_s = hints->GetEstimatedRemainingTimeMs() / 1000.0;
  const auto remaining_playouts =
      std::min(remaining_time_s * nps / smart_pruning_factor_,
               hints->GetEstimatedRemainingPlayouts() / smart_pruning_factor_);

  // May overflow if (nps/smart_pruning_factor) > 180 000 000, but that's not
  // very realistic.
  hints->UpdateEstimatedRemainingPlayouts(remaining_playouts);
  if (stats.batches_since_movestart < minimum_batches_) return false;

  // Don't stop early unless node with highest visits also has the
  // highest Expected Q.

  // When calculating Expected Q, what prior is suitable? If we accept
  // pruning, then we should also play the move that made us think
  // pruning is appropriate, ie reject pruning if another move will be
  // played. When move selection is done,
  // stats.move_selection_visits_scaling_power and total node budget
  // is used. It is thus safe to reject when those priors give another
  // move than the most visited child. But we are free to reject at
  // lower level of certainty, e.g. at evaluted_nodes (aggressive), or
  // (evaluted_nodes + budget_nodes) / 2, or min(evaluated_nodes *
  // 1.2, budget_nodes).

  // If we reject we should also override
  // PUCT, otherwise most new nodes will be wasted on the most visited
  // child.

  // Can we do better than that by override PUCT even
  // when no pruning is suggested? The parameter
  // stats.override_PUCT_node_budget_threshold gives a threshold for
  // when we are allowed to try that. But what prior is suitable in that case?
  // for now just use the same prior.

  float beta_prior_base = std::min(nodes * 1.2, nodes + remaining_playouts);
  float beta_prior_scaler = stats.move_selection_visits_scaling_power;
  // float beta_prior_scaler = pow(nodes/(nodes + remaining_playouts), 0.5);
  const float beta_prior = pow(beta_prior_base, beta_prior_scaler);

  float highest_q = -1.0f;
  uint32_t my_largest_n = 0;
  long unsigned int index_of_highest_q = 0;
  long unsigned int index_of_largest_n = 0;

  // Calculate expected Q
  const float alpha_prior = 0.0f; // if set to 1.0f then there will be problems when Q == -1 (Expected Q will be larger than -1)
  float winrate = 0.0f;
  int visits = 0;
  float alpha = 0.0f;
  float beta = 0.0f;
  std::vector<float> expected_q(stats.q.size());
  // float expected_q = 0.0f;
  
  for (long unsigned int i = 0; i < stats.q.size(); i++) {
      winrate = (stats.q[i] + 1) * 0.5;
      visits = stats.edge_n[i];
      alpha = winrate * visits + alpha_prior;
      beta = visits - alpha + beta_prior;
      expected_q[i] = alpha / (alpha + beta);
      // transpose back to [-1, 1] to ease comparison with raw Q while debugging
      expected_q[i] = expected_q[i] * 2 - 1;
    
      if(expected_q[i] > highest_q){
	index_of_highest_q = i;
	highest_q = expected_q[i];
      }

    if(stats.edge_n[i] > my_largest_n){
      index_of_largest_n = i;
      my_largest_n = stats.edge_n[i];
    }
  }

  uint32_t largest_n = 0;
  uint32_t second_largest_n = 0;
  for (auto n : stats.edge_n) {
    if (n > largest_n) {
      second_largest_n = largest_n;
      largest_n = n;
    } else if (n > second_largest_n) {
      second_largest_n = n;
    }
  }

  float proportion_left = 1 - nodes/(nodes + remaining_playouts);

  if (remaining_playouts < (largest_n - second_largest_n)) {

    // Reject early stop if Expected Q and N disagrees.
    if(index_of_largest_n != index_of_highest_q){
      LOGFILE << "ratio evaluated/budgeted=" << nodes/(nodes + remaining_playouts) << " Rejected smart pruning since child (" << index_of_largest_n << ") is the child with largest n=" << stats.edge_n[index_of_largest_n] << ", but has lower Expected Q=" << expected_q[index_of_largest_n] << "(raw Q=" << stats.q[index_of_largest_n] << ") than child (" << index_of_highest_q << ") which has Expected Q=" << expected_q[index_of_highest_q] << "(raw Q=" << stats.q[index_of_highest_q] << ") and n=" << stats.edge_n[index_of_highest_q] << " beta_prior=" << beta_prior << " beta_prior_base=" << beta_prior_base << " beta_prior_scaler=" << beta_prior_scaler << " nodes=" << nodes << " remaining playouts=" << remaining_playouts; 
      // Help search to focus on this child:
      hints->UpdateIndexOfBestEdge(index_of_highest_q);
      return false;
    } else {
      LOGFILE << "ratio evaluated/budgeted=" << nodes/(nodes + remaining_playouts) << " Accepted smart pruning since child with largest n: " <<
    	index_of_largest_n << ", which has " << my_largest_n << " visits also has highest Expected Q=" << expected_q[index_of_largest_n] << " (raw Q=" << stats.q[index_of_largest_n] << ", beta_prior=" << beta_prior << ") beta_prior=" << beta_prior << " beta_prior_base=" << beta_prior_base << " beta_prior_scaler=" << beta_prior_scaler << " nodes=" << nodes << " remaining playouts=" << remaining_playouts; 
    }

    LOGFILE << remaining_playouts << " playouts remaining. Best move has "
            << largest_n << " visits, second best -- " << second_largest_n
            << ". Difference is " << (largest_n - second_largest_n)
            << ", so stopping the search after "
            << stats.batches_since_movestart << " batches.";

    return true;
  }

  if(index_of_largest_n != index_of_highest_q){
    if( proportion_left < 1 - stats.override_PUCT_node_budget_threshold ){
      // Help search to focus on this child:
      hints->UpdateIndexOfBestEdge(index_of_highest_q);
      LOGFILE << "ratio evaluated/budgeted=" << nodes/(nodes + remaining_playouts) << " Interfering with PUCT since remaining nodes is less than " << 1 - stats.override_PUCT_node_budget_threshold << " of budget and best root-edge hasn't the most visits: promising node has " << stats.edge_n[index_of_highest_q] << " nodes and most visited node has " << stats.edge_n[index_of_largest_n] << " visits." << " beta_prior=" << beta_prior << " beta_prior_base=" << beta_prior_base << " beta_prior_scaler=" << beta_prior_scaler << " nodes=" << nodes << " remaining playouts=" << remaining_playouts; 
    }
  }

  return false;
}

}  // namespace lczero
