/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020-2021 The LCZero Authors

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

#include "mcts/stoppers/smooth.h"

#include <functional>
#include <iomanip>
#include <optional>

#include "mcts/stoppers/legacy.h"
#include "mcts/stoppers/stoppers.h"
#include "utils/mutex.h"

namespace lczero {
namespace {

class Params {
 public:
  Params(const OptionsDict& /* params */, int64_t move_overhead);

  using MovesLeftEstimator = std::function<float(const NodeTree&)>;

  // Which fraction of the tree is reuse after a full move. Initial guess.
  float initial_tree_reuse() const { return initial_tree_reuse_; }
  // Do not allow tree reuse expectation to go above this value.
  float max_tree_reuse() const { return max_tree_reuse_; }
  // Number of moves needed to update tree reuse estimation halfway.
  float tree_reuse_halfupdate_moves() const {
    return tree_reuse_halfupdate_moves_;
  }
  // Number of seconds to update nps estimation.
  float nps_update_seconds() const { return nps_update_seconds_; }
  // Fraction of the allocated time the engine uses, initial estimation.
  float initial_smartpruning_timeuse() const {
    return initial_smartpruning_timeuse_;
  }
  // Do not allow timeuse estimation to fall below this.
  float min_smartpruning_timeuse() const { return min_smartpruning_timeuse_; }
  // Number of moves to update timeuse estimation halfway.
  float smartpruning_timeuse_halfupdate_moves() const {
    return smartpruning_timeuse_halfupdate_moves_;
  }
  // Fraction of a total available move time that is allowed to use for a single
  // move.
  float max_single_move_time_fraction() const {
    return max_single_move_time_fraction_;
  }
  // How much of the initial time of the game (without increments) is sent to a
  // piggybank.
  float initial_piggybank_fraction() const {
    return initial_piggybank_fraction_;
  }
  // How much of the move time (for every move) it taxed into a piggybank.
  float per_move_piggybank_fraction() const {
    return per_move_piggybank_fraction_;
  }
  // Maximum time in piggybank to use for a single move.
  float max_piggybank_use() const { return max_piggybank_use_; }

  // Max number of avg move times in piggybank.
  float max_piggybank_moves() const { return max_piggybank_moves_; }

  int64_t trend_nps_update_period_ms() const {
    return trend_nps_update_period_ms_;
  }

  // Expected ration of the best move nps in future, to the current nps.
  float bestmove_optimism() const { return bestmove_optimism_; }

  // Expected ration of the non-best move nps in future, to the current nps.
  float overtaker_optimism() const { return overtaker_optimism_; }

  // Force a use of piggybank during the first few milliseconds of the move.
  float force_piggybank_ms() const { return force_piggybank_ms_; }

  // Move overhead.
  int64_t move_overhead_ms() const { return move_overhead_ms_; }
  // Returns a function function that estimates remaining moves.
  MovesLeftEstimator moves_left_estimator() const {
    return moves_left_estimator_;
  }

 private:
  const int64_t move_overhead_ms_;
  const float initial_tree_reuse_;
  const float max_tree_reuse_;
  const float tree_reuse_halfupdate_moves_;
  const float nps_update_seconds_;
  const float initial_smartpruning_timeuse_;
  const float min_smartpruning_timeuse_;
  const float smartpruning_timeuse_halfupdate_moves_;
  const float max_single_move_time_fraction_;
  const float initial_piggybank_fraction_;
  const float per_move_piggybank_fraction_;
  const float max_piggybank_use_;
  const float max_piggybank_moves_;
  const float trend_nps_update_period_ms_;
  const float bestmove_optimism_;
  const float overtaker_optimism_;
  const float force_piggybank_ms_;
  const MovesLeftEstimator moves_left_estimator_;
};

Params::MovesLeftEstimator CreateMovesLeftEstimator(const OptionsDict& params) {
  // The only estimator we have now is MLE-legacy (Moves left estimator).
  const OptionsDict& mle_dict = params.HasSubdict("mle-legacy")
                                    ? params.GetSubdict("mle-legacy")
                                    : params;
  return [midpoint = mle_dict.GetOrDefault<float>("midpoint", 45.2f),
          steepness = mle_dict.GetOrDefault<float>("steepness", 5.93f)](
             const NodeTree& tree) {
    const auto ply = tree.HeadPosition().GetGamePly();
    return ComputeEstimatedMovesToGo(ply, midpoint, steepness);
  };
}

Params::Params(const OptionsDict& params, int64_t move_overhead)
    : move_overhead_ms_(move_overhead),
      initial_tree_reuse_(params.GetOrDefault<float>("init-tree-reuse", 0.52f)),
      max_tree_reuse_(params.GetOrDefault<float>("max-tree-reuse", 0.73f)),
      tree_reuse_halfupdate_moves_(
          params.GetOrDefault<float>("tree-reuse-update-rate", 3.39f)),
      nps_update_seconds_(
          params.GetOrDefault<float>("nps-update-period", 20.0f)),
      initial_smartpruning_timeuse_(
          params.GetOrDefault<float>("init-timeuse", 0.7f)),
      min_smartpruning_timeuse_(
          params.GetOrDefault<float>("min-timeuse", 0.34f)),
      smartpruning_timeuse_halfupdate_moves_(
          params.GetOrDefault<float>("timeuse-update-rate", 5.51f)),
      max_single_move_time_fraction_(
          params.GetOrDefault<float>("max-move-budget", 0.42f)),
      initial_piggybank_fraction_(
          params.GetOrDefault<float>("init-piggybank", 0.09f)),
      per_move_piggybank_fraction_(
          params.GetOrDefault<float>("per-move-piggybank", 0.12f)),
      max_piggybank_use_(
          params.GetOrDefault<float>("max-piggybank-use", 0.94f)),
      max_piggybank_moves_(
          params.GetOrDefault<float>("max-piggybank-moves", 36.5f)),
      trend_nps_update_period_ms_(
          params.GetOrDefault<int>("trend-nps-update-period-ms", 3000)),
      bestmove_optimism_(params.GetOrDefault<float>("bestmove-optimism", 0.2f)),
      overtaker_optimism_(
          params.GetOrDefault<float>("overtaker-optimism", 4.0f)),
      force_piggybank_ms_(params.GetOrDefault<int>("force-piggybank-ms", 1000)),
      moves_left_estimator_(CreateMovesLeftEstimator(params)) {}

// Returns the updated value of @from, towards @to by the number of halves
// equal to number of @steps in @value. E.g. if value=1*step, returns
// (from+to)/2, if value=2*step, return (1*from + 3*to)/4, if
// value=3*step, return (1*from + 7*to)/7, if value=0, returns from.
float ExponentialDecay(float from, float to, float step, float value) {
  return to - (to - from) * std::pow(0.5f, value / step);
}

float LinearInterpolate(float a, float b, float theta) {
  if (theta <= 0) return a;
  if (theta >= 1) return b;
  return a + (b - a) * theta;
}

// @cur_value -- current "average", returned from the previous call of this
//   function.
// @target_value -- current "correct but noisy" value, that we slowly go to.
// @time_since_reliable -- time since we had reliable nps in target_value. For
//    the case of nps, end of the previous move is considered reliable point).
// @time_since_last_update_ms -- how long ago was this function last called.
// @window_size_ms -- in what time should the output converge to @target_value.
float LinearDecay(float cur_value, float target_value,
                  float time_since_reliable_ms, float time_since_last_update_ms,
                  float window_size_ms) {
  // If target is higher than cur value, just return target without any
  // smoothing. That makes sense for nps, but for other averages we should look
  // a bit closer.
  if (cur_value <= target_value) return target_value;
  // If no time passed since last update, return previous value.
  if (time_since_last_update_ms <= 0) return cur_value;
  // If window is large enough, we consider
  if (time_since_reliable_ms >= window_size_ms) return target_value;
  if (time_since_last_update_ms > time_since_reliable_ms) {
    // Should never happen, but just for the case.
    time_since_last_update_ms = time_since_reliable_ms;
  }
  // time_since_reliable_ms during the previous call of this function.
  const float prev_time_since_reliable_ms =
      time_since_reliable_ms - time_since_last_update_ms;
  // The same, but as fraction of window size.
  const float prev_reliable_frac = prev_time_since_reliable_ms / window_size_ms;
  // What was the value when it was reliable.
  const float reliable_value = (prev_reliable_frac * target_value - cur_value) /
                               (prev_reliable_frac - 1);
  // Now we know window width, nps when it was reliable, and time since that,
  // and time to converge, interpolate.
  return LinearInterpolate(reliable_value, target_value,
                           time_since_reliable_ms / window_size_ms);
}

class SmoothTimeManager;

class VisitsTrendWatcher {
 public:
  VisitsTrendWatcher(float nps_update_period, float bestmove_optimism,
                     float overtaker_optimism)
      : nps_update_period_(nps_update_period),
        bestmove_optimism_(bestmove_optimism),
        overtaker_optimism_(overtaker_optimism) {}

  void Update(uint64_t timestamp, const std::vector<uint32_t>& visits);
  bool IsBestmoveBeingOvertaken(uint64_t by_which_time) const;

 private:
  const float nps_update_period_;
  const float bestmove_optimism_;
  const float overtaker_optimism_;

  mutable Mutex mutex_;
  uint64_t prev_timestamp_ GUARDED_BY(mutex_) = 0;
  std::vector<uint32_t> prev_visits_ GUARDED_BY(mutex_);
  uint64_t cur_timestamp_ GUARDED_BY(mutex_) = 0;
  std::vector<uint32_t> cur_visits_ GUARDED_BY(mutex_);
  uint64_t last_timestamp_ GUARDED_BY(mutex_) = 0;
  std::vector<uint32_t> last_visits_ GUARDED_BY(mutex_);
};

class SmoothStopper : public SearchStopper {
 public:
  SmoothStopper(int64_t deadline_ms, int64_t allowed_piggybank_use_ms,
                float nps_update_period, float bestmove_optimism,
                float overtaker_optimism, int64_t forces_piggybank_ms,
                SmoothTimeManager* manager);

 private:
  bool ShouldStop(const IterationStats& stats, StoppersHints* hints) override;
  void OnSearchDone(const IterationStats& stats) override;

  const int64_t deadline_ms_;
  const int64_t allowed_piggybank_use_ms_;
  const int64_t forced_piggybank_use_ms_;

  VisitsTrendWatcher visits_trend_watcher_;
  SmoothTimeManager* const manager_;
  std::atomic_flag used_piggybank_;
};

class SmoothTimeManager : public TimeManager {
 public:
  SmoothTimeManager(int64_t move_overhead, const OptionsDict& params)
      : params_(params, move_overhead) {}

  float UpdateNps(int64_t time_since_movestart_ms,
                  int64_t nodes_since_movestart) {
    Mutex::Lock lock(mutex_);
    if (time_since_movestart_ms <= 0) return nps_;
    if (nps_is_reliable_ && time_since_movestart_ms > last_time_) {
      const float nps =
          1000.0f * nodes_since_movestart / time_since_movestart_ms;
      nps_ = LinearDecay(nps_, nps, time_since_movestart_ms,
                         time_since_movestart_ms - last_time_,
                         params_.nps_update_seconds() * 1000.0f);
    } else {
      nps_ = 1000.0f * nodes_since_movestart / time_since_movestart_ms;
    }
    last_time_ = time_since_movestart_ms;
    return nps_;
  }

  void UpdateEndOfMoveStats(int64_t total_move_time, bool used_piggybank,
                            int64_t time_budget, int64_t total_nodes) {
    Mutex::Lock lock(mutex_);
    // Whatever is in nps_ after the first move, is truth now.
    nps_is_reliable_ = true;
    // How different was this move from an average move
    const float this_move_time_fraction =
        avg_ms_per_move_ <= 0.0f ? 0.0f : total_move_time / avg_ms_per_move_;
    // Update time_use estimation.
    const float this_move_time_use =
        move_allocated_time_ms_ <= 0.0f
            ? 1.0f
            : total_move_time / move_allocated_time_ms_;
    // Recompute expected move time for logging.
    const float expected_move_time = move_allocated_time_ms_ * timeuse_;

    int64_t piggybank_time_used = 0;
    if (used_piggybank) {
      piggybank_time_used = std::max(int64_t(), total_move_time - time_budget);
      piggybank_time_ -= piggybank_time_used;
    }
    // If piggybank was used, time use is 100%.
    timeuse_ =
        ExponentialDecay(timeuse_, used_piggybank ? 1.0f : this_move_time_use,
                         params_.smartpruning_timeuse_halfupdate_moves(),
                         this_move_time_fraction);
    if (timeuse_ < params_.min_smartpruning_timeuse()) {
      timeuse_ = params_.min_smartpruning_timeuse();
    }
    // Remember final number of nodes for tree reuse estimation.
    last_move_final_nodes_ = total_nodes;

    LOGFILE << std::fixed << "TMGR: Updating endmove stats. actual_move_time="
            << total_move_time
            << "ms, allocated_move_time=" << move_allocated_time_ms_
            << "ms (ratio=" << this_move_time_use
            << "), expected_move_time=" << expected_move_time
            << "ms. New time_use=" << timeuse_
            << ", update_rate=" << this_move_time_fraction
            << " (avg_move_time=" << avg_ms_per_move_ << "ms)."
            << " piggybank_used=" << piggybank_time_used << "ms";
  }

 private:
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const NodeTree& tree) override {
    const Position& position = tree.HeadPosition();
    const bool is_black = position.IsBlackToMove();
    const std::optional<int64_t>& time =
        (is_black ? params.btime : params.wtime);
    // If no time limit is given, don't stop on this condition.
    if (params.infinite || params.ponder || !time) return nullptr;

    Mutex::Lock lock(mutex_);

    if (is_first_move_) {
      piggybank_time_ =
          static_cast<int64_t>(*time * params_.initial_piggybank_fraction());
      is_first_move_ = false;
    }

    const auto current_nodes = tree.GetCurrentHead()->GetN();
    if (last_move_final_nodes_ && last_time_ && avg_ms_per_move_ >= 0.0f) {
      UpdateTreeReuseFactor(current_nodes);
    }

    last_time_ = 0;

    // Get remaining moves estimation.
    float remaining_moves = params_.moves_left_estimator()(tree);

    // If the number of moves remaining until the time control are less than
    // the estimated number of moves left in the game, then use the number of
    // moves until the time control instead.
    if (params.movestogo &&
        *params.movestogo > 0 &&  // Ignore non-standard uci command.
        *params.movestogo < remaining_moves) {
      remaining_moves = *params.movestogo;
    }

    const std::optional<int64_t>& inc = is_black ? params.binc : params.winc;
    const int increment = inc ? std::max(int64_t(0), *inc) : 0;

    // Maximum time allowed in piggybank.
    const float max_piggybank_time =
        params_.max_piggybank_moves() *
        std::max(0.0f, *time + increment * (remaining_moves - 1) -
                           params_.move_overhead_ms()) /
        remaining_moves;

    if (piggybank_time_ > max_piggybank_time) {
      piggybank_time_ = max_piggybank_time;
    }

    // Total time, including increments, until time control.
    const auto total_remaining_ms = std::max(
        0.0f, *time - piggybank_time_ + increment * (remaining_moves - 1) -
                  params_.move_overhead_ms());

    // Total remaining nodes that we'll have chance to compute in a game.
    const float remaining_game_nodes = total_remaining_ms * nps_ / 1000.0f;
    // Total (fresh) nodes, in average, to processed per move.
    const float avg_nodes_per_move = remaining_game_nodes / remaining_moves;
    // Average time that will be spent per move.
    avg_ms_per_move_ = total_remaining_ms / remaining_moves;
    // As some part of a tree is usually reused, we can aim to a larger target.
    const float nodes_per_move_including_reuse =
        avg_nodes_per_move / (1.0f - tree_reuse_);
    // Subtract what we already have, and get what we need to compute.
    const float move_estimate_nodes =
        std::max(0.0f, nodes_per_move_including_reuse - current_nodes);
    // This is what time we think will be really spent thinking.
    const float expected_movetime_ms_brutto =
        move_estimate_nodes / nps_ * 1000.0f;
    // Share of this move that will go to piggybank.
    const float time_to_piggybank_ms = std::min(
        max_piggybank_time - piggybank_time_,
        expected_movetime_ms_brutto * params_.per_move_piggybank_fraction());
    // Some share of this time will go to a piggybank, so a move gets less.
    const float expected_movetime_ms =
        expected_movetime_ms_brutto - time_to_piggybank_ms;
    // When we need to use piggybank, we can use it.
    int64_t allowed_piggybank_time_ms =
        piggybank_time_ * params_.max_piggybank_use();
    // This is what is the actual budget as we hope that the search will be
    // shorter due to smart pruning.
    move_allocated_time_ms_ = expected_movetime_ms / timeuse_;

    if (move_allocated_time_ms_ >
        *time * params_.max_single_move_time_fraction()) {
      move_allocated_time_ms_ = *time * params_.max_single_move_time_fraction();
    }
    if (move_allocated_time_ms_ > *time - params_.move_overhead_ms()) {
      move_allocated_time_ms_ = std::max(static_cast<int64_t>(0LL),
                                         *time - params_.move_overhead_ms());
    }
    if (allowed_piggybank_time_ms >
        *time - params_.move_overhead_ms() - move_allocated_time_ms_) {
      allowed_piggybank_time_ms =
          *time - params_.move_overhead_ms() - move_allocated_time_ms_;
    }
    piggybank_time_ += time_to_piggybank_ms;

    LOGFILE << std::fixed << std::setprecision(1)
            << "TMGR: MOVE TIME: allocated()=" << move_allocated_time_ms_
            << "ms, expected(brutto)=" << expected_movetime_ms_brutto
            << "ms, expected(netto)=" << expected_movetime_ms
            << std::setprecision(5) << "ms, timeuse=" << timeuse_;

    LOGFILE << std::fixed << std::setprecision(1)
            << "TMGR: MOVE NODES: avg_total_with_reuse(brutto)="
            << nodes_per_move_including_reuse
            << ", avg_total_without_reuse(brutto)=" << avg_nodes_per_move
            << ", reused=" << current_nodes
            << ", expected_new(brutto)=" << move_estimate_nodes
            << std::setprecision(5) << ", tree_reuse=" << tree_reuse_;

    LOGFILE << std::fixed << std::setprecision(1)
            << "TMGR: PIGGYBANK: total=" << piggybank_time_
            << "ms, increment=" << time_to_piggybank_ms
            << "ms, longthink=" << allowed_piggybank_time_ms
            << "ms, max=" << max_piggybank_time << "ms";

    LOGFILE << std::fixed << std::setprecision(1)
            << "TMGR: REMAINING GAME: nodes=" << remaining_game_nodes
            << ", moves=" << remaining_moves << ", time=" << total_remaining_ms
            << "ms, nps=" << nps_;

    return std::make_unique<SmoothStopper>(
        move_allocated_time_ms_, allowed_piggybank_time_ms,
        params_.trend_nps_update_period_ms(), params_.bestmove_optimism(),
        params_.overtaker_optimism(), params_.force_piggybank_ms(), this);
  }

  void UpdateTreeReuseFactor(int64_t new_move_nodes) REQUIRES(mutex_) {
    // How different was this move from an average move
    const float this_move_time_fraction =
        avg_ms_per_move_ <= 0.0f ? 0.0f : last_time_ / avg_ms_per_move_;

    const float this_move_tree_reuse =
        static_cast<float>(new_move_nodes) / last_move_final_nodes_;
    tree_reuse_ = ExponentialDecay(tree_reuse_, this_move_tree_reuse,
                                   params_.tree_reuse_halfupdate_moves(),
                                   this_move_time_fraction);
    if (tree_reuse_ > params_.max_tree_reuse()) {
      tree_reuse_ = params_.max_tree_reuse();
    }
    LOGFILE << std::fixed << "TMGR: Updating tree reuse. last_move_nodes="
            << last_move_final_nodes_ << ", this_move_nodes=" << new_move_nodes
            << " (tree_reuse=" << this_move_tree_reuse
            << "). avg_tree_reuse=" << tree_reuse_
            << ", update_rate=" << this_move_time_fraction
            << " (avg_move_time=" << avg_ms_per_move_
            << "ms, actual_move_time=" << last_time_ << "ms)";
  }

  const Params params_;

  Mutex mutex_;
  // Fraction of a tree which usually survives a full move (and is reused).
  float tree_reuse_ GUARDED_BY(mutex_) = params_.initial_tree_reuse();
  // Current NPS estimation.
  float nps_ GUARDED_BY(mutex_) = 20000.0f;
  // NPS is unreliable until the end of the first move.
  bool nps_is_reliable_ GUARDED_BY(mutex_) = false;
  // Fraction of a allocated time usually used.
  float timeuse_ GUARDED_BY(mutex_) = params_.initial_smartpruning_timeuse();

  // Average amount of time per move. Used to compute ratio for timeuse and
  // tree reuse updates.
  float avg_ms_per_move_ GUARDED_BY(mutex_) = 0.0f;
  // Total amount of time allocated for the current move. Used to update
  // timeuse_ when the move ends.
  float move_allocated_time_ms_ GUARDED_BY(mutex_) = 0.0f;
  // Total amount of nodes in the end of the previous search. Used to compute
  // tree reuse factor when a new search starts.
  int64_t last_move_final_nodes_ GUARDED_BY(mutex_) = 0;
  // Time of the last report, since the beginning of the move.
  int64_t last_time_ GUARDED_BY(mutex_) = 0;
  // Time in piggybank (to use when search requests it).
  int64_t piggybank_time_ GUARDED_BY(mutex_) = 0;
  // Tracks whether it's called for the first time, meaning there's a need to
  // do initialization.
  bool is_first_move_ GUARDED_BY(mutex_) = true;
};

void VisitsTrendWatcher::Update(uint64_t timestamp,
                                const std::vector<uint32_t>& visits) {
  Mutex::Lock lock(mutex_);
  if (timestamp <= last_timestamp_) return;
  if (prev_visits_.empty()) {
    prev_visits_ = visits;
    cur_visits_ = visits;
    prev_timestamp_ = timestamp;
    cur_timestamp_ = timestamp;
  }
  last_timestamp_ = timestamp;
  last_visits_ = visits;
  if (cur_timestamp_ + nps_update_period_ >= timestamp) {
    prev_timestamp_ = cur_timestamp_;
    prev_visits_ = std::move(cur_visits_);
    cur_visits_ = last_visits_;
    cur_timestamp_ = last_timestamp_;
  }
}

bool VisitsTrendWatcher::IsBestmoveBeingOvertaken(
    uint64_t by_which_time) const {
  Mutex::Lock lock(mutex_);
  // If we don't have any stats yet, we cannot stop the search.
  if (prev_timestamp_ >= last_timestamp_) return false;
  if (by_which_time <= last_timestamp_) return false;
  std::vector<float> npms;
  npms.reserve(last_visits_.size());
  for (size_t i = 0; i < last_visits_.size(); ++i) {
    npms.push_back(static_cast<float>(last_visits_[i] - prev_visits_[i]) /
                   (last_timestamp_ - prev_timestamp_));
  }
  const size_t bestmove_idx =
      std::max_element(last_visits_.begin(), last_visits_.end()) -
      last_visits_.begin();
  const auto planned_bestmove_visits =
      last_visits_[bestmove_idx] + bestmove_optimism_ * npms[bestmove_idx] *
                                       (by_which_time - last_timestamp_);
  for (size_t i = 0; i < last_visits_.size(); ++i) {
    if (i == bestmove_idx) continue;
    const auto planned_visits =
        last_visits_[i] +
        overtaker_optimism_ * npms[i] * (by_which_time - last_timestamp_);
    if (planned_visits > planned_bestmove_visits) return true;
  }
  return false;
}

SmoothStopper::SmoothStopper(int64_t deadline_ms,
                             int64_t allowed_piggybank_use_ms,
                             float nps_update_period, float bestmove_optimism,
                             float overtaker_optimism,
                             int64_t forced_piggybank_use_ms,
                             SmoothTimeManager* manager)
    : deadline_ms_(deadline_ms),
      allowed_piggybank_use_ms_(allowed_piggybank_use_ms),
      forced_piggybank_use_ms_(forced_piggybank_use_ms),
      visits_trend_watcher_(nps_update_period, bestmove_optimism,
                            overtaker_optimism),
      manager_(manager) {
  used_piggybank_.clear();
}

bool SmoothStopper::ShouldStop(const IterationStats& stats,
                               StoppersHints* hints) {
  const auto nps = manager_->UpdateNps(stats.time_since_first_batch,
                                       stats.nodes_since_movestart);
  if (stats.time_usage_hint_ == IterationStats::TimeUsageHint::kImmediateMove) {
    LOGFILE << "Search requested immediate stop, alrite.";
    return true;
  }

  visits_trend_watcher_.Update(stats.time_since_movestart, stats.edge_n);
  const auto deadline_with_piggybank = deadline_ms_ + allowed_piggybank_use_ms_;
  const bool force_use_piggybank =
      stats.time_since_first_batch <= forced_piggybank_use_ms_;
  const bool use_piggybank =
      (stats.time_usage_hint_ == IterationStats::TimeUsageHint::kNeedMoreTime ||
       force_use_piggybank ||
       visits_trend_watcher_.IsBestmoveBeingOvertaken(deadline_with_piggybank));
  const int64_t time_limit =
      use_piggybank ? deadline_with_piggybank : deadline_ms_;
  hints->UpdateEstimatedNps(nps);
  hints->UpdateEstimatedRemainingTimeMs(time_limit -
                                        stats.time_since_movestart);
  if (use_piggybank && stats.time_since_movestart >= deadline_ms_) {
    // It's not entirely correct as due to extended remaining time smart pruning
    // will trigger later and we spend more time than if use_piggyback was
    // false, even before reaching the deadline.
    if (!used_piggybank_.test_and_set()) {
      LOGFILE << "Entering piggybank, reason: "
              << (stats.time_usage_hint_ ==
                          IterationStats::TimeUsageHint::kNeedMoreTime
                      ? "requested by search."
                  : force_use_piggybank
                      ? "forced used in the beginning of the move."
                      : "bestmove can be overtaken.");
    }
  }
  if (stats.time_since_movestart >= time_limit) {
    LOGFILE << "Stopping search: Ran out of time. elapsed=" << std::fixed
            << stats.time_since_movestart << " limit=" << time_limit
            << " piggy=" << use_piggybank;
    return true;
  }
  return false;
}

void SmoothStopper::OnSearchDone(const IterationStats& stats) {
  manager_->UpdateEndOfMoveStats(stats.time_since_movestart,
                                 used_piggybank_.test_and_set(), deadline_ms_,
                                 stats.total_nodes);
}

}  // namespace

std::unique_ptr<TimeManager> MakeSmoothTimeManager(int64_t move_overhead,
                                                   const OptionsDict& params) {
  return std::make_unique<SmoothTimeManager>(move_overhead, params);
}

}  // namespace lczero
