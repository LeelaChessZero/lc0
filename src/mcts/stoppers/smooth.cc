/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include "mcts/stoppers/stoppers.h"
#include "utils/mutex.h"

namespace lczero {
namespace {

class Params {
 public:
  Params(const OptionsDict& /* params */, int64_t /* move_overhead */) {}

  // Which fraction of the tree is reuse after a full move. Initial guess.
  float initial_tree_reuse() const { return 0.5f; }
  // Do not allow tree reuse expectation to go above this value.
  float max_tree_reuse() const { return 0.7f; }
  // Number of moves needed to update tree reuse estimation halfway.
  float tree_reuse_halfupdate_moves() const { return 4.0f; }
  // Initial NPS guess.
  float initial_nps() const { return 20000.0f; }
  // Number of seconds to update nps estimation halfway.
  float nps_halfupdate_seconds() const { return 5.0f; }
  // Fraction of the budgeted time the engine uses, initial estimation.
  float initial_smartpruning_timeuse() const { return 0.7f; }
  // Do not allow timeuse estimation to fall below this.
  float min_smartpruning_timeuse() const { return 0.3f; }
  // Number of moves to update timeuse estimation halfway.
  float smartpruning_timeuse_halfupdate_moves() const { return 10.0f; }
  // Fraction of a total available move time that is allowed to use for a single
  // move.
  float max_single_move_time_fraction() const { return 0.5f; }

  int64_t move_overhead_ms() const;

 private:
};

// Returns the updated value of @from, towards @to by the number of halves equal
// to number of @steps in @value.
// E.g. if value=1*step, returns (from+to)/2,
// if value=2*step, return (1*from + 3*to)/4,
// if value=3*step, return (1*from + 7*to)/7,
// if value=0, returns from.
float ExponentialDecay(float from, float to, float step, float value) {
  return to - (to - from) * std::pow(0.5f, value / step);
}

class SmoothTimeManager;

class SmoothStopper : public TimeLimitStopper {
 public:
  SmoothStopper(int64_t deadline_ms, SmoothTimeManager* manager);

 private:
  SmoothTimeManager* const manager_;
};

class SmoothTimeManager : public TimeManager {
 public:
  SmoothTimeManager(int64_t move_overhead, const OptionsDict& params)
      : params_(params, move_overhead) {}

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

    const auto current_nodes = tree.GetCurrentHead()->GetN();
    if (last_move_final_nodes_ && last_time_ && last_expected_movetime_ms_) {
      UpdateTreeReuseFactor(current_nodes);
    }

    last_time_ = 0.0f;

    // Get remaining moves estimation.
    float remaining_moves = GetRemainingMoves();

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

    // Total time, including increments, until time control.
    auto total_remaining_ms =
        std::max(0.0f, *time + increment * (remaining_moves - 1) -
                           params_.move_overhead_ms());

    // Total remaining nodes that we'll have chance to compute in a game.
    float remaining_game_nodes = total_remaining_ms * nps_ / 1000.0f;
    // Total (fresh) nodes, in average, to processed per move.
    float avg_nodes_per_move = remaining_game_nodes / remaining_moves;
    // As some part of a tree is usually reused, we can aim to a larger target.
    float nodes_per_move_including_reuse =
        avg_nodes_per_move / (1.0f - tree_reuse_);
    // Subtract what we already have, and get what we need to compute.
    float move_estimate_nodes = nodes_per_move_including_reuse - current_nodes;
    // This is what time we think will be really spent thinking.
    last_expected_movetime_ms_ = move_estimate_nodes / nps_ * 1000.0f;
    // This is what is the actual budget as we hope that the search will be
    // shorter due to smart pruning.
    move_budgeted_time_ms_ = last_expected_movetime_ms_ / timeuse_;

    if (move_budgeted_time_ms_ >
        *time * params_.max_single_move_time_fraction()) {
      move_budgeted_time_ms_ = *time * params_.max_single_move_time_fraction();
    }

    return std::make_unique<SmoothStopper>(move_budgeted_time_ms_, this);
  }

  void UpdateTreeReuseFactor(int64_t new_move_nodes) REQUIRES(mutex_) {
    tree_reuse_ = ExponentialDecay(
        tree_reuse_,
        static_cast<float>(new_move_nodes) / last_move_final_nodes_,
        last_time_ / last_expected_movetime_ms_,
        params_.tree_reuse_halfupdate_moves());
    if (tree_reuse_ > params_.max_tree_reuse()) {
      tree_reuse_ = params_.max_tree_reuse();
    }
  }

  float GetRemainingMoves() const { return -4; }  // DO NOT SUBMIT

  const Params params_;

  Mutex mutex_;
  // Fraction of a tree which usually survives a full move (and is reused).
  float tree_reuse_ GUARDED_BY(mutex_) = params_.initial_tree_reuse();
  // Current NPS estimation.
  float nps_ GUARDED_BY(mutex_) = params_.initial_nps();
  // Fraction of a budgeted time usually used.
  float timeuse_ GUARDED_BY(mutex_) = params_.initial_smartpruning_timeuse();

  // Total amount of time budgeted for the current move. Used to update timeuse_
  // when the move ends.
  float move_budgeted_time_ms_ GUARDED_BY(mutex_) = 0.0f;
  // Total amount of nodes in the end of the previous search. Used to compute
  // tree reuse factor when a new search starts.
  int64_t last_move_final_nodes_ GUARDED_BY(mutex_) = 0;
  // Time of the last report, since the beginning of the move.
  int64_t last_time_ GUARDED_BY(mutex_) = 0;

  // According to the recent calculations, how much time should be spent in
  // average per move.
  float last_expected_movetime_ms_ GUARDED_BY(mutex_) = 0.0f;
};

/*class SmoothStopper : public TimeLimitStopper {
  SmoothStopper(SmoothTimeManager* manager);

 private:
  SmoothTimeManager* const manager_;
};
*/

SmoothStopper::SmoothStopper(int64_t deadline_ms, SmoothTimeManager* manager)
    : TimeLimitStopper(deadline_ms), manager_(manager) {}

}  // namespace

std::unique_ptr<TimeManager> MakeSmoothTimeManager(int64_t move_overhead,
                                                   const OptionsDict& params) {
  return std::make_unique<SmoothTimeManager>(move_overhead, params);
}

}  // namespace lczero