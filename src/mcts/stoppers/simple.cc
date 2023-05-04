/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2022 The LCZero Authors

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

namespace lczero {

namespace {

class SimpleTimeManager : public TimeManager {
 public:
  SimpleTimeManager(int64_t move_overhead, const OptionsDict& params)
      : move_overhead_(move_overhead),
        basepct_(params.GetOrDefault<float>("base-pct", 1.83f)),
        plypct_(params.GetOrDefault<float>("ply-pct", 0.0454f)),
        timefactor_(params.GetOrDefault<float>("time-factor", 33.4f)),
        opening_bonus_(params.GetOrDefault<float>("opening-bonus", 82.5f)) {
    if (basepct_ <= 0.0f || basepct_ > 100.0f) {
      throw Exception("base-pct value to be in range [0.0, 100.0]");
    }
    if (plypct_ < 0.0f || plypct_ > 1.0f) {
      throw Exception("ply-pct value to be in range [0.0, 1.0]");
    }
    if (timefactor_ < 0.0f || timefactor_ > 100.0f) {
      throw Exception("time-factor value to be in range [0.0, 100.0]");
    }
    if (opening_bonus_ < 0.0f || opening_bonus_ > 1000.0f) {
      throw Exception("opening-bonus value to be in range [0.0, 1000.0]");
    }
  }
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const NodeTree& tree) override;

 private:
  const int64_t move_overhead_;
  const float basepct_;
  const float plypct_;
  const float timefactor_;
  const float opening_bonus_;
  float prev_move_time = 0.0f;
  float prev_total_moves_time = 0.0f;
  bool bonus_applied = false;
};

std::unique_ptr<SearchStopper> SimpleTimeManager::GetStopper(
    const GoParams& params, const NodeTree& tree) {
  const Position& position = tree.HeadPosition();
  const bool is_black = position.IsBlackToMove();
  const std::optional<int64_t>& time = (is_black ? params.btime : params.wtime);

  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  const std::optional<int64_t>& inc = is_black ? params.binc : params.winc;
  const int increment = inc ? std::max(int64_t(0), *inc) : 0;

  const float total_moves_time =
      static_cast<float>(*time) - static_cast<float>(move_overhead_);

  // increase percentage as ply count increases
  float pct = (basepct_ + position.GetGamePly() * plypct_) * 0.01f;

  // increase percentage as ratio of increment time to total time gets smaller
  pct += pct * (static_cast<float>(increment) /
                static_cast<float>(total_moves_time) * timefactor_);

  float this_move_time = total_moves_time * pct;

  // immediately spend time saved from smart pruning during previous move
  if (prev_move_time > 0.0f) {
    const float time_saved =
        prev_move_time - (prev_total_moves_time -
                          (total_moves_time - static_cast<float>(increment)));

    this_move_time += time_saved;
  }

  // apply any opening bonus and note the next move will also benefit
  // from an increased time_saved as a result
  if (!bonus_applied) {
    this_move_time += this_move_time * opening_bonus_ * 0.01f;
    bonus_applied = true;
  }

  this_move_time = std::min(this_move_time, total_moves_time);

  prev_move_time = this_move_time;
  prev_total_moves_time = total_moves_time;

  LOGFILE << "Budgeted time for the move: " << this_move_time << "ms"
          << "Remaining time " << *time << "ms(-" << move_overhead_
          << "ms overhead)";

  return std::make_unique<TimeLimitStopper>(this_move_time);
}

}  // namespace

std::unique_ptr<TimeManager> MakeSimpleTimeManager(
    int64_t move_overhead, const OptionsDict& params) {
  return std::make_unique<SimpleTimeManager>(move_overhead, params);
}
}  // namespace lczero
