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

#include "mcts/stoppers/stoppers.h"

namespace lczero {

namespace {

class AlphazeroTimeManager : public TimeManager {
 public:
  AlphazeroTimeManager(int64_t move_overhead, const OptionsDict& params)
      : move_overhead_(move_overhead),
        minpct_(params.GetOrDefault<float>("min-pct", 2.52f)),
        timemult_(params.GetOrDefault<float>("time-mult", 2.17f)),
        plymult_(params.GetOrDefault<float>("ply-mult", 2.72f)) {
    if (minpct_ <= 0.0f || minpct_ > 10.0f)
      throw Exception("min-pct value to be in range [0.0, 10.0]");
    if (timemult_ <= 0.0f || timemult_ > 10.0f)
      throw Exception("time-mult value to be in range [0.0, 10.0]");
    if (plymult_ <= 0.0f || plymult_ > 10.0f)
      throw Exception("ply-mult value to be in range [0.0, 10.0]");
  }
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const NodeTree& tree) override;

 private:
  const int64_t move_overhead_;
  const float minpct_;
  const float timemult_;
  const float plymult_;
};

std::unique_ptr<SearchStopper> AlphazeroTimeManager::GetStopper(
    const GoParams& params, const NodeTree& tree) {
  const Position& position = tree.HeadPosition();
  const bool is_black = position.IsBlackToMove();
  const std::optional<int64_t>& time = (is_black ? params.btime : params.wtime);
  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  const std::optional<int64_t>& inc = is_black ? params.binc : params.winc;
  const int increment = inc ? std::max(int64_t(0), *inc) : 0;

  auto total_moves_time = *time - move_overhead_;

  const float timeratio_ = (float)increment / (float)total_moves_time;

  const float pct = minpct_ * 0.01f + timeratio_ * timemult_ +
                    (float)(position.GetGamePly() + 1) * 0.001f * plymult_;

  float this_move_time = total_moves_time * std::min(pct, 0.99f);

  LOGFILE << "Budgeted time for the move: " << this_move_time << "ms"
          << "Remaining time " << *time << "ms(-" << move_overhead_
          << "ms overhead)";

  return std::make_unique<TimeLimitStopper>(this_move_time);
}

}  // namespace

std::unique_ptr<TimeManager> MakeAlphazeroTimeManager(
    int64_t move_overhead, const OptionsDict& params) {
  return std::make_unique<AlphazeroTimeManager>(move_overhead, params);
}
}  // namespace lczero
