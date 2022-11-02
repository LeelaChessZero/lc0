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
        alphazerotimepct_(
            params.GetOrDefault<float>("alphazero-time-pct", 12.0f)) {
    if (alphazerotimepct_ < 0.0f || alphazerotimepct_ > 100.0f)
      throw Exception("alphazero-time-pct value to be in range [0.0, 100.0]");
  }
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const NodeTree& tree) override;

 private:
  const int64_t move_overhead_;
  const float alphazerotimepct_;
};

std::unique_ptr<SearchStopper> AlphazeroTimeManager::GetStopper(
    const GoParams& params, const NodeTree& tree) {
  const Position& position = tree.HeadPosition();
  const bool is_black = position.IsBlackToMove();
  const std::optional<int64_t>& time = (is_black ? params.btime : params.wtime);
  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  auto total_moves_time = *time - move_overhead_;

  float this_move_time = total_moves_time * (alphazerotimepct_ / 100.0f);

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
