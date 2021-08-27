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
  float new_alphazerotimepct_;
  float alphazeroincrementpct_;
  const float expected_moves_ = 75.0f;
  bool alphazero_modified_= false;
  float alphazero_decay_;
  float initial_time_;
  int64_t moves_played_= 0;
};

std::unique_ptr<SearchStopper> AlphazeroTimeManager::GetStopper(
    const GoParams& params, const NodeTree& tree) {
  const Position& position = tree.HeadPosition();
  const bool is_black = position.IsBlackToMove();

  const std::optional<int64_t>& time = (is_black ? params.btime : params.wtime);
  const std::optional<int64_t>& increment = (is_black ? params.binc : params.winc);

  // Transforming the alphazero percentage to make it play as if it has tuner conditions. (It's only done once)
  if (!alphazero_modified_) {
    const float tuned_initial_time_ = 216.0f;
    const float tuned_increment_ = 0.3f;
    initial_time_= *time;
    const float initial_time_sec_ = initial_time_ / 1000.0f;
    const float expected_tuned_game_time_ = tuned_initial_time_ + (tuned_increment_ * expected_moves_);
    const float expected_game_time_ = initial_time_sec_ + ((*increment/1000.f) * expected_moves_);   
    new_alphazerotimepct_ = std::min<float>(100, (alphazerotimepct_ * (tuned_initial_time_ / initial_time_sec_) * (expected_game_time_ / expected_tuned_game_time_)));
    alphazero_decay_ = (1 / expected_moves_) * (new_alphazerotimepct_ - alphazerotimepct_);
    alphazero_modified_ = true;
  } else {
    // Decaying new Alphazero percentage back to the input value
    if (moves_played_ < expected_moves_) {
      new_alphazerotimepct_ -= alphazero_decay_;
    }
  }
  moves_played_++;


  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  auto total_moves_time = *time - move_overhead_;
  // Using part of the increment based on the difference between initial total time and current total time.
  alphazeroincrementpct_ = (1 - std::min<float>(1, std::max<float>(0,(total_moves_time - *increment))/initial_time_)) * 100.0f;

  float this_move_time = std::max<unsigned long>(0, total_moves_time - *increment) * (new_alphazerotimepct_ / 100.0f) + *increment * (alphazeroincrementpct_ / 100.0f);

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
