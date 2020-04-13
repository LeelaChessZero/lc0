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

class AlphazeroStopper : public TimeLimitStopper {
 public:
  AlphazeroStopper(int64_t deadline_ms, int64_t* time_piggy_bank)
      : TimeLimitStopper(deadline_ms), time_piggy_bank_(time_piggy_bank) {}
  virtual void OnSearchDone(const IterationStats& stats) override {
    *time_piggy_bank_ += GetTimeLimitMs() - stats.time_since_movestart;
  }

 private:
  int64_t* const time_piggy_bank_;
};

class AlphazeroTimeManager : public TimeManager {
 public:
  AlphazeroTimeManager(int64_t move_overhead, const OptionsDict& params)
      : move_overhead_(move_overhead),
        alphazerotimevalue_(params.GetOrDefault<float>("alphazero-time-value", 20.0f)),
        spend_saved_time_(params.GetOrDefault<float>("immediate-use", 1.0f)) {}
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const Position& position) override;

 private:
  const int64_t move_overhead_;
  const float alphazerotimevalue_;
  const float spend_saved_time_;
  // No need to be atomic as only one thread will update it.
  int64_t time_spared_ms_ = 0;
};

std::unique_ptr<SearchStopper> AlphazeroTimeManager::GetStopper(
    const GoParams& params, const Position& position) {
  const bool is_black = position.IsBlackToMove();
  std::optional<int64_t> time = (is_black ? params.btime : params.wtime);
  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  std::optional<int64_t> inc = is_black ? params.binc : params.winc;
  const int increment = inc ? std::max(int64_t(0), *inc) : 0;

  // How to scale moves time.
  float this_move_time = 1.0f;
  int time_to_squander = 0;

  auto total_moves_time = *time - move_overhead_;
  this_move_time =
      increment +
      (total_moves_time / alphazerotimevalue_);

  LOGFILE << "Budgeted time for the move: " << this_move_time << "ms(+"
          << time_to_squander << "ms to squander). Remaining time " << *time
          << "ms(-" << move_overhead_ << "ms overhead)";
  // Use `time_to_squander` time immediately.
  this_move_time += time_to_squander;

  // Make sure we don't exceed current time limit with what we calculated.
  auto deadline =
      std::min(static_cast<int64_t>(this_move_time), *time - move_overhead_);
  return std::make_unique<AlphazeroStopper>(deadline, &time_spared_ms_);
}

}  // namespace

std::unique_ptr<TimeManager> MakeAlphazeroTimeManager(int64_t move_overhead,
                                                   const OptionsDict& params) {
  return std::make_unique<AlphazeroTimeManager>(move_overhead, params);
}
}  // namespace lczero