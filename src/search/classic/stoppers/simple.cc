/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2022-2023 The LCZero Authors

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

#include "search/classic/stoppers/stoppers.h"

namespace lczero {
namespace classic {

namespace {

class SimpleTimeManager : public TimeManager {
 public:
  SimpleTimeManager(int64_t move_overhead, const OptionsDict& params)
      : move_overhead_(move_overhead),
        base_pct_(params.GetOrDefault<float>("base-pct", 1.4f)),
        ply_pct_(params.GetOrDefault<float>("ply-pct", 0.049f)),
        time_factor_(params.GetOrDefault<float>("time-factor", 1.5f)),
        opening_bonus_pct_(
            params.GetOrDefault<float>("opening-bonus-pct", 0.0f)) {
    if (base_pct_ <= 0.0f || base_pct_ > 100.0f) {
      throw Exception("base-pct value to be in range [0.0, 100.0]");
    }
    if (ply_pct_ < 0.0f || ply_pct_ > 1.0f) {
      throw Exception("ply-pct value to be in range [0.0, 1.0]");
    }
    if (time_factor_ < 0.0f || time_factor_ > 100.0f) {
      throw Exception("time-factor value to be in range [0.0, 100.0]");
    }
    if (opening_bonus_pct_ < 0.0f || opening_bonus_pct_ > 1000.0f) {
      throw Exception("opening-bonus-pct value to be in range [0.0, 1000.0]");
    }
  }
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const Position& position,
                                            size_t /*total_memory*/,
                                            size_t /*avg_node_size*/,
                                            uint32_t /*nodes*/) override;

 private:
  const int64_t move_overhead_;
  const float base_pct_;
  const float ply_pct_;
  const float time_factor_;
  const float opening_bonus_pct_;
  float prev_time_budgeted_ = 0.0f;
  float prev_time_available_ = 0.0f;
  bool bonus_applied_ = false;
};

std::unique_ptr<SearchStopper> SimpleTimeManager::GetStopper(
    const GoParams& params, const Position& position, size_t /*total_memory*/,
    size_t /*avg_node_size*/, uint32_t /*nodes*/) {
  const bool is_black = position.IsBlackToMove();
  const std::optional<int64_t>& time = (is_black ? params.btime : params.wtime);

  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  const std::optional<int64_t>& inc = is_black ? params.binc : params.winc;
  const int increment = inc ? std::max(int64_t(0), *inc) : 0;

  const float time_available =
      static_cast<float>(*time) - static_cast<float>(move_overhead_);

  const float time_ratio =
      static_cast<float>(increment) / static_cast<float>(*time);

  // increase percentage as ply count increases
  float pct = (base_pct_ + position.GetGamePly() * ply_pct_) * 0.01f;

  // increase percentage as ratio of increment to total time reaches equality
  pct += time_ratio * time_factor_;

  float time_budgeted = time_available * pct;

  // apply any opening bonus and note the next move will also benefit
  // from an increased time_saved as a result
  if (!bonus_applied_) {
    time_budgeted += time_budgeted * opening_bonus_pct_ * 0.01f;
    bonus_applied_ = true;
  }

  // immediately spend time saved from smart pruning during previous move
  if (prev_time_budgeted_ > 0.0f) {
    const float time_saved = prev_time_budgeted_ -
                             (prev_time_available_ -
                              (time_available - static_cast<float>(increment)));

    time_budgeted += std::max(time_saved, 0.0f);
  }

  time_budgeted = std::min(time_budgeted, time_available);

  LOGFILE << "Budgeted time for the move: " << time_budgeted << "ms "
          << "Remaining time " << *time << "ms(-" << move_overhead_
          << "ms overhead)";

  prev_time_budgeted_ = time_budgeted;
  prev_time_available_ = time_available;

  return std::make_unique<TimeLimitStopper>(time_budgeted);
}

}  // namespace

std::unique_ptr<TimeManager> MakeSimpleTimeManager(int64_t move_overhead,
                                                   const OptionsDict& params) {
  return std::make_unique<SimpleTimeManager>(move_overhead, params);
}
}  // namespace classic
}  // namespace lczero
