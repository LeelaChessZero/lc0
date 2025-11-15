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

#include "search/classic/stoppers/stoppers.h"

#include <cmath>

namespace lczero {
namespace classic {

namespace {

class AlphazeroAutoscaledTimeManager : public TimeManager {
 public:
  AlphazeroAutoscaledTimeManager(int64_t move_overhead,
                                 const OptionsDict& params)
      : move_overhead_(move_overhead),
        opening_increments_(
            params.GetOrDefault<bool>("opening-increments", false)) {
    if (opening_increments_ != true && opening_increments_ != false) {
      throw Exception("opening-increments can only be set to true or false.");
    }
  }
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const Position& position,
                                            size_t /*total_memory*/,
                                            size_t /*avg_node_size*/,
                                            uint32_t /*nodes*/) override;

 private:
  const int64_t move_overhead_;
  const bool opening_increments_;
  bool time_control_ratio_set_ = false;
  float inc_time_ratio_;
};

std::unique_ptr<SearchStopper> AlphazeroAutoscaledTimeManager::GetStopper(
    const GoParams& params, const Position& position, size_t /*total_memory*/,
    size_t /*avg_node_size*/, uint32_t /*nodes*/) {
  const bool is_black = position.IsBlackToMove();
  const auto& board = position.GetBoard();
  const std::optional<int64_t> time = (is_black ? params.btime : params.wtime);
  const std::optional<int64_t> inc = (is_black ? params.binc : params.winc);
  const int increment = std::max<int64_t>(0LL, inc.value_or(0));
  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  if (!time_control_ratio_set_) {
    inc_time_ratio_ = opening_increments_ ? increment * 1.0f / (*time -
        (std::floor(position.GetGamePly() / 2.0f) + 1.0f) * increment) :
        increment * 1.0f / *time;
    time_control_ratio_set_ = true;
  }

  auto total_moves_time = *time - move_overhead_;

  int pieces_on_board = (board.ours() | board.theirs()).count();

  float time_factor = 28.0f * inc_time_ratio_ + 0.089f;
  float pieces_factor = (0.81f * inc_time_ratio_ + 0.00097f) * pieces_on_board;

  LOGFILE << "Increment/basetime ratio: " << inc_time_ratio_
          << ", Time factor: " << time_factor
          << ", Pieces factor: " << pieces_factor << ".";

  float this_move_time = total_moves_time * std::min(time_factor -
                                                     pieces_factor, 1.0f);

  // Try to detect if increment time is added only after a move has been made
  // and recalculate this_move_time (can only happen in the first move of the
  // game or in the first move of a new game phase with increment).
  if (*time < increment) {
    this_move_time = std::min(
        (total_moves_time + increment * 1.0f) *
        std::min(time_factor - pieces_factor, 1.0f), total_moves_time * 1.0f);
  }

  LOGFILE << "Budgeted time for the move: " << this_move_time << "ms."
          << " Remaining time " << *time << "ms (-" << move_overhead_
          << "ms overhead).";

  return std::make_unique<TimeLimitStopper>(this_move_time);
}

}  // namespace

std::unique_ptr<TimeManager> MakeAlphazeroAutoscaledTimeManager(
    int64_t move_overhead, const OptionsDict& params) {
  return std::make_unique<AlphazeroAutoscaledTimeManager>(move_overhead,
                                                          params);
}
}  // namespace classic
}  // namespace lczero
