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

#include "search/classic/stoppers/legacy.h"

#include <cmath>

#include "search/classic/stoppers/stoppers.h"

namespace lczero {
namespace classic {

float ComputeEstimatedMovesToGo(int ply, float midpoint, float steepness) {
  // An analysis of chess games shows that the distribution of game lengths
  // looks like a log-logistic distribution. The mean residual time function
  // calculates how many more moves are expected in the game given that we are
  // at the current ply. Given that this function can be expensive to compute,
  // we calculate the median residual time function instead. This is derived and
  // shown to be similar to the mean residual time in "Some Useful Properties of
  // Log-Logistic Random Variables for Health Care Simulations" (Clark &
  // El-Taha, 2015).
  // midpoint: The median length of games.
  // steepness: How quickly the function drops off from its maximum value,
  // around the midpoint.
  const float move = ply / 2.0f;
  return midpoint * std::pow(1 + 2 * std::pow(move / midpoint, steepness),
                             1 / steepness) -
         move;
}

namespace {

class LegacyStopper : public TimeLimitStopper {
 public:
  LegacyStopper(int64_t deadline_ms, int64_t* time_piggy_bank)
      : TimeLimitStopper(deadline_ms), time_piggy_bank_(time_piggy_bank) {}
  virtual void OnSearchDone(const IterationStats& stats) override {
    *time_piggy_bank_ += GetTimeLimitMs() - stats.time_since_movestart;
  }

 private:
  int64_t* const time_piggy_bank_;
};

class LegacyTimeManager : public TimeManager {
 public:
  LegacyTimeManager(int64_t move_overhead, const OptionsDict& params)
      : move_overhead_(move_overhead),
        slowmover_(params.GetOrDefault<float>("slowmover", 1.0f)),
        time_curve_midpoint_(
            params.GetOrDefault<float>("midpoint-move", 51.5f)),
        time_curve_steepness_(params.GetOrDefault<float>("steepness", 7.0f)),
        spend_saved_time_(params.GetOrDefault<float>("immediate-use", 1.0f)),
        first_move_bonus_(params.GetOrDefault<float>("first-move-bonus", 1.8f)),
        book_ply_bonus_(params.GetOrDefault<float>("book-ply-bonus", 0.25f)) {}
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const Position& position,
                                            size_t /*total_memory*/,
                                            size_t /*avg_node_size*/,
                                            uint32_t /*nodes*/) override;

 private:
  const int64_t move_overhead_;
  const float slowmover_;
  const float time_curve_midpoint_;
  const float time_curve_steepness_;
  const float spend_saved_time_;
  // When starting a game from a book, add bonus time per ply of the book.
  const float first_move_bonus_;
  const float book_ply_bonus_;
  bool first_move_of_game_ = true;
  // No need to be atomic as only one thread will update it.
  int64_t time_spared_ms_ = 0;
};

std::unique_ptr<SearchStopper> LegacyTimeManager::GetStopper(
    const GoParams& params, const Position& position, size_t /*total_memory*/,
    size_t /*avg_node_size*/, uint32_t /*nodes*/) {
  const bool is_black = position.IsBlackToMove();
  const std::optional<int64_t>& time = (is_black ? params.btime : params.wtime);
  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  const std::optional<int64_t>& inc = is_black ? params.binc : params.winc;
  const int increment = inc ? std::max(int64_t(0), *inc) : 0;

  float movestogo = ComputeEstimatedMovesToGo(
      position.GetGamePly(), time_curve_midpoint_, time_curve_steepness_);

  // If the number of moves remaining until the time control are less than
  // the estimated number of moves left in the game, then use the number of
  // moves until the time control instead.
  if (params.movestogo &&
      *params.movestogo > 0 &&  // Ignore non-standard uci command.
      *params.movestogo < movestogo) {
    movestogo = *params.movestogo;
  }

  // Total time, including increments, until time control.
  auto total_moves_time =
      std::max(0.0f, *time + increment * (movestogo - 1) - move_overhead_);

  // If there is time spared from previous searches, the `time_to_squander` part
  // of it will be used immediately, remove that from planning.
  int time_to_squander = 0;
  if (time_spared_ms_ > 0) {
    total_moves_time = std::max(0.0f, total_moves_time - time_spared_ms_);
    time_to_squander = time_spared_ms_ * spend_saved_time_;
    time_spared_ms_ -= time_to_squander;
  }

  // Evenly split total time between all moves.
  float this_move_time = total_moves_time / movestogo;

  // Add bonus time per ply of the opening book to compensate starting from an
  // uncommon position without a tree to reuse.
  // Limit the bonus to max. 12 plies, which also prevents spending too much
  // time on the first move in resumed games.
  if (first_move_of_game_) {
    this_move_time *= (1.0f + first_move_bonus_ +
                       book_ply_bonus_ * std::min(12, position.GetGamePly()));
    first_move_of_game_ = false;
  }

  // Only extend thinking time with slowmover if smart pruning can potentially
  // reduce it.
  constexpr int kSmartPruningToleranceMs = 200;
  if (slowmover_ < 1.0 ||
      this_move_time * slowmover_ > kSmartPruningToleranceMs) {
    // If time is planned to be overused because of slowmover, remove excess
    // of that time from spared time.
    time_spared_ms_ -= this_move_time * (slowmover_ - 1);
    this_move_time *= slowmover_;
  }

  LOGFILE << "Budgeted time for the move: " << this_move_time << "ms(+"
          << time_to_squander << "ms to squander). Remaining time " << *time
          << "ms(-" << move_overhead_ << "ms overhead)";
  // Use `time_to_squander` time immediately.
  this_move_time += time_to_squander;

  // Make sure we don't exceed current time limit with what we calculated.
  auto deadline =
      std::min(static_cast<int64_t>(this_move_time), *time - move_overhead_);
  return std::make_unique<LegacyStopper>(deadline, &time_spared_ms_);
}

}  // namespace

std::unique_ptr<TimeManager> MakeLegacyTimeManager(int64_t move_overhead,
                                                   const OptionsDict& params) {
  return std::make_unique<LegacyTimeManager>(move_overhead, params);
}
}  // namespace classic
}  // namespace lczero
