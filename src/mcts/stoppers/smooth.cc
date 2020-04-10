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
  // Fraction of a total available move time that is allowed to use.
  float max_time_spend_fraction() const { return 0.5f; }

 private:
};

class SmoothTimeManager : public TimeManager {
 public:
  SmoothTimeManager(int64_t move_overhead, const OptionsDict& params)
      : params_(params, move_overhead) {}

 private:
  std::unique_ptr<SearchStopper> GetStopper(
      const GoParams& /* params */, const Position& /* position */) override {
    return nullptr;
  }

  const Params params_;

  // Fraction of a tree which usually survives a full move (and is reused).
  float tree_reuse_ = params_.initial_tree_reuse();
  // Current NPS estimation.
  float nps_ = params_.initial_nps();
  // Fraction of a budgeted time usually used.
  float timeuse_ = params_.initial_smartpruning_timeuse();
};

}  // namespace

std::unique_ptr<TimeManager> MakeSmoothTimeManager(int64_t move_overhead,
                                                   const OptionsDict& params) {
  return std::make_unique<SmoothTimeManager>(move_overhead, params);
}

}  // namespace lczero