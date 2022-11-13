/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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

#include "mcts/stoppers/timemgr.h"

#include "mcts/stoppers/stoppers.h"

namespace lczero {

StoppersHints::StoppersHints() { Reset(); }

void StoppersHints::UpdateIndexOfBestEdge(int64_t v) {
  index_of_best_edge_ = v;
}
int64_t StoppersHints::GetIndexOfBestEdge() const {
  return index_of_best_edge_;
}

void StoppersHints::UpdateEstimatedRemainingTimeMs(int64_t v) {
  if (v < remaining_time_ms_) remaining_time_ms_ = v;
}
int64_t StoppersHints::GetEstimatedRemainingTimeMs() const {
  return remaining_time_ms_;
}

void StoppersHints::UpdateEstimatedRemainingPlayouts(int64_t v) {
  if (v < remaining_playouts_) remaining_playouts_ = v;
}
int64_t StoppersHints::GetEstimatedRemainingPlayouts() const {
  // Even if we exceeded limits, don't go crazy by not allowing any playouts.
  return std::max(decltype(remaining_playouts_){1}, remaining_playouts_);
}

void StoppersHints::UpdateEstimatedNps(float v) { estimated_nps_ = v; }

std::optional<float> StoppersHints::GetEstimatedNps() const {
  return estimated_nps_;
}

void StoppersHints::Reset() {
  // Slightly more than 3 years.
  remaining_time_ms_ = 100000000000;
  // Type for N in nodes is currently uint32_t, so set limit in order not to
  // overflow it.
  remaining_playouts_ = 4000000000;
  // NPS is not known.
  estimated_nps_.reset();
}

}  // namespace lczero
