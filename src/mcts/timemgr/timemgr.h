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

#pragma once

#include <chrono>
#include <memory>
#include "chess/uciloop.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

// DO NOT SUBMIT actually fill all that!
struct IterationStats {
  void Reset() { *this = IterationStats(); }
  bool is_watchdog = false;
  int64_t total_nodes = 0;
  int average_depth = 0;
  int maximum_depath = 0;
};

class TimeManagerHints {
 public:
  TimeManagerHints() { Reset(); }
  void Reset();

  void UpdateEstimatedRemainingTime(std::chrono::milliseconds v) {
    if (v < remaining_time_) remaining_time_ = v;
  }
  std::chrono::milliseconds GetEstimatedRemainingTime() const {
    return remaining_time_;
  }

  void UpdateEstimatedRemainingRemainingPlayouts(int64_t v) {
    if (v < remaining_playouts_) remaining_playouts_ = v;
  }
  int64_t GetEstimatedRemainingPlayouts() const { return remaining_playouts_; }

 private:
  std::chrono::milliseconds remaining_time_;
  int64_t remaining_playouts_;
};

class SearchStopper {
 public:
  virtual ~SearchStopper() = default;
  virtual bool ShouldStop(const IterationStats&, TimeManagerHints*) = 0;
};

class TimeManager {};

void PopulateTimeManagementOptions(OptionsParser* options);
std::unique_ptr<SearchStopper> MakeSearchStopper(
    const OptionsDict& dict, const GoParams& params,
    const TimeManager& time_mgr,
    std::chrono::steady_clock::time_point start_time, int cache_size_mb);

}  // namespace lczero

// DO NOT SUBMIT
// - Only one possible move left (but not during infinte!)