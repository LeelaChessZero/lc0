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

#include "mcts/timemgr/timemgr.h"
#include "mcts/timemgr/stoppers.h"

namespace lczero {
namespace {

const OptionId kRamLimitMbId{
    "ramlimit-mb", "RamLimitMb",
    "Maximum memory usage for the engine, in megabytes. The estimation is very "
    "rough, and can be off by a lot. For example, multiple visits to a "
    "terminal node counted several times, and the estimation assumes that all "
    "positions have 30 possible moves. When set to 0, no RAM limit is "
    "enforced."};
const OptionId kMoveOverheadId{
    "move-overhead", "MoveOverheadMs",
    "Amount of time, in milliseconds, that the engine subtracts from it's "
    "total available time (to compensate for slow connection, interprocess "
    "communication, etc)."};

template <class T>
void Maximize(T* x) {
  *x = std::numeric_limits<T>::max();
}
}  // namespace

void TimeManagerHints::Reset() {
  Maximize(&remaining_time_);
  Maximize(&remaining_playouts_);
}

void PopulateTimeManagementOptions(OptionsParser* options) {
  options->Add<IntOption>(kRamLimitMbId, 0, 100000000) = 0;
  options->Add<IntOption>(kMoveOverheadId, 0, 100000000) = 200;
}

std::unique_ptr<SearchStopper> MakeSearchStopper(
    const OptionsDict& options, const GoParams& params, const TimeManager&,
    std::chrono::steady_clock::time_point start_time, int cache_size_mb) {
  const bool infinite = params.infinite || params.ponder;
  const int64_t move_overhead = options.Get<int>(kMoveOverheadId.GetId());

  // Stopper container.
  auto result = std::make_unique<ChainedSearchStopper>();

  // RAM limit watching stopper.
  const int ram_limit = options.Get<int>(kRamLimitMbId.GetId());
  if (ram_limit) {
    result->AddStopper(
        std::make_unique<MemoryWatchingStopper>(cache_size_mb, ram_limit));
  }

  // "go nodes" stopper.
  if (params.nodes) {
    result->AddStopper(std::make_unique<VisitsStopper>(*params.nodes));
  }

  // "go movetime" stopper.
  if (params.movetime && !infinite) {
    result->AddStopper(std::make_unique<DeadlineStopper>(
        start_time +
        std::chrono::milliseconds(*params.movetime - move_overhead)));
  }

  return std::move(result);
}

}  // namespace lczero