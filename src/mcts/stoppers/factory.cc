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

#include "mcts/stoppers/factory.h"

#include <optional>

#include "factory.h"
#include "mcts/stoppers/stoppers.h"
#include "utils/exception.h"

namespace lczero {

const OptionId kNNCacheSizeId{
    "nncache", "NNCacheSize",
    "Number of positions to store in a memory cache. A large cache can speed "
    "up searching, but takes memory."};

namespace {

const OptionId kMoveOverheadId{
    "move-overhead", "MoveOverheadMs",
    "Amount of time, in milliseconds, that the engine subtracts from it's "
    "total available time (to compensate for slow connection, interprocess "
    "communication, etc)."};
const OptionId kTimeManagerId{"time-manager", "TimeManager",
                              "Name and config of atime manager."};

}  // namespace

void PopulateTimeManagementOptions(RunType for_what, OptionsParser* options) {
  PopulateCommonStopperOptions(for_what, options);
  if (for_what == RunType::kUci) {
    options->Add<IntOption>(kMoveOverheadId, 0, 100000000) = 200;
    options->Add<StringOption>(kTimeManagerId) = "legacy";
  }
}

std::unique_ptr<TimeManager> MakeTimeManager() {
  const int64_t move_overhead = options.Get<int>(kMoveOverheadId);
  const std::string time_manager_config =
      options.Get<std::string>(kTimeManagerId);
  const auto managers = options.ListSubdicts();

  if (managers.size() != 1) {
    throw Exception("Exactly one time manager should be specified, " +
                    std::to_string(managers.size()) + " specified instead.");
  }
  if (manager[0] == "legacy") {
  } else {
    throw Exception("Unknown time manager: [" + manager[0] + "]");
  }
}

}  // namespace lczero
