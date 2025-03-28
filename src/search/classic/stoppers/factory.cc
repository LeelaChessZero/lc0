/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2022 The LCZero Authors

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

#include "search/classic/stoppers/factory.h"

#include <optional>

#include "factory.h"
#include "search/classic/stoppers/alphazero.h"
#include "search/classic/stoppers/legacy.h"
#include "search/classic/stoppers/simple.h"
#include "search/classic/stoppers/smooth.h"
#include "search/classic/stoppers/stoppers.h"
#include "utils/exception.h"

namespace lczero {
namespace classic {
namespace {

const OptionId kMoveOverheadId{
    "move-overhead", "MoveOverheadMs",
    "Amount of time, in milliseconds, that the engine subtracts from it's "
    "total available time (to compensate for slow connection, interprocess "
    "communication, etc)."};
const OptionId kTimeManagerId{
    "time-manager", "TimeManager",
    "Name and config of a time manager. "
    "Possible names are 'legacy' (default), 'smooth', 'alphazero', and simple."
    "See https://lc0.org/timemgr for configuration details."};
const OptionId kSlowMoverId{
    "slowmover", "Slowmover",
    "Budgeted time for a move is multiplied by this value, causing the engine "
    "to spend more time (if value is greater than 1) or less time (if the "
    "value is less than 1)."};
}  // namespace

void PopulateTimeManagementOptions(RunType for_what, OptionsParser* options) {
  PopulateCommonStopperOptions(for_what, options);
  if (for_what == RunType::kUci || for_what == RunType::kSimpleUci) {
    options->Add<IntOption>(kMoveOverheadId, 0, 100000000) = 200;
    if (for_what == RunType::kUci) {
      options->Add<StringOption>(kTimeManagerId) = "legacy";
    } else {
      options->Add<FloatOption>(kSlowMoverId, 0.0f, 100.0f) = 1.0f;
    }
  }
}

std::unique_ptr<TimeManager> MakeTimeManager(const OptionsDict& options) {
  const int64_t move_overhead = options.Get<int>(kMoveOverheadId);

  OptionsDict tm_options;
  if (options.Exists<std::string>(kTimeManagerId)) {
    tm_options.AddSubdictFromString(options.Get<std::string>(kTimeManagerId));
  } else {
    float slowmover = options.Get<float>(kSlowMoverId);
    tm_options.AddSubdict("legacy")->Set("slowmover", slowmover);
  }
  const auto managers = tm_options.ListSubdicts();

  std::unique_ptr<TimeManager> time_manager;
  if (managers.size() != 1) {
    throw Exception("Exactly one time manager should be specified, " +
                    std::to_string(managers.size()) + " specified instead.");
  }

  if (managers[0] == "legacy") {
    time_manager =
        MakeLegacyTimeManager(move_overhead, tm_options.GetSubdict("legacy"));
  } else if (managers[0] == "alphazero") {
    time_manager = MakeAlphazeroTimeManager(move_overhead,
                                            tm_options.GetSubdict("alphazero"));
  } else if (managers[0] == "smooth") {
    time_manager =
        MakeSmoothTimeManager(move_overhead, tm_options.GetSubdict("smooth"));
  } else if (managers[0] == "simple") {
    time_manager =
        MakeSimpleTimeManager(move_overhead, tm_options.GetSubdict("simple"));
  }

  if (!time_manager) {
    throw Exception("Unknown time manager: [" + managers[0] + "]");
  }
  tm_options.CheckAllOptionsRead("");

  return MakeCommonTimeManager(std::move(time_manager), options, move_overhead);
}

}  // namespace classic
}  // namespace lczero
