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

#include "mcts/stoppers/stoppers.h"

namespace lczero {

const OptionId kNNCacheSizeId{
    "nncache", "NNCacheSize",
    "Number of positions to store in a memory cache. A large cache can speed "
    "up searching, but takes memory."};

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
const OptionId kSlowMoverId{
    "slowmover", "Slowmover",
    "Budgeted time for a move is multiplied by this value, causing the engine "
    "to spend more time (if value is greater than 1) or less time (if the "
    "value is less than 1)."};
const OptionId kTimeMidpointMoveId{
    "time-midpoint-move", "TimeMidpointMove",
    "The move where the time budgeting algorithm guesses half of all "
    "games to be completed by. Half of the time allocated for the first move "
    "is allocated at approximately this move."};
const OptionId kTimeSteepnessId{
    "time-steepness", "TimeSteepness",
    "\"Steepness\" of the function the time budgeting algorithm uses to "
    "consider when games are completed. Lower values leave more time for "
    "the endgame, higher values use more time for each move before the "
    "midpoint."};
const OptionId kSpendSavedTimeId{
    "immediate-time-use", "ImmediateTimeUse",
    "Fraction of time saved by smart pruning, which is added to the budget to "
    "the next move rather than to the entire game. When 1, all saved time is "
    "added to the next move's budget; when 0, saved time is distributed among "
    "all future moves."};
const OptionId kMinimumKLDGainPerNodeId{
    "minimum-kldgain-per-node", "MinimumKLDGainPerNode",
    "If greater than 0 search will abort unless the last "
    "KLDGainAverageInterval nodes have an average gain per node of at least "
    "this much."};
const OptionId kKLDGainAverageIntervalId{
    "kldgain-average-interval", "KLDGainAverageInterval",
    "Used to decide how frequently to evaluate the average KLDGainPerNode to "
    "check the MinimumKLDGainPerNode, if specified."};
const OptionId kSmartPruningFactorId{
    "smart-pruning-factor", "SmartPruningFactor",
    "Do not spend time on the moves which cannot become bestmove given the "
    "remaining time to search. When no other move can overtake the current "
    "best, the search stops, saving the time. Values greater than 1 stop less "
    "promising moves from being considered even earlier. Values less than 1 "
    "causes hopeless moves to still have some attention. When set to 0, smart "
    "pruning is deactivated."};

}  // namespace

void PopulateTimeManagementOptions(RunType for_what, OptionsParser* options) {
  options->Add<IntOption>(kKLDGainAverageIntervalId, 1, 10000000) = 100;
  options->Add<FloatOption>(kMinimumKLDGainPerNodeId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kSmartPruningFactorId, 0.0f, 10.0f) =
      (for_what == RunType::kUci ? 1.33f : 0.00f);

  if (for_what == RunType::kUci) {
    options->Add<IntOption>(kRamLimitMbId, 0, 100000000) = 0;
    options->Add<IntOption>(kMoveOverheadId, 0, 100000000) = 200;
    options->Add<FloatOption>(kSlowMoverId, 0.0f, 100.0f) = 1.0f;
    options->Add<FloatOption>(kTimeMidpointMoveId, 1.0f, 100.0f) = 51.5f;
    options->Add<FloatOption>(kTimeSteepnessId, 1.0f, 100.0f) = 7.0f;
    options->Add<FloatOption>(kSpendSavedTimeId, 0.0f, 1.0f) = 1.0f;

    // Hide time curve options.
    options->HideOption(kTimeMidpointMoveId);
    options->HideOption(kTimeSteepnessId);
  }
}

// Parameters needed for selfplay and uci, but not benchmark nor infinite mode.
void PopulateIntrinsicStoppers(ChainedSearchStopper* stopper,
                               const OptionsDict& options) {
  // KLD gain.
  const auto min_kld_gain =
      options.Get<float>(kMinimumKLDGainPerNodeId.GetId());
  if (min_kld_gain > 0.0f) {
    stopper->AddStopper(std::make_unique<KldGainStopper>(
        min_kld_gain, options.Get<int>(kKLDGainAverageIntervalId.GetId())));
  }

  // Should be last in the chain.
  const auto smart_pruning_factor =
      options.Get<float>(kSmartPruningFactorId.GetId());
  if (smart_pruning_factor > 0.0f) {
    stopper->AddStopper(
        std::make_unique<SmartPruningStopper>(smart_pruning_factor));
  }
}

namespace {
// Stoppers for uci mode only.
void PopulateStoppers(ChainedSearchStopper* stopper, const OptionsDict& options,
                      const GoParams& params) {
  const bool infinite = params.infinite || params.ponder;
  const int64_t move_overhead = options.Get<int>(kMoveOverheadId.GetId());

  // RAM limit watching stopper.
  const auto cache_size_mb = options.Get<int>(kNNCacheSizeId.GetId());
  const int ram_limit = options.Get<int>(kRamLimitMbId.GetId());
  if (ram_limit) {
    stopper->AddStopper(
        std::make_unique<MemoryWatchingStopper>(cache_size_mb, ram_limit));
  }

  // "go nodes" stopper.
  if (params.nodes) {
    stopper->AddStopper(std::make_unique<VisitsStopper>(*params.nodes));
  }

  // "go movetime" stopper.
  if (params.movetime && !infinite) {
    stopper->AddStopper(
        std::make_unique<TimeLimitStopper>(*params.movetime - move_overhead));
  }

  // "go depth" stopper.
  if (params.depth) {
    stopper->AddStopper(std::make_unique<DepthStopper>(*params.depth));
  }

  // Add internal search tree stoppers when we want to automatically stop.
  if (!infinite) PopulateIntrinsicStoppers(stopper, options);
}

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

class LegacyTimeManager : public TimeManager {
 public:
  void ResetGame() override;
  std::unique_ptr<SearchStopper> GetStopper(const OptionsDict& options,
                                            const GoParams& params,
                                            const Position& position) override;

 private:
  std::unique_ptr<SearchStopper> CreateTimeManagementStopper(
      const OptionsDict& options, const GoParams& params,
      const Position& position);
  // No need to be atomic as only one thread will update it.
  int64_t time_spared_ms_ = 0;
};
}  // namespace

std::unique_ptr<TimeManager> MakeLegacyTimeManager() {
  return std::make_unique<LegacyTimeManager>();
}

void LegacyTimeManager::ResetGame() { time_spared_ms_ = 0; }

std::unique_ptr<SearchStopper> LegacyTimeManager::CreateTimeManagementStopper(
    const OptionsDict& options, const GoParams& params,
    const Position& position) {
  const bool is_black = position.IsBlackToMove();
  const optional<int64_t>& time = (is_black ? params.btime : params.wtime);
  // If no time limit is given, don't stop on this condition.
  if (params.infinite || params.ponder || !time) return nullptr;

  const int64_t move_overhead = options.Get<int>(kMoveOverheadId.GetId());
  const optional<int64_t>& inc = is_black ? params.binc : params.winc;
  const int increment = inc ? std::max(int64_t(0), *inc) : 0;

  // How to scale moves time.
  const float slowmover = options.Get<float>(kSlowMoverId.GetId());
  const float time_curve_midpoint =
      options.Get<float>(kTimeMidpointMoveId.GetId());
  const float time_curve_steepness =
      options.Get<float>(kTimeSteepnessId.GetId());

  float movestogo = ComputeEstimatedMovesToGo(
      position.GetGamePly(), time_curve_midpoint, time_curve_steepness);

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
      std::max(0.0f, *time + increment * (movestogo - 1) - move_overhead);

  // If there is time spared from previous searches, the `time_to_squander` part
  // of it will be used immediately, remove that from planning.
  int time_to_squander = 0;
  if (time_spared_ms_ > 0) {
    time_to_squander =
        time_spared_ms_ * options.Get<float>(kSpendSavedTimeId.GetId());
    time_spared_ms_ -= time_to_squander;
    total_moves_time -= time_to_squander;
  }

  // Evenly split total time between all moves.
  float this_move_time = total_moves_time / movestogo;

  // Only extend thinking time with slowmover if smart pruning can potentially
  // reduce it.
  constexpr int kSmartPruningToleranceMs = 200;
  if (slowmover < 1.0 ||
      this_move_time * slowmover > kSmartPruningToleranceMs) {
    this_move_time *= slowmover;
    // If time is planned to be overused because of slowmover, remove excess
    // of that time from spared time.
    time_spared_ms_ -= this_move_time * (slowmover - 1);
  }

  LOGFILE << "Budgeted time for the move: " << this_move_time << "ms(+"
          << time_to_squander << "ms to squander). Remaining time " << *time
          << "ms(-" << move_overhead << "ms overhead)";
  // Use `time_to_squander` time immediately.
  this_move_time += time_to_squander;

  // Make sure we don't exceed current time limit with what we calculated.
  auto deadline =
      std::min(static_cast<int64_t>(this_move_time), *time - move_overhead);
  return std::make_unique<LegacyStopper>(deadline, &time_spared_ms_);
}

std::unique_ptr<SearchStopper> LegacyTimeManager::GetStopper(
    const OptionsDict& options, const GoParams& params,
    const Position& position) {
  auto result = std::make_unique<ChainedSearchStopper>();

  // Time management stopper.
  result->AddStopper(CreateTimeManagementStopper(options, params, position));
  // All the standard stoppers (go nodes, RAM limit, smart pruning, etc).
  PopulateStoppers(result.get(), options, params);
  return std::move(result);
}

}  // namespace lczero
