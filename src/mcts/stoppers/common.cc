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
const OptionId kMinimumSmartPruningBatchesId{
    "smart-pruning-minimum-batches", "SmartPruningMinimumBatches",
    "Only allow smart pruning to stop search after at least this many batches "
    "have been evaluated. It may be useful to have this value greater than the "
    "number of search threads in use."};

}  // namespace

void PopulateCommonStopperOptions(RunType for_what, OptionsParser* options) {
  options->Add<IntOption>(kKLDGainAverageIntervalId, 1, 10000000) = 100;
  options->Add<FloatOption>(kMinimumKLDGainPerNodeId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kSmartPruningFactorId, 0.0f, 10.0f) =
      (for_what == RunType::kUci ? 1.33f : 0.00f);
  options->Add<IntOption>(kMinimumSmartPruningBatchesId, 0, 10000) = 0;

  if (for_what == RunType::kUci) {
    options->Add<IntOption>(kRamLimitMbId, 0, 100000000) = 0;
    options->Add<IntOption>(kMoveOverheadId, 0, 100000000) = 200;
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
    stopper->AddStopper(std::make_unique<SmartPruningStopper>(
        smart_pruning_factor,
        options.Get<int>(kMinimumSmartPruningBatchesId.GetId())));
  }
}

// Stoppers for uci mode only.
void PopulateCommonUciStoppers(ChainedSearchStopper* stopper,
                               const OptionsDict& options,
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

}  // namespace lczero