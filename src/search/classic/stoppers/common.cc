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

#include "search/classic/stoppers/common.h"

#include "neural/shared_params.h"

namespace lczero {
namespace classic {

namespace {
const OptionId kRamLimitMbId{
    {.long_flag = "ramlimit-mb",
     .uci_option = "RamLimitMb",
     .help_text =
         "Maximum memory usage for the engine, in megabytes. The estimation is "
         "very rough, and can be off by a lot. For example, multiple visits to "
         "a terminal node counted several times, and the estimation assumes "
         "that all positions have 30 possible moves. When set to 0, no RAM "
         "limit is enforced.",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kMinimumKLDGainPerNodeId{
    {.long_flag = "minimum-kldgain-per-node",
     .uci_option = "MinimumKLDGainPerNode",
     .help_text = "If greater than 0 search will abort unless the last "
                  "KLDGainAverageInterval nodes have an average gain per node "
                  "of at least this much.",
     .visibility = OptionId::kProOnly}};
const OptionId kKLDGainAverageIntervalId{
    {.long_flag = "kldgain-average-interval",
     .uci_option = "KLDGainAverageInterval",
     .help_text =
         "Used to decide how frequently to evaluate the average KLDGainPerNode "
         "to check the MinimumKLDGainPerNode, if specified.",
     .visibility = OptionId::kProOnly}};
const OptionId kSmartPruningFactorId{
    {.long_flag = "smart-pruning-factor",
     .uci_option = "SmartPruningFactor",
     .help_text =
         "Do not spend time on the moves which cannot become bestmove given "
         "the remaining time to search. When no other move can overtake the "
         "current best, the search stops, saving the time. Values greater than "
         "1 stop less promising moves from being considered even earlier. "
         "Values less than 1 causes hopeless moves to still have some "
         "attention. When set to 0, smart pruning is deactivated.",
     .visibility = OptionId::kDefaultVisibility}};
const OptionId kMinimumSmartPruningBatchesId{
    {.long_flag = "smart-pruning-minimum-batches",
     .uci_option = "SmartPruningMinimumBatches",
     .help_text =
         "Only allow smart pruning to stop search after at least this many "
         "batches have been evaluated. It may be useful to have this value "
         "greater than the number of search threads in use.",
     .visibility = OptionId::kDefaultVisibility}};
const OptionId kNodesAsPlayoutsId{
    {.long_flag = "nodes-as-playouts",
     .uci_option = "NodesAsPlayouts",
     .help_text = "Treat UCI `go nodes` command as referring to playouts "
                  "instead of visits.",
     .visibility = OptionId::kProOnly}};

}  // namespace

void PopulateCommonStopperOptions(RunType for_what, OptionsParser* options) {
  options->Add<IntOption>(kKLDGainAverageIntervalId, 1, 10000000) = 100;
  options->Add<FloatOption>(kMinimumKLDGainPerNodeId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kSmartPruningFactorId, 0.0f, 10.0f) =
      (for_what == RunType::kUci ? 1.33f : 0.00f);
  options->Add<IntOption>(kMinimumSmartPruningBatchesId, 0, 10000) = 0;
  options->Add<BoolOption>(kNodesAsPlayoutsId) = false;
  options->Add<IntOption>(kRamLimitMbId, 0, 100000000) = 0;
}

// Parameters needed for selfplay and uci, but not benchmark nor infinite mode.
void PopulateIntrinsicStoppers(ChainedSearchStopper* stopper,
                               const OptionsDict& options) {
  // KLD gain.
  const auto min_kld_gain = options.Get<float>(kMinimumKLDGainPerNodeId);
  if (min_kld_gain > 0.0f) {
    stopper->AddStopper(std::make_unique<KldGainStopper>(
        min_kld_gain, options.Get<int>(kKLDGainAverageIntervalId)));
  }

  // Should be last in the chain.
  const auto smart_pruning_factor = options.Get<float>(kSmartPruningFactorId);
  if (smart_pruning_factor > 0.0f) {
    stopper->AddStopper(std::make_unique<SmartPruningStopper>(
        smart_pruning_factor, options.Get<int>(kMinimumSmartPruningBatchesId)));
  }
}

namespace {
// Stoppers for uci mode only.
void PopulateCommonUciStoppers(ChainedSearchStopper* stopper,
                               const OptionsDict& options,
                               const GoParams& params, size_t total_memory,
                               size_t avg_node_size, uint32_t nodes,
                               int64_t move_overhead) {
  const bool infinite = params.infinite || params.ponder || params.mate;

  // RAM limit watching stopper.
  const int ram_limit_mb = options.Get<int>(kRamLimitMbId);
  if (ram_limit_mb) {
    stopper->AddStopper(std::make_unique<MemoryWatchingStopper>(
        ram_limit_mb, total_memory, avg_node_size, nodes,
        options.Get<float>(kSmartPruningFactorId) > 0.0f));
  }

  // "go nodes" stopper.
  int64_t node_limit = 4000000000;
  if (params.nodes.has_value()) {
    if (options.Get<bool>(kNodesAsPlayoutsId)) {
      stopper->AddStopper(std::make_unique<PlayoutsStopper>(
          *params.nodes, options.Get<float>(kSmartPruningFactorId) > 0.0f));
    } else {
      node_limit = *params.nodes;
    }
  }
  // Always limit nodes to avoid exceeding the limit 4000000000.
  stopper->AddStopper(std::make_unique<VisitsStopper>(
      node_limit, options.Get<float>(kSmartPruningFactorId) > 0.0f));

  // "go movetime" stopper.
  if (params.movetime && !infinite) {
    stopper->AddStopper(
        std::make_unique<TimeLimitStopper>(*params.movetime - move_overhead));
  }

  // "go depth" stopper.
  if (params.depth) {
    stopper->AddStopper(std::make_unique<DepthStopper>(*params.depth));
  }

  // "go mate" stopper.
  if (params.mate) {
    stopper->AddStopper(std::make_unique<MateStopper>(*params.mate));
  }

  // Add internal search tree stoppers when we want to automatically stop.
  if (!infinite) PopulateIntrinsicStoppers(stopper, options);
}

class CommonTimeManager : public TimeManager {
 public:
  CommonTimeManager(std::unique_ptr<TimeManager> child_mgr,
                    const OptionsDict& options, int64_t move_overhead)
      : child_mgr_(std::move(child_mgr)),
        options_(options),
        move_overhead_(move_overhead) {}

 private:
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const Position& position,
                                            size_t avg_node_size,
                                            size_t total_memory,
                                            uint32_t nodes) override {
    auto result = std::make_unique<ChainedSearchStopper>();
    if (child_mgr_)
      result->AddStopper(child_mgr_->GetStopper(params, position, avg_node_size,
                                                total_memory, nodes));
    PopulateCommonUciStoppers(result.get(), options_, params, avg_node_size,
                              total_memory, nodes, move_overhead_);
    return result;
  }

  const std::unique_ptr<TimeManager> child_mgr_;
  const OptionsDict& options_;
  const int64_t move_overhead_;
};

}  // namespace

std::unique_ptr<TimeManager> MakeCommonTimeManager(
    std::unique_ptr<TimeManager> child_manager, const OptionsDict& options,
    int64_t move_overhead) {
  return std::make_unique<CommonTimeManager>(std::move(child_manager), options,
                                             move_overhead);
}

}  // namespace classic
}  // namespace lczero
