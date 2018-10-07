/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "mcts/params.h"

namespace lczero {
const char* SearchParams::kMiniBatchSizeStr = "Minibatch size for NN inference";
const char* SearchParams::kMaxPrefetchBatchStr =
    "Max prefetch nodes, per NN call";
const char* SearchParams::kCpuctStr = "Cpuct MCTS option";
const char* SearchParams::kTemperatureStr = "Initial temperature";
const char* SearchParams::kTempDecayMovesStr = "Moves with temperature decay";
const char* SearchParams::kTemperatureVisitOffsetStr =
    "Temperature visit offset";
const char* SearchParams::kNoiseStr = "Add Dirichlet noise at root node";
const char* SearchParams::kVerboseStatsStr = "Display verbose move stats";
const char* SearchParams::kAggressiveTimePruningStr =
    "Aversion to search if change unlikely";
const char* SearchParams::kFpuReductionStr = "First Play Urgency Reduction";
const char* SearchParams::kCacheHistoryLengthStr =
    "Length of history to include in cache";
const char* SearchParams::kPolicySoftmaxTempStr = "Policy softmax temperature";
const char* SearchParams::kAllowedNodeCollisionEventsStr =
    "Allowed node collision events, per batch";
const char* SearchParams::kAllowedTotalNodeCollisionsStr =
    "Total allowed node collisions, per batch";
const char* SearchParams::kOutOfOrderEvalStr =
    "Out-of-order cache backpropagation";
const char* SearchParams::kMultiPvStr = "MultiPV";

void SearchParams::Populate(OptionsParser* options) {
  // Here the "safe defaults" are listed.
  // Many of them are overridden with optimized defaults in engine.cc and
  // tournament.cc
  options->Add<IntOption>(kMiniBatchSizeStr, 1, 1024, "minibatch-size") = 1;
  options->Add<IntOption>(kMaxPrefetchBatchStr, 0, 1024, "max-prefetch") = 32;
  options->Add<FloatOption>(kCpuctStr, 0.0f, 100.0f, "cpuct") = 1.2f;
  options->Add<FloatOption>(kTemperatureStr, 0.0f, 100.0f, "temperature") =
      0.0f;
  options->Add<FloatOption>(kTemperatureVisitOffsetStr, -0.99999f, 1000.0f,
                            "temp-visit-offset") = 0.0f;
  options->Add<IntOption>(kTempDecayMovesStr, 0, 100, "tempdecay-moves") = 0;
  options->Add<BoolOption>(kNoiseStr, "noise", 'n') = false;
  options->Add<BoolOption>(kVerboseStatsStr, "verbose-move-stats") = false;
  options->Add<FloatOption>(kAggressiveTimePruningStr, 0.0f, 10.0f,
                            "futile-search-aversion") = 1.33f;
  options->Add<FloatOption>(kFpuReductionStr, -100.0f, 100.0f,
                            "fpu-reduction") = 0.0f;
  options->Add<IntOption>(kCacheHistoryLengthStr, 0, 7,
                          "cache-history-length") = 7;
  options->Add<FloatOption>(kPolicySoftmaxTempStr, 0.1f, 10.0f,
                            "policy-softmax-temp") = 1.0f;
  options->Add<IntOption>(kAllowedNodeCollisionEventsStr, 0, 1024,
                          "allowed-node-collision-events") = 32;
  options->Add<IntOption>(kAllowedTotalNodeCollisionsStr, 0, 1000000,
                          "allowed-total-node-collisions") = 10000;
  options->Add<BoolOption>(kOutOfOrderEvalStr, "out-of-order-eval") = false;
  options->Add<IntOption>(kMultiPvStr, 1, 500, "multipv") = 1;
}

SearchParams::SearchParams(const OptionsDict& options)
    : options_(options),
      kCpuct(options.Get<float>(kCpuctStr)),
      kNoise(options.Get<bool>(kNoiseStr)),
      kAggressiveTimePruning(options.Get<float>(kAggressiveTimePruningStr)),
      kFpuReduction(options.Get<float>(kFpuReductionStr)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthStr)),
      kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempStr)),
      kAllowedNodeCollisionEvents(
          options.Get<int>(kAllowedNodeCollisionEventsStr)),
      kAllowedTotalNodeCollisions(
          options.Get<int>(kAllowedTotalNodeCollisionsStr)),
      kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalStr)) {}

}  // namespace lczero