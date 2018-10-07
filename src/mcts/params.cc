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
const OptionId SearchParams::kMiniBatchSizeId{
    "minibatch-size", "MinibatchSize",
    "Now many positions the engine tries to batch together for computation.\n"
    "Theoretically larger batches may reduce strengths a bit, especially on "
    "small number of playouts."};
const OptionId SearchParams::kMaxPrefetchBatchId{
    "max-prefetch", "MaxPrefetch",
    "When engine cannot gather large enough batch for immediate use, try to "
    "prefetch up to X positions which are likely to be useful soon, and put "
    "them into cache."};
const OptionId SearchParams::kCpuctId{
    "cpuct", "CPuct",
    "C_puct constant from \"Upper confidence trees search\" "
    "algorithm. Higher values promote more exploration/wider search, lower "
    "values promote more confidence/deeper search."};
const OptionId SearchParams::kTemperatureId{
    "temperature", "Temperature",
    "Tau value from softmax formula for the first move. If equal to 0, the "
    "engine also picks the best move to make. Larger values increase "
    "randomness while making the move."};
const OptionId SearchParams::kTempDecayMovesId{
    "tempdecay-moves", "TempDecayMoves",
    "Reduce temperature for every move linearly from initial temperature to 0, "
    "during this number of moves since game start. 0 disables tempdecay."};
const OptionId SearchParams::kTemperatureVisitOffsetId{
    "temp-visit-offset", "TempVisitOffset", "Temperature visit offset."};
const OptionId SearchParams::kNoiseId{
    "noise", "Noise",
    "Add noise to root node prior probabilities. That allows engine to explore "
    "moves which are known to be very bad, which is useful to discover new "
    "ideas during training.",
    'n'};
const OptionId SearchParams::kVerboseStatsId{
    "verbose-move-stats", "VerboseMoveStats",
    "Display Q, V, N, U and P values of every move candidate after each move."};
const OptionId SearchParams::kAggressiveTimePruningId{
    "smart-pruning-factor", "SmartPruningFactor",
    "Aversion to search if change unlikely."};
const OptionId SearchParams::kFpuReductionId{"fpu-reduction", "FpuReduction",
                                             "First Play Urgency Reduction."};
const OptionId SearchParams::kCacheHistoryLengthId{
    "cache-history-length", "CacheHistoryLength",
    "Length of history to include in cache."};
const OptionId SearchParams::kPolicySoftmaxTempId{
    "policy-softmax-temp", "PolicySoftMaxTemp", "Policy softmax temperature."};
const OptionId SearchParams::kAllowedNodeCollisionsId{
    "allowed-node-collisions", "AllowedNodeCollisions",
    "Allowed node collisions, per batch."};
const OptionId SearchParams::kOutOfOrderEvalId{
    "out-of-order-eval", "OutOfOrderEval",
    "Out-of-order cache backpropagation."};
const OptionId SearchParams::kMultiPvId{
    "multipv", "MultiPV", "Number of moves to show in UCI info output."};

void SearchParams::Populate(OptionsParser* options) {
  // Here the "safe defaults" are listed.
  // Many of them are overridden with optimized defaults in engine.cc and
  // tournament.cc
  options->Add<IntOption>(kMiniBatchSizeId, 1, 1024) = 1;
  options->Add<IntOption>(kMaxPrefetchBatchId, 0, 1024) = 32;
  options->Add<FloatOption>(kCpuctId, 0.0f, 100.0f) = 1.2f;
  options->Add<FloatOption>(kTemperatureId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kTempDecayMovesId, 0, 100) = 0;
  options->Add<FloatOption>(kTemperatureVisitOffsetId, -0.99999f, 1000.0f) =
      0.0f;
  options->Add<BoolOption>(kNoiseId) = false;
  options->Add<BoolOption>(kVerboseStatsId) = false;
  options->Add<FloatOption>(kAggressiveTimePruningId, 0.0f, 10.0f) = 1.33f;
  options->Add<FloatOption>(kFpuReductionId, -100.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kCacheHistoryLengthId, 0, 7) = 7;
  options->Add<FloatOption>(kPolicySoftmaxTempId, 0.1f, 10.0f) = 1.0f;
  options->Add<IntOption>(kAllowedNodeCollisionsId, 0, 1024) = 0;
  options->Add<BoolOption>(kOutOfOrderEvalId) = false;
  options->Add<IntOption>(kMultiPvId, 1, 500) = 1;
}

SearchParams::SearchParams(const OptionsDict& options)
    : options_(options),
      kCpuct(options.Get<float>(kCpuctId.GetId())),
      kNoise(options.Get<bool>(kNoiseId.GetId())),
      kAggressiveTimePruning(
          options.Get<float>(kAggressiveTimePruningId.GetId())),
      kFpuReduction(options.Get<float>(kFpuReductionId.GetId())),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthId.GetId())),
      kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempId.GetId())),
      kAllowedNodeCollisions(
          options.Get<int>(kAllowedNodeCollisionsId.GetId())),
      kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalId.GetId())) {}

}  // namespace lczero