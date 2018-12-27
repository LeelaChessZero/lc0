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

namespace {
FillEmptyHistory EncodeHistoryFill(std::string history_fill) {
  if (history_fill == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (history_fill == "always") return FillEmptyHistory::ALWAYS;
  assert(history_fill == "no");
  return FillEmptyHistory::NO;
}

}  // namespace

const OptionId SearchParams::kMiniBatchSizeId{
    "minibatch-size", "MinibatchSize",
    "How many positions the engine tries to batch together for parallel NN "
    "computation. Larger batches may reduce strength a bit, especially with a "
    "small number of playouts."};
const OptionId SearchParams::kMaxPrefetchBatchId{
    "max-prefetch", "MaxPrefetch",
    "When the engine cannot gather a large enough batch for immediate use, try "
    "to prefetch up to X positions which are likely to be useful soon, and put "
    "them into cache."};
const OptionId SearchParams::kCpuctId{
    "cpuct", "CPuct",
    "cpuct_init constant from \"UCT search\" algorithm. Higher values promote "
    "more exploration/wider search, lower values promote more "
    "confidence/deeper search."};
const OptionId SearchParams::kCpuctBaseId{
    "cpuct-base", "CPuctBase",
    "cpuct_base constant from \"UCT search\" algorithm. Lower value means "
    "higher growth of Cpuct as number of node visits grows."};
const OptionId SearchParams::kCpuctFactorId{
    "cpuct-factor", "CPuctFactor", "Multiplier for the cpuct growth formula."};
const OptionId SearchParams::kTemperatureId{
    "temperature", "Temperature",
    "Tau value from softmax formula for the first move. If equal to 0, the "
    "engine picks the best move to make. Larger values increase randomness "
    "while making the move."};
const OptionId SearchParams::kTempDecayMovesId{
    "tempdecay-moves", "TempDecayMoves",
    "Reduce temperature for every move from the game start to this number of "
    "moves, decreasing linearly from initial temperature to 0. A value of 0 "
    "disables tempdecay."};
const OptionId SearchParams::kTemperatureCutoffMoveId{
    "temp-cutoff-move", "TempCutoffMove",
    "Move number, starting from which endgame temperature is used rather "
    "than initial temperature. Setting it to 0 disables cutoff."};
const OptionId SearchParams::kTemperatureEndgameId{
    "temp-endgame", "TempEndgame",
    "Temperature used during endgame (starting from cutoff move). Endgame "
    "temperature doesn't decay."};
const OptionId SearchParams::kTemperatureWinpctCutoffId{
    "temp-value-cutoff", "TempValueCutoff",
    "When move is selected using temperature, bad moves (with win "
    "probability less than X than the best move) are not considered at all."};
const OptionId SearchParams::kTemperatureVisitOffsetId{
    "temp-visit-offset", "TempVisitOffset",
    "Reduces visits by this value when picking a move with a temperature. When "
    "the offset is less than number of visits for a particular move, that move "
    "is not picked at all."};
const OptionId SearchParams::kNoiseId{
    "noise", "DirichletNoise",
    "Add Dirichlet noise to root node prior probabilities. This allows the "
    "engine to discover new ideas during training by exploring moves which are "
    "known to be bad. Not normally used during play.",
    'n'};
const OptionId SearchParams::kVerboseStatsId{
    "verbose-move-stats", "VerboseMoveStats",
    "Display Q, V, N, U and P values of every move candidate after each move."};
const OptionId SearchParams::kSmartPruningFactorId{
    "smart-pruning-factor", "SmartPruningFactor",
    "Do not spend time on the moves which cannot become bestmove given the "
    "remaining time to search. When no other move can overtake the current "
    "best, the search stops, saving the time. Values greater than 1 stop less "
    "promising moves from being considered even earlier. Values less than 1 "
    "causes hopeless moves to still have some attention. When set to 0, smart "
    "pruning is deactivated."};
const OptionId SearchParams::kFpuStrategyId{
    "fpu-strategy", "FpuStrategy",
    "How is an eval of unvisited node determined. \"reduction\" subtracts "
    "--fpu-reduction value from the parent eval. \"absolute\" sets eval of "
    "unvisited nodes to the value specified in --fpu-value."};
// TODO(crem) Make FPU in "reduction" mode use fpu-value too. For now it's kept
// for backwards compatibility.
const OptionId SearchParams::kFpuReductionId{
    "fpu-reduction", "FpuReduction",
    "\"First Play Urgency\" reduction (used when FPU strategy is "
    "\"reduction\"). Normally when a move has no visits, "
    "it's eval is assumed to be equal to parent's eval. With non-zero FPU "
    "reduction, eval of unvisited move is decreased by that value, "
    "discouraging visits of unvisited moves, and saving those visits for "
    "(hopefully) more promising moves."};
const OptionId SearchParams::kFpuValueId{
    "fpu-value", "FpuValue",
    "\"First Play Urgency\" value. When FPU strategy is \"absolute\", value of "
    "unvisited node is assumed to be equal to this value, and does not depend "
    "on parent eval."};
const OptionId SearchParams::kCacheHistoryLengthId{
    "cache-history-length", "CacheHistoryLength",
    "Length of history, in half-moves, to include into the cache key. When "
    "this value is less than history that NN uses to eval a position, it's "
    "possble that the search will use eval of the same position with different "
    "history taken from cache."};
const OptionId SearchParams::kPolicySoftmaxTempId{
    "policy-softmax-temp", "PolicyTemperature",
    "Policy softmax temperature. Higher values make priors of move candidates "
    "closer to each other, widening the search."};
const OptionId SearchParams::kMaxCollisionVisitsId{
    "max-collision-visits", "MaxCollisionVisits",
    "Total allowed node collision visits, per batch."};
const OptionId SearchParams::kMaxCollisionEventsId{
    "max-collision-events", "MaxCollisionEvents",
    "Allowed node collision events, per batch."};
const OptionId SearchParams::kOutOfOrderEvalId{
    "out-of-order-eval", "OutOfOrderEval",
    "During the gathering of a batch for NN to eval, if position happens to be "
    "in the cache or is terminal, evaluate it right away without sending the "
    "batch to the NN. When off, this may only happen with the very first node "
    "of a batch; when on, this can happen with any node."};
const OptionId SearchParams::kMultiPvId{
    "multipv", "MultiPV",
    "Number of game play lines (principal variations) to show in UCI info "
    "output."};
const OptionId SearchParams::kScoreTypeId{
    "score-type", "ScoreType",
    "What to display as score. Either centipawns (the UCI default), win "
    "percentage or Q (the actual internal score) multiplied by 100."};
const OptionId SearchParams::kHistoryFillId{
    "history-fill", "HistoryFill",
    "Neural network uses 7 previous board positions in addition to the current "
    "one. During the first moves of the game such historical positions don't "
    "exist, but they can be synthesized. This parameter defines when to "
    "synthesize them (always, never, or only at non-standard fen position)."};

void SearchParams::Populate(OptionsParser* options) {
  // Here the uci optimized defaults" are set.
  // Many of them are overridden with training specific values in tournament.cc.
  options->Add<IntOption>(kMiniBatchSizeId, 1, 1024) = 256;
  options->Add<IntOption>(kMaxPrefetchBatchId, 0, 1024) = 32;
  options->Add<FloatOption>(kCpuctId, 0.0f, 100.0f) = 3.0f;
  options->Add<FloatOption>(kCpuctBaseId, 1.0f, 1000000000.0f) = 19652.0f;
  options->Add<FloatOption>(kCpuctFactorId, 0.0f, 1000.0f) = 2.0f;
  options->Add<FloatOption>(kTemperatureId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kTempDecayMovesId, 0, 100) = 0;
  options->Add<IntOption>(kTemperatureCutoffMoveId, 0, 1000) = 0;
  options->Add<FloatOption>(kTemperatureEndgameId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kTemperatureWinpctCutoffId, 0.0f, 100.0f) = 100.0f;
  options->Add<FloatOption>(kTemperatureVisitOffsetId, -0.99999f, 1000.0f) =
      0.0f;
  options->Add<BoolOption>(kNoiseId) = false;
  options->Add<BoolOption>(kVerboseStatsId) = false;
  options->Add<FloatOption>(kSmartPruningFactorId, 0.0f, 10.0f) = 1.33f;
  std::vector<std::string> fpu_strategy = {"reduction", "absolute"};
  options->Add<ChoiceOption>(kFpuStrategyId, fpu_strategy) = "reduction";
  options->Add<FloatOption>(kFpuReductionId, -100.0f, 100.0f) = 1.2f;
  options->Add<FloatOption>(kFpuValueId, -1.0f, 1.0f) = -1.0f;
  options->Add<IntOption>(kCacheHistoryLengthId, 0, 7) = 0;
  options->Add<FloatOption>(kPolicySoftmaxTempId, 0.1f, 10.0f) = 2.2f;
  options->Add<IntOption>(kMaxCollisionEventsId, 1, 1024) = 32;
  options->Add<IntOption>(kMaxCollisionVisitsId, 1, 1000000) = 9999;
  options->Add<BoolOption>(kOutOfOrderEvalId) = true;
  options->Add<IntOption>(kMultiPvId, 1, 500) = 1;
  std::vector<std::string> score_type = {"centipawn", "win_percentage", "Q"};
  options->Add<ChoiceOption>(kScoreTypeId, score_type) = "centipawn";
  std::vector<std::string> history_fill_opt{"no", "fen_only", "always"};
  options->Add<ChoiceOption>(kHistoryFillId, history_fill_opt) = "fen_only";
}

SearchParams::SearchParams(const OptionsDict& options)
    : options_(options),
      kCpuct(options.Get<float>(kCpuctId.GetId())),
      kCpuctBase(options.Get<float>(kCpuctBaseId.GetId())),
      kCpuctFactor(options.Get<float>(kCpuctFactorId.GetId())),
      kNoise(options.Get<bool>(kNoiseId.GetId())),
      kSmartPruningFactor(options.Get<float>(kSmartPruningFactorId.GetId())),
      kFpuAbsolute(options.Get<std::string>(kFpuStrategyId.GetId()) ==
                   "absolute"),
      kFpuReduction(options.Get<float>(kFpuReductionId.GetId())),
      kFpuValue(options.Get<float>(kFpuValueId.GetId())),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthId.GetId())),
      kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempId.GetId())),
      kMaxCollisionEvents(options.Get<int>(kMaxCollisionEventsId.GetId())),
      kMaxCollisionVisits(options.Get<int>(kMaxCollisionVisitsId.GetId())),
      kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalId.GetId())),
      kHistoryFill(
          EncodeHistoryFill(options.Get<std::string>(kHistoryFillId.GetId()))),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeId.GetId())){
}

}  // namespace lczero
