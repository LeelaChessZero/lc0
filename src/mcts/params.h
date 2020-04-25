/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include "neural/encoder.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

class SearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;

  // Populates UciOptions with search parameters.
  static void Populate(OptionsParser* options);

  // Parameter getters.
  int GetMiniBatchSize() const { return kMiniBatchSize; }
  int GetMaxPrefetchBatch() const {
    return options_.Get<int>(kMaxPrefetchBatchId);
  }
  bool GetLogitQ() const { return kLogitQ; }
  float GetPolicyDecayShift() const { return kPolicyDecayShift; }
  float GetPolicyDecaySlope() const { return kPolicyDecaySlope; }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctBase(bool at_root) const {
    return at_root ? kCpuctBaseAtRoot : kCpuctBase;
  }
  float GetCpuctFactor(bool at_root) const {
    return at_root ? kCpuctFactorAtRoot : kCpuctFactor;
  }
  float GetTemperature() const { return options_.Get<float>(kTemperatureId); }
  float GetTemperatureVisitOffset() const {
    return options_.Get<float>(kTemperatureVisitOffsetId);
  }
  int GetTempDecayMoves() const { return options_.Get<int>(kTempDecayMovesId); }
  int GetTempDecayDelayMoves() const {
    return options_.Get<int>(kTempDecayDelayMovesId);
  }
  int GetTemperatureCutoffMove() const {
    return options_.Get<int>(kTemperatureCutoffMoveId);
  }
  float GetTemperatureEndgame() const {
    return options_.Get<float>(kTemperatureEndgameId);
  }
  float GetTemperatureWinpctCutoff() const {
    return options_.Get<float>(kTemperatureWinpctCutoffId);
  }

  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetNoiseAlpha() const { return kNoiseAlpha; }
  bool GetVerboseStats() const { return options_.Get<bool>(kVerboseStatsId); }
  bool GetLogLiveStats() const { return options_.Get<bool>(kLogLiveStatsId); }
  bool GetFpuAbsolute(bool at_root) const {
    return at_root ? kFpuAbsoluteAtRoot : kFpuAbsolute;
  }
  float GetFpuValue(bool at_root) const {
    return at_root ? kFpuValueAtRoot : kFpuValue;
  }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicySoftmaxTemp() const { return kPolicySoftmaxTemp; }
  float GetShortSightedness() const { return kShortSightedness; }
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisitsId() const { return kMaxCollisionVisits; }
  bool GetOutOfOrderEval() const { return kOutOfOrderEval; }
  bool GetStickyEndgames() const { return kStickyEndgames; }
  bool GetSyzygyFastPlay() const { return kSyzygyFastPlay; }
  int GetMultiPv() const { return options_.Get<int>(kMultiPvId); }
  bool GetPerPvCounters() const { return options_.Get<bool>(kPerPvCountersId); }
  std::string GetScoreType() const {
    return options_.Get<std::string>(kScoreTypeId);
  }
  FillEmptyHistory GetHistoryFill() const { return kHistoryFill; }
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftSlope() const { return kMovesLeftSlope; }
  float GetMovesLeftConstantFactor() const { return kMovesLeftConstantFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const {
    return kMovesLeftQuadraticFactor;
  }
  bool GetDisplayCacheUsage() const { return kDisplayCacheUsage; }
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetSidetomoveDrawScore() const { return kDrawScoreSidetomove; }
  float GetOpponentDrawScore() const { return kDrawScoreOpponent; }
  float GetWhiteDrawDelta() const { return kDrawScoreWhite; }
  float GetBlackDrawDelta() const { return kDrawScoreBlack; }

  int GetMaxOutOfOrderEvals() const { return kMaxOutOfOrderEvals; }

  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId;
  static const OptionId kLogitQId;
  static const OptionId kPolicyDecayShiftId;
  static const OptionId kPolicyDecaySlopeId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctBaseAtRootId;
  static const OptionId kCpuctFactorId;
  static const OptionId kCpuctFactorAtRootId;
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kTemperatureId;
  static const OptionId kTempDecayMovesId;
  static const OptionId kTempDecayDelayMovesId;
  static const OptionId kTemperatureCutoffMoveId;
  static const OptionId kTemperatureEndgameId;
  static const OptionId kTemperatureWinpctCutoffId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kNoiseEpsilonId;
  static const OptionId kNoiseAlphaId;
  static const OptionId kVerboseStatsId;
  static const OptionId kLogLiveStatsId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuValueId;
  static const OptionId kFpuStrategyAtRootId;
  static const OptionId kFpuValueAtRootId;
  static const OptionId kCacheHistoryLengthId;
  static const OptionId kPolicySoftmaxTempId;
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kOutOfOrderEvalId;
  static const OptionId kStickyEndgamesId;
  static const OptionId kSyzygyFastPlayId;
  static const OptionId kMultiPvId;
  static const OptionId kPerPvCountersId;
  static const OptionId kScoreTypeId;
  static const OptionId kHistoryFillId;
  static const OptionId kMovesLeftMaxEffectId;
  static const OptionId kMovesLeftThresholdId;
  static const OptionId kMovesLeftConstantFactorId;
  static const OptionId kMovesLeftScaledFactorId;
  static const OptionId kMovesLeftQuadraticFactorId;
  static const OptionId kMovesLeftSlopeId;
  static const OptionId kShortSightednessId;
  static const OptionId kDisplayCacheUsageId;
  static const OptionId kMaxConcurrentSearchersId;
  static const OptionId kDrawScoreSidetomoveId;
  static const OptionId kDrawScoreOpponentId;
  static const OptionId kDrawScoreWhiteId;
  static const OptionId kDrawScoreBlackId;
  static const OptionId kMaxOutOfOrderEvalsId;

 private:
  const OptionsDict& options_;
  // Cached parameter values. Values have to be cached if either:
  // 1. Parameter is accessed often and has to be cached for performance
  // reasons.
  // 2. Parameter has to stay the say during the search.
  // TODO(crem) Some of those parameters can be converted to be dynamic after
  //            trivial search optimizations.
  const bool kLogitQ;
  const float kPolicyDecayShift;
  const float kPolicyDecaySlope;
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctBase;
  const float kCpuctBaseAtRoot;
  const float kCpuctFactor;
  const float kCpuctFactorAtRoot;
  const float kNoiseEpsilon;
  const float kNoiseAlpha;
  const bool kFpuAbsolute;
  const float kFpuValue;
  const bool kFpuAbsoluteAtRoot;
  const float kFpuValueAtRoot;
  const int kCacheHistoryLength;
  const float kPolicySoftmaxTemp;
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kOutOfOrderEval;
  const bool kStickyEndgames;
  const bool kSyzygyFastPlay;
  const FillEmptyHistory kHistoryFill;
  const int kMiniBatchSize;
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftSlope;
  const float kMovesLeftConstantFactor;
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const float kShortSightedness;
  const bool kDisplayCacheUsage;
  const int kMaxConcurrentSearchers;
  const float kDrawScoreSidetomove;
  const float kDrawScoreOpponent;
  const float kDrawScoreWhite;
  const float kDrawScoreBlack;
  const int kMaxOutOfOrderEvals;
};

}  // namespace lczero
