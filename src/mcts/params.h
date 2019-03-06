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
  int GetMiniBatchSize() const {
    return kMiniBatchSize;
  }
  int GetMaxPrefetchBatch() const {
    return options_.Get<int>(kMaxPrefetchBatchId.GetId());
  }
  float GetCpuct() const { return kCpuct; }
  float GetCpuctBase() const { return kCpuctBase; }
  float GetCpuctFactor() const { return kCpuctFactor; }
  float GetTemperature() const {
    return options_.Get<float>(kTemperatureId.GetId());
  }
  float GetTemperatureVisitOffset() const {
    return options_.Get<float>(kTemperatureVisitOffsetId.GetId());
  }
  int GetTempDecayMoves() const {
    return options_.Get<int>(kTempDecayMovesId.GetId());
  }
  int GetTemperatureCutoffMove() const {
    return options_.Get<int>(kTemperatureCutoffMoveId.GetId());
  }
  float GetTemperatureEndgame() const {
    return options_.Get<float>(kTemperatureEndgameId.GetId());
  }
  float GetTemperatureWinpctCutoff() const {
    return options_.Get<float>(kTemperatureWinpctCutoffId.GetId());
  }

  bool GetNoise() const { return kNoise; }
  bool GetVerboseStats() const {
    return options_.Get<bool>(kVerboseStatsId.GetId());
  }
  bool GetLogLiveStats() const {
    return options_.Get<bool>(kLogLiveStatsId.GetId());
  }
  float GetSmartPruningFactor() const { return kSmartPruningFactor; }
  bool GetFpuAbsolute() const { return kFpuAbsolute; }
  float GetFpuReduction() const { return kFpuReduction; }
  float GetFpuValue() const { return kFpuValue; }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicySoftmaxTemp() const { return kPolicySoftmaxTemp; }
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisitsId() const { return kMaxCollisionVisits; }
  bool GetOutOfOrderEval() const { return kOutOfOrderEval; }
  bool GetSyzygyFastPlay() const { return kSyzygyFastPlay; }
  int GetMultiPv() const { return options_.Get<int>(kMultiPvId.GetId()); }
  std::string GetScoreType() const {
    return options_.Get<std::string>(kScoreTypeId.GetId());
  }
  FillEmptyHistory GetHistoryFill() const { return kHistoryFill; }
  int GetKLDGainAverageInterval() const {
    return options_.Get<int>(kKLDGainAverageInterval.GetId());
  }
  float GetMinimumKLDGainPerNode() const {
    return options_.Get<float>(kMinimumKLDGainPerNode.GetId());
  }

  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctFactorId;
  static const OptionId kTemperatureId;
  static const OptionId kTempDecayMovesId;
  static const OptionId kTemperatureCutoffMoveId;
  static const OptionId kTemperatureEndgameId;
  static const OptionId kTemperatureWinpctCutoffId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kNoiseId;
  static const OptionId kVerboseStatsId;
  static const OptionId kLogLiveStatsId;
  static const OptionId kSmartPruningFactorId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuReductionId;
  static const OptionId kFpuValueId;
  static const OptionId kCacheHistoryLengthId;
  static const OptionId kPolicySoftmaxTempId;
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kOutOfOrderEvalId;
  static const OptionId kSyzygyFastPlayId;
  static const OptionId kMultiPvId;
  static const OptionId kScoreTypeId;
  static const OptionId kHistoryFillId;
  static const OptionId kMinimumKLDGainPerNode;
  static const OptionId kKLDGainAverageInterval;

 private:
  const OptionsDict& options_;
  // Cached parameter values. Values have to be cached if either:
  // 1. Parameter is accessed often and has to be cached for performance
  // reasons.
  // 2. Parameter has to stay the say during the search.
  // TODO(crem) Some of those parameters can be converted to be dynamic after
  //            trivial search optimiations.
  const float kCpuct;
  const float kCpuctBase;
  const float kCpuctFactor;
  const bool kNoise;
  const float kSmartPruningFactor;
  const bool kFpuAbsolute;
  const float kFpuReduction;
  const float kFpuValue;
  const int kCacheHistoryLength;
  const float kPolicySoftmaxTemp;
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kOutOfOrderEval;
  const bool kSyzygyFastPlay;
  const FillEmptyHistory kHistoryFill;
  const int kMiniBatchSize;
};

}  // namespace lczero
