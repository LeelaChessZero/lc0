/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

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

#include "search/classic/params.h"

#include <cmath>
#include <vector>

#include "utils/random.h"

namespace lczero {
namespace contempt {

using ContemptMode = classic::ContemptMode;

// START: ADDED FOR DYNAMIC HYBRID RATIO
enum class HybridRatioMode {
  STATIC,
  MANUAL_SCHEDULE,
  LINEAR,
  LOGARITHMIC,
  POWER,
  ROOT,
  SIGMOID,
  EXPONENTIAL,
  STEP_DECAY,
  INVERSE_SIGMOID,
  STEPS,
  PLATEAU,
  GAUSSIAN_PEAK,
  DOUBLE_PEAK,
  SAWTOOTH_WAVE,
  OSCILLATING,
  HEARTBEAT,
  MULTI_TIMESCALE,
  THERMAL_ANNEALING,
  ASYMPTOTIC_APPROACH,
  CHAOTIC
  // Fibonacci and Golden Ratio are complex and stateful, omitted for simplicity
  // unless a stateful evaluation mechanism is added.
};
// END: ADDED FOR DYNAMIC HYBRID RATIO

class SearchParams : public classic::SearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;

  // Populates UciOptions with search parameters.
  static void Populate(OptionsParser* options);

  // START: ADDED FOR DYNAMIC HYBRID RATIO
  // The main function to calculate the ratio based on the selected mode.
  float GetDynamicHybridRatio(int node_count) const;
  // END: ADDED FOR DYNAMIC HYBRID RATIO

  // Parameter getters.
  int GetScLimit() const { return options_.Get<int>(kScLimitId); }
  float GetHybridSamplingRatio() const { return options_.Get<float>(kHybridSamplingRatioId); }
  HybridRatioMode GetHybridRatioMode() const { return kHybridRatioMode; }
  const std::vector<std::pair<int, float>>& GetHybridRatioSchedule() const { return kHybridRatioSchedule; }
  float GetHybridMinRatio() const { return options_.Get<float>(kHybridMinRatioId); }
  float GetHybridMaxRatio() const { return options_.Get<float>(kHybridMaxRatioId); }
  int GetHybridScalingFactor() const { return options_.Get<int>(kHybridScalingFactorId); }
  float GetHybridShapeParam1() const { return options_.Get<float>(kHybridShapeParam1Id); }
  float GetHybridShapeParam2() const { return options_.Get<float>(kHybridShapeParam2Id); }
  bool GetContemptModeTBEnable() const { return kContemptModeTBEnable; }

  // Search parameter IDs.
  static const OptionId kScLimitId;
  static const OptionId kContemptModeTBEnableId;
  static const OptionId kHybridSamplingRatioId;
  // START: ADDED FOR DYNAMIC HYBRID RATIO
  static const OptionId kHybridRatioModeId;
  static const OptionId kHybridRatioScheduleId;
  static const OptionId kHybridMinRatioId;
  static const OptionId kHybridMaxRatioId;
  static const OptionId kHybridScalingFactorId;
  static const OptionId kHybridShapeParam1Id;
  static const OptionId kHybridShapeParam2Id;
  // END: ADDED FOR DYNAMIC HYBRID RATIO

  const bool kContemptModeTBEnable;
  // START: ADDED FOR DYNAMIC HYBRID RATIO
  const HybridRatioMode kHybridRatioMode;
  const std::vector<std::pair<int, float>> kHybridRatioSchedule;
  mutable float chaotic_state_{0.5f}; // Mutable for chaotic function state
  // END: ADDED FOR DYNAMIC HYBRID RATIO
};

}  // namespace contempt
}  // namespace lczero
