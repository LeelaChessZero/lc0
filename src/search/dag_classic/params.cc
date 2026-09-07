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

#include "search/dag_classic/params.h"

namespace lczero {
namespace dag_classic {

const OptionId SearchParams::kUseUncertaintyWeightingId{
    {.long_flag = "use-uncertainty-weighting",
     .uci_option = "UseUncertaintyWeighting",
     .help_text =
         "Enable uncertainty weighting in MCTS. It uses logistic function to "
         "convert network uncertainty to a weight factor."}};
const OptionId SearchParams::kUncertaintyWeightingMidPointId{
    {.long_flag = "uncertainty-weighting-mid-point",
     .uci_option = "UncertaintyWeightingMidPoint",
     .help_text = "The mid point of logistic function. It gets subtracted from "
                  "network uncertainty."}};
const OptionId SearchParams::kUncertaintyWeightingMinusExponentId{
    {.long_flag = "uncertainty-weighting-minus-exponent",
     .uci_option = "UncertaintyWeightingMinusExponent",
     .help_text = "Minus exponent in the logistic function. Multiplies mid "
                  "point corrected uncertainty for exponent."}};
const OptionId SearchParams::kUncertaintyWeightingScaleId{
    {.long_flag = "uncertainty-weighting-scale",
     .uci_option = "UncertaintyWeightingScale",
     .help_text = "Numerator in the logistic function."}};
const OptionId SearchParams::kUncertaintyWeightingBaseId{
    {.long_flag = "uncertainty-weighting-base",
     .uci_option = "UncertaintyWeightingBase",
     .help_text = "Base is added to logistic function output."}};
const OptionId SearchParams::kUncertaintyWeightingCapId{
    {.long_flag = "uncertainty-weighting-cap",
     .uci_option = "UncertaintyWeightingCap",
     .help_text = "Maximum value of uncertainty weighting. It is used for "
                  "non-network evaluations like tablebase."}};

void SearchParams::Populate(OptionsParser* options) {
  BaseSearchParams::Populate(options);
  options->Add<BoolOption>(kUseUncertaintyWeightingId) = true;
  options->Add<FloatOption>(kUncertaintyWeightingMidPointId, 0.0f, 1.0f) =
      0.2086f;
  options->Add<FloatOption>(kUncertaintyWeightingMinusExponentId, 0.0f,
                            1000000.0f) = 51.34f;
  options->Add<FloatOption>(kUncertaintyWeightingScaleId, 0.0f, 10.0f) =
      0.6523f;
  options->Add<FloatOption>(kUncertaintyWeightingBaseId, 0.0f, 10.0f) = 0.5388f;
  options->Add<FloatOption>(kUncertaintyWeightingCapId, 0.0f, 10.0f) = 1.216f;
}

SearchParams::SearchParams(const OptionsDict& options)
    : BaseSearchParams(options),
      kUseUncertaintyWeighting(options.Get<bool>(kUseUncertaintyWeightingId)),
      kUncertaintyWeightingMidPoint(
          options.Get<float>(kUncertaintyWeightingMidPointId)),
      kUncertaintyWeightingMinusExponent(
          options.Get<float>(kUncertaintyWeightingMinusExponentId)),
      kUncertaintyWeightingScale(
          options.Get<float>(kUncertaintyWeightingScaleId)),
      kUncertaintyWeightingBase(
          options.Get<float>(kUncertaintyWeightingBaseId)),
      kUncertaintyWeightingCap(options.Get<float>(kUncertaintyWeightingCapId)) {
}
}  // namespace dag_classic
}  // namespace lczero
