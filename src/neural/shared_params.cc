/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include "neural/shared_params.h"

#include "neural/factory.h"

namespace lczero {
const OptionId SharedBackendParams::kPolicySoftmaxTemp{
    "policy-softmax-temp", "PolicyTemperature",
    "Policy softmax temperature. Higher values make priors of move candidates "
    "closer to each other, widening the search."};
const OptionId SharedBackendParams::kHistoryFill{
    "history-fill-new", "HistoryFill",
    "Neural network uses 7 previous board positions in addition to the current "
    "one. During the first moves of the game such historical positions don't "
    "exist, but they can be synthesized. This parameter defines when to "
    "synthesize them (always, never, or only at non-standard fen position)."};
const OptionId SharedBackendParams::kWeightsId{
    {.long_flag = "weights",
     .uci_option = "WeightsFile",
     .help_text =
         "Path from which to load network weights.\nSetting it to "
         "<autodiscover> makes it search in ./ and ./weights/ subdirectories "
         "for the latest (by file date) file which looks like weights.",
     .short_flag = 'w',
     .visibility = OptionId::kAlwaysVisible}};
const OptionId SharedBackendParams::kBackendId{{
    .long_flag = "backend",
    .uci_option = "Backend",
    .help_text = "Neural network computational backend to use.",
    .short_flag = 'b',
}};
const OptionId SharedBackendParams::kBackendOptionsId{
    "backend-opts", "BackendOptions",
    "Parameters of neural network backend. Exact parameters differ per "
    "backend.",
    'o'};
const OptionId SharedBackendParams::kNNCacheSizeId{
    "nncache", "NNCacheSize",
    "Number of positions to store in a memory cache. A large cache can speed "
    "up searching, but takes memory."};

void SharedBackendParams::Populate(OptionsParser* options) {
  options->Add<FloatOption>(kPolicySoftmaxTemp, 0.1f, 10.0f) = 1.359f;
  std::vector<std::string> history_fill_opt{"no", "fen_only", "always"};
  options->Add<ChoiceOption>(kHistoryFill, history_fill_opt) = "fen_only";

#if defined(EMBED)
  options->Add<StringOption>(SharedBackendParams::kWeightsId) = kEmbed;
#else
  options->Add<StringOption>(SharedBackendParams::kWeightsId) = kAutoDiscover;
#endif
  const auto backends = NetworkFactory::Get()->GetBackendsList();
  options->Add<ChoiceOption>(SharedBackendParams::kBackendId, backends) =
      backends.empty() ? "<none>" : backends[0];
  options->Add<StringOption>(SharedBackendParams::kBackendOptionsId);
  options->Add<IntOption>(SharedBackendParams::kNNCacheSizeId, 0, 999999999) =
      2000000;
}

}  // namespace lczero