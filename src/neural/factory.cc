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

#include "neural/factory.h"
#include "neural/loader.h"

#include <algorithm>
#include "utils/logging.h"

namespace lczero {

const OptionId NetworkFactory::kWeightsId{
    "weights", "WeightsFile",
    "Path from which to load network weights.\nSetting it to <autodiscover> "
    "makes it search in ./ and ./weights/ subdirectories for the latest (by "
    "file date) file which looks like weights.",
    'w'};
const OptionId NetworkFactory::kBackendId{
    "backend", "Backend", "Neural network computational backend to use.", 'b'};
const OptionId NetworkFactory::kBackendOptionsId{
    "backend-opts", "BackendOptions",
    "Parameters of neural network backend. "
    "Exact parameters differ per backend.",
    'o'};
const OptionId NetworkFactory::kSecondWeightsId{
    "secondweights", "SecondWeightsFile",
    "Path from which to load network weights for the second NN.",
    's'};
const OptionId NetworkFactory::kSecondWeightsSwitchAtId{
    "switchat", "SwitchAt",
    "Switch to the second NN when this number of pieces are left on the board. "
    "Switch will take place _after_ lc0 has made a move, when it is the "
    "opponents time to think."};
const OptionId NetworkFactory::kSecondWeightsCpuctId{
    "SWcpuct", "SWCPuct",
    "cpuct_init constant from \"UCT search\" algorithm. Higher values promote "
    "more exploration/wider search, lower values promote more "
    "confidence/deeper search. "
    "This value is used for the Secondary NN."};
const OptionId NetworkFactory::kSecondWeightsFpuReductionId{
    "SWfpu-Reduction", "SWFpuReduction",
      "First Play Urgency reduction (used when FPU strategy is 'reduction'). Normally "
      "when a move has no visits, it's eval is assumed to be equal to parent's eval. "
      "With non-zero FPU reduction, eval of unvisited move is decreased by that value, "
      "discouraging visits of unvisited moves, and saving those visits for (hopefully) "
      "more promising moves. "
    "This value is used for the Secondary NN."};      
const OptionId NetworkFactory::kSecondWeightsPolicySoftmaxTempId{
    "SWpolicy-softmax-temp", "SWPolicyTemperature",
    "Policy softmax temperature. Higher values make priors of move candidates "
    "closer to each other, widening the search. "
    "This value is used for the Secondary NN."};          
  
const char* kAutoDiscover = "<autodiscover>";

NetworkFactory* NetworkFactory::Get() {
  static NetworkFactory factory;
  return &factory;
}

NetworkFactory::Register::Register(const std::string& name, FactoryFunc factory,
                                   int priority) {
  NetworkFactory::Get()->RegisterNetwork(name, factory, priority);
}

void NetworkFactory::PopulateOptions(OptionsParser* options) {
  options->Add<StringOption>(NetworkFactory::kWeightsId) = kAutoDiscover;
  const auto backends = NetworkFactory::Get()->GetBackendsList();
  options->Add<ChoiceOption>(NetworkFactory::kBackendId, backends) =
      backends.empty() ? "<none>" : backends[0];
  options->Add<StringOption>(NetworkFactory::kBackendOptionsId);
  options->Add<StringOption>(NetworkFactory::kSecondWeightsId);
  options->Add<IntOption>(NetworkFactory::kSecondWeightsSwitchAtId, 5, 16) = 12;
  options->Add<FloatOption>(kSecondWeightsCpuctId, 0.0f, 100.0f) = 3.5f;
  options->Add<FloatOption>(kSecondWeightsFpuReductionId, -100.0f, 100.0f) = 1.02f;
  options->Add<FloatOption>(kSecondWeightsPolicySoftmaxTempId, 0.1f, 10.0f) = 1.7f;
}

void NetworkFactory::RegisterNetwork(const std::string& name,
                                     FactoryFunc factory, int priority) {
  factories_.emplace_back(name, factory, priority);
  std::sort(factories_.begin(), factories_.end());
}

std::vector<std::string> NetworkFactory::GetBackendsList() const {
  std::vector<std::string> result;
  for (const auto& x : factories_) result.emplace_back(x.name);
  return result;
}

std::unique_ptr<Network> NetworkFactory::Create(const std::string& network,
                                                const WeightsFile& weights,
                                                const OptionsDict& options) {
  CERR << "Creating backend [" << network << "]...";
  for (const auto& factory : factories_) {
    if (factory.name == network) {
      return factory.factory(weights, options);
    }
  }
  throw Exception("Unknown backend: " + network);
}

NetworkFactory::BackendConfiguration::BackendConfiguration(
    const OptionsDict& options)
    : weights_path(options.Get<std::string>(kWeightsId.GetId())),
      backend(options.Get<std::string>(kBackendId.GetId())),
      backend_options(options.Get<std::string>(kBackendOptionsId.GetId())) {}

bool NetworkFactory::BackendConfiguration::operator==(
    const BackendConfiguration& other) const {
  return (weights_path == other.weights_path && backend == other.backend &&
          backend_options == other.backend_options);
}

std::unique_ptr<Network> NetworkFactory::LoadNetwork(
    const OptionsDict& options) {
  std::string net_path = options.Get<std::string>(kWeightsId.GetId());
  std::string backend = options.Get<std::string>(kBackendId.GetId());
  std::string backend_options =
      options.Get<std::string>(kBackendOptionsId.GetId());

  if (net_path == kAutoDiscover) {
    net_path = DiscoverWeightsFile();
  } else {
    CERR << "Loading weights file from: " << net_path;
  }
  WeightsFile weights = LoadWeightsFromFile(net_path);

  OptionsDict network_options(&options);
  network_options.AddSubdictFromString(backend_options);

  return NetworkFactory::Get()->Create(backend, weights, network_options);
}

}  // namespace lczero
