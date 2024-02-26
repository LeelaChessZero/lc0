/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

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

#include <algorithm>

#include "neural/loader.h"
#include "utils/commandline.h"
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
const OptionId NetworkFactory::kGpusId{
    "gpus", "Gpus", "Number of gpus to use. Can be overriden by backend-opts."};
const char* kAutoDiscover = "<autodiscover>";
const char* kEmbed = "<built in>";

NetworkFactory* NetworkFactory::Get() {
  static NetworkFactory factory;
  return &factory;
}

NetworkFactory::Register::Register(const std::string& name, FactoryFunc factory,
                                   int priority) {
  NetworkFactory::Get()->RegisterNetwork(name, factory, priority);
}

void NetworkFactory::PopulateOptions(OptionsParser* options) {
#if defined(EMBED)
  options->Add<StringOption>(NetworkFactory::kWeightsId) = kEmbed;
#else
  options->Add<StringOption>(NetworkFactory::kWeightsId) = kAutoDiscover;
#endif
  const auto backends = NetworkFactory::Get()->GetBackendsList();
  options->Add<ChoiceOption>(NetworkFactory::kBackendId, backends) =
      backends.empty() ? "<none>" : backends[0];
  options->Add<StringOption>(NetworkFactory::kBackendOptionsId);
#if !defined(NO_GPUS_OPT)
  options->Add<IntOption>(NetworkFactory::kGpusId, 0, 8) = 1;
#endif
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

std::unique_ptr<Network> NetworkFactory::Create(
    const std::string& network, const std::optional<WeightsFile>& weights,
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
    : weights_path(options.Get<std::string>(kWeightsId)),
      backend(options.Get<std::string>(kBackendId)),
      backend_options(options.Get<std::string>(kBackendOptionsId)) {
#if !defined(NO_GPUS_OPT)
  gpus = options.Get<int>(kGpusId);
#endif
}

bool NetworkFactory::BackendConfiguration::operator==(
    const BackendConfiguration& other) const {
  return (weights_path == other.weights_path && backend == other.backend &&
          backend_options == other.backend_options && gpus == other.gpus);
}

std::unique_ptr<Network> NetworkFactory::LoadNetwork(
    const OptionsDict& options) {
  std::string net_path = options.Get<std::string>(kWeightsId);
  std::string backend = options.Get<std::string>(kBackendId);
  std::string backend_options = options.Get<std::string>(kBackendOptionsId);

  if (net_path == kAutoDiscover) {
    net_path = DiscoverWeightsFile();
  } else if (net_path == kEmbed) {
    net_path = CommandLine::BinaryName();
  } else {
    CERR << "Loading weights file from: " << net_path;
  }
  std::optional<WeightsFile> weights;
  if (!net_path.empty()) {
    weights = LoadWeightsFromFile(net_path);
  }

  if (backend_options.empty()) {
    if (backend == "check") {
      throw Exception("The check backend needs backend options");
    }
#if !defined(NO_GPUS_OPT)
    int gpus = options.Get<int>(kGpusId);
    if (gpus == 1) {
      backend_options = "gpu=0";
    } else if (gpus > 1) {
      std::string gpu_backend = backend;
      if (backend == "multiplexing" || backend == "demux" ||
          backend == "roundrobin") {
        gpu_backend = NetworkFactory::Get()->GetBackendsList()[0];
      } else {
        backend = "multiplexing";
      }
      for (int i = 0; i < gpus; i++) {
        backend_options += "(backend=" + gpu_backend + ",";
        backend_options += "gpu=" + std::to_string(i) + "),";
      }
      backend_options.pop_back();
    }
#endif
  }

  OptionsDict network_options(&options);
  network_options.AddSubdictFromString(backend_options);

  auto ptr = NetworkFactory::Get()->Create(backend, weights, network_options);
  network_options.CheckAllOptionsRead(backend);
  return ptr;
}

}  // namespace lczero
