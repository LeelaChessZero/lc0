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

#pragma once

#include <functional>
#include <optional>
#include <string>

#include "neural/loader.h"
#include "neural/network.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

class NetworkFactory {
 public:
  using FactoryFunc = std::function<std::unique_ptr<Network>(
      const std::optional<WeightsFile>&, const OptionsDict&)>;

  static NetworkFactory* Get();

  // Registers network so it can be created by name.
  // @name -- name
  // @options -- options to pass to the network
  // @priority -- how high should be the network in the list. The network with
  //              the highest priority is the default.
  class Register {
   public:
    Register(const std::string& name, FactoryFunc factory, int priority = 0);
  };

  // Add the network/backend parameters to the options dictionary.
  static void PopulateOptions(OptionsParser* options);

  // Returns list of backend names, sorted by priority (higher priority first).
  std::vector<std::string> GetBackendsList() const;

  // Creates a backend given name and config.
  std::unique_ptr<Network> Create(const std::string& network,
                                  const std::optional<WeightsFile>&,
                                  const OptionsDict& options);

  // Helper function to load the network from the options. Returns nullptr
  // if no network options changed since the previous call.
  static std::unique_ptr<Network> LoadNetwork(const OptionsDict& options);

  // Parameter IDs.
  static const OptionId kWeightsId;
  static const OptionId kBackendId;
  static const OptionId kBackendOptionsId;

  struct BackendConfiguration {
    BackendConfiguration() = default;
    BackendConfiguration(const OptionsDict& options);
    std::string weights_path;
    std::string backend;
    std::string backend_options;
    bool operator==(const BackendConfiguration& other) const;
    bool operator!=(const BackendConfiguration& other) const {
      return !operator==(other);
    }
    bool operator<(const BackendConfiguration& other) const {
      return std::tie(weights_path, backend, backend_options) <
             std::tie(other.weights_path, other.backend, other.backend_options);
    }
  };

 private:
  void RegisterNetwork(const std::string& name, FactoryFunc factory,
                       int priority);

  NetworkFactory() {}

  struct Factory {
    Factory(const std::string& name, FactoryFunc factory, int priority)
        : name(name), factory(factory), priority(priority) {}

    bool operator<(const Factory& other) const {
      if (priority != other.priority) return priority > other.priority;
      return name < other.name;
    }

    std::string name;
    FactoryFunc factory;
    int priority;
  };

  std::vector<Factory> factories_;
  friend class Register;
};

#define REGISTER_NETWORK_WITH_COUNTER2(name, func, priority, counter) \
  namespace {                                                         \
  static NetworkFactory::Register regH38fhs##counter(                 \
      name,                                                           \
      [](const std::optional<WeightsFile>& w, const OptionsDict& o) { \
        return func(w, o);                                            \
      },                                                              \
      priority);                                                      \
  }
#define REGISTER_NETWORK_WITH_COUNTER(name, func, priority, counter) \
  REGISTER_NETWORK_WITH_COUNTER2(name, func, priority, counter)

// Registers a Network.
// Constructor of a network class must have parameters:
// (const Weights& w, const OptionsDict& o)
// @name -- name under which the backend will be known in configs.
// @func -- Factory function for a backend.
//          std::unique_ptr<Network>(const WeightsFile&, const OptionsDict&)
// @priority -- numeric priority of a backend. Higher is higher, highest number
// is the default backend.
#define REGISTER_NETWORK(name, func, priority) \
  REGISTER_NETWORK_WITH_COUNTER(name, func, priority, __LINE__)
}  // namespace lczero
