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

#include <algorithm>
#include <iostream>

namespace lczero {

NetworkFactory* NetworkFactory::Get() {
  static NetworkFactory factory;
  return &factory;
}

NetworkFactory::Register::Register(const std::string& name, FactoryFunc factory,
                                   int priority) {
  NetworkFactory::Get()->RegisterNetwork(name, factory, priority);
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
                                                const Weights& weights,
                                                const OptionsDict& options) {
  std::cerr << "Creating backend [" << network << "]..." << std::endl;
  for (const auto& factory : factories_) {
    if (factory.name == network) {
      return factory.factory(weights, options);
    }
  }
  throw Exception("Unknown backend: " + network);
}

}  // namespace lczero
