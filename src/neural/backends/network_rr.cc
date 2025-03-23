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

#include <condition_variable>
#include <queue>
#include <thread>

#include "neural/factory.h"
#include "utils/exception.h"

namespace lczero {
namespace {

class RoundRobinNetwork : public Network {
 public:
  RoundRobinNetwork(const std::optional<WeightsFile>& weights,
                    const OptionsDict& options) {
    const auto parents = options.ListSubdicts();
    if (parents.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = NetworkFactory::Get()->GetBackendsList();
      AddBackend(backends[0], weights, options);
    }

    for (const auto& name : parents) {
      AddBackend(name, weights, options.GetSubdict(name));
    }
  }

  void AddBackend(const std::string& name,
                  const std::optional<WeightsFile>& weights,
                  const OptionsDict& opts) {
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    networks_.emplace_back(
        NetworkFactory::Get()->Create(backend, weights, opts));

    min_batch_size_ =
        std::min(min_batch_size_, networks_.back()->GetMiniBatchSize());
    is_cpu_ &= networks_.back()->IsCpu();

    if (networks_.size() == 1) {
      capabilities_ = networks_.back()->GetCapabilities();
    } else {
      capabilities_.Merge(networks_.back()->GetCapabilities());
    }
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    const long long val = ++counter_;
    return networks_[val % networks_.size()]->NewComputation();
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  int GetMiniBatchSize() const override { return min_batch_size_; }

  int GetThreads() const override { return networks_.size(); }

  bool IsCpu() const override { return is_cpu_; }

  ~RoundRobinNetwork() {}

 private:
  std::vector<std::unique_ptr<Network>> networks_;
  std::atomic<long long> counter_;
  NetworkCapabilities capabilities_;
  int min_batch_size_ = std::numeric_limits<int>::max();
  bool is_cpu_ = true;
};

std::unique_ptr<Network> MakeRoundRobinNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<RoundRobinNetwork>(weights, options);
}

REGISTER_NETWORK("roundrobin", MakeRoundRobinNetwork, -999)

}  // namespace
}  // namespace lczero
