/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2021 The LCZero Authors

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
#include "neural/network.h"
#include "utils/logging.h"

namespace lczero {

namespace {

class BlendNetwork;

class BlendComputation : public NetworkComputation {
 public:
  BlendComputation(std::unique_ptr<NetworkComputation> work_comp,
                   std::unique_ptr<NetworkComputation> policy_comp)
      : work_comp_(std::move(work_comp)),
        policy_comp_(std::move(policy_comp)) {}

  void AddInput(InputPlanes&& input) override {
    InputPlanes x = input;
    InputPlanes y = input;
    work_comp_->AddInput(std::move(x));
    policy_comp_->AddInput(std::move(y));
  }

  void ComputeBlocking() override {
    work_comp_->ComputeBlocking();
    policy_comp_->ComputeBlocking();
  }

  int GetBatchSize() const override {
    return static_cast<int>(work_comp_->GetBatchSize());
  }

  float GetQVal(int sample) const override {
    return work_comp_->GetQVal(sample);
  }

  float GetDVal(int sample) const override {
    return work_comp_->GetDVal(sample);
  }

  float GetMVal(int sample) const override {
    return work_comp_->GetMVal(sample);
  }

  float GetPVal(int sample, int move_id) const override {
    return policy_comp_->GetPVal(sample, move_id);
  }

 private:
  std::unique_ptr<NetworkComputation> work_comp_;
  std::unique_ptr<NetworkComputation> policy_comp_;
};

class BlendNetwork : public Network {
 public:
  BlendNetwork(const std::optional<WeightsFile>& weights,
               const OptionsDict& options) {
    auto backends = NetworkFactory::Get()->GetBackendsList();

    OptionsDict dict1;
    std::string backendName1 = backends[0];
    OptionsDict& backend1_dict = dict1;
    std::string networkName1 = "<default>";

    OptionsDict dict2;
    std::string backendName2 = backends[0];
    OptionsDict& backend2_dict = dict2;
    std::string networkName2 = "<default>";

    const auto parents = options.ListSubdicts();

    if (parents.size() == 0) {
      backendName1 = options.GetOrDefault<std::string>("backend", backendName1);
      networkName1 = options.GetOrDefault<std::string>("weights", networkName1);
    } else {
      backend1_dict = options.GetSubdict(parents[0]);
      backendName1 =
          backend1_dict.GetOrDefault<std::string>("backend", backendName1);
      networkName1 =
          backend1_dict.GetOrDefault<std::string>("weights", networkName1);
    }
    if (parents.size() > 1) {
      backend2_dict = options.GetSubdict(parents[1]);
      backendName2 =
          backend2_dict.GetOrDefault<std::string>("backend", backendName2);
      networkName2 =
          backend2_dict.GetOrDefault<std::string>("weights", networkName2);
    }
    if (parents.size() > 2) {
      CERR << "Warning, cannot blend more than two backends";
    }

    if (networkName1 == "<default>") {
      policy_net_ =
          NetworkFactory::Get()->Create(backendName1, weights, backend1_dict);
    } else {
      CERR << "Policy net set to " << networkName1 << ".";
      std::optional<WeightsFile> weights1;
      weights1 = LoadWeightsFromFile(networkName1);
      policy_net_ =
          NetworkFactory::Get()->Create(backendName1, weights1, backend1_dict);
    }

    if (networkName2 == "<default>") {
      work_net_ =
          NetworkFactory::Get()->Create(backendName2, weights, backend2_dict);
    } else {
      CERR << "Working net set to " << networkName2 << ".";
      std::optional<WeightsFile> weights2;
      weights2 = LoadWeightsFromFile(networkName2);
      work_net_ =
          NetworkFactory::Get()->Create(backendName2, weights2, backend2_dict);
    }
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    std::unique_ptr<NetworkComputation> work_comp = work_net_->NewComputation();
    std::unique_ptr<NetworkComputation> policy_comp =
        policy_net_->NewComputation();
    return std::make_unique<BlendComputation>(std::move(work_comp),
                                              std::move(policy_comp));
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return work_net_->GetCapabilities();
  }

 private:
  std::unique_ptr<Network> work_net_;
  std::unique_ptr<Network> policy_net_;
};

std::unique_ptr<Network> MakeBlendNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<BlendNetwork>(weights, options);
}

REGISTER_NETWORK("blend", MakeBlendNetwork, -800)

}  // namespace
}  // namespace lczero
