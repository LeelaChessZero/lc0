/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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
#include "neural/network_legacy.h"
#include "utils/bititer.h"
#include "utils/optionsdict.h"
#include "utils/transpose.h"

namespace lczero {

namespace {

class ExternalNetworkComputation;

class ExternalNetwork : public Network {
 public:
  ExternalNetwork(const WeightsFile& file, const OptionsDict& options) {
    // Serialize file to bytes.
    // Make large memory mapped file big enough to contain plus some extra and
    // also at least max batch size times size of inputs, wdl and policies.
    // write weights bytes at small offset.
    // Write 'weights ready' flag.
    // Spin Wait for 'dest ready' flag.
  }

  std::unique_ptr<NetworkComputation> NewComputation() override;

  void Compute(const std::vector<InputPlanes>& raw_input,
               std::vector<std::vector<float>>* wdls,
               std::vector<std::vector<float>>* policies) const {
    // Take lock.
    // Write raw_input at small offset with length.
    // Write 'input ready' flag value.
    // Otherside clears input ready flag.
    // Spin Wait for 'output ready' flag value.
    // Clear 'output ready' flag. (Maybe atomic_compare_swap in spin wait?)
    // Copy output in wdls/policies.
  }

  const NetworkCapabilities& GetCapabilities() const override {
    // TODO: use same capabilities as weights file implies.
    static NetworkCapabilities capabilities;
    return capabilities;
  }

 private:
};

class ExternalNetworkComputation : public NetworkComputation {
 public:
  ExternalNetworkComputation(const ExternalNetwork* network)
      : network_(network) {}
  void AddInput(InputPlanes&& input) override {
    raw_input_.emplace_back(input);
  }
  void ComputeBlocking() override {
    network_->Compute(raw_input_, &wdls_, &policies_);
  }

  int GetBatchSize() const override { return raw_input_.size(); }
  float GetQVal(int sample) const override {
    return wdls_[sample][0] - wdls_[sample][2];
  }
  float GetDVal(int sample) const override { return wdls_[sample][1]; }
  float GetPVal(int sample, int move_id) const override {
    return policies_[sample][move_id];
  }

 private:
  std::vector<InputPlanes> raw_input_;
  std::vector<std::vector<float>> wdls_;
  std::vector<std::vector<float>> policies_;
  const ExternalNetwork* network_;
};

std::unique_ptr<NetworkComputation> ExternalNetwork::NewComputation() {
  return std::make_unique<ExternalNetworkComputation>(this);
}

std::unique_ptr<Network> MakeExternalNetwork(const WeightsFile& weights,
                                             const OptionsDict& options) {
  return std::make_unique<ExternalNetwork>(weights, options);
}

REGISTER_NETWORK("external", MakeExternalNetwork, -999)

}  // namespace
}  // namespace lczero
