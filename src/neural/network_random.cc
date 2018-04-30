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
*/

#include <functional>
#include "neural/factory.h"
#include "utils/hashcat.h"

namespace lczero {

class RandomNetworkComputation : public NetworkComputation {
 public:
  RandomNetworkComputation() {}
  void AddInput(InputPlanes&& input) override {
    std::uint64_t hash = 0;
    for (const auto& plane : input) {
      hash = HashCat({hash, plane.mask});
    }
    inputs_.push_back(hash);
  }
  void ComputeBlocking() override { return; }

  int GetBatchSize() const override { return inputs_.size(); }
  float GetQVal(int sample) const override {
    return (int(inputs_[sample] % 200000) - 100000) / 100000.0;
  }
  float GetPVal(int sample, int move_id) const override {
    return (HashCat({inputs_[sample], static_cast<unsigned long>(move_id)}) %
            10000) /
           10000.0;
  }

 private:
  std::vector<std::uint64_t> inputs_;
};

class RandomNetwork : public Network {
 public:
  RandomNetwork(const Weights& weights, const OptionsDict& options) {}
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<RandomNetworkComputation>();
  }
};

REGISTER_NETWORK("random", RandomNetwork, -1000);

}  // namespace lczero