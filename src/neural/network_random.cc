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

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <thread>
#include "neural/factory.h"
#include "utils/hashcat.h"

namespace lczero {
namespace {

class RandomNetworkComputation : public NetworkComputation {
 public:
  RandomNetworkComputation(int delay, int seed, bool uniform_mode)
      : delay_ms_(delay), seed_(seed), uniform_mode_(uniform_mode) {}

  void AddInput(InputPlanes&& input) override {
    std::uint64_t hash = seed_;
    for (const auto& plane : input) {
      hash = HashCat({hash, plane.mask});
      std::uint32_t tmp;
      std::memcpy(&tmp, &plane.value, sizeof(float));
      const std::uint64_t value_hash = tmp;
      hash = HashCat({hash, value_hash});
    }
    inputs_.push_back(hash);
  }

  void ComputeBlocking() override {
    if (delay_ms_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
    }
  }

  int GetBatchSize() const override { return inputs_.size(); }

  float GetQVal(int sample) const override {
    if (uniform_mode_) return 0.0f;
    return (int(inputs_[sample] % 200000) - 100000) / 100000.0;
  }

  float GetDVal(int sample) const override {
    if (uniform_mode_) return 0.0f;
    // Maximum D value is 1 - abs(Q) for W, D, L to be in range [0.0, 1.0].
    float q = GetQVal(sample);
    float max_d = 1.0f - std::fabs(q);
    // Hash in arbitrary constant to make D return different value from Q.
    float d = max_d * (HashCat({inputs_[sample], 1234}) % 10000) / 10000.0;
    return d;
  }

  float GetPVal(int sample, int move_id) const override {
    if (uniform_mode_) return 1.0f;
    return (HashCat({inputs_[sample], static_cast<unsigned long>(move_id)}) %
            10000) /
           10000.0;
  }

 private:
  std::vector<std::uint64_t> inputs_;
  int delay_ms_ = 0;
  int seed_ = 0;
  bool uniform_mode_ = false;
};

class RandomNetwork : public Network {
 public:
  RandomNetwork(const OptionsDict& options)
      : delay_ms_(options.GetOrDefault<int>("delay", 0)),
        seed_(options.GetOrDefault<int>("seed", 0)),
        uniform_mode_(options.GetOrDefault<bool>("uniform", false)) {}
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<RandomNetworkComputation>(delay_ms_, seed_, uniform_mode_);
  }

 private:
  int delay_ms_ = 0;
  int seed_ = 0;
  bool uniform_mode_ = false;
};
}  // namespace

std::unique_ptr<Network> MakeRandomNetwork(const WeightsFile& /*weights*/,
                                           const OptionsDict& options) {
  return std::make_unique<RandomNetwork>(options);
}

REGISTER_NETWORK("random", MakeRandomNetwork, -900)

}  // namespace lczero
