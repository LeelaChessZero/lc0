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

#include "neural/factory.h"
#include "neural/network.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <thread>

#include "utils/exception.h"
#include "utils/random.h"

namespace lczero {

namespace {

static constexpr int NUM_OUTPUT_POLICY = 1858;

class CheckNetwork;

class CheckComputation : public NetworkComputation {
 public:
  CheckComputation(std::unique_ptr<NetworkComputation> refComp,
                   std::unique_ptr<NetworkComputation> checkComp)
      : refComp_(std::move(refComp)), checkComp_(std::move(checkComp)) {}

  void AddInput(InputPlanes&& input) override {
    InputPlanes x = input;
    InputPlanes y = input;
    refComp_->AddInput(std::move(x));
    checkComp_->AddInput(std::move(y));
  }

  void ComputeBlocking() override {
    refComp_->ComputeBlocking();
    checkComp_->ComputeBlocking();
    Check();
  }

  int GetBatchSize() const override { return refComp_->GetBatchSize(); }

  float GetQVal(int sample) const override { return refComp_->GetQVal(sample); }

  float GetPVal(int sample, int move_id) const override {
    return refComp_->GetPVal(sample, move_id);
  }

 private:
  void Check() {
    bool valueAlmostEqual = true;
    int size = GetBatchSize();
    for (int i = 0; i < size && valueAlmostEqual; i++) {
      float v1 = refComp_->GetQVal(i);
      float v2 = checkComp_->GetQVal(i);
      valueAlmostEqual &= IsAlmostEqual(v1, v2);
    }

    bool policyAlmostEqual = true;
    for (int i = 0; i < size && policyAlmostEqual; i++) {
      for (int j = 0; j < NUM_OUTPUT_POLICY; j++) {
        float v1 = refComp_->GetPVal(i, j);
        float v2 = checkComp_->GetPVal(i, j);
        policyAlmostEqual &= IsAlmostEqual(v1, v2);
      }
    }

    // We could also print on standard output using the uci info string
    // example:
    //
    // info string Check passed
    // info string ***** error chech failed
    //

    if (valueAlmostEqual && policyAlmostEqual) {
      std::cerr << "Check passed for a batch of " << size << std::endl;
      return;
    }

    if (!valueAlmostEqual && !policyAlmostEqual) {
      std::cerr << "*** ERROR check failed for a batch of " << size
                << " both value and policy incorrect" << std::endl;
      return;
    }

    if (!valueAlmostEqual) {
      std::cerr << "*** ERROR check failed for a batch of " << size
                << " value incorrect (but policy ok)" << std::endl;
      return;
    }

    std::cerr << "*** ERROR check failed for a batch of " << size
              << " policy incorrect (but value ok)" << std::endl;
  }

  static bool IsAlmostEqual(float a, float b) {
    constexpr float ABSOLUTE_TOLERANCE = 1e-6;
    constexpr float RELATIVE_TOLERANCE = 1e-2;

    return std::abs(a - b) <=
           std::max(RELATIVE_TOLERANCE * std::max(std::abs(a), std::abs(b)),
                    ABSOLUTE_TOLERANCE);
  }

  std::unique_ptr<NetworkComputation> refComp_;
  std::unique_ptr<NetworkComputation> checkComp_;
};

class CheckNetwork : public Network {
 public:
  CheckNetwork(const Weights& weights, const OptionsDict& options) {
    // TODO:
    //
    // Parse the option string and create backends on demand.
    //
    // For now, only checking opencl against blas.

    OptionsDict dict1;
    refNet_ = NetworkFactory::Get()->Create("opencl", weights, dict1);

    OptionsDict dict;
    checkNet_ = NetworkFactory::Get()->Create("blas", weights, options);
  }

  static constexpr int CHECK_PROBABILITY = 20;

  std::unique_ptr<NetworkComputation> NewComputation() override {
    bool check = Random::Get().GetInt(0, CHECK_PROBABILITY) == 0;
    if (check) {
      std::unique_ptr<NetworkComputation> refComp = refNet_->NewComputation();
      std::unique_ptr<NetworkComputation> checkComp =
          checkNet_->NewComputation();
      return std::make_unique<CheckComputation>(std::move(refComp),
                                                std::move(checkComp));
    }
    return refNet_->NewComputation();
  }

 private:
  std::unique_ptr<Network> refNet_;
  std::unique_ptr<Network> checkNet_;
};

}  // namespace

REGISTER_NETWORK("check", CheckNetwork, -800)

}  // namespace lczero
