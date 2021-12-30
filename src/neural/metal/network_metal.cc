/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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
#include "network_metal.h"
#include "mps/MetalNetworkDelegate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/shared/policy_map.h"
#include "utils/bititer.h"
#include "utils/exception.h"

namespace lczero {
namespace metal_backend {

#define DUMP_VEC(vector) for (auto it: vector) CERR << it << ' '

MetalNetwork::MetalNetwork(const WeightsFile& file, const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().moves_left()} {
  LegacyWeights weights(file.weights());


  CERR << "Weights in first input layer: ";
  //for (auto it: file.weights().input().weights()) CERR << it << ' ';
  //CERR << file.weights().input().weights().params();

  bool conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  delegate_ = new MetalNetworkDelegate();
  //delegate_->init(weights, options);
}

//void MetalNetwork::forwardEval(InputsOutputs* io, int inputBatchSize) {
void MetalNetwork::forwardEval(int* io, int inputBatchSize) {
  CERR << "io: " << io;
  CERR << "batchsize: " << inputBatchSize;
//  delegate_->forwardEval(io, inputBatchSize);
  delegate_->logTestData("Just testing some stuff");
}

std::unique_ptr<Network> MakeMetalNetwork(const std::optional<WeightsFile>& w,
                                          const OptionsDict& options) {
  if (!w) {
    throw Exception("The Metal backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by the Metal backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION) {
    throw Exception("Policy format " +
                    std::to_string(weights.format().network_format().policy()) +
                    " is not supported by the Metal backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    std::to_string(weights.format().network_format().value()) +
                    " is not supported by the Metal backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception(
        "Movest left head format " +
        std::to_string(weights.format().network_format().moves_left()) +
        " is not supported by the Metal backend.");
  }
  return std::make_unique<MetalNetwork>(weights, options);
}

REGISTER_NETWORK("metal", MakeMetalNetwork, 95)

}  // namespace backend_metal
}  // namespace lczero
