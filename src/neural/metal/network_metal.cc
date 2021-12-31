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
#include "mps/MetalNetworkBuilder.h"

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

void describeWeights(LegacyWeights::ConvBlock &conv, int inputs) {
  const int channelSize = conv.weights.size() / inputs / 9;

  CERR << "Size of conv weights: " << conv.weights.size() << "; Filters: " << channelSize;
  float * p = &conv.weights[0];
  //  for (int i=0; i<input.weights.size(); i++) {
  for (int i=2000; i<3000; i++) {
    CERR << "From Weight[" << i << "]: 1) " << conv.weights[i]
         << "; 2) " << *(p + i);
  }

  float * q = &conv.biases[0];
  for (int i=0; i<channelSize; i++) {
    CERR << "From Bias[" << i << "]: 1) " << conv.biases[i]
         << "; 2) " << *(q + i);
  }
}

void MetalNetworkComputation::ComputeBlocking() {
  //network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
  network_->forwardEval(GetBatchSize());
}

MetalNetwork::MetalNetwork(const WeightsFile& file, const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().moves_left()} {

  LegacyWeights weights(file.weights());

  bool conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  try {
    builder = new MetalNetworkBuilder();
    builder->init();
  } catch (...) {
    throw Exception("There was an error initializing the GPU device.");
  }

  bool isConvolutionPolicyHead = true;
  bool isWdl = true;
  bool hasMlh = true;
  const int channelSize = weights.input.weights.size() / kInputPlanes / 9;
  int kernelSize = 3;

  // Pointer to last layer in MPS NN graph.
  void * layer;

  //describeWeights(weights.input, kInputPlanes);

  // 1. Input layer
  layer = builder->makeConvolutionBlock(nullptr, kInputPlanes, channelSize, kernelSize,
                                        &weights.input.weights[0],
                                        &weights.input.biases[0],
                                        true, "input/conv");

  // 2. Residual blocks
  for (int i = 0; i < weights.residual.size(); i++) {
    //describeWeights(weights.residual[i].conv1, channelSize);
    //describeWeights(weights.residual[i].conv2, channelSize);
    layer = builder->makeResidualBlock(layer, channelSize, channelSize, kernelSize,
                                       &weights.residual[i].conv1.weights[0],
                                       &weights.residual[i].conv1.biases[0],
                                       &weights.residual[i].conv2.weights[0],
                                       &weights.residual[i].conv2.biases[0],
                                       &weights.residual[i].se.w1[0],
                                       &weights.residual[i].se.b1[0],
                                       &weights.residual[i].se.w2[0],
                                       &weights.residual[i].se.b2[0],
                                       weights.residual[i].has_se,
                                       true, "block_" + std::to_string(i)
                                       );
  }

  // 3. Policy head.
  void * policy;
  if (isConvolutionPolicyHead) {
    policy = builder->makeConvolutionBlock(layer, channelSize, channelSize, kernelSize,
                                           &weights.policy1.weights[0],
                                           &weights.policy1.biases[0],
                                           true, "policy/conv1");


    policy = builder->makeConvolutionBlock(policy, channelSize, 80, kernelSize,
                                           &weights.policy.weights[0],
                                           &weights.policy.biases[0],
                                           false, "policy/conv2");

    // [1858 -> HWC or CHW]
    const HWC = true;
    std::vector<int> policy_map(1858);
    for (const auto& mapping : kConvPolicyMap) {
      if (mapping == -1) continue;
      const auto index = &mapping - kConvPolicyMap;
      const auto displacement = index / 64;
      const auto square = index % 64;
      const auto row = square / 8;
      const auto col = square % 8;
      if (HWC) {
        policy_map[mapping] = ((row * 8) + col) * 80 + displacement;
      } else {
        policy_map[mapping] = ((displacement * 8) + row) * 8 + col;
      }
    }
    policy = builder->makePolicyMapLayer(policy, &policy_map);
    // @todo Policy mapping in GPU.
    /*auto mapping = MakeIntConst(scope, {1858}, policy_map);
    auto flattened_conv =
        Reshape(scope, conv_pol, Const(scope, {-1, 80 * 8 * 8}));
    policy_head = GatherV2(scope, flattened_conv, mapping, 1);

    mapping.node()->set_name("policy/mapping_table");
    flattened_conv.node()->set_name("policy/flatten");*/

  }
  else {
    const int policySize = weights.policy.biases.size();
    policy = builder->makeConvolutionBlock(layer, channelSize, policySize, 1,
                                           &weights.policy.weights[0],
                                           &weights.policy.biases[0],
                                           true, "policy/conv");

    policy = builder->makeReshapeLayer(policy, 1, 1, policySize * 8 * 8);
    // @todo check if the weights are correctly aligned.
    policy = builder->makeFullyConnectedLayer(policy, policySize * 8 * 8, 1858,
                                              &weights.ip_pol_w[0],
                                              &weights.ip_pol_b[0],
                                              nullptr, "policy/fc");

  }

  // 4. Value head.
  void * value;
  value = builder->makeConvolutionBlock(layer, channelSize, 32,
                                        &weights.value.weights[0],
                                        &weights.value.biases[0],
                                        true, "value/conv");
  value = builder->makeReshapeLayer(value, 1, 1, 32 * 8 * 8);
  value = builder->makeFullyConnectedLayer(value, 32 * 8 * 8, 128,
                                           &weights.ip1_val_w[0],
                                           &weights.ip1_val_b[0],
                                           "relu", "value/fc1");
  if (isWdl) {
    value = builder->makeFullyConnectedLayer(value, 128, 3,
                                             &weights.ip2_val_w[0],
                                             &weights.ip2_val_b[0],
                                             "softmax", "value/fc2")
  }
  else {
    value = builder->makeFullyConnectedLayer(value, 128, 1,
                                             &weights.ip2_val_w[0],
                                             &weights.ip2_val_b[0],
                                             "tanh", "value/fc2");
  }

  // 5. Moves left head.
  void * mlh;
  if (hasMlh) {
    const int mlhChannels = weights.moves_left.biases.size();
    mlh = builder->makeConvolutionBlock(layer, channelSize, mlhChannels, 1,
                                        &weights.moves_left.weights[0],
                                        &weights.moves_left.biases[0],
                                        true, "mlh/conv");
    mlh = builder->makeReshapeLayer(mlh, 1, 1, mlhChannels * 8 * 8);
    mlh = builder->makeFullyConnectedLayer(mlh, mlhChannels * 8 * 8, weights.ip1_mov_b.size(),
                                           &weights.ip1_mov_w[0],
                                           &weights.ip1_mov_b[0],
                                           "relu", "mlh/fc1");
    mlh = builder->makeFullyConnectedLayer(mlh, weights.ip1_mov_b.size(), 1,
                                           &weights.ip2_mov_w[0],
                                           &weights.ip2_mov_b[0],
                                           "relu", "mlh/fc2");
  }

  // MPSNNGraph requires all three to be joined into one operation.
  //builder->buildGraph({policy, value, mlh});
  builder->buildGraph(policy);
}

//void MetalNetwork::forwardEval(InputsOutputs* io, int inputBatchSize) {
//void MetalNetwork::forwardEval(int* io, int inputBatchSize) {
void MetalNetwork::forwardEval(int inputBatchSize) {
//  CERR << "io: " << io;
  CERR << "batchsize: " << inputBatchSize;
//  builder->forwardEval(io, inputBatchSize);
  std::string str_obj = "lc0 invading";
  char * char_arr = &str_obj[0];
  CERR << "response: " << builder->getTestData(char_arr);
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
