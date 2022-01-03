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

void describeInputs(uint64_t * masks, float * vals, int batchSize, int numPerBatch) {
  CERR << "Inputs: batchsize: " << batchSize;
  uint64_t * p = masks;
  float * q = vals;
  for (int i=0; i<batchSize; i++) {
    for (int j=0; j<numPerBatch; j++) {
      CERR << "batch[" << i << "]: layer[" << j << "]: mask) "
           << *(p + (i * numPerBatch) + j) << "; val) "
           << *(q + (i * numPerBatch) + j);
    }
  }
}

MetalNetworkComputation::MetalNetworkComputation(MetalNetwork* network, bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

MetalNetworkComputation::~MetalNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void MetalNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

MetalNetwork::MetalNetwork(const WeightsFile& file, const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().moves_left()} {

  LegacyWeights weights(file.weights());

  try {
    // @todo better implementation with unique_ptr??
    builder = new MetalNetworkBuilder();
    builder->init();
  } catch (...) {
    throw Exception("There was an error initializing the GPU device.");
  }

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  const int channelSize = weights.input.weights.size() / kInputPlanes / 9;
  const int kernelSize = 3;

  bool has_se_ = false;
  if (weights.residual[0].has_se) {
    has_se_ = true;
  }

  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
  batch_size_ = options.GetOrDefault<int>("batch", 64);
  steps_ = options.GetOrDefault<int>("steps", 2);

  if (batch_size_ <= 0) {
    steps_ = 1;
  } else if (steps_ > max_batch_size_ / batch_size_) {
    steps_ = max_batch_size_ / batch_size_;
  }

  // Pointer to last layer in MPS NN graph.
  void * layer;

  //describeWeights(weights.input, kInputPlanes);

  // 1. Input layer
  layer = builder->makeConvolutionBlock(nullptr, kInputPlanes, channelSize, kernelSize,
                                        &weights.input.weights[0],
                                        &weights.input.biases[0],
                                        true, "input/conv");

  // 2. Residual blocks
  for (size_t i = 0; i < weights.residual.size(); i++) {
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
  if (conv_policy_) {
    policy = builder->makeConvolutionBlock(layer, channelSize, channelSize, kernelSize,
                                           &weights.policy1.weights[0],
                                           &weights.policy1.biases[0],
                                           true, "policy/conv1");

    // No relu.
    policy = builder->makeConvolutionBlock(policy, channelSize, 80, kernelSize,
                                           &weights.policy.weights[0],
                                           &weights.policy.biases[0],
                                           false, "policy/conv2");

    // [1858 -> HWC or CHW]
    const bool HWC = true;
    std::vector<short> policy_map(1858);
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
    // @todo Policy mapping in GPU.
    policy = builder->makePolicyMapLayer(policy, &policy_map);
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
  value = builder->makeConvolutionBlock(layer, channelSize, 32, 1,
                                        &weights.value.weights[0],
                                        &weights.value.biases[0],
                                        true, "value/conv");
  value = builder->makeReshapeLayer(value, 1, 1, 32 * 8 * 8);
  value = builder->makeFullyConnectedLayer(value, 32 * 8 * 8, 128,
                                           &weights.ip1_val_w[0],
                                           &weights.ip1_val_b[0],
                                           "relu", "value/fc1");
  wdl_ = file.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;
  if (wdl_) {
    value = builder->makeFullyConnectedLayer(value, 128, 3,
                                             &weights.ip2_val_w[0],
                                             &weights.ip2_val_b[0],
                                             "softmax", "value/fc2");
  }
  else {
    value = builder->makeFullyConnectedLayer(value, 128, 1,
                                             &weights.ip2_val_w[0],
                                             &weights.ip2_val_b[0],
                                             "tanh", "value/fc2");
  }

  // 5. Moves left head.
  moves_left_ = (file.format().network_format().moves_left() ==
                 pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                options.GetOrDefault<bool>("mlh", true);
  void * mlh;
  if (moves_left_) {
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

  // MPSNNGraph requires all three heads to be joined into one optimized graph
  // operation.
  std::vector<void*> outputs;
  if (moves_left_) {
    outputs = {policy, value, mlh};
  }
  else {
    outputs = {policy, value};
  }
  builder->buildGraph(&outputs);
}

void MetalNetwork::forwardEval(InputsOutputs* io, int batchSize) {
  CERR << "Forwarding eval to graph adapter: batchsize: " << batchSize;
  //describeInputs(io->input_masks_mem_, io->input_val_mem_, batchSize, kInputPlanes);
  //std::vector<float*> * output_mems;
  /*memset(io->op_policy_mem_, 0, max_batch_size_ * kNumOutputPolicy * sizeof(uint64_t));
  memset(io->op_value_mem_, 0, max_batch_size_ * (wdl_ ? 3 : 1) * sizeof(float));
  if (moves_left_) {
    memset(io->op_moves_left_mem_, 0, max_batch_size_ * sizeof(float));
    output_mems = {io->op_policy_mem_, io->op_value_mem_, io->op_moves_left_mem_};
  }
  else {
    output_mems = {io->op_policy_mem_, io->op_value_mem_};
  }*/
  std::vector<float*> output_mems = builder->forwardEval(io->input_masks_mem_, io->input_val_mem_, nullptr,
                       batchSize, kInputPlanes);

  CERR << "Completed forwarding";
//  CERR << "Return vector: size: " << output_mems.size();
//  CERR << "policy: N:" << *output_mems[0];
//  CERR << "value: N:" << *output_mems[1];
//  CERR << "mlh: N:" << *output_mems[2];
  //CERR << "Return vector[1]: " << output_mems[1];

  return;

  // Copy memory to output buffers and do final transformations.
  if (wdl_) {
    // Value softmax done cpu side.
    float* opVal = output_mems[1];
    for (int i = 0; i < batchSize; i++) {
      float w = opVal[3 * i + 0];
      float d = opVal[3 * i + 1];
      float l = opVal[3 * i + 2];
      float m = std::max({w, d, l});
      w = std::exp(w - m);
      d = std::exp(d - m);
      l = std::exp(l - m);
      float sum = w + d + l;
      w /= sum;
      l /= sum;
      d = 1.0f - w - l;
      io->op_value_mem_[3 * i + 0] = w;
      io->op_value_mem_[3 * i + 1] = d;
      io->op_value_mem_[3 * i + 2] = l;
    }
  } else {
    memcpy(io->op_value_mem_, output_mems[1], batchSize * sizeof(float));
  }
  CERR << "Completed value";

  if (conv_policy_) {
    float* opPol = output_mems[0];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int i = 0; i < 73 * 8 * 8; i++) {
        auto j = kConvPolicyMap[i];
        if (j >= 0) {
          io->op_policy_mem_[batch * kNumOutputPolicy + j] = opPol[batch * 80 * 64 + i];
        }
      }
    }
  } else {
    memcpy(io->op_policy_mem_, output_mems[0], batchSize * kNumOutputPolicy * sizeof(float));
  }
  CERR << "Completed policy";

  if (moves_left_) {
    memcpy(io->op_moves_left_mem_, output_mems[2], batchSize * sizeof(float));
  }
  CERR << "Completed mlh";

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
