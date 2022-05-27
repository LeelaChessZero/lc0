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
    const int gpu_id = options.GetOrDefault<int>("gpu", 0);
    builder_ = std::make_unique<MetalNetworkBuilder>();
    std::string device = builder_->init(gpu_id);
    CERR << "Initialized metal backend on device " << device;
  } catch (...) {
    throw Exception("There was an error initializing the GPU device.");
  }

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  const int channelSize = weights.input.weights.size() / kInputPlanes / 9;
  const int kernelSize = 3;

  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
  batch_size_ = options.GetOrDefault<int>("batch", 64);
  steps_ = options.GetOrDefault<int>("steps", 2);

  if (batch_size_ <= 0) {
    steps_ = 1;
  } else if (steps_ > max_batch_size_ / batch_size_) {
    steps_ = max_batch_size_ / batch_size_;
  }

  // Pointer to last layer in MPS Graph.
  void * layer;

  // 0. Input placeholder.
  layer = builder_->getInputPlaceholder(8, 8, kInputPlanes, "inputs");

  // 1. Input layer
  layer = builder_->makeConvolutionBlock(layer, kInputPlanes, channelSize, kernelSize,
                                        &weights.input.weights[0],
                                        &weights.input.biases[0],
                                        true, "input/conv");

  // 2. Residual blocks
  for (size_t i = 0; i < weights.residual.size(); i++) {
    layer = builder_->makeResidualBlock(layer, channelSize, channelSize, kernelSize,
                                       &weights.residual[i].conv1.weights[0],
                                       &weights.residual[i].conv1.biases[0],
                                       &weights.residual[i].conv2.weights[0],
                                       &weights.residual[i].conv2.biases[0],
                                       true, "block_" + std::to_string(i),
                                       weights.residual[i].has_se,
                                       weights.residual[i].se.b1.size(),
                                       &weights.residual[i].se.w1[0],
                                       &weights.residual[i].se.b1[0],
                                       &weights.residual[i].se.w2[0],
                                       &weights.residual[i].se.b2[0]);
  }

  // 3. Policy head.
  void * policy;
  if (conv_policy_) {
    policy = builder_->makeConvolutionBlock(layer, channelSize, channelSize, kernelSize,
                                           &weights.policy1.weights[0],
                                           &weights.policy1.biases[0],
                                           true, "policy/conv1");

    // No relu.
    policy = builder_->makeConvolutionBlock(policy, channelSize, 80, kernelSize,
                                           &weights.policy.weights[0],
                                           &weights.policy.biases[0],
                                           false, "policy/conv2");

    /**
     * @todo policy map implementation has bug in MPSGraph (GatherND not working in graph).
     * Implementation of policy map to be done in CPU for now.
     *
     * Reinstate this section when bug is fixed. See comments below.
     *
    // [1858 -> HWC or CHW]
    const bool HWC = false;
    std::vector<uint32_t> policy_map(1858);
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
    policy = builder_->makePolicyMapLayer(policy, &policy_map[0], "policy_map");
    */
  }
  else {
    const int policySize = weights.policy.biases.size();
    policy = builder_->makeConvolutionBlock(layer, channelSize, policySize, 1,
                                           &weights.policy.weights[0],
                                           &weights.policy.biases[0],
                                           true, "policy/conv");
    policy = builder_->makeFullyConnectedLayer(policy, policySize * 8 * 8, 1858,
                                              &weights.ip_pol_w[0],
                                              &weights.ip_pol_b[0],
                                              "", "policy/fc");
  }

  // 4. Value head.
  void * value;
  value = builder_->makeConvolutionBlock(layer, channelSize, 32, 1,
                                        &weights.value.weights[0],
                                        &weights.value.biases[0],
                                        true, "value/conv");
  value = builder_->makeFullyConnectedLayer(value, 32 * 8 * 8, 128,
                                           &weights.ip1_val_w[0],
                                           &weights.ip1_val_b[0],
                                           "relu", "value/fc1");
  wdl_ = file.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;
  if (wdl_) {
    value = builder_->makeFullyConnectedLayer(value, 128, 3,
                                             &weights.ip2_val_w[0],
                                             &weights.ip2_val_b[0],
                                             "softmax", "value/fc2");
  }
  else {
    value = builder_->makeFullyConnectedLayer(value, 128, 1,
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
    mlh = builder_->makeConvolutionBlock(layer, channelSize, mlhChannels, 1,
                                        &weights.moves_left.weights[0],
                                        &weights.moves_left.biases[0],
                                        true, "mlh/conv");
    mlh = builder_->makeFullyConnectedLayer(mlh, mlhChannels * 8 * 8, weights.ip1_mov_b.size(),
                                           &weights.ip1_mov_w[0],
                                           &weights.ip1_mov_b[0],
                                           "relu", "mlh/fc1");
    mlh = builder_->makeFullyConnectedLayer(mlh, weights.ip1_mov_b.size(), 1,
                                           &weights.ip2_mov_w[0],
                                           &weights.ip2_mov_b[0],
                                           "relu", "mlh/fc2");
  }

  // Select the outputs to be run through the inference graph.
  std::vector<void*> outputs;
  if (moves_left_) {
    outputs = {policy, value, mlh};
  }
  else {
    outputs = {policy, value};
  }

  builder_->setSelectedOutputs(&outputs);
}

void MetalNetwork::forwardEval(InputsOutputs* io, int batchSize) {
  // Expand encoded input into N x 112 x 8 x 8.
  float * dptr = io->input_val_mem_expanded_;
  for (size_t i = 0; i < batchSize; i++) {
      for (size_t j = 0; j < kInputPlanes; j++) {
          const float value = io->input_val_mem_[j + i * kInputPlanes];
          const uint64_t mask = io->input_masks_mem_[j + i * kInputPlanes];
          for (auto k = 0; k < 64; k++) {
              *(dptr++) = (mask & (((uint64_t)1) << k)) != 0 ? value : 0;
          }
      }
  }

  // Metal is not thread-safe, so lock is needed.
  lock_.lock();

  if (conv_policy_) {
    /**
     * @todo policy map implementation has bug in MPSGraph (GatherND not working in graph).
     * Implementation of policy map to be done in CPU for now.
     *
     * Remove this if-branch when bug is fixed. See comments above.
     */

    if (moves_left_) {
      builder_->forwardEval(io->input_val_mem_expanded_, batchSize);
      builder_->copyResults({io->op_policy_raw_mem_, io->op_value_mem_, io->op_moves_left_mem_});
    }
    else {
      builder_->forwardEval(io->input_val_mem_expanded_, batchSize);
      builder_->copyResults({io->op_policy_raw_mem_, io->op_value_mem_});
    }

    // Mapping from convolutional policy to lc0 policy
    for (size_t batch = 0; batch < batchSize; batch++) {
      for (size_t i = 0; i < 73 * 64; i++) {
        short j = kConvPolicyMap[i];
        if (j >= 0) {
          io->op_policy_mem_[batch * 1858 + j] =
              io->op_policy_raw_mem_[batch * 32 * 64 + i];
        }
      }
    }

  }
  else {
    if (moves_left_) {
      builder_->forwardEval(io->input_val_mem_expanded_, batchSize);
      builder_->copyResults({io->op_policy_mem_, io->op_value_mem_, io->op_moves_left_mem_});
    }
    else {
      builder_->forwardEval(io->input_val_mem_expanded_, batchSize);
      builder_->copyResults({io->op_policy_mem_, io->op_value_mem_});
    }
  }

  // The next thread can start using the GPU now.
  lock_.unlock();

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
