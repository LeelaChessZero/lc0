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
#include "neural/shared/attention_policy_map.h"
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

  const int channelSize = weights.input.weights.size() / kInputPlanes / 9;
  const int kernelSize = 3;

  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
  batch_size_ = options.GetOrDefault<int>("batch", 64);

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  attn_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_ATTENTION;

  wdl_ = file.format().network_format().value() == pblczero::NetworkFormat::VALUE_WDL;

  moves_left_ = (file.format().network_format().moves_left() ==
                 pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                options.GetOrDefault<bool>("mlh", true);

  policy_d_model_ = weights.ip2_pol_b.size();

  // Build MPS Graph.
  builder_->build(kInputPlanes, channelSize, kernelSize, weights, attn_policy_, conv_policy_, wdl_, moves_left_,
    file.format().network_format().default_activation() == pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH ? "mish" : "relu"
  );
}

void MetalNetwork::forwardEval(InputsOutputs* io, int batchSize) {
  // Expand encoded input into N x 112 x 8 x 8.
  float * dptr = &io->input_val_mem_expanded_[0];
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

  if (attn_policy_ || conv_policy_) {
    /**
     * @todo policy map implementation has bug in MPSGraph (GatherND not working in graph).
     * Implementation of policy map to be done in CPU for now.
     *
     * Remove this if-branch when bug is fixed. See comments above.
     */

    if (moves_left_) {
      builder_->forwardEval(
          &io->input_val_mem_expanded_[0], batchSize,
          {&io->op_policy_raw_mem_[0], &io->op_value_mem_[0], &io->op_moves_left_mem_[0]});
    }
    else {
      builder_->forwardEval(
          &io->input_val_mem_expanded_[0], batchSize,
          {&io->op_policy_raw_mem_[0], &io->op_value_mem_[0]});
    }
    // The next thread can start using the GPU now.
    lock_.unlock();

    if (attn_policy_) {
      // Promotion offset calculation.
      for (size_t batch = 0; batch < batchSize; batch++) {
        for (int k = 0; k < 8; k++) {      // y in cuda
          for (int j = 0; j < 8; j++) {    // w in cuda
            for (int i = 0; i < 3; i++) {  // c in cuda
              // Promotion offsets already precalculated and stored in GPU.
              // Just the main policy offsets need to be added here.
              io->op_policy_raw_mem_[batch * (64 * 64 + 8 * 24) + 64 * 64 + 24 * k +
                                 3 * j + i] +=
                  io->op_policy_raw_mem_[batch * (64 * 64 + 8 * 24) +
                                     (48 + k) * 64 + 56 + j];
            }
          }
        }
      }
      // Mapping from attention policy to lc0 policy
      for (size_t batch = 0; batch < batchSize; batch++) {
        for (size_t i = 0; i < 64 * 64 + 8 * 24; i++) {
          size_t j = kAttnPolicyMap[i];
          if (j >= 0) {
            io->op_policy_mem_[batch * 1858 + j] =
                io->op_policy_raw_mem_[batch * (64 * 64 + 8 * 24) + i];
          }
        }
      }
    }
    else if (conv_policy_) {
      // Mapping from convolutional policy to lc0 policy
      for (size_t batch = 0; batch < batchSize; batch++) {
        for (size_t i = 0; i < 73 * 64; i++) {
          short j = kConvPolicyMap[i];
          if (j >= 0) {
            io->op_policy_mem_[batch * 1858 + j] =
                io->op_policy_raw_mem_[batch * 80 * 64 + i];
          }
        }
      }
    }

  }
  else {
    if (moves_left_) {
      builder_->forwardEval(
          &io->input_val_mem_expanded_[0], batchSize,
          {&io->op_policy_mem_[0], &io->op_value_mem_[0], &io->op_moves_left_mem_[0]});
    }
    else {
      builder_->forwardEval(
          &io->input_val_mem_expanded_[0], batchSize,
          {&io->op_policy_mem_[0], &io->op_value_mem_[0]});
    }

    // The next thread can start using the GPU now.
    lock_.unlock();
  }

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
    throw Exception("Network format " +
                    pblczero::NetworkFormat::NetworkStructure_Name(
                      weights.format().network_format().network()) +
                    " is not supported by the Metal backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_ATTENTION) {
    throw Exception("Policy format " +
                    pblczero::NetworkFormat::PolicyFormat_Name(
                      weights.format().network_format().policy()) +
                    " is not supported by the Metal backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    pblczero::NetworkFormat::ValueFormat_Name(
                      weights.format().network_format().value()) +
                    " is not supported by the Metal backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception("Moves left head format " +
                    pblczero::NetworkFormat::MovesLeftFormat_Name(
                      weights.format().network_format().moves_left()) +
                    " is not supported by the Metal backend.");
  }
  if (weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
      weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
    throw Exception("Default activation " +
                    pblczero::NetworkFormat::DefaultActivation_Name(
                        weights.format().network_format().default_activation()) +
                    " is not supported by the Metal backend.");
  }
  return std::make_unique<MetalNetwork>(weights, options);
}

REGISTER_NETWORK("metal", MakeMetalNetwork, 105)

}  // namespace backend_metal
}  // namespace lczero
