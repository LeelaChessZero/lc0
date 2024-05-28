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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "mps/MetalNetworkBuilder.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/shared/attention_policy_map.h"
#include "neural/shared/policy_map.h"
#include "utils/bititer.h"
#include "utils/exception.h"

namespace lczero {
namespace metal_backend {

MetalNetworkComputation::MetalNetworkComputation(MetalNetwork* network,
                                                 bool wdl, bool moves_left)
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

std::string activationString(pblczero::NetworkFormat::ActivationFunction act) {
  switch (act) {
    case pblczero::NetworkFormat::ACTIVATION_RELU:
      return "relu";
    case pblczero::NetworkFormat::ACTIVATION_MISH:
      return "mish";
    case pblczero::NetworkFormat::ACTIVATION_NONE:
      return "none";
    case pblczero::NetworkFormat::ACTIVATION_TANH:
      return "tanh";
    case pblczero::NetworkFormat::ACTIVATION_SIGMOID:
      return "sigmoid";
    case pblczero::NetworkFormat::ACTIVATION_SELU:
      return "selu";
    case pblczero::NetworkFormat::ACTIVATION_SWISH:
      return "swish";
    case pblczero::NetworkFormat::ACTIVATION_RELU_2:
      return "relu_2";
    case pblczero::NetworkFormat::ACTIVATION_SOFTMAX:
      return "softmax";
    default:
      return "";
  }
}

MetalNetwork::MetalNetwork(const WeightsFile& file, const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()} {
  MultiHeadWeights weights(file.weights());

  try {
    const int gpu_id = options.GetOrDefault<int>("gpu", 0);
    builder_ = std::make_unique<MetalNetworkBuilder>();
    std::string device = builder_->init(gpu_id);
    CERR << "Initialized metal backend on device " << device;
  } catch (...) {
    throw Exception("There was an error initializing the GPU device.");
  }

  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
  batch_size_ = options.GetOrDefault<int>("batch", 64);

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  attn_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_ATTENTION;

  wdl_ = file.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;

  moves_left_ = (file.format().network_format().moves_left() ==
                 pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                options.GetOrDefault<bool>("mlh", true);

  bool attn_body =
      (file.format().network_format().network()) ==
          pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
      (file.format().network_format().network()) ==
          pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;

  // Build MPS Graph.
  Activations activations;
  activations.default_activation =
      file.format().network_format().default_activation() ==
              pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH
          ? "mish"
          : "relu";
  const auto smolgen_activation =
      file.format().network_format().smolgen_activation();
  activations.smolgen_activation =
      smolgen_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
          ? activations.default_activation
          : activationString(
                static_cast<pblczero::NetworkFormat::ActivationFunction>(
                    smolgen_activation));
  const auto ffn_activation = file.format().network_format().ffn_activation();
  activations.ffn_activation =
      ffn_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
          ? activations.default_activation
          : activationString(
                static_cast<pblczero::NetworkFormat::ActivationFunction>(
                    ffn_activation));

  std::string policy_head =
      options.GetOrDefault<std::string>("policy_head", "vanilla");
  // Check that selected policy head exists.
  if (weights.policy_heads.count(policy_head) == 0) {
    throw Exception("The policy head you specified '" + policy_head +
                    "' does not exist in this net.");
  }
  std::string value_head =
      options.GetOrDefault<std::string>("value_head", "winner");
  // Check that selected value head exists.
  if (weights.value_heads.count(value_head) == 0) {
    throw Exception("The value head you specified '" + value_head +
                    "' does not exist in this net.");
  }

  auto embedding = static_cast<InputEmbedding>(file.format().network_format().input_embedding());
  builder_->build(kInputPlanes, weights, embedding, attn_body, attn_policy_, conv_policy_,
                  wdl_, moves_left_, activations, policy_head, value_head);
}

void MetalNetwork::forwardEval(InputsOutputs* io, int batchSize) {
  // Expand encoded input into N x 112 x 8 x 8.
  float* dptr = &io->input_val_mem_expanded_[0];
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
     * @todo policy map implementation has bug in MPSGraph (GatherND not working
     * in graph). Implementation of policy map to be done in CPU for now.
     *
     * Remove this if-branch when bug is fixed. See comments above.
     */

    if (moves_left_) {
      builder_->forwardEval(&io->input_val_mem_expanded_[0], batchSize,
                            {&io->op_policy_raw_mem_[0], &io->op_value_mem_[0],
                             &io->op_moves_left_mem_[0]});
    } else {
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
              io->op_policy_raw_mem_[batch * (64 * 64 + 8 * 24) + 64 * 64 +
                                     24 * k + 3 * j + i] +=
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
    } else if (conv_policy_) {
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

  } else {
    if (moves_left_) {
      builder_->forwardEval(&io->input_val_mem_expanded_[0], batchSize,
                            {&io->op_policy_mem_[0], &io->op_value_mem_[0],
                             &io->op_moves_left_mem_[0]});
    } else {
      builder_->forwardEval(&io->input_val_mem_expanded_[0], batchSize,
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
  auto nf = weights.format().network_format();
  using NF = pblczero::NetworkFormat;
  switch (nf.network()) {
    case NF::NETWORK_CLASSICAL_WITH_HEADFORMAT:
    case NF::NETWORK_SE_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
      break;
    default:
      throw Exception("Network format " +
                      NF::NetworkStructure_Name(nf.network()) +
                      " is not supported by the Metal backend.");
  }
  switch (nf.policy()) {
    case NF::POLICY_CLASSICAL:
    case NF::POLICY_CONVOLUTION:
    case NF::POLICY_ATTENTION:
      break;
    default:
      throw Exception("Policy format " + NF::PolicyFormat_Name(nf.policy()) +
                      " is not supported by the Metal backend.");
  }
  switch (nf.value()) {
    case NF::VALUE_CLASSICAL:
    case NF::VALUE_WDL:
      break;
    default:
      throw Exception("Value format " + NF::ValueFormat_Name(nf.value()) +
                      " is not supported by the Metal backend.");
  }
  switch (nf.moves_left()) {
    case NF::MOVES_LEFT_NONE:
    case NF::MOVES_LEFT_V1:
      break;
    default:
      throw Exception("Moves left head format " +
                      NF::MovesLeftFormat_Name(nf.moves_left()) +
                      " is not supported by the Metal backend.");
  }
  switch (nf.default_activation()) {
    case NF::DEFAULT_ACTIVATION_RELU:
    case NF::DEFAULT_ACTIVATION_MISH:
      break;
    default:
      throw Exception("Default activation " +
                      NF::DefaultActivation_Name(nf.default_activation()) +
                      " is not supported by the Metal backend.");
  }
  switch (nf.input_embedding()) {
    case NF::INPUT_EMBEDDING_NONE:
    case NF::INPUT_EMBEDDING_PE_MAP:
    case NF::INPUT_EMBEDDING_PE_DENSE:
      break;
    default:
      throw Exception("Input embedding " +
                      NF::InputEmbeddingFormat_Name(nf.input_embedding()) +
                      " is not supported by the Metal backend.");
  }
  return std::make_unique<MetalNetwork>(weights, options);
}

REGISTER_NETWORK("metal", MakeMetalNetwork, 105)

}  // namespace metal_backend
}  // namespace lczero
