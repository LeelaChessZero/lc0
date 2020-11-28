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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "simple_common.h"
#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/shared/policy_map.h"
#include "utils/bititer.h"
#include "utils/exception.h"

//#define DEBUG_RAW_NPS

namespace lczero {
using namespace simple_backend;

static constexpr int kNumOutputPolicy = 1858;

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left) {
    input_masks_mem_ =
        (uint64_t*)malloc(maxBatchSize * kInputPlanes * sizeof(uint64_t));

    input_val_mem_ =
        (float*)malloc(maxBatchSize * kInputPlanes * sizeof(float));

    op_policy_mem_ =
        (float*)malloc(maxBatchSize * kNumOutputPolicy * sizeof(float));

    op_value_mem_ =
        (float*)malloc(maxBatchSize * (wdl ? 3 : 1) * sizeof(float));

    if (moves_left) {
      op_moves_left_mem_ = (float*)malloc(maxBatchSize * sizeof(float));
    } else
      op_moves_left_mem_ = nullptr;
  }
  ~InputsOutputs() {
    free(input_masks_mem_);
    free(input_val_mem_);
    free(op_policy_mem_);
    free(op_value_mem_);
    if (op_moves_left_mem_) {
      free(op_moves_left_mem_);
    }
  }
  uint64_t* input_masks_mem_;
  float* input_val_mem_;
  float* op_policy_mem_;
  float* op_value_mem_;
  float* op_moves_left_mem_;
};

class SimpleNetwork;

class SimpleNetworkComputation : public NetworkComputation {
 public:
  SimpleNetworkComputation(SimpleNetwork* network, bool wdl, bool moves_left);
  ~SimpleNetworkComputation();

  void AddInput(InputPlanes&& input) override {
    const auto iter_mask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    const auto iter_val =
        &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    int i = 0;
    for (const auto& plane : input) {
      iter_mask[i] = plane.mask;
      iter_val[i] = plane.value;
      i++;
    }

    batch_size_++;
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    if (wdl_) {
      auto w = inputs_outputs_->op_value_mem_[3 * sample + 0];
      auto l = inputs_outputs_->op_value_mem_[3 * sample + 2];
      return w - l;
    } else {
      return inputs_outputs_->op_value_mem_[sample];
    }
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      auto d = inputs_outputs_->op_value_mem_[3 * sample + 1];
      return d;
    } else {
      return 0.0f;
    }
  }

  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
  }

  float GetMVal(int sample) const override {
    if (moves_left_) {
      return inputs_outputs_->op_moves_left_mem_[sample];
    }
    return 0.0f;
  }

 private:
  // Memory holding inputs, outputs.
  std::unique_ptr<InputsOutputs> inputs_outputs_;
  int batch_size_;
  bool wdl_;
  bool moves_left_;

  SimpleNetwork* network_;
};

class SimpleNetwork : public Network {
 public:
  SimpleNetwork(const WeightsFile& file, const OptionsDict& options)
      : capabilities_{file.format().network_format().input(),
                      file.format().network_format().moves_left()} {
    LegacyWeights weights(file.weights());

    conv_policy_ = file.format().network_format().policy() ==
                   pblczero::NetworkFormat::POLICY_CONVOLUTION;

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

    // Default layout is nchw.

    const int kNumInputPlanes = kInputPlanes;
    const int kNumFilters = (int)weights.input.biases.size();
    numBlocks_ = (int)weights.residual.size();
    size_t residual_single_layer_weight_size =
        3 * 3 * kNumFilters * kNumFilters * sizeof(float);

    // 0. Check for SE.

    has_se_ = false;
    if (weights.residual[0].has_se) {
      has_se_ = true;
    }

    // 1. Allocate scratch space.

    // Have some minumum as we also use this for transforming weights.
    size_t max_weight_size = 128 * 1024 * 1024;

    // Parts from scratch allocation are suballocated to hold various weights
    // and biases when transforming winograd weights (one layer at a time), 128
    // MB is way more than that what we need but make sure it's at least 3x of
    // single layer's weight size to be safe.
    if (max_weight_size < 3 * residual_single_layer_weight_size)
      max_weight_size = 3 * residual_single_layer_weight_size;

    scratch_size_ = max_weight_size;

    scratch_mem_ = malloc(scratch_size_);

    // 2. Build the network, and copy the weights to GPU memory.

    // Input.
    {
      auto inputConv = std::make_unique<ConvLayer>(nullptr, kNumFilters, 8, 8,
                                                   3, kNumInputPlanes, true);
      inputConv->LoadWeights(&weights.input.weights[0],
                             &weights.input.biases[0], scratch_mem_);
      network_.emplace_back(std::move(inputConv));
    }

    // Residual block.
    for (size_t block = 0; block < weights.residual.size(); block++) {
      auto conv1 = std::make_unique<ConvLayer>(getLastLayer(), kNumFilters, 8,
                                               8, 3, kNumFilters, true);
      conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                         &weights.residual[block].conv1.biases[0],
                         scratch_mem_);
      network_.emplace_back(std::move(conv1));

      // Relu of second convolution and skip connection is handled by SELayer.
      bool has_se = weights.residual[block].has_se;

      auto conv2 = std::make_unique<ConvLayer>(
          getLastLayer(), kNumFilters, 8, 8, 3, kNumFilters, !has_se, !has_se);
      conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                         &weights.residual[block].conv2.biases[0],
                         scratch_mem_);
      network_.emplace_back(std::move(conv2));

      if (has_se) {
        int numFCOut = (int)weights.residual[block].se.b1.size();
        auto se = std::make_unique<SELayer>(getLastLayer(), numFCOut);
        se->LoadWeights(&weights.residual[block].se.w1[0],
                        &weights.residual[block].se.b1[0],
                        &weights.residual[block].se.w2[0],
                        &weights.residual[block].se.b2[0], scratch_mem_);
        network_.emplace_back(std::move(se));
      }
    }

    BaseLayer* resi_last_ = getLastLayer();

    // Policy head.
    if (conv_policy_) {
      auto conv1 = std::make_unique<ConvLayer>(resi_last_, kNumFilters, 8, 8, 3,
                                               kNumFilters, true);
      conv1->LoadWeights(&weights.policy1.weights[0],
                         &weights.policy1.biases[0], scratch_mem_);
      network_.emplace_back(std::move(conv1));

      auto pol_channels = weights.policy.biases.size();

      // No relu
      auto conv2 = std::make_unique<ConvLayer>(getLastLayer(), pol_channels, 8,
                                               8, 3, kNumFilters, false);
      conv2->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                         scratch_mem_);
      network_.emplace_back(std::move(conv2));

      auto policymap = std::make_unique<PolicyMapLayer>(
          getLastLayer(), kNumOutputPolicy, 1, 1, 73 * 8 * 8);
      policymap->LoadWeights(kConvPolicyMap, scratch_mem_);

      network_.emplace_back(std::move(policymap));
    } else {
      auto convPol = std::make_unique<ConvLayer>(
          resi_last_, weights.policy.biases.size(), 8, 8, 1, kNumFilters, true);
      convPol->LoadWeights(&weights.policy.weights[0],
                           &weights.policy.biases[0], scratch_mem_);
      network_.emplace_back(std::move(convPol));

      auto FCPol = std::make_unique<FCLayer>(
          getLastLayer(), weights.ip_pol_b.size(), 1, 1, false, true);
      FCPol->LoadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0],
                         scratch_mem_);
      network_.emplace_back(std::move(FCPol));
    }

    // Value head.
    {
      auto convVal = std::make_unique<ConvLayer>(
          resi_last_, weights.value.biases.size(), 8, 8, 1, kNumFilters, true);
      convVal->LoadWeights(&weights.value.weights[0], &weights.value.biases[0],
                           scratch_mem_);
      network_.emplace_back(std::move(convVal));

      auto FCVal1 = std::make_unique<FCLayer>(
          getLastLayer(), weights.ip1_val_b.size(), 1, 1, true, true);
      FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCVal1));

      wdl_ = file.format().network_format().value() ==
             pblczero::NetworkFormat::VALUE_WDL;
      auto fc2_tanh = !wdl_;

      auto FCVal2 =
          std::make_unique<FCLayer>(getLastLayer(), weights.ip2_val_b.size(), 1,
                                    1, false, true, fc2_tanh);
      FCVal2->LoadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCVal2));
    }

    // Moves left head
    moves_left_ = (file.format().network_format().moves_left() ==
                   pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                  options.GetOrDefault<bool>("mlh", true);
    if (moves_left_) {
      auto convMov = std::make_unique<ConvLayer>(
          resi_last_, weights.moves_left.biases.size(), 8, 8, 1, kNumFilters,
          true);
      convMov->LoadWeights(&weights.moves_left.weights[0],
                           &weights.moves_left.biases[0], scratch_mem_);
      network_.emplace_back(std::move(convMov));

      auto FCMov1 = std::make_unique<FCLayer>(
          getLastLayer(), weights.ip1_mov_b.size(), 1, 1, true, true);
      FCMov1->LoadWeights(&weights.ip1_mov_w[0], &weights.ip1_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov1));

      auto FCMov2 =
          std::make_unique<FCLayer>(getLastLayer(), 1, 1, 1, true, true);
      FCMov2->LoadWeights(&weights.ip2_mov_w[0], &weights.ip2_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov2));
    }

    // 3. Allocate memory for running the network:
    //    - three buffers of max size are enough (one to hold input, second to
    //      hold output and third to hold skip connection's input).

    // size of input to the network
    size_t maxSize = max_batch_size_ * kNumInputPlanes * 64 * sizeof(float);

    // take max size of all layers
    for (auto& layer : network_) {
      maxSize = std::max(maxSize, layer->GetOutputSize(max_batch_size_));
    }

    for (auto& mem : tensor_mem_) {
      mem = (float*)malloc(maxSize);
      memset(mem, 0, maxSize);
    }
  }

  void forwardEval(InputsOutputs* io, int batchSize) {
    std::lock_guard<std::mutex> lock(lock_);

    // Expand packed planes to full planes.
    uint64_t* ipDataMasks = io->input_masks_mem_;
    float* ipDataValues = io->input_val_mem_;
    float* buffer = tensor_mem_[0];
    for (int j = 0; j < batchSize * kInputPlanes; j++) {
      const float value = ipDataValues[j];
      const uint64_t mask = ipDataMasks[j];
      for (auto i = 0; i < 64; i++)
        *(buffer++) = (mask & (((uint64_t)1) << i)) != 0 ? value : 0;
    }

    float* opPol = io->op_policy_mem_;
    float* opVal = io->op_value_mem_;
    float* opMov = io->op_moves_left_mem_;

    int l = 0;
    // Input.
    network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0], scratch_mem_,
                        scratch_size_);  // input conv

    // Residual block.
    for (int block = 0; block < numBlocks_; block++) {
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2],
                          scratch_mem_, scratch_size_);  // conv1

      // For SE Resnet, skip connection is added after SE (and bias is added
      // as part of SE).
      if (has_se_) {
        network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0],
                            scratch_mem_, scratch_size_);  // conv2
      } else {
        network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0],
                            scratch_mem_, scratch_size_);  // conv2
      }

      if (has_se_) {
        network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[1],
                            scratch_mem_, scratch_size_);  // SE layer
      }
    }

    // Policy head.
    if (conv_policy_) {
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2],
                          scratch_mem_, scratch_size_);  // policy conv1

      network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0],
                          scratch_mem_, scratch_size_);  // policy conv2

      network_[l++]->Eval(batchSize, (float*)opPol, tensor_mem_[1],
                          scratch_mem_,
                          scratch_size_);  // policy map layer  // POLICY output
    } else {
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2],
                          scratch_mem_, scratch_size_);  // pol conv

      network_[l++]->Eval(batchSize, (float*)opPol, tensor_mem_[0],
                          scratch_mem_, scratch_size_);  // pol FC  // POLICY
    }

    // value head
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], scratch_mem_,
                        scratch_size_);  // value conv

    network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], scratch_mem_,
                        scratch_size_);  // value FC1

    network_[l++]->Eval(batchSize, (float*)opVal, tensor_mem_[1], scratch_mem_,
                        scratch_size_);  // value FC2    // VALUE

    if (moves_left_) {
      // Moves left head
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2],
                          scratch_mem_, scratch_size_);  // moves conv

      network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0],
                          scratch_mem_, scratch_size_);  // moves FC1

      // Moves left FC2
      network_[l++]->Eval(batchSize, (float*)opMov, tensor_mem_[1],
                          scratch_mem_, scratch_size_);
    }

    if (wdl_) {
      // Value softmax done cpu side.
      for (int i = 0; i < batchSize; i++) {
        float w = std::exp(io->op_value_mem_[3 * i + 0]);
        float d = std::exp(io->op_value_mem_[3 * i + 1]);
        float l = std::exp(io->op_value_mem_[3 * i + 2]);
        float sum = w + d + l;
        w /= sum;
        l /= sum;
        d = 1.0f - w - l;
        io->op_value_mem_[3 * i + 0] = w;
        io->op_value_mem_[3 * i + 1] = d;
        io->op_value_mem_[3 * i + 2] = l;
      }
    }
  }

  ~SimpleNetwork() {
    for (auto mem : tensor_mem_) {
      if (mem) free(mem);
    }
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    // Set correct gpu id for this computation (as it might have been called
    // from a different thread).
    return std::make_unique<SimpleNetworkComputation>(this, wdl_, moves_left_);
  }

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(max_batch_size_, wdl_,
                                             moves_left_);
    } else {
      std::unique_ptr<InputsOutputs> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

 private:
  const NetworkCapabilities capabilities_;
  int max_batch_size_;
  bool wdl_;
  bool moves_left_;

  std::mutex lock_;

  int numBlocks_;
  bool has_se_;
  bool conv_policy_;
  std::vector<std::unique_ptr<BaseLayer>> network_;
  BaseLayer* getLastLayer() { return network_.back().get(); }

  float* tensor_mem_[3];
  void* scratch_mem_;
  size_t scratch_size_;

  std::mutex inputs_outputs_lock_;

  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
};

SimpleNetworkComputation::SimpleNetworkComputation(SimpleNetwork* network,
                                                   bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

SimpleNetworkComputation::~SimpleNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void SimpleNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

std::unique_ptr<Network> MakeSimpleNetwork(const std::optional<WeightsFile>& w,
                                           const OptionsDict& options) {
  if (!w) {
    throw Exception("The simple backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by the simple backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION) {
    throw Exception("Policy format " +
                    std::to_string(weights.format().network_format().policy()) +
                    " is not supported by the simple backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    std::to_string(weights.format().network_format().value()) +
                    " is not supported by the simple backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception(
        "Movest left head format " +
        std::to_string(weights.format().network_format().moves_left()) +
        " is not supported by the simple backend.");
  }
  return std::make_unique<SimpleNetwork>(weights, options);
}

REGISTER_NETWORK("simple", MakeSimpleNetwork, 10)

}  // namespace lczero
