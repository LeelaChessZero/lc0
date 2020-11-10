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

#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/shared/policy_map.h"
#include "utils/bititer.h"
#include "utils/exception.h"

namespace lczero {
using namespace dnnl_backend;

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

class DnnlNetwork;

class DnnlNetworkComputation : public NetworkComputation {
 public:
  DnnlNetworkComputation(DnnlNetwork* network, bool wdl, bool moves_left);
  ~DnnlNetworkComputation();

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

  DnnlNetwork* network_;
};

class DnnlNetwork : public Network {
 public:
  DnnlNetwork(const WeightsFile& file, const OptionsDict& options)
      : capabilities_{file.format().network_format().input(),
                      file.format().network_format().moves_left()} {
    LegacyWeights weights(file.weights());

    conv_policy_ = file.format().network_format().policy() ==
                   pblczero::NetworkFormat::POLICY_CONVOLUTION;

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

    batch_size_ = options.GetOrDefault<int>("batch", 32);

#if DNNL_VERSION_MAJOR * 100 + DNNL_VERSION_MINOR >= 105
    dnnl::set_primitive_cache_capacity(
        options.GetOrDefault<int>("jit_cache", 1024));
#endif

    if (!options.IsDefault<int>("gpu")) {
      eng_ = dnnl::engine(dnnl::engine::kind::gpu, options.Get<int>("gpu"));
    } else {
      eng_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    }
    eng_stream_ = dnnl::stream(eng_);

    auto data_type = dnnl::memory::data_type::f32;
    if (options.GetOrDefault<bool>("fp16", false)) {
      if (eng_.get_kind() == dnnl::engine::kind::cpu) {
        data_type = dnnl::memory::data_type::bf16;
      } else {
        data_type = dnnl::memory::data_type::f16;
      }
    }

    // Default layout is nchw.
    const int kNumInputPlanes = kInputPlanes;
    numFilters_ = (int)weights.input.biases.size();
    numBlocks_ = (int)weights.residual.size();

    pol_channels_ = weights.policy.biases.size();

    // 1. Check for SE.
    has_se_ = false;
    if (weights.residual[0].has_se) {
      has_se_ = true;
    }

    // 2. Build the network, and copy the weights to GPU memory.

    // Input.
    {
      auto inputConv = std::make_unique<ConvLayer>(nullptr, numFilters_, 8, 8,
                                                   3, kNumInputPlanes, true);
      // Set the data type first, the following layers will pick it up.
      inputConv->SetDataType(data_type);
      inputConv->LoadWeights(&weights.input.weights[0],
                             &weights.input.biases[0], eng_, eng_stream_);
      network_.emplace_back(std::move(inputConv));
    }

    // Residual block.
    for (size_t block = 0; block < weights.residual.size(); block++) {
      auto conv1 = std::make_unique<ConvLayer>(getLastLayer(), numFilters_, 8,
                                               8, 3, numFilters_, true);
      conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                         &weights.residual[block].conv1.biases[0], eng_,
                         eng_stream_);
      network_.emplace_back(std::move(conv1));

      // Relu of second convolution and skip connection is handled by SELayer.
      bool has_se = weights.residual[block].has_se;

      auto conv2 = std::make_unique<ConvLayer>(
          getLastLayer(), numFilters_, 8, 8, 3, numFilters_, !has_se, !has_se);
      conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                         &weights.residual[block].conv2.biases[0], eng_,
                         eng_stream_);
      network_.emplace_back(std::move(conv2));

      if (has_se) {
        int numFCOut = (int)weights.residual[block].se.b1.size();
        auto se = std::make_unique<SELayer>(getLastLayer(), numFCOut);
        se->LoadWeights(&weights.residual[block].se.w1[0],
                        &weights.residual[block].se.b1[0],
                        &weights.residual[block].se.w2[0],
                        &weights.residual[block].se.b2[0], eng_, eng_stream_);
        network_.emplace_back(std::move(se));
      }
    }

    resi_last_ = getLastLayer();

    // Policy head.
    if (conv_policy_) {
      auto conv1 = std::make_unique<ConvLayer>(resi_last_, numFilters_, 8, 8, 3,
                                               numFilters_, true);
      conv1->LoadWeights(&weights.policy1.weights[0],
                         &weights.policy1.biases[0], eng_, eng_stream_);
      network_.emplace_back(std::move(conv1));

      // No relu
      auto conv2 = std::make_unique<ConvLayer>(getLastLayer(), pol_channels_, 8,
                                               8, 3, numFilters_, false);
      conv2->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                         eng_, eng_stream_);
      network_.emplace_back(std::move(conv2));
    } else {
      auto convPol = std::make_unique<ConvLayer>(resi_last_, pol_channels_, 8,
                                                 8, 1, numFilters_, true);
      convPol->LoadWeights(&weights.policy.weights[0],
                           &weights.policy.biases[0], eng_, eng_stream_);
      network_.emplace_back(std::move(convPol));

      auto FCPol = std::make_unique<FCLayer>(getLastLayer(), kNumOutputPolicy,
                                             1, 1, false);
      FCPol->LoadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0], eng_,
                         eng_stream_);
      network_.emplace_back(std::move(FCPol));
    }
    policy_out_ = getLastLayer();

    // Value head.
    {
      value_channels_ = weights.ip1_val_b.size();
      value_input_planes_ = weights.value.biases.size();

      auto convVal = std::make_unique<ConvLayer>(
          resi_last_, value_input_planes_, 8, 8, 1, numFilters_, true);
      convVal->LoadWeights(&weights.value.weights[0], &weights.value.biases[0],
                           eng_, eng_stream_);
      network_.emplace_back(std::move(convVal));

      auto FCVal1 = std::make_unique<FCLayer>(getLastLayer(), value_channels_,
                                              1, 1, true);
      FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0], eng_,
                          eng_stream_);
      network_.emplace_back(std::move(FCVal1));

      wdl_ = file.format().network_format().value() ==
             pblczero::NetworkFormat::VALUE_WDL;
      auto fc2_tanh = !wdl_;

      auto FCVal2 = std::make_unique<FCLayer>(getLastLayer(), wdl_ ? 3 : 1, 1,
                                              1, false, fc2_tanh);
      FCVal2->LoadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0], eng_,
                          eng_stream_);
      network_.emplace_back(std::move(FCVal2));
    }
    value_out_ = getLastLayer();

    // Moves left head
    moves_left_ = (file.format().network_format().moves_left() ==
                   pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                  options.GetOrDefault<bool>("mlh", true);
    if (moves_left_) {
      moves_channels_ = weights.ip1_mov_b.size();
      moves_input_planes_ = weights.moves_left.biases.size();

      auto convMov = std::make_unique<ConvLayer>(
          resi_last_, moves_input_planes_, 8, 8, 1, numFilters_, true);
      convMov->LoadWeights(&weights.moves_left.weights[0],
                           &weights.moves_left.biases[0], eng_, eng_stream_);
      network_.emplace_back(std::move(convMov));

      auto FCMov1 = std::make_unique<FCLayer>(getLastLayer(), moves_channels_,
                                              1, 1, true);
      FCMov1->LoadWeights(&weights.ip1_mov_w[0], &weights.ip1_mov_b[0], eng_,
                          eng_stream_);
      network_.emplace_back(std::move(FCMov1));

      auto FCMov2 = std::make_unique<FCLayer>(getLastLayer(), 1, 1, 1, true);
      FCMov2->LoadWeights(&weights.ip2_mov_w[0], &weights.ip2_mov_b[0], eng_,
                          eng_stream_);
      network_.emplace_back(std::move(FCMov2));
    }
    moves_left_out_ = getLastLayer();
  }

  void forwardEval(InputsOutputs* io, int inputBatchSize) {
    std::lock_guard<std::mutex> lock(lock_);

    // Expand packed planes to full planes.
    uint64_t* ipDataMasks = io->input_masks_mem_;
    float* ipDataValues = io->input_val_mem_;

    dnnl::engine cpu_eng;
    if (eng_.get_kind() == dnnl::engine::kind::cpu) {
      cpu_eng = eng_;
    } else {
      cpu_eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
    }

    int batchSize = batch_size_;
    if (batchSize <= 0) {
      // Use just one batch of variable size.
      batchSize = inputBatchSize;
    }

    // Break input batch in smaller batches.
    for (int start = 0; start < inputBatchSize; start += batchSize) {
      int currentBatchSize = inputBatchSize - start;
      if (currentBatchSize > batchSize) {
        currentBatchSize = batchSize;
      }

      auto input_desc = dnnl::memory::desc({batchSize, kInputPlanes, 8, 8},
                                           dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::nchw);
      dnnl::memory input_mem = dnnl::memory(input_desc, cpu_eng);

      float* buffer = (float*)input_mem.get_data_handle();
      for (int j = 0; j < currentBatchSize * kInputPlanes; j++) {
        const float value = ipDataValues[j + start * kInputPlanes];
        const uint64_t mask = ipDataMasks[j + start * kInputPlanes];
        for (auto i = 0; i < 64; i++)
          *(buffer++) = (mask & (((uint64_t)1) << i)) != 0 ? value : 0;
      }
      // Clear remaining buffer (if any).
      memset(buffer, 0, (batchSize - currentBatchSize) * kInputPlanes * 64 *
                            sizeof(float));

      // Move input to the gpu.
      if (eng_.get_kind() != dnnl::engine::kind::cpu) {
        auto tmp = dnnl::memory(input_desc, eng_);
        dnnl::reorder(input_mem, tmp).execute(eng_stream_, input_mem, tmp);
        input_mem = tmp;
      }

      // Output descriptors.
      dnnl::memory::desc opPol_desc;
      if (conv_policy_) {
        opPol_desc = dnnl::memory::desc({batchSize, pol_channels_, 8, 8},
                                        dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::nchw);
      } else {
        opPol_desc = dnnl::memory::desc({batchSize, kNumOutputPolicy, 1, 1},
                                        dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::nchw);
      }
      auto opVal_desc = dnnl::memory::desc({batchSize, wdl_ ? 3 : 1, 1, 1},
                                           dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::nchw);
      auto opMov_desc =
          dnnl::memory::desc({batchSize, 1, 1, 1}, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::nchw);
      // Output memory.
      dnnl::memory opPol_mem;
      dnnl::memory opVal_mem;
      dnnl::memory opMov_mem;

      // Intermediate tensors.
      dnnl::memory tensor_mem[3];

      int l = 0;
      // Input.
      network_[l++]->Eval(batchSize, tensor_mem[2], input_mem, eng_,
                          eng_stream_);  // input conv

      // Residual block.
      for (int block = 0; block < numBlocks_; block++) {
        network_[l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], eng_,
                            eng_stream_);  // conv1

        // For SE Resnet, skip connection is added after SE.
        if (has_se_) {
          network_[l++]->Eval(batchSize, tensor_mem[1], tensor_mem[0], eng_,
                              eng_stream_);  // conv2
        } else {
          network_[l++]->Eval(batchSize, tensor_mem[2], tensor_mem[0], eng_,
                              eng_stream_);  // conv2
        }

        if (has_se_) {
          network_[l++]->Eval(batchSize, tensor_mem[2], tensor_mem[1], eng_,
                              eng_stream_);  // SE layer
        }
      }

      // Policy head.
      if (conv_policy_) {
        network_[l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], eng_,
                            eng_stream_);  // policy conv1

        network_[l++]->Eval(batchSize, opPol_mem, tensor_mem[0], eng_,
                            eng_stream_);  // policy conv2
      } else {
        dnnl::memory policy_mem;
        network_[l++]->Eval(batchSize, policy_mem, tensor_mem[2], eng_,
                            eng_stream_);  // pol conv

        network_[l++]->Eval(batchSize, opPol_mem, policy_mem, eng_,
                            eng_stream_);  // pol FC  // POLICY
      }

      // value head
      {
        dnnl::memory tmp1_mem;
        dnnl::memory tmp2_mem;
        network_[l++]->Eval(batchSize, tmp1_mem, tensor_mem[2], eng_,
                            eng_stream_);  // value conv

        network_[l++]->Eval(batchSize, tmp2_mem, tmp1_mem, eng_,
                            eng_stream_);  // value FC1

        network_[l++]->Eval(batchSize, opVal_mem, tmp2_mem, eng_,
                            eng_stream_);  // value FC2    // VALUE
      }

      if (moves_left_) {
        // Moves left head
        dnnl::memory tmp1_mem;
        dnnl::memory tmp2_mem;
        network_[l++]->Eval(batchSize, tmp1_mem, tensor_mem[2], eng_,
                            eng_stream_);  // moves conv

        network_[l++]->Eval(batchSize, tmp2_mem, tmp1_mem, eng_,
                            eng_stream_);  // moves FC1

        // Moves left FC2
        network_[l++]->Eval(batchSize, opMov_mem, tmp2_mem, eng_, eng_stream_);
      }

      // Convert output data to nchw and if on gpu move them to the cpu.
      if (opPol_desc != opPol_mem.get_desc() ||
          eng_.get_kind() != dnnl::engine::kind::cpu) {
        auto tmp = dnnl::memory(opPol_desc, cpu_eng);
        dnnl::reorder(opPol_mem, tmp).execute(eng_stream_, opPol_mem, tmp);
        opPol_mem = tmp;
      }

      if (opVal_desc != opVal_mem.get_desc() ||
          eng_.get_kind() != dnnl::engine::kind::cpu) {
        auto tmp = dnnl::memory(opVal_desc, cpu_eng);
        dnnl::reorder(opVal_mem, tmp).execute(eng_stream_, opVal_mem, tmp);
        opVal_mem = tmp;
      }

      if (moves_left_ && (opMov_desc != opMov_mem.get_desc() ||
                          eng_.get_kind() != dnnl::engine::kind::cpu)) {
        auto tmp = dnnl::memory(opMov_desc, cpu_eng);
        dnnl::reorder(opMov_mem, tmp).execute(eng_stream_, opMov_mem, tmp);
        opMov_mem = tmp;
      }

      eng_stream_.wait();

      // Copy memopy to output buffers and do final transformations.
      if (wdl_) {
        // Value softmax done cpu side.
        float* opVal = (float*)opVal_mem.get_data_handle();
        for (int i = 0; i < currentBatchSize; i++) {
          float w = std::exp(opVal[3 * i + 0]);
          float d = std::exp(opVal[3 * i + 1]);
          float l = std::exp(opVal[3 * i + 2]);
          float sum = w + d + l;
          w /= sum;
          l /= sum;
          d = 1.0f - w - l;
          io->op_value_mem_[3 * (i + start) + 0] = w;
          io->op_value_mem_[3 * (i + start) + 1] = d;
          io->op_value_mem_[3 * (i + start) + 2] = l;
        }
      } else {
        memcpy(io->op_value_mem_ + start, opVal_mem.get_data_handle(),
               currentBatchSize * sizeof(float));
      }

      if (conv_policy_) {
        float* opPol = (float*)opPol_mem.get_data_handle();
        for (int batch = 0; batch < currentBatchSize; batch++) {
          for (int i = 0; i < 73 * 8 * 8; i++) {
            auto j = kConvPolicyMap[i];
            if (j >= 0) {
              io->op_policy_mem_[(batch + start) * kNumOutputPolicy + j] =
                  opPol[batch * pol_channels_ * 64 + i];
            }
          }
        }
      } else {
        memcpy(io->op_policy_mem_ + start * kNumOutputPolicy,
               opPol_mem.get_data_handle(),
               currentBatchSize * kNumOutputPolicy * sizeof(float));
      }

      if (moves_left_) {
        memcpy(io->op_moves_left_mem_ + start, opMov_mem.get_data_handle(),
               currentBatchSize * sizeof(float));
      }
    }
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<DnnlNetworkComputation>(this, wdl_, moves_left_);
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
  dnnl::engine eng_;
  dnnl::stream eng_stream_;
  int max_batch_size_;
  int batch_size_;
  bool wdl_;
  bool moves_left_;

  mutable std::mutex lock_;

  int numBlocks_;
  int numFilters_;
  int pol_channels_;
  int value_channels_;
  int value_input_planes_;
  int moves_channels_;
  int moves_input_planes_;

  bool has_se_;
  bool conv_policy_;
  std::vector<std::unique_ptr<BaseLayer>> network_;
  BaseLayer* getLastLayer() { return network_.back().get(); }

  BaseLayer* resi_last_;
  BaseLayer* policy_out_;
  BaseLayer* value_out_;
  BaseLayer* moves_left_out_;

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
};

DnnlNetworkComputation::DnnlNetworkComputation(DnnlNetwork* network, bool wdl,
                                               bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

DnnlNetworkComputation::~DnnlNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void DnnlNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

std::unique_ptr<Network> MakeDnnlNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& options) {
  if (!w) {
    throw Exception("The dnnl backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by DNNL backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION) {
    throw Exception("Policy format " +
                    std::to_string(weights.format().network_format().policy()) +
                    " is not supported by DNNL backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    std::to_string(weights.format().network_format().value()) +
                    " is not supported by DNNL backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception(
        "Movest left head format " +
        std::to_string(weights.format().network_format().moves_left()) +
        " is not supported by DNNL backend.");
  }
  return std::make_unique<DnnlNetwork>(weights, options);
}

REGISTER_NETWORK("dnnl", MakeDnnlNetwork, 110)

}  // namespace lczero
