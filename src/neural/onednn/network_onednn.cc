/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021-2022 The LCZero Authors

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
#include "neural/shared/attention_policy_map.h"
#include "neural/shared/policy_map.h"
#include "utils/bititer.h"
#include "utils/exception.h"

#include <omp.h>

namespace lczero {
using namespace onednn_backend;

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

class OnednnNetwork;

class OnednnNetworkComputation : public NetworkComputation {
 public:
  OnednnNetworkComputation(OnednnNetwork* network, bool wdl, bool moves_left);
  ~OnednnNetworkComputation();

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

  OnednnNetwork* network_;
};

class OnednnNetwork : public Network {
 public:
  OnednnNetwork(const WeightsFile& file, const OptionsDict& options)
      : capabilities_{file.format().network_format().input(),
                      file.format().network_format().moves_left()} {
    LegacyWeights weights(file.weights());

    conv_policy_ = file.format().network_format().policy() ==
                   pblczero::NetworkFormat::POLICY_CONVOLUTION;

    attn_policy_ = file.format().network_format().policy() ==
                   pblczero::NetworkFormat::POLICY_ATTENTION;

    default_activation_ =
        file.format().network_format().default_activation() ==
                pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH
            ? MISH
            : RELU;

#if DNNL_VERSION_MAJOR * 100 + DNNL_VERSION_MINOR >= 105
    dnnl::set_primitive_cache_capacity(
        options.GetOrDefault<int>("jit_cache", 1024));
#endif

    if (!options.IsDefault<int>("threads")) {
      omp_set_num_threads(options.Get<int>("threads"));
    }

    cpu_eng_ = dnnl::engine(dnnl::engine::kind::cpu, 0);

    if (!options.IsDefault<int>("gpu")) {
      eng_ = dnnl::engine(dnnl::engine::kind::gpu, options.Get<int>("gpu"));
    } else {
      eng_ = cpu_eng_;
    }
    eng_stream_ = dnnl::stream(eng_);

    auto data_type = dnnl::memory::data_type::f32;
    if (options.GetOrDefault<bool>(
            "fp16", eng_.get_kind() == dnnl::engine::kind::gpu)) {
      if (eng_.get_kind() == dnnl::engine::kind::cpu) {
        data_type = dnnl::memory::data_type::bf16;
      } else {
        data_type = dnnl::memory::data_type::f16;
      }
    }

    // Unfortunately current oneDNN versions get this wrong, selecting Winograd
    // on gpu and not on cpu (last tested with version 2.6.0). So for the time
    // being this will be overriden in every case.
    auto convolution_type = dnnl::algorithm::convolution_auto;
    if (!options.IsDefault<bool>("winograd")) {
      if (options.Get<bool>("winograd")) {
        convolution_type = dnnl::algorithm::convolution_winograd;
      } else {
        convolution_type = dnnl::algorithm::convolution_direct;
      }
    } else {
      // Heuristic: only use Winograd convolution on cpu newer than avx2.
      if (eng_.get_kind() == dnnl::engine::kind::cpu &&
          dnnl::get_effective_cpu_isa() > dnnl::cpu_isa::avx2) {
        convolution_type = dnnl::algorithm::convolution_winograd;
      } else {
        convolution_type = dnnl::algorithm::convolution_direct;
      }
    }

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

    batch_size_ = options.GetOrDefault<int>(
        "batch", data_type == dnnl::memory::data_type::f32 ? 32 : 64);

    steps_ = options.GetOrDefault<int>("steps", 2);
    if (batch_size_ <= 0) {
      steps_ = 1;
    } else if (steps_ > max_batch_size_ / batch_size_) {
      steps_ = max_batch_size_ / batch_size_;
    }

    // Default layout is nchw.
    numFilters_ = (int)weights.input.biases.size();
    numBlocks_ = (int)weights.residual.size();

    pol_channels_ = weights.policy.biases.size();

    // 1. Check for SE.
    has_se_ = false;
    if (weights.residual[0].has_se) {
      has_se_ = true;
    }

    // 2. Build the network, and copy the weights to GPU memory.

    layers_.resize(steps_);
    for (int idx = 0; idx < steps_; idx++) {
      // Input.
      {
        auto inputConv = std::make_unique<ConvLayer>(
            nullptr, numFilters_, 8, 8, 3, kInputPlanes, default_activation_);
        // Set the data type first, the following layers will pick it up.
        inputConv->SetDataType(data_type);
        inputConv->SetConvolutionType(convolution_type);
        auto w_md = dnnl::memory::desc({numFilters_, kInputPlanes, 3, 3},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::oihw);
        auto w_mem = dnnl::memory(w_md, cpu_eng_, &weights.input.weights[0]);
        auto b_md =
            dnnl::memory::desc({numFilters_}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        auto b_mem = dnnl::memory(b_md, cpu_eng_, &weights.input.biases[0]);
        inputConv->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(inputConv));
      }

      // Residual block.
      for (size_t block = 0; block < weights.residual.size(); block++) {
        auto conv1 =
            std::make_unique<ConvLayer>(getLastLayer(idx), numFilters_, 8, 8, 3,
                                        numFilters_, default_activation_);
        auto w_md = dnnl::memory::desc({numFilters_, numFilters_, 3, 3},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::oihw);
        auto w_mem = dnnl::memory(w_md, cpu_eng_,
                                  &weights.residual[block].conv1.weights[0]);
        auto b_md =
            dnnl::memory::desc({numFilters_}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        auto b_mem = dnnl::memory(b_md, cpu_eng_,
                                  &weights.residual[block].conv1.biases[0]);
        conv1->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(conv1));

        // Activation of second convolution and skip connection is handled by
        // SELayer.
        bool has_se = weights.residual[block].has_se;

        auto conv2 = std::make_unique<ConvLayer>(
            getLastLayer(idx), numFilters_, 8, 8, 3, numFilters_,
            has_se ? NONE : default_activation_, !has_se);
        w_mem = dnnl::memory(w_md, cpu_eng_,
                             &weights.residual[block].conv2.weights[0]);
        b_mem = dnnl::memory(b_md, cpu_eng_,
                             &weights.residual[block].conv2.biases[0]);
        conv2->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(conv2));

        if (has_se) {
          int numFCOut = (int)weights.residual[block].se.b1.size();

          auto se = std::make_unique<SELayer>(getLastLayer(idx), numFCOut,
                                              default_activation_);
          w_md = dnnl::memory::desc({numFCOut, numFilters_},
                                    dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::ab);
          w_mem =
              dnnl::memory(w_md, cpu_eng_, &weights.residual[block].se.w1[0]);
          b_md = dnnl::memory::desc({numFCOut}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::a);
          b_mem =
              dnnl::memory(b_md, cpu_eng_, &weights.residual[block].se.b1[0]);
          auto w2_md = dnnl::memory::desc({2 * numFilters_, numFCOut},
                                          dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::ab);
          auto w2_mem =
              dnnl::memory(w2_md, cpu_eng_, &weights.residual[block].se.w2[0]);
          auto b2_md = dnnl::memory::desc({2 * numFilters_},
                                          dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::a);
          auto b2_mem =
              dnnl::memory(b2_md, cpu_eng_, &weights.residual[block].se.b2[0]);
          se->LoadWeights(w_mem, b_mem, w2_mem, b2_mem, eng_, eng_stream_);
          layers_[idx].emplace_back(std::move(se));
        }
      }

      BaseLayer* resi_last = getLastLayer(idx);

      // Policy head.
      if (attn_policy_) {
        for (auto layer : weights.pol_encoder) {
          // TODO: support encoder heads.
          throw Exception(
              "Encoder heads are not yet supported by the oneDNN backend.");
        }
        const int embedding_size = weights.ip_pol_b.size();
        const int policy_d_model = weights.ip2_pol_b.size();

        auto attn = std::make_unique<AttentionPolicyHead>(
            resi_last, embedding_size, policy_d_model);
        auto ip_w_md = dnnl::memory::desc({numFilters_, embedding_size},
                                          dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::ab);
        auto ip_w_mem =
            dnnl::memory(ip_w_md, cpu_eng_, weights.ip_pol_w.data());
        auto ip_b_md =
            dnnl::memory::desc({embedding_size}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        auto ip_b_mem =
            dnnl::memory(ip_b_md, cpu_eng_, weights.ip_pol_b.data());
        auto ip23_w_md = dnnl::memory::desc({embedding_size, policy_d_model},
                                            dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::ab);
        auto ip2_w_mem =
            dnnl::memory(ip23_w_md, cpu_eng_, weights.ip2_pol_w.data());
        auto ip23_b_md =
            dnnl::memory::desc({policy_d_model}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        auto ip2_b_mem =
            dnnl::memory(ip23_b_md, cpu_eng_, weights.ip2_pol_b.data());
        auto ip3_w_mem =
            dnnl::memory(ip23_w_md, cpu_eng_, weights.ip3_pol_w.data());
        auto ip3_b_mem =
            dnnl::memory(ip23_b_md, cpu_eng_, weights.ip3_pol_b.data());
        auto ip4_w_md = dnnl::memory::desc({1, 4, policy_d_model},
                                           dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::abc);
        auto ip4_w_mem =
            dnnl::memory(ip4_w_md, cpu_eng_, weights.ip4_pol_w.data());
        attn->LoadWeights(ip_w_mem, ip_b_mem, ip2_w_mem, ip2_b_mem, ip3_w_mem,
                          ip3_b_mem, ip4_w_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(attn));
      } else if (conv_policy_) {
        auto conv1 = std::make_unique<ConvLayer>(
            resi_last, numFilters_, 8, 8, 3, numFilters_, default_activation_);
        auto w_md = dnnl::memory::desc({numFilters_, numFilters_, 3, 3},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::oihw);
        auto w_mem = dnnl::memory(w_md, cpu_eng_, &weights.policy1.weights[0]);
        auto b_md =
            dnnl::memory::desc({numFilters_}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        auto b_mem = dnnl::memory(b_md, cpu_eng_, &weights.policy1.biases[0]);
        conv1->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(conv1));

        // No Activation
        auto conv2 = std::make_unique<ConvLayer>(
            getLastLayer(idx), pol_channels_, 8, 8, 3, numFilters_, NONE);
        w_md = dnnl::memory::desc({pol_channels_, numFilters_, 3, 3},
                                  dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::oihw);
        w_mem = dnnl::memory(w_md, cpu_eng_, &weights.policy.weights[0]);
        b_md = dnnl::memory::desc({pol_channels_}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::a);
        b_mem = dnnl::memory(b_md, cpu_eng_, &weights.policy.biases[0]);
        conv2->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(conv2));
      } else {
        auto convPol =
            std::make_unique<ConvLayer>(resi_last, pol_channels_, 8, 8, 1,
                                        numFilters_, default_activation_);
        auto w_md = dnnl::memory::desc({pol_channels_, numFilters_, 1, 1},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::oihw);
        auto w_mem = dnnl::memory(w_md, cpu_eng_, &weights.policy.weights[0]);
        auto b_md =
            dnnl::memory::desc({pol_channels_}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        auto b_mem = dnnl::memory(b_md, cpu_eng_, &weights.policy.biases[0]);
        convPol->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(convPol));

        auto FCPol = std::make_unique<FCLayer>(getLastLayer(idx),
                                               kNumOutputPolicy, 1, 1, NONE);
        w_md = dnnl::memory::desc({kNumOutputPolicy, pol_channels_, 8, 8},
                                  dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::abcd);
        w_mem = dnnl::memory(w_md, cpu_eng_, &weights.ip_pol_w[0]);
        b_md =
            dnnl::memory::desc({kNumOutputPolicy}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        b_mem = dnnl::memory(b_md, cpu_eng_, &weights.ip_pol_b[0]);
        FCPol->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(FCPol));
      }

      // Value head.
      {
        value_channels_ = weights.ip1_val_b.size();
        value_input_planes_ = weights.value.biases.size();

        auto convVal =
            std::make_unique<ConvLayer>(resi_last, value_input_planes_, 8, 8, 1,
                                        numFilters_, default_activation_);
        auto w_md = dnnl::memory::desc({value_input_planes_, numFilters_, 1, 1},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::oihw);
        auto w_mem = dnnl::memory(w_md, cpu_eng_, &weights.value.weights[0]);
        auto b_md = dnnl::memory::desc({value_input_planes_},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::a);
        auto b_mem = dnnl::memory(b_md, cpu_eng_, &weights.value.biases[0]);
        convVal->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(convVal));

        auto FCVal1 = std::make_unique<FCLayer>(
            getLastLayer(idx), value_channels_, 1, 1, default_activation_);
        w_md = dnnl::memory::desc({value_channels_, value_input_planes_, 8, 8},
                                  dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::abcd);
        w_mem = dnnl::memory(w_md, cpu_eng_, &weights.ip1_val_w[0]);
        b_md =
            dnnl::memory::desc({value_channels_}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        b_mem = dnnl::memory(b_md, cpu_eng_, &weights.ip1_val_b[0]);
        FCVal1->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(FCVal1));

        wdl_ = file.format().network_format().value() ==
               pblczero::NetworkFormat::VALUE_WDL;
        auto fc2_tanh = !wdl_;

        auto FCVal2 = std::make_unique<FCLayer>(getLastLayer(idx), wdl_ ? 3 : 1,
                                                1, 1, fc2_tanh ? TANH : NONE);
        w_md = dnnl::memory::desc({wdl_ ? 3 : 1, value_channels_, 1, 1},
                                  dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::abcd);
        w_mem = dnnl::memory(w_md, cpu_eng_, &weights.ip2_val_w[0]);
        b_md = dnnl::memory::desc({wdl_ ? 3 : 1}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::a);
        b_mem = dnnl::memory(b_md, cpu_eng_, &weights.ip2_val_b[0]);
        FCVal2->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(FCVal2));
      }

      // Moves left head
      moves_left_ = (file.format().network_format().moves_left() ==
                     pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                    options.GetOrDefault<bool>("mlh", true);
      if (moves_left_) {
        moves_channels_ = weights.ip1_mov_b.size();
        moves_input_planes_ = weights.moves_left.biases.size();

        auto convMov =
            std::make_unique<ConvLayer>(resi_last, moves_input_planes_, 8, 8, 1,
                                        numFilters_, default_activation_);
        auto w_md = dnnl::memory::desc({moves_input_planes_, numFilters_, 1, 1},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::oihw);
        auto w_mem =
            dnnl::memory(w_md, cpu_eng_, &weights.moves_left.weights[0]);

        auto b_md = dnnl::memory::desc({moves_input_planes_},
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::a);
        auto b_mem =
            dnnl::memory(b_md, cpu_eng_, &weights.moves_left.biases[0]);
        convMov->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(convMov));

        auto FCMov1 = std::make_unique<FCLayer>(
            getLastLayer(idx), moves_channels_, 1, 1, default_activation_);
        w_md = dnnl::memory::desc({moves_channels_, moves_input_planes_, 8, 8},
                                  dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::abcd);
        w_mem = dnnl::memory(w_md, cpu_eng_, &weights.ip1_mov_w[0]);
        b_md =
            dnnl::memory::desc({moves_channels_}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a);
        b_mem = dnnl::memory(b_md, cpu_eng_, &weights.ip1_mov_b[0]);
        FCMov1->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(FCMov1));

        auto FCMov2 = std::make_unique<FCLayer>(getLastLayer(idx), 1, 1, 1,
                                                default_activation_);
        w_md = dnnl::memory::desc({1, moves_channels_, 1, 1},
                                  dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::abcd);
        w_mem = dnnl::memory(w_md, cpu_eng_, &weights.ip2_mov_w[0]);
        b_md = dnnl::memory::desc({1}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::a);
        b_mem = dnnl::memory(b_md, cpu_eng_, &weights.ip2_mov_b[0]);
        FCMov2->LoadWeights(w_mem, b_mem, eng_, eng_stream_);
        layers_[idx].emplace_back(std::move(FCMov2));
      }

      // Initialize layers if batch size fixed.
      if (options.GetOrDefault<bool>("init", true) && batch_size_ > 0) {
        int batchSize = (idx + 1) * batch_size_;
        InputsOutputs io(batchSize, wdl_, moves_left_);
        memset(io.input_masks_mem_, 0,
               batchSize * kInputPlanes * sizeof(uint64_t));
        memset(io.input_val_mem_, 0, batchSize * kInputPlanes * sizeof(float));
        forwardEval(&io, batchSize);
      }
    }
  }

  void forwardEval(InputsOutputs* io, int inputBatchSize) {
    // Expand packed planes to full planes.
    uint64_t* ipDataMasks = io->input_masks_mem_;
    float* ipDataValues = io->input_val_mem_;

    int batchSize = steps_ * batch_size_;
    if (batchSize <= 0) {
      // Use just one batch of variable size.
      batchSize = inputBatchSize;
    }

    // Break input batch in smaller batches.
    for (int start = 0; start < inputBatchSize; start += batchSize) {
      int idx = steps_ - 1;

      int currentBatchSize = inputBatchSize - start;
      if (currentBatchSize > batchSize) {
        currentBatchSize = batchSize;
      } else {
        idx = (currentBatchSize - 1) / batch_size_;
        batchSize = (idx + 1) * batch_size_;
      }

      auto input_desc = dnnl::memory::desc({batchSize, kInputPlanes, 8, 8},
                                           dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::nchw);
      dnnl::memory input_mem = dnnl::memory(input_desc, cpu_eng_);

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
        dnnl::reorder in_reorder = dnnl::reorder(input_mem, tmp);
        in_reorder.execute(eng_stream_, input_mem, tmp);
        input_mem = tmp;
      }

      // Output descriptors.
      dnnl::memory::desc opPol_desc;
      if (attn_policy_) {
        opPol_desc = dnnl::memory::desc({batchSize, 67, 8, 8},
                                        dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::nchw);
      } else if (conv_policy_) {
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
      layers_[idx][l++]->Eval(batchSize, tensor_mem[2], input_mem, eng_,
                              eng_stream_);  // input conv

      // Residual block.
      for (int block = 0; block < numBlocks_; block++) {
        layers_[idx][l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], eng_,
                                eng_stream_);  // conv1

        // For SE Resnet, skip connection is added after SE.
        if (has_se_) {
          layers_[idx][l++]->Eval(batchSize, tensor_mem[1], tensor_mem[0], eng_,
                                  eng_stream_);  // conv2
        } else {
          layers_[idx][l++]->Eval(batchSize, tensor_mem[2], tensor_mem[0], eng_,
                                  eng_stream_);  // conv2
        }

        if (has_se_) {
          layers_[idx][l++]->Eval(batchSize, tensor_mem[2], tensor_mem[1], eng_,
                                  eng_stream_);  // SE layer
        }
      }

      // Policy head.
      if (attn_policy_) {
        layers_[idx][l++]->Eval(batchSize, opPol_mem, tensor_mem[2], eng_,
                                eng_stream_);  // attention head
      } else if (conv_policy_) {
        layers_[idx][l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], eng_,
                                eng_stream_);  // policy conv1

        layers_[idx][l++]->Eval(batchSize, opPol_mem, tensor_mem[0], eng_,
                                eng_stream_);  // policy conv2
      } else {
        dnnl::memory policy_mem;
        layers_[idx][l++]->Eval(batchSize, policy_mem, tensor_mem[2], eng_,
                                eng_stream_);  // pol conv

        layers_[idx][l++]->Eval(batchSize, opPol_mem, policy_mem, eng_,
                                eng_stream_);  // pol FC  // POLICY
      }

      // value head
      {
        dnnl::memory tmp1_mem;
        dnnl::memory tmp2_mem;
        layers_[idx][l++]->Eval(batchSize, tmp1_mem, tensor_mem[2], eng_,
                                eng_stream_);  // value conv

        layers_[idx][l++]->Eval(batchSize, tmp2_mem, tmp1_mem, eng_,
                                eng_stream_);  // value FC1

        layers_[idx][l++]->Eval(batchSize, opVal_mem, tmp2_mem, eng_,
                                eng_stream_);  // value FC2    // VALUE
      }

      if (moves_left_) {
        // Moves left head
        dnnl::memory tmp1_mem;
        dnnl::memory tmp2_mem;
        layers_[idx][l++]->Eval(batchSize, tmp1_mem, tensor_mem[2], eng_,
                                eng_stream_);  // moves conv

        layers_[idx][l++]->Eval(batchSize, tmp2_mem, tmp1_mem, eng_,
                                eng_stream_);  // moves FC1

        // Moves left FC2
        layers_[idx][l++]->Eval(batchSize, opMov_mem, tmp2_mem, eng_,
                                eng_stream_);
      }

      // Convert output data to nchw and if on gpu move them to the cpu.
      if (opPol_desc != opPol_mem.get_desc() ||
          eng_.get_kind() != dnnl::engine::kind::cpu) {
        auto tmp = dnnl::memory(opPol_desc, cpu_eng_);
        dnnl::reorder pol_reorder = dnnl::reorder(opPol_mem, tmp);
        pol_reorder.execute(eng_stream_, opPol_mem, tmp);
        opPol_mem = tmp;
      }

      if (opVal_desc != opVal_mem.get_desc() ||
          eng_.get_kind() != dnnl::engine::kind::cpu) {
        auto tmp = dnnl::memory(opVal_desc, cpu_eng_);
        dnnl::reorder val_reorder_ = dnnl::reorder(opVal_mem, tmp);
        val_reorder_.execute(eng_stream_, opVal_mem, tmp);
        opVal_mem = tmp;
      }

      if (moves_left_ && (opMov_desc != opMov_mem.get_desc() ||
                          eng_.get_kind() != dnnl::engine::kind::cpu)) {
        auto tmp = dnnl::memory(opMov_desc, cpu_eng_);
        dnnl::reorder mov_reorder_ = dnnl::reorder(opMov_mem, tmp);
        mov_reorder_.execute(eng_stream_, opMov_mem, tmp);
        opMov_mem = tmp;
      }

      eng_stream_.wait();

      // Copy memory to output buffers and do final transformations.
      if (wdl_) {
        // Value softmax done cpu side.
        float* opVal = (float*)opVal_mem.get_data_handle();
        for (int i = 0; i < currentBatchSize; i++) {
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
          io->op_value_mem_[3 * (i + start) + 0] = w;
          io->op_value_mem_[3 * (i + start) + 1] = d;
          io->op_value_mem_[3 * (i + start) + 2] = l;
        }
      } else {
        memcpy(io->op_value_mem_ + start, opVal_mem.get_data_handle(),
               currentBatchSize * sizeof(float));
      }
      if (attn_policy_) {
        float* opPol = (float*)opPol_mem.get_data_handle();
        // The promotion offsets are extracted from the output tensor.
        float promotion_offsets[3][8];
        for (int batch = 0; batch < currentBatchSize; batch++) {
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 8; j++) {
              promotion_offsets[i][j] =
                  opPol[batch * (64 * 64 + 8 * 24) + 64 * 64 + i * 8 + j] +
                  opPol[batch * (64 * 64 + 8 * 24) + 64 * 64 + 24 + j];
            }
          }
          for (int x = 0; x < 64 * 64; x++) {
            auto y = kAttnPolicyMap[x];
            if (y >= 0) {
              io->op_policy_mem_[(batch + start) * kNumOutputPolicy + y] =
                  opPol[batch * (64 * 64 + 8 * 24) + x];
            }
          }
          for (int k = 0; k < 8; k++) {
            for (int j = 0; j < 8; j++) {
              for (int i = 0; i < 3; i++) {
                auto y = kAttnPolicyMap[64 * 64 + 24 * k + 3 * j + i];
                if (y >= 0) {
                  io->op_policy_mem_[(batch + start) * kNumOutputPolicy + y] =
                      opPol[batch * (64 * 64 + 8 * 24) + (48 + k) * 64 + 56 +
                            j] +
                      promotion_offsets[i][j];
                }
              }
            }
          }
        }
      } else if (conv_policy_) {
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
    return std::make_unique<OnednnNetworkComputation>(this, wdl_, moves_left_);
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
  dnnl::engine cpu_eng_;
  dnnl::engine eng_;
  dnnl::stream eng_stream_;
  int max_batch_size_;
  int batch_size_;
  int steps_;
  bool wdl_;
  bool moves_left_;

  std::mutex lock_;

  int numBlocks_;
  int numFilters_;
  int pol_channels_;
  int value_channels_;
  int value_input_planes_;
  int moves_channels_;
  int moves_input_planes_;

  bool has_se_;
  bool conv_policy_;
  bool attn_policy_;
  ActivationFunction default_activation_;

  std::vector<std::vector<std::unique_ptr<BaseLayer>>> layers_;
  BaseLayer* getLastLayer(int idx) { return layers_[idx].back().get(); }

  std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
};

OnednnNetworkComputation::OnednnNetworkComputation(OnednnNetwork* network,
                                                   bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

OnednnNetworkComputation::~OnednnNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void OnednnNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

std::unique_ptr<Network> MakeOnednnNetwork(const std::optional<WeightsFile>& w,
                                           const OptionsDict& options) {
  if (!w) {
    throw Exception("The oneDNN backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception("Network format " +
                    pblczero::NetworkFormat::NetworkStructure_Name(
                        weights.format().network_format().network()) +
                    " is not supported by the oneDNN backend.");
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
                    " is not supported by the oneDNN backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    pblczero::NetworkFormat::ValueFormat_Name(
                        weights.format().network_format().value()) +
                    " is not supported by the oneDNN backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception("Moves left head format " +
                    pblczero::NetworkFormat::MovesLeftFormat_Name(
                        weights.format().network_format().moves_left()) +
                    " is not supported by the oneDNN backend.");
  }
  if (weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
      weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
    throw Exception(
        "Default activation " +
        pblczero::NetworkFormat::DefaultActivation_Name(
            weights.format().network_format().default_activation()) +
        " is not supported by the oneDNN backend.");
  }
  return std::make_unique<OnednnNetwork>(weights, options);
}

REGISTER_NETWORK("onednn", MakeOnednnNetwork, 110)

}  // namespace lczero
