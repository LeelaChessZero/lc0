/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2022 The LCZero Authors

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

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "neural/blas/blas.h"
#include "neural/blas/convolution1.h"
#include "neural/blas/encoder.h"
#include "neural/blas/fully_connected_layer.h"
#include "neural/blas/se_unit.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "neural/network_legacy.h"
#include "neural/shared/activation.h"
#include "neural/shared/attention_policy_map.h"
#include "neural/shared/policy_map.h"
#include "neural/shared/winograd_filter.h"
#include "utils/numa.h"

#ifdef USE_DNNL
#include <omp.h>
#endif

#ifdef USE_ISPC
#include "activation_ispc.h"
#endif

namespace lczero {
namespace {

template <bool use_eigen>
class BlasComputation : public NetworkComputation {
 public:
  BlasComputation(const LegacyWeights& weights, const size_t max_batch_size,
                  const bool wdl, const bool moves_left, const bool conv_policy,
                  const ActivationFunction default_activation,
                  const ActivationFunction smolgen_activation,
                  const ActivationFunction ffn_activation,
                  const bool attn_policy, const bool attn_body);

  virtual ~BlasComputation() {}

  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  // Do the computation.
  void ComputeBlocking() override;

  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return static_cast<int>(planes_.size()); }

  // Returns Q value of @sample.
  float GetQVal(int sample) const override {
    if (wdl_) {
      auto w = q_values_[3 * sample + 0];
      auto l = q_values_[3 * sample + 2];
      return w - l;
    } else {
      return q_values_[sample];
    }
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      auto d = q_values_[3 * sample + 1];
      return d;
    } else {
      return 0.0f;
    }
  }

  float GetMVal(int sample) const override {
    if (moves_left_) {
      return m_values_[sample];
    } else {
      return 0.0f;
    }
  }

  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override {
    return policies_[sample][move_id];
  }

 private:
  void EncodePlanes(const InputPlanes& sample, float* buffer);
  void MakeEncoderLayer(std::vector<float>& head_buffer,
                        std::vector<float>& head_buffer2,
                        std::vector<float>& head_buffer3, size_t batch_size,
                        const LegacyWeights::EncoderLayer& layer,
                        int embedding_size, int heads,
                        ActivationFunction smolgen_activation = SWISH,
                        ActivationFunction ffn_activation = RELU_2,
                        float alpha = 1.0f);

  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kSquares = kWidth * kHeight;
  static constexpr auto kPolicyOutputs = 1858;
  // Number of used planes with convolutional policy.
  // The real number of planes is higher because of padding.
  static constexpr auto kPolicyUsedPlanes = 73;

  const LegacyWeights& weights_;
  size_t max_batch_size_;
  std::vector<InputPlanes> planes_;
  std::vector<std::vector<float>> policies_;
  std::vector<float> q_values_;
  std::vector<float> m_values_;
  bool wdl_;
  bool moves_left_;
  bool conv_policy_;
  ActivationFunction default_activation_;
  ActivationFunction smolgen_activation_;
  ActivationFunction ffn_activation_;
  bool attn_policy_;
  bool attn_body_;
};

template <bool use_eigen>
class BlasNetwork : public Network {
 public:
  BlasNetwork(const WeightsFile& weights, const OptionsDict& options);
  virtual ~BlasNetwork(){};

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasComputation<use_eigen>>(
        weights_, max_batch_size_, wdl_, moves_left_, conv_policy_,
        default_activation_, smolgen_activation_, ffn_activation_, attn_policy_,
        attn_body_);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  void InitThread(int id) override { Numa::BindThread(id); }

 private:
  // A cap on the max batch size since it consumes a lot of memory
  static constexpr auto kHardMaxBatchSize = 2048;

  const NetworkCapabilities capabilities_;
  LegacyWeights weights_;
  size_t max_batch_size_;
  bool wdl_;
  bool moves_left_;
  bool conv_policy_;
  ActivationFunction default_activation_;
  ActivationFunction smolgen_activation_;
  ActivationFunction ffn_activation_;
  bool attn_policy_;
  bool attn_body_;
};

template <bool use_eigen>
BlasComputation<use_eigen>::BlasComputation(
    const LegacyWeights& weights, const size_t max_batch_size, const bool wdl,
    const bool moves_left, const bool conv_policy,
    const ActivationFunction default_activation,
    const ActivationFunction smolgen_activation,
    const ActivationFunction ffn_activation, const bool attn_policy,
    const bool attn_body)
    : weights_(weights),
      max_batch_size_(max_batch_size),
      policies_(0),
      q_values_(0),
      wdl_(wdl),
      moves_left_(moves_left),
      conv_policy_(conv_policy),
      default_activation_(default_activation),
      smolgen_activation_(smolgen_activation),
      ffn_activation_(ffn_activation),
      attn_policy_(attn_policy),
      attn_body_(attn_body) {
#ifdef USE_DNNL
  omp_set_num_threads(1);
#endif
}

template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenStridedMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
               Eigen::OuterStride<>>;
template <typename T>
using ConstEigenStridedMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
               Eigen::OuterStride<>>;

template <bool use_eigen>
void BlasComputation<use_eigen>::MakeEncoderLayer(
    std::vector<float>& head_buffer, std::vector<float>& head_buffer2,
    std::vector<float>& head_buffer3, size_t batch_size,
    const LegacyWeights::EncoderLayer& layer, int embedding_size, int heads,
    ActivationFunction smolgen_activation, ActivationFunction ffn_activation,
    float alpha) {
  const int d_model = layer.mha.q_b.size();
  const int dff_size = layer.ffn.dense1_b.size();
  std::vector<float> head_buffer4(batch_size * std::max(d_model, dff_size) *
                                  kSquares);

  // Smolgen.
  if (layer.mha.has_smolgen) {
    const float* input = &head_buffer[0];
    float* QK = &head_buffer4[0];

    // Compress.
    const auto hidden_channels =
        layer.mha.smolgen.compress.size() / embedding_size;
    std::vector<float> temp1(batch_size * kSquares * hidden_channels);
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * kSquares, embedding_size, hidden_channels, input,
        layer.mha.smolgen.compress.data(), (const float*)nullptr, NONE,
        temp1.data());

    // Dense 1.
    const auto hidden_sz = layer.mha.smolgen.dense1_b.size();
    std::vector<float> temp2(batch_size * hidden_sz);
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, kSquares * hidden_channels, hidden_sz, temp1.data(),
        layer.mha.smolgen.dense1_w.data(), layer.mha.smolgen.dense1_b.data(),
        smolgen_activation, temp2.data());
    // Layer Norm + skip connection.
    LayerNorm2DWithSkipConnection(batch_size, hidden_sz, temp2.data(), 0.0f,
                                  (const float*)nullptr,
                                  layer.mha.smolgen.ln1_gammas.data(),
                                  layer.mha.smolgen.ln1_betas.data(), 1e-3);

    // Dense 2.
    const auto gen_sz_outputs = layer.mha.smolgen.dense2_b.size();
    std::vector<float> temp3(batch_size * gen_sz_outputs);
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, hidden_sz, gen_sz_outputs, temp2.data(),
        layer.mha.smolgen.dense2_w.data(), layer.mha.smolgen.dense2_b.data(),
        smolgen_activation, temp3.data());
    // Layer Norm + skip connection.
    LayerNorm2DWithSkipConnection(batch_size, gen_sz_outputs, temp3.data(),
                                  0.0f, (const float*)nullptr,
                                  layer.mha.smolgen.ln2_gammas.data(),
                                  layer.mha.smolgen.ln2_betas.data(), 1e-3);

    // Global smolgen weights.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * heads, gen_sz_outputs / heads, kSquares * kSquares, temp3.data(),
        weights_.smolgen_w.data(), (const float*)nullptr, NONE, QK);
  }

  // Q
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, d_model, head_buffer.data(),
      layer.mha.q_w.data(), layer.mha.q_b.data(), NONE, head_buffer2.data());
  // K
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, d_model, head_buffer.data(),
      layer.mha.k_w.data(), layer.mha.k_b.data(), NONE, head_buffer3.data());

  // MHA (Q, K, V)
  const int depth = d_model / heads;
  const float scaling = 1.0f / sqrtf(depth);

  // MHA is done per batch since there's a fourth dimension introduced.
  for (auto batch = size_t{0}; batch < batch_size; batch++) {
    auto batchStart = batch * kSquares * d_model;

    float* QK = &head_buffer4[batch * kSquares * kSquares * heads];

    const float* Q = &head_buffer2[batchStart];
    const float* K = &head_buffer3[batchStart];

    // matmul(Q, K) for all heads per batch.

    for (auto h = 0; h < heads; h++) {
      const float* A = &Q[h * depth];
      const float* B = &K[h * depth];
      float* C = &QK[h * kSquares * kSquares];
      const float beta = layer.mha.has_smolgen ? 1.0f : 0.0f;
      if (use_eigen) {
        auto C_mat = EigenMatrixMap<float>(C, kSquares, kSquares);
        C_mat.noalias() =
            beta * C_mat +
            scaling *
                ConstEigenStridedMatrixMap<float>(
                    B, depth, kSquares, Eigen::OuterStride<>(heads * depth))
                    .transpose() *
                ConstEigenStridedMatrixMap<float>(
                    A, depth, kSquares, Eigen::OuterStride<>(heads * depth));
      } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kSquares, kSquares,
                    depth, scaling, A, heads * depth, B, heads * depth, beta, C,
                    kSquares);
#else
        // Should never get here.
        throw Exception("Blas backend internal error");
#endif
      }
    }
  }

  // Apply Softmax.
  float* QK = &head_buffer4[0];
  for (size_t h = 0; h < batch_size * heads * kSquares * kSquares;
       h += kSquares) {
#if defined(USE_ISPC)
    if (!use_eigen) {
      ispc::SoftmaxActivation(kSquares, QK + h, QK + h);
      continue;
    }
#endif
    SoftmaxActivation(kSquares, QK + h, QK + h);
  }

  // V
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, d_model, head_buffer.data(),
      layer.mha.v_w.data(), layer.mha.v_b.data(), NONE, head_buffer3.data());

  for (auto batch = size_t{0}; batch < batch_size; batch++) {
    auto batchStart = batch * kSquares * d_model;
    // matmul(softmax(QK), V) for all heads per batch.
    float* attn = &head_buffer2[batchStart];
    const float* V = &head_buffer3[batchStart];
    const float* QK = &head_buffer4[batch * kSquares * kSquares * heads];
    for (auto h = 0; h < heads; h++) {
      const float* A = &QK[h * kSquares * kSquares];
      const float* B = &V[h * depth];
      float* C = &attn[h * depth];
      if (use_eigen) {
        auto C_mat = EigenStridedMatrixMap<float>(
            C, depth, kSquares, Eigen::OuterStride<>(heads * depth));
        C_mat.noalias() =
            ConstEigenStridedMatrixMap<float>(
                B, depth, kSquares, Eigen::OuterStride<>(heads * depth)) *
            ConstEigenMatrixMap<float>(A, kSquares, kSquares);
      } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kSquares, depth,
                    kSquares, 1.0f, A, kSquares, B, heads * depth, 0.0f, C,
                    heads * depth);
#endif
      }
    }
  }

  // Fully connected final MHA layer.
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, d_model, embedding_size, head_buffer2.data(),
      layer.mha.dense_w.data(), layer.mha.dense_b.data(), NONE,
      head_buffer3.data());

  // Layer Norm + skip connection.
  LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size,
                                head_buffer.data(), 1.0f / alpha,
                                head_buffer3.data(), layer.ln1_gammas.data(),
                                layer.ln1_betas.data(), 1e-6);

  // FFN.
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, dff_size, head_buffer.data(),
      layer.ffn.dense1_w.data(), layer.ffn.dense1_b.data(), ffn_activation,
      head_buffer4.data());

  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, dff_size, layer.ffn.dense2_b.size(),
      head_buffer4.data(), layer.ffn.dense2_w.data(), layer.ffn.dense2_b.data(),
      NONE, head_buffer3.data());

  // Layer Norm + skip connection.
  LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size,
                                head_buffer.data(), 1.0f / alpha,
                                head_buffer3.data(), layer.ln2_gammas.data(),
                                layer.ln2_betas.data(), 1e-6);
}

template <bool use_eigen>
void BlasComputation<use_eigen>::ComputeBlocking() {
  // Retrieve network key dimensions from the weights structure.
  const auto num_value_channels = weights_.ip1_val_b.size();
  const auto num_moves_channels = weights_.ip1_mov_b.size();
  const auto num_value_input_planes =
      attn_body_ ? weights_.ip_val_b.size() : weights_.value.biases.size();
  const auto num_policy_input_planes = weights_.policy.biases.size();
  const auto num_moves_input_planes =
      attn_body_ ? weights_.ip_mov_b.size() : weights_.moves_left.biases.size();
  const auto num_output_policy = static_cast<size_t>(kPolicyOutputs);
  const auto output_channels =
      attn_body_ ? weights_.ip_emb_b.size() : weights_.input.biases.size();
  const auto num_res_blocks = weights_.residual.size();

  // max_channels is the maximum number of input channels of any
  // convolution.
  // Residual blocks are identical, but the first convolution might be bigger
  // when the network has very few filters
  const auto input_channels = static_cast<size_t>(
      kInputPlanes + (attn_body_ ? kNumPosEncodingChannels : 0));
  const auto max_channels = std::max(output_channels, input_channels);

  // The policy head may increase convolution max output size.
  const auto max_output_channels =
      (conv_policy_ && weights_.policy.biases.size() > output_channels)
          ? weights_.policy.biases.size()
          : output_channels;

  // Determine the largest batch for allocations.
  const auto total_batches = planes_.size();
  const auto largest_batch_size = std::min(max_batch_size_, total_batches);

  /* Typically
   input_channels = 112
   output_channels = 192
   max_channels = 192
   num_value_input_planes = 32
   num_policy_input_planes = 32
   num_value_channels = 128
   num_output_policy = 1858
   */

  // Allocate data for the whole batch.
  size_t max_fc_channels = std::max(
      num_value_channels, std::max(num_output_policy, num_moves_channels));
  std::vector<float> output_fc(largest_batch_size * max_fc_channels);

  std::vector<float> res_buffer1(largest_batch_size * max_channels * kSquares);
  std::vector<float> res_buffer2(largest_batch_size * max_channels * kSquares);
  std::vector<float> res_buffer3(largest_batch_size * max_channels * kSquares);

  WinogradConvolution3<use_eigen> convolve3(largest_batch_size, max_channels,
                                            max_output_channels);

  size_t max_head_planes =
      std::max(num_policy_input_planes,
               std::max(num_value_input_planes, num_moves_input_planes));
  if (attn_policy_) {
    max_head_planes = std::max(std::max(max_head_planes, size_t{67}),
                               weights_.ip_pol_b.size());
  }
  std::vector<float> head_buffer(largest_batch_size * max_head_planes *
                                 kSquares);

  // These ones will rotate during the computation.
  float* conv_in = res_buffer1.data();
  float* conv_out = res_buffer2.data();
  float* res = res_buffer3.data();

  for (size_t i = 0; i < total_batches; i += largest_batch_size) {
    const auto batch_size = std::min(total_batches - i, largest_batch_size);
    for (size_t j = 0; j < batch_size; j++) {
      EncodePlanes(planes_[i + j], &conv_in[j * kSquares * kInputPlanes]);
    }

    if (num_res_blocks > 0) {
      // Input convolution

      convolve3.Forward(batch_size, kInputPlanes, output_channels, conv_in,
                        weights_.input.weights.data(), conv_out);

      BiasActivate(batch_size, output_channels, conv_out,
                   weights_.input.biases.data(), default_activation_);

      // Residual tower

      for (auto& residual : weights_.residual) {
        const auto& conv1 = residual.conv1;
        const auto& conv2 = residual.conv2;
        const auto& se = residual.se;

        std::swap(conv_out, conv_in);

        convolve3.Forward(batch_size, output_channels, output_channels, conv_in,
                          conv1.weights.data(), conv_out);

        BiasActivate(batch_size, output_channels, &conv_out[0],
                     conv1.biases.data(), default_activation_);

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);

        convolve3.Forward(batch_size, output_channels, output_channels, conv_in,
                          conv2.weights.data(), conv_out);

        if (residual.has_se) {
          // No relu if followed by SE-unit and residual/bias is added later
          std::swap(conv_out, conv_in);

          auto se_fc_outputs = se.b1.size();
          ApplySEUnit<use_eigen>(batch_size, output_channels, se_fc_outputs,
                                 conv_in, conv2.biases.data(), res,
                                 se.w1.data(), se.b1.data(), se.w2.data(),
                                 se.b2.data(), conv_out, default_activation_);
        } else {
          BiasResidual(batch_size, output_channels, &conv_out[0],
                       conv2.biases.data(), res, default_activation_);
        }
      }
    }

    if (attn_body_) {
      const auto embedding_size = weights_.ip_emb_b.size();
      assert(embedding_size > 0);
      const auto input_size =
          num_res_blocks == 0 ? input_channels : weights_.input.biases.size();

      if (num_res_blocks == 0) {
        // No residual means pure transformer, so process input position
        // encoding.
        // Preprocess for attention body.
        for (auto batch = size_t{0}; batch < batch_size; batch++) {
          for (auto i = 0; i < kSquares; i++) {
            // NCHW to NHWC conversion.
            for (size_t j = 0; j < kInputPlanes; j++) {
              res[batch * kSquares * input_size + i * input_size + j] =
                  conv_in[batch * kSquares * kInputPlanes + j * kSquares + i];
            }
            // Position encoding.
            for (size_t j = kInputPlanes; j < input_size; j++) {
              res[batch * kSquares * input_size + i * input_size + j] =
                  kPosEncoding[i][j - kInputPlanes];
            }
          }
        }
      }

      // Input embedding.
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, input_size, embedding_size, res_buffer3.data(),
          weights_.ip_emb_w.data(), weights_.ip_emb_b.data(),
          default_activation_, res_buffer1.data());

      // Input gating
      if (weights_.ip_mult_gate.size() > 0 && weights_.ip_add_gate.size() > 0) {
        int idx;
        for (auto batch = size_t{0}; batch < batch_size; batch++) {
          for (auto i = 0; i < kSquares; i++) {
            for (size_t j = 0; j < embedding_size; j++) {
              idx = batch * kSquares * embedding_size + i * embedding_size + j;
              res_buffer1[idx] =
                  res_buffer1[idx] * weights_.ip_mult_gate[j * kSquares + i] +
                  weights_.ip_add_gate[j * kSquares + i];
            }
          }
        };
      }

      // Attention body encoders.
      float alpha = (float)pow(2.0 * weights_.encoder.size(), 0.25);
      for (auto& layer : weights_.encoder) {
        MakeEncoderLayer(res_buffer1, res_buffer2, res_buffer3, batch_size,
                         layer, embedding_size, weights_.encoder_head_count,
                         smolgen_activation_, ffn_activation_, alpha);
      }

      res = res_buffer1.data();
      conv_in = res_buffer2.data();
      conv_out = res_buffer3.data();
    }

    // Need to preserve conv_out which is used for value and moves left heads.
    if (attn_policy_) {
      if (!attn_body_) {
        // NCHW to NHWC conversion.
        for (auto batch = size_t{0}; batch < batch_size; batch++) {
          for (auto i = 0; i < kSquares; i++) {
            for (size_t j = 0; j < output_channels; j++) {
              res[batch * kSquares * output_channels + i * output_channels + j] =
                  conv_out[batch * kSquares * output_channels + j * kSquares + i];
            }
          }
        }
      }
      const size_t policy_embedding_size = weights_.ip_pol_b.size();
      // Policy Embedding.
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, output_channels, policy_embedding_size, res,
          weights_.ip_pol_w.data(), weights_.ip_pol_b.data(),
          attn_body_ ? default_activation_
                     : SELU,  // SELU activation hardcoded for apmish nets.
          head_buffer.data());

      const size_t policy_d_model = weights_.ip2_pol_b.size();
      const size_t max_channel_size =
          weights_.pol_encoder.size() > 0
              ? weights_.pol_encoder[0].ffn.dense1_b.size()  // DFF size
              : policy_d_model;
      std::vector<float> head_buffer2(largest_batch_size * max_channel_size *
                                      kSquares);
      std::vector<float> head_buffer3(largest_batch_size * max_channel_size *
                                      kSquares);

      for (auto& layer : weights_.pol_encoder) {
        MakeEncoderLayer(head_buffer, head_buffer2, head_buffer3, batch_size,
                         layer, policy_embedding_size,
                         weights_.pol_encoder_head_count,
                         attn_body_ ? smolgen_activation_ : NONE,
                         attn_body_ ? ffn_activation_ : SELU, 1.0f);
      }

      // Q
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, policy_embedding_size, policy_d_model,
          head_buffer.data(), weights_.ip2_pol_w.data(),
          weights_.ip2_pol_b.data(), NONE, head_buffer2.data());
      // K
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, policy_embedding_size, policy_d_model,
          head_buffer.data(), weights_.ip3_pol_w.data(),
          weights_.ip3_pol_b.data(), NONE, head_buffer3.data());
      const float scaling = 1.0f / sqrtf(policy_d_model);
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        const float* A = &head_buffer2[batch * 64 * policy_d_model];
        const float* B = &head_buffer3[batch * 64 * policy_d_model];
        float* C = &head_buffer[batch * (64 * 64 + 8 * 24)];
        if (use_eigen) {
          auto C_mat = EigenMatrixMap<float>(C, kSquares, kSquares);
          C_mat.noalias() =
              scaling *
              ConstEigenMatrixMap<float>(B, policy_d_model, kSquares)
                  .transpose() *
              ConstEigenMatrixMap<float>(A, policy_d_model, kSquares);
        } else {
#ifdef USE_BLAS
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kSquares,
                      kSquares, policy_d_model, scaling, A, policy_d_model, B,
                      policy_d_model, 0.0f, C, 64);
#else
          // Should never get here.
          throw Exception("Blas backend internal error");
#endif
        }
      }
      // Promotion offset calculation.
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        float promotion_offsets[4][8];
        // This is so small that SGEMM seems slower.
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (size_t k = 0; k < policy_d_model; k++) {
              sum += head_buffer3.data()[batch * kSquares * policy_d_model +
                                         (56 + j) * policy_d_model + k] *
                     weights_.ip4_pol_w.data()[i * policy_d_model + k];
            }
            promotion_offsets[i][j] = sum;
          }
        }
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 8; j++) {
            promotion_offsets[i][j] += promotion_offsets[3][j];
          }
        }
        for (int k = 0; k < 8; k++) {      // y in cuda
          for (int j = 0; j < 8; j++) {    // w in cuda
            for (int i = 0; i < 3; i++) {  // c in cuda
              head_buffer.data()[batch * (64 * 64 + 8 * 24) + 64 * 64 + 24 * k +
                                 3 * j + i] =
                  head_buffer.data()[batch * (64 * 64 + 8 * 24) +
                                     (48 + k) * 64 + 56 + j] +
                  promotion_offsets[i][j];
            }
          }
        }
      }
      // Mapping from attention policy to lc0 policy
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        for (auto i = 0; i < 64 * 64 + 8 * 24; i++) {
          auto j = kAttnPolicyMap[i];
          if (j >= 0) {
            output_fc[batch * num_output_policy + j] =
                head_buffer[batch * (64 * 64 + 8 * 24) + i];
          }
        }
      }
    } else if (conv_policy_) {
      assert(!attn_body_);  // not supported with attention body
      convolve3.Forward(batch_size, output_channels, output_channels, conv_out,
                        weights_.policy1.weights.data(), res);

      BiasActivate(batch_size, output_channels, &res[0],
                   weights_.policy1.biases.data(), default_activation_);

      convolve3.Forward(batch_size, output_channels, num_policy_input_planes,
                        res, weights_.policy.weights.data(),
                        head_buffer.data());

      BiasActivate(batch_size, num_policy_input_planes, &head_buffer.data()[0],
                   weights_.policy.biases.data(), NONE);

      // Mapping from convolutional policy to lc0 policy
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        for (auto i = 0; i < kPolicyUsedPlanes * kSquares; i++) {
          auto j = kConvPolicyMap[i];
          if (j >= 0) {
            output_fc[batch * num_output_policy + j] =
                head_buffer[batch * num_policy_input_planes * kSquares + i];
          }
        }
      }

    } else {
      assert(!attn_body_);  // not supported with attention body
      Convolution1<use_eigen>::Forward(
          batch_size, output_channels, num_policy_input_planes, conv_out,
          weights_.policy.weights.data(), head_buffer.data());

      BiasActivate(batch_size, num_policy_input_planes, &head_buffer[0],
                   weights_.policy.biases.data(), default_activation_);

      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_policy_input_planes * kSquares, num_output_policy,
          head_buffer.data(), weights_.ip_pol_w.data(),
          weights_.ip_pol_b.data(),
          NONE,  // Activation Off
          output_fc.data());
    }

    for (size_t j = 0; j < batch_size; j++) {
      std::vector<float> policy(num_output_policy);

      // Get the moves
      policy.assign(output_fc.begin() + j * num_output_policy,
                    output_fc.begin() + (j + 1) * num_output_policy);
      policies_.emplace_back(std::move(policy));
    }

    // Value head
    if (attn_body_) {
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, weights_.ip_emb_b.size(),
          num_value_input_planes, res, weights_.ip_val_w.data(),
          weights_.ip_val_b.data(), default_activation_, head_buffer.data());
    } else {
      Convolution1<use_eigen>::Forward(
          batch_size, output_channels, num_value_input_planes, conv_out,
          weights_.value.weights.data(), head_buffer.data());

      BiasActivate(batch_size, num_value_input_planes, &head_buffer[0],
                   weights_.value.biases.data(), default_activation_);
    }

    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, num_value_input_planes * kSquares, num_value_channels,
        head_buffer.data(), weights_.ip1_val_w.data(),
        weights_.ip1_val_b.data(),
        default_activation_,  // Activation On
        output_fc.data());

    // Now get the score
    if (wdl_) {
      std::vector<float> wdl(3 * batch_size);
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_value_channels, 3, output_fc.data(),
          weights_.ip2_val_w.data(), weights_.ip2_val_b.data(),
          NONE,  // Activation Off
          wdl.data());

      for (size_t j = 0; j < batch_size; j++) {
        std::vector<float> wdl_softmax(3);
        SoftmaxActivation(3, &wdl[j * 3], wdl_softmax.data());

        q_values_.emplace_back(wdl_softmax[0]);
        q_values_.emplace_back(wdl_softmax[1]);
        q_values_.emplace_back(wdl_softmax[2]);
      }
    } else {
      for (size_t j = 0; j < batch_size; j++) {
        double winrate = FullyConnectedLayer<use_eigen>::Forward0D(
                             num_value_channels, weights_.ip2_val_w.data(),
                             &output_fc[j * num_value_channels]) +
                         weights_.ip2_val_b[0];

        q_values_.emplace_back(std::tanh(winrate));
      }
    }
    if (moves_left_) {
      if (attn_body_) {
        FullyConnectedLayer<use_eigen>::Forward1D(
            batch_size * kSquares, weights_.ip_emb_b.size(),
            num_moves_input_planes, res, weights_.ip_mov_w.data(),
            weights_.ip_mov_b.data(), default_activation_, head_buffer.data());
      } else {
        Convolution1<use_eigen>::Forward(
            batch_size, output_channels, num_moves_input_planes, conv_out,
            weights_.moves_left.weights.data(), head_buffer.data());

        BiasActivate(batch_size, num_moves_input_planes, &head_buffer[0],
                     weights_.moves_left.biases.data(), default_activation_);
      }

      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_moves_input_planes * kSquares, num_moves_channels,
          head_buffer.data(), weights_.ip1_mov_w.data(),
          weights_.ip1_mov_b.data(),
          default_activation_,  // Activation On
          output_fc.data());

      std::vector<float> output_moves_left(batch_size);
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_moves_channels, 1, output_fc.data(),
          weights_.ip2_mov_w.data(), weights_.ip2_mov_b.data(),
          RELU,  // Specifically Relu
          output_moves_left.data());

      for (size_t j = 0; j < batch_size; j++) {
        m_values_.emplace_back(output_moves_left[j]);
      }
    }
  }
}

template <bool use_eigen>
void BlasComputation<use_eigen>::EncodePlanes(const InputPlanes& sample,
                                              float* buffer) {
  for (const InputPlane& plane : sample) {
    const float value = plane.value;
    for (auto i = 0; i < kSquares; i++)
      *(buffer++) = (plane.mask & (((uint64_t)1) << i)) != 0 ? value : 0;
  }
}

template <bool use_eigen>
BlasNetwork<use_eigen>::BlasNetwork(const WeightsFile& file,
                                    const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().moves_left()},
      weights_(file.weights()) {
  Numa::Init();

  max_batch_size_ =
      static_cast<size_t>(options.GetOrDefault<int>("batch_size", 256));

  wdl_ = file.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;

  moves_left_ = (file.format().network_format().moves_left() ==
                 pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                options.GetOrDefault<bool>("mlh", true);

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  attn_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_ATTENTION;

  attn_body_ = file.format().network_format().network() ==
               pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT;

  default_activation_ = file.format().network_format().default_activation() ==
                                pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH
                            ? MISH
                            : RELU;

  if (attn_body_) {
    const auto smol_act = file.format().network_format().smolgen_activation();
    smolgen_activation_ =
        smol_act == pblczero::NetworkFormat::SMOLGEN_ACTIVATION_INHERIT
            ? default_activation_
            : static_cast<ActivationFunction>(smol_act);
    const auto ffn_act = file.format().network_format().ffn_activation();
    ffn_activation_ = ffn_act == pblczero::NetworkFormat::FFN_ACTIVATION_INHERIT
                          ? default_activation_
                          : static_cast<ActivationFunction>(ffn_act);
  }

  if (max_batch_size_ > kHardMaxBatchSize) {
    max_batch_size_ = kHardMaxBatchSize;
  }

  const auto inputChannels = kInputPlanes;
  const auto channels = static_cast<int>(weights_.input.biases.size());
  const auto residual_blocks = weights_.residual.size();

  weights_.input.weights =
      WinogradFilterTransformF(weights_.input.weights, channels, inputChannels);

  // residual blocks
  for (size_t i = 0; i < residual_blocks; i++) {
    auto& residual = weights_.residual[i];
    auto& conv1 = residual.conv1;
    auto& conv2 = residual.conv2;

    conv1.weights = WinogradFilterTransformF(conv1.weights, channels, channels);
    conv2.weights = WinogradFilterTransformF(conv2.weights, channels, channels);
  }

  if (conv_policy_) {
    weights_.policy1.weights =
        WinogradFilterTransformF(weights_.policy1.weights, channels, channels);
    auto pol_channels = weights_.policy.biases.size();
    weights_.policy.weights = WinogradFilterTransformF(weights_.policy.weights,
                                                       pol_channels, channels);
  }

  if (use_eigen) {
    CERR << "Using Eigen version " << EIGEN_WORLD_VERSION << "."
         << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION;
    CERR << "Eigen max batch size is " << max_batch_size_ << ".";
  } else {
#ifdef USE_OPENBLAS
    int num_procs = openblas_get_num_procs();
    openblas_set_num_threads(1);
    const char* core_name = openblas_get_corename();
    const char* config = openblas_get_config();
    CERR << "BLAS vendor: OpenBLAS.";
    CERR << "OpenBLAS [" << config << "].";
    CERR << "OpenBLAS found " << num_procs << " " << core_name << " core(s).";
#endif

#ifdef USE_MKL
    mkl_set_num_threads(1);
    CERR << "BLAS vendor: MKL.";
    constexpr int len = 256;
    char versionbuf[len];
    mkl_get_version_string(versionbuf, len);
    CERR << "MKL " << versionbuf << ".";
    MKLVersion version;
    mkl_get_version(&version);
    CERR << "MKL platform: " << version.Platform
         << ", processor: " << version.Processor << ".";
#endif

#ifdef USE_DNNL
    const dnnl_version_t* ver = dnnl_version();
    CERR << "BLAS functions from DNNL version " << ver->major << "."
         << ver->minor << "." << ver->patch;
#endif

#ifdef USE_ACCELERATE
    CERR << "BLAS vendor: Apple vecLib.";
#endif
    CERR << "BLAS max batch size is " << max_batch_size_ << ".";
  }
}

template <bool use_eigen>
std::unique_ptr<Network> MakeBlasNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& options) {
  if (!w) {
    throw Exception("The " + std::string(use_eigen ? "eigen" : "blas") +
                    " backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT) {
    throw Exception("Network format " +
                    pblczero::NetworkFormat::NetworkStructure_Name(
                        weights.format().network_format().network()) +
                    " is not supported by BLAS backend.");
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
                    " is not supported by BLAS backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    pblczero::NetworkFormat::ValueFormat_Name(
                        weights.format().network_format().value()) +
                    " is not supported by BLAS backend.");
  }
  if (weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
      weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
    throw Exception(
        "Default activation " +
        pblczero::NetworkFormat::DefaultActivation_Name(
            weights.format().network_format().default_activation()) +
        " is not supported by BLAS backend.");
  }

  // @todo Hack for old encoding compatibility. REMOVE BEFORE MERGING.
  if (w->format().network_format().network() ==
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT &&
      w->weights().encoder().size() > 0) {
    CERR << "Attention body detected, hacking network format.";
    WeightsFile x = *w;
    x.mutable_format()->mutable_network_format()->set_network(
        pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT);
    if (w->weights().has_smolgen_w()) {
      CERR << "BT2 detected, hacking activations.";
      x.mutable_format()->mutable_network_format()->set_ffn_activation(
          pblczero::NetworkFormat::FFN_ACTIVATION_RELU_2);
      x.mutable_format()->mutable_network_format()->set_smolgen_activation(
          pblczero::NetworkFormat::SMOLGEN_ACTIVATION_SWISH);
    }
    return std::make_unique<BlasNetwork<use_eigen>>(x, options);
  }

  return std::make_unique<BlasNetwork<use_eigen>>(weights, options);
}

#ifdef USE_BLAS
REGISTER_NETWORK("blas", MakeBlasNetwork<false>, 50)
#endif
REGISTER_NETWORK("eigen", MakeBlasNetwork<true>, 49)

}  // namespace
}  // namespace lczero
