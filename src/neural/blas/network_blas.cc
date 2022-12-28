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

namespace lczero {
namespace {

template <bool use_eigen>
class BlasComputation : public NetworkComputation {
 public:
  BlasComputation(const LegacyWeights& weights, const size_t max_batch_size,
                  const bool wdl, const bool moves_left, const bool conv_policy,
                  const ActivationFunction default_activation,
                  const bool attn_policy);

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
  bool attn_policy_;
};

template <bool use_eigen>
class BlasNetwork : public Network {
 public:
  BlasNetwork(const WeightsFile& weights, const OptionsDict& options);
  virtual ~BlasNetwork(){};

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasComputation<use_eigen>>(
        weights_, max_batch_size_, wdl_, moves_left_, conv_policy_,
        default_activation_, attn_policy_);
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
  bool attn_policy_;
};

template <bool use_eigen>
BlasComputation<use_eigen>::BlasComputation(
    const LegacyWeights& weights, const size_t max_batch_size, const bool wdl,
    const bool moves_left, const bool conv_policy,
    const ActivationFunction default_activation, const bool attn_policy)
    : weights_(weights),
      max_batch_size_(max_batch_size),
      policies_(0),
      q_values_(0),
      wdl_(wdl),
      moves_left_(moves_left),
      conv_policy_(conv_policy),
      default_activation_(default_activation),
      attn_policy_(attn_policy) {
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

template <bool use_eigen>
void BlasComputation<use_eigen>::ComputeBlocking() {
  // Retrieve network key dimensions from the weights structure.
  const auto num_value_channels = weights_.ip1_val_b.size();
  const auto num_moves_channels = weights_.ip1_mov_b.size();
  const auto num_value_input_planes = weights_.value.biases.size();
  const auto num_policy_input_planes = weights_.policy.biases.size();
  const auto num_moves_input_planes = weights_.moves_left.biases.size();
  const auto num_output_policy = static_cast<size_t>(kPolicyOutputs);
  const auto output_channels = weights_.input.biases.size();

  // max_channels is the maximum number of input channels of any
  // convolution.
  // Residual blocks are identical, but the first convolution might be bigger
  // when the network has very few filters
  const auto input_channels = static_cast<size_t>(kInputPlanes);
  const auto max_channels = std::max(output_channels, input_channels);

  // The policy head may increase convolution max output size.
  const auto max_output_channels =
      (conv_policy_ && weights_.policy.biases.size() > output_channels)
          ? weights_.policy.biases.size()
          : output_channels;

  // Determine the largest batch for allocations.
  const auto plane_count = planes_.size();
  const auto largest_batch_size = std::min(max_batch_size_, plane_count);

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
  std::vector<float> res_buffer2(largest_batch_size * output_channels *
                                 kSquares);
  std::vector<float> res_buffer3(largest_batch_size * output_channels *
                                 kSquares);

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

  for (size_t i = 0; i < plane_count; i += largest_batch_size) {
    const auto batch_size = std::min(plane_count - i, largest_batch_size);
    for (size_t j = 0; j < batch_size; j++) {
      EncodePlanes(planes_[i + j], &conv_in[j * kSquares * kInputPlanes]);
    }

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
                               conv_in, conv2.biases.data(), res, se.w1.data(),
                               se.b1.data(), se.w2.data(), se.b2.data(),
                               conv_out, default_activation_);
      } else {
        BiasResidual(batch_size, output_channels, &conv_out[0],
                     conv2.biases.data(), res, default_activation_);
      }
    }

    // Need to preserve conv_out which is used for value and moves left heads.
    if (attn_policy_) {
      // NCHW to NHWC conversion.
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        for (auto i = 0; i < kSquares; i++) {
          for (size_t j = 0; j < output_channels; j++) {
            res[batch * kSquares * output_channels + i * output_channels + j] =
                conv_out[batch * kSquares * output_channels + j * kSquares + i];
          }
        }
      }
      const size_t embedding_size = weights_.ip_pol_b.size();
      // Embedding.
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, output_channels, embedding_size, res,
          weights_.ip_pol_w.data(), weights_.ip_pol_b.data(),
          SELU,  // SELU activation for attention head.
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

      if (weights_.pol_encoder.size() > 0) {
        std::vector<float> head_buffer4(largest_batch_size * max_channel_size *
                                        kSquares);
        std::vector<float> temp_buffer1(policy_d_model * kSquares);
        std::vector<float> temp_buffer2(policy_d_model * kSquares);
        std::vector<float> temp_buffer3(policy_d_model * kSquares);

        for (auto layer : weights_.pol_encoder) {
          // Q
          FullyConnectedLayer<use_eigen>::Forward1D(
              batch_size * kSquares, embedding_size, layer.mha.q_b.size(),
              head_buffer.data(), layer.mha.q_w.data(), layer.mha.q_b.data(),
              NONE, head_buffer2.data());
          // K
          FullyConnectedLayer<use_eigen>::Forward1D(
              batch_size * kSquares, embedding_size, layer.mha.k_b.size(),
              head_buffer.data(), layer.mha.k_w.data(), layer.mha.k_b.data(),
              NONE, head_buffer3.data());
          // V
          FullyConnectedLayer<use_eigen>::Forward1D(
              batch_size * kSquares, embedding_size, layer.mha.v_b.size(),
              head_buffer.data(), layer.mha.v_w.data(), layer.mha.v_b.data(),
              NONE, head_buffer4.data());

          // MHA (Q, K, V)
          const int d_model = layer.mha.q_b.size();
          const int heads = weights_.pol_encoder_head_count;
          const int depth = d_model / heads;
          const float scaling = 1.0f / sqrtf(depth);

          // MHA is done per batch since there's a fourth dimension introduced.
          for (auto batch = size_t{0}; batch < batch_size; batch++) {
            auto batchStart = batch * kSquares * d_model;

            // Reshape and transpose for each head.
            const float* Q = temp_buffer1.data();
            const float* K = temp_buffer2.data();
            const float* V = temp_buffer3.data();

            for (int head = 0; head < heads; head++) {
              for (int j = 0; j < kSquares; j++) {
                auto channelStart = batchStart + j * d_model + head * depth;
                auto transposeStart = head * kSquares * depth + j * depth;
                std::copy(head_buffer2.begin() + channelStart,
                          head_buffer2.begin() + channelStart + depth,
                          temp_buffer1.begin() + transposeStart);
                std::copy(head_buffer3.begin() + channelStart,
                          head_buffer3.begin() + channelStart + depth,
                          temp_buffer2.begin() + transposeStart);
                std::copy(head_buffer4.begin() + channelStart,
                          head_buffer4.begin() + channelStart + depth,
                          temp_buffer3.begin() + transposeStart);
              }
            }

            // matmul(Q, K) for all heads per batch.
            float* QK = &head_buffer2[batchStart];
            AttentionMatmul2D<use_eigen>(false, true, heads, kSquares, kSquares,
                                         depth, scaling, Q, K, QK);

            // Apply Softmax.
            for (int h = 0; h < heads * kSquares * kSquares; h += kSquares) {
              SoftmaxActivation(kSquares, QK + h, QK + h);
            }

            // matmul(softmax(QK), V) for all heads per batch.
            float* attn = &head_buffer3[batchStart];
            AttentionMatmul2D<use_eigen>(false, false, heads, kSquares, depth,
                                         kSquares, 1.0, QK, V, attn);

            // Transpose back into N x 64 x H x D.
            for (int j = 0; j < kSquares; j++) {
              for (int head = 0; head < heads; head++) {
                auto transposeStart =
                    batchStart + head * kSquares * depth + j * depth;
                std::copy(head_buffer3.begin() + transposeStart,
                          head_buffer3.begin() + transposeStart + depth,
                          head_buffer2.begin() + batchStart + j * d_model +
                              head * depth);
              }
            }
          }

          // Fully connected final MHA layer.
          FullyConnectedLayer<use_eigen>::Forward1D(
              batch_size * kSquares, d_model, embedding_size,
              head_buffer2.data(), layer.mha.dense_w.data(),
              layer.mha.dense_b.data(), NONE, head_buffer3.data());

          // Layer Norm + skip connection.
          LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size,
                                        head_buffer.data(), head_buffer3.data(),
                                        layer.ln1_gammas.data(),
                                        layer.ln1_betas.data(), 1e-6);

          // FFN.
          const size_t dff_size = layer.ffn.dense1_b.size();
          FullyConnectedLayer<use_eigen>::Forward1D(
              batch_size * kSquares, embedding_size, dff_size,
              head_buffer.data(), layer.ffn.dense1_w.data(),
              layer.ffn.dense1_b.data(), SELU, head_buffer2.data());

          FullyConnectedLayer<use_eigen>::Forward1D(
              batch_size * kSquares, dff_size, layer.ffn.dense2_b.size(),
              head_buffer2.data(), layer.ffn.dense2_w.data(),
              layer.ffn.dense2_b.data(), NONE, head_buffer3.data());

          // Layer Norm + skip connection.
          LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size,
                                        head_buffer.data(), head_buffer3.data(),
                                        layer.ln2_gammas.data(),
                                        layer.ln2_betas.data(), 1e-6);
        }
      }

      // Q
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, embedding_size, policy_d_model,
          head_buffer.data(), weights_.ip2_pol_w.data(),
          weights_.ip2_pol_b.data(), NONE, head_buffer2.data());
      // K
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, embedding_size, policy_d_model,
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
    Convolution1<use_eigen>::Forward(
        batch_size, output_channels, num_value_input_planes, conv_out,
        weights_.value.weights.data(), head_buffer.data());

    BiasActivate(batch_size, num_value_input_planes, &head_buffer[0],
                 weights_.value.biases.data(), default_activation_);

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
      Convolution1<use_eigen>::Forward(
          batch_size, output_channels, num_moves_input_planes, conv_out,
          weights_.moves_left.weights.data(), head_buffer.data());

      BiasActivate(batch_size, num_moves_input_planes, &head_buffer[0],
                   weights_.moves_left.biases.data(), default_activation_);

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

  default_activation_ = file.format().network_format().default_activation() ==
                                pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH
                            ? MISH
                            : RELU;

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
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
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
  return std::make_unique<BlasNetwork<use_eigen>>(weights, options);
}

#ifdef USE_BLAS
REGISTER_NETWORK("blas", MakeBlasNetwork<false>, 50)
#endif
REGISTER_NETWORK("eigen", MakeBlasNetwork<true>, 49)

}  // namespace
}  // namespace lczero
