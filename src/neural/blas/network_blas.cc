/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2020 The LCZero Authors

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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "neural/blas/blas.h"
#include "neural/blas/convolution1.h"
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

#include <Eigen/Core>

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
                  const ActivationFunction default_activation, const bool attn_policy);

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
        // No relu if followed by SE-unit and residual is added later
        BiasActivate(batch_size, output_channels, &conv_out[0],
                     conv2.biases.data(), NONE);

        std::swap(conv_out, conv_in);

        auto se_fc_outputs = se.b1.size();
        ApplySEUnit<use_eigen>(batch_size, output_channels, se_fc_outputs,
                               conv_in, res, se.w1.data(), se.b1.data(),
                               se.w2.data(), se.b2.data(), conv_out,
                               default_activation_);
      } else {
        BiasResidual(batch_size, output_channels, &conv_out[0],
                     conv2.biases.data(), res, default_activation_);
      }
    }

    if (attn_policy_) {
      // NCHW to NHWC conversion.
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        for (auto i = 0; i < kSquares; i++) {
          for (auto j = 0; j < output_channels; j++) {
            res[batch * kSquares * output_channels + i * output_channels + j] =
                conv_out[batch * kSquares * output_channels + j * kSquares + i];
          }
        }
      }
      const size_t embedding_size = weights_.ip_pol_b.size();
      // Embedding.
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, output_channels, embedding_size,
          res, weights_.ip_pol_w.data(),
          weights_.ip_pol_b.data(),
          SELU,  // SELU activation for attention head.
          head_buffer.data());

      for (auto layer : weights_.pol_encoder) {
        // TODO: support encoder heds.
        throw Exception("Eigen/Blas backend doesn't support encoder heads yet.");
      }
      const size_t policy_d_model = weights_.ip2_pol_b.size();
      std::vector<float> head_buffer2(largest_batch_size * policy_d_model *
                                     kSquares);
      std::vector<float> head_buffer3(largest_batch_size * policy_d_model *
                                      kSquares);
      // Q
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, embedding_size, policy_d_model, head_buffer.data(),
          weights_.ip2_pol_w.data(), weights_.ip2_pol_b.data(),
          NONE,
          head_buffer2.data());
      // K
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, embedding_size, policy_d_model,
          head_buffer.data(), weights_.ip3_pol_w.data(),
          weights_.ip3_pol_b.data(), NONE, head_buffer3.data());
      // TODO: perform dotproduct with Eigen/Blas rather than handrolled?
      const float scaling = sqrtf(policy_d_model);
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        for (auto i = 0; i < kSquares; i++) {
          for (auto j = 0; j < kSquares; j++) {
            float sum = 0.0f;
            for (auto k = 0; k < policy_d_model; k++) {
              sum += head_buffer2.data()[batch * 64 * policy_d_model +
                                         i * policy_d_model + k] *
                     head_buffer3.data()[batch * 64 * policy_d_model +
                                         j * policy_d_model + k];
            }
            head_buffer.data()[batch * (64 * 64 + 8 * 24) + i * 64 + j] =
                sum / scaling;
          }
        }
      }
      // Promotion offset calculation.
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        float promotion_offsets[4][8];
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (int k = 0; k < policy_d_model; k++) {
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
        for (auto i = 0; i < 64*64+8*24; i++) {
          auto j = kAttnPolicyMap[i];
          if (j >= 0) {
            output_fc[batch * num_output_policy + j] =
                head_buffer[batch * (64 * 64 + 8 * 24) + i];
          }
        }
      }
    } else if (conv_policy_) {
      // Need to preserve conv_out which is used for value head
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
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by BLAS backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_ATTENTION) {
    throw Exception("Policy format " +
                    std::to_string(weights.format().network_format().policy()) +
                    " is not supported by BLAS backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    std::to_string(weights.format().network_format().value()) +
                    " is not supported by BLAS backend.");
  }
  if (weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
      weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
    throw Exception(
        "Default activation " +
        std::to_string(weights.format().network_format().default_activation()) +
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
