/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2019 The LCZero Authors

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

#include "neural/blas/blas.h"
#include "neural/blas/convolution1.h"
#include "neural/blas/fully_connected_layer.h"
#include "neural/blas/se_unit.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "neural/network_legacy.h"
#include "neural/shared/activation.h"
#include "neural/shared/policy_map.h"
#include "neural/shared/winograd_filter.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace lczero {
namespace {

class BlasComputation : public NetworkComputation {
 public:
  BlasComputation(const LegacyWeights& weights, const size_t max_batch_size,
                  const bool wdl, const bool conv_policy);

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
  bool wdl_;
  bool conv_policy_;
};

class BlasNetwork : public Network {
 public:
  BlasNetwork(const WeightsFile& weights, const OptionsDict& options);
  virtual ~BlasNetwork(){};

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasComputation>(weights_, max_batch_size_, wdl_,
                                             conv_policy_);
  }

 private:
  // A cap on the max batch size since it consumes a lot of memory
  static constexpr auto kHardMaxBatchSize = 2048;

  LegacyWeights weights_;
  size_t max_batch_size_;
  bool wdl_;
  bool conv_policy_;
};

BlasComputation::BlasComputation(const LegacyWeights& weights,
                                 const size_t max_batch_size, const bool wdl,
                                 const bool conv_policy)
    : weights_(weights),
      max_batch_size_(max_batch_size),
      policies_(0),
      q_values_(0),
      wdl_(wdl),
      conv_policy_(conv_policy) {}

void BlasComputation::ComputeBlocking() {
  // Retrieve network key dimensions from the weights structure.
  const auto num_value_channels = weights_.ip1_val_b.size();
  const auto num_value_input_planes = weights_.value.biases.size();
  const auto num_policy_input_planes = weights_.policy.biases.size();
  const auto num_output_policy = kPolicyOutputs;
  const auto output_channels = weights_.input.biases.size();

  // max_channels is the maximum number of input channels of any
  // convolution.
  // Residual blocks are identical, but the first convolution might be bigger
  // when the network has very few filters
  const auto input_channels = static_cast<size_t>(kInputPlanes);
  const auto max_channels = std::max(output_channels, input_channels);

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
  std::vector<float> output_val(largest_batch_size * num_value_channels);
  std::vector<float> output_pol(largest_batch_size * num_output_policy);

  std::vector<float> res_buffer1(largest_batch_size * max_channels * kSquares);
  std::vector<float> res_buffer2(largest_batch_size * output_channels *
                                 kSquares);
  std::vector<float> res_buffer3(largest_batch_size * output_channels *
                                 kSquares);

  WinogradConvolution3 convolve3(largest_batch_size, max_channels,
                                 output_channels);

  std::vector<float> policy_buffer(largest_batch_size *
                                   num_policy_input_planes * kSquares);
  std::vector<float> value_buffer(largest_batch_size * num_value_input_planes *
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

    BiasResidualRelu(batch_size, output_channels, conv_out,
                     weights_.input.biases.data());

    // Residual tower

    for (auto& residual : weights_.residual) {
      const auto& conv1 = residual.conv1;
      const auto& conv2 = residual.conv2;
      const auto& se = residual.se;

      std::swap(conv_out, conv_in);

      convolve3.Forward(batch_size, output_channels, output_channels, conv_in,
                        conv1.weights.data(), conv_out);

      BiasResidualRelu(batch_size, output_channels, &conv_out[0],
                       conv1.biases.data());

      std::swap(conv_in, res);
      std::swap(conv_out, conv_in);

      convolve3.Forward(batch_size, output_channels, output_channels, conv_in,
                        conv2.weights.data(), conv_out);

      if (residual.has_se) {
        // No relu if followed by SE-unit and residual is added later
        BiasResidualRelu(batch_size, output_channels, &conv_out[0],
                         conv2.biases.data(), nullptr, false);

        std::swap(conv_out, conv_in);

        auto se_fc_outputs = se.b1.size();
        ApplySEUnit(batch_size, output_channels, se_fc_outputs, conv_in, res,
                    se.w1.data(), se.b1.data(), se.w2.data(), se.b2.data(),
                    conv_out);
      } else {
        BiasResidualRelu(batch_size, output_channels, &conv_out[0],
                         conv2.biases.data(), res);
      }
    }

    if (conv_policy_) {
      // Need to preserve conv_out which is used for value head
      convolve3.Forward(batch_size, output_channels, output_channels, conv_out,
                        weights_.policy1.weights.data(), res);

      BiasResidualRelu(batch_size, output_channels, &res[0],
                       weights_.policy1.biases.data());

      convolve3.Forward(batch_size, output_channels, num_policy_input_planes,
                        res, weights_.policy.weights.data(),
                        policy_buffer.data());

      BiasResidualRelu(batch_size, num_policy_input_planes,
                       &policy_buffer.data()[0], weights_.policy.biases.data(),
                       nullptr, false);

      // Mapping from convolutional policy to lc0 policy
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        for (auto i = 0; i < kPolicyUsedPlanes * kSquares; i++) {
          auto j = kConvPolicyMap[i];
          if (j >= 0) {
            output_pol[batch * num_output_policy + j] =
                policy_buffer[batch * num_policy_input_planes * kSquares + i];
          }
        }
      }

    } else {
      Convolution1::Forward(
          batch_size, output_channels, num_policy_input_planes, conv_out,
          weights_.policy.weights.data(), policy_buffer.data());

      BiasResidualRelu(batch_size, num_policy_input_planes, &policy_buffer[0],
                       weights_.policy.biases.data());

      FullyConnectedLayer::Forward1D(
          batch_size, num_policy_input_planes * kSquares, num_output_policy,
          policy_buffer.data(), weights_.ip_pol_w.data(),
          weights_.ip_pol_b.data(),
          false,  // Relu Off
          output_pol.data());
    }

    // Value head
    Convolution1::Forward(batch_size, output_channels, num_value_input_planes,
                          conv_out, weights_.value.weights.data(),
                          value_buffer.data());

    BiasResidualRelu(batch_size, num_value_input_planes, &value_buffer[0],
                     weights_.value.biases.data());

    FullyConnectedLayer::Forward1D(
        batch_size, num_value_input_planes * kSquares, num_value_channels,
        value_buffer.data(), weights_.ip1_val_w.data(),
        weights_.ip1_val_b.data(),
        true,  // Relu On
        output_val.data());

    for (size_t j = 0; j < batch_size; j++) {
      std::vector<float> policy(num_output_policy);

      // Get the moves
      SoftmaxActivation(num_output_policy, &output_pol[j * num_output_policy],
                        policy.data());

      policies_.emplace_back(std::move(policy));
    }

    // Now get the score
    if (wdl_) {
      std::vector<float> wdl(3 * batch_size);
      FullyConnectedLayer::Forward1D(
          batch_size, num_value_channels, 3, output_val.data(),
          weights_.ip2_val_w.data(), weights_.ip2_val_b.data(),
          false,  // Relu Off
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
        double winrate = FullyConnectedLayer::Forward0D(
                             num_value_channels, weights_.ip2_val_w.data(),
                             &output_val[j * num_value_channels]) +
                         weights_.ip2_val_b[0];

        q_values_.emplace_back(std::tanh(winrate));
      }
    }
  }
}

void BlasComputation::EncodePlanes(const InputPlanes& sample, float* buffer) {
  for (const InputPlane& plane : sample) {
    const float value = plane.value;
    for (auto i = 0; i < kSquares; i++)
      *(buffer++) = (plane.mask & (((uint64_t)1) << i)) != 0 ? value : 0;
  }
}

BlasNetwork::BlasNetwork(const WeightsFile& file, const OptionsDict& options)
    : weights_(file.weights()) {
  int blas_cores = options.GetOrDefault<int>("blas_cores", 1);
  max_batch_size_ =
      static_cast<size_t>(options.GetOrDefault<int>("batch_size", 256));

  wdl_ = file.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  if (max_batch_size_ > kHardMaxBatchSize) {
    max_batch_size_ = kHardMaxBatchSize;
  }
  std::cerr << "BLAS, maximum batch size set to " << max_batch_size_ << '\n';

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

#ifdef USE_OPENBLAS
  int num_procs = openblas_get_num_procs();
  blas_cores = std::min(num_procs, blas_cores);
  openblas_set_num_threads(blas_cores);
  const char* core_name = openblas_get_corename();
  const char* config = openblas_get_config();
  std::cerr << "BLAS vendor: OpenBlas.\n";
  std::cerr << "OpenBlas [" << config << "].\n";
  std::cerr << "OpenBlas found " << num_procs << " " << core_name
            << " core(s).\n";
  std::cerr << "OpenBLAS using " << blas_cores
            << " core(s) for this backend.\n";
#endif

#ifdef USE_MKL
  int max_procs = mkl_get_max_threads();
  blas_cores = std::min(max_procs, blas_cores);
  mkl_set_num_threads(blas_cores);
  std::cerr << "BLAS vendor: MKL.\n";
  constexpr int len = 256;
  char versionbuf[len];
  mkl_get_version_string(versionbuf, len);
  std::cerr << "MKL " << versionbuf << ".\n";
  MKLVersion version;
  mkl_get_version(&version);
  std::cerr << "MKL platform: " << version.Platform
            << ", processor: " << version.Processor << ".\n";
  std::cerr << "MKL can use up to " << max_procs << " thread(s).\n";
  std::cerr << "MKL using " << blas_cores << " thread(s) for this backend.\n";
#endif

#ifdef USE_ACCELERATE
  std::cerr << "BLAS vendor: Apple vecLib.\n";
  std::cerr << "Apple vecLib ignores blas_cores (" << blas_cores
            << ") parameter.\n";
#endif

  std::cerr << "BLAS max batch size is " << max_batch_size_ << ".\n";
}

std::unique_ptr<Network> MakeBlasNetwork(const WeightsFile& weights,
                                         const OptionsDict& options) {
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
          pblczero::NetworkFormat::POLICY_CONVOLUTION) {
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
  return std::make_unique<BlasNetwork>(weights, options);
}

REGISTER_NETWORK("blas", MakeBlasNetwork, 50)

}  // namespace
}  // namespace lczero
