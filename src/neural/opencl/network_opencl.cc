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

#include "neural/factory.h"
#include "neural/network.h"
#include "neural/opencl/OpenCL.h"
#include "neural/opencl/OpenCLParams.h"
#include "neural/shared/activation.h"
#include "neural/shared/policy_map.h"
#include "neural/shared/winograd_filter.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <thread>

#include "neural/network_legacy.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/logging.h"
#include "utils/weights_adapter.h"

namespace lczero {

namespace {

class OpenCLNetwork;

// Copy the vectors we need after weights is deallocated.
struct OpenCLWeights {
  const std::vector<float> ip2_val_w;
  const std::vector<float> ip2_val_b;
  const size_t num_output_policies = 1858;
  const size_t num_value_channels;

  OpenCLWeights(const WeightsFile& file)
      : ip2_val_w(LayerAdapter(file.weights().ip2_val_w()).as_vector()),
        ip2_val_b(LayerAdapter(file.weights().ip2_val_b()).as_vector()),
        num_value_channels(LayerAdapter(file.weights().ip1_val_b()).size()) {}
};

class OpenCLComputation : public NetworkComputation {
 public:
  OpenCLComputation(const OpenCL_Network& opencl_net,
                    const OpenCLWeights& weights, const bool wdl)
      : opencl_net_(opencl_net),
        weights_(weights),
        policies_(),
        q_values_(),
        wdl_(wdl) {
    buffers_ = opencl_net.acquire_buffers();
  }

  virtual ~OpenCLComputation() {
    opencl_net_.release_buffers(std::move(buffers_));
  }

  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  // Do the computation.
  void ComputeBlocking() override {
    // Determine the largest batch for allocations.
    const auto plane_count = planes_.size();
    const auto max_batch_size = opencl_net_.getMaxMatchSize();
    const auto largest_batch_size = std::min(max_batch_size, plane_count);

    const auto num_output_policies = weights_.num_output_policies;
    const auto num_value_channels = weights_.num_value_channels;

    // Typically
    // input_channels = 112
    // num_value_channels = 128
    // num_output_policy = 1858

    std::vector<float> output_pol(largest_batch_size * num_output_policies);
    std::vector<float> output_val(largest_batch_size * num_value_channels);
    std::vector<float> input_data(largest_batch_size * kInputPlanes * kSquares);

    for (size_t i = 0; i < plane_count; i += largest_batch_size) {
      const auto batch_size = std::min(plane_count - i, largest_batch_size);
      for (size_t j = 0; j < batch_size; j++) {
        EncodePlanes(planes_[i + j], &input_data[j * kSquares * kInputPlanes]);
      }

      buffers_->forward(input_data, output_pol, output_val, batch_size);

      for (size_t j = 0; j < batch_size; j++) {
        std::vector<float> policy(weights_.num_output_policies);

        // Get the moves.
        SoftmaxActivation(num_output_policies,
                          &output_pol[j * num_output_policies], policy.data());

        policies_.emplace_back(std::move(policy));

        // Now get the score.
        if (wdl_) {
          std::vector<float> wdl(weights_.ip2_val_b);
          auto ptr_weights = weights_.ip2_val_w.data();
          auto ptr_outputs = &output_val[j * num_value_channels];
          for (size_t q = 0; q < 3; q++) {
            for (size_t i = 0; i < num_value_channels; i++) {
              wdl[q] +=
                  ptr_weights[i + q * num_value_channels] * ptr_outputs[i];
            }
          }

          std::vector<float> wdl_softmax(3);
          SoftmaxActivation(3, wdl.data(), wdl_softmax.data());

          q_values_.emplace_back(wdl_softmax[0]);
          q_values_.emplace_back(wdl_softmax[1]);
          q_values_.emplace_back(wdl_softmax[2]);
        } else {
          auto winrate = weights_.ip2_val_b[0];
          auto ptr_weights = weights_.ip2_val_w.data();
          auto ptr_outputs = &output_val[j * num_value_channels];
          for (size_t i = 0; i < num_value_channels; i++)
            winrate += ptr_weights[i] * ptr_outputs[i];

          q_values_.emplace_back(std::tanh(winrate));
        }
      }
    }
  }

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
  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kSquares = kWidth * kHeight;

  void EncodePlanes(const InputPlanes& sample, float* buffer);

  const OpenCL_Network& opencl_net_;
  const OpenCLWeights& weights_;

  std::vector<InputPlanes> planes_;

  std::vector<std::vector<float>> policies_;
  std::vector<float> q_values_;

  std::unique_ptr<OpenCLBuffers> buffers_;
  bool wdl_;
};

void OpenCLComputation::EncodePlanes(const InputPlanes& sample, float* buffer) {
  for (const InputPlane& plane : sample) {
    const float value = plane.value;
    for (auto i = 0; i < kSquares; i++) {
      *(buffer++) = (plane.mask & (((uint64_t)1) << i)) != 0 ? value : 0;
    }
  }
}

class OpenCLNetwork : public Network {
 public:
  virtual ~OpenCLNetwork(){};

  OpenCLNetwork(const WeightsFile& file, const OptionsDict& options)
      : weights_(file), params_(), opencl_(), opencl_net_(opencl_) {
    LegacyWeights weights(file.weights());
    params_.gpuId = options.GetOrDefault<int>("gpu", -1);
    params_.force_tune = options.GetOrDefault<bool>("force_tune", false);
    params_.tune_only = options.GetOrDefault<bool>("tune_only", false);
    params_.tune_exhaustive =
        options.GetOrDefault<bool>("tune_exhaustive", false);

    wdl_ = file.format().network_format().output() ==
           pblczero::NetworkFormat::OUTPUT_WDL;

    auto max_batch_size_ =
        static_cast<size_t>(options.GetOrDefault<int>("batch_size", 16));
    if (max_batch_size_ > kHardMaxBatchSize) {
      max_batch_size_ = kHardMaxBatchSize;
    }
    CERR << "OpenCL, maximum batch size set to " << max_batch_size_ << ".";

    // By default, the max batch size used for tuning is the max batch size
    // used for computations.
    // It may not be the optimal value, then use tune_batch_size for fine
    // tune.
    params_.tune_batch_size =
        options.GetOrDefault<int>("tune_batch_size", max_batch_size_);

    const auto inputChannels = static_cast<size_t>(kInputPlanes);
    const auto channels = weights.input.biases.size();
    const auto residual_blocks = weights.residual.size();

    const auto num_value_input_planes = weights.value.biases.size();
    const auto num_policy_input_planes = weights.policy.biases.size();
    const auto num_output_policy = kPolicyOutputs;
    const auto num_value_channels = weights.ip1_val_b.size();

    // Typically
    // input_channels = 112
    // output_channels = 192
    // num_value_input_planes = 32
    // num_policy_input_planes = 32
    // num_value_channels = 128
    // num_output_policy = 1858

    static constexpr auto kWinogradAlpha = 4;

    opencl_.initialize(channels, params_);

    auto tuners = opencl_.get_sgemm_tuners();

    auto mwg = tuners[0];
    auto kwg = tuners[2];
    auto vwm = tuners[3];

    size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
    size_t k_ceil = ceilMultiple(ceilMultiple(inputChannels, kwg), vwm);

    std::vector<float> input_conv_weights = WinogradFilterTransformF(
        weights.input.weights, channels, inputChannels);

    auto Upad = WinogradFilterZeropadU(input_conv_weights, channels,
                                       inputChannels, m_ceil, k_ceil);

    // Winograd filter transformation changes filter size to 4x4.
    opencl_net_.push_input_convolution(kWinogradAlpha, inputChannels, channels,
                                       Upad, weights.input.biases);

    auto conv_policy = file.format().network_format().policy() ==
                       pblczero::NetworkFormat::POLICY_CONVOLUTION;

    // Residual blocks.
    for (auto i = size_t{0}; i < residual_blocks; i++) {
      auto& residual = weights.residual[i];
      auto& conv1 = residual.conv1;
      auto& conv2 = residual.conv2;
      auto& se = residual.se;

      std::vector<float> conv_weights_1 =
          WinogradFilterTransformF(conv1.weights, channels, channels);
      std::vector<float> conv_weights_2 =
          WinogradFilterTransformF(conv2.weights, channels, channels);

      auto Upad1 = WinogradFilterZeropadU(conv_weights_1, channels, channels,
                                          m_ceil, m_ceil);
      auto Upad2 = WinogradFilterZeropadU(conv_weights_2, channels, channels,
                                          m_ceil, m_ceil);

      opencl_net_.push_residual(kWinogradAlpha, channels, channels, Upad1,
                                conv1.biases, Upad2, conv2.biases);
      if (residual.has_se) {
        auto se_fc_outputs = se.w1.size() / channels;
        if (se.b2.size() != 2 * channels) {
          throw Exception("SE-unit output bias is not right size.");
        }
        opencl_net_.push_se(channels, se_fc_outputs, se.w1, se.b1, se.w2,
                            se.b2);
      }
    }

    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;

    if (conv_policy) {
      auto& policy1 = weights.policy1;
      auto& policy = weights.policy;
      auto pol_channels = policy.biases.size();

      std::vector<float> conv_weights_1 =
          WinogradFilterTransformF(policy1.weights, channels, channels);
      auto W1 = WinogradFilterZeropadU(conv_weights_1, channels, channels,
                                       m_ceil, m_ceil);

      size_t m_ceil_pol = ceilMultiple(ceilMultiple(pol_channels, mwg), vwm);
      size_t k_ceil_pol = ceilMultiple(ceilMultiple(channels, kwg), vwm);
      std::vector<float> conv_weights_2 =
          WinogradFilterTransformF(policy.weights, pol_channels, channels);
      auto W2 = WinogradFilterZeropadU(conv_weights_2, pol_channels, channels,
                                       m_ceil_pol, k_ceil_pol);

      std::vector<short> indices;
      for (auto i = size_t{0}; i < kPolicyUsedPlanes * 8 * 8; i++) {
        indices.emplace_back(kConvPolicyMap[i]);
      }

      opencl_net_.push_conv_policy(
          channels, pol_channels, kPolicyUsedPlanes * width * height,
          num_output_policy, W1, weights.policy1.biases, W2,
          weights.policy.biases, indices);
    } else {
      opencl_net_.push_policy(channels, num_policy_input_planes,
                              num_policy_input_planes * width * height,
                              num_output_policy, weights.policy.weights,
                              weights.policy.biases, weights.ip_pol_w,
                              weights.ip_pol_b);
    }
    opencl_net_.push_value(channels, num_value_input_planes,
                           num_value_input_planes * width * height,
                           num_value_channels, weights.value.weights,
                           weights.value.biases, weights.ip1_val_w,
                           weights.ip1_val_b);

    opencl_net_.setMaxMatchSize(max_batch_size_);
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<OpenCLComputation>(opencl_net_, weights_, wdl_);
  }

 private:
  static constexpr auto kHardMaxBatchSize = 16;
  static constexpr auto kPolicyUsedPlanes = 73;
  static constexpr auto kPolicyOutputs = 1858;

  OpenCLWeights weights_;
  OpenCLParams params_;
  OpenCL opencl_;
  OpenCL_Network opencl_net_;
  bool wdl_;
};

std::unique_ptr<Network> MakeOpenCLNetwork(const WeightsFile& weights,
                                           const OptionsDict& options) {
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by OpenCL backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION) {
    throw Exception("Policy format " +
                    std::to_string(weights.format().network_format().policy()) +
                    " is not supported by OpenCL backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    std::to_string(weights.format().network_format().value()) +
                    " is not supported by OpenCL backend.");
  }
  return std::make_unique<OpenCLNetwork>(weights, options);
}

REGISTER_NETWORK("opencl", MakeOpenCLNetwork, 100)

}  // namespace
}  // namespace lczero
