/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors

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

#include "neural/network.h"
#include "neural/blas/batchnorm.h"
#include "neural/blas/blas.h"
#include "neural/blas/fully_connected_layer.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/factory.h"
#include "neural/opencl/OpenCL.h"
#include "neural/opencl/OpenCLParams.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <thread>

#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {

namespace {

class OpenCLNetwork;

// Copy the vectors we need after weights is deallocated.
struct OpenCLWeights {
  const std::vector<float> ip2_val_w;
  const std::vector<float> ip2_val_b;
  const size_t num_output_policies;
  const size_t num_value_channels;

  OpenCLWeights(const Weights& weights)
      : ip2_val_w(weights.ip2_val_w),
        ip2_val_b(weights.ip2_val_b),
        num_output_policies(weights.ip_pol_b.size()),
        num_value_channels(weights.ip1_val_b.size()) {}
};

class OpenCLComputation : public NetworkComputation {
 public:
  OpenCLComputation(const OpenCL_Network& opencl_net, const OpenCLWeights& weights)
      : opencl_net_(opencl_net), weights_(weights), policies_(), q_values_() {
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
        FullyConnectedLayer::Softmax(num_output_policies,
                                     &output_pol[j * num_output_policies],
                                     policy.data());

        policies_.emplace_back(std::move(policy));

        // Now get the score.
        auto winrate = FullyConnectedLayer::Forward0D(
                           num_value_channels, weights_.ip2_val_w.data(),
                           &output_val[j * num_value_channels]) +
                       weights_.ip2_val_b[0];

        q_values_.emplace_back(std::tanh(winrate));
      }
    }
  }

  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return static_cast<int>(planes_.size()); }

  // Returns Q value of @sample.
  float GetQVal(int sample) const override { return q_values_[sample]; }

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

  OpenCLNetwork(const Weights& weights, const OptionsDict& options)
      : weights_(weights), params_(), opencl_(), opencl_net_(opencl_) {
    params_.gpuId = options.GetOrDefault<int>("gpu", -1);
    params_.force_tune = options.GetOrDefault<bool>("force_tune", false);
    params_.tune_only = options.GetOrDefault<bool>("tune_only", false);
    params_.tune_exhaustive =
        options.GetOrDefault<bool>("tune_exhaustive", false);

    // By default batch size is 1, as many old cards may not support more.
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

    const auto num_value_input_planes = weights.value.bn_means.size();
    const auto num_policy_input_planes = weights.policy.bn_means.size();
    const auto num_output_policy = weights.ip_pol_b.size();
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

    std::vector<float> input_conv_weights = WinogradConvolution3::TransformF(
        weights.input.weights, channels, inputChannels);

    auto Upad = WinogradConvolution3::ZeropadU(input_conv_weights, channels,
                                               inputChannels, m_ceil, k_ceil);

    std::vector<float> input_batchnorm_means =
        Batchnorm::OffsetMeans(weights.input);
    std::vector<float> input_batchnorm_stddivs =
        Batchnorm::InvertStddev(weights.input);

    // Winograd filter transformation changes filter size to 4x4.
    opencl_net_.push_input_convolution(kWinogradAlpha, inputChannels, channels,
                                       Upad, input_batchnorm_means,
                                       input_batchnorm_stddivs);

    // Residual blocks.
    for (auto i = size_t{0}; i < residual_blocks; i++) {
      auto& residual = weights.residual[i];
      auto& conv1 = residual.conv1;
      auto& conv2 = residual.conv2;

      std::vector<float> conv_weights_1 =
          WinogradConvolution3::TransformF(conv1.weights, channels, channels);
      std::vector<float> conv_weights_2 =
          WinogradConvolution3::TransformF(conv2.weights, channels, channels);

      auto Upad1 = WinogradConvolution3::ZeropadU(conv_weights_1, channels,
                                                  channels, m_ceil, m_ceil);
      auto Upad2 = WinogradConvolution3::ZeropadU(conv_weights_2, channels,
                                                  channels, m_ceil, m_ceil);

      std::vector<float> batchnorm_means_1 = Batchnorm::OffsetMeans(conv1);
      std::vector<float> batchnorm_means_2 = Batchnorm::OffsetMeans(conv2);

      std::vector<float> batchnorm_stddivs_1 = Batchnorm::InvertStddev(conv1);
      std::vector<float> batchnorm_stddivs_2 = Batchnorm::InvertStddev(conv2);

      opencl_net_.push_residual(kWinogradAlpha, channels, channels, Upad1,
                                batchnorm_means_1, batchnorm_stddivs_1, Upad2,
                                batchnorm_means_2, batchnorm_stddivs_2);
    }

    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;

    std::vector<float> bn_pol_means = Batchnorm::OffsetMeans(weights.policy);
    std::vector<float> bn_pol_stddivs = Batchnorm::InvertStddev(weights.policy);

    opencl_net_.push_policy(channels, num_policy_input_planes,
                            num_policy_input_planes * width * height,
                            num_output_policy, weights.policy.weights,
                            bn_pol_means, bn_pol_stddivs, weights.ip_pol_w,
                            weights.ip_pol_b);

    std::vector<float> bn_val_means = Batchnorm::OffsetMeans(weights.value);
    std::vector<float> bn_val_stddivs = Batchnorm::InvertStddev(weights.value);

    opencl_net_.push_value(channels, num_value_input_planes,
                           num_value_input_planes * width * height,
                           num_value_channels, weights.value.weights,
                           bn_val_means, bn_val_stddivs, weights.ip1_val_w,
                           weights.ip1_val_b);

    opencl_net_.setMaxMatchSize(max_batch_size_);
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<OpenCLComputation>(opencl_net_, weights_);
  }

 private:
  static constexpr auto kHardMaxBatchSize = 16;

  OpenCLWeights weights_;
  OpenCLParams params_;
  OpenCL opencl_;
  OpenCL_Network opencl_net_;
};

}  // namespace

REGISTER_NETWORK("opencl", OpenCLNetwork, 100)

}  // namespace lczero
