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
#include "neural/factory.h"
#include "neural/CL/OpenCL.h"
#include "neural/CL/OpenCLParams.h"
#include "neural/BLAS/blas.h"
#include "neural/BLAS/transforms.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <thread>

#include "utils/bititer.h"
#include "utils/exception.h"


namespace lczero {

namespace {

class OpenCLNetwork;

// Copy the vectors we need after weights is deallocated
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
  OpenCLComputation(const OpenCL_Network& opencl_net,
                    const OpenCLWeights& weights)
      : opencl_net_(opencl_net),
        weights_(weights),
        input_data_(kInputPlanes * 64),
        value_data_(weights_.num_value_channels),
        policy_data_(),
        q_value_(0) {}

  virtual ~OpenCLComputation() {}

  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

 public:
  // Do the computation.
  void ComputeBlocking() override {
    for (auto& sample : planes_) ComputeBlocking(sample);
  }

  void ComputeBlocking(const InputPlanes& sample) {
    int index = 0;
    for (const InputPlane& plane : sample) {
      float value = plane.value;
      const uint64_t one = 1;
      for (int i = 0; i < 64; i++)
        input_data_[index++] = (plane.mask & (one << i)) != 0 ? value : 0;
    }

    std::vector<float> policy_data(weights_.num_output_policies);
    opencl_net_.forward(input_data_, policy_data, value_data_);

    // Get the moves
    Transforms::Softmax(policy_data, policy_data);
    policy_data_.emplace_back(move(policy_data));

    // Now get the score
    double winrate = Transforms::Innerproduct(weights_.ip2_val_w, value_data_) +
                     weights_.ip2_val_b[0];
    q_value_.emplace_back(std::tanh(winrate));
  }

  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return planes_.size(); }

  // Returns Q value of @sample.
  float GetQVal(int sample) const override { return q_value_[sample]; }

  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override {
    return policy_data_[sample][move_id];
  }

 private:
  const OpenCL_Network& opencl_net_;
  const OpenCLWeights& weights_;

  std::vector<InputPlanes> planes_;
  std::vector<float> input_data_;
  std::vector<float> value_data_;

  std::vector<std::vector<float>> policy_data_;
  std::vector<float> q_value_;
};

class OpenCLNetwork : public Network {
 public:
  virtual ~OpenCLNetwork(){};

  OpenCLNetwork(const Weights& weights, const OptionsDict& options)
      : weights_(weights), params_(), opencl_(), opencl_net_(opencl_) {
    params_.gpuId = options.GetOrDefault<int>("gpu", -1);
    params_.verbose = options.GetOrDefault<bool>("verbose", true);
    params_.force_tune = options.GetOrDefault<bool>("force_tune", false);
    params_.tune_only = options.GetOrDefault<bool>("tune_only", false);
    params_.tune_exhaustive =
        options.GetOrDefault<bool>("tune_exhaustive", false);

    const int inputChannels = kInputPlanes;
    const int channels = weights.input.biases.size();
    const size_t residual_blocks = weights.residual.size();

    /*
     static constexpr int NUM_VALUE_INPUT_PLANES = 32;
     static constexpr int NUM_POLICY_INPUT_PLANES = 32;
     static constexpr int NUM_OUTPUT_POLICY = 1858;
     static constexpr int NUM_VALUE_CHANNELS = 128;
     */

    int NUM_VALUE_INPUT_PLANES = weights.value.bn_means.size();
    int NUM_POLICY_INPUT_PLANES = weights.policy.bn_means.size();
    int NUM_OUTPUT_POLICY = weights.ip_pol_b.size();
    int NUM_VALUE_CHANNELS = weights.ip1_val_b.size();

    static constexpr auto kWinogradAlpha = 4;

    opencl_.initialize(channels, params_);

    auto tuners = opencl_.get_sgemm_tuners();

    auto mwg = tuners[0];
    auto kwg = tuners[2];
    auto vwm = tuners[3];

    size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
    size_t k_ceil = ceilMultiple(ceilMultiple(inputChannels, kwg), vwm);

    std::vector<float> input_conv_weights = Transforms::WinogradTransformF(
        weights.input.weights, channels, inputChannels);

    auto Upad = Transforms::ZeropadU(input_conv_weights, channels,
                                     inputChannels, m_ceil, k_ceil);

    std::vector<float> input_batchnorm_means =
        weights.input.bn_means;  // copy ctor
    Transforms::OffsetBatchNormMeans(input_batchnorm_means,
                                     weights.input.biases);

    std::vector<float> input_batchnorm_stddivs = weights.input.bn_stddivs;
    Transforms::InvertBatchNormStddev(input_batchnorm_stddivs);

    // Winograd filter transformation changes filter size to 4x4
    opencl_net_.push_input_convolution(kWinogradAlpha, inputChannels, channels,
                                       Upad, input_batchnorm_means,
                                       input_batchnorm_stddivs);

    // residual blocks
    for (auto i = size_t{0}; i < residual_blocks; i++) {
      auto& residual = weights.residual[i];
      auto& conv1 = residual.conv1;
      auto& conv2 = residual.conv2;

      std::vector<float> conv_weights_1 =
          Transforms::WinogradTransformF(conv1.weights, channels, channels);
      std::vector<float> conv_weights_2 =
          Transforms::WinogradTransformF(conv2.weights, channels, channels);

      auto Upad1 = Transforms::ZeropadU(conv_weights_1, channels, channels,
                                        m_ceil, m_ceil);
      auto Upad2 = Transforms::ZeropadU(conv_weights_2, channels, channels,
                                        m_ceil, m_ceil);

      std::vector<float> batchnorm_means_1 = conv1.bn_means;  // copy ctor
      Transforms::OffsetBatchNormMeans(batchnorm_means_1, conv1.biases);

      std::vector<float> batchnorm_means_2 = conv2.bn_means;  // copy ctor
      Transforms::OffsetBatchNormMeans(batchnorm_means_2, conv2.biases);

      std::vector<float> batchnorm_stddivs_1 = conv1.bn_stddivs;  // copy ctor
      Transforms::InvertBatchNormStddev(batchnorm_stddivs_1);

      std::vector<float> batchnorm_stddivs_2 = conv2.bn_stddivs;  // copy ctor
      Transforms::InvertBatchNormStddev(batchnorm_stddivs_2);

      opencl_net_.push_residual(kWinogradAlpha, channels, channels, Upad1,
                                batchnorm_means_1, batchnorm_stddivs_1, Upad2,
                                batchnorm_means_2, batchnorm_stddivs_2);
    }

    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;

    std::vector<float> bn_pol_means = weights.policy.bn_means;  // copy ctor
    Transforms::OffsetBatchNormMeans(bn_pol_means, weights.policy.biases);

    std::vector<float> bn_pol_stddivs = weights.policy.bn_stddivs;
    Transforms::InvertBatchNormStddev(bn_pol_stddivs);

    opencl_net_.push_policy(channels, NUM_POLICY_INPUT_PLANES,
                            NUM_POLICY_INPUT_PLANES * width * height,
                            NUM_OUTPUT_POLICY, weights.policy.weights,
                            bn_pol_means, bn_pol_stddivs, weights.ip_pol_w,
                            weights.ip_pol_b);

    std::vector<float> bn_val_means = weights.value.bn_means;
    Transforms::OffsetBatchNormMeans(bn_val_means, weights.value.biases);

    std::vector<float> bn_val_stddivs = weights.value.bn_stddivs;
    Transforms::InvertBatchNormStddev(bn_val_stddivs);

    opencl_net_.push_value(channels, NUM_VALUE_INPUT_PLANES,
                           NUM_VALUE_INPUT_PLANES * width * height,
                           NUM_VALUE_CHANNELS, weights.value.weights,
                           bn_val_means, bn_val_stddivs, weights.ip1_val_w,
                           weights.ip1_val_b);
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<OpenCLComputation>(opencl_net_, weights_);
  }

 private:
  OpenCLWeights weights_;
  OpenCLParams params_;
  OpenCL opencl_;
  OpenCL_Network opencl_net_;
};

}  // namespace

REGISTER_NETWORK("opencl", OpenCLNetwork, 100)

}  // namespace lczero
