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

#include "neural/CL/transforms.h"
#include "neural/factory.h"
#include "neural/network.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <thread>

#include "utils/blas.h"
#include "utils/exception.h"

namespace lczero {

namespace {

class BlasNetwork;

class BlasComputation : public NetworkComputation {
 public:
  BlasComputation(const Weights& weights)
      : weights_(weights),
        input_data_(kInputPlanes * 64),
        value_data_(weights_.ip1_val_b.size()),
        policy_data_(),
        q_value_(0) {}

  virtual ~BlasComputation() {}

  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

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
        input_data_[index++] = ((plane.mask & (one << i)) == 0) ? 0 : value;
    }

    std::vector<float> policy_data(weights_.ip_pol_b.size());
    forward(input_data_, policy_data, value_data_);

    // Get the moves
    Transforms::Softmax(policy_data, policy_data);
    policy_data_.emplace_back(move(policy_data));

    // Now get the score
    double winrate = Transforms::Innerproduct(weights_.ip2_val_w, value_data_) +
                     weights_.ip2_val_b[0];
    q_value_.emplace_back(std::tanh(winrate));
  }

  void forward(std::vector<float>& input, std::vector<float>& output_pol,
               std::vector<float>& output_val) {
    // Input convolution
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int tiles = width * height / 4;

    int NUM_VALUE_INPUT_PLANES = weights_.value.bn_means.size();
    int NUM_POLICY_INPUT_PLANES = weights_.policy.bn_means.size();

    static constexpr auto kWinogradAlpha = 4;
    static constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;

    // Calculate output channels
    const auto output_channels = weights_.input.biases.size();
    // input_channels is the maximum number of input channels of any
    // convolution.
    // Residual blocks are identical, but the first convolution might be bigger
    // when the network has very few filters
    const auto input_channels = std::max(static_cast<size_t>(output_channels),
                                         static_cast<size_t>(kInputPlanes));
    auto conv_out = std::vector<float>(output_channels * width * height);

    auto V = std::vector<float>(kWinogradTile * input_channels * tiles);
    auto M = std::vector<float>(kWinogradTile * output_channels * tiles);

    std::vector<float> policy_data(NUM_POLICY_INPUT_PLANES * width * height);
    std::vector<float> value_data(NUM_VALUE_INPUT_PLANES * width * height);

    Transforms::WinogradConvolve3(output_channels, input,
                                  weights_.input.weights, V, M, conv_out);
    Transforms::Batchnorm<64>(output_channels, conv_out,
                              weights_.input.bn_means.data(),
                              weights_.input.bn_stddivs.data());

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * width * height);
    auto res = std::vector<float>(output_channels * width * height);

    for (auto& residual : weights_.residual) {
      auto& conv1 = residual.conv1;
      auto output_channels = conv1.biases.size();
      std::swap(conv_out, conv_in);
      std::copy(begin(conv_in), end(conv_in), begin(res));

      Transforms::WinogradConvolve3(output_channels, conv_in, conv1.weights, V,
                                    M, conv_out);
      Transforms::Batchnorm<64>(output_channels, conv_out,
                                conv1.bn_means.data(), conv1.bn_stddivs.data());

      auto& conv2 = residual.conv2;
      output_channels = conv2.biases.size();
      std::swap(conv_out, conv_in);
      Transforms::WinogradConvolve3(output_channels, conv_in, conv2.weights, V,
                                    M, conv_out);
      Transforms::Batchnorm<64>(output_channels, conv_out,
                                conv2.bn_means.data(), conv2.bn_stddivs.data(),
                                res.data());
    }

    Transforms::Convolve<1>(NUM_POLICY_INPUT_PLANES, conv_out,
                            weights_.policy.weights, weights_.policy.biases,
                            policy_data);

    Transforms::Convolve<1>(NUM_VALUE_INPUT_PLANES, conv_out,
                            weights_.value.weights, weights_.value.biases,
                            value_data);

    Transforms::Batchnorm<width * height>(NUM_POLICY_INPUT_PLANES, policy_data,
                                          weights_.policy.bn_means.data(),
                                          weights_.policy.bn_stddivs.data());

    Transforms::Batchnorm<width * height>(NUM_VALUE_INPUT_PLANES, value_data,
                                          weights_.value.bn_means.data(),
                                          weights_.value.bn_stddivs.data());

    // NUM_POLICY_INPUT_PLANES*width*height x NUM_OUTPUT_POLICY
    Transforms::Innerproduct(policy_data, weights_.ip_pol_w, weights_.ip_pol_b,
                             output_pol);

    // NUM_VALUE_INPUT_PLANES*width*height x NUM_VALUE_CHANNELS,
    Transforms::Innerproduct(value_data, weights_.ip1_val_w, weights_.ip1_val_b,
                             output_val, true);
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
  const Weights& weights_;

  std::vector<InputPlanes> planes_;
  std::vector<float> input_data_;
  std::vector<float> value_data_;

  std::vector<std::vector<float>> policy_data_;
  std::vector<float> q_value_;
};

class BlasNetwork : public Network {
 public:
  virtual ~BlasNetwork(){};

  BlasNetwork(const Weights& weights, const OptionsDict& /* options */)
      : weights_(weights) {
    const int inputChannels = kInputPlanes;
    const int channels = weights.input.biases.size();
    const size_t residual_blocks = weights.residual.size();

    weights_.input.weights = Transforms::WinogradTransformF(
        weights_.input.weights, channels, inputChannels);

    std::vector<float>& input_batchnorm_means = weights_.input.bn_means;
    Transforms::OffsetBatchNormMeans(input_batchnorm_means,
                                     weights_.input.biases);

    std::vector<float>& input_batchnorm_stddivs = weights_.input.bn_stddivs;
    Transforms::InvertBatchNormStddev(input_batchnorm_stddivs);

    // residual blocks
    for (size_t i = 0; i < residual_blocks; i++) {
      auto& residual = weights_.residual[i];
      auto& conv1 = residual.conv1;
      auto& conv2 = residual.conv2;

      conv1.weights =
          Transforms::WinogradTransformF(conv1.weights, channels, channels);
      conv2.weights =
          Transforms::WinogradTransformF(conv2.weights, channels, channels);

      std::vector<float>& batchnorm_means_1 = conv1.bn_means;
      Transforms::OffsetBatchNormMeans(batchnorm_means_1, conv1.biases);

      std::vector<float>& batchnorm_means_2 = conv2.bn_means;
      Transforms::OffsetBatchNormMeans(batchnorm_means_2, conv2.biases);

      std::vector<float>& batchnorm_stddivs_1 = conv1.bn_stddivs;
      Transforms::InvertBatchNormStddev(batchnorm_stddivs_1);

      std::vector<float>& batchnorm_stddivs_2 = conv2.bn_stddivs;
      Transforms::InvertBatchNormStddev(batchnorm_stddivs_2);
    }

    std::vector<float>& bn_pol_means = weights_.policy.bn_means;
    Transforms::OffsetBatchNormMeans(bn_pol_means, weights_.policy.biases);

    std::vector<float>& bn_pol_stddivs = weights_.policy.bn_stddivs;
    Transforms::InvertBatchNormStddev(bn_pol_stddivs);

    std::vector<float>& bn_val_means = weights_.value.bn_means;
    Transforms::OffsetBatchNormMeans(bn_val_means, weights_.value.biases);

    std::vector<float>& bn_val_stddivs = weights_.value.bn_stddivs;
    Transforms::InvertBatchNormStddev(bn_val_stddivs);

#ifdef USE_OPENBLAS
// openblas_set_num_threads(1);
// printf("BLAS Core: %s\n", openblas_get_corename());
#endif

#ifdef USE_MKL
    // mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    printf("BLAS core: MKL %s\n", Version.Processor);
#endif
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasComputation>(weights_);
  }

 private:
  Weights weights_;
};

}  // namespace

REGISTER_NETWORK("blas", BlasNetwork, 50)

}  // namespace lczero
