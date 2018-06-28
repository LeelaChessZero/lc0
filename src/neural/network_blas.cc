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
#include "neural/blas/blas.h"
#include "neural/blas/transforms.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <thread>

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
        input_data_[index++] = (plane.mask & (one << i)) != 0 ? value : 0;
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

  void forward(std::vector<float>& input,
               std::vector<float>& output_pol,
               std::vector<float>& output_val) {

    ////////////////////////////////////////////////////////////////////////////
    // Input convolution

    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int tiles = width * height / 4;

    // input_channels is the max number of input channels of any convolution.
    // Residual blocks are identical, but the first convolution might be bigger
    // when the network has very few filters
    const auto input_channels = std::max(static_cast<size_t>(weights_.input.biases.size()),
                                         static_cast<size_t>(kInputPlanes));

    std::vector<float> conv_out(weights_.input.biases.size() * width * height);

    std::vector<float> V(kWinogradTile * input_channels * tiles);
    std::vector<float> M(kWinogradTile * weights_.input.biases.size() * tiles);

    std::vector<float> policy_data(weights_.policy.bn_means.size() * width * height);
    std::vector<float> value_data(weights_.value.bn_means.size() * width * height);

    Transforms::WinogradConvolve3(weights_.input.biases.size(), input,
                                  weights_.input.weights, V, M, conv_out);
    Transforms::Batchnorm(weights_.input.biases.size(), conv_out,
                              weights_.input.bn_means,
                              weights_.input.bn_stddivs);

    ////////////////////////////////////////////////////////////////////////////
    // Residual tower

    std::vector<float> conv_in(weights_.residual[0].conv1.biases.size() * width * height);
    std::vector<float> res(weights_.residual[0].conv1.biases.size() * width * height);

    for (auto& residual : weights_.residual) {
      auto& conv1 = residual.conv1;
      std::swap(conv_out, conv_in);
      std::copy(begin(conv_in), end(conv_in), begin(res));

      Transforms::WinogradConvolve3(conv1.biases.size(), conv_in, conv1.weights, V,
                                    M, conv_out);
      Transforms::Batchnorm(conv1.biases.size(), conv_out,
                                conv1.bn_means, conv1.bn_stddivs);

      auto& conv2 = residual.conv2;
      std::swap(conv_out, conv_in);
      Transforms::WinogradConvolve3(conv2.biases.size(), conv_in, conv2.weights, V,
                                    M, conv_out);
      Transforms::Batchnorm(conv2.biases.size(), conv_out,
                                conv2.bn_means, conv2.bn_stddivs,
                                res.data());
    }

    ////////////////////////////////////////////////////////////////////////////
    // Value/policy heads

    Transforms::Convolve(weights_.policy.bn_means.size(), conv_out,
                         weights_.policy.weights, weights_.policy.biases,
                         policy_data);

    Transforms::Convolve(weights_.value.bn_means.size(), conv_out,
                         weights_.value.weights, weights_.value.biases,
                         value_data);

    Transforms::Batchnorm(weights_.policy.bn_means.size(), policy_data,
                          weights_.policy.bn_means,
                          weights_.policy.bn_stddivs);

    Transforms::Batchnorm(weights_.value.bn_means.size(), value_data,
                          weights_.value.bn_means,
                          weights_.value.bn_stddivs);


    Transforms::Innerproduct(policy_data, weights_.ip_pol_w, weights_.ip_pol_b,
                             output_pol);

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

  BlasNetwork(const Weights& weights, const OptionsDict& options)
      : weights_(weights) {
    bool verbose = options.GetOrDefault<bool>("verbose", true);
    int blas_cores = options.GetOrDefault<int>("blas_cores", 1);

    ////////////////////////////////////////////////////////////////////////////
    // Input convolution

    weights_.input.weights = Transforms::WinogradTransformF(
        weights_.input.weights, weights.input.biases.size(), kInputPlanes);

    Transforms::OffsetBatchNormMeans(weights_.input.bn_means,
                                     weights_.input.biases);

    Transforms::InvertBatchNormStddev(weights_.input.bn_stddivs);

    ////////////////////////////////////////////////////////////////////////////
    // Residual tower

    for (auto& residual : weights_.residual) {
      auto& conv1 = residual.conv1;
      auto& conv2 = residual.conv2;

      conv1.weights =
          Transforms::WinogradTransformF(conv1.weights, conv1.biases.size(), conv1.biases.size());
      conv2.weights =
          Transforms::WinogradTransformF(conv2.weights, conv2.biases.size(), conv2.biases.size());

      Transforms::OffsetBatchNormMeans(conv1.bn_means, conv1.biases);
      Transforms::OffsetBatchNormMeans(conv2.bn_means, conv2.biases);

      Transforms::InvertBatchNormStddev(conv1.bn_stddivs);
      Transforms::InvertBatchNormStddev(conv2.bn_stddivs);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Value/policy heads

    Transforms::OffsetBatchNormMeans(weights_.policy.bn_means,
                                     weights_.policy.biases);
    Transforms::InvertBatchNormStddev(weights_.policy.bn_stddivs);

    Transforms::OffsetBatchNormMeans(weights_.value.bn_means,
                                     weights_.value.biases);
    Transforms::InvertBatchNormStddev(weights_.value.bn_stddivs);

#ifdef USE_OPENBLAS
    int num_procs = openblas_get_num_procs();
    blas_cores = std::min(num_procs, blas_cores);
    openblas_set_num_threads(blas_cores);
    if (verbose) {
      const char* core_name = openblas_get_corename();
      const char* config = openblas_get_config();
      fprintf(stderr, "BLAS vendor: OpenBlas.\n");
      fprintf(stderr, "OpenBlas [%s].\n", config);
      fprintf(stderr, "OpenBlas found %d %s core(s).\n", num_procs, core_name);
      fprintf(stderr, "OpenBLAS using %d core(s) for this backend.\n",
              blas_cores);
    }
#endif

#ifdef USE_MKL
    int max_procs = mkl_get_max_threads();
    blas_cores = std::min(max_procs, blas_cores);
    mkl_set_num_threads(blas_cores);
    if (verbose) {
      fprintf(stderr, "BLAS vendor: MKL.\n");
      constexpr int len = 256;
      char versionbuf[len];
      mkl_get_version_string(versionbuf, len);
      fprintf(stderr, "MKL %s.\n", versionbuf);
      MKLVersion version;
      mkl_get_version(&version);
      fprintf(stderr, "MKL platform: %s, processor: %s.\n", version.Platform,
              version.Processor);
      fprintf(stderr, "MKL can use up to  %d thread(s).\n", max_procs);
      fprintf(stderr, "MKL using %d thread(s) for this backend.\n", blas_cores);
    }
#endif

#ifdef USE_ACCELERATE
    if (verbose) {
      fprintf(stderr, "BLAS vendor: Apple vecLib.\n");
    }
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
