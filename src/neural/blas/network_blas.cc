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

#include "neural/blas/batchnorm.h"
#include "neural/blas/blas.h"
#include "neural/blas/convolution1.h"
#include "neural/blas/fully_connected_layer.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/factory.h"
#include "neural/network.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace lczero {

class BlasComputation : public NetworkComputation {
 public:
  BlasComputation(const Weights& weights, const size_t max_batch_size);

  virtual ~BlasComputation() {}

  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  // Do the computation.
  void ComputeBlocking() override;

  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return static_cast<int>(planes_.size()); }

  // Returns Q value of @sample.
  float GetQVal(int sample) const override { return q_values_[sample]; }

  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override {
    return policies_[sample][move_id];
  }

 private:
  void EncodePlanes(const InputPlanes& sample, float* buffer);

  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kSquares = kWidth * kHeight;

  const Weights& weights_;
  size_t max_batch_size_;
  std::vector<InputPlanes> planes_;
  std::vector<std::vector<float>> policies_;
  std::vector<float> q_values_;
};

class BlasNetwork : public Network {
 public:
  BlasNetwork(const Weights& weights, const OptionsDict& options);
  virtual ~BlasNetwork(){};

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasComputation>(weights_, max_batch_size_);
  }

 private:
  // A cap on the max batch size since it consumes a lot of memory
  static constexpr auto kHardMaxBatchSize = 2048;

  Weights weights_;
  size_t max_batch_size_;
};

BlasComputation::BlasComputation(const Weights& weights,
                                 const size_t max_batch_size)
    : weights_(weights),
      max_batch_size_(max_batch_size),
      policies_(0),
      q_values_(0) {}

void BlasComputation::ComputeBlocking() {
  // Retrieve network key dimensions from the weights structure.
  const auto num_value_channels = weights_.ip1_val_b.size();
  const auto num_value_input_planes = weights_.value.bn_means.size();
  const auto num_policy_input_planes = weights_.policy.bn_means.size();
  const auto num_output_policy = weights_.ip_pol_b.size();
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
                      &weights_.input.weights[0], conv_out);

    Batchnorm::Apply(batch_size, output_channels, conv_out,
                     weights_.input.bn_means.data(),
                     weights_.input.bn_stddivs.data());

    // Residual tower

    for (auto& residual : weights_.residual) {
      auto& conv1 = residual.conv1;
      auto& conv2 = residual.conv2;

      std::swap(conv_out, conv_in);

      convolve3.Forward(batch_size, output_channels, output_channels, conv_in,
                        &conv1.weights[0], conv_out);

      Batchnorm::Apply(batch_size, output_channels, &conv_out[0],
                       conv1.bn_means.data(), conv1.bn_stddivs.data());

      std::swap(conv_in, res);
      std::swap(conv_out, conv_in);

      convolve3.Forward(batch_size, output_channels, output_channels, conv_in,
                        &conv2.weights[0], conv_out);

      Batchnorm::Apply(batch_size, output_channels, conv_out,
                       conv2.bn_means.data(), conv2.bn_stddivs.data(), res);
    }

    Convolution1::Forward(batch_size, output_channels, num_policy_input_planes,
                          conv_out, weights_.policy.weights.data(),
                          policy_buffer.data());

    Convolution1::Forward(batch_size, output_channels, num_value_input_planes,
                          conv_out, weights_.value.weights.data(),
                          value_buffer.data());

    Batchnorm::Apply(batch_size, num_policy_input_planes, &policy_buffer[0],
                     weights_.policy.bn_means.data(),
                     weights_.policy.bn_stddivs.data());

    Batchnorm::Apply(batch_size, num_value_input_planes, &value_buffer[0],
                     weights_.value.bn_means.data(),
                     weights_.value.bn_stddivs.data());

    FullyConnectedLayer::Forward1D(
        batch_size, num_policy_input_planes * kSquares, num_output_policy,
        policy_buffer.data(), weights_.ip_pol_w.data(),
        weights_.ip_pol_b.data(),
        false,  // Relu Off
        output_pol.data());

    FullyConnectedLayer::Forward1D(
        batch_size, num_value_input_planes * kSquares, num_value_channels,
        value_buffer.data(), weights_.ip1_val_w.data(),
        weights_.ip1_val_b.data(),
        true,  // Relu On
        output_val.data());

    for (size_t j = 0; j < batch_size; j++) {
      std::vector<float> policy(num_output_policy);

      // Get the moves
      FullyConnectedLayer::Softmax(
          num_output_policy, &output_pol[j * num_output_policy], policy.data());

      policies_.emplace_back(std::move(policy));

      // Now get the score
      double winrate = FullyConnectedLayer::Forward0D(
                           num_value_channels, weights_.ip2_val_w.data(),
                           &output_val[j * num_value_channels]) +
                       weights_.ip2_val_b[0];

      q_values_.emplace_back(std::tanh(winrate));
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

BlasNetwork::BlasNetwork(const Weights& weights, const OptionsDict& options)
    : weights_(weights) {
  bool verbose = options.GetOrDefault<bool>("verbose", true);
  int blas_cores = options.GetOrDefault<int>("blas_cores", 1);
  max_batch_size_ =
      static_cast<size_t>(options.GetOrDefault<int>("batch_size", 256));

  if (max_batch_size_ > kHardMaxBatchSize) {
    max_batch_size_ = kHardMaxBatchSize;
  }
  fprintf(stderr, "BLAS, maximum batch size set to %ld.\n", max_batch_size_);

  const auto inputChannels = kInputPlanes;
  const auto channels = static_cast<int>(weights.input.biases.size());
  const auto residual_blocks = weights.residual.size();

  weights_.input.weights = WinogradConvolution3::TransformF(
      weights_.input.weights, channels, inputChannels);

  Batchnorm::OffsetMeans(&weights_.input);
  Batchnorm::InvertStddev(&weights_.input);

  // residual blocks
  for (size_t i = 0; i < residual_blocks; i++) {
    auto& residual = weights_.residual[i];
    auto& conv1 = residual.conv1;
    auto& conv2 = residual.conv2;

    conv1.weights =
        WinogradConvolution3::TransformF(conv1.weights, channels, channels);
    conv2.weights =
        WinogradConvolution3::TransformF(conv2.weights, channels, channels);

    Batchnorm::OffsetMeans(&conv1);
    Batchnorm::OffsetMeans(&conv2);

    Batchnorm::InvertStddev(&conv1);
    Batchnorm::InvertStddev(&conv2);
  }

  Batchnorm::OffsetMeans(&weights_.policy);
  Batchnorm::InvertStddev(&weights_.policy);

  Batchnorm::OffsetMeans(&weights_.value);
  Batchnorm::InvertStddev(&weights_.value);

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
    fprintf(stderr, "Apple vecLib ignores blas_cores (%d) parameter.\n",
            blas_cores);
  }
#endif

  fprintf(stderr, "BLAS max batch size is %ld.\n", max_batch_size_);
}

REGISTER_NETWORK("blas", BlasNetwork, 50)

}  // namespace lczero
