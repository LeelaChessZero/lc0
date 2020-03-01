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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "utils/bititer.h"
#include "utils/optionsdict.h"
#include "utils/transpose.h"

namespace lczero {

using namespace tensorflow;
using namespace tensorflow::ops;

namespace {

Output MakeConst(const Scope& scope, TensorShape shape,
                 const std::vector<float>& values,
                 const std::vector<int>& order = {}) {
  auto tensor = Tensor(DataType::DT_FLOAT, shape);
  CHECK_EQ(tensor.NumElements(), static_cast<int>(values.size()))
      << shape.DebugString();

  std::vector<int> dims;
  for (const auto& x : shape) {
    dims.push_back(x.size);
  }
  TransposeTensor(dims, order, values, tensor.flat<float>().data());

  return Const(scope, tensor);
}

template <bool CPU>
Output SqueezeAndExcite(const Scope& scope, Input input, int channels,
                        const LegacyWeights::SEunit& weights) {
  const int se_channels = weights.b1.size();
  // NCHW ("NHWC" for CPU case) format reduced to NC.
  auto pooled = Mean(scope, input, CPU ? Input({1, 2}) : Input({2, 3}));
  auto w1 = MakeConst(scope, {channels, se_channels}, weights.w1);
  auto b1 = MakeConst(scope, {se_channels}, weights.b1);
  auto fc1 = Add(scope, MatMul(scope, pooled, w1), b1);
  auto relu = Relu(scope, fc1);
  auto w2 = MakeConst(scope, {se_channels, 2 * channels}, weights.w2);
  auto b2 = MakeConst(scope, {2 * channels}, weights.b2);
  auto fc2 = Add(scope, MatMul(scope, relu, w2), b2);
  auto reshape = Reshape(
      scope, fc2,
      CPU ? Input({-1, 1, 1, 2 * channels}) : Input({-1, 2 * channels, 1, 1}));
  auto outputs = Split(scope, CPU ? 3 : 1, reshape, 2);
  auto sigmoid = Sigmoid(scope, outputs[0]);
  return Add(scope, Mul(scope, sigmoid, input), outputs[1]);
}

template <bool CPU>
Output MakeConvBlock(const Scope& scope, Input input, int channels,
                     int input_channels, int output_channels,
                     const LegacyWeights::ConvBlock& weights,
                     const LegacyWeights::SEunit* const seunit = nullptr,
                     Input* mixin = nullptr) {
  // CPU only supports "NHWC", while for GPU "NCHW" is better.
  const char* const kDataFormat = CPU ? "NHWC" : "NCHW";
  auto w_conv =
      MakeConst(scope, {channels, channels, input_channels, output_channels},
                weights.weights, {3, 2, 0, 1});
  auto conv2d = Conv2D(scope, input, w_conv, {1, 1, 1, 1}, "SAME",
                       Conv2D::DataFormat(kDataFormat).Dilations({1, 1, 1, 1}));
  auto b_conv = MakeConst(scope, {output_channels}, weights.biases);
  Output conv_b =
      BiasAdd(scope, conv2d, b_conv, BiasAdd::DataFormat(kDataFormat));
  if (seunit)
    conv_b = SqueezeAndExcite<CPU>(scope, conv_b, output_channels, *seunit);
  if (mixin) conv_b = Add(scope, conv_b, *mixin);
  return Relu(scope, conv_b);
}

template <bool CPU>
Output MakeResidualBlock(const Scope& scope, Input input, int channels,
                         const LegacyWeights::Residual& weights) {
  auto block1 =
      MakeConvBlock<CPU>(scope, input, 3, channels, channels, weights.conv1);
  auto block2 =
      MakeConvBlock<CPU>(scope, block1, 3, channels, channels, weights.conv2,
                         weights.has_se ? &weights.se : nullptr, &input);
  return block2;
}

template <bool CPU>
std::pair<Output, Output> MakeNetwork(const Scope& scope, Input input,
                                      const LegacyWeights& weights) {
  const int filters = weights.input.weights.size() / kInputPlanes / 9;

  // Input convolution.
  auto flow =
      MakeConvBlock<CPU>(scope, input, 3, kInputPlanes, filters, weights.input);

  // Residual tower
  for (const auto& block : weights.residual) {
    flow = MakeResidualBlock<CPU>(scope, flow, filters, block);
  }

  // Policy head
  const int policy_conv_size = weights.policy.biases.size();
  auto conv_pol = MakeConvBlock<CPU>(scope, flow, 1, filters, policy_conv_size,
                                     weights.policy);
  if (CPU) {
    // conv_pol = Transpose(scope, conv_pol, {0, 3, 1, 2});
  }
  conv_pol =
      Reshape(scope, conv_pol, Const(scope, {-1, policy_conv_size * 8 * 8}));
  auto ip_pol_w = CPU ? MakeConst(scope, {8, 8, policy_conv_size, 1858},
                                  weights.ip_pol_w, {3, 2, 0, 1})
                      : MakeConst(scope, {policy_conv_size, 8, 8, 1858},
                                  weights.ip_pol_w, {3, 0, 1, 2});
  ip_pol_w =
      Reshape(scope, ip_pol_w, Const(scope, {policy_conv_size * 8 * 8, 1858}));
  auto ip_pol_b = MakeConst(scope, {1858}, weights.ip_pol_b);
  auto policy_fc = Add(scope, MatMul(scope, conv_pol, ip_pol_w), ip_pol_b);

  // Value head
  auto conv_val =
      MakeConvBlock<CPU>(scope, flow, 1, filters, 32, weights.value);
  conv_val = Reshape(scope, conv_val, Const(scope, {-1, 32 * 8 * 8}));

  auto ip1_val_w =
      CPU ? MakeConst(scope, {8, 8, 32, 128}, weights.ip1_val_w, {3, 2, 0, 1})
          : MakeConst(scope, {32, 8, 8, 128}, weights.ip1_val_w, {3, 0, 1, 2});
  ip1_val_w = Reshape(scope, ip1_val_w, Const(scope, {32 * 8 * 8, 128}));
  auto ip1_val_b = MakeConst(scope, {128}, weights.ip1_val_b);
  auto value_flow =
      Relu(scope, Add(scope, MatMul(scope, conv_val, ip1_val_w), ip1_val_b));
  auto ip2_val_w = MakeConst(scope, {128, 1}, weights.ip2_val_w);
  auto ip2_val_b = MakeConst(scope, {1}, weights.ip2_val_b);
  auto value_head =
      Tanh(scope, Add(scope, MatMul(scope, value_flow, ip2_val_w), ip2_val_b));

  return {policy_fc, value_head};
}

template <bool CPU>
class TFNetworkComputation;

template <bool CPU>
class TFNetwork : public Network {
 public:
  TFNetwork(const WeightsFile& file, const OptionsDict& options);

  std::unique_ptr<NetworkComputation> NewComputation() override;

  tensorflow::Status Compute(tensorflow::Tensor& input,
                             std::vector<tensorflow::Tensor>* outputs) const;

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

 private:
  tensorflow::Scope scope_;
  std::unique_ptr<tensorflow::ClientSession> session_;

  std::unique_ptr<tensorflow::ops::Placeholder> input_;
  std::unique_ptr<tensorflow::Output> policy_head_;
  std::unique_ptr<tensorflow::Output> value_head_;
  const NetworkCapabilities capabilities_;
};

template <bool CPU>
class TFNetworkComputation : public NetworkComputation {
 public:
  TFNetworkComputation(const TFNetwork<CPU>* network) : network_(network) {}
  void AddInput(InputPlanes&& input) override {
    raw_input_.emplace_back(input);
  }
  void ComputeBlocking() override {
    PrepareInput();
    status_ = network_->Compute(input_, &output_);
    CHECK(status_.ok()) << status_.ToString();
  }

  int GetBatchSize() const override { return raw_input_.size(); }
  float GetQVal(int sample) const override {
    return output_[0].template matrix<float>()(sample, 0);
  }
  float GetDVal(int) const override { return 0.0f; }
  float GetPVal(int sample, int move_id) const override {
    return output_[1].template matrix<float>()(sample, move_id);
  }

 private:
  void PrepareInput();

  const TFNetwork<CPU>* network_;
  std::vector<InputPlanes> raw_input_;

  tensorflow::Tensor input_;
  std::vector<tensorflow::Tensor> output_;
  tensorflow::Status status_;
};

// Version for GPU.
template <>
void TFNetworkComputation<false>::PrepareInput() {
  input_ = tensorflow::Tensor(
      tensorflow::DataType::DT_FLOAT,
      {static_cast<int>(raw_input_.size()), kInputPlanes, 8, 8});

  auto flat = input_.flat<float>();
  memset(flat.data(), 0, flat.size() * sizeof(*flat.data()));
  auto iter = flat.data();
  for (const auto& sample : raw_input_) {
    CHECK_EQ(sample.size(), kInputPlanes);
    for (const auto& plane : sample) {
      for (auto bit : IterateBits(plane.mask)) {
        *(iter + bit) = plane.value;
      }
      iter += 64;
    }
  }
}

// Version for CPU.
template <>
void TFNetworkComputation<true>::PrepareInput() {
  input_ = tensorflow::Tensor(
      tensorflow::DataType::DT_FLOAT,
      {static_cast<int>(raw_input_.size()), 8, 8, kInputPlanes});

  auto flat = input_.flat<float>();
  memset(flat.data(), 0, flat.size() * sizeof(*flat.data()));
  auto* data = flat.data();
  for (size_t input_idx = 0; input_idx < raw_input_.size(); ++input_idx) {
    const auto& sample = raw_input_[input_idx];
    int base = kInputPlanes * 8 * 8 * input_idx;

    CHECK_EQ(sample.size(), kInputPlanes);
    for (int plane_idx = 0; plane_idx < kInputPlanes; ++plane_idx) {
      const auto& plane = sample[plane_idx];
      for (auto bit : IterateBits(plane.mask)) {
        data[base + bit * kInputPlanes + plane_idx] = plane.value;
      }
    }
  }
}  // namespace

template <bool CPU>
TFNetwork<CPU>::TFNetwork(const WeightsFile& file,
                          const OptionsDict& /*options*/)
    : scope_(Scope::NewRootScope()),
      capabilities_{file.format().network_format().input()} {
  const LegacyWeights weights(file.weights());
  tensorflow::SessionOptions session_options;
  if (CPU) (*session_options.config.mutable_device_count())["GPU"] = 0;
  session_ =
      std::make_unique<tensorflow::ClientSession>(scope_, session_options);

  if (CPU) {
    input_ = std::make_unique<Placeholder>(
        scope_, DataType::DT_FLOAT,
        Placeholder::Shape({-1, 8, 8, kInputPlanes}));
  } else {
    input_ = std::make_unique<Placeholder>(
        scope_, DataType::DT_FLOAT,
        Placeholder::Shape({-1, kInputPlanes, 8, 8}));
  }

  auto output = MakeNetwork<CPU>(scope_, *input_, weights);
  CHECK(scope_.ok()) << scope_.status().ToString();
  policy_head_ = std::make_unique<Output>(output.first);
  value_head_ = std::make_unique<Output>(output.second);

  // First request to tensorflow is slow (0.6s), so doing an empty request for
  // preheating.
  auto fake_request = NewComputation();
  fake_request->AddInput(InputPlanes(kInputPlanes));
  fake_request->ComputeBlocking();
}

template <bool CPU>
tensorflow::Status TFNetwork<CPU>::Compute(tensorflow::Tensor& input,
                                           std::vector<Tensor>* outputs) const {
  return session_->Run({{*input_, input}}, {*value_head_, *policy_head_},
                       outputs);
}

template <bool CPU>
std::unique_ptr<NetworkComputation> TFNetwork<CPU>::NewComputation() {
  return std::make_unique<TFNetworkComputation<CPU>>(this);
}

template <bool CPU>
std::unique_ptr<Network> MakeTFNetwork(const WeightsFile& weights,
                                       const OptionsDict& options) {
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by Tensorflow backend.");
  }
  if (weights.format().network_format().policy() !=
      pblczero::NetworkFormat::POLICY_CLASSICAL) {
    throw Exception("Policy format " +
                    std::to_string(weights.format().network_format().policy()) +
                    " is not supported by Tensorflow backend.");
  }
  if (weights.format().network_format().value() !=
      pblczero::NetworkFormat::VALUE_CLASSICAL) {
    throw Exception("Value format " +
                    std::to_string(weights.format().network_format().value()) +
                    " is not supported by Tensorflow backend.");
  }
  return std::make_unique<TFNetwork<CPU>>(weights, options);
}

REGISTER_NETWORK("tensorflow-cc-cpu", MakeTFNetwork<true>, 90)
REGISTER_NETWORK("tensorflow-cc", MakeTFNetwork<false>, 80)

}  // namespace
}  // namespace lczero
