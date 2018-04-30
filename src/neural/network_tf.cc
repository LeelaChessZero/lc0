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

#include "neural/factory.h"
#include "utils/bititer.h"
#include "utils/optionsdict.h"
#include "utils/transpose.h"

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

namespace lczero {

using namespace tensorflow;
using namespace tensorflow::ops;

namespace {

Output MakeConst(const Scope& scope, TensorShape shape,
                 const std::vector<float>& values,
                 const std::vector<int>& order = {}) {
  auto tensor = Tensor(DataType::DT_FLOAT, shape);
  CHECK_EQ(tensor.NumElements(), values.size()) << shape.DebugString();

  std::vector<int> dims;
  for (const auto& x : shape) {
    dims.push_back(x.size);
  }
  TransposeTensor(dims, order, values, tensor.flat<float>().data());

  return Const(scope, tensor);
}

Output MakeVals(const Scope& scope, TensorShape shape, float val) {
  auto tensor = Tensor(DataType::DT_FLOAT, shape);
  std::fill_n(tensor.flat<float>().data(), tensor.NumElements(), val);
  return Const(scope, tensor);
}

Output Zeros(const Scope& scope, TensorShape shape) {
  return MakeVals(scope, shape, 0.0f);
}

Output Ones(const Scope& scope, TensorShape shape) {
  return MakeVals(scope, shape, 1.0f);
}

Output MakeConvBlock(const Scope& scope, Input input, int channels,
                     int input_channels, int output_channels,
                     const Weights::ConvBlock& weights,
                     Input* mixin = nullptr) {
  auto w_conv =
      MakeConst(scope, {channels, channels, input_channels, output_channels},
                weights.weights, {3, 2, 0, 1});

  // auto b_conv = MakeConst(scope, {output_channels}, weights.biases);
  auto conv2d = Conv2D(scope, input, w_conv, {1, 1, 1, 1}, "SAME",
                       Conv2D::DataFormat("NCHW"));

  auto batch_norm =
      FusedBatchNorm(scope, conv2d, Ones(scope, {output_channels}),
                     Zeros(scope, {output_channels}),
                     MakeConst(scope, {output_channels}, weights.bn_means),
                     MakeConst(scope, {output_channels}, weights.bn_stddivs),
                     FusedBatchNorm::DataFormat("NCHW").IsTraining(false))
          .y;

  if (mixin) {
    batch_norm = Add(scope, batch_norm, *mixin);
  }
  return Relu(scope, batch_norm);
}

Output MakeResidualBlock(const Scope& scope, Input input, int channels,
                         const Weights::Residual& weights) {
  auto block1 =
      MakeConvBlock(scope, input, 3, channels, channels, weights.conv1);
  auto block2 = MakeConvBlock(scope, block1, 3, channels, channels,
                              weights.conv2, &input);
  return block2;
}

std::pair<Output, Output> MakeNetwork(const Scope& scope, Input input,
                                      const Weights& weights) {
  const int filters = weights.input.weights.size() / kInputPlanes / 9;

  // Input convolution.
  auto flow =
      MakeConvBlock(scope, input, 3, kInputPlanes, filters, weights.input);

  // Residual tower
  for (const auto& block : weights.residual) {
    flow = MakeResidualBlock(scope, flow, filters, block);
  }

  // Policy head
  auto conv_pol = MakeConvBlock(scope, flow, 1, filters, 32, weights.policy);
  conv_pol = Reshape(scope, conv_pol, Const(scope, {-1, 32 * 8 * 8}));
  auto ip_pol_w = MakeConst(scope, {32 * 8 * 8, 1858}, weights.ip_pol_w);
  auto ip_pol_b = MakeConst(scope, {1858}, weights.ip_pol_b);
  auto policy_fc = Add(scope, MatMul(scope, conv_pol, ip_pol_w), ip_pol_b);
  auto policy_head = Softmax(scope, policy_fc);

  // Value head
  auto conv_val = MakeConvBlock(scope, flow, 1, filters, 32, weights.value);
  conv_val = Reshape(scope, conv_val, Const(scope, {-1, 32 * 8 * 8}));
  auto ip1_val_w = MakeConst(scope, {32 * 8 * 8, 128}, weights.ip1_val_w);
  auto ip1_val_b = MakeConst(scope, {128}, weights.ip1_val_b);
  auto value_flow =
      Relu(scope, Add(scope, MatMul(scope, conv_val, ip1_val_w), ip1_val_b));
  auto ip2_val_w = MakeConst(scope, {128, 1}, weights.ip2_val_w);
  auto ip2_val_b = MakeConst(scope, {1}, weights.ip2_val_b);
  auto value_head =
      Tanh(scope, Add(scope, MatMul(scope, value_flow, ip2_val_w), ip2_val_b));

  return {policy_head, value_head};
}

class TFNetworkComputation;
class TFNetwork : public Network {
 public:
  TFNetwork(const Weights& weights, const OptionsDict& options);

  std::unique_ptr<NetworkComputation> NewComputation() override;

  tensorflow::Status Compute(tensorflow::Tensor& input,
                             std::vector<tensorflow::Tensor>* outputs) const;

 private:
  tensorflow::Scope scope_;
  tensorflow::ClientSession session_;

  std::unique_ptr<tensorflow::ops::Placeholder> input_;
  std::unique_ptr<tensorflow::Output> policy_head_;
  std::unique_ptr<tensorflow::Output> value_head_;
};

class TFNetworkComputation : public NetworkComputation {
 public:
  TFNetworkComputation(const TFNetwork* network) : network_(network) {}
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
    return output_[0].matrix<float>()(sample, 0);
  }
  float GetPVal(int sample, int move_id) const override {
    return output_[1].matrix<float>()(sample, move_id);
  }

 private:
  void PrepareInput() {
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

  const TFNetwork* network_;
  std::vector<InputPlanes> raw_input_;

  tensorflow::Tensor input_;
  std::vector<tensorflow::Tensor> output_;
  tensorflow::Status status_;
};

TFNetwork::TFNetwork(const Weights& weights, const OptionsDict& options)
    : scope_(Scope::NewRootScope()), session_(scope_) {
  input_ = std::make_unique<Placeholder>(
      scope_, DataType::DT_FLOAT, Placeholder::Shape({-1, kInputPlanes, 8, 8}));

  auto output = MakeNetwork(scope_, *input_, weights);
  CHECK(scope_.ok()) << scope_.status().ToString();

  policy_head_ = std::make_unique<Output>(output.first);
  value_head_ = std::make_unique<Output>(output.second);

  // First request to tensorflow is slow (0.6s), so doing an empty request for
  // preheating.
  auto fake_request = NewComputation();
  fake_request->AddInput(InputPlanes(kInputPlanes));
  fake_request->ComputeBlocking();
}

tensorflow::Status TFNetwork::Compute(tensorflow::Tensor& input,
                                      std::vector<Tensor>* outputs) const {
  return session_.Run({{*input_, input}}, {*value_head_, *policy_head_},
                      outputs);
}

std::unique_ptr<NetworkComputation> TFNetwork::NewComputation() {
  return std::make_unique<TFNetworkComputation>(this);
}

}  // namespace

REGISTER_NETWORK("tensorflow", TFNetwork, 100);

}  // namespace lczero