/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include <fstream>
#include <iostream>
#include <sstream>

#include "neural/factory.h"
#include "neural/tensorflow/tf_wrappers.h"
#include "utils/bititer.h"
#include "utils/exception.h"

namespace lczero {
namespace {

std::string LoadFile() {
  std::ifstream fin(
      "/home/crem/dev/lczero-training/tf/FrozenModel/FrozenModel.pb",
      std::ios::binary);
  std::stringstream sstr;
  sstr << fin.rdbuf();
  return sstr.str();
}

class TFV2Network;
class TFV2NetworkComputation : public NetworkComputation {
 public:
  TFV2NetworkComputation(TFV2Network* network) : network_(network) {}

  void AddInput(InputPlanes&& input) override {
    raw_input_.emplace_back(std::move(input));
  }
  int GetBatchSize() const override { return raw_input_.size(); }
  void ComputeBlocking() override;

  float GetQVal(int sample) const override {
    float* buf = static_cast<float*>(value_output_.GetBuffer());
    return buf[3 * sample] - buf[3 * sample + 2];
  }
  float GetDVal(int sample) const override {
    float* buf = static_cast<float*>(value_output_.GetBuffer());
    return buf[3 * sample + 1];
  }
  float GetPVal(int sample, int move_id) const override {
    float* buf = static_cast<float*>(policy_output_.GetBuffer());
    return buf[sample * 1858 + move_id];
  }

 private:
  void PrepareInput();
  TFV2Network* const network_;
  std::vector<InputPlanes> raw_input_;
  TFTensor value_output_;
  TFTensor policy_output_;
};

class TFV2Network : public Network {
 public:
  TFV2Network(const WeightsFile& file, const OptionsDict&)
      : capabilities_{file.format().network_format().input()} {
    std::cout << "TensorFlow Version: " << TF_Version() << std::endl;
    graph_.ImportGraphDef(TFBuffer(LoadFile()));
    session_ = std::make_unique<TFSession>(graph_);

    input_head_ = TF_Output{graph_.GetOperationByName("input_1"), 0};
    output_value_head_ =
        TF_Output{graph_.GetOperationByName("softmax/Softmax"), 0};
    output_policy_head_ =
        TF_Output{graph_.GetOperationByName("tf_op_layer_MatMul/MatMul"), 0};

    // First request to tensorflow is slow (0.6s), so doing an empty request for
    // preheating.
    auto fake_request = NewComputation();
    fake_request->AddInput(InputPlanes(kInputPlanes));
    fake_request->ComputeBlocking();
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<TFV2NetworkComputation>(this);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

 private:
  std::vector<TFTensor> Compute(TFTensor input) {
    return session_->Run({input_head_}, {&input},
                         {output_value_head_, output_policy_head_});
  }

  TF_Output input_head_;
  TF_Output output_value_head_;
  TF_Output output_policy_head_;

  NetworkCapabilities capabilities_;
  TFGraph graph_;
  std::unique_ptr<TFSession> session_;
  friend class TFV2NetworkComputation;
};

std::unique_ptr<Network> MakeTFV2Network(const WeightsFile& weights,
                                         const OptionsDict& options) {
  return std::make_unique<TFV2Network>(weights, options);
}

void TFV2NetworkComputation::ComputeBlocking() {
  TFTensor input(TF_FLOAT,
                 std::vector<int64_t>{static_cast<int64_t>(raw_input_.size()),
                                      kInputPlanes, 8, 8});

  auto data = static_cast<float*>(input.GetBuffer());
  std::memset(data, 0, input.GetByteSize());
  for (size_t input_idx = 0; input_idx < raw_input_.size(); ++input_idx) {
    const auto& sample = raw_input_[input_idx];
    int base = kInputPlanes * 8 * 8 * input_idx;
    for (int plane_idx = 0; plane_idx < kInputPlanes; ++plane_idx) {
      const auto& plane = sample[plane_idx];
      for (auto bit : IterateBits(plane.mask)) {
        // data[base + bit * kInputPlanes + plane_idx] = plane.value;
        data[base + 8 * 8 * plane_idx + bit] = plane.value;
      }
    }
  }

  auto outputs = network_->Compute(std::move(input));
  value_output_ = std::move(outputs[0]);
  policy_output_ = std::move(outputs[1]);
  // LOGFILE << value_output_.DebugString();
  // LOGFILE << value_output_.Dump();
  // LOGFILE << policy_output_.DebugString();
  // LOGFILE << policy_output_.Dump();
}

REGISTER_NETWORK("tensorflow-v2", MakeTFV2Network, 100000)

}  // namespace
}  // namespace lczero