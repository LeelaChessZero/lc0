/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include <cassert>

#include "neural/factory.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "neural/xla/onnx2hlo.h"
#include "neural/xla/xla_runner.h"
#include "utils/bititer.h"

namespace lczero {
namespace {

class XlaNetwork;
class XlaComputation : public NetworkComputation {
 public:
  XlaComputation();

  void AddInput(InputPlanes&& input) override;
  int GetBatchSize() const override;
  void ComputeBlocking() override;
  float GetQVal(int sample) const override { return 0.0f; }
  float GetDVal(int sample) const override { return 0.0f; }
  float GetPVal(int sample, int move_id) const override { return 0.0f; }
  float GetMVal(int sample) const override { return 0.0f; }

 private:
  const XlaNetwork* network_;
  constexpr static size_t kBatchSize = kInputPlanes * 8 * 8;
  size_t batch_size_ = 0;
  std::vector<float> raw_input_planes_;
};

struct XlaNetworkOptions {
  std::optional<size_t> output_value_idx;
  std::optional<size_t> output_wdl_idx;
  std::optional<size_t> output_policy_idx;
  std::optional<size_t> output_mlh_idx;
};

class XlaNetwork : public Network {
 public:
  XlaNetwork(std::unique_ptr<XlaRunner> runner,
             const XlaNetworkOptions& options,
             const pblczero::NetworkFormat& format);

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<XlaComputation>();
  }

 private:
  std::unique_ptr<XlaRunner> runner_;
  XlaNetworkOptions options_;
  NetworkCapabilities capabilities_;

  friend class XlaComputation;
};

XlaComputation::XlaComputation() {
  raw_input_planes_.reserve(GetBatchSize() * kBatchSize);
}

int XlaComputation::GetBatchSize() const {
  return network_->runner_->GetMaxBatchSize();
}

void XlaComputation::AddInput(InputPlanes&& input) {
  assert(batch_size_ < (size_t)GetBatchSize());
  ++batch_size_;
  float* start = raw_input_planes_.data() + raw_input_planes_.size();
  raw_input_planes_.resize(raw_input_planes_.size() + kBatchSize);
  for (const auto& plane : input) {
    float* ptr = start;
    for (auto bit : IterateBits(plane.mask)) ptr[bit] = plane.value;
    start += 8 * 8;
  }
}

XlaNetwork::XlaNetwork(std::unique_ptr<XlaRunner> runner,
                       const XlaNetworkOptions& options,
                       const pblczero::NetworkFormat& format)
    : runner_(std::move(runner)),
      options_(options),
      capabilities_{format.input(), format.moves_left()} {}

XlaNetworkOptions FillXlaRunnerFromOnnx(const pblczero::OnnxModel& onnx_model,
                                        XlaRunner* runner) {
  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_model.model());

  std::unordered_map<std::string, size_t> constant_to_parameter_idx;
  std::unordered_map<std::string, size_t> input_to_parameter_idx;
  std::unordered_map<std::string, size_t> output_to_parameter_idx;

  auto add_tensors = [](const std::vector<Onnx2HloResult::NamedTensor>& tensors,
                        std::unordered_map<std::string, size_t>& map) {
    for (const auto& tensor : tensors) {
      auto iter = map.find(tensor.name);
      if (iter == map.end()) {
        map[tensor.name] = tensor.param_idx;
      } else if (iter->second != tensor.param_idx) {
        throw Exception("Inconsistent index for " + tensor.name);
      }
    }
  };

  // DO NOT SUBMIT, pass the correct batch size.
  for (size_t batch_size : {512}) {
    CERR << "Building HLO for batch size " << batch_size << "...";
    auto conversion = ConvertOnnxToHlo(onnx, batch_size, {});
    add_tensors(conversion.constants, constant_to_parameter_idx);
    add_tensors(conversion.inputs, input_to_parameter_idx);
    add_tensors(conversion.outputs, output_to_parameter_idx);
    runner->AddModule(batch_size, conversion.hlo_module);
  }

  std::vector<std::unique_ptr<XlaTensor>> constants;
  constants.resize(constant_to_parameter_idx.size() +
                   input_to_parameter_idx.size());
  for (const auto& initializer : onnx.graph().initializer()) {
    auto iter = constant_to_parameter_idx.find(std::string(initializer.name()));
    if (iter == constant_to_parameter_idx.end()) continue;
    auto idx = iter->second;
    assert(idx < constants.size());
    constants[idx] = OnnxTensorToXlaTensor(initializer);
  }

  CERR << "Transferring constants...";
  runner->SetFrozenInputs(std::move(constants));
  CERR << "Done.";

  XlaNetworkOptions options;
  if (input_to_parameter_idx.size() != 1 ||
      input_to_parameter_idx.begin()->first != onnx_model.input_planes()) {
    throw Exception("Expected a single input named " +
                    std::string(onnx_model.input_planes()));
  }
  if (onnx_model.has_output_value()) {
    options.output_value_idx =
        output_to_parameter_idx.at(std::string(onnx_model.output_value()));
  }
  if (onnx_model.has_output_wdl()) {
    options.output_wdl_idx =
        output_to_parameter_idx.at(std::string(onnx_model.output_wdl()));
  }
  if (onnx_model.has_output_policy()) {
    options.output_policy_idx =
        output_to_parameter_idx.at(std::string(onnx_model.output_policy()));
  }
  if (onnx_model.has_output_mlh()) {
    options.output_mlh_idx =
        output_to_parameter_idx.at(std::string(onnx_model.output_mlh()));
  }
  return options;
}

std::unique_ptr<Network> MakeXlaNetwork(const std::optional<WeightsFile>& w,
                                        const OptionsDict&) {
  if (!w) throw Exception("The XLA backend requires a network file.");
  auto runner = std::make_unique<XlaRunner>(
      "/home/crem/dev/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so");
  XlaNetworkOptions options;
  if (w->has_onnx_model()) {
    options = FillXlaRunnerFromOnnx(w->onnx_model(), runner.get());
  } else {
    CERR << "Converting weights to ONNX first.";
    WeightsToOnnxConverterOptions onnx_converter_options;
    auto converted = ConvertWeightsToOnnx(*w, onnx_converter_options);
    options = FillXlaRunnerFromOnnx(converted.onnx_model(), runner.get());
  }

  return std::make_unique<XlaNetwork>(std::move(runner), options,
                                      w->format().network_format());
}

REGISTER_NETWORK("xla", MakeXlaNetwork, 143)

}  // namespace
}  // namespace lczero
