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

class Lc0InputTensor : public XlaTensor {
 public:
  Lc0InputTensor(size_t max_batch_size)
      : max_batch_size_(max_batch_size),
        // TODO replace with make_unique_for_overwrite() once C++20 is
        // available.
        data_(new float[GetTensorByteSizeForBatch(max_batch_size)]),
        shape_{0, kInputPlanes, 8, 8} {}

  const std::vector<int64_t>& shape() const override { return shape_; }
  const void* data() const override { return data_.get(); }
  size_t size() const override { return GetTensorByteSizeForBatch(shape_[0]); }
  size_t capacity() const override {
    return GetTensorByteSizeForBatch(max_batch_size_);
  }
  pblczero::XlaShapeProto::Type type() const override {
    return pblczero::XlaShapeProto::F32;
  }

  // Adds a batch to the tensor and returns a pointer to the start of the its
  // part in the buffer. Does NOT initialize the data with zeros.
  float* AddBatch() {
    assert(size_t(shape_[0]) < max_batch_size_);
    auto ret = data_.get() + shape_[0] * GetTensorByteSizeForBatch(1);
    ++shape_[0];
    return ret;
  }
  size_t GetBatchSize() const { return shape_[0]; }

 private:
  static size_t GetTensorByteSizeForBatch(size_t batch_size) {
    return kInputPlanes * 8 * 8 * batch_size * sizeof(float);
  }

  const size_t max_batch_size_;
  std::unique_ptr<float[]> data_;
  std::vector<int64_t> shape_;
};

class XlaNetwork;
class XlaComputation : public NetworkComputation {
 public:
  XlaComputation(const XlaNetwork* network);
  void AddInput(InputPlanes&& input) override;
  int GetBatchSize() const override;
  void ComputeBlocking() override;
  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  const XlaNetwork* network_;
  Lc0InputTensor input_tensor_;
  std::vector<std::unique_ptr<XlaTensor>> outputs_;
};

// Indices of various heads in the HLO output.
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
    return std::make_unique<XlaComputation>(this);
  }
  int GetMiniBatchSize() const override {
    // 32 is the default prefetch size, subtract it so that backend doesn't
    // crash.
    // TODO make it better when we have a proper way to query the batch size.
    return runner_->GetMaxBatchSize() - 32;
  }

 private:
  std::unique_ptr<XlaRunner> runner_;
  XlaNetworkOptions options_;
  NetworkCapabilities capabilities_;

  friend class XlaComputation;
};

XlaComputation::XlaComputation(const XlaNetwork* network)
    : network_(network), input_tensor_(network->runner_->GetMaxBatchSize()) {}

void XlaComputation::AddInput(InputPlanes&& input) {
  float* ptr = input_tensor_.AddBatch();
  memset(ptr, 0, 8 * 8 * kInputPlanes * sizeof(float));
  for (const auto& plane : input) {
    for (auto bit : IterateBits(plane.mask)) ptr[bit] = plane.value;
    ptr += 8 * 8;
  }
}

float XlaComputation::GetQVal(int sample) const {
  if (network_->options_.output_wdl_idx) {
    const float* data = reinterpret_cast<const float*>(
        outputs_[*network_->options_.output_wdl_idx]->data());
    return data[sample * 3 + 0] - data[sample * 3 + 2];
  } else {
    const float* data = reinterpret_cast<const float*>(
        outputs_[*network_->options_.output_value_idx]->data());
    return data[sample];
  }
}

float XlaComputation::GetDVal(int sample) const {
  if (network_->options_.output_wdl_idx) {
    const float* data = reinterpret_cast<const float*>(
        outputs_[*network_->options_.output_wdl_idx]->data());
    return data[sample * 3 + 1];
  }
  return 0.0f;
}

float XlaComputation::GetPVal(int sample, int move_id) const {
  const float* data = reinterpret_cast<const float*>(
      outputs_[*network_->options_.output_policy_idx]->data());
  return data[sample * 1858 + move_id];
}

float XlaComputation::GetMVal(int sample) const {
  if (network_->options_.output_mlh_idx) {
    const float* data = reinterpret_cast<const float*>(
        outputs_[*network_->options_.output_mlh_idx]->data());
    return data[sample];
  }
  return 0.0f;
}

int XlaComputation::GetBatchSize() const {
  return input_tensor_.GetBatchSize();
}

void XlaComputation::ComputeBlocking() {
  outputs_ = network_->runner_->ExecuteBlocking({&input_tensor_});
}

XlaNetwork::XlaNetwork(std::unique_ptr<XlaRunner> runner,
                       const XlaNetworkOptions& options,
                       const pblczero::NetworkFormat& format)
    : runner_(std::move(runner)),
      options_(options),
      capabilities_{format.input(), format.moves_left()} {}

// Converts ONNX model to HLO (for various batch sizes) and adds them to the
// XlaRunner.
XlaNetworkOptions FillXlaRunnerFromOnnx(const pblczero::OnnxModel& onnx_model,
                                        XlaRunner* runner,
                                        size_t max_batch_size, size_t steps) {
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

  for (size_t i = 0; i < steps; ++i) {
    size_t batch_size = max_batch_size * (i + 1) / steps;
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

// Makes an XLA network. First converts the weights to ONNX, and then calls
// FillXlaRunnerFromOnnx to convert them further to HLO and them compile them.
std::unique_ptr<Network> MakeXlaNetwork(const std::optional<WeightsFile>& w,
                                        const OptionsDict& opts) {
  if (!w) throw Exception("The XLA backend requires a network file.");
  int device = opts.GetOrDefault<int>("device", 0);
  // Note: if the plugin_path does NOT contain a slash, it's looked up in the
  // LD_LIBRARY_PATH (and a few other system defined places). If it does contain
  // a slash, it's looked up at the exact relative or absolute path.
  auto runner = std::make_unique<XlaRunner>(
      opts.GetOrDefault<std::string>("plugin_path",
                                     "./pjrt_c_api_gpu_plugin.so")
          .c_str(),
      device);
  int max_batch_size = opts.GetOrDefault<int>("max_batch", 739);
  int steps = opts.GetOrDefault<int>("steps", 13);

  XlaNetworkOptions options;
  if (w->has_onnx_model()) {
    options = FillXlaRunnerFromOnnx(w->onnx_model(), runner.get(),
                                    max_batch_size, steps);
  } else {
    CERR << "Converting weights to ONNX first.";
    WeightsToOnnxConverterOptions onnx_converter_options;
    auto converted = ConvertWeightsToOnnx(*w, onnx_converter_options);
    options = FillXlaRunnerFromOnnx(converted.onnx_model(), runner.get(),
                                    max_batch_size, steps);
  }

  return std::make_unique<XlaNetwork>(std::move(runner), options,
                                      w->format().network_format());
}

REGISTER_NETWORK("xla", MakeXlaNetwork, -34)

}  // namespace
}  // namespace lczero
