/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#if __has_include("dml_provider_factory.h")
#include "dml_provider_factory.h"
#define USE_DML
#endif

#include "cpu_provider_factory.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "onnxruntime_cxx_api.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/logging.h"

namespace lczero {
namespace {

enum class OnnxProvider { CPU, CUDA, DML };

class OnnxNetwork;

template <typename DataType>
class OnnxComputation : public NetworkComputation {
 public:
  OnnxComputation(OnnxNetwork* network);
  void AddInput(InputPlanes&& input) override;
  int GetBatchSize() const override { return raw_input_.size(); }
  void ComputeBlocking() override;
  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  Ort::Value PrepareInputs(int start, int batch_size);

  OnnxNetwork* network_;
  std::vector<InputPlanes> raw_input_;
  std::vector<DataType> input_tensor_data_;
  std::vector<Ort::Value> output_tensors_;
  std::vector<std::vector<DataType>> output_tensors_data_;
  std::vector<size_t> output_tensors_step_;
};

class OnnxNetwork : public Network {
 public:
  OnnxNetwork(const WeightsFile& file, const OptionsDict& options,
              OnnxProvider provider, int gpu, bool fp16, int batch_size,
              int steps);
  std::unique_ptr<NetworkComputation> NewComputation() override {
    if (fp16_) {
      return std::make_unique<OnnxComputation<Ort::Float16_t>>(this);
    } else {
      return std::make_unique<OnnxComputation<float>>(this);
    }
  }
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  Ort::Env onnx_env_;
  // Prepare sessions for this many multiples of the batch size;
  int steps_;
  std::vector<Ort::Session> session_;
  std::vector<std::string> inputs_;
  // Points to strings in inputs_.
  std::vector<const char*> inputs_cstr_;
  std::vector<std::string> outputs_;
  // Points to strings in outputs_.
  std::vector<const char*> outputs_cstr_;
  // Indices in output_cstr_ vector.
  int policy_head_ = -1;
  int wdl_head_ = -1;
  int value_head_ = -1;
  int mlh_head_ = -1;
  NetworkCapabilities capabilities_;
  bool fp16_;
  // The batch size to use, or -1 for variable.
  int batch_size_;
  static constexpr int max_batch_size_ = 1024;
  // For conditional locking if running the DML provider.
  OnnxProvider provider_;
  std::mutex lock_;
};

template <typename DataType>
OnnxComputation<DataType>::OnnxComputation(OnnxNetwork* network)
    : network_(network) {
  output_tensors_data_.resize(network_->outputs_.size());
  output_tensors_step_.resize(network_->outputs_.size());
  output_tensors_step_[network_->policy_head_] = 1858;
  output_tensors_data_[network_->policy_head_] =
      std::vector<DataType>(1858 * network_->max_batch_size_);
  if (network_->wdl_head_ != -1) {
    output_tensors_step_[network_->wdl_head_] = 3;
    output_tensors_data_[network_->wdl_head_] =
        std::vector<DataType>(3 * network_->max_batch_size_);
  }
  if (network_->value_head_ != -1) {
    output_tensors_step_[network_->value_head_] = 1;
    output_tensors_data_[network_->value_head_] =
        std::vector<DataType>(network_->max_batch_size_);
  }
  if (network_->mlh_head_ != -1) {
    output_tensors_step_[network_->mlh_head_] = 1;
    output_tensors_data_[network_->mlh_head_] =
        std::vector<DataType>(network_->max_batch_size_);
  }
}

template <typename DataType>
void OnnxComputation<DataType>::AddInput(InputPlanes&& input) {
  raw_input_.emplace_back(input);
  if (raw_input_.size() > network_->max_batch_size_) {
    throw Exception("NN input exceeds max batch size of " +
                    std::to_string(network_->max_batch_size_) + ".");
  }
}

float AsFloat(float x) { return x; }
float AsFloat(Ort::Float16_t x) { return FP16toFP32(x); }

template <typename DataType>
float OnnxComputation<DataType>::GetQVal(int sample) const {
  if (network_->wdl_head_ != -1) {
    const auto& data = output_tensors_data_[network_->wdl_head_];
    return AsFloat(data[sample * 3 + 0]) - AsFloat(data[sample * 3 + 2]);
  } else {
    const auto& data = output_tensors_data_[network_->value_head_];
    return AsFloat(data[sample]);
  }
}

template <typename DataType>
float OnnxComputation<DataType>::GetDVal(int sample) const {
  if (network_->wdl_head_ == -1) return 0.0f;
  const auto& data = output_tensors_data_[network_->wdl_head_];
  return AsFloat(data[sample * 3 + 1]);
}

template <typename DataType>
float OnnxComputation<DataType>::GetPVal(int sample, int move_id) const {
  const auto& data = output_tensors_data_[network_->policy_head_];
  return AsFloat(data[sample * 1858 + move_id]);
}

template <typename DataType>
float OnnxComputation<DataType>::GetMVal(int sample) const {
  if (network_->mlh_head_ == -1) return 0.0f;
  const auto& data = output_tensors_data_[network_->mlh_head_];
  return AsFloat(data[sample]);
}

template <typename DataType>
Ort::Value OnnxComputation<DataType>::PrepareInputs(int start, int batch_size) {
  input_tensor_data_.clear();
  input_tensor_data_.resize(batch_size * kInputPlanes * 8 * 8);
  auto iter = input_tensor_data_.data();
  int end = std::min(start + batch_size, static_cast<int>(raw_input_.size()));
  for (int i = start; i < end; i++) {
    for (const auto& plane : raw_input_[i]) {
      for (auto bit : IterateBits(plane.mask)) {
        if (std::is_same<Ort::Float16_t, DataType>::value) {
          *(iter + bit) = FP32toFP16(plane.value);
        } else {
          *(iter + bit) = plane.value;
        }
      }
      iter += 64;
    }
  }
  for (int i = end; i < start + batch_size; i++) {
    for (int j = 0; j < kInputPlanes * 64; j++) {
      *iter++ = 0;
    }
  }

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  output_tensors_.clear();
  for (size_t i = 0; i < output_tensors_step_.size(); i++) {
    int size = output_tensors_step_[i];
    int64_t dims[] = {batch_size, size};
    output_tensors_.emplace_back(Ort::Value::CreateTensor<DataType>(
        memory_info, output_tensors_data_[i].data() + start * size,
        size * batch_size, dims, 2));
  }

  int64_t dims[] = {batch_size, kInputPlanes, 8, 8};
  return Ort::Value::CreateTensor<DataType>(memory_info,
                                            input_tensor_data_.data(),
                                            input_tensor_data_.size(), dims, 4);
}

template <typename DataType>
void OnnxComputation<DataType>::ComputeBlocking() {
  int batch_size = network_->batch_size_;
  if (batch_size < 0) batch_size = raw_input_.size();

  for (size_t i = 0; i < raw_input_.size();) {
    int step = (raw_input_.size() - i + batch_size - 1) / batch_size;
    if (step > network_->steps_) step = network_->steps_;
    int batch = batch_size * step;

    auto input_tensor = PrepareInputs(i, batch);
    if (network_->provider_ == OnnxProvider::DML) network_->lock_.lock();
    network_->session_[step - 1].Run(
        {}, network_->inputs_cstr_.data(), &input_tensor, 1,
        network_->outputs_cstr_.data(), output_tensors_.data(),
        output_tensors_.size());
    if (network_->provider_ == OnnxProvider::DML) network_->lock_.unlock();
    i += batch;
  }
}

Ort::SessionOptions GetOptions(OnnxProvider provider, int gpu, int batch_size) {
  Ort::SessionOptions options;
  OrtCUDAProviderOptions cuda_options;
  // options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (batch_size > 0) {
    // Override the default (variable) batch size.
    Ort::ThrowOnError(
        OrtGetApiBase()
            ->GetApi(ORT_API_VERSION)
            ->AddFreeDimensionOverrideByName(options, "batch", batch_size));
  }

  switch (provider) {
    case OnnxProvider::DML:
      options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      options.DisableMemPattern();
#ifdef USE_DML
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_DML(options, gpu));
#else
      throw Exception("ONNX backend internal error.");
#endif
      break;
    case OnnxProvider::CUDA:
      cuda_options.device_id = gpu;
      options.AppendExecutionProvider_CUDA(cuda_options);
      break;
    case OnnxProvider::CPU:
      // Doesn't really work. :-( There are two execution providers (CUDA and
      // CPU) already added, don't know how to force it to use CPU.
      auto status = OrtSessionOptionsAppendExecutionProvider_CPU(options, 0);
      if (status) {
        std::string error_message = Ort::GetApi().GetErrorMessage(status);
        OrtErrorCode error_code = Ort::GetApi().GetErrorCode(status);
        Ort::GetApi().ReleaseStatus(status);
        throw Exception("ONNX CPU error " + std::to_string(error_code) + ": " +
                        error_message);
      }
      break;
  }
  return options;
}

OnnxNetwork::OnnxNetwork(const WeightsFile& file, const OptionsDict&,
                         OnnxProvider provider, int gpu, bool fp16,
                         int batch_size, int steps)
    : onnx_env_(ORT_LOGGING_LEVEL_WARNING, "lc0"),
      steps_(steps),
      capabilities_{file.format().network_format().input(),
                    file.format().network_format().moves_left()},
      fp16_(fp16),
      batch_size_(batch_size),
      provider_(provider) {
  // Sanity checks.
  if (batch_size_ < 0) steps_ = 1;
  if (batch_size_ * steps > max_batch_size_) {
    batch_size_ = max_batch_size_ / steps_;
  }

  for (int step = 1; step <= steps_; step++)
    session_.emplace_back(onnx_env_, file.onnx_model().model().data(),
                          file.onnx_model().model().size(),
                          GetOptions(provider, gpu, batch_size_ * step));

  const auto& md = file.onnx_model();
  if (!md.has_input_planes()) {
    throw Exception("NN doesn't have input planes defined.");
  }
  inputs_.emplace_back(md.input_planes());
  if (!md.has_output_policy()) {
    throw Exception("NN doesn't have policy head defined.");
  }
  policy_head_ = outputs_.size();
  outputs_.emplace_back(md.output_policy());
  if (md.has_output_wdl()) {
    wdl_head_ = outputs_.size();
    outputs_.emplace_back(md.output_wdl());
  } else if (md.has_output_value()) {
    value_head_ = outputs_.size();
    outputs_.emplace_back(md.output_value());
  } else {
    throw Exception("NN doesn't have value head.");
  }
  if (md.has_output_mlh()) {
    mlh_head_ = outputs_.size();
    outputs_.emplace_back(md.output_mlh());
  }
  std::transform(inputs_.begin(), inputs_.end(),
                 std::back_inserter(inputs_cstr_),
                 [](const auto& x) { return x.c_str(); });
  std::transform(outputs_.begin(), outputs_.end(),
                 std::back_inserter(outputs_cstr_),
                 [](const auto& x) { return x.c_str(); });
}

template <OnnxProvider kProvider>
std::unique_ptr<Network> MakeOnnxNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& opts) {
  if (!w) throw Exception("The ONNX backend requires a network file.");

  int gpu = opts.GetOrDefault<int>("gpu", 0);

  int batch_size =
      opts.GetOrDefault<int>("batch", kProvider == OnnxProvider::DML ? 16 : -1);

  int steps =
      opts.GetOrDefault<int>("steps", kProvider == OnnxProvider::DML ? 8 : 1);

  if (batch_size <= 0) batch_size = -1;  // Variable batch size.

  bool fp16 = opts.GetOrDefault<bool>(
      "fp16", kProvider == OnnxProvider::CPU ? false : true);

  if (w->has_onnx_model()) {
    return std::make_unique<OnnxNetwork>(*w, opts, kProvider, gpu, false,
                                         batch_size, steps);
  } else {
    if (w->format().network_format().network() !=
            pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
        w->format().network_format().network() !=
            pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
      throw Exception("Network format " +
                      pblczero::NetworkFormat::NetworkStructure_Name(
                          w->format().network_format().network()) +
                      " is not supported by the ONNX backend.");
    }
    if (w->format().network_format().policy() !=
            pblczero::NetworkFormat::POLICY_CLASSICAL &&
        w->format().network_format().policy() !=
            pblczero::NetworkFormat::POLICY_CONVOLUTION &&
        w->format().network_format().policy() !=
            pblczero::NetworkFormat::POLICY_ATTENTION) {
      throw Exception("Policy format " +
                      pblczero::NetworkFormat::PolicyFormat_Name(
                          w->format().network_format().policy()) +
                      " is not supported by the ONNX backend.");
    }
    if (w->format().network_format().value() !=
            pblczero::NetworkFormat::VALUE_CLASSICAL &&
        w->format().network_format().value() !=
            pblczero::NetworkFormat::VALUE_WDL) {
      throw Exception("Value format " +
                      pblczero::NetworkFormat::ValueFormat_Name(
                          w->format().network_format().value()) +
                      " is not supported by the ONNX backend.");
    }
    if (w->format().network_format().default_activation() !=
            pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
        w->format().network_format().default_activation() !=
            pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
      throw Exception("Default activation " +
                      pblczero::NetworkFormat::DefaultActivation_Name(
                          w->format().network_format().default_activation()) +
                      " is not supported by the ONNX backend.");
    }
    WeightsToOnnxConverterOptions converter_options;
    converter_options.opset = opts.GetOrDefault<int>("opset", 17);
    converter_options.data_type_ =
        fp16 ? WeightsToOnnxConverterOptions::DataType::kFloat16
             : WeightsToOnnxConverterOptions::DataType::kFloat32;
    auto converted = ConvertWeightsToOnnx(*w, converter_options);
    return std::make_unique<OnnxNetwork>(converted, opts, kProvider, gpu, fp16,
                                         batch_size, steps);
  }
}

#ifdef USE_DML
REGISTER_NETWORK("onnx-dml", MakeOnnxNetwork<OnnxProvider::DML>, 63)
#endif
REGISTER_NETWORK("onnx-cuda", MakeOnnxNetwork<OnnxProvider::CUDA>, 61)
REGISTER_NETWORK("onnx-cpu", MakeOnnxNetwork<OnnxProvider::CPU>, 62)

}  // namespace
}  // namespace lczero
