/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021-2023 The LCZero Authors

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
#include <iomanip>
#include <iterator>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if __has_include("dml_provider_factory.h")
#include "dml_provider_factory.h"
#define USE_DML
#endif

#if __has_include("cuda_runtime.h")
#include "cuda_runtime.h"
#endif

#include "cpu_provider_factory.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "onnxruntime_cxx_api.h"
#include "utils/bf16_utils.h"
#include "utils/bititer.h"
#include "utils/commandline.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/logging.h"

namespace lczero {
namespace {

enum class OnnxProvider { CPU, CUDA, DML, ROCM, TRT };

static constexpr int kNumOutputPolicy = 1858;

#ifdef CUDART_VERSION
void CudaError(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    auto err = std::string("CUDA error: ") + cudaGetErrorString(status) + " (" +
               file + ":" + std::to_string(line) + ") ";
    throw Exception(err);
  }
}
#define ReportCUDAErrors(status) CudaError(status, __FILE__, __LINE__)
#endif

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, int value_head, int wdl_head, int policy_head,
                int mlh_head, int data_size, OnnxProvider provider,
                [[maybe_unused]] int gpu)
      : provider_(provider) {
    int outputs_size =
        std::max({value_head, wdl_head, policy_head, mlh_head}) + 1;
    output_tensors_data_.resize(outputs_size);
    output_tensors_data_device_.resize(outputs_size);
    output_tensors_step_.resize(outputs_size);
    if (wdl_head != -1) {
      wdl_output_data_.resize(3 * maxBatchSize);
    }
    switch (provider) {
      case OnnxProvider::CUDA:
      case OnnxProvider::TRT:
#ifdef CUDART_VERSION
        output_tensors_step_[policy_head] = kNumOutputPolicy;
        ReportCUDAErrors(cudaHostAlloc(
            &output_tensors_data_[policy_head],
            maxBatchSize * kNumOutputPolicy * data_size, cudaHostAllocMapped));
        if (wdl_head != -1) {
          output_tensors_step_[wdl_head] = 3;
          ReportCUDAErrors(cudaHostAlloc(&output_tensors_data_[wdl_head],
                                         maxBatchSize * 3 * data_size,
                                         cudaHostAllocMapped));
        }
        if (value_head != -1) {
          output_tensors_step_[value_head] = 1;
          ReportCUDAErrors(cudaHostAlloc(&output_tensors_data_[value_head],
                                         maxBatchSize * data_size,
                                         cudaHostAllocMapped));
        }
        if (mlh_head != -1) {
          output_tensors_step_[mlh_head] = 1;
          ReportCUDAErrors(cudaHostAlloc(&output_tensors_data_[mlh_head],
                                         maxBatchSize * data_size,
                                         cudaHostAllocMapped));
        }
        ReportCUDAErrors(
            cudaHostAlloc(&input_tensor_data_,
                          maxBatchSize * kInputPlanes * 8 * 8 * data_size,
                          cudaHostAllocMapped));
        ReportCUDAErrors(cudaHostGetDevicePointer(&input_tensor_data_device_,
                                                  input_tensor_data_, 0));
        for (int i = 0; i < outputs_size; i++) {
          ReportCUDAErrors(cudaHostGetDevicePointer(
              &output_tensors_data_device_[i], output_tensors_data_[i], 0));
        }
        memory_info_ =
            Ort::MemoryInfo{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
        break;
#endif
      default:
        output_tensors_step_[policy_head] = kNumOutputPolicy;
        output_tensors_data_[policy_head] =
            malloc(maxBatchSize * kNumOutputPolicy * data_size);
        if (wdl_head != -1) {
          output_tensors_step_[wdl_head] = 3;
          output_tensors_data_[wdl_head] = malloc(maxBatchSize * 3 * data_size);
        }
        if (value_head != -1) {
          output_tensors_step_[value_head] = 1;
          output_tensors_data_[value_head] = malloc(maxBatchSize * data_size);
        }
        if (mlh_head != -1) {
          output_tensors_step_[mlh_head] = 1;
          output_tensors_data_[mlh_head] = malloc(maxBatchSize * data_size);
        }
        input_tensor_data_ =
            malloc(maxBatchSize * kInputPlanes * 8 * 8 * data_size);
        input_tensor_data_device_ = input_tensor_data_;
        for (int i = 0; i < outputs_size; i++) {
          output_tensors_data_device_[i] = output_tensors_data_[i];
        }
        memory_info_ =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }
  }
  ~InputsOutputs() {
    switch (provider_) {
      case OnnxProvider::CUDA:
      case OnnxProvider::TRT:
#ifdef CUDART_VERSION
        ReportCUDAErrors(cudaFreeHost(input_tensor_data_));
        for (void* ptr : output_tensors_data_) {
          ReportCUDAErrors(cudaFreeHost(ptr));
        }
        break;
#endif
      default:
        free(input_tensor_data_);
        for (void* ptr : output_tensors_data_) {
          free(ptr);
        }
    }
  }
  OnnxProvider provider_;
  void* input_tensor_data_;
  void* input_tensor_data_device_;
  std::vector<void*> output_tensors_data_;
  std::vector<void*> output_tensors_data_device_;
  std::vector<size_t> output_tensors_step_;
  std::vector<float> wdl_output_data_;
  Ort::MemoryInfo memory_info_{nullptr};
};

class OnnxNetwork;

template <typename DataType>
class OnnxComputation : public NetworkComputation {
 public:
  OnnxComputation(OnnxNetwork* network);
  ~OnnxComputation();
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
  std::vector<Ort::Value> output_tensors_;
  std::unique_ptr<InputsOutputs> inputs_outputs_;
};

class OnnxNetwork : public Network {
 public:
  OnnxNetwork(const WeightsFile& file, const OptionsDict& options,
              OnnxProvider provider, bool cpu_wdl);
  std::unique_ptr<NetworkComputation> NewComputation() override {
    if (fp16_) {
      return std::make_unique<OnnxComputation<Ort::Float16_t>>(this);
    } else if (bf16_) {
      return std::make_unique<OnnxComputation<Ort::BFloat16_t>>(this);
    } else {
      return std::make_unique<OnnxComputation<float>>(this);
    }
  }
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  int GetMiniBatchSize() const override {
    return batch_size_ == -1 ? Network::GetMiniBatchSize()
                             : batch_size_ * steps_;
  }
  bool IsCpu() const override { return provider_ == OnnxProvider::CPU; }

  Ort::SessionOptions GetOptions(int threads, int batch_size, uint64_t hash);

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(
          max_batch_size_, value_head_, wdl_head_, policy_head_, mlh_head_,
          (fp16_ | bf16_) ? 2 : 4, provider_, gpu_);
    } else {
      std::unique_ptr<InputsOutputs> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

  Ort::Env onnx_env_;
  // Prepare sessions for this many multiples of the batch size;
  int steps_;
  std::vector<Ort::Session> session_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  // Indices in output_ vector.
  int policy_head_ = -1;
  int wdl_head_ = -1;
  int value_head_ = -1;
  int mlh_head_ = -1;
  NetworkCapabilities capabilities_;
  bool fp16_;
  bool bf16_;
  bool cpu_wdl_;
  // The batch size to use, or -1 for variable.
  int batch_size_;
  // The lower limit for variable batch size.
  int min_batch_size_;
  int gpu_;
  static constexpr int max_batch_size_ = 1024;
  // For conditional locking if running the DML/ROCM/TRT provider.
  OnnxProvider provider_;
  std::mutex lock_;

 private:
  std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
};

template <typename DataType>
OnnxComputation<DataType>::OnnxComputation(OnnxNetwork* network)
    : network_(network) {
  inputs_outputs_ = network_->GetInputsOutputs();
}

template <typename DataType>
OnnxComputation<DataType>::~OnnxComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
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
float AsFloat(Ort::Float16_t x) {
  uint16_t tmp;
  std::memcpy(&tmp, reinterpret_cast<uint16_t*>(&x), sizeof(uint16_t));
  return FP16toFP32(tmp);
}
float AsFloat(Ort::BFloat16_t x) {
  uint16_t tmp;
  std::memcpy(&tmp, reinterpret_cast<uint16_t*>(&x), sizeof(uint16_t));
  return BF16toFP32(tmp);
}

template <typename DataType>
float OnnxComputation<DataType>::GetQVal(int sample) const {
  if (network_->wdl_head_ != -1) {
    return inputs_outputs_->wdl_output_data_[sample * 3 + 0] -
           inputs_outputs_->wdl_output_data_[sample * 3 + 2];
  } else {
    DataType* data = static_cast<DataType*>(
        inputs_outputs_->output_tensors_data_[network_->value_head_]);
    return AsFloat(data[sample]);
  }
}

template <typename DataType>
float OnnxComputation<DataType>::GetDVal(int sample) const {
  if (network_->wdl_head_ == -1) return 0.0f;
  return inputs_outputs_->wdl_output_data_[sample * 3 + 1];
}

template <typename DataType>
float OnnxComputation<DataType>::GetPVal(int sample, int move_id) const {
  DataType* data = static_cast<DataType*>(
      inputs_outputs_->output_tensors_data_[network_->policy_head_]);
  return AsFloat(data[sample * kNumOutputPolicy + move_id]);
}

template <typename DataType>
float OnnxComputation<DataType>::GetMVal(int sample) const {
  if (network_->mlh_head_ == -1) return 0.0f;
  DataType* data = static_cast<DataType*>(
      inputs_outputs_->output_tensors_data_[network_->mlh_head_]);
  return AsFloat(data[sample]);
}

void AsDataType(float x, float* y) { *y = x; }
void AsDataType(float x, Ort::Float16_t* y) {
  uint16_t tmp = FP32toFP16(x);
  std::memcpy(reinterpret_cast<uint16_t*>(y), &tmp, sizeof(uint16_t));
}
void AsDataType(float x, Ort::BFloat16_t* y) {
  uint16_t tmp = FP32toBF16(x);
  std::memcpy(reinterpret_cast<uint16_t*>(y), &tmp, sizeof(uint16_t));
}

template <typename DataType>
Ort::Value OnnxComputation<DataType>::PrepareInputs(int start, int batch_size) {
  std::memset(inputs_outputs_->input_tensor_data_, 0,
              batch_size * kInputPlanes * 8 * 8 * sizeof(DataType));
  DataType* iter = static_cast<DataType*>(inputs_outputs_->input_tensor_data_);
  int end = std::min(start + batch_size, static_cast<int>(raw_input_.size()));
  for (int i = start; i < end; i++) {
    for (const auto& plane : raw_input_[i]) {
      DataType value;
      AsDataType(plane.value, &value);
      for (auto bit : IterateBits(plane.mask)) {
        *(iter + bit) = value;
      }
      iter += 64;
    }
  }

  output_tensors_.clear();
  for (size_t i = 0; i < inputs_outputs_->output_tensors_step_.size(); i++) {
    int size = inputs_outputs_->output_tensors_step_[i];
    int64_t dims[] = {batch_size, size};
    output_tensors_.emplace_back(Ort::Value::CreateTensor<DataType>(
        inputs_outputs_->memory_info_,
        static_cast<DataType*>(
            inputs_outputs_->output_tensors_data_device_[i]) +
            start * size,
        size * batch_size, dims, 2));
  }

  int64_t dims[] = {batch_size, kInputPlanes, 8, 8};
  return Ort::Value::CreateTensor<DataType>(
      inputs_outputs_->memory_info_,
      static_cast<DataType*>(inputs_outputs_->input_tensor_data_device_),
      batch_size * kInputPlanes * 8 * 8, dims, 4);
}

template <typename DataType>
void OnnxComputation<DataType>::ComputeBlocking() {
  int batch_size = network_->batch_size_;
  if (batch_size < 0) {
    batch_size = std::max(static_cast<int>(raw_input_.size()),
                          network_->min_batch_size_);
  }
  for (size_t i = 0; i < raw_input_.size();) {
    int step = (raw_input_.size() - i + batch_size - 1) / batch_size;
    if (step > network_->steps_) step = network_->steps_;
    int batch = batch_size * step;

    auto input_tensor = PrepareInputs(i, batch);

    Ort::IoBinding binding{network_->session_[step - 1]};
    binding.BindInput(network_->inputs_[0].c_str(), input_tensor);
    for (size_t i = 0; i < output_tensors_.size(); i++) {
      binding.BindOutput(network_->outputs_[i].c_str(), output_tensors_[i]);
    }
    binding.SynchronizeInputs();
    // The DML onnxruntime execution provider is documented as not supporting
    // multi-threaded calls to Run on the same inference session. We found the
    // same to be true for the ROCm execution provider (at least for CNNs).
    // TODO: This may be a onnxruntime/ROCm bug, check onnxruntime 1.16 release.
    if (network_->provider_ == OnnxProvider::DML ||
        network_->provider_ == OnnxProvider::ROCM ||
        network_->provider_ == OnnxProvider::TRT) {
      network_->lock_.lock();
    }
    network_->session_[step - 1].Run({}, binding);
    binding.SynchronizeOutputs();
    if (network_->provider_ == OnnxProvider::DML ||
        network_->provider_ == OnnxProvider::ROCM ||
        network_->provider_ == OnnxProvider::TRT) {
      network_->lock_.unlock();
    }
    i += batch;
  }
  if (network_->wdl_head_ != -1) {
    const DataType* data = static_cast<DataType*>(
        inputs_outputs_->output_tensors_data_[network_->wdl_head_]);
    for (size_t i = 0; i < (size_t)GetBatchSize(); i++) {
      float w = AsFloat(data[i * 3 + 0]);
      float d = AsFloat(data[i * 3 + 1]);
      float l = AsFloat(data[i * 3 + 2]);
      if (network_->cpu_wdl_) {
        // Value softmax done cpu side.
        float m = std::max({w, d, l});
        w = std::exp(w - m);
        d = std::exp(d - m);
        l = std::exp(l - m);
        float sum = w + d + l;
        w /= sum;
        l /= sum;
        d = 1.0f - w - l;
      }
      inputs_outputs_->wdl_output_data_[3 * i + 0] = w;
      inputs_outputs_->wdl_output_data_[3 * i + 1] = d;
      inputs_outputs_->wdl_output_data_[3 * i + 2] = l;
    }
  }
}

Ort::SessionOptions OnnxNetwork::GetOptions(int threads, int batch_size,
                                            uint64_t hash) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (batch_size > 0) {
    // Override the default (variable) batch size.
    Ort::ThrowOnError(
        OrtGetApiBase()
            ->GetApi(ORT_API_VERSION)
            ->AddFreeDimensionOverrideByName(options, "batch", batch_size));
  }

  switch (provider_) {
    case OnnxProvider::DML:
      options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      options.DisableMemPattern();
#ifdef USE_DML
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_DML(options, gpu_));
#else
      throw Exception("ONNX backend internal error.");
#endif
      break;
    case OnnxProvider::TRT: {
      options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

      std::string cache_dir = CommandLine::BinaryDirectory() + "/trt_cache";
      std::map<std::string, std::string> trt_options;
      trt_options["device_id"] = std::to_string(gpu_);
      trt_options["trt_fp16_enable"] = fp16_ ? "1" : "0";
      trt_options["trt_int8_enable"] = "0";
      trt_options["trt_max_partition_iterations"] = "1000";
      trt_options["trt_min_subgraph_size"] = "1";
      trt_options["trt_engine_cache_enable"] = "1";
      // We need the batch size as well as the hash, as it is set after loading.
      std::ostringstream oss;
      oss << std::hex << hash;
      trt_options["trt_engine_cache_prefix"] =
          "Lc0_ONNX_TRT_ORT_" + Ort::GetVersionString() + "_batch_" +
          std::to_string(batch_size) + "_" + oss.str() + "_";
      trt_options["trt_engine_cache_path"] = cache_dir;
      trt_options["trt_timing_cache_enable"] = "1";
      trt_options["trt_timing_cache_path"] = cache_dir;
      trt_options["trt_layer_norm_fp32_fallback"] = "1";
      trt_options["trt_force_sequential_engine_build"] = "1";
      // Looks like we need I/O binding to enable this.
      // trt_options["trt_cuda_graph_enable"] = "1";
      if (batch_size < 0) {
        trt_options["trt_profile_min_shapes"] =
            inputs_[0] + ":" + std::to_string(min_batch_size_) + "x112x8x8";
        trt_options["trt_profile_max_shapes"] =
            inputs_[0] + ":" + std::to_string(max_batch_size_) + "x112x8x8";
        trt_options["trt_profile_opt_shapes"] =
            inputs_[0] + ":" + std::to_string(max_batch_size_ / 4) + "x112x8x8";
      } else {
        trt_options["trt_profile_min_shapes"] =
            inputs_[0] + ":" + std::to_string(batch_size_) + "x112x8x8";
        trt_options["trt_profile_max_shapes"] =
            inputs_[0] + ":" + std::to_string(batch_size_ * steps_) +
            "x112x8x8";
        trt_options["trt_profile_opt_shapes"] =
            inputs_[0] + ":" + std::to_string(batch_size_ * steps_) +
            "x112x8x8";
      }
      std::vector<const char*> keys;
      std::vector<const char*> values;
      for (const auto& [key, value] : trt_options) {
        keys.push_back(key.c_str());
        values.push_back(value.c_str());
      }

      const auto& api = Ort::GetApi();
      OrtTensorRTProviderOptionsV2* trt_options_v2;
      Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options_v2));
      Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
          trt_options_v2, keys.data(), values.data(), keys.size()));
      options.AppendExecutionProvider_TensorRT_V2(*trt_options_v2);
      api.ReleaseTensorRTProviderOptions(trt_options_v2);
      break;
    }
    case OnnxProvider::ROCM: {
      OrtROCMProviderOptions rocm_options;
      rocm_options.device_id = gpu_;
      options.AppendExecutionProvider_ROCM(rocm_options);
      break;
    }
    case OnnxProvider::CUDA: {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = gpu_;
      options.AppendExecutionProvider_CUDA(cuda_options);
      break;
    }
    case OnnxProvider::CPU:
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

OnnxNetwork::OnnxNetwork(const WeightsFile& file, const OptionsDict& opts,
                         OnnxProvider provider, bool cpu_wdl)
    : onnx_env_(ORT_LOGGING_LEVEL_WARNING, "lc0"),
      capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()},
      fp16_(file.onnx_model().data_type() == pblczero::OnnxModel::FLOAT16),
      bf16_(file.onnx_model().data_type() == pblczero::OnnxModel::BFLOAT16),
      cpu_wdl_(cpu_wdl),
      provider_(provider) {
  onnx_env_.DisableTelemetryEvents();
  batch_size_ =
      opts.GetOrDefault<int>("batch", provider == OnnxProvider::DML ? 16 : -1);
  steps_ =
      opts.GetOrDefault<int>("steps", provider == OnnxProvider::DML ? 4 : 1);
  min_batch_size_ = opts.GetOrDefault<int>(
      "min_batch", provider == OnnxProvider::TRT ? 4 : 1);
  gpu_ = opts.GetOrDefault<int>("gpu", 0);
  int threads =
      opts.GetOrDefault<int>("threads", provider == OnnxProvider::CPU ? 1 : 0);

  // Sanity checks.
  if (batch_size_ <= 0) {
    batch_size_ = -1;  // Variable batch size.
    steps_ = 1;
  }
  if (batch_size_ * steps_ > max_batch_size_) {
    batch_size_ = max_batch_size_ / steps_;
  }

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
  uint64_t hash = 0;
  if (provider == OnnxProvider::TRT) {
    hash = std::hash<std::string_view>()(md.model());
  }

  for (int step = 1; step <= steps_; step++)
    session_.emplace_back(onnx_env_, file.onnx_model().model().data(),
                          file.onnx_model().model().size(),
                          GetOptions(threads, batch_size_ * step, hash));
}

template <OnnxProvider kProvider>
std::unique_ptr<Network> MakeOnnxNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& opts) {
  if (!w) throw Exception("The ONNX backend requires a network file.");

  if (w->has_onnx_model()) {
    return std::make_unique<OnnxNetwork>(*w, opts, kProvider, false);
  } else {
    WeightsToOnnxConverterOptions converter_options;
    converter_options.opset = opts.GetOrDefault<int>("opset", 17);
    converter_options.ir = opts.GetOrDefault<int>("ir", -1);
    converter_options.alt_mish = opts.GetOrDefault<bool>(
        "alt_mish", kProvider == OnnxProvider::CPU ? true : false);
    converter_options.alt_layernorm = opts.GetOrDefault<bool>(
        "alt_layernorm", kProvider == OnnxProvider::DML ? true : false);
    converter_options.no_shape = opts.GetOrDefault<bool>("no_shape", false);
    converter_options.policy_head =
        opts.GetOrDefault<std::string>("policy_head", "vanilla");
    converter_options.value_head =
        opts.GetOrDefault<std::string>("value_head", "winner");
    converter_options.no_wdl_softmax = true;

    std::string datatype;
    if (opts.Exists<std::string>("datatype")) {
      datatype = opts.Get<std::string>("datatype");
    } else {
      bool fp16 = opts.GetOrDefault<bool>(
          "fp16", kProvider == OnnxProvider::CPU ? false : true);
      datatype = fp16 ? "f16" : "f32";
    }
    converter_options.data_type =
        WeightsToOnnxConverterOptions::StringToDataType(datatype);

    auto converted = ConvertWeightsToOnnx(*w, converter_options);
    return std::make_unique<OnnxNetwork>(converted, opts, kProvider, true);
  }
}

#ifdef USE_ROCM
REGISTER_NETWORK("onnx-rocm", MakeOnnxNetwork<OnnxProvider::ROCM>, 64)
#endif
#ifdef USE_DML
REGISTER_NETWORK("onnx-dml", MakeOnnxNetwork<OnnxProvider::DML>, 63)
#endif
REGISTER_NETWORK("onnx-trt", MakeOnnxNetwork<OnnxProvider::TRT>, 60)
REGISTER_NETWORK("onnx-cuda", MakeOnnxNetwork<OnnxProvider::CUDA>, 61)
REGISTER_NETWORK("onnx-cpu", MakeOnnxNetwork<OnnxProvider::CPU>, 62)

}  // namespace
}  // namespace lczero
