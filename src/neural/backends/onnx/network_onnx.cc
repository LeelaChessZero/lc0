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
#include <chrono>
#include <filesystem>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "onnx_config.h"

#if __has_include("dml_provider_factory.h")
#include "dml_provider_factory.h"
#define USE_DML
#endif

#include "cpu_provider_factory.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "onnx_cuda.h"
#include "onnxruntime_cxx_api.h"
#include "utils/bf16_utils.h"
#include "utils/bititer.h"
#include "utils/commandline.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/logging.h"

namespace lczero::onnx {
namespace {

template <typename Provider>
class OnnxNetwork;

static constexpr int kNumOutputPolicy = 1858;

void AsDataType(float x, float* y) { *y = x; }
void AsDataType(float x, Ort::Float16_t* y) {
  uint16_t tmp = FP32toFP16(x);
  std::memcpy(reinterpret_cast<uint16_t*>(y), &tmp, sizeof(uint16_t));
}
void AsDataType(float x, Ort::BFloat16_t* y) {
  uint16_t tmp = FP32toBF16(x);
  std::memcpy(reinterpret_cast<uint16_t*>(y), &tmp, sizeof(uint16_t));
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

template <typename NetworkInfoType>
struct OnnxNetworkBase;
template <typename NetworkInfoType>
struct OnnxInputsOutputsBase {
  void Clear() {}
};

template <typename NetworkInfoType>
struct OnnxComputationBase : public NetworkComputation {
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;

  std::tuple<int, int> GetBatchStep(int start, int batch_size, int n,
                                    int max_steps) {
    int step = std::min(max_steps, (n - start + batch_size - 1) / batch_size);
    int batch = batch_size * step;
    return {batch, step};
  }

  std::unique_lock<std::mutex> LockComputeSession(std::mutex&) { return {}; }

  static constexpr bool UseCPUExpandPlanes() { return true; }

  void UploadInputs(Ort::IoBinding& binding, OnnxNetworkBase<NetworkInfo>&,
                    OnnxInputsOutputsBase<NetworkInfo>&, int, int) {
    binding.SynchronizeInputs();
  }

  Ort::RunOptions GetRunOptions() { return {}; }

  void DownloadOutputs(Ort::IoBinding& binding, OnnxNetworkBase<NetworkInfo>&,
                       OnnxInputsOutputsBase<NetworkInfo>&, int, int) {
    binding.SynchronizeOutputs();
  }

  void WaitForWDL(OnnxInputsOutputsBase<NetworkInfo>&) {}
  void WaitForOutputs(OnnxInputsOutputsBase<NetworkInfo>&) {}
};

template <typename NetworkInfoType>
struct OnnxNetworkBase : public Network {
  struct SessionParams {
    int threads_;
    int batch_size_;
    int steps_;
    int min_batch_size_;
    int max_batch_size_;
    int opt_batch_size_;

    void ValidateOptions() {
      // Sanity checks.
      if (batch_size_ <= 0) {
        batch_size_ = -1;  // Variable batch size.
        steps_ = 1;
      }
      if (batch_size_ * steps_ > max_batch_size_) {
        batch_size_ = max_batch_size_ / steps_;
      }
    }

    SessionParams(const WeightsFile&, const OptionsDict& opts)
        : threads_(opts.GetOrDefault<int>("threads", 0)),
          batch_size_(opts.GetOrDefault<int>("batch", -1)),
          steps_(opts.GetOrDefault<int>("steps", 1)),
          min_batch_size_(opts.GetOrDefault<int>("min_batch", 1)),
          max_batch_size_(opts.GetOrDefault<int>("max_batch", 1024)),
          opt_batch_size_(opts.GetOrDefault<int>(
              "opt_batch", batch_size_ < 0 ? 256 : batch_size_ * steps_)) {
      ValidateOptions();
    }
  };

  Ort::MemoryInfo GetMemoryInfo() const {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  }

  bool IsCpu() const override { return false; }

  bool IsSessionGood(const SessionParams&, size_t, int) const { return true; }

  void FixedSizeBatchOptions(Ort::SessionOptions& options,
                             const SessionParams& params) {
    if (params.batch_size_ < 0) return;
    Ort::ThrowOnError(OrtGetApiBase()
                          ->GetApi(ORT_API_VERSION)
                          ->AddFreeDimensionOverrideByName(
                              options, "batch", params.max_batch_size_));
  }
};

template <typename NetworkInfoType>
struct OnnxCPUInputsOutputs : public OnnxInputsOutputsBase<NetworkInfoType> {
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;

  template <typename Provider>
  OnnxCPUInputsOutputs(const OnnxNetwork<Provider>& network) {
    const auto max_batch_size = network.max_batch_size_;
    input_tensor_data_device_ =
        std::make_unique<DataType[]>(max_batch_size * kInputPlanes * 8 * 8);
    for (int i = 0; i < NetworkInfo::output_size_; i++) {
      const auto step = NetworkInfo::output_tensors_step_[i];
      output_tensors_data_device_[i] =
          std::make_unique<DataType[]>(max_batch_size * step);
    }
  }

  DataType* GetInputData() { return input_tensor_data_device_.get(); }

  const DataType* GetOutputData(int index) const {
    assert(index >= 0);
    assert(index < NetworkInfo::output_size_);
    return output_tensors_data_device_[index].get();
  }

  void Clear() { raw_input_.clear(); }

  std::vector<InputPlanes> raw_input_;
  std::unique_ptr<DataType[]> input_tensor_data_device_;
  std::unique_ptr<DataType[]>
      output_tensors_data_device_[NetworkInfo::output_size_];
};

template <typename NetworkInfoType>
struct OnnxCPUNetwork : public OnnxNetworkBase<NetworkInfoType> {
  using Base = OnnxNetworkBase<NetworkInfoType>;
  using SessionParams = Base::SessionParams;

  OnnxCPUNetwork(const OptionsDict&) {}

  bool IsCpu() const override { return true; }

  Ort::SessionOptions GetOptions(Ort::SessionOptions options,
                                 const SessionParams& params,
                                 const std::vector<std::string>&, int) {
    Base::FixedSizeBatchOptions(options, params);
    auto status = OrtSessionOptionsAppendExecutionProvider_CPU(options, 0);
    if (status) {
      std::string error_message = Ort::GetApi().GetErrorMessage(status);
      OrtErrorCode error_code = Ort::GetApi().GetErrorCode(status);
      Ort::GetApi().ReleaseStatus(status);
      throw Exception("ONNX CPU error " + std::to_string(error_code) + ": " +
                      error_message);
    }
    return options;
  }
};

template <typename NetworkInfoType>
struct OnnxCPUComputation : public OnnxComputationBase<NetworkInfoType> {
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;

  OnnxCPUComputation(OnnxNetworkBase<NetworkInfo>*) {}

  int GetBatchSizeImplement(
      const OnnxCPUInputsOutputs<NetworkInfo>& inputs_outputs) const {
    return inputs_outputs.raw_input_.size();
  }
  void AddInputImplement(InputPlanes&& input,
                         OnnxCPUInputsOutputs<NetworkInfo>& inputs_outputs) {
    inputs_outputs.raw_input_.emplace_back(input);
  }
};

template <typename WrappedIO>
struct OnnxWDLIOWrapper : public WrappedIO {
  template <typename Provider>
  OnnxWDLIOWrapper(const OnnxNetwork<Provider>& network) : WrappedIO(network) {
    const auto max_batch_size = network.max_batch_size_;
    wdl_output_data_ = std::make_unique<float[]>(max_batch_size * 2);
  }
  std::unique_ptr<float[]> wdl_output_data_;
};

template <typename WrappedComputation>
struct OnnxExclusiveComputationWrapper : public WrappedComputation {
  using Base = WrappedComputation;
  using NetworkInfo = typename Base::NetworkInfo;

  OnnxExclusiveComputationWrapper(OnnxNetworkBase<NetworkInfo>* network)
      : Base(network) {}

  std::unique_lock<std::mutex> LockComputeSession(std::mutex& mutex) {
    return std::unique_lock<std::mutex>{mutex};
  }
};

template <typename T>
struct OnnxCPUProvider {
  using NetworkInfo = T;
  using DataType = NetworkInfo::DataType;
  static constexpr bool cpu_wdl_ = NetworkInfo::cpu_wdl_;

  using NetworkBase = OnnxCPUNetwork<NetworkInfo>;
  using ComputationBase = OnnxCPUComputation<NetworkInfo>;
  using InputsOutputsBase =
      std::conditional_t<cpu_wdl_,
                         OnnxWDLIOWrapper<OnnxCPUInputsOutputs<NetworkInfo>>,
                         OnnxCPUInputsOutputs<NetworkInfo>>;
};

#if USE_ONNX_CUDART
template <typename DataType>
struct OnnxToCUDAType {};

template <>
struct OnnxToCUDAType<float> {
  using Type = float;
};
template <>
struct OnnxToCUDAType<Ort::Float16_t> {
  using Type = __half;
};
template <>
struct OnnxToCUDAType<Ort::BFloat16_t> {
  using Type = __nv_bfloat16;
};

template <typename NetworkInfoType>
struct OnnxCUDAInputsOutputs : public OnnxInputsOutputsBase<NetworkInfoType> {
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;
  using CUDADataType = typename OnnxToCUDAType<DataType>::Type;

  template <typename Provider>
  OnnxCUDAInputsOutputs(const OnnxNetwork<Provider>& network) {
    const auto max_batch_size = network.max_batch_size_;
    const int outputs_size = NetworkInfo::output_size_;
    const size_t data_size = sizeof(DataType);

    ReportCUDAErrors(
        cudaEventCreate(&inputs_processed_event_, cudaEventDisableTiming));
    ReportCUDAErrors(
        cudaEventCreate(&inputs_uploaded_event_, cudaEventDisableTiming));
    ReportCUDAErrors(
        cudaEventCreate(&evaluation_done_event_, cudaEventDisableTiming));
    ReportCUDAErrors(
        cudaEventCreate(&wdl_download_event_, cudaEventDisableTiming));
    ReportCUDAErrors(
        cudaEventCreate(&outputs_download_event_, cudaEventDisableTiming));
    ReportCUDAErrors(cudaHostAlloc(
        &input_mask_data_,
        max_batch_size * kInputPlanes * sizeof(*input_mask_data_), 0));
    ReportCUDAErrors(cudaHostAlloc(
        &input_value_data_,
        max_batch_size * kInputPlanes * sizeof(*input_value_data_), 0));

    for (int i = 0; i < outputs_size; i++) {
      ReportCUDAErrors(cudaHostAlloc(
          &output_tensors_data_[i],
          max_batch_size * NetworkInfo::output_tensors_step_[i] * data_size,
          0));
    }

    ReportCUDAErrors(
        cudaMalloc(&input_tensor_upload_device_,
                   max_batch_size * kInputPlanes * sizeof(InputPlane)));
    ReportCUDAErrors(
        cudaMalloc(&input_tensor_data_device_,
                   max_batch_size * kInputPlanes * 8 * 8 * data_size));
    for (int i = 0; i < outputs_size; i++) {
      ReportCUDAErrors(cudaMalloc(
          &output_tensors_data_device_[i],
          max_batch_size * NetworkInfo::output_tensors_step_[i] * data_size));
    }
  }
  ~OnnxCUDAInputsOutputs() {
    ReportCUDAErrors(cudaEventDestroy(inputs_uploaded_event_));
    ReportCUDAErrors(cudaEventDestroy(inputs_processed_event_));
    ReportCUDAErrors(cudaEventDestroy(evaluation_done_event_));
    ReportCUDAErrors(cudaEventDestroy(wdl_download_event_));
    ReportCUDAErrors(cudaEventDestroy(outputs_download_event_));
    ReportCUDAErrors(cudaFree(input_tensor_upload_device_));
    ReportCUDAErrors(cudaFree(input_tensor_data_device_));
    for (void* ptr : output_tensors_data_device_) {
      ReportCUDAErrors(cudaFree(ptr));
    }
    ReportCUDAErrors(cudaFreeHost(input_mask_data_));
    ReportCUDAErrors(cudaFreeHost(input_value_data_));
    for (void* ptr : output_tensors_data_) {
      ReportCUDAErrors(cudaFreeHost(ptr));
    }
  }

  const DataType* GetOutputData(int index) const {
    assert(index >= 0);
    assert(index < NetworkInfo::output_size_);
    return output_tensors_data_[index];
  }

  uint64_t* input_mask_data_ = nullptr;
  DataType* input_value_data_ = nullptr;
  void* input_tensor_upload_device_ = nullptr;
  DataType* input_tensor_data_device_ = nullptr;
  DataType* output_tensors_data_[NetworkInfo::output_size_] = {nullptr};
  DataType* output_tensors_data_device_[NetworkInfo::output_size_] = {nullptr};
  cudaEvent_t inputs_uploaded_event_ = nullptr;
  cudaEvent_t inputs_processed_event_ = nullptr;
  cudaEvent_t evaluation_done_event_ = nullptr;
  cudaEvent_t wdl_download_event_ = nullptr;
  cudaEvent_t outputs_download_event_ = nullptr;
};

#endif

template <typename NetworkInfoType>
struct OnnxCUDANetwork : public OnnxNetworkBase<NetworkInfoType> {
  using Base = OnnxNetworkBase<NetworkInfoType>;
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;
  struct SessionParams : public Base::SessionParams {
    using Base = Base::SessionParams;
    SessionParams(const WeightsFile& file, const OptionsDict& opts)
        : Base(file, opts) {
#ifdef USE_ONNX_CUDART
      int gpu = opts.GetOrDefault<int>("gpu", 0);
      cudaDeviceProp deviceProp = {};
      if (!cudaGetDeviceProperties(&deviceProp, gpu)) {
        CERR << "GPU: " << deviceProp.name;
        CERR << "GPU memory: " << deviceProp.totalGlobalMem / std::pow(2.0f, 30)
             << " Gb";
        CERR << "GPU SM count: " << deviceProp.multiProcessorCount;

        const int divisor = deviceProp.multiProcessorCount >= 128 ? 2 : 1;

        Base::opt_batch_size_ = opts.GetOrDefault<int>(
            "opt_batch",
            std::max(Base::batch_size_ * Base::steps_,
                     (deviceProp.multiProcessorCount & ~3) / divisor));

        Base::ValidateOptions();

        int clockRate = 0;
        ReportCUDAErrors(
            cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, gpu));
        CERR << "GPU clock frequency: " << clockRate / 1e3f << " MHz";
      }
#if CUDART_VERSION >= 12080
      int runtime_version;
      ReportCUDAErrors(cudaRuntimeGetVersion(&runtime_version));
      if (runtime_version >= 12080) {
        int attr;
        ReportCUDAErrors(
            cudaDeviceGetAttribute(&attr, cudaDevAttrGpuPciDeviceId, gpu));
        uint32_t pci_device = attr;
        CERR << "GPU device ID: " << std::hex << (pci_device & 0xffff) << ":"
             << (pci_device >> 16);
        ReportCUDAErrors(
            cudaDeviceGetAttribute(&attr, cudaDevAttrGpuPciSubsystemId, gpu));
        uint32_t pci_subsystem = attr;
        CERR << "GPU subsystem ID: " << std::hex << (pci_subsystem & 0xffff)
             << ":" << (pci_subsystem >> 16) << std::dec;
      }
#endif
#endif
    }
  };

  OnnxCUDANetwork(const OptionsDict& opts)
      : gpu_(opts.GetOrDefault<int>("gpu", 0)) {
#ifdef USE_ONNX_CUDART
    ReportCUDAErrors(cudaSetDevice(gpu_));
    ReportCUDAErrors(cudaStreamCreate(&compute_stream_));
    ReportCUDAErrors(cudaStreamCreate(&upload_stream_));
    ReportCUDAErrors(cudaStreamCreate(&download_stream_));
#else
    CERR << "WARNING: CUDA support missing. Enable plain_cuda build option "
            "for CUDA optimisations.";
#endif
  }

  ~OnnxCUDANetwork() {
#ifdef USE_ONNX_CUDART
    ReportCUDAErrors(cudaStreamDestroy(compute_stream_));
    ReportCUDAErrors(cudaStreamDestroy(upload_stream_));
    ReportCUDAErrors(cudaStreamDestroy(download_stream_));
#endif
  }

  Ort::SessionOptions GetOptions(Ort::SessionOptions options,
                                 const SessionParams&,
                                 const std::vector<std::string>&, int) {
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = gpu_;
#if USE_ONNX_CUDART
    cuda_options.has_user_compute_stream = true;
    cuda_options.user_compute_stream = compute_stream_;
#endif
    options.AppendExecutionProvider_CUDA(cuda_options);
    return options;
  }

#ifdef USE_ONNX_CUDART
  Ort::MemoryInfo GetMemoryInfo() const {
    return Ort::MemoryInfo{"Cuda", OrtDeviceAllocator, gpu_, OrtMemTypeDefault};
  }
#endif

  int gpu_;

#if USE_ONNX_CUDART
  cudaStream_t compute_stream_ = nullptr;
  cudaStream_t upload_stream_ = nullptr;
  cudaStream_t download_stream_ = nullptr;
#endif
};

#if USE_ONNX_CUDART
template <typename NetworkInfoType>
struct OnnxCUDAComputation : public OnnxComputationBase<NetworkInfoType> {
  using Base = OnnxComputationBase<NetworkInfoType>;
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;
  using CUDADataType = typename OnnxToCUDAType<DataType>::Type;

  OnnxCUDAComputation(OnnxNetworkBase<NetworkInfo>* network) {
    OnnxCUDANetwork<NetworkInfo>* cuda_network =
        static_cast<OnnxCUDANetwork<NetworkInfo>*>(network);
    ReportCUDAErrors(cudaSetDevice(cuda_network->gpu_));
  }

  int GetBatchSizeImplement(const OnnxCUDAInputsOutputs<NetworkInfo>&) const {
    return input_size_;
  }

  void AddInputImplement(InputPlanes&& input,
                         OnnxCUDAInputsOutputs<NetworkInfo>& inputs_outputs) {
    assert(input.size() == kInputPlanes);
    uint64_t* masks =
        &inputs_outputs.input_mask_data_[input_size_ * kInputPlanes];
    DataType* values =
        &inputs_outputs.input_value_data_[input_size_ * kInputPlanes];
    for (size_t i = 0; i < kInputPlanes; i++) {
      masks[i] = input[i].mask;
      DataType value;
      AsDataType(input[i].value, &value);
      values[i] = value;
    }
    input_size_++;
    return;
  }

  std::tuple<int, int> GetBatchStep(int start, int batch_size, int n,
                                    int max_steps) {
    auto [batch, step] = Base::GetBatchStep(start, batch_size, n, max_steps);
    if (batch_size < 0) {
      batch = std::min(n - start, batch);
    }
    return {batch, step};
  }

  static constexpr bool UseCPUExpandPlanes() { return false; }

  void UploadInputs(Ort::IoBinding&, OnnxCUDANetwork<NetworkInfo>& network,
                    OnnxCUDAInputsOutputs<NetworkInfo>& inputs_outputs,
                    int start, int batch) {
    const auto* src_masks =
        &inputs_outputs.input_mask_data_[start * kInputPlanes];
    char* dst_masks =
        static_cast<char*>(inputs_outputs.input_tensor_upload_device_);
    dst_masks += start * kInputPlanes * (sizeof(uint64_t) + sizeof(DataType));
    ReportCUDAErrors(cudaMemcpyAsync(
        dst_masks, src_masks, batch * kInputPlanes * sizeof(uint64_t),
        cudaMemcpyHostToDevice, network.upload_stream_));
    const auto* src_values =
        &inputs_outputs.input_value_data_[start * kInputPlanes];
    char* dst_values = dst_masks + batch * kInputPlanes * sizeof(uint64_t);
    ReportCUDAErrors(cudaMemcpyAsync(
        dst_values, src_values, batch * kInputPlanes * sizeof(DataType),
        cudaMemcpyHostToDevice, network.upload_stream_));
    ReportCUDAErrors(cudaEventRecord(inputs_outputs.inputs_uploaded_event_,
                                     network.upload_stream_));
    ReportCUDAErrors(cudaStreamWaitEvent(
        network.compute_stream_, inputs_outputs.inputs_uploaded_event_));
    CUDADataType* dst = reinterpret_cast<CUDADataType*>(
        inputs_outputs.input_tensor_data_device_);
    dst += start * kInputPlanes * 8 * 8;
    expandPlanes(dst, dst_masks, batch * kInputPlanes, network.compute_stream_);

    ReportCUDAErrors(cudaEventRecord(inputs_outputs.inputs_processed_event_,
                                     network.upload_stream_));
  }

  Ort::RunOptions GetRunOptions() {
    Ort::RunOptions run_options = {};
    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
    return run_options;
  }

  void DownloadOutputs(Ort::IoBinding&, OnnxCUDANetwork<NetworkInfo>& network,
                       OnnxCUDAInputsOutputs<NetworkInfo>& inputs_outputs,
                       int start, int batch) {
    for (size_t i = 0; i < NetworkInfo::output_size_; i++) {
      size_t step = NetworkInfo::output_tensors_step_[i];
      ReportCUDAErrors(cudaEventRecord(inputs_outputs.evaluation_done_event_,
                                       network.compute_stream_));
      ReportCUDAErrors(cudaStreamWaitEvent(
          network.download_stream_, inputs_outputs.evaluation_done_event_));
      size_t offset = start * step;
      ReportCUDAErrors(cudaMemcpyAsync(
          inputs_outputs.output_tensors_data_[i] + offset,
          inputs_outputs.output_tensors_data_device_[i] + offset,
          batch * step * sizeof(DataType), cudaMemcpyDeviceToHost,
          network.download_stream_));
      if (NetworkInfo::cpu_wdl_ && i == (size_t)NetworkInfo::wdl_head_) {
        ReportCUDAErrors(cudaEventRecord(inputs_outputs.wdl_download_event_,
                                         network.download_stream_));
      }
    }
    ReportCUDAErrors(cudaEventRecord(inputs_outputs.outputs_download_event_,
                                     network.download_stream_));
  }

  void WaitForWDL(OnnxCUDAInputsOutputs<NetworkInfo>& inputs_outputs) {
    ReportCUDAErrors(cudaEventSynchronize(inputs_outputs.wdl_download_event_));
  }

  void WaitForOutputs(OnnxCUDAInputsOutputs<NetworkInfo>& inputs_outputs) {
    ReportCUDAErrors(
        cudaEventSynchronize(inputs_outputs.outputs_download_event_));
  }

  size_t input_size_ = 0;
};
#endif

template <typename T>
struct OnnxCUDAProvider {
  using NetworkInfo = T;
  using DataType = NetworkInfo::DataType;
  static constexpr bool cpu_wdl_ = NetworkInfo::cpu_wdl_;

  using NetworkBase = OnnxCUDANetwork<T>;
#if USE_ONNX_CUDART
  using ComputationBase =
      OnnxExclusiveComputationWrapper<OnnxCUDAComputation<T>>;
  using InputsOutputsBase =
      std::conditional_t<cpu_wdl_, OnnxWDLIOWrapper<OnnxCUDAInputsOutputs<T>>,
                         OnnxCUDAInputsOutputs<T>>;
#else
  using ComputationBase = OnnxCPUComputation<T>;
  using InputsOutputsBase =
      std::conditional_t<cpu_wdl_, OnnxWDLIOWrapper<OnnxCPUInputsOutputs<T>>,
                         OnnxCPUInputsOutputs<T>>;
#endif
};

template <typename NetworkInfoType>
struct OnnxTRTNetwork : public OnnxCUDANetwork<NetworkInfoType> {
  using Base = OnnxCUDANetwork<NetworkInfoType>;
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;

  using Base::Base;

  struct SessionParams : public Base::SessionParams {
    using Base = Base::SessionParams;
    uint64_t hash_;
    std::filesystem::file_time_type start_time_ =
        std::chrono::file_clock::now();

    SessionParams(const WeightsFile& file, const OptionsDict& opts)
        : Base(file, opts), hash_(GetModelHash(file.onnx_model())) {
      Base::min_batch_size_ = opts.GetOrDefault<int>("min_batch", 4);
      Base::ValidateOptions();
    }
  };

  static uint64_t GetModelHash(const pblczero::OnnxModel& md) {
    return std::hash<std::string_view>()(md.model());
  }

  std::string TRTCachePrefix(const SessionParams& params) {
    std::ostringstream oss;
    const auto hash = params.hash_;
    const int min_batch_size = params.min_batch_size_;
    const int max_batch_size = params.max_batch_size_;
    const int opt_batch_size = params.opt_batch_size_;
    const int batch_size = params.batch_size_;
    oss << std::hex << hash;
    // We need the batch size as well as the hash, as it is set after
    // loading.
    return "Lc0_ONNX_TRT_ORT_" + Ort::GetVersionString() + "_batch_" +
           std::to_string(min_batch_size) + "-" +
           (batch_size < 0 ? std::to_string(opt_batch_size)
                           : std::to_string(max_batch_size)) +
           "-" + std::to_string(max_batch_size) + "_" + oss.str() + "_";
  }

  bool IsSessionGood(const SessionParams& params, size_t onnx_model_size,
                     int attempt) {
    const auto start = params.start_time_;
    std::filesystem::path cache_dir =
        CommandLine::BinaryDirectory() + "/trt_cache";
    const auto prefix = TRTCachePrefix(params);
    std::filesystem::file_time_type last_edit{};
    std::filesystem::directory_entry latest_matching{};
    std::filesystem::path timing_cache{};
    bool found = false;
    for (const auto& dir_entry :
         std::filesystem::directory_iterator(cache_dir)) {
      const auto& filename = dir_entry.path().filename().string();
      const auto& extension = dir_entry.path().extension();
      if (extension == ".timing") {
        timing_cache = dir_entry.path();
        continue;
      }
      if (dir_entry.is_regular_file() && filename.starts_with(prefix) &&
          extension == ".engine" &&
          (!found || last_edit < dir_entry.last_write_time())) {
        latest_matching = dir_entry;
        last_edit = dir_entry.last_write_time();
        found = true;
      }
    }
    if (!found || !latest_matching.exists()) {
      throw Exception("TRT engine cache file not found: " +
                      (cache_dir / prefix).string() + "*.engine");
    }
    if (latest_matching.last_write_time() < start) {
      // Reusing an engine. We don't know which one was used if there is more
      // than one.
      return true;
    }
    if (latest_matching.file_size() > onnx_model_size * 7 / 4) {
      CERR << "TRT engine is bad: " << latest_matching.path() << " size "
           << latest_matching.file_size() << " vs model size "
           << onnx_model_size;
      if (!std::filesystem::remove(latest_matching.path())) {
        throw Exception("Failed to remove slow TRT engine file: " +
                        latest_matching.path().string());
      }
      std::filesystem::path profile = latest_matching.path();
      profile.replace_extension(".profile");
      if (!std::filesystem::remove(profile)) {
        throw Exception("Failed to remove slow TRT profile file: " +
                        profile.string());
      }
      if (attempt < 3) {
        return false;
      }
      if (!std::filesystem::remove(timing_cache)) {
        throw Exception("Failed to remove TRT timing cache file: " +
                        timing_cache.string());
      }
      return false;
    }
    return true;
  }

  Ort::SessionOptions GetOptions(Ort::SessionOptions options,
                                 const SessionParams& params,
                                 const std::vector<std::string>& inputs,
                                 int attempt) {
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    const int batch_size = params.batch_size_;
    const int min_batch_size = params.min_batch_size_;
    const int max_batch_size = params.max_batch_size_;
    const int opt_batch_size = params.opt_batch_size_;

    std::string cache_dir = CommandLine::BinaryDirectory() + "/trt_cache";
    std::map<std::string, std::string> trt_options;
    trt_options["device_id"] = std::to_string(Base::gpu_);
    trt_options["trt_fp16_enable"] = sizeof(DataType) == 2 ? "1" : "0";
    trt_options["trt_int8_enable"] = "0";
    trt_options["trt_max_partition_iterations"] = "1000";
    trt_options["trt_min_subgraph_size"] = "1";
    trt_options["trt_engine_cache_enable"] = "1";
    trt_options["trt_engine_cache_prefix"] = TRTCachePrefix(params);
    trt_options["trt_engine_cache_path"] = cache_dir;
    trt_options["trt_timing_cache_enable"] = "1";
    trt_options["trt_timing_cache_path"] = cache_dir;
    trt_options["trt_layer_norm_fp32_fallback"] = "1";
    trt_options["trt_force_sequential_engine_build"] = "1";
    trt_options["trt_context_memory_sharing_enable"] = "1";
    trt_options["trt_builder_optimization_level"] = attempt < 2 ? "4" : "5";
    // Looks like we need I/O binding to enable this.
#if USE_ONNX_CUDART
    trt_options["has_user_compute_stream"] = "1";
#endif
    if (batch_size < 0) {
      trt_options["trt_profile_min_shapes"] =
          inputs[0] + ":" + std::to_string(min_batch_size) + "x112x8x8";
      trt_options["trt_profile_max_shapes"] =
          inputs[0] + ":" + std::to_string(max_batch_size) + "x112x8x8";
      trt_options["trt_profile_opt_shapes"] =
          inputs[0] + ":" + std::to_string(opt_batch_size) + "x112x8x8";
    } else {
      trt_options["trt_profile_min_shapes"] =
          inputs[0] + ":" + std::to_string(min_batch_size) + "x112x8x8";
      trt_options["trt_profile_max_shapes"] =
          inputs[0] + ":" + std::to_string(max_batch_size) + "x112x8x8";
      trt_options["trt_profile_opt_shapes"] =
          inputs[0] + ":" + std::to_string(max_batch_size) + "x112x8x8";
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
#if USE_ONNX_CUDART
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(
        trt_options_v2, "user_compute_stream", Base::compute_stream_));
#endif
    options.AppendExecutionProvider_TensorRT_V2(*trt_options_v2);
    api.ReleaseTensorRTProviderOptions(trt_options_v2);
    return options;
  }
};

template <typename T>
struct OnnxTRTProvider {
  using NetworkInfo = T;
  using DataType = NetworkInfo::DataType;
  static constexpr bool cpu_wdl_ = NetworkInfo::cpu_wdl_;

  using NetworkBase = OnnxTRTNetwork<T>;
#if USE_ONNX_CUDART
  using ComputationBase =
      OnnxExclusiveComputationWrapper<OnnxCUDAComputation<T>>;
  using InputsOutputsBase =
      std::conditional_t<cpu_wdl_, OnnxWDLIOWrapper<OnnxCUDAInputsOutputs<T>>,
                         OnnxCUDAInputsOutputs<T>>;
#else
  using ComputationBase =
      OnnxExclusiveComputationWrapper<OnnxCPUComputation<T>>;
  using InputsOutputsBase =
      std::conditional_t<cpu_wdl_, OnnxWDLIOWrapper<OnnxCPUInputsOutputs<T>>,
                         OnnxCPUInputsOutputs<T>>;
#endif
};

template <typename NetworkInfoType>
struct OnnxDMLNetwork : public OnnxNetworkBase<NetworkInfoType> {
  using Base = OnnxNetworkBase<NetworkInfoType>;
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;

  struct SessionParams : public Base::SessionParams {
    using Base = Base::SessionParams;

    SessionParams(const WeightsFile& file, const OptionsDict& opts)
        : Base(file, opts) {
      Base::batch_size_ = opts.GetOrDefault<int>("batch", 16);
      Base::steps_ = opts.GetOrDefault<int>("steps", 4);
      Base::opt_batch_size_ = opts.GetOrDefault<int>(
          "opt_batch",
          Base::batch_size_ < 0 ? 256 : Base::batch_size_ * Base::steps_);
      Base::ValidateOptions();
    }
  };

  OnnxDMLNetwork(const OptionsDict& opts)
      : gpu_(opts.GetOrDefault<int>("gpu", 0)) {}

  Ort::SessionOptions GetOptions(Ort::SessionOptions options,
                                 const SessionParams& params,
                                 const std::vector<std::string>&, int) {
    Base::FixedSizeBatchOptions(options, params);

    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.DisableMemPattern();
#ifdef USE_DML
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_DML(options, gpu_));
#else
    throw Exception("ONNX backend internal error.");
#endif
    return options;
  }

  int gpu_ = 0;
};

template <typename T>
struct OnnxDMLProvider {
  using NetworkInfo = T;
  using DataType = NetworkInfo::DataType;
  static constexpr bool cpu_wdl_ = NetworkInfo::cpu_wdl_;

  using NetworkBase = OnnxDMLNetwork<T>;
  using ComputationBase =
      OnnxExclusiveComputationWrapper<OnnxCPUComputation<T>>;
  using InputsOutputsBase =
      std::conditional_t<cpu_wdl_, OnnxWDLIOWrapper<OnnxCPUInputsOutputs<T>>,
                         OnnxCPUInputsOutputs<T>>;
};

template <typename NetworkInfoType>
struct OnnxROCMNetwork : public OnnxNetworkBase<NetworkInfoType> {
  using Base = OnnxNetworkBase<NetworkInfoType>;
  using NetworkInfo = NetworkInfoType;
  using DataType = typename NetworkInfo::DataType;
  using SessionParams = Base::SessionParams;

  OnnxROCMNetwork(const OptionsDict& opts)
      : gpu_(opts.GetOrDefault<int>("gpu", 0)) {}

  Ort::SessionOptions GetOptions(Ort::SessionOptions options,
                                 const SessionParams&,
                                 const std::vector<std::string>&, int) {
    OrtROCMProviderOptions rocm_options = {};
    rocm_options.device_id = gpu_;
    options.AppendExecutionProvider_ROCM(rocm_options);
    return options;
  }

  int gpu_ = 0;
};

template <typename T>
struct OnnxROCMProvider {
  using NetworkInfo = T;
  using DataType = NetworkInfo::DataType;
  static constexpr bool cpu_wdl_ = NetworkInfo::cpu_wdl_;

  using NetworkBase = OnnxROCMNetwork<T>;
  using ComputationBase =
      OnnxExclusiveComputationWrapper<OnnxCPUComputation<T>>;
  using InputsOutputsBase =
      std::conditional_t<cpu_wdl_, OnnxWDLIOWrapper<OnnxCPUInputsOutputs<T>>,
                         OnnxCPUInputsOutputs<T>>;
};

enum class OnnxProvider { CPU, CUDA, TRT, DML, ROCM };

template <typename Provider>
struct InputsOutputs : public Provider::InputsOutputsBase {
  using Base = typename Provider::InputsOutputsBase;
  InputsOutputs(const OnnxNetwork<Provider>& network);
};

template <typename Provider>
class OnnxComputation final : public Provider::ComputationBase {
  using Base = typename Provider::ComputationBase;
  using NetworkInfo = typename Provider::NetworkInfo;
  using DataType = typename NetworkInfo::DataType;

 public:
  OnnxComputation(OnnxNetwork<Provider>* network);
  ~OnnxComputation();
  void AddInput(InputPlanes&& input) override;
  int GetBatchSize() const override;
  void ComputeBlocking() override;
  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  Ort::IoBinding PrepareInputs(int start, int batch_size, int step);

  OnnxNetwork<Provider>* network_;
  std::unique_ptr<InputsOutputs<Provider>> inputs_outputs_;
};

template <typename Provider>
class OnnxNetwork final : public Provider::NetworkBase {
  using Base = typename Provider::NetworkBase;
  using NetworkInfo = typename Provider::NetworkInfo;
  using DataType = typename NetworkInfo::DataType;
  using SessionParams = typename Base::SessionParams;

 public:
  OnnxNetwork(const WeightsFile& file, const OptionsDict& options);
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<OnnxComputation<Provider>>(this);
  }
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  int GetMiniBatchSize() const override {
    return batch_size_ == -1 ? opt_batch_size_
                             : std::max(batch_size_ * steps_, opt_batch_size_);
  }

  Ort::SessionOptions GetOptions(const SessionParams& params, int attempt);

  std::unique_ptr<InputsOutputs<Provider>> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs<Provider>>(*this);
    } else {
      std::unique_ptr<InputsOutputs<Provider>> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs<Provider>> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

  std::string TRTCachePrefix(int batch_size, uint64_t hash);
  bool IsTRTEngineGood(std::filesystem::file_time_type start, int batch_size,
                       uint64_t hash, size_t onnx_model_size, int attempt);

  Ort::Env onnx_env_;
  // Prepare sessions for this many multiples of the batch size;
  int steps_;
  std::vector<Ort::Session> session_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  NetworkCapabilities capabilities_;
  // The batch size to use, or -1 for variable.
  int batch_size_;
  // The lower limit for variable batch size.
  int min_batch_size_ = 1;
  int opt_batch_size_ = 256;
  int max_batch_size_ = 1024;
  Ort::MemoryInfo memory_info_{nullptr};
  // For conditional locking if running the DML/ROCM/TRT provider.
  std::mutex lock_;
  // For shared device addresses.

 private:
  std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs<Provider>>> free_inputs_outputs_;
};

template <typename Provider>
InputsOutputs<Provider>::InputsOutputs(const OnnxNetwork<Provider>& network)
    : Base(network) {}

template <typename Provider>
OnnxComputation<Provider>::OnnxComputation(OnnxNetwork<Provider>* network)
    : Base(network), network_(network) {
  inputs_outputs_ = network_->GetInputsOutputs();
  inputs_outputs_->Clear();
}

template <typename Provider>
OnnxComputation<Provider>::~OnnxComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}
template <typename Provider>
int OnnxComputation<Provider>::GetBatchSize() const {
  return Base::GetBatchSizeImplement(*inputs_outputs_);
}

template <typename NetworkInfoType>
void OnnxComputation<NetworkInfoType>::AddInput(InputPlanes&& input) {
  if (GetBatchSize() >= network_->max_batch_size_) {
    throw Exception("NN input exceeds max batch size of " +
                    std::to_string(network_->max_batch_size_) + ".");
  }
  Base::AddInputImplement(std::move(input), *inputs_outputs_);
}

template <typename NetworkInfo>
float OnnxComputation<NetworkInfo>::GetQVal(int sample) const {
  if constexpr (NetworkInfo::cpu_wdl_) {
    return inputs_outputs_->wdl_output_data_[sample * 2 + 0];
  } else if constexpr (NetworkInfo::wdl_head_ != -1) {
    auto step = NetworkInfo::output_tensors_step_[NetworkInfo::wdl_head_];
    DataType w = inputs_outputs_->GetOutputData(
        NetworkInfo::wdl_head_)[sample * step + 0];
    DataType l = inputs_outputs_->GetOutputData(
        NetworkInfo::wdl_head_)[sample * step + 2];
    return AsFloat(w) - AsFloat(l);
  }
  auto step = NetworkInfo::output_tensors_step_[NetworkInfo::value_head_];
  DataType value =
      inputs_outputs_->GetOutputData(NetworkInfo::value_head_)[sample * step];
  return AsFloat(value);
}

template <typename NetworkInfo>
float OnnxComputation<NetworkInfo>::GetDVal(int sample) const {
  if constexpr (NetworkInfo::wdl_head_ == -1) {
    return 0.0f;
  } else if constexpr (NetworkInfo::cpu_wdl_) {
    return inputs_outputs_->wdl_output_data_[sample * 2 + 1];
  }
  return AsFloat(
      inputs_outputs_->GetOutputData(NetworkInfo::wdl_head_)[sample * 3 + 1]);
}

template <typename NetworkInfo>
float OnnxComputation<NetworkInfo>::GetPVal(int sample, int move_id) const {
  const DataType* data =
      inputs_outputs_->GetOutputData(NetworkInfo::policy_head_);
  return AsFloat(data[sample * kNumOutputPolicy + move_id]);
}

template <typename NetworkInfoType>
float OnnxComputation<NetworkInfoType>::GetMVal(int sample) const {
  if (NetworkInfo::mlh_head_ == -1) return 0.0f;
  const DataType* data = inputs_outputs_->GetOutputData(NetworkInfo::mlh_head_);
  return AsFloat(data[sample]);
}

template <typename NetworkInfoType>
Ort::IoBinding OnnxComputation<NetworkInfoType>::PrepareInputs(int start,
                                                               int batch_size,
                                                               int step) {
  if constexpr (Base::UseCPUExpandPlanes()) {
    DataType* iter = inputs_outputs_->GetInputData();
    const auto& raw_input = inputs_outputs_->raw_input_;
    iter += start * kInputPlanes * 8 * 8;
    std::memset(iter, 0, batch_size * kInputPlanes * 8 * 8 * sizeof(DataType));
    int end = std::min(start + batch_size, static_cast<int>(raw_input.size()));
    for (int i = start; i < end; i++) {
      for (const auto& plane : raw_input[i]) {
        DataType value;
        AsDataType(plane.value, &value);
        for (auto bit : IterateBits(plane.mask)) {
          *(iter + bit) = value;
        }
        iter += 64;
      }
    }
  }

  Ort::IoBinding binding{network_->session_[step - 1]};
  for (size_t i = 0; i < NetworkInfo::output_size_; i++) {
    int size = NetworkInfo::output_tensors_step_[i];
    int64_t dims[] = {batch_size, size};
    binding.BindOutput(
        network_->outputs_[i].c_str(),
        Ort::Value::CreateTensor<DataType>(
            network_->memory_info_,
            &inputs_outputs_->output_tensors_data_device_[i][start * size],
            size * batch_size, dims, std::size(dims)));
  }

  int64_t dims[] = {batch_size, kInputPlanes, 8, 8};
  binding.BindInput(
      network_->inputs_[0].c_str(),
      Ort::Value::CreateTensor<DataType>(
          network_->memory_info_,
          &inputs_outputs_
               ->input_tensor_data_device_[start * kInputPlanes * 8 * 8],
          batch_size * kInputPlanes * 8 * 8, dims, std::size(dims)));
  return binding;
}

template <typename NetworkInfoType>
void OnnxComputation<NetworkInfoType>::ComputeBlocking() {
  int batch_size = network_->batch_size_;
  if (batch_size < 0) {
    batch_size = std::max(GetBatchSize(), network_->min_batch_size_);
  }
  for (size_t i = 0; i < (size_t)GetBatchSize();) {
    auto [batch, step] =
        Base::GetBatchStep(i, batch_size, GetBatchSize(), network_->steps_);

    auto binding = PrepareInputs(i, batch, step);

    auto lock = Base::LockComputeSession(network_->lock_);
    Base::UploadInputs(binding, *network_, *inputs_outputs_, i, batch);

    network_->session_[step - 1].Run(Base::GetRunOptions(), binding);

    Base::DownloadOutputs(binding, *network_, *inputs_outputs_, i, batch);
    i += batch;
  }
  if constexpr (NetworkInfo::cpu_wdl_) {
    static_assert(NetworkInfo::wdl_head_ != -1,
                  "WDL head required for CPU softmax.");

    Base::WaitForWDL(*inputs_outputs_);

    const DataType* data =
        inputs_outputs_->GetOutputData(NetworkInfo::wdl_head_);

    for (size_t i = 0; i < (size_t)GetBatchSize(); i++) {
      float w = AsFloat(data[i * 3 + 0]);
      float d = AsFloat(data[i * 3 + 1]);
      float l = AsFloat(data[i * 3 + 2]);
      // Value softmax done cpu side.
      float m = std::max({w, d, l});
      w = std::exp(w - m);
      d = std::exp(d - m);
      l = std::exp(l - m);
      float sum = w + d + l;
      w /= sum;
      l /= sum;
      d = 1.0f - w - l;
      inputs_outputs_->wdl_output_data_[2 * i + 0] = w - l;
      inputs_outputs_->wdl_output_data_[2 * i + 1] = d;
    }
  }
  Base::WaitForOutputs(*inputs_outputs_);
}

template <typename Provider>
Ort::SessionOptions OnnxNetwork<Provider>::GetOptions(
    const SessionParams& params, int attempt) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(params.threads_);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  return Base::GetOptions(std::move(options), params, inputs_, attempt);
}

template <typename Provider>
OnnxNetwork<Provider>::OnnxNetwork(const WeightsFile& file,
                                   const OptionsDict& opts)
    : Base(opts),
      onnx_env_(ORT_LOGGING_LEVEL_WARNING, "lc0"),
      capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()} {
  onnx_env_.DisableTelemetryEvents();

  SessionParams params(file, opts);

  batch_size_ = params.batch_size_;
  steps_ = params.steps_;
  min_batch_size_ = params.min_batch_size_;
  max_batch_size_ = params.max_batch_size_;
  opt_batch_size_ = params.opt_batch_size_;

  const auto& md = file.onnx_model();
  if (!md.has_input_planes()) {
    throw Exception("NN doesn't have input planes defined.");
  }
  inputs_.emplace_back(md.input_planes());
  if (NetworkInfo::wdl_head_ != -1) {
    outputs_.emplace_back(md.output_wdl());
  } else if (md.has_output_value()) {
    outputs_.emplace_back(md.output_value());
  } else {
    throw Exception("NN doesn't have value head.");
  }
  if (!md.has_output_policy()) {
    throw Exception("NN doesn't have policy head defined.");
  }
  outputs_.emplace_back(md.output_policy());
  if (NetworkInfo::mlh_head_ != -1) {
    outputs_.emplace_back(md.output_mlh());
  }

  memory_info_ = Base::GetMemoryInfo();

  int attempt = 0;
  for (int step = 1; step <= steps_; step++) {
    params.max_batch_size_ =
        batch_size_ > 0 ? batch_size_ * step : max_batch_size_;
    params.min_batch_size_ = batch_size_ > 0
                                 ? params.max_batch_size_ - batch_size_ + 1
                                 : min_batch_size_;
    CERR << "Building engine for step " << step << " with batch size "
         << params.min_batch_size_ << "-" << params.max_batch_size_
         << " with optimization target " << params.opt_batch_size_ << ".";
    session_.emplace_back(onnx_env_, file.onnx_model().model().data(),
                          file.onnx_model().model().size(),
                          GetOptions(params, attempt++));

    if (!Base::IsSessionGood(params, file.onnx_model().model().size(),
                             attempt)) {
      if (attempt > 3) {
        throw Exception("TensorRT failed to build a good engine after " +
                        std::to_string(attempt) + " attempts.");
      }
      CERR << "WARNING: TensorRT build a bad engine! Deleted the bad engine "
              "and retrying.";
      session_.pop_back();
      step--;
      continue;
    }
    attempt = 0;
  }
}

template <typename T, bool cpu_wdl, bool wdl_head, bool mlh_head>
struct NetworkInfo {
  using DataType = T;
  static constexpr bool cpu_wdl_ = cpu_wdl && wdl_head;
  static constexpr int wdl_head_ = wdl_head ? 0 : -1;
  static constexpr int value_head_ = !wdl_head ? 0 : -1;
  static constexpr int policy_head_ = 1;
  static constexpr int mlh_head_ = mlh_head ? 2 : -1;
  static constexpr int output_size_ = mlh_head ? 3 : 2;
  static constexpr size_t output_tensors_step_[3] = {
      wdl_head ? 3 : 1, kNumOutputPolicy, mlh_head ? 1 : 0};
};

template <typename NetworkInfoType>
std::unique_ptr<Network> MakeOnnxNetwork(const WeightsFile& file,
                                         const OptionsDict& opts,
                                         OnnxProvider provider) {
  switch (provider) {
    case OnnxProvider::CPU:
      return std::make_unique<OnnxNetwork<OnnxCPUProvider<NetworkInfoType>>>(
          file, opts);
    case OnnxProvider::CUDA:
      return std::make_unique<OnnxNetwork<OnnxCUDAProvider<NetworkInfoType>>>(
          file, opts);
    case OnnxProvider::TRT:
      return std::make_unique<OnnxNetwork<OnnxTRTProvider<NetworkInfoType>>>(
          file, opts);
    case OnnxProvider::ROCM:
      return std::make_unique<OnnxNetwork<OnnxROCMProvider<NetworkInfoType>>>(
          file, opts);
    case OnnxProvider::DML:
      return std::make_unique<OnnxNetwork<OnnxDMLProvider<NetworkInfoType>>>(
          file, opts);
    default:
      throw Exception("Unsupported ONNX provider.");
  }
}

template <bool cpu_wdl, bool wdl_head, bool mlh_head>
std::unique_ptr<Network> MakeOnnxNetwork(const WeightsFile& file,
                                         const OptionsDict& opts,
                                         OnnxProvider provider) {
  switch (file.onnx_model().data_type()) {
    case pblczero::OnnxModel::FLOAT:
      return MakeOnnxNetwork<NetworkInfo<float, cpu_wdl, wdl_head, mlh_head>>(
          file, opts, provider);
    case pblczero::OnnxModel::FLOAT16:
      return MakeOnnxNetwork<
          NetworkInfo<Ort::Float16_t, cpu_wdl, wdl_head, mlh_head>>(file, opts,
                                                                    provider);
    case pblczero::OnnxModel::BFLOAT16:
      return MakeOnnxNetwork<
          NetworkInfo<Ort::BFloat16_t, cpu_wdl, wdl_head, mlh_head>>(file, opts,
                                                                     provider);
    default:
      throw Exception("Unsupported ONNX data type.");
  }
}

template <bool cpu_wdl, bool wdl_head>
std::unique_ptr<Network> MakeOnnxNetwork(const WeightsFile& file,
                                         const OptionsDict& opts,
                                         OnnxProvider provider) {
  if (file.onnx_model().has_output_mlh()) {
    return MakeOnnxNetwork<cpu_wdl, wdl_head, true>(file, opts, provider);
  } else {
    return MakeOnnxNetwork<cpu_wdl, wdl_head, false>(file, opts, provider);
  }
}

template <bool cpu_wdl>
std::unique_ptr<Network> MakeOnnxNetwork(const WeightsFile& file,
                                         const OptionsDict& opts,
                                         OnnxProvider provider) {
  if (file.onnx_model().has_output_wdl()) {
    return MakeOnnxNetwork<cpu_wdl, true>(file, opts, provider);
  } else {
    return MakeOnnxNetwork<false, false>(file, opts, provider);
  }
}

std::unique_ptr<Network> MakeOnnxNetwork(const WeightsFile& file,
                                         const OptionsDict& opts,
                                         OnnxProvider provider, bool cpu_wdl) {
  if (cpu_wdl) {
    return MakeOnnxNetwork<true>(file, opts, provider);
  } else {
    return MakeOnnxNetwork<false>(file, opts, provider);
  }
}

template <OnnxProvider kProvider>
std::unique_ptr<Network> MakeOnnxNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& opts) {
  if (!w) throw Exception("The ONNX backend requires a network file.");

  if (w->has_onnx_model()) {
    return MakeOnnxNetwork(*w, opts, kProvider, false);
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
    return MakeOnnxNetwork(converted, opts, kProvider, true);
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
}  // namespace lczero::onnx
