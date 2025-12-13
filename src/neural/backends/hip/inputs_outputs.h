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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <cassert>
#include <memory>

#include "hip_common.h"
#include "neural/network.h"
#include "utils/bit.h"

namespace lczero {
namespace hip_backend {

inline void ToType(float& dst, float src) { dst = src; }
inline void ToType(half& dst, float src) {
  auto temp = FP32toFP16(src);
  dst = bit_cast<half>(temp);
}

inline float FromType(float src) { return src; }
inline float FromType(half src) {
  uint16_t temp = bit_cast<uint16_t>(src);
  return FP16toFP32(temp);
}

template <typename DataType>
struct CudaGraphCapture;

template <typename DataType>
struct CudaGraphExec {
  ~CudaGraphExec() {
    if (graph_exec_ != nullptr) {
      ReportHIPErrors(hipGraphExecDestroy(graph_exec_));
    }
  }

  CudaGraphExec& operator=(const CudaGraphCapture<DataType>&);
  explicit operator bool() const { return graph_exec_ != nullptr; }

  void Launch(hipStream_t stream) {
    ReportHIPErrors(hipGraphLaunch(graph_exec_, stream));
  }
  hipGraphExec_t graph_exec_ = nullptr;
};

template <typename DataType>
struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left,
                size_t tensor_mem_size = 0, size_t scratch_size = 0,
                bool cublasDisableTensorCores = false) {
    ReportHIPErrors(hipHostAlloc(
        &input_masks_mem_, maxBatchSize * kInputPlanes * sizeof(uint64_t),
        hipHostMallocMapped));
    ReportHIPErrors(hipMalloc(
        &input_masks_mem_gpu_, maxBatchSize * kInputPlanes * sizeof(uint64_t)));

    ReportHIPErrors(
        hipHostAlloc(&input_val_mem_,
                      maxBatchSize * kInputPlanes * sizeof(input_val_mem_[0]),
                      hipHostMallocMapped));
    ReportHIPErrors(hipMalloc(
        &input_val_mem_gpu_,
        maxBatchSize * kInputPlanes * sizeof(input_val_mem_gpu_[0])));

    ReportHIPErrors(hipHostAlloc(
        &op_policy_mem_,
        maxBatchSize * kNumOutputPolicy * sizeof(op_policy_mem_[0]), 0));

    // Seperate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it.
    ReportHIPErrors(hipMalloc(
        &op_policy_mem_gpu_,
        maxBatchSize * kNumOutputPolicy * sizeof(op_policy_mem_[0])));
    ReportHIPErrors(hipHostAlloc(
        &op_value_mem_, maxBatchSize * (wdl ? 3 : 1) * sizeof(op_value_mem_[0]),
        hipHostMallocMapped));
    ReportHIPErrors(hipMalloc(
        &op_value_mem_gpu_,
        maxBatchSize * (wdl ? 3 : 1) * sizeof(op_value_mem_gpu_[0])));
    if (wdl && sizeof(DataType) != sizeof(float)) {
      wdl_cpu_softmax_ = std::make_unique<float[]>(maxBatchSize * 2);
    }
    ReportHIPErrors(
        hipEventCreateWithFlags(&upload_done_event_, hipEventDisableTiming));
    ReportHIPErrors(
        hipEventCreateWithFlags(&policy_done_event_, hipEventDisableTiming));
    ReportHIPErrors(
        hipEventCreateWithFlags(&value_done_event_, hipEventDisableTiming));
    ReportHIPErrors(hipEventCreateWithFlags(&wdl_download_done_event_,
                                              hipEventDisableTiming));
    ReportHIPErrors(hipEventCreateWithFlags(&download_done_event_,
                                              hipEventDisableTiming));
    if (moves_left) {
      ReportHIPErrors(hipHostAlloc(
          &op_moves_left_mem_, maxBatchSize * sizeof(op_moves_left_mem_[0]),
          hipHostMallocMapped));
      ReportHIPErrors(
          hipMalloc(&op_moves_left_mem_gpu_,
                     maxBatchSize * sizeof(op_moves_left_mem_gpu_[0])));
      ReportHIPErrors(hipEventCreateWithFlags(&moves_left_done_event_,
                                                hipEventDisableTiming));
    }

    ReportHIPErrors(
        hipStreamCreateWithFlags(&exec_stream_, hipStreamNonBlocking));
    ReportHIPErrors(
        hipEventCreateWithFlags(&join_capture_event_, hipEventDisableTiming));
    cuda_graphs_ = std::make_unique<CudaGraphExec<DataType>[]>(maxBatchSize);

    // memory for network execution managed inside this structure
    if (tensor_mem_size) {
      multi_stream_ = true;
      ReportHIPErrors(
          hipStreamCreateWithFlags(&compute_stream_, hipStreamNonBlocking));
      ReportHIPErrors(
          hipStreamCreateWithFlags(&upload_stream_, hipStreamNonBlocking));
      ReportHIPErrors(
          hipStreamCreateWithFlags(&download_stream_, hipStreamNonBlocking));
      ReportHIPErrors(hipMalloc(&scratch_mem_, scratch_size));
      for (auto& mem : tensor_mem_) {
        ReportHIPErrors(hipMalloc(&mem, tensor_mem_size));
        ReportHIPErrors(
            hipMemsetAsync(mem, 0, tensor_mem_size, compute_stream_));
      }
      ReportHIPBLASErrors(hipblasCreate(&cublas_));
      ReportHIPBLASErrors(hipblasSetMathMode(
          cublas_, cublasDisableTensorCores ? HIPBLAS_PEDANTIC_MATH
                                            : HIPBLAS_TENSOR_OP_MATH));
      ReportHIPBLASErrors(hipblasSetStream(cublas_, compute_stream_));
    } else {
      multi_stream_ = false;
    }
  }
  ~InputsOutputs() {
    ReportHIPErrors(hipHostFree(input_masks_mem_));
    ReportHIPErrors(hipFree(input_masks_mem_gpu_));
    ReportHIPErrors(hipHostFree(input_val_mem_));
    ReportHIPErrors(hipFree(input_val_mem_gpu_));
    ReportHIPErrors(hipHostFree(op_policy_mem_));
    ReportHIPErrors(hipFree(op_policy_mem_gpu_));
    ReportHIPErrors(hipHostFree(op_value_mem_));
    ReportHIPErrors(hipFree(op_value_mem_gpu_));
    ReportHIPErrors(hipEventDestroy(upload_done_event_));
    ReportHIPErrors(hipEventDestroy(policy_done_event_));
    ReportHIPErrors(hipEventDestroy(value_done_event_));
    ReportHIPErrors(hipEventDestroy(wdl_download_done_event_));
    ReportHIPErrors(hipEventDestroy(download_done_event_));
    if (op_moves_left_mem_ != nullptr) {
      ReportHIPErrors(hipHostFree(op_moves_left_mem_));
      ReportHIPErrors(hipFree(op_moves_left_mem_gpu_));
      ReportHIPErrors(hipEventDestroy(moves_left_done_event_));
    }
    ReportHIPErrors(hipEventDestroy(join_capture_event_));
    ReportHIPErrors(hipStreamDestroy(exec_stream_));

    if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) ReportHIPErrors(hipFree(mem));
      }
      if (scratch_mem_) ReportHIPErrors(hipFree(scratch_mem_));
      if (offset_pointers_) ReportHIPErrors(hipFree(offset_pointers_));
      if (head_offset_pointers_) {
        ReportHIPErrors(hipFree(head_offset_pointers_));
      }
      ReportHIPErrors(hipStreamDestroy(compute_stream_));
      ReportHIPErrors(hipStreamDestroy(upload_stream_));
      ReportHIPErrors(hipStreamDestroy(download_stream_));
      ReportHIPBLASErrors(hipblasDestroy(cublas_));
    }
  }
  uint64_t* input_masks_mem_;
  DataType* input_val_mem_;
  DataType* op_policy_mem_;
  DataType* op_value_mem_;
  DataType* op_moves_left_mem_ = nullptr;

  // Copies in VRAM.
  uint64_t* input_masks_mem_gpu_;
  DataType* input_val_mem_gpu_;
  DataType* op_policy_mem_gpu_;
  DataType* op_value_mem_gpu_;
  DataType* op_moves_left_mem_gpu_ = nullptr;

  std::unique_ptr<float[]> wdl_cpu_softmax_;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_;
  void* tensor_mem_[3];
  void* scratch_mem_;
  void** offset_pointers_ = nullptr;
  void** head_offset_pointers_ = nullptr;

  // cuda stream used to run the network
  hipStream_t compute_stream_ = nullptr;
  hipStream_t upload_stream_ = nullptr;
  hipStream_t download_stream_ = nullptr;

  // cuda events to synchronize between streams
  hipEvent_t upload_done_event_ = nullptr;
  hipEvent_t policy_done_event_ = nullptr;
  hipEvent_t value_done_event_ = nullptr;
  hipEvent_t moves_left_done_event_ = nullptr;
  hipEvent_t wdl_download_done_event_ = nullptr;
  hipEvent_t download_done_event_ = nullptr;

  // cuda graph support
  hipStream_t exec_stream_ = nullptr;
  std::unique_ptr<CudaGraphExec<DataType>[]> cuda_graphs_;
  hipEvent_t join_capture_event_ = nullptr;

  // cublas handle used to run the network
  hipblasHandle_t cublas_ = nullptr;
};

template <typename DataType>
struct CudaGraphCapture {
  static constexpr int kMinimumFreeMemory = 100 * 1024 * 1024;

  CudaGraphCapture(InputsOutputs<DataType>& io, hipStream_t upload_stream,
                   hipStream_t download_stream)
      : io_(io),
        upload_stream_(upload_stream),
        download_stream_(download_stream) {
    ReportHIPErrors(hipStreamBeginCapture(upload_stream_,
                                            hipStreamCaptureModeThreadLocal));
  }

  ~CudaGraphCapture() {
    if (graph_ != nullptr) {
      ReportHIPErrors(hipGraphDestroy(graph_));
    }
  }

  static bool EnsureEnoughFreeMemory() {
    size_t free_mem = 0;
    size_t total_mem = 0;
    ReportHIPErrors(hipMemGetInfo(&free_mem, &total_mem));
    return free_mem > kMinimumFreeMemory;
  }

  void EndCapture() {
    ReportHIPErrors(
        hipEventRecord(io_.join_capture_event_, download_stream_));
    ReportHIPErrors(
        hipStreamWaitEvent(upload_stream_, io_.join_capture_event_, 0));
    ReportHIPErrors(hipStreamEndCapture(upload_stream_, &graph_));
  }

  InputsOutputs<DataType>& io_;
  hipStream_t upload_stream_;
  hipStream_t download_stream_;

  hipGraph_t graph_ = nullptr;
};

template <typename DataType>
inline CudaGraphExec<DataType>& CudaGraphExec<DataType>::operator=(
    const CudaGraphCapture<DataType>& graph) {
  assert(graph_exec_ == nullptr);
  if (graph.graph_ == nullptr) {
    throw Exception("Trying to instantiate an nullptr cuda graph");
  }
  ReportHIPErrors(
      hipGraphInstantiate(&graph_exec_, graph.graph_, nullptr, nullptr, 0));
#if CUDART_VERSION >= 11010
  ReportHIPErrors(hipGraphUpload(graph_exec_, graph.io_.exec_stream_));
#endif
  return *this;
}

}  // namespace hip_backend
}  // namespace lczero
