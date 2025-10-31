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

#include "cuda_common.h"
#include "neural/network.h"

namespace lczero {
namespace cudnn_backend {

template <typename NetworkInfo>
struct CudaGraphCapture;

template <typename NetworkInfo>
struct CudaGraphExec {
  ~CudaGraphExec() {
    if (graph_exec_ != nullptr) {
      ReportCUDAErrors(cudaGraphExecDestroy(graph_exec_));
    }
  }

  CudaGraphExec& operator=(const CudaGraphCapture<NetworkInfo>&);
  explicit operator bool() const { return graph_exec_ != nullptr; }

  void Launch(cudaStream_t stream) {
    ReportCUDAErrors(cudaGraphLaunch(graph_exec_, stream));
  }
  cudaGraphExec_t graph_exec_ = nullptr;
};

template <typename NetworkInfo>
struct InputsOutputs {
  using DataType = std::conditional_t<NetworkInfo::fast_cpu_conversion_,
                                      typename NetworkInfo::DataType, float>;
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left,
                size_t tensor_mem_size = 0, size_t scratch_size = 0,
                bool cublasDisableTensorCores = false) {
    ReportCUDAErrors(cudaHostAlloc(
        &input_masks_mem_, maxBatchSize * kInputPlanes * sizeof(uint64_t),
        cudaHostAllocMapped));
    ReportCUDAErrors(cudaMalloc(
        &input_masks_mem_gpu_, maxBatchSize * kInputPlanes * sizeof(uint64_t)));

    ReportCUDAErrors(cudaHostAlloc(
        &input_val_mem_, maxBatchSize * kInputPlanes * sizeof(input_val_mem_[0]),
        cudaHostAllocMapped));
    ReportCUDAErrors(cudaMalloc(
        &input_val_mem_gpu_, maxBatchSize * kInputPlanes * sizeof(input_val_mem_gpu_[0])));

    ReportCUDAErrors(
        cudaHostAlloc(&op_policy_mem_,
                      maxBatchSize * kNumOutputPolicy * sizeof(op_policy_mem_[0]), 0));

    // Seperate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it.
    ReportCUDAErrors(
        cudaMalloc(&op_policy_mem_gpu_,
                   maxBatchSize * kNumOutputPolicy * sizeof(op_policy_mem_[0])));
    ReportCUDAErrors(cudaHostAlloc(
        &op_value_mem_, maxBatchSize * (wdl ? 3 : 1) * sizeof(op_value_mem_[0]),
        cudaHostAllocMapped));
    ReportCUDAErrors(cudaMalloc(
        &op_value_mem_gpu_, maxBatchSize * (wdl ? 3 : 1) * sizeof(op_value_mem_gpu_[0])));
    if (wdl && sizeof(DataType) != sizeof(float)) {
      wdl_cpu_softmax_ = std::make_unique<float[]>(maxBatchSize * 2);
    }
    ReportCUDAErrors(
        cudaEventCreateWithFlags(&upload_done_event_, cudaEventDisableTiming));
    ReportCUDAErrors(
        cudaEventCreateWithFlags(&policy_done_event_, cudaEventDisableTiming));
    ReportCUDAErrors(
        cudaEventCreateWithFlags(&value_done_event_, cudaEventDisableTiming));
    ReportCUDAErrors(cudaEventCreateWithFlags(&wdl_download_done_event_,
                                              cudaEventDisableTiming));
    ReportCUDAErrors(cudaEventCreateWithFlags(&download_done_event_,
                                              cudaEventDisableTiming));
    if (moves_left) {
      ReportCUDAErrors(cudaHostAlloc(&op_moves_left_mem_,
                                     maxBatchSize * sizeof(op_moves_left_mem_[0]),
                                     cudaHostAllocMapped));
      ReportCUDAErrors(
          cudaMalloc(&op_moves_left_mem_gpu_, maxBatchSize * sizeof(op_moves_left_mem_gpu_[0])));
      ReportCUDAErrors(cudaEventCreateWithFlags(&moves_left_done_event_,
                                                cudaEventDisableTiming));
    }

    ReportCUDAErrors(
        cudaStreamCreateWithFlags(&exec_stream_, cudaStreamNonBlocking));
    ReportCUDAErrors(
        cudaEventCreateWithFlags(&join_capture_event_, cudaEventDisableTiming));
    cuda_graphs_ = std::make_unique<CudaGraphExec<NetworkInfo>[]>(maxBatchSize);

    // memory for network execution managed inside this structure
    if (tensor_mem_size) {
      multi_stream_ = true;
      ReportCUDAErrors(
          cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
      ReportCUDAErrors(
          cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking));
      ReportCUDAErrors(
          cudaStreamCreateWithFlags(&download_stream_, cudaStreamNonBlocking));
      ReportCUDAErrors(cudaMalloc(&scratch_mem_, scratch_size));
      for (auto& mem : tensor_mem_) {
        ReportCUDAErrors(cudaMalloc(&mem, tensor_mem_size));
        ReportCUDAErrors(
            cudaMemsetAsync(mem, 0, tensor_mem_size, compute_stream_));
      }
      ReportCUBLASErrors(cublasCreate(&cublas_));
      ReportCUBLASErrors(cublasSetMathMode(
          cublas_, cublasDisableTensorCores ? CUBLAS_PEDANTIC_MATH
                                            : CUBLAS_TENSOR_OP_MATH));
      ReportCUBLASErrors(cublasSetStream(cublas_, compute_stream_));
    } else {
      multi_stream_ = false;
    }
  }
  ~InputsOutputs() {
    ReportCUDAErrors(cudaFreeHost(input_masks_mem_));
    ReportCUDAErrors(cudaFree(input_masks_mem_gpu_));
    ReportCUDAErrors(cudaFreeHost(input_val_mem_));
    ReportCUDAErrors(cudaFree(input_val_mem_gpu_));
    ReportCUDAErrors(cudaFreeHost(op_policy_mem_));
    ReportCUDAErrors(cudaFree(op_policy_mem_gpu_));
    ReportCUDAErrors(cudaFreeHost(op_value_mem_));
    ReportCUDAErrors(cudaFree(op_value_mem_gpu_));
    ReportCUDAErrors(cudaEventDestroy(upload_done_event_));
    ReportCUDAErrors(cudaEventDestroy(policy_done_event_));
    ReportCUDAErrors(cudaEventDestroy(value_done_event_));
    ReportCUDAErrors(cudaEventDestroy(wdl_download_done_event_));
    ReportCUDAErrors(cudaEventDestroy(download_done_event_));
    if (op_moves_left_mem_ != nullptr) {
      ReportCUDAErrors(cudaFreeHost(op_moves_left_mem_));
      ReportCUDAErrors(cudaFree(op_moves_left_mem_gpu_));
      ReportCUDAErrors(cudaEventDestroy(moves_left_done_event_));
    }
    ReportCUDAErrors(cudaEventDestroy(join_capture_event_));
    ReportCUDAErrors(cudaStreamDestroy(exec_stream_));

    if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) ReportCUDAErrors(cudaFree(mem));
      }
      if (scratch_mem_) ReportCUDAErrors(cudaFree(scratch_mem_));
      if (offset_pointers_) ReportCUDAErrors(cudaFree(offset_pointers_));
      if (head_offset_pointers_) {
        ReportCUDAErrors(cudaFree(head_offset_pointers_));
      }
      ReportCUDAErrors(cudaStreamDestroy(compute_stream_));
      ReportCUDAErrors(cudaStreamDestroy(upload_stream_));
      ReportCUDAErrors(cudaStreamDestroy(download_stream_));
      ReportCUBLASErrors(cublasDestroy(cublas_));
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

  std::unique_ptr<float []> wdl_cpu_softmax_;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_;
  void* tensor_mem_[3];
  void* scratch_mem_;
  void** offset_pointers_ = nullptr;
  void** head_offset_pointers_ = nullptr;

  // cuda stream used to run the network
  cudaStream_t compute_stream_ = nullptr;
  cudaStream_t upload_stream_ = nullptr;
  cudaStream_t download_stream_ = nullptr;

  // cuda events to synchronize between streams
  cudaEvent_t upload_done_event_ = nullptr;
  cudaEvent_t policy_done_event_ = nullptr;
  cudaEvent_t value_done_event_ = nullptr;
  cudaEvent_t moves_left_done_event_ = nullptr;
  cudaEvent_t wdl_download_done_event_ = nullptr;
  cudaEvent_t download_done_event_ = nullptr;

  // cuda graph support
  cudaStream_t exec_stream_ = nullptr;
  std::unique_ptr<CudaGraphExec<NetworkInfo>[]> cuda_graphs_;
  cudaEvent_t join_capture_event_ = nullptr;

  // cublas handle used to run the network
  cublasHandle_t cublas_ = nullptr;
};

template <typename NetworkInfo>
struct CudaGraphCapture {
  static constexpr int kMinimumFreeMemory = 100 * 1024 * 1024;

  CudaGraphCapture(InputsOutputs<NetworkInfo>& io, cudaStream_t upload_stream,
                   cudaStream_t download_stream)
      : io_(io),
        upload_stream_(upload_stream),
        download_stream_(download_stream) {
    ReportCUDAErrors(cudaStreamBeginCapture(upload_stream_,
                                            cudaStreamCaptureModeThreadLocal));
  }

  ~CudaGraphCapture() {
    if (graph_ != nullptr) {
      ReportCUDAErrors(cudaGraphDestroy(graph_));
    }
  }

  static bool EnsureEnoughFreeMemory() {
    size_t free_mem = 0;
    size_t total_mem = 0;
    ReportCUDAErrors(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem > kMinimumFreeMemory;
  }

  void EndCapture() {
    ReportCUDAErrors(
        cudaEventRecord(io_.join_capture_event_, download_stream_));
    ReportCUDAErrors(
        cudaStreamWaitEvent(upload_stream_, io_.join_capture_event_, 0));
    ReportCUDAErrors(cudaStreamEndCapture(upload_stream_, &graph_));
  }

  InputsOutputs<NetworkInfo>& io_;
  cudaStream_t upload_stream_;
  cudaStream_t download_stream_;

  cudaGraph_t graph_ = nullptr;
};

template <typename NetworkInfo>
inline CudaGraphExec<NetworkInfo>& CudaGraphExec<NetworkInfo>::operator=(
    const CudaGraphCapture<NetworkInfo>& graph) {
  assert(graph_exec_ == nullptr);
  if (graph.graph_ == nullptr) {
    throw Exception("Trying to instantiate an nullptr cuda graph");
  }
  ReportCUDAErrors(
      cudaGraphInstantiate(&graph_exec_, graph.graph_, nullptr, nullptr, 0));
  ReportCUDAErrors(cudaGraphUpload(graph_exec_, graph.io_.exec_stream_));
  return *this;
}

}  // namespace cudnn_backend
}  // namespace lczero
