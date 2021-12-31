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

#include "neural/network.h"

namespace lczero {
namespace cudnn_backend {

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left,
                size_t tensor_mem_size = 0, size_t scratch_size = 0) {
    ReportCUDAErrors(cudaHostAlloc(
        &input_masks_mem_, maxBatchSize * kInputPlanes * sizeof(uint64_t),
        cudaHostAllocMapped));
    ReportCUDAErrors(
        cudaHostGetDevicePointer(&input_masks_mem_gpu_, input_masks_mem_, 0));

    ReportCUDAErrors(cudaHostAlloc(&input_val_mem_,
                                   maxBatchSize * kInputPlanes * sizeof(float),
                                   cudaHostAllocMapped));
    ReportCUDAErrors(
        cudaHostGetDevicePointer(&input_val_mem_gpu_, input_val_mem_, 0));

    ReportCUDAErrors(cudaHostAlloc(
        &op_policy_mem_, maxBatchSize * kNumOutputPolicy * sizeof(float), 0));

    // Seperate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it.
    ReportCUDAErrors(cudaMalloc(
        &op_policy_mem_gpu_, maxBatchSize * kNumOutputPolicy * sizeof(float)));

    ReportCUDAErrors(cudaHostAlloc(&op_value_mem_,
                                   maxBatchSize * (wdl ? 3 : 1) * sizeof(float),
                                   cudaHostAllocMapped));
    ReportCUDAErrors(
        cudaHostGetDevicePointer(&op_value_mem_gpu_, op_value_mem_, 0));
    if (moves_left) {
      ReportCUDAErrors(cudaHostAlloc(&op_moves_left_mem_,
                                     maxBatchSize * sizeof(float),
                                     cudaHostAllocMapped));
      ReportCUDAErrors(cudaHostGetDevicePointer(&op_moves_left_mem_gpu_,
                                                op_moves_left_mem_, 0));
    }

    memset(graph_created_, 0, sizeof(graph_created_));
    memset(graph_, 0, sizeof(graph_));
    memset(graph_instance_, 0, sizeof(graph_instance_));

    // memory for network execution managed inside this structure
    if (tensor_mem_size) {
      multi_stream_ = true;
      ReportCUDAErrors(cudaStreamCreate(&stream_));
      ReportCUDAErrors(cudaStreamCreate(&stream_copy_));
      ReportCUDAErrors(cudaEventCreate(&policy_ready_event_));
      ReportCUDAErrors(cudaEventCreate(&policy_copied_event_));
      ReportCUDAErrors(cudaMalloc(&scratch_mem_, scratch_size));
      for (auto& mem : tensor_mem_) {
        ReportCUDAErrors(cudaMalloc(&mem, tensor_mem_size));
        ReportCUDAErrors(cudaMemsetAsync(mem, 0, tensor_mem_size, stream_));
      }
      ReportCUBLASErrors(cublasCreate(&cublas_));
      ReportCUBLASErrors(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
      ReportCUBLASErrors(cublasSetStream(cublas_, stream_));
    } else {
      multi_stream_ = false;
    }
  }
  ~InputsOutputs() {
    ReportCUDAErrors(cudaFreeHost(input_masks_mem_));
    ReportCUDAErrors(cudaFreeHost(input_val_mem_));
    ReportCUDAErrors(cudaFreeHost(op_policy_mem_));
    ReportCUDAErrors(cudaFree(op_policy_mem_gpu_));
    ReportCUDAErrors(cudaFreeHost(op_value_mem_));

    if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) ReportCUDAErrors(cudaFree(mem));
      }
      if (scratch_mem_) ReportCUDAErrors(cudaFree(scratch_mem_));

      cudaStreamDestroy(stream_);
      cudaStreamDestroy(stream_copy_);
      cudaEventDestroy(policy_ready_event_);
      cudaEventDestroy(policy_copied_event_);
      cublasDestroy(cublas_);
    }

    for (int i = 0; i < 1024; i++)
      if (graph_created_[i]) {
        cudaGraphDestroy(graph_[i]);
        cudaGraphExecDestroy(graph_instance_[i]);
      }
  }
  uint64_t* input_masks_mem_;
  float* input_val_mem_;
  float* op_policy_mem_;
  float* op_value_mem_;
  float* op_moves_left_mem_;

  // GPU pointers for the above allocations.
  uint64_t* input_masks_mem_gpu_;
  float* input_val_mem_gpu_;
  float* op_value_mem_gpu_;
  float* op_moves_left_mem_gpu_;

  // This is a seperate copy.
  float* op_policy_mem_gpu_;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_;
  void* tensor_mem_[3];
  void* scratch_mem_;

  // cuda stream used to run the network
  cudaStream_t stream_, stream_copy_;
  // cublas handle used to run the network
  cublasHandle_t cublas_;
  cudaEvent_t policy_ready_event_;
  cudaEvent_t policy_copied_event_;


  // cuda-graph related stuff
  // for each batch-size
  bool graph_created_[1024];
  cudaGraph_t graph_[1024];
  cudaGraphExec_t graph_instance_[1024];
};

}  // namespace cudnn_backend
}  // namespace lczero
