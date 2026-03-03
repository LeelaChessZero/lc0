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
  combining it with AMD ROCm libraries from the ROCm toolkit
  (or a modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include "neural/network.h"
#include "rocm_common.h"

namespace lczero {
namespace rocm_backend {

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left,
                size_t tensor_mem_size = 0, size_t scratch_size = 0,
                bool rocblasDisableTensorCores = false, bool use_fp16 = false,
                int gpu_id = 0) {
    (void)rocblasDisableTensorCores;

    // Ensure correct device context (needed for multi-stream resource creation)
    if (tensor_mem_size > 0) {
      ReportHIPErrors(hipSetDevice(gpu_id));
    }
    ReportHIPErrors(hipHostMalloc(
        &input_masks_mem_, maxBatchSize * kInputPlanes * sizeof(uint64_t),
        hipHostMallocMapped));
    ReportHIPErrors(hipHostGetDevicePointer((void**)&input_masks_mem_gpu_,
                                            input_masks_mem_, 0));

    ReportHIPErrors(hipHostMalloc(&input_val_mem_,
                                  maxBatchSize * kInputPlanes * sizeof(float),
                                  hipHostMallocMapped));
    ReportHIPErrors(hipHostGetDevicePointer((void**)&input_val_mem_gpu_,
                                            input_val_mem_, 0));

    ReportHIPErrors(hipHostMalloc(
        &op_policy_mem_, maxBatchSize * kNumOutputPolicy * sizeof(float), 0));

    // Seperate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it.
    ReportHIPErrors(hipMalloc(&op_policy_mem_gpu_,
                              maxBatchSize * kNumOutputPolicy * sizeof(float)));

    ReportHIPErrors(hipHostMalloc(&op_value_mem_,
                                  maxBatchSize * (wdl ? 3 : 1) * sizeof(float),
                                  hipHostMallocMapped));
    ReportHIPErrors(
        hipHostGetDevicePointer((void**)&op_value_mem_gpu_, op_value_mem_, 0));

    // Separate FP16 buffer for value output when using FP16 backend
    op_value_mem_fp16_ = nullptr;
    op_value_mem_gpu_fp16_ = nullptr;
    if (use_fp16) {
      ReportHIPErrors(hipHostMalloc(&op_value_mem_fp16_,
                                    maxBatchSize * (wdl ? 3 : 1) * sizeof(half),
                                    hipHostMallocMapped));
      ReportHIPErrors(hipHostGetDevicePointer((void**)&op_value_mem_gpu_fp16_,
                                              op_value_mem_fp16_, 0));
    }
    if (moves_left) {
      ReportHIPErrors(hipHostMalloc(&op_moves_left_mem_,
                                    maxBatchSize * sizeof(float),
                                    hipHostMallocMapped));
      ReportHIPErrors(hipHostGetDevicePointer((void**)&op_moves_left_mem_gpu_,
                                              op_moves_left_mem_, 0));
    }

    // memory for network execution managed inside this structure
    if (tensor_mem_size) {
      multi_stream_ = true;
      // Create stream with NonBlocking flag to match CUDA behavior
      ReportHIPErrors(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));

      ReportHIPErrors(hipMalloc(&scratch_mem_, scratch_size));
      for (auto& mem : tensor_mem_) {
        ReportHIPErrors(hipMalloc(&mem, tensor_mem_size));
       ReportHIPErrors(hipMemsetAsync(mem, 0, tensor_mem_size, stream_));
      }

      // Create per-stream rocBLAS handle for thread-safe concurrent execution
      ReportROCBLASErrors(rocblas_create_handle(&rocblas_));
      ReportROCBLASErrors(rocblas_set_stream(rocblas_, stream_));
    } else {
      multi_stream_ = false;
    }
  }
  ~InputsOutputs() {
    ReportHIPErrors(hipHostFree(input_masks_mem_));
    ReportHIPErrors(hipHostFree(input_val_mem_));
    ReportHIPErrors(hipHostFree(op_policy_mem_));
    ReportHIPErrors(hipFree(op_policy_mem_gpu_));
    ReportHIPErrors(hipHostFree(op_value_mem_));
    if (op_value_mem_fp16_ != nullptr)
      ReportHIPErrors(hipHostFree(op_value_mem_fp16_));
    // Note: op_value_mem_gpu_fp16_ is obtained via hipHostGetDevicePointer
    // and should NOT be manually freed - it's automatically freed with
    // op_value_mem_fp16_
    if (op_moves_left_mem_ != nullptr)
      ReportHIPErrors(hipHostFree(op_moves_left_mem_));

    if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) ReportHIPErrors(hipFree(mem));
      }
      if (scratch_mem_) ReportHIPErrors(hipFree(scratch_mem_));
      if (offset_pointers_) ReportHIPErrors(hipFree(offset_pointers_));
      if (head_offset_pointers_) {
        ReportHIPErrors(hipFree(head_offset_pointers_));
      }
      (void)hipStreamDestroy(stream_);
      if (rocblas_) rocblas_destroy_handle(rocblas_);
    }
  }
  uint64_t* input_masks_mem_;
  float* input_val_mem_;
  float* op_policy_mem_;
  float* op_value_mem_;
  float* op_moves_left_mem_ = nullptr;

  // GPU pointers for the above allocations.
  uint64_t* input_masks_mem_gpu_;
  float* input_val_mem_gpu_;
  float* op_value_mem_gpu_;
  float* op_moves_left_mem_gpu_;

  // This is a seperate copy.
  float* op_policy_mem_gpu_;

  // FP16 buffers for value output (used by FP16 backend)
  half* op_value_mem_fp16_;
  half* op_value_mem_gpu_fp16_;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_;
  void* tensor_mem_[3];
  void* scratch_mem_;
  void** offset_pointers_ = nullptr;
  void** head_offset_pointers_ = nullptr;

  // HIP stream used to run the network
  hipStream_t stream_;

  // rocBLAS handle used to run the network
  rocblas_handle rocblas_;
};

}  // namespace rocm_backend
}  // namespace lczero
