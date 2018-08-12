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
  Toolkit and the the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

// common functions/structures for cudnn and TRT backends

namespace lczero {

  // Hard-coded for now, no point in going above this anyway (can possibly save
  // memory by reducing this).
  static constexpr int kMaxBatchSize = 1024;
  static constexpr int kNumOutputPolicy = 1858;

  void CudaError(cudaError_t status, const char *file, const int &line);
  #define ReportCUDAErrors(status) CudaError(status, __FILE__, __LINE__)

  struct InputsOutputs {
    InputsOutputs() {
      ReportCUDAErrors(cudaHostAlloc(
          &input_masks_mem_, kMaxBatchSize * kInputPlanes * sizeof(uint64_t),
          cudaHostAllocMapped));
      ReportCUDAErrors(
          cudaHostGetDevicePointer(&input_masks_mem_gpu_, input_masks_mem_, 0));

      ReportCUDAErrors(cudaHostAlloc(&input_val_mem_,
                                    kMaxBatchSize * kInputPlanes * sizeof(float),
                                    cudaHostAllocMapped));
      ReportCUDAErrors(
          cudaHostGetDevicePointer(&input_val_mem_gpu_, input_val_mem_, 0));

      ReportCUDAErrors(cudaHostAlloc(
          &op_policy_mem_, kMaxBatchSize * kNumOutputPolicy * sizeof(float),
          cudaHostAllocMapped));
      ReportCUDAErrors(
          cudaHostGetDevicePointer(&op_policy_mem_gpu_, op_policy_mem_, 0));

      ReportCUDAErrors(cudaHostAlloc(
          &op_value_mem_, kMaxBatchSize * sizeof(float), cudaHostAllocMapped));
      ReportCUDAErrors(
          cudaHostGetDevicePointer(&op_value_mem_gpu_, op_value_mem_, 0));
    }
    ~InputsOutputs() {
      ReportCUDAErrors(cudaFreeHost(input_masks_mem_));
      ReportCUDAErrors(cudaFreeHost(input_val_mem_));
      ReportCUDAErrors(cudaFreeHost(op_policy_mem_));
      ReportCUDAErrors(cudaFreeHost(op_value_mem_));
    }
    uint64_t *input_masks_mem_;
    float *input_val_mem_;
    float *op_policy_mem_;
    float *op_value_mem_;

    // GPU pointers for the above allocations
    uint64_t *input_masks_mem_gpu_;
    float *input_val_mem_gpu_;
    float *op_policy_mem_gpu_;
    float *op_value_mem_gpu_;
  };  

  void processConvBlock(Weights::ConvBlock &block, bool foldBNLayer = false);

  void expandPlanes_Fp32_NCHW(float *output, const uint64_t *masks, const float *values, int n);
}  // namespace lczero
