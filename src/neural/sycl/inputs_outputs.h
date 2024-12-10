/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors
  Copyright (C) 2023 Intel Corporation

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#include <sycl/sycl.hpp>
#include "neural/network.h"
#include "cuBlasContext.h"

namespace lczero {
namespace sycldnn_backend {

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left, sycl::queue& m_ct1,
                size_t tensor_mem_size = 0, size_t scratch_size = 0,
                bool cublasDisableTensorCores = false): q_ct1(m_ct1) {
  #ifdef USE_CUBLAS
    cublasHandle_t h= cuBlasContextManager::getcuBlasHandle_t();
  #endif                
    input_masks_mem_shared_ = malloc_host<uint64_t>(maxBatchSize * kInputPlanes, q_ct1);
    input_val_mem_shared_ = malloc_host<float>(maxBatchSize * kInputPlanes, q_ct1);
    // Seperate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it.
    op_policy_mem_ = malloc_host<float>(maxBatchSize * kNumOutputPolicy, q_ct1);
    op_policy_mem_gpu_ = malloc_device<float>(maxBatchSize * kNumOutputPolicy, q_ct1);
    op_value_mem_shared_ = malloc_host<float>(maxBatchSize * (wdl ? 3 : 1), q_ct1);

    if (moves_left) {
      op_moves_left_mem_shared_ = malloc_host<float>(maxBatchSize, q_ct1);
    }

    // memory for network execution managed inside this structure
    if (tensor_mem_size) {
      multi_stream_ = true;
      scratch_mem_ = (void*)sycl::malloc_device( scratch_size, q_ct1);
      for (auto& mem : tensor_mem_) {
        mem = (void*)sycl::malloc_device(tensor_mem_size, q_ct1);
        q_ct1.memset(mem, 0, tensor_mem_size);
      }
    } else {
      multi_stream_ = false;
    }
  }


  ~InputsOutputs() {
    /*
    sycl::free(input_masks_mem_shared_, q_ct1);
    sycl::free(input_val_mem_shared_, q_ct1);
    sycl::free(op_value_mem_shared_, q_ct1);
    if (op_moves_left_mem_shared_ != nullptr)
        sycl::free(op_moves_left_mem_shared_, q_ct1);
    sycl::free(op_policy_mem_gpu_, q_ct1);

    if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) 
            sycl::free(mem, q_ct1);
      }
      if (scratch_mem_) 
          sycl::free(scratch_mem_, q_ct1);
      if (offset_pointers_) 
          sycl::free(offset_pointers_, q_ct1);
      if (head_offset_pointers_) {
          sycl::free(head_offset_pointers_, q_ct1);
      } 
      //dpct::get_current_device().destroy_queue(stream_);
      //cublas_ = nullptr;
    } */
  }
  uint64_t* input_masks_mem_shared_;
  float* input_val_mem_shared_;
  float* op_value_mem_shared_;
  float* op_moves_left_mem_shared_ = nullptr;

  // GPU pointers for the above allocations.
  //uint64_t* input_masks_mem_gpu_;
  //float* input_val_mem_gpu_;
  //float* op_value_mem_gpu_;
  //float* op_moves_left_mem_gpu_;

  // This is a seperate copy.
  float* op_policy_mem_gpu_;
  float* op_policy_mem_;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_;
  void* tensor_mem_[3];
  void* scratch_mem_;
  void** offset_pointers_ = nullptr;
  void** head_offset_pointers_ = nullptr;

  // cuda stream used to run the network
  sycl::queue& q_ct1;
};

}  // namespace cudnn_backend
}  // namespace lczero
