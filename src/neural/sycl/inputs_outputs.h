/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

/*   This file is part of Leela Chess Zero.
    Modifications Copyright (C) 2023 Intel Corporation

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
   
   SPDX-License-Identifier: GNU General Public License v3.0 only
*/

#include <sycl/sycl.hpp>
#include "neural/network.h"

#ifdef USE_CUBLAS
#include <sycl/backend/cuda.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuBlasContext.h"
#endif


namespace lczero {
namespace sycldnn_backend {

struct InputsOutputs {
  
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left, sycl::queue& m_ct1, 
                size_t tensor_mem_size = 0, size_t scratch_size = 0,
                bool cublasDisableTensorCores = false): q_ct1(m_ct1)
                
 {


  
  

  #ifdef USE_CUBLAS
    cublasHandle_t h= cuBlasContextManager::getcuBlasHandle_t();
  #endif

  //cudaHostAlloc(&input_masks_mem_, maxBatchSize * kInputPlanes * sizeof(uint64_t), 0));
  //ReportCUDAErrors(cudaHostGetDevicePointer(&input_masks_mem_gpu_, input_masks_mem_, 0));

  //input_masks_mem_ = malloc_host<uint64_t>(maxBatchSize * kInputPlanes, q_ct1);
  //input_masks_mem_gpu_ = (uint64_t *)malloc_device(maxBatchSize * kInputPlanes, q_ct1);
   
   input_masks_mem_shared_ = malloc_host<uint64_t>(maxBatchSize * kInputPlanes, q_ct1);
   //input_masks_mem_shared_ = malloc_shared<uint64_t>(maxBatchSize * kInputPlanes, q_ct1);


  //ReportCUDAErrors(cudaHostAlloc(&input_val_mem_, maxBatchSize * kInputPlanes * sizeof(float), 0));
  //ReportCUDAErrors(cudaHostGetDevicePointer(&input_val_mem_gpu_, input_val_mem_, 0));

  //input_val_mem_ = malloc_host<float>(maxBatchSize * kInputPlanes, q_ct1);
  //input_val_mem_gpu_ = (float *)malloc_device(maxBatchSize * kInputPlanes, q_ct1);

   input_val_mem_shared_ = malloc_host<float>(maxBatchSize * kInputPlanes, q_ct1);
   //input_val_mem_shared_ = malloc_shared<float>(maxBatchSize * kInputPlanes, q_ct1);


   // ReportCUDAErrors(cudaHostAlloc(&op_policy_mem_, maxBatchSize * kNumOutputPolicy * sizeof(float), 0));

    // Seperate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it.
    //ReportCUDAErrors(cudaMalloc(&op_policy_mem_gpu_, maxBatchSize * kNumOutputPolicy * sizeof(float)));

    op_policy_mem_ = malloc_host<float>(maxBatchSize * kNumOutputPolicy, q_ct1);
    op_policy_mem_gpu_ = malloc_device<float>(maxBatchSize * kNumOutputPolicy, q_ct1);
   // op_policy_mem_shared_ = (float *)malloc_shared(maxBatchSize * kNumOutputPolicy, q_ct1);
    
    //ReportCUDAErrors(cudaHostAlloc( &op_value_mem_, maxBatchSize * (wdl ? 3 : 1) * sizeof(float), 0));
    
    //ReportCUDAErrors(cudaHostGetDevicePointer(&op_value_mem_gpu_, op_value_mem_, 0));
    
  //op_value_mem_ = malloc_host<float>(maxBatchSize * (wdl ? 3 : 1), q_ct1);
  //op_value_mem_gpu_ = (float *)malloc_device(maxBatchSize * (wdl ? 3 : 1), q_ct1);
    op_value_mem_shared_ = malloc_host<float>(maxBatchSize * (wdl ? 3 : 1), q_ct1);
    //op_value_mem_shared_ = malloc_shared<float>(maxBatchSize * (wdl ? 3 : 1), q_ct1);
    //printf("%i\n", maxBatchSize * (wdl ? 3 : 1));
    
  if (moves_left) {

      //ReportCUDAErrors(cudaHostAlloc(&op_moves_left_mem_, maxBatchSize * sizeof(float), 0));
      //ReportCUDAErrors(cudaHostGetDevicePointer(&op_moves_left_mem_gpu_, op_moves_left_mem_, 0));
    
      //op_moves_left_mem_ = malloc_shared<float>(maxBatchSize, q_ct1);  
      //op_moves_left_mem_gpu_ = (float *)malloc_device(maxBatchSize, q_ct1);
      
      op_moves_left_mem_shared_ = malloc_host<float>(maxBatchSize, q_ct1);
      //op_moves_left_mem_shared_ = malloc_shared<float>(maxBatchSize, q_ct1);

  }

    // memory for network execution managed inside this structure
    
    /*
    if (tensor_mem_size) {
      multi_stream_ = true;
      
      scratch_mem_ = (void*)sycl::malloc_device( scratch_size, q_ct1);
      
      
      for (auto& mem : tensor_mem_) {
      
        mem = (void*)sycl::malloc_device(tensor_mem_size, q_ct1);
      
        q_ct1.memset(mem, 0, tensor_mem_size);
      }
      
      //cublas_ = &dpct::get_default_queue(), 0));
      
      //ReportCUBLASErrors(cublasSetMathMode(
        //  cublas_, cublasDisableTensorCores ? CUBLAS_PEDANTIC_MATH
                                            : CUBLAS_TENSOR_OP_MATH));
      
      //ReportCUBLASErrors((cublas_ = stream_, 0));
    
    } else { */
      multi_stream_ = false;
    //}
  }


  ~InputsOutputs() {
  
  //#ifdef USE_CUBLAS
  //  cuBlasContextManager::destroycuBlasHandle_t();
  //#endif
  
  sycl::free(input_masks_mem_shared_, q_ct1);
  sycl::free(input_val_mem_shared_, q_ct1);
  sycl::free(op_value_mem_shared_, q_ct1);
  sycl::free(op_moves_left_mem_shared_, q_ct1);
  sycl::free(op_policy_mem_gpu_, q_ct1);

  /*
  if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        

        if (mem)
            ReportCUDAErrors((sycl::free(mem, q_ct1), 0));
      }
      

      if (scratch_mem_) 
          sycl::free(scratch_mem_, q_ct1);

      dpct::get_current_device().destroy_queue(stream_);
      
      cublas_ = nullptr;
  }*/
  
  }
  //uint64_t* input_masks_mem_;
  //float* input_val_mem_;
  float* op_policy_mem_;
  //float* op_value_mem_;
  //float* op_moves_left_mem_;

  // GPU pointers for the above allocations.
  //uint64_t* input_masks_mem_gpu_;
  //float* input_val_mem_gpu_;
  //float* op_value_mem_gpu_;
  //float* op_moves_left_mem_gpu_;

  sycl::queue& q_ct1;

  uint64_t* input_masks_mem_shared_ = NULL;
  float* input_val_mem_shared_ = NULL;
  float* op_value_mem_shared_ = NULL;
  float* op_moves_left_mem_shared_ = NULL;


  // This is a seperate copy.
  float* op_policy_mem_gpu_;
  //float* op_policy_mem_shared_;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_;
  void* tensor_mem_[3] = {NULL, NULL, NULL};
  void* scratch_mem_ = NULL;

  // cublas handle used to run the network

};

}  // namespace cudnn_backend
}  // namespace lczero
