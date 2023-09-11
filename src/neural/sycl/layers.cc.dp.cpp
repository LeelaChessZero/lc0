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
#include <iostream>


#ifdef USE_HIPBLAS 
#include "hipblas.h"
#include "cuBlasContext.h"
#elifdef USE_CUBLAS
#include <sycl/backend/cuda.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuBlasContext.h"
#else
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/blas.hpp"
//#include "dpct/lib_common_utils.hpp"
//#include "dpct/blas_utils.hpp"
#endif

#include "layers.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "sycl_common.h"
#include "kernels.h"
#include "utils/fp16_utils.h"

#ifdef USE_HIPBLAS
#define transpose_type hipblasOperation_t 
#define transpose_type_transpose HIPBLAS_OP_T  
#define transpose_type_notranspose HIPBLAS_OP_N 
#elifdef USE_CUBLAS
#define transpose_type cublasOperation_t 
#define transpose_type_transpose CUBLAS_OP_T  
#define transpose_type_notranspose CUBLAS_OP_N 
#else
#define transpose_type oneapi::mkl::transpose 
#define transpose_type_transpose oneapi::mkl::transpose::trans
#define transpose_type_notranspose oneapi::mkl::transpose::nontrans
#endif



namespace lczero {
// void dumpTensor(void* memory, int elements, const char* message, bool fp16 =
// false);

namespace sycldnn_backend {

// Use Single kernel for entire SE operation.
// Right now supported only for fp16 with nhwc and it's quite a bit faster
// than using multiple passes. The flag can be set to false for debugging.
static constexpr bool kUseFusedSELayer = true;

template <typename DataType> BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, sycl::queue& sycl_queue) : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(false), sycl_queue_(sycl_queue) {}

template <typename DataType> BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, bool gemm_ex, sycl::queue& sycl_queue) : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(gemm_ex), sycl_queue_(sycl_queue) {}

template <typename DataType> BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, sycl::queue& sycl_queue) : input_(ip), C(c), H(h), W(w), nhwc_(ip->nhwc_), use_gemm_ex_(false), sycl_queue_(sycl_queue) {}

template <typename DataType> SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs, bool addPrevLayerBias, ActivationFunction activation, sycl::queue& sycl_queue) : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip, sycl_queue),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias),
      act_(activation)
{

  w1_ = sycl::malloc_device<DataType>(ip->GetC() * numFc1Out_, sycl_queue_);
  
  w2_ = sycl::malloc_device<DataType>(2 * C * numFc1Out_, sycl_queue_);

  if (kUseFusedSELayer && nhwc_) {
    
    w1_t_ = sycl::malloc_device<DataType>(C * numFc1Out_, sycl_queue_);
    
    w2_t_ = sycl::malloc_device<DataType>(2 * C * numFc1Out_, sycl_queue_);
  }

  b1_ = sycl::malloc_device<DataType>(numFc1Out_, sycl_queue_);
  
  b2_ = sycl::malloc_device<DataType>(2 * C, sycl_queue_);

  bPrev_ = sycl::malloc_device<DataType>(C, sycl_queue_);
}


template <typename DataType> SELayer<DataType>::~SELayer() {
  

  sycl::free(w1_, sycl_queue_);

  sycl::free(w2_, sycl_queue_);

  sycl::free(b1_, sycl_queue_);

  sycl::free(b2_, sycl_queue_);

  sycl::free(bPrev_, sycl_queue_);
}

template <> void SELayer<float>::LoadWeights(float* w1, float* b1, float* w2, float* b2, float* prevLayerBias, void* /*scratch*/) {
  
  
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t weight_size2 = 2 * weight_size1;

  // Weight for the first FC layer.
  sycl_queue_.memcpy(w1_, w1, weight_size1);

  // Weight for the second FC layer.
  sycl_queue_.memcpy(w2_, w2, weight_size2);

  // Bias for the first FC layer.
  sycl_queue_.memcpy(b1_, b1, numFc1Out_ * sizeof(float));

  // Bias for the second FC layer.
  sycl_queue_.memcpy(b2_, b2, 2 * C * sizeof(float));

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    sycl_queue_.memcpy(bPrev_, prevLayerBias, C * sizeof(float));
  }

  sycl_queue_.wait();

}

//#ifdef USE_CUBLAS
//inline cublasOperation_t convertTranspose(oneapi::mkl::transpose mkl_tran) {
  //  if (mkl_tran == oneapi::mkl::transpose::trans)
   //     return CUBLAS_OP_T;
   // else 
   //     return CUBLAS_OP_N;
//}
//#endif

void cpuTranspose(float* op, float* ip, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}


template <>
void SELayer<sycl::half>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                float* prevLayerBias, void* scratch) {
  const size_t num_weights1 = C * numFc1Out_;
  size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t num_weights2 = 2 * num_weights1;
  size_t weight_size2 = 2 * weight_size1;

  // Transpose the weight matrices for the fused path.
  std::vector<float> temp(weight_size2);

  // Weight for the first FC layer.
 
  sycl_queue_.memcpy(scratch, w1, weight_size1).wait();
  
  copyTypeConverted((sycl::half*)w1_, (float*)scratch, (int)num_weights1, sycl_queue_);

  if (kUseFusedSELayer && nhwc_) {
    // transposed copy for fused SE kernel
    cpuTranspose(temp.data(), w1, numFc1Out_, C);
    
    sycl_queue_.memcpy(scratch, temp.data(), weight_size1).wait();    
    
    copyTypeConverted((sycl::half*)w1_t_, (float*)scratch, (int)num_weights1, sycl_queue_);
  }

  // Weight for the second FC layer.
  sycl_queue_.memcpy(scratch, w2, weight_size2).wait();
  
  copyTypeConverted((sycl::half*)w2_, (float*)scratch, (int)num_weights2, sycl_queue_);
  if (kUseFusedSELayer && nhwc_) {
    cpuTranspose(temp.data(), w2, 2 * C, numFc1Out_);
    
    sycl_queue_.memcpy(scratch, temp.data(), weight_size2).wait();
    copyTypeConverted((sycl::half*)w2_t_, (float*)scratch, (int)num_weights2, sycl_queue_);
  }

  // Bias for the first FC layer.
    
  sycl_queue_.memcpy(scratch, b1, numFc1Out_ * sizeof(float)).wait();
  
  copyTypeConverted((sycl::half*)b1_, (float*)scratch, numFc1Out_, sycl_queue_);

  // Bias for the second FC layer.
  sycl_queue_.memcpy(scratch, b2, 2 * C * sizeof(float)).wait();

  copyTypeConverted((sycl::half*)b2_, (float*)scratch, 2 * C, sycl_queue_);

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    
    sycl_queue_.memcpy(scratch, prevLayerBias, C * sizeof(float)).wait();
    copyTypeConverted((sycl::half*)bPrev_, (float*)scratch, C, sycl_queue_);
  }

} 

template <>
void SELayer<float>::Eval(int N, float* output, const float* input,
                          const float* /*input2*/, void* scratch,
                          size_t scratch_size) {
                            
  // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
  float* op1 = (float*)scratch;
  float* op2 = (float*)scratch + scratch_size / sizeof(float) / 2;

  // 1. Global avg pooling (also adds previous layer bias before computing
  // averages).
  globalAvgPool(N, C, op2, input, bPrev_, false, sycl_queue_);

  // 2. First fully connected layer.
  float alpha = 1.0f, beta = 0.0f;

  #ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
   

  sycl_queue_.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
        cublasSetStream(handle, cudaStreamHandle);  

        ReportCUBLASErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                 N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                 numFc1Out_));

        cudaStreamSynchronize(cudaStreamHandle);
        
        
        });
  });
  #elifdef USE_HIPBLAS
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
   

  sycl_queue_.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue_);
        hipblasSetStream(handle, hipStreamHandle);  

        hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, numFc1Out_,
                                 N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                 numFc1Out_);

        hipStreamSynchronize(hipStreamHandle);
        
        
        });
  });  
  #else
  
  oneapi::mkl::blas::column_major::gemm(sycl_queue_, oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, numFc1Out_, N, C, alpha, w1_, C, op2,
        C, beta, op1, numFc1Out_);

  
  

  #endif 

  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_, sycl_queue_);

  // 3. Second fully connected layer.

  #ifdef USE_CUBLAS
  sycl_queue_.submit([&](sycl::handler &cgh) {
        
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
        cublasSetStream(handle, cudaStreamHandle);  

        ReportCUBLASErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C));

        cudaStreamSynchronize(cudaStreamHandle);
        
        });
  });

  
  #elifdef USE_HIPBLAS
  sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue_);
        hipblasSetStream(handle, hipStreamHandle);  

        hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C);

        hipStreamSynchronize(hipStreamHandle);
        
        });
  });
  #else
    oneapi::mkl::blas::column_major::gemm(sycl_queue_, oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 2 * C, N, numFc1Out_, alpha, w2_,
        numFc1Out_, op1, numFc1Out_, beta, op2, 2 * C);

   

  #endif

  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, NONE, sycl_queue_);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  globalScale(N, C, output, input, op2, bPrev_, false, act_, sycl_queue_);

}



template <>
void SELayer<sycl::half>::Eval(int N, sycl::half* output, const sycl::half* input,
                         const sycl::half* input2, void* scratch, size_t scratch_size) {

  bool se_done = false;
  if (kUseFusedSELayer && nhwc_) {
    se_done = Se_Fp16_NHWC(N, C, numFc1Out_, output, input2, input, w1_t_, b1_,
                           w2_t_, b2_, bPrev_, act_, sycl_queue_);
  }
  if (!se_done) {
    assert(output == input2);
    // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
    sycl::half* op1 = (sycl::half*)scratch;
    sycl::half* op2 = (sycl::half*)scratch + scratch_size / sizeof(sycl::half) / 2;

    // 1. Global avg pooling (also adds previous layer bias before computing
    // averages).
    globalAvgPool(N, C, op2, input, bPrev_, nhwc_, sycl_queue_);

    // 2. First fully connected layer.
    //half_raw one_h{0x3C00};
    //half_raw zero_h{0};

    
    
    #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

    #ifdef USE_CUBLAS
  
    cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

    sycl_queue_.submit([&](sycl::handler &cgh) {
       
        cgh.host_task([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
        cublasSetStream(handle, cudaStreamHandle);  
    
        ReportCUBLASErrors(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                   N, C, &alpha, ((const half *)w1_), C, ((const half *)op2), C, &beta, ((half *)op1),
                                   numFc1Out_));
    
        cudaStreamSynchronize(cudaStreamHandle);
        
        });
    });

    #endif

    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_, sycl_queue_);

    
    #ifdef USE_CUBLAS

    sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
        cublasSetStream(handle, cudaStreamHandle);   
    
        // 3. Second fully connected layer.
        ReportCUBLASErrors(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                   numFc1Out_, &alpha, ((const half *)w2_), numFc1Out_, ((const half *)op1),
                                   numFc1Out_, &beta, ((half *)op2), 2 * C));
  
        cudaStreamSynchronize(cudaStreamHandle);
        
        });
    });  
    
    #endif
    
    
    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, NONE, sycl_queue_);


      
  
    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    globalScale(N, C, output, input, op2, bPrev_, nhwc_, act_, sycl_queue_);
  }
} 

template <typename DataType> FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, sycl_queue), use_bias_(bias), act_(activation) {
  
  
  const size_t weight_size = sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(DataType) * C * H * W;

  weights_ = (DataType *)sycl::malloc_device(weight_size, sycl_queue_);
  

  if (use_bias_) {
    biases_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  } else {
    biases_ = nullptr;
  }
}


template <>
void FCLayer<sycl::half>::LoadWeights(float* cpuWeight, float* cpuBias,
                                void* scratch) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  // also need to convert from fp32 to fp16
  assert(scratch);
  
  sycl_queue_.memcpy(scratch, cpuWeight, weight_size).wait();

  if (nhwc_) {
    convertNCHWtoNHWC((sycl::half*)weights_, (float*)scratch, (int)num_biases,
                      input_->GetC(), (int)num_biases, input_->GetC(),
                      input_->GetH(), input_->GetW(), sycl_queue_);
  } else {
    copyTypeConverted((sycl::half*)weights_, (float*)scratch, (int)num_weights, sycl_queue_);
  }

  if (cpuBias) {
    sycl_queue_.memcpy(scratch, cpuBias, bias_size).wait();
    copyTypeConverted((sycl::half*)biases_, (float*)scratch, (int)num_biases, sycl_queue_);
  } 
} 

template <>
void FCLayer<float>::LoadWeights(float* cpuWeight, float* cpuBias,
                                 void* /*scratch*/) {

  

  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  
  sycl_queue_.memcpy(weights_, cpuWeight, weight_size);
  
  if (use_bias_) {
    sycl_queue_.memcpy(biases_, cpuBias, bias_size);
  }

  //sycl_queue_.wait();
}


 template <>
 void FCLayer<sycl::half>::Eval(int N, sycl::half* output_tensor, const sycl::half* input_tensor,
                          const sycl::half* /*input2*/, void* /*scratch*/,
                          size_t /*scratch_size*/) {
   const int num_outputs = C * H * W;
   const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

   //sycl::half alpha = float2half_rn(1.0f), 
   //beta = float2half_rn(0.0f);
   
   #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

   #ifdef USE_CUBLAS
    cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

    sycl_queue_.submit([&](sycl::handler &cgh) {
        
         cgh.host_task([=](sycl::interop_handle ih) {

         auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
         cublasSetStream(handle, cudaStreamHandle);    
  
         ReportCUBLASErrors(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                  N, num_inputs, &alpha, ((const half *)weights_), num_inputs,
                                  ((const half *)input_tensor), num_inputs, &beta, ((half *)output_tensor),
                                  num_outputs));

         cudaStreamSynchronize(cudaStreamHandle);
        
       });
   });  
   #endif     
   

   if (use_bias_ || (act_ != NONE)) {
     addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
                num_outputs, num_outputs * N, act_, sycl_queue_);
   }
 } 


template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;

  #ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

  sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
        cublasSetStream(handle, cudaStreamHandle);    


        ReportCUBLASErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

        cudaStreamSynchronize(cudaStreamHandle);
        
      });
  });  
  #elifdef USE_HIPBLAS
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue_);
        hipblasSetStream(handle, hipStreamHandle);    


        hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs);

        hipStreamSynchronize(hipStreamHandle);
        
      });
  });
  #else
    

   //printf("3\n");
   oneapi::mkl::blas::column_major::gemm(sycl_queue_, oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, num_outputs, N, num_inputs, alpha,
        weights_, num_inputs, input_tensor, num_inputs, beta, output_tensor,
        num_outputs);
    
    //event.wait();
  

  #endif


  if (use_bias_ || (act_ != NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, sycl_queue_);
  }
}

template <typename DataType> FCLayer<DataType>::~FCLayer() {
  
  free(weights_, sycl_queue_);
  free(biases_, sycl_queue_);
  //ReportCUDAErrors(cudaFree(weights_));
  //ReportCUDAErrors(cudaFree(biases_));


}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, int usedSize, bool attention, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, sycl_queue),
      used_size_(usedSize),
      attention_map_(attention) {

  size_t weight_size = sizeof(short) * this->input_->GetC() * 64;

  if (attention) weight_size = sizeof(short) * usedSize;
  
  weights_ = (short *)sycl::malloc_device(weight_size, sycl_queue_);
}

template <typename DataType>
void PolicyMapLayer<DataType>::LoadWeights(const short* cpuWeight,
                                           void* /*scratch*/) {
  size_t weight_size = sizeof(short) * used_size_;

  if (nhwc_ && !attention_map_) {
    // convert CHW to HWC
    int C = used_size_ / 64;
    int Cin = this->input_->GetC();

    // C is the no. of channels actually used (typically 73).
    // Cin the the no. of channels in previous layer (padded up to 80).
    // Weights of this layer is a mapping to select which output index of the
    // policy vector (1858 elements) maps to every element of input
    // tensor (assuming NCHW layout). Note that there are 73x64 valid inputs
    // (80x64 taking padding), and only 1858 outputs so the mapping isn't
    // one to one. Only few of the indices point to valid index in policy
    // vector. Invalid entries are set to -1.

    // In fp16 mode, the tensor layout is NHWC so the weights need to be
    // adjusted to make them work as intended.

    // This is how the original weights looks like (CHW layout):
    /*
               HW (64)
       ----|-------------|
           |             |
           |             |
    C (73) |             |
           |             |
           |             |
       ------------------|   Cin (80)
           |  padding    |
           |-------------|
    */
    // The padding is not part of the weights provided (used_size_ is 73 x 64).
    //
    // The weights converted to HWC looks like this
    /*
                 C (73)
            |-------------|---|
            |             | P |
            |             | a |
    HW (64) |             | d |
            |             |   |
            |             |   |
            |-----------------|
                     Cin (80)
    */
    // In HWC, because the padding is now part of each row
    // we need to increase the size of weights to account
    // for it.
    // The pad elements point to -1 (invalid output index) and the
    // same kernel works for both HWC and CHW layouts after used_size_
    // is updated to include padding (80x64).

    used_size_ = Cin * 64;
    std::vector<short> convertedWeights(used_size_);

    for (int hw = 0; hw < 64; hw++)
      for (int c = 0; c < Cin; c++) {
        if (c < C)
          convertedWeights[hw * Cin + c] = cpuWeight[c * 64 + hw];
        else
          convertedWeights[hw * Cin + c] = -1;
      }
    
    sycl_queue_.memcpy(weights_, convertedWeights.data(), used_size_ * sizeof(short));
    
  } else {
    
     sycl_queue_.memcpy(weights_, cpuWeight, weight_size);
    
  }

  sycl_queue_.wait();
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(
    int N, DataType* output_tensor, const DataType* input_tensor,
    const DataType* /*input2*/, void* /*scratch*/, size_t /*scratch_size*/) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  if (attention_map_) inputSize = used_size_;
  int outputSize = this->C * this->H * this->W;

  PolicyMap(N, output_tensor, input_tensor, weights_, inputSize, used_size_, outputSize, sycl_queue_);
}

template <typename DataType> PolicyMapLayer<DataType>::~PolicyMapLayer() {
  
  
  free(weights_, sycl_queue_);
  
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::FusedWinogradConvSELayer(
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
    ActivationFunction activation, bool bias, bool skip_add, bool se, int se_k,
    bool use_gemm_ex, sycl::queue &sycl_queue, bool op_nhcw)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex, sycl_queue),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      op_nhcw_(op_nhcw) {
  if (act_ != RELU && act_ != MISH && act_ != NONE) {
    throw Exception("Unsupported activation for fused winograd conv SE layer.");
  }

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 3 * 3;

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    biases_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  }

  // 6x6 transformed filter size, for 3x3 convolution
  transformed_weights_ = (DataType *)sycl::malloc_device(weight_size * 4, sycl_queue_);

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    w1_ = (DataType *)sycl::malloc_device(weight_size1 * 4, sycl_queue_);
    w2_ = (DataType *)sycl::malloc_device(weight_size2 * 4, sycl_queue_);
    b1_ = (DataType *)sycl::malloc_device(biases_size1 * 4, sycl_queue_);
    b2_ = (DataType *)sycl::malloc_device(biases_size2 * 4, sycl_queue_);
  }
}

template <typename DataType> void FusedWinogradConvSELayer<DataType>::LoadWeights(float* pfilter,
                                                     float* pBias,
                                                     void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size + bias_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  //sycl_queue_.memcpy(scratch, pfilter, weight_size).wait_and_throw();
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3, sycl_queue_);

  if (pBias) {
    
    
    //sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    sycl_queue_.memcpy(scratch, pBias, bias_size);  

    float total = 0;
    for(int i = 0; i < C; i++)
      total = pBias[i] + total;

    copyTypeConverted((DataType*)biases_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights_, weights, sycl_queue_);
}value

// TODO: Do this on the GPU to improve network load time!
static inline void CpuTranspose(float* op, float* ip, size_t rows,
                                size_t cols) {
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadSEWeights(float* w1, float* b1,
                                                       float* w2, float* b2,
                                                       void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  //sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float)).wait();
  
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, sycl_queue_);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);

  //sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, sycl_queue_);

  //sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, sycl_queue_);

  //sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, sycl_queue_);

  

}

 template <>
 void BaseLayer<sycl::half>::cublasRowMajorMatrixMul(const sycl::half* A, const sycl::half* B,
                                               sycl::half* Out, int M, int N, int K,
                                               int batchSize) {
   
  
   

   // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
   // float to sycl::half type doesn't work before CUDA 10.0
   #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

   // dimensions of matrix A = M x K
   // dimensions of matrix B = K x N
   // dimensions of output   = M x N

   // cublas supports only col major output
   // to multiply row major matrices, use the trick below
  
  #ifdef USE_CUBLAS
   cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  
   sycl_queue_.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
         cgh.host_task([=](sycl::interop_handle ih) {
  
          auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
          cublasSetStream(handle, cudaStreamHandle);

          ReportCUBLASErrors(cublasGemmStridedBatchedEx(
             handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one_h, B, CUDA_R_16F, N,
             N * K, A, CUDA_R_16F, K, K * M, &zero_h, Out, CUDA_R_16F, N, N * M,
             batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));

          
           cudaStreamSynchronize(cudaStreamHandle);
        
         });   
   });
  
  #endif  

 }

template <> void BaseLayer<float>::cublasRowMajorMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;

  int64_t M_ = M;
  int64_t N_ = N;
  int64_t K_ = K;

  #ifdef USE_CUBLAS
  //static cublasHandle_t handle;
  //ReportCUBLASErrors(cublasCreate(&handle)); 
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  #endif

  #ifdef USE_HIPBLAS
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  #endif

  if (use_gemm_ex_) {

   // printf("use_gemm_ex_\n");
    #ifdef USE_CUBLAS
    sycl_queue_.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        cgh.host_task([=](sycl::interop_handle ih) {
            auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
            cublasSetStream(handle, cudaStreamHandle);   

          ReportCUBLASErrors(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
            N * K, A, CUDA_R_32F, K, K * M, &floatZero, Out, CUDA_R_32F, N, N * M,
          batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

          
          cudaStreamSynchronize(cudaStreamHandle);

        });
    });
    #elifdef USE_HIPBLAS
    sycl_queue_.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        cgh.host_task([=](sycl::interop_handle ih) {
            auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue_);
            hipblasSetStream(handle, hipStreamHandle);   

          hipblasGemmStridedBatchedEx(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &floatOne, B, HIPBLAS_R_32F, N,
            N * K, A, HIPBLAS_R_32F, K, K * M, &floatZero, Out, HIPBLAS_R_32F, N, N * M,
          batchSize, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);

          
          hipStreamSynchronize(hipStreamHandle);

        });
    });  
    #else
        
      oneapi::mkl::blas::column_major::gemm_batch(sycl_queue_, oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans, N_, M_, K_, floatOne, B, N_, N_ * K_, A, K_, K_ * M_, floatZero, Out, N_, N_ * M_, batchSize);

      

    #endif
  }
  


  else {

    #ifdef USE_CUBLAS
     sycl_queue_.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        cgh.host_task([=](sycl::interop_handle ih) {
            auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
            cublasSetStream(handle, cudaStreamHandle); 

          // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
          ReportCUBLASErrors(cublasSgemmStridedBatched( handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
              K * M, &floatZero, Out, N, N * M, batchSize));

          
          cudaStreamSynchronize(cudaStreamHandle);

        });
     });
     #elifdef USE_HIPBLAS
     sycl_queue_.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        cgh.host_task([=](sycl::interop_handle ih) {
            auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue_);
            hipblasSetStream(handle, hipStreamHandle); 

          // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
          hipblasSgemmStridedBatched( handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
              K * M, &floatZero, Out, N, N * M, batchSize);

          
          hipStreamSynchronize(hipStreamHandle);

        });
     });
     #else
      oneapi::mkl::blas::column_major::gemm_batch(sycl_queue_, oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans, N_, M_, K_, floatOne, B, N_, N_ * K_, A, K_, K_ * M_, floatZero, Out, N_, N_ * M_, batchSize);

       

        

      #endif
     

  }

 
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size) {
  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType, false>(N, c_input_, transformed_input, input, sycl_queue_);
  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights_, transformed_output, N * 4, C, c_input_, 36);

  if (act_ == NONE) {
    if (!has_se_ && use_bias_ && !skip_add_)
      OutputTransform<DataType, false, NONE, true, false, false, false>(
          N, C, 0, output, transformed_output, nullptr, biases_, nullptr, nullptr, nullptr, nullptr, sycl_queue_);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == RELU) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, RELU, true, true, false, false>(
          N, C, se_k_, output, transformed_output, input2, biases_, w1_, b1_,
          w2_, b2_, sycl_queue_);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, RELU, true, false, false, true>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue_);
      else
        OutputTransform<DataType, false, RELU, true, false, false, false>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue_);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, RELU, true, true, false, false>(
          N, C, 0, output, transformed_output, input2, biases_, nullptr,
          nullptr, nullptr, nullptr, sycl_queue_);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == MISH) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, MISH, true, true, false, false>(
          N, C, se_k_, output, transformed_output, input2, biases_, w1_, b1_,
          w2_, b2_, sycl_queue_);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, MISH, true, false, false, true>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue_);
      else
        OutputTransform<DataType, false, MISH, true, false, false, false>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue_);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, MISH, true, true, false, false>(
          N, C, 0, output, transformed_output, input2, biases_, nullptr,
          nullptr, nullptr, nullptr, sycl_queue_);
    else
      throw Exception("unsupported network type!");
  } else
    throw Exception("unsupported network type!");
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::~FusedWinogradConvSELayer() {
  
  free(transformed_weights_, sycl_queue_);
  
  if (use_bias_) 
    free(biases_, sycl_queue_);
  
  if (has_se_) {
    free(w1_, sycl_queue_);
    free(w2_, sycl_queue_);
    free(b1_, sycl_queue_);
    free(b2_, sycl_queue_);
  }
}

template <typename DataType>
Conv1Layer<DataType>::Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W,
                                 int Cin, ActivationFunction activation,
                                 bool bias, bool use_gemm_ex, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex, sycl_queue),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias) {

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 1 * 1;
  weights_ = (DataType *)sycl::malloc_device(weight_size, sycl_queue_);

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    biases_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  }
}

template <typename DataType> void Conv1Layer<DataType>::LoadWeights(float* pfilter, float* pBias, void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 1 * 1;
  const size_t bias_size = sizeof(float) * C;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel

  assert(scratch);

  
  
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1, sycl_queue_);

  if (pBias) {
    
    
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, sycl_queue_);
  }
}

 template <>
  void Conv1Layer<sycl::half>::cublasSpecialMatrixMul(const sycl::half* A, const sycl::half* B,
                                               sycl::half* Out, int M, int N, int K,
                                               int batchSize) {

   // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
  // float to sycl::half type doesn't work before CUDA 10.0

  #ifdef USE_CUBLAS
   cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  #endif

  #ifdef USE_HIPBLAS
   hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  #endif


   #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

   // dimensions of matrix A = M x K
   // dimensions of matrix B = K x N
   // dimensions of output   = M x N

   // cublas supports only col major output
   // to multiply row major matrices, use the trick below
   // NOTE strideB set to 0 below!

   sycl_queue_.submit([&](sycl::handler &cgh) {
         
         cgh.host_task([=](sycl::interop_handle ih) {
  
          auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
          cublasSetStream(handle, cudaStreamHandle);


         ReportCUBLASErrors(cublasGemmStridedBatchedEx(
         handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one_h, B, CUDA_R_16F, N,
         N * K, A, CUDA_R_16F, K, 0, &zero_h, Out, CUDA_R_16F, N, N * M,
         batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));

         cudaStreamSynchronize(cudaStreamHandle);
        
         });   
   });
}

template <>
void Conv1Layer<float>::cublasSpecialMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;


  int64_t M_ = M;
  int64_t N_ = N;
  int64_t K_ = K;

  #ifdef USE_CUBLAS
   cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  #endif

  #ifdef USE_HIPBLAS
   hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  #endif


  // NOTE strideB set to 0 below!
  if (use_gemm_ex_){

   
    #ifdef USE_CUBLAS
    sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {
  
         auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
         cublasSetStream(handle, cudaStreamHandle);


        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
          N * K, A, CUDA_R_32F, K, 0, &floatZero, Out, CUDA_R_32F, N, N * M,
          batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

         cudaStreamSynchronize(cudaStreamHandle);

        });   
    });
    #elifdef USE_HIPBLAS
    sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {
  
         auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue_);
         hipblasSetStream(handle, hipStreamHandle);


        hipblasGemmStridedBatchedEx(
          handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &floatOne, B, HIPBLAS_R_32F, N,
          N * K, A, HIPBLAS_R_32F, K, 0, &floatZero, Out, HIPBLAS_R_32F, N, N * M,
          batchSize, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);

         hipStreamSynchronize(hipStreamHandle);

        });   
    });
    #else
      oneapi::mkl::blas::column_major::gemm_batch(
        sycl_queue_, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, N_, M_, K_, floatOne, B, N_, N_ * K_, A, K_,
        0, floatZero, Out, N_, N_ * M_, batchSize); 
    #endif

        
  } else {
    

    #ifdef USE_CUBLAS
    sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {
  
         auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue_);
         cublasSetStream(handle, cudaStreamHandle);
    
        // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
        ReportCUBLASErrors(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
        0, &floatZero, Out, N, N * M, batchSize));

        cudaStreamSynchronize(cudaStreamHandle);

        });
    });
    #elifdef USE_HIPBLAS
    sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {
  
         auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue_);
         hipblasSetStream(handle, hipStreamHandle);
    
        // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
        hipblasSgemmStridedBatched(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K, 0, &floatZero, Out, N, N * M, batchSize);

        hipStreamSynchronize(hipStreamHandle);

        });
    });
    #else
      oneapi::mkl::blas::column_major::gemm_batch(
        sycl_queue_, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, N_, M_, K_, floatOne, B, N_, N_ * K_, A, K_,
        0, floatZero, Out, N_, N_ * M_, batchSize);    
      
     
    #endif

  }
}

template <typename DataType>
void Conv1Layer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                const DataType* /*input2*/, void* /*scratch*/,
                                size_t /*scratch_size*/) {

  cublasSpecialMatrixMul(weights_, input, output, C, H * W, c_input_, N);

  if (use_bias_)
    addBias_NCHW(output, output, biases_, N, C, H, W, act_, sycl_queue_);
  else if (act_ != NONE)
    addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W, 0, act_, sycl_queue_);
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
 
  free(weights_, sycl_queue_);
  if (use_bias_) 
    free(biases_, sycl_queue_);
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(BaseLayer<DataType>* ip, int C, bool se,
                                       int se_k, bool use_gemm_ex, bool first,
                                       bool last, ActivationFunction activation,
                                       int shared_mem_size, sycl::queue& sycl_queue) : BaseLayer<DataType>(C, 8, 8, ip, ip->isNHWC(), use_gemm_ex, sycl_queue),
      has_se_(se),
      se_k_(se_k),
      c_input_(C),
      first_block_(first),
      last_block_(last),
      shared_mem_size_(shared_mem_size),
      act_(activation) {

  if (act_ != RELU && act_ != MISH) {
    throw Exception("Unsupported activation for residual block.");
  }

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * C * C * 3 * 3;

  const size_t bias_size = sizeof(DataType) * C;
  biases0_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  biases1_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);

  // 6x6 transformed filter size, for 3x3 convolution
  transformed_weights0_ = (DataType *)sycl::malloc_device(weight_size * 4, sycl_queue_);
  transformed_weights1_ = (DataType *)sycl::malloc_device(weight_size * 4, sycl_queue_);  


  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;


    w1_ = (DataType *)sycl::malloc_device(weight_size1, sycl_queue_);
    w2_ = (DataType *)sycl::malloc_device(weight_size2, sycl_queue_);
    b1_ = (DataType *)sycl::malloc_device(biases_size1, sycl_queue_);
    b2_ = (DataType *)sycl::malloc_device(biases_size2, sycl_queue_);

  }
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights0(float* pfilter, float* pBias,
                                           void* scratch) {

  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();

  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3, sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    copyTypeConverted((DataType*)biases0_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights0_, weights, sycl_queue_);
}

template <typename DataType> void ResidualBlock<DataType>::LoadWeights1(float* pfilter, float* pBias, void* scratch) {
  
  const size_t weight_size = sizeof(float) * C * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();

  copyTypeConverted((DataType*)weights, (float*)scratch, C * C * 3 * 3, sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size);
    copyTypeConverted((DataType*)biases1_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, C, transformed_weights1_, weights, sycl_queue_);
}

template <typename DataType> void ResidualBlock<DataType>::LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch) {

  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, sycl_queue_);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float)).wait(); 
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, sycl_queue_);

  
  sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, sycl_queue_);

  
  sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, sycl_queue_);
}

template <typename DataType>
void ResidualBlock<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* /*input2*/, void* scratch,
                                   size_t scratch_size) {
  // normally:
  // - "output" initially contains the transformed input,
  //    and after this layer, it contains the transformed input for next layer
  // - "input" contains the original/untransformed input
  // special cases:
  //   - for first_block_, input is real input (untransformed)
  //   - for last_block_, output is the final output of this block
  //   (untransformed)

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  if (first_block_) {
    InputTransform<DataType, true>(N, c_input_, transformed_input, input, sycl_queue_);

    BaseLayer<DataType>::cublasRowMajorMatrixMul(
        transformed_input, transformed_weights0_, transformed_output, N * 4, C,
        c_input_, 36);
  } else {
    BaseLayer<DataType>::cublasRowMajorMatrixMul(output, transformed_weights0_,
                                                 transformed_output, N * 4, C,
                                                 c_input_, 36);
  }

  if (act_ == RELU) {
    OutputInputTransform<DataType, false, RELU, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, sycl_queue_);
  } else if (act_ == MISH) {
    OutputInputTransform<DataType, false, MISH, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, sycl_queue_);
  }
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights1_, transformed_output, N * 4, C, C,
      36);

  const bool fp16 = std::is_same<sycl::half, DataType>::value;
  bool allowFusing =
      (C <= kMaxResBlockFusingChannels) ||
      (fp16 && (shared_mem_size_ >= kMaxResBlockFusingSeFp16AmpereSmem) &&
       (C <= kMaxResBlockFusingSeKFp16Ampere));

  if (act_ == RELU) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, RELU, true, true, true, false>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue_);
      else
        OutputTransform<DataType, false, RELU, true, true, true, false>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue_);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, RELU, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, sycl_queue_);
        } else {
          OutputTransform<DataType, true, RELU, true, true, true, true>(
              N, C, se_k_, (DataType*)input, transformed_output, input,
              biases1_, w1_, b1_, w2_, b2_, sycl_queue_);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         sycl_queue_);
        }
      } else
        OutputInputTransform<DataType, false, RELU, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue_);
    }
  } else if (act_ == MISH) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, MISH, true, true, true, false>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue_);
      else
        OutputTransform<DataType, false, MISH, true, true, true, false>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue_);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, MISH, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, sycl_queue_);
        } else {
          OutputTransform<DataType, true, MISH, true, true, true, true>(
              N, C, se_k_, (DataType*)input, transformed_output, input,
              biases1_, w1_, b1_, w2_, b2_, sycl_queue_);
          InputTransform<DataType, true>(N, C, output, (DataType*)input, sycl_queue_);
        }
      } else
        OutputInputTransform<DataType, false, MISH, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue_);
    }
  }
  // "output" tensor now contains transformed input for the next
  // convolution
}

template <typename DataType>
ResidualBlock<DataType>::~ResidualBlock() {

  free(transformed_weights0_, sycl_queue_);
  free(biases0_, sycl_queue_);
  free(transformed_weights1_, sycl_queue_);
  free(biases1_, sycl_queue_);
  if (has_se_) {
    free(w1_, sycl_queue_);
    free(w2_, sycl_queue_);
    free(b1_, sycl_queue_);
    free(b2_, sycl_queue_);
  }
}

template <typename DataType>
void allocAndUpload(DataType** gpu_dest, std::vector<float> cpu_src,
                    void* scratch, sycl::queue &sycl_queue) {

  //TODO: Take a look at this.                    
  size_t size = cpu_src.size() * sizeof(DataType);
  if (size == 0) {
    *gpu_dest = nullptr;
    return;
  }
  
  gpu_dest = (DataType **)sycl::malloc_device(size, sycl_queue);

  sycl_queue.memcpy(scratch, &cpu_src[0], cpu_src.size() * sizeof(float)).wait();

  copyTypeConverted((DataType*)(*gpu_dest), (float*)scratch,
                    (int)cpu_src.size(), sycl_queue);
}

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(BaseLayer<DataType>* ip,
                                                   const LegacyWeights& weights,
                                                   void* scratch, sycl::queue &sycl_queue)
    : BaseLayer<DataType>(64 * 64 + 24 * 8, 1, 1, ip, sycl_queue) {


  embedding_op_size_ = weights.ip_pol_b.size();
  wq_op_size_ = weights.ip2_pol_b.size();
  wk_op_size_ = weights.ip3_pol_b.size();

  encoder_heads_ = weights.pol_encoder_head_count;
  policy_d_model_ = wq_op_size_;

  allocAndUpload<DataType>(&ip_pol_w_, weights.ip_pol_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip_pol_b_, weights.ip_pol_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ip2_pol_w_, weights.ip2_pol_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip2_pol_b_, weights.ip2_pol_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ip3_pol_w_, weights.ip3_pol_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip3_pol_b_, weights.ip3_pol_b, scratch, sycl_queue_);

  // big allocation to hold wq and wk weights one after the other
  {
    size_t elements = weights.ip2_pol_w.size();
    assert(elements == weights.ip3_pol_w.size());

    size_t size = elements * sizeof(DataType) * 2;
    wqk_w_ = (DataType *)sycl::malloc_device(size, sycl_queue_);

    sycl_queue_.memcpy(wqk_w_, ip2_pol_w_, size / 2).wait();
    sycl_queue_.memcpy(wqk_w_ + elements, ip3_pol_w_, size / 2).wait();


    elements = weights.ip2_pol_b.size();
    size = elements * sizeof(DataType) * 2;
    wqk_b_ = (DataType *)sycl::malloc_device(size, sycl_queue_);

    sycl_queue_.memcpy(wqk_b_, ip2_pol_b_, size / 2).wait();
    sycl_queue_.memcpy(wqk_b_ + elements, ip3_pol_b_, size / 2).wait();
  }

  allocAndUpload<DataType>(&ip4_pol_w_, weights.ip4_pol_w, scratch, sycl_queue_);

  for (const auto& enc : weights.pol_encoder) {
    EncoderWeights* pW = new EncoderWeights(enc, scratch, sycl_queue_);
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
AttentionPolicyHead<DataType>::EncoderWeights::EncoderWeights(
    const LegacyWeights::EncoderLayer& cpu_weights, void* scratch, sycl::queue &sycl_queue): 
    sycl_queue_(sycl_queue) 
{
  
  mha_q_size_ = cpu_weights.mha.q_b.size();
  mha_k_size_ = cpu_weights.mha.k_b.size();
  mha_v_size_ = cpu_weights.mha.v_b.size();
  mha_dense_size_ = cpu_weights.mha.dense_b.size();
  ffn_dense1_size_ = cpu_weights.ffn.dense1_b.size();
  ffn_dense2_size_ = cpu_weights.ffn.dense2_b.size();

  allocAndUpload<DataType>(&mha_q_w, cpu_weights.mha.q_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_q_b, cpu_weights.mha.q_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&mha_k_w, cpu_weights.mha.k_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_k_b, cpu_weights.mha.k_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&mha_v_w, cpu_weights.mha.v_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_v_b, cpu_weights.mha.v_b, scratch, sycl_queue_);

  // big allocation to hold qkv weights one after the other
  {
    size_t elements = cpu_weights.mha.q_w.size();
    size_t size = elements * sizeof(DataType) * 3;
    
    mha_qkv_w = (DataType *)sycl::malloc_device(size, sycl_queue_);

    sycl_queue_.memcpy(mha_qkv_w, mha_q_w, size / 3).wait();
    sycl_queue_.memcpy(mha_qkv_w + elements, mha_k_w, size / 3).wait();
    sycl_queue_.memcpy(mha_qkv_w + elements * 2, mha_v_w, size / 3).wait();

    elements = cpu_weights.mha.q_b.size();
    size = elements * sizeof(DataType) * 3;
    mha_qkv_b = (DataType *)sycl::malloc_device(size, sycl_queue_);
    
    sycl_queue_.memcpy(mha_qkv_b, mha_q_b, size / 3).wait();
    sycl_queue_.memcpy(mha_qkv_b + elements, mha_k_b, size / 3).wait();
    sycl_queue_.memcpy(mha_qkv_b + elements * 2, mha_v_b, size / 3).wait();
  }

  allocAndUpload<DataType>(&mha_dense_w, cpu_weights.mha.dense_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_dense_b, cpu_weights.mha.dense_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ln1_gammas, cpu_weights.ln1_gammas, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ln1_betas, cpu_weights.ln1_betas, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ffn_dense1_w, cpu_weights.ffn.dense1_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ffn_dense1_b, cpu_weights.ffn.dense1_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ffn_dense2_w, cpu_weights.ffn.dense2_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ffn_dense2_b, cpu_weights.ffn.dense2_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ln2_gammas, cpu_weights.ln2_gammas, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ln2_betas, cpu_weights.ln2_betas, scratch, sycl_queue_);
}

template <typename DataType>
static void cublasXgemm(transpose_type transa,
                        transpose_type transb, int m, int n, int k,
                        float alpha, const DataType* A, int lda,
                        const DataType* B, int ldb, float beta, DataType* C,
                        int ldc, sycl::queue &sycl_queue) {



  const bool fp16 = std::is_same<sycl::half, DataType>::value;
  
  #ifdef USE_CUBLAS

  
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);


    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue);
        cublasSetStream(handle, cudaStreamHandle);    


        ReportCUBLASErrors(cublasHgemm(
          handle, transa, transb, m, n, k, (const half*)&alpha_h, ((const half *)A),
          lda, ((const half *)B), ldb, (const half*)&beta_h, ((half *)C), ldc));

         cudaStreamSynchronize(cudaStreamHandle); 

        });
    });

  } else { 

    

    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {  

        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue);
        cublasSetStream(handle, cudaStreamHandle);  

        ReportCUBLASErrors(cublasSgemm(handle, transa, transb, m, n, k, &alpha,
                                   (const float*)A, lda, (const float*)B, ldb,
                                   &beta, (float*)C, ldc));

         cudaStreamSynchronize(cudaStreamHandle);

        });
      });
  }
    #elifdef USE_HIPBLAS

    hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {  

        auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue);
        hipblasSetStream(handle, hipStreamHandle);  

        hipblasSgemm(handle, transa, transb, m, n, k, &alpha, (const float*)A, lda, (const float*)B, ldb, &beta, (float*)C, ldc);

         hipStreamSynchronize(hipStreamHandle);

        });
      });

    #else
      oneapi::mkl::blas::column_major::gemm(sycl_queue, transa, transb, m, n, k, alpha, (const float *)A, lda,
        (const float *)B, ldb, beta, (float *)C, ldc);

       
      
    #endif

}

template <typename DataType>
static void cublasXGemmStridedBatched(transpose_type transa, transpose_type transb,
    int m, int n, int k, float alpha, const void* A, int lda,
    long long int strideA, const void* B, int ldb, long long int strideB,
    float beta, void* C, int ldc, long long int strideC, int batchCount, sycl::queue &sycl_queue) {

  const bool fp16 = std::is_same<sycl::half, DataType>::value;
  
  
  #ifdef USE_CUBLAS
  
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);


    
    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {
    
        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue);
        cublasSetStream(handle, cudaStreamHandle);    

        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, transa, transb, m, n, k, &alpha_h, A, CUDA_R_16F, lda, strideA,
          B, CUDA_R_16F, ldb, strideB, &beta_h, C, CUDA_R_16F, ldc, strideC,
          batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
        
        cudaStreamSynchronize(cudaStreamHandle);

      });

    });
  
  } else { 
    
    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.host_task([=](sycl::interop_handle ih) {
    
        auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue);
        cublasSetStream(handle, cudaStreamHandle);    
    
        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
        CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC,
        batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  
        cudaStreamSynchronize(cudaStreamHandle);
  
      });
    });
  }
    
  #elifdef USE_HIPBLAS
    hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

     sycl_queue.submit([&](sycl::handler &cgh) {

        cgh.host_task([=](sycl::interop_handle ih) {
    
        auto hipStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_hip>(sycl_queue);
        hipblasSetStream(handle, hipStreamHandle);    
    
        hipblasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha, A, HIPBLAS_R_32F, lda, strideA, B,
        HIPBLAS_R_32F, ldb, strideB, &beta, C, HIPBLAS_R_32F, ldc, strideC,
        batchCount, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
  
        hipStreamSynchronize(hipStreamHandle);
  
      });
    });


    #else

      oneapi::mkl::blas::column_major::gemm_batch(sycl_queue, transa, transb, m, n, k,  alpha, (const float *)A, lda, strideA, (const float *)B, ldb, strideB, beta, (float *)C, ldc, strideC, batchCount); 
     
   #endif
  
}


template <typename DataType>
void AttentionPolicyHead<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size) {


  DataType* scratch0 = (DataType*)scratch;
  DataType* scratch1 = (DataType*)input2;
  DataType* scratch2 = output + scratch_size / (2 * sizeof(DataType));
  DataType* scratch3 = scratch1 + scratch_size / (2 * sizeof(DataType));

  int inputC = this->input_->GetC();
  convertNCHWtoNHWC(scratch0, input, N, inputC, N, inputC, 8, 8, sycl_queue_);

  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
  DataType* pol_embedding = scratch1;
  {
    const int num_outputs = embedding_op_size_;
    const int num_inputs = inputC;
    const int batch = N * 64;
    cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip_pol_w_,
                          num_inputs, scratch0, num_inputs, 0.0f, pol_embedding,
                          num_outputs, sycl_queue_);
    addBiasBatched(pol_embedding, pol_embedding, ip_pol_b_, 1, batch,
                   num_outputs, SELU, sycl_queue_);
  }

  // 2. Encoder layers
  for (const auto pEnc : encoder_weights_) {
    const auto& enc = *pEnc;
    const int d_model = enc.mha_q_size_;
    const int depth = d_model / encoder_heads_;

    DataType* mha_q;
    DataType* mha_k;
    DataType* mha_v;

    {
      const int num_inputs = embedding_op_size_;
      const int num_outputs = d_model;
      const int batch = N * 64;

      mha_q = scratch0;
      mha_k = mha_q + num_outputs * batch;
      mha_v = mha_k + num_outputs * batch;

      cublasXGemmStridedBatched<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch, num_inputs,
          1.0f, enc.mha_qkv_w, num_inputs, num_inputs * num_outputs,
          pol_embedding, num_inputs, 0, 0.0f, mha_q, num_outputs,
          num_outputs * batch, 3, sycl_queue_);
      addBiasBatched<DataType>(mha_q, mha_q, enc.mha_qkv_b, 3, batch,
                               num_outputs, NONE, sycl_queue_);
    }

    // Apply split_heads() to q, k and v
    // which basically transposes (batch_size, 64, num_heads, depth)
    // to (batch_size, num_heads, 64, depth)
    // Do we really need to transpose here?
    // (Maybe not, we can play with strides of the gemm and do independent gemms
    // for each encoder head)

    // Apply scaled dot product attention:
    /*
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
    */

    // shape(k)[-1] = depth
    float factor = 1.0f / sqrt((float)depth);

    // matmul_qk = tf.matmul(q, k, transpose_b=True)
    for (int i = 0; i < encoder_heads_; i++) {
      int offset = i * depth;
      // layout of the output: encoder_heads_ * Batch * 64 * 64
      int outOffset = i * N * 64 * 64;
      cublasXGemmStridedBatched<DataType>(transpose_type_transpose, transpose_type_notranspose, 64 /*M*/, 64 /*N*/,
          depth /*K*/,  // A/B, and M/N are swapped for row-major to col-major
                        // transform
          factor,       // to handle "/ tf.math.sqrt(dk)"
          mha_k + offset /*A*/,
          d_model /*LDA*/,  // (d_model = depth * encoder_heads_) to skip over
                            // other "depth" slices / heads
          64 * d_model,     /*strideA*/
          mha_q + offset /*B*/,
          d_model /*LDB*/,  // to skip over other other "depth" slices / heads
          64 * d_model,     /*strideB*/
          0.0f,
          scratch2 + outOffset /*C*/,  // output (matmul_qk) goes to scratch2
          64 /*LDC*/, 64 * 64 /*strideC*/, N, sycl_queue_);
    }

    // attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
    // attention_weights -> scratch2
    Softmax(encoder_heads_ * N * 64, 64, scratch2, scratch2, sycl_queue_);

    // output = tf.matmul(attention_weights, v)
    for (int i = 0; i < encoder_heads_; i++) {
      int offset = i * depth;  // for output and "v" matrix
      // layout: encoder_heads_ * Batch*64*64
      int weightsOffset = i * N * 64 * 64;
      cublasXGemmStridedBatched<DataType>(transpose_type_transpose, transpose_type_notranspose, depth /*M*/, 64 /*N*/, 64 /*K*/,
          1.0f, mha_v + offset /*A*/,  // "v" matrix
          d_model /*LDA*/,  // to skip over other "depth" slices / heads
          64 * d_model,     /*strideA*/
          scratch2 + weightsOffset /*B*/, 64 /*LDB*/, 64 * 64, /*strideB*/
          0.0f, scratch3 + offset /*C*/,  // output goes to scratch3
          d_model /*LDC*/, 64 * d_model /*strideC*/, N, sycl_queue_);
    }

    // #final dense layer (mha_dense), scratch3 -> scratch2
    {
      const int num_inputs = d_model;
      const int num_outputs = embedding_op_size_;
      const int batch = N * 64;

      cublasXgemm(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)enc.mha_dense_w,
                  num_inputs, scratch3, num_inputs, 0.0f, scratch2,
                  num_outputs, sycl_queue_);
    }

    // LN1: skip connection and layer normalization (also bias add of prev gemm)
    // scratch2/scratch1 -> scratch0
    LayerNorm<DataType>(N * 64, embedding_op_size_, scratch0, scratch2,
                        enc.mha_dense_b, scratch1, enc.ln1_gammas,
                        enc.ln1_betas, 1e-6, sycl_queue_);

    // #FFN dense 1, scratch0 -> scratch1
    const int encoder_dff = enc.ffn_dense1_size_;
    {
      const int num_inputs = embedding_op_size_;
      const int num_outputs = encoder_dff;
      const int batch = N * 64;
      cublasXgemm(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)enc.ffn_dense1_w,
                  num_inputs, scratch0, num_inputs, 0.0f, scratch1,
                  num_outputs, sycl_queue_);
      addBiasBatched(scratch1, scratch1, enc.ffn_dense1_b, 1, batch,
                     num_outputs, SELU, sycl_queue_);
    }

    // #FFN dense 2, scratch1 -> scratch2
    {
      const int num_inputs = encoder_dff;
      const int num_outputs = embedding_op_size_;
      const int batch = N * 64;
      cublasXgemm(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)enc.ffn_dense2_w,
                  num_inputs, scratch1, num_inputs, 0.0f, scratch2,
                  num_outputs, sycl_queue_);
    }

    // LN2: skip connection and layer normilization (also bias add of prev gemm)
    // scratch2/scratch0 -> scratch1
    LayerNorm<DataType>(N * 64, embedding_op_size_, scratch1, scratch2,
                        enc.ffn_dense2_b, scratch0, enc.ln2_gammas,
                        enc.ln2_betas, 1e-6, sycl_queue_);

  }  // End of encoder blocks

  DataType* wq;
  DataType* wk;
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = policy_d_model_;
    const int batch = N * 64;
    wq = scratch0;
    wk = wq + num_outputs * batch;

    cublasXGemmStridedBatched<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch, num_inputs, 1.0f,
        wqk_w_, num_inputs, num_inputs * num_outputs, scratch1, num_inputs, 0,
        0.0f, wq, num_outputs, num_outputs * batch, 2, sycl_queue_);

    addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs, NONE,
                             sycl_queue_);
  }

  // dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))
  // policy matmul_qk = tf.matmul(queries, keys, transpose_b=True)
  // policy_attn_logits = matmul_qk / dk
  {
    // shape(keys)[-1] = policy_d_model_
    float factor = 1.0f / sqrt((float)policy_d_model_);

    // A/B, and M/N are swapped for row-major to col-major transform
    // leave 8*24 after each batch to interleave promotion_logits (computed
    // later below)
    cublasXGemmStridedBatched<DataType>(transpose_type_transpose, transpose_type_notranspose, 64 /*M*/, 64 /*N*/,
        policy_d_model_ /*K*/,
        factor,  // to handle "/ tf.math.sqrt(dk)"
        wk /*A*/, policy_d_model_ /*LDA*/, 64 * policy_d_model_, /*strideA*/
        wq /*B*/, policy_d_model_ /*LDB*/, 64 * policy_d_model_, /*strideB*/
        0.0f, output /*C*/,  // output (policy_attn_logits)
        64 /*LDC*/, 64 * 64 + 8 * 24 /*strideC*/, N, sycl_queue_);
  }

  // Compute promotion_logits in a single kernel (and put the result just after
  // policy_attn_logits interleaved to get concat for free)
  DataType* promotion_logits = output + 64 * 64;

  ComputePromotionLogits<DataType>(N, policy_d_model_, promotion_logits, wk,
                                   ip4_pol_w_, output, sycl_queue_);
}

template <typename DataType>
AttentionPolicyHead<DataType>::~AttentionPolicyHead() {
 
  free(ip_pol_w_, sycl_queue_);
  free(ip_pol_b_, sycl_queue_);
  free(ip2_pol_w_, sycl_queue_);
  free(ip2_pol_b_, sycl_queue_);
  free(ip3_pol_w_, sycl_queue_);
  free(ip3_pol_b_, sycl_queue_);
  free(ip4_pol_w_, sycl_queue_);
  free(wqk_w_, sycl_queue_);
  free(wqk_b_, sycl_queue_);
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
AttentionPolicyHead<DataType>::EncoderWeights::~EncoderWeights() {

  free(mha_q_w, sycl_queue_);
  free(mha_q_b, sycl_queue_);
  free(mha_k_w, sycl_queue_);
  free(mha_k_b, sycl_queue_);
  free(mha_v_w, sycl_queue_);
  free(mha_v_b, sycl_queue_);
  free(mha_qkv_w, sycl_queue_);
  free(mha_qkv_b, sycl_queue_);
  free(mha_dense_w, sycl_queue_);
  free(mha_dense_b, sycl_queue_);
  free(ln1_gammas, sycl_queue_);
  free(ln1_betas, sycl_queue_);
  free(ffn_dense1_w, sycl_queue_);
  free(ffn_dense1_b, sycl_queue_);
  free(ffn_dense2_w, sycl_queue_);
  free(ffn_dense2_b, sycl_queue_);
  free(ln2_gammas, sycl_queue_);
  free(ln2_betas, sycl_queue_);
}

// Template instantiation.
#ifdef USE_CUDNN
//template class ConvLayer<sycl::half>;
template class ConvLayer<float>;
#endif

template class FCLayer<sycl::half>;
template class FCLayer<float>;

template class SELayer<sycl::half>;
template class SELayer<float>;

template class PolicyMapLayer<sycl::half>;
template class PolicyMapLayer<float>;

template class FusedWinogradConvSELayer<sycl::half>;
template class FusedWinogradConvSELayer<float>;

template class Conv1Layer<sycl::half>;
template class Conv1Layer<float>;

template class ResidualBlock<sycl::half>;
template class ResidualBlock<float>;

template class AttentionPolicyHead<sycl::half>;
template class AttentionPolicyHead<float>;

// Misc error handling stuff.
#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status, const char* file, const int& line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", cudnnGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}
#endif

const char* CublasGetErrorString(int status) {
  switch (status) {
    case 0:
      return "CUBLAS_STATUS_SUCCESS";
    case 1:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case 3:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case 7:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case 8:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case 11:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case 13:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case 14:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case 15:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case 16:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown cublas error";
}

void CublasError(int status, const char* file, const int& line) {
  if (status != 0) {
    char message[128];
    sprintf(message, "CUBLAS error: %s (%s:%d) ", CublasGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

void CudaError(int status, const char* file, const int& line) {
 
  if (status != 0) {
    char message[128];
    
    sprintf(
        message, "CUDA error: %s (%s:%d) ",
        "cudaGetErrorString is not supported" /*cudaGetErrorString(status)*/,
        file, line);
    throw Exception(message);
  }
}

}  // namespace sycldnn_backend
}  // namespace lczero
