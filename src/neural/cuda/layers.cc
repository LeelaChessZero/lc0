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
#include "layers.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "cuda_common.h"
#include "kernels.h"
#include "neural/network.h"
#include "neural/shared/activation.h"
#include "neural/shared/attention_policy_map.h"
#include "utils/fp16_utils.h"

namespace lczero {

#if 0
// debug code to dump allocation in GPU memory
template <typename T>
void dumpTensor(T* memory, int elements, const char* message, bool only_summary = false) {
    const bool fp16 = std::is_same<half, T>::value;
    printf("\n%s\n", message);
    int elementSize = (int) (fp16 ? sizeof(half) : sizeof(float));
    int bytes = elements * elementSize;
    void *temp = malloc(bytes);
    cudaMemcpy(temp, memory, bytes, cudaMemcpyDeviceToHost);
    float maxval = -std::numeric_limits<float>::max();
    float minval = std::numeric_limits<float>::max();
    int nans = 0;
    int nanss[10] {};

    for (int i = 0; i < elements; i++)
    {
        float val;
        if (fp16) 
        {
            half *arr = (half*)temp;
            val = (float)arr[i];
        }
        else
        {
            float *arr = (float *)temp;
            val = arr[i];
        }
        maxval = std::max(maxval, val);
        minval = std::min(minval, val);

        if (std::isnan(val)) {
          if (nans < 10) nanss[nans] = i;
          nans++;
        }

        if (!only_summary || i < 2 || i == elements - 1) {
          // printf("%8.4f ", val);
          // if ((i % 8) == 7) printf("\n");
          printf("%i;%.6f\n", i, val);
        }
    }
    free(temp);
    if (maxval == -std::numeric_limits<float>::max())
       maxval = std::numeric_limits<double>::quiet_NaN();
    if (minval == std::numeric_limits<float>::max())
       minval = std::numeric_limits<double>::quiet_NaN();

    printf("Max: %.6f, Min: %.6f, NaNs: %i of %i", maxval, minval, nans, elements);
    printf("\nNaN indices: ");
    for (int i=0; i<nans && i<10; i++) printf("%i ", nanss[i]);
    if (nans > 10) printf("......");
    printf("\n");
}
#endif

namespace cudnn_backend {

// Use Single kernel for entire SE operation.
// Right now supported only for fp16 with nhwc and it's quite a bit faster
// than using multiple passes. The flag can be set to false for debugging.
static constexpr bool kUseFusedSELayer = true;

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(false) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc,
                               bool gemm_ex)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(gemm_ex) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip),
      C(c),
      H(h),
      W(w),
      nhwc_(ip ? ip->nhwc_ : false),
      use_gemm_ex_(false) {}

#ifdef USE_CUDNN
template <typename DataType>
void ConvLayer<DataType>::init() {
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size =
      sizeof(DataType) * c_input_ * C * filter_size_ * filter_size_;
  ReportCUDAErrors(cudaMalloc(&weights, weight_size));

  const size_t bias_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases, bias_size));

  const bool fp16 = std::is_same<half, DataType>::value;
  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  // Create cudnn objects for various tensors, algorithms, etc.
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
  cudnnCreateTensorDescriptor(&in_tensor_desc_);
  cudnnCreateTensorDescriptor(&bias_desc_);
  cudnnCreateActivationDescriptor(&activation_);

  cudnnSetFilter4dDescriptor(filter_desc_, dataType, layout, GetC(), c_input_,
                             filter_size_, filter_size_);

  ReportCUDNNErrors(
      cudnnSetTensor4dDescriptor(bias_desc_, layout, dataType, 1, C, 1, 1));

  const int padding = filter_size_ / 2;
  const bool crossCorr = 1;

  ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
      conv_desc_, padding, padding, 1, 1, 1, 1,
      crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION, dataType));

  if (fp16 && nhwc_)
    ReportCUDNNErrors(
        cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

  // TODO: dynamic selection of algorithm!
  if ((C > 32) && (!nhwc_) && (filter_size_ > 1)) {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  } else {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }

  if (act_ == ACTIVATION_RELU) {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_RELU,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
#if CUDNN_MAJOR != 7 || CUDNN_MINOR != 0
  else {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_IDENTITY,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
#endif
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                               int filter, int Cin,
                               ActivationFunction activation, bool bias)
    : BaseLayer<DataType>(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      act_(activation),
      use_bias_(bias) {
  init();
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(bool nhwc, int C, int H, int W, int filter,
                               int Cin, ActivationFunction activation,
                               bool bias)
    : BaseLayer<DataType>(C, H, W, nullptr, nhwc),
      c_input_(Cin),
      filter_size_(filter),
      act_(activation),
      use_bias_(bias) {
  init();
}

template <>
void ConvLayer<half>::LoadWeights(float* pfilter, float* pBias, void* scratch) {
  const size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  const size_t bias_size = sizeof(float) * C;
  // Also need to convert from fp32 NCHW to fp16 NHWC
  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type / layout conversion using a kernel.
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (nhwc_) {
    convertNCHWtoNHWC((half*)weights, (float*)scratch, C, c_input_, C, c_input_,
                      filter_size_, filter_size_);
  } else {
    copyTypeConverted((half*)weights, (float*)scratch,
                      C * c_input_ * filter_size_ * filter_size_, 0);
  }

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));

    copyTypeConverted((half*)biases, (float*)scratch, C, 0);
  }
}

template <>
void ConvLayer<float>::LoadWeights(float* pfilter, float* pBias,
                                   void* /*scratch*/) {
  const size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  const size_t bias_size = sizeof(float) * C;
  ReportCUDAErrors(
      cudaMemcpy(weights, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(biases, pBias, bias_size, cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(cudaMemset(biases, 0, bias_size));
  }
}

template <typename DataType>
void ConvLayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t cudnn,
                               cublasHandle_t /*cublas*/, cudaStream_t stream,
                               DataType***) {
  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_, layout,
                                               dataType, N, C, H, W));

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_, layout,
                                               dataType, N, c_input_, H, W));

  float alpha = 1.0f, beta = 0.0f;

  if (!(act_ != ACTIVATION_NONE || use_bias_ || input2)) {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
        output));
  }
#if CUDNN_MAJOR != 7 || CUDNN_MINOR != 0
  else if (input2 && (act_ == ACTIVATION_RELU || act_ == ACTIVATION_NONE) &&
           use_bias_) {
    // fused bias + sum + relu!
    ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &alpha, out_tensor_desc_,
        input2, bias_desc_, biases, activation_, out_tensor_desc_, output));
  } else {
    // For some reason cudnn doesn't support just Convolution + Bias with nchw
    // (winograd algorithm) it works fine when RELU is also needed which is
    // somewhat strange.
    if ((act_ == ACTIVATION_RELU || (act_ == ACTIVATION_NONE && nhwc_)) &&
        !input2 && use_bias_) {
      ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output, bias_desc_, biases, activation_,
          out_tensor_desc_, output));
    } else {
      // The no special case path...
      ReportCUDNNErrors(cudnnConvolutionForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output));
      bool act_done = false;
      if (input2 && input2 != output) {
        // Merge act with residual add unless there is bias.
        addVectors(output, output, (DataType*)input2, N * C * H * W,
                   N * C * H * W, N * C * H * W,
                   use_bias_ ? ACTIVATION_NONE : act_, stream);
        act_done = !use_bias_;
      }
      // Merge act with bias.
      if (use_bias_) {
        if (!nhwc_) {
          // add bias
          addBias_NCHW(output, output, biases, N, C, H, W, act_, stream);
        } else {
          addVectors(output, output, biases, N * C * H * W, N * C * H * W, C,
                     act_, stream);
        }
      } else if (!act_done && act_ != ACTIVATION_NONE) {
        addVectors(output, output, (DataType*)nullptr, N * C * H * W,
                   N * C * H * W, 0, act_, stream);
      }
    }
  }
#else
  else {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size,
        (input2 == output) ? &alpha : &beta, out_tensor_desc_, output));
    if (input2 && input2 != output) {
      ReportCUDNNErrors(cudnnAddTensor(cudnn, &alpha, out_tensor_desc_, input2,
                                       &alpha, out_tensor_desc_, output));
    }
    if (use_bias_) {
      ReportCUDNNErrors(cudnnAddTensor(cudnn, &alpha, bias_desc_, biases,
                                       &alpha, out_tensor_desc_, output));
    }
    if (act_ == ACTIVATION_RELU) {
      ReportCUDNNErrors(cudnnActivationForward(cudnn, activation_, &alpha,
                                               out_tensor_desc_, output, &beta,
                                               out_tensor_desc_, output));
    }
    if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_NONE) {
      addVectors(output, output, nullptr, N * C * H * W, N * C * H * W, 0, act_,
                 stream);
      // TODO: check this actually compiles?
    }
  }
#endif
}

template <typename DataType>
ConvLayer<DataType>::~ConvLayer() {
  ReportCUDAErrors(cudaFree(weights));
  ReportCUDAErrors(cudaFree(biases));

  cudnnDestroyFilterDescriptor(filter_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
  cudnnDestroyTensorDescriptor(bias_desc_);
  cudnnDestroyTensorDescriptor(in_tensor_desc_);
  cudnnDestroyTensorDescriptor(out_tensor_desc_);
  cudnnDestroyActivationDescriptor(activation_);
}
#endif

template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs,
                           bool addPrevLayerBias, ActivationFunction activation)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias),
      act_(activation) {
  ReportCUDAErrors(cudaMalloc(&w1_, C * numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&w2_, 2 * C * numFc1Out_ * sizeof(DataType)));

  if (kUseFusedSELayer && nhwc_) {
    ReportCUDAErrors(cudaMalloc(&w1_t_, C * numFc1Out_ * sizeof(DataType)));
    ReportCUDAErrors(cudaMalloc(&w2_t_, 2 * C * numFc1Out_ * sizeof(DataType)));
  }

  ReportCUDAErrors(cudaMalloc(&b1_, numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&b2_, 2 * C * sizeof(DataType)));

  ReportCUDAErrors(cudaMalloc(&bPrev_, C * sizeof(DataType)));
}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  ReportCUDAErrors(cudaFree(w1_));
  ReportCUDAErrors(cudaFree(w2_));
  ReportCUDAErrors(cudaFree(b1_));
  ReportCUDAErrors(cudaFree(b2_));
  ReportCUDAErrors(cudaFree(bPrev_));
}

template <>
void SELayer<float>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                 float* prevLayerBias, void* /*scratch*/) {
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t weight_size2 = 2 * weight_size1;

  // Weight for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(w1_, w1, weight_size1, cudaMemcpyHostToDevice));

  // Weight for the second FC layer.
  ReportCUDAErrors(cudaMemcpy(w2_, w2, weight_size2, cudaMemcpyHostToDevice));

  // Bias for the first FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b1_, b1, numFc1Out_ * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b2_, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(bPrev_, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
  }
}

void cpuTranspose(float* op, float* ip, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <>
void SELayer<half>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                float* prevLayerBias, void* scratch) {
  const size_t num_weights1 = C * numFc1Out_;
  size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t num_weights2 = 2 * num_weights1;
  size_t weight_size2 = 2 * weight_size1;

  // Transpose the weight matrices for the fused path.
  std::vector<float> temp(weight_size2);

  // Weight for the first FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, w1, weight_size1, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w1_, (float*)scratch, (int)num_weights1, 0);
  if (kUseFusedSELayer && nhwc_) {
    // transposed copy for fused SE kernel
    cpuTranspose(temp.data(), w1, numFc1Out_, C);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size1, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w1_t_, (float*)scratch, (int)num_weights1, 0);
  }

  // Weight for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, w2, weight_size2, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w2_, (float*)scratch, (int)num_weights2, 0);
  if (kUseFusedSELayer && nhwc_) {
    cpuTranspose(temp.data(), w2, 2 * C, numFc1Out_);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size2, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w2_t_, (float*)scratch, (int)num_weights2, 0);
  }

  // Bias for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(scratch, b1, numFc1Out_ * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b1_, (float*)scratch, numFc1Out_, 0);

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b2_, (float*)scratch, 2 * C, 0);

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(scratch, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
    copyTypeConverted((half*)bPrev_, (float*)scratch, C, 0);
  }
}

template <>
void SELayer<float>::Eval(int N, float* output, const float* input,
                          const float* /*input2*/, void* scratch,
                          size_t scratch_size, cudnnHandle_t /*cudnn*/,
                          cublasHandle_t cublas, cudaStream_t stream,
                          float***) {
  // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
  float* op1 = (float*)scratch;
  float* op2 = (float*)scratch + scratch_size / sizeof(float) / 2;

  // 1. Global avg pooling (also adds previous layer bias before computing
  // averages).
  globalAvgPool(N, C, op2, input, bPrev_, false);

  // 2. First fully connected layer.
  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                 N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                 numFc1Out_));
  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
             stream);

  // 3. Second fully connected layer.
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C));
  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
             stream);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  globalScale(N, C, output, input, op2, bPrev_, false, act_);
}

template <>
void SELayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                         cudaStream_t stream, half***) {
  bool se_done = false;
  if (kUseFusedSELayer && nhwc_) {
    se_done = Se_Fp16_NHWC(N, C, numFc1Out_, output, input2, input, w1_t_, b1_,
                           w2_t_, b2_, bPrev_, act_);
  }
  if (!se_done) {
    assert(output == input2);
    // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
    half* op1 = (half*)scratch;
    half* op2 = (half*)scratch + scratch_size / sizeof(half) / 2;

    // 1. Global avg pooling (also adds previous layer bias before computing
    // averages).
    globalAvgPool(N, C, op2, input, bPrev_, nhwc_);

    // 2. First fully connected layer.
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                   N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                   numFc1Out_));
    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
               stream);

    // 3. Second fully connected layer.
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                   numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                   numFc1Out_, &beta, op2, 2 * C));
    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
               stream);

    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    globalScale(N, C, output, input, op2, bPrev_, nhwc_, act_);
  }
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation)
    : BaseLayer<DataType>(C, H, W, ip), use_bias_(bias), act_(activation) {
  const size_t weight_size =
      sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(DataType) * C * H * W;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
  if (use_bias_) {
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  } else {
    biases_ = nullptr;
  }
}

template <>
void FCLayer<half>::LoadWeights(float* cpuWeight, float* cpuBias,
                                void* scratch) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  // also need to convert from fp32 to fp16
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, cpuWeight, weight_size, cudaMemcpyHostToDevice));

  if (nhwc_) {
    convertNCHWtoNHWC((half*)weights_, (float*)scratch, (int)num_biases,
                      input_->GetC(), (int)num_biases, input_->GetC(),
                      input_->GetH(), input_->GetW());
  } else {
    copyTypeConverted((half*)weights_, (float*)scratch, (int)num_weights, 0);
  }

  if (cpuBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, cpuBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)biases_, (float*)scratch, (int)num_biases, 0);
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

  ReportCUDAErrors(
      cudaMemcpy(weights_, cpuWeight, weight_size, cudaMemcpyHostToDevice));
  if (use_bias_) {
    ReportCUDAErrors(
        cudaMemcpy(biases_, cpuBias, bias_size, cudaMemcpyHostToDevice));
  }
}

template <>
void FCLayer<half>::Eval(int N, half* output_tensor, const half* input_tensor,
                         const half* /*input2*/, void* /*scratch*/,
                         size_t /*scratch_size*/, cudnnHandle_t /*cudnn*/,
                         cublasHandle_t cublas, cudaStream_t stream, half***) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  // half alpha = float2half_rn(1.0f), beta = float2half_rn(0.0f);
  const __half_raw one_h{0x3C00};
  const __half_raw zero_h{0};
  half alpha = one_h;
  half beta = zero_h;
  ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, stream);
  }
}

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/,
                          cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                          cudaStream_t stream, float***) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, stream);
  }
}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, int usedSize, bool attention)
    : BaseLayer<DataType>(C, H, W, ip),
      used_size_(usedSize),
      attention_map_(attention) {
  size_t weight_size = sizeof(short) * this->input_->GetC() * 64;
  if (attention) weight_size = sizeof(short) * usedSize;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
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
    ReportCUDAErrors(cudaMemcpy(weights_, convertedWeights.data(),
                                used_size_ * sizeof(short),
                                cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(
        cudaMemcpy(weights_, cpuWeight, weight_size, cudaMemcpyHostToDevice));
  }
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(int N, DataType* output_tensor,
                                    const DataType* input_tensor,
                                    const DataType* /*input2*/,
                                    void* /*scratch*/, size_t /*scratch_size*/,
                                    cudnnHandle_t /*cudnn*/,
                                    cublasHandle_t /*cublas*/,
                                    cudaStream_t stream, DataType***) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  if (attention_map_) inputSize = used_size_;
  int outputSize = this->C * this->H * this->W;
  PolicyMap(N, output_tensor, input_tensor, weights_, inputSize, used_size_,
            outputSize, stream);
}

template <typename DataType>
PolicyMapLayer<DataType>::~PolicyMapLayer() {
  ReportCUDAErrors(cudaFree(weights_));
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::FusedWinogradConvSELayer(
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
    ActivationFunction activation, bool bias, bool skip_add, bool se, int se_k,
    bool use_gemm_ex, bool op_nhcw)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      op_nhcw_(op_nhcw) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH &&
      act_ != ACTIVATION_NONE) {
    throw Exception("Unsupported activation for fused winograd conv SE layer.");
  }
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 3 * 3;

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  }

  // 6x6 transformed filter size, for 3x3 convolution
  ReportCUDAErrors(cudaMalloc(&transformed_weights_, weight_size * 4));

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    ReportCUDAErrors(cudaMalloc(&w1_, weight_size1));
    ReportCUDAErrors(cudaMalloc(&w2_, weight_size2));
    ReportCUDAErrors(cudaMalloc(&b1_, biases_size1));
    ReportCUDAErrors(cudaMalloc(&b2_, biases_size2));
  }
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadWeights(float* pfilter,
                                                     float* pBias,
                                                     void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size + bias_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights_, weights);
}

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
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, 0);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, 0);
}

template <>
void BaseLayer<half>::cublasRowMajorMatrixMul(const half* A, const half* B,
                                              half* Out, int M, int N, int K,
                                              int batchSize,
                                              cublasHandle_t cublas) {
  // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
  // float to half type doesn't work before CUDA 10.0
  __half_raw one_h{0x3C00};
  __half_raw zero_h{0};
  half halfOne = one_h;
  half halfZero = zero_h;

  // dimensions of matrix A = M x K
  // dimensions of matrix B = K x N
  // dimensions of output   = M x N

  // cublas supports only col major output
  // to multiply row major matrices, use the trick below
  ReportCUBLASErrors(cublasGemmStridedBatchedEx(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &halfOne, B, CUDA_R_16F, N,
      N * K, A, CUDA_R_16F, K, K * M, &halfZero, Out, CUDA_R_16F, N, N * M,
      batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
}

template <>
void BaseLayer<float>::cublasRowMajorMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;
  if (use_gemm_ex_)
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
        N * K, A, CUDA_R_32F, K, K * M, &floatZero, Out, CUDA_R_32F, N, N * M,
        batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  else
    // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
    ReportCUBLASErrors(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
        K * M, &floatZero, Out, N, N * M, batchSize));
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType***) {
  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType, false>(N, c_input_, transformed_input, input,
                                  stream);
  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights_, transformed_output, N * 4, C,
      c_input_, 36, cublas);

  if (act_ == ACTIVATION_NONE) {
    if (!has_se_ && use_bias_ && !skip_add_)
      OutputTransform<DataType, false, ACTIVATION_NONE, true, false, false,
                      false>(N, C, 0, output, transformed_output, nullptr,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_RELU) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_RELU, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, stream);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              stream);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               stream);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_RELU, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_MISH) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_MISH, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, stream);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              stream);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               stream);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_MISH, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else
    throw Exception("unsupported network type!");
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::~FusedWinogradConvSELayer() {
  ReportCUDAErrors(cudaFree(transformed_weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
  if (has_se_) {
    ReportCUDAErrors(cudaFree(w1_));
    ReportCUDAErrors(cudaFree(w2_));
    ReportCUDAErrors(cudaFree(b1_));
    ReportCUDAErrors(cudaFree(b2_));
  }
}

template <typename DataType>
Conv1Layer<DataType>::Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W,
                                 int Cin, ActivationFunction activation,
                                 bool bias, bool use_gemm_ex)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias) {
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 1 * 1;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  }
}

template <typename DataType>
void Conv1Layer<DataType>::LoadWeights(float* pfilter, float* pBias,
                                       void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 1 * 1;
  const size_t bias_size = sizeof(float) * C;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }
}
template <>
void Conv1Layer<half>::cublasSpecialMatrixMul(const half* A, const half* B,
                                              half* Out, int M, int N, int K,
                                              int batchSize,
                                              cublasHandle_t cublas) {
  // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
  // float to half type doesn't work before CUDA 10.0
  __half_raw one_h{0x3C00};
  __half_raw zero_h{0};
  half halfOne = one_h;
  half halfZero = zero_h;

  // dimensions of matrix A = M x K
  // dimensions of matrix B = K x N
  // dimensions of output   = M x N

  // cublas supports only col major output
  // to multiply row major matrices, use the trick below
  // NOTE strideB set to 0 below!
  ReportCUBLASErrors(cublasGemmStridedBatchedEx(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &halfOne, B, CUDA_R_16F, N,
      N * K, A, CUDA_R_16F, K, 0, &halfZero, Out, CUDA_R_16F, N, N * M,
      batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
}

template <>
void Conv1Layer<float>::cublasSpecialMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;

  // NOTE strideB set to 0 below!
  if (use_gemm_ex_)
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
        N * K, A, CUDA_R_32F, K, 0, &floatZero, Out, CUDA_R_32F, N, N * M,
        batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  else
    // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
    ReportCUBLASErrors(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
        0, &floatZero, Out, N, N * M, batchSize));
}

template <typename DataType>
void Conv1Layer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                const DataType* /*input2*/, void* /*scratch*/,
                                size_t /*scratch_size*/,
                                cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                                cudaStream_t stream, DataType***) {
  cublasSpecialMatrixMul(weights_, input, output, C, H * W, c_input_, N,
                         cublas);

  if (use_bias_)
    addBias_NCHW(output, output, biases_, N, C, H, W, act_, stream);
  else if (act_ != ACTIVATION_NONE)
    addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W,
               0, act_, stream);
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
  ReportCUDAErrors(cudaFree(weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(BaseLayer<DataType>* ip, int C, bool se,
                                       int se_k, bool use_gemm_ex, bool first,

                                       bool last, ActivationFunction activation,
                                       int shared_mem_size)
    : BaseLayer<DataType>(C, 8, 8, ip, ip->isNHWC(), use_gemm_ex),
      has_se_(se),
      se_k_(se_k),
      c_input_(C),
      first_block_(first),
      last_block_(last),
      shared_mem_size_(shared_mem_size),
      act_(activation) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH) {
    throw Exception("Unsupported activation for residual block.");
  }
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * C * C * 3 * 3;

  const size_t bias_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases0_, bias_size));
  ReportCUDAErrors(cudaMalloc(&biases1_, bias_size));

  // 6x6 transformed filter size, for 3x3 convolution
  ReportCUDAErrors(cudaMalloc(&transformed_weights0_, weight_size * 4));
  ReportCUDAErrors(cudaMalloc(&transformed_weights1_, weight_size * 4));

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    ReportCUDAErrors(cudaMalloc(&w1_, weight_size1));
    ReportCUDAErrors(cudaMalloc(&w2_, weight_size2));
    ReportCUDAErrors(cudaMalloc(&b1_, biases_size1));
    ReportCUDAErrors(cudaMalloc(&b2_, biases_size2));
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
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases0_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights0_, weights);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights1(float* pfilter, float* pBias,
                                           void* scratch) {
  const size_t weight_size = sizeof(float) * C * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * C * 3 * 3, 0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases1_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, C, transformed_weights1_, weights);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadSEWeights(float* w1, float* b1, float* w2,
                                            float* b2, void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, 0);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, 0);
}

template <typename DataType>
void ResidualBlock<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* /*input2*/, void* scratch,
                                   size_t scratch_size, cudnnHandle_t /*cudnn*/,
                                   cublasHandle_t cublas, cudaStream_t stream,
                                   DataType***) {
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
  DataType* transformed_input;
  DataType* transformed_output;
  if (!scratch) {
    // Caller wants us to sub-allocate all memory we need from "output" tensor.
    transformed_input = output;  // This is true in normal cases too!
    transformed_output = transformed_input + (N * C * 8 * 8 * 36 / 16);
  } else {
    transformed_input = (DataType*)scratch;
    transformed_output =
        transformed_input + scratch_size / (2 * sizeof(DataType));
  }

  if (first_block_) {
    InputTransform<DataType, true>(N, c_input_, transformed_input, input,
                                   stream);
    BaseLayer<DataType>::cublasRowMajorMatrixMul(
        transformed_input, transformed_weights0_, transformed_output, N * 4, C,
        c_input_, 36, cublas);
  } else {
    BaseLayer<DataType>::cublasRowMajorMatrixMul(output, transformed_weights0_,
                                                 transformed_output, N * 4, C,
                                                 c_input_, 36, cublas);
  }

  if (act_ == ACTIVATION_RELU) {
    OutputInputTransform<DataType, false, ACTIVATION_RELU, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  } else if (act_ == ACTIVATION_MISH) {
    OutputInputTransform<DataType, false, ACTIVATION_MISH, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  }
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights1_, transformed_output, N * 4, C, C,
      36, cublas);

  const bool fp16 = std::is_same<half, DataType>::value;
  bool allowFusing =
      (C <= kMaxResBlockFusingChannels) ||
      (fp16 && (shared_mem_size_ >= kMaxResBlockFusingSeFp16AmpereSmem) &&
       (C <= kMaxResBlockFusingSeKFp16Ampere));

  if (act_ == ACTIVATION_RELU) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_RELU, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, stream);
        } else {
          OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, stream);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         stream);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_RELU, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, stream);
    }
  } else if (act_ == ACTIVATION_MISH) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_MISH, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, stream);
        } else {
          OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, stream);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         stream);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_MISH, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, stream);
    }
  }
  // "output" tensor now contains transformed input for the next
  // convolution
}

template <typename DataType>
ResidualBlock<DataType>::~ResidualBlock() {
  ReportCUDAErrors(cudaFree(transformed_weights0_));
  ReportCUDAErrors(cudaFree(biases0_));
  ReportCUDAErrors(cudaFree(transformed_weights1_));
  ReportCUDAErrors(cudaFree(biases1_));
  if (has_se_) {
    ReportCUDAErrors(cudaFree(w1_));
    ReportCUDAErrors(cudaFree(w2_));
    ReportCUDAErrors(cudaFree(b1_));
    ReportCUDAErrors(cudaFree(b2_));
  }
}

template <typename DataType>
void allocAndUpload(DataType** gpu_dest, std::vector<float> cpu_src,
                    void* scratch) {
  size_t size = cpu_src.size() * sizeof(DataType);
  if (size == 0) {
    *gpu_dest = nullptr;
    return;
  }
  ReportCUDAErrors(cudaMalloc(gpu_dest, size));
  ReportCUDAErrors(cudaMemcpy(scratch, &cpu_src[0],
                              cpu_src.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)(*gpu_dest), (float*)scratch,
                    (int)cpu_src.size(), 0);
}

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(
    BaseLayer<DataType>* ip, const MultiHeadWeights::PolicyHead& weights,
    void* scratch, bool attention_body, ActivationFunction act,
    int max_batch_size)
    : BaseLayer<DataType>(64 * 64 + 24 * 8, 1, 1, ip),
      attention_body_(attention_body),
      // Old networks without attention body (e.g. T79) use hardcoded SELU
      // activations.
      act_(attention_body ? act : ACTIVATION_SELU) {
  embedding_op_size_ = weights.ip_pol_b.size();
  wq_op_size_ = weights.ip2_pol_b.size();
  wk_op_size_ = weights.ip3_pol_b.size();

  encoder_heads_ = weights.pol_encoder_head_count;
  policy_d_model_ = wq_op_size_;

  allocAndUpload<DataType>(&ip_pol_w_, weights.ip_pol_w, scratch);
  allocAndUpload<DataType>(&ip_pol_b_, weights.ip_pol_b, scratch);

  allocAndUpload<DataType>(&ip2_pol_w_, weights.ip2_pol_w, scratch);
  allocAndUpload<DataType>(&ip2_pol_b_, weights.ip2_pol_b, scratch);

  allocAndUpload<DataType>(&ip3_pol_w_, weights.ip3_pol_w, scratch);
  allocAndUpload<DataType>(&ip3_pol_b_, weights.ip3_pol_b, scratch);

  // big allocation to hold wq and wk weights one after the other
  {
    size_t elements = weights.ip2_pol_w.size();
    assert(elements == weights.ip3_pol_w.size());

    size_t size = elements * sizeof(DataType) * 2;
    ReportCUDAErrors(cudaMalloc(&wqk_w_, size));
    ReportCUDAErrors(
        cudaMemcpy(wqk_w_, ip2_pol_w_, size / 2, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(wqk_w_ + elements, ip3_pol_w_, size / 2,
                                cudaMemcpyDeviceToDevice));

    elements = weights.ip2_pol_b.size();
    size = elements * sizeof(DataType) * 2;
    ReportCUDAErrors(cudaMalloc(&wqk_b_, size));
    ReportCUDAErrors(
        cudaMemcpy(wqk_b_, ip2_pol_b_, size / 2, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(wqk_b_ + elements, ip3_pol_b_, size / 2,
                                cudaMemcpyDeviceToDevice));
  }

  allocAndUpload<DataType>(&ip4_pol_w_, weights.ip4_pol_w, scratch);

  for (const auto& enc : weights.pol_encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_heads_, embedding_op_size_,
        1.0f,        // using alpha = 1 for now (TODO: may change?)
        nullptr, 0,  // smolgen weights not implemented in
                     // policy encoder heads yet.
        max_batch_size, ACTIVATION_SWISH, act_,
        1e-6);  // attentionbody nets don't have policy encoders, so using old
                // epsilon for backward compatibility with T78.
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
EncoderBlock<DataType>::EncoderBlock(
    const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch, int heads,
    int size, float alpha, DataType* smolgen_global_scratch,
    int smolgen_global_size, int max_batch_size, ActivationFunction smolgen_act,
    ActivationFunction ffn_act, float default_eps)
    : embedding_op_size_(size),
      encoder_heads_(heads),
      alpha_(alpha),
      default_eps_(default_eps),
      has_smolgen_(cpu_weights.mha.has_smolgen),
      smolgen_activation_(smolgen_act),
      ffn_activation_(ffn_act),
      max_batch_size_(max_batch_size) {
  mha_q_size_ = cpu_weights.mha.q_b.size();
  mha_k_size_ = cpu_weights.mha.k_b.size();
  mha_v_size_ = cpu_weights.mha.v_b.size();
  mha_dense_size_ = cpu_weights.mha.dense_b.size();
  ffn_dense1_size_ = cpu_weights.ffn.dense1_b.size();
  ffn_dense2_size_ = cpu_weights.ffn.dense2_b.size();

  allocAndUpload<DataType>(&mha_q_w, cpu_weights.mha.q_w, scratch);
  allocAndUpload<DataType>(&mha_q_b, cpu_weights.mha.q_b, scratch);

  allocAndUpload<DataType>(&mha_k_w, cpu_weights.mha.k_w, scratch);
  allocAndUpload<DataType>(&mha_k_b, cpu_weights.mha.k_b, scratch);

  allocAndUpload<DataType>(&mha_v_w, cpu_weights.mha.v_w, scratch);
  allocAndUpload<DataType>(&mha_v_b, cpu_weights.mha.v_b, scratch);

  // big allocation to hold qkv weights one after the other
  {
    size_t elements = cpu_weights.mha.q_w.size();
    size_t size = elements * sizeof(DataType) * 3;
    ReportCUDAErrors(cudaMalloc(&mha_qkv_w, size));
    ReportCUDAErrors(
        cudaMemcpy(mha_qkv_w, mha_q_w, size / 3, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_w + elements, mha_k_w, size / 3,
                                cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_w + elements * 2, mha_v_w, size / 3,
                                cudaMemcpyDeviceToDevice));

    elements = cpu_weights.mha.q_b.size();
    size = elements * sizeof(DataType) * 3;
    ReportCUDAErrors(cudaMalloc(&mha_qkv_b, size));
    ReportCUDAErrors(
        cudaMemcpy(mha_qkv_b, mha_q_b, size / 3, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_b + elements, mha_k_b, size / 3,
                                cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_b + elements * 2, mha_v_b, size / 3,
                                cudaMemcpyDeviceToDevice));
  }

  allocAndUpload<DataType>(&mha_dense_w, cpu_weights.mha.dense_w, scratch);
  allocAndUpload<DataType>(&mha_dense_b, cpu_weights.mha.dense_b, scratch);

  allocAndUpload<DataType>(&ln1_gammas, cpu_weights.ln1_gammas, scratch);
  allocAndUpload<DataType>(&ln1_betas, cpu_weights.ln1_betas, scratch);

  allocAndUpload<DataType>(&ffn_dense1_w, cpu_weights.ffn.dense1_w, scratch);
  allocAndUpload<DataType>(&ffn_dense1_b, cpu_weights.ffn.dense1_b, scratch);

  allocAndUpload<DataType>(&ffn_dense2_w, cpu_weights.ffn.dense2_w, scratch);
  allocAndUpload<DataType>(&ffn_dense2_b, cpu_weights.ffn.dense2_b, scratch);

  allocAndUpload<DataType>(&ln2_gammas, cpu_weights.ln2_gammas, scratch);
  allocAndUpload<DataType>(&ln2_betas, cpu_weights.ln2_betas, scratch);

  // Smolgen weights.
  if (has_smolgen_) {
    smol_compress_size_ = cpu_weights.mha.smolgen.compress.size() / mha_q_size_;
    smol_dense_1_size_ = cpu_weights.mha.smolgen.dense1_b.size();
    smol_dense_2_size_ = cpu_weights.mha.smolgen.dense2_b.size();
    smol_global_size_ = smolgen_global_size;

    allocAndUpload<DataType>(&smol_compress, cpu_weights.mha.smolgen.compress,
                             scratch);
    allocAndUpload<DataType>(&smol_dense1_w, cpu_weights.mha.smolgen.dense1_w,
                             scratch);
    allocAndUpload<DataType>(&smol_dense1_b, cpu_weights.mha.smolgen.dense1_b,
                             scratch);
    allocAndUpload<DataType>(&smol_dense2_w, cpu_weights.mha.smolgen.dense2_w,
                             scratch);
    allocAndUpload<DataType>(&smol_dense2_b, cpu_weights.mha.smolgen.dense2_b,
                             scratch);

    allocAndUpload<DataType>(&smol_ln1_gammas,
                             cpu_weights.mha.smolgen.ln1_gammas, scratch);
    allocAndUpload<DataType>(&smol_ln1_betas, cpu_weights.mha.smolgen.ln1_betas,
                             scratch);
    allocAndUpload<DataType>(&smol_ln2_gammas,
                             cpu_weights.mha.smolgen.ln2_gammas, scratch);
    allocAndUpload<DataType>(&smol_ln2_betas, cpu_weights.mha.smolgen.ln2_betas,
                             scratch);

    // GPU memory already allocated in AttentionBody.
    smol_global = smolgen_global_scratch;
  }
}

template <typename DataType>
static void cublasXgemm(cublasHandle_t handle, cublasOperation_t transa,
                        cublasOperation_t transb, int m, int n, int k,
                        float alpha, const DataType* A, int lda,
                        const DataType* B, int ldb, float beta, DataType* C,
                        int ldc) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasHgemm(
        handle, transa, transb, m, n, k, (const half*)&alpha_h, (const half*)A,
        lda, (const half*)B, ldb, (const half*)&beta_h, (half*)C, ldc));
  } else {
    ReportCUBLASErrors(cublasSgemm(handle, transa, transb, m, n, k, &alpha,
                                   (const float*)A, lda, (const float*)B, ldb,
                                   &beta, (float*)C, ldc));
  }
}

template <typename DataType>
static void cublasXGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha, const void* A, int lda,
    long long int strideA, const void* B, int ldb, long long int strideB,
    float beta, void* C, int ldc, long long int strideC, int batchCount) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha_h, A, CUDA_R_16F, lda, strideA,
        B, CUDA_R_16F, ldb, strideB, &beta_h, C, CUDA_R_16F, ldc, strideC,
        batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
  } else {
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
        CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC,
        batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  }
}

template <typename DataType>
static void cublasXGemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               float alpha, DataType** A, int lda, DataType** B,
                               int ldb, float beta, DataType** C, int ldc,
                               int batchCount) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasHgemmBatched(
        handle, transa, transb, m, n, k, (const half*)&alpha_h, (half**)A, lda,
        (half**)B, ldb, (const half*)&beta_h, (half**)C, ldc, batchCount));
  } else {
    ReportCUBLASErrors(cublasSgemmBatched(
        handle, transa, transb, m, n, k, &alpha, (float**)A, lda, (float**)B,
        ldb, &beta, (float**)C, ldc, batchCount));
  }
}

// input/output tensor is in_out_tensor, others are used as scratch.
template <typename DataType>
void EncoderBlock<DataType>::Eval(int N, DataType* in_out_tensor,
                                  DataType* scratch, DataType* buffer1,
                                  DataType* buffer2, cublasHandle_t cublas,
                                  cudaStream_t stream,
                                  DataType*** offset_pointers) const {
  const int d_model = mha_q_size_;
  const int depth = d_model / encoder_heads_;

  // Calculate smolgen weights. Do this first so we can make use of
  // scratch, buffer1 and buffer2.
  if (has_smolgen_) {
    {
      // Compress.
      // input shape: N, 64, d_model
      // output shape: N, 64, hidden_channels
      const int num_inputs = d_model;
      const int num_outputs = smol_compress_size_;
      const int batch = N * 64;
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)smol_compress, num_inputs, in_out_tensor,
          num_inputs, 0.0f, scratch, num_outputs);
    }

    {
      // Hidden 1 dense.
      // input shape: N, 64 * hidden_channels
      // output shape: N, hidden_sz
      const int num_inputs = 64 * smol_compress_size_;
      const int num_outputs = smol_dense_1_size_;
      const int batch = N;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense1_w, num_inputs, scratch,
                            num_inputs, 0.0f, buffer1, num_outputs);

      LayerNorm<DataType>(batch, num_outputs, scratch, buffer1, smol_dense1_b,
                          (DataType*)nullptr, smol_ln1_gammas, smol_ln1_betas,
                          1e-3, 1.0, smolgen_activation_, stream);
    }

    {
      // Hidden 2 dense (gen_from)
      // input shape: N, hidden_sz
      // output shape: N, heads * gen_sz
      const int num_inputs = smol_dense_1_size_;
      const int num_outputs = smol_dense_2_size_;
      const int batch = N;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense2_w, num_inputs, scratch,
                            num_inputs, 0.0f, buffer1, num_outputs);

      LayerNorm<DataType>(batch, num_outputs, scratch, buffer1, smol_dense2_b,
                          (DataType*)nullptr, smol_ln2_gammas, smol_ln2_betas,
                          1e-3, 1.0, smolgen_activation_, stream);
    }

    {
      // Final smolgen weights generation.
      /*
        gen_from = tf.reshape(gen_from, [-1, heads, gen_sz])
        out = self.smol_weight_gen_dense(gen_from)
      */
      const int num_inputs =
          smol_dense_2_size_ / encoder_heads_; /* num_inputs == gen_sz == 256 */
      const int num_outputs = smol_global_size_; /* hwhw: 64 * 64 */
      const int batch = N * encoder_heads_;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_global, num_inputs, scratch,
                            num_inputs, 0.0f, buffer2, num_outputs);
    }
  }

  DataType* mha_q;
  DataType* mha_k;
  DataType* mha_v;

  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = d_model;
    const int batch = N * 64;
    const int max_batch = max_batch_size_ * 64;

    mha_q = scratch;
    mha_k = mha_q + num_outputs * max_batch;
    mha_v = mha_k + num_outputs * max_batch;

    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f,
        mha_qkv_w, num_inputs, num_inputs * num_outputs, in_out_tensor,
        num_inputs, 0, 0.0f, mha_q, num_outputs, num_outputs * max_batch, 3);
    addBiasBatched<DataType>(mha_q, mha_q, mha_qkv_b, 3, batch, num_outputs,
                             max_batch, ACTIVATION_NONE, stream);
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
  {
    if (*offset_pointers == nullptr) {
      std::vector<DataType*> offsets(encoder_heads_ * max_batch_size_ * 5);
      for (int i = 0; i < encoder_heads_ * max_batch_size_; i++) {
        int h = i % encoder_heads_;
        int n = i / encoder_heads_;
        offsets[i] = mha_k + h * depth + 64 * d_model * n;
        offsets[i + encoder_heads_ * max_batch_size_] =
            mha_q + h * depth + 64 * d_model * n;
        offsets[i + 2 * encoder_heads_ * max_batch_size_] =
            buffer1 + i * 64 * 64;
        offsets[i + 3 * encoder_heads_ * max_batch_size_] =
            mha_v + h * depth + 64 * d_model * n;
        offsets[i + 4 * encoder_heads_ * max_batch_size_] =
            buffer2 + h * depth + 64 * d_model * n;
      }
      ReportCUDAErrors(
          cudaMalloc((void**)offset_pointers,
                     encoder_heads_ * max_batch_size_ * 5 * sizeof(DataType*)));
      ReportCUDAErrors(
          cudaMemcpy(*offset_pointers, offsets.data(),
                     encoder_heads_ * max_batch_size_ * 5 * sizeof(DataType*),
                     cudaMemcpyHostToDevice));
    }
    cublasXGemmBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64 /*M*/, 64 /*N*/,
        depth /*K*/,  // A/B, and M/N are swapped for row-major to col-major
                      // transform
        factor,       // to handle "/ tf.math.sqrt(dk)"
        *offset_pointers,  // mha_k + offset /*A*/,
        d_model /*LDA*/,   // (d_model = depth * encoder_heads_) to skip over
                           // other "depth" slices / heads
        // 64 * d_model,     /*strideA*/
        *offset_pointers +
            encoder_heads_ * max_batch_size_,  // mha_q + offset /*B*/,
        d_model /*LDB*/,  // to skip over other other "depth" slices / heads
        // 64 * d_model,     /*strideB*/
        0.0f,
        *offset_pointers + encoder_heads_ * max_batch_size_ *
                               2,  // buffer1 + outOffset /*C*/,  // output
                                   // (matmul_qk) goes to buffer1
        64 /*LDC*/,
        // 64 * 64 /*strideC*/,
        N * encoder_heads_);
  }

  // attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
  // attention_weights -> buffer1
  if (has_smolgen_) {
    // Add smolgen weights to the scaled matmul_qk attention logits before
    // softmax.
    Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1, buffer2, stream);
  } else {
    Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1,
            (const DataType*)nullptr, stream);
  }

  {
    cublasXGemmBatched<DataType>(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, depth /*M*/, 64 /*N*/, 64 /*K*/, 1.0f,
        *offset_pointers + encoder_heads_ * max_batch_size_ *
                               3,  // mha_v + offset /*A*/,  // "v" matrix
        d_model /*LDA*/,           // to skip over other "depth" slices / heads
        // 64 * d_model,          /*strideA*/
        *offset_pointers + encoder_heads_ * max_batch_size_ *
                               2,  // buffer1 + weightsOffset /*B*/,
        64 /*LDB*/,                // 64 * 64, /*strideB*/
        0.0f,
        *offset_pointers +
            encoder_heads_ * max_batch_size_ *
                4,  // buffer2 + offset /*C*/,  // output goes to buffer2
        d_model /*LDC*/,
        // 64 * d_model /*strideC*/,
        N * encoder_heads_);
  }

  // #final dense layer (mha_dense), buffer2 -> buffer1
  {
    const int num_inputs = d_model;
    const int num_outputs = embedding_op_size_;
    const int batch = N * 64;
    cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                num_inputs, 1.0f, (const DataType*)mha_dense_w, num_inputs,
                buffer2, num_inputs, 0.0f, buffer1, num_outputs);
  }

  // LN1: skip connection and layer normalization (also bias add of prev gemm)
  // buffer1/in_out_tensor -> scratch
  LayerNorm<DataType>(N * 64, embedding_op_size_, scratch, buffer1, mha_dense_b,
                      in_out_tensor, ln1_gammas, ln1_betas, default_eps_,
                      alpha_, ACTIVATION_NONE, stream);

  // #FFN dense 1, scratch -> in_out_tensor
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = ffn_dense1_size_;  // encoder_dff
    const int batch = N * 64;
    cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                num_inputs, 1.0f, (const DataType*)ffn_dense1_w, num_inputs,
                scratch, num_inputs, 0.0f, in_out_tensor, num_outputs);
    addBiasBatched(in_out_tensor, in_out_tensor, ffn_dense1_b, 1, batch,
                   num_outputs, ffn_activation_, stream);
  }

  // #FFN dense 2, in_out_tensor -> buffer1
  {
    const int num_inputs = ffn_dense1_size_;  // encoder_dff
    const int num_outputs = embedding_op_size_;
    const int batch = N * 64;
    cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                num_inputs, 1.0f, (const DataType*)ffn_dense2_w, num_inputs,
                in_out_tensor, num_inputs, 0.0f, buffer1, num_outputs);
  }

  // LN2: skip connection and layer normilization (also bias add of prev gemm)
  // buffer1/scratch -> in_out_tensor
  LayerNorm<DataType>(N * 64, embedding_op_size_, in_out_tensor, buffer1,
                      ffn_dense2_b, scratch, ln2_gammas, ln2_betas,
                      default_eps_, alpha_, ACTIVATION_NONE, stream);
}

template <typename DataType>
void AttentionPolicyHead<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType*** offset_pointers) {
  DataType* input2_tensor = (DataType*)input2;
  DataType* buffer1 = output + scratch_size / (2 * sizeof(DataType));
  DataType* buffer2 = input2_tensor + scratch_size / (2 * sizeof(DataType));

  int inputC = this->input_->GetC();
  if (!attention_body_)
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8);

  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
  DataType* pol_embedding = input2_tensor;
  {
    const int num_outputs = embedding_op_size_;
    const int num_inputs = inputC;
    const int batch = N * 64;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip_pol_w_,
                          num_inputs,
                          attention_body_ ? input : (DataType*)scratch,
                          num_inputs, 0.0f, pol_embedding, num_outputs);
    addBiasBatched(pol_embedding, pol_embedding, ip_pol_b_, 1, batch,
                   num_outputs, act_, stream);
  }

  // 2. Encoder layers
  for (const auto pEnc : encoder_weights_) {
    pEnc->Eval(N, input2_tensor, (DataType*)scratch, buffer1, buffer2, cublas,
               stream, offset_pointers);
  }  // End of encoder blocks

  DataType* wq;
  DataType* wk;
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = policy_d_model_;
    const int batch = N * 64;
    wq = (DataType*)scratch;
    wk = wq + num_outputs * batch;

    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f,
        wqk_w_, num_inputs, num_inputs * num_outputs, input2_tensor, num_inputs,
        0, 0.0f, wq, num_outputs, num_outputs * batch, 2);

    addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs,
                             ACTIVATION_NONE, stream);
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
    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64 /*M*/, 64 /*N*/,
        policy_d_model_ /*K*/,
        factor,  // to handle "/ tf.math.sqrt(dk)"
        wk /*A*/, policy_d_model_ /*LDA*/, 64 * policy_d_model_, /*strideA*/
        wq /*B*/, policy_d_model_ /*LDB*/, 64 * policy_d_model_, /*strideB*/
        0.0f, output /*C*/,  // output (policy_attn_logits)
        64 /*LDC*/, 64 * 64 + 8 * 24 /*strideC*/, N);
  }

  // Compute promotion_logits in a single kernel (and put the result just after
  // policy_attn_logits interleaved to get concat for free)
  DataType* promotion_logits = output + 64 * 64;

  ComputePromotionLogits<DataType>(N, policy_d_model_, promotion_logits, wk,
                                   ip4_pol_w_, output, stream);
}

template <typename DataType>
AttentionPolicyHead<DataType>::~AttentionPolicyHead() {
  ReportCUDAErrors(cudaFree(ip_pol_w_));
  ReportCUDAErrors(cudaFree(ip_pol_b_));
  ReportCUDAErrors(cudaFree(ip2_pol_w_));
  ReportCUDAErrors(cudaFree(ip2_pol_b_));
  ReportCUDAErrors(cudaFree(ip3_pol_w_));
  ReportCUDAErrors(cudaFree(ip3_pol_b_));
  ReportCUDAErrors(cudaFree(ip4_pol_w_));
  ReportCUDAErrors(cudaFree(wqk_w_));
  ReportCUDAErrors(cudaFree(wqk_b_));
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
EncoderBlock<DataType>::~EncoderBlock() {
  ReportCUDAErrors(cudaFree(mha_q_w));
  ReportCUDAErrors(cudaFree(mha_q_b));
  ReportCUDAErrors(cudaFree(mha_k_w));
  ReportCUDAErrors(cudaFree(mha_k_b));
  ReportCUDAErrors(cudaFree(mha_v_w));
  ReportCUDAErrors(cudaFree(mha_v_b));
  ReportCUDAErrors(cudaFree(mha_qkv_w));
  ReportCUDAErrors(cudaFree(mha_qkv_b));
  ReportCUDAErrors(cudaFree(mha_dense_w));
  ReportCUDAErrors(cudaFree(mha_dense_b));
  ReportCUDAErrors(cudaFree(ln1_gammas));
  ReportCUDAErrors(cudaFree(ln1_betas));
  ReportCUDAErrors(cudaFree(ffn_dense1_w));
  ReportCUDAErrors(cudaFree(ffn_dense1_b));
  ReportCUDAErrors(cudaFree(ffn_dense2_w));
  ReportCUDAErrors(cudaFree(ffn_dense2_b));
  ReportCUDAErrors(cudaFree(ln2_gammas));
  ReportCUDAErrors(cudaFree(ln2_betas));
  if (has_smolgen_) {
    ReportCUDAErrors(cudaFree(smol_compress));
    ReportCUDAErrors(cudaFree(smol_dense1_w));
    ReportCUDAErrors(cudaFree(smol_dense1_b));
    ReportCUDAErrors(cudaFree(smol_dense2_w));
    ReportCUDAErrors(cudaFree(smol_dense2_b));
    ReportCUDAErrors(cudaFree(smol_ln1_gammas));
    ReportCUDAErrors(cudaFree(smol_ln1_betas));
    ReportCUDAErrors(cudaFree(smol_ln2_gammas));
    ReportCUDAErrors(cudaFree(smol_ln2_betas));
  }
}

template <typename DataType>
EmbeddingLayer<DataType>::EmbeddingLayer(BaseLayer<DataType>* ip,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         void* scratch, ActivationFunction act)
    : BaseLayer<DataType>(biases.size(), 8, 8, ip), act_(act) {
  allocAndUpload<DataType>(&weights_, weights, scratch);
  allocAndUpload<DataType>(&biases_, biases, scratch);
}

template <typename DataType>
EmbeddingLayer<DataType>::~EmbeddingLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
void EmbeddingLayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* /*input2*/,
    void* /*scratch*/, size_t /*scratch_size*/, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType***) {
  const int num_outputs = this->GetC();
  const int num_inputs = this->input_->GetC();
  const int batch = N * 64;
  cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                        num_inputs, 1.0f, weights_, num_inputs, input,
                        num_inputs, 0.0f, output, num_outputs);
  addBiasBatched(output, output, biases_, 1, batch, num_outputs, act_, stream);
}

template <typename DataType>
AttentionBody<DataType>::AttentionBody(const MultiHeadWeights& weights,
                                       void* scratch, Activations activations,
                                       int num_res_blocks, int input_c,
                                       int max_batch_size,
                                       bool is_pe_dense_embedding)
    : BaseLayer<DataType>(weights.ip_emb_b.size(), 8, 8, nullptr),
      embedding_op_size_(weights.ip_emb_b.size()),
      encoder_head_count_(weights.encoder_head_count),
      activations_(activations),
      num_resi_blocks_(num_res_blocks),
      input_c_(input_c),
      has_gating_(weights.ip_mult_gate.size() > 0 &&
                  weights.ip_add_gate.size() > 0),
      has_smolgen_(weights.has_smolgen),
      is_pe_dense_embedding_(is_pe_dense_embedding) {
  allocAndUpload<DataType>(&ip_emb_w_, weights.ip_emb_w, scratch);
  allocAndUpload<DataType>(&ip_emb_b_, weights.ip_emb_b, scratch);

  if (is_pe_dense_embedding_) {
    allocAndUpload<DataType>(&ip_emb_pre_w_, weights.ip_emb_preproc_w, scratch);
    allocAndUpload<DataType>(&ip_emb_pre_b_, weights.ip_emb_preproc_b, scratch);

    allocAndUpload<DataType>(&ip_emb_ln_g_, weights.ip_emb_ln_gammas, scratch);
    allocAndUpload<DataType>(&ip_emb_ln_b_, weights.ip_emb_ln_betas, scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_d1_w_, weights.ip_emb_ffn.dense1_w,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_d1_b_, weights.ip_emb_ffn.dense1_b,
                             scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_d2_w_, weights.ip_emb_ffn.dense2_w,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_d2_b_, weights.ip_emb_ffn.dense2_b,
                             scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_ln_g_, weights.ip_emb_ffn_ln_gammas,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_ln_b_, weights.ip_emb_ffn_ln_betas,
                             scratch);

    // 12 is the number of input channels used for the input encoding.
    embedding_dense_size_ = weights.ip_emb_preproc_b.size() / 64;
    embedding_ffn_size_ = weights.ip_emb_ffn.dense2_b.size();
    embedding_ffn_dff_ = weights.ip_emb_ffn.dense1_b.size();
  } else {
    size_t size = 64 * kNumPosEncodingChannels * sizeof(float);
    ReportCUDAErrors(cudaMalloc(&pos_encoding_, size));
    ReportCUDAErrors(
        cudaMemcpy(scratch, kPosEncoding, size, cudaMemcpyHostToDevice));
    copyTypeConverted(pos_encoding_, (float*)scratch, size, 0);
  }

  if (has_gating_) {
    allocAndUpload<DataType>(&ip_mult_gate_, weights.ip_mult_gate, scratch);
    allocAndUpload<DataType>(&ip_add_gate_, weights.ip_add_gate, scratch);
  }

  if (has_smolgen_) {
    allocAndUpload<DataType>(&smolgen_global_, weights.smolgen_w, scratch);
    smolgen_global_size_ = 64 * 64;
  }

  int num_encoders = weights.encoder.size();
  float alpha = (float)pow(2.0 * num_encoders, -0.25);
  for (const auto& enc : weights.encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_head_count_, embedding_op_size_, alpha,
        smolgen_global_, smolgen_global_size_, max_batch_size,
        activations_.smolgen_activation, activations_.ffn_activation,
        is_pe_dense_embedding_ ? 1e-3 : 1e-6);
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
AttentionBody<DataType>::~AttentionBody() {
  ReportCUDAErrors(cudaFree(ip_emb_w_));
  ReportCUDAErrors(cudaFree(ip_emb_b_));
  if (is_pe_dense_embedding_) {
    ReportCUDAErrors(cudaFree(ip_emb_pre_w_));
    ReportCUDAErrors(cudaFree(ip_emb_pre_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ln_g_));
    ReportCUDAErrors(cudaFree(ip_emb_ln_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d1_w_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d1_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d2_w_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d2_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_ln_g_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_ln_b_));
  } else {
    ReportCUDAErrors(cudaFree(pos_encoding_));
  }
  if (has_gating_) {
    ReportCUDAErrors(cudaFree(ip_mult_gate_));
    ReportCUDAErrors(cudaFree(ip_add_gate_));
  }
  if (has_smolgen_) {
    ReportCUDAErrors(cudaFree(smolgen_global_));
  }
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
void AttentionBody<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* input2, void* scratch,
                                   size_t scratch_size, cudnnHandle_t /*cudnn*/,
                                   cublasHandle_t cublas, cudaStream_t stream,
                                   DataType*** offset_pointers) {
  DataType* output_tensor = (DataType*)output;
  DataType* buffer1 = (DataType*)input2;
  DataType* buffer2 = buffer1 + scratch_size / (2 * sizeof(DataType));

  int inputC = input_c_;
  if (num_resi_blocks_ == 0) {
    assert(inputC == kInputPlanes);
    /*
      # if there are no residual blocks (pure transformer), do some input
      processing
    */
    if (is_pe_dense_embedding_) {
      // New encoding is made of dense layer fed with input from a 12-channel
      // slice of the input tensor.
      // pos_info = flow[..., :12]
      // pos_info_flat = tf.reshape(pos_info, [-1, 64 * 12])
      // pos_info_processed = tf.keras.layers.Dense(64*self.embedding_dense_sz,
      //                                            name=name+"embedding/preprocess")(pos_info_flat)
      const int num_outputs = 64 * embedding_dense_size_;
      const int num_inputs = 64 * 12;
      const int batch = N;

      convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, 12, 8, 8);
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)ip_emb_pre_w_, num_inputs,
          (const DataType*)scratch, num_inputs, 0.0f, buffer1, num_outputs);

      // addBiasBatched(buffer1, buffer1, ip_emb_pre_b_, batch, N, num_outputs,
      //               ACTIVATION_NONE, stream);
      const int size = num_outputs * N;
      // @todo addBiasBatched has a 4096 channel limit, needs refactoring.
      addVectors(buffer1, buffer1, ip_emb_pre_b_, size, size, num_outputs,
                 ACTIVATION_NONE, stream);
      inputPreprocessForAttentionBody((DataType*)scratch, input, buffer1, N,
                                      kInputPlanes, embedding_dense_size_, true,
                                      stream);
      inputC += embedding_dense_size_;
    } else {
      /*
      flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
      flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])
      # add positional encoding for each square to the input
      positional_encoding = tf.broadcast_to(tf.convert_to_tensor(self.POS_ENC,
      dtype=self.model_dtype), [tf.shape(flow)[0], 64,
      tf.shape(self.POS_ENC)[2]]) flow = tf.concat([flow, positional_encoding],
      axis=2)
      */
      inputPreprocessForAttentionBody((DataType*)scratch, input, pos_encoding_,
                                      N, kInputPlanes, kNumPosEncodingChannels,
                                      false, stream);
      inputC += kNumPosEncodingChannels;
    }
  } else {
    // #redirect flow through encoder blocks
    // flow = tf.transpose(flow, perm = [ 0, 2, 3, 1 ])
    // flow = tf.reshape(flow, [ -1, 64, self.RESIDUAL_FILTERS ])
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8);
  }

  if (is_pe_dense_embedding_) {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    DataType* temp = (DataType*)scratch;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, temp, num_inputs, 0.0f, embedding,
                            num_outputs);
      // embedding layer norm with fused in bias add of previous gemm.
      LayerNorm<DataType>(N * 64, embedding_op_size_, temp, embedding,
                          ip_emb_b_, (DataType*)nullptr, ip_emb_ln_g_,
                          ip_emb_ln_b_, 1e-3, 1.0,
                          activations_.default_activation, stream);
    }

    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(temp, temp, ip_mult_gate_, ip_add_gate_, N, 64,
                                 embedding_op_size_, stream);
    }

    // embedding FFN dense 1
    {
      const int num_inputs = embedding_ffn_size_;
      const int num_outputs = embedding_ffn_dff_;  // encoder_dff
      const int batch = N * 64;
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d1_w_,
                  num_inputs, temp, num_inputs, 0.0f, buffer1, num_outputs);
      addBiasBatched(buffer1, buffer1, ip_emb_ffn_d1_b_, 1, batch, num_outputs,
                     activations_.ffn_activation, stream);
    }

    // embedding FFN dense 2
    {
      const int num_inputs = embedding_ffn_dff_;  // encoder_dff
      const int num_outputs = embedding_ffn_size_;
      const int batch = N * 64;
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d2_w_,
                  num_inputs, buffer1, num_inputs, 0.0f, buffer2, num_outputs);
      // Embedding LN: skip connection and layer normilization (also bias add of
      // prev gemm) buffer2 -> embedding
      float alpha = (float)pow(2. * encoder_weights_.size(), -0.25);
      LayerNorm<DataType>(N * 64, embedding_ffn_size_, embedding, buffer2,
                          ip_emb_ffn_d2_b_, temp, ip_emb_ffn_ln_g_,
                          ip_emb_ffn_ln_b_, 1e-3, alpha, ACTIVATION_NONE,
                          stream);
    }

  } else {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, (DataType*)scratch, num_inputs, 0.0f,
                            embedding, num_outputs);
      addBiasBatched(embedding, embedding, ip_emb_b_, 1, batch, num_outputs,
                     activations_.default_activation, stream);
    }
    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(embedding, embedding, ip_mult_gate_,
                                 ip_add_gate_, N, 64, embedding_op_size_,
                                 stream);
    }
  }

  // 2. Encoder blocks
  for (const auto pEnc : encoder_weights_) {
    pEnc->Eval(N, output_tensor, (DataType*)scratch, buffer1, buffer2, cublas,
               stream, offset_pointers);
  }  // End of encoder blocks
}

template <typename DataType>
ValueHead<DataType>::ValueHead(BaseLayer<DataType>* ip,
                               const MultiHeadWeights::ValueHead& weights,
                               void* scratch, bool attention_body, bool wdl,
                               ActivationFunction act, int /*max_batch_size*/,
                               bool use_gemm_ex)
    : BaseLayer<DataType>(weights.ip_val_b.size(), 8, 8, ip),
      embedding_size_(attention_body ? weights.ip_val_b.size()
                                     : weights.value.biases.size()),
      value_hidden_size_(weights.ip1_val_b.size()),
      wdl_(wdl),
      attention_body_(attention_body),
      act_(act) {
  if (attention_body_) {
    allocAndUpload<DataType>(&ip_val_w_, weights.ip_val_w, scratch);
    allocAndUpload<DataType>(&ip_val_b_, weights.ip_val_b, scratch);
  } else {
    conv_ = std::make_unique<Conv1Layer<DataType>>(
        ip, weights.value.biases.size(), 8, 8, ip->GetC(), act, true,
        use_gemm_ex);
    conv_->LoadWeights((float*)&weights.value.weights[0],
                       (float*)&weights.value.biases[0], scratch);
  }

  allocAndUpload<DataType>(&ip1_val_w_, weights.ip1_val_w, scratch);
  allocAndUpload<DataType>(&ip1_val_b_, weights.ip1_val_b, scratch);

  allocAndUpload<DataType>(&ip2_val_w_, weights.ip2_val_w, scratch);
  allocAndUpload<DataType>(&ip2_val_b_, weights.ip2_val_b, scratch);
}

template <typename DataType>
ValueHead<DataType>::~ValueHead() {
  if (attention_body_) {
    ReportCUDAErrors(cudaFree(ip_val_w_));
    ReportCUDAErrors(cudaFree(ip_val_b_));
  }
  ReportCUDAErrors(cudaFree(ip1_val_w_));
  ReportCUDAErrors(cudaFree(ip1_val_b_));
  ReportCUDAErrors(cudaFree(ip2_val_w_));
  ReportCUDAErrors(cudaFree(ip2_val_b_));
}

template <typename DataType>
void ValueHead<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t /*cudnn*/,
                               cublasHandle_t cublas, cudaStream_t stream,
                               DataType***) {
  DataType* buffer = (DataType*)input2;
  {
    const int num_inputs = this->input_->GetC();
    const int num_outputs = embedding_size_;
    const int batch = N * 64;
    if (attention_body_) {
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_val_w_,
                            num_inputs, input, num_inputs, 0.0f, buffer,
                            num_outputs);
      addBiasBatched<DataType>(buffer, buffer, ip_val_b_, 1, batch, num_outputs,
                               act_, stream);

    } else {
      conv_->Eval(N, buffer, input, nullptr, scratch, scratch_size, nullptr,
                  cublas, stream);
    }
  }

  {
    // Value dense 1
    const int num_inputs = embedding_size_ * 64;
    const int num_outputs = value_hidden_size_;
    const int batch = N;
    DataType* layer_out = (DataType*)scratch;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip1_val_w_,
                          num_inputs, buffer, num_inputs, 0.0f, layer_out,
                          num_outputs);
    addBiasBatched<DataType>(layer_out, layer_out, ip1_val_b_, 1, batch,
                             num_outputs, act_, stream);
  }

  {
    // Value dense 2
    const int num_inputs = value_hidden_size_;
    const int num_outputs = wdl_ ? 3 : 1;
    const int batch = N;
    DataType* layer_out = (DataType*)output;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip2_val_w_,
                          num_inputs, (DataType*)scratch, num_inputs, 0.0f,
                          layer_out, num_outputs);
    addVectors(layer_out, layer_out, ip2_val_b_, num_outputs * batch,
               num_outputs * batch, num_outputs,
               wdl_ ? ACTIVATION_NONE : ACTIVATION_TANH, stream);
  }
}

// Template instantiation.
#ifdef USE_CUDNN
template class ConvLayer<half>;
template class ConvLayer<float>;
#endif

template class FCLayer<half>;
template class FCLayer<float>;

template class SELayer<half>;
template class SELayer<float>;

template class PolicyMapLayer<half>;
template class PolicyMapLayer<float>;

template class FusedWinogradConvSELayer<half>;
template class FusedWinogradConvSELayer<float>;

template class Conv1Layer<half>;
template class Conv1Layer<float>;

template class ResidualBlock<half>;
template class ResidualBlock<float>;

template class AttentionPolicyHead<half>;
template class AttentionPolicyHead<float>;

template class EncoderBlock<half>;
template class EncoderBlock<float>;

template class AttentionBody<half>;
template class AttentionBody<float>;

template class EmbeddingLayer<half>;
template class EmbeddingLayer<float>;

template class ValueHead<half>;
template class ValueHead<float>;

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

const char* CublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown cublas error";
}

void CublasError(cublasStatus_t status, const char* file, const int& line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUBLAS error: %s (%s:%d) ", CublasGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

void CudaError(cudaError_t status, const char* file, const int& line) {
  if (status != cudaSuccess) {
    char message[128];
    sprintf(message, "CUDA error: %s (%s:%d) ", cudaGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

}  // namespace cudnn_backend
}  // namespace lczero
