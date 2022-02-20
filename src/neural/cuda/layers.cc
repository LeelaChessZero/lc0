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
namespace lczero {
//void dumpTensor(void* memory, int elements, const char* message, bool fp16 = false);

namespace cudnn_backend {

// Use Single kernel for entire SE operation.
// Right now supported only for fp16 with nhwc and it's quite a bit faster
// than using multiple passes. The flag can be set to false for debugging.
static constexpr bool kUseFusedSELayer = true;

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(false) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, bool gemm_ex)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(gemm_ex) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip), C(c), H(h), W(w), nhwc_(ip->nhwc_), use_gemm_ex_(false) {}

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

  if (use_relu_) {
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
                               int filter, int Cin, bool relu, bool bias)
    : BaseLayer<DataType>(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_bias_(bias) {
  init();
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(bool nhwc, int C, int H, int W, int filter,
                               int Cin, bool relu, bool bias)
    : BaseLayer<DataType>(C, H, W, nullptr, nhwc),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
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
    fp32NCHWtofp16NHWC((half*)weights, (float*)scratch, C, c_input_, C,
                       c_input_, filter_size_, filter_size_);
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
                               cublasHandle_t /*cublas*/, cudaStream_t stream) {
  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_, layout,
                                               dataType, N, C, H, W));

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_, layout,
                                               dataType, N, c_input_, H, W));

  float alpha = 1.0f, beta = 0.0f;

  if (!(use_relu_ || use_bias_ || input2)) {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
        output));
  }
#if CUDNN_MAJOR != 7 || CUDNN_MINOR != 0
  else if (input2) {
    // fused bias + sum + relu!
    ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &alpha, out_tensor_desc_,
        input2, bias_desc_, biases, activation_, out_tensor_desc_, output));
  } else {
    // For some reason cudnn doesn't support just Convolution + Bias with nchw
    // (winograd algorithm) it works fine when RELU is also needed which is
    // somewhat strange.
    if ((!nhwc_) && (!use_relu_)) {
      ReportCUDNNErrors(cudnnConvolutionForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output));
      // add bias
      addBias_NCHW(output, output, biases, N, C, H, W, false, stream);
    } else {
      ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output, bias_desc_, biases, activation_,
          out_tensor_desc_, output));
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
    if (use_relu_) {
      ReportCUDNNErrors(cudnnActivationForward(cudnn, activation_, &alpha,
                                               out_tensor_desc_, output, &beta,
                                               out_tensor_desc_, output));
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
                           bool addPrevLayerBias)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias) {
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
                          cublasHandle_t cublas, cudaStream_t stream) {
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
  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, RELU, stream);

  // 3. Second fully connected layer.
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C));
  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, NONE, stream);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  globalScale(N, C, output, input, op2, bPrev_, false);
}

template <>
void SELayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                         cudaStream_t stream) {
  bool se_done = false;
  if (kUseFusedSELayer && nhwc_) {
    se_done = Se_Fp16_NHWC(N, C, numFc1Out_, output, input2, input, w1_t_, b1_,
                           w2_t_, b2_, bPrev_);
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
    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, RELU, stream);

    // 3. Second fully connected layer.
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                   numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                   numFc1Out_, &beta, op2, 2 * C));
    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, NONE, stream);

    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    globalScale(N, C, output, input, op2, bPrev_, nhwc_);
  }
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation)
    : BaseLayer<DataType>(C, H, W, ip),
      use_bias_(bias), act_(activation)
  {
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
    fp32NCHWtofp16NHWC((half*)weights_, (float*)scratch, (int)num_biases,
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
                         cublasHandle_t cublas, cudaStream_t stream) {
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

  if (use_bias_ || (act_ != NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, stream);
  }
}

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/,
                          cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                          cudaStream_t stream) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || (act_ != NONE)) {
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
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size)); }

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
                                    cudnnHandle_t /*cudnn*/, cublasHandle_t /*cublas*/, cudaStream_t stream) {
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
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin, bool relu, bool bias,
    bool skip_add, bool se, int se_k, bool use_gemm_ex, bool op_nhcw)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex),
      c_input_(Cin),
      use_relu_(relu),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      op_nhcw_(op_nhcw) {
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
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3, 0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights_, weights);
}

// TODO: Do this on the GPU to improve network load time!
static inline void CpuTranspose(float* op, float* ip, size_t rows, size_t cols) {
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
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(), num_weights1*sizeof(float),
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
void BaseLayer<half>::cublasRowMajorMatrixMul(
    const half* A, const half* B, half* Out, int M, int N, int K, int batchSize,
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
void BaseLayer<float>::cublasRowMajorMatrixMul(
    const float* A, const float* B, float* Out, int M, int N, int K,
    int batchSize, cublasHandle_t cublas) {

  float floatOne  = 1.0f;
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
    cublasHandle_t cublas, cudaStream_t stream) {

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType, false>(N, c_input_, transformed_input, input, stream);
  BaseLayer<DataType>::cublasRowMajorMatrixMul(transformed_input, transformed_weights_, transformed_output, N*4, C, c_input_, 36, cublas);  

  if (has_se_ && use_relu_ && use_bias_ && skip_add_)
    OutputTransform<DataType, true, true, true, true, false, false>(
        N, C, se_k_, output, transformed_output, input2, biases_, w1_, b1_, w2_,
        b2_, stream);
  else if (!has_se_ && use_relu_ && use_bias_ && !skip_add_) {
    if (op_nhcw_)
      OutputTransform<DataType, false, true, true, false, false, true>(
          N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
          nullptr, nullptr, nullptr, stream);
    else
      OutputTransform<DataType, false, true, true, false, false, false>(
          N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
          nullptr, nullptr, nullptr, stream);
  } else if (!has_se_ && use_relu_ && use_bias_ && skip_add_)
    OutputTransform<DataType, false, true, true, true, false, false>(
        N, C, 0, output, transformed_output, input2, biases_, nullptr, nullptr,
        nullptr, nullptr, stream);
  else if (!has_se_ && !use_relu_ && use_bias_ && !skip_add_)
    OutputTransform<DataType, false, false, true, false, false, false>(
        N, C, 0, output, transformed_output, nullptr, biases_, nullptr, nullptr,
        nullptr, nullptr, stream);
  else
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
                                 int Cin, bool relu, bool bias,
                                 bool use_gemm_ex)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex),
      c_input_(Cin),
      use_relu_(relu),
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
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1, 0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }
}

template <typename DataType>
void Conv1Layer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                const DataType* /*input2*/, void* /*scratch*/,
                                size_t /*scratch_size*/,
                                cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                                cudaStream_t stream) {
  BaseLayer<DataType>::cublasRowMajorMatrixMul(weights_, input, output, C,
                                               H * W, c_input_, N, cublas);

  if (use_bias_)
    addBias_NCHW(output, output, biases_, N, C, H, W, use_relu_, stream);
  else if (use_relu_)
    addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W,
               0, use_relu_ ? RELU : NONE, stream);
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
  ReportCUDAErrors(cudaFree(weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(
    BaseLayer<DataType>* ip, int C, bool se, int se_k, bool use_gemm_ex, bool first, bool last)
    : BaseLayer<DataType>(C, 8, 8, ip, ip->isNHWC(), use_gemm_ex),
      has_se_(se),
      se_k_(se_k),
      c_input_(C),
      first_block_(first),
      last_block_(last) {
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
void ResidualBlock<DataType>::LoadWeights0(float* pfilter,
                                           float* pBias,
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
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3, 0);

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
void ResidualBlock<DataType>::LoadSEWeights(float* w1, float* b1,
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

template <typename DataType>
void ResidualBlock<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* /*input2*/,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
                                   cublasHandle_t cublas, cudaStream_t stream) {
  // normally:
  // - "output" initially contains the transformed input, 
  //    and after this layer, it contains the transformed input for next layer
  // - "input" contains the original/untransformed input
  // special cases:
  //   - for first_block_, input is real input (untransformed)
  //   - for last_block_, output is the final output of this block (untransformed)

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  if (first_block_) {
    InputTransform<DataType, true>(N, c_input_, transformed_input, input, stream);
    BaseLayer<DataType>::cublasRowMajorMatrixMul(
        transformed_input, transformed_weights0_, transformed_output, N * 4, C,
        c_input_, 36, cublas);
  } else {
    BaseLayer<DataType>::cublasRowMajorMatrixMul(output, transformed_weights0_,
                                                 transformed_output, N * 4, C,
                                                 c_input_, 36, cublas);
  }

  OutputInputTransform<DataType, false, true, true, false>(
      N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
      nullptr, nullptr, nullptr, nullptr, stream);
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights1_, transformed_output, N * 4, C, C,
      36, cublas);

  if (last_block_) {
    if (has_se_)
      OutputTransform<DataType, true, true, true, true, true, false>(
          N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_, stream);
    else
      OutputTransform<DataType, false, true, true, true, true, false>(
          N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_, stream);
  } else {
    if (has_se_)
      OutputInputTransform<DataType, true, true, true, true>(
        N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_, stream);
    else
      OutputInputTransform<DataType, false, true, true, true>(
        N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_, stream);
    // "output" tensor now contains transformed input for the next
    // convolution
  }
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
  ReportCUDAErrors(
      cudaMemcpy(scratch, &cpu_src[0], cpu_src.size() * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)(*gpu_dest), (float*)scratch,
                    (int)cpu_src.size(), 0);
}

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(BaseLayer<DataType>* ip,
                                                   const LegacyWeights& weights,
                                                   void* scratch)
    : BaseLayer<DataType>(ip->GetC(), 8, 8, ip) {
  embedding_op_size_ = weights.ip_pol_b.size();
  wq_op_size_ = weights.ip2_pol_b.size();
  wk_op_size_ = weights.ip3_pol_b.size();
  ppo_op_size_ = weights.ip4_pol_b.size();

  allocAndUpload<DataType>(&ip_pol_w_, weights.ip_pol_w, scratch);
  allocAndUpload<DataType>(&ip_pol_b_, weights.ip_pol_b, scratch);

  allocAndUpload<DataType>(&ip2_pol_w_, weights.ip2_pol_w, scratch);
  allocAndUpload<DataType>(&ip2_pol_b_, weights.ip2_pol_b, scratch);

  allocAndUpload<DataType>(&ip3_pol_w_, weights.ip3_pol_w, scratch);
  allocAndUpload<DataType>(&ip3_pol_b_, weights.ip3_pol_b, scratch);

  allocAndUpload<DataType>(&ip4_pol_w_, weights.ip4_pol_w, scratch);
  allocAndUpload<DataType>(&ip4_pol_b_, weights.ip4_pol_b, scratch);

  for (const auto& enc : weights.encoder) {
    EncoderWeights* pW = new EncoderWeights(enc, scratch);
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
AttentionPolicyHead<DataType>::EncoderWeights::EncoderWeights(
    const LegacyWeights::EncoderLayer& cpu_weights, void* scratch) {
  mha_q_size_ = cpu_weights.mha.q_b.size();
  mha_k_size_ = cpu_weights.mha.k_b.size();
  mha_v_size_ = cpu_weights.mha.v_b.size();
  mha_dense_size_ = cpu_weights.mha.dense_b.size();

  // debug!
  printf("\nsize of weight mha.q_b/w: %d, %d\n",
         (int)cpu_weights.mha.q_b.size(), (int)cpu_weights.mha.q_w.size());
  printf("\nsize of weight mha.k_b/w: %d, %d\n",
         (int)cpu_weights.mha.k_b.size(), (int)cpu_weights.mha.k_w.size());
  printf("\nsize of weight mha.v_b/w: %d, %d\n",
         (int)cpu_weights.mha.v_b.size(), (int)cpu_weights.mha.v_w.size());
  printf("\nsize of weight mha.dense_b/w: %d, %d\n",
         (int)cpu_weights.mha.dense_b.size(),
         (int)cpu_weights.mha.dense_w.size());
  printf("\nsize of ln1 betas/gammas: %d, %d\n",
         (int)cpu_weights.ln1_betas.size(), (int)cpu_weights.ln1_gammas.size());
  printf("\nsize of ln2 betas/gammas: %d, %d\n",
         (int)cpu_weights.ln2_betas.size(), (int)cpu_weights.ln2_gammas.size());
  printf("\nsize of weight ffn.dense1_b/w: %d, %d\n",
         (int)cpu_weights.ffn.dense1_b.size(),
         (int)cpu_weights.ffn.dense1_w.size());
  printf("\nsize of weight ffn.dense2_b/w: %d, %d\n",
         (int)cpu_weights.ffn.dense2_b.size(),
         (int)cpu_weights.ffn.dense2_w.size());


  allocAndUpload<DataType>(&mha_q_w, cpu_weights.mha.q_w, scratch);
  allocAndUpload<DataType>(&mha_q_b, cpu_weights.mha.q_b, scratch);

  allocAndUpload<DataType>(&mha_k_w, cpu_weights.mha.k_w, scratch);
  allocAndUpload<DataType>(&mha_k_b, cpu_weights.mha.k_b, scratch);

  allocAndUpload<DataType>(&mha_v_w, cpu_weights.mha.v_w, scratch);
  allocAndUpload<DataType>(&mha_v_b, cpu_weights.mha.v_b, scratch);

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
}

// taken from https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
unsigned short float_to_half(const float x) {
  const unsigned int b = (*(unsigned int*)&x) + 0x00001000;
  const unsigned int e = (b & 0x7F800000) >> 23;
  const unsigned int m = b & 0x007FFFFF;
  return (b & 0x80000000) >> 16 |
         (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
         ((e < 113) & (e > 101)) *
             ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
         (e > 143) * 0x7FFF;
}


template <>
void AttentionPolicyHead<half>::Eval(
    int N, half* output, const half* input, const half* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t cudnn,
    cublasHandle_t cublas, cudaStream_t stream) {

  half* scratch0 = (half*)scratch;
  half* scratch1 = (half*)scratch + scratch_size / (2 * sizeof(half));
  half* scratch2 = (half*)input2;
  half* scratch3 = (half*)input2 + scratch_size / (2 * sizeof(half));
  half* scratch4 = output + scratch_size / (2 * sizeof(half));

  // half alpha = float2half_rn(1.0f), beta = float2half_rn(0.0f);
  const __half_raw one_h{0x3C00};
  const __half_raw zero_h{0};
  half alpha = one_h;
  half beta = zero_h;


  fp16NCHWtoNHWC(scratch1, input, N, 64, N, 64, 8, 8);
  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_

  {
    const int num_outputs = embedding_op_size_;
    const int num_inputs = input_->GetC();  // 64 * C
    const int batch = N * 64;
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                   num_outputs, batch, num_inputs, &alpha,
                                   (const half*)ip_pol_w_, num_inputs, scratch1,
                                   num_inputs, &beta, scratch0, num_outputs));
    addVectors(scratch0, (half*)ip_pol_b_, scratch0, num_outputs * batch,
               num_outputs, num_outputs * batch, SELU, stream);
  }

  // 2. Encoder layers
  for (const auto pEnc : encoder_weights_) {
    const auto& enc = *pEnc;
    const int depth = d_model_ / encoder_heads_;

    // MHA q (scratch1)
    {
      const int num_inputs = embedding_op_size_;
      const int num_outputs = d_model_;
      const int batch = N * 64;
      ReportCUBLASErrors(
          cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                      num_inputs, &alpha, (const half*)enc.mha_q_w, num_inputs,
                      scratch0, num_inputs, &beta, scratch1, num_outputs));
      addVectors(scratch1, (half*)enc.mha_q_b, scratch1, num_outputs * batch,
                 num_outputs, num_outputs * batch, NONE, stream);
    }

    // MHA k (scratch2)
    {
      const int num_inputs = d_model_;
      const int num_outputs = d_model_;
      const int batch = N * 64;
      ReportCUBLASErrors(
          cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                      num_inputs, &alpha, (const half*)enc.mha_k_w, num_inputs,
                      scratch0, num_inputs, &beta, scratch2, num_outputs));
      addVectors(scratch2, (half*)enc.mha_k_b, scratch2, num_outputs * batch,
                 num_outputs, num_outputs * batch, NONE, stream);
    }

    // MHA v (scratch3)
    {
      const int num_inputs = d_model_;
      const int num_outputs = d_model_;
      const int batch = N * 64;
      ReportCUBLASErrors(
          cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                      num_inputs, &alpha, (const half*)enc.mha_v_w, num_inputs,
                      scratch0, num_inputs, &beta, scratch3, num_outputs));
      addVectors(scratch3, (half*)enc.mha_v_b, scratch3, num_outputs * batch,
                 num_outputs, num_outputs * batch, NONE, stream);
    }

    // Apply split_heads() to q, k and v
    // which basically transposes (batch_size, 64, num_heads, depth)
    // to (batch_size, num_heads, 64, depth)
    // Ankan - do we really need to transpose here?
    // (Maybe not, we can play with strides of the gemm and do independent gemms for each encoder head)

    // Apply scaled dot product attention:
    /*
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
    */

    // shape(k)[-1] = depth
    unsigned short factor = float_to_half(1.0f / sqrt((float)depth));

    // matmul_qk = tf.matmul(q, k, transpose_b=True)
    // q -> scratch1, k -> scratch2, v -> scratch3
    for (int i = 0; i < encoder_heads_; i++) {
      int offset = i * depth;
      int outOffset = i * N * 64 * 64;      // layout of the output: encoder_heads_ * Batch * 64 * 64
      cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
                                 64 /*M*/, 64 /*N*/, depth /*K*/,    // A/B, and M/N are swapped for row-major to col-major transform
                                 &factor,                            // to handle "/ tf.math.sqrt(dk)"
                                 scratch2 + offset /*A*/,
                                 CUDA_R_16F, 
                                 d_model_ /*LDA*/,  // (d_model_ = depth * encoder_heads_) to skip over
                                                    // other "depth" slices / heads
                                 64 * d_model_,    /*strideA*/
                                 scratch1 + offset /*B*/, 
                                 CUDA_R_16F, 
                                 d_model_ /*LDB*/,  // to skip over other other "depth" slices / heads
                                 64 * d_model_,     /*strideB*/
                                 &beta, 
                                 scratch4 + outOffset /*C*/,  // output (matmul_qk) goes to scratch4
                                 CUDA_R_16F, 
                                 64 /*LDC*/, 
                                 64 * 64 /*strideC*/,
                                 N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
    }

    // attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
    // attention_weights -> scratch4
    Softmax(encoder_heads_ * N * 64, 64, scratch4, scratch4, stream);

    // output = tf.matmul(attention_weights, v)
    for (int i = 0; i < encoder_heads_; i++) {
      int offset = i * depth;               // for output and "v" matrix
      int weightsOffset = i * N * 64 * 64;  // layout: encoder_heads_ * Batch*64*64
      cublasGemmStridedBatchedEx(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, 
          depth /*M*/, 64 /*N*/, 64 /*K*/,
          &alpha,
          scratch3 + offset /*A*/,          // "v" matrix
          CUDA_R_16F,
          d_model_ /*LDA*/,  // to skip over other "depth" slices / heads
          64 * d_model_,     /*strideA*/
          scratch4 + weightsOffset /*B*/, 
          CUDA_R_16F,
          64 /*LDB*/,
          64 * 64, /*strideB*/
          &beta,
          scratch1 + offset /*C*/,  // output goes to scratch1 again
          CUDA_R_16F, 
          d_model_ /*LDC*/, 
          64 * d_model_ /*strideC*/, 
          N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
    }

    // #final dense layer (mha_dense), scratch1 -> scratch2
    {
      const int num_inputs = d_model_;
      const int num_outputs = embedding_op_size_;
      const int batch = N * 64;
      ReportCUBLASErrors(
          cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                      num_inputs, &alpha, (const half*)enc.mha_dense_w, num_inputs,
                      scratch1, num_inputs, &beta, scratch2, num_outputs));
      addVectors(scratch2, (half*)enc.mha_dense_b, scratch2, num_outputs * batch,
                 num_outputs, num_outputs * batch, NONE, stream);
    }

    // LN1: skip connection and layer normilization
    // scratch2/scratch0 -> scratch3
    LayerNorm(N * 64, embedding_op_size_, scratch3, scratch2, scratch0,
              enc.ln1_gammas, enc.ln1_betas, 1e-6, stream);

    // #FFN dense 1, scratch3 -> scratch1
    {
      const int num_inputs = embedding_op_size_;
      const int num_outputs = encoder_dff_;
      const int batch = N * 64;
      ReportCUBLASErrors(cublasHgemm(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          &alpha, (const half*)enc.ffn_dense1_w, num_inputs, scratch3,
          num_inputs, &beta, scratch1, num_outputs));
      addVectors(scratch1, (half*)enc.ffn_dense1_b, scratch1,
                 num_outputs * batch, num_outputs, num_outputs * batch, SELU,
                 stream);
    }

    // #FFN dense 2, scratch1 -> scratch2
    {
      const int num_inputs = encoder_dff_;
      const int num_outputs = embedding_op_size_;
      const int batch = N * 64;
      ReportCUBLASErrors(cublasHgemm(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          &alpha, (const half*)enc.ffn_dense2_w, num_inputs, scratch1,
          num_inputs, &beta, scratch2, num_outputs));
      addVectors(scratch2, (half*)enc.ffn_dense2_b, scratch2,
                 num_outputs * batch, num_outputs, num_outputs * batch, NONE,
                 stream);
    }

    // LN2: skip connection and layer normilization
    // scratch2/scratch3 -> scratch0
    LayerNorm(N * 64, embedding_op_size_, scratch0, scratch2, scratch3,
              enc.ln2_gammas, enc.ln2_betas, 1e-6, stream);

  }  // End of encoder blocks

  // queries (policy/attention/wq) -> scratch 1
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = policy_d_model_;
    const int batch = N * 64;
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                   num_outputs, batch, num_inputs, &alpha,
                                   ip2_pol_w_, num_inputs, scratch0, num_inputs,
                                   &beta, scratch1, num_outputs));
    addVectors(scratch1, ip2_pol_b_, scratch1, num_outputs * batch, num_outputs,
               num_outputs * batch, NONE, stream);
  }

  // keys (policy/attention/wk) -> scratch 2
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = policy_d_model_;
    const int batch = N * 64;
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                   num_outputs, batch, num_inputs, &alpha,
                                   ip3_pol_w_, num_inputs, scratch0, num_inputs,
                                   &beta, scratch2, num_outputs));
    addVectors(scratch2, ip3_pol_b_, scratch2, num_outputs * batch, num_outputs,
               num_outputs * batch, NONE, stream);
  }

  // dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))  # constant for scaling
  // # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
  // matmul_qk = tf.matmul(queries, keys, transpose_b=True)  # Bx64x64 (from 64 queries, 64 keys)
  // policy_attn_logits = matmul_qk / dk       # Bx64x64 (64 from-squares, 64 to-squares)
  {
    // shape(keys)[-1] = policy_d_model_
    unsigned short factor = float_to_half(1.0f / sqrt((float)policy_d_model_));
    cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
                               64 /*M*/, 64 /*N*/, policy_d_model_ /*K*/,    // A/B, and M/N are swapped for row-major to col-major transform
                               &factor,                            // to handle "/ tf.math.sqrt(dk)"
                               scratch2 /*A*/,
                               CUDA_R_16F, 
                               policy_d_model_ /*LDA*/,
                               64 * policy_d_model_,    /*strideA*/
                               scratch1 /*B*/, 
                               CUDA_R_16F, 
                               policy_d_model_ /*LDB*/,
                               64 * policy_d_model_,     /*strideB*/
                               &beta, 
                               output /*C*/,  // output (policy_attn_logits)
                               CUDA_R_16F, 
                               64 /*LDC*/, 
                               64 * 64 + 8 * 24 /*strideC*/,            // leave 8*24 after each batch to interleave promotion_logits (computed below)
                               N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
  }


  // Compute promotion_logits in a single kernel (and put the result just after policy_attn_logits interleaved to get concat for free)
  half* promotion_logits = output + 64 * 64;

  ComputePromotionLogits<half>(N, policy_d_model_, promotion_logits, scratch2,
                               ip4_pol_w_, output, stream);
}


template <>
void AttentionPolicyHead<float>::Eval(
    int N, float* output, const float* input, const float* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t cudnn,
    cublasHandle_t cublas, cudaStream_t stream) {
  // convert to nhwc! TODO

  throw Exception("Not supported yet!");
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
  ReportCUDAErrors(cudaFree(ip4_pol_b_));
  for (const auto pEnc : encoder_weights_)
    delete pEnc;
}

template <typename DataType>AttentionPolicyHead<DataType>::EncoderWeights::~EncoderWeights() {
  ReportCUDAErrors(cudaFree(mha_q_w));
  ReportCUDAErrors(cudaFree(mha_q_b));
  ReportCUDAErrors(cudaFree(mha_k_w));
  ReportCUDAErrors(cudaFree(mha_k_b));
  ReportCUDAErrors(cudaFree(mha_v_w));
  ReportCUDAErrors(cudaFree(mha_v_b));
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
