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
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip), C(c), H(h), W(w), nhwc_(ip->nhwc_) {}

#ifdef USE_CUDNN
template <typename DataType>
SoftMaxLayer<DataType>::SoftMaxLayer(BaseLayer<DataType>* ip)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip) {
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
}

template <typename DataType>
SoftMaxLayer<DataType>::~SoftMaxLayer() {
  cudnnDestroyTensorDescriptor(out_tensor_desc_);
}

template <typename DataType>
void SoftMaxLayer<DataType>::Eval(int N, DataType* output,
                                  const DataType* input,
                                  const DataType* /*input2*/, void* /*scratch*/,
                                  size_t /*scratch_size*/, cudnnHandle_t cudnn,
                                  cublasHandle_t /*cublas*/) {
  float alpha = 1.0f, beta = 0.0f;

  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  // Need to call this at Eval as 'N' changes :-/
  cudnnSetTensor4dDescriptor(out_tensor_desc_, layout, dataType, N, GetC(),
                             GetH(), GetW());

  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out_tensor_desc_,
                      input, &beta, out_tensor_desc_, output);
}

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
                      C * c_input_ * filter_size_ * filter_size_);
  }

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));

    copyTypeConverted((half*)biases, (float*)scratch, C);
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
                               cublasHandle_t /*cublas*/) {
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
      addBias_NCHW(output, output, biases, N, C, H, W, false);
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
  copyTypeConverted((half*)w1_, (float*)scratch, (int)num_weights1);
  if (kUseFusedSELayer && nhwc_) {
    // transposed copy for fused SE kernel
    cpuTranspose(temp.data(), w1, numFc1Out_, C);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size1, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w1_t_, (float*)scratch, (int)num_weights1);
  }

  // Weight for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, w2, weight_size2, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w2_, (float*)scratch, (int)num_weights2);
  if (kUseFusedSELayer && nhwc_) {
    cpuTranspose(temp.data(), w2, 2 * C, numFc1Out_);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size2, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w2_t_, (float*)scratch, (int)num_weights2);
  }

  // Bias for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(scratch, b1, numFc1Out_ * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b1_, (float*)scratch, numFc1Out_);

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b2_, (float*)scratch, 2 * C);

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(scratch, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
    copyTypeConverted((half*)bPrev_, (float*)scratch, C);
  }
}

template <>
void SELayer<float>::Eval(int N, float* output, const float* input,
                          const float* /*input2*/, void* scratch,
                          size_t scratch_size, cudnnHandle_t /*cudnn*/,
                          cublasHandle_t cublas) {
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
  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, true,
             false, false);

  // 3. Second fully connected layer.
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C));
  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, false, false, false);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  globalScale(N, C, output, input, op2, bPrev_, false);
}

template <>
void SELayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t /*cudnn*/, cublasHandle_t cublas) {
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
    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, true,
               false, false);

    // 3. Second fully connected layer.
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                   numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                   numFc1Out_, &beta, op2, 2 * C));
    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, false, false, false);

    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    globalScale(N, C, output, input, op2, bPrev_, nhwc_);
  }
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool relu, bool bias, bool tanh, bool sigmoid)
    : BaseLayer<DataType>(C, H, W, ip),
      use_bias_(bias),
      use_relu_(relu),
      use_tanh_(tanh),
      use_sigmoid_(sigmoid) {
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
    copyTypeConverted((half*)weights_, (float*)scratch, (int)num_weights);
  }

  if (cpuBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, cpuBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)biases_, (float*)scratch, (int)num_biases);
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
                         cublasHandle_t cublas) {
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

  if (use_bias_ || use_relu_ || use_tanh_ || use_sigmoid_) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, use_relu_, use_tanh_,
               use_sigmoid_);
  }
}

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/,
                          cudnnHandle_t /*cudnn*/, cublasHandle_t cublas) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || use_relu_ || use_tanh_ || use_sigmoid_) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, use_relu_, use_tanh_,
               use_sigmoid_);
  }
}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, int usedSize)
    : BaseLayer<DataType>(C, H, W, ip), used_size_(usedSize) {
  size_t weight_size = sizeof(short) * this->input_->GetC() * 64;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
}

template <typename DataType>
void PolicyMapLayer<DataType>::LoadWeights(const short* cpuWeight,
                                           void* /*scratch*/) {
  size_t weight_size = sizeof(short) * used_size_;

  if (nhwc_) {
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
                                    cublasHandle_t /*cublas*/) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  int outputSize = this->C * this->H * this->W;
  PolicyMap(N, output_tensor, input_tensor, weights_, inputSize, used_size_,
            outputSize);
}

template <typename DataType>
PolicyMapLayer<DataType>::~PolicyMapLayer() {
  ReportCUDAErrors(cudaFree(weights_));
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::FusedWinogradConvSELayer(
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin, bool relu, bool bias,
    bool skip_add, bool se, int se_k, bool use_gemm_ex)
    : BaseLayer<DataType>(C, H, W, ip, false),
      c_input_(Cin),
      use_relu_(relu),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      use_gemm_ex_(use_gemm_ex) {
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
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C);
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
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2);



  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2);
}

template <>
void FusedWinogradConvSELayer<half>::cublasRowMajorMatrixMul(
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
void FusedWinogradConvSELayer<float>::cublasRowMajorMatrixMul(
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
    cublasHandle_t cublas) {

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType>(N, c_input_, transformed_input, input);

  cublasRowMajorMatrixMul(transformed_input, transformed_weights_, transformed_output, N*4, C, c_input_, 36, cublas);  

  if (has_se_ && use_relu_ && use_bias_ && skip_add_)
    OutputTransform<DataType, true, true, true, true>(
        N, C, se_k_, output, transformed_output, input2, biases_, w1_, b1_, w2_,
        b2_);
  else if (!has_se_ && use_relu_ && use_bias_ && !skip_add_)
    OutputTransform<DataType, false, true, true, false>(
        N, C, 0, output, transformed_output, nullptr, biases_, nullptr, nullptr,
        nullptr, nullptr);
  else if (!has_se_ && use_relu_ && use_bias_ && skip_add_)
    OutputTransform<DataType, false, true, true, true>(
        N, C, 0, output, transformed_output, input2, biases_, nullptr, nullptr,
        nullptr, nullptr);
  else if (!has_se_ && !use_relu_ && use_bias_ && !skip_add_)
    OutputTransform<DataType, false, false, true, false>(
        N, C, 0, output, transformed_output, nullptr, biases_, nullptr, nullptr,
        nullptr, nullptr);
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
    : BaseLayer<DataType>(C, H, W, ip, false),
      c_input_(Cin),
      use_relu_(relu),
      use_bias_(bias),
      use_gemm_ex_(use_gemm_ex) {
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
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C);
  }
}

template <>
void Conv1Layer<half>::cublasRowMajorMatrixMul(const half* A, const half* B,
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
      N * K, A, CUDA_R_16F, K, 0, &halfZero, Out, CUDA_R_16F, N, N * M,
      batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
}

template <>
void Conv1Layer<float>::cublasRowMajorMatrixMul(const float* A, const float* B,
                                                float* Out, int M, int N, int K,
                                                int batchSize,
                                                cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;
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
                                cudnnHandle_t /*cudnn*/,
                                cublasHandle_t cublas) {
  cublasRowMajorMatrixMul(weights_, input, output, C, H * W, c_input_, N,
                          cublas);

  if (use_bias_)
    addBias_NCHW(output, output, biases_, N, C, H, W, use_relu_);
  else if (use_relu_)
    addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W,
               0, use_relu_, false, false);
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
  ReportCUDAErrors(cudaFree(weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(
    BaseLayer<DataType>* ip, int C, bool se, int se_k, bool use_gemm_ex, bool first, bool last)
    : BaseLayer<DataType>(C, 8, 8, ip),
      has_se_(se),
      se_k_(se_k),
      use_gemm_ex_(use_gemm_ex),
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
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases0_, (float*)scratch, C);
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
  copyTypeConverted((DataType*)weights, (float*)scratch, C * C * 3 * 3);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases1_, (float*)scratch, C);
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
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2);

  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2);
}

template <>
void ResidualBlock<half>::cublasRowMajorMatrixMul(
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
void ResidualBlock<float>::cublasRowMajorMatrixMul(
    const float* A, const float* B, float* Out, int M, int N, int K,
    int batchSize, cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;
  if (use_gemm_ex_)
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
        N * K, A, CUDA_R_32F, K, K * M, &floatZero, Out, CUDA_R_32F, N, N * M,
        batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  else
    ReportCUBLASErrors(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
        K * M, &floatZero, Out, N, N * M, batchSize));
}

template <typename DataType>
void ResidualBlock<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* /*input2*/,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas) {

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
    InputTransform<DataType>(N, c_input_, transformed_input, input);

    cublasRowMajorMatrixMul(transformed_input, transformed_weights0_,
                            transformed_output, N * 4, C, c_input_, 36, cublas);
  } else {
    cublasRowMajorMatrixMul(output, transformed_weights0_,
                            transformed_output, N * 4, C, c_input_, 36, cublas);
 
  }

  OutputInputTransform<DataType, false, true, true, false>(
      N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
      nullptr, nullptr, nullptr, nullptr);
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  cublasRowMajorMatrixMul(transformed_input, transformed_weights1_,
                          transformed_output, N * 4, C, C, 36, cublas);

  if (last_block_) {
    if (has_se_)
      OutputTransform<DataType, true, true, true, true>(
          N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_);
    else
      OutputTransform<DataType, false, true, true, true>(
          N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_);
  } else {
    if (has_se_)
      OutputInputTransform<DataType, true, true, true, true>(
          N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_);
    else
      OutputInputTransform<DataType, false, true, true, true>(
          N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
          w2_, b2_);
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


// Template instantiation.
#ifdef USE_CUDNN
template class ConvLayer<half>;
template class ConvLayer<float>;
#endif

template class FCLayer<half>;
template class FCLayer<float>;

#ifdef USE_CUDNN
template class SoftMaxLayer<half>;
template class SoftMaxLayer<float>;
#endif

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
