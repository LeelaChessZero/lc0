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
#include "cuda_common.h"
#include "kernels.h"
#include "layers.h"
#include <cassert>
#include <cstring>

namespace lczero {
namespace cudnn_backend {

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : C(c), H(h), W(w), input_(ip) {}

template <typename DataType>
SoftMaxLayer<DataType>::SoftMaxLayer(BaseLayer<DataType>* ip)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip) {
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
}

template <typename DataType>
void SoftMaxLayer<DataType>::Eval(int N, DataType* output,
                                  const DataType* input, const DataType* input2,
                                  void* scratch, size_t scratch_size,
                                  cudnnHandle_t cudnn, cublasHandle_t cublas) {
  float alpha = 1.0f, beta = 0.0f;

  // Need to call this at Eval as 'N' changes :-/
  if (std::is_same<half, DataType>::value) {
    cudnnSetTensor4dDescriptor(out_tensor_desc_, CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_HALF, N, GetC(), GetH(), GetW());
  } else {
    cudnnSetTensor4dDescriptor(out_tensor_desc_, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, N, GetC(), GetH(), GetW());
  }

  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out_tensor_desc_,
                      input, &beta, out_tensor_desc_, output);
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                               int filter, int Cin, bool relu, bool bias)
    : BaseLayer<DataType>(C, H, W, ip),
      filter_size_(filter),
      c_input_(Cin),
      use_relu_(relu),
      use_bias_(bias) {
  // Allocate memory for weights (filter tensor) and biases.
  size_t weight_size = sizeof(DataType) * Cin * C * filter_size_ * filter_size_;
  ReportCUDAErrors(cudaMalloc(&weights, weight_size));

  size_t blas_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases, blas_size));

  const bool fp16 = std::is_same<half, DataType>::value;

  // Create cudnn objects for various tensors, algorithms, etc.
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
  cudnnCreateTensorDescriptor(&in_tensor_desc_);
  cudnnCreateTensorDescriptor(&bias_desc_);
  cudnnCreateActivationDescriptor(&activation_);

  cudnnSetFilter4dDescriptor(filter_desc_,
                             fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
                             fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
                             GetC(), Cin, filter_size_, filter_size_);

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
      bias_desc_, fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, 1, C, 1, 1));

  int padding = filter_size_ / 2;
  const bool crossCorr = 1;

  ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
      conv_desc_, padding, padding, 1, 1, 1, 1,
      crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));

  if (fp16)
    ReportCUDNNErrors(
        cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

  // TODO: dynamic selection of algorithm!
  if ((C > 32) && (!fp16)) {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  } else {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }

  if (use_relu_) {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_RELU,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  } else {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_IDENTITY,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
}

template <>
void ConvLayer<half>::LoadWeights(float* pfilter, float* pBias, void* scratch) {
  size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  size_t blas_size = sizeof(float) * C;
  // Also need to convert from fp32 NCHW to fp16 NHWC
  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type / layout conversion using a kernel.
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpyAsync(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  fp32NCHWtofp16NHWC((half*)weights, (float*)scratch, C, c_input_, C, c_input_,
                     filter_size_, filter_size_);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpyAsync(scratch, pBias, blas_size, cudaMemcpyHostToDevice));

    copyTypeConverted((half*)biases, (float*)scratch, C);
  }
}

template <>
void ConvLayer<float>::LoadWeights(float* pfilter, float* pBias,
                                   void* scratch) {
  size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  size_t blas_size = sizeof(float) * C;
  ReportCUDAErrors(
      cudaMemcpyAsync(weights, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpyAsync(biases, pBias, blas_size, cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(cudaMemset(biases, blas_size, 0));
  }
}

template <typename DataType>
void ConvLayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t cudnn,
                               cublasHandle_t cublas) {
  const bool fp16 = std::is_same<half, DataType>::value;

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
      out_tensor_desc_, fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, N, C, H, W));

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
      in_tensor_desc_, fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, N, c_input_, H, W));

  float alpha = 1.0f, beta = 0.0f;

  if (!(use_relu_ || use_bias_)) {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
        output));
  } else if (input2) {
    // fused bias + sum + relu!
    ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &alpha, out_tensor_desc_,
        input2, bias_desc_, biases, activation_, out_tensor_desc_, output));
  } else {
    // For some reason cudnn doesn't support just Convolution + Bias with fp32
    // (winograd algorithm) it works fine when RELU is also needed which is
    // somewhat strange.
    if ((std::is_same<float, DataType>::value) && (!use_relu_)) {
      ReportCUDNNErrors(cudnnConvolutionForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output));
      // add bias
      addBias_NCHW(output, output, biases, N, C, H, W);
    } else {
      ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output, bias_desc_, biases, activation_,
          out_tensor_desc_, output));
    }
  }
}

template <typename DataType>
ConvLayer<DataType>::~ConvLayer() {
  ReportCUDAErrors(cudaFree(weights));
  ReportCUDAErrors(cudaFree(biases));
}

template <typename DataType>
BNLayer<DataType>::BNLayer(BaseLayer<DataType>* ip, bool relu)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      use_relu_(relu) {
  size_t weight_size = sizeof(float) * C;

  ReportCUDAErrors(cudaMalloc(&means_, weight_size));
  ReportCUDAErrors(cudaMalloc(&variances_, weight_size));
}

template <typename DataType>
void BNLayer<DataType>::LoadWeights(float* cpuMeans, float* cpuVar) {
  size_t weight_size = sizeof(float) * C;
  ReportCUDAErrors(
      cudaMemcpyAsync(means_, cpuMeans, weight_size, cudaMemcpyHostToDevice));
  ReportCUDAErrors(
      cudaMemcpyAsync(variances_, cpuVar, weight_size, cudaMemcpyHostToDevice));
}

template <>
void BNLayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t cudnn, cublasHandle_t cublas) {
  batchNorm(output, input, input2, N, C, H, W, means_, variances_,
                   use_relu_);
}

template <>
void BNLayer<float>::Eval(int N, float* output, const float* input,
                          const float* input2, void* scratch,
                          size_t scratch_size, cudnnHandle_t cudnn,
                          cublasHandle_t cublas) {
  batchNorm(output, input, input2, N, C, H, W, means_, variances_,
                   use_relu_);
}

template <typename DataType>
BNLayer<DataType>::~BNLayer() {
  ReportCUDAErrors(cudaFree(means_));
  ReportCUDAErrors(cudaFree(variances_));
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool relu, bool bias, bool tanh, bool sigmoid)
    : BaseLayer<DataType>(C, H, W, ip),
      use_relu_(relu),
      use_bias_(bias),
      use_tanh_(tanh),
      use_sigmoid_(sigmoid) {
  size_t weight_size =
      sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  size_t blas_size = sizeof(DataType) * C * H * W;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
  if (use_bias_) {
    ReportCUDAErrors(cudaMalloc(&biases_, blas_size));
  } else {
    biases_ = nullptr;
  }
}

template <>
void FCLayer<half>::LoadWeights(float* cpuWeight, float* cpuBias,
                                void* scratch) {
  size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  size_t weight_size = sizeof(float) * num_weights;
  size_t num_biases = C * H * W;
  size_t blas_size = sizeof(float) * num_biases;

  // also need to convert from fp32 to fp16
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpyAsync(scratch, cpuWeight, weight_size, cudaMemcpyHostToDevice));

  fp32NCHWtofp16NHWC((half*)weights_, (float*)scratch, num_biases,
                     input_->GetC(), num_biases, input_->GetC(), input_->GetH(),
                     input_->GetW());

  if (cpuBias) {
    ReportCUDAErrors(
        cudaMemcpyAsync(scratch, cpuBias, blas_size, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)biases_, (float*)scratch, num_biases);
  }
}

template <>
void FCLayer<float>::LoadWeights(float* cpuWeight, float* cpuBias,
                                 void* scratch) {
  size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  size_t weight_size = sizeof(float) * num_weights;
  size_t num_biases = C * H * W;
  size_t blas_size = sizeof(float) * num_biases;

  ReportCUDAErrors(cudaMemcpyAsync(weights_, cpuWeight, weight_size,
                                   cudaMemcpyHostToDevice));
  if (use_bias_) {
    ReportCUDAErrors(
        cudaMemcpyAsync(biases_, cpuBias, blas_size, cudaMemcpyHostToDevice));
  }
}

template <>
void FCLayer<half>::Eval(int N, half* output_tensor, const half* input_tensor,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t cudnn, cublasHandle_t cublas) {
  int num_outputs = C * H * W;
  int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  // half alpha = float2half_rn(1.0f), beta = float2half_rn(0.0f);
  std::uint16_t one_h = 0x3c00;
  std::uint16_t zero_h = 0;
  half alpha;
  half beta;
  memcpy(&alpha, &one_h, sizeof(half));
  memcpy(&beta, &zero_h, sizeof(half));
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
                          const float* input_tensor, const float* input2,
                          void* scratch, size_t scratch_size,
                          cudnnHandle_t cudnn, cublasHandle_t cublas) {
  int num_outputs = C * H * W;
  int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

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


// Template instantiation
template class ConvLayer<half>;
template class ConvLayer<float>;

template class FCLayer<half>;
template class FCLayer<float>;

template class BNLayer<half>;
template class BNLayer<float>;

template class SoftMaxLayer<half>;
template class SoftMaxLayer<float>;

// misc error handling stuff
void CudnnError(cudnnStatus_t status, const char* file, const int& line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", cudnnGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

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
    sprintf(message, "CUDNN error: %s (%s:%d) ", CublasGetErrorString(status),
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
