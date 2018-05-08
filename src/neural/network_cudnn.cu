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
*/
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include "neural/factory.h"
#include "utils/bititer.h"
#include "utils/exception.h"

#include <cublas_v2.h>
#include <cudnn.h>

#define DEBUG_RAW_NPS 0

namespace lczero {
namespace {

void cudnnError(cudnnStatus_t status, const char *file, const int &line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", cudnnGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

const char *cublasGetErrorString(cublasStatus_t status) {
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

void cublasError(cublasStatus_t status, const char *file, const int &line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", cublasGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

void cudaError(cudaError_t status, const char *file, const int &line) {
  if (status != cudaSuccess) {
    char message[128];
    sprintf(message, "CUDA error: %s (%s:%d) ", cudaGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

#define reportCUDNNErrors(status) cudnnError(status, __FILE__, __LINE__)
#define reportCUBLASErrors(status) cublasError(status, __FILE__, __LINE__)
#define reportCUDAErrors(status) cudaError(status, __FILE__, __LINE__)

// 256 MB fixed scratch memory size (hardcoded for now)
static constexpr int kCudaScratchSize = 256 * 1024 * 1024;

// hard-coded for now, no point in going above this anyway (can possibly save
// memory by reducing this)
static constexpr int kMaxBatchSize = 1024;

static constexpr int kNumOutputPolicy = 1858;

// the Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval

class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }

  BaseLayer(int c, int h, int w, BaseLayer *ip);
  size_t GetOutputSize(int N) const { return bpe_ * N * C * H * W; }

  // input2 is optional (skip connection)
  virtual void Eval(int N, float *output, const float *input,
                    const float *input2, float *scratch, cudnnHandle_t cudnn,
                    cublasHandle_t cublas) = 0;

 protected:
  static bool fp16_;
  static size_t bpe_;  // size of each element
  BaseLayer *input_;

  int C;  // output tensor dimensions
  int H;
  int W;
};

class ConvLayer : public BaseLayer {
 public:
  ConvLayer(BaseLayer *ip, int C, int H, int W, int size, int Cin,
            bool relu = false, bool bias = false);
  ~ConvLayer();
  void LoadWeights(float *pfilter, float *pBias = nullptr);
  void Eval(int N, float *output, const float *input, const float *input2,
            float *scratch, cudnnHandle_t cudnn,
            cublasHandle_t cublas) override;

 private:
  const int c_input_;
  const int filter_size_;
  const bool use_relu_;
  const bool use_bias_;

  float *biases = nullptr;
  float *weights = nullptr;

  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t convAlgo;

  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t in_tensor_desc_;
  cudnnTensorDescriptor_t out_tensor_desc_;
  cudnnActivationDescriptor_t activation_;
};

class SoftMaxLayer : public BaseLayer {
 public:
  SoftMaxLayer(BaseLayer *ip);
  void Eval(int N, float *output, const float *input, const float *input2,
            float *scratch, cudnnHandle_t cudnn,
            cublasHandle_t cublas) override;

 private:
  cudnnTensorDescriptor_t out_tensor_desc_;
};

class BNLayer : public BaseLayer {
 public:
  BNLayer(BaseLayer *ip, bool relu);
  ~BNLayer();

  void LoadWeights(float *cpuMeans, float *cpuVar);
  void Eval(int N, float *output, const float *input, const float *input2,
            float *scratch, cudnnHandle_t cudnn,
            cublasHandle_t cublas) override;

 private:
  const bool use_relu_;
  float *means_ = nullptr;
  float *variances_ = nullptr;
};

class FCLayer : public BaseLayer {
 public:
  FCLayer(BaseLayer *ip, int C, int H, int W, bool relu, bool bias,
          bool tanh = false);
  ~FCLayer();

  void LoadWeights(float *cpuWeight, float *cpuBias);
  void Eval(int N, float *output, const float *input, const float *input2,
            float *scratch, cudnnHandle_t cudnn,
            cublasHandle_t cublas) override;

 private:
  const bool use_bias_;
  const bool use_relu_;
  const bool use_tanh_;
  float *weights_ = nullptr;
  float *biases_ = nullptr;
};

// Need memory for 3 data buffers
//  1. input for the layer
//  2. output of the layer
//  3. data from old layer for skip connection

/////////////////////////////////////////////////////////////////////////////
//                      Static variable Definations                        //
/////////////////////////////////////////////////////////////////////////////

// TODO: fp16 support
bool BaseLayer::fp16_ = false;
size_t BaseLayer::bpe_ = sizeof(float);

int divUp(int a, int b) { return (a + b - 1) / b; }

/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void addVectors_kernel(T *c, T *a, T *b, int size, int asize,
                                  int bsize, bool relu, bool useTanh) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    T aVal = 0;
    T bVal = 0;
    if (a) aVal = a[i % asize];
    if (b) bVal = b[i % bsize];

    T cVal = aVal + bVal;

    if (relu && (cVal < 0)) cVal = 0;

    if (useTanh) {
      // Ankan: actually it's sigmoid in leela-zero main branch??
      // see code in Network.cpp
      //    auto winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;
      // Different from lc0 branch? WHY ???
      // cVal = (1.0f + tanh(cVal)) / 2.0f;
      cVal = tanh(cVal);
    }

    c[i] = cVal;
  }
}

// adds two vectors (possibly of different sizes), also do optional relu
// activation_
template <typename T>
void addVectors(T *c, T *a, T *b, int size, int asize, int bsize, bool relu,
                bool useTanh) {
  const int blockSize = 256;
  int blocks = divUp(size, blockSize);

  addVectors_kernel<<<blocks, blockSize>>>(c, a, b, size, asize, bsize, relu,
                                           useTanh);
  reportCUDAErrors(cudaGetLastError());
}

__global__ void batchNormForward_kernel(float *output, const float *input,
                                        const float *skipInput, int N, int C,
                                        int H, int W, const float *means,
                                        const float *varMultipliers,
                                        bool relu) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int wIndex = (index / (H * W)) % C;

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  // TODO: figure out order of relu and skip connection
  if (skipInput) el += skipInput[index];

  if (relu && (el < 0)) el = 0;

  output[index] = el;
}

// works only on NCHW tensors
// each thread processes single element
void batchNormForward(float *output, const float *input, const float *skipInput,
                      int N, int C, int H, int W, float *means,
                      float *varMultipliers, bool relu) {
  int totalElements = N * C * H * W;
  const int blockSize = 256;
  int blocks = divUp(totalElements, blockSize);

  batchNormForward_kernel<<<blocks, blockSize>>>(
      output, input, skipInput, N, C, H, W, means, varMultipliers, relu);

  reportCUDAErrors(cudaGetLastError());
}

__global__ void expandPlanes_kernel(float *output, const uint64_t *masks,
                                    const float *values, int n) {
  // block size of 256, same mask/val for 64 consecutive threads
  constexpr int kNumShmemElments = 256 / 64;

  __shared__ uint64_t shMasks[kNumShmemElments];
  __shared__ float shVals[kNumShmemElments];

  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // load inputs to shared memory
  if (threadIdx.x < kNumShmemElments) {
    shMasks[threadIdx.x] = masks[planeIndex + threadIdx.x];
    shVals[threadIdx.x] = values[planeIndex + threadIdx.x];
  }
  __syncthreads();

  uint64_t mask = shMasks[threadIdx.x >> 6];

  int sqIndex = index & 0x3F;
  float op = 0;

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = shVals[threadIdx.x >> 6];
  }
  output[index] = op;
}
void expandPlanes(float *output, const uint64_t *masks, const float *values,
                  int n) {
  int threads = n * 8 * 8;  // each thread writes a single element
  const int blockSize = 256;
  int blocks = divUp(threads, blockSize);

  expandPlanes_kernel<<<blocks, blockSize>>>(output, masks, values, n);

  reportCUDAErrors(cudaGetLastError());
}

BaseLayer::BaseLayer(int c, int h, int w, BaseLayer *ip)
    : C(c), H(h), W(w), input_(ip) {}

SoftMaxLayer::SoftMaxLayer(BaseLayer *ip)
    : BaseLayer(ip->GetC(), ip->GetH(), ip->GetW(), ip) {
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
}

void SoftMaxLayer::Eval(int N, float *output, const float *input,
                        const float *input2, float *scratch,
                        cudnnHandle_t cudnn, cublasHandle_t cublas) {
  float alpha = 1.0f, beta = 0.0f;

  // need to call this at Eval as 'N' changes :-/
  cudnnSetTensor4dDescriptor(
      out_tensor_desc_, fp16_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16_ ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, N, GetC(), GetH(), GetW());

  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out_tensor_desc_,
                      input, &beta, out_tensor_desc_, output);
}

ConvLayer::ConvLayer(BaseLayer *ip, int C, int H, int W, int filter, int Cin,
                     bool relu, bool bias)
    : BaseLayer(C, H, W, ip),
      filter_size_(filter),
      c_input_(Cin),
      use_relu_(relu),
      use_bias_(bias) {
  // allocate memory for weights (filter tensor) and biases
  size_t weightSize = bpe_ * Cin * C * filter_size_ * filter_size_;
  reportCUDAErrors(cudaMalloc(&weights, weightSize));

  size_t biasSize = bpe_ * C;
  reportCUDAErrors(cudaMalloc(&biases, biasSize));

  // create cudnn objects for various tensors, algorithms, etc
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
  cudnnCreateTensorDescriptor(&in_tensor_desc_);
  cudnnCreateTensorDescriptor(&bias_desc_);
  cudnnCreateActivationDescriptor(&activation_);

  cudnnSetFilter4dDescriptor(
      filter_desc_, fp16_ ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
      fp16_ ? CUDNN_TENSOR_NHWC
            : CUDNN_TENSOR_NCHW,  // TODO: support fp16 evaluation
      GetC(), Cin, filter_size_, filter_size_);

  reportCUDNNErrors(cudnnSetTensor4dDescriptor(
      bias_desc_, fp16_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16_ ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, 1, C, 1, 1));

  int padding = filter_size_ / 2;
  const bool crossCorr = 1;

  cudnnSetConvolution2dDescriptor(
      conv_desc_, padding, padding, 1, 1, 1, 1,
      crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION,
      fp16_ ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT);

  // TODO: dynamic selection of algorithm!
  if (C > 32) {
    convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  } else {
    convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }

  if (use_relu_) {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_RELU,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  } else {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_IDENTITY,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
}

void ConvLayer::LoadWeights(float *pfilter, float *pBias) {
  size_t weightSize = bpe_ * c_input_ * C * filter_size_ * filter_size_;
  reportCUDAErrors(
      cudaMemcpyAsync(weights, pfilter, weightSize, cudaMemcpyHostToDevice));

  size_t biasSize = bpe_ * C;
  if (pBias) {
    reportCUDAErrors(
        cudaMemcpyAsync(biases, pBias, biasSize, cudaMemcpyHostToDevice));
  } else {
    reportCUDAErrors(cudaMemset(biases, biasSize, 0));
  }
}

void ConvLayer::Eval(int N, float *output, const float *input,
                     const float *input2, float *scratch, cudnnHandle_t cudnn,
                     cublasHandle_t cublas) {
  reportCUDNNErrors(cudnnSetTensor4dDescriptor(
      out_tensor_desc_, fp16_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16_ ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, N, C, H, W));

  reportCUDNNErrors(cudnnSetTensor4dDescriptor(
      in_tensor_desc_, fp16_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16_ ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, N, c_input_, H, W));

  float alpha = 1.0f, beta = 0.0f;

  if (!(use_relu_ || use_bias_)) {
    reportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, convAlgo, scratch, kCudaScratchSize, &beta,
        out_tensor_desc_, output));
  } else if (input2) {
    // fused bias + sum + relu!
    reportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, convAlgo, scratch, kCudaScratchSize, &alpha,
        out_tensor_desc_, input2, bias_desc_, biases, activation_,
        out_tensor_desc_, output));
  } else {
    reportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, convAlgo, scratch, kCudaScratchSize, &beta,
        out_tensor_desc_, output, bias_desc_, biases, activation_,
        out_tensor_desc_, output));
  }
}

ConvLayer::~ConvLayer() {
  reportCUDAErrors(cudaFree(weights));
  reportCUDAErrors(cudaFree(biases));
}

BNLayer::BNLayer(BaseLayer *ip, bool relu)
    : BaseLayer(ip->GetC(), ip->GetH(), ip->GetW(), ip), use_relu_(relu) {
  size_t weightSize = bpe_ * C;

  reportCUDAErrors(cudaMalloc(&means_, weightSize));
  reportCUDAErrors(cudaMalloc(&variances_, weightSize));
}

void BNLayer::LoadWeights(float *cpuMeans, float *cpuVar) {
  size_t weightSize = bpe_ * C;
  reportCUDAErrors(
      cudaMemcpyAsync(means_, cpuMeans, weightSize, cudaMemcpyHostToDevice));
  reportCUDAErrors(
      cudaMemcpyAsync(variances_, cpuVar, weightSize, cudaMemcpyHostToDevice));
}

void BNLayer::Eval(int N, float *output, const float *input,
                   const float *input2, float *scratch, cudnnHandle_t cudnn,
                   cublasHandle_t cublas) {
  batchNormForward(output, input, input2, N, C, H, W, means_, variances_,
                   use_relu_);
}

BNLayer::~BNLayer() {
  reportCUDAErrors(cudaFree(means_));
  reportCUDAErrors(cudaFree(variances_));
}

FCLayer::FCLayer(BaseLayer *ip, int C, int H, int W, bool relu, bool bias,
                 bool tanh)
    : BaseLayer(C, H, W, ip),
      use_relu_(relu),
      use_bias_(bias),
      use_tanh_(tanh) {
  size_t weightSize = bpe_ * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  size_t biasSize = bpe_ * C * H * W;
  reportCUDAErrors(cudaMalloc(&weights_, weightSize));
  if (use_bias_) {
    reportCUDAErrors(cudaMalloc(&biases_, biasSize));
  } else {
    biases_ = nullptr;
  }
}

void FCLayer::LoadWeights(float *cpuWeight, float *cpuBias) {
  size_t weightSize =
      bpe_ * C * H * W * input_->GetC() * input_->GetH() * input_->GetW();

  reportCUDAErrors(
      cudaMemcpyAsync(weights_, cpuWeight, weightSize, cudaMemcpyHostToDevice));
  if (use_bias_) {
    size_t biasSize = bpe_ * C * H * W;
    reportCUDAErrors(
        cudaMemcpyAsync(biases_, cpuBias, biasSize, cudaMemcpyHostToDevice));
  }
}

void FCLayer::Eval(int N, float *outputTensor, const float *inputTensor,
                   const float *input2, float *scratch, cudnnHandle_t cudnn,
                   cublasHandle_t cublas) {
  float alpha = 1.0f, beta = 0.0f;
  int numOutputs = C * H * W;
  int numInputs = input_->GetC() * input_->GetH() * input_->GetW();

  if (fp16_) {
    // TODO: implement this!
    assert(0);
  } else {
    reportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, numOutputs,
                                   N, numInputs, &alpha, weights_, numInputs,
                                   inputTensor, numInputs, &beta, outputTensor,
                                   numOutputs));

    if (use_bias_ || use_relu_ || use_tanh_) {
      addVectors(outputTensor, biases_, outputTensor, numOutputs * N,
                 numOutputs, numOutputs * N, use_relu_, use_tanh_);
    }
  }
}

FCLayer::~FCLayer() {
  reportCUDAErrors(cudaFree(weights_));
  reportCUDAErrors(cudaFree(biases_));
}

class CudnnNetwork;

struct InputsOutputs {
  InputsOutputs() {
    reportCUDAErrors(cudaHostAlloc(
        &input_masks_mem_, kMaxBatchSize * kInputPlanes * sizeof(uint64_t),
        cudaHostAllocMapped));
    reportCUDAErrors(
        cudaHostGetDevicePointer(&input_masks_mem_gpu_, input_masks_mem_, 0));

    reportCUDAErrors(cudaHostAlloc(&input_val_mem_,
                                   kMaxBatchSize * kInputPlanes * sizeof(float),
                                   cudaHostAllocMapped));
    reportCUDAErrors(
        cudaHostGetDevicePointer(&input_val_mem_gpu_, input_val_mem_, 0));

    reportCUDAErrors(cudaHostAlloc(
        &op_policy_mem_, kMaxBatchSize * kNumOutputPolicy * sizeof(float),
        cudaHostAllocMapped));
    reportCUDAErrors(
        cudaHostGetDevicePointer(&op_policy_mem_gpu_, op_policy_mem_, 0));

    reportCUDAErrors(cudaHostAlloc(
        &op_value_mem_, kMaxBatchSize * sizeof(float), cudaHostAllocMapped));
    reportCUDAErrors(
        cudaHostGetDevicePointer(&op_value_mem_gpu_, op_value_mem_, 0));
  }
  ~InputsOutputs() {
    reportCUDAErrors(cudaFreeHost(input_masks_mem_));
    reportCUDAErrors(cudaFreeHost(input_val_mem_));
    reportCUDAErrors(cudaFreeHost(op_policy_mem_));
    reportCUDAErrors(cudaFreeHost(op_value_mem_));
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

class CudnnNetworkComputation : public NetworkComputation {
 public:
  CudnnNetworkComputation(CudnnNetwork *network);
  ~CudnnNetworkComputation();

  void AddInput(InputPlanes &&input) override {
    auto iterMask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    auto iterVal = &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    int i = 0;
    for (const auto &plane : input) {
      iterMask[i] = plane.mask;
      iterVal[i] = plane.value;
      i++;
    }

    batch_size_++;
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    return inputs_outputs_->op_value_mem_[sample];
  }
  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
  }

 private:
  // memory holding inputs, outputs
  std::unique_ptr<InputsOutputs> inputs_outputs_;
  int batch_size_;

  CudnnNetwork *network_;
};

class CudnnNetwork : public Network {
 public:
  CudnnNetwork(Weights weights, const OptionsDict &options) {
    gpuId_ = options.GetOrDefault<int>("gpu", 0);

    int totalGPUs;
    reportCUDAErrors(cudaGetDeviceCount(&totalGPUs));

    if (gpuId_ >= totalGPUs)
      throw Exception("Invalid GPU Id: " + std::to_string(gpuId_));

    // select GPU to run on (for *the current* thread)
    reportCUDAErrors(cudaSetDevice(gpuId_));

    reportCUDNNErrors(cudnnCreate(&cudnn_));
    reportCUBLASErrors(cublasCreate(&cublas_));

    const int numInputPlanes = kInputPlanes;
    const int numFilters = weights.input.biases.size();

    numBlocks_ = weights.residual.size();

    // 0. process weights
    processConvBlock(weights.input, true);
    for (auto i = size_t{0}; i < numBlocks_; i++) {
      processConvBlock(weights.residual[i].conv1, true);
      processConvBlock(weights.residual[i].conv2, true);
    }
    processConvBlock(weights.policy);
    processConvBlock(weights.value);

    // 1. build the network, and copy the weights to GPU memory
    // input
    {
      auto inputConv = std::make_unique<ConvLayer>(nullptr, numFilters, 8, 8, 3,
                                                   numInputPlanes, true, true);
      inputConv->LoadWeights(&weights.input.weights[0],
                             &weights.input.biases[0]);
      network_.emplace_back(std::move(inputConv));
    }

    // residual block
    for (int block = 0; block < weights.residual.size(); block++) {
      auto conv1 = std::make_unique<ConvLayer>(getLastLayer(), numFilters, 8, 8,
                                               3, numFilters, true, true);
      conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                         &weights.residual[block].conv1.biases[0]);
      network_.emplace_back(std::move(conv1));

      auto conv2 = std::make_unique<ConvLayer>(getLastLayer(), numFilters, 8, 8,
                                               3, numFilters, true, true);
      conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                         &weights.residual[block].conv2.biases[0]);
      network_.emplace_back(std::move(conv2));
    }

    resi_last_ = getLastLayer();

    // policy head
    {
      auto convPol = std::make_unique<ConvLayer>(
          resi_last_, weights.policy.bn_means.size(), 8, 8, 1, numFilters);
      convPol->LoadWeights(&weights.policy.weights[0]);
      network_.emplace_back(std::move(convPol));

      auto BNPol = std::make_unique<BNLayer>(getLastLayer(), true);
      BNPol->LoadWeights(&weights.policy.bn_means[0],
                         &weights.policy.bn_stddivs[0]);
      network_.emplace_back(std::move(BNPol));

      auto FCPol = std::make_unique<FCLayer>(
          getLastLayer(), weights.ip_pol_b.size(), 1, 1, false, true);
      FCPol->LoadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0]);
      network_.emplace_back(std::move(FCPol));

      auto softmaxPol = std::make_unique<SoftMaxLayer>(getLastLayer());
      network_.emplace_back(std::move(softmaxPol));
    }
    policy_out_ = getLastLayer();

    // Value head
    {
      auto convVal = std::make_unique<ConvLayer>(
          resi_last_, weights.value.bn_means.size(), 8, 8, 1, numFilters);
      convVal->LoadWeights(&weights.value.weights[0]);
      network_.emplace_back(std::move(convVal));

      auto BNVal = std::make_unique<BNLayer>(getLastLayer(), true);
      BNVal->LoadWeights(&weights.value.bn_means[0],
                         &weights.value.bn_stddivs[0]);
      network_.emplace_back(std::move(BNVal));

      auto FCVal1 = std::make_unique<FCLayer>(
          getLastLayer(), weights.ip1_val_b.size(), 1, 1, true, true);
      FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0]);
      network_.emplace_back(std::move(FCVal1));

      auto FCVal2 =
          std::make_unique<FCLayer>(getLastLayer(), 1, 1, 1, false, true, true);
      FCVal2->LoadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0]);
      network_.emplace_back(std::move(FCVal2));
    }
    value_out_ = getLastLayer();

    // 2. allocate GPU memory for running the network
    //    - three buffers of max size are enough (one to hold input, second to
    //    hold output and third to hold skip connection's input)
    size_t maxSize = resi_last_->GetOutputSize(kMaxBatchSize);
    for (auto &mem : tensor_mem_) {
      reportCUDAErrors(cudaMalloc(&mem, maxSize));
      reportCUDAErrors(cudaMemset(mem, 0, maxSize));
    }

    // printf("Allocated %d bytes of GPU memory to run the network\n", 3 *
    // maxSize);

    // 3. allocate scratch space (used internally by cudnn to run convolutions)
    reportCUDAErrors(cudaMalloc(&scratch_mem_, kCudaScratchSize));
  }

  void forwardEval(InputsOutputs *io, int batchSize) {
    std::lock_guard<std::mutex> lock(lock_);

#if DEBUG_RAW_NPS == 1
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // expand packed planes to full planes
    uint64_t *ipDataMasks = io->input_masks_mem_gpu_;
    float *ipDataValues = io->input_val_mem_gpu_;
    expandPlanes(tensor_mem_[0], ipDataMasks, ipDataValues,
                 batchSize * kInputPlanes);

    float *opPol = io->op_policy_mem_gpu_;
    float *opVal = io->op_value_mem_gpu_;

    int l = 0;
    // input
    network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0], nullptr,
                        scratch_mem_, cudnn_, cublas_);  // input conv

    // residual block
    for (int block = 0; block < numBlocks_; block++) {
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                          scratch_mem_, cudnn_, cublas_);  // conv1
      network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0],
                          tensor_mem_[2], scratch_mem_, cudnn_,
                          cublas_);  // conv2
    }

    // policy head
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                        scratch_mem_, cudnn_, cublas_);  // pol conv
    network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                        scratch_mem_, cudnn_, cublas_);  // pol BN
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[1], nullptr,
                        scratch_mem_, cudnn_, cublas_);  // pol FC
    network_[l++]->Eval(batchSize, opPol, tensor_mem_[0], nullptr, scratch_mem_,
                        cudnn_,
                        cublas_);  // pol softmax  // POLICY

    // value head
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                        scratch_mem_, cudnn_, cublas_);  // value conv
    network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0], nullptr,
                        scratch_mem_, cudnn_, cublas_);  // value BN
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                        scratch_mem_, cudnn_, cublas_);  // value FC1
    network_[l++]->Eval(batchSize, opVal, tensor_mem_[0], nullptr, scratch_mem_,
                        cudnn_,
                        cublas_);  // value FC2    // VALUE

    reportCUDAErrors(cudaDeviceSynchronize());

#if DEBUG_RAW_NPS == 1
    const int reportingCalls = 100;
    static int numCalls = 0;
    static int sumBatchSize = 0;
    static double totalTime = 0;

    sumBatchSize += batchSize;
    numCalls++;

    auto t_end = std::chrono::high_resolution_clock::now();

    double dt = std::chrono::duration<double>(t_end - t_start).count();
    totalTime += dt;
    if (numCalls == reportingCalls) {
      double avgBatchSize = ((double)sumBatchSize) / numCalls;
      printf("\nAvg batch size: %lf, NN eval time: %lf seconds per %d evals\n",
             avgBatchSize, totalTime, sumBatchSize);
      sumBatchSize = 0;
      totalTime = 0;
      numCalls = 0;
    }
#endif
  }

  ~CudnnNetwork() {
    for (auto mem : tensor_mem_) {
      if (mem) reportCUDAErrors(cudaFree(mem));
    }
    if (scratch_mem_) reportCUDAErrors(cudaFree(scratch_mem_));
    cudnnDestroy(cudnn_);
    cublasDestroy(cublas_);
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    // set correct gpu id for this computation (as it might have been called
    // from a different thread)
    reportCUDAErrors(cudaSetDevice(gpuId_));
    return std::make_unique<CudnnNetworkComputation>(this);
  }

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>();
    } else {
      std::unique_ptr<InputsOutputs> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

 private:
  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;
  int gpuId_;

  // currently only one NN Eval can happen a time (we can fix this if needed by
  // allocating more memory)
  mutable std::mutex lock_;

  int numBlocks_;
  std::vector<std::unique_ptr<BaseLayer>> network_;
  BaseLayer *getLastLayer() { return network_.back().get(); }

  BaseLayer *resi_last_;
  BaseLayer *policy_out_;
  BaseLayer *value_out_;

  float *tensor_mem_[3];
  float *scratch_mem_;

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;

  void processConvBlock(Weights::ConvBlock &block, bool foldBNLayer = false) {
    const float epsilon = 1e-5f;

    // compute reciprocal of std-dev from the variances (so that it can be just
    // multiplied)
    std::vector<float> &stddev = block.bn_stddivs;
    for (auto &&w : stddev) {
      w = 1.0f / std::sqrt(w + epsilon);
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    for (auto j = size_t{0}; j < block.bn_means.size(); j++) {
      block.bn_means[j] -= block.biases[j];
      block.biases[j] = 0.0f;
    }

    // get rid of the BN layer by adjusting weights and biases of the
    // convolution idea proposed by Henrik Forstén and first implemented in
    // leela go zero
    if (foldBNLayer) {
      const int outputs = block.biases.size();
      const int channels = block.weights.size() / (outputs * 3 * 3);

      for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
          for (auto i = 0; i < 9; i++) {
            block.weights[o * channels * 9 + c * 9 + i] *= block.bn_stddivs[o];
          }
        }

        block.bn_means[o] *= block.bn_stddivs[o];
        block.bn_stddivs[o] = 1.0f;

        // Move means to convolution biases
        block.biases[o] = -block.bn_means[o];
        block.bn_means[o] = 0.0f;
      }
    }
  }
};

CudnnNetworkComputation::CudnnNetworkComputation(CudnnNetwork *network)
    : network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

CudnnNetworkComputation::~CudnnNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void CudnnNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

}  // namespace

REGISTER_NETWORK("cudnn", CudnnNetwork, 110);

}  // namespace lczero
