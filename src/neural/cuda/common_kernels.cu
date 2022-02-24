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

#include <cassert>

#include "cuda_common.h"
#include "winograd_helper.inc"

namespace lczero {
namespace cudnn_backend {
namespace {
constexpr int kInputPlanes = 112;
}  // namespace

/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void addVectors_kernel(T* c, T* a, T* b, int size, int asize,
                                  int bsize, ActivationFunction activation) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    switch (activation) { 
      case RELU:
        if (cVal < 0) cVal = 0;
        break;
      case TANH:
        cVal = tanh(cVal);
        break;
      case SIGMOID:
        cVal = 1.0f / (1.0f + exp(-cVal));
        break;
      case SELU:
        float alpha = 1.67326324f, scale = 1.05070098f;
        if (cVal > 0)
          cVal = scale * cVal;
        else
          cVal = scale * alpha * (exp(cVal) - 1);
        break;
    }

    c[i] = (T)cVal;
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addVectors_kernel<<<blocks, kBlockSize, 0, stream>>>(c, a, b, size, asize,
                                                       bsize, activation);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void addBias_NCHW_kernel(T* c, T* a, T* b, int N, int C, int H,
                                    int W, bool relu) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int size = N * C * H * W;
  if (i < size) {
    float aVal = (float)a[i];

    // All this math can be optimized, but the kernel is memory bound anyway.
    int biasIndex = (i / (H * W)) % C;
    float bVal = (float)b[biasIndex];

    float cVal = aVal + bVal;

    if (relu && (cVal < 0)) cVal = 0;

    c[i] = (T)cVal;
  }
}

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W, bool relu, cudaStream_t stream) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addBias_NCHW_kernel<<<blocks, kBlockSize, 0, stream>>>(c, a, b, N, C, H, W, relu);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename dT, typename sT>
__device__ dT readNCHW(const sT* input_tensor, int n, int c, int h, int w,
                         int Nin, int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index;
  index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return (dT)(input_tensor[index]);
}

template <typename dT, typename sT>
__global__ void NCHWtoNHWC_kernel(dT* output_tensor,
                                  const sT* input_tensor, int Nin, int Cin,
                                  int Nout, int Cout, int H, int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= Nout * Cout * H * W) return;

  int index = tid;

  int c = (index % Cout);
  index /= Cout;
  int w = index % W;
  index /= W;
  int h = index % H;
  index /= H;
  int n = index;

  output_tensor[tid] = readNCHW<dT, sT>(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
    int Nin, int Cin, int Nout, int Cout, int H, int W) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);
  NCHWtoNHWC_kernel<<<blocks, blockSize>>>(output_tensor, input_tensor, Nin,
                                           Cin, Nout, Cout, H, W);
}

template <typename DstType, typename SrcType>
__global__ void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N) return;

  DstType el = (DstType)ip[tid];
  op[tid] = el;
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  copyTypeConverted_kernel<<<blocks, kBlockSize, 0, stream>>>(op, ip, N);
}

template <typename T>
__global__ void batchNorm_kernel(T* output, const T* input, const T* skipInput,
                                 int N, int C, int H, int W, const float* means,
                                 const float* varMultipliers, bool relu) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int wIndex = 0;
  if (sizeof(T) == sizeof(float))
    wIndex = (index / (H * W)) % C;  // NCHW for fp32.
  else
    wIndex = index % C;  // NHWC for fp16.

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  if (skipInput) el += (float)skipInput[index];

  if (relu && (el < 0)) el = 0;

  output[index] = (T)el;
}

// Every thread processes single element.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers, bool relu) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  batchNorm_kernel<<<blocks, kBlockSize>>>(output, input, skipInput, N, C, H, W,
                                           means, var_multipliers, relu);

  ReportCUDAErrors(cudaGetLastError());
}

__global__ void expandPlanes_kernel_Fp32_NCHW(float* output,
                                              const uint64_t* masks,
                                              const float* values, int n) {
  // Block size of 256, same mask/val for 64 consecutive threads.
  constexpr int kNumShmemElements = 256 / 64;

  __shared__ uint64_t shMasks[kNumShmemElements];
  __shared__ float shVals[kNumShmemElements];

  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // Load inputs to shared memory.
  if (threadIdx.x < kNumShmemElements) {
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

void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n, cudaStream_t stream) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  expandPlanes_kernel_Fp32_NCHW<<<blocks, blockSize, 0, stream>>>(output, masks,
                                                                  values, n);
  ReportCUDAErrors(cudaGetLastError());
}

// TODO: Can optimize using shared memory if this becomes a bottleneck.
__global__ void expandPlanes_kernel_Fp16_NHWC(half* output,
                                              const uint64_t* masks,
                                              const float* values, int n) {
  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= n * 8 * 8) return;

  const int planeIndex = index % kInputPlanes;
  const int boardIndex = index / (kInputPlanes * 8 * 8);
  const int sqIndex = (index / kInputPlanes) & 0x3F;

  uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

  half op = 0;
  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    float val = values[boardIndex * kInputPlanes + planeIndex];
    op = (half)val;
  }
  output[index] = op;
}

void expandPlanes_Fp16_NHWC(half* output, const uint64_t* masks,
                            const float* values, int n, cudaStream_t stream) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  expandPlanes_kernel_Fp16_NHWC<<<blocks, kBlockSize, 0, stream>>>(
      output, masks, values, n);
  ReportCUDAErrors(cudaGetLastError());
}

__global__ void expandPlanes_kernel_Fp16_NCHW(half* output,
                                              const uint64_t* masks,
                                              const float* values, int n) {
  // block size of 256, same mask/val for 64 consecutive threads
  constexpr int kNumShmemElements = 256 / 64;

  __shared__ uint64_t shMasks[kNumShmemElements];
  __shared__ half shVals[kNumShmemElements];

  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // load inputs to shared memory
  if (threadIdx.x < kNumShmemElements) {
    shMasks[threadIdx.x] = masks[planeIndex + threadIdx.x];
    shVals[threadIdx.x] = values[planeIndex + threadIdx.x];
  }
  __syncthreads();

  uint64_t mask = shMasks[threadIdx.x >> 6];

  int sqIndex = index & 0x3F;
  half op = 0;

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = (half)shVals[threadIdx.x >> 6];
  }
  output[index] = op;
}

void expandPlanes_Fp16_NCHW(half* output, const uint64_t* masks,
                            const float* values, int n, cudaStream_t stream) {
  int threads = n * 8 * 8;  // each thread writes a single element
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  expandPlanes_kernel_Fp16_NCHW<<<blocks, blockSize, 0, stream>>>(output, masks,
                                                                  values, n);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void globalScale_kernel(T* output, const T* input,
                                   const T* scaleBias, const T* prevLayerBias,
                                   int inputSize, int C) {
  const int kPlaneSize = 64;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > inputSize) return;

  int nc = tid / kPlaneSize;
  int n = nc / C;
  int c = nc % C;

  float val1 = input[tid];   // Output of residual block to be scaled.
  float val2 = output[tid];  // Skip connection to be added directly.

  if (prevLayerBias) {
    val1 += (float)(prevLayerBias[c]);
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  if (op < 0) op = 0;
  output[tid] = (T)op;
}

__global__ void globalScale_kernel_fp16_nhwc(half* output, const half* input,
                                             const half* scaleBias,
                                             const half* prevLayerBias,
                                             int inputSize, int C, int HWC) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > inputSize) return;

  int c = tid % C;
  int n = tid / (HWC);

  float val1 = (float)input[tid];   // Output of residual block to be scaled.
  float val2 = (float)output[tid];  // Skip connection to be added directly.
  if (prevLayerBias) {
    val1 += (float)prevLayerBias[c];
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  if (op < 0) op = 0;

  output[tid] = (half)op;
}

// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread writes a single output.
__global__ void globalAvgPool_kernel_NHWC_fp16(half* output, const half* input,
                                               const half* prevLayerBias,
                                               int inputSize, int outputSize) {
  const int elementsPerThread = 64;  // 8x8 board.

  int blockStart = blockIdx.x * blockDim.x;

  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * blockDim.x + threadIdx.x;
    int inputIndex = blockStart * elementsPerThread + localIndex;
    if (inputIndex < inputSize) S += (float)(input[inputIndex]);
  }

  float avg = S / elementsPerThread;

  // Add bias from previous layer.
  if (prevLayerBias) avg += (float)(prevLayerBias[threadIdx.x]);

  int opIndex = blockStart + threadIdx.x;
  if (opIndex < outputSize) output[opIndex] = (half)avg;
}

// Each thread reads 2 inputs (8x8/32), and each warp writes a single output.
template <typename T>
__global__ void globalAvgPool_kernel(T* output, const T* input,
                                     const T* prevLayerBias, int inputSize,
                                     int outputSize, int C) {
  const int elementsPerWarp = 64;
  const int elementsPerThread = 2;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int laneId = threadIdx.x & 0x1F;
  int laneStartIndex = (tid - laneId) * elementsPerThread;

  // Compute per-thread sum for elementsPerThread elements.
  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerWarp; i += 32) {
    int index = laneStartIndex + laneId + i;
    if (index < inputSize) S += (float)(input[index]);
  }

// Compute warp wide sum (for entire plane - elementsPerWarp elements).
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    S += __shfl_down_sync(0xFFFFFFFF, S, offset);
  }

  float avg = S / elementsPerWarp;
  int opIndex = tid >> 5;

  // First thread in warp has the sum, write it in output.
  if (laneId == 0) {
    if (opIndex < outputSize) {
      if (prevLayerBias) avg += (float)prevLayerBias[opIndex % C];
      output[opIndex] = (T)avg;
    }
  }
}

template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc) {
  const int kPlaneSize = 64;

  const bool fp16 = std::is_same<half, T>::value;
  if (nhwc) {
    assert(fp16);
    // For NHWC fp16, simply launch N blocks, each with C threads.
    globalAvgPool_kernel_NHWC_fp16<<<N, C>>>((half*)output, (half*)input,
                                             (half*)prevLayerBias,
                                             N * C * kPlaneSize, N * C);
  } else {
    // For NCHW layout (used with fp32),
    // each warp processes a full plane (64 elements), and writes a single
    // average N*C warps are launched.

    const int kTotalWarps = N * C;
    const int kWarpsPerBlock = 8;
    const int kBlockSize = kWarpsPerBlock * 32;

    int blocks = DivUp(kTotalWarps, kWarpsPerBlock);
    globalAvgPool_kernel<<<blocks, kBlockSize>>>(output, input, prevLayerBias,
                                                 N * C * kPlaneSize, N * C, C);
  }
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc) {
  const bool fp16 = std::is_same<half, T>::value;

  // Each thread writes one output.
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  if (nhwc) {
    assert(fp16);
    globalScale_kernel_fp16_nhwc<<<kBlocks, kBlockSize>>>(
        (half*)output, (half*)input, (half*)scaleBias, (half*)prevLayerBias,
        N * C * 8 * 8, C, 8 * 8 * C);
  } else {
    globalScale_kernel<<<kBlocks, kBlockSize>>>(
        output, input, scaleBias, prevLayerBias, N * C * 8 * 8, C);
  }
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void policyMap_kernel(T* output, const T* input,
                                 const short* indices, int N, int inputSize,
                                 int usedSize, int outputSize) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int n = tid / usedSize;
  int i = tid % usedSize;

  if (n >= N) return;

  int j = indices[i];

  if (j >= 0) {
    output[n * outputSize + j] = input[n * inputSize + i];
  }
}

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize, cudaStream_t stream) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  policyMap_kernel<T><<<kBlocks, kBlockSize, 0, stream>>>((T*)output, (T*)input,
                                               (short*)indices, N, inputSize,
                                               usedSize, outputSize);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T = float, bool use_se, bool relu, bool use_bias,
          bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          cudaStream_t stream) {
  // Each thread processes entire chess board
  if (C > kMaxResBlockFusingChannels) {
      throw Exception(
          "res block fusing opt not supported for the given data type and no "
          "of filters\n");
  } else {
    OutputTransform_SE_relu_InputTransform_kernel<float, use_se, relu, use_bias,
                                                  use_skip>
        <<<N, C, 0, stream>>>(N, C, se_K, output, input, (float*)skip, bias, w1,
                              b1, w2, b2);
  }
  ReportCUDAErrors(cudaGetLastError());
}


// N * C Tensors
// performs softmax along the C dimension
// Each thread processes one element
// Sums are computed in shared memory
// C threads per block, N blocks
template <typename T>
__global__ void softmax_kernel(T* output, const T* input) {
  int n = blockIdx.x;
  int c = threadIdx.x;
  int C = blockDim.x;
  int index = n * C + c;

  __shared__ float sum;
  if (c == 0) sum = 0;
  __syncthreads();

  // softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

  float x = (float)input[index];
  float ex = exp(x);

  // compute warp wide sums first
  float val = warpReduce(ex);

  // update shared memory sum across C dimension
  if ((c & 0x1F) == 0) atomicAdd(&sum, val);

  __syncthreads();

  float op = ex / sum;

  output[index] = (T) op;
}

template <typename T>
void Softmax(int N, int C, T* output, const T* input, cudaStream_t stream) {
  softmax_kernel<T><<<N, C, 0, stream>>>(output, input);
  ReportCUDAErrors(cudaGetLastError());
}

// N * C Tensors
// performs layer normalization along the C dimension
// Each thread processes one element
// Sums/variences are computed in shared memory
// C threads per block, N blocks
template <typename T>
__global__ void layer_norm_kernel(T* output, const T* input, const T* skip,
                                  const T* gammas, const T* betas, float ep) {
  int n = blockIdx.x;
  int c = threadIdx.x;
  int C = blockDim.x;

  __shared__ float sum, sum_sq;
  if (c == 0) {
    sum = 0;
    sum_sq = 0;
  }
  __syncthreads();

  int index = n * C + c;

  // From: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization
  // mean_i = sum(x_i[j] for j in range(k)) / k
  // var_i  = sum((x_i[j] - mean_i) ^ 2 for j in range(k)) / k
  // x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
  // output_i = x_i_normalized * gamma + beta

  float x = (float)input[index];
  if (skip) x += (float)skip[index];

  float s = warpReduce(x);
  if ((c & 0x1F) == 0) atomicAdd(&sum, s);

  __syncthreads();

  float mean = sum / C;
  float d = x - mean;
  float d_sq = d * d;

  s = warpReduce(d_sq);
  if ((c & 0x1F) == 0) atomicAdd(&sum_sq, s);
  __syncthreads();

  float var = sum_sq / C;

  float norm = d / sqrt(var + ep);
  float op = norm * (float)gammas[c] + (float)betas[c];

  output[index] = (T)op;
}

// add (optional) skip connection to input, and then perform Layer normalization
// normalization is done across C dimension (i.e, sums and std deviations taken over elements in C dim)
template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* skip,
               const T* gammas, const T* betas, float ep, cudaStream_t stream) {
  layer_norm_kernel<T><<<N, C, 0, stream>>>(output, input, skip, gammas, betas, ep);
  ReportCUDAErrors(cudaGetLastError());
}



// Compute promotion logits in a single kernel
// keys matrix is of N * 64 * C (but we use only last 8 from the 'rows' dimension, so N * 8 * C)
// ppo matrix is 4 * C (weights for dense layer / matrix multiplication)
// policy_attn_logits matrix is N * 64 * 64, but we use only 8x8 part of it from each batch dimension (so, N * 8 * 8)
// output matrix (promotion logits) is of N * 8 * 24 size
template <typename T>
__global__ void promotion_logits_kernel(int C, T* output, const T* keys,
                                        const T* ppo,
                                        const T* policy_attn_logits) {

  constexpr int output_stride = 64 * 64 + 8 * 24;
  int n = blockIdx.x;    // [0..N)
  int y = threadIdx.y;   // [0..8)
  int x = threadIdx.x;   // [0..24)     // Can split into 8 * 3

  int threadInGroup = threadIdx.y * 24 + threadIdx.x;

  // phase 1 : compute promotion_offsets by multiplying keys and ppo matrices
  const T* keys_start = keys + n * 64 * C + C * 56;      // we are interested only in last 8 out of 64 'rows' of keys matrix
  __shared__ float promotion_offsets[4][8];

  // only 32 threads out of 192 in the group are active in this phase, and each thread computes one element of the promotion_offsets matrix
  // TODO: opt idea1, can use more threads to reduce the length of the loop for the matrix multiply (do parallel reduction of partial sums later)
  //       opt idea2, the below loop for matrix mul has very poor memory access pattern, can do the loop over 32, and do parallel reductions
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;

    float S = 0;
    for (int i = 0; i < C; i++) {               // TODO: modify to loop over 32 instead of C (doing parallel reductions for the 32 sums)
      float a = (float) keys_start[y * C + i];
      float b = (float) ppo[x * C + i];  // weight matrix is transposed (col major)
      S += a * b;
    }

    // write the product (promotion_offsets) in shared memory
    promotion_offsets[x][y] = S;
  }

  __syncthreads();

  // phase 2: add the last "row" to the other 3
  // #knight offset is added to the other three
  // promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4, :]
  // Only 24 threads in the group are active in this phase
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;
    if (x < 3) {
      promotion_offsets[x][y] += promotion_offsets[3][y];
    }
  }

  __syncthreads();

  // phase 3: add 8x8 chunk of policy_attn_logits matrix to promotion offsets
  //          the output is 3x8x8 (written as 8 * 24)
  // All threads are active in this phase and they compute one element each
  int w = x / 3;
  int c = x % 3;

  // n_promo_logits = matmul_qk[:, -16:-8, -8:]  # default traversals from rank 7 to rank 8
  float n_promo_logit =
      (float)policy_attn_logits[n * output_stride + (48 + y) * 64 + (56 + w)];
  float promo_offset = promotion_offsets[c][w];

  float op = n_promo_logit + promo_offset;

  output[n * output_stride + threadInGroup] = (T)op;

}


template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
    const T* ppo, const T* policy_attn_logits,
    cudaStream_t stream) {

  // N blocks
  // 8 * 24 threads
  // Each thread computes a single output element
  dim3 blockDim(24, 8, 1);
  promotion_logits_kernel<T>
      <<<N, blockDim, 0, stream>>>(C, output, keys, ppo, policy_attn_logits);
}


// Template instantiation.
template void copyTypeConverted<half, float>(half* op, float* ip, int N,
                                             cudaStream_t stream);
template void copyTypeConverted<float, half>(float* op, half* ip, int N,
                                             cudaStream_t stream);
template void copyTypeConverted<float, float>(float* op, float* ip, int N,
                                              cudaStream_t stream);
template void copyTypeConverted<half, half>(half* op, half* ip, int N,
                                            cudaStream_t stream);

template void batchNorm<float>(float* output, const float* input,
                               const float* skipInput, int N, int C, int H,
                               int W, float* means, float* var_multipliers,
                               bool relu);
template void batchNorm<half>(half* output, const half* input,
                              const half* skipInput, int N, int C, int H, int W,
                              float* means, float* var_multipliers, bool relu);

template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, ActivationFunction act,
                                cudaStream_t stream);
template void addVectors<half>(half* c, half* a, half* b, int size, int asize,
                               int bsize, ActivationFunction act,
                               cudaStream_t stream);

template void addBias_NCHW<float>(float* c, float* a, float* b, int N, int C,
                                  int H, int W, bool relu, cudaStream_t stream);

template void addBias_NCHW<half>(half* c, half* a, half* b, int N, int C, int H,
                                 int W, bool relu, cudaStream_t stream);

template void globalAvgPool<float>(int N, int C, float* output,
                                   const float* input,
                                   const float* prevLayerBias, bool nhwc);
template void globalAvgPool<half>(int N, int C, half* output, const half* input,
                                  const half* prevLayerBias, bool nhwc);

template void globalScale<float>(int N, int C, float* output,
                                 const float* input, const float* scaleBias,
                                 const float* prevLayerBias, bool nhwc);
template void globalScale<half>(int N, int C, half* output, const half* input,
                                const half* scaleBias,
                                const half* prevLayerBias, bool nhwc);

template void PolicyMap<float>(int N, float* output, const float* input,
                               const short* indices, int inputSize,
                               int usedSize, int outputSize,
                               cudaStream_t stream);

template void PolicyMap<half>(int N, half* output, const half* input,
                              const short* indices, int inputSize, int usedSize,
                              int outputSize, cudaStream_t stream);

template void FilterTransform<float>(int N, int C, float* transformedFilter,
                                     const float* filter);

template void InputTransform<float, true>(int N, int C,
                                          float* transformed_input,
                                          const float* input,
                                          cudaStream_t stream);

template void InputTransform<float, false>(int N, int C,
                                           float* transformed_input,
                                           const float* input,
                                           cudaStream_t stream);

template void OutputTransform<float, true, true, true, true, false, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, true, true, true, false, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputTransform<float, true, true, true, true, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, true, true, true, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, true, true, false, false, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, true, true, false, false, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, false, true, false, false, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, true, true, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, false, true, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, false, true, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);


template void Softmax<half>(int N, int C, half* output, const half* input,
                            cudaStream_t stream);
template void Softmax<float>(int N, int C, float* output, const float* input,
                            cudaStream_t stream);

template void LayerNorm<half>(int N, int C, half* output, const half* input,
                              const half* skip, const half* gammas,
                              const half* betas, float ep, cudaStream_t stream);
template void LayerNorm<float>(int N, int C, float* output, const float* input,
                               const float* skip, const float* gammas,
                               const float* betas, float ep,
                               cudaStream_t stream);

template void ComputePromotionLogits<half>(int N, int C, half* output,
                                           const half* keys, const half* ppo,
                                           const half* policy_attn_logits,
                                           cudaStream_t stream);
template void ComputePromotionLogits<float>(int N, int C, float* output,
                                            const float* keys, const float* ppo,
                                            const float* policy_attn_logits,
                                            cudaStream_t stream);

template void convertNCHWtoNHWC<half, float>(half* output_tensor,
                                             const float* input_tensor, int Nin,
                                             int Cin, int Nout, int Cout, int H,
                                             int W);
template void convertNCHWtoNHWC<float, float>(float* output_tensor,
                                              const float* input_tensor,
                                              int Nin, int Cin, int Nout,
                                              int Cout, int H, int W);
template void convertNCHWtoNHWC<half, half>(half* output_tensor,
                                            const half* input_tensor, int Nin,
                                            int Cin, int Nout, int Cout, int H,
                                            int W);
}  // namespace cudnn_backend
}  // namespace lczero
