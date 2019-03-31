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

#include "cuda_common.h"
#include "neural/network.h"

namespace lczero {
namespace cudnn_backend {

/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void addVectors_kernel(T* c, T* a, T* b, int size, int asize,
                                  int bsize, bool relu, bool useTanh,
                                  bool useSigmoid) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    if (relu && (cVal < 0)) cVal = 0;

    if (useTanh) {
      cVal = tanh(cVal);
    }

    if (useSigmoid) {
      cVal = 1.0f / (1.0f + exp(-cVal));
    }

    c[i] = (T)cVal;
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize, bool relu,
                bool use_tanh, bool use_sigmoid) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addVectors_kernel<<<blocks, kBlockSize>>>(c, a, b, size, asize, bsize, relu,
                                            use_tanh, use_sigmoid);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void addBias_NCHW_kernel(T* c, T* a, T* b, int N, int C, int H,
                                    int W) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int size = N * C * H * W;
  if (i < size) {
    float aVal = (float)a[i];

    // All this math can be optimized, but the kernel is memory bound anyway.
    int biasIndex = (i / (H * W)) % C;
    float bVal = (float)b[biasIndex];

    float cVal = aVal + bVal;
    c[i] = (T)cVal;
  }
}

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addBias_NCHW_kernel<<<blocks, kBlockSize>>>(c, a, b, N, C, H, W);
  ReportCUDAErrors(cudaGetLastError());
}

__device__ half readNCHW(float* input_tensor, int n, int c, int h, int w,
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

  return (half)(input_tensor[index]);
}

__global__ void fp32NCHWtofp16NHWC_kernel(half* output_tensor,
                                          float* input_tensor, int Nin, int Cin,
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

  output_tensor[tid] = readNCHW(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

void fp32NCHWtofp16NHWC(half* output_tensor, float* input_tensor, int Nin,
                        int Cin, int Nout, int Cout, int H, int W) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);
  fp32NCHWtofp16NHWC_kernel<<<blocks, blockSize>>>(output_tensor, input_tensor,
                                                   Nin, Cin, Nout, Cout, H, W);
}

template <typename DstType, typename SrcType>
__global__ void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N) return;

  DstType el = (DstType)ip[tid];
  op[tid] = el;
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  copyTypeConverted_kernel<<<blocks, kBlockSize>>>(op, ip, N);
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
  constexpr int kNumShmemElments = 256 / 64;

  __shared__ uint64_t shMasks[kNumShmemElments];
  __shared__ float shVals[kNumShmemElments];

  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // Load inputs to shared memory.
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

void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  expandPlanes_kernel_Fp32_NCHW<<<blocks, blockSize>>>(output, masks, values,
                                                       n);
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
                            const float* values, int n) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  expandPlanes_kernel_Fp16_NHWC<<<blocks, kBlockSize>>>(output, masks, values,
                                                        n);
  ReportCUDAErrors(cudaGetLastError());
}

__global__ void globalScale_kernel(float* output, const float* input,
                                   const float* scaleBias,
                                   const float* prevLayerBias, int inputSize,
                                   int C) {
  const int kPlaneSize = 64;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > inputSize) return;

  int nc = tid / kPlaneSize;
  int n = nc / C;
  int c = nc % C;

  float val1 = input[tid];   // Output of residual block to be scaled.
  float val2 = output[tid];  // Skip connection to be added directly.

  if (prevLayerBias) {
    val1 += prevLayerBias[c];
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  if (op < 0) op = 0;
  output[tid] = op;
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
__global__ void globalAvgPool_kernel(float* output, const float* input,
                                     const float* prevLayerBias, int inputSize,
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
    if (index < inputSize) S += input[index];
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
      if (prevLayerBias) avg += prevLayerBias[opIndex % C];
      output[opIndex] = avg;
    }
  }
}

template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias) {
  const int kPlaneSize = 64;

  const bool fp16 = std::is_same<half, T>::value;
  if (fp16) {
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
    globalAvgPool_kernel<<<blocks, kBlockSize>>>((float*)output, (float*)input,
                                                 (float*)prevLayerBias,
                                                 N * C * kPlaneSize, N * C, C);
  }
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias) {
  const bool fp16 = std::is_same<half, T>::value;

  // Each thread writes one output.
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  if (fp16) {
    globalScale_kernel_fp16_nhwc<<<kBlocks, kBlockSize>>>(
        (half*)output, (half*)input, (half*)scaleBias, (half*)prevLayerBias,
        N * C * 8 * 8, C, 8 * 8 * C);
  } else {
    globalScale_kernel<<<kBlocks, kBlockSize>>>(
        (float*)output, (float*)input, (float*)scaleBias, (float*)prevLayerBias,
        N * C * 8 * 8, C);
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
               int inputSize, int usedSize, int outputSize) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  policyMap_kernel<T><<<kBlocks, kBlockSize>>>((T*)output, (T*)input,
                                               (short*)indices, N, inputSize,
                                               usedSize, outputSize);
  ReportCUDAErrors(cudaGetLastError());
}

// Template instantiation.
template void copyTypeConverted<half, float>(half* op, float* ip, int N);
template void copyTypeConverted<float, half>(float* op, half* ip, int N);

template void batchNorm<float>(float* output, const float* input,
                               const float* skipInput, int N, int C, int H,
                               int W, float* means, float* var_multipliers,
                               bool relu);
template void batchNorm<half>(half* output, const half* input,
                              const half* skipInput, int N, int C, int H, int W,
                              float* means, float* var_multipliers, bool relu);

template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, bool relu, bool use_tanh,
                                bool use_sigmoid);
template void addVectors<half>(half* c, half* a, half* b, int size, int asize,
                               int bsize, bool relu, bool use_tanh,
                               bool use_sigmoid);

template void addBias_NCHW<float>(float* c, float* a, float* b, int N, int C,
                                  int H, int W);

template void addBias_NCHW<half>(half* c, half* a, half* b, int N, int C, int H,
                                 int W);

template void globalAvgPool<float>(int N, int C, float* output,
                                   const float* input,
                                   const float* prevLayerBias);
template void globalAvgPool<half>(int N, int C, half* output, const half* input,
                                  const half* prevLayerBias);

template void globalScale<float>(int N, int C, float* output,
                                 const float* input, const float* scaleBias,
                                 const float* prevLayerBias);
template void globalScale<half>(int N, int C, half* output, const half* input,
                                const half* scaleBias,
                                const half* prevLayerBias);

template void PolicyMap<float>(int N, float* output, const float* input,
                               const short* indices, int inputSize,
                               int usedSize, int outputSize);

template void PolicyMap<half>(int N, half* output, const half* input,
                              const short* indices, int inputSize, int usedSize,
                              int outputSize);

}  // namespace cudnn_backend
}  // namespace lczero
