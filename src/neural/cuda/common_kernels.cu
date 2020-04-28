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
                            const float* values, int n) {
  int threads = n * 8 * 8;  // each thread writes a single element
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  expandPlanes_kernel_Fp16_NCHW<<<blocks, blockSize>>>(output, masks, values,
                                                       n);
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


template <int C>
__device__ constexpr inline int board_index_nchw(int n, int c, int h, int w) {
    return n*C * 64 + c * 64 + h * 8 + w;
}

template <int C>
__device__ constexpr inline int filter_index_kchw(int k, int c, int h, int w) {
    return k*C * 9 + c * 9 + h * 3 + w;
}

__device__ constexpr inline int div_up(int a, int b) {
    return  (a + b - 1) / b;
}

namespace ConvKernel0 {
// Special kernel to get better performance for small batch size (N=1)
//
// Optimizations:
// - Do multiple elements in H and W dimensions (4x4) per thread, to get some
//   spatial reuse for input tensor as well as filter.
// - More spatial reuse for input tensor using shfl for reading elements outside
//   the 4x4 'tile'.
// - The sum across C dimension is split across multiple threads (8), and
//    summed in parallel using SHFL.

// Do multiple elements in H and W dimensions (4x4).
constexpr int wPerThread = 4;
constexpr int hPerThread = 4;

// 32 threads in block.
// One board (4 threads, with 16 elements per thread).
constexpr int blockWidth = 4; 

// different 'C' dimensions
constexpr int blockHeight = 8;

template <int K, int C, bool doRelu, bool biasAdd>
__global__ void convKernel(float* output, const float* input,
                           const float* weight, const float* bias,
                           float alpha, float beta) {
  int n = blockIdx.y;
  int k = blockIdx.x;

  int hStart = (threadIdx.x >> 1) * hPerThread;
  int wStart = (threadIdx.x & 1) * wPerThread;
  constexpr int cPerThread = C / blockHeight;
  int cBase = threadIdx.y * cPerThread;

  int threadInBlock = threadIdx.y * blockWidth + threadIdx.x;

  __shared__ float shFilter[C * 9];

  // Accumulators.
  float op[4][4];
  #pragma unroll
  for (int y = 0; y < hPerThread; y++)
    #pragma unroll
    for (int x = 0; x < wPerThread; x++) op[y][x] = 0;

  // load filters into shared memory
  #pragma unroll
  for (int i = 0; i < div_up(C * 9, 32); i++) {
    int localIndex = (32) * i + threadInBlock;
    if (localIndex < C * 9)
      shFilter[localIndex] = weight[k * (C * 9) + localIndex];
  }

  #pragma unroll 8
  for (int lc = 0; lc < cPerThread; lc++) {
    int c = cBase + lc;

    // hardcoded for 3x3 filter, and 4x4 spatial elements per thread
    float inEl[hPerThread + 2][wPerThread + 2];
    #pragma unroll
    for (int y = 0; y < hPerThread + 2; y++)
      #pragma unroll
      for (int x = 0; x < wPerThread + 2; x++) inEl[y][x] = 0.0f;

    // assume wPerThread == 4, and use a 128 bit reads
    *((uint4*)(&inEl[1][1])) =
        *((uint4*)(&input[board_index_nchw<C>(n, c, hStart, wStart)]));
    *((uint4*)(&inEl[2][1])) =
        *((uint4*)(&input[board_index_nchw<C>(n, c, hStart + 1, wStart)]));
    *((uint4*)(&inEl[3][1])) =
        *((uint4*)(&input[board_index_nchw<C>(n, c, hStart + 2, wStart)]));
    *((uint4*)(&inEl[4][1])) =
        *((uint4*)(&input[board_index_nchw<C>(n, c, hStart + 3, wStart)]));

    // need temps because shfl needs all threads in warp to participate
    float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[4][1], 2);
    float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[4][2], 2);
    float t03 = __shfl_up_sync(0xFFFFFFFF, inEl[4][3], 2);
    float t04 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 2);
    if (hStart != 0) {
      inEl[0][1] = t01;
      inEl[0][2] = t02;
      inEl[0][3] = t03;
      inEl[0][4] = t04;
    }

    float t51 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 2);
    float t52 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 2);
    float t53 = __shfl_down_sync(0xFFFFFFFF, inEl[1][3], 2);
    float t54 = __shfl_down_sync(0xFFFFFFFF, inEl[1][4], 2);
    if (hStart == 0) {
      inEl[5][1] = t51;
      inEl[5][2] = t52;
      inEl[5][3] = t53;
      inEl[5][4] = t54;
    }

    float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][4], 1);
    float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][4], 1);
    float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][4], 1);
    float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][4], 1);
    float t40 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 1);
    float t50 = __shfl_up_sync(0xFFFFFFFF, inEl[5][4], 1);
    if (wStart != 0) {
      inEl[0][0] = t00;
      inEl[1][0] = t10;
      inEl[2][0] = t20;
      inEl[3][0] = t30;
      inEl[4][0] = t40;
      inEl[5][0] = t50;
    }

    float t05 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
    float t15 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
    float t25 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
    float t35 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
    float t45 = __shfl_down_sync(0xFFFFFFFF, inEl[4][1], 1);
    float t55 = __shfl_down_sync(0xFFFFFFFF, inEl[5][1], 1);
    if (wStart == 0) {
      inEl[0][5] = t05;
      inEl[1][5] = t15;
      inEl[2][5] = t25;
      inEl[3][5] = t35;
      inEl[4][5] = t45;
      inEl[5][5] = t55;
    }

    #pragma unroll
    for (int s = 0; s < 3; s++) {
      #pragma unroll
      for (int r = 0; r < 3; r++) {
        float weight = (float)(shFilter[filter_index_kchw<1>(0, c, s, r)]);
        #pragma unroll
        for (int y = 0; y < hPerThread; y++) {
          #pragma unroll
          for (int x = 0; x < wPerThread; x++) {
            op[y][x] += inEl[y + s][x + r] * weight;
          }
        }
      }
    }
  }  // lc / c

  float b = biasAdd ? bias[k] : 0;

  #pragma unroll
  for (int y = 0; y < hPerThread; y++) {
    #pragma unroll
    for (int x = 0; x < wPerThread; x++) {
      // sum across C dimension
      op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 4);
      op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 8);
      op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 16);

      op[y][x] += b;

      if (doRelu && op[y][x] < 0) op[y][x] = 0;
    }
  }

  if (threadIdx.y == 0) {
    ((uint4*)output)[board_index_nchw<K>(n, k, hStart, wStart) >> 2] =
        *((uint4*)&op[0][0]);
    ((uint4*)output)[board_index_nchw<K>(n, k, hStart + 1, wStart) >> 2] =
        *((uint4*)&op[1][0]);
    ((uint4*)output)[board_index_nchw<K>(n, k, hStart + 2, wStart) >> 2] =
        *((uint4*)&op[2][0]);
    ((uint4*)output)[board_index_nchw<K>(n, k, hStart + 3, wStart) >> 2] =
        *((uint4*)&op[3][0]);
  }
}

};  // ConvKernel0

namespace ConvKernel1 {
// Fastest kernel for small batch size (N=1).
//  - Requires C to be a multiple of 64.
//
// Optimizations:
// - Do multiple elements in H and W dimensions (4x4) per thread, to get some
//   spatial reuse for input tensor as well as filter.
// - More spatial reuse for input tensor using shfl for reading elements outside
//   the 4x4 'tile'.
// - C dimension is sliced into multiple chunks (of 32) to reduce shared memory
//   usage and get better occupancy (allow more blocks per SM).
// - Multiple elements (2) in K dimension processed by thread.
//   -- This gets more reuse of the input tensor.
// - Another slicing along C dimension to increase occupancy (using shared
// memory)

// output elements processed per thread
constexpr int wPerThread = 4;
constexpr int hPerThread = 4;

// 32 threads in block.
// One board (4 threads, with 16 elements per thread).
constexpr int blockWidth = 4;
// Different 'C' dimensions (summed in parallel using SHFL).
constexpr int blockHeight = 8;

// The code to do final sums and to write outputs below assumes this to be 2.
// Again different 'C' dimension (in a block of
// shared memory accessed by different warps).
constexpr int blockDepth = 2;

// These many filter elements from c dimension are loaded into
// shared memory at a time (should be a multiple of warp size).
constexpr int cPerIter = 32;
constexpr int cPerIterPerThread = cPerIter / (blockHeight);

constexpr int kPerBlock = 2;

template <int K, int C, bool doRelu, bool biasAdd, bool skipAdd>
__global__ void convKernel(float* output, const float* input,
                           const float* weight, const float* bias,
                           const float* skip, float alpha, float beta) {
  int n = blockIdx.y;
  int kStart = blockIdx.x * kPerBlock;

  int hStart = (threadIdx.x >> 1) * hPerThread;
  int wStart = (threadIdx.x & 1) * wPerThread;

  // extra offset
  int cPerSlice = C / blockDepth;
  int cStart = threadIdx.z * cPerSlice;
  int shStart = (kPerBlock * cPerIter * 9) * threadIdx.z;

  // offset to be added to get C index
  int cOffset = threadIdx.y * cPerIterPerThread;

  int threadInWarp = threadIdx.y * blockWidth + threadIdx.x;

  __shared__ float shData[blockDepth * kPerBlock * cPerIter * 9];

  // accumulators
  float op[kPerBlock][4][4];

  #pragma unroll
  for (int lk = 0; lk < kPerBlock; lk++)
    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
      #pragma unroll
      for (int x = 0; x < wPerThread; x++) op[lk][y][x] = 0;

  // outer loop
  for (int cBase = 0; cBase < cPerSlice; cBase += cPerIter) {
    // load filters into shared memory
    #pragma unroll
    for (int lk = 0; lk < kPerBlock; lk++) {
      int k = kStart + lk;
      #pragma unroll
      for (int i = 0; i < cPerIter * 9 / 32; i++) {
        int localIndex = 32 * i + threadInWarp;
        int sharedIndex = shStart + lk * (cPerIter * 9) + localIndex;
        int globalIndex = k * C * 9 + (cStart + cBase) * 9 + localIndex;
        shData[sharedIndex] = weight[globalIndex];
      }
    }

    #pragma unroll
    for (int lc = 0; lc < cPerIterPerThread; lc++) {
      int shc = cOffset + lc;  // offset of filter for index c in shared memory
      int c = cStart + cBase + shc;  // real c dimension

      // hardcoded for 3x3 filter, and 4x4 spatial elements per thread
      float inEl[hPerThread + 2][wPerThread + 2];
      #pragma unroll
      for (int y = 0; y < hPerThread + 2; y++)
        #pragma unroll
        for (int x = 0; x < wPerThread + 2; x++) inEl[y][x] = 0.0f;

      // assume wPerThread == 4, and use a 128 bit reads
      *((uint4*)(&inEl[1][1])) =
          *((uint4*)(&input[board_index_nchw<C>(n, c, hStart, wStart)]));
      *((uint4*)(&inEl[2][1])) =
          *((uint4*)(&input[board_index_nchw<C>(n, c, hStart + 1, wStart)]));
      *((uint4*)(&inEl[3][1])) =
          *((uint4*)(&input[board_index_nchw<C>(n, c, hStart + 2, wStart)]));
      *((uint4*)(&inEl[4][1])) =
          *((uint4*)(&input[board_index_nchw<C>(n, c, hStart + 3, wStart)]));

      // need temps because shfl needs all threads in warp to participate
      float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[4][1], 2);
      float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[4][2], 2);
      float t03 = __shfl_up_sync(0xFFFFFFFF, inEl[4][3], 2);
      float t04 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 2);
      if (hStart != 0) {
        inEl[0][1] = t01;
        inEl[0][2] = t02;
        inEl[0][3] = t03;
        inEl[0][4] = t04;
      }

      float t51 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 2);
      float t52 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 2);
      float t53 = __shfl_down_sync(0xFFFFFFFF, inEl[1][3], 2);
      float t54 = __shfl_down_sync(0xFFFFFFFF, inEl[1][4], 2);
      if (hStart == 0) {
        inEl[5][1] = t51;
        inEl[5][2] = t52;
        inEl[5][3] = t53;
        inEl[5][4] = t54;
      }

      float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][4], 1);
      float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][4], 1);
      float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][4], 1);
      float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][4], 1);
      float t40 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 1);
      float t50 = __shfl_up_sync(0xFFFFFFFF, inEl[5][4], 1);
      if (wStart != 0) {
        inEl[0][0] = t00;
        inEl[1][0] = t10;
        inEl[2][0] = t20;
        inEl[3][0] = t30;
        inEl[4][0] = t40;
        inEl[5][0] = t50;
      }

      float t05 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
      float t15 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
      float t25 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
      float t35 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
      float t45 = __shfl_down_sync(0xFFFFFFFF, inEl[4][1], 1);
      float t55 = __shfl_down_sync(0xFFFFFFFF, inEl[5][1], 1);
      if (wStart == 0) {
        inEl[0][5] = t05;
        inEl[1][5] = t15;
        inEl[2][5] = t25;
        inEl[3][5] = t35;
        inEl[4][5] = t45;
        inEl[5][5] = t55;
      }

      #pragma unroll
      for (int s = 0; s < 3; s++) {
        #pragma unroll
        for (int r = 0; r < 3; r++) {
          #pragma unroll
          for (int lk = 0; lk < kPerBlock; lk++) {
            float wt = (float)(shData[shStart + filter_index_kchw<cPerIter>(lk, shc, s, r)]);
            #pragma unroll
            for (int y = 0; y < hPerThread; y++) {
              #pragma unroll
              for (int x = 0; x < wPerThread; x++) {
                op[lk][y][x] += inEl[y + s][x + r] * wt;
              }  // x
            }    // y
          }      // k
        }        // r
      }          // s
    }            // lc
  }              // cBase

  #pragma unroll
  for (int y = 0; y < hPerThread; y++) {
    #pragma unroll
    for (int x = 0; x < wPerThread; x++) {
      // sum across C dimension
      #pragma unroll
      for (int lk = 0; lk < kPerBlock; lk++) {
        op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 4);
        op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 8);
        op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 16);
      }
    }
  }

  //__shared__ float shResult[blockWidth][kPerBlock][hPerThread][wPerThread];
  static_assert(sizeof(shData) >= 2 * sizeof(float) * blockWidth * kPerBlock *
                                      hPerThread * wPerThread,
                "shared mem not enough");

  if (threadIdx.y == 0 && threadIdx.z == 0) {
    #pragma unroll
    for (int lk = 0; lk < kPerBlock; lk++)
      #pragma unroll
      for (int y = 0; y < hPerThread; y++)
        #pragma unroll
        for (int x = 0; x < wPerThread; x++) {
          // shResult[threadIdx.x][lk][y][x] = op[lk][y][x];
          shData[threadIdx.x * kPerBlock * hPerThread * wPerThread +
                 lk * hPerThread * wPerThread + y * wPerThread + x] =
              op[lk][y][x];
        }
  }

  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.z == 1) {
    float b[kPerBlock];

    if (biasAdd) {
      #pragma unroll
      for (int lk = 0; lk < kPerBlock; lk++)
        b[lk] = bias ? bias[kStart + lk] : 0;
    }

    float sk[kPerBlock][4][4];
    if (skipAdd) {
      #pragma unroll
      for (int lk = 0; lk < kPerBlock; lk++) {
        int k = kStart + lk;
        *((uint4*)&sk[lk][0][0]) =
            ((uint4*)skip)[board_index_nchw<K>(n, k, hStart + 0, wStart) >> 2];
        *((uint4*)&sk[lk][1][0]) =
            ((uint4*)skip)[board_index_nchw<K>(n, k, hStart + 1, wStart) >> 2];
        *((uint4*)&sk[lk][2][0]) =
            ((uint4*)skip)[board_index_nchw<K>(n, k, hStart + 2, wStart) >> 2];
        *((uint4*)&sk[lk][3][0]) =
            ((uint4*)skip)[board_index_nchw<K>(n, k, hStart + 3, wStart) >> 2];
      }
    }

    #pragma unroll
    for (int y = 0; y < hPerThread; y++) {
      // sum across C dimension
      #pragma unroll
      for (int lk = 0; lk < kPerBlock; lk++) {
        // apply bias and relu
        #pragma unroll
        for (int x = 0; x < wPerThread; x++) {
          // op[lk][y][x] += shResult[threadIdx.x][lk][y][x];
          op[lk][y][x] +=
              shData[threadIdx.x * kPerBlock * hPerThread * wPerThread +
                     lk * hPerThread * wPerThread + y * wPerThread + x];

          if (skipAdd) {
            op[lk][y][x] = op[lk][y][x] * alpha + beta * sk[lk][y][x];
          }

          if (biasAdd) {
            op[lk][y][x] += b[lk];
          }

          if (doRelu && op[lk][y][x] < 0) op[lk][y][x] = 0;
        }
      }
    }

    // final memory write
    #pragma unroll
    for (int lk = 0; lk < kPerBlock; lk++) {
      int k = kStart + lk;
      ((uint4*)output)[board_index_nchw<K>(n, k, hStart, wStart) >> 2] =
          *((uint4*)&op[lk][0][0]);
      ((uint4*)output)[board_index_nchw<K>(n, k, hStart + 1, wStart) >> 2] =
          *((uint4*)&op[lk][1][0]);
      ((uint4*)output)[board_index_nchw<K>(n, k, hStart + 2, wStart) >> 2] =
          *((uint4*)&op[lk][2][0]);
      ((uint4*)output)[board_index_nchw<K>(n, k, hStart + 3, wStart) >> 2] =
          *((uint4*)&op[lk][3][0]);
    }
  }
}

};  // namespace ConvKernel1

template <int K, int C>
void launch_convCuda3x3_1(dim3 blockDim, dim3 gridDim, float* output,
                          const float* input, const float* weight,
                          const float* bias, const float* skip, bool relu) {
  static_assert((C % 64 == 0) && (K % 64 == 0), "Channel count not supported");
  if (skip) {
    if (relu)
      ConvKernel1::convKernel<K, C, true, true, true>
          <<<gridDim, blockDim>>>(output, input, weight, bias, skip, 1, 1);
    else
      ConvKernel1::convKernel<K, C, false, false, true>
          <<<gridDim, blockDim>>>(output, input, weight, bias, skip, 1, 1);
  } else {
    if (relu)
      ConvKernel1::convKernel<K, C, true, true, false>
          <<<gridDim, blockDim>>>(output, input, weight, bias, skip, 1, 1);
    else
      ConvKernel1::convKernel<K, C, false, false, false>
          <<<gridDim, blockDim>>>(output, input, weight, bias, skip, 1, 1);
  }
}


template <int K, int C>
void launch_convCuda3x3_0(dim3 blockDim, dim3 gridDim, float* output,
                          const float* input, const float* weight,
                          const float* bias, bool relu) {
  static_assert(C % 8 == 0, "Channel count not supported");
  if (relu)
    ConvKernel0::convKernel<K, C, true, true>
        <<<gridDim, blockDim>>>(output, input, weight, bias, 1, 1);
  else
    ConvKernel0::convKernel<K, C, false, false>
        <<<gridDim, blockDim>>>(output, input, weight, bias, 1, 1);
}

// Use specialized kernels for 3x3 convolutions.
// Faster for small batch sizes.
// Returns false if the kernel isn't able to handle the given params.
bool convCuda3x3(float* output, const float* input, const float* weight,
                 const float* bias, const float* skip, bool relu, int N, int K,
                 int C) {
  // For computing each each of them need to do (3 * 3 * C) multiple-adds  (e.g:
  // 2304 for C=256) Need to re-use input and filter elements to avoid making
  // everything memory bound.

  // Either need both bias and relu, or none of them (this is not a limitation
  // of the kernels but just to avoid too much template param hard-coding).
  if ((bias && !relu) || (!bias && relu))
    return false;

  if (C == 112) {
    // This kernel is slower than the one used below, but supports not-multiple
    // of 64 C. We use it only for the first convolution.

    // N * K blocks used.
    // Each thread block processes 8x8 elements.
    dim3 gridDim(K, N);
    dim3 blockDim(ConvKernel0::blockWidth, ConvKernel0::blockHeight, 1);

    if (skip) return false; // Doesn't support skip connection add.

    if (K == 64)
      launch_convCuda3x3_0<64, 112>(blockDim, gridDim, output, input, weight,
                                   bias, relu);
    else if (K == 128)
      launch_convCuda3x3_0<128, 112>(blockDim, gridDim, output, input, weight,
                                     bias, relu);
    else if (K == 192)
      launch_convCuda3x3_0<192, 112>(blockDim, gridDim, output, input, weight,
                                     bias, relu);
    else if (K == 256)
      launch_convCuda3x3_0<256, 112>(blockDim, gridDim, output, input, weight,
                                     bias, relu);
    else if (K == 320)
      launch_convCuda3x3_0<320, 112>(blockDim, gridDim, output, input, weight,
                                     bias, relu);
    else if (K == 384)
      launch_convCuda3x3_0<384, 112>(blockDim, gridDim, output, input, weight,
                                     bias, relu);

    else
      return false;  // Add more template instantiations as needed
  } else {
    // N * (K/2) blocks used.
    // Each thread block processes 2x8x8 elements.
    dim3 gridDim(K / ConvKernel1::kPerBlock, N);
    dim3 blockDim(ConvKernel1::blockWidth, ConvKernel1::blockHeight,
                  ConvKernel1::blockDepth);

    // Supports only multiple of 64 channel counts
    if (C == 64 && K == 64)
      launch_convCuda3x3_1<64, 64>(blockDim, gridDim, output, input, weight,
                                   bias, skip, relu);
    else if (C == 128 && K == 128)
      launch_convCuda3x3_1<128, 128>(blockDim, gridDim, output, input, weight,
                                     bias, skip, relu);
    else if (C == 192 && K == 192)
      launch_convCuda3x3_1<192, 192>(blockDim, gridDim, output, input, weight,
                                     bias, skip, relu);
    else if (C == 256 && K == 256)
      launch_convCuda3x3_1<256, 256>(blockDim, gridDim, output, input, weight,
                                     bias, skip, relu);
    else if (C == 320 && K == 320)
      launch_convCuda3x3_1<320, 320>(blockDim, gridDim, output, input, weight,
                                     bias, skip, relu);
    else if (C == 384 && K == 384)
      launch_convCuda3x3_1<384, 384>(blockDim, gridDim, output, input, weight,
                                     bias, skip, relu);
    else
      return false; // Add more template instantiations as needed
  }

  ReportCUDAErrors(cudaGetLastError());

  return true;
}

// Template instantiation.
template void copyTypeConverted<half, float>(half* op, float* ip, int N);
template void copyTypeConverted<float, half>(float* op, half* ip, int N);
template void copyTypeConverted<float, float>(float* op, float* ip, int N);
template void copyTypeConverted<half, half>(half* op, half* ip, int N);

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
                               int usedSize, int outputSize);

template void PolicyMap<half>(int N, half* output, const half* input,
                              const short* indices, int inputSize, int usedSize,
                              int outputSize);

template void FilterTransform<float>(int N, int C, float* transformedFilter,
                                     const float* filter);

template void InputTransform<float>(int N, int C, float* transformed_input,
                                    const float* input);

template void OutputTransform<float, true, true, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2);

template void OutputTransform<float, false, true, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2);

template void OutputTransform<float, false, true, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2);


}  // namespace cudnn_backend
}  // namespace lczero
