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
#include "winograd_helper.inc"

namespace lczero {
namespace cudnn_backend {

/////////////////////////////////////////////////////////////////////////////
//          fp16-specific kernels used by certain layers                   //
/////////////////////////////////////////////////////////////////////////////

// SE layer implementation using single fused kernel.

// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread processes 8x8 elements.
// K is the no. of outputs of first fully connected layer (same as no. of inputs
// for second fully connected layer).
// The kernel assumes K <= C.

template <int C, int K>
__global__ void SE_Layer_NHWC(half* output, const half* skip, const half* input,
                              const half* w1, const half* b1, const half* w2,
                              const half* b2, const half* bPrev,
                              ActivationFunction activation) {
  const int elementsPerThread = 64;  // 8x8 board
  const int se_K = K;

  int n = blockIdx.x;
  int c = threadIdx.x;

  __shared__ half sharedData[C];

  half2 localData[elementsPerThread];

  half S = 0;

  half bias = 0;
  if (bPrev) bias = bPrev[c];

// 1. Global avg (1 avg per thread).
#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    localData[i].x = input[inputIndex] + bias;
    localData[i].y = skip[inputIndex];
    S += localData[i].x;
  }

  half avg = S / (half)elementsPerThread;
  sharedData[c] = avg;

  __syncthreads();

  // 2. First fully connected layer.
  if (c < K) {
    S = 0;

#pragma unroll
    for (int i = 0; i < C; i++) {
      S += sharedData[i] * readw1(i, c);
    }

    S += b1[c];

    S = activate(S, activation);

    sharedData[c] = S;
  }
  __syncthreads();

  // 3. Second fully connected layer.
  S = 0;
  half B = 0;
#pragma unroll
  for (int i = 0; i < K; i++) {
    half val = sharedData[i];
    S += val * readw2(i, c);
    B += val * readw2(i, c + C);
  }
  S += b2[c];
  B += b2[c + C];

  // Sigmoid (only on the scale part).
  S = (half)(1.0f / (1.0f + exp(-(float)(S))));

// 4. Scale, and add skip connection, perform relu, and write to output.
#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    half val = localData[i].y + localData[i].x * S + B;

    // Relu activation function.
    val = (half)activate((float)val, activation);

    output[inputIndex] = val;
  }
}

bool Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2, const half* bPrev,
                  ActivationFunction activation) {
  // TODO: Think of more elegant way to avoid this hardcoding :-/
  if (numFc1Out == 16) {
    if (C == 64) {
      SE_Layer_NHWC<64, 16>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else {
      // TODO: support other channel counts.
      throw Exception("channel count unsupported by SE layer");
    }
  } else if (numFc1Out == 32) {
    if (C == 64) {
      SE_Layer_NHWC<64, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 320) {
      SE_Layer_NHWC<320, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 352) {
      SE_Layer_NHWC<352, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 384) {
      SE_Layer_NHWC<384, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else {
      // TODO: support other channel counts.
      return false;
    }
  } else if (numFc1Out == 64) {
    if (C == 64) {
      SE_Layer_NHWC<64, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 320) {
      SE_Layer_NHWC<320, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 384) {
      SE_Layer_NHWC<384, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else {
      // TODO: support other channel counts.
      return false;
    }
  } else {
    // TODO: support other sizes.
    return false;
  }
  ReportCUDAErrors(cudaGetLastError());
  return true;
}

// Get board for this thread from shared memory.
// We are just using shared memory to store local thread data in this kernel to
// help reduce some register pressure and spills to local memory.
#define BOARD(y, x) shboard[(y)*8 + (x)]

// input is in transformed space (HWNC layout) --- output of GEMM
// output is also in transformed space (HWNC layout) --- input to GEMM (for
// next layer)
// 'C' threads per block
// 'N' blocks
// Every thread generates an entire board/plane (8x8 elements).
template <ActivationFunction activation, bool use_bias,
          bool use_skip>
__global__ __launch_bounds__(kMaxResBlockFusingSeKFp16Ampere,1)
void OutputInputTransformKernel_fp16_shmem_board(
        int N, int C, int se_K, half* output, const half* input, half* skip,
        const half* bias, const half* w1, const half* b1, const half* w2,
        const half* b2) {
  int k = threadIdx.x;
  int n = blockIdx.x;

  extern __shared__ half _sboard[];
  half* shboard = &_sboard[k * 72];  // 72 instead of 64 to reduce shared
                                     // memory bank conflicts.
  half b = bias[k];

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4)
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      //  i) read to per thread registers (for doing output transform)
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      half outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++)
#pragma unroll
        for (int x = 0; x < 6; x++)
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];

      // ii) transform it
      half outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++)
        copyAs<uint2>(&BOARD(hStart + y, wStart), &outEl[y][0]);
    }

  // Add bias, and compute the average for SE.
  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++) {
    half boardRow[8];
    copyAs<uint4>(&boardRow, &BOARD(y, 0));
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) boardRow[x] += b;
      S += (float)boardRow[x];
    }
    if (use_bias) copyAs<uint4>(&BOARD(y, 0), &boardRow);
  }

  __shared__ float shared_data[kMaxResBlockFusingSeKFp16Ampere];
  float avg = S / 64;
  shared_data[k] = avg;

  int lane = k & 0x1F;
  int warp = k >> 5;
  __syncthreads();

  // First fully-connected layer for SE

  // As se_K << C, we want to loop over se_K instead of C
  // even if it means taking the sum across threads

  __shared__ float shared_sums[kMaxResBlockFusingSeKFp16Ampere / 32]
                              [kMaxResBlockFusingSeK];  // per-warp sums

  for (int i = 0; i < se_K; i++) {
    float val = shared_data[k] * float(readw1(k, i));
    val = warpReduce(val);
    if (lane == 0) shared_sums[warp][i] = val;
  }
  __syncthreads();
  if (k < se_K) {
    S = 0;
    for (int i = 0; i < C / 32; i++) S += shared_sums[i][k];

    S += (float)b1[k];
    S = activate(S, activation);
    shared_data[k] = S;
  }

  __syncthreads();

  // Second fully-connected layer for SE
  S = 0;
  for (int i = 0; i < se_K; i++) {
    float val = shared_data[i];
    S += val * float(readw2(i, k));
    B += val * float(readw2(i, k + C));
  }
  S += (float)b2[k];
  B += (float)b2[k + C];

  // Sigmoid (only on the scale part).
  S = 1.0f / (1.0f + exp(-S));

  // Scale/bias, add skip connection, perform activation, and write to output.
  for (int h = 0; h < 8; h++) {
    half boardRow[8];
    copyAs<uint4>(&boardRow[0], &BOARD(h, 0));

#pragma unroll
    for (int w = 0; w < 8; w++) {
      boardRow[w] = (half)(float(boardRow[w]) * S + B);
    }

    // residual add
    if (use_skip) {
      half skipInp[8];
      copyAs<uint4>(&skipInp[0], &skip[INDEX_NHCW(n, k, h, 0)]);
#pragma unroll
      for (int w = 0; w < 8; w++) boardRow[w] += skipInp[w];
    }

    if (activation != NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        boardRow[w] = (half)activate((float)boardRow[w], activation);
    }

    // write un-transformed output to 'skip' if required
    if (use_skip) {
      copyAs<uint4>(&skip[INDEX_NHCW(n, k, h, 0)], &boardRow[0]);
    }

    copyAs<uint4>(&BOARD(h, 0), &boardRow);
  }


  // Perform input transform.

  int c = k;
  // top-left
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = BOARD(i, j);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = BOARD(i, j + 3);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = BOARD(i + 3, j);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = BOARD(i + 3, j + 3);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
}

template <typename T = half, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          cudaStream_t stream) {
  // Each thread processes entire chess board.
  if (use_se == false) {
    dim3 grid_dim(DivUp(C, kOpInpTransformBlockSize), N, 1);
    OutputTransform_relu_InputTransform_kernel<half, activation, use_bias,
                                               use_skip>
        <<<grid_dim, kOpInpTransformBlockSize, 0, stream>>>(N, C, output, input,
                                                            (half*)skip, bias);
  } else if (C > kMaxResBlockFusingChannels) {
    // Use special kernel with reduced register pressure - only works on Ampere,
    // and only for fp16.
    if (C <= kMaxResBlockFusingSeKFp16Ampere) {
      cudaFuncSetAttribute(
          OutputInputTransformKernel_fp16_shmem_board<activation,
                                                      use_bias, use_skip>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          72 * C * sizeof(half));
      OutputInputTransformKernel_fp16_shmem_board<activation, use_bias,
                                                  use_skip>
          <<<N, C, 72 * C * sizeof(half), stream>>>(
              N, C, se_K, (half*)output, (const half*)input, (half*)skip,
              (half*)bias, (half*)w1, (half*)b1, (half*)w2, (half*)b2);
    } else {
      throw Exception(
          "res block fusing opt not supported for the given data type and no "
          "of filters\n");
    }
  } else {
    OutputTransform_SE_relu_InputTransform_kernel<half, activation,
                                                  use_bias, use_skip>
        <<<N, C, 0, stream>>>(N, C, se_K, output, input, (half*)skip, bias, w1,
                              b1, w2, b2);
  }
  ReportCUDAErrors(cudaGetLastError());
}

template void FilterTransform<half>(int N, int C, half* transformedFilter,
                                    const half* filter);

template void InputTransform<half, true>(int N, int C, half* transformed_input,
                                         const half* input,
                                         cudaStream_t stream);
template void InputTransform<half, false>(int N, int C, half* transformed_input,
                                          const half* input,
                                          cudaStream_t stream);

template void OutputTransform<half, true, RELU, true, true, false, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, RELU, true, true, false, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, RELU, true, true, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, RELU, true, true, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, RELU, true, false, false, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, RELU, true, false, false, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, RELU, true, true, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, MISH, true, true, false, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, MISH, true, true, false, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, MISH, true, true, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, MISH, true, true, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, MISH, true, false, false, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, MISH, true, false, false, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, MISH, true, true, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, NONE, true, false, false, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, true, RELU, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, RELU, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, RELU, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, true, MISH, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, MISH, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, MISH, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

}  // namespace cudnn_backend
}  // namespace lczero
