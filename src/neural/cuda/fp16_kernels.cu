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

#define readw1(row, col) (w1[(row)*K + (col)])
#define readw2(row, col) (w2[(row)*2 * C + (col)])

template <int C, int K>
__global__ void SE_Layer_NHWC(half* output, const half* skip, const half* input,
                              const half* w1, const half* b1, const half* w2,
                              const half* b2, const half *bPrev) {
  const int elementsPerThread = 64;  // 8x8 board

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

    // relu
    if (S < (half)0) S = 0;

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
    if (val < (half)0) val = 0;

    output[inputIndex] = val;
  }
}

bool Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2, const half* bPrev) {
  // TODO: Think of more elegant way to avoid this hardcoding :-/
  if (numFc1Out == 16) {
    if (C == 64) {
      SE_Layer_NHWC<64, 16>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else {
      // TODO: support other channel counts.
      throw Exception("channel count unsupported by SE layer");
    }
  } else if (numFc1Out == 32) {
    if (C == 64) {
      SE_Layer_NHWC<64, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else {
      // TODO: support other channel counts.
      return false;
    }
  } else if (numFc1Out == 64) {
    if (C == 64) {
      SE_Layer_NHWC<64, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev);
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

}   // namespace cudnn_backend
}   // namespace lczero
