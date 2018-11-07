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


// This kernel is for OLD plain SE, doesn't work with SE-SiLK
// TODO: modify the kernel to support SE-SiLK 
// (need a different strategy as we can't fit the entire weight matrix in shared memory)

// N blocks
// C threads per block
// 'HWC' input data processed by thread block
// each thread processes 8x8 elements
// K is the no. of outputs of first fully connected layer (same as no. of inputs
// for second fully connected layer) the kernel assumes K <= C

// the weights matrix are transposed in reality (K rows, and C columns for fc1,
// and C rows and K columns for fc2)
#define shw1(row, col) (((half*)sharedWeights1)[(col)*C + (row)])
#define shw2(row, col) (((half*)sharedWeights2)[(col)*K + (row)])

template <int C, int K>
__global__ void SE_Layer_NHWC(half* output, const half* skip, const half* input,
                              const half* w1, const half* b1, const half* w2,
                              const half* b2) {
  const int elementsPerThread = 64;  // 8x8 board

  int n = blockIdx.x;
  int c = threadIdx.x;

  __shared__ half sharedData[C];

  half localInput[elementsPerThread];
  half localskip[elementsPerThread];

  // This acutally doesn't save on any global memory reads (each thread block
  // still reads entire weights array redundantly :-/)
  // TODO: can try processing multiple C (multiple planes) in single thread
  // block to get some savings
  //
  // it's only to make the reads faster (better memory coleasing)
  static_assert(((C * K) % 8) == 0, "K*C must be multiple of 8");

  // don't really NEED two shared memory arrays, as the same shared memory can
  // be re-used to hold weights for FC2 after FC1 is done, but loading all
  // weights early seems to improve performance by about 5%
  __shared__ uint4 sharedWeights1[C * K / 8];
  __shared__ uint4 sharedWeights2[C * K / 8];

  half S = 0;

  // 1. global avg (1 avg per thread)
  #pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    localInput[i] = input[inputIndex];
    localskip[i] = skip[inputIndex];
    S += localInput[i];
  }

  half avg = S / (half)elementsPerThread;
  sharedData[c] = avg;

  // load weights for the FC layers in shared memory
  // use uint4 loads to make it faster
  const int numSharedReadsPerThread = K / 8;  // K * C weights, divided by C
                                              // threads, divided by 8 halfs
                                              // (uint4) read per thread
  uint4* w1raw = (uint4*)w1;
  uint4* w2raw = (uint4*)w2;

  #pragma unroll
  for (int i = 0; i < numSharedReadsPerThread; i++) {
    sharedWeights1[c + i * C] = w1raw[c + i * C];
    sharedWeights2[c + i * C] = w2raw[c + i * C];
  }
  __syncthreads();

  // 2. first fully connected layer
  if (c < K) {
    S = 0;

    #pragma unroll
    for (int i = 0; i < C; i++) {
      S += sharedData[i] * shw1(i, c);
    }

    S += b1[c];

    // relu
    if (S < (half)0) S = 0;

    sharedData[c] = S;
  }

  __syncthreads();

  // 3. second fully connected layer
  S = 0;
  #pragma unroll
  for (int i = 0; i < K; i++) {
    S += sharedData[i] * shw2(i, c);
  }
  S += b2[c];

  // sigmoid
  S = (half)(1.0f / (1.0f + exp(-(float)(S))));

  // 4. scale, and add skip connection, perform relu, and write to output
  #pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    half val = localskip[i] + localInput[i] * S;

    // relu
    if (val < (half)0) val = 0;

    output[inputIndex] = val;
  }
}

void Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2) {
  // TODO: think of more elegant way to avoid this hardcoding :-/
  if (numFc1Out == 32) {
    if (C == 64) {
      SE_Layer_NHWC<64, 32><<<N, C>>>(output, skip, input, w1, b1, w2, b2);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 32><<<N, C>>>(output, skip, input, w1, b1, w2, b2);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 32><<<N, C>>>(output, skip, input, w1, b1, w2, b2);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 32><<<N, C>>>(output, skip, input, w1, b1, w2, b2);
    } else {
      // TODO: support other channel counts
      throw Exception("channel count unsupported by SE layer");
    }
  } else {
    // TODO: support other sizes
    throw Exception("numOutputs unsupported by SE layer");
  }
  ReportCUDAErrors(cudaGetLastError());
}

}   // namespace cudnn_backend
}   // namespace lczero
