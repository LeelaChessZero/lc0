/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

namespace lczero {
namespace cudnn_backend {

__device__ __forceinline__ float mishActivate(float el) {
  auto e = __expf(el);
  auto n = e * e + 2.0f * e;
  auto d = __fdividef(el, n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}
__device__ __forceinline__ float activate(float cVal,
                                          ActivationFunction activation) {
  switch (activation) {
    case RELU:
      if (cVal < 0) cVal = 0;
      break;
    case TANH:
      cVal = tanh(cVal);
      break;
    case SIGMOID:
      cVal = 1.0f / (1.0f + __expf(-cVal));
      break;
    case SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (cVal > 0)
        cVal = scale * cVal;
      else
        cVal = scale * alpha * (__expf(cVal) - 1.0f);
      break;
    }
    case MISH:
      cVal = mishActivate(cVal);
      break;
  }
  return cVal;
}

template <typename T, int M, int N, int K>
__device__ __forceinline__ void matrixMul_gpu_serial(T* c, const T* a,
                                                     const T* b) {
#pragma unroll
  for (int i = 0; i < M; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j) {
      T S = 0;
#pragma unroll
      for (int k = 0; k < K; ++k) S += a[i * K + k] * b[k * N + j];
      c[i * N + j] = S;
    }
}

template <typename T>
__device__ __forceinline__ void FilterTransform4x4(T* transformed_filter,
                                                   const T* filter) {
  // transform applied to filter (of size 3x3)
  T G[6 * 3] = {1.0f / 4,  0,         0,         -1.0f / 6,  -1.0f / 6,
                -1.0f / 6, -1.0f / 6, 1.0f / 6,  -1.0f / 6,  1.0f / 24,
                1.0f / 12, 1.0f / 6,  1.0f / 24, -1.0f / 12, 1.0f / 6,
                0,         0,         1};

  T Gt[3 * 6] = {1.0f / 4, -1.0f / 6, -1.0f / 6, 1.0f / 24, 1.0f / 24,  0,
                 0,        -1.0f / 6, 1.0f / 6,  1.0f / 12, -1.0f / 12, 0,
                 0,        -1.0f / 6, -1.0f / 6, 1.0f / 6,  1.0f / 6,   1};

  T temp_filter[6 * 3];
  matrixMul_gpu_serial<T, 6, 3, 3>(temp_filter, G, filter);
  matrixMul_gpu_serial<T, 6, 6, 3>(transformed_filter, temp_filter, Gt);
}

template <typename T>
__device__ __forceinline__ void InputTransform4x4(T* transformedInput,
                                                  const T* input) {
  // transform applied to input tile (of size 4x4)
  const T Bt[6 * 6] = {4, 0, -5, 0,  1, 0, 0, -4, -4, 1,  1, 0,
                       0, 4, -4, -1, 1, 0, 0, -2, -1, 2,  1, 0,
                       0, 2, -1, -2, 1, 0, 0, 4,  0,  -5, 0, 1};

  const T B[6 * 6] = {4,  0,  0,  0,  0,  0, 0, -4, 4,  -2, 2,  4,
                      -5, -4, -4, -1, -1, 0, 0, 1,  -1, 2,  -2, -5,
                      1,  1,  1,  1,  1,  0, 0, 0,  0,  0,  0,  1};

  T tempIp1[6 * 6];
  matrixMul_gpu_serial<T, 6, 6, 6>(tempIp1, Bt, input);
  matrixMul_gpu_serial<T, 6, 6, 6>(transformedInput, tempIp1, B);
}

template <typename T>
__device__ __forceinline__ void OutputTransform4x4(T* output,
                                                   const T* transformedOutput) {
  // transform applied to result
  const T At[4 * 6] = {1, 1, 1, 1, 1, 0, 0, 1, -1, 2, -2, 0,
                       0, 1, 1, 4, 4, 0, 0, 1, -1, 8, -8, 1};

  const T A[6 * 4] = {1, 0, 0, 0, 1, 1,  1, 1,  1, -1, 1, -1,
                      1, 2, 4, 8, 1, -2, 4, -8, 0, 0,  0, 1};

  T tempOp[4 * 6];
  matrixMul_gpu_serial<T, 4, 6, 6>(tempOp, At, transformedOutput);
  matrixMul_gpu_serial<T, 4, 4, 6>(output, tempOp, A);
}

#define FILTER_IDX_NCHW(k, c, h, w) ((k)*C * S * R + (c)*S * R + (h)*R + w)
template <typename T>
__global__ void filterTransform_kernel(int K, int C, int elements,
                                       T* transformed_filter, const T* filter) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= elements) return;

  constexpr int S = 3;
  constexpr int R = 3;

  int c = tid % C;
  int k = tid / C;

  T filter_tile[3][3];
  T transformed_tile[6][6];

  // read input from memory
  for (int s = 0; s < S; s++)
    for (int r = 0; r < R; r++) {
      filter_tile[s][r] = filter[FILTER_IDX_NCHW(k, c, s, r)];
    }

  // transform it
  FilterTransform4x4(&(transformed_tile[0][0]), &(filter_tile[0][0]));

  // write to output (output is in HWCK layout)
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) {
      transformed_filter[i * 6 * C * K + j * C * K + c * K + k] =
          transformed_tile[i][j];
    }
}

#define INDEX_NCHW(n, c, h, w) ((n)*C * 8 * 8 + (c)*8 * 8 + (h)*8 + w)
#define INDEX_NHCW(n, c, h, w) ((n)*C * 8 * 8 + (h)*C * 8 + (c)*8 + w)

// index in intermediate/temp tensor
// W, H == 6 here! (6x6 transformed blocks)
// N also includes part of dimension (2x2)
#define GemmN (N * 4)
#define TEMP_INDEX_HWNC(h, w, n, c) \
  ((h)*6 * GemmN * C + (w)*GemmN * C + (n)*C + c)

// 'C' threads per block
// 'N' blocks
// every thread transforms an entire board/plane (8x8 elements)
// - producing 4 x 6x6 elements
template <typename T, bool nhcw>
__global__ void InputTransform_kernel(int N, int C, const T* input, T* output) {
  int c = threadIdx.x;
  int n = blockIdx.x;

  T board[8][8];

  const bool fp16 = std::is_same<half, T>::value;

// read the board (a row at a time for fp16)
#pragma unroll
  for (int y = 0; y < 8; y++) {
    if (nhcw) {
      *((uint4*)(&board[y][0])) = *((uint4*)(&input[INDEX_NHCW(n, c, y, 0)]));
      if (!fp16)
        *((uint4*)(&board[y][4])) = *((uint4*)(&input[INDEX_NHCW(n, c, y, 4)]));
    } else {
      *((uint4*)(&board[y][0])) = *((uint4*)(&input[INDEX_NCHW(n, c, y, 0)]));
      if (!fp16)
        *((uint4*)(&board[y][4])) = *((uint4*)(&input[INDEX_NCHW(n, c, y, 4)]));
    }
  }

  // top-left
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = board[i][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = board[i][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = board[i + 3][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = board[i + 3][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
}

#define readw1(row, col) (w1[(row)*se_K + (col)])
#define readw2(row, col) (w2[(row)*2 * C + (col)])

// input is in transformed space (HWNC layout)
// output is NCHW
// 'C' threads per block
// 'N' blocks
// every thread generates an entire board/plane (8x8 elements)
template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
__global__ void OutputTransform_kernel(int N, int C, int se_K, T* output,
                                       const T* input, const T* skip,
                                       const T* bias, const T* w1, const T* b1,
                                       const T* w2, const T* b2) {
  const bool fp16 = std::is_same<half, T>::value;

  int k = threadIdx.x;
  int n = blockIdx.x;

  T board[8][8];
  T b = bias[k];

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4)
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      //  i) read to per thread registers (for doing output transform)
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      T outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++)
#pragma unroll
        for (int x = 0; x < 6; x++)
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];

      // ii) transform it
      T outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++)
#pragma unroll
        for (int x = 0; x < 4; x++) board[hStart + y][wStart + x] = outEl[y][x];
    }

  // Add bias, and compute the average for SE.
  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++)
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) board[y][x] += b;
      if (use_se) S += (float)board[y][x];
    }

  if (use_se) {
    __shared__ float shared_data[1024];
    float avg = S / 64;
    shared_data[k] = avg;
    __syncthreads();

    // First fully-connected layer for SE
    if (k < se_K) {
      S = 0;
      for (int i = 0; i < C; i++) {
        S += shared_data[i] * float(readw1(i, k));
      }
      S += (float)b1[k];
      S = activate(S, activation);
    }
    __syncthreads();
    if (k < se_K) {
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
  }

  // Scale/bias, add skip connection, perform relu, and write to output.
  for (int h = 0; h < 8; h++) {
    if (use_se)
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] = (T)(float(board[h][w]) * S + B);

    // residual add
    if (use_skip) {
      T skipInp[8];
      if (skipInput_nhcw) {
        *((uint4*)(&skipInp[0])) = *((uint4*)(&skip[INDEX_NHCW(n, k, h, 0)]));
        if (!fp16)
          *((uint4*)(&skipInp[4])) = *((uint4*)(&skip[INDEX_NHCW(n, k, h, 4)]));
      } else {
        *((uint4*)(&skipInp[0])) = *((uint4*)(&skip[INDEX_NCHW(n, k, h, 0)]));
        if (!fp16)
          *((uint4*)(&skipInp[4])) = *((uint4*)(&skip[INDEX_NCHW(n, k, h, 4)]));
      }
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[w];
    }

    // relu
    if (activation != NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        board[h][w] = (T)activate((float)board[h][w], activation);
    }

    // Write to output (use 128 bit writes to store one row a time)
    if (output_nhcw) {
      *((uint4*)(&output[INDEX_NHCW(n, k, h, 0)])) = *((uint4*)&board[h][0]);
      if (!fp16)
        *((uint4*)(&output[INDEX_NHCW(n, k, h, 4)])) = *((uint4*)&board[h][4]);
    } else {
      *((uint4*)(&output[INDEX_NCHW(n, k, h, 0)])) = *((uint4*)&board[h][0]);
      if (!fp16)
        *((uint4*)(&output[INDEX_NCHW(n, k, h, 4)])) = *((uint4*)&board[h][4]);
    }
  }
}

// fast reduction for the warp
__device__ __forceinline__ float warpReduce(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    x += __shfl_xor_sync(0xFFFFFFFF, x, mask);

  return x;
}

// Helper fuction to do vector loads/stores
template <typename T>
__device__ __forceinline__ void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

// input is in transformed space (HWNC layout) --- output of GEMM
// output is also in transformed space (HWNC layout) --- input to GEMM (for next
// layer)
// 'C' threads per block
// 'N' blocks
// every thread generates an entire board/plane (8x8 elements)
template <typename T, ActivationFunction activation, bool use_bias,
          bool use_skip>
__global__ __launch_bounds__(kMaxResBlockFusingChannels, 1)
void OutputTransform_SE_relu_InputTransform_kernel(
        int N, int C, int se_K, T* output, const T* input, T* skip,
        const T* bias, const T* w1, const T* b1, const T* w2, const T* b2) {
  const bool fp16 = std::is_same<half, T>::value;

  int k = threadIdx.x;
  int n = blockIdx.x;

  T board[8][8];
  T b = bias[k];

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4)
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      //  i) read to per thread registers (for doing output transform)
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      T outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++)
#pragma unroll
        for (int x = 0; x < 6; x++)
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];

      // ii) transform it
      T outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++)
#pragma unroll
        for (int x = 0; x < 4; x++) board[hStart + y][wStart + x] = outEl[y][x];
    }

  // Add bias, and compute the average for SE.
  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++)
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) board[y][x] += b;
      S += (float)board[y][x];
    }

  {
    __shared__ float shared_data[kMaxResBlockFusingChannels];
    float avg = S / 64;
    shared_data[k] = avg;

    int lane = k & 0x1F;
    int warp = k >> 5;
    __syncthreads();

    // First fully-connected layer for SE

    // As se_K << C, we want to loop over se_K instead of C
    // even if it means taking the sum across threads

    __shared__ float shared_sums[kMaxResBlockFusingChannels / 32]
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
  }

  // Scale/bias, add skip connection, perform relu, and write to output.
  for (int h = 0; h < 8; h++) {
#pragma unroll
    for (int w = 0; w < 8; w++) board[h][w] = (T)(float(board[h][w]) * S + B);

    // residual add
    if (use_skip) {
      T skipInp[8];
      copyAs<uint4>(&skipInp[0], &skip[INDEX_NHCW(n, k, h, 0)]);
      if (!fp16) copyAs<uint4>(&skipInp[4], &skip[INDEX_NHCW(n, k, h, 4)]);
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[w];
    }

    // relu
    if (activation != NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        board[h][w] = (T)activate((float)board[h][w], activation);
    }

    // write un-transformed output to 'skip' if required
    if (use_skip) {
      // Write to skip (use 128 bit writes to store one row a time)
      copyAs<uint4>(&skip[INDEX_NHCW(n, k, h, 0)], &board[h][0]);
      if (!fp16) copyAs<uint4>(&skip[INDEX_NHCW(n, k, h, 4)], &board[h][4]);
    }
  }

  // perform input transform

  int c = k;
  // top-left
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = board[i][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = board[i][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = board[i + 3][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = board[i + 3][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
}


constexpr int kOpInpTransformBlockSize = 64;
template <typename T, ActivationFunction activation, bool use_bias, bool use_skip>
__global__ __launch_bounds__(kOpInpTransformBlockSize, 4)
void OutputTransform_relu_InputTransform_kernel(int N, int C,
                                                T* output, const T* input,
                                                T* skip, const T* bias) {
  const bool fp16 = std::is_same<half, T>::value;

  int k = threadIdx.x + blockIdx.x * kOpInpTransformBlockSize;
  if (k >= C) return;   // wasted threads (for non-multiple of 64 channel counts)
  int n = blockIdx.y;

  T board[8][8];
  T b = bias[k];

  T skipInp[8][8];
#pragma unroll
  for (int h = 0; h < 8; h++) {
    copyAs<uint4>(&skipInp[h][0], &skip[INDEX_NHCW(n, k, h, 0)]);
    if (!fp16) copyAs<uint4>(&skipInp[h][4], &skip[INDEX_NHCW(n, k, h, 4)]);
  }

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4)
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      //  i) read to per thread registers (for doing output transform)
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      T outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++)
#pragma unroll
        for (int x = 0; x < 6; x++)
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];

      // ii) transform it
      T outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++)
#pragma unroll
        for (int x = 0; x < 4; x++) board[hStart + y][wStart + x] = outEl[y][x];
    }

    // Add bias
#pragma unroll
  for (int y = 0; y < 8; y++)
#pragma unroll
    for (int x = 0; x < 8; x++)
      if (use_bias) board[y][x] += b;


  // Add skip connection, perform relu, and write to output.
  for (int h = 0; h < 8; h++) {
    // residual add
    if (use_skip) {
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[h][w];
    }

    // activation
    if (activation != NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        board[h][w] = (T) activate((float)board[h][w], activation);
    }

    // write un-transformed output to 'skip' if required
    if (use_skip) {
      // Write to skip (use 128 bit writes to store one row a time)
      copyAs<uint4>(&skip[INDEX_NHCW(n, k, h, 0)], &board[h][0]);
      if (!fp16) copyAs<uint4>(&skip[INDEX_NHCW(n, k, h, 4)], &board[h][4]);
    }
  }

  // perform input transform

  int c = k;
  // top-left
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = board[i][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = board[i][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = board[i + 3][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = board[i + 3][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
}


template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter) {
  // Each thread processes entire filter block (input 3x3 elements -> output 6x6
  // elements)
  const int kBlockSize = 64;
  const int kBlocks = DivUp(N * C, kBlockSize);

  filterTransform_kernel<<<kBlocks, kBlockSize>>>(N, C, N * C,
                                                  transformedFilter, filter);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformed_input, const T* input,
                    cudaStream_t stream) {
  // Each thread processes entire chess board (input 8x8 elements -> outputs
  // 2x2, 6x6 elements)
  InputTransform_kernel<T, nhcw>
      <<<N, C, 0, stream>>>(N, C, input, transformed_input);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, cudaStream_t stream) {
  // Each thread processes entire chess board
  OutputTransform_kernel<T, use_se, activation, use_bias, use_skip,
                         skipInput_nhcw, output_nhcw><<<N, C, 0, stream>>>(
      N, C, se_K, output, input, skip, bias, w1, b1, w2, b2);
  ReportCUDAErrors(cudaGetLastError());
}

}  // namespace cudnn_backend
}  // namespace lczero
