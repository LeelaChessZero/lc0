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
__device__ __forceinline__ void OutputTransform4x4(
    T* output, const T* transformedOutput) {
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


// index in intermediate/temp tensor
// W, H == 6 here! (6x6 transformed blocks)
// N also includes part of dimension (2x2)
#define GemmN (N * 4)
#define INDEX_NCHW(n, c, h, w) ((n)*C * 8 * 8 + (c)*8 * 8 + (h)*8 + w)
#define TEMP_INDEX_HWNC(h, w, n, c) \
  ((h)*6 * GemmN * C + (w)*GemmN * C + (n)*C + c)

// 'C' threads per block
// 'N' blocks
// every thread transforms an entire board/plane (8x8 elements)
// - producing 4 x 6x6 elements
template <typename T>
__global__ void InputTransform_kernel(int N, int C, const T* input, T* output) {
  int c = threadIdx.x;
  int n = blockIdx.x;

  T board[8][8];

  const bool fp16 = std::is_same<half, T>::value;

// read the board (a row at a time for fp16)
#pragma unroll
  for (int y = 0; y < 8; y++) {
    *((uint4*)(&board[y][0])) = *((uint4*)(&input[INDEX_NCHW(n, c, y, 0)]));
    if (!fp16)
      *((uint4*)(&board[y][4])) = *((uint4*)(&input[INDEX_NCHW(n, c, y, 4)]));
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
template <typename T, bool use_se, bool relu, bool use_bias, bool use_skip>
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
      if (S < 0) S = 0;  // relu
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
      *((uint4*)(&skipInp[0])) = *((uint4*)(&skip[INDEX_NCHW(n, k, h, 0)]));
      if (!fp16)
        *((uint4*)(&skipInp[4])) = *((uint4*)(&skip[INDEX_NCHW(n, k, h, 4)]));
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[w];
    }

    // relu
    if (relu) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        if (board[h][w] < (T)0) board[h][w] = 0;
    }

    // Write to output (use 128 bit writes to store one row a time)
    *((uint4*)(&output[INDEX_NCHW(n, k, h, 0)])) = *((uint4*)&board[h][0]);
    if (!fp16)
      *((uint4*)(&output[INDEX_NCHW(n, k, h, 4)])) = *((uint4*)&board[h][4]);
  }
}


// input is in transformed space (HWNC layout) --- output of GEMM
// output is also in transformed space (HWNC layout) --- input to GEMM (for next layer)
// 'C' threads per block
// 'N' blocks
// every thread generates an entire board/plane (8x8 elements)
template <typename T, bool use_se, bool relu, bool use_bias, bool use_skip>
__global__ 
void OutputTransform_SE_relu_InputTransform_kernel(int N, int C, int se_K, T* output,
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
      if (S < 0) S = 0;  // relu
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
      *((uint4*)(&skipInp[0])) = *((uint4*)(&skip[INDEX_NCHW(n, k, h, 0)]));
      if (!fp16)
        *((uint4*)(&skipInp[4])) = *((uint4*)(&skip[INDEX_NCHW(n, k, h, 4)]));
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[w];
    }

    // relu
    if (relu) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        if (board[h][w] < (T)0) board[h][w] = 0;
    }

    // write un-transformed output to 'skip' if required
    if (use_skip)
    {
      // Write to skip (use 128 bit writes to store one row a time)
      *((uint4*)(&skip[INDEX_NCHW(n, k, h, 0)])) = *((uint4*)&board[h][0]);
      if (!fp16)
        *((uint4*)(&skip[INDEX_NCHW(n, k, h, 4)])) = *((uint4*)&board[h][4]);
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

template <typename T>
void InputTransform(int N, int C, T* transformed_input, const T* input) {
  // Each thread processes entire chess board (input 8x8 elements -> outputs
  // 2x2, 6x6 elements)
  InputTransform_kernel<<<N, C>>>(N, C, input, transformed_input);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T, bool use_se, bool relu, bool use_bias, bool use_skip>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2) {
  // Each thread processes entire chess board
  OutputTransform_kernel<T, use_se, relu, use_bias, use_skip>
      <<<N, C>>>(N, C, se_K, output, input, skip, bias, w1, b1, w2, b2);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T, bool use_se, bool relu, bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                         const T* skip, const T* bias, const T* w1, const T* b1,
                         const T* w2, const T* b2) {
  // Each thread processes entire chess board
  OutputTransform_SE_relu_InputTransform_kernel<T, use_se, relu, use_bias, use_skip>
                    <<<N, C>>>(N, C, se_K, output, input, skip, bias, w1, b1, w2, b2);
  ReportCUDAErrors(cudaGetLastError());
}


}  // namespace cudnn_backend
}  // namespace lczero
