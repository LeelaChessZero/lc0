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

#include "neural/blas/transforms.h"
#include "neural/blas/blas.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace lczero {



template <unsigned int filter_size>
void Transforms::Convolve(const int batch_size, const int input_channels,
                          const int output_channels, const float* input,
                          const float* weights, const float* biases,
                          float* output) {
 constexpr unsigned int filter_len = filter_size * filter_size;
  const auto filter_dim = filter_len * input_channels;
  float col[filter_dim * kWidth * kHeight];

  for (int i = 0; i < batch_size; i++) {
    Im2Col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 22 3 3
    // outputs[96,8x8] = weights[96,22x3x3] x col[22x3x3,8x8]
    // C←αAB + βC
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M                  N            K
                output_channels, kSquares, filter_dim, 1.0f, weights,
                filter_dim, col, kSquares, 0.0f, output, kSquares);

    int index = 0;
    for (unsigned int o = 0; o < output_channels; o++) {
      for (unsigned int b = 0; b < kSquares; b++) {
        output[index++] += biases[o];
      }
    }

    input += 64 * input_channels;
    output += 64 * output_channels;
  }
}

template <>
void Transforms::Convolve<1>(const int batch_size, const int input_channels,
                             const int output_channels, const float* input,
                             const float* weights, const float* biases,
                             float* output) {
  for (int i = 0; i < batch_size; i++) {
    // C←αAB + βC
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    //             C                               A                     B
    //
    //           outputs             :=          weights        x      input
    //
    //   cols:     64 (N)                   input_channels (K)         64 (N)
    //
    //   rows:  output_channels (M)         output_channels (M)
    //   input_channels (K)

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M              N         K         alpha
                output_channels, 64, input_channels, 1.0f,
                // A     lda
                weights, input_channels,
                // B    ldb   beta,
                input, 64, 0.0f,
                // C   ldc
                output, 64);

    int index = 0;
    for (int o = 0; o < output_channels; o++) {
      const auto bias=biases[o];
      for (unsigned int b = 0; b < 64; b++) {
        output[index++] += bias;
      }
    }

    input += 64 * input_channels;
    output += 64 * output_channels;
  }
}

void Transforms::Innerproduct(int batch_size, const int input_size,
                              const int output_size, const float* inputs,
                              const float* weights, const float* biases,
                              bool apply_relu, float* outputs) {
  if (batch_size == 1) {
    // Just matrix-vecttor multplication
    //
    //             C                A                     B
    //
    //         outputs    :=     weights      x       inputs
    //
    //   cols:   1               input_size            1
    //
    //   rows  output_size      output_size          input_size
    //

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                output_size, input_size, 1.0f, weights, input_size, inputs, 1,
                0.0f, outputs, 1);
  } else {
    // matrix-matrix multplication
    //
    //             C                     A                         B
    //
    //            outputs      :=       weights        x         inputs
    //
    //   cols:   batch_size (N)       input_size  (K)          batch_size (N)
    //
    //   rows  output_size (M)        output_size (M)         input_size (K)
    //

    // C←αAB + βC
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                // M              N         K         alpha
                output_size, batch_size, input_size, 1.0f,
                // A     lda
                weights, input_size,
                // B    ldb   beta,
                inputs, input_size, 0.0f,
                // C   ldc
                outputs, output_size);
  }
  for (int i = 0; i < batch_size; i++) {
    if (apply_relu) {
      for (int o = 0; o < output_size; o++) {
        float val = biases[o] + outputs[o];
        outputs[o] = val >= 0 ? val : 0;
      }
    } else {
      for (int o = 0; o < output_size; o++) {
        outputs[o] += biases[o];
      }
    }

    outputs += output_size;
    inputs += input_size;
  }
}

void Transforms::Batchnorm(const int batch_size, const int channels,
                           float* data, const float* means,
                           const float* stddivs, const float* eltwise) {
  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < channels; ++c) {
      auto mean = means[c];
      auto scale_stddiv = stddivs[c];

      if (eltwise == nullptr) {
        // Classical BN
        auto arr = &data[c * 64];
        for (int b = 0; b < 64; b++) {
          float val = scale_stddiv * (arr[b] - mean);
          arr[b] = val > 0 ? val : 0;
        }
      } else {
        // BN + residual add
        auto arr = &data[c * 64];
        auto res = &eltwise[c * 64];
        for (int b = 0; b < 64; b++) {
          float val = res[b] + (scale_stddiv * (arr[b] - mean));
          arr[b] = val > 0 ? val : 0;
        }
      }
    }
    data += channels * 64;
    if (eltwise != nullptr) eltwise += channels * 64;
  }
}

template <unsigned long filter_size>
void Transforms::Im2Col(const int channels, const float* data_im,
                        float* data_col) {
  constexpr unsigned int height = 8;
  constexpr unsigned int width = 8;
  constexpr unsigned int channel_size = height * width;

  constexpr int pad = (filter_size / 2);
  constexpr unsigned int output_h = height + 2 * pad - filter_size + 1;
  constexpr unsigned int output_w = width + 2 * pad - filter_size + 1;

  for (int channel = channels; channel--; data_im += channel_size) {
    for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
      for (unsigned int kernel_col = 0; kernel_col < filter_size;
           kernel_col++) {
        int input_row = -pad + kernel_row;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if ((unsigned)input_row < height) {
            int input_col = -pad + kernel_col;
            for (int output_col = output_w; output_col; output_col--) {
              if ((unsigned)input_col < width) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col++;
            }
          } else {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          }
          input_row++;
        }
      }
    }
  }
}

float Transforms::DotProduct(const int size, const float* x, const float* y) {
  // float cblas_sdot(const int __N, const float *__X, const int __incX, const
  // float *__Y, const int __incY);
  return cblas_sdot(size, x, 1, y, 1);
}

void Transforms::Softmax(const int size, const float* input, float* output) {
  auto alpha = *std::max_element(input, input + size);

  auto denom = 0.0f;
  for (int i = 0; i < size; i++) {
    auto val = std::exp(input[i] - alpha);
    output[i] = val;
    denom += val;
  }
  for (int i = 0; i < size; i++) {
    output[i] = output[i] / denom;
  }
}

void Transforms::OffsetBatchNormMeans(std::vector<float>& bn_means,
                                      const std::vector<float>& biases) {
  // Biases are not calculated and are typically zero but some networks might
  // still have non-zero biases.
  // Move biases to batchnorm means to make the output match without having
  // to separately add the biases.
  for (size_t i = 0; i < bn_means.size(); i++) bn_means[i] -= biases[i];
}

void Transforms::InvertBatchNormStddev(std::vector<float>& weights) {
  constexpr float EPSILON = 1e-5f;
  for (auto& w : weights) w = 1.0f / std::sqrt(w + EPSILON);
}

/* Template instantiations and specializations */

template <>
void Transforms::Im2Col<1>(const int channels, const float* input,
                           float* output) {
  constexpr unsigned int boardsize = 8;
  auto outSize = size_t{channels * boardsize * boardsize};
  std::copy(input, input + outSize, output);
}
}
