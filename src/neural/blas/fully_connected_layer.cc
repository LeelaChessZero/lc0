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

#include "neural/blas/fully_connected_layer.h"
#include "neural/blas/blas.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace lczero {

void FullyConnectedLayer::Forward1D(size_t batch_size, const size_t input_size,
                             const size_t output_size, const float* inputs,
                             const float* weights, const float* biases,
                             bool apply_relu, float* outputs) {
  if (batch_size == 1) {
    
    // Just a matrix-vector multiplication
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
                (int) output_size, (int) input_size, 1.0f, weights, (int) input_size, inputs, 1,
                0.0f, outputs, 1);
  } else {
    
    // more columns, matrix-matrix multiplication
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
                (int) output_size, (int)  batch_size, (int) input_size, 1.0f,
                // A     lda
                weights, (int) input_size,
                // B    ldb   beta,
                inputs, (int) input_size, 0.0f,
                // C   ldc
                outputs, (int)  output_size);
  }
  for (size_t i = 0; i < batch_size; i++) {
    if (apply_relu) {
      for (size_t o = 0; o < output_size; o++) {
        float val = biases[o] + outputs[o];
        outputs[o] = val >= 0 ? val : 0;
      }
    } else {
      for (size_t o = 0; o < output_size; o++) {
        outputs[o] += biases[o];
      }
    }

    outputs += output_size;
    inputs += input_size;
  }
}

float FullyConnectedLayer::Forward0D(const size_t size, const float* x, const float* y) {
  
  // That a scalar product, also known as a dot-produt.
  
  // float cblas_sdot(const int __N, const float *__X, const int __incX, const
  // float *__Y, const int __incY);
  return cblas_sdot((int) size, x, 1, y, 1);
}

void FullyConnectedLayer::Softmax(const size_t size, const float* input,
                             float* output) {
  auto alpha = *std::max_element(input, input + size);

  auto denom = 0.0f;
  for (size_t i = 0; i < size; i++) {
    auto val = std::exp(input[i] - alpha);
    output[i] = val;
    denom += val;
  }
  for (size_t i = 0; i < size; i++) {
    output[i] = output[i] / denom;
  }
}
}
