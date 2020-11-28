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

#include "fully_connected_layer.h"
#include "simple_common.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace lczero {
namespace simple_backend {
namespace {
void ApplyBias(size_t batch_size, const size_t output_size, const float* biases,
               bool apply_relu, bool apply_tanh, float* outputs) {
  if (apply_relu) {
    for (size_t i = 0; i < batch_size; i++) {
      float* batch_outputs = outputs + i * output_size;
      for (size_t o = 0; o < output_size; o++) {
        float val = biases[o] + batch_outputs[o];
        batch_outputs[o] = val >= 0 ? val : 0;
      }
    }
  } else if (apply_tanh) {
    for (size_t i = 0; i < batch_size; i++) {
      float* batch_outputs = outputs + i * output_size;
      for (size_t o = 0; o < output_size; o++) {
        float val = biases[o] + batch_outputs[o];
        batch_outputs[o] = tanh(val);
      }
    }
  } else {
    for (size_t i = 0; i < batch_size; i++) {
      float* batch_outputs = outputs + i * output_size;
      for (size_t o = 0; o < output_size; o++) {
        batch_outputs[o] += biases[o];
      }
    }
  }
}
} // namespace


void FullyConnectedLayer::Forward1D(
    size_t batch_size, const size_t input_size, const size_t output_size,
    const float* inputs, const float* weights, const float* biases,
    bool apply_relu, bool apply_tanh, float* outputs) {

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
    // lda The size of the first dimension of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                (int)output_size,   // M
                (int)batch_size,    // N
                (int)input_size,    // K
                1.0f,               // alpha
                weights,            // A
                (int)input_size,    // lda, leading rank of A
                inputs,             // B
                (int)input_size,    // ldb, leading rank of B
                0.0f,               // beta
                outputs,            // C
                (int)output_size);  // ldc, leading rank of C

    ApplyBias(batch_size, output_size, biases, apply_relu, apply_tanh, outputs);
}

}  // namespace simple_backend
}  // namespace lczero
