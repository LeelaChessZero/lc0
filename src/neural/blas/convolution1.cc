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

#include "neural/blas/convolution1.h"
#include "neural/blas/blas.h"

namespace lczero {


void Convolution1::Forward(const int batch_size, const int input_channels,
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

    const float* input_batch=input+i*kSquares * input_channels;
    float* output_batch=output+i*kSquares * output_channels;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M              N         K         alpha
                output_channels, 64, input_channels, 1.0f,
                // A     lda
                weights, input_channels,
                // B    ldb   beta,
                input_batch, 64, 0.0f,
                // C   ldc
                output_batch, 64);

    int index = 0;
    for (int o = 0; o < output_channels; o++) {
      const auto bias = biases[o];
      for (unsigned int b = 0; b < 64; b++) {
        output_batch[index++] += bias;
      }
    }
  }
}

}  // namespace lczerp
