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

#include <Eigen/Dense>

namespace lczero {
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

#ifdef USE_BLAS
template <>
void Convolution1<false>::Forward(const size_t batch_size,
                                  const size_t input_channels,
                                  const size_t output_channels,
                                  const float* input, const float* weights,
                                  float* output) {
  for (size_t i = 0; i < batch_size; i++) {
    // C←αAB + βC
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimension of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    //             C                          A                     B
    //
    //           outputs       :=          weights        x      input
    //
    //   cols:  kSquares (N)         input_channels (K)        kSquares(N)
    //
    //   rows:  output_channels (M)   output_channels (M)  input_channels (K)

    const float* batch_input = input + i * kSquares * input_channels;
    float* batch_output = output + i * kSquares * output_channels;
    cblas_sgemm(CblasRowMajor,         // Row major formar
                CblasNoTrans,          // A not transposed
                CblasNoTrans,          // B not transposed
                (int)output_channels,  // M
                kSquares,              // N
                (int)input_channels,   // K
                1.0f,                  // Alpha
                weights,               // A
                (int)input_channels,   // lda, leading rank of A
                batch_input,           // B
                kSquares,              // ldb, leading rank of B
                0.0f,                  // beta
                batch_output,          // C
                kSquares);             // ldc, leading rank of B
  }
}
#endif

template <>
void Convolution1<true>::Forward(const size_t batch_size,
                                 const size_t input_channels,
                                 const size_t output_channels,
                                 const float* input, const float* weights,
                                 float* output) {
  for (size_t i = 0; i < batch_size; i++) {
    const float* batch_input = input + i * kSquares * input_channels;
    float* batch_output = output + i * kSquares * output_channels;
    auto C_mat = EigenMatrixMap<float>(batch_output, kSquares, output_channels);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(batch_input, kSquares, input_channels) *
        ConstEigenMatrixMap<float>(weights, input_channels, output_channels);
  }
}

}  // namespace lczero
