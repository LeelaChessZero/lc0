/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2022 The LCZero Authors

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
#include "neural/shared/activation.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include <Eigen/Dense>

namespace lczero {
namespace {
void ApplyBias(size_t batch_size, const size_t output_size, const float* biases,
               const ActivationFunction activation, float* outputs) {
  for (size_t i = 0; i < batch_size; i++) {
    float* batch_outputs = outputs + i * output_size;
    Activate(output_size, batch_outputs, biases, batch_outputs, activation);
  }
}
}  // namespace

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

#ifdef USE_BLAS
template <>
void FullyConnectedLayer<false>::Forward1D(
    size_t batch_size, const size_t input_size, const size_t output_size,
    const float* inputs, const float* weights, const float* biases,
    const ActivationFunction activation, float* outputs) {
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
                (int)output_size, (int)input_size, 1.0f, weights,
                (int)input_size, inputs, 1, 0.0f, outputs, 1);
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
  }
  ApplyBias(batch_size, output_size, biases, activation, outputs);
}

template <>
float FullyConnectedLayer<false>::Forward0D(const size_t size, const float* x,
                                            const float* y) {
  // A scalar product, also known as a dot-product.
  // float cblas_sdot(const int N, const float *X, const int incX, const float
  // *Y,
  // const int incY);
  return cblas_sdot((int)size, x, 1, y, 1);
}
#endif

template <>
void FullyConnectedLayer<true>::Forward1D(
    size_t batch_size, const size_t input_size, const size_t output_size,
    const float* inputs, const float* weights, const float* biases,
    const ActivationFunction activation, float* outputs) {
  if (batch_size == 1) {
    EigenVectorMap<float> y(outputs, output_size);
    y.noalias() = ConstEigenMatrixMap<float>(weights, input_size, output_size)
                      .transpose() *
                  ConstEigenVectorMap<float>(inputs, input_size);
  } else {
    auto C_mat = EigenMatrixMap<float>(outputs, output_size, batch_size);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(weights, input_size, output_size)
            .transpose() *
        ConstEigenMatrixMap<float>(inputs, input_size, batch_size);
  }
  ApplyBias(batch_size, output_size, biases, activation, outputs);
}

template <>
float FullyConnectedLayer<true>::Forward0D(const size_t size, const float* x,
                                           const float* y) {
  return ConstEigenVectorMap<float>(x, size).dot(
      ConstEigenVectorMap<float>(y, size));
}

}  // namespace lczero
