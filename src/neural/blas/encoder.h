/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2019 The LCZero Authors

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

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstddef>

#include "neural/shared/activation.h"
#include "utils/exception.h"

namespace lczero {

namespace {

template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

}  // namespace

void LayerNorm2DWithSkipConnection(const size_t batch_size,
                                   const size_t channels, float* data,
                                   const float* skip, const float* gammas,
                                   const float* betas, float epsilon) {
  for (size_t i = 0; i < batch_size; i++) {
    // Mean taken in dimension C.
    float mean = 0;
    for (size_t c = 0; c < channels; ++c) {
      data[i * channels + c] += skip[i * channels + c];
      mean += data[i * channels + c];
    }
    mean /= channels;

    // Variance.
    float var = 0;
    for (size_t c = 0; c < channels; ++c) {
      auto diff = data[i * channels + c] - mean;
      var += diff * diff;
    }
    var /= channels;

    // Norm.
    for (size_t c = 0; c < channels; ++c) {
      data[i * channels + c] = betas[c] + gammas[c] *
                                              (data[i * channels + c] - mean) /
                                              std::sqrt(var + epsilon);
    }
  }
}

template <bool use_eigen>
void AttentionMatmul2D(const bool transpose_a, const bool transpose_b,
                       const size_t batch_size, const size_t M, const size_t N,
                       const size_t K, const float scaling, const float* input1,
                       const float* input2, float* output) {
  for (auto batch = size_t{0}; batch < batch_size; batch++) {
    const float* A = &input1[batch * M * K];
    const float* B = &input2[batch * N * K];
    float* C = &output[batch * M * N];
    if (use_eigen) {
      auto C_mat = EigenMatrixMap<float>(C, N, M);

      if (transpose_a && transpose_b) {
        C_mat.noalias() = scaling *
                          ConstEigenMatrixMap<float>(B, K, N).transpose() *
                          ConstEigenMatrixMap<float>(A, M, K).transpose();
      } else if (transpose_a) {
        C_mat.noalias() = scaling * ConstEigenMatrixMap<float>(B, N, K) *
                          ConstEigenMatrixMap<float>(A, M, K).transpose();
      } else if (transpose_b) {
        C_mat.noalias() = scaling *
                          ConstEigenMatrixMap<float>(B, K, N).transpose() *
                          ConstEigenMatrixMap<float>(A, K, M);
      } else {
        C_mat.noalias() = scaling * ConstEigenMatrixMap<float>(B, N, K) *
                          ConstEigenMatrixMap<float>(A, K, M);
      }
    } else {
#ifdef USE_BLAS
      cblas_sgemm(CblasRowMajor, transpose_a ? CblasTrans : CblasNoTrans,
                  transpose_b ? CblasTrans : CblasNoTrans, M, N, K, scaling, A,
                  transpose_a ? M : K, B, transpose_b ? K : N, 0.0f, C, N);
#else
      // Should never get here.
      throw Exception("Blas backend internal error");
#endif
    }
  }
}

}  // namespace lczero
