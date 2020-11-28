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
#pragma once

#if defined(USE_OPENBLAS)
#include <cblas.h>

#elif defined(USE_DNNL)
#include <dnnl.h>

// Implement the cblas subset needed using dnnl_sgemm().
extern "C" {
#define CblasRowMajor 0
#define CblasColMajor 1
#define CblasNoTrans 'N'
#define CblasTrans 'T'
static inline void cblas_sgemm(char order, char transa, char transb,
                               dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K,
                               float alpha, const float *A, dnnl_dim_t lda,
                               const float *B, dnnl_dim_t ldb, float beta,
                               float *C, dnnl_dim_t ldc) {
  // DNNL only has row major sgemm.
  if (order == CblasRowMajor) {
    dnnl_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    dnnl_sgemm(transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
  }
}

}

#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>

#endif
