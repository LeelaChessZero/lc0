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

// Select the BLAS vendor based on defines

#ifdef USE_MKL
#include <mkl.h>
#else

#ifdef USE_OPENBLAS
#include <cblas.h>

// Specific openblas routines.
extern "C" {
int openblas_get_num_procs(void);
void openblas_set_num_threads(int num_threads);
char* openblas_get_corename(void);
char* openblas_get_config(void);
}

#else

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE

#else

#ifdef USE_DNNL
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

static inline void cblas_sgemv(char order, char transa, dnnl_dim_t M,
                               dnnl_dim_t N, float alpha, const float *A,
                               dnnl_dim_t lda, const float *x, dnnl_dim_t incx,
                               float beta, float *y, dnnl_dim_t incy) {
  cblas_sgemm(order, transa, 'N', M, 1, N, alpha, A, lda, x, incx, beta, y,
              incy);
}

static inline float cblas_sdot(dnnl_dim_t N, const float *x, dnnl_dim_t incx,
                               const float *y, dnnl_dim_t incy) {
  float r = 0;
  dnnl_sgemm('T', 'N', 1, 1, N, 1.0, x, incx, y, incy, 0.0, &r, 1);
  return r;
}
}

#endif

#endif  // __APPLE__

#endif  // USE_OPENBLAS

#endif  // USE_MKL
