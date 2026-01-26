/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <stdexcept>
#include <string>
#include "rocm_common.h"

namespace lczero {
namespace rocm_backend {

// Global handle cache for hipBLASLt (to avoid expensive handle creation per call)
static hipblasLtHandle_t& GetCachedHipblasLtHandle() {
  static hipblasLtHandle_t handle = nullptr;
  static bool initialized = false;

  if (!initialized) {
    if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
      handle = nullptr;
    }
    initialized = true;
  }
  return handle;
}

// Global workspace for Split-K algorithms
static void*& GetCachedWorkspace() {
  static void* workspace = nullptr;
  static bool initialized = false;
  static constexpr size_t WORKSPACE_SIZE = 8 * 1024 * 1024;  // 8MB

  if (!initialized) {
    if (hipMalloc(&workspace, WORKSPACE_SIZE) != hipSuccess) {
      workspace = nullptr;
    }
    initialized = true;
  }
  return workspace;
}

// Simple helper for GEMM with bias fusion using hipBLASLt
// Fuses matrix multiply + bias add into single kernel call
//
// For now, we only support bias (no activation fusion) as SWISH is not
// natively supported. This still provides significant speedup by eliminating
// one kernel launch overhead.

template <typename DataType>
inline bool HipblasLtGemmWithBias(
    hipStream_t stream,
    rocblas_operation transpose_a,
    rocblas_operation transpose_b,
    int m, int n, int k,
    float alpha,
    const DataType* A, int lda,
    const DataType* B, int ldb,
    float beta,
    DataType* C, int ldc,
    const DataType* bias) {

  // Only FP16 supported for now - return false for other types
  if (!std::is_same<DataType, half>::value) {
    return false;
  }

  // Use cached handle
  hipblasLtHandle_t handle = GetCachedHipblasLtHandle();
  if (handle == nullptr) {
    return false;  // Failed to create handle
  }

  // Convert to hipBLAS operations
  hipblasOperation_t opA = (transpose_a == rocblas_operation_transpose)
                              ? HIPBLAS_OP_T
                              : HIPBLAS_OP_N;
  hipblasOperation_t opB = (transpose_b == rocblas_operation_transpose)
                              ? HIPBLAS_OP_T
                              : HIPBLAS_OP_N;

  // Matrix dimensions
  int rows_A = (opA == HIPBLAS_OP_T) ? k : m;
  int cols_A = (opA == HIPBLAS_OP_T) ? m : k;
  int rows_B = (opB == HIPBLAS_OP_T) ? n : k;
  int cols_B = (opB == HIPBLAS_OP_T) ? k : n;

  // Create matrix layouts
  hipblasLtMatrixLayout_t matA, matB, matC;
  hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, rows_A, cols_A, lda);
  hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, rows_B, cols_B, ldb);
  hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, ldc);

  // Create matmul descriptor
  hipblasLtMatmulDesc_t matmul_desc;
  hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  hipblasLtMatmulDescSetAttribute(
      matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
  hipblasLtMatmulDescSetAttribute(
      matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

  // Set epilogue for bias
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
  hipblasLtMatmulDescSetAttribute(
      matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  hipblasLtMatmulDescSetAttribute(
      matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));

  // Get algorithm with Split-K support for better GPU utilization
  hipblasLtMatmulPreference_t pref;
  hipblasLtMatmulPreferenceCreate(&pref);

  // Set max workspace to enable Split-K algorithms (8MB should be plenty)
  size_t max_workspace = 8 * 1024 * 1024;
  hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace, sizeof(max_workspace));

  // Request multiple algorithms so heuristic can consider Split-K
  constexpr int MAX_ALGOS = 10;
  hipblasLtMatmulHeuristicResult_t heuristic_results[MAX_ALGOS];
  int returned_algo_count = 0;
  hipblasLtMatmulAlgoGetHeuristic(
      handle, matmul_desc, matA, matB, matC, matC, pref, MAX_ALGOS,
      heuristic_results, &returned_algo_count);

  bool success = false;
  if (returned_algo_count > 0) {
    // Convert alpha/beta to FP16
    half alpha_h = __float2half(alpha);
    half beta_h = __float2half(beta);

    // Get cached workspace for Split-K
    void* workspace = GetCachedWorkspace();
    size_t workspace_size = (workspace != nullptr) ? (8 * 1024 * 1024) : 0;

    // Try algorithms in order returned by heuristic (best first)
    // Heuristic should prioritize Split-K for small batch sizes
    for (int i = 0; i < returned_algo_count; i++) {
      hipblasStatus_t status = hipblasLtMatmul(
          handle, matmul_desc, &alpha_h, A, matA, B, matB, &beta_h, C, matC, C,
          matC, &heuristic_results[i].algo, workspace, workspace_size, stream);

      if (status == HIPBLAS_STATUS_SUCCESS) {
        success = true;
        break;
      }
    }
  }

  // Cleanup (but keep handle cached)
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatmulDescDestroy(matmul_desc);
  hipblasLtMatrixLayoutDestroy(matC);
  hipblasLtMatrixLayoutDestroy(matB);
  hipblasLtMatrixLayoutDestroy(matA);
  // Note: handle is cached globally, not destroyed here

  return success;
}

}  // namespace rocm_backend
}  // namespace lczero
