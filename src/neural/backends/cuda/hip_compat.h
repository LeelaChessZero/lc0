/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors
  Copyright (c) 2026 Advanced Micro Devices, Inc.

  Author: Jeff Daily <jeff.daily@amd.com>

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

// Single CUDA->HIP/ROCm compatibility shim for the cuda backend. It is
// force-included into every HIP translation unit by the meson -Dhip build and
// is pulled in by cuda_common.h under USE_HIP, so the rest of the backend keeps
// its CUDA spelling. The NVIDIA build never sees this file.
#pragma once

#if defined(USE_HIP)

// Pull host string/memory functions in before the HIP runtime so plain memcpy/
// memset resolve to the host overloads, not HIP device builtins.
#include <cstdlib>
#include <cstring>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <hipblas/hipblas.h>

// fp16_kernels.cu and the fp16 bodies in winograd_helper.inc gate their device
// code on HAS_FP16_SUPPORT. The NVIDIA build defines it only for compute
// capability >= 5.3, where native fp16 arithmetic exists. Every AMD GPU that ROCm
// targets has native fp16 (GCN 5.0 / gfx900 and newer), so define it
// unconditionally here. Without it those fp16 SE / fused conv-transform kernels
// would compile to empty bodies and leave their output uninitialized.
#define HAS_FP16_SUPPORT 1

// Deliberately report a pre-11.0 CUDA runtime so every `CUDART_VERSION >= 11000`
// / `>= 11010` block in the backend (NVIDIA L2-persistence cache hints, CUDA
// graph external-event flags, the >= 13000 clock-rate path) compiles out, while
// the plain arithmetic uses of CUDART_VERSION in showInfo() still work. There
// is no ROCm equivalent for those NVIDIA-only features.
#ifndef CUDART_VERSION
#define CUDART_VERSION 10020
#endif

// --- runtime: types ---------------------------------------------------------
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorInitializationError hipErrorNotInitialized
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureMode hipStreamCaptureMode
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t

// --- runtime: enums / flags -------------------------------------------------
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaDevAttrClockRate hipDeviceAttributeClockRate

// --- runtime: device / stream / event / memory ------------------------------
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaFreeHost hipFreeHost

#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture

#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventRecordWithFlags hipEventRecordWithFlags
#define cudaEventSynchronize hipEventSynchronize

#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload

// hipFuncSetAttribute only takes the kernel as a const void* (CUDA also has a
// templated T* overload). The single call site casts its kernel pointer.
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
  hipFuncAttributeMaxDynamicSharedMemorySize

// --- device intrinsics ------------------------------------------------------
// HIP's device runtime does not declare CUDA's __trap(); the builtin is the
// portable device-abort and is accepted by both nvcc and hipcc.
#define __trap __builtin_trap

// --- fp16 -------------------------------------------------------------------
// __half / __half_raw / half come from <hip/hip_fp16.h> with matching names.

// --- library data types -----------------------------------------------------
// hipDataType from <hip/library_types.h>. These are correct in the GEMM *data*
// type positions; the compute-type position is handled by the GEMM shims below.
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F

// --- cuBLAS: handle / ops / status ------------------------------------------
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasOperation_t hipblasOperation_t
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
// hipBLAS has no LICENSE_ERROR; fold it into the catch-all status code.
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_UNKNOWN

// --- cuBLAS: plain GEMM entry points (1:1) ----------------------------------
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasSgemmBatched hipblasSgemmBatched

#define NS_BACKEND hip_backend
#define BACKEND_NAME "HIP"
#define BACKEND_NAME_LC "hip"
namespace lczero {
namespace hip_backend {

// CUDA's cublasHgemm / cublasHgemmBatched take __half*, but hipBLAS types its
// fp16 GEMMs on hipblasHalf (uint16_t). The bit layouts are identical; these
// shims accept the __half* the call sites already cast to and reinterpret to
// hipblasHalf so the call sites stay unchanged.
inline hipblasStatus_t lc0HipHgemm(hipblasHandle_t handle,
                                   hipblasOperation_t transa,
                                   hipblasOperation_t transb, int m, int n,
                                   int k, const __half* alpha, const __half* A,
                                   int lda, const __half* B, int ldb,
                                   const __half* beta, __half* C, int ldc) {
  return hipblasHgemm(handle, transa, transb, m, n, k,
                      reinterpret_cast<const hipblasHalf*>(alpha),
                      reinterpret_cast<const hipblasHalf*>(A), lda,
                      reinterpret_cast<const hipblasHalf*>(B), ldb,
                      reinterpret_cast<const hipblasHalf*>(beta),
                      reinterpret_cast<hipblasHalf*>(C), ldc);
}

inline hipblasStatus_t lc0HipHgemmBatched(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k, const __half* alpha, __half* const A[], int lda,
    __half* const B[], int ldb, const __half* beta, __half* const C[], int ldc,
    int batchCount) {
  return hipblasHgemmBatched(
      handle, transa, transb, m, n, k,
      reinterpret_cast<const hipblasHalf*>(alpha),
      reinterpret_cast<const hipblasHalf* const*>(A), lda,
      reinterpret_cast<const hipblasHalf* const*>(B), ldb,
      reinterpret_cast<const hipblasHalf*>(beta),
      reinterpret_cast<hipblasHalf* const*>(C), ldc, batchCount);
}

// The cublas*Ex calls pass CUDA_R_16F / CUDA_R_32F in BOTH the data-type and the
// compute-type argument slots. In hipBLAS v2 (ROCm 7.x) the data-type slots take
// hipDataType (CUDA_R_* already maps there) but the compute-type slot takes a
// distinct hipblasComputeType_t. These shims accept the compute argument as a
// hipDataType (what the macro expansion produces) and translate it, so the call
// sites stay in CUDA spelling and pick the precision-correct compute type.
inline hipblasComputeType_t hipblasComputeFromDataType(hipDataType t) {
  switch (t) {
    case HIP_R_16F:
      return HIPBLAS_COMPUTE_16F;
    case HIP_R_32F:
    default:
      return HIPBLAS_COMPUTE_32F;
  }
}

inline hipblasStatus_t lc0HipGemmStridedBatchedEx(
    hipblasHandle_t handle, hipblasOperation_t transa,
    hipblasOperation_t transb, int m, int n, int k, const void* alpha,
    const void* A, hipDataType aType, int lda, long long int strideA,
    const void* B, hipDataType bType, int ldb, long long int strideB,
    const void* beta, void* C, hipDataType cType, int ldc,
    long long int strideC, int batchCount, hipDataType computeType,
    hipblasGemmAlgo_t algo) {
  return hipblasGemmStridedBatchedEx(
      handle, transa, transb, m, n, k, alpha, A, aType, lda, strideA, B, bType,
      ldb, strideB, beta, C, cType, ldc, strideC, batchCount,
      hipblasComputeFromDataType(computeType), algo);
}

}  // namespace hip_backend
}  // namespace lczero

#define cublasGemmStridedBatchedEx \
  ::lczero::hip_backend::lc0HipGemmStridedBatchedEx
#define cublasHgemm ::lczero::hip_backend::lc0HipHgemm
#define cublasHgemmBatched ::lczero::hip_backend::lc0HipHgemmBatched

#endif  // USE_HIP
