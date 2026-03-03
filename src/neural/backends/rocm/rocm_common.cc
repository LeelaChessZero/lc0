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

#include "rocm_common.h"

#include <cstdio>

namespace lczero {
namespace rocm_backend {

const char* RocblasGetErrorString(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    case rocblas_status_perf_degraded:
      return "rocblas_status_perf_degraded";
    case rocblas_status_size_query_mismatch:
      return "rocblas_status_size_query_mismatch";
    case rocblas_status_size_increased:
      return "rocblas_status_size_increased";
    case rocblas_status_size_unchanged:
      return "rocblas_status_size_unchanged";
    case rocblas_status_invalid_value:
      return "rocblas_status_invalid_value";
    case rocblas_status_continue:
      return "rocblas_status_continue";
    case rocblas_status_check_numerics_fail:
      return "rocblas_status_check_numerics_fail";
    default:
      return "unknown rocblas error";
  }
}

void RocblasError(rocblas_status status, const char* file, const int& line) {
  if (status != rocblas_status_success) {
    char message[256];
    sprintf(message, "rocBLAS error: %s (%s:%d)", RocblasGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

void HipError(hipError_t status, const char* file, const int& line) {
  if (status != hipSuccess) {
    char message[256];
    sprintf(message, "HIP error: %s (%s:%d)", hipGetErrorString(status), file,
            line);
    throw Exception(message);
  }
}

#ifdef USE_MIOPEN
const char* MiopenGetErrorString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "miopenStatusSuccess";
    case miopenStatusNotInitialized:
      return "miopenStatusNotInitialized";
    case miopenStatusAllocFailed:
      return "miopenStatusAllocFailed";
    case miopenStatusBadParm:
      return "miopenStatusBadParm";
    case miopenStatusInternalError:
      return "miopenStatusInternalError";
    case miopenStatusInvalidValue:
      return "miopenStatusInvalidValue";
    case miopenStatusUnknownError:
      return "miopenStatusUnknownError";
    case miopenStatusNotImplemented:
      return "miopenStatusNotImplemented";
    case miopenStatusUnsupportedOp:
      return "miopenStatusUnsupportedOp";
    default:
      return "unknown MIOpen error";
  }
}

void MiopenError(miopenStatus_t status, const char* file, const int& line) {
  if (status != miopenStatusSuccess) {
    char message[256];
    sprintf(message, "MIOpen error: %s (%s:%d)", MiopenGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}
#endif

}  // namespace rocm_backend
}  // namespace lczero
