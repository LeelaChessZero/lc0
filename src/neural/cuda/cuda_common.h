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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "utils/exception.h"

namespace lczero {
namespace cudnn_backend {

void CudnnError(cudnnStatus_t status, const char* file, const int& line);
void CublasError(cublasStatus_t status, const char* file, const int& line);
void CudaError(cudaError_t status, const char* file, const int& line);

#define ReportCUDNNErrors(status) CudnnError(status, __FILE__, __LINE__)
#define ReportCUBLASErrors(status) CublasError(status, __FILE__, __LINE__)
#define ReportCUDAErrors(status) CudaError(status, __FILE__, __LINE__)

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

}  // namespace cudnn_backend
}  // namespace lczero
