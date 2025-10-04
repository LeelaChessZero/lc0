/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include <cuda_bf16.h>

#include "cuda_common.h"
#include "onnx_kernels.h"

namespace lczero {
namespace cudnn_backend {

template <unsigned bits_per_thread, typename DataType>
__global__ void expandPlanes_kernel(DataType* output, const uint64_t* masks,
                                    const DataType* values, unsigned n) {
  unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
  index *= bits_per_thread;
  unsigned planeIndex = index >> 6;
  if (planeIndex >= n) return;

  uint64_t mask = masks[planeIndex];
  unsigned sqIndex = index & 0x3F;
  DataType value = static_cast<DataType>(values[planeIndex]);
  DataType op[bits_per_thread] = {};
  mask >>= sqIndex;
  for (unsigned i = 0; i < bits_per_thread; i++) {
    if (mask & 0x1) {
      op[i] = value;
    }
    mask >>= 1;
  }
  for (unsigned i = 0; i < bits_per_thread; i++) {
    output[index + i] = op[i];
  }
}

template <typename DataType>
void expandPlanesOnnx(DataType* output, const void* input, unsigned n,
                      cudaStream_t stream) {
  constexpr unsigned bits_per_thread = 2;
  int threads = n * 8 * 8 / bits_per_thread;
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);

  const uint64_t* masks = static_cast<const uint64_t*>(input);
  const DataType* values = reinterpret_cast<const DataType*>(masks + n);

  expandPlanes_kernel<bits_per_thread>
      <<<blocks, blockSize, 0, stream>>>(output, masks, values, n);

  ReportCUDAErrors(cudaGetLastError());
}

template void expandPlanesOnnx<half>(half* output, const void* input,
                                     unsigned n, cudaStream_t stream);
template void expandPlanesOnnx<float>(float* output, const void* input,
                                      unsigned n, cudaStream_t stream);
template void expandPlanesOnnx<__nv_bfloat16>(__nv_bfloat16* output,
                                              const void* input, unsigned n,
                                              cudaStream_t stream);

}  // namespace cudnn_backend
}  // namespace lczero
