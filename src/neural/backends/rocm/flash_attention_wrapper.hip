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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with AMD ROCm libraries from the ROCm toolkit
  (or a modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "kernels.h"
#include "flash_attention.hip"
#include <type_traits>

namespace lczero {
namespace rocm_backend {

// Flash Attention wrapper implementation
// This function wraps the flash attention kernel and provides a clean interface
// for the lc0 backend to use fused attention computation
template <typename T>
bool flash_attention_wrapper(
    const T* Q,           // Query: [batch, seq_len, head0_depth, head1_depth, ..., headN_depth]
    const T* K,           // Key: same layout as Q
    const T* V,           // Value: same layout as Q
    T* output,            // Output: same layout as Q/K/V
    float scale,          // Scale factor: 1.0 / sqrt(depth)
    int batch,            // Batch size (N)
    int num_heads,        // Number of attention heads
    int d_model,          // Total embedding dimension (depth * num_heads)
    int depth,            // Per-head dimension
    rocblas_handle cublas,
    hipStream_t stream
) {
    // Verify input parameters
    if (batch <= 0 || num_heads <= 0 || d_model <= 0 || depth <= 0) {
        static bool param_log_once = false;
        if (!param_log_once) {
            printf("[FLASH ATTENTION DEBUG] Invalid parameters: batch=%d, heads=%d, d_model=%d, depth=%d\n",
                   batch, num_heads, d_model, depth);
            param_log_once = true;
        }
        return false;  // Invalid parameters, fallback to rocBLAS
    }

    // Verify d_model == depth * num_heads
    if (d_model != depth * num_heads) {
        static bool dim_log_once = false;
        if (!dim_log_once) {
            printf("[FLASH ATTENTION DEBUG] Dimension mismatch: d_model=%d != depth=%d * heads=%d = %d\n",
                   d_model, depth, num_heads, depth * num_heads);
            dim_log_once = true;
        }
        return false;  // Inconsistent dimensions, fallback to rocBLAS
    }

    // Check if flash attention supports this configuration
    // Currently supported: T82 configurations (depth = 32, 64, 96, 128)
    if (depth != 32 && depth != 64 && depth != 96 && depth != 128) {
        static bool depth_log_once = false;
        if (!depth_log_once) {
            printf("[FLASH ATTENTION DEBUG] Unsupported depth=%d (only 32,64,96,128 supported)\n", depth);
            depth_log_once = true;
        }
        return false;  // Unsupported depth, fallback to rocBLAS
    }

    // For T82 models, seq_len is always 64
    const int seq_len = 64;

    try {
        // Call the flash attention kernel launcher from flash_attention.hip
        // Note: launch function is in lczero_mma namespace and only supports half (FP16)
        if (std::is_same<T, half>::value) {
            static bool type_log_once = false;
            if (!type_log_once) {
                printf("[FLASH ATTENTION DEBUG] Type check passed, calling kernel launcher...\n");
                type_log_once = true;
            }
            lczero_mma::launch_flash_attention_lc0(
                reinterpret_cast<const half*>(Q),
                reinterpret_cast<const half*>(K),
                reinterpret_cast<const half*>(V),
                reinterpret_cast<half*>(output),
                batch,
                num_heads,
                seq_len,
                depth,
                stream
            );
        } else {
            // Flash attention only supports FP16, fallback for other types
            static bool type_fail_log_once = false;
            if (!type_fail_log_once) {
                printf("[FLASH ATTENTION DEBUG] Type check failed: not FP16\n");
                type_fail_log_once = true;
            }
            return false;
        }

        // Check for kernel launch errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            // Kernel launch failed, fallback to rocBLAS
            static bool error_log_once = false;
            if (!error_log_once) {
                printf("[FLASH ATTENTION ERROR] Kernel launch failed: %s (code %d)\n",
                       hipGetErrorString(err), err);
                error_log_once = true;
            }
            return false;
        }

        // Synchronize to catch any kernel execution errors
        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            static bool exec_error_log_once = false;
            if (!exec_error_log_once) {
                printf("[FLASH ATTENTION ERROR] Kernel execution failed: %s (code %d)\n",
                       hipGetErrorString(err), err);
                exec_error_log_once = true;
            }
            return false;
        }

        // Flash attention succeeded
        static bool success_log_once = false;
        if (!success_log_once) {
            printf("[FLASH ATTENTION] ✓ Successfully using fused attention (depth=%d, heads=%d)\n",
                   depth, num_heads);
            success_log_once = true;
        }
        return true;

    } catch (...) {
        // Exception during kernel launch, fallback to rocBLAS
        return false;
    }
}

// Explicit template instantiation for half precision (FP16)
template bool flash_attention_wrapper<half>(
    const half* Q, const half* K, const half* V, half* output,
    float scale, int batch, int num_heads, int d_model, int depth,
    rocblas_handle cublas, hipStream_t stream
);

// Explicit template instantiation for single precision (FP32)
template bool flash_attention_wrapper<float>(
    const float* Q, const float* K, const float* V, float* output,
    float scale, int batch, int num_heads, int d_model, int depth,
    rocblas_handle cublas, hipStream_t stream
);

}  // namespace rocm_backend
}  // namespace lczero
