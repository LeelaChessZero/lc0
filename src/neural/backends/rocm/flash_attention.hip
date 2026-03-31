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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with AMD ROCm libraries from the ROCm toolkit
  (or a modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

// Flash Attention implementation for lc0's T82 model
// Ported from llama.cpp's CUDA flash attention kernel to HIP
// Adapted for lc0's interleaved multi-head attention layout

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "mma.hip"

#define WARP_SIZE 32
#define FATTN_KQ_MAX_OFFSET 0.0f
#define SOFTMAX_FTZ_THRESHOLD -20.0f

// Tunable parameters for optimization (can be overridden with -D flags)
// For depth=32 (24-head T82 configuration)
// Optimized for RDNA 3.5 (8060S): 2,272 nps (+18% vs rocBLAS, +3% vs initial)
#ifndef FATTN_NTHREADS_D32
#define FATTN_NTHREADS_D32 256      // Optimized: 256 threads per block (was 128)
#endif

#ifndef FATTN_OCCUPANCY_D32
#define FATTN_OCCUPANCY_D32 2       // Optimal: occupancy target of 2
#endif

#ifndef FATTN_NBATCH_FA_D32
#define FATTN_NBATCH_FA_D32 64      // Optimal: 64 rows per softmax rescaling
#endif

#ifndef FATTN_NBATCH_K2_D32
#define FATTN_NBATCH_K2_D32 32      // Optimal: 32 for K2 parallel loading
#endif

#ifndef FATTN_NBATCH_V2_D32
#define FATTN_NBATCH_V2_D32 32      // Optimal: 32 for V2 parallel loading
#endif

#ifndef FATTN_NBATCH_COMBINE_D32
#define FATTN_NBATCH_COMBINE_D32 32 // Optimal: 32 for combine parameter
#endif

#ifndef FATTN_NSTAGES_D32
#define FATTN_NSTAGES_D32 2         // Optimal: 2 pipeline stages
#endif

namespace lczero_mma {

// Use the mma tile definitions from mma.hip
using namespace lczero_mma;

// Config options for the MMA kernel
// Should not affect results, only speed/register pressure/shared memory use
struct fattn_mma_config {
    int  nthreads;       // Number of threads per HIP block
    int  occupancy;      // Targeted occupancy for the MMA kernel
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2,
            int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2),
        nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define LCZ_FATTN_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                           \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                        \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                       \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                       \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                       \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                       \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                  \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                  \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)}; \
    }

// RDNA3-specific configuration function
static constexpr __host__ __device__ fattn_mma_config get_fattn_config_rdna3(
    const int DKQ, const int DV, const int ncols) {
    // Standard RDNA3 configurations
    LCZ_FATTN_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    LCZ_FATTN_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    LCZ_FATTN_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

// T82 model-specific configurations for lc0
// T82 uses d_model=768 with varying number of heads and corresponding per-head depth
static constexpr __host__ __device__ fattn_mma_config get_fattn_config_t82(
    const int DKQ, const int DV, const int ncols) {

    // T82 24-head configuration: depth=32, d_model=768
    // Each head has 32 dimensions, 24 heads × 32 = 768
    // Use tunable parameters for all depth=32 configs
    LCZ_FATTN_CONFIG_CASE( 32,  32,  8, FATTN_NTHREADS_D32, FATTN_OCCUPANCY_D32, 128, FATTN_NBATCH_K2_D32, FATTN_NBATCH_V2_D32, FATTN_NBATCH_COMBINE_D32, FATTN_NSTAGES_D32, true);
    LCZ_FATTN_CONFIG_CASE( 32,  32, 16, FATTN_NTHREADS_D32, FATTN_OCCUPANCY_D32, FATTN_NBATCH_FA_D32, FATTN_NBATCH_K2_D32, FATTN_NBATCH_V2_D32, FATTN_NBATCH_COMBINE_D32, FATTN_NSTAGES_D32, true);
    LCZ_FATTN_CONFIG_CASE( 32,  32, 32, FATTN_NTHREADS_D32, FATTN_OCCUPANCY_D32, FATTN_NBATCH_FA_D32, FATTN_NBATCH_K2_D32, FATTN_NBATCH_V2_D32, FATTN_NBATCH_COMBINE_D32, FATTN_NSTAGES_D32, true);
    LCZ_FATTN_CONFIG_CASE( 32,  32, 64, FATTN_NTHREADS_D32, FATTN_OCCUPANCY_D32, FATTN_NBATCH_FA_D32, FATTN_NBATCH_K2_D32, FATTN_NBATCH_V2_D32, FATTN_NBATCH_COMBINE_D32, FATTN_NSTAGES_D32, true);

    // T82 12-head configuration: depth=64, d_model=768
    // Each head has 64 dimensions, 12 heads × 64 = 768
    LCZ_FATTN_CONFIG_CASE( 64,  64,  8, 128, 2, 128, 32, 32, 32, 2, true);
    LCZ_FATTN_CONFIG_CASE( 64,  64, 16, 128, 2,  64, 32, 32, 32, 2, true);
    LCZ_FATTN_CONFIG_CASE( 64,  64, 32, 128, 2,  64, 32, 32, 32, 2, true);
    LCZ_FATTN_CONFIG_CASE( 64,  64, 64, 128, 2,  64, 32, 32, 32, 2, true);

    // T82 8-head configuration: depth=96, d_model=768
    // Each head has 96 dimensions, 8 heads × 96 = 768
    LCZ_FATTN_CONFIG_CASE( 96,  96,  8, 128, 2, 128, 48, 48, 48, 2, true);
    LCZ_FATTN_CONFIG_CASE( 96,  96, 16, 128, 2,  64, 48, 48, 48, 2, true);
    LCZ_FATTN_CONFIG_CASE( 96,  96, 32, 128, 2,  64, 48, 48, 48, 2, true);
    LCZ_FATTN_CONFIG_CASE( 96,  96, 64, 128, 2,  64, 48, 48, 48, 2, true);

    // T82 6-head configuration: depth=128, d_model=768
    // Each head has 128 dimensions, 6 heads × 128 = 768
    LCZ_FATTN_CONFIG_CASE(128, 128,  8, 128, 2, 128, 64, 64, 64, 2, true);
    LCZ_FATTN_CONFIG_CASE(128, 128, 16, 128, 2,  64, 64, 64, 64, 2, true);
    LCZ_FATTN_CONFIG_CASE(128, 128, 32, 128, 2,  64, 64, 64, 64, 2, true);
    LCZ_FATTN_CONFIG_CASE(128, 128, 64, 128, 2,  64, 64, 64, 64, 2, true);

    // Unsupported config - will fallback to rocBLAS
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __device__ fattn_mma_config get_fattn_config(
    const int DKQ, const int DV, const int ncols) {
#if defined(RDNA3)
    // Try T82-specific configs first
    fattn_mma_config t82_config = get_fattn_config_t82(DKQ, DV, ncols);
    if (t82_config.nbatch_fa > 0) {
        return t82_config;
    }
    // Fallback to generic RDNA3
    return get_fattn_config_rdna3(DKQ, DV, ncols);
#else
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif
}

// Helper functions to extract config parameters
static constexpr __device__ int get_fattn_nthreads(const int DKQ, const int DV, const int ncols) {
    return get_fattn_config(DKQ, DV, ncols).nthreads;
}

static constexpr __device__ int get_fattn_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return get_fattn_config(DKQ, DV, ncols).nbatch_fa;
}

static constexpr __device__ int get_fattn_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return get_fattn_config(DKQ, DV, ncols).nbatch_K2;
}

static constexpr __device__ int get_fattn_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return get_fattn_config(DKQ, DV, ncols).nbatch_V2;
}

static constexpr __device__ bool get_fattn_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return get_fattn_config(DKQ, DV, ncols).Q_in_reg;
}

// ------------------------------------------------------------------------------------------------------------------
// Memory loading functions adapted for lc0's interleaved layout
// ------------------------------------------------------------------------------------------------------------------

// Load tile from interleaved lc0 memory layout
// lc0 layout: [batch, seq, head0_depth, head1_depth, ..., headN_depth]
// stride_KV must be d_model (NOT depth) for proper row indexing
// head_offset indicates which head's data to load: head_idx * depth
template<int stride_tile, int nwarps, int nbatch_fa>
static __device__ __forceinline__ void load_tile_lc0(
        const half2 * const __restrict__ KV_global,
        half2 * __restrict__ tile_KV,
        const int D2,                 // depth/2 for this head
        const int stride_KV,          // d_model (total embedding dimension)
        const int head_offset,        // offset to this head's data: head_idx * depth
        const int i_sup) {

    // Load with decreasing granularity for better memory bandwidth
    // Minimum granularity is 4 bytes (1 half2) for synchronous loading
    auto load = [&] __device__ (const int n) {
        const int stride_k = WARP_SIZE >> n;
        const int k0_start = stride_k == WARP_SIZE ? 0 : D2 - D2 % (2*stride_k);
        const int k0_stop  =                             D2 - D2 % (1*stride_k);
        const int stride_i = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                // CRITICAL: Load from interleaved layout using stride_KV (d_model) for row offset
                // Memory layout: [..., i-th token at position (i*stride_KV), then head_offset, then k-th element...]
                if (i < i_sup) {
                    const int global_idx = i*stride_KV + head_offset + k;
                    tile_KV[i*stride_tile + k] = KV_global[global_idx];
                } else {
                    tile_KV[i*stride_tile + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    };

    // Unroll for different granularities
    // 1: max 32* 4=128 bytes,  64 half
    // 2: max 16* 4= 64 bytes,  32 half
    // 3: max  8* 4= 32 bytes,  16 half
    // 4: max  4* 4= 16 bytes,   8 half
    load(0);
    load(1);
    load(2);
    load(3);
}

// ------------------------------------------------------------------------------------------------------------------
// Online Softmax Flash Attention Iteration
// ------------------------------------------------------------------------------------------------------------------

// Single iteration of flash attention with online softmax
// Implements the fused attention kernel with running max and sum tracking
template<int DKQ, int DV, int ncols, int nwarps,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_iter_lc0(
        const half2  * const __restrict__ Q_h2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        half2        * const __restrict__ output,
        const float scale,
        const int batch_size,
        const int num_heads,
        const int d_model,          // Total embedding dimension (depth * num_heads)
        const int depth,            // Per-head dimension
        const int seq_len,
        const int head_idx,         // Which head we're computing
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int kb0,
        const int k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE)
    constexpr int  cols_per_thread = 1; // RDNA has single column
    constexpr int  nbatch_fa       = get_fattn_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = get_fattn_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = get_fattn_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = get_fattn_Q_in_reg (DKQ, DV, ncols);

    constexpr int stride_tile_Q = DKQ/2     + 1;
    constexpr int stride_tile_K = nbatch_K2 + 1;
    constexpr int stride_tile_V = nbatch_V2 + 1;

    const int k_VKQ_0 = kb0 * nbatch_fa;
    const int head_offset = head_idx * depth / 2; // In half2 units

    // KQ accumulator tiles
    constexpr int np = nwarps; // Number of parallel warps
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];

    // Load K tile from interleaved memory
    load_tile_lc0<stride_tile_K, nwarps, nbatch_fa>
        (K_h2 + k_VKQ_0*d_model, tile_K, nbatch_K2, d_model, head_offset, k_VKQ_sup);
    __syncthreads();

    // Calculate tile of KQ = Q @ K^T
    if constexpr (Q_in_reg) {
#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < DKQ/2; k_KQ_0 += T_A_KQ::J) {
                T_A_KQ K_A;
                load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + k_KQ_0, stride_tile_K);
                // RDNA matrix C is column-major
                mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
            }
        }
    } else {
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < DKQ/2; k_KQ_0 += T_A_KQ::J) {
            load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                T_A_KQ K_A;
                load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + k_KQ_0, stride_tile_K);

                // RDNA matrix C is column-major
                mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
            }
        }
    }

    // Online softmax: compute running max and sum
    float KQ_max_new = KQ_max[0];
    float KQ_rowsum_add = 0.0f;

    // Calculate softmax for each KQ column using the current max value
    static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                KQ_max_new = fmaxf(KQ_max_new, KQ_C[(k0/(np*T_C_KQ::J))].x[l] * scale + FATTN_KQ_MAX_OFFSET);
            }
        }
    }

    // Reduce max across threads in warp
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        KQ_max_new = fmaxf(KQ_max_new, __shfl_xor(KQ_max_new, offset, WARP_SIZE));
    }

    // Apply exp with new max
    static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] * scale - KQ_max_new);
                KQ_rowsum_add += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
            } else {
                KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
            }
        }
    }

    // Update running max and sum with rescaling
    {
        const float KQ_max_diff = KQ_max[0] - KQ_max_new;
        float KQ_max_scale = expf(KQ_max_diff);
        KQ_max[0] = KQ_max_new;

        // Flush to zero if difference is too large
        *((uint32_t *) &KQ_max_scale) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

        // Scale previous KQ_rowsum to account for potential increase in KQ_max
        KQ_rowsum[0] = KQ_max_scale * KQ_rowsum[0] + KQ_rowsum_add;

        // Scale previous VKQ accumulator
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
        B[k] = get_half2(KQ_C[k]);
    }

    // Load V tile from interleaved memory
    load_tile_lc0<stride_tile_V, nwarps, nbatch_fa>
        (V_h2 + k_VKQ_0*d_model, tile_V, nbatch_V2, d_model, head_offset, k_VKQ_sup);
    __syncthreads();

    // Calculate VKQ tile = softmax(QK^T) @ V
#pragma unroll
    for (int i0_stop = DV; i0_stop > 0; i0_stop -= 2*nbatch_V2) {
        const int i0_start = i0_stop - 2*nbatch_V2 > 0 ? i0_stop - 2*nbatch_V2 : 0;

        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A;
                load_ldmatrix(A, tile_V + 2*k0*stride_tile_V + i_VKQ_0/2, stride_tile_V);

                // RDNA matrix C is column-major
                mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
            }
        }

        // Barrier only needed if loop runs multiple times
        // With DV=32, nbatch_V2=24: loop runs once, barrier unnecessary
        if constexpr (DV > 2*nbatch_V2) {
            __syncthreads();
        }
    }
#else
    // No device code for non-RDNA architectures
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// Main Flash Attention Kernel
// ------------------------------------------------------------------------------------------------------------------

// Flash attention kernel for lc0's interleaved multi-head layout
// Q, K, V layout: [batch, seq_len, head0_depth, head1_depth, ..., headN_depth]
template<int DKQ, int DV, int ncols>
__global__ void flash_attention_kernel_lc0(
        const half2 * __restrict__ Q,
        const half2 * __restrict__ K,
        const half2 * __restrict__ V,
        half2       * __restrict__ output,
        const int batch_size,
        const int num_heads,
        const int seq_len,
        const int d_model,      // Total embedding dimension (depth * num_heads)
        const int depth,        // Per-head dimension
        const float scale) {    // 1/sqrt(depth)

    constexpr int nwarps_raw = get_fattn_nthreads(DKQ, DV, ncols) / WARP_SIZE;
    constexpr int nbatch_fa_raw = get_fattn_nbatch_fa(DKQ, DV, ncols);

    // Ensure minimum size of 1 to avoid zero-length arrays
    constexpr int nwarps = nwarps_raw > 0 ? nwarps_raw : 1;
    constexpr int nbatch_fa = nbatch_fa_raw > 0 ? nbatch_fa_raw : 1;
    constexpr int safe_ncols = ncols > 0 ? ncols : 1;
    constexpr int safe_DKQ = DKQ > 0 ? DKQ : 16;
    constexpr int safe_DV = DV > 0 ? DV : 16;

    // Shared memory tiles (with safe minimum sizes)
    __shared__ half2 tile_Q_smem[safe_ncols * (safe_DKQ/2 + 2)];
    __shared__ half2 tile_K_smem[nbatch_fa * (safe_DKQ/2 + 2)];
    __shared__ half2 tile_V_smem[nbatch_fa * (safe_DV/2 + 2)];

    // Early return for invalid configurations
    if (nwarps_raw == 0 || nbatch_fa_raw == 0) {
        return;
    }

    // Register accumulators for online softmax
    float KQ_max[1] = {-INFINITY};
    float KQ_rowsum[1] = {0.0f};

    // Get batch and head indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int col_idx = blockIdx.x * ncols + threadIdx.y / nwarps;

    if (col_idx >= seq_len) return;

    // Calculate offsets for this head in interleaved layout
    const int head_offset = head_idx * depth / 2; // In half2 units
    const int batch_offset = batch_idx * seq_len * d_model / 2; // In half2 units

    // Pointers to Q, K, V for this batch
    const half2 * Q_batch = Q + batch_offset;
    const half2 * K_batch = K + batch_offset;
    const half2 * V_batch = V + batch_offset;

    // Load Q tile for this column from interleaved layout
    // Q is at position: [batch][col_idx][head_offset + k]
    half2 * tile_Q = tile_Q_smem + (threadIdx.y % nwarps) * (DKQ/2 + 2);
#pragma unroll
    for (int k = threadIdx.x; k < DKQ/2; k += WARP_SIZE) {
        const int global_idx = col_idx * d_model + head_offset + k;
        tile_Q[k] = Q_batch[global_idx];
    }

    // MMA tile types - using RDNA3 WMMA
#if defined(AMD_WMMA_AVAILABLE)
    using T_A_KQ  = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_B_KQ  = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
    using T_A_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_B_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_C_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR>;

    // Q matrix in registers
    T_B_KQ Q_B[DKQ/(2*T_B_KQ::J)];

    // Load Q into registers if configured
    constexpr bool Q_in_reg = get_fattn_Q_in_reg(DKQ, DV, ncols);
    if constexpr (Q_in_reg) {
        constexpr int stride_tile_Q = DKQ/2 + 2;
#pragma unroll
        for (int k = 0; k < DKQ/(2*T_B_KQ::J); ++k) {
            load_ldmatrix(Q_B[k], tile_Q + k*T_B_KQ::J, stride_tile_Q);
        }
    }

    // VKQ accumulator
    T_C_VKQ VKQ_C[DV/(2*T_C_VKQ::J)];

    // Process K, V in chunks
    const int num_iters = (seq_len + nbatch_fa - 1) / nbatch_fa;
    for (int kb0 = 0; kb0 < num_iters; ++kb0) {
        const int k_VKQ_sup = min(seq_len - kb0 * nbatch_fa, nbatch_fa);

        flash_attn_iter_lc0<DKQ, DV, ncols, nwarps,
            T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>(
            Q_batch, K_batch, V_batch, output,
            scale, batch_size, num_heads, d_model, depth, seq_len, head_idx,
            tile_Q, tile_K_smem, tile_V_smem,
            Q_B, VKQ_C, KQ_max, KQ_rowsum,
            kb0, k_VKQ_sup);
    }

    // Final normalization by rowsum and write back to interleaved output
    const float inv_rowsum = 1.0f / KQ_rowsum[0];
    const int output_offset = batch_offset + col_idx * d_model + head_offset;

#pragma unroll
    for (int i = 0; i < DV/(2*T_C_VKQ::J); ++i) {
#pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
            const int output_idx = output_offset + i * T_C_VKQ::J + T_C_VKQ::get_j(l);
            VKQ_C[i].x[l] *= inv_rowsum;
            // Write back to interleaved output
            if (output_idx < batch_offset + seq_len * d_model) {
                output[output_idx] = VKQ_C[i].x[l];
            }
        }
    }
#endif
}

// Host launch function
void launch_flash_attention_lc0(
        const half * Q,
        const half * K,
        const half * V,
        half * output,
        int batch_size,
        int num_heads,
        int seq_len,
        int depth,
        hipStream_t stream) {

    const int d_model = num_heads * depth;
    const float scale = 1.0f / sqrtf(static_cast<float>(depth));

    // Determine ncols based on sequence length (tunable parameter)
    const int ncols = seq_len <= 16 ? 8 : (seq_len <= 64 ? 16 : 32);

    // Get configuration
    const auto config = get_fattn_config_t82(depth, depth, ncols);

    if (config.nbatch_fa == 0) {
        // Unsupported configuration, fallback to rocBLAS
        // (Implementation would call rocBLAS-based attention here)
        return;
    }

    const int nthreads = config.nthreads;
    const int nwarps = nthreads / WARP_SIZE;

    dim3 blocks(
        (seq_len + ncols - 1) / ncols,  // x: columns
        num_heads,                       // y: heads
        batch_size                       // z: batches
    );
    dim3 threads(WARP_SIZE, nwarps, 1);

    // Launch kernel based on depth configuration
    const half2 * Q_h2 = reinterpret_cast<const half2 *>(Q);
    const half2 * K_h2 = reinterpret_cast<const half2 *>(K);
    const half2 * V_h2 = reinterpret_cast<const half2 *>(V);
    half2 * output_h2 = reinterpret_cast<half2 *>(output);

    if (depth == 32) {
        // T82 24-head configuration
        hipLaunchKernelGGL(
            (flash_attention_kernel_lc0<32, 32, 16>), blocks, threads, 0, stream,
            Q_h2, K_h2, V_h2, output_h2,
            batch_size, num_heads, seq_len, d_model, depth, scale);
    } else if (depth == 64) {
        // T82 12-head configuration
        hipLaunchKernelGGL(
            (flash_attention_kernel_lc0<64, 64, 16>), blocks, threads, 0, stream,
            Q_h2, K_h2, V_h2, output_h2,
            batch_size, num_heads, seq_len, d_model, depth, scale);
    } else if (depth == 96) {
        // T82 8-head configuration
        hipLaunchKernelGGL(
            (flash_attention_kernel_lc0<96, 96, 16>), blocks, threads, 0, stream,
            Q_h2, K_h2, V_h2, output_h2,
            batch_size, num_heads, seq_len, d_model, depth, scale);
    } else if (depth == 128) {
        // T82 6-head configuration
        hipLaunchKernelGGL(
            (flash_attention_kernel_lc0<128, 128, 16>), blocks, threads, 0, stream,
            Q_h2, K_h2, V_h2, output_h2,
            batch_size, num_heads, seq_len, d_model, depth, scale);
    }
}

} // namespace lczero_mma
