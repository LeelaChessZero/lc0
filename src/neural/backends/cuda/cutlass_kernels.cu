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

#include "neural/backends/cuda/cuda_common.h"

// Fused MHA implementation from cutlass example #41
#include "fused_multi_head_attention/kernel_forward.h"
#include "utils/exception.h"

namespace lczero {
namespace cudnn_backend {

template <bool bias>
void fusedMHACutlass(void* output, void* q, void* k, void* v, void* skip,
                     int batch_size, int num_heads, int depth,
                     cudaStream_t stream) {
  cutlass::half_t* mha_q = (cutlass::half_t*)q;
  cutlass::half_t* mha_k = (cutlass::half_t*)k;
  cutlass::half_t* mha_v = (cutlass::half_t*)v;

  constexpr int kQueriesPerBlock = 64;
  constexpr int kKeysPerBlock = 64;
  constexpr bool kSingleValueIteration = true;

  using Attention =
      AttentionKernel<cutlass::half_t,      // scalar_t
                      cutlass::arch::Sm80,  // ArchTag
                      true,                 // Memory is aligned
                      kQueriesPerBlock, kKeysPerBlock, kSingleValueIteration,
                      false,  // Supports dropout
                      bias    // Supports bias
                      >;
  static_assert(
      !Attention::kNeedsOutputAccumulatorBuffer,
      "Unhandled case in cutlass MHA: needs output accumulator buffer");

  typename Attention::Params p;
  {  // set parameters
    p.query_ptr = mha_q;
    p.key_ptr = mha_k;
    p.value_ptr = mha_v;
    p.logsumexp_ptr = nullptr;  // Only needed for bw
    p.output_accum_ptr = nullptr;
    p.output_ptr = (cutlass::half_t*)output;
    p.attn_bias_ptr = (cutlass::half_t*)skip;

    p.scale = 1.0f / sqrt((float)depth);

    p.num_heads = num_heads;
    p.num_batches = batch_size;
    p.head_dim = depth;
    p.head_dim_value = depth;
    p.num_queries = 64;
    p.num_keys = 64;

    // All tensors are in BMHK shapes
    p.q_strideH = depth;
    p.k_strideH = depth;
    p.v_strideH = depth;
    p.q_strideM = depth * num_heads;
    p.k_strideM = depth * num_heads;
    p.v_strideM = depth * num_heads;
    p.q_strideB = p.q_strideM * 64;
    p.k_strideB = p.k_strideM * 64;
    p.v_strideB = p.v_strideM * 64;
    p.o_strideM = p.head_dim_value * p.num_heads;

    p.bias_strideH = 64 * 64;
    p.bias_strideM = 64;
    p.bias_strideB = num_heads * p.bias_strideH;
  }

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    ReportCUDAErrors(cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  if (!Attention::check_supported(p)) {
    throw Exception("Unhandled case in cutlass MHA: check_supported failed.");
  }

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

  ReportCUDAErrors(cudaGetLastError());
}

void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, cudaStream_t stream) {
  if (skip == nullptr) {
    fusedMHACutlass<false>(output, mha_q, mha_k, mha_v, skip, batch_size,
                           num_heads, depth, stream);
  } else {
    fusedMHACutlass<true>(output, mha_q, mha_k, mha_v, skip, batch_size,
                          num_heads, depth, stream);
  }
}

}  // namespace cudnn_backend
}  // namespace lczero
