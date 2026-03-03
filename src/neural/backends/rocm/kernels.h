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

#pragma once

#include "neural/tables/activation_function.h"
#include "rocm_common.h"

namespace lczero {
namespace rocm_backend {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, hipStream_t stream);

// Adds two vectors of equal size overwriting the first with the sum.
// This specialisation performs a transposition of the first 2 indexes
// of the second while performing the addition.
template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, hipStream_t stream);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, hipStream_t stream);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation,
                    hipStream_t stream);

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, hipStream_t stream);

// Conversion from NCHW to NHWC, can also change datatype depending on template
// params, also pad/un-pad elements from Batch or Channel dimensions
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W);

// Plain data-type conversion (no layout conversion).
template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, hipStream_t stream);

// Perform batch normilization.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation);

// Unpack planes (input to network).
void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n, hipStream_t stream);

void expandPlanes_Fp16_NHWC(half* output, const uint64_t* masks,
                            const float* values, int n, hipStream_t stream);

void expandPlanes_Fp16_NCHW(half* output, const uint64_t* masks,
                            const float* values, int n, hipStream_t stream);

// Perform global avg pool.
template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc);

// Perform global scale.
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation);

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
// Returns false if the fused kernel can't handle the sizes.
bool Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2, const half* bPrev,
                  ActivationFunction activation);

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize, hipStream_t stream);

// PolicyMap with FP32 output - converts FP16 input to FP32 output during copy
void PolicyMapFp32Output(int N, float* output, const half* input,
                         const short* indices, int inputSize, int usedSize,
                         int outputSize, hipStream_t stream);

// Custom winograd helper functions
template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter);

template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformedInput, const T* input,
                    hipStream_t stream);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, hipStream_t stream);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          hipStream_t stream);

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             hipStream_t stream);

template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, hipStream_t stream);

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits,
                            hipStream_t stream);

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     hipStream_t stream);

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, hipStream_t stream);

// Flash Attention kernel wrapper (fused Q·K^T → softmax → ·V)
// Implements tiled flash attention algorithm for memory-efficient attention computation
// Returns true if flash attention was used, false if fallback to rocBLAS is needed
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
);
}  // namespace rocm_backend
}  // namespace lczero
