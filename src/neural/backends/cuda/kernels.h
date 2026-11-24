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
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include "cuda_common.h"
#include "neural/tables/activation_function.h"

namespace lczero {
namespace NS_BACKEND {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, cudaStream_t stream);

// The heads pooled into the selected one, passed by value to the kernel.
template <typename T>
struct PolicyBlendHeads {
  static constexpr int kMax = 8;
  const T* head[kMax];
  float weight[kMax];
  int count = 0;
};

// Geometric pooling of the policy heads, as a weighted sum of their logits
// written over the first vector: main = w_main * main + sum_k w_k * head_k,
// summed in float in the order of the heads.
template <typename T>
void blendPolicyLogits(T* main, const PolicyBlendHeads<T>& heads,
                       float w_main, int count, cudaStream_t stream);

// Adds two vectors of equal size overwriting the first with the sum.
// This specialisation performs a transposition of the first 2 indexes
// of the second while performing the addition.
template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, cudaStream_t stream);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, cudaStream_t stream);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation,
                    cudaStream_t stream);

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, cudaStream_t stream);

// Conversion from NCHW to NHWC, can also change datatype depending on template
// params, also pad/un-pad elements from Batch or Channel dimensions
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       cudaStream_t stream);

// Plain data-type conversion (no layout conversion).
template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, cudaStream_t stream);

// Perform batch normilization.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation, cudaStream_t stream);

// Unpack planes (input to network).
template <typename T>
void expandPlanes_NHWC(T* output, const uint64_t* masks, const T* values, int n,
                       cudaStream_t stream);

template <typename T>
void expandPlanes_NCHW(T* output, const uint64_t* masks, const T* values, int n,
                       cudaStream_t stream);

// Perform global avg pool.
template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, cudaStream_t steam);

// Perform global scale.
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, cudaStream_t steam);

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
// Returns false if the fused kernel can't handle the sizes.
bool Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2, const half* bPrev,
                  ActivationFunction activation, cudaStream_t stream);

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize,
               cudaStream_t stream);

// Custom winograd helper functions
template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter,
                     cudaStream_t stream);

template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformedInput, const T* input,
                    cudaStream_t stream);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, cudaStream_t stream);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          cudaStream_t stream);

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             cudaStream_t stream);

template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, cudaStream_t stream);

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits,
                            cudaStream_t stream);

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     cudaStream_t stream);

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, cudaStream_t stream);

// output = (input + bias) * sigmoid(gate), the gated FFN (ACTIVATION_SWIGLU).
template <typename T>
void addBiasSwiGLU(T* output, const T* input, const T* bias, const T* gate,
                   int N, int C, cudaStream_t stream);

// Attack graph and static pair stream (NETWORK_ATTENTIONBODY_BT6); see
// common_kernels.cu for the channels. bits: one uint64 per (position, i, j),
// bit c = attack-graph channel c at (i, j).
constexpr int kAttackGraphChannels = 34;

// bits [N, 64, 64] from the NCHW input planes [N, input_planes, 64].
template <typename T>
void computeAttackGraph(uint64_t* bits, const T* planes, int N,
                        int input_planes, cudaStream_t stream);

// z [N, 64 * 64, pair_d_e] = rms(E @ p_in + t_off[rel(i, j)]);
// p_in [kAttackGraphChannels, pair_d_e], t_off [225, pair_d_e].
template <typename T>
void computePairStream(T* z, const uint64_t* bits, const float* p_in,
                       const float* t_off, int N, int pair_d_e,
                       cudaStream_t stream);

// logits [N, heads, 64, 64] (+)= the attack-graph terms of the attention
// logits: attack_w [heads, C], gate_q / gate_k [heads, C, depth].
template <typename T>
void addAttackGraphBias(T* logits, const uint64_t* bits, const T* q,
                        const T* k, const float* attack_w, const float* gate_q,
                        const float* gate_k, int N, int heads, int depth,
                        int d_model, bool accumulate, cudaStream_t stream);

// x = x * scale * sigmoid(gate), elementwise: the attention output gate.
template <typename T>
void applySigmoidGate(T* x, const T* gate, float scale, int size,
                      cudaStream_t stream);

// logits[n * stride + p] += factor * (u . z[n, p, :]), p < 64 * 64.
template <typename T>
void addPairPolicyLogits(T* logits, const T* z, const float* u, float factor,
                         int N, int pair_d_e, int stride,
                         cudaStream_t stream);

// Layer norm over the N rows of C: v = (in + bias) * alpha + skip (bias and
// skip may be null); residual_out = v when non-null (it may alias skip);
// out = (v - mean) / sqrt(var + eps) * gammas + betas. The statistics are
// computed in float. C <= 8192.
template <typename TIn, typename TSkip, typename TOut, typename T>
void layerNormRows(TOut* out, float* residual_out, const TIn* in,
                   const T* bias, const TSkip* skip, const T* gammas,
                   const T* betas, int N, int C, float eps, float alpha,
                   cudaStream_t stream);

// residual += alpha * (y + bias), bias [C] broadcast over the N rows.
template <typename T>
void addToResidual(float* residual, const T* y, const T* bias, float alpha,
                   int N, int C, cudaStream_t stream);

// The edge-stream doors of a block's attention, in place of the softmax:
// logits [N, heads, 64, 64] = q k^T / sqrt(depth) on entry (bias, the same
// shape, added when non-null), the gated attention weights on exit. For row
// (n, h, i), with e the edge state [N, 64 * 64, d_e] and doors[h] = [edge_e_w
// (d_e), edge_m_w (d_e), edge_g_w (d_e), edge_t_node, edge_t_edge, edge_g_b]:
//   L = (t_node L + t_edge (edge_e_w . e)) * (1 + edge_m_w . e)
//   W = softmax_j(L) * gate_scale * sigmoid(edge_g_w . e + edge_g_b).
// With o_e [heads, d_e] (an edge-stream site follows the block) it also adds
// the readback e += sum_h L[h] o_e[h, :]. d_e 8, 16 or 32.
template <typename T>
void edgeDoorSoftmax(T* logits, const T* bias, float* e, const float* doors,
                     const float* o_e, float gate_scale, int N, int heads,
                     int d_e, cudaStream_t stream);

// The weights of an edge-stream site (net.proto EdgeStreamUpdate), row-major
// float on the device.
struct EdgeStreamUpdateWeights {
  const float* tri_v = nullptr;     // [d_e, 2 d_e]
  const float* tri_eg = nullptr;    // [d_e, 4 triplet_heads]
  const float* tri_eg_b = nullptr;  // [4 triplet_heads]
  const float* tri_o = nullptr;     // [2 d_e, d_e]
  const float* ffn_in = nullptr;    // [d_e, hidden]
  const float* ffn_in_b = nullptr;  // [hidden]
  const float* ffn_rev = nullptr;   // [d_e, hidden] or null
  const float* ffn_out = nullptr;   // [hidden, d_e]
  int hidden = 0;
  int triplet_heads = 0;
  bool path = true;  // triplet path form (else TGT-Ag)
};

// The scratch an edge-stream site needs for N positions.
size_t edgeStreamUpdateScratchBytes(int N, int d_e, int triplet_heads);

// An edge-stream site after the readback: e += triplet(rms(e)); e = rms(e +
// FFN(rms(e))). e [N, 64 * 64, d_e] float, in place. d_e 16 (triplet heads 4)
// or 32 (triplet heads 4 or 8).
void edgeStreamUpdate(float* e, void* scratch,
                      const EdgeStreamUpdateWeights& weights, int N, int d_e,
                      cudaStream_t stream);

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2,
                       cudaStream_t stream);

template <typename DataType>
void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, cudaStream_t stream);

}  // namespace cudnn_backend
}  // namespace lczero
