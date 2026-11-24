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

#if !defined(USE_HIP)
#include <cublas_v2.h>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cuda_common.h"
#include "kernels.h"
#include "neural/network_legacy.h"
#include "neural/tables/activation_function.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#else
typedef void* cudnnHandle_t;
#endif

namespace lczero {
namespace NS_BACKEND {

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

// Per-position state of nets whose attention reads the board's attack graph
// (NETWORK_ATTENTIONBODY_BT6): the attack-graph bits [N, 64, 64] (one uint64
// per square pair, see kernels.h) and the static pair stream Z
// [N, 64 * 64, pair_d_e]. AttentionBody computes them once per evaluation;
// the encoder blocks and the policy head read them. They live after the
// scratch area: the network allocates scratch_size + Bytes() and every layer
// is still given scratch_size.
template <typename DataType>
struct AttackGraphState {
  uint64_t* bits;
  DataType* pair;

  static size_t Bytes(int max_batch_size, int pair_d_e) {
    return kAlign + BitsBytes(max_batch_size) +
           static_cast<size_t>(max_batch_size) * 64 * 64 * pair_d_e *
               sizeof(DataType);
  }
  AttackGraphState(void* scratch, size_t scratch_size, int max_batch_size) {
    char* base = static_cast<char*>(scratch) +
                 (scratch_size + kAlign - 1) / kAlign * kAlign;
    bits = reinterpret_cast<uint64_t*>(base);
    pair = reinterpret_cast<DataType*>(base + BitsBytes(max_batch_size));
  }

 private:
  static constexpr size_t kAlign = 256;
  static size_t BitsBytes(int max_batch_size) {
    return static_cast<size_t>(max_batch_size) * 64 * 64 * sizeof(uint64_t);
  }
};

// The float residual stream of pre-norm encoder blocks, N x 64 x C floats.
// It lives after the scratch area and the AttackGraphState (when the net has
// one): the network allocates scratch_size + attack graph bytes + Bytes().
struct PrenormResidual {
  static size_t Bytes(int max_batch_size, int C) {
    return kAlign +
           static_cast<size_t>(max_batch_size) * 64 * C * sizeof(float);
  }
  static float* At(void* scratch, size_t scratch_size,
                   size_t attack_graph_bytes) {
    const size_t offset = (scratch_size + attack_graph_bytes + kAlign - 1) /
                          kAlign * kAlign;
    return reinterpret_cast<float*>(static_cast<char*>(scratch) + offset);
  }

 private:
  static constexpr size_t kAlign = 256;
};

// The edge stream's state e, N x 64 * 64 x d_e floats, then the scratch of
// its sites, after the PrenormResidual (when the net has one): the network
// allocates scratch_size + attack graph bytes + residual bytes + Bytes().
struct EdgeStreamState {
  float* e;
  void* site_scratch;

  static size_t Bytes(int max_batch_size, int d_e, size_t site_scratch_bytes) {
    return 2 * kAlign + StateBytes(max_batch_size, d_e) + site_scratch_bytes;
  }
  EdgeStreamState(void* scratch, size_t scratch_size, size_t preceding_bytes,
                  int max_batch_size, int d_e) {
    char* base = static_cast<char*>(scratch);
    const size_t offset = AlignUp(scratch_size + preceding_bytes);
    e = reinterpret_cast<float*>(base + offset);
    site_scratch = base + AlignUp(offset + StateBytes(max_batch_size, d_e));
  }

 private:
  static constexpr size_t kAlign = 256;
  static size_t AlignUp(size_t x) { return (x + kAlign - 1) / kAlign * kAlign; }
  static size_t StateBytes(int max_batch_size, int d_e) {
    return static_cast<size_t>(max_batch_size) * 64 * 64 * d_e * sizeof(float);
  }
};

// Whether the attention body computes an AttackGraphState for these weights:
// a pair stream, an edge stream, or attack-graph / pair stream terms in an
// encoder block.
inline bool UsesAttackGraph(const MultiHeadWeights& weights) {
  bool uses = !weights.pair_p_in.empty() || !weights.edge_p_in.empty();
  for (const auto& enc : weights.encoder) {
    uses = uses || !enc.mha.attack_w.empty() || !enc.mha.pair_att_w.empty();
  }
  return uses;
}

// Constants of the attention body that a net format sets; zero (false) means
// the value older formats imply.
struct AttentionBodyConstants {
  float encoder_alpha = 0.0f;        // (2 * encoders)^-0.25
  float embedding_ffn_alpha = 0.0f;  // (2 * encoders)^-0.25
  float epsilon = 0.0f;              // 1e-3 (PE-dense nets) or 1e-6
  float output_gate_scale = 0.0f;    // 1 (blocks with an output gate)
  bool input_gate_relu = false;      // multiplicative input gate max(g, 0)
  // Pre-norm blocks (x += a * MHA(ln1(x)); x += a * FFN(ln2(x))) on a float
  // residual stream, then the final norm; post-norm blocks otherwise.
  bool prenorm = false;
  // Edge stream: site k follows block edge_update_after[k].
  std::vector<int> edge_update_after;
  bool triplet_path = false;     // triplet path form (else TGT-Ag)
  float edge_gate_scale = 0.0f;  // 1 (the doors' G gate)
};

template <typename DataType>
class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }
  bool isNHWC() const { return nhwc_; }

  BaseLayer(int c, int h, int w, BaseLayer* ip);
  BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc);
  BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, bool use_gemm_ex);
  virtual ~BaseLayer() = default;
  size_t GetOutputSize(int N) const { return sizeof(DataType) * N * C * H * W; }

  // Input2 is optional (skip connection).
  virtual void Eval(int N, DataType* output, const DataType* input,
                    const DataType* input2, void* scratch, size_t scratch_size,
                    cudnnHandle_t cudnn, cublasHandle_t cublas,
                    cudaStream_t stream, DataType*** = nullptr) = 0;

 protected:
  BaseLayer* input_;

  int C;  // Output tensor dimensions.
  int H;
  int W;

  bool nhwc_;  // tensor layout
  const bool use_gemm_ex_;

  void cublasRowMajorMatrixMul(const DataType* A, const DataType* B,
                               DataType* Out, int M, int N, int K,
                               int batchSize, cublasHandle_t cublas);
};

#ifdef USE_CUDNN
template <typename DataType>
class ConvLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;

 public:
  ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W, int size, int Cin,
            ActivationFunction activation = ACTIVATION_NONE, bool bias = false);

  ConvLayer(bool nhwc, int C, int H, int W, int size, int Cin,
            ActivationFunction activation = ACTIVATION_NONE, bool bias = false);

  ~ConvLayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const int c_input_;
  const int filter_size_;
  const ActivationFunction act_;
  const bool use_bias_;

  DataType* biases = nullptr;
  DataType* weights = nullptr;

  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t conv_algo_;

  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t in_tensor_desc_;
  cudnnTensorDescriptor_t out_tensor_desc_;
  cudnnActivationDescriptor_t activation_;

  void init();
};

#endif

template <typename DataType>
class FCLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;

 public:
  FCLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool bias,
          ActivationFunction activation);
  ~FCLayer();

  void LoadWeights(float* cpuWeight, float* cpuBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const bool use_bias_;
  const ActivationFunction act_;
  DataType* weights_ = nullptr;
  DataType* biases_ = nullptr;
};

template <typename DataType>
class PolicyMapLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;

 public:
  PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H, int W, int usedSize,
                 bool attention);
  ~PolicyMapLayer();

  void LoadWeights(const short* cpuWeight, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  int used_size_;  // Size of the input without padding (typically 73x64).
                   // This is over-written to contain size with padding
                   // (typically 80x64) after CHW->HWC conversion for fp16.
  const bool attention_map_;
  short* weights_ = nullptr;
};

// Fused SE layer:
// (optional bias add +) global avg -> FC1 -> FC2 -> global scale -> add skip
// connection -> RELU.
template <typename DataType>
class SELayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::nhwc_;

 public:
  SELayer(BaseLayer<DataType>* ip, int numFc1Out, bool addPrevLayerBias,
          ActivationFunction activation);
  ~SELayer();

  void LoadWeights(float* w1, float* b1, float* w2, float* b2,
                   float* prevLayerBias, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  DataType* w1_ = nullptr;
  DataType* w1_t_ = nullptr;  // transposed copy used by fused SE kernel
  DataType* b1_ = nullptr;
  DataType* w2_ = nullptr;
  DataType* w2_t_ = nullptr;
  DataType* b2_ = nullptr;
  DataType* bPrev_ = nullptr;
  int numFc1Out_;
  bool addPrevLayerBias_;
  const ActivationFunction act_;
};

// Multi-pass Winograd Conv fused with (optional) SE
template <typename DataType>
class FusedWinogradConvSELayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;

 public:
  FusedWinogradConvSELayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           int Cin, ActivationFunction activation, bool bias,
                           bool skipAdd, bool se, int se_k, bool use_gemm_ex,
                           bool op_nhcw = false);

  ~FusedWinogradConvSELayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const int c_input_;
  const ActivationFunction act_;
  const bool use_bias_;
  const bool skip_add_;
  const bool has_se_;
  const int se_k_;
  const bool op_nhcw_;

  DataType* biases_ = nullptr;
  DataType* transformed_weights_ = nullptr;  // After winograd transform.

  // Weights and Biases for (optional) SE.
  DataType* w1_;
  DataType* w2_;
  DataType* b1_;
  DataType* b2_;
};

template <typename DataType>
class Conv1Layer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;

 public:
  Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
             ActivationFunction activation, bool bias, bool use_gemm_ex);

  ~Conv1Layer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const int c_input_;
  const ActivationFunction act_;
  const bool use_bias_;

  DataType* biases_ = nullptr;
  DataType* weights_ = nullptr;

  // uses stride of 0 to read a vector as a matrix
  void cublasSpecialMatrixMul(const DataType* A, const DataType* B,
                              DataType* Out, int M, int N, int K, int batchSize,
                              cublasHandle_t cublas);
};

// Multi-pass Winograd Conv fused with (optional) SE
template <typename DataType>
class ResidualBlock : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;

 public:
  ResidualBlock(BaseLayer<DataType>* ip, int C, bool se, int se_k,
                bool use_gemm_ex, bool first, bool last,
                ActivationFunction activation, int shared_mem_size);

  ~ResidualBlock();
  void LoadWeights0(float* pfilter, float* pBias, void* scratch);
  void LoadWeights1(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const bool has_se_;
  const int se_k_;
  const int c_input_;
  const bool first_block_;
  const bool last_block_;
  const int shared_mem_size_;
  const ActivationFunction act_;

  DataType* biases0_ = nullptr;
  DataType* biases1_ = nullptr;
  DataType* transformed_weights0_ = nullptr;  // After winograd transform.
  DataType* transformed_weights1_ = nullptr;  // After winograd transform.

  // Weights and Biases for (optional) SE.
  DataType* w1_;
  DataType* w2_;
  DataType* b1_;
  DataType* b2_;
};

template <typename DataType>
class EncoderBlock {
 public:
  EncoderBlock(const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch,
               int heads, int size, float alpha,
               DataType* smolgen_global_scratch, int smolgen_global_size,
               int max_batch_size, ActivationFunction smolgen_act,
               ActivationFunction ffn_act, float default_eps, bool use_gemm_ex,
               bool fused_mha, float output_gate_scale = 1.0f,
               bool prenorm = false, float edge_gate_scale = 1.0f);
  ~EncoderBlock();

  // attack_bits and pair_stream: the AttackGraphState, for blocks with
  // attack-graph terms or pair_att_w. residual: the PrenormResidual of a
  // pre-norm block, which reads and updates it (inpop is then scratch too).
  // edge: the EdgeStreamState, for blocks with edge doors; edge_readback: the
  // o_e of the edge-stream site after this block, whose readback the block
  // adds into edge.
  void Eval(int N, DataType* inpop, DataType* scratch0, DataType* scratch1,
            DataType* scratch2, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** offset_pointers,
            const uint64_t* attack_bits = nullptr,
            const DataType* pair_stream = nullptr,
            float* residual = nullptr, float* edge = nullptr,
            const float* edge_readback = nullptr) const;

  // all GPU side pointers
  DataType *mha_q_w, *mha_q_b;
  DataType *mha_k_w, *mha_k_b;
  DataType *mha_v_w, *mha_v_b;
  DataType *mha_og_w = nullptr, *mha_og_b = nullptr;  // output gate
  // q, k, v (and the output gate's when present) one after the other.
  DataType *mha_qkv_w, *mha_qkv_b;
  DataType *mha_dense_w, *mha_dense_b;

  DataType *ln1_gammas, *ln1_betas;

  DataType *ffn_dense1_w, *ffn_dense1_b;
  DataType *ffn_dense_gate_w = nullptr;  // ACTIVATION_SWIGLU
  DataType *ffn_dense2_w, *ffn_dense2_b;

  DataType *ln2_gammas, *ln2_betas;

  DataType *smol_compress;
  DataType *smol_dense1_w, *smol_dense1_b;
  DataType *smol_dense2_w, *smol_dense2_b;
  DataType *smol_ln1_gammas, *smol_ln1_betas;
  DataType *smol_ln2_gammas, *smol_ln2_betas;
  DataType *smol_global;

  // Attention-logit terms from the attack graph and the pair stream.
  float *attack_w = nullptr, *attack_gate_q = nullptr,
        *attack_gate_k = nullptr;
  DataType *pair_att_w = nullptr;

  int mha_q_size_;
  int mha_k_size_;
  int mha_v_size_;
  int mha_dense_size_;

  int ffn_dense1_size_;
  int ffn_dense2_size_;

  int embedding_op_size_;
  int encoder_heads_;

  float alpha_;  // scale to apply to skip connection add
  float default_eps_;  // value of epsilon where it wasn't specified in training

  const bool has_smolgen_;
  const ActivationFunction smolgen_activation_;
  const ActivationFunction ffn_activation_;

  // Output sizes for smolgen layers.
  int smol_compress_size_;
  int smol_dense_1_size_;
  int smol_dense_2_size_;
  int smol_global_size_;

  int pair_d_e_ = 0;  // pair_att_w: [heads, pair_d_e_]

  bool has_output_gate_ = false;
  float output_gate_scale_;
  bool prenorm_ = false;

  // Edge doors [heads][3 d_e + 3] (kernels.h edgeDoorSoftmax).
  float* edge_doors_ = nullptr;
  int edge_d_e_ = 0;
  float edge_gate_scale_ = 1.0f;

  const int max_batch_size_;
  const bool use_fused_mha_;
  const bool use_gemm_ex_;
};

// The Attention policy head implementation
// Responsible for loading weights into GPU memory, and evaluating the entire
// policy head
template <typename DataType>
class AttentionPolicyHead : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::use_gemm_ex_;

 public:
  // With shared_embedding the head does not own the policy embedding: Eval
  // expects input2 to already hold the embedding another head computed from
  // the same body output, and only runs the query/key projections and the
  // promotion logits on top of it. Requires a head without policy encoder
  // blocks, since those would rewrite the embedding in place.
  AttentionPolicyHead(BaseLayer<DataType>* ip,
                      const MultiHeadWeights::PolicyHead& weights,
                      void* scratch, bool attention_body,
                      ActivationFunction act, int max_batch_size,
                      bool use_gemm_ex, bool shared_embedding = false);
  ~AttentionPolicyHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  // GPU allocations to hold various weights used by the attention policy head
  DataType *ip_pol_w_, *ip_pol_b_;    // "embedding" in policy attention
  DataType *ip2_pol_w_, *ip2_pol_b_;  // "wq" in policy attention
  DataType *ip3_pol_w_, *ip3_pol_b_;  // "wk" in policy attention
  DataType *ip4_pol_w_;               // "ppo" in policy attention

  DataType *wqk_w_, *wqk_b_;  // allocation containing both "wq" and "wq"
  float *pair_u_ = nullptr;   // pair stream term [pair_d_e_]

  int embedding_op_size_;
  int wq_op_size_;
  int wk_op_size_;

  int encoder_heads_;
  int policy_d_model_;
  int pair_d_e_ = 0;
  int max_batch_size_;
  bool attention_body_;
  // The embedding is computed by another head and handed over in input2.
  bool shared_embedding_;
  ActivationFunction act_;

  std::vector<EncoderBlock<DataType>*> encoder_weights_;
};

template <typename DataType>
class EmbeddingLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;

 public:
  EmbeddingLayer(BaseLayer<DataType>* ip, const std::vector<float>& weights,
                 const std::vector<float>& biases, void* scratch,
                 ActivationFunction activation);
  ~EmbeddingLayer();

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  DataType *weights_, *biases_;
  ActivationFunction act_;
};

// The Attention body implementation
// Responsible for loading weights into GPU memory, and evaluating the entire
// attention network part of the body including the stack of encoder layers
template <typename DataType>
class AttentionBody : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;

 public:
  AttentionBody(const MultiHeadWeights& weights, void* scratch,
                Activations activations, int num_res_blocks, int input_c,
                int max_batch_size, bool is_pe_dense_embedding,
                bool use_gemm_ex, bool fused_mha,
                const AttentionBodyConstants& constants = {});
  ~AttentionBody();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  // GPU allocations to hold various weights used by the attention net body.
  DataType *ip_emb_pre_w_, *ip_emb_pre_b_;  // input position preprocessing weights.
  DataType *ip_emb_w_, *ip_emb_b_;          // "embedding" layer in net body
  DataType *ip_emb_ln_g_, *ip_emb_ln_b_;  // input embedding layernorm gamma and beta
  DataType *ip_mult_gate_, *ip_add_gate_;   // input gating
  DataType *ip_emb_ffn_d1_w_, *ip_emb_ffn_d1_b_;  // input embedding FFN dense1 weights
  DataType *ip_emb_ffn_dg_w_ = nullptr;  // input embedding FFN gate (SwiGLU)
  DataType *ip_emb_ffn_d2_w_, *ip_emb_ffn_d2_b_;  // input embedding FFN dense2 weights
  DataType *ip_emb_ffn_ln_g_, *ip_emb_ffn_ln_b_;  // input embedding FFN layernorm gamma and beta
  DataType *smolgen_global_;  // global smolgen weights for all encoder layers
  DataType *pos_encoding_;
  float *pair_p_in_ = nullptr, *pair_t_off_ = nullptr;  // static pair stream
  // Final norm of a pre-norm tower.
  DataType *final_norm_g_ = nullptr, *final_norm_b_ = nullptr;
  // Edge stream: the seed's weights and the sites.
  float *edge_p_in_ = nullptr, *edge_t_off_ = nullptr;
  int edge_d_e_ = 0;
  struct EdgeSite {
    int after_block;
    const float* o_e;  // [heads, d_e]
    EdgeStreamUpdateWeights weights;
  };
  std::vector<EdgeSite> edge_sites_;
  std::vector<float*> edge_site_memory_;  // every site's device arrays
  int embedding_dense_size_;
  int embedding_op_size_;
  int embedding_ffn_size_;
  int embedding_ffn_dff_;
  int encoder_head_count_;
  std::vector<EncoderBlock<DataType>*> encoder_weights_;
  Activations activations_;
  int num_resi_blocks_;
  int input_c_;
  int smolgen_global_size_;
  int max_batch_size_;
  int pair_d_e_ = 0;
  bool has_attack_graph_ = false;  // computes the AttackGraphState
  bool prenorm_ = false;
  float embedding_ffn_alpha_;
  float eps_;
  const bool has_gating_;
  const bool has_smolgen_;
  bool is_pe_dense_embedding_;  // flag for dense position encoding
  const bool use_fused_mha_;
};

// The value head implementation
// Responsible for loading weights into GPU memory, and evaluating the value
// head and value error head
template <typename DataType>
class ValueHead : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;

 public:
  ValueHead(BaseLayer<DataType>* ip, const MultiHeadWeights::ValueHead& weights,
            void* scratch, bool attention_body, bool wdl,
            ActivationFunction act, int max_batch_size, bool use_gemm_ex);
  ~ValueHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  // "convolution" in value head (legacy)
  std::unique_ptr<Conv1Layer<DataType>> conv_;

  // GPU allocations to hold various weights used by the attention policy head
  DataType *ip_val_w_, *ip_val_b_;          // "embedding" in value head
  DataType *ip1_val_w_, *ip1_val_b_;        // "FC1" in value head
  DataType *ip2_val_w_, *ip2_val_b_;        // "FC2" in value head
  DataType *ip_val_err_w_, *ip_val_err_b_;  // value error "FC" weights

  int embedding_size_;
  int value_hidden_size_;
  bool wdl_;
  bool attention_body_;
  ActivationFunction act_;
};

}  // namespace cudnn_backend
}  // namespace lczero
