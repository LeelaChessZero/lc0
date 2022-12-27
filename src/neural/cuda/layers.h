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

#include <cublas_v2.h>

#include <cstddef>

#include "cuda_common.h"
#include "neural/network_legacy.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#else
typedef void* cudnnHandle_t;
#endif

namespace lczero {
namespace cudnn_backend {

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

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
                    cudaStream_t stream) = 0;

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
            ActivationFunction activation = NONE, bool bias = false);

  ConvLayer(bool nhwc, int C, int H, int W, int size, int Cin,
            ActivationFunction activation = NONE, bool bias = false);

  ~ConvLayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

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
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

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
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

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
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

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
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

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
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

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
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

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
  EncoderBlock(const LegacyWeights::EncoderLayer& cpu_weights, void* scratch,
               int heads, int size, float alpha, DataType* smolgen_global_scratch,
               int smolgen_global_size, int max_batch_size);
  ~EncoderBlock();

  void Eval(int N, DataType* inpop, DataType* scratch0, DataType* scratch1,
            DataType* scratch2, cublasHandle_t cublas,
            cudaStream_t stream, ActivationFunction act) const;

  // all GPU side pointers
  DataType *mha_q_w, *mha_q_b;
  DataType *mha_k_w, *mha_k_b;
  DataType *mha_v_w, *mha_v_b;
  DataType *mha_qkv_w, *mha_qkv_b;
  DataType *mha_dense_w, *mha_dense_b;

  DataType *ln1_gammas, *ln1_betas;

  DataType *ffn_dense1_w, *ffn_dense1_b;
  DataType *ffn_dense2_w, *ffn_dense2_b;

  DataType *ln2_gammas, *ln2_betas;

  DataType *smol_compress;
  DataType *smol_dense1_w, *smol_dense1_b;
  DataType *smol_dense2_w, *smol_dense2_b;
  DataType *smol_ln1_gammas, *smol_ln1_betas;
  DataType *smol_ln2_gammas, *smol_ln2_betas;
  DataType *smol_global;

  int mha_q_size_;
  int mha_k_size_;
  int mha_v_size_;
  int mha_dense_size_;

  int ffn_dense1_size_;
  int ffn_dense2_size_;

  int embedding_op_size_;
  int encoder_heads_;

  float alpha_; // scale to apply to skip connection add

  const bool has_smolgen_;

  // Output sizes for smolgen layers.
  int smol_compress_size_;
  int smol_dense_1_size_;
  int smol_dense_2_size_;
  int smol_global_size_;

  const int max_batch_size_;
  mutable DataType** scratch_rel_ptrs_;
  mutable DataType* last_known_scratch_;
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

 public:
  AttentionPolicyHead(BaseLayer<DataType>* ip, const LegacyWeights& weights,
                      void* scratch, bool attention_body, ActivationFunction act, int max_batch_size);
  ~AttentionPolicyHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

 private:
  // GPU allocations to hold various weights used by the attention policy head
  DataType *ip_pol_w_, *ip_pol_b_;    // "embedding" in policy attention
  DataType *ip2_pol_w_, *ip2_pol_b_;  // "wq" in policy attention
  DataType *ip3_pol_w_, *ip3_pol_b_;  // "wk" in policy attention
  DataType* ip4_pol_w_;               // "ppo" in policy attention

  DataType *wqk_w_, *wqk_b_;          // allocation containing both "wq" and "wq"

  int embedding_op_size_;
  int wq_op_size_;
  int wk_op_size_;

  int encoder_heads_;
  int policy_d_model_;
  bool attention_body_;
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
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;
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
  AttentionBody(const LegacyWeights& weights, void* scratch,
                ActivationFunction default_act, int num_res_blocks,
                int input_c, int max_batch_size);
  ~AttentionBody();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas,
            cudaStream_t stream) override;

 private:
  // GPU allocations to hold various weights used by the attention policy head
  DataType *ip_emb_w_, *ip_emb_b_;    // "embedding" layer in net body
  DataType *ip_mult_gate_, *ip_add_gate_; // input gating
  DataType *smolgen_global_; // global smolgen weights for all encoder layers
  int embedding_op_size_;
  int encoder_head_count_;
  std::vector<EncoderBlock<DataType>*> encoder_weights_;
  ActivationFunction default_act_;
  int num_resi_blocks_;
  int input_c_;
  int smolgen_global_size_;
  const bool has_gating_;
  const bool has_smolgen_;
};


}  // namespace cudnn_backend
}  // namespace lczero
