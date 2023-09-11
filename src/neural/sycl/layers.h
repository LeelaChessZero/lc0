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

/*   This file is part of Leela Chess Zero.
    Modifications Copyright (C) 2023 Intel Corporation

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>. 
   
   SPDX-License-Identifier: GNU General Public License v3.0 only
*/




#pragma once

#include <sycl/sycl.hpp>

//#include <oneapi/mkl.hpp>
//#include <dpct/blas_utils.hpp>

#include <cstddef>

#include "sycl_common.h"
#include "neural/network_legacy.h"

namespace lczero {
namespace sycldnn_backend {

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

template <typename DataType>
class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }
  sycl::queue GetSycl_Queue() const { return sycl_queue_;}
 
  bool isNHWC() const { return nhwc_; }

  BaseLayer(int c, int h, int w, BaseLayer* ip, sycl::queue& sycl_queue);
  BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, sycl::queue& sycl_queue);
  BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, bool use_gemm_ex, sycl::queue& sycl_queue);
  virtual ~BaseLayer() = default;
  size_t GetOutputSize(int N) const { return sizeof(DataType) * N * C * H * W; }

  // Input2 is optional (skip connection).
  virtual void Eval(int N, DataType* output, const DataType* input,
                    const DataType* input2, void* scratch, size_t scratch_size) = 0;

 protected:
  BaseLayer* input_;
  sycl::queue& sycl_queue_;

  int C;  // Output tensor dimensions.
  int H;
  int W;

  bool nhwc_;  // tensor layout
  const bool use_gemm_ex_;

  

  void cublasRowMajorMatrixMul(const DataType* A, const DataType* B,
                               DataType* Out, int M, int N, int K,
                               int batchSize);
};

template <typename DataType>
class FCLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  FCLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool bias,
          ActivationFunction activation, sycl::queue &sycl_queue);
  ~FCLayer();

  void LoadWeights(float* cpuWeight, float* cpuBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size) override;

 private:
  const bool use_bias_;
  const ActivationFunction act_;
  DataType* weights_ = nullptr;
  DataType* biases_ = nullptr;
};

template <typename DataType>
class PolicyMapLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;
  using BaseLayer<DataType>::sycl_queue_;
  

 public:
  PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H, int W, int usedSize,
                 bool attention, sycl::queue &sycl_queue);
  ~PolicyMapLayer();

  void LoadWeights(const short* cpuWeight, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size) override;

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
  using BaseLayer<DataType>::sycl_queue_;

 public:
  SELayer(BaseLayer<DataType>* ip, int numFc1Out, bool addPrevLayerBias,
          ActivationFunction activation, sycl::queue &sycl_queue);
  ~SELayer();

  void LoadWeights(float* w1, float* b1, float* w2, float* b2,
                   float* prevLayerBias, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size) override;

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
  using BaseLayer<DataType>::sycl_queue_;

 public:
  FusedWinogradConvSELayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           int Cin, ActivationFunction activation, bool bias,
                           bool skipAdd, bool se, int se_k, bool use_gemm_ex, sycl::queue &sycl_queue,
                           bool op_nhcw = false);

  ~FusedWinogradConvSELayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size) override;

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
  using BaseLayer<DataType>::sycl_queue_;

 public:
  Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
             ActivationFunction activation, bool bias, bool use_gemm_ex, sycl::queue &sycl_queue);

  ~Conv1Layer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size) override;

 private:
  const int c_input_;
  const ActivationFunction act_;
  const bool use_bias_;

  DataType* biases_ = nullptr;
  DataType* weights_ = nullptr;

  // uses stride of 0 to read a vector as a matrix
  void cublasSpecialMatrixMul(const DataType* A, const DataType* B,
                              DataType* Out, int M, int N, int K, int batchSize);
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
  using BaseLayer<DataType>::sycl_queue_;

 public:
  ResidualBlock(BaseLayer<DataType>* ip, int C, bool se, int se_k,
                bool use_gemm_ex, bool first, bool last,
                ActivationFunction activation, int shared_mem_size, sycl::queue &sycl_queue);

  ~ResidualBlock();
  void LoadWeights0(float* pfilter, float* pBias, void* scratch);
  void LoadWeights1(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size) override;

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
  using BaseLayer<DataType>::sycl_queue_;

 public:
  AttentionPolicyHead(BaseLayer<DataType>* ip, const LegacyWeights& weights,
                      void* scratch, sycl::queue &sycl_queue);
  ~AttentionPolicyHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size) override;

 private:
  struct EncoderWeights {
    EncoderWeights(const LegacyWeights::EncoderLayer& cpu_weights,
                   void* scratch, sycl::queue &sycl_queue);
    ~EncoderWeights();
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
    sycl::queue &sycl_queue_;

    int mha_q_size_;
    int mha_k_size_;
    int mha_v_size_;
    int mha_dense_size_;

    int ffn_dense1_size_;
    int ffn_dense2_size_;
  };

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

  std::vector<EncoderWeights*> encoder_weights_;
};

}  // namespace sycldnn_backend
}  // namespace lczero
