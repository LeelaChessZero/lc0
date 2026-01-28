/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include "network_mlx.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/tables/attention_policy_map.h"
#include "neural/tables/policy_map.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace mlx_backend {

namespace mx = mlx::core;

// Use optional for MLX arrays since mx::array has no default constructor.
using OptArray = std::optional<mx::array>;

// Helper to load FC weights from BLAS column-major format.
// BLAS stores FC weights in column-major layout [output_size, input_size].
// We load as [output_size, input_size] and transpose to get row-major [input_size, output_size].
mx::array MakeFCWeights(const std::vector<float>& data, int input_size, int output_size) {
  return mx::transpose(MakeArray(data, {output_size, input_size}));
}

// MLXGraphBuilder holds the network weights and performs forward evaluation.
class MLXGraphBuilder {
 public:
  MLXGraphBuilder() = default;
  ~MLXGraphBuilder() = default;

  void Build(int input_planes, MultiHeadWeights& weights,
             InputEmbedding embedding, bool attn_body, bool attn_policy,
             bool conv_policy, bool wdl, bool moves_left,
             Activations& activations, const std::string& policy_head,
             const std::string& value_head);

  void ForwardEval(float* values, uint64_t* masks, int batch_size,
                   std::vector<float*> output_mems);

 private:
  // Pre-converted MLX weight tensors for the network.
  // Input convolution.
  OptArray input_conv_weights_;
  OptArray input_conv_biases_;

  // Residual tower weights.
  struct ResidualWeights {
    mx::array conv1_weights;
    mx::array conv1_biases;
    mx::array conv2_weights;
    mx::array conv2_biases;
    bool has_se = false;
    OptArray se_fc1_weights;
    OptArray se_fc1_biases;
    OptArray se_fc2_weights;
    OptArray se_fc2_biases;

    ResidualWeights(mx::array c1w, mx::array c1b, mx::array c2w, mx::array c2b)
        : conv1_weights(std::move(c1w)),
          conv1_biases(std::move(c1b)),
          conv2_weights(std::move(c2w)),
          conv2_biases(std::move(c2b)) {}
  };
  std::vector<ResidualWeights> residual_weights_;

  // Encoder stack weights (for attention body).
  struct EncoderWeights {
    mx::array mha_q_w, mha_q_b;
    mx::array mha_k_w, mha_k_b;
    mx::array mha_v_w, mha_v_b;
    mx::array mha_dense_w, mha_dense_b;
    mx::array ln1_gammas, ln1_betas;
    mx::array ffn_dense1_w, ffn_dense1_b;
    mx::array ffn_dense2_w, ffn_dense2_b;
    mx::array ln2_gammas, ln2_betas;
    bool has_smolgen = false;
    OptArray smolgen_compress;
    OptArray smolgen_dense1_w, smolgen_dense1_b;
    OptArray smolgen_ln1_gammas, smolgen_ln1_betas;
    OptArray smolgen_dense2_w, smolgen_dense2_b;
    OptArray smolgen_ln2_gammas, smolgen_ln2_betas;

    EncoderWeights(mx::array q_w, mx::array q_b, mx::array k_w, mx::array k_b,
                   mx::array v_w, mx::array v_b, mx::array dense_w,
                   mx::array dense_b, mx::array ln1_g, mx::array ln1_b,
                   mx::array ffn1_w, mx::array ffn1_b, mx::array ffn2_w,
                   mx::array ffn2_b, mx::array ln2_g, mx::array ln2_b)
        : mha_q_w(std::move(q_w)),
          mha_q_b(std::move(q_b)),
          mha_k_w(std::move(k_w)),
          mha_k_b(std::move(k_b)),
          mha_v_w(std::move(v_w)),
          mha_v_b(std::move(v_b)),
          mha_dense_w(std::move(dense_w)),
          mha_dense_b(std::move(dense_b)),
          ln1_gammas(std::move(ln1_g)),
          ln1_betas(std::move(ln1_b)),
          ffn_dense1_w(std::move(ffn1_w)),
          ffn_dense1_b(std::move(ffn1_b)),
          ffn_dense2_w(std::move(ffn2_w)),
          ffn_dense2_b(std::move(ffn2_b)),
          ln2_gammas(std::move(ln2_g)),
          ln2_betas(std::move(ln2_b)) {}
  };
  std::vector<EncoderWeights> encoder_weights_;
  int encoder_head_count_ = 0;

  // Embedding weights (for attention body).
  OptArray ip_emb_preproc_w_, ip_emb_preproc_b_;
  OptArray ip_emb_w_, ip_emb_b_;
  OptArray ip_emb_ln_gammas_, ip_emb_ln_betas_;
  OptArray ip_mult_gate_, ip_add_gate_;
  OptArray ip_emb_ffn_dense1_w_, ip_emb_ffn_dense1_b_;
  OptArray ip_emb_ffn_dense2_w_, ip_emb_ffn_dense2_b_;
  OptArray ip_emb_ffn_ln_gammas_, ip_emb_ffn_ln_betas_;

  // Global smolgen weights.
  OptArray smolgen_global_w_;
  bool has_smolgen_ = false;

  // Policy head weights.
  struct PolicyWeights {
    // Classical/conv policy.
    OptArray policy_conv_weights;
    OptArray policy_conv_biases;
    OptArray policy1_conv_weights;
    OptArray policy1_conv_biases;
    OptArray ip_pol_w;
    OptArray ip_pol_b;
    // Attention policy.
    OptArray ip2_pol_w, ip2_pol_b;
    OptArray ip3_pol_w, ip3_pol_b;
    OptArray ip4_pol_w;
    int pol_encoder_head_count = 0;
    std::vector<EncoderWeights> pol_encoder;
  };
  PolicyWeights policy_weights_;

  // Value head weights.
  struct ValueWeights {
    OptArray value_conv_weights;
    OptArray value_conv_biases;
    OptArray ip_val_w, ip_val_b;
    OptArray ip1_val_w, ip1_val_b;
    OptArray ip2_val_w, ip2_val_b;
  };
  ValueWeights value_weights_;

  // Moves left head weights.
  OptArray moves_left_conv_weights_;
  OptArray moves_left_conv_biases_;
  OptArray ip_mov_w_, ip_mov_b_;
  OptArray ip1_mov_w_, ip1_mov_b_;
  OptArray ip2_mov_w_, ip2_mov_b_;

  // Configuration.
  bool attn_body_ = false;
  bool attn_policy_ = false;
  bool conv_policy_ = false;
  bool wdl_ = false;
  bool moves_left_ = false;
  InputEmbedding embedding_ = INPUT_EMBEDDING_NONE;
  Activations activations_;
  int num_filters_ = 0;
  int embedding_size_ = 0;

  // Position encoding data.
  std::vector<float> pos_enc_data_;
};

void MLXGraphBuilder::Build(int input_planes, MultiHeadWeights& weights,
                            InputEmbedding embedding, bool attn_body,
                            bool attn_policy, bool conv_policy, bool wdl,
                            bool moves_left, Activations& activations,
                            const std::string& policy_head,
                            const std::string& value_head) {
  attn_body_ = attn_body;
  attn_policy_ = attn_policy;
  conv_policy_ = conv_policy;
  wdl_ = wdl;
  moves_left_ = moves_left;
  embedding_ = embedding;
  activations_ = activations;

  // Get filter count from input convolution.
  if (!attn_body) {
    num_filters_ = static_cast<int>(weights.input.biases.size());
    // Convert input convolution weights from OIHW to OHWI format.
    input_conv_weights_ =
        ConvertConvWeightsOIHWtoOHWI(weights.input.weights,
                                     num_filters_, input_planes, 3, 3);
    input_conv_biases_ =
        MakeArray(weights.input.biases, {num_filters_, 1, 1});

    // Convert residual tower weights.
    residual_weights_.reserve(weights.residual.size());
    for (const auto& res : weights.residual) {
      residual_weights_.emplace_back(
          ConvertConvWeightsOIHWtoOHWI(res.conv1.weights, num_filters_, num_filters_, 3, 3),
          MakeArray(res.conv1.biases, {num_filters_, 1, 1}),
          ConvertConvWeightsOIHWtoOHWI(res.conv2.weights, num_filters_, num_filters_, 3, 3),
          MakeArray(res.conv2.biases, {num_filters_, 1, 1}));
      auto& rw = residual_weights_.back();

      rw.has_se = res.has_se;
      if (res.has_se) {
        int se_channels = static_cast<int>(res.se.b1.size());
        // SE FC weights - use BLAS column-major format loader.
        rw.se_fc1_weights = MakeFCWeights(res.se.w1, num_filters_, se_channels);
        rw.se_fc1_biases = MakeArray(res.se.b1, {se_channels});
        rw.se_fc2_weights = MakeFCWeights(res.se.w2, se_channels, 2 * num_filters_);
        rw.se_fc2_biases = MakeArray(res.se.b2, {2 * num_filters_});
      }
    }
  } else {
    // Attention body - load embedding and encoder weights.
    embedding_size_ = static_cast<int>(weights.ip_emb_b.size());
    encoder_head_count_ = weights.encoder_head_count;

    // Embedding preprocessing (PE_DENSE only).
    if (!weights.ip_emb_preproc_w.empty()) {
      int preproc_out = static_cast<int>(weights.ip_emb_preproc_b.size());
      ip_emb_preproc_w_ = MakeFCWeights(weights.ip_emb_preproc_w, 64 * 12, preproc_out);
      ip_emb_preproc_b_ = MakeArray(weights.ip_emb_preproc_b, {preproc_out});
    }

    // Embedding layer.
    int ip_emb_in = static_cast<int>(weights.ip_emb_w.size()) / embedding_size_;
    ip_emb_w_ = MakeFCWeights(weights.ip_emb_w, ip_emb_in, embedding_size_);
    ip_emb_b_ = MakeArray(weights.ip_emb_b, {embedding_size_});

    // Embedding layer norm.
    if (!weights.ip_emb_ln_gammas.empty()) {
      ip_emb_ln_gammas_ =
          MakeArray(weights.ip_emb_ln_gammas, {embedding_size_});
      ip_emb_ln_betas_ = MakeArray(weights.ip_emb_ln_betas, {embedding_size_});
    }

    // Input gating.
    // BLAS stores as [embedding_size, 64] in row-major.
    if (!weights.ip_mult_gate.empty()) {
      ip_mult_gate_ = MakeArray(weights.ip_mult_gate, {embedding_size_, 64});
      ip_add_gate_ = MakeArray(weights.ip_add_gate, {embedding_size_, 64});
    }

    // Embedding FFN.
    if (!weights.ip_emb_ffn.dense1_w.empty()) {
      int ffn_hidden =
          static_cast<int>(weights.ip_emb_ffn.dense1_b.size());
      ip_emb_ffn_dense1_w_ = MakeFCWeights(weights.ip_emb_ffn.dense1_w, embedding_size_, ffn_hidden);
      ip_emb_ffn_dense1_b_ = MakeArray(weights.ip_emb_ffn.dense1_b, {ffn_hidden});
      ip_emb_ffn_dense2_w_ = MakeFCWeights(weights.ip_emb_ffn.dense2_w, ffn_hidden, embedding_size_);
      ip_emb_ffn_dense2_b_ = MakeArray(weights.ip_emb_ffn.dense2_b, {embedding_size_});
      ip_emb_ffn_ln_gammas_ = MakeArray(weights.ip_emb_ffn_ln_gammas, {embedding_size_});
      ip_emb_ffn_ln_betas_ = MakeArray(weights.ip_emb_ffn_ln_betas, {embedding_size_});
    }

    // Global smolgen weights.
    has_smolgen_ = weights.has_smolgen;
    if (has_smolgen_) {
      int smolgen_out = static_cast<int>(weights.smolgen_w.size()) / 64 / 64;
      smolgen_global_w_ = MakeFCWeights(weights.smolgen_w, smolgen_out, 64 * 64);
    }

    // Encoder layers - use MakeFCWeights for all FC weight matrices.
    encoder_weights_.reserve(weights.encoder.size());
    for (const auto& enc : weights.encoder) {
      int qkv_size = static_cast<int>(enc.mha.q_b.size());
      int ffn_hidden = static_cast<int>(enc.ffn.dense1_b.size());


      encoder_weights_.emplace_back(
          MakeFCWeights(enc.mha.q_w, embedding_size_, qkv_size),
          MakeArray(enc.mha.q_b, {qkv_size}),
          MakeFCWeights(enc.mha.k_w, embedding_size_, qkv_size),
          MakeArray(enc.mha.k_b, {qkv_size}),
          MakeFCWeights(enc.mha.v_w, embedding_size_, qkv_size),
          MakeArray(enc.mha.v_b, {qkv_size}),
          MakeFCWeights(enc.mha.dense_w, qkv_size, embedding_size_),
          MakeArray(enc.mha.dense_b, {embedding_size_}),
          MakeArray(enc.ln1_gammas, {embedding_size_}),
          MakeArray(enc.ln1_betas, {embedding_size_}),
          MakeFCWeights(enc.ffn.dense1_w, embedding_size_, ffn_hidden),
          MakeArray(enc.ffn.dense1_b, {ffn_hidden}),
          MakeFCWeights(enc.ffn.dense2_w, ffn_hidden, embedding_size_),
          MakeArray(enc.ffn.dense2_b, {embedding_size_}),
          MakeArray(enc.ln2_gammas, {embedding_size_}),
          MakeArray(enc.ln2_betas, {embedding_size_}));
      auto& ew = encoder_weights_.back();

      ew.has_smolgen = enc.mha.has_smolgen;
      if (enc.mha.has_smolgen) {
        int hidden =
            static_cast<int>(enc.mha.smolgen.compress.size()) / embedding_size_;
        ew.smolgen_compress = MakeFCWeights(enc.mha.smolgen.compress, embedding_size_, hidden);
        int dense1_out = static_cast<int>(enc.mha.smolgen.dense1_b.size());
        ew.smolgen_dense1_w = MakeFCWeights(enc.mha.smolgen.dense1_w, 64 * hidden, dense1_out);
        ew.smolgen_dense1_b = MakeArray(enc.mha.smolgen.dense1_b, {dense1_out});
        ew.smolgen_ln1_gammas = MakeArray(enc.mha.smolgen.ln1_gammas, {dense1_out});
        ew.smolgen_ln1_betas = MakeArray(enc.mha.smolgen.ln1_betas, {dense1_out});
        int dense2_out = static_cast<int>(enc.mha.smolgen.dense2_b.size());
        ew.smolgen_dense2_w = MakeFCWeights(enc.mha.smolgen.dense2_w, dense1_out, dense2_out);
        ew.smolgen_dense2_b = MakeArray(enc.mha.smolgen.dense2_b, {dense2_out});
        ew.smolgen_ln2_gammas = MakeArray(enc.mha.smolgen.ln2_gammas, {dense2_out});
        ew.smolgen_ln2_betas = MakeArray(enc.mha.smolgen.ln2_betas, {dense2_out});
      }
    }
  }

  // Policy head weights.
  // BLAS stores FC weights in column-major layout.
  auto& pol_head = weights.policy_heads.at(policy_head);
  if (attn_policy) {
    int pol_emb_size = static_cast<int>(pol_head.ip_pol_b.size());
    policy_weights_.ip_pol_w = MakeFCWeights(pol_head.ip_pol_w, embedding_size_, pol_emb_size);
    policy_weights_.ip_pol_b = MakeArray(pol_head.ip_pol_b, {pol_emb_size});

    int pol_dmodel = static_cast<int>(pol_head.ip2_pol_b.size());
    policy_weights_.ip2_pol_w = MakeFCWeights(pol_head.ip2_pol_w, pol_emb_size, pol_dmodel);
    policy_weights_.ip2_pol_b = MakeArray(pol_head.ip2_pol_b, {pol_dmodel});
    policy_weights_.ip3_pol_w = MakeFCWeights(pol_head.ip3_pol_w, pol_emb_size, pol_dmodel);
    policy_weights_.ip3_pol_b = MakeArray(pol_head.ip3_pol_b, {pol_dmodel});
    policy_weights_.ip4_pol_w = MakeFCWeights(pol_head.ip4_pol_w, pol_dmodel, 4);
    policy_weights_.pol_encoder_head_count = pol_head.pol_encoder_head_count;

    // Policy encoder layers.
    policy_weights_.pol_encoder.reserve(pol_head.pol_encoder.size());
    for (const auto& enc : pol_head.pol_encoder) {
      int qkv_size = static_cast<int>(enc.mha.q_b.size());
      int ffn_hidden = static_cast<int>(enc.ffn.dense1_b.size());

      policy_weights_.pol_encoder.emplace_back(
          MakeFCWeights(enc.mha.q_w, pol_emb_size, qkv_size),
          MakeArray(enc.mha.q_b, {qkv_size}),
          MakeFCWeights(enc.mha.k_w, pol_emb_size, qkv_size),
          MakeArray(enc.mha.k_b, {qkv_size}),
          MakeFCWeights(enc.mha.v_w, pol_emb_size, qkv_size),
          MakeArray(enc.mha.v_b, {qkv_size}),
          MakeFCWeights(enc.mha.dense_w, qkv_size, pol_emb_size),
          MakeArray(enc.mha.dense_b, {pol_emb_size}),
          MakeArray(enc.ln1_gammas, {pol_emb_size}),
          MakeArray(enc.ln1_betas, {pol_emb_size}),
          MakeFCWeights(enc.ffn.dense1_w, pol_emb_size, ffn_hidden),
          MakeArray(enc.ffn.dense1_b, {ffn_hidden}),
          MakeFCWeights(enc.ffn.dense2_w, ffn_hidden, pol_emb_size),
          MakeArray(enc.ffn.dense2_b, {pol_emb_size}),
          MakeArray(enc.ln2_gammas, {pol_emb_size}),
          MakeArray(enc.ln2_betas, {pol_emb_size}));
      policy_weights_.pol_encoder.back().has_smolgen = enc.mha.has_smolgen;
    }
  } else if (conv_policy) {
    int pol1_channels = static_cast<int>(pol_head.policy1.biases.size());
    policy_weights_.policy1_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(pol_head.policy1.weights, pol1_channels, num_filters_, 3, 3);
    policy_weights_.policy1_conv_biases =
        MakeArray(pol_head.policy1.biases, {pol1_channels, 1, 1});
    int pol_channels = static_cast<int>(pol_head.policy.biases.size());
    policy_weights_.policy_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(pol_head.policy.weights, pol_channels, pol1_channels, 3, 3);
    policy_weights_.policy_conv_biases =
        MakeArray(pol_head.policy.biases, {pol_channels, 1, 1});
  } else {
    // Classical policy.
    int pol_channels = static_cast<int>(pol_head.policy.biases.size());
    policy_weights_.policy_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(pol_head.policy.weights, pol_channels, num_filters_, 1, 1);
    policy_weights_.policy_conv_biases =
        MakeArray(pol_head.policy.biases, {pol_channels, 1, 1});
    int pol_outputs = static_cast<int>(pol_head.ip_pol_b.size());
    policy_weights_.ip_pol_w = MakeFCWeights(pol_head.ip_pol_w, pol_channels * 64, pol_outputs);
    policy_weights_.ip_pol_b = MakeArray(pol_head.ip_pol_b, {pol_outputs});
  }

  // Value head weights.
  // BLAS stores FC weights in column-major layout.
  auto& val_head = weights.value_heads.at(value_head);
  if (attn_body) {
    int val_emb_size = static_cast<int>(val_head.ip_val_b.size());
    value_weights_.ip_val_w = MakeFCWeights(val_head.ip_val_w, embedding_size_, val_emb_size);
    value_weights_.ip_val_b = MakeArray(val_head.ip_val_b, {val_emb_size});
  } else {
    int val_channels = static_cast<int>(val_head.value.biases.size());
    value_weights_.value_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(val_head.value.weights, val_channels, num_filters_, 1, 1);
    value_weights_.value_conv_biases =
        MakeArray(val_head.value.biases, {val_channels, 1, 1});
  }
  int val_hidden = static_cast<int>(val_head.ip1_val_b.size());
  int ip1_val_in = static_cast<int>(val_head.ip1_val_w.size()) / val_hidden;
  value_weights_.ip1_val_w = MakeFCWeights(val_head.ip1_val_w, ip1_val_in, val_hidden);
  value_weights_.ip1_val_b = MakeArray(val_head.ip1_val_b, {val_hidden});
  int val_outputs = static_cast<int>(val_head.ip2_val_b.size());
  value_weights_.ip2_val_w = MakeFCWeights(val_head.ip2_val_w, val_hidden, val_outputs);
  value_weights_.ip2_val_b = MakeArray(val_head.ip2_val_b, {val_outputs});

  // Moves left head weights.
  // BLAS stores FC weights in column-major layout.
  if (moves_left) {
    if (attn_body) {
      int mov_emb_size = static_cast<int>(weights.ip_mov_b.size());
      ip_mov_w_ = MakeFCWeights(weights.ip_mov_w, embedding_size_, mov_emb_size);
      ip_mov_b_ = MakeArray(weights.ip_mov_b, {mov_emb_size});
    } else {
      int mov_channels = static_cast<int>(weights.moves_left.biases.size());
      moves_left_conv_weights_ =
          ConvertConvWeightsOIHWtoOHWI(weights.moves_left.weights, mov_channels, num_filters_, 1, 1);
      moves_left_conv_biases_ =
          MakeArray(weights.moves_left.biases, {mov_channels, 1, 1});
    }
    int mov_hidden = static_cast<int>(weights.ip1_mov_b.size());
    int ip1_mov_in = static_cast<int>(weights.ip1_mov_w.size()) / mov_hidden;
    ip1_mov_w_ = MakeFCWeights(weights.ip1_mov_w, ip1_mov_in, mov_hidden);
    ip1_mov_b_ = MakeArray(weights.ip1_mov_b, {mov_hidden});
    ip2_mov_w_ = MakeFCWeights(weights.ip2_mov_w, mov_hidden, 1);
    ip2_mov_b_ = MakeArray(weights.ip2_mov_b, {1});
  }
}

void MLXGraphBuilder::ForwardEval(float* values, uint64_t* masks, int batch_size,
                                  std::vector<float*> output_mems) {
  // Create input arrays.
  mx::array input_vals = mx::array(values,
                                   mx::Shape{batch_size, kInputPlanes},
                                   mx::float32);
  mx::array input_masks = mx::array(masks,
                                    mx::Shape{batch_size, kInputPlanes},
                                    mx::uint64);

  // Expand input to [batch, 112, 8, 8].
  mx::array flow = ExpandInput(input_masks, input_vals, batch_size);

  if (!attn_body_) {
    // Classical/SE network: input convolution.
    flow = ConvBlock(flow, *input_conv_weights_, *input_conv_biases_, 3,
                     activations_.default_activation);

    // Residual tower.
    for (const auto& rw : residual_weights_) {
      mx::array se_fc1_w = rw.se_fc1_weights.value_or(mx::array(0.0f));
      mx::array se_fc1_b = rw.se_fc1_biases.value_or(mx::array(0.0f));
      mx::array se_fc2_w = rw.se_fc2_weights.value_or(mx::array(0.0f));
      mx::array se_fc2_b = rw.se_fc2_biases.value_or(mx::array(0.0f));
      flow = ResidualBlock(flow, rw.conv1_weights, rw.conv1_biases,
                           rw.conv2_weights, rw.conv2_biases, rw.has_se,
                           se_fc1_w, se_fc1_b, se_fc2_w, se_fc2_b,
                           activations_.default_activation);
    }
  } else {
    // Attention body network.
    // Reshape from NCHW [batch, 112, 8, 8] to NHWC [batch, 64, 112].
    flow = mx::transpose(flow, {0, 2, 3, 1});
    flow = mx::reshape(flow, {batch_size, 64, kInputPlanes});

    // Handle position encoding based on embedding type.
    if (embedding_ == INPUT_EMBEDDING_PE_DENSE && ip_emb_preproc_w_.has_value()) {
      // PE_DENSE: Take first 12 channels, flatten, FC, reshape, concat.
      // Extract first 12 channels (piece positions).
      mx::array input_12 = mx::slice(flow, {0, 0, 0}, {batch_size, 64, 12});
      // Flatten to [batch, 64*12].
      input_12 = mx::reshape(input_12, {batch_size, 64 * 12});
      // FC preproc to [batch, preproc_out] where preproc_out = 64 * enc_channels.
      mx::array pos_enc = FullyConnected(input_12, *ip_emb_preproc_w_,
                                         *ip_emb_preproc_b_, "");
      // Reshape to [batch, 64, enc_channels].
      int enc_channels = static_cast<int>(pos_enc.size()) / (batch_size * 64);
      pos_enc = mx::reshape(pos_enc, {batch_size, 64, enc_channels});
      // Concat with original 112 channels: [batch, 64, 112 + enc_channels].
      flow = mx::concatenate({flow, pos_enc}, 2);
    } else if (embedding_ == INPUT_EMBEDDING_PE_MAP) {
      // PE_MAP: Concat static position encoding (64 channels per square).
      // kPosEncoding is [64][64] = [square][channel].
      mx::array pos_enc = mx::array(reinterpret_cast<const float*>(kPosEncoding),
                                    mx::Shape{64, kNumPosEncodingChannels},
                                    mx::float32);
      // Broadcast to [batch, 64, 64].
      pos_enc = mx::broadcast_to(pos_enc, {batch_size, 64, kNumPosEncodingChannels});
      // Concat with 112 input channels: [batch, 64, 112 + 64].
      flow = mx::concatenate({flow, pos_enc}, 2);
    }
    // For INPUT_EMBEDDING_NONE, just use the 112 channels directly.

    // Main embedding.
    flow = FullyConnected(flow, *ip_emb_w_, *ip_emb_b_,
                          activations_.default_activation);

    // Embedding layer norm (for PE_DENSE).
    if (ip_emb_ln_gammas_.has_value()) {
      flow = LayerNorm(flow, *ip_emb_ln_gammas_, *ip_emb_ln_betas_, 1e-3f);
    }

    // Input gating.
    if (ip_mult_gate_.has_value()) {
      flow = GatingLayer(flow, *ip_mult_gate_, "mult");
      flow = GatingLayer(flow, *ip_add_gate_, "add");
    }

    // Compute alpha for encoder skip connections (BLAS style).
    // alpha = (2 * num_encoders)^(-0.25)
    float alpha = static_cast<float>(std::pow(2.0 * encoder_weights_.size(), -0.25));
    // Use 1e-3 epsilon for PE_DENSE, 1e-6 otherwise.
    float default_eps = (embedding_ == INPUT_EMBEDDING_PE_DENSE) ? 1e-3f : 1e-6f;

    // Embedding FFN (for PE_DENSE).
    if (ip_emb_ffn_dense1_w_.has_value()) {
      mx::array ffn = FullyConnected(flow, *ip_emb_ffn_dense1_w_,
                                     *ip_emb_ffn_dense1_b_,
                                     activations_.ffn_activation);
      ffn = FullyConnected(ffn, *ip_emb_ffn_dense2_w_, *ip_emb_ffn_dense2_b_, "");
      flow = LayerNormWithSkip(flow, ffn, *ip_emb_ffn_ln_gammas_,
                               *ip_emb_ffn_ln_betas_, alpha, 1e-3f);
    }

    // Encoder layers.
    for (size_t i = 0; i < encoder_weights_.size(); i++) {
      const auto& ew = encoder_weights_[i];

      // Q, K, V projections.
      mx::array q = FullyConnected(flow, ew.mha_q_w, ew.mha_q_b, "");
      mx::array k = FullyConnected(flow, ew.mha_k_w, ew.mha_k_b, "");
      mx::array v = FullyConnected(flow, ew.mha_v_w, ew.mha_v_b, "");

      // Compute smolgen attention weights if needed.
      std::optional<mx::array> smolgen_attn;
      if (ew.has_smolgen && smolgen_global_w_.has_value()) {
        smolgen_attn = ComputeSmolgen(
            flow, encoder_head_count_,
            *ew.smolgen_compress,
            *ew.smolgen_dense1_w, *ew.smolgen_dense1_b,
            *ew.smolgen_ln1_gammas, *ew.smolgen_ln1_betas,
            *ew.smolgen_dense2_w, *ew.smolgen_dense2_b,
            *ew.smolgen_ln2_gammas, *ew.smolgen_ln2_betas,
            *smolgen_global_w_,
            activations_.smolgen_activation);
      }

      // Multi-head attention with optional smolgen.
      mx::array mha = smolgen_attn.has_value()
          ? MultiHeadAttention(q, k, v, encoder_head_count_, &*smolgen_attn)
          : MultiHeadAttention(q, k, v, encoder_head_count_, nullptr);

      // MHA dense layer.
      mha = FullyConnected(mha, ew.mha_dense_w, ew.mha_dense_b, "");

      // Skip connection + layer norm.
      flow = LayerNormWithSkip(flow, mha, ew.ln1_gammas, ew.ln1_betas, alpha, default_eps);

      // FFN.
      mx::array ffn =
          FullyConnected(flow, ew.ffn_dense1_w, ew.ffn_dense1_b,
                         activations_.ffn_activation);
      ffn = FullyConnected(ffn, ew.ffn_dense2_w, ew.ffn_dense2_b, "");

      // Skip connection + layer norm.
      flow = LayerNormWithSkip(flow, ffn, ew.ln2_gammas, ew.ln2_betas, alpha, default_eps);
    }
  }

  // Policy head.
  mx::array policy = mx::array(0.0f);  // Dummy init, will be assigned.
  if (attn_policy_) {
    // Attention policy head.
    mx::array pol_flow = flow;

    // Square embedding.
    pol_flow = FullyConnected(pol_flow, *policy_weights_.ip_pol_w,
                              *policy_weights_.ip_pol_b,
                              attn_body_ ? activations_.default_activation
                                         : "selu");

    // Policy encoder layers.
    for (size_t i = 0; i < policy_weights_.pol_encoder.size(); i++) {
      const auto& ew = policy_weights_.pol_encoder[i];

      mx::array q = FullyConnected(pol_flow, ew.mha_q_w, ew.mha_q_b, "");
      mx::array k = FullyConnected(pol_flow, ew.mha_k_w, ew.mha_k_b, "");
      mx::array v = FullyConnected(pol_flow, ew.mha_v_w, ew.mha_v_b, "");

      mx::array mha = MultiHeadAttention(
          q, k, v, policy_weights_.pol_encoder_head_count, nullptr);
      mha = FullyConnected(mha, ew.mha_dense_w, ew.mha_dense_b, "");

      pol_flow =
          LayerNormWithSkip(pol_flow, mha, ew.ln1_gammas, ew.ln1_betas, 1.0f);

      mx::array ffn =
          FullyConnected(pol_flow, ew.ffn_dense1_w, ew.ffn_dense1_b,
                         attn_body_ ? activations_.ffn_activation : "selu");
      ffn = FullyConnected(ffn, ew.ffn_dense2_w, ew.ffn_dense2_b, "");

      pol_flow =
          LayerNormWithSkip(pol_flow, ffn, ew.ln2_gammas, ew.ln2_betas, 1.0f);
    }

    // Self-attention Q and K.
    mx::array queries = FullyConnected(pol_flow, *policy_weights_.ip2_pol_w,
                                       *policy_weights_.ip2_pol_b, "");
    mx::array keys = FullyConnected(pol_flow, *policy_weights_.ip3_pol_w,
                                    *policy_weights_.ip3_pol_b, "");

    // Scaled Q*K matmul.
    int pol_dmodel = static_cast<int>(policy_weights_.ip2_pol_b->size());
    policy = ScaledQKMatmul(queries, keys, 1.0f / std::sqrt(static_cast<float>(pol_dmodel)));

    // Promotion logits.
    policy = AttentionPolicyPromoMatmulConcat(
        policy, keys, *policy_weights_.ip4_pol_w, 56, pol_dmodel);

    // Reshape and apply policy map on CPU.
    policy = mx::reshape(policy, {batch_size, -1});
  } else if (conv_policy_) {
    // Convolution policy head.
    policy = ConvBlock(flow, *policy_weights_.policy1_conv_weights,
                       *policy_weights_.policy1_conv_biases, 3,
                       activations_.default_activation);
    policy = ConvBlock(policy, *policy_weights_.policy_conv_weights,
                       *policy_weights_.policy_conv_biases, 3, "");
    // ConvBlock returns NCHW [batch, C, H, W].
    // The kConvPolicyMap expects NCHW layout where index = plane * 64 + square.
    // Just reshape - no transpose needed since data is already NCHW.
    policy = mx::reshape(policy, {batch_size, -1});
  } else {
    // Classical policy head.
    policy = ConvBlock(flow, *policy_weights_.policy_conv_weights,
                       *policy_weights_.policy_conv_biases, 1,
                       activations_.default_activation);
    policy = mx::reshape(policy, {batch_size, -1});
    policy = FullyConnected(policy, *policy_weights_.ip_pol_w,
                            *policy_weights_.ip_pol_b, "");
  }

  // Value head.
  mx::array value = mx::array(0.0f);  // Dummy init, will be assigned.
  if (attn_body_) {
    // Attention body value head - use embedding from encoder output.
    value = FullyConnected(flow, *value_weights_.ip_val_w,
                           *value_weights_.ip_val_b,
                           activations_.default_activation);
    value = mx::reshape(value, {batch_size, -1});
  } else {
    value = ConvBlock(flow, *value_weights_.value_conv_weights,
                      *value_weights_.value_conv_biases, 1,
                      activations_.default_activation);
    value = mx::reshape(value, {batch_size, -1});
  }

  value = FullyConnected(value, *value_weights_.ip1_val_w,
                         *value_weights_.ip1_val_b,
                         activations_.default_activation);
  value = FullyConnected(value, *value_weights_.ip2_val_w,
                         *value_weights_.ip2_val_b, wdl_ ? "softmax" : "tanh");

  // Moves left head.
  mx::array moves_left_out = mx::array(0.0f);  // Dummy init.
  if (moves_left_) {
    if (attn_body_) {
      moves_left_out =
          FullyConnected(flow, *ip_mov_w_, *ip_mov_b_,
                         activations_.default_activation);
      moves_left_out = mx::reshape(moves_left_out, {batch_size, -1});
    } else {
      moves_left_out = ConvBlock(flow, *moves_left_conv_weights_,
                                 *moves_left_conv_biases_, 1,
                                 activations_.default_activation);
      moves_left_out = mx::reshape(moves_left_out, {batch_size, -1});
    }
    moves_left_out = FullyConnected(moves_left_out, *ip1_mov_w_, *ip1_mov_b_,
                                    activations_.default_activation);
    moves_left_out =
        FullyConnected(moves_left_out, *ip2_mov_w_, *ip2_mov_b_, "relu");
  }

  // Trigger MLX lazy evaluation.
  if (moves_left_) {
    mx::eval({policy, value, moves_left_out});
  } else {
    mx::eval({policy, value});
  }

  // Verify output arrays are valid before accessing data.
  assert(policy.size() > 0 && policy.size() % batch_size == 0);
  assert(value.size() == static_cast<size_t>(batch_size) * (wdl_ ? 3 : 1));
  if (moves_left_) {
    assert(moves_left_out.size() == static_cast<size_t>(batch_size));
  }

  // Copy results to output buffers.
  // Policy map is applied on CPU for attention/conv policy.
  if (attn_policy_) {
    // Attention policy: tensor is [batch, 64*64 + 8*24], input_stride = map_size.
    constexpr size_t kAttnPolicySize = 64 * 64 + 8 * 24;
    ApplyPolicyMap(
        std::span<const float>(policy.data<float>(), batch_size * kAttnPolicySize),
        std::span<float>(output_mems[0], batch_size * kNumOutputPolicy),
        kAttnPolicyMap, kAttnPolicySize);
  } else if (conv_policy_) {
    // Conv policy: tensor is [batch, num_channels * 64] where num_channels >= 73.
    // The kConvPolicyMap reads 73*64 elements, but tensor stride is larger.
    size_t tensor_stride = policy.size() / batch_size;  // Actual elements per batch.
    ApplyPolicyMap(
        std::span<const float>(policy.data<float>(), policy.size()),
        std::span<float>(output_mems[0], batch_size * kNumOutputPolicy),
        kConvPolicyMap, tensor_stride);
  } else {
    // Classical policy: directly copy 1858 outputs.
    std::memcpy(output_mems[0], policy.data<float>(),
                batch_size * kNumOutputPolicy * sizeof(float));
  }

  // Value output.
  std::memcpy(output_mems[1], value.data<float>(),
              batch_size * (wdl_ ? 3 : 1) * sizeof(float));

  // Moves left output.
  if (moves_left_) {
    std::memcpy(output_mems[2], moves_left_out.data<float>(),
                batch_size * sizeof(float));
  }
}

// MLXNetworkComputation implementation.
MLXNetworkComputation::MLXNetworkComputation(MLXNetwork* network, bool wdl,
                                             bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

MLXNetworkComputation::~MLXNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void MLXNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

// Helper to convert activation string.
std::string activationString(pblczero::NetworkFormat::ActivationFunction act) {
  switch (act) {
    case pblczero::NetworkFormat::ACTIVATION_RELU:
      return "relu";
    case pblczero::NetworkFormat::ACTIVATION_MISH:
      return "mish";
    case pblczero::NetworkFormat::ACTIVATION_NONE:
      return "";
    case pblczero::NetworkFormat::ACTIVATION_TANH:
      return "tanh";
    case pblczero::NetworkFormat::ACTIVATION_SIGMOID:
      return "sigmoid";
    case pblczero::NetworkFormat::ACTIVATION_SELU:
      return "selu";
    case pblczero::NetworkFormat::ACTIVATION_SWISH:
      return "swish";
    case pblczero::NetworkFormat::ACTIVATION_RELU_2:
      return "relu_2";
    case pblczero::NetworkFormat::ACTIVATION_SOFTMAX:
      return "softmax";
    default:
      return "";
  }
}

// MLXNetwork implementation.
MLXNetwork::MLXNetwork(const WeightsFile& file, const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()} {
  MultiHeadWeights weights(file.weights());

  builder_ = std::make_unique<MLXGraphBuilder>();
  CERR << "Initialized MLX backend";

  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  attn_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_ATTENTION;

  wdl_ = file.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;

  moves_left_ = (file.format().network_format().moves_left() ==
                 pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                options.GetOrDefault<bool>("mlh", true);

  bool attn_body =
      (file.format().network_format().network()) ==
          pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
      (file.format().network_format().network()) ==
          pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;

  // Build activations configuration.
  Activations activations;
  activations.default_activation =
      file.format().network_format().default_activation() ==
              pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH
          ? "mish"
          : "relu";
  const auto smolgen_activation =
      file.format().network_format().smolgen_activation();
  activations.smolgen_activation =
      smolgen_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
          ? activations.default_activation
          : activationString(
                static_cast<pblczero::NetworkFormat::ActivationFunction>(
                    smolgen_activation));
  const auto ffn_activation = file.format().network_format().ffn_activation();
  activations.ffn_activation =
      ffn_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
          ? activations.default_activation
          : activationString(
                static_cast<pblczero::NetworkFormat::ActivationFunction>(
                    ffn_activation));

  std::string policy_head =
      options.GetOrDefault<std::string>("policy_head", "vanilla");
  if (weights.policy_heads.count(policy_head) == 0) {
    throw Exception("The policy head you specified '" + policy_head +
                    "' does not exist in this net.");
  }
  std::string value_head =
      options.GetOrDefault<std::string>("value_head", "winner");
  if (weights.value_heads.count(value_head) == 0) {
    throw Exception("The value head you specified '" + value_head +
                    "' does not exist in this net.");
  }

  auto embedding = static_cast<InputEmbedding>(
      file.format().network_format().input_embedding());
  builder_->Build(kInputPlanes, weights, embedding, attn_body, attn_policy_,
                  conv_policy_, wdl_, moves_left_, activations, policy_head,
                  value_head);
}

MLXNetwork::~MLXNetwork() = default;

void MLXNetwork::forwardEval(InputsOutputs* io, int batchSize) {
  // MLX evaluation - use lock for thread safety.
  std::lock_guard<std::mutex> lock(lock_);

  if (moves_left_) {
    builder_->ForwardEval(&io->input_val_mem_[0], &io->input_masks_mem_[0],
                          batchSize,
                          {&io->op_policy_mem_[0], &io->op_value_mem_[0],
                           &io->op_moves_left_mem_[0]});
  } else {
    builder_->ForwardEval(&io->input_val_mem_[0], &io->input_masks_mem_[0],
                          batchSize,
                          {&io->op_policy_mem_[0], &io->op_value_mem_[0]});
  }
}

std::unique_ptr<Network> MakeMLXNetwork(const std::optional<WeightsFile>& w,
                                        const OptionsDict& options) {
  if (!w) {
    throw Exception("The MLX backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  auto nf = weights.format().network_format();
  using NF = pblczero::NetworkFormat;
  switch (nf.network()) {
    case NF::NETWORK_CLASSICAL_WITH_HEADFORMAT:
    case NF::NETWORK_SE_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
      break;
    default:
      throw Exception("Network format " +
                      NF::NetworkStructure_Name(nf.network()) +
                      " is not supported by the MLX backend.");
  }
  switch (nf.policy()) {
    case NF::POLICY_CLASSICAL:
    case NF::POLICY_CONVOLUTION:
    case NF::POLICY_ATTENTION:
      break;
    default:
      throw Exception("Policy format " + NF::PolicyFormat_Name(nf.policy()) +
                      " is not supported by the MLX backend.");
  }
  switch (nf.value()) {
    case NF::VALUE_CLASSICAL:
    case NF::VALUE_WDL:
      break;
    default:
      throw Exception("Value format " + NF::ValueFormat_Name(nf.value()) +
                      " is not supported by the MLX backend.");
  }
  switch (nf.moves_left()) {
    case NF::MOVES_LEFT_NONE:
    case NF::MOVES_LEFT_V1:
      break;
    default:
      throw Exception("Moves left head format " +
                      NF::MovesLeftFormat_Name(nf.moves_left()) +
                      " is not supported by the MLX backend.");
  }
  switch (nf.default_activation()) {
    case NF::DEFAULT_ACTIVATION_RELU:
    case NF::DEFAULT_ACTIVATION_MISH:
      break;
    default:
      throw Exception("Default activation " +
                      NF::DefaultActivation_Name(nf.default_activation()) +
                      " is not supported by the MLX backend.");
  }
  switch (nf.input_embedding()) {
    case NF::INPUT_EMBEDDING_NONE:
    case NF::INPUT_EMBEDDING_PE_MAP:
    case NF::INPUT_EMBEDDING_PE_DENSE:
      break;
    default:
      throw Exception("Input embedding " +
                      NF::InputEmbeddingFormat_Name(nf.input_embedding()) +
                      " is not supported by the MLX backend.");
  }
  return std::make_unique<MLXNetwork>(weights, options);
}

REGISTER_NETWORK("mlx", MakeMLXNetwork, 110)

}  // namespace mlx_backend
}  // namespace lczero
