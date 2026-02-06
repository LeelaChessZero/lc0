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
using OptQuantized = std::optional<QuantizedWeight>;

// Helper to load FC weights from BLAS column-major format.
// BLAS stores FC weights in column-major layout [output_size, input_size].
// We load as [output_size, input_size] and transpose to get row-major [input_size, output_size].
mx::array MakeFCWeights(const std::vector<float>& data, int input_size, int output_size,
                        mx::Dtype dtype = mx::float32) {
  return mx::transpose(MakeArray(data, {output_size, input_size}, dtype));
}

// Helper to dispatch to quantized or float FC based on weight availability.
// Uses quantized weights when available, otherwise falls back to float weights.
// Version for optional float weight (e.g., embedding weights that may be quantized).
mx::array DispatchFC(const mx::array& input,
                     const OptQuantized& q_weight,
                     const OptArray& f_weight,
                     const mx::array& biases,
                     const std::string& activation) {
  if (q_weight.has_value()) {
    return QuantizedFullyConnected(input, *q_weight, biases, activation);
  }
  assert(f_weight.has_value());
  return FullyConnected(input, *f_weight, biases, activation);
}

// Version for always-present float weight (e.g., encoder weights).
mx::array DispatchFC(const mx::array& input,
                     const OptQuantized& q_weight,
                     const mx::array& f_weight,
                     const mx::array& biases,
                     const std::string& activation) {
  if (q_weight.has_value()) {
    return QuantizedFullyConnected(input, *q_weight, biases, activation);
  }
  return FullyConnected(input, f_weight, biases, activation);
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
             const std::string& value_head, Precision precision,
             int group_size = 64);

  void ForwardEval(float* values, uint64_t* masks, int batch_size,
                   std::vector<float*> output_mems);

 private:
  // Forward computation logic (pure graph construction, no eval/memcpy).
  std::vector<mx::array> ForwardPass(const std::vector<mx::array>& inputs);

  // Compiled version of ForwardPass (set in Build()).
  std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
      compiled_forward_;

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

    // Quantized weight versions (populated when precision is Q8).
    OptQuantized mha_q_w_q, mha_k_w_q, mha_v_w_q, mha_dense_w_q;
    OptQuantized ffn_dense1_w_q, ffn_dense2_w_q;
    OptQuantized smolgen_compress_q;
    OptQuantized smolgen_dense1_w_q;
    OptQuantized smolgen_dense2_w_q;

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

  // Quantized embedding weights (for Q8 precision).
  OptQuantized ip_emb_preproc_w_q_;
  OptQuantized ip_emb_w_q_;
  OptQuantized ip_emb_ffn_dense1_w_q_, ip_emb_ffn_dense2_w_q_;

  // Global smolgen weights.
  OptArray smolgen_global_w_;
  OptQuantized smolgen_global_w_q_;
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

    // Quantized policy weights (for Q8 precision).
    OptQuantized ip_pol_w_q;
    OptQuantized ip2_pol_w_q, ip3_pol_w_q;
    // Note: ip4_pol_w is NOT quantized - used by AttentionPolicyPromoMatmulConcat.
  };
  PolicyWeights policy_weights_;

  // Value head weights.
  struct ValueWeights {
    OptArray value_conv_weights;
    OptArray value_conv_biases;
    OptArray ip_val_w, ip_val_b;
    OptArray ip1_val_w, ip1_val_b;
    OptArray ip2_val_w, ip2_val_b;

    // Quantized value weights (for Q8 precision).
    OptQuantized ip_val_w_q;
    OptQuantized ip1_val_w_q, ip2_val_w_q;
  };
  ValueWeights value_weights_;

  // Moves left head weights.
  OptArray moves_left_conv_weights_;
  OptArray moves_left_conv_biases_;
  OptArray ip_mov_w_, ip_mov_b_;
  OptArray ip1_mov_w_, ip1_mov_b_;
  OptArray ip2_mov_w_, ip2_mov_b_;

  // Quantized moves left weights (for Q8 precision).
  OptQuantized ip_mov_w_q_;
  OptQuantized ip1_mov_w_q_, ip2_mov_w_q_;

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

  // Precision configuration.
  Precision precision_ = Precision::FP32;
  mx::Dtype compute_dtype_ = mx::float32;
  int group_size_ = 64;  // Quantization group size for Q8.

  // Position encoding data.
  std::vector<float> pos_enc_data_;

  // Pre-computed values from Build() for ForwardEval() optimization.
  // Position encoding array for PE_MAP embedding (static, create once).
  OptArray pos_enc_base_;  // Shape: [64, 64] in compute_dtype_

  // Alpha scalar for encoder skip connections: (2 * num_encoders)^(-0.25).
  float encoder_alpha_ = 1.0f;

  // Epsilon value for layer normalization (depends on embedding type).
  float default_epsilon_ = 1e-6f;

  // Policy head dimension and attention scale (for attention policy).
  int pol_dmodel_ = 0;
  float attn_policy_scale_ = 1.0f;

  // Pre-computed bit tensor for ExpandInput (static data, create once).
  OptArray bit_tensor_;  // Shape: [1, 1, 64] uint64

  // Pre-computed encoder MHA scale: 1.0f / sqrt(embedding_size_ / encoder_head_count_).
  float encoder_mha_scale_ = 1.0f;

  // Pre-computed policy encoder MHA scale: 1.0f / sqrt(pol_emb_size / pol_encoder_head_count).
  float policy_mha_scale_ = 1.0f;

  // Pre-computed network structure booleans (fixed at build time).
  bool has_preproc_ = false;
  bool has_emb_ln_ = false;
  bool has_gates_ = false;
  bool has_emb_ffn_ = false;

  // Pre-computed policy activation strings.
  std::string pol_act_;
  std::string pol_ffn_act_;

  // Pre-computed dummy scalar for non-SE residual blocks.
  OptArray dummy_scalar_;

  // Pre-computed encoder alpha as mx::array (avoids creating from float each call).
  OptArray encoder_alpha_array_;

  // Epsilon values for different network sections.
  float pe_dense_epsilon_ = kPeDenseEpsilon;
  float smolgen_epsilon_ = kSmolgenEpsilon;
  float policy_epsilon_ = kDefaultEpsilon;

  // Pre-computed zero uint64 for ExpandInput comparison.
  OptArray zero_uint64_;  // mx::zeros({1}, mx::uint64)

  // Pre-computed attention policy scale as mx::array.
  OptArray attn_policy_scale_array_;

  // Pre-computed policy gather indices for GPU-based policy mapping.
  // Maps from output index (1858) to input index, enabling mx::take gather.
  OptArray policy_gather_indices_;

  // Pre-computed smolgen weight variants for each encoder layer.
  // Avoids runtime variant construction in ForwardEval.
  struct SmolgenVariants {
    WeightVariant compress;
    WeightVariant dense1_w;
    WeightVariant dense2_w;

    SmolgenVariants(WeightVariant c, WeightVariant d1, WeightVariant d2)
        : compress(std::move(c)), dense1_w(std::move(d1)), dense2_w(std::move(d2)) {}
  };
  std::vector<std::optional<SmolgenVariants>> encoder_smolgen_variants_;
  std::optional<WeightVariant> smolgen_global_variant_;
};

void MLXGraphBuilder::Build(int input_planes, MultiHeadWeights& weights,
                            InputEmbedding embedding, bool attn_body,
                            bool attn_policy, bool conv_policy, bool wdl,
                            bool moves_left, Activations& activations,
                            const std::string& policy_head,
                            const std::string& value_head,
                            Precision precision, int group_size) {
  attn_body_ = attn_body;
  attn_policy_ = attn_policy;
  conv_policy_ = conv_policy;
  wdl_ = wdl;
  moves_left_ = moves_left;
  embedding_ = embedding;
  activations_ = activations;
  precision_ = precision;
  group_size_ = group_size;

  // For quantized precision, use float16 for activations and float32 for weight loading.
  // For non-quantized, use the precision-mapped dtype.
  compute_dtype_ = ActivationDtype(precision);
  bool quantize = IsQuantized(precision);

  // Helper to quantize weights with fallback to compute_dtype_.
  // Returns {quantized_opt, float_opt} where exactly one has value.
  auto try_quantize = [quantize, group_size = group_size_,
                       compute_dtype = compute_dtype_](
                          mx::array w, const std::string& name, int in_dim,
                          int out_dim) -> std::pair<OptQuantized, OptArray> {
    if (!quantize) {
      return {std::nullopt, std::move(w)};
    }
    auto q = QuantizeWeights(w, group_size, kDefaultQuantizationBits);
    if (q.has_value()) {
      return {std::move(q), std::nullopt};
    }
    CERR << "Warning: " << name << " [" << in_dim << ", " << out_dim
         << "] not divisible by group_size=" << group_size
         << ", using float16.";
    return {std::nullopt, mx::astype(w, compute_dtype)};
  };

  // Silent variant for encoder loops - doesn't warn (caller handles warning).
  auto try_quantize_silent = [quantize, group_size = group_size_,
                              compute_dtype = compute_dtype_](
                                 mx::array w) -> std::pair<OptQuantized, OptArray> {
    if (!quantize) {
      return {std::nullopt, std::move(w)};
    }
    auto q = QuantizeWeights(w, group_size, kDefaultQuantizationBits);
    if (q.has_value()) {
      return {std::move(q), std::nullopt};
    }
    return {std::nullopt, mx::astype(w, compute_dtype)};
  };

  // Helper to apply silent quantization directly to member variables.
  auto apply_quantize_silent = [&try_quantize_silent](mx::array& w,
                                                      OptQuantized& q_out) {
    auto [q, f] = try_quantize_silent(std::move(w));
    q_out = std::move(q);
    if (f.has_value()) w = std::move(*f);
  };

  // Variant for OptArray output (used by smolgen weights).
  auto apply_quantize_silent_opt = [&try_quantize_silent](
                                       mx::array w, OptQuantized& q_out,
                                       OptArray& f_out) {
    auto [q, f] = try_quantize_silent(std::move(w));
    q_out = std::move(q);
    f_out = std::move(f);
  };

  // Helper to warn about non-quantizable weights.
  auto warn_if_not_quantized = [group_size = group_size_](
                                   const OptQuantized& q, const std::string& name,
                                   int in_dim, int out_dim) {
    if (!q.has_value()) {
      CERR << "Warning: " << name << " [" << in_dim << ", " << out_dim
           << "] not divisible by group_size=" << group_size
           << ", using float16.";
    }
  };

  // Helper to warn once about non-quantizable weights.
  // Executes warn_fn only on first call (when warned is false and quantize is
  // true).
  auto warn_once = [quantize](bool& warned, auto warn_fn) {
    if (quantize && !warned) {
      warn_fn();
      warned = true;
    }
  };

  // Get filter count from input convolution.
  if (!attn_body) {
    num_filters_ = static_cast<int>(weights.input.biases.size());
    // Convert input convolution weights from OIHW to OHWI format.
    input_conv_weights_ =
        ConvertConvWeightsOIHWtoOHWI(weights.input.weights,
                                     num_filters_, input_planes, 3, 3, compute_dtype_);
    input_conv_biases_ =
        MakeArray(weights.input.biases, {num_filters_}, compute_dtype_);

    // Convert residual tower weights.
    residual_weights_.reserve(weights.residual.size());
    for (const auto& res : weights.residual) {
      residual_weights_.emplace_back(
          ConvertConvWeightsOIHWtoOHWI(res.conv1.weights, num_filters_, num_filters_, 3, 3, compute_dtype_),
          MakeArray(res.conv1.biases, {num_filters_}, compute_dtype_),
          ConvertConvWeightsOIHWtoOHWI(res.conv2.weights, num_filters_, num_filters_, 3, 3, compute_dtype_),
          MakeArray(res.conv2.biases, {num_filters_}, compute_dtype_));
      auto& rw = residual_weights_.back();

      rw.has_se = res.has_se;
      if (res.has_se) {
        int se_channels = static_cast<int>(res.se.b1.size());
        // SE FC weights - use BLAS column-major format loader.
        rw.se_fc1_weights = MakeFCWeights(res.se.w1, num_filters_, se_channels, compute_dtype_);
        rw.se_fc1_biases = MakeArray(res.se.b1, {se_channels}, compute_dtype_);
        rw.se_fc2_weights = MakeFCWeights(res.se.w2, se_channels, 2 * num_filters_, compute_dtype_);
        rw.se_fc2_biases = MakeArray(res.se.b2, {2 * num_filters_}, compute_dtype_);
      }
    }
  } else {
    // Attention body - load embedding and encoder weights.
    embedding_size_ = static_cast<int>(weights.ip_emb_b.size());
    encoder_head_count_ = weights.encoder_head_count;

    // Embedding preprocessing (PE_DENSE only).
    if (!weights.ip_emb_preproc_w.empty()) {
      int preproc_out = static_cast<int>(weights.ip_emb_preproc_b.size());
      auto w = MakeFCWeights(weights.ip_emb_preproc_w, 64 * 12, preproc_out,
                             quantize ? mx::float32 : compute_dtype_);
      auto [q, f] = try_quantize(w, "ip_emb_preproc_w", 64 * 12, preproc_out);
      ip_emb_preproc_w_q_ = std::move(q);
      if (f.has_value()) ip_emb_preproc_w_ = std::move(*f);
      ip_emb_preproc_b_ = MakeArray(weights.ip_emb_preproc_b, {preproc_out}, compute_dtype_);
    }

    // Embedding layer.
    int ip_emb_in = static_cast<int>(weights.ip_emb_w.size()) / embedding_size_;
    {
      auto w = MakeFCWeights(weights.ip_emb_w, ip_emb_in, embedding_size_,
                             quantize ? mx::float32 : compute_dtype_);
      auto [q, f] = try_quantize(w, "ip_emb_w", ip_emb_in, embedding_size_);
      ip_emb_w_q_ = std::move(q);
      if (f.has_value()) ip_emb_w_ = std::move(*f);
    }
    ip_emb_b_ = MakeArray(weights.ip_emb_b, {embedding_size_}, compute_dtype_);

    // Embedding layer norm.
    if (!weights.ip_emb_ln_gammas.empty()) {
      ip_emb_ln_gammas_ =
          MakeArray(weights.ip_emb_ln_gammas, {embedding_size_}, compute_dtype_);
      ip_emb_ln_betas_ = MakeArray(weights.ip_emb_ln_betas, {embedding_size_}, compute_dtype_);
    }

    // Input gating.
    // BLAS stores as [embedding_size, 64] in row-major.
    if (!weights.ip_mult_gate.empty()) {
      ip_mult_gate_ = MakeArray(weights.ip_mult_gate, {embedding_size_, 64}, compute_dtype_);
      ip_add_gate_ = MakeArray(weights.ip_add_gate, {embedding_size_, 64}, compute_dtype_);
    }

    // Embedding FFN.
    if (!weights.ip_emb_ffn.dense1_w.empty()) {
      int ffn_hidden = static_cast<int>(weights.ip_emb_ffn.dense1_b.size());
      {
        auto w1 = MakeFCWeights(weights.ip_emb_ffn.dense1_w, embedding_size_, ffn_hidden,
                                quantize ? mx::float32 : compute_dtype_);
        auto [q1, f1] = try_quantize(w1, "ip_emb_ffn_dense1_w", embedding_size_, ffn_hidden);
        ip_emb_ffn_dense1_w_q_ = std::move(q1);
        if (f1.has_value()) ip_emb_ffn_dense1_w_ = std::move(*f1);
      }
      {
        auto w2 = MakeFCWeights(weights.ip_emb_ffn.dense2_w, ffn_hidden, embedding_size_,
                                quantize ? mx::float32 : compute_dtype_);
        auto [q2, f2] = try_quantize(w2, "ip_emb_ffn_dense2_w", ffn_hidden, embedding_size_);
        ip_emb_ffn_dense2_w_q_ = std::move(q2);
        if (f2.has_value()) ip_emb_ffn_dense2_w_ = std::move(*f2);
      }
      ip_emb_ffn_dense1_b_ = MakeArray(weights.ip_emb_ffn.dense1_b, {ffn_hidden}, compute_dtype_);
      ip_emb_ffn_dense2_b_ = MakeArray(weights.ip_emb_ffn.dense2_b, {embedding_size_}, compute_dtype_);
      ip_emb_ffn_ln_gammas_ = MakeArray(weights.ip_emb_ffn_ln_gammas, {embedding_size_}, compute_dtype_);
      ip_emb_ffn_ln_betas_ = MakeArray(weights.ip_emb_ffn_ln_betas, {embedding_size_}, compute_dtype_);
    }

    // Global smolgen weights.
    has_smolgen_ = weights.has_smolgen;
    if (has_smolgen_) {
      int smolgen_out = static_cast<int>(weights.smolgen_w.size()) / 64 / 64;
      auto w = MakeFCWeights(weights.smolgen_w, smolgen_out, 64 * 64,
                             quantize ? mx::float32 : compute_dtype_);
      auto [q, f] = try_quantize(w, "smolgen_global_w", smolgen_out, 64 * 64);
      smolgen_global_w_q_ = std::move(q);
      if (f.has_value()) smolgen_global_w_ = std::move(*f);
    }

    // Encoder layers - use MakeFCWeights for all FC weight matrices.
    encoder_weights_.reserve(weights.encoder.size());
    bool smolgen_quant_warned = false;  // Only warn once for smolgen quantization failures.
    bool encoder_quant_warned = false;  // Only warn once for encoder layer quantization failures.
    for (const auto& enc : weights.encoder) {
      int qkv_size = static_cast<int>(enc.mha.q_b.size());
      int ffn_hidden = static_cast<int>(enc.ffn.dense1_b.size());

      // Load weights in float32 for quantization, then convert/quantize.
      encoder_weights_.emplace_back(
          MakeFCWeights(enc.mha.q_w, embedding_size_, qkv_size, mx::float32),
          MakeArray(enc.mha.q_b, {qkv_size}, compute_dtype_),
          MakeFCWeights(enc.mha.k_w, embedding_size_, qkv_size, mx::float32),
          MakeArray(enc.mha.k_b, {qkv_size}, compute_dtype_),
          MakeFCWeights(enc.mha.v_w, embedding_size_, qkv_size, mx::float32),
          MakeArray(enc.mha.v_b, {qkv_size}, compute_dtype_),
          MakeFCWeights(enc.mha.dense_w, qkv_size, embedding_size_, mx::float32),
          MakeArray(enc.mha.dense_b, {embedding_size_}, compute_dtype_),
          MakeArray(enc.ln1_gammas, {embedding_size_}, compute_dtype_),
          MakeArray(enc.ln1_betas, {embedding_size_}, compute_dtype_),
          MakeFCWeights(enc.ffn.dense1_w, embedding_size_, ffn_hidden, mx::float32),
          MakeArray(enc.ffn.dense1_b, {ffn_hidden}, compute_dtype_),
          MakeFCWeights(enc.ffn.dense2_w, ffn_hidden, embedding_size_, mx::float32),
          MakeArray(enc.ffn.dense2_b, {embedding_size_}, compute_dtype_),
          MakeArray(enc.ln2_gammas, {embedding_size_}, compute_dtype_),
          MakeArray(enc.ln2_betas, {embedding_size_}, compute_dtype_));
      auto& ew = encoder_weights_.back();

      // Quantize encoder FC weights if requested, convert to compute_dtype_ if quantization fails.
      // Use silent helper, then warn once per layer type.
      apply_quantize_silent(ew.mha_q_w, ew.mha_q_w_q);
      apply_quantize_silent(ew.mha_k_w, ew.mha_k_w_q);
      apply_quantize_silent(ew.mha_v_w, ew.mha_v_w_q);
      apply_quantize_silent(ew.mha_dense_w, ew.mha_dense_w_q);
      apply_quantize_silent(ew.ffn_dense1_w, ew.ffn_dense1_w_q);
      apply_quantize_silent(ew.ffn_dense2_w, ew.ffn_dense2_w_q);

      // Warn once if any encoder weight can't be quantized.
      warn_once(encoder_quant_warned, [&] {
        warn_if_not_quantized(ew.mha_q_w_q, "encoder mha_q_w", embedding_size_, qkv_size);
        warn_if_not_quantized(ew.mha_k_w_q, "encoder mha_k_w", embedding_size_, qkv_size);
        warn_if_not_quantized(ew.mha_v_w_q, "encoder mha_v_w", embedding_size_, qkv_size);
        warn_if_not_quantized(ew.mha_dense_w_q, "encoder mha_dense_w", qkv_size, embedding_size_);
        warn_if_not_quantized(ew.ffn_dense1_w_q, "encoder ffn_dense1_w", embedding_size_, ffn_hidden);
        warn_if_not_quantized(ew.ffn_dense2_w_q, "encoder ffn_dense2_w", ffn_hidden, embedding_size_);
      });

      ew.has_smolgen = enc.mha.has_smolgen;
      if (enc.mha.has_smolgen) {
        int hidden =
            static_cast<int>(enc.mha.smolgen.compress.size()) / embedding_size_;
        int dense1_out = static_cast<int>(enc.mha.smolgen.dense1_b.size());
        int dense2_out = static_cast<int>(enc.mha.smolgen.dense2_b.size());

        apply_quantize_silent_opt(
            MakeFCWeights(enc.mha.smolgen.compress, embedding_size_, hidden, mx::float32),
            ew.smolgen_compress_q, ew.smolgen_compress);
        apply_quantize_silent_opt(
            MakeFCWeights(enc.mha.smolgen.dense1_w, 64 * hidden, dense1_out, mx::float32),
            ew.smolgen_dense1_w_q, ew.smolgen_dense1_w);
        apply_quantize_silent_opt(
            MakeFCWeights(enc.mha.smolgen.dense2_w, dense1_out, dense2_out, mx::float32),
            ew.smolgen_dense2_w_q, ew.smolgen_dense2_w);

        // Warn once if any smolgen weight can't be quantized.
        warn_once(smolgen_quant_warned, [&] {
          warn_if_not_quantized(ew.smolgen_compress_q, "smolgen_compress", embedding_size_, hidden);
          warn_if_not_quantized(ew.smolgen_dense1_w_q, "smolgen_dense1_w", 64 * hidden, dense1_out);
          warn_if_not_quantized(ew.smolgen_dense2_w_q, "smolgen_dense2_w", dense1_out, dense2_out);
        });

        ew.smolgen_dense1_b = MakeArray(enc.mha.smolgen.dense1_b, {dense1_out}, compute_dtype_);
        ew.smolgen_ln1_gammas = MakeArray(enc.mha.smolgen.ln1_gammas, {dense1_out}, compute_dtype_);
        ew.smolgen_ln1_betas = MakeArray(enc.mha.smolgen.ln1_betas, {dense1_out}, compute_dtype_);
        ew.smolgen_dense2_b = MakeArray(enc.mha.smolgen.dense2_b, {dense2_out}, compute_dtype_);
        ew.smolgen_ln2_gammas = MakeArray(enc.mha.smolgen.ln2_gammas, {dense2_out}, compute_dtype_);
        ew.smolgen_ln2_betas = MakeArray(enc.mha.smolgen.ln2_betas, {dense2_out}, compute_dtype_);
      }
    }
  }

  // Policy head weights.
  // BLAS stores FC weights in column-major layout.
  auto& pol_head = weights.policy_heads.at(policy_head);
  if (attn_policy) {
    int pol_emb_size = static_cast<int>(pol_head.ip_pol_b.size());
    {
      auto w = MakeFCWeights(pol_head.ip_pol_w, embedding_size_, pol_emb_size,
                             quantize ? mx::float32 : compute_dtype_);
      auto [q, f] = try_quantize(w, "ip_pol_w", embedding_size_, pol_emb_size);
      policy_weights_.ip_pol_w_q = std::move(q);
      if (f.has_value()) policy_weights_.ip_pol_w = std::move(*f);
    }
    policy_weights_.ip_pol_b = MakeArray(pol_head.ip_pol_b, {pol_emb_size}, compute_dtype_);

    int pol_dmodel = static_cast<int>(pol_head.ip2_pol_b.size());
    {
      auto w2 = MakeFCWeights(pol_head.ip2_pol_w, pol_emb_size, pol_dmodel,
                              quantize ? mx::float32 : compute_dtype_);
      auto w3 = MakeFCWeights(pol_head.ip3_pol_w, pol_emb_size, pol_dmodel,
                              quantize ? mx::float32 : compute_dtype_);
      // ip4_pol_w is NOT quantized - it's used by AttentionPolicyPromoMatmulConcat
      // which doesn't support quantized weights.
      auto w4 = MakeFCWeights(pol_head.ip4_pol_w, pol_dmodel, 4, compute_dtype_);

      auto [q2, f2] = try_quantize(w2, "ip2_pol_w", pol_emb_size, pol_dmodel);
      auto [q3, f3] = try_quantize(w3, "ip3_pol_w", pol_emb_size, pol_dmodel);
      policy_weights_.ip2_pol_w_q = std::move(q2);
      policy_weights_.ip3_pol_w_q = std::move(q3);
      if (f2.has_value()) policy_weights_.ip2_pol_w = std::move(*f2);
      if (f3.has_value()) policy_weights_.ip3_pol_w = std::move(*f3);
      policy_weights_.ip4_pol_w = std::move(w4);
    }
    policy_weights_.ip2_pol_b = MakeArray(pol_head.ip2_pol_b, {pol_dmodel}, compute_dtype_);
    policy_weights_.ip3_pol_b = MakeArray(pol_head.ip3_pol_b, {pol_dmodel}, compute_dtype_);
    policy_weights_.pol_encoder_head_count = pol_head.pol_encoder_head_count;

    // Policy encoder layers.
    policy_weights_.pol_encoder.reserve(pol_head.pol_encoder.size());
    bool pol_encoder_quant_warned = false;  // Only warn once for policy encoder quantization failures.
    for (const auto& enc : pol_head.pol_encoder) {
      int qkv_size = static_cast<int>(enc.mha.q_b.size());
      int ffn_hidden = static_cast<int>(enc.ffn.dense1_b.size());

      policy_weights_.pol_encoder.emplace_back(
          MakeFCWeights(enc.mha.q_w, pol_emb_size, qkv_size, mx::float32),
          MakeArray(enc.mha.q_b, {qkv_size}, compute_dtype_),
          MakeFCWeights(enc.mha.k_w, pol_emb_size, qkv_size, mx::float32),
          MakeArray(enc.mha.k_b, {qkv_size}, compute_dtype_),
          MakeFCWeights(enc.mha.v_w, pol_emb_size, qkv_size, mx::float32),
          MakeArray(enc.mha.v_b, {qkv_size}, compute_dtype_),
          MakeFCWeights(enc.mha.dense_w, qkv_size, pol_emb_size, mx::float32),
          MakeArray(enc.mha.dense_b, {pol_emb_size}, compute_dtype_),
          MakeArray(enc.ln1_gammas, {pol_emb_size}, compute_dtype_),
          MakeArray(enc.ln1_betas, {pol_emb_size}, compute_dtype_),
          MakeFCWeights(enc.ffn.dense1_w, pol_emb_size, ffn_hidden, mx::float32),
          MakeArray(enc.ffn.dense1_b, {ffn_hidden}, compute_dtype_),
          MakeFCWeights(enc.ffn.dense2_w, ffn_hidden, pol_emb_size, mx::float32),
          MakeArray(enc.ffn.dense2_b, {pol_emb_size}, compute_dtype_),
          MakeArray(enc.ln2_gammas, {pol_emb_size}, compute_dtype_),
          MakeArray(enc.ln2_betas, {pol_emb_size}, compute_dtype_));
      auto& pew = policy_weights_.pol_encoder.back();
      pew.has_smolgen = enc.mha.has_smolgen;

      // Quantize policy encoder FC weights if requested.
      // Use silent helper, then warn once per layer type.
      apply_quantize_silent(pew.mha_q_w, pew.mha_q_w_q);
      apply_quantize_silent(pew.mha_k_w, pew.mha_k_w_q);
      apply_quantize_silent(pew.mha_v_w, pew.mha_v_w_q);
      apply_quantize_silent(pew.mha_dense_w, pew.mha_dense_w_q);
      apply_quantize_silent(pew.ffn_dense1_w, pew.ffn_dense1_w_q);
      apply_quantize_silent(pew.ffn_dense2_w, pew.ffn_dense2_w_q);

      // Warn once if any policy encoder weight can't be quantized.
      warn_once(pol_encoder_quant_warned, [&] {
        warn_if_not_quantized(pew.mha_q_w_q, "pol_encoder mha_q_w", pol_emb_size, qkv_size);
        warn_if_not_quantized(pew.mha_k_w_q, "pol_encoder mha_k_w", pol_emb_size, qkv_size);
        warn_if_not_quantized(pew.mha_v_w_q, "pol_encoder mha_v_w", pol_emb_size, qkv_size);
        warn_if_not_quantized(pew.mha_dense_w_q, "pol_encoder mha_dense_w", qkv_size, pol_emb_size);
        warn_if_not_quantized(pew.ffn_dense1_w_q, "pol_encoder ffn_dense1_w", pol_emb_size, ffn_hidden);
        warn_if_not_quantized(pew.ffn_dense2_w_q, "pol_encoder ffn_dense2_w", ffn_hidden, pol_emb_size);
      });
    }
  } else if (conv_policy) {
    int pol1_channels = static_cast<int>(pol_head.policy1.biases.size());
    policy_weights_.policy1_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(pol_head.policy1.weights, pol1_channels, num_filters_, 3, 3, compute_dtype_);
    policy_weights_.policy1_conv_biases =
        MakeArray(pol_head.policy1.biases, {pol1_channels}, compute_dtype_);
    int pol_channels = static_cast<int>(pol_head.policy.biases.size());
    policy_weights_.policy_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(pol_head.policy.weights, pol_channels, pol1_channels, 3, 3, compute_dtype_);
    policy_weights_.policy_conv_biases =
        MakeArray(pol_head.policy.biases, {pol_channels}, compute_dtype_);
  } else {
    // Classical policy.
    int pol_channels = static_cast<int>(pol_head.policy.biases.size());
    policy_weights_.policy_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(pol_head.policy.weights, pol_channels, num_filters_, 1, 1, compute_dtype_);
    policy_weights_.policy_conv_biases =
        MakeArray(pol_head.policy.biases, {pol_channels}, compute_dtype_);
    int pol_outputs = static_cast<int>(pol_head.ip_pol_b.size());
    policy_weights_.ip_pol_w = MakeFCWeights(pol_head.ip_pol_w, pol_channels * 64, pol_outputs, compute_dtype_);
    policy_weights_.ip_pol_b = MakeArray(pol_head.ip_pol_b, {pol_outputs}, compute_dtype_);
  }

  // Value head weights.
  // BLAS stores FC weights in column-major layout.
  auto& val_head = weights.value_heads.at(value_head);
  if (attn_body) {
    int val_emb_size = static_cast<int>(val_head.ip_val_b.size());
    {
      auto w = MakeFCWeights(val_head.ip_val_w, embedding_size_, val_emb_size,
                             quantize ? mx::float32 : compute_dtype_);
      auto [q, f] = try_quantize(w, "ip_val_w", embedding_size_, val_emb_size);
      value_weights_.ip_val_w_q = std::move(q);
      if (f.has_value()) value_weights_.ip_val_w = std::move(*f);
    }
    value_weights_.ip_val_b = MakeArray(val_head.ip_val_b, {val_emb_size}, compute_dtype_);
  } else {
    int val_channels = static_cast<int>(val_head.value.biases.size());
    value_weights_.value_conv_weights =
        ConvertConvWeightsOIHWtoOHWI(val_head.value.weights, val_channels, num_filters_, 1, 1, compute_dtype_);
    value_weights_.value_conv_biases =
        MakeArray(val_head.value.biases, {val_channels}, compute_dtype_);
  }
  int val_hidden = static_cast<int>(val_head.ip1_val_b.size());
  int ip1_val_in = static_cast<int>(val_head.ip1_val_w.size()) / val_hidden;
  int val_outputs = static_cast<int>(val_head.ip2_val_b.size());
  {
    auto w1 = MakeFCWeights(val_head.ip1_val_w, ip1_val_in, val_hidden,
                            quantize ? mx::float32 : compute_dtype_);
    auto w2 = MakeFCWeights(val_head.ip2_val_w, val_hidden, val_outputs,
                            quantize ? mx::float32 : compute_dtype_);
    auto [q1, f1] = try_quantize(w1, "ip1_val_w", ip1_val_in, val_hidden);
    auto [q2, f2] = try_quantize(w2, "ip2_val_w", val_hidden, val_outputs);
    value_weights_.ip1_val_w_q = std::move(q1);
    value_weights_.ip2_val_w_q = std::move(q2);
    if (f1.has_value()) value_weights_.ip1_val_w = std::move(*f1);
    if (f2.has_value()) value_weights_.ip2_val_w = std::move(*f2);
  }
  value_weights_.ip1_val_b = MakeArray(val_head.ip1_val_b, {val_hidden}, compute_dtype_);
  value_weights_.ip2_val_b = MakeArray(val_head.ip2_val_b, {val_outputs}, compute_dtype_);

  // Moves left head weights.
  // BLAS stores FC weights in column-major layout.
  if (moves_left) {
    if (attn_body) {
      int mov_emb_size = static_cast<int>(weights.ip_mov_b.size());
      {
        auto w = MakeFCWeights(weights.ip_mov_w, embedding_size_, mov_emb_size,
                               quantize ? mx::float32 : compute_dtype_);
        auto [q, f] = try_quantize(w, "ip_mov_w", embedding_size_, mov_emb_size);
        ip_mov_w_q_ = std::move(q);
        if (f.has_value()) ip_mov_w_ = std::move(*f);
      }
      ip_mov_b_ = MakeArray(weights.ip_mov_b, {mov_emb_size}, compute_dtype_);
    } else {
      int mov_channels = static_cast<int>(weights.moves_left.biases.size());
      moves_left_conv_weights_ =
          ConvertConvWeightsOIHWtoOHWI(weights.moves_left.weights, mov_channels, num_filters_, 1, 1, compute_dtype_);
      moves_left_conv_biases_ =
          MakeArray(weights.moves_left.biases, {mov_channels}, compute_dtype_);
    }
    int mov_hidden = static_cast<int>(weights.ip1_mov_b.size());
    int ip1_mov_in = static_cast<int>(weights.ip1_mov_w.size()) / mov_hidden;
    {
      auto w1 = MakeFCWeights(weights.ip1_mov_w, ip1_mov_in, mov_hidden,
                              quantize ? mx::float32 : compute_dtype_);
      auto w2 = MakeFCWeights(weights.ip2_mov_w, mov_hidden, 1,
                              quantize ? mx::float32 : compute_dtype_);
      auto [q1, f1] = try_quantize(w1, "ip1_mov_w", ip1_mov_in, mov_hidden);
      auto [q2, f2] = try_quantize(w2, "ip2_mov_w", mov_hidden, 1);
      ip1_mov_w_q_ = std::move(q1);
      ip2_mov_w_q_ = std::move(q2);
      if (f1.has_value()) ip1_mov_w_ = std::move(*f1);
      if (f2.has_value()) ip2_mov_w_ = std::move(*f2);
    }
    ip1_mov_b_ = MakeArray(weights.ip1_mov_b, {mov_hidden}, compute_dtype_);
    ip2_mov_b_ = MakeArray(weights.ip2_mov_b, {1}, compute_dtype_);
  }

  // === Pre-compute values for ForwardEval optimization ===

  // 1. Position encoding array for PE_MAP (static data, create once).
  if (embedding_ == INPUT_EMBEDDING_PE_MAP) {
    mx::array pos_enc = mx::array(
        reinterpret_cast<const float*>(kPosEncoding),
        mx::Shape{64, kNumPosEncodingChannels}, mx::float32);
    if (compute_dtype_ != mx::float32) {
      pos_enc = mx::astype(pos_enc, compute_dtype_);
    }
    pos_enc_base_ = std::move(pos_enc);
  }

  // 2. Encoder alpha scalar for skip connections.
  if (!encoder_weights_.empty()) {
    encoder_alpha_ = static_cast<float>(
        std::pow(2.0 * encoder_weights_.size(), -0.25));
  }

  // 3. Default epsilon based on embedding type.
  default_epsilon_ = (embedding_ == INPUT_EMBEDDING_PE_DENSE)
                         ? kPeDenseEpsilon : kDefaultEpsilon;

  // 4. Policy head dimensions and scale (for attention policy).
  if (attn_policy_ && policy_weights_.ip2_pol_b.has_value()) {
    pol_dmodel_ = static_cast<int>(policy_weights_.ip2_pol_b->size());
    attn_policy_scale_ = 1.0f / std::sqrt(static_cast<float>(pol_dmodel_));
  }

  // 6. Pre-compute bit tensor for ExpandInput.
  {
    std::vector<uint64_t> bit_indices(64);
    for (int i = 0; i < 64; i++) {
      bit_indices[i] = 1ULL << i;
    }
    bit_tensor_ = mx::array(bit_indices.data(), mx::Shape{1, 1, 64}, mx::uint64);
  }

  // 7. Pre-transpose gating weights for element-wise operations.
  if (ip_mult_gate_.has_value()) {
    ip_mult_gate_ = mx::transpose(*ip_mult_gate_);
    ip_add_gate_ = mx::transpose(*ip_add_gate_);
  }

  // 8. Pre-compute encoder MHA scale.
  if (encoder_head_count_ > 0 && embedding_size_ > 0) {
    int depth = embedding_size_ / encoder_head_count_;
    encoder_mha_scale_ = 1.0f / std::sqrt(static_cast<float>(depth));
  }

  // 9. Pre-compute policy encoder MHA scale.
  if (attn_policy_ && policy_weights_.pol_encoder_head_count > 0 &&
      policy_weights_.ip_pol_b.has_value()) {
    int pol_emb_size = static_cast<int>(policy_weights_.ip_pol_b->size());
    int pol_depth = pol_emb_size / policy_weights_.pol_encoder_head_count;
    policy_mha_scale_ = 1.0f / std::sqrt(static_cast<float>(pol_depth));
  }

  // 10. Pre-compute network structure booleans.
  has_preproc_ = ip_emb_preproc_w_.has_value() || ip_emb_preproc_w_q_.has_value();
  has_emb_ln_ = ip_emb_ln_gammas_.has_value();
  has_gates_ = ip_mult_gate_.has_value();
  has_emb_ffn_ = ip_emb_ffn_dense1_w_.has_value() || ip_emb_ffn_dense1_w_q_.has_value();

  // 11. Pre-compute policy activation strings.
  pol_act_ = attn_body_ ? activations_.default_activation : "selu";
  pol_ffn_act_ = attn_body_ ? activations_.ffn_activation : "selu";

  // 12. Pre-compute dummy scalar for non-SE residual blocks.
  dummy_scalar_ = mx::array(0.0f);

  // 13. Pre-compute encoder alpha as mx::array.
  encoder_alpha_array_ = mx::array(encoder_alpha_);

  // 14. Epsilon values are now plain floats (used directly by mx::fast::layer_norm).

  // 15. Pre-compute zero uint64 for ExpandInput.
  zero_uint64_ = mx::zeros({1}, mx::uint64);

  // 16. Pre-compute attention policy scale as mx::array.
  attn_policy_scale_array_ = mx::array(attn_policy_scale_);

  // 17. Pre-compute policy gather indices for GPU-based policy mapping.
  // Inverts the scatter map (input_idx → output_idx) into a gather map
  // (output_idx → input_idx) so we can use mx::take inside the compiled graph.
  if (attn_policy_) {
    constexpr size_t kAttnPolicySize = 64 * 64 + 8 * 24;  // 4288
    // Default index = kAttnPolicySize (padding slot, will be zero).
    std::vector<int32_t> gather(kNumOutputPolicy,
                                static_cast<int32_t>(kAttnPolicySize));
    for (size_t i = 0; i < kAttnPolicySize; i++) {
      short j = kAttnPolicyMap[i];
      if (j >= 0 && static_cast<size_t>(j) < kNumOutputPolicy) {
        gather[j] = static_cast<int32_t>(i);
      }
    }
    policy_gather_indices_ = mx::array(
        gather.data(), {static_cast<int>(kNumOutputPolicy)}, mx::int32);
  } else if (conv_policy_) {
    constexpr size_t kConvPolicySize = 73 * 8 * 8;  // 4672
    std::vector<int32_t> gather(kNumOutputPolicy,
                                static_cast<int32_t>(kConvPolicySize));
    for (size_t i = 0; i < kConvPolicySize; i++) {
      short j = kConvPolicyMap[i];
      if (j >= 0 && static_cast<size_t>(j) < kNumOutputPolicy) {
        gather[j] = static_cast<int32_t>(i);
      }
    }
    policy_gather_indices_ = mx::array(
        gather.data(), {static_cast<int>(kNumOutputPolicy)}, mx::int32);
  }

  // 5. Pre-compute smolgen weight variants for encoder layers.
  bool has_smolgen_weights = smolgen_global_w_.has_value() || smolgen_global_w_q_.has_value();
  if (has_smolgen_weights) {
    auto make_variant = [](const OptQuantized& q,
                           const OptArray& f) -> WeightVariant {
      if (q.has_value()) return *q;
      assert(f.has_value());
      return std::cref(*f);
    };
    smolgen_global_variant_ = make_variant(smolgen_global_w_q_, smolgen_global_w_);

    encoder_smolgen_variants_.resize(encoder_weights_.size());
    for (size_t i = 0; i < encoder_weights_.size(); ++i) {
      const auto& ew = encoder_weights_[i];
      if (ew.has_smolgen) {
        bool use_quantized = ew.smolgen_compress_q.has_value() ||
                             ew.smolgen_dense1_w_q.has_value() ||
                             ew.smolgen_dense2_w_q.has_value() ||
                             smolgen_global_w_q_.has_value();
        if (use_quantized) {
          encoder_smolgen_variants_[i] = SmolgenVariants(
              make_variant(ew.smolgen_compress_q, ew.smolgen_compress),
              make_variant(ew.smolgen_dense1_w_q, ew.smolgen_dense1_w),
              make_variant(ew.smolgen_dense2_w_q, ew.smolgen_dense2_w));
        }
      }
    }
  }

  // 18. Compile the forward pass for graph caching and kernel fusion.
  compiled_forward_ = mx::compile(
      std::function<std::vector<mx::array>(const std::vector<mx::array>&)>(
          [this](const std::vector<mx::array>& inputs) {
            return ForwardPass(inputs);
          }));
}

std::vector<mx::array> MLXGraphBuilder::ForwardPass(
    const std::vector<mx::array>& inputs) {
  mx::array input_vals = inputs[0];   // [batch, 112] float32
  mx::array input_masks = inputs[1];  // [batch, 112] uint64
  int batch_size = input_vals.shape(0);

  // Expand input to [batch, 112, 8, 8].
  mx::array flow = ExpandInput(input_masks, input_vals, batch_size, *bit_tensor_, *zero_uint64_);

  // Convert to compute dtype if not float32.
  if (compute_dtype_ != mx::float32) {
    flow = mx::astype(flow, compute_dtype_);
  }

  if (!attn_body_) {
    // Classical/SE network: input convolution.
    flow = ConvBlock(flow, *input_conv_weights_, *input_conv_biases_, 3,
                     activations_.default_activation);

    // Residual tower.
    for (const auto& rw : residual_weights_) {
      if (rw.has_se) {
        flow = ResidualBlock(flow, rw.conv1_weights, rw.conv1_biases,
                             rw.conv2_weights, rw.conv2_biases, true,
                             *rw.se_fc1_weights, *rw.se_fc1_biases,
                             *rw.se_fc2_weights, *rw.se_fc2_biases,
                             activations_.default_activation);
      } else {
        flow = ResidualBlock(flow, rw.conv1_weights, rw.conv1_biases,
                             rw.conv2_weights, rw.conv2_biases, false,
                             *dummy_scalar_, *dummy_scalar_,
                             *dummy_scalar_, *dummy_scalar_,
                             activations_.default_activation);
      }
    }
  } else {
    // Attention body network.
    // Reshape from NCHW [batch, 112, 8, 8] to NHWC [batch, 64, 112].
    flow = mx::transpose(flow, {0, 2, 3, 1});
    flow = mx::reshape(flow, {batch_size, 64, kInputPlanes});

    // Handle position encoding based on embedding type.
    if (embedding_ == INPUT_EMBEDDING_PE_DENSE && has_preproc_) {
      // PE_DENSE: Take first 12 channels, flatten, FC, reshape, concat.
      mx::array input_12 = mx::slice(flow, {0, 0, 0}, {batch_size, 64, 12});
      input_12 = mx::reshape(input_12, {batch_size, 64 * 12});
      mx::array pos_enc = DispatchFC(input_12, ip_emb_preproc_w_q_,
                                     ip_emb_preproc_w_, *ip_emb_preproc_b_, "");
      // Use -1 to let MLX infer enc_channels dimension.
      pos_enc = mx::reshape(pos_enc, {batch_size, 64, -1});
      flow = mx::concatenate({flow, pos_enc}, 2);
    } else if (embedding_ == INPUT_EMBEDDING_PE_MAP) {
      // PE_MAP: Concat static position encoding (64 channels per square).
      mx::array pos_enc = mx::broadcast_to(*pos_enc_base_,
                                           {batch_size, 64, kNumPosEncodingChannels});
      flow = mx::concatenate({flow, pos_enc}, 2);
    }

    // Main embedding.
    if (!ip_emb_w_q_.has_value() && !ip_emb_w_.has_value()) {
      throw Exception("Neither ip_emb_w_ nor ip_emb_w_q_ has value!");
    }
    flow = DispatchFC(flow, ip_emb_w_q_, ip_emb_w_, *ip_emb_b_,
                      activations_.default_activation);

    // Embedding layer norm (for PE_DENSE).
    if (has_emb_ln_) {
      flow = LayerNorm(flow, *ip_emb_ln_gammas_, *ip_emb_ln_betas_, pe_dense_epsilon_);
    }

    // Input gating (weights pre-transposed in Build()).
    if (has_gates_) {
      flow = mx::multiply(flow, *ip_mult_gate_);
      flow = mx::add(flow, *ip_add_gate_);
    }

    // Embedding FFN (for PE_DENSE).
    if (has_emb_ffn_) {
      mx::array ffn = DispatchFC(flow, ip_emb_ffn_dense1_w_q_, ip_emb_ffn_dense1_w_,
                                 *ip_emb_ffn_dense1_b_, activations_.ffn_activation);
      ffn = DispatchFC(ffn, ip_emb_ffn_dense2_w_q_, ip_emb_ffn_dense2_w_,
                       *ip_emb_ffn_dense2_b_, "");
      flow = LayerNormWithSkip(flow, ffn, *ip_emb_ffn_ln_gammas_,
                               *ip_emb_ffn_ln_betas_, *encoder_alpha_array_, pe_dense_epsilon_);
    }

    // Encoder layers.
    for (size_t i = 0; i < encoder_weights_.size(); i++) {
      const auto& ew = encoder_weights_[i];

      mx::array q = DispatchFC(flow, ew.mha_q_w_q, ew.mha_q_w, ew.mha_q_b, "");
      mx::array k = DispatchFC(flow, ew.mha_k_w_q, ew.mha_k_w, ew.mha_k_b, "");
      mx::array v = DispatchFC(flow, ew.mha_v_w_q, ew.mha_v_w, ew.mha_v_b, "");

      std::optional<mx::array> smolgen_attn;
      if (i < encoder_smolgen_variants_.size() && encoder_smolgen_variants_[i].has_value()) {
        const auto& sv = *encoder_smolgen_variants_[i];
        smolgen_attn = ComputeSmolgenQuantized(
            flow, encoder_head_count_,
            sv.compress, sv.dense1_w, *ew.smolgen_dense1_b,
            *ew.smolgen_ln1_gammas, *ew.smolgen_ln1_betas,
            sv.dense2_w, *ew.smolgen_dense2_b,
            *ew.smolgen_ln2_gammas, *ew.smolgen_ln2_betas,
            *smolgen_global_variant_, activations_.smolgen_activation,
            smolgen_epsilon_);
      } else if (ew.has_smolgen && smolgen_global_w_.has_value()) {
        smolgen_attn = ComputeSmolgen(
            flow, encoder_head_count_,
            *ew.smolgen_compress,
            *ew.smolgen_dense1_w, *ew.smolgen_dense1_b,
            *ew.smolgen_ln1_gammas, *ew.smolgen_ln1_betas,
            *ew.smolgen_dense2_w, *ew.smolgen_dense2_b,
            *ew.smolgen_ln2_gammas, *ew.smolgen_ln2_betas,
            *smolgen_global_w_,
            activations_.smolgen_activation,
            smolgen_epsilon_);
      }

      mx::array mha = smolgen_attn.has_value()
          ? MultiHeadAttention(q, k, v, encoder_head_count_, encoder_mha_scale_, &*smolgen_attn)
          : MultiHeadAttention(q, k, v, encoder_head_count_, encoder_mha_scale_);

      mha = DispatchFC(mha, ew.mha_dense_w_q, ew.mha_dense_w, ew.mha_dense_b, "");

      flow = LayerNormWithSkip(flow, mha, ew.ln1_gammas, ew.ln1_betas, *encoder_alpha_array_, default_epsilon_);

      mx::array ffn = DispatchFC(flow, ew.ffn_dense1_w_q, ew.ffn_dense1_w,
                                 ew.ffn_dense1_b, activations_.ffn_activation);
      ffn = DispatchFC(ffn, ew.ffn_dense2_w_q, ew.ffn_dense2_w,
                       ew.ffn_dense2_b, "");

      flow = LayerNormWithSkip(flow, ffn, ew.ln2_gammas, ew.ln2_betas, *encoder_alpha_array_, default_epsilon_);
    }
  }

  // Policy head.
  mx::array policy = [&]() -> mx::array {
    if (attn_policy_) {
      mx::array pol = flow;
      pol = DispatchFC(pol, policy_weights_.ip_pol_w_q, policy_weights_.ip_pol_w,
                       *policy_weights_.ip_pol_b, pol_act_);

      for (size_t i = 0; i < policy_weights_.pol_encoder.size(); i++) {
        const auto& ew = policy_weights_.pol_encoder[i];

        mx::array q = DispatchFC(pol, ew.mha_q_w_q, ew.mha_q_w, ew.mha_q_b, "");
        mx::array k = DispatchFC(pol, ew.mha_k_w_q, ew.mha_k_w, ew.mha_k_b, "");
        mx::array v = DispatchFC(pol, ew.mha_v_w_q, ew.mha_v_w, ew.mha_v_b, "");

        mx::array mha = MultiHeadAttention(
            q, k, v, policy_weights_.pol_encoder_head_count, policy_mha_scale_);
        mha = DispatchFC(mha, ew.mha_dense_w_q, ew.mha_dense_w, ew.mha_dense_b, "");

        pol = LayerNormWithSkip(pol, mha, ew.ln1_gammas, ew.ln1_betas, 1.0f, policy_epsilon_);

        mx::array ffn = DispatchFC(pol, ew.ffn_dense1_w_q, ew.ffn_dense1_w,
                                   ew.ffn_dense1_b, pol_ffn_act_);
        ffn = DispatchFC(ffn, ew.ffn_dense2_w_q, ew.ffn_dense2_w,
                         ew.ffn_dense2_b, "");

        pol = LayerNormWithSkip(pol, ffn, ew.ln2_gammas, ew.ln2_betas, 1.0f, policy_epsilon_);
      }

      mx::array queries = DispatchFC(pol, policy_weights_.ip2_pol_w_q,
                                     policy_weights_.ip2_pol_w,
                                     *policy_weights_.ip2_pol_b, "");
      mx::array keys = DispatchFC(pol, policy_weights_.ip3_pol_w_q,
                                  policy_weights_.ip3_pol_w,
                                  *policy_weights_.ip3_pol_b, "");

      pol = ScaledQKMatmul(queries, keys, *attn_policy_scale_array_);
      pol = AttentionPolicyPromoMatmulConcat(pol, keys, *policy_weights_.ip4_pol_w,
                                             56, pol_dmodel_);
      return mx::reshape(pol, {batch_size, -1});
    } else if (conv_policy_) {
      mx::array pol = ConvBlock(flow, *policy_weights_.policy1_conv_weights,
                                *policy_weights_.policy1_conv_biases, 3,
                                activations_.default_activation);
      pol = ConvBlock(pol, *policy_weights_.policy_conv_weights,
                      *policy_weights_.policy_conv_biases, 3, "");
      return mx::reshape(pol, {batch_size, -1});
    } else {
      mx::array pol = ConvBlock(flow, *policy_weights_.policy_conv_weights,
                                *policy_weights_.policy_conv_biases, 1,
                                activations_.default_activation);
      pol = mx::reshape(pol, {batch_size, -1});
      return FullyConnected(pol, *policy_weights_.ip_pol_w,
                            *policy_weights_.ip_pol_b, "");
    }
  }();

  // Apply GPU-based policy mapping via gather (replaces CPU ApplyPolicyMap).
  if (policy_gather_indices_) {
    // Broadcast 1D indices [1858] to 2D [batch, 1858] for take_along_axis.
    mx::array indices_2d = mx::broadcast_to(
        mx::reshape(*policy_gather_indices_, {1, kNumOutputPolicy}),
        {batch_size, kNumOutputPolicy});
    policy = mx::take_along_axis(policy, indices_2d, 1);
  }

  // Value head.
  mx::array value = [&]() -> mx::array {
    if (attn_body_) {
      mx::array val = DispatchFC(flow, value_weights_.ip_val_w_q,
                                 value_weights_.ip_val_w, *value_weights_.ip_val_b,
                                 activations_.default_activation);
      return mx::reshape(val, {batch_size, -1});
    } else {
      mx::array val = ConvBlock(flow, *value_weights_.value_conv_weights,
                                *value_weights_.value_conv_biases, 1,
                                activations_.default_activation);
      return mx::reshape(val, {batch_size, -1});
    }
  }();

  value = DispatchFC(value, value_weights_.ip1_val_w_q, value_weights_.ip1_val_w,
                     *value_weights_.ip1_val_b, activations_.default_activation);
  value = DispatchFC(value, value_weights_.ip2_val_w_q, value_weights_.ip2_val_w,
                     *value_weights_.ip2_val_b, wdl_ ? "softmax" : "tanh");

  // Cast outputs back to float32 for memcpy if needed.
  if (compute_dtype_ != mx::float32) {
    policy = mx::astype(policy, mx::float32);
    value = mx::astype(value, mx::float32);
  }

  // Moves left head.
  if (moves_left_) {
    mx::array mleft = [&]() -> mx::array {
      if (attn_body_) {
        return DispatchFC(flow, ip_mov_w_q_, ip_mov_w_, *ip_mov_b_,
                          activations_.default_activation);
      } else {
        return ConvBlock(flow, *moves_left_conv_weights_,
                         *moves_left_conv_biases_, 1,
                         activations_.default_activation);
      }
    }();
    mleft = mx::reshape(mleft, {batch_size, -1});
    mleft = DispatchFC(mleft, ip1_mov_w_q_, ip1_mov_w_, *ip1_mov_b_,
                       activations_.default_activation);
    mleft = DispatchFC(mleft, ip2_mov_w_q_, ip2_mov_w_, *ip2_mov_b_,
                       "relu");
    if (compute_dtype_ != mx::float32) {
      mleft = mx::astype(mleft, mx::float32);
    }
    return {policy, value, mleft};
  }

  return {policy, value};
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

  // Call compiled forward pass.
  auto outputs = compiled_forward_({input_vals, input_masks});

  mx::array& policy = outputs[0];
  mx::array& value = outputs[1];

  // Trigger MLX lazy evaluation.
  mx::eval(outputs);

  // Verify output arrays are valid before accessing data.
  assert(policy.size() > 0 && policy.size() % batch_size == 0);
  assert(value.size() == static_cast<size_t>(batch_size) * (wdl_ ? 3 : 1));
  if (moves_left_ && outputs.size() > 2) {
    assert(outputs[2].size() == static_cast<size_t>(batch_size));
  }

  // Validate output buffer pointers.
  assert(output_mems.size() >= 2);
  assert(output_mems[0] != nullptr && output_mems[1] != nullptr);
  if (moves_left_) {
    assert(output_mems.size() >= 3 && output_mems[2] != nullptr);
  }

  // Policy output — ForwardPass already mapped to [batch, 1858].
  std::memcpy(output_mems[0], policy.data<float>(),
              batch_size * kNumOutputPolicy * sizeof(float));

  // Value output.
  std::memcpy(output_mems[1], value.data<float>(),
              batch_size * (wdl_ ? 3 : 1) * sizeof(float));

  // Moves left output.
  if (moves_left_ && outputs.size() > 2) {
    std::memcpy(output_mems[2], outputs[2].data<float>(),
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

  // Parse precision option: fp32 (default), fp16, bf16, or q8 (int8 quantized).
  std::string precision_str = options.GetOrDefault<std::string>("precision", "fp32");
  // Parse group_size option for quantization (default 64).
  int group_size = options.GetOrDefault<int>("group_size", 64);

  Precision precision = Precision::FP32;
  if (precision_str == "fp16") {
    precision = Precision::FP16;
    CERR << "Using fp16 precision.";
  } else if (precision_str == "bf16") {
    precision = Precision::BF16;
    CERR << "Using bf16 precision.";
  } else if (precision_str == "q8" || precision_str == "int8") {
    precision = Precision::Q8;
    CERR << "Using int8 (q8) precision with group_size=" << group_size << ".";
    CERR << "NOTE: Layers with dimensions not divisible by " << group_size
         << " will use float16 instead.";
  }

  builder_->Build(kInputPlanes, weights, embedding, attn_body, attn_policy_,
                  conv_policy_, wdl_, moves_left_, activations, policy_head,
                  value_head, precision, group_size);
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
