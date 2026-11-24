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
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "proto/net.pb.h"

namespace lczero {

struct BaseWeights {
  explicit BaseWeights(const pblczero::Weights& weights);

  using Vec = std::vector<float>;
  struct ConvBlock {
    explicit ConvBlock(const pblczero::Weights::ConvBlock& block);

    Vec weights;
    Vec biases;
    Vec bn_gammas;
    Vec bn_betas;
    Vec bn_means;
    Vec bn_stddivs;
  };

  struct SEunit {
    explicit SEunit(const pblczero::Weights::SEunit& se);
    Vec w1;
    Vec b1;
    Vec w2;
    Vec b2;
  };

  struct Residual {
    explicit Residual(const pblczero::Weights::Residual& residual);
    ConvBlock conv1;
    ConvBlock conv2;
    SEunit se;
    bool has_se;
  };

  struct Smolgen {
    explicit Smolgen(const pblczero::Weights::Smolgen& smolgen);
    Vec compress;
    Vec dense1_w;
    Vec dense1_b;
    Vec ln1_gammas;
    Vec ln1_betas;
    Vec dense2_w;
    Vec dense2_b;
    Vec ln2_gammas;
    Vec ln2_betas;
  };

  struct MHA {
    explicit MHA(const pblczero::Weights::MHA& mha);
    Vec q_w;
    Vec q_b;
    Vec k_w;
    Vec k_b;
    Vec v_w;
    Vec v_b;
    Vec dense_w;
    Vec dense_b;
    Smolgen smolgen;
    bool has_smolgen;
    // Attention-logit terms from the attack graph E [C, 64, 64] and the pair
    // stream Z [pair_d_e, 64, 64] (net.proto MHA); row-major, empty if absent.
    Vec attack_w;       // [heads, C]
    Vec attack_gate_q;  // [heads, C, depth]
    Vec attack_gate_k;  // [heads, C, depth]
    Vec pair_att_w;     // [heads, pair_d_e]
    // Attention output gate: the attention output is multiplied by
    // scale * sigmoid(x @ output_gate_w + output_gate_b) before dense_w.
    Vec output_gate_w;  // like q_w
    Vec output_gate_b;  // like q_b
    // Edge-stream doors: the logits L and weights W of a block read the edge
    // state e [d_e, 64, 64] as L = (edge_t_node L + edge_t_edge (edge_e_w . e))
    // * (1 + edge_m_w . e) and W = softmax(L) * scale * sigmoid(edge_g_w . e +
    // edge_g_b) (net.proto MHA); row-major, empty if absent.
    Vec edge_e_w;     // [heads, d_e]
    Vec edge_m_w;     // [heads, d_e]
    Vec edge_g_w;     // [heads, d_e]
    Vec edge_g_b;     // [heads]
    Vec edge_t_edge;  // [heads]
    Vec edge_t_node;  // [heads]
  };

  struct FFN {
    explicit FFN(const pblczero::Weights::FFN& mha);
    Vec dense1_w;
    Vec dense1_b;
    Vec dense2_w;
    Vec dense2_b;
    Vec dense_gate_w;  // ACTIVATION_SWIGLU only
  };

  struct EncoderLayer {
    explicit EncoderLayer(const pblczero::Weights::EncoderLayer& encoder);
    MHA mha;
    Vec ln1_gammas;
    Vec ln1_betas;
    FFN ffn;
    Vec ln2_gammas;
    Vec ln2_betas;
  };

  // Input convnet.
  ConvBlock input;

  // Embedding preprocess layer.
  Vec ip_emb_preproc_w;
  Vec ip_emb_preproc_b;

  // Embedding layer
  Vec ip_emb_w;
  Vec ip_emb_b;

  // Embedding layernorm
  // @todo can this be folded into weights?
  Vec ip_emb_ln_gammas;
  Vec ip_emb_ln_betas;

  // Input gating
  Vec ip_mult_gate;
  Vec ip_add_gate;

  // Embedding feedforward network
  FFN ip_emb_ffn;
  Vec ip_emb_ffn_ln_gammas;
  Vec ip_emb_ffn_ln_betas;

  // Encoder stack.
  std::vector<EncoderLayer> encoder;
  int encoder_head_count;

  // Residual tower.
  std::vector<Residual> residual;

  // Moves left head
  ConvBlock moves_left;
  Vec ip_mov_w;
  Vec ip_mov_b;
  Vec ip1_mov_w;
  Vec ip1_mov_b;
  Vec ip2_mov_w;
  Vec ip2_mov_b;

  // Smolgen global weights
  Vec smolgen_w;
  bool has_smolgen;

  // Static pair stream Z = rms(E @ p_in + t_off[rel(i, j)]) (net.proto
  // PairStream); row-major, empty if absent.
  Vec pair_p_in;   // [C, pair_d_e]
  Vec pair_t_off;  // [225, pair_d_e]

  // Edge stream (net.proto EdgeStream): seed e = rms(E @ p_in +
  // t_off[rel(i, j)]), then one update per site; row-major, empty if absent.
  struct EdgeStreamUpdate {
    explicit EdgeStreamUpdate(const pblczero::Weights::EdgeStreamUpdate& u);
    Vec o_e;       // [heads, d_e] logit readback
    Vec tri_v;     // [d_e, 2 d_e]
    Vec tri_eg;    // [d_e, 4 triplet_heads]
    Vec tri_eg_b;  // [4 triplet_heads]
    Vec tri_o;     // [2 d_e, d_e]
    Vec ffn_in;    // [d_e, hidden]
    Vec ffn_in_b;  // [hidden]
    Vec ffn_rev;   // [d_e, hidden], empty without rev_edge
    Vec ffn_out;   // [hidden, d_e]
  };
  Vec edge_p_in;   // [C, d_e]
  Vec edge_t_off;  // [225, d_e]
  std::vector<EdgeStreamUpdate> edge_updates;  // in site order

  // Final norm of a pre-norm encoder tower (net.proto encoder_final_norm);
  // empty if absent.
  Vec encoder_final_norm_gammas;
  Vec encoder_final_norm_betas;
};

struct LegacyWeights : public BaseWeights {
  explicit LegacyWeights(const pblczero::Weights& weights);

  // Policy head
  // Extra convolution for AZ-style policy head
  ConvBlock policy1;
  ConvBlock policy;
  Vec ip_pol_w;
  Vec ip_pol_b;
  // Extra params for attention policy head
  Vec ip2_pol_w;
  Vec ip2_pol_b;
  Vec ip3_pol_w;
  Vec ip3_pol_b;
  Vec ip4_pol_w;
  int pol_encoder_head_count;
  std::vector<EncoderLayer> pol_encoder;

  // Value head
  ConvBlock value;
  Vec ip_val_w;
  Vec ip_val_b;
  Vec ip1_val_w;
  Vec ip1_val_b;
  Vec ip2_val_w;
  Vec ip2_val_b;
};

struct MultiHeadWeights : public BaseWeights {
  explicit MultiHeadWeights(const pblczero::Weights& weights);

  struct PolicyHead {
    explicit PolicyHead(const pblczero::Weights::PolicyHead& policyhead, Vec& w,
                        Vec& b);
    // Policy head
   private:
    // Storage in case _ip_pol_w/b are not shared among heads.
    Vec _ip_pol_w;
    Vec _ip_pol_b;

   public:
    // Reference to possibly shared value (to avoid unnecessary copies).
    Vec& ip_pol_w;
    Vec& ip_pol_b;
    // Extra convolution for AZ-style policy head
    ConvBlock policy1;
    ConvBlock policy;
    // Extra params for attention policy head
    Vec ip2_pol_w;
    Vec ip2_pol_b;
    Vec ip3_pol_w;
    Vec ip3_pol_b;
    Vec ip4_pol_w;
    int pol_encoder_head_count;
    std::vector<EncoderLayer> pol_encoder;
    // logits[from, to] += pair_u . Z[:, from, to]; empty if absent.
    Vec pair_u;  // [pair_d_e]
  };

  struct ValueHead {
    explicit ValueHead(const pblczero::Weights::ValueHead& valuehead);
    // Value head
    ConvBlock value;
    Vec ip_val_w;
    Vec ip_val_b;
    Vec ip1_val_w;
    Vec ip1_val_b;
    Vec ip2_val_w;
    Vec ip2_val_b;
    Vec ip_val_err_w;
    Vec ip_val_err_b;
  };

 private:
  Vec ip_pol_w;
  Vec ip_pol_b;

 public:
  // Policy and value multiheads
  std::unordered_map<std::string, ValueHead> value_heads;
  std::unordered_map<std::string, PolicyHead> policy_heads;
  // Heads of the head maps that need more than the structs above hold (child-Q
  // bins, depth attention): not loaded, named here for error messages.
  std::vector<std::string> unsupported_policy_heads;
  std::vector<std::string> unsupported_value_heads;
};

enum InputEmbedding {
  INPUT_EMBEDDING_NONE = 0,
  INPUT_EMBEDDING_PE_MAP = 1,
  INPUT_EMBEDDING_PE_DENSE = 2,
};

}  // namespace lczero
