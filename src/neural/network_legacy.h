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

#include <vector>

#include "proto/net.pb.h"

namespace lczero {

struct LegacyWeights {
  explicit LegacyWeights(const pblczero::Weights& weights);

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
  };

  struct FFN {
    explicit FFN(const pblczero::Weights::FFN& mha);
    Vec dense1_w;
    Vec dense1_b;
    Vec dense2_w;
    Vec dense2_b;
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

  // Embedding layer
  Vec ip_emb_w;
  Vec ip_emb_b;

  // Input gating
  Vec ip_mult_gate;
  Vec ip_add_gate;

  // Encoder stack.
  std::vector<EncoderLayer> encoder;
  int encoder_head_count;


  // Residual tower.
  std::vector<Residual> residual;

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
  Vec smolgen_b;
  bool has_smolgen;
};

}  // namespace lczero
