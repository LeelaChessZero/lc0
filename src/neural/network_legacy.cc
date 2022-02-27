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

#include "neural/network_legacy.h"

#include <algorithm>
#include <cmath>

#include "utils/weights_adapter.h"

namespace lczero {
namespace {
static constexpr float kEpsilon = 1e-5f;
}  // namespace

LegacyWeights::LegacyWeights(const pblczero::Weights& weights)
    : input(weights.input()),
      policy1(weights.policy1()),
      policy(weights.policy()),
      ip_pol_w(LayerAdapter(weights.ip_pol_w()).as_vector()),
      ip_pol_b(LayerAdapter(weights.ip_pol_b()).as_vector()),
      ip2_pol_w(LayerAdapter(weights.ip2_pol_w()).as_vector()),
      ip2_pol_b(LayerAdapter(weights.ip2_pol_b()).as_vector()),
      ip3_pol_w(LayerAdapter(weights.ip3_pol_w()).as_vector()),
      ip3_pol_b(LayerAdapter(weights.ip3_pol_b()).as_vector()),
      ip4_pol_w(LayerAdapter(weights.ip4_pol_w()).as_vector()),
      value(weights.value()),
      ip1_val_w(LayerAdapter(weights.ip1_val_w()).as_vector()),
      ip1_val_b(LayerAdapter(weights.ip1_val_b()).as_vector()),
      ip2_val_w(LayerAdapter(weights.ip2_val_w()).as_vector()),
      ip2_val_b(LayerAdapter(weights.ip2_val_b()).as_vector()),
      moves_left(weights.moves_left()),
      ip1_mov_w(LayerAdapter(weights.ip1_mov_w()).as_vector()),
      ip1_mov_b(LayerAdapter(weights.ip1_mov_b()).as_vector()),
      ip2_mov_w(LayerAdapter(weights.ip2_mov_w()).as_vector()),
      ip2_mov_b(LayerAdapter(weights.ip2_mov_b()).as_vector()) {
  for (const auto& res : weights.residual()) {
    residual.emplace_back(res);
  }
  pol_encoder_head_count = weights.pol_headcount();
  for (const auto& enc : weights.pol_encoder()) {
    pol_encoder.emplace_back(enc);
  }
}

LegacyWeights::SEunit::SEunit(const pblczero::Weights::SEunit& se)
    : w1(LayerAdapter(se.w1()).as_vector()),
      b1(LayerAdapter(se.b1()).as_vector()),
      w2(LayerAdapter(se.w2()).as_vector()),
      b2(LayerAdapter(se.b2()).as_vector()) {}

LegacyWeights::Residual::Residual(const pblczero::Weights::Residual& residual)
    : conv1(residual.conv1()),
      conv2(residual.conv2()),
      se(residual.se()),
      has_se(residual.has_se()) {}

LegacyWeights::ConvBlock::ConvBlock(const pblczero::Weights::ConvBlock& block)
    : weights(LayerAdapter(block.weights()).as_vector()),
      biases(LayerAdapter(block.biases()).as_vector()),
      bn_gammas(LayerAdapter(block.bn_gammas()).as_vector()),
      bn_betas(LayerAdapter(block.bn_betas()).as_vector()),
      bn_means(LayerAdapter(block.bn_means()).as_vector()),
      bn_stddivs(LayerAdapter(block.bn_stddivs()).as_vector()) {
  if (weights.size() == 0) {
    // Empty ConvBlock.
    return;
  }

  if (bn_betas.size() == 0) {
    // Old net without gamma and beta.
    for (auto i = size_t{0}; i < bn_means.size(); i++) {
      bn_betas.emplace_back(0.0f);
      bn_gammas.emplace_back(1.0f);
    }
  }
  if (biases.size() == 0) {
    for (auto i = size_t{0}; i < bn_means.size(); i++) {
      biases.emplace_back(0.0f);
    }
  }

  if (bn_means.size() == 0) {
    // No batch norm.
    return;
  }

  // Fold batch norm into weights and biases.
  // Variance to gamma.
  for (auto i = size_t{0}; i < bn_stddivs.size(); i++) {
    bn_gammas[i] *= 1.0f / std::sqrt(bn_stddivs[i] + kEpsilon);
    bn_means[i] -= biases[i];
  }

  auto outputs = biases.size();

  // We can treat the [inputs, filter_size, filter_size] dimensions as one.
  auto inputs = weights.size() / outputs;

  for (auto o = size_t{0}; o < outputs; o++) {
    for (auto c = size_t{0}; c < inputs; c++) {
      weights[o * inputs + c] *= bn_gammas[o];
    }

    biases[o] = -bn_gammas[o] * bn_means[o] + bn_betas[o];
  }

  // Batch norm weights are not needed anymore.
  bn_stddivs.clear();
  bn_means.clear();
  bn_betas.clear();
  bn_gammas.clear();
}

LegacyWeights::MHA::MHA(const pblczero::Weights::MHA& mha)
    : q_w(LayerAdapter(mha.q_w()).as_vector()),
      q_b(LayerAdapter(mha.q_b()).as_vector()),
      k_w(LayerAdapter(mha.k_w()).as_vector()),
      k_b(LayerAdapter(mha.k_b()).as_vector()),
      v_w(LayerAdapter(mha.v_w()).as_vector()),
      v_b(LayerAdapter(mha.v_b()).as_vector()),
      dense_w(LayerAdapter(mha.dense_w()).as_vector()),
      dense_b(LayerAdapter(mha.dense_b()).as_vector()) {}

LegacyWeights::FFN::FFN(const pblczero::Weights::FFN& ffn)
    : dense1_w(LayerAdapter(ffn.dense1_w()).as_vector()),
      dense1_b(LayerAdapter(ffn.dense1_b()).as_vector()),
      dense2_w(LayerAdapter(ffn.dense2_w()).as_vector()),
      dense2_b(LayerAdapter(ffn.dense2_b()).as_vector()) {}

LegacyWeights::EncoderLayer::EncoderLayer(
    const pblczero::Weights::EncoderLayer& encoder)
    : mha(MHA(encoder.mha())),
      ln1_gammas(LayerAdapter(encoder.ln1_gammas()).as_vector()),
      ln1_betas(LayerAdapter(encoder.ln1_betas()).as_vector()),
      ffn(FFN(encoder.ffn())),
      ln2_gammas(LayerAdapter(encoder.ln2_gammas()).as_vector()),
      ln2_betas(LayerAdapter(encoder.ln2_betas()).as_vector()) {}

}  // namespace lczero
