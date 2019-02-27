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

#include <cmath>
#include <algorithm>
#include "utils/weights_adapter.h"

namespace lczero {
namespace {
static constexpr float kEpsilon = 1e-5f;

void InvertVector(std::vector<float>* vec) {
  for (auto& x : *vec) x = 1.0f / std::sqrt(x + kEpsilon);
}

void OffsetVector(std::vector<float>* means, const std::vector<float>& biases) {
  std::transform(means->begin(), means->end(), biases.begin(), means->begin(),
                 std::minus<float>());
}

}  // namespace

LegacyWeights::LegacyWeights(const pblczero::Weights& weights)
    : input(weights.input()),
      policy1(weights.policy1()),
      policy(weights.policy()),
      ip_pol_w(LayerAdapter(weights.ip_pol_w()).as_vector()),
      ip_pol_b(LayerAdapter(weights.ip_pol_b()).as_vector()),
      value(weights.value()),
      ip1_val_w(LayerAdapter(weights.ip1_val_w()).as_vector()),
      ip1_val_b(LayerAdapter(weights.ip1_val_b()).as_vector()),
      ip2_val_w(LayerAdapter(weights.ip2_val_w()).as_vector()),
      ip2_val_b(LayerAdapter(weights.ip2_val_b()).as_vector()) {
  for (const auto& res : weights.residual()) {
    residual.emplace_back(res);
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
  // Merge gammas to stddivs and beta to means for backwards compatibility
  auto channels = bn_betas.size();
  auto epsilon = 1e-5;
  for (auto i = size_t{0}; i < channels; i++) {
    auto s = bn_gammas[i] / std::sqrt(bn_stddivs[i] + epsilon);
    bn_stddivs[i] = 1.0f / (s * s) - epsilon;
    bn_means[i] -= bn_betas[i] / s;
    bn_gammas[i] = 1.0f;
    bn_betas[i] = 0.0f;
    biases.emplace_back(0.0f);
  }
}

void LegacyWeights::ConvBlock::InvertStddev() { InvertVector(&bn_stddivs); }

void LegacyWeights::ConvBlock::OffsetMeans() {
  OffsetVector(&bn_means, biases);
}

std::vector<float> LegacyWeights::ConvBlock::GetInvertedStddev() const {
  std::vector<float> stddivs = bn_stddivs;  // Copy.
  InvertVector(&stddivs);
  return stddivs;
}

std::vector<float> LegacyWeights::ConvBlock::GetOffsetMeans() const {
  std::vector<float> means = bn_means;  // Copy.
  OffsetVector(&means, biases);
  return means;
}

}  // namespace lczero
