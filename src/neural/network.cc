/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors
 
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

#include "neural/network.h"

#include <cmath>

namespace  lczero {
  
void Weights::ConvBlock::InvertStddev() {
  InvertStddev(bn_stddivs.size(), bn_stddivs.data());
}

void Weights::ConvBlock::OffsetMeans() {
  OffsetMeans(bn_means.size(), bn_means.data());
}

std::vector<float> Weights::ConvBlock::InvertStddev() const {
  Vec output=bn_stddivs;
  InvertStddev(output.size(), output.data());
  return output;
}

std::vector<float> Weights::ConvBlock::OffsetMeans() const {
  Vec output=bn_means;
  OffsetMeans(output.size(), output.data());
  return output;
}

void Weights::ConvBlock::InvertStddev(const size_t size, float* array) const {
  for (auto i = 0; i < size; i++)
    array[i] = 1.0f / std::sqrt(array[i] + kEpsilon);
}

void Weights::ConvBlock::OffsetMeans(const size_t size, float* means) const {
  for (auto i = 0; i < size; i++)
    means[i] -= biases[i];
}

} //namespace  lczero




