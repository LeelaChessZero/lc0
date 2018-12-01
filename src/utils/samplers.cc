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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "utils/samplers.h"
#include <algorithm>

namespace lczero {

SoftmaxSampler::SoftmaxSampler(double theta)
    : base_(std::exp(1.0 / theta)),
      gen_(Random::Get().GetInt(0, std::numeric_limits<int>::max())) {
  Reset();
}

void SoftmaxSampler::Reset() {
  cumulative_values_.clear();
  cumulative_values_.push_back(0.0);
}

void SoftmaxSampler::Add(double val) {
  cumulative_values_.push_back(cumulative_values_.back() +
                               std::pow(base_, val));
}

void SoftmaxSampler::AddImprobable() { cumulative_values_.push_back(0.0); }

int SoftmaxSampler::Toss() const {
  const double rnd = std::uniform_real_distribution<double>(
      0.0, cumulative_values_.back())(gen_);
  return std::upper_bound(cumulative_values_.begin(), cumulative_values_.end(),
                          rnd) -
         cumulative_values_.begin() - 1;
}

PowSampler::PowSampler(double theta)
    : exponent_(1.0 / theta),
      gen_(Random::Get().GetInt(0, std::numeric_limits<int>::max())) {
  Reset();
}

void PowSampler::Reset() {
  cumulative_values_.clear();
  cumulative_values_.push_back(0.0);
}

void PowSampler::Add(double val) {
  cumulative_values_.push_back(cumulative_values_.back() +
                               std::pow(val, exponent_));
}

void PowSampler::AddImprobable() { cumulative_values_.push_back(0.0); }

int PowSampler::Toss() const {
  const double rnd = std::uniform_real_distribution<double>(
      0.0, cumulative_values_.back())(gen_);
  return std::upper_bound(cumulative_values_.begin(), cumulative_values_.end(),
                          rnd) -
         cumulative_values_.begin() - 1;
}

MaxSampler::MaxSampler()
    : gen_(Random::Get().GetInt(0, std::numeric_limits<int>::max())) {
  Reset();
}

void MaxSampler::Reset() {
  num_ = 0;
  best_idx_ = -1;
}

void MaxSampler::Add(double val) {
  if (best_idx_ == -1 || best_val_ < val) {
    best_idx_ = num_;
    best_val_ = val;
    best_count_ = 1;
  } else if (best_val_ == val) {
    if (std::uniform_int_distribution<int>(0, best_count_)(gen_) == 0) {
      best_idx_ = num_;
    }
    ++best_count_;
  }
  ++num_;
}

void MaxSampler::AddImprobable() { ++num_; }

int MaxSampler::Toss() const { return best_idx_; }

std::unique_ptr<Sampler> MakeSoftmaxSampler(double theta) {
  if (theta > 0) {
    return std::make_unique<SoftmaxSampler>(theta);
  } else {
    return std::make_unique<MaxSampler>();
  }
}

std::unique_ptr<Sampler> MakePowSampler(double theta) {
  if (theta > 0) {
    return std::make_unique<PowSampler>(theta);
  } else {
    return std::make_unique<MaxSampler>();
  }
}

}  // namespace lczero