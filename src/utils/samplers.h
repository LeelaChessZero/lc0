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

#include <vector>
#include "utils/random.h"

namespace lczero {

template <class T>
class Sampler {
 public:
  virtual ~Sampler() = default;

  // Resets sampler to it can start fresh.
  virtual void Reset() = 0;
  // Adds one sample.
  virtual void Add(double weight, const T& val) = 0;
  // Picks a random sample according to sampler's distribution and returns its
  // value.
  virtual T Toss() const = 0;
};

// Returns random sample proportional to
// P(x) = exp(x / theta) / sum{over all y}(exp(y / theta))
// Not thread safe! (has own RNG without mutex).
template <class T>
class SoftmaxSampler : public Sampler<T> {
 public:
  SoftmaxSampler(double theta)
      : base_(std::exp(1.0 / theta)),
        gen_(Random::Get().GetInt(0, std::numeric_limits<int>::max())) {}

  void Reset() override {
    weights_.clear();
    values_.clear();
    cumulative_weights_.clear();
  }

  void Add(double weight, const T& val) override {
    if (weights_.empty() || weight > max_weight_) max_weight_ = weight;
    weights_.push_back(weight);
    values_.push_back(val);
  }

  T Toss() const override {
    if (weights_.size() != cumulative_weights_.size()) {
      cumulative_weights_.clear();
      double sum = 0.0;
      for (auto weight : weights_) {
        sum += std::pow(base_, weight - max_weight_);
        cumulative_weights_.push_back(sum);
      }
    }

    const double rnd = std::uniform_real_distribution<double>(
        0.0, cumulative_weights_.back())(gen_);
    int idx = std::lower_bound(cumulative_weights_.begin(),
                               cumulative_weights_.end(), rnd) -
              cumulative_weights_.begin();
    return values_[idx];
  }

 private:
  const double base_;
  std::vector<double> weights_;
  std::vector<T> values_;
  mutable std::vector<double> cumulative_weights_;
  double max_weight_;
  mutable std::mt19937 gen_;
};

// Returns random sample proportional to
// P(x) = x ^ (1/theta) / sum{over all y}(y ^ (1/theta))
// Not thread safe! (has own RNG without mutex).
template <class T>
class PowSampler : public Sampler<T> {
 public:
  PowSampler(double theta)
      : exponent_(1.0 / theta),
        gen_(Random::Get().GetInt(0, std::numeric_limits<int>::max())) {
    Reset();
  }

  void Reset() override {
    cumulative_weights_.clear();
    cumulative_weights_.push_back(0.0);
    values_.clear();
  }

  void Add(double weight, const T& val) override {
    cumulative_weights_.push_back(cumulative_weights_.back() +
                                  std::pow(weight, exponent_));
    values_.push_back(val);
  }

  T Toss() const override {
    const double rnd = std::uniform_real_distribution<double>(
        0.0, cumulative_weights_.back())(gen_);
    int idx = std::upper_bound(cumulative_weights_.begin(),
                               cumulative_weights_.end(), rnd) -
              cumulative_weights_.begin() - 1;
    return values_[idx];
  }

 private:
  const double exponent_;
  std::vector<double> cumulative_weights_;
  std::vector<T> values_;
  mutable std::mt19937 gen_;
};

// Returns index of the maximum element.
// If several samples have the same maximum value, random index of them is
// returned.
// Not thread safe! (has own RNG without mutex).
template <class T>
class MaxSampler : public Sampler<T> {
 public:
  void Reset() override { have_value_ = false; }
  void Add(double weight, const T& val) override {
    if (!have_value_ || best_weight_ < weight) {
      have_value_ = true;
      best_value_ = val;
      best_weight_ = weight;
    }
  }

  T Toss() const override { return best_value_; }

 private:
  // TODO(crem): Replace it with std::optional when we have proper
  // std::optional.
  bool have_value_ = false;
  T best_value_;
  double best_weight_;
};  // namespace lczero

// Creates SoftmaxSampler or MaxSampler, depending on whether theta is 0.
template <class T>
std::unique_ptr<Sampler<T>> MakeSoftmaxSampler(double theta) {
  if (theta > 0) {
    return std::make_unique<SoftmaxSampler<T>>(theta);
  } else {
    return std::make_unique<MaxSampler<T>>();
  }
}

// Creates PowSampler or MaxSampler, depending on whether theta is 0.
template <class T>
std::unique_ptr<Sampler<T>> MakePowSampler(double theta) {
  if (theta > 0) {
    return std::make_unique<PowSampler<T>>(theta);
  } else {
    return std::make_unique<MaxSampler<T>>();
  }
}

}  // namespace lczero