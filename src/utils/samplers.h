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

class Sampler {
 public:
  virtual ~Sampler() = default;

  // Resets sampler to it can start fresh.
  virtual void Reset() = 0;
  // Adds one sample.
  virtual void Add(double val) = 0;
  // Adds one sample which should never be returned as a result of Toss() (but
  // still consumes sample index).
  virtual void AddImprobable() = 0;
  // Picks a random sample according to sampler's distribution and returns its
  // index.
  virtual int Toss() const = 0;
};

// Returns random sample proportional to
// P(x) = exp(x / theta) / sum{over all y}(exp(y / theta))
// Not thread safe! (has own RNG without mutex).
class SoftmaxSampler : public Sampler {
 public:
  // Initializes softmaxer with a given beta parameter.
  SoftmaxSampler(double theta);
  void Reset() override;
  void Add(double val) override;
  void AddImprobable() override;
  int Toss() const override;

 private:
  const double base_;
  std::vector<double> cumulative_values_;
  mutable std::mt19937 gen_;
};

// Returns random sample proportional to
// P(x) = x ^ (1/theta) / sum{over all y}(y ^ (1/theta))
// Not thread safe! (has own RNG without mutex).
class PowSampler : public Sampler {
 public:
  // Initializes softmaxer with a given beta parameter.
  PowSampler(double theta);
  void Reset() override;
  void Add(double val) override;
  void AddImprobable() override;
  int Toss() const override;

 private:
  const double exponent_;
  std::vector<double> cumulative_values_;
  mutable std::mt19937 gen_;
};

// Returns index of the maximum element.
// If several samples have the same maximum value, random index of them is
// returned.
// Not thread safe! (has own RNG without mutex).
class MaxSampler : public Sampler {
 public:
  MaxSampler();
  void Reset() override;
  void Add(double val) override;
  void AddImprobable() override;
  int Toss() const override;

 private:
  int num_;
  int best_idx_;
  int best_count_;
  double best_val_;
  mutable std::mt19937 gen_;
};

// Creates SoftmaxSampler or MaxSampler, depending on whether theta is 0.
std::unique_ptr<Sampler> MakeSoftmaxSampler(double theta);

// Creates PowSampler or MaxSampler, depending on whether theta is 0.
std::unique_ptr<Sampler> MakePowSampler(double theta);

}  // namespace lczero