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

#pragma once

#include <algorithm>
#include <random>
#include <string>
#include "utils/mutex.h"

namespace lczero {

class Random {
 public:
  static Random& Get();
  double GetDouble(double max_val);
  float GetFloat(float max_val);
  double GetGamma(double alpha, double beta);
  // Both sides are included.
  int GetInt(int min, int max);
  std::string GetString(int length);
  bool GetBool();
  template <class RandomAccessIterator>
  void Shuffle(RandomAccessIterator s, RandomAccessIterator e);

 private:
  Random();

  Mutex mutex_;
  std::mt19937 gen_ GUARDED_BY(mutex_);
};

template <class RandomAccessIterator>
void Random::Shuffle(RandomAccessIterator s, RandomAccessIterator e) {
  Mutex::Lock lock(mutex_);
  std::shuffle(s, e, gen_);
}

}  // namespace lczero
