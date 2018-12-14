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

#include "random.h"
#include <random>

namespace lczero {

Random::Random() : gen_(std::random_device()()) {}

Random& Random::Get() {
  static Random rand;
  return rand;
}

int Random::GetInt(int min, int max) {
  Mutex::Lock lock(mutex_);
  std::uniform_int_distribution<> dist(min, max);
  return dist(gen_);
}

bool Random::GetBool() { return GetInt(0, 1) != 0; }

double Random::GetDouble(double maxval) {
  Mutex::Lock lock(mutex_);
  std::uniform_real_distribution<> dist(0.0, maxval);
  return dist(gen_);
}

float Random::GetFloat(float maxval) {
  Mutex::Lock lock(mutex_);
  std::uniform_real_distribution<> dist(0.0, maxval);
  return dist(gen_);
}

std::string Random::GetString(int length) {
  std::string result;
  for (int i = 0; i < length; ++i) {
    result += 'a' + GetInt(0, 25);
  }
  return result;
}

double Random::GetGamma(double alpha, double beta) {
  Mutex::Lock lock(mutex_);
  std::gamma_distribution<double> dist(alpha, beta);
  return dist(gen_);
}

}  // namespace lczero
