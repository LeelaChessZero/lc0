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

#include "random.h"
#include <random>

namespace lczero {

Random::Random() : gen_(std::random_device()()) {}

Random& Random::Get() {
  static Random rand = Random();
  return rand;
}

bool Random::GetBool() {
  std::uniform_int_distribution<> dist(0, 1);
  return dist(gen_) != 0;
}

double Random::GetDouble(double maxval) {
  std::uniform_real_distribution<> dist(0.0, maxval);
  return dist(gen_);
}

}  // namespace lczero