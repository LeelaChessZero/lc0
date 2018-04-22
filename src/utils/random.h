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

#pragma once

#include <random>
#include <string>
#include "utils/mutex.h"

namespace lczero {

class Random {
 public:
  static Random& Get();
  double GetDouble(double max_val);
  double GetGamma(double alpha, double beta);
  // Both sides are included.
  int GetInt(int min, int max);
  std::string GetString(int length);
  bool GetBool();

 private:
  Random();

  Mutex mutex_;
  std::mt19937 gen_ GUARDED_BY(mutex_);
};

}  // namespace lczero