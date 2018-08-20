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

#include <cstddef>
#include <string>
#include <vector>

namespace lczero {

// Histogram with a logarithmic x-axis.
//
//    0.50   +
//           |
//           |
//           |
//    0.40   |
//
//          ....
//
//           |
//    0.10   +
//           |
//           |#
//           |#         ##                     #|
//           |#  #   # #### #  #     #         #|
//    0.00   +----+----+----+----+---- ... +----+
//
//         -inf  -15  -14  -13  -12        5   inf

class Histogram {
 public:
  // Creates a histogram with default scales.
  Histogram();

  // Creates a histogram from 10^min_exp to 10^max_exp
  // with minor_scales spacing.
  Histogram(int min_exp, int max_exp, int minor_scales);

  void Clear();

  // Adds a sample.
  void Add(double value);

  // Dumps the histogram to stderr.
  void Dump() const;

 private:
  int GetIndex(double val) const;

  static constexpr int kDefaultMinExp = -15;
  static constexpr int kDefaultMaxExp = 5;
  static constexpr int kDefaultMinorScales = 5;

  const int min_exp_;
  const int max_exp_;
  const int minor_scales_;
  const int major_scales_;
  const int total_scales_;
  std::vector<double> buckets_;
  double total_;
  double max_;
};

}  // namespace lczero
