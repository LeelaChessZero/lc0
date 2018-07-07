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

/* Histogram with a logarithmic x-axis
 *
 *    0.50   +
 *           |
 *           |
 *           |
 *    0.40   |
 *
 *          ....
 *
 *           |
 *    0.10   +
 *           |
 *           |#
 *           |#         ##                     #|
 *           |#  #   # #### #  #     #         #|
 *    0.00   +----+----+----+----+---- ... +----+
 *
 *         -inf  -15  -14  -13  -12        5   inf
 */

class Histogram {
 public:
  // Create a histogram with default scales.
  Histogram();

  // Create a histogram from 10^minExp to 10^maxExp with minorScales
  // spacing.
  Histogram(int minExp, int maxExp, int minorScales);

  void Clear();

  // Add a sample.
  void Add(double value);

  // Dump the histogram to stderr.
  void Dump();

 private:
  int GetIndex(double val);

  void Print(const char* what);
  void Print(const char* what, int aligned);
  void Print(const char* format, double value, int aligned);

  static constexpr int kDefaultMinExp = -15;
  static constexpr int kDefaultMaxExp = 5;
  static constexpr int kDefaultMinorScales = 5;

  const int minExp_;
  const int maxExp_;
  const int majorScales_;
  const int minorScales_;
  const int totalScales_;
  const int fullScales_;

  double* const buckets_;
  double total_;
  double max_;
};
