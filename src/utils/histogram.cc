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

#include "utils/histogram.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace lczero {

namespace {
void Print(const std::string& what) { std::cerr << what; }

void PrintAligned(const std::string& what, int aligned) {
  std::cerr << std::right << std::setw(aligned) << what;
}

std::string Format(const std::string& format, double value) {
  static const int kMaxBufferSize = 32;
  char buffer[kMaxBufferSize];
  const int len = snprintf(buffer, kMaxBufferSize, format.c_str(), value);
  return std::string(buffer, buffer + len);
}
}  // namespace

Histogram::Histogram()
    : Histogram(kDefaultMinExp, kDefaultMaxExp, kDefaultMinorScales) {}

Histogram::Histogram(int min_exp, int max_exp, int minor_scales)
    : min_exp_(min_exp),
      max_exp_(max_exp),
      minor_scales_(minor_scales),
      major_scales_(max_exp_ - min_exp_ + 1),
      total_scales_(major_scales_ * minor_scales_),
      buckets_(total_scales_ + 4) {
  Clear();
}

void Histogram::Clear() {
  std::fill(buckets_.begin(), buckets_.end(), 0);
  total_ = 0;
  max_ = 0;
}

void Histogram::Add(double value) {
  const int index = GetIndex(std::abs(value));
  const int count = ++buckets_[index];
  total_++;
  if (count > max_) max_ = count;
}

void Histogram::Dump() const {
  const double ymax = 0.02 + max_ / (double)total_;
  for (int i = 0; i < 100; i++) {
    const double yscale = 1 - i * 0.01;
    if (yscale > ymax) continue;
    const bool scale = i % 5 == 0;
    if (scale) {
      PrintAligned(Format("%.2g", yscale), 5);
      Print(" +");
    } else {
      Print("      |");
    }
    const double ymin = (99 - i) * 0.01;
    for (size_t j = 0; j < buckets_.size(); j++) {
      const double val = buckets_[j] / (double)total_;
      if (val > ymin) {
        Print("#");
      } else {
        Print(" ");
      }
    }
    if (scale) {
      Print("+");
    } else {
      Print("|");
    }
    Print("\n");
  }
  Print("      +");
  for (int j = 0; j <= major_scales_; j++) {
    const int size = j == 0 ? 5 : minor_scales_;
    for (int k = 0; k < size - 1; k++) Print("-");
    Print("+");
  }
  Print("\n");
  Print("   -inf");
  for (int j = 0; j < major_scales_; j++) {
    const int size = j == 0 ? 5 : minor_scales_;
    Print(" ");
    PrintAligned(Format("%g", min_exp_ + j), size - 1);
  }
  Print("  ");
  PrintAligned("+inf", minor_scales_ - 2);
  Print(" \n");
}

int Histogram::GetIndex(double val) const {
  if (val <= 0) return 0;
  const double log10 = std::log10(val);
  // 2: -15 :    -15.1 ... -14.9          2 ... 3
  // 1:          -15.3 ... -15.1
  // 0:          -15.5 ... -15.3          0 ... 1
  const int index =
      static_cast<int>(std::floor(2.5 + minor_scales_ * (log10 - min_exp_)));
  if (index < 0) return 0;
  if (index >= total_scales_) return total_scales_ + 3;
  return index + 2;
}

}  // namespace lczero
