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

#include "utils/histogram.h"

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cmath>

Histogram::Histogram()
    : Histogram(kDefaultMinExp, kDefaultMaxExp, kDefaultMinorScales) {}

Histogram::Histogram(int minExp, int maxExp, int minorScales)
    : minExp_(minExp),
      maxExp_(maxExp),
      minorScales_(minorScales),
      majorScales_(maxExp_ - minExp_ + 1),
      totalScales_(majorScales_ * minorScales_),
      fullScales_(totalScales_ + 4),
      buckets_(new double[fullScales_]) {
  Clear();
}

void Histogram::Clear() {
  for (int i = 0; i < fullScales_; i++) buckets_[i] = 0;
  total_ = 0;
  max_ = 0;
}

void Histogram::Add(double value) {
  int index = GetIndex(std::abs(value));
  int count = ++buckets_[index];
  total_++;
  if (count > max_) max_ = count;
}

void Histogram::Dump() {
  double ymax = 0.02 + max_ / (double)total_;
  for (int i = 0; i < 100; i++) {
    double yscale = 1 - i * 0.01;
    if (yscale > ymax) continue;
    bool scale = i % 5 == 0;
    if (scale) {
      Print("%.2g", yscale, 5);
      Print(" +");
    } else {
      Print("      |");
    }
    double ymin = (99 - i) * 0.01;
    for (int j = 0; j < fullScales_; j++) {
      double val = buckets_[j] / (double)total_;
      if (val > ymin)
        Print("#");
      else
        Print(" ");
    }
    if (scale)
      Print("+");
    else
      Print("|");
    Print("\n");
  }
  Print("      +");
  for (int j = 0; j <= majorScales_; j++) {
    int size = j == 0 ? 5 : minorScales_;
    for (int k = 0; k < size - 1; k++) Print("-");
    Print("+");
  }
  Print("\n");
  Print("   -inf");
  for (int j = 0; j < majorScales_; j++) {
    int size = j == 0 ? 5 : minorScales_;
    Print(" ");
    Print("%g", minExp_ + j, size - 1);
  }
  Print("  ");
  Print("+inf", minorScales_ - 2);
  Print(" \n");
}

int Histogram::GetIndex(double val) {
  if (val <= 0) return 0;
  double log10 = std::log10(val);
  // 2: -15 :    -15.1 ... -14.9          2 ... 3
  // 1:          -15.3 ... -15.1
  // 0:          -15.5 ... -15.3          0 ... 1

  int index = (int)std::floor(2.5 + minorScales_ * (log10 - minExp_));
  if (index < 0) return 0;

  if (index >= totalScales_) return totalScales_ + 3;

  return index + 2;
}

void Histogram::Print(const char* what) {
  // fprintf(stderr, what); is disallowed
  fprintf(stderr, "%s", what);
}

void Histogram::Print(const char* what, size_t aligned) {
  int remain = aligned - (int)strlen(what);
  for (int i = 0; i < remain; i++) Print(" ");
  Print(what);
}

void Histogram::Print(const char* format, double value, size_t aligned) {
  static const size_t kMaxBufferSize = 32;
  aligned = std::min(aligned, kMaxBufferSize);
  char buffer[kMaxBufferSize];
  snprintf(buffer, aligned, format, value);
  Print(buffer, aligned);
}
