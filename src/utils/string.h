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

#include <string>
#include <vector>

namespace lczero {

// Joins strings using @delim as delimiter.
std::string StrJoin(const std::vector<std::string>& strings,
                    const std::string& delim = " ");

// Splits strings at whitespace.
std::vector<std::string> StrSplitAtWhitespace(const std::string& str);

// Split string by delimiter.
std::vector<std::string> StrSplit(const std::string& str,
                                  const std::string& delim);

// Parses comma-separated list of integers.
std::vector<int> ParseIntList(const std::string& str);

// Trims a string of whitespace from the start.
std::string LeftTrim(std::string str);

// Trims a string of whitespace from the end.
std::string RightTrim(std::string str);

// Trims a string of whitespace from both ends.
std::string Trim(std::string str);

// Returns whether strings are equal, ignoring case.
bool StringsEqualIgnoreCase(const std::string& a, const std::string& b);

// Flow text into lines of width up to @width.
std::vector<std::string> FlowText(const std::string& src, size_t width);

}  // namespace lczero
