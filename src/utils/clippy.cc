/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "utils/clippy.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

namespace lczero {

// This is based on a vibe coded first draft, handle with care.
void Clippy(const std::string& formatted) {
  // Split into lines.
  std::vector<std::string> lines;
  std::istringstream ss(formatted);
  std::string ln;
  while (std::getline(ss, ln)) lines.push_back(ln);

  size_t maxlen = 0;
  for (const auto& l : lines) maxlen = std::max(maxlen, l.size());
  // content padding: one space on each side
  const size_t bubble_inner =
      maxlen + 2;  // inner width used for top/side calculations

  // Build speech bubble:
  std::vector<std::string> bubble;
  bubble.push_back("  " + std::string(bubble_inner, '_') + " ");
  bubble.push_back((lines.size() == 1 ? "_/" : " /") +
                   std::string(bubble_inner, ' ') + "\\");
  for (size_t i = 0; const auto& l : lines) {
    std::string header;
    header = i == lines.size() / 2 - 1 + (lines.size() > 4) ? "_| "
             : i == lines.size() / 2 + (lines.size() > 4)   ? "\\  "
                                                            : " | ";
    bubble.push_back(header + l +
                     std::string(bubble_inner - 2 - l.size(), ' ') + " |");
    i++;
  }
  bubble.push_back(" \\" + std::string(bubble_inner, '_') + "/");

  // Clippy ASCII art.
  std::vector<std::string> clippy = {"  __",    " /  \\", " |  |",  " +  +",
                                     "(@)(@)",  " |  |",  " || |/", " || ||",
                                     " |\\_/|", " \\___/"};
  size_t clippy_width = 0;
  for (const auto& c : clippy) clippy_width = std::max(clippy_width, c.size());
  clippy_width++;  // One space extra padding.
  const size_t rows = std::max(clippy.size(), bubble.size());
  // Pad one side's top as needed to center spech bubble to Clippy's edge.
  if (clippy.size() < bubble.size()) {
    clippy.insert(clippy.begin(), (bubble.size() - clippy.size() + 1) / 2, "");
  } else if (clippy.size() > bubble.size()) {
    bubble.insert(bubble.begin(),
                  (clippy.size() - bubble.size()) / 2 + (lines.size() <= 4),
                  "");
  }
  // Print side-by-side, aligning rows.
  for (size_t i = 0; i < rows; ++i) {
    std::string left;
    if (i < clippy.size()) {
      left = clippy[i];
    }
    // pad left to fixed width
    if (left.size() < clippy_width) {
      left += std::string(clippy_width - left.size(), ' ');
    }
    std::string right;
    if (i < bubble.size()) {
      right = bubble[i];
    }
    std::cout << left << right << std::endl;
  }
}

}  // namespace lczero
