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

#include <string>
#include <vector>

namespace lczero {

class CommandLine {
 public:
  CommandLine() = delete;

  static const std::string& BinaryName() { return binary_; }
  static const std::vector<std::string>& Arguments() { return arguments_; }
  static void Init(int argc, const char** argv);

 private:
  static std::string binary_;
  static std::vector<std::string> arguments_;
};

}  // namespace lczero