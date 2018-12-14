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

class CommandLine {
 public:
  CommandLine() = delete;

  // This function must be called before any other.
  static void Init(int argc, const char** argv);

  // Name of the executable filename that was run.
  static const std::string& BinaryName() { return binary_; }

  // Directory where the binary is run. Without trailing slash.
  static std::string BinaryDirectory();

  // If the first command line parameter is @command, remove it and return
  // true. Otherwise return false.
  static bool ConsumeCommand(const std::string& command);

  // Command line arguments.
  static const std::vector<std::string>& Arguments() { return arguments_; }

  static void RegisterMode(const std::string& mode,
                           const std::string& description);

  static const std::vector<std::pair<std::string, std::string>>& GetModes() {
    return modes_;
  }

 private:
  static std::string binary_;
  static std::vector<std::string> arguments_;
  static std::vector<std::pair<std::string, std::string>> modes_;
};

}  // namespace lczero
