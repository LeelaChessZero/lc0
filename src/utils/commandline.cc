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

#include "utils/commandline.h"
#include "utils/logging.h"

namespace lczero {

std::string CommandLine::binary_;
std::vector<std::string> CommandLine::arguments_;
std::vector<std::pair<std::string, std::string>> CommandLine::modes_;

void CommandLine::Init(int argc, const char** argv) {
  binary_ = argv[0];
  arguments_.clear();
  std::ostringstream params;
  for (int i = 1; i < argc; ++i) {
    params << ' ' << argv[i];
    arguments_.push_back(argv[i]);
  }
  LOGFILE << "Command line: " << binary_ << params.str();
}

bool CommandLine::ConsumeCommand(const std::string& command) {
  if (arguments_.empty()) return false;
  if (arguments_[0] != command) return false;
  arguments_.erase(arguments_.begin());
  return true;
}

void CommandLine::RegisterMode(const std::string& mode,
                               const std::string& description) {
  modes_.emplace_back(mode, description);
}

std::string CommandLine::BinaryDirectory() {
  std::string path = binary_;
  const auto pos = path.find_last_of("\\/");
  if (pos == std::string::npos) {
    return ".";
  }
  path.resize(pos);
  return path;
}

}  // namespace lczero
