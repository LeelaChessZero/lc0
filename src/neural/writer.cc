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

#include "neural/writer.h"

#include <experimental/filesystem>
#include <iomanip>
#include <sstream>
#include "utils/commandline.h"
#include "utils/random.h"

namespace lczero {

TrainingDataWriter::TrainingDataWriter(int game_id) {
  using namespace std::experimental::filesystem;
  static std::string directory =
      CommandLine::BinaryDirectory() + "/data-" + Random::Get().GetString(12);
  // It's fine if it already exists.
  create_directories(directory.c_str());

  std::ostringstream oss;
  oss << directory << '/' << "game_" << std::setfill('0') << std::setw(6)
      << game_id;

  filename_ = oss.str();
  fout_.open(filename_.c_str(), std::ios::binary);
}

void TrainingDataWriter::WriteChunk(const V3TrainingData& data) {
  fout_.write(reinterpret_cast<const char*>(&data), sizeof(data));
}

}  // namespace lczero