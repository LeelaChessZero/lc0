/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors

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

#include "trainingdata/writer.h"

#include "trainingdata/trainingdata.h"
#include "utils/exception.h"
#include "utils/filesystem.h"
#include "utils/random.h"

namespace lczero {
namespace {
std::string GetLc0CacheDirectory() {
  std::string user_cache_path = GetUserCacheDirectory();
  if (!user_cache_path.empty()) {
    user_cache_path += "lc0/";
    CreateDirectory(user_cache_path);
  }
  return user_cache_path;
}

}  // namespace

TrainingDataWriter::TrainingDataWriter(int game_id) {
  static std::string directory =
      GetLc0CacheDirectory() + "data-" + Random::Get().GetString(12);
  // It's fine if it already exists.
  CreateDirectory(directory.c_str());

  std::ostringstream oss;
  oss << directory << '/' << "game_" << std::setfill('0') << std::setw(6)
      << game_id << ".gz";

  filename_ = oss.str();
  fout_ = gzopen(filename_.c_str(), "wb");
  if (!fout_) throw Exception("Cannot create gzip file " + filename_);
}

TrainingDataWriter::TrainingDataWriter(std::string filename)
    : filename_(filename) {
  fout_ = gzopen(filename_.c_str(), "wb");
  if (!fout_) throw Exception("Cannot create gzip file " + filename_);
}

void TrainingDataWriter::WriteChunk(const V6TrainingData& data) {
  auto bytes_written =
      gzwrite(fout_, reinterpret_cast<const char*>(&data), sizeof(data));
  if (bytes_written != sizeof(data)) {
    throw Exception("Unable to write into " + filename_);
  }
}

void TrainingDataWriter::Finalize() {
  gzclose(fout_);
  fout_ = nullptr;
}

}  // namespace lczero
