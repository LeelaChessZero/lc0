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

#include "neural/writer.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include "utils/commandline.h"
#include "utils/exception.h"
#include "utils/filesystem.h"
#include "utils/random.h"

namespace lczero {
namespace {
// Reverse bits in every byte of a number
uint64_t ReverseBitsInBytes(uint64_t v) {
  v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
  v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
  v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
  return v;
}
}  // namespace

InputPlanes PlanesFromTrainingData(const V5TrainingData& data) {
  InputPlanes result;
  if (data.input_format !=
      pblczero::NetworkFormat::InputFormat::INPUT_CLASSICAL_112_PLANE) {
    throw Exception("FRC input variant not yet supported.");
  }
  for (int i = 0; i < 104; i++) {
    result.emplace_back();
    result.back().mask = ReverseBitsInBytes(data.planes[i]);
  }
  result.emplace_back();
  result.back().mask = data.castling_us_ooo != 0 ? ~0LL : 0LL;
  result.emplace_back();
  result.back().mask = data.castling_us_oo != 0 ? ~0LL : 0LL;
  result.emplace_back();
  result.back().mask = data.castling_them_ooo != 0 ? ~0LL : 0LL;
  result.emplace_back();
  result.back().mask = data.castling_them_oo != 0 ? ~0LL : 0LL;
  result.emplace_back();
  result.back().mask = data.side_to_move != 0 ? ~0LL : 0LL;
  result.emplace_back();
  result.back().Fill(data.rule50_count);
  result.emplace_back();
  // Empty plane.
  result.emplace_back();
  // All ones plane.
  result.back().SetAll();
  return result;
}

TrainingDataReader::TrainingDataReader(std::string filename)
    : filename_(filename) {
  fin_ = gzopen(filename_.c_str(), "rb");
  if (!fin_) {
    throw Exception("Cannot open gzip file " + filename_);
  }
}

TrainingDataReader::~TrainingDataReader() { gzclose(fin_); }

bool TrainingDataReader::ReadChunk(V5TrainingData* data) {
  if (format_v5) {
    int read_size = gzread(fin_, reinterpret_cast<void*>(data), sizeof(*data));
    if (read_size < 0) throw Exception("Corrupt read.");
    return read_size == sizeof(*data);
  } else {
    size_t v5_extra = 16;
    size_t v4_extra = 16;
    size_t v3_size = sizeof(*data) - v4_extra - v5_extra;
    int read_size = gzread(fin_, reinterpret_cast<void*>(data), v3_size);
    if (read_size < 0) throw Exception("Corrupt read.");
    if (read_size != v3_size) return false;
    auto orig_version = data->version;
    switch (data->version) {
      case 3: {
        data->version = 4;
        // First convert 3 to 4 to reduce code duplication.
        char* v4_extra_start = reinterpret_cast<char*>(data) + v3_size;
        // Write 0 bytes for 16 extra bytes - corresponding to 4 floats of 0.0f.
        for (int i = 0; i < v4_extra; i++) {
          v4_extra_start[i] = 0;
        }
        // Deliberate fallthrough.
      }
      case 4: {
        // If actually 4, we need to read the additional data first.
        if (orig_version == 4) {
          read_size = gzread(
              fin_,
              reinterpret_cast<void*>(reinterpret_cast<char*>(data) + v3_size),
              v4_extra);
          if (read_size < 0) throw Exception("Corrupt read.");
          if (read_size != v4_extra) return false;
        }
        data->version = 5;
        char* data_ptr = reinterpret_cast<char*>(data);
        // Shift data after version back 4 bytes.
        memmove(data_ptr + 2 * sizeof(uint32_t), data_ptr + sizeof(uint32_t),
                v3_size + v4_extra - sizeof(uint32_t));
        data->input_format = pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE;
        data->root_m = 0.0f;
        data->best_m = 0.0f;
        data->plies_left = 0.0f;
        return true;
      }
      case 5: {
        format_v5 = true;
        read_size = gzread(
            fin_,
            reinterpret_cast<void*>(reinterpret_cast<char*>(data) + v3_size),
            v4_extra + v5_extra);
        if (read_size < 0) throw Exception("Corrupt read.");
        return read_size == v4_extra + v5_extra;
      }
    }
  }
}

TrainingDataWriter::TrainingDataWriter(int game_id) {
  static std::string directory =
      CommandLine::BinaryDirectory() + "/data-" + Random::Get().GetString(12);
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

void TrainingDataWriter::WriteChunk(const V5TrainingData& data) {
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
