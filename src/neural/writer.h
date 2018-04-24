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

#include <fstream>
#include "utils/cppattributes.h"

#pragma once

namespace lczero {

#pragma pack(push, 1)

struct V3TrainingData {
  uint32_t version;
  float probabilities[1858];
  uint64_t planes[104];
  uint8_t castling_us_ooo;
  uint8_t castling_us_oo;
  uint8_t castling_them_ooo;
  uint8_t castling_them_oo;
  uint8_t side_to_move;
  uint8_t move_count;
  uint8_t rule50_count;
  int8_t result;
} PACKED_STRUCT;
static_assert(sizeof(V3TrainingData) == 8276, "Wrong struct size");

#pragma pack(pop)

class TrainingDataWriter {
 public:
  // Creates a new file to write in data directory. It will has @game_id
  // somewhere in the filename.
  TrainingDataWriter(int game_id);

  // Writes a chunk.
  void WriteChunk(const V3TrainingData& data);

  // Flushes file and closes it.
  void Finalize() { fout_.close(); }

  // Gets full filename of the file written.
  std::string GetFileName() const { return filename_; }

 private:
  std::string filename_;
  std::ofstream fout_;
};

}  // namespace lczero