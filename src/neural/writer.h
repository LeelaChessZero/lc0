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

#include <zlib.h>

#include <fstream>

#include "utils/cppattributes.h"

#pragma once

namespace lczero {

#pragma pack(push, 1)

struct V5TrainingData {
  uint32_t version;
  uint32_t input_format;
  float probabilities[1858];
  uint64_t planes[104];
  uint8_t castling_us_ooo;
  uint8_t castling_us_oo;
  uint8_t castling_them_ooo;
  uint8_t castling_them_oo;
  // For input type 3 contains enpassant column as a mask.
  uint8_t side_to_move_or_enpassant;
  uint8_t rule50_count;
  // For input type 3 contains a bit field indicating the transform that was
  // used and the original side to move info.
  // Side to move is in the top bit, transform in the lower bits.
  // In versions prior to v5 this spot contained an unused move count field.
  uint8_t invariance_info;
  int8_t result;
  float root_q;
  float best_q;
  float root_d;
  float best_d;
  float root_m;      // In plies.
  float best_m;      // In plies.
  float plies_left;  // This is the training target for MLH.
} PACKED_STRUCT;
static_assert(sizeof(V5TrainingData) == 8308, "Wrong struct size");

#pragma pack(pop)

class TrainingDataWriter {
 public:
  // Creates a new file to write in data directory. It will has @game_id
  // somewhere in the filename.
  TrainingDataWriter(int game_id);

  ~TrainingDataWriter() {
    if (fout_) Finalize();
  }

  // Writes a chunk.
  void WriteChunk(const V5TrainingData& data);

  // Flushes file and closes it.
  void Finalize();

  // Gets full filename of the file written.
  std::string GetFileName() const { return filename_; }

 private:
  std::string filename_;
  gzFile fout_;
};

}  // namespace lczero
