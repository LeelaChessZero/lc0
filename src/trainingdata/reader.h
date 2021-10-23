/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "trainingdata/trainingdata.h"

namespace lczero {

// Constructs InputPlanes from training data.
//
// NOTE: If the training data is a cannonical type, the canonicalization
// transforms are reverted before returning, since it is assumed that the data
// will be used with DecodeMoveFromInput or PopulateBoard which assume the
// InputPlanes are not transformed.
InputPlanes PlanesFromTrainingData(const V6TrainingData& data);

class TrainingDataReader {
 public:
  // Opens the given file to read chunk data from.
  TrainingDataReader(std::string filename);

  ~TrainingDataReader();

  // Reads a chunk. Returns true if a chunk was read.
  bool ReadChunk(V6TrainingData* data);

  // Gets full filename of the file being read.
  std::string GetFileName() const { return filename_; }

 private:
  std::string filename_;
  gzFile fin_;
  bool format_v6 = false;
};

}  // namespace lczero
