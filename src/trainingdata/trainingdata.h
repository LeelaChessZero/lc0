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

#include "neural/backend.h"
#include "search/classic/node.h"
#include "trainingdata/writer.h"
#include "trainingdata/trainingdata_v6.h"

namespace lczero {

class V6TrainingDataArray {
 public:
  V6TrainingDataArray(FillEmptyHistory white_fill_empty_history,
                      FillEmptyHistory black_fill_empty_history,
                      pblczero::NetworkFormat::InputFormat input_format)
      : fill_empty_history_{white_fill_empty_history, black_fill_empty_history},
        input_format_(input_format) {}

  // Add a chunk.
  void Add(const classic::Node* node, const PositionHistory& history,
           classic::Eval best_eval, classic::Eval played_eval,
           bool best_is_proven, Move best_move, Move played_move,
           std::span<Move> legal_moves,
           const std::optional<EvalResult>& nneval, float policy_softmax_temp);

  // Writes training data to a file.
  void Write(TrainingDataWriter* writer, GameResult result,
             bool adjudicated) const;

 private:
  std::vector<V6TrainingData> training_data_;
  FillEmptyHistory fill_empty_history_[2];
  pblczero::NetworkFormat::InputFormat input_format_;
};

}  // namespace lczero
