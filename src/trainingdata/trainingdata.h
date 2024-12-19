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

#include "search/classic/node.h"
#include "trainingdata/writer.h"

namespace lczero {

#pragma pack(push, 1)

struct V6TrainingData {
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
  // Bitfield with the following allocation:
  //  bit 7: side to move (input type 3)
  //  bit 6: position marked for deletion by the rescorer (never set by lc0)
  //  bit 5: game adjudicated (v6)
  //  bit 4: max game length exceeded (v6)
  //  bit 3: best_q is for proven best move (v6)
  //  bit 2: transpose transform (input type 3)
  //  bit 1: mirror transform (input type 3)
  //  bit 0: flip transform (input type 3)
  // In versions prior to v5 this spot contained an unused move count field.
  uint8_t invariance_info;
  // In versions prior to v6 this spot contained thr result as an int8_t.
  uint8_t dummy;
  float root_q;
  float best_q;
  float root_d;
  float best_d;
  float root_m;      // In plies.
  float best_m;      // In plies.
  float plies_left;  // This is the training target for MLH.
  float result_q;
  float result_d;
  float played_q;
  float played_d;
  float played_m;
  // The folowing may be NaN if not found in cache.
  float orig_q;  // For value repair.
  float orig_d;
  float orig_m;
  uint32_t visits;
  // Indices in the probabilities array.
  uint16_t played_idx;
  uint16_t best_idx;
  // Kullback-Leibler divergence between visits and policy (denominator)
  float policy_kld;
  uint32_t reserved;
} PACKED_STRUCT;
static_assert(sizeof(V6TrainingData) == 8356, "Wrong struct size");

#pragma pack(pop)

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
           const NNCacheLock& nneval);

  // Writes training data to a file.
  void Write(TrainingDataWriter* writer, GameResult result,
             bool adjudicated) const;

 private:
  std::vector<V6TrainingData> training_data_;
  FillEmptyHistory fill_empty_history_[2];
  pblczero::NetworkFormat::InputFormat input_format_;
};

}  // namespace lczero
