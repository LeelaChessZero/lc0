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
#include <vector>

namespace lczero {
namespace metal_backend {

static int kNumOutputPolicy = 1858;
static int kInputPlanes = 112;

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left, bool conv_policy, bool attn_policy) {
    input_masks_mem_.reserve(maxBatchSize * kInputPlanes);
    input_val_mem_.reserve(maxBatchSize * kInputPlanes);
    input_val_mem_expanded_.reserve(maxBatchSize * kInputPlanes * 64);
    op_policy_mem_.reserve(maxBatchSize * kNumOutputPolicy);
    op_value_mem_.reserve(maxBatchSize * (wdl ? 3 : 1));

    if (moves_left) {
      op_moves_left_mem_.reserve(maxBatchSize);
    };

    /**
     * @todo policy map implementation has bug in MPSGraph (GatherND not working in graph).
     * Implementation of policy map to be done in CPU for now.
     *
     * Remove this op_policy_raw_mem_ memory allocation when bug is fixed.
     */
    if (attn_policy) {
      op_policy_raw_mem_.reserve(maxBatchSize * (64 * 64 + 8 * 24));
    }
    else if (conv_policy) {
      op_policy_raw_mem_.reserve(maxBatchSize * 73 * 64);
    }
  }
  ~InputsOutputs() {}

  std::vector<uint64_t> input_masks_mem_;
  std::vector<float> input_val_mem_;
  std::vector<float> input_val_mem_expanded_;
  std::vector<float> op_policy_mem_;
  std::vector<float> op_value_mem_;
  std::vector<float> op_moves_left_mem_;
  std::vector<float> op_policy_raw_mem_;
};

}  // namespace metal_backend
}  // namespace lczero
