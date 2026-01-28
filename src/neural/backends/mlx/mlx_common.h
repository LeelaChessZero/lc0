/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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
namespace mlx_backend {

static constexpr int kNumOutputPolicy = 1858;
static constexpr int kInputPlanes = 112;

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left, bool conv_policy,
                bool attn_policy) {
    input_masks_mem_.resize(maxBatchSize * kInputPlanes);
    input_val_mem_.resize(maxBatchSize * kInputPlanes);
    op_policy_mem_.resize(maxBatchSize * kNumOutputPolicy);
    op_value_mem_.resize(maxBatchSize * (wdl ? 3 : 1));

    if (moves_left) {
      op_moves_left_mem_.resize(maxBatchSize);
    }

    // Policy map implementation - raw policy before mapping.
    if (attn_policy) {
      op_policy_raw_mem_.resize(maxBatchSize * (64 * 64 + 8 * 24));
    } else if (conv_policy) {
      op_policy_raw_mem_.resize(maxBatchSize * 73 * 64);
    }
  }
  ~InputsOutputs() = default;

  std::vector<uint64_t> input_masks_mem_;
  std::vector<float> input_val_mem_;
  std::vector<float> op_policy_mem_;
  std::vector<float> op_value_mem_;
  std::vector<float> op_moves_left_mem_;
  std::vector<float> op_policy_raw_mem_;
};

// Activation configuration for the network.
struct Activations {
  std::string default_activation = "relu";
  std::string smolgen_activation = "swish";
  std::string ffn_activation = "relu_2";
};

}  // namespace mlx_backend
}  // namespace lczero
