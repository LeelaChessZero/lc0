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

#include <mlx/mlx.h>

#include <functional>
#include <variant>
#include <vector>

namespace lczero {
namespace mlx_backend {

namespace mx = mlx::core;

// Precision configuration for compute and storage.
enum class Precision { FP32, FP16, BF16, Q8 };

// Convert Precision enum to MLX dtype for non-quantized weights.
inline mx::Dtype PrecisionToDtype(Precision p) {
  switch (p) {
    case Precision::FP16:
      return mx::float16;
    case Precision::BF16:
      return mx::bfloat16;
    default:
      return mx::float32;
  }
}

// Check if precision is quantized.
inline bool IsQuantized(Precision p) { return p == Precision::Q8; }

// Get activation dtype for quantized inference.
// Quantized weights use float16 activations for efficiency.
inline mx::Dtype ActivationDtype(Precision p) {
  return IsQuantized(p) ? mx::float16 : PrecisionToDtype(p);
}

// Quantized weight storage for int8 (or int4) quantization.
// Holds packed quantized weights, per-group scales, and biases.
struct QuantizedWeight {
  mx::array packed;     // Packed quantized weights (uint32)
  mx::array scales;     // Per-group scale factors
  mx::array biases;     // Per-group bias values
  int group_size;       // Number of elements per quantization group
  int bits;             // Quantization bits (8 or 4)

  QuantizedWeight(mx::array p, mx::array s, mx::array b, int gs, int bt)
      : packed(std::move(p)),
        scales(std::move(s)),
        biases(std::move(b)),
        group_size(gs),
        bits(bt) {}
};

// A weight can be either quantized or a float array reference.
// Using variant ensures exactly one must be present (no null pointers).
using WeightVariant =
    std::variant<QuantizedWeight, std::reference_wrapper<const mx::array>>;

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

// Epsilon values for layer normalization.
// Smolgen layers always use 1e-3 epsilon.
static constexpr float kSmolgenEpsilon = 1e-3f;
// Default epsilon for older networks.
static constexpr float kDefaultEpsilon = 1e-6f;
// PE_DENSE networks use 1e-3 epsilon.
static constexpr float kPeDenseEpsilon = 1e-3f;

// Default quantization parameters.
static constexpr int kDefaultQuantizationGroupSize = 64;
static constexpr int kDefaultQuantizationBits = 8;

// Activation configuration for the network.
struct Activations {
  std::string default_activation = "relu";
  std::string smolgen_activation = "swish";
  std::string ffn_activation = "relu_2";
};

}  // namespace mlx_backend
}  // namespace lczero
