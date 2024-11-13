/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include <string_view>
#include <vector>

#include "neural/onnx/onnx.pb.h"
#include "neural/xla/hlo.pb.h"
#include "neural/xla/xla_runner.h"

namespace lczero {

struct Onnx2HloOptions {
  // Constants larger that this size in bytes will be passed as parameters
  // instead. This allows them to be shared between different modules.
  size_t max_inline_constant_size = 1024;
  // It set, ensures that the input/output tensors have given type (does not
  // affect constants passed as parameters).
  std::optional<pblczero::XlaShapeProto::Type> io_type = std::nullopt;
  // If error occurs during conversion, return a partial result instead of
  // failing. Only to be used for debugging.
  bool debugging_allow_partial_result = false;
  // If not empty, uses these nodes as outputs instead of the ones from the ONNX
  // model.
  std::vector<std::string> outputs_override;
};

struct Onnx2HloResult {
  struct NamedTensor {
    // Index of the tensor in the input or output tuple.
    size_t param_idx;
    // Name of the tensor from the ONNX model.
    std::string name;
    pblczero::XlaShapeProto shape;
  };
  // Constants that are passed as inputs to the module.
  std::vector<NamedTensor> constants;
  std::vector<NamedTensor> inputs;
  std::vector<NamedTensor> outputs;
  pblczero::HloModuleProto hlo_module;
};

// Converts an ONNX model to an HLO module.
Onnx2HloResult ConvertOnnxToHlo(const pblczero::ModelProto& onnx_model,
                                size_t minibatch_size,
                                const Onnx2HloOptions& options);

// Converts an ONNX tensor to an XLA tensor (thanks GitHub Copilot for the
// comment suggestion).
std::unique_ptr<XlaTensor> OnnxTensorToXlaTensor(
    const pblczero::TensorProto& onnx_tensor);

}  // namespace lczero
