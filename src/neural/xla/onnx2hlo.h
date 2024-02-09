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
  size_t max_inline_constant_size = 1024;
  pblczero::XlaShapeProto::Type io_type = pblczero::XlaShapeProto::F32;
};

struct Onnx2HloResult {
  struct NamedTensor {
    size_t param_idx;
    std::string name;
    pblczero::XlaShapeProto shape;
  };
  std::vector<NamedTensor> constants;
  std::vector<NamedTensor> inputs;
  std::vector<NamedTensor> outputs;
  pblczero::HloModuleProto hlo_module;
};

Onnx2HloResult ConvertOnnxToHlo(const pblczero::ModelProto& onnx_model,
                                size_t minibatch_size,
                                const Onnx2HloOptions& options);

XlaTensor OnnxConstantToXlaTensor(const pblczero::ModelProto& onnx_model,
                                  std::string_view name);

}  // namespace lczero
