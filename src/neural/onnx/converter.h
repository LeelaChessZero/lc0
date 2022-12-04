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

#include "neural/onnx/onnx.pb.h"
#include "proto/net.pb.h"

namespace lczero {

// Options to use when converting "old" weights to ONNX weights format.
struct WeightsToOnnxConverterOptions {
  enum class DataType { kFloat32, kFloat16 };
  DataType data_type_ = DataType::kFloat32;
  std::string input_planes_name = "/input/planes";
  std::string output_policy_head = "/output/policy";
  std::string output_wdl = "/output/wdl";
  std::string output_value = "/output/value";
  std::string output_mlh = "/output/mlh";
  int batch_size = -1;
  int opset = 17;
};

// Converts "classical" weights file to weights file with embedded ONNX model.
pblczero::Net ConvertWeightsToOnnx(const pblczero::Net&,
                                   const WeightsToOnnxConverterOptions&);

}  // namespace lczero
