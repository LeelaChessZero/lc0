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

#include <initializer_list>

#include "neural/onnx/onnx.pb.h"

namespace lczero {

class OnnxWeights {
 public:
  virtual ~OnnxWeights() = default;
  virtual pblczero::TensorProto::DataType GetDataType() const = 0;
  virtual std::vector<int> GetDimensions() const = 0;
  virtual std::string GetRawData() const = 0;
};

class OnnxBuilder {
 public:
  OnnxBuilder();
  void AddInput(const std::string& name, std::initializer_list<int> dims,
                pblczero::TensorProto::DataType datatype);

  std::string AddConvLayer(const std::string& input_name,
                           const std::string& name,
                           const OnnxWeights& kernel_weights,
                           const OnnxWeights& bias_weights);

  const pblczero::ModelProto& as_proto() const { return model_; }
  std::string OutputAsString() const { return model_.OutputAsString(); }

 private:
  std::string AddInitializer(const std::string& name,
                             const OnnxWeights& weights);
  pblczero::ModelProto model_;
};

}  // namespace lczero