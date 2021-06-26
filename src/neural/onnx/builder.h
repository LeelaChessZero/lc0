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
#include <string>

#include "neural/onnx/onnx.pb.h"

namespace lczero {

class OnnxConst {
 public:
  virtual ~OnnxConst() = default;
  virtual pblczero::TensorProto::DataType GetDataType() const = 0;
  virtual std::vector<int> GetDimensions() const = 0;
  virtual std::string GetRawData() const = 0;
};

class OnnxBuilder {
 public:
  OnnxBuilder();
  void AddInput(const std::string& name, std::initializer_list<int> dims,
                pblczero::TensorProto::DataType datatype);
  void AddOutput(const std::string& name, std::initializer_list<int> dims,
                 pblczero::TensorProto::DataType datatype);

  std::string Conv(const std::string& name, const std::string& input_name,
                   const OnnxConst& kernel_weights,
                   const OnnxConst& bias_weights);
  std::string Add(const std::string& name, const std::string& input1,
                  const std::string& input2);
  std::string Add(const std::string& name, const std::string& input1,
                  const OnnxConst&);
  std::string GlobalAveragePool(const std::string& name,
                                const std::string& input);
  std::string Squeeze(const std::string& name, const std::string& input);
  std::string MatMul(const std::string& name, const std::string& input1,
                     const OnnxConst& input2);
  std::string MatMul(const std::string& name, const std::string& input1,
                     const std::string& input2);
  std::string Relu(const std::string& name, const std::string& input);
  std::string Tanh(const std::string& name, const std::string& input);
  std::string Softmax(const std::string& name, const std::string& input);
  std::string AddInitializer(const std::string& name, const OnnxConst& weights);
  std::string Reshape(const std::string& name, const std::string& input,
                      const std::string& shape);
  std::pair<std::string, std::string> Split(const std::string& name,
                                            const std::string& input, int axis);
  std::string Sigmoid(const std::string& name, const std::string& input);
  std::string Gather(const std::string& name, const std::string& input1,
                     const std::string& input2, int axis);

  const pblczero::ModelProto& as_proto() const { return model_; }
  std::string OutputAsString() const { return model_.OutputAsString(); }

 private:
  pblczero::ModelProto model_;
};

}  // namespace lczero