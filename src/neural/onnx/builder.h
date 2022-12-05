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

// Builds Onnx::ModelProto.
class OnnxBuilder {
 public:
  OnnxBuilder(int opset);
  void AddInput(const std::string& name, std::initializer_list<int> dims,
                pblczero::TensorProto::DataType datatype);
  void AddOutput(const std::string& name, std::initializer_list<int> dims,
                 pblczero::TensorProto::DataType datatype);

  // Functions to add operators.
  // Every function appends one node to the graph.
  // @name parameter is used to name both the added node, and the output edge.
  // @input{,1,2} input tensor.
  //     - if std::string, contains the name of input tensor.
  //     - if OnnxConst, adds it as initializer, and then uses.
  // Return the name of the output edge, which is in most cases the same as the
  // node name.
  std::string Conv(const std::string& name, const std::string& input_name,
                   const OnnxConst& kernel_weights,
                   const OnnxConst& bias_weights, int pads = 1);
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
  std::string Mul(const std::string& name, const std::string& input1,
                  const std::string& input2);
  std::string Mul(const std::string& name, const std::string& input1,
                  const OnnxConst&);
  std::string Relu(const std::string& name, const std::string& input);
  std::string Tanh(const std::string& name, const std::string& input);
  std::string Softmax(const std::string& name, const std::string& input,
                      int axis = 1);
  std::string AddInitializer(const std::string& name, const OnnxConst& weights);
  std::string Reshape(const std::string& name, const std::string& input,
                      const std::string& shape);
  std::vector<std::string> Split(const std::string& name,
                                 const std::string& input, int axis,
                                 std::initializer_list<int> split = {});
  std::string Sigmoid(const std::string& name, const std::string& input);
  std::string Gather(const std::string& name, const std::string& input1,
                     const std::string& input2, int axis);
  std::string Softplus(const std::string& name, const std::string& input);
  std::string Identity(const std::string& name, const std::string& input);
  std::string Transpose(const std::string& name, const std::string& input,
                        std::initializer_list<int> perm = {});
  std::string Pad(const std::string& name, const std::string& input,
                  std::initializer_list<int> pads);
  std::string Selu(const std::string& name, const std::string& input);
  std::string Slice(const std::string& name, const std::string& input,
                    std::initializer_list<int> starts,
                    std::initializer_list<int> ends);
  std::string Concat(const std::string& name,
                     const std::vector<std::string>& input, int axis);
  std::string LayerNormalization(const std::string& name,
                                 const std::string& input,
                                 const OnnxConst& scale, const OnnxConst& bias,
                                 int axis, float epsilon = 1e-6);
  // Returns ONNX model as protobuf.
  const pblczero::ModelProto& as_proto() const { return model_; }
  // Returns serialized model.
  std::string OutputAsString() const { return model_.OutputAsString(); }

 private:
  const int opset_;
  pblczero::ModelProto model_;
};

}  // namespace lczero
