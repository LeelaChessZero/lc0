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

#include "neural/xla/onnx2hlo.h"

#include <unordered_map>

#include "neural/onnx/onnx.pb.h"
#include "neural/xla/hlo.pb.h"
#include "neural/xla/hlo_builder.h"
#include "utils/exception.h"

namespace lczero {
namespace {

pblczero::XlaShapeProto::Type OnnxTypeToXlaType(
    const pblczero::TensorProto::DataType& type) {
  switch (type) {
    case pblczero::TensorProto::FLOAT:
      return pblczero::XlaShapeProto::F32;
    case pblczero::TensorProto::UINT8:
      return pblczero::XlaShapeProto::U8;
    case pblczero::TensorProto::INT8:
      return pblczero::XlaShapeProto::S8;
    case pblczero::TensorProto::UINT16:
      return pblczero::XlaShapeProto::U16;
    case pblczero::TensorProto::INT16:
      return pblczero::XlaShapeProto::S16;
    case pblczero::TensorProto::INT32:
      return pblczero::XlaShapeProto::S32;
    case pblczero::TensorProto::INT64:
      return pblczero::XlaShapeProto::S64;
    case pblczero::TensorProto::BOOL:
      return pblczero::XlaShapeProto::PRED;
    case pblczero::TensorProto::FLOAT16:
      return pblczero::XlaShapeProto::F16;
    case pblczero::TensorProto::DOUBLE:
      return pblczero::XlaShapeProto::F64;
    case pblczero::TensorProto::UINT32:
      return pblczero::XlaShapeProto::U32;
    case pblczero::TensorProto::UINT64:
      return pblczero::XlaShapeProto::U64;
    case pblczero::TensorProto::COMPLEX64:
      return pblczero::XlaShapeProto::C64;
    case pblczero::TensorProto::COMPLEX128:
      return pblczero::XlaShapeProto::C128;
    case pblczero::TensorProto::BFLOAT16:
      return pblczero::XlaShapeProto::BF16;
    case pblczero::TensorProto::STRING:
    case pblczero::TensorProto::UNDEFINED:
      throw Exception("Unsupported ONNX type " +
                      pblczero::TensorProto::DataType_Name(type));
  }
}

pblczero::XlaShapeProto OnnxShapeToXlaShape(
    const pblczero::TypeProto& type,
    std::optional<size_t> batch_size = std::nullopt) {
  pblczero::XlaShapeProto shape;
  shape.set_element_type(OnnxTypeToXlaType(type.tensor_type().elem_type()));
  for (const auto& dim : type.tensor_type().shape().dim()) {
    if (dim.has_dim_value()) {
      shape.add_dimensions(dim.dim_value());
      continue;
    }
    if (dim.dim_param() == "batch") {
      if (batch_size.has_value()) {
        shape.add_dimensions(batch_size.value());
        continue;
      }
      throw Exception("Batch size not provided");
    }
    throw Exception("Unsupported dimension type " + type.OutputAsJson());
  }
  return shape;
}

pblczero::XlaShapeProto OnnxTypeProtoToXlaShape(
    const pblczero::TypeProto& type) {}

class Onnx2HloConverter {
 public:
  Onnx2HloConverter(const Onnx2HloOptions& options) : options_(options) {}
  void Convert(const pblczero::ModelProto& onnx_model, size_t minibatch_size) {
    BuildInputs(onnx_model.graph().input());
  }

 private:
  void BuildInputs(const std::vector<pblczero::ValueInfoProto>& inputs) {
    for (const auto& input : inputs) {
      auto ctx = builder_.ScopedContext().OpType("input").OpName(input.name());
      auto out_shape = OnnxShapeToXlaShape(input.type());
      auto in_shape = out_shape;
      in_shape.set_element_type(options_.io_type);
      auto* flow = builder_.Parameter(in_shape);
      flow = builder_.Convert(flow, out_shape.element_type());
    }
  }

  std::unordered_map<std::string, size_t> onnx_to_xla_name_;
  HloBuilder builder_;
  Onnx2HloOptions options_;
};

}  // namespace

Onnx2HloResult ConvertOnnxToHlo(const pblczero::ModelProto& onnx_model,
                                size_t minibatch_size,
                                const Onnx2HloOptions& options) {
  Onnx2HloConverter converter(options);
  converter.Convert(onnx_model, minibatch_size);
}

}  // namespace lczero