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

#include "neural/xla/onnx_builder_proto_utils.h"

namespace lczero {

TensorType::ElementType ElementTypeFromProto(
    pblczero::XlaShapeProto::Type element_type) {
  switch (element_type) {
    case pblczero::XlaShapeProto::F16:
      return TensorType::ElementType::kF16;
    case pblczero::XlaShapeProto::BF16:
      return TensorType::ElementType::kBF16;
    case pblczero::XlaShapeProto::F32:
      return TensorType::ElementType::kF32;
    case pblczero::XlaShapeProto::S32:
      return TensorType::ElementType::kS32;
    case pblczero::XlaShapeProto::S64:
      return TensorType::ElementType::kS64;
    case pblczero::XlaShapeProto::PRED:
      return TensorType::ElementType::kPred;
    default:
      return TensorType::ElementType::kInvalid;
  }
}

pblczero::XlaShapeProto::Type ElementTypeToProto(
    TensorType::ElementType element_type) {
  switch (element_type) {
    case TensorType::ElementType::kF16:
      return pblczero::XlaShapeProto::F16;
    case TensorType::ElementType::kBF16:
      return pblczero::XlaShapeProto::BF16;
    case TensorType::ElementType::kF32:
      return pblczero::XlaShapeProto::F32;
    case TensorType::ElementType::kS32:
      return pblczero::XlaShapeProto::S32;
    case TensorType::ElementType::kS64:
      return pblczero::XlaShapeProto::S64;
    case TensorType::ElementType::kPred:
      return pblczero::XlaShapeProto::PRED;
    default:
      return pblczero::XlaShapeProto::PRIMITIVE_TYPE_INVALID;
  }
}

TensorType TensorTypeFromProto(const pblczero::XlaShapeProto& shape) {
  TensorType result;
  result.element_type = ElementTypeFromProto(shape.element_type());
  result.dimensions.reserve(shape.dimensions_size());
  for (const int64_t dim : shape.dimensions()) {
    result.dimensions.push_back(dim);
  }
  return result;
}

pblczero::XlaShapeProto TensorTypeToProto(const TensorType& type) {
  pblczero::XlaShapeProto result;
  result.set_element_type(ElementTypeToProto(type.element_type));
  result.mutable_layout();
  const size_t rank = type.dimensions.size();
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim = type.dimensions[i];
    result.add_dimensions(dim);
    result.add_is_dynamic_dimension(false);
    result.mutable_layout()->add_minor_to_major(rank - i - 1);
  }
  return result;
}

}  // namespace lczero
