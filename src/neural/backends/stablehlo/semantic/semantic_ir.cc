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

#include "neural/backends/stablehlo/semantic/semantic_ir.h"

#include <algorithm>

#include "neural/xla/tensor_literal_utils.h"
#include "utils/exception.h"

namespace lczero::stablehlo::semantic {

namespace {

std::string FormatPermutation(const std::vector<int64_t>& permutation) {
  std::string result = "[";
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (i > 0) result += ",";
    result += std::to_string(permutation[i]);
  }
  result += "]";
  return result;
}

ElementType ToSemanticElement(::lczero::TensorType::ElementType element_type) {
  switch (element_type) {
    case ::lczero::TensorType::ElementType::kPred:
      return ElementType::kPred;
    case ::lczero::TensorType::ElementType::kF16:
      return ElementType::kF16;
    case ::lczero::TensorType::ElementType::kBF16:
      return ElementType::kBF16;
    case ::lczero::TensorType::ElementType::kF32:
      return ElementType::kF32;
    case ::lczero::TensorType::ElementType::kS32:
      return ElementType::kS32;
    case ::lczero::TensorType::ElementType::kS64:
      return ElementType::kS64;
    case ::lczero::TensorType::ElementType::kInvalid:
    default:
      return ElementType::kInvalid;
  }
}

::lczero::TensorType::ElementType ToBuilderElement(ElementType element_type) {
  switch (element_type) {
    case ElementType::kPred:
      return ::lczero::TensorType::ElementType::kPred;
    case ElementType::kF16:
      return ::lczero::TensorType::ElementType::kF16;
    case ElementType::kBF16:
      return ::lczero::TensorType::ElementType::kBF16;
    case ElementType::kF32:
      return ::lczero::TensorType::ElementType::kF32;
    case ElementType::kS32:
      return ::lczero::TensorType::ElementType::kS32;
    case ElementType::kS64:
      return ::lczero::TensorType::ElementType::kS64;
    case ElementType::kInvalid:
    default:
      return ::lczero::TensorType::ElementType::kInvalid;
  }
}

int64_t ComputeConvOutputDim(int64_t input_dim, int64_t kernel_dim,
                             int64_t stride, int64_t pad_low,
                             int64_t pad_high, int64_t lhs_dilation,
                             int64_t rhs_dilation) {
  if (stride <= 0 || lhs_dilation <= 0 || rhs_dilation <= 0) {
    throw Exception("Invalid non-positive convolution stride/dilation");
  }
  if (input_dim <= 0 || kernel_dim <= 0) {
    throw Exception("Invalid non-positive convolution dimension");
  }
  const int64_t dilated_input = (input_dim - 1) * lhs_dilation + 1;
  const int64_t dilated_kernel = (kernel_dim - 1) * rhs_dilation + 1;
  const int64_t numerator = pad_low + dilated_input + pad_high - dilated_kernel;
  if (numerator < 0) {
    throw Exception("Convolution output dimension computed as negative");
  }
  return numerator / stride + 1;
}

}  // namespace

SemanticBuilder::SemanticBuilder() { EnsureEntryFunction(); }

void SemanticBuilder::EnsureEntryFunction() {
  if (!module_.functions.empty()) return;
  SemanticFunction entry;
  entry.name = "main";
  module_.functions.push_back(std::move(entry));
  metadata_scopes_.push_back(Metadata{});
}

void SemanticBuilder::EnsureValueExists(::lczero::ValueId value) const {
  if (value >= value_types_.size()) {
    throw Exception("Invalid ValueId: " + std::to_string(value));
  }
}

TensorType SemanticBuilder::ToSemanticType(const ::lczero::TensorType& type) const {
  TensorType result;
  result.element_type = ToSemanticElement(type.element_type);
  result.dimensions = type.dimensions;
  return result;
}

::lczero::TensorType SemanticBuilder::ToBuilderType(const TensorType& type) const {
  ::lczero::TensorType result;
  result.element_type = ToBuilderElement(type.element_type);
  result.dimensions = type.dimensions;
  return result;
}

::lczero::ValueId SemanticBuilder::Emit(
    OpKind kind, const std::vector<::lczero::ValueId>& operands,
    const std::vector<TensorType>& result_types, AttrPayload attrs) {
  EnsureEntryFunction();
  for (const ::lczero::ValueId operand : operands) EnsureValueExists(operand);

  SemanticOp op;
  op.kind = kind;
  op.operands.assign(operands.begin(), operands.end());
  op.result_types = result_types;
  if (!metadata_scopes_.empty()) {
    op.source_op_type = metadata_scopes_.back().op_type;
    op.source_op_name = metadata_scopes_.back().op_name;
  }
  op.attrs = std::move(attrs);
  module_.functions.front().ops.push_back(std::move(op));

  if (result_types.empty()) return 0;
  const ::lczero::ValueId first_id = value_types_.size();
  for (const TensorType& result_type : result_types) {
    value_types_.push_back(result_type);
    value_kinds_.push_back(::lczero::BuilderOpKind::kUnknown);
    value_literals_.push_back(std::nullopt);
  }
  return first_id;
}

::lczero::ValueId SemanticBuilder::EmitElementwise(
    OpKind kind, ::lczero::ValueId lhs, ::lczero::ValueId rhs) {
  EnsureValueExists(lhs);
  EnsureValueExists(rhs);
  return Emit(kind, {lhs, rhs}, {value_types_[lhs]});
}

::lczero::ValueId SemanticBuilder::Parameter(const ::lczero::TensorType& shape) {
  EnsureEntryFunction();
  const TensorType type = ToSemanticType(shape);
  module_.functions.front().param_types.push_back(type);
  const ::lczero::ValueId id = value_types_.size();
  value_types_.push_back(type);
  value_kinds_.push_back(::lczero::BuilderOpKind::kParameter);
  value_literals_.push_back(std::nullopt);
  return id;
}

::lczero::ValueId SemanticBuilder::Constant(const ::lczero::TensorLiteral& literal) {
  const TensorType result_type = ToSemanticType(literal.type);
  const ::lczero::ValueId id =
      Emit(OpKind::kConstant, {}, {result_type}, literal.bytes);
  value_kinds_[id] = ::lczero::BuilderOpKind::kConstant;
  value_literals_[id] = ::lczero::ToLiteralProto(literal);
  return id;
}

::lczero::ValueId SemanticBuilder::Convert(
    ::lczero::ValueId input, ::lczero::TensorType::ElementType type) {
  EnsureValueExists(input);
  TensorType result_type = value_types_[input];
  result_type.element_type = ToSemanticElement(type);
  const ::lczero::ValueId id = Emit(OpKind::kConvert, {input}, {result_type});
  value_kinds_[id] = ::lczero::BuilderOpKind::kConvert;
  return id;
}

::lczero::ValueId SemanticBuilder::Convolution(
    ::lczero::ValueId input, ::lczero::ValueId filter,
    const ::lczero::ConvolutionParams& params) {
  EnsureValueExists(input);
  EnsureValueExists(filter);
  TensorType result_type = value_types_[input];
  const TensorType& filter_type = value_types_[filter];
  const auto& dn = params.dimension_numbers;
  if (result_type.dimensions.size() == filter_type.dimensions.size() &&
      !result_type.dimensions.empty()) {
    result_type.dimensions[dn.output_batch_dimension()] =
        value_types_[input].dimensions[dn.input_batch_dimension()];
    result_type.dimensions[dn.output_feature_dimension()] =
        filter_type.dimensions[dn.kernel_output_feature_dimension()];
    for (int i = 0; i < dn.input_spatial_dimensions_size(); ++i) {
      int64_t stride = 1;
      int64_t pad_low = 0;
      int64_t pad_high = 0;
      int64_t lhs_dilation = 1;
      int64_t rhs_dilation = 1;
      if (i < params.window.dimensions_size()) {
        const auto& window_dim = params.window.dimensions(i);
        stride = window_dim.stride();
        pad_low = window_dim.padding_low();
        pad_high = window_dim.padding_high();
        lhs_dilation = window_dim.base_dilation();
        rhs_dilation = window_dim.window_dilation();
      }
      const int64_t input_dim =
          value_types_[input].dimensions[dn.input_spatial_dimensions(i)];
      const int64_t kernel_dim =
          filter_type.dimensions[dn.kernel_spatial_dimensions(i)];
      result_type.dimensions[dn.output_spatial_dimensions(i)] =
          ComputeConvOutputDim(input_dim, kernel_dim, stride, pad_low, pad_high,
                               lhs_dilation, rhs_dilation);
    }
  }

  ConvParams conv_params;
  conv_params.input_batch_dim = dn.input_batch_dimension();
  conv_params.input_feature_dim = dn.input_feature_dimension();
  for (int i = 0; i < dn.input_spatial_dimensions_size(); ++i) {
    conv_params.input_spatial_dims.push_back(dn.input_spatial_dimensions(i));
    conv_params.kernel_spatial_dims.push_back(dn.kernel_spatial_dimensions(i));
    conv_params.output_spatial_dims.push_back(dn.output_spatial_dimensions(i));
  }
  conv_params.kernel_input_feature_dim = dn.kernel_input_feature_dimension();
  conv_params.kernel_output_feature_dim = dn.kernel_output_feature_dimension();
  conv_params.output_batch_dim = dn.output_batch_dimension();
  conv_params.output_feature_dim = dn.output_feature_dimension();
  for (const auto& dim : params.window.dimensions()) {
    conv_params.window_strides.push_back(dim.stride());
    conv_params.padding.emplace_back(dim.padding_low(), dim.padding_high());
    conv_params.lhs_dilation.push_back(dim.base_dilation());
    conv_params.rhs_dilation.push_back(dim.window_dilation());
  }

  const ::lczero::ValueId id = Emit(OpKind::kConvolution, {input, filter},
                                    {result_type}, conv_params);
  value_kinds_[id] = ::lczero::BuilderOpKind::kConvolution;
  return id;
}

::lczero::ValueId SemanticBuilder::Broadcast(
    ::lczero::ValueId input, const ::lczero::TensorType& target_shape,
    const std::vector<int64_t>& broadcast_dimensions) {
  EnsureValueExists(input);
  BroadcastInDimParams params;
  params.broadcast_dimensions = broadcast_dimensions;
  const ::lczero::ValueId id =
      Emit(OpKind::kBroadcastInDim, {input}, {ToSemanticType(target_shape)}, params);
  value_kinds_[id] = ::lczero::BuilderOpKind::kBroadcast;
  return id;
}

::lczero::ValueId SemanticBuilder::Add(::lczero::ValueId lhs,
                                       ::lczero::ValueId rhs) {
  const ::lczero::ValueId id = EmitElementwise(OpKind::kAdd, lhs, rhs);
  value_kinds_[id] = ::lczero::BuilderOpKind::kAdd;
  return id;
}

::lczero::ValueId SemanticBuilder::Subtract(::lczero::ValueId lhs,
                                            ::lczero::ValueId rhs) {
  const ::lczero::ValueId id = EmitElementwise(OpKind::kSubtract, lhs, rhs);
  value_kinds_[id] = ::lczero::BuilderOpKind::kSubtract;
  return id;
}

::lczero::ValueId SemanticBuilder::Multiply(::lczero::ValueId lhs,
                                            ::lczero::ValueId rhs) {
  const ::lczero::ValueId id = EmitElementwise(OpKind::kMultiply, lhs, rhs);
  value_kinds_[id] = ::lczero::BuilderOpKind::kMultiply;
  return id;
}

::lczero::ValueId SemanticBuilder::Divide(::lczero::ValueId lhs,
                                          ::lczero::ValueId rhs) {
  const ::lczero::ValueId id = EmitElementwise(OpKind::kDivide, lhs, rhs);
  value_kinds_[id] = ::lczero::BuilderOpKind::kDivide;
  return id;
}

::lczero::ValueId SemanticBuilder::Maximum(::lczero::ValueId lhs,
                                           ::lczero::ValueId rhs) {
  const ::lczero::ValueId id = EmitElementwise(OpKind::kMaximum, lhs, rhs);
  value_kinds_[id] = ::lczero::BuilderOpKind::kMaximum;
  return id;
}

::lczero::ValueId SemanticBuilder::Reshape(
    ::lczero::ValueId input, const ::lczero::TensorType& new_shape) {
  EnsureValueExists(input);
  const ::lczero::ValueId id =
      Emit(OpKind::kReshape, {input}, {ToSemanticType(new_shape)});
  value_kinds_[id] = ::lczero::BuilderOpKind::kReshape;
  return id;
}

::lczero::ValueId SemanticBuilder::Dot(::lczero::ValueId lhs,
                                       ::lczero::ValueId rhs,
                                       const ::lczero::DotParams& params) {
  EnsureValueExists(lhs);
  EnsureValueExists(rhs);
  DotParams dot_params;
  dot_params.lhs_batch_dims = params.dimension_numbers.lhs_batch_dimensions();
  dot_params.rhs_batch_dims = params.dimension_numbers.rhs_batch_dimensions();
  dot_params.lhs_contracting_dims =
      params.dimension_numbers.lhs_contracting_dimensions();
  dot_params.rhs_contracting_dims =
      params.dimension_numbers.rhs_contracting_dimensions();

  const TensorType& lhs_type = value_types_[lhs];
  const TensorType& rhs_type = value_types_[rhs];
  if (lhs_type.element_type != rhs_type.element_type) {
    throw Exception("Dot operands must have the same element type");
  }

  TensorType result_type;
  result_type.element_type = lhs_type.element_type;

  if (dot_params.lhs_batch_dims.size() != dot_params.rhs_batch_dims.size()) {
    throw Exception("Dot batch dimension counts must match");
  }
  if (dot_params.lhs_contracting_dims.size() !=
      dot_params.rhs_contracting_dims.size()) {
    throw Exception("Dot contracting dimension counts must match");
  }

  for (size_t i = 0; i < dot_params.lhs_batch_dims.size(); ++i) {
    const int64_t lhs_dim_idx = dot_params.lhs_batch_dims[i];
    const int64_t rhs_dim_idx = dot_params.rhs_batch_dims[i];
    if (lhs_dim_idx < 0 || rhs_dim_idx < 0 ||
        lhs_dim_idx >= static_cast<int64_t>(lhs_type.dimensions.size()) ||
        rhs_dim_idx >= static_cast<int64_t>(rhs_type.dimensions.size())) {
      throw Exception("Dot batch dimension index out of range");
    }
    if (lhs_type.dimensions[lhs_dim_idx] != rhs_type.dimensions[rhs_dim_idx]) {
      throw Exception("Dot batch dimensions must have equal sizes");
    }
    result_type.dimensions.push_back(lhs_type.dimensions[lhs_dim_idx]);
  }

  for (size_t i = 0; i < dot_params.lhs_contracting_dims.size(); ++i) {
    const int64_t lhs_dim_idx = dot_params.lhs_contracting_dims[i];
    const int64_t rhs_dim_idx = dot_params.rhs_contracting_dims[i];
    if (lhs_dim_idx < 0 || rhs_dim_idx < 0 ||
        lhs_dim_idx >= static_cast<int64_t>(lhs_type.dimensions.size()) ||
        rhs_dim_idx >= static_cast<int64_t>(rhs_type.dimensions.size())) {
      throw Exception("Dot contracting dimension index out of range");
    }
    if (lhs_type.dimensions[lhs_dim_idx] != rhs_type.dimensions[rhs_dim_idx]) {
      throw Exception("Dot contracting dimensions must have equal sizes");
    }
  }

  for (size_t i = 0; i < lhs_type.dimensions.size(); ++i) {
    if (std::find(dot_params.lhs_batch_dims.begin(), dot_params.lhs_batch_dims.end(),
                  static_cast<int64_t>(i)) == dot_params.lhs_batch_dims.end() &&
        std::find(dot_params.lhs_contracting_dims.begin(),
                  dot_params.lhs_contracting_dims.end(),
                  static_cast<int64_t>(i)) == dot_params.lhs_contracting_dims.end()) {
      result_type.dimensions.push_back(lhs_type.dimensions[i]);
    }
  }

  for (size_t i = 0; i < rhs_type.dimensions.size(); ++i) {
    if (std::find(dot_params.rhs_batch_dims.begin(), dot_params.rhs_batch_dims.end(),
                  static_cast<int64_t>(i)) == dot_params.rhs_batch_dims.end() &&
        std::find(dot_params.rhs_contracting_dims.begin(),
                  dot_params.rhs_contracting_dims.end(),
                  static_cast<int64_t>(i)) == dot_params.rhs_contracting_dims.end()) {
      result_type.dimensions.push_back(rhs_type.dimensions[i]);
    }
  }

  const ::lczero::ValueId id =
      Emit(OpKind::kDotGeneral, {lhs, rhs}, {result_type}, dot_params);
  value_kinds_[id] = ::lczero::BuilderOpKind::kDot;
  return id;
}

::lczero::ValueId SemanticBuilder::Slice(
    ::lczero::ValueId input, const ::lczero::SliceParams& params) {
  EnsureValueExists(input);
  SliceParams slice_params;
  TensorType result_type = value_types_[input];
  result_type.dimensions.clear();
  for (const auto& dim : params.dimensions) {
    slice_params.start_indices.push_back(dim.start());
    slice_params.limit_indices.push_back(dim.limit());
    slice_params.strides.push_back(dim.stride());
    const int64_t size = (dim.limit() - dim.start()) / dim.stride();
    result_type.dimensions.push_back(size);
  }
  const ::lczero::ValueId id =
      Emit(OpKind::kSlice, {input}, {result_type}, slice_params);
  value_kinds_[id] = ::lczero::BuilderOpKind::kSlice;
  return id;
}

::lczero::ValueId SemanticBuilder::Concatenate(
    const std::vector<::lczero::ValueId>& inputs, int64_t dimension) {
  if (inputs.empty()) throw Exception("Concatenate requires at least one input");
  for (const auto input : inputs) EnsureValueExists(input);
  TensorType result_type = value_types_[inputs.front()];
  int64_t sum = 0;
  for (const auto input : inputs) sum += value_types_[input].dimensions[dimension];
  result_type.dimensions[dimension] = sum;
  const ::lczero::ValueId id =
      Emit(OpKind::kConcatenate, inputs, {result_type}, ConcatenateParams{dimension});
  value_kinds_[id] = ::lczero::BuilderOpKind::kConcatenate;
  return id;
}

::lczero::ValueId SemanticBuilder::Tanh(::lczero::ValueId input) {
  EnsureValueExists(input);
  const ::lczero::ValueId id = Emit(OpKind::kTanh, {input}, {value_types_[input]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kTanh;
  return id;
}

::lczero::ValueId SemanticBuilder::LogPlusOne(::lczero::ValueId input) {
  EnsureValueExists(input);
  const ::lczero::ValueId id =
      Emit(OpKind::kLogPlusOne, {input}, {value_types_[input]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kLogPlusOne;
  return id;
}

::lczero::ValueId SemanticBuilder::ExponentialMinusOne(::lczero::ValueId input) {
  EnsureValueExists(input);
  const ::lczero::ValueId id =
      Emit(OpKind::kExponentialMinusOne, {input}, {value_types_[input]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kExponentialMinusOne;
  return id;
}

::lczero::ValueId SemanticBuilder::Negate(::lczero::ValueId input) {
  EnsureValueExists(input);
  const ::lczero::ValueId id = Emit(OpKind::kNegate, {input}, {value_types_[input]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kNegate;
  return id;
}

::lczero::ValueId SemanticBuilder::Exponential(::lczero::ValueId input) {
  EnsureValueExists(input);
  const ::lczero::ValueId id =
      Emit(OpKind::kExponential, {input}, {value_types_[input]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kExponential;
  return id;
}

::lczero::ValueId SemanticBuilder::Sqrt(::lczero::ValueId input) {
  EnsureValueExists(input);
  const ::lczero::ValueId id = Emit(OpKind::kSqrt, {input}, {value_types_[input]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kSqrt;
  return id;
}

::lczero::ValueId SemanticBuilder::Rsqrt(::lczero::ValueId input) {
  EnsureValueExists(input);
  const ::lczero::ValueId id = Emit(OpKind::kRsqrt, {input}, {value_types_[input]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kRsqrt;
  return id;
}

::lczero::ValueId SemanticBuilder::Tuple(
    const std::vector<::lczero::ValueId>& elements) {
  if (elements.empty()) throw Exception("Tuple requires at least one element");
  for (const auto element : elements) EnsureValueExists(element);
  // Known limitation: TensorType cannot represent tuple types. We store the
  // first element's type as a placeholder for tuple results. GetType on tuple
  // ValueIds is not a reliable tuple-type query; see PR11 guard task below.
  // This is tracked debt for post-cutover cleanup.
  const TensorType result_type = value_types_[elements.front()];
  const ::lczero::ValueId id = Emit(OpKind::kTuple, elements, {result_type});
  value_kinds_[id] = ::lczero::BuilderOpKind::kTuple;
  return id;
}

::lczero::ValueId SemanticBuilder::Reduce(
    ::lczero::ValueId input, ::lczero::ValueId initial,
    const ::lczero::ReduceParams& params) {
  EnsureValueExists(input);
  EnsureValueExists(initial);
  TensorType result_type = value_types_[input];
  ReduceParams reduce_params;
  reduce_params.dimensions = params.reduction_dimensions;
  switch (params.computation) {
    case ::lczero::ReduceParams::Computation::kAdd:
      reduce_params.reduce_op = ReduceParams::ReduceOp::kAdd;
      break;
    case ::lczero::ReduceParams::Computation::kMaximum:
      reduce_params.reduce_op = ReduceParams::ReduceOp::kMaximum;
      break;
    case ::lczero::ReduceParams::Computation::kMultiply:
      reduce_params.reduce_op = ReduceParams::ReduceOp::kMultiply;
      break;
  }
  for (auto dim_it = reduce_params.dimensions.rbegin();
       dim_it != reduce_params.dimensions.rend(); ++dim_it) {
    if (*dim_it >= 0 && static_cast<size_t>(*dim_it) < result_type.dimensions.size()) {
      result_type.dimensions.erase(result_type.dimensions.begin() + *dim_it);
    }
  }
  const ::lczero::ValueId id =
      Emit(OpKind::kReduce, {input, initial}, {result_type}, reduce_params);
  value_kinds_[id] = ::lczero::BuilderOpKind::kReduce;
  return id;
}

::lczero::ValueId SemanticBuilder::Transpose(
    ::lczero::ValueId input, const std::vector<int64_t>& permutation) {
  EnsureValueExists(input);
  const TensorType& input_type = value_types_[input];
  const size_t rank = input_type.dimensions.size();
  if (permutation.size() != rank) {
    throw Exception("Transpose invariant failed: operand rank=" +
                    std::to_string(rank) + ", perm.size()=" +
                    std::to_string(permutation.size()) + ", operand_id=V" +
                    std::to_string(input) + ", perm=" +
                    FormatPermutation(permutation));
  }
  std::vector<bool> seen(rank, false);
  for (const int64_t dim : permutation) {
    if (dim < 0 || static_cast<size_t>(dim) >= rank) {
      throw Exception("Transpose invariant failed: invalid dimension " +
                      std::to_string(dim) + " for rank=" +
                      std::to_string(rank) + ", operand_id=V" +
                      std::to_string(input) + ", perm=" +
                      FormatPermutation(permutation));
    }
    if (seen[dim]) {
      throw Exception("Transpose invariant failed: duplicate dimension " +
                      std::to_string(dim) + ", operand_id=V" +
                      std::to_string(input) + ", perm=" +
                      FormatPermutation(permutation));
    }
    seen[dim] = true;
  }

  TensorType result_type = input_type;
  std::vector<int64_t> dims = input_type.dimensions;
  for (size_t i = 0; i < permutation.size(); ++i) {
    result_type.dimensions[i] = dims[permutation[i]];
  }
  const ::lczero::ValueId id =
      Emit(OpKind::kTranspose, {input}, {result_type}, TransposeParams{permutation});
  value_kinds_[id] = ::lczero::BuilderOpKind::kTranspose;
  return id;
}

::lczero::ValueId SemanticBuilder::Gather(
    ::lczero::ValueId input, ::lczero::ValueId indices,
    const ::lczero::GatherParams& params) {
  EnsureValueExists(input);
  EnsureValueExists(indices);
  GatherParams gather_params;
  gather_params.index_vector_dim = static_cast<int64_t>(params.index_vector_dim);
  gather_params.offset_dims = params.offset_dims;
  gather_params.collapsed_slice_dims = params.collapsed_slice_dims;
  gather_params.start_index_map = params.start_index_map;
  gather_params.slice_sizes = params.slice_sizes;
  gather_params.indices_are_sorted = params.indices_are_sorted;
  gather_params.unique_indices = params.unique_indices;

  const TensorType& input_type = value_types_[input];
  const TensorType& indices_type = value_types_[indices];
  std::vector<int64_t> indices_dims = indices_type.dimensions;
  if (indices_dims.size() == params.index_vector_dim) {
    indices_dims.push_back(1);
  }

  TensorType result_type;
  result_type.element_type = input_type.element_type;
  const size_t output_rank =
      gather_params.offset_dims.size() + indices_dims.size() - 1;
  size_t offset_dims_idx = 0;
  size_t gather_dims_idx = 0;
  for (size_t i = 0; i < output_rank; ++i) {
    const bool is_offset_dim =
        std::find(gather_params.offset_dims.begin(), gather_params.offset_dims.end(),
                  static_cast<int64_t>(i)) != gather_params.offset_dims.end();
    if (is_offset_dim) {
      while (std::find(gather_params.collapsed_slice_dims.begin(),
                       gather_params.collapsed_slice_dims.end(),
                       static_cast<int64_t>(offset_dims_idx)) !=
             gather_params.collapsed_slice_dims.end()) {
        ++offset_dims_idx;
      }
      if (offset_dims_idx >= gather_params.slice_sizes.size()) {
        throw Exception("Gather slice size index out of range");
      }
      result_type.dimensions.push_back(gather_params.slice_sizes[offset_dims_idx++]);
    } else {
      if (gather_dims_idx == gather_params.index_vector_dim) {
        ++gather_dims_idx;
      }
      if (gather_dims_idx >= indices_dims.size()) {
        throw Exception("Gather indices dimension index out of range");
      }
      result_type.dimensions.push_back(indices_dims[gather_dims_idx++]);
    }
  }

  const ::lczero::ValueId id =
      Emit(OpKind::kGather, {input, indices}, {result_type}, gather_params);
  value_kinds_[id] = ::lczero::BuilderOpKind::kGather;
  return id;
}

::lczero::ValueId SemanticBuilder::Compare(
    ::lczero::ValueId lhs, ::lczero::ValueId rhs,
    const ::lczero::CompareParams& params) {
  EnsureValueExists(lhs);
  EnsureValueExists(rhs);
  CompareParams compare_params;
  if (params.direction == "EQ") {
    compare_params.direction = CompareParams::Direction::kEq;
  } else if (params.direction == "NE") {
    compare_params.direction = CompareParams::Direction::kNe;
  } else if (params.direction == "LT") {
    compare_params.direction = CompareParams::Direction::kLt;
  } else if (params.direction == "LE") {
    compare_params.direction = CompareParams::Direction::kLe;
  } else if (params.direction == "GT") {
    compare_params.direction = CompareParams::Direction::kGt;
  } else if (params.direction == "GE") {
    compare_params.direction = CompareParams::Direction::kGe;
  } else {
    throw Exception("Unrecognized compare direction: " + params.direction);
  }

  TensorType result_type = value_types_[lhs];
  result_type.element_type = ElementType::kPred;
  const ::lczero::ValueId id =
      Emit(OpKind::kCompare, {lhs, rhs}, {result_type}, compare_params);
  value_kinds_[id] = ::lczero::BuilderOpKind::kCompare;
  return id;
}

::lczero::ValueId SemanticBuilder::Select(::lczero::ValueId condition,
                                          ::lczero::ValueId on_true,
                                          ::lczero::ValueId on_false) {
  EnsureValueExists(condition);
  EnsureValueExists(on_true);
  EnsureValueExists(on_false);
  const ::lczero::ValueId id =
      Emit(OpKind::kSelect, {condition, on_true, on_false}, {value_types_[on_true]});
  value_kinds_[id] = ::lczero::BuilderOpKind::kSelect;
  return id;
}

::lczero::TensorType SemanticBuilder::GetType(::lczero::ValueId value) const {
  EnsureValueExists(value);
  if (value_kinds_[value] == ::lczero::BuilderOpKind::kTuple) {
    throw Exception("GetType on tuple ValueId is unsupported");
  }
  return ToBuilderType(value_types_[value]);
}

::lczero::BuilderOpKind SemanticBuilder::GetOpKind(::lczero::ValueId value) const {
  EnsureValueExists(value);
  return value_kinds_[value];
}

const pblczero::XlaLiteralProto* SemanticBuilder::TryGetLiteral(
    ::lczero::ValueId value) const {
  EnsureValueExists(value);
  if (!value_literals_[value].has_value()) return nullptr;
  return &(*value_literals_[value]);
}

void SemanticBuilder::PushMetadataScope() {
  if (metadata_scopes_.empty()) {
    metadata_scopes_.push_back(Metadata{});
  } else {
    metadata_scopes_.push_back(metadata_scopes_.back());
  }
}

void SemanticBuilder::PopMetadataScope() {
  if (metadata_scopes_.size() <= 1) return;
  metadata_scopes_.pop_back();
}

void SemanticBuilder::SetOpType(std::string_view op_type) {
  if (metadata_scopes_.empty()) metadata_scopes_.push_back(Metadata{});
  metadata_scopes_.back().op_type = std::string(op_type);
}

void SemanticBuilder::SetOpName(std::string_view op_name) {
  if (metadata_scopes_.empty()) metadata_scopes_.push_back(Metadata{});
  metadata_scopes_.back().op_name = std::string(op_name);
}

void SemanticBuilder::Return(const std::vector<::lczero::ValueId>& values) {
  EnsureEntryFunction();
  std::vector<TensorType> result_types;
  result_types.reserve(values.size());
  for (const auto value : values) {
    EnsureValueExists(value);
    result_types.push_back(value_types_[value]);
  }
  module_.functions.front().result_types = result_types;
  (void)Emit(OpKind::kReturn, values, {});
}

SemanticModule SemanticBuilder::BuildModule() const { return module_; }

}  // namespace lczero::stablehlo::semantic
