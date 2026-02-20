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

#include "neural/xla/hlo_builder_adapter.h"

#include "neural/xla/tensor_literal_utils.h"
#include "utils/exception.h"

namespace lczero {

namespace {

std::string ReduceName(ReduceParams::Computation computation,
                       pblczero::XlaShapeProto::Type type) {
  const char* prefix = nullptr;
  switch (computation) {
    case ReduceParams::Computation::kAdd:
      prefix = "reduce_add_";
      break;
    case ReduceParams::Computation::kMaximum:
      prefix = "reduce_max_";
      break;
    case ReduceParams::Computation::kMultiply:
      prefix = "reduce_mul_";
      break;
  }
  return std::string(prefix) + pblczero::XlaShapeProto::Type_Name(type);
}

}  // namespace

ValueId HloBuilderAdapter::Store(HloFlow flow) {
  values_.push_back(flow);
  return values_.size() - 1;
}

HloFlow HloBuilderAdapter::Flow(ValueId value) const {
  if (value >= values_.size()) {
    throw Exception("Invalid ValueId: " + std::to_string(value));
  }
  return values_[value];
}

HloTensorType HloBuilderAdapter::ToHloTensorType(const TensorType& type) const {
  return HloTensorType(ElementTypeToProto(type.element_type), type.dimensions);
}

pblczero::XlaLiteralProto HloBuilderAdapter::ToLiteralProto(
    const TensorLiteral& literal) const {
  return ::lczero::ToLiteralProto(literal);
}

HloComputation HloBuilderAdapter::EnsureReduceComputation(ReduceKey key) {
  if (auto iter = reduce_computations_.find(key); iter != reduce_computations_.end()) {
    return iter->second;
  }
  const std::string name = ReduceName(key.first, key.second);
  if (auto comp = builder_.GetComputationId(name)) {
    reduce_computations_.emplace(key, *comp);
    return *comp;
  }

  HloBuilder computation_builder;
  HloTensorType scalar_type(key.second);
  HloFlow lhs = computation_builder.Parameter(scalar_type);
  HloFlow rhs = computation_builder.Parameter(scalar_type);
  switch (key.first) {
    case ReduceParams::Computation::kAdd:
      computation_builder.Add(lhs, rhs);
      break;
    case ReduceParams::Computation::kMaximum:
      computation_builder.Maximum(lhs, rhs);
      break;
    case ReduceParams::Computation::kMultiply:
      computation_builder.Multiply(lhs, rhs);
      break;
  }
  HloComputation computation = builder_.AddComputation(name, computation_builder);
  reduce_computations_.emplace(key, computation);
  return computation;
}

BuilderOpKind HloBuilderAdapter::OpcodeToKind(std::string_view opcode) const {
  if (opcode == "parameter") return BuilderOpKind::kParameter;
  if (opcode == "constant") return BuilderOpKind::kConstant;
  if (opcode == "convert") return BuilderOpKind::kConvert;
  if (opcode == "convolution") return BuilderOpKind::kConvolution;
  if (opcode == "broadcast") return BuilderOpKind::kBroadcast;
  if (opcode == "add") return BuilderOpKind::kAdd;
  if (opcode == "subtract") return BuilderOpKind::kSubtract;
  if (opcode == "multiply") return BuilderOpKind::kMultiply;
  if (opcode == "divide") return BuilderOpKind::kDivide;
  if (opcode == "maximum") return BuilderOpKind::kMaximum;
  if (opcode == "reshape") return BuilderOpKind::kReshape;
  if (opcode == "dot") return BuilderOpKind::kDot;
  if (opcode == "slice") return BuilderOpKind::kSlice;
  if (opcode == "concatenate") return BuilderOpKind::kConcatenate;
  if (opcode == "tanh") return BuilderOpKind::kTanh;
  if (opcode == "log-plus-one") return BuilderOpKind::kLogPlusOne;
  if (opcode == "exponential-minus-one") return BuilderOpKind::kExponentialMinusOne;
  if (opcode == "negate") return BuilderOpKind::kNegate;
  if (opcode == "exponential") return BuilderOpKind::kExponential;
  if (opcode == "sqrt") return BuilderOpKind::kSqrt;
  if (opcode == "rsqrt") return BuilderOpKind::kRsqrt;
  if (opcode == "tuple") return BuilderOpKind::kTuple;
  if (opcode == "reduce") return BuilderOpKind::kReduce;
  if (opcode == "transpose") return BuilderOpKind::kTranspose;
  if (opcode == "gather") return BuilderOpKind::kGather;
  if (opcode == "compare") return BuilderOpKind::kCompare;
  if (opcode == "select") return BuilderOpKind::kSelect;
  throw Exception("Unmapped HLO opcode: " + std::string(opcode));
}

ValueId HloBuilderAdapter::Parameter(const TensorType& shape) {
  return Store(builder_.Parameter(ToHloTensorType(shape)));
}

ValueId HloBuilderAdapter::Constant(const TensorLiteral& literal) {
  return Store(builder_.Constant(ToLiteralProto(literal)));
}

ValueId HloBuilderAdapter::Convert(ValueId input, TensorType::ElementType type) {
  return Store(builder_.Convert(Flow(input), ElementTypeToProto(type)));
}

ValueId HloBuilderAdapter::Convolution(ValueId input, ValueId filter,
                                       const ConvolutionParams& params) {
  return Store(builder_.Convolution(Flow(input), Flow(filter), params.window,
                                    params.dimension_numbers));
}

ValueId HloBuilderAdapter::Broadcast(
    ValueId input, const TensorType& target_shape,
    const std::vector<int64_t>& broadcast_dimensions) {
  return Store(builder_.Broadcast(Flow(input), ToHloTensorType(target_shape),
                                  broadcast_dimensions));
}

ValueId HloBuilderAdapter::Add(ValueId lhs, ValueId rhs) {
  return Store(builder_.Add(Flow(lhs), Flow(rhs)));
}

ValueId HloBuilderAdapter::Subtract(ValueId lhs, ValueId rhs) {
  return Store(builder_.Subtract(Flow(lhs), Flow(rhs)));
}

ValueId HloBuilderAdapter::Multiply(ValueId lhs, ValueId rhs) {
  return Store(builder_.Multiply(Flow(lhs), Flow(rhs)));
}

ValueId HloBuilderAdapter::Divide(ValueId lhs, ValueId rhs) {
  return Store(builder_.Divide(Flow(lhs), Flow(rhs)));
}

ValueId HloBuilderAdapter::Maximum(ValueId lhs, ValueId rhs) {
  return Store(builder_.Maximum(Flow(lhs), Flow(rhs)));
}

ValueId HloBuilderAdapter::Reshape(ValueId input, const TensorType& new_shape) {
  return Store(builder_.Reshape(Flow(input), ToHloTensorType(new_shape)));
}

ValueId HloBuilderAdapter::Dot(ValueId lhs, ValueId rhs, const DotParams& params) {
  return Store(builder_.Dot(Flow(lhs), Flow(rhs), params.dimension_numbers));
}

ValueId HloBuilderAdapter::Slice(ValueId input, const SliceParams& params) {
  return Store(builder_.Slice(Flow(input), params.dimensions));
}

ValueId HloBuilderAdapter::Concatenate(const std::vector<ValueId>& inputs,
                                       int64_t dimension) {
  std::vector<HloFlow> hlo_inputs;
  hlo_inputs.reserve(inputs.size());
  for (const ValueId value : inputs) {
    hlo_inputs.push_back(Flow(value));
  }
  return Store(builder_.Concatenate(hlo_inputs, dimension));
}

ValueId HloBuilderAdapter::Tanh(ValueId input) {
  return Store(builder_.Tanh(Flow(input)));
}

ValueId HloBuilderAdapter::LogPlusOne(ValueId input) {
  return Store(builder_.LogPlusOne(Flow(input)));
}

ValueId HloBuilderAdapter::ExponentialMinusOne(ValueId input) {
  return Store(builder_.ExponentialMinusOne(Flow(input)));
}

ValueId HloBuilderAdapter::Negate(ValueId input) {
  return Store(builder_.Negate(Flow(input)));
}

ValueId HloBuilderAdapter::Exponential(ValueId input) {
  return Store(builder_.Exponential(Flow(input)));
}

ValueId HloBuilderAdapter::Sqrt(ValueId input) {
  return Store(builder_.Sqrt(Flow(input)));
}

ValueId HloBuilderAdapter::Rsqrt(ValueId input) {
  return Store(builder_.Rsqrt(Flow(input)));
}

ValueId HloBuilderAdapter::Tuple(const std::vector<ValueId>& elements) {
  std::vector<HloFlow> hlo_elements;
  hlo_elements.reserve(elements.size());
  for (const ValueId value : elements) {
    hlo_elements.push_back(Flow(value));
  }
  return Store(builder_.Tuple(hlo_elements));
}

ValueId HloBuilderAdapter::Reduce(ValueId input, ValueId initial,
                                  const ReduceParams& params) {
  const pblczero::XlaShapeProto::Type type = Flow(input)->shape().element_type();
  const HloComputation computation =
      EnsureReduceComputation({params.computation, type});
  return Store(builder_.Reduce(Flow(input), Flow(initial), computation,
                               params.reduction_dimensions));
}

ValueId HloBuilderAdapter::Transpose(
    ValueId input, const std::vector<int64_t>& permutation) {
  return Store(builder_.Transpose(Flow(input), permutation));
}

ValueId HloBuilderAdapter::Gather(ValueId input, ValueId indices,
                                  const GatherParams& params) {
  return Store(builder_.Gather(
      Flow(input), Flow(indices), params.index_vector_dim, params.offset_dims,
      params.slice_sizes, params.collapsed_slice_dims, params.start_index_map,
      params.indices_are_sorted, params.unique_indices));
}

ValueId HloBuilderAdapter::Compare(ValueId lhs, ValueId rhs,
                                   const CompareParams& params) {
  return Store(builder_.Compare(Flow(lhs), Flow(rhs), params.direction));
}

ValueId HloBuilderAdapter::Select(ValueId condition, ValueId on_true,
                                  ValueId on_false) {
  return Store(builder_.Select(Flow(condition), Flow(on_true), Flow(on_false)));
}

TensorType HloBuilderAdapter::GetType(ValueId value) const {
  return TensorTypeFromProto(Flow(value)->shape());
}

BuilderOpKind HloBuilderAdapter::GetOpKind(ValueId value) const {
  return OpcodeToKind(Flow(value)->opcode());
}

const pblczero::XlaLiteralProto* HloBuilderAdapter::TryGetLiteral(
    ValueId value) const {
  const HloFlow flow = Flow(value);
  if (flow->opcode() != "constant") return nullptr;
  return &flow->literal();
}

void HloBuilderAdapter::PushMetadataScope() {
  metadata_scopes_.push_back(std::make_unique<HloContext>(&builder_));
}

void HloBuilderAdapter::PopMetadataScope() {
  if (metadata_scopes_.size() <= 1) return;
  metadata_scopes_.pop_back();
}

void HloBuilderAdapter::SetOpType(std::string_view op_type) {
  if (metadata_scopes_.empty()) return;
  metadata_scopes_.back()->SetOpType(op_type);
}

void HloBuilderAdapter::SetOpName(std::string_view op_name) {
  if (metadata_scopes_.empty()) return;
  metadata_scopes_.back()->SetOpName(op_name);
}

pblczero::HloModuleProto HloBuilderAdapter::BuildModule(std::string_view name) {
  return builder_.BuildModule(name);
}

}  // namespace lczero
