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

#include "neural/backends/stablehlo/lowering/lowering.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "neural/backends/stablehlo/lowering/op_specs.h"
#include "utils/exception.h"

namespace lczero::stablehlo::lowering {
namespace {

TypePtr LowerElementType(semantic::ElementType element_type) {
  switch (element_type) {
    case semantic::ElementType::kF32:
      return f32Type();
    case semantic::ElementType::kS32:
      return i32Type();
    case semantic::ElementType::kS64:
      return i64Type();
    case semantic::ElementType::kPred:
      return boolType();
    case semantic::ElementType::kF16:
      return f16Type();
    case semantic::ElementType::kBF16:
      return bf16Type();
    case semantic::ElementType::kInvalid:
      break;
  }
  throw Exception("Unsupported semantic element type in lowering");
}

std::vector<TypePtr> LowerResultTypes(const semantic::SemanticOp& op) {
  std::vector<TypePtr> result_types;
  result_types.reserve(op.result_types.size());
  for (const auto& type : op.result_types) {
    result_types.push_back(LowerType(type));
  }
  return result_types;
}

uint64_t LowerCompareDirection(semantic::CompareParams::Direction direction) {
  switch (direction) {
    case semantic::CompareParams::Direction::kEq:
      return 0;
    case semantic::CompareParams::Direction::kNe:
      return 1;
    case semantic::CompareParams::Direction::kGe:
      return 2;
    case semantic::CompareParams::Direction::kGt:
      return 3;
    case semantic::CompareParams::Direction::kLe:
      return 4;
    case semantic::CompareParams::Direction::kLt:
      return 5;
  }
  throw Exception("Unsupported semantic compare direction");
}

uint64_t LowerCompareType(const TypePtr& lhs_type) {
  if (lhs_type->kind() != TypeKind::kRankedTensor) return 1;
  const auto& ranked_type = static_cast<const RankedTensorType&>(*lhs_type);
  if (ranked_type.elementType()->kind() != TypeKind::kElement) return 1;
  const auto element_type = static_cast<const ElementTypeWrapper&>(
                                *ranked_type.elementType())
                                .elementType();
  switch (element_type) {
    case ElementType::kF16:
    case ElementType::kBF16:
    case ElementType::kF32:
    case ElementType::kF64:
    case ElementType::kTF32:
      return 0;  // FLOAT
    default:
      return 1;  // INTEGER/SIGNED fallback
  }
}

template <typename T>
const T& GetAttr(const semantic::SemanticOp& op) {
  const auto* attr = std::get_if<T>(&op.attrs);
  if (attr == nullptr) {
    throw Exception("Missing semantic attrs for op");
  }
  return *attr;
}

WireRegion makeReduceBodyMul(TypePtr scalar_tensor) {
  return makeWireReduceBodyRegion("multiply_v1", /*locationIndex=*/kInvalidIndex,
                                  std::move(scalar_tensor), /*lhs=*/0, /*rhs=*/1,
                                  /*result=*/2);
}

WireOp LowerStraightLineWireOp(const semantic::SemanticOp& op,
                               const std::vector<ValueRef>& operand_refs,
                               const std::vector<TypePtr>& operand_types,
                               std::vector<TypePtr> result_types,
                               size_t location_index) {
  const OpSpec& spec = GetOpSpec(op.kind);
  if (spec.num_operands != 0 && spec.num_operands != operand_refs.size()) {
    throw Exception("Operand arity mismatch while lowering op");
  }

  switch (op.kind) {
    case semantic::OpKind::kAdd:
    case semantic::OpKind::kSubtract:
    case semantic::OpKind::kMultiply:
    case semantic::OpKind::kDivide:
    case semantic::OpKind::kMaximum:
    case semantic::OpKind::kTanh:
    case semantic::OpKind::kLogPlusOne:
    case semantic::OpKind::kExponentialMinusOne:
    case semantic::OpKind::kNegate:
    case semantic::OpKind::kExponential:
    case semantic::OpKind::kSqrt:
    case semantic::OpKind::kRsqrt:
    case semantic::OpKind::kSelect:
    case semantic::OpKind::kTuple:
    case semantic::OpKind::kConvert:
    case semantic::OpKind::kReshape:
      return makeWireOpWithResultsAndOperands(
          std::string(spec.vhlo_name), "vhlo", location_index,
          std::move(result_types), operand_refs);

    case semantic::OpKind::kBroadcastInDim: {
      const auto& attr = GetAttr<semantic::BroadcastInDimParams>(op);
      BroadcastInDimProps props;
      props.broadcastDimensions = attr.broadcast_dimensions;
      return makeWireBroadcastInDimOp(location_index, std::move(result_types),
                                      operand_refs, std::move(props));
    }

    case semantic::OpKind::kSlice: {
      const auto& attr = GetAttr<semantic::SliceParams>(op);
      SliceProps props;
      props.startIndices = attr.start_indices;
      props.limitIndices = attr.limit_indices;
      props.strides = attr.strides;
      WireOp wire_op = makeWireOpWithResultsAndOperands(
          std::string(spec.vhlo_name), "vhlo", location_index,
          std::move(result_types), operand_refs);
      wire_op.sliceProps = std::move(props);
      return wire_op;
    }

    case semantic::OpKind::kConcatenate: {
      const auto& attr = GetAttr<semantic::ConcatenateParams>(op);
      ConcatenateProps props;
      props.dimension = attr.dimension;
      WireOp wire_op = makeWireOpWithResultsAndOperands(
          std::string(spec.vhlo_name), "vhlo", location_index,
          std::move(result_types), operand_refs);
      wire_op.concatenateProps = std::move(props);
      return wire_op;
    }

    case semantic::OpKind::kCompare: {
      const auto& attr = GetAttr<semantic::CompareParams>(op);
      const uint64_t ctype =
          operand_types.empty() ? 1 : LowerCompareType(operand_types.front());
      const uint64_t cdir = LowerCompareDirection(attr.direction);
      return makeWireCompareOp(location_index, std::move(result_types),
                               operand_refs, ctype, cdir);
    }

    case semantic::OpKind::kDotGeneral: {
      const auto& attr = GetAttr<semantic::DotParams>(op);
      DotGeneralProps props;
      props.lhsBatchingDimensions = attr.lhs_batch_dims;
      props.rhsBatchingDimensions = attr.rhs_batch_dims;
      props.lhsContractingDimensions = attr.lhs_contracting_dims;
      props.rhsContractingDimensions = attr.rhs_contracting_dims;
      props.precisionConfig = {0};
      return makeWireDotGeneralOp(location_index, std::move(result_types),
                                  operand_refs, std::move(props));
    }

    case semantic::OpKind::kGather: {
      const auto& attr = GetAttr<semantic::GatherParams>(op);
      GatherProps props;
      props.offsetDims = attr.offset_dims;
      props.collapsedSliceDims = attr.collapsed_slice_dims;
      props.startIndexMap = attr.start_index_map;
      props.indexVectorDim = attr.index_vector_dim;
      props.sliceSizes = attr.slice_sizes;
      props.indicesAreSorted = attr.indices_are_sorted;
      return makeWireGatherOp(location_index, std::move(result_types),
                              operand_refs, std::move(props));
    }

    case semantic::OpKind::kConvolution: {
      const auto& attr = GetAttr<semantic::ConvParams>(op);
      ConvolutionProps props;
      props.inputBatchDimension = attr.input_batch_dim;
      props.inputFeatureDimension = attr.input_feature_dim;
      props.kernelInputFeatureDimension = attr.kernel_input_feature_dim;
      props.kernelOutputFeatureDimension = attr.kernel_output_feature_dim;
      props.outputBatchDimension = attr.output_batch_dim;
      props.outputFeatureDimension = attr.output_feature_dim;
      props.featureGroupCount = attr.feature_group_count;
      props.batchGroupCount = attr.batch_group_count;
      props.windowStrides = attr.window_strides;
      props.lhsDilation = attr.lhs_dilation;
      props.rhsDilation = attr.rhs_dilation;
      props.inputSpatialDimensions = attr.input_spatial_dims;
      props.kernelSpatialDimensions = attr.kernel_spatial_dims;
      props.outputSpatialDimensions = attr.output_spatial_dims;
      for (const auto& [low, high] : attr.padding) {
        props.padding.push_back(low);
        props.padding.push_back(high);
      }
      props.windowReversal.assign(props.windowStrides.size(), false);
      props.precisionConfig = {0};
      return makeWireConvolutionOp(location_index, std::move(result_types),
                                   operand_refs, std::move(props));
    }

    case semantic::OpKind::kTranspose: {
      const auto& attr = GetAttr<semantic::TransposeParams>(op);
      TransposeProps props;
      props.permutation = attr.permutation;
      return makeWireTransposeOp(location_index, std::move(result_types),
                                 operand_refs, std::move(props));
    }

    case semantic::OpKind::kParameter:
    case semantic::OpKind::kConstant:
    case semantic::OpKind::kReduce:
    case semantic::OpKind::kReturn:
      break;
  }
  throw Exception("Unsupported straight-line semantic op in lowering");
}

TypePtr LowerScalarTensorFromRanked(const TypePtr& ranked_tensor) {
  if (ranked_tensor->kind() != TypeKind::kRankedTensor) {
    throw Exception("Reduce lowering expects ranked tensor operand");
  }
  const auto& ranked = static_cast<const RankedTensorType&>(*ranked_tensor);
  return makeRankedTensor({}, ranked.elementType());
}

TargetVersion ParseTargetVersion(std::string_view value) {
  auto parse_component = [&](size_t begin, size_t end) -> int {
    if (begin >= end) {
      throw Exception("Empty component in LC0_STABLEHLO_TARGET_VERSION");
    }
    int parsed = 0;
    for (size_t i = begin; i < end; ++i) {
      const char c = value[i];
      if (c < '0' || c > '9') {
        throw Exception(
            "Invalid LC0_STABLEHLO_TARGET_VERSION: expected digits only");
      }
      parsed = parsed * 10 + (c - '0');
    }
    return parsed;
  };

  const size_t dot1 = value.find('.');
  const size_t dot2 =
      dot1 == std::string_view::npos ? std::string_view::npos : value.find('.', dot1 + 1);
  if (dot1 == std::string_view::npos || dot2 == std::string_view::npos ||
      value.find('.', dot2 + 1) != std::string_view::npos) {
    throw Exception(
        "Invalid LC0_STABLEHLO_TARGET_VERSION: expected major.minor.patch");
  }
  return TargetVersion{
      parse_component(0, dot1),
      parse_component(dot1 + 1, dot2),
      parse_component(dot2 + 1, value.size()),
  };
}

}  // namespace

TypePtr LowerType(const semantic::TensorType& type) {
  if (type.element_type == semantic::ElementType::kInvalid) {
    throw Exception("Cannot lower invalid semantic tensor type");
  }

  std::vector<int64_t> dims = type.dimensions;
  for (const int64_t dim : dims) {
    if (dim < 0 && dim != -1 && dim != kDynamicDim) {
      throw Exception("Unsupported semantic tensor dimension: " +
                      std::to_string(dim));
    }
  }

  return makeRankedTensor(std::move(dims), LowerElementType(type.element_type));
}

TargetVersion ResolveTargetVersion() {
  std::string env_value;
#ifdef _MSC_VER
  char* raw = nullptr;
  size_t len = 0;
  if (_dupenv_s(&raw, &len, "LC0_STABLEHLO_TARGET_VERSION") == 0 && raw != nullptr) {
    env_value.assign(raw);
    free(raw);
  }
#else
  const char* raw = std::getenv("LC0_STABLEHLO_TARGET_VERSION");
  if (raw != nullptr) env_value = raw;
#endif
  if (env_value.empty()) {
    return TargetVersion{};
  }
  return ParseTargetVersion(env_value);
}

std::string BuildProducerString(const TargetVersion& version) {
  return "StableHLO_v" + std::to_string(version.major) + "." +
         std::to_string(version.minor) + "." + std::to_string(version.patch);
}

std::vector<uint8_t> ApplyTargetVersionToBytecode(
    const std::vector<uint8_t>& bytecode) {
  if (bytecode.size() < 5) {
    throw Exception("Bytecode too short while rewriting header producer");
  }

  size_t pos = 4;  // after magic bytes
  while (true) {
    if (pos >= bytecode.size()) {
      throw Exception("Malformed bytecode header: truncated version varint");
    }
    const uint8_t byte = bytecode[pos++];
    if ((byte & 0x80) == 0) break;
  }
  const size_t version_end = pos;

  while (true) {
    if (pos >= bytecode.size()) {
      throw Exception("Malformed bytecode header: missing producer terminator");
    }
    if (bytecode[pos++] == 0) break;
  }
  const size_t header_end = pos;

  const std::string producer = BuildProducerString(ResolveTargetVersion());
  std::vector<uint8_t> rewritten;
  rewritten.reserve(4 + (version_end - 4) + producer.size() + 1 +
                    (bytecode.size() - header_end));

  rewritten.insert(rewritten.end(), bytecode.begin(), bytecode.begin() + 4);
  rewritten.insert(rewritten.end(), bytecode.begin() + 4,
                   bytecode.begin() + version_end);
  rewritten.insert(rewritten.end(), producer.begin(), producer.end());
  rewritten.push_back(0);
  rewritten.insert(rewritten.end(), bytecode.begin() + header_end, bytecode.end());
  return rewritten;
}

bool LowerStraightLineOpIntoBlock(const semantic::SemanticOp& op,
                                  WireBlock* block,
                                  std::vector<ValueRef>* value_refs,
                                  std::vector<TypePtr>* value_types,
                                  size_t location_index) {
  if (block == nullptr || value_refs == nullptr || value_types == nullptr) {
    throw Exception("LowerStraightLineOpIntoBlock received null output pointer");
  }

  switch (op.kind) {
    case semantic::OpKind::kParameter:
    case semantic::OpKind::kConstant:
    case semantic::OpKind::kReduce:
    case semantic::OpKind::kReturn:
      return false;
    default:
      break;
  }

  std::vector<ValueRef> operand_refs;
  std::vector<TypePtr> operand_types;
  operand_refs.reserve(op.operands.size());
  operand_types.reserve(op.operands.size());
  for (const semantic::ValueId operand : op.operands) {
    if (operand >= value_refs->size() || operand >= value_types->size()) {
      throw Exception("Semantic operand ValueId is not mapped");
    }
    operand_refs.push_back((*value_refs)[operand]);
    operand_types.push_back((*value_types)[operand]);
  }

  std::vector<TypePtr> result_types = LowerResultTypes(op);
  WireOp wire_op =
      LowerStraightLineWireOp(op, operand_refs, operand_types, result_types,
                              location_index);
  block->ops.push_back(std::move(wire_op));

  const ValueRef first_result_ref = value_refs->size();
  for (size_t i = 0; i < result_types.size(); ++i) {
    value_refs->push_back(first_result_ref + i);
    value_types->push_back(result_types[i]);
  }
  return true;
}

bool LowerReduceOpIntoBlock(const semantic::SemanticOp& op, WireBlock* block,
                            std::vector<ValueRef>* value_refs,
                            std::vector<TypePtr>* value_types,
                            size_t location_index) {
  if (block == nullptr || value_refs == nullptr || value_types == nullptr) {
    throw Exception("LowerReduceOpIntoBlock received null output pointer");
  }
  if (op.kind != semantic::OpKind::kReduce) {
    return false;
  }

  std::vector<ValueRef> operand_refs;
  std::vector<TypePtr> operand_types;
  operand_refs.reserve(op.operands.size());
  operand_types.reserve(op.operands.size());
  for (const semantic::ValueId operand : op.operands) {
    if (operand >= value_refs->size() || operand >= value_types->size()) {
      throw Exception("Reduce operand ValueId is not mapped");
    }
    operand_refs.push_back((*value_refs)[operand]);
    operand_types.push_back((*value_types)[operand]);
  }
  if (operand_types.empty()) {
    throw Exception("Reduce lowering requires at least one operand");
  }

  const auto& attr = GetAttr<semantic::ReduceParams>(op);
  ReduceProps props;
  props.dimensions = attr.dimensions;

  const TypePtr scalar_tensor = LowerScalarTensorFromRanked(operand_types.front());
  WireRegion body;
  switch (attr.reduce_op) {
    case semantic::ReduceParams::ReduceOp::kAdd:
      body = makeReduceBodyAdd(scalar_tensor);
      break;
    case semantic::ReduceParams::ReduceOp::kMaximum:
      body = makeReduceBodyMax(scalar_tensor);
      break;
    case semantic::ReduceParams::ReduceOp::kMultiply:
      body = makeReduceBodyMul(scalar_tensor);
      break;
  }

  std::vector<TypePtr> result_types = LowerResultTypes(op);
  WireOp wire_op = makeWireReduceOp(location_index, std::move(result_types),
                                    operand_refs, std::move(props), body);

  const std::vector<TypePtr> lowered_result_types = wire_op.resultTypes;
  block->ops.push_back(std::move(wire_op));

  const ValueRef first_result_ref = value_refs->size();
  for (size_t i = 0; i < lowered_result_types.size(); ++i) {
    value_refs->push_back(first_result_ref + i);
    value_types->push_back(lowered_result_types[i]);
  }
  return true;
}

bool LowerConstantOpIntoBlock(const semantic::SemanticOp& op, WireBlock* block,
                              std::vector<ValueRef>* value_refs,
                              std::vector<TypePtr>* value_types,
                              size_t location_index) {
  if (block == nullptr || value_refs == nullptr || value_types == nullptr) {
    throw Exception("LowerConstantOpIntoBlock received null output pointer");
  }
  if (op.kind != semantic::OpKind::kConstant) {
    return false;
  }

  const auto* raw_data = std::get_if<std::vector<uint8_t>>(&op.attrs);
  if (raw_data == nullptr) {
    throw Exception("Constant lowering requires raw byte attrs");
  }

  std::vector<TypePtr> result_types = LowerResultTypes(op);
  if (result_types.size() != 1) {
    throw Exception("Constant lowering expects exactly one result type");
  }
  WireOp wire_op =
      makeWireConstantOp(location_index, result_types.front(), *raw_data);

  block->ops.push_back(std::move(wire_op));
  const ValueRef next_result_ref = value_refs->size();
  value_refs->push_back(next_result_ref);
  value_types->push_back(result_types.front());
  return true;
}

WireModule LowerToWireModule(const semantic::SemanticModule& module) {
  if (module.functions.empty()) {
    throw Exception("Cannot lower empty semantic module");
  }

  constexpr size_t kLoc = kInvalidIndex;
  const auto& function = module.functions.front();
  const std::string function_name = function.name.empty() ? "main" : function.name;
  if (function_name != "main") {
    throw Exception("Semantic entry function must be named 'main'");
  }

  WireBlock function_block;
  std::vector<ValueRef> value_refs;
  std::vector<TypePtr> value_types;
  value_refs.reserve(function.param_types.size());
  value_types.reserve(function.param_types.size() + function.ops.size());

  for (size_t i = 0; i < function.param_types.size(); ++i) {
    TypePtr type = LowerType(function.param_types[i]);
    function_block.args.push_back(makeWireBlockArg(type, kLoc));
    value_refs.push_back(i);
    value_types.push_back(std::move(type));
  }

  bool saw_return = false;
  for (const auto& op : function.ops) {
    if (LowerConstantOpIntoBlock(op, &function_block, &value_refs, &value_types,
                                 kLoc)) {
      continue;
    }
    if (LowerReduceOpIntoBlock(op, &function_block, &value_refs, &value_types,
                               kLoc)) {
      continue;
    }
    if (LowerStraightLineOpIntoBlock(op, &function_block, &value_refs, &value_types,
                                     kLoc)) {
      continue;
    }

    if (op.kind != semantic::OpKind::kReturn) {
      throw Exception("Unsupported semantic op in module lowering");
    }
    std::vector<ValueRef> return_operands;
    return_operands.reserve(op.operands.size());
    for (const semantic::ValueId operand : op.operands) {
      if (operand >= value_refs.size()) {
        throw Exception("Return operand ValueId is not mapped");
      }
      return_operands.push_back(value_refs[operand]);
    }
    function_block.ops.push_back(makeWireReturnOp(kLoc, std::move(return_operands)));
    saw_return = true;
  }
  if (!saw_return) {
    throw Exception("Semantic function is missing return op");
  }

  WireRegion function_region;
  function_region.blocks.push_back(std::move(function_block));

  WireOp function_op = makeWireFuncOp(kLoc, std::nullopt, std::move(function_region),
                                      /*symName=*/"main",
                                      /*symVisibility=*/"");

  WireBlock module_block;
  module_block.ops.push_back(std::move(function_op));

  WireRegion module_region;
  module_region.blocks.push_back(std::move(module_block));

  WireOp module_op = makeWireModuleOp(
      kLoc, std::nullopt, std::move(module_region),
      /*symName=*/"stablehlo_semantic_module");
  return WireModule{std::move(module_op)};
}

}  // namespace lczero::stablehlo::lowering
