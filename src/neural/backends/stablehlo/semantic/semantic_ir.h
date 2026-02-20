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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "neural/xla/onnx_builder_interface.h"

namespace lczero::stablehlo::semantic {

// Dialect-agnostic scalar element type.
enum class ElementType : uint8_t {
  kInvalid = 0,
  kPred,
  kF16,
  kBF16,
  kF32,
  kS32,
  kS64,
};

// Dialect-agnostic tensor type.
struct TensorType {
  ElementType element_type = ElementType::kInvalid;
  std::vector<int64_t> dimensions;
};

// SSA-style value reference used by semantic ops.
using ValueId = size_t;

// Semantic operation kind (M9 scope: existing HLO builder surface + Return).
enum class OpKind : uint8_t {
  kParameter = 0,
  kConstant,
  kConvert,
  kConvolution,
  kBroadcastInDim,
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kMaximum,
  kReshape,
  kDotGeneral,
  kSlice,
  kConcatenate,
  kTanh,
  kLogPlusOne,
  kExponentialMinusOne,
  kNegate,
  kExponential,
  kSqrt,
  kRsqrt,
  kTuple,
  kReduce,
  kTranspose,
  kGather,
  kCompare,
  kSelect,
  kReturn,
};

struct ConvParams {
  int64_t input_batch_dim = 0;
  int64_t input_feature_dim = 0;
  std::vector<int64_t> input_spatial_dims;
  int64_t kernel_input_feature_dim = 0;
  int64_t kernel_output_feature_dim = 0;
  std::vector<int64_t> kernel_spatial_dims;
  int64_t output_batch_dim = 0;
  int64_t output_feature_dim = 0;
  std::vector<int64_t> output_spatial_dims;
  std::vector<int64_t> window_strides;
  std::vector<std::pair<int64_t, int64_t>> padding;
  std::vector<int64_t> lhs_dilation;
  std::vector<int64_t> rhs_dilation;
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;
};

struct DotParams {
  std::vector<int64_t> lhs_batch_dims;
  std::vector<int64_t> rhs_batch_dims;
  std::vector<int64_t> lhs_contracting_dims;
  std::vector<int64_t> rhs_contracting_dims;
};

struct GatherParams {
  int64_t index_vector_dim = 0;
  std::vector<int64_t> offset_dims;
  std::vector<int64_t> collapsed_slice_dims;
  std::vector<int64_t> start_index_map;
  std::vector<int64_t> slice_sizes;
  bool indices_are_sorted = false;
  bool unique_indices = false;
};

struct ReduceParams {
  enum class ReduceOp : uint8_t { kAdd = 0, kMaximum, kMultiply };
  std::vector<int64_t> dimensions;
  ReduceOp reduce_op = ReduceOp::kAdd;
};

struct CompareParams {
  enum class Direction : uint8_t { kEq = 0, kNe, kLt, kLe, kGt, kGe };
  Direction direction = Direction::kEq;
};

struct SliceParams {
  std::vector<int64_t> start_indices;
  std::vector<int64_t> limit_indices;
  std::vector<int64_t> strides;
};

struct BroadcastInDimParams {
  std::vector<int64_t> broadcast_dimensions;
};

struct TransposeParams {
  std::vector<int64_t> permutation;
};

struct ConcatenateParams {
  int64_t dimension = 0;
};

// Semantic op attribute payload. Each registered op kind uses exactly one
// typed param struct (or monostate for ops with no attrs). The raw byte
// variant (vector<uint8_t>) is reserved for constant tensor data only --
// it is not a generic escape hatch for unregistered ops.
using AttrPayload =
    std::variant<std::monostate, ConvParams, DotParams, GatherParams,
                 ReduceParams, CompareParams, SliceParams,
                 BroadcastInDimParams, TransposeParams, ConcatenateParams,
                 std::vector<uint8_t>>;

struct SemanticOp {
  OpKind kind = OpKind::kParameter;
  std::vector<ValueId> operands;
  std::vector<TensorType> result_types;
  std::string source_op_type;
  std::string source_op_name;
  AttrPayload attrs;
};

struct SemanticFunction {
  std::string name;
  std::vector<TensorType> param_types;
  std::vector<TensorType> result_types;
  std::vector<SemanticOp> ops;
};

struct SemanticModule {
  std::vector<SemanticFunction> functions;
};

class SemanticBuilder final : public ::lczero::IBuilder {
 public:
  SemanticBuilder();
  ~SemanticBuilder() override = default;

  ::lczero::ValueId Parameter(const ::lczero::TensorType& shape) override;
  ::lczero::ValueId Constant(const ::lczero::TensorLiteral& literal) override;
  ::lczero::ValueId Convert(::lczero::ValueId input,
                            ::lczero::TensorType::ElementType type) override;
  ::lczero::ValueId Convolution(
      ::lczero::ValueId input, ::lczero::ValueId filter,
      const ::lczero::ConvolutionParams& params) override;
  ::lczero::ValueId Broadcast(
      ::lczero::ValueId input, const ::lczero::TensorType& target_shape,
      const std::vector<int64_t>& broadcast_dimensions) override;
  ::lczero::ValueId Add(::lczero::ValueId lhs, ::lczero::ValueId rhs) override;
  ::lczero::ValueId Subtract(::lczero::ValueId lhs,
                             ::lczero::ValueId rhs) override;
  ::lczero::ValueId Multiply(::lczero::ValueId lhs,
                             ::lczero::ValueId rhs) override;
  ::lczero::ValueId Divide(::lczero::ValueId lhs,
                           ::lczero::ValueId rhs) override;
  ::lczero::ValueId Maximum(::lczero::ValueId lhs,
                            ::lczero::ValueId rhs) override;
  ::lczero::ValueId Reshape(::lczero::ValueId input,
                            const ::lczero::TensorType& new_shape) override;
  ::lczero::ValueId Dot(::lczero::ValueId lhs, ::lczero::ValueId rhs,
                        const ::lczero::DotParams& params) override;
  ::lczero::ValueId Slice(::lczero::ValueId input,
                          const ::lczero::SliceParams& params) override;
  ::lczero::ValueId Concatenate(const std::vector<::lczero::ValueId>& inputs,
                                int64_t dimension) override;
  ::lczero::ValueId Tanh(::lczero::ValueId input) override;
  ::lczero::ValueId LogPlusOne(::lczero::ValueId input) override;
  ::lczero::ValueId ExponentialMinusOne(::lczero::ValueId input) override;
  ::lczero::ValueId Negate(::lczero::ValueId input) override;
  ::lczero::ValueId Exponential(::lczero::ValueId input) override;
  ::lczero::ValueId Sqrt(::lczero::ValueId input) override;
  ::lczero::ValueId Rsqrt(::lczero::ValueId input) override;
  ::lczero::ValueId Tuple(const std::vector<::lczero::ValueId>& elements) override;
  ::lczero::ValueId Reduce(::lczero::ValueId input, ::lczero::ValueId initial,
                           const ::lczero::ReduceParams& params) override;
  ::lczero::ValueId Transpose(
      ::lczero::ValueId input,
      const std::vector<int64_t>& permutation) override;
  ::lczero::ValueId Gather(::lczero::ValueId input, ::lczero::ValueId indices,
                           const ::lczero::GatherParams& params) override;
  ::lczero::ValueId Compare(::lczero::ValueId lhs, ::lczero::ValueId rhs,
                            const ::lczero::CompareParams& params) override;
  ::lczero::ValueId Select(::lczero::ValueId condition,
                           ::lczero::ValueId on_true,
                           ::lczero::ValueId on_false) override;

  ::lczero::TensorType GetType(::lczero::ValueId value) const override;
  ::lczero::BuilderOpKind GetOpKind(::lczero::ValueId value) const override;
  std::optional<::lczero::TensorLiteral> TryGetLiteral(
      ::lczero::ValueId value) const override;

  void PushMetadataScope() override;
  void PopMetadataScope() override;
  void SetOpType(std::string_view op_type) override;
  void SetOpName(std::string_view op_name) override;

  void Return(const std::vector<::lczero::ValueId>& values);
  SemanticModule BuildModule() const;

 private:
  ::lczero::ValueId Emit(
      OpKind kind, const std::vector<::lczero::ValueId>& operands,
      const std::vector<TensorType>& result_types,
      AttrPayload attrs = std::monostate{});
  ::lczero::ValueId EmitElementwise(
      OpKind kind, ::lczero::ValueId lhs, ::lczero::ValueId rhs);
  TensorType ToSemanticType(const ::lczero::TensorType& type) const;
  ::lczero::TensorType ToBuilderType(const TensorType& type) const;
  void EnsureValueExists(::lczero::ValueId value) const;
  void EnsureEntryFunction();

  struct Metadata {
    std::string op_type;
    std::string op_name;
  };

  SemanticModule module_;
  std::vector<TensorType> value_types_;
  std::vector<::lczero::BuilderOpKind> value_kinds_;
  std::vector<std::optional<::lczero::TensorLiteral>> value_literals_;
  std::vector<Metadata> metadata_scopes_;
};

}  // namespace lczero::stablehlo::semantic
