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
#include <string_view>
#include <utility>
#include <vector>

namespace lczero {

// Shared SSA-style identifier used by ONNX conversion handlers.
using ValueId = size_t;

// Builder-facing tensor type contract used by both adapter and semantic builders.
struct TensorType {
  enum class ElementType : uint8_t {
    kInvalid = 0,
    kF16,
    kBF16,
    kF32,
    kS32,
    kS64,
    kPred,
  };

  ElementType element_type = ElementType::kInvalid;
  std::vector<int64_t> dimensions;
};

// Constant payload contract at IBuilder boundary.
// - bytes are contiguous
// - row-major layout
// - little-endian encoding
// - ownership is explicit: implementations must copy or take ownership
// - intended for small constants (scalars/shape literals), not bulk weights
struct TensorLiteral {
  TensorType type;
  std::vector<uint8_t> bytes;
};

// Intentional naming differences vs semantic::OpKind (semantic uses StableHLO names):
//   BuilderOpKind::kBroadcast  <->  semantic::OpKind::kBroadcastInDim
//   BuilderOpKind::kDot        <->  semantic::OpKind::kDotGeneral
// These are 1:1 mappings; the semantic names match the VHLO op names.
// Minimal op kind surface used by conversion-time constant inspection.
enum class BuilderOpKind : uint8_t {
  kUnknown = 0,
  kParameter,
  kConstant,
  kConvert,
  kConvolution,
  kBroadcast,
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kMaximum,
  kReshape,
  kDot,
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
};

struct ConvolutionParams {
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

struct SliceParams {
  std::vector<int64_t> start_indices;
  std::vector<int64_t> limit_indices;
  std::vector<int64_t> strides;
};

struct GatherParams {
  size_t index_vector_dim = 0;
  std::vector<int64_t> offset_dims;
  std::vector<int64_t> slice_sizes;
  std::vector<int64_t> collapsed_slice_dims;
  std::vector<int64_t> start_index_map;
  bool indices_are_sorted = false;
  bool unique_indices = false;
};

struct ReduceParams {
  enum class Computation : uint8_t { kAdd = 0, kMaximum, kMultiply };
  Computation computation = Computation::kAdd;
  std::vector<int64_t> reduction_dimensions;
};

struct CompareParams {
  // One of: EQ, NE, LT, LE, GT, GE.
  std::string direction;
};

// Narrow interface used by ONNX handlers. Do not extend this interface unless
// a concrete ONNX conversion need is proven.
class IBuilder {
 public:
  virtual ~IBuilder() = default;

  // Emitters (current handler-used surface).
  virtual ValueId Parameter(const TensorType& shape) = 0;
  virtual ValueId Constant(const TensorLiteral& literal) = 0;
  virtual ValueId Convert(ValueId input, TensorType::ElementType type) = 0;
  virtual ValueId Convolution(ValueId input, ValueId filter,
                              const ConvolutionParams& params) = 0;
  virtual ValueId Broadcast(
      ValueId input, const TensorType& target_shape,
      const std::vector<int64_t>& broadcast_dimensions) = 0;
  virtual ValueId Add(ValueId lhs, ValueId rhs) = 0;
  virtual ValueId Subtract(ValueId lhs, ValueId rhs) = 0;
  virtual ValueId Multiply(ValueId lhs, ValueId rhs) = 0;
  virtual ValueId Divide(ValueId lhs, ValueId rhs) = 0;
  virtual ValueId Maximum(ValueId lhs, ValueId rhs) = 0;
  virtual ValueId Reshape(ValueId input, const TensorType& new_shape) = 0;
  virtual ValueId Dot(ValueId lhs, ValueId rhs, const DotParams& params) = 0;
  virtual ValueId Slice(ValueId input, const SliceParams& params) = 0;
  virtual ValueId Concatenate(const std::vector<ValueId>& inputs,
                              int64_t dimension) = 0;
  virtual ValueId Tanh(ValueId input) = 0;
  virtual ValueId LogPlusOne(ValueId input) = 0;
  virtual ValueId ExponentialMinusOne(ValueId input) = 0;
  virtual ValueId Negate(ValueId input) = 0;
  virtual ValueId Exponential(ValueId input) = 0;
  virtual ValueId Sqrt(ValueId input) = 0;
  virtual ValueId Rsqrt(ValueId input) = 0;
  virtual ValueId Tuple(const std::vector<ValueId>& elements) = 0;
  virtual ValueId Reduce(ValueId input, ValueId initial,
                         const ReduceParams& params) = 0;
  virtual ValueId Transpose(ValueId input,
                            const std::vector<int64_t>& permutation) = 0;
  virtual ValueId Gather(ValueId input, ValueId indices,
                         const GatherParams& params) = 0;
  virtual ValueId Compare(ValueId lhs, ValueId rhs,
                          const CompareParams& params) = 0;
  virtual ValueId Select(ValueId condition, ValueId on_true,
                         ValueId on_false) = 0;

  // Type query used by handlers for dtype+shape access.
  virtual TensorType GetType(ValueId value) const = 0;

  // Constant inspection hooks used by constant-folding paths.
  virtual BuilderOpKind GetOpKind(ValueId value) const = 0;
  virtual std::optional<TensorLiteral> TryGetLiteral(ValueId value) const = 0;

  // Metadata/context hooks equivalent to HloContext.
  // Contract: PopMetadataScope is a no-op when scope depth <= 1.
  // Implementations should maintain at least one active scope during normal
  // BuilderContext RAII usage.
  virtual void PushMetadataScope() = 0;
  virtual void PopMetadataScope() = 0;
  virtual void SetOpType(std::string_view op_type) = 0;
  virtual void SetOpName(std::string_view op_name) = 0;
};

class BuilderContext {
 public:
  explicit BuilderContext(IBuilder* builder) : builder_(builder) {
    builder_->PushMetadataScope();
  }
  ~BuilderContext() { builder_->PopMetadataScope(); }

  void SetOpType(std::string_view op_type) const {
    builder_->SetOpType(op_type);
  }
  void SetOpName(std::string_view op_name) const { builder_->SetOpName(op_name); }

 private:
  IBuilder* builder_;
};

}  // namespace lczero
