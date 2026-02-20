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

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "neural/xla/hlo_builder.h"
#include "neural/xla/onnx_builder_interface.h"

namespace lczero {

// Thin compatibility adapter used to validate IBuilder contract against the
// legacy HloBuilder API.
class HloBuilderAdapter : public IBuilder {
 public:
  HloBuilderAdapter() = default;
  ~HloBuilderAdapter() override = default;

  ValueId Parameter(const TensorType& shape) override;
  ValueId Constant(const TensorLiteral& literal) override;
  ValueId Convert(ValueId input, TensorType::ElementType type) override;
  ValueId Convolution(ValueId input, ValueId filter,
                      const ConvolutionParams& params) override;
  ValueId Broadcast(ValueId input, const TensorType& target_shape,
                    const std::vector<int64_t>& broadcast_dimensions) override;
  ValueId Add(ValueId lhs, ValueId rhs) override;
  ValueId Subtract(ValueId lhs, ValueId rhs) override;
  ValueId Multiply(ValueId lhs, ValueId rhs) override;
  ValueId Divide(ValueId lhs, ValueId rhs) override;
  ValueId Maximum(ValueId lhs, ValueId rhs) override;
  ValueId Reshape(ValueId input, const TensorType& new_shape) override;
  ValueId Dot(ValueId lhs, ValueId rhs, const DotParams& params) override;
  ValueId Slice(ValueId input, const SliceParams& params) override;
  ValueId Concatenate(const std::vector<ValueId>& inputs,
                      int64_t dimension) override;
  ValueId Tanh(ValueId input) override;
  ValueId LogPlusOne(ValueId input) override;
  ValueId ExponentialMinusOne(ValueId input) override;
  ValueId Negate(ValueId input) override;
  ValueId Exponential(ValueId input) override;
  ValueId Sqrt(ValueId input) override;
  ValueId Rsqrt(ValueId input) override;
  ValueId Tuple(const std::vector<ValueId>& elements) override;
  ValueId Reduce(ValueId input, ValueId initial,
                 const ReduceParams& params) override;
  ValueId Transpose(ValueId input,
                    const std::vector<int64_t>& permutation) override;
  ValueId Gather(ValueId input, ValueId indices,
                 const GatherParams& params) override;
  ValueId Compare(ValueId lhs, ValueId rhs,
                  const CompareParams& params) override;
  ValueId Select(ValueId condition, ValueId on_true, ValueId on_false) override;

  TensorType GetType(ValueId value) const override;
  BuilderOpKind GetOpKind(ValueId value) const override;
  const pblczero::XlaLiteralProto* TryGetLiteral(ValueId value) const override;

  void PushMetadataScope() override;
  void PopMetadataScope() override;
  void SetOpType(std::string_view op_type) override;
  void SetOpName(std::string_view op_name) override;

  // Helper for adapter-only smoke/probing.
  pblczero::HloModuleProto BuildModule(std::string_view name);

 private:
  using ReduceKey =
      std::pair<ReduceParams::Computation, pblczero::XlaShapeProto::Type>;

  ValueId Store(HloFlow flow);
  HloFlow Flow(ValueId value) const;
  HloTensorType ToHloTensorType(const TensorType& type) const;
  pblczero::XlaLiteralProto ToLiteralProto(const TensorLiteral& literal) const;
  HloComputation EnsureReduceComputation(ReduceKey key);
  BuilderOpKind OpcodeToKind(std::string_view opcode) const;

  HloBuilder builder_;
  std::vector<HloFlow> values_;
  std::vector<std::unique_ptr<HloContext>> metadata_scopes_;
  std::map<ReduceKey, HloComputation> reduce_computations_;
};

}  // namespace lczero

