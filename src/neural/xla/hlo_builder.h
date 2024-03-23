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

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "neural/xla/hlo.pb.h"
#include "utils/logging.h"

namespace lczero {

class HloContext;
class HloBuilder;

using HloFlow = const pblczero::HloInstructionProto*;
using InstructionList =
    std::vector<std::unique_ptr<pblczero::HloInstructionProto>>;

class HloComputation {
 public:
  HloComputation(const HloComputation&) = default;
  HloComputation& operator=(const HloComputation&) = default;
  size_t idx() const { return idx_; }

 private:
  explicit HloComputation(size_t idx) : idx_(idx) {}
  size_t idx_;
  friend class HloBuilder;
};

class HloType {
 public:
  virtual ~HloType() = default;
  virtual pblczero::XlaShapeProto ToProto() const = 0;
  virtual std::string ToString() const = 0;
};

class HloTensorType : public HloType {
 public:
  HloTensorType() = default;
  HloTensorType(const HloTensorType&) = default;
  explicit HloTensorType(pblczero::XlaShapeProto::Type el_type)
      : type_(el_type) {}
  explicit HloTensorType(pblczero::XlaShapeProto::Type el_type,
                         const std::vector<int64_t>& dimensions)
      : type_(el_type), dimensions_(dimensions) {}
  explicit HloTensorType(const pblczero::XlaShapeProto&);
  pblczero::XlaShapeProto ToProto() const override;
  std::string ToString() const override;

  void SetElementType(pblczero::XlaShapeProto::Type el_type) {
    type_ = el_type;
  }
  pblczero::XlaShapeProto::Type GetElementType() const { return type_; }
  void AddDimension(int64_t size) { dimensions_.push_back(size); }
  const std::vector<int64_t>& GetDimensions() const { return dimensions_; }
  int64_t GetDimension(size_t idx) const { return dimensions_[idx]; }
  void SetDimension(size_t idx, int64_t size) { dimensions_[idx] = size; }
  void SetDimensions(const std::vector<int64_t>& dimensions) {
    dimensions_ = dimensions;
  }
  size_t Rank() const { return dimensions_.size(); }
  size_t NumElements() const;

 private:
  pblczero::XlaShapeProto::Type type_ =
      pblczero::XlaShapeProto::PRIMITIVE_TYPE_INVALID;
  std::vector<int64_t> dimensions_;
};

// A builder class for constructing HloModuleProto.
class HloBuilder {
 public:
  // HLO operations.
  HloFlow Parameter(const HloType& shape);
  HloFlow Constant(const pblczero::XlaLiteralProto& literal);
  HloFlow Convert(HloFlow input, const pblczero::XlaShapeProto::Type type);
  HloFlow Convolution(
      HloFlow input, HloFlow filter, const pblczero::XlaWindow& window,
      const pblczero::XlaConvolutionDimensionNumbers& dimension_numbers);
  HloFlow Broadcast(HloFlow input, const HloTensorType& target_shape,
                    const std::vector<int64_t>& broadcast_dimensions);
  HloFlow Add(HloFlow lhs, HloFlow rhs);
  HloFlow Subtract(HloFlow lhs, HloFlow rhs);
  HloFlow Multiply(HloFlow lhs, HloFlow rhs);
  HloFlow Divide(HloFlow lhs, HloFlow rhs);
  HloFlow Maximum(HloFlow lhs, HloFlow rhs);
  HloFlow Reshape(HloFlow input, const HloTensorType& new_shape);
  HloFlow Dot(HloFlow lhs, HloFlow rhs,
              const pblczero::XlaDotDimensionNumbers& dimension_numbers);
  HloFlow Slice(
      HloFlow input,
      const std::vector<pblczero::HloInstructionProto::SliceDimensions>& slice);
  HloFlow Concatenate(const std::vector<HloFlow>& inputs, int64_t dimension);
  HloFlow Tanh(HloFlow input);
  HloFlow LogPlusOne(HloFlow input);
  HloFlow ExponentialMinusOne(HloFlow input);
  HloFlow Negate(HloFlow input);
  HloFlow Exponential(HloFlow input);
  HloFlow Sqrt(HloFlow input);
  HloFlow Rsqrt(HloFlow input);
  HloFlow Tuple(const std::vector<HloFlow>& elements);
  HloFlow Reduce(HloFlow input, HloFlow initial, HloComputation function,
                 const std::vector<int64_t>& reduction_dimensions);
  HloFlow Transpose(HloFlow input, const std::vector<int64_t>& permutation);
  HloFlow Gather(HloFlow input, HloFlow indices, size_t index_vector_dim,
                 const std::vector<int64_t>& offset_dims,
                 const std::vector<int64_t>& slice_sizes,
                 const std::vector<int64_t>& collapsed_slice_dims,
                 const std::vector<int64_t>& start_index_map,
                 bool indices_are_sorted, bool unique_indicies);
  // Direction is one of "EQ", "NE", "LT", "LE", "GT", "GE".
  HloFlow Compare(HloFlow lhs, HloFlow rhs, std::string_view direction);
  HloFlow Select(HloFlow condition, HloFlow on_true, HloFlow on_false);
  // Insert a computation into the module, under given name. Dependent
  // computations are also merged into the module.
  HloComputation AddComputation(std::string_view name,
                                const HloBuilder& builder);
  std::optional<HloComputation> GetComputationId(std::string_view name) const;
  // Build the HloModuleProto with a given name.
  pblczero::HloModuleProto BuildModule(std::string_view name);

 private:
  pblczero::HloInstructionProto* MakeInstruction(
      std::string_view opcode, const pblczero::XlaShapeProto& shape,
      const std::vector<const pblczero::HloInstructionProto*> operands);
  pblczero::HloInstructionProto* MakeElementwiseInstruction(
      std::string_view opcode, HloFlow lhs, HloFlow rhs);
  void AssignInstructionNames();
  pblczero::HloComputationProto ReassignInstructionIds(
      pblczero::HloComputationProto computation);

  size_t next_instruction_id_ = 0;
  InstructionList entry_computation_;
  std::unordered_map<std::string, size_t> computation_names_;
  std::vector<pblczero::HloComputationProto> dependent_computations_;
  pblczero::XlaOpMetadata metadata_;
  friend class HloContext;
};

// A context class for annotating parts of the HLO computation with metadata,
// like original ONNX op, its name, and source file name and line.
// The class saves the current metadata in constructor and restores it in
// destructor, making it possible to use it in a scoped way.
class HloContext {
 public:
  HloContext(HloBuilder* builder)
      : builder_(builder), saved_metadata_(builder->metadata_) {}
  ~HloContext() { builder_->metadata_ = saved_metadata_; }
  void SetOpType(std::string_view op_type) const {
    builder_->metadata_.set_op_type(op_type);
  }
  void SetOpName(std::string_view op_name) const {
    builder_->metadata_.set_op_name(op_name);
  }

 private:
  HloBuilder* builder_;
  pblczero::XlaOpMetadata saved_metadata_;
};

}  // namespace lczero