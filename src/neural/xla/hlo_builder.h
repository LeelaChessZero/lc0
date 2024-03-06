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
using HloComputation =
    std::vector<std::unique_ptr<pblczero::HloInstructionProto>>;

// A builder class for constructing HloModuleProto.
class HloBuilder {
 public:
  // HLO operations.
  HloFlow Parameter(const pblczero::XlaShapeProto& shape);
  HloFlow Constant(const pblczero::XlaLiteralProto& literal);
  HloFlow Convert(HloFlow input, const pblczero::XlaShapeProto::Type type);
  HloFlow Convolution(
      HloFlow input, HloFlow filter, const pblczero::XlaWindow& window,
      const pblczero::XlaConvolutionDimensionNumbers& dimension_numbers);
  HloFlow Broadcast(HloFlow input, const pblczero::XlaShapeProto& target_shape,
                    const std::vector<int64_t>& broadcast_dimensions);
  HloFlow Add(HloFlow lhs, HloFlow rhs);
  HloFlow Maximum(HloFlow lhs, HloFlow rhs);
  HloFlow Reshape(HloFlow input, const pblczero::XlaShapeProto& new_shape);
  HloFlow Dot(HloFlow lhs, HloFlow rhs,
              const pblczero::XlaDotDimensionNumbers& dimension_numbers);
  HloFlow Tanh(HloFlow input);
  HloFlow Tuple(const std::vector<HloFlow>& elements);

  // Build the HloModuleProto with a given name.
  pblczero::HloModuleProto Build(std::string_view name);

 private:
  pblczero::HloInstructionProto* MakeInstruction(
      std::string_view opcode, const pblczero::XlaShapeProto& shape,
      const std::vector<const pblczero::HloInstructionProto*> operands);
  pblczero::HloInstructionProto* MakeElementwiseInstruction(
      std::string_view opcode, HloFlow lhs, HloFlow rhs);
  void AssignInstructionNames();

  HloComputation entry_computation_;
  std::unordered_map<std::string, pblczero::HloComputationProto>
      dependent_computations_;
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

// A helper function to reset a shape of a layout. Marks all dimensions as
// non-dynamic, and sets layout to major_to_minor.
void ResetXlaShapeProtoLayout(pblczero::XlaShapeProto* shape);

}  // namespace lczero