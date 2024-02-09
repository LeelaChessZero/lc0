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
#include <vector>

#include "neural/xla/hlo.pb.h"

namespace lczero {

class HloContext;
class HloBuilder;

using HloFlow = pblczero::HloInstructionProto;

using HloComputation = std::vector<std::unique_ptr<HloFlow>>;

class HloBuilder {
 public:
  HloContext ScopedContext();

  HloFlow* Parameter(const pblczero::XlaShapeProto& shape);
  HloFlow* Constant(const pblczero::XlaShapeProto& shape,
                    const std::string& raw_data);
  HloFlow* Convert(HloFlow* input, const pblczero::XlaShapeProto::Type type);

  pblczero::HloModuleProto Build(std::string_view name);

 private:
  HloFlow* MakeInstruction(std::string_view opcode,
                           const pblczero::XlaShapeProto& shape);
  void AssignInstructionNames();

  HloComputation entry_computation_;
  std::unordered_map<std::string, pblczero::HloComputationProto>
      dependent_computations_;
  pblczero::XlaOpMetadata metadata_;
  friend class HloContext;
};

class HloContext {
 public:
  HloContext(HloBuilder* builder) : builder_(builder) {}
  ~HloContext() { builder_->metadata_ = saved_metadata_; }
  const HloContext& OpType(std::string_view op_type) const {
    builder_->metadata_.set_op_type(op_type);
    return *this;
  }
  const HloContext& OpName(std::string_view op_name) const {
    builder_->metadata_.set_op_name(op_name);
    return *this;
  }

 private:
  HloBuilder* builder_;
  pblczero::XlaOpMetadata saved_metadata_;
};

}  // namespace lczero