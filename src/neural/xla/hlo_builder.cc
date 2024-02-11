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

#include "neural/xla/hlo_builder.h"

#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {

HloFlow HloBuilder::Parameter(const pblczero::XlaShapeProto& shape) {
  return MakeInstruction("parameter", shape, {});
}

HloFlow HloBuilder::Convert(HloFlow input,
                            const pblczero::XlaShapeProto::Type type) {
  if (input->shape().element_type() == type) return input;
  pblczero::XlaShapeProto shape = input->shape();
  shape.set_element_type(type);
  return MakeInstruction("convert", shape, {input});
}

HloFlow HloBuilder::Constant(const pblczero::XlaLiteralProto& literal) {
  auto* flow = MakeInstruction("constant", literal.shape(), {});
  *flow->mutable_literal() = literal;
  return flow;
}

HloFlow HloBuilder::Convolution(
    HloFlow input, HloFlow filter, const pblczero::XlaWindow& window,
    const pblczero::XlaConvolutionDimensionNumbers& dn) {
  if (input->shape().dimensions_size() != filter->shape().dimensions_size()) {
    throw Exception(
        "Convolution input and filter shapes must have the "
        "same number of dimensions");
  }

  pblczero::XlaShapeProto shape = input->shape();
  auto* out_dims = shape.mutable_dimensions();
  const auto& in_dims = input->shape().dimensions();
  const auto& filter_dims = filter->shape().dimensions();
  (*out_dims)[dn.output_batch_dimension()] =
      in_dims[dn.input_batch_dimension()];
  (*out_dims)[dn.output_feature_dimension()] =
      filter_dims[dn.kernel_output_feature_dimension()];
  for (size_t i = 0; i < dn.input_spatial_dimensions_size(); ++i) {
    (*out_dims)[dn.output_spatial_dimensions(i)] =
        in_dims[dn.input_spatial_dimensions(i)];
  }

  auto* flow = MakeInstruction("convolution", shape, {input, filter});
  *flow->mutable_window() = window;
  *flow->mutable_convolution_dimension_numbers() = dn;
  return flow;
}

HloFlow HloBuilder::Broadcast(
    HloFlow input, const pblczero::XlaShapeProto& target_shape,
    const std::vector<int64_t>& broadcast_dimensions) {
  auto flow = MakeInstruction("broadcast", target_shape, {input});
  if (broadcast_dimensions.size() != input->shape().dimensions_size()) {
    throw Exception("Broadcast must have the same size as the input shape");
  }
  const auto& input_shape = input->shape();
  for (size_t i = 0; i < broadcast_dimensions.size(); ++i) {
    auto dim = broadcast_dimensions[i];
    const auto& input_dim = input_shape.dimensions(i);
    if (input_dim != 1 && input_dim != target_shape.dimensions(dim)) {
      throw Exception(
          "Broadcast dimension must be 1 or equal to the target shape "
          "dimension");
    }
    flow->add_dimensions(dim);
  }

  return flow;
}

HloFlow HloBuilder::Add(HloFlow lhs, HloFlow rhs) {
  return MakeElementwiseInstruction("add", lhs, rhs);
}

HloFlow HloBuilder::Maximum(HloFlow lhs, HloFlow rhs) {
  return MakeElementwiseInstruction("maximum", lhs, rhs);
}

pblczero::HloInstructionProto* HloBuilder::MakeElementwiseInstruction(
    std::string_view opcode, HloFlow lhs, HloFlow rhs) {
  if (lhs->shape().dimensions() != rhs->shape().dimensions()) {
    throw Exception("Elementwise operands must have the same shape");
  }
  return MakeInstruction(opcode, lhs->shape(), {lhs, rhs});
}

pblczero::HloInstructionProto* HloBuilder::MakeInstruction(
    std::string_view opcode, const pblczero::XlaShapeProto& shape,
    const std::vector<const pblczero::HloInstructionProto*> operands) {
  auto instr = std::make_unique<pblczero::HloInstructionProto>();
  auto ret = instr.get();
  ret->set_opcode(opcode);
  *ret->mutable_shape() = shape;
  *ret->mutable_metadata() = metadata_;
  ret->set_id(entry_computation_.size());
  for (const auto& operand : operands) {
    ret->add_operand_ids(operand->id());
  }
  entry_computation_.push_back(std::move(instr));
  return ret;
}

namespace {

std::pair<std::vector<pblczero::XlaShapeProto>, std::vector<std::string>>
AssignParameterIndices(const HloComputation& comp) {
  std::vector<pblczero::XlaShapeProto> parameter_shapes;
  std::vector<std::string> parameter_names;
  size_t idx = 0;
  for (const auto& instr : comp) {
    if (instr->opcode() == "parameter") {
      instr->set_parameter_number(idx++);
      parameter_shapes.push_back(instr->shape());
      parameter_names.push_back(std::string(instr->name()));
    }
  }
  return {parameter_shapes, parameter_names};
}

pblczero::HloComputationProto MakeComputation(const HloComputation& comp,
                                              std::string_view name,
                                              size_t id) {
  pblczero::HloComputationProto ret;
  ret.set_id(id);
  ret.set_name(name);
  auto [shapes, names] = AssignParameterIndices(comp);
  for (auto& instr : comp) *ret.add_instructions() = *instr;
  *ret.mutable_program_shape()->mutable_parameters() = shapes;
  *ret.mutable_program_shape()->mutable_parameter_names() = names;
  *ret.mutable_program_shape()->mutable_result() = comp.back()->shape();
  ret.set_root_id(comp.back()->id());
  return ret;
}

}  // namespace

pblczero::HloModuleProto HloBuilder::Build(std::string_view name) {
  AssignInstructionNames();
  pblczero::HloModuleProto module;
  module.set_name(name);
  module.set_entry_computation_name("main");
  module.set_entry_computation_id(0);
  *module.add_computations() = MakeComputation(entry_computation_, "main", 0);
  for (auto& [name, comp] : dependent_computations_) {
    *module.add_computations() = comp;
  }
  *module.mutable_host_program_shape() = module.computations(0).program_shape();
  return module;
}

void HloBuilder::AssignInstructionNames() {
  // Every instruction in the module should have an unique name, numeric names
  // are allowed.
  size_t idx = 0;
  for (auto& instr : entry_computation_) {
    instr->set_name("i" + std::to_string(idx++));
  }
  for (auto& [_, comp] : dependent_computations_) {
    for (auto& instr : *comp.mutable_instructions()) {
      instr.set_name("i" + std::to_string(idx++));
    }
  }
}

}  // namespace lczero