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

#include "utils/logging.h"

namespace lczero {

HloFlow* HloBuilder::Parameter(const pblczero::XlaShapeProto& shape) {
  return MakeInstruction("parameter", shape);
}

HloFlow* HloBuilder::Convert(HloFlow* input,
                             const pblczero::XlaShapeProto::Type type) {
  if (input->shape().element_type() == type) return input;
  pblczero::XlaShapeProto shape = input->shape();
  shape.set_element_type(type);
  return MakeInstruction("convert", shape);
}

HloFlow* HloBuilder::Constant(const pblczero::XlaLiteralProto& literal) {
  auto* flow = MakeInstruction("constant", literal.shape());
  *flow->mutable_literal() = literal;
  return flow;
}

HloFlow* HloBuilder::MakeInstruction(std::string_view opcode,
                                     const pblczero::XlaShapeProto& shape) {
  auto instr = std::make_unique<HloFlow>();
  auto ret = instr.get();
  ret->set_opcode(opcode);
  *ret->mutable_shape() = shape;
  *ret->mutable_metadata() = metadata_;
  ret->set_id(entry_computation_.size());
  entry_computation_.push_back(std::move(instr));
  return ret;
}

namespace {

std::pair<std::vector<pblczero::XlaShapeProto>, std::vector<std::string>>
AssignParameterIndices(const HloComputation& comp) {
  std::vector<pblczero::XlaShapeProto> parameter_shapes;
  std::vector<std::string> parameter_names;
  size_t idx = 0;
  for (const std::unique_ptr<HloFlow>& instr : comp) {
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
  for (auto& instr : entry_computation_) instr->set_name(std::to_string(idx++));
  for (auto& [_, comp] : dependent_computations_) {
    for (auto& instr : *comp.mutable_instructions()) {
      instr.set_name(std::to_string(idx++));
    }
  }
}

}  // namespace lczero