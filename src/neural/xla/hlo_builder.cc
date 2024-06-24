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

#include <algorithm>
#include <numeric>

#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {

HloTensorType::HloTensorType(const pblczero::XlaShapeProto& proto)
    : type_(proto.element_type()), dimensions_(proto.dimensions()) {
  switch (type_) {
    case pblczero::XlaShapeProto::PRIMITIVE_TYPE_INVALID:
    case pblczero::XlaShapeProto::TUPLE:
    case pblczero::XlaShapeProto::OPAQUE_TYPE:
    case pblczero::XlaShapeProto::TOKEN:
      throw Exception("Invalid element type for tensor type");
    default:
      break;
  }
}

pblczero::XlaShapeProto HloTensorType::ToProto() const {
  pblczero::XlaShapeProto ret;
  ret.set_element_type(type_);
  *ret.mutable_dimensions() = dimensions_;
  ret.mutable_layout();
  for (size_t i = 0; i < dimensions_.size(); ++i) {
    ret.add_is_dynamic_dimension(false);
    ret.mutable_layout()->add_minor_to_major(dimensions_.size() - i - 1);
  }
  return ret;
}

std::string HloTensorType::ToString() const {
  std::string ret = pblczero::XlaShapeProto::Type_Name(type_);
  ret += "[";
  for (size_t i = 0; i < dimensions_.size(); ++i) {
    ret += (i == 0 ? "" : ", ") + std::to_string(dimensions_[i]);
  }
  return ret + "]";
}

size_t HloTensorType::NumElements() const {
  return std::accumulate(dimensions_.begin(), dimensions_.end(), 1,
                         std::multiplies<int64_t>());
}

// Creates an instruction and populates required fields of the
// HloInstructionProto: result shape, opcode and operands.
// Appends the instruction to the entry computation.
pblczero::HloInstructionProto* HloBuilder::MakeInstruction(
    std::string_view opcode, const pblczero::XlaShapeProto& shape,
    const std::vector<const pblczero::HloInstructionProto*> operands) {
  auto instr = std::make_unique<pblczero::HloInstructionProto>();
  auto ret = instr.get();
  ret->set_opcode(opcode);
  *ret->mutable_shape() = shape;
  if (!metadata_.OutputAsString().empty()) {
    *ret->mutable_metadata() = metadata_;
  }
  ret->set_id(next_instruction_id_++);
  for (const auto& operand : operands) {
    ret->add_operand_ids(operand->id());
  }
  entry_computation_.push_back(std::move(instr));
  return ret;
}

// Creates an elementwise instruction, which always have two operands of the
// same shape.
pblczero::HloInstructionProto* HloBuilder::MakeElementwiseInstruction(
    std::string_view opcode, HloFlow lhs, HloFlow rhs) {
  if (lhs->shape().dimensions() != rhs->shape().dimensions()) {
    throw Exception("Elementwise operands must have the same shape");
  }
  return MakeInstruction(opcode, lhs->shape(), {lhs, rhs});
}

////////////////////////////////////////////////////////////////////////////
// Instructions.
////////////////////////////////////////////////////////////////////////////

HloFlow HloBuilder::Parameter(const HloType& shape) {
  return MakeInstruction("parameter", shape.ToProto(), {});
}

// Converts the element types while keeping the shape.
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
    HloFlow input, const HloTensorType& target_shape,
    const std::vector<int64_t>& broadcast_dimensions) {
  auto flow = MakeInstruction("broadcast", target_shape.ToProto(), {input});
  if (broadcast_dimensions.size() != input->shape().dimensions_size()) {
    throw Exception(
        "Broadcast must have the same size as the input shape: "
        "broadcast_dimensions=" +
        std::to_string(broadcast_dimensions.size()) +
        ", input_shape=" + HloTensorType(input->shape()).ToString());
  }
  const auto& input_shape = input->shape();
  for (size_t i = 0; i < broadcast_dimensions.size(); ++i) {
    auto dim = broadcast_dimensions[i];
    const auto& input_dim = input_shape.dimensions(i);
    if (input_dim != target_shape.GetDimension(dim)) {
      throw Exception(
          "Broadcast dimension must be equal to the target shape dimension");
    }
    flow->add_dimensions(dim);
  }
  return flow;
}

HloFlow HloBuilder::Reduce(HloFlow input, HloFlow initial,
                           HloComputation function,
                           const std::vector<int64_t>& reduction_dimensions) {
  HloTensorType target_shape(input->shape().element_type());
  for (size_t i = 0; i < input->shape().dimensions_size(); ++i) {
    if (std::find(reduction_dimensions.begin(), reduction_dimensions.end(),
                  i) == reduction_dimensions.end()) {
      target_shape.AddDimension(input->shape().dimensions(i));
    }
  }
  auto flow =
      MakeInstruction("reduce", target_shape.ToProto(), {input, initial});
  *flow->mutable_dimensions() = {reduction_dimensions.begin(),
                                 reduction_dimensions.end()};
  flow->add_called_computation_ids(function.idx());
  return flow;
}

HloFlow HloBuilder::Transpose(HloFlow input,
                              const std::vector<int64_t>& permutation) {
  HloTensorType target_shape(input->shape().element_type());
  for (size_t i = 0; i < permutation.size(); ++i) {
    target_shape.AddDimension(input->shape().dimensions(permutation[i]));
  }
  auto flow = MakeInstruction("transpose", target_shape.ToProto(), {input});
  *flow->mutable_dimensions() = permutation;
  return flow;
}

HloFlow HloBuilder::Add(HloFlow lhs, HloFlow rhs) {
  return MakeElementwiseInstruction("add", lhs, rhs);
}

HloFlow HloBuilder::Subtract(HloFlow lhs, HloFlow rhs) {
  return MakeElementwiseInstruction("subtract", lhs, rhs);
}

HloFlow HloBuilder::Multiply(HloFlow lhs, HloFlow rhs) {
  return MakeElementwiseInstruction("multiply", lhs, rhs);
}

HloFlow HloBuilder::Divide(HloFlow lhs, HloFlow rhs) {
  return MakeElementwiseInstruction("divide", lhs, rhs);
}

HloFlow HloBuilder::Maximum(HloFlow lhs, HloFlow rhs) {
  return MakeElementwiseInstruction("maximum", lhs, rhs);
}

HloFlow HloBuilder::Reshape(HloFlow input, const HloTensorType& new_shape) {
  if (input->shape().element_type() != new_shape.GetElementType()) {
    throw Exception("Reshape must have the same element type");
  }
  auto old_shape = HloTensorType(input->shape());
  if (old_shape.NumElements() != new_shape.NumElements()) {
    throw Exception("Reshape must have the same number of elements: " +
                    old_shape.ToString() + " vs " + new_shape.ToString());
  }
  return MakeInstruction("reshape", new_shape.ToProto(), {input});
}

HloFlow HloBuilder::Gather(HloFlow input, HloFlow indices,
                           size_t index_vector_dim,
                           const std::vector<int64_t>& offset_dims,
                           const std::vector<int64_t>& slice_sizes,
                           const std::vector<int64_t>& collapsed_slice_dims,
                           const std::vector<int64_t>& start_index_map,
                           bool indices_are_sorted, bool unique_indicies) {
  HloTensorType input_shape(input->shape());
  HloTensorType indices_shape(indices->shape());

  if (indices_shape.Rank() == index_vector_dim) {
    indices_shape.AddDimension(1);
  }

  HloTensorType output_shape(input_shape.GetElementType());
  size_t output_rank = offset_dims.size() + indices_shape.Rank() - 1;
  size_t offset_dims_idx = 0;
  size_t gather_dims_idx = 0;

  for (size_t i = 0; i < output_rank; ++i) {
    const bool is_in_offset = std::find(offset_dims.begin(), offset_dims.end(),
                                        i) != offset_dims.end();
    if (is_in_offset) {
      while (std::find(collapsed_slice_dims.begin(), collapsed_slice_dims.end(),
                       offset_dims_idx) != collapsed_slice_dims.end()) {
        offset_dims_idx++;
      }
      output_shape.AddDimension(slice_sizes[offset_dims_idx++]);
    } else {
      if (gather_dims_idx == index_vector_dim) ++gather_dims_idx;
      output_shape.AddDimension(indices_shape.GetDimension(gather_dims_idx++));
    }
  }

  auto flow =
      MakeInstruction("gather", output_shape.ToProto(), {input, indices});
  *flow->mutable_gather_slice_sizes() = slice_sizes;
  *flow->mutable_gather_dimension_numbers()->mutable_offset_dims() =
      offset_dims;
  *flow->mutable_gather_dimension_numbers()->mutable_collapsed_slice_dims() =
      collapsed_slice_dims;
  *flow->mutable_gather_dimension_numbers()->mutable_start_index_map() =
      start_index_map;
  flow->mutable_gather_dimension_numbers()->set_index_vector_dim(
      index_vector_dim);
  flow->set_indices_are_sorted(indices_are_sorted);
  flow->set_unique_indices(unique_indicies);
  return flow;
}

HloFlow HloBuilder::Dot(HloFlow lhs, HloFlow rhs,
                        const pblczero::XlaDotDimensionNumbers& dn) {
  HloTensorType lhs_shape(lhs->shape());
  HloTensorType rhs_shape(rhs->shape());
  HloTensorType new_shape(lhs_shape.GetElementType());
  if (lhs_shape.GetElementType() != rhs_shape.GetElementType()) {
    throw Exception("Dot operands must have the same element type");
  }
  if (dn.lhs_batch_dimensions_size() != dn.rhs_batch_dimensions_size()) {
    throw Exception("Dot batch dimensions must have the same size");
  }
  for (size_t i = 0; i < dn.lhs_batch_dimensions_size(); ++i) {
    auto lhs_dim = lhs_shape.GetDimension(dn.lhs_batch_dimensions(i));
    auto rhs_dim = rhs_shape.GetDimension(dn.rhs_batch_dimensions(i));
    if (lhs_dim != rhs_dim) {
      throw Exception("Dot batch dimensions must have the same size");
    }
    new_shape.AddDimension(lhs_dim);
  }
  if (dn.lhs_contracting_dimensions_size() !=
      dn.rhs_contracting_dimensions_size()) {
    throw Exception("Dot contracting dimensions must have the same size");
  }
  for (size_t i = 0; i < dn.lhs_contracting_dimensions_size(); ++i) {
    auto lhs_dim = lhs_shape.GetDimension(dn.lhs_contracting_dimensions(i));
    auto rhs_dim = rhs_shape.GetDimension(dn.rhs_contracting_dimensions(i));
    if (lhs_dim != rhs_dim) {
      throw Exception("Dot contracting dimensions must have the same size");
    }
  }
  // Sorry, github copilot generated the code below (well, above too). Enjoy!
  for (size_t i = 0; i < lhs->shape().dimensions_size(); ++i) {
    if (std::find(dn.lhs_batch_dimensions().begin(),
                  dn.lhs_batch_dimensions().end(),
                  i) == dn.lhs_batch_dimensions().end() &&
        std::find(dn.lhs_contracting_dimensions().begin(),
                  dn.lhs_contracting_dimensions().end(),
                  i) == dn.lhs_contracting_dimensions().end()) {
      new_shape.AddDimension(lhs_shape.GetDimension(i));
    }
  }
  for (size_t i = 0; i < rhs->shape().dimensions_size(); ++i) {
    if (std::find(dn.rhs_batch_dimensions().begin(),
                  dn.rhs_batch_dimensions().end(),
                  i) == dn.rhs_batch_dimensions().end() &&
        std::find(dn.rhs_contracting_dimensions().begin(),
                  dn.rhs_contracting_dimensions().end(),
                  i) == dn.rhs_contracting_dimensions().end()) {
      new_shape.AddDimension(rhs_shape.GetDimension(i));
    }
  }
  auto flow = MakeInstruction("dot", new_shape.ToProto(), {lhs, rhs});
  *flow->mutable_dot_dimension_numbers() = dn;
  return flow;
}

HloFlow HloBuilder::Tanh(HloFlow input) {
  return MakeInstruction("tanh", input->shape(), {input});
}

HloFlow HloBuilder::Sqrt(HloFlow input) {
  return MakeInstruction("sqrt", input->shape(), {input});
}

HloFlow HloBuilder::Rsqrt(HloFlow input) {
  return MakeInstruction("rsqrt", input->shape(), {input});
}

HloFlow HloBuilder::LogPlusOne(HloFlow input) {
  return MakeInstruction("log-plus-one", input->shape(), {input});
}

HloFlow HloBuilder::ExponentialMinusOne(HloFlow input) {
  return MakeInstruction("exponential-minus-one", input->shape(), {input});
}
HloFlow HloBuilder::Compare(HloFlow lhs, HloFlow rhs,
                            std::string_view direction) {
  if (lhs->shape().dimensions() != rhs->shape().dimensions()) {
    throw Exception("Elementwise operands must have the same shape");
  }
  HloTensorType shape(lhs->shape());
  shape.SetElementType(pblczero::XlaShapeProto::PRED);
  auto* flow = MakeInstruction("compare", shape.ToProto(), {lhs, rhs});
  flow->set_comparison_direction(direction);
  return flow;
}

HloFlow HloBuilder::Select(HloFlow condition, HloFlow on_true,
                           HloFlow on_false) {
  if (condition->shape().element_type() != pblczero::XlaShapeProto::PRED) {
    throw Exception("Select condition must have the PRED element type");
  }
  if (on_true->shape().dimensions() != on_false->shape().dimensions()) {
    throw Exception("Select operands must have the same shape");
  }
  if (on_true->shape().element_type() != on_false->shape().element_type()) {
    throw Exception("Select operands must have the same element type");
  }
  if (condition->shape().dimensions() != on_true->shape().dimensions()) {
    throw Exception("Select condition and operands must have the same shape");
  }
  return MakeInstruction("select", on_true->shape(),
                         {condition, on_true, on_false});
}

HloFlow HloBuilder::Negate(HloFlow input) {
  return MakeInstruction("negate", input->shape(), {input});
}

HloFlow HloBuilder::Exponential(HloFlow input) {
  return MakeInstruction("exponential", input->shape(), {input});
}

HloFlow HloBuilder::Tuple(const std::vector<HloFlow>& elements) {
  pblczero::XlaShapeProto shape;
  shape.set_element_type(pblczero::XlaShapeProto::TUPLE);
  for (const auto& element : elements) {
    *shape.add_tuple_shapes() = element->shape();
  }
  return MakeInstruction("tuple", shape, elements);
}

HloFlow HloBuilder::Concatenate(const std::vector<HloFlow>& inputs,
                                int64_t dimension) {
  if (inputs.empty()) {
    throw Exception("Concatenate must have at least one input");
  }
  HloTensorType shape(inputs[0]->shape());
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i]->shape().element_type() != shape.GetElementType()) {
      throw Exception("Concatenate operands must have the same element type");
    }
    if (inputs[i]->shape().dimensions_size() != shape.Rank()) {
      throw Exception("Concatenate operands must have the same rank");
    }
    for (size_t j = 0; j < shape.Rank(); ++j) {
      if (j == static_cast<size_t>(dimension)) {
        shape.SetDimension(
            j, shape.GetDimension(j) + inputs[i]->shape().dimensions(j));
      } else if (inputs[i]->shape().dimensions(j) != shape.GetDimension(j)) {
        std::string shapes;
        for (const auto& input : inputs) {
          shapes += HloTensorType(input->shape()).ToString() + ", ";
        }
        throw Exception("Concatenate operands must have the same shape, got " +
                        shapes + "axis=" + std::to_string(dimension));
      }
    }
  }
  auto flow = MakeInstruction("concatenate", shape.ToProto(), inputs);
  flow->add_dimensions(dimension);
  return flow;
}

HloFlow HloBuilder::Slice(
    HloFlow input,
    const std::vector<pblczero::HloInstructionProto::SliceDimensions>& slice) {
  HloTensorType current_shape(input->shape());
  if (slice.size() != current_shape.Rank()) {
    throw Exception(
        "Slice dimensions must have the same size as the input shape");
  }
  HloTensorType new_shape(current_shape.GetElementType());
  for (size_t i = 0; i < slice.size(); ++i) {
    const auto& dim = slice[i];
    if (dim.start() < 0 || dim.start() >= current_shape.GetDimension(i) ||
        dim.limit() < 0 || dim.limit() > current_shape.GetDimension(i) ||
        dim.start() >= dim.limit() || dim.stride() != 1) {
      throw Exception("Invalid slice dimensions, input shape=" +
                      current_shape.ToString() + ", dim=" + std::to_string(i) +
                      ", slice_start=" + std::to_string(dim.start()) +
                      ", slice_limit=" + std::to_string(dim.limit()));
    }
    // This / dim.stride() is pretty approximate, therefore there's a check
    // above.
    new_shape.AddDimension((dim.limit() - dim.start()) / dim.stride());
  }
  auto flow = MakeInstruction("slice", new_shape.ToProto(), {input});
  *flow->mutable_slice_dimensions() = slice;
  return flow;
}

namespace {
// Go over all "parameter" instructions of the computation and assign
// "parameter_number" field with increasing numbers.
// Normally it's not requiredm but in our case it's simpler.
// Outputs shapes and instruction names of parameters.
std::pair<std::vector<pblczero::XlaShapeProto>, std::vector<std::string>>
AssignParameterIndices(const InstructionList& comp) {
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

// Finalizes HloComputationProto (sets name, renumbers parameters, adds
// computation shape and root instruction).
pblczero::HloComputationProto MakeComputation(const InstructionList& comp,
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

// Assigns unique names to all instructions in the module.
// In StableHLO instructions are allowed to have numeric names, but in XLA HLO
// they are not, so we use "i"+number.
void HloBuilder::AssignInstructionNames() {
  // Every instruction in the module should have an unique name, numeric names
  // are allowed.
  size_t idx = 0;
  for (auto& instr : entry_computation_) {
    instr->set_name("i" + std::to_string(idx++));
  }
  for (auto& comp : dependent_computations_) {
    for (auto& instr : *comp.mutable_instructions()) {
      instr.set_name("i" + std::to_string(idx++));
    }
  }
}

std::optional<HloComputation> HloBuilder::GetComputationId(
    std::string_view name) const {
  auto iter = computation_names_.find(std::string(name));
  if (iter == computation_names_.end()) return std::nullopt;
  return HloComputation(iter->second);
}

HloComputation HloBuilder::AddComputation(std::string_view name,
                                          const HloBuilder& builder) {
  std::unordered_map<size_t, size_t> id_map;
  if (computation_names_.count(std::string(name))) {
    throw Exception("Computation with name " + std::string(name) +
                    " already exists");
  }

  const size_t computation_add_idx = dependent_computations_.size();

  // Insert all dependent computations of the passed builder.
  for (const auto& [name, id] : builder.computation_names_) {
    auto iter = computation_names_.find(name);
    if (iter != computation_names_.end()) {
      // TODO check that the computation is the same.
      id_map[id] = iter->second;
      continue;
    }
    const size_t new_id = dependent_computations_.size();
    id_map[id] = new_id;
    computation_names_[name] = new_id;
    dependent_computations_.push_back(
        ReassignInstructionIds(builder.dependent_computations_[id]));
    dependent_computations_.back().set_id(new_id);
  }

  // Insert passed builder's entry computation as current builder's dependent
  // computation.
  {
    const size_t new_id = dependent_computations_.size();
    computation_names_[std::string(name)] = new_id;
    dependent_computations_.push_back(ReassignInstructionIds(
        MakeComputation(builder.entry_computation_, name, new_id)));
  }

  // Remap operand ids in the dependent computations.
  for (size_t i = computation_add_idx; i < dependent_computations_.size();
       ++i) {
    auto* comp = &dependent_computations_[i];
    for (auto& instr : *comp->mutable_instructions()) {
      for (int64_t& called_id : *instr.mutable_called_computation_ids()) {
        called_id = id_map.at(called_id);
      }
    }
  }

  return HloComputation(dependent_computations_.back().id());
}

pblczero::HloComputationProto HloBuilder::ReassignInstructionIds(
    pblczero::HloComputationProto computation) {
  std::unordered_map<size_t, size_t> id_map;
  for (auto& instr : *computation.mutable_instructions()) {
    id_map[instr.id()] = next_instruction_id_++;
  }
  for (auto& instr : *computation.mutable_instructions()) {
    instr.set_id(id_map.at(instr.id()));
    for (int64_t& operand_id : *instr.mutable_operand_ids()) {
      operand_id = id_map.at(operand_id);
    }
  }
  computation.set_root_id(id_map.at(computation.root_id()));
  return computation;
}

pblczero::HloModuleProto HloBuilder::BuildModule(std::string_view name) {
  AssignInstructionNames();
  pblczero::HloModuleProto module;
  for (auto& comp : dependent_computations_) {
    *module.add_computations() = comp;
  }
  module.set_name(name);
  module.set_entry_computation_name("main");
  const size_t entry_computation_id = dependent_computations_.size();
  module.set_entry_computation_id(entry_computation_id);
  *module.add_computations() =
      MakeComputation(entry_computation_, "main", entry_computation_id);
  *module.mutable_host_program_shape() =
      module.computations().back().program_shape();
  return module;
}

}  // namespace lczero
