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

#include "neural/xla/onnx2hlo.h"

#include <unordered_map>

#include "neural/onnx/onnx.pb.h"
#include "neural/xla/hlo.pb.h"
#include "neural/xla/hlo_builder.h"
#include "neural/xla/print_hlo.h"
#include "utils/exception.h"

namespace lczero {
namespace {

pblczero::XlaShapeProto::Type OnnxTypeToXlaType(
    const pblczero::TensorProto::DataType& type) {
  switch (type) {
    case pblczero::TensorProto::FLOAT:
      return pblczero::XlaShapeProto::F32;
    case pblczero::TensorProto::UINT8:
      return pblczero::XlaShapeProto::U8;
    case pblczero::TensorProto::INT8:
      return pblczero::XlaShapeProto::S8;
    case pblczero::TensorProto::UINT16:
      return pblczero::XlaShapeProto::U16;
    case pblczero::TensorProto::INT16:
      return pblczero::XlaShapeProto::S16;
    case pblczero::TensorProto::INT32:
      return pblczero::XlaShapeProto::S32;
    case pblczero::TensorProto::INT64:
      return pblczero::XlaShapeProto::S64;
    case pblczero::TensorProto::BOOL:
      return pblczero::XlaShapeProto::PRED;
    case pblczero::TensorProto::FLOAT16:
      return pblczero::XlaShapeProto::F16;
    case pblczero::TensorProto::DOUBLE:
      return pblczero::XlaShapeProto::F64;
    case pblczero::TensorProto::UINT32:
      return pblczero::XlaShapeProto::U32;
    case pblczero::TensorProto::UINT64:
      return pblczero::XlaShapeProto::U64;
    case pblczero::TensorProto::COMPLEX64:
      return pblczero::XlaShapeProto::C64;
    case pblczero::TensorProto::COMPLEX128:
      return pblczero::XlaShapeProto::C128;
    case pblczero::TensorProto::BFLOAT16:
      return pblczero::XlaShapeProto::BF16;
    case pblczero::TensorProto::STRING:
    default:
      throw Exception("Unsupported ONNX type " +
                      pblczero::TensorProto::DataType_Name(type));
  }
}

// Converts an ONNX shape to an XLA shape, replacing the batch dimension with
// the provided batch size.
pblczero::XlaShapeProto OnnxShapeToXlaShape(const pblczero::TypeProto& type,
                                            std::optional<size_t> batch_size) {
  pblczero::XlaShapeProto shape;
  shape.set_element_type(OnnxTypeToXlaType(type.tensor_type().elem_type()));
  for (const auto& dim : type.tensor_type().shape().dim()) {
    if (dim.has_dim_value()) {
      shape.add_dimensions(dim.dim_value());
      continue;
    }
    if (dim.dim_param() == "batch") {
      if (batch_size.has_value()) {
        shape.add_dimensions(batch_size.value());
        continue;
      }
      throw Exception("Batch size not provided");
    }
    throw Exception("Unsupported dimension type " + type.OutputAsJson());
  }
  ResetXlaShapeProtoLayout(&shape);
  return shape;
}

// Type is not a field of the ONNX tensor, so this function extracts the shape
// and converts it (discarding the data).
pblczero::XlaShapeProto OnnxTensorToXlaShape(
    const pblczero::TensorProto& tensor) {
  pblczero::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(tensor.data_type());
  for (const auto& dim : tensor.dims()) {
    type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return OnnxShapeToXlaShape(type, std::nullopt);
}

// Converts an ONNX tensor to an XLA literal (which is a shape and a data).
pblczero::XlaLiteralProto OnnxTensorToXlaLiteral(
    const pblczero::TensorProto& tensor) {
  pblczero::XlaLiteralProto literal;
  *literal.mutable_shape() = OnnxTensorToXlaShape(tensor);

  auto convert = [&](std::string_view src, /*std::vector<T>*/ auto* dst) {
    using value_type =
        typename std::remove_pointer<decltype(dst)>::type::value_type;
    dst->assign(reinterpret_cast<const value_type*>(src.data()),
                reinterpret_cast<const value_type*>(src.data() + src.size()));
  };

  switch (tensor.data_type()) {
    case pblczero::TensorProto::FLOAT:
      convert(tensor.raw_data(), literal.mutable_f32s());
      break;
    case pblczero::TensorProto::INT64:
      convert(tensor.raw_data(), literal.mutable_s64s());
      break;
    default:
      throw Exception("Cannot convert ONNX tensor to XLA literal for type " +
                      pblczero::XlaShapeProto::Type_Name(
                          OnnxTypeToXlaType(tensor.data_type())));
  }
  return literal;
}

class Onnx2HloConverter {
 public:
  Onnx2HloConverter(const Onnx2HloOptions& options) : options_(options) {
    onnx_op_to_builder_["Add"] = &Onnx2HloConverter::OpAdd;
    onnx_op_to_builder_["Conv"] = &Onnx2HloConverter::OpConv;
    onnx_op_to_builder_["MatMul"] = &Onnx2HloConverter::OpMatMul;
    onnx_op_to_builder_["Relu"] = &Onnx2HloConverter::OpRelu;
    onnx_op_to_builder_["Reshape"] = &Onnx2HloConverter::OpReshape;
    onnx_op_to_builder_["Tanh"] = &Onnx2HloConverter::OpTanh;
  }

  Onnx2HloResult Convert(const pblczero::ModelProto& onnx_model,
                         size_t minibatch_size) {
    batch_size_ = minibatch_size;
    // Populate the set of ONNX initializers (constants), but not emit them for
    // now. They are emitted lazily so that they appear close to the first use.
    BuildInitializerMapping(onnx_model);
    // Convert ONNX inputs to HLO parameters.
    BuildInputs(onnx_model.graph().input());
    Onnx2HloResult result;
    try {
      BuildGraph(onnx_model.graph());
      // Convert ONNX outputs to HLO result.
      result.outputs = BuildOutputs(onnx_model.graph().output());
      for (size_t i = 0; i < params_.size(); ++i) {
        const auto& param = params_[i];
        auto& dst = param.is_constant ? result.constants : result.inputs;
        dst.push_back({i, param.name, param.flow->shape()});
      }
    } catch (Exception& e) {
      if (!options_.debugging_allow_partial_result) throw;
      CERR << "Ignoring error in ONNX to HLO conversion: " << e.what();
    }
    result.hlo_module = builder_.Build("onnx_model");
    return result;
  }

 private:
  std::vector<Onnx2HloResult::NamedTensor> BuildOutputs(
      const std::vector<pblczero::ValueInfoProto>& graph_output) {
    // Gathers outputs into the root tuple, optionally converting their type if
    // I/O type is different from the instruction output.
    std::vector<Onnx2HloResult::NamedTensor> result;
    std::vector<HloFlow> outputs;
    for (size_t i = 0; i < graph_output.size(); ++i) {
      const auto& output = graph_output[i];
      auto flow = GetFlowByName(std::string(output.name()));
      if (flow->shape().element_type() != options_.io_type) {
        auto ctx = HloContext(&builder_);
        ctx.SetOpType("output");
        ctx.SetOpName(output.name());
        flow = builder_.Convert(flow, options_.io_type);
      }
      result.push_back({i, std::string(output.name()), flow->shape()});
      outputs.push_back(flow);
    }
    builder_.Tuple(outputs);
    return result;
  }

  void BuildInitializerMapping(const pblczero::ModelProto& onnx_model) {
    for (const auto& tensor : onnx_model.graph().initializer()) {
      initializers_[std::string(tensor.name())] = &tensor;
    }
  }

  // Checks that the ONNX node doesn't have any unknown attributes.
  void CheckKnownAttributes(
      const pblczero::NodeProto& node,
      const std::initializer_list<std::string_view> attributes) {
    for (const auto& attribute : node.attribute()) {
      if (std::find(attributes.begin(), attributes.end(), attribute.name()) ==
          attributes.end()) {
        throw Exception("Unknown attribute " + std::string(attribute.name()));
      }
    }
  }

  // Fetches an HloFlow by name. If the name is not in the map, check whether
  // there is an initializer for it, and either create a constant or a parameter
  // depending on its size.
  HloFlow GetFlowByName(const std::string& name) {
    auto iter = onnx_name_to_hlo_flow_.find(name);
    if (iter != onnx_name_to_hlo_flow_.end()) return iter->second;

    auto iter2 = initializers_.find(name);
    if (iter2 == initializers_.end()) {
      throw Exception("Unknown input " + name);
    }
    auto ctx = HloContext(&builder_);
    ctx.SetOpType("initializer");
    ctx.SetOpName(name);

    HloFlow flow = nullptr;
    if (iter2->second->raw_data().size() <= options_.max_inline_constant_size) {
      flow = builder_.Constant(OnnxTensorToXlaLiteral(*iter2->second));
    } else {
      const auto shape = OnnxTensorToXlaShape(*iter2->second);
      flow = MakeParameter(name, shape, true);
    }
    onnx_name_to_hlo_flow_[name] = flow;
    return flow;
  }

  // A helper function to fetch an input of ONNX node by index.
  HloFlow GetInput(const pblczero::NodeProto& node, size_t idx,
                   bool optional = false) {
    if (idx >= node.input_size()) {
      if (optional) return nullptr;
      throw Exception("Input " + std::to_string(idx) + " not set");
    }
    return GetFlowByName(std::string(node.input(idx)));
  }

  // A helper function to fetch an attribute of ONNX node by name.
  const pblczero::AttributeProto* GetAttribute(const pblczero::NodeProto& node,
                                               std::string_view name,
                                               bool optional = false) {
    for (const auto& attribute : node.attribute()) {
      if (attribute.name() == name) return &attribute;
    }
    if (optional) return nullptr;
    throw Exception("Attribute " + std::string(name) + " not set");
  }

  /////////////////////////////////////////////////////////////////////////////
  // ONNX operations
  /////////////////////////////////////////////////////////////////////////////

  std::vector<HloFlow> OpConv(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, {"pads", "kernel_shape"});
    auto* input = GetInput(node, 0);
    auto* kernel = GetInput(node, 1);
    auto* bias = GetInput(node, 2, true);

    pblczero::XlaConvolutionDimensionNumbers dn;
    dn.set_input_batch_dimension(0);
    dn.set_input_feature_dimension(1);
    dn.set_kernel_input_feature_dimension(1);
    dn.set_kernel_output_feature_dimension(0);
    dn.set_output_batch_dimension(0);
    dn.set_output_feature_dimension(1);
    const size_t num_dims = input->shape().dimensions_size() - 2;
    for (size_t i = 0; i < num_dims; ++i) {
      dn.add_input_spatial_dimensions(i + 2);
      dn.add_kernel_spatial_dimensions(i + 2);
      dn.add_output_spatial_dimensions(i + 2);
    }

    const auto* pads = GetAttribute(node, "pads");
    const auto* kernel_shape = GetAttribute(node, "kernel_shape");
    if (!pads || pads->ints_size() != 2 * num_dims) {
      throw Exception("'pads' attribute not set or wrong size");
    }
    if (!kernel_shape || kernel_shape->ints_size() != num_dims) {
      throw Exception("'kernel_shape' attribute not set or wrong size");
    }
    pblczero::XlaWindow window;
    for (size_t i = 0; i < input->shape().dimensions_size() - 2; ++i) {
      auto* dim = window.add_dimensions();
      dim->set_size(kernel_shape->ints(i));
      dim->set_stride(1);
      dim->set_padding_low(pads->ints(i));
      dim->set_padding_high(pads->ints(i + num_dims));
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
    }

    auto* conv = builder_.Convolution(input, kernel, window, dn);

    if (!bias) return {conv};
    auto* flow = builder_.Broadcast(bias, conv->shape(), {1});
    return {builder_.Add(conv, flow)};
  }

  std::vector<HloFlow> OpRelu(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, {});
    auto* input = GetInput(node, 0);
    auto* zero = MakeScalar(0, input->shape().element_type());
    zero = builder_.Broadcast(zero, input->shape(), {});
    return {builder_.Maximum(input, zero)};
  }

  std::vector<HloFlow> OpTanh(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, {});
    auto* input = GetInput(node, 0);
    return {builder_.Tanh(input)};
  }

  std::vector<HloFlow> OpAdd(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, {});
    auto* lhs = GetInput(node, 0);
    auto* rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_.Add(lhs, rhs)};
  }

  std::vector<HloFlow> OpReshape(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, {});
    auto* input = GetInput(node, 0);
    if (node.input_size() < 2) {
      throw Exception("Reshape requires a shape input");
    }
    auto dims_tensor = initializers_.find(std::string(node.input(1)));
    if (dims_tensor == initializers_.end()) {
      throw Exception("Reshape only supports constant shape");
    }
    auto new_dims = OnnxTensorToXlaLiteral(*dims_tensor->second).s64s();
    pblczero::XlaShapeProto new_shape;
    new_shape.set_element_type(input->shape().element_type());
    for (size_t i = 0; i < new_dims.size(); ++i) {
      auto dim = new_dims[i];
      if (dim == -1) dim = batch_size_;
      if (dim == 0) {
        if (new_dims.size() != input->shape().dimensions_size()) {
          throw Exception("Reshape cannot infer shape when rank changes");
        }
        dim = input->shape().dimensions(i);
      }
      new_shape.add_dimensions(dim);
    }
    ResetXlaShapeProtoLayout(&new_shape);
    return {builder_.Reshape(input, new_shape)};
  }

  std::vector<HloFlow> OpMatMul(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, {});
    auto* lhs = GetInput(node, 0);
    auto* rhs = GetInput(node, 1);
    if (lhs->shape().dimensions_size() != 2 ||
        rhs->shape().dimensions_size() != 2) {
      throw Exception("MatMul only implemented for 2D inputs so far");
    }
    pblczero::XlaDotDimensionNumbers dn;
    dn.add_lhs_contracting_dimensions(1);
    dn.add_rhs_contracting_dimensions(0);
    return {builder_.Dot(lhs, rhs, dn)};
  }

  /////////////////////////////////////////////////////////////////////////////

  // Makes a scalar constant (usually 0 or 1) of a given type.
  template <typename T>
  HloFlow MakeScalar(T value, pblczero::XlaShapeProto::Type type) {
    pblczero::XlaLiteralProto literal;
    literal.mutable_shape()->set_element_type(type);
    literal.mutable_shape()->mutable_layout();
    switch (type) {
      case pblczero::XlaShapeProto::F32:
        literal.add_f32s(value);
        break;
      default:
        throw Exception("Unsupported type for zero constant");
    }
    return builder_.Constant(literal);
  }

  // Take two inputs and optionally performs numpy-style broadcasting to make
  // them equal shape.
  std::pair<HloFlow, HloFlow> EqualizeShape(HloFlow lhs, HloFlow rhs) {
    const auto& lhs_dims = lhs->shape().dimensions();
    const auto& rhs_dims = rhs->shape().dimensions();

    const size_t num_dims = std::max(lhs_dims.size(), rhs_dims.size());
    std::vector<int64_t> output_dims(num_dims);
    std::vector<int64_t> lhs_broadcast_dims;
    std::vector<int64_t> rhs_broadcast_dims;
    bool lhs_broadcast = lhs_dims.size() < num_dims;
    bool rhs_broadcast = rhs_dims.size() < num_dims;

    for (size_t i = 0; i < num_dims; ++i) {
      int lhs_idx = i + lhs_dims.size() - num_dims;
      int rhs_idx = i + rhs_dims.size() - num_dims;
      const auto lhs_dim = (lhs_idx < 0) ? 1 : lhs_dims[lhs_idx];
      const auto rhs_dim = (rhs_idx < 0) ? 1 : rhs_dims[rhs_idx];
      if (lhs_dim != rhs_dim) {
        if (lhs_dim != 1 && rhs_dim != 1) {
          throw Exception("Incompatible shapes for broadcast");
        }
        if (lhs_dim == 1) lhs_broadcast = true;
        if (rhs_dim == 1) rhs_broadcast = true;
      }
      if (lhs_idx >= 0) lhs_broadcast_dims.push_back(i);
      if (rhs_idx >= 0) rhs_broadcast_dims.push_back(i);
    }

    if (lhs_broadcast) {
      lhs = builder_.Broadcast(lhs, rhs->shape(), lhs_broadcast_dims);
    }
    if (rhs_broadcast) {
      rhs = builder_.Broadcast(rhs, lhs->shape(), rhs_broadcast_dims);
    }
    return {lhs, rhs};
  }

  // Convert ONNX inputs to HLO parameters.
  void BuildInputs(const std::vector<pblczero::ValueInfoProto>& inputs) {
    for (const auto& input : inputs) {
      auto ctx = HloContext(&builder_);
      ctx.SetOpType("input");
      ctx.SetOpName(input.name());
      auto out_shape = OnnxShapeToXlaShape(input.type(), batch_size_);
      auto in_shape = out_shape;
      in_shape.set_element_type(options_.io_type);
      const auto* flow =
          MakeParameter(std::string(input.name()), in_shape, false);
      flow = builder_.Convert(flow, out_shape.element_type());
      onnx_name_to_hlo_flow_[std::string(input.name())] = flow;
    }
  }

  // Makes a parameter instruction (for inputs or large constants).
  HloFlow MakeParameter(const std::string& name,
                        const pblczero::XlaShapeProto& shape,
                        bool is_constant) {
    auto* res = builder_.Parameter(shape);
    params_.push_back({name, res, is_constant});
    return res;
  }

  void BuildGraph(const pblczero::GraphProto& graph) {
    for (const auto& node : graph.node()) {
      // Set up the context so that nodes have metadata from the original ONNX.
      auto ctx = HloContext(&builder_);
      ctx.SetOpType(node.op_type());
      ctx.SetOpName(node.name());
      DispatchNode(node);
    }
  }

  // Calls the correct function to handle the ONNX node, and stores output in
  // the map.
  void DispatchNode(const pblczero::NodeProto& node) {
    auto iter = onnx_op_to_builder_.find(std::string(node.op_type()));
    if (iter == onnx_op_to_builder_.end()) {
      throw Exception("Unsupported ONNX op[" + std::string(node.op_type()) +
                      "] name=[" + std::string(node.name()) + "]");
    }
    try {
      auto outputs = (this->*iter->second)(node);
      if (outputs.size() != node.output_size()) {
        throw Exception("Node produced wrong number of outputs");
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        onnx_name_to_hlo_flow_[std::string(node.output(i))] = outputs[i];
      }
    } catch (Exception& e) {
      throw Exception("Error in ONNX op[" + std::string(node.op_type()) +
                      "] name=[" + std::string(node.name()) + "]: " + e.what());
    }
  }

  std::unordered_map<std::string, HloFlow> onnx_name_to_hlo_flow_;
  std::unordered_map<std::string, std::vector<HloFlow> (Onnx2HloConverter::*)(
                                      const pblczero::NodeProto&)>
      onnx_op_to_builder_;
  std::unordered_map<std::string, const pblczero::TensorProto*> initializers_;
  HloBuilder builder_;
  size_t batch_size_ = 0;
  Onnx2HloOptions options_;
  struct Param {
    std::string name;
    HloFlow flow;
    bool is_constant;
  };
  std::vector<Param> params_;
};

}  // namespace

Onnx2HloResult ConvertOnnxToHlo(const pblczero::ModelProto& onnx_model,
                                size_t minibatch_size,
                                const Onnx2HloOptions& options) {
  Onnx2HloConverter converter(options);
  return converter.Convert(onnx_model, minibatch_size);
}

std::unique_ptr<XlaTensor> OnnxTensorToXlaTensor(
    const pblczero::TensorProto& onnx_tensor) {
  switch (onnx_tensor.data_type()) {
    case pblczero::TensorProto::FLOAT:
      return std::make_unique<XlaTensorNotOwned>(onnx_tensor.dims(),
                                                 onnx_tensor.raw_data(),
                                                 pblczero::XlaShapeProto::F32);
    default:
      throw Exception(
          "Unsupported ONNX tensor type for buffer conversion " +
          pblczero::TensorProto::DataType_Name(onnx_tensor.data_type()));
  }
}

}  // namespace lczero