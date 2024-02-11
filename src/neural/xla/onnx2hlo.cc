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
    case pblczero::TensorProto::UNDEFINED:
      throw Exception("Unsupported ONNX type " +
                      pblczero::TensorProto::DataType_Name(type));
  }
}
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
  for (size_t i = 0; i < shape.dimensions_size(); ++i) {
    shape.add_is_dynamic_dimension(false);
    shape.mutable_layout()->add_minor_to_major(shape.dimensions_size() - i - 1);
  }

  return shape;
}

pblczero::XlaShapeProto OnnxTensorToXlaShape(
    const pblczero::TensorProto& tensor) {
  pblczero::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(tensor.data_type());
  for (const auto& dim : tensor.dims()) {
    type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return OnnxShapeToXlaShape(type, std::nullopt);
}

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
    PopulateOpMapping();
  }

  Onnx2HloResult Convert(const pblczero::ModelProto& onnx_model,
                         size_t minibatch_size) {
    batch_size_ = minibatch_size;
    BuildInitializerMapping(onnx_model);
    BuildInputs(onnx_model.graph().input());
    try {
      BuildGraph(onnx_model.graph());
    } catch (Exception& e) {
      CERR << "Error in ONNX graph: " << e.what();  // DO NOT SUBMIT
    }

    Onnx2HloResult result;
    result.hlo_module = builder_.Build("onnx_model");
    // TO DELETE, DEBUG ONLY
    PrettyPrintHlo(result.hlo_module, {}, std::cout);
    return result;
  }

 private:
  void BuildInitializerMapping(const pblczero::ModelProto& onnx_model) {
    for (const auto& tensor : onnx_model.graph().initializer()) {
      initializers_[std::string(tensor.name())] = &tensor;
    }
  }

  void PopulateOpMapping() {
    onnx_op_to_builder_["Conv"] = &Onnx2HloConverter::OpConv;
  }

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

  HloFlow GetInput(const pblczero::NodeProto& node, size_t idx,
                   bool optional = false) {
    if (idx >= node.input_size()) {
      if (optional) return nullptr;
      throw Exception("Input " + std::to_string(idx) + " not set");
    }
    return GetFlowByName(std::string(node.input(idx)));
  }

  const pblczero::AttributeProto* GetAttribute(const pblczero::NodeProto& node,
                                               std::string_view name,
                                               bool optional = false) {
    for (const auto& attribute : node.attribute()) {
      if (attribute.name() == name) return &attribute;
    }
    if (optional) return nullptr;
    throw Exception("Attribute " + std::string(name) + " not set");
  }

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

  // std::pair<HloFlow, HloFlow> EqualizeShape(
  //     HloFlow lhs, HloFlow rhs);

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

  HloFlow MakeParameter(const std::string& name,
                        const pblczero::XlaShapeProto& shape,
                        bool is_constant) {
    auto* res = builder_.Parameter(shape);
    params_.push_back({name, res, is_constant});
    return res;
  }

  void BuildGraph(const pblczero::GraphProto& graph) {
    for (const auto& node : graph.node()) {
      auto ctx = HloContext(&builder_);
      ctx.SetOpType(node.op_type());
      ctx.SetOpName(node.name());
      DispatchNode(node);
    }
  }

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

}  // namespace lczero