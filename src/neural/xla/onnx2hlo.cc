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

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "neural/onnx/onnx.pb.h"
#include "neural/xla/hlo.pb.h"
#include "neural/xla/hlo_builder.h"
#include "neural/xla/print_hlo.h"
#include "utils/exception.h"

namespace lczero {
namespace {

template <typename T>
void LiteralOp2(pblczero::XlaLiteralProto* dst,
                const pblczero::XlaLiteralProto& lhs, T&& func) {
  switch (lhs.shape().element_type()) {
    case pblczero::XlaShapeProto::S64:
      func(dst->mutable_s64s(), lhs.s64s());
      break;
    default:
      throw Exception(
          "Unsupported type for constant input " +
          pblczero::XlaShapeProto::Type_Name(lhs.shape().element_type()));
  }
}

pblczero::XlaLiteralProto ConstOpConcat(
    const std::vector<pblczero::XlaLiteralProto>& inputs, int axis) {
  if (inputs.empty()) {
    throw Exception("Concat requires at least one input");
  }
  if (axis != 0) {
    throw Exception("Concat only supports axis 0 for now");
  }
  HloTensorType shape(inputs[0].shape());
  shape.SetDimension(axis, 0);
  pblczero::XlaLiteralProto result;
  for (const auto& input : inputs) {
    if (input.shape().dimensions_size() != 1) {
      throw Exception(
          "For constant concat, only 1D inputs are supported for now");
    }
    if (input.shape().element_type() != shape.GetElementType()) {
      throw Exception("All inputs must have the same type");
    }
    shape.SetDimension(
        axis, shape.GetDimension(axis) + input.shape().dimensions(axis));
    LiteralOp2(&result, input, [](auto* dst, const auto& src) {
      dst->insert(dst->end(), src.begin(), src.end());
    });
  }
  *result.mutable_shape() = shape.ToProto();
  return result;
}

pblczero::XlaLiteralProto ConstOpSlice(
    const pblczero::XlaLiteralProto& input,
    const std::vector<pblczero::HloInstructionProto::SliceDimensions>& slice) {
  if (input.shape().dimensions_size() != 1 || slice.size() != 1) {
    throw Exception(
        "For constant slices, only 1D inputs are supported for now");
  }
  if (slice[0].stride() != 1) {
    throw Exception("For constant slices, only stride 1 is supported for now");
  }
  HloTensorType shape(input.shape().element_type());
  shape.AddDimension(slice[0].limit() - slice[0].start());
  pblczero::XlaLiteralProto result;
  LiteralOp2(&result, input, [&slice](auto* dst, const auto& src) {
    dst->insert(dst->end(), src.begin() + slice[0].start(),
                src.begin() + slice[0].limit());
  });
  *result.mutable_shape() = shape.ToProto();
  return result;
}

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

std::pair<bool, bool> IsConstantSortedUnique(
    const pblczero::XlaLiteralProto& literal) {
  auto is_sorted_unique = [](const auto& values) {
    using type =
        typename std::remove_reference<decltype(values)>::type::value_type;
    std::unordered_set<type> seen;
    type prev = values[0];
    bool sorted = true;
    bool unique = true;
    for (const auto& value : values) {
      if (value < prev) {
        sorted = false;
      }
      prev = value;
      if (seen.insert(value).second == false) {
        unique = false;
      }
    }
    return std::make_pair(sorted, unique);
  };

  switch (literal.shape().element_type()) {
    case pblczero::XlaShapeProto::S32:
      return is_sorted_unique(literal.s32s());
    default:
      throw Exception(
          "Unsupported type for constant input " +
          pblczero::XlaShapeProto::Type_Name(literal.shape().element_type()));
  }
}

// Converts an ONNX shape to an XLA shape, replacing the batch dimension with
// the provided batch size.
HloTensorType OnnxShapeToHloTensorType(const pblczero::TypeProto& type,
                                       std::optional<size_t> batch_size) {
  HloTensorType shape;
  shape.SetElementType(OnnxTypeToXlaType(type.tensor_type().elem_type()));
  for (const auto& dim : type.tensor_type().shape().dim()) {
    if (dim.has_dim_value()) {
      shape.AddDimension(dim.dim_value());
      continue;
    }
    if (dim.dim_param() == "batch") {
      if (batch_size.has_value()) {
        shape.AddDimension(batch_size.value());
        continue;
      }
      throw Exception("Batch size not provided");
    }
    throw Exception("Unsupported dimension type " + type.OutputAsJson());
  }
  return shape;
}

// Type is not a field of the ONNX tensor, so this function extracts the shape
// and converts it (discarding the data).
HloTensorType OnnxTensorToHloTensorType(const pblczero::TensorProto& tensor) {
  pblczero::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(tensor.data_type());
  for (const auto& dim : tensor.dims()) {
    type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return OnnxShapeToHloTensorType(type, std::nullopt);
}

// Converts an ONNX tensor to an XLA literal (which is a shape and a data).
pblczero::XlaLiteralProto OnnxTensorToXlaLiteral(
    const pblczero::TensorProto& tensor) {
  pblczero::XlaLiteralProto literal;
  *literal.mutable_shape() = OnnxTensorToHloTensorType(tensor).ToProto();

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
    case pblczero::TensorProto::INT32:
      convert(tensor.raw_data(), literal.mutable_s32s());
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
    onnx_op_to_builder_["Cast"] = &Onnx2HloConverter::OpCast;
    onnx_op_to_builder_["Concat"] = &Onnx2HloConverter::OpConcat;
    onnx_op_to_builder_["Conv"] = &Onnx2HloConverter::OpConv;
    onnx_op_to_builder_["Gather"] = &Onnx2HloConverter::OpGather;
    onnx_op_to_builder_["GlobalAveragePool"] =
        &Onnx2HloConverter::OpGlobalAveragePool;
    onnx_op_to_builder_["Expand"] = &Onnx2HloConverter::OpExpand;
    onnx_op_to_builder_["Identity"] = &Onnx2HloConverter::OpIdentity;
    onnx_op_to_builder_["LayerNormalization"] =
        &Onnx2HloConverter::OpLayerNormalization;
    onnx_op_to_builder_["MatMul"] = &Onnx2HloConverter::OpMatMul;
    onnx_op_to_builder_["Mul"] = &Onnx2HloConverter::OpMul;
    onnx_op_to_builder_["Reciprocal"] = &Onnx2HloConverter::OpReciprocal;
    onnx_op_to_builder_["ReduceMean"] = &Onnx2HloConverter::OpReduceMean;
    onnx_op_to_builder_["Relu"] = &Onnx2HloConverter::OpRelu;
    onnx_op_to_builder_["Reshape"] = &Onnx2HloConverter::OpReshape;
    onnx_op_to_builder_["Selu"] = &Onnx2HloConverter::OpSelu;
    onnx_op_to_builder_["Sigmoid"] = &Onnx2HloConverter::OpSigmoid;
    onnx_op_to_builder_["Shape"] = &Onnx2HloConverter::OpShape;
    onnx_op_to_builder_["Slice"] = &Onnx2HloConverter::OpSlice;
    onnx_op_to_builder_["Softmax"] = &Onnx2HloConverter::OpSoftmax;
    onnx_op_to_builder_["Softplus"] = &Onnx2HloConverter::OpSoftplus;
    onnx_op_to_builder_["Split"] = &Onnx2HloConverter::OpSplit;
    onnx_op_to_builder_["Sqrt"] = &Onnx2HloConverter::OpSqrt;
    onnx_op_to_builder_["Squeeze"] = &Onnx2HloConverter::OpSqueeze;
    onnx_op_to_builder_["Sub"] = &Onnx2HloConverter::OpSub;
    onnx_op_to_builder_["Tanh"] = &Onnx2HloConverter::OpTanh;
    onnx_op_to_builder_["Transpose"] = &Onnx2HloConverter::OpTranspose;
  }

  Onnx2HloResult Convert(const pblczero::ModelProto& onnx_model,
                         size_t minibatch_size) {
    batch_size_ = minibatch_size;
    opset_version_ = onnx_model.opset_import(0).version();
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
    result.hlo_module = builder_.BuildModule("onnx_model");
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
      const pblczero::NodeProto& node, size_t max_inputs,
      const std::initializer_list<std::string_view> attributes) {
    if (node.input_size() > max_inputs) {
      throw Exception("Too many inputs for " + std::string(node.op_type()));
    }
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
    if (auto iter = onnx_name_to_hlo_flow_.find(name);
        iter != onnx_name_to_hlo_flow_.end()) {
      return iter->second;
    }

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
      const auto shape = OnnxTensorToHloTensorType(*iter2->second);
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

  std::optional<pblczero::XlaLiteralProto> GetConstantInput(
      const pblczero::NodeProto& node, size_t idx, bool optional = false) {
    if (idx >= node.input_size()) {
      if (optional) return std::nullopt;
      throw Exception("Input " + std::to_string(idx) + " not set");
    }
    const std::string name(node.input(idx));
    if (auto tensor = initializers_.find(name); tensor != initializers_.end()) {
      return OnnxTensorToXlaLiteral(*tensor->second);
    }
    if (auto iter = onnx_name_to_hlo_flow_.find(name);
        iter != onnx_name_to_hlo_flow_.end() &&
        iter->second->opcode() == "constant") {
      return iter->second->literal();
    }
    throw Exception("Constant input " + std::string(node.input(idx)) +
                    " not found");
  }

  bool AllInputsConstant(const pblczero::NodeProto& node) {
    for (const auto& input : node.input()) {
      const std::string name(input);
      if (initializers_.count(name)) continue;
      if (auto iter = onnx_name_to_hlo_flow_.find(name);
          iter != onnx_name_to_hlo_flow_.end() &&
          iter->second->opcode() == "constant") {
        continue;
      }
      return false;
    }
    return true;
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
    CheckKnownAttributes(node, 3, {"pads", "kernel_shape"});
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
    auto* flow = builder_.Broadcast(bias, HloTensorType(conv->shape()), {1});
    return {builder_.Add(conv, flow)};
  }

  std::vector<HloFlow> OpRelu(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* input = GetInput(node, 0);
    auto* zero = MakeScalar(0, input->shape().element_type());
    zero = builder_.Broadcast(zero, HloTensorType(input->shape()), {});
    return {builder_.Maximum(input, zero)};
  }

  std::vector<HloFlow> OpSelu(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"alpha", "gamma"});
    auto* input = GetInput(node, 0);
    auto* palpha = GetAttribute(node, "alpha", true);
    double alpha = palpha ? palpha->f() : 1.6732632423543772848170429916717;
    auto* pgamma = GetAttribute(node, "gamma", true);
    double gamma = pgamma ? pgamma->f() : 1.0507009873554804934193349852946;
    auto* neg = builder_.Multiply(
        builder_.Broadcast(MakeScalar(alpha, input->shape().element_type()),
                           HloTensorType(input->shape()), {}),
        builder_.ExponentialMinusOne(input));
    auto* zeros =
        builder_.Broadcast(MakeScalar(0, input->shape().element_type()),
                           HloTensorType(input->shape()), {});
    auto* preds = builder_.Compare(input, zeros, "GE");
    auto* flow = builder_.Select(preds, input, neg);
    return {builder_.Multiply(
        flow,
        builder_.Broadcast(MakeScalar(gamma, input->shape().element_type()),
                           HloTensorType(input->shape()), {}))};
  }

  std::vector<HloFlow> OpIdentity(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    return {GetInput(node, 0)};
  }

  std::vector<HloFlow> OpTranspose(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"perm"});
    auto* input = GetInput(node, 0);
    auto perm = GetAttribute(node, "perm")->ints();
    return {builder_.Transpose(input, perm)};
  }

  std::vector<HloFlow> OpShape(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* input = GetInput(node, 0);

    pblczero::XlaLiteralProto literal;
    HloTensorType shape(pblczero::XlaShapeProto::S64);
    shape.AddDimension(input->shape().dimensions_size());
    *literal.mutable_shape() = shape.ToProto();
    for (auto dim : input->shape().dimensions()) {
      literal.add_s64s(dim == -1 ? batch_size_ : dim);
    }
    return {builder_.Constant(literal)};
  }

  std::vector<HloFlow> OpExpand(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto* input = GetInput(node, 0);
    auto shape = GetConstantInput(node, 1)->s64s();
    return {DoBroadcast(input, shape)};
  }

  std::vector<HloFlow> OpSoftmax(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"axis"});
    const auto axis = GetAttribute(node, "axis")->i();
    auto* input = GetInput(node, 0);

    // Normalize each batch by subtracting the maximum value.
    auto* max = builder_.Reduce(
        input,
        MakeScalar(-std::numeric_limits<float>::infinity(),
                   input->shape().element_type()),
        MakeMaxComputation(input->shape().element_type()), {axis});
    std::vector<int64_t> broadcast_dims;
    for (size_t i = 0; i < input->shape().dimensions_size(); ++i) {
      if (i != static_cast<size_t>(axis)) broadcast_dims.push_back(i);
    }
    max =
        builder_.Broadcast(max, HloTensorType(input->shape()), broadcast_dims);
    input = builder_.Subtract(input, max);

    auto exp = builder_.Exponential(input);
    auto sum = builder_.Reduce(
        exp, MakeScalar(0, input->shape().element_type()),
        MakeAddComputation(input->shape().element_type()), {axis});
    sum =
        builder_.Broadcast(sum, HloTensorType(input->shape()), broadcast_dims);
    return {builder_.Divide(exp, sum)};
  }

  std::vector<HloFlow> OpGather(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {"axis"});
    auto* input = GetInput(node, 0);
    const auto axis = GetAttribute(node, "axis")->i();
    bool is_sorted = false;
    bool is_unique = false;
    HloFlow indices;
    if (auto indices_constant = GetConstantInput(node, 1)) {
      std::tie(is_sorted, is_unique) =
          IsConstantSortedUnique(*indices_constant);
      indices = builder_.Constant(*indices_constant);
    } else {
      indices = GetInput(node, 1);
    }
    return {builder_.Gather(input, indices, axis, {0},
                            {input->shape().dimensions(0), 1}, {1}, {1},
                            is_sorted, is_unique)};
  }

  HloFlow DoReduceMean(HloFlow input, const std::vector<int64_t>& axes) {
    HloFlow zero = MakeScalar(0, input->shape().element_type());
    auto flow = builder_.Reduce(
        input, zero, MakeAddComputation(input->shape().element_type()), axes);
    size_t count = std::accumulate(
        axes.begin(), axes.end(), 1, [&](size_t acc, size_t axis) {
          return acc * input->shape().dimensions(axis);
        });
    auto denominator =
        DoBroadcast(MakeScalar(count, input->shape().element_type()),
                    flow->shape().dimensions());
    flow = builder_.Divide(flow, denominator);
    HloTensorType target_shape(input->shape());
    for (auto axis : axes) target_shape.SetDimension(axis, 1);
    return builder_.Reshape(flow, target_shape);
  }

  std::vector<HloFlow> OpReduceMean(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"axes"});
    auto* input = GetInput(node, 0);
    auto axes = GetAttribute(node, "axes")->ints();
    return {DoReduceMean(input, axes)};
  }

  std::vector<HloFlow> OpCast(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"to"});
    auto* input = GetInput(node, 0);
    const auto onnx_type = static_cast<pblczero::TensorProto::DataType>(
        GetAttribute(node, "to")->i());
    const auto hlo_type = OnnxTypeToXlaType(onnx_type);
    if (input->shape().element_type() == hlo_type) return {input};
    return {builder_.Convert(input, hlo_type)};
  }

  std::vector<HloFlow> OpLayerNormalization(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 3, {"axis", "epsilon"});
    auto* input = GetInput(node, 0);
    const auto axis = GetAttribute(node, "axis")->i();
    const auto epsilon = GetAttribute(node, "epsilon")->f();
    auto* scale = GetInput(node, 1);
    auto* bias = GetInput(node, 2, true);
    const bool need_conv =
        input->shape().element_type() != pblczero::XlaShapeProto::F32;
    auto* flow = need_conv
                     ? builder_.Convert(input, pblczero::XlaShapeProto::F32)
                     : input;
    flow = DoBroadcast(DoReduceMean(flow, {axis}), input->shape().dimensions());
    auto* norm = builder_.Subtract(input, flow);
    flow = builder_.Multiply(norm, norm);
    flow = DoReduceMean(flow, {axis});
    flow = builder_.Add(
        flow, DoBroadcast(MakeScalar(epsilon, pblczero::XlaShapeProto::F32),
                          flow->shape().dimensions()));
    flow = builder_.Rsqrt(flow);
    flow =
        builder_.Multiply(norm, DoBroadcast(flow, norm->shape().dimensions()));
    if (need_conv) {
      flow = builder_.Convert(flow, input->shape().element_type());
    }
    flow =
        builder_.Multiply(flow, DoBroadcast(scale, flow->shape().dimensions()));
    if (bias) {
      flow = builder_.Add(flow, DoBroadcast(bias, flow->shape().dimensions()));
    }
    return {flow};
  }

  std::vector<HloFlow> OpConcat(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, std::numeric_limits<size_t>::max(), {"axis"});
    const auto axis = GetAttribute(node, "axis")->i();
    if (AllInputsConstant(node)) {
      std::vector<pblczero::XlaLiteralProto> constants;
      for (size_t i = 0; i < node.input_size(); ++i) {
        constants.push_back(*GetConstantInput(node, i));
      }
      return {builder_.Constant(ConstOpConcat(constants, axis))};
    }
    std::vector<HloFlow> inputs;
    for (size_t i = 0; i < node.input_size(); ++i) {
      inputs.push_back(GetInput(node, i));
    }
    return {builder_.Concatenate(inputs, axis)};
  }

  std::vector<HloFlow> OpSigmoid(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* input = GetInput(node, 0);
    auto* one = MakeScalar(1, input->shape().element_type());
    one = builder_.Broadcast(one, HloTensorType(input->shape()), {});
    return {builder_.Divide(
        one, builder_.Add(one, builder_.Exponential(builder_.Negate(input))))};
  }

  std::vector<HloFlow> OpTanh(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* input = GetInput(node, 0);
    return {builder_.Tanh(input)};
  }

  std::vector<HloFlow> OpSqrt(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* input = GetInput(node, 0);
    return {builder_.Sqrt(input)};
  }

  std::vector<HloFlow> OpReciprocal(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* input = GetInput(node, 0);
    auto* one = MakeScalar(1, input->shape().element_type());
    return {
        builder_.Divide(DoBroadcast(one, input->shape().dimensions()), input)};
  }

  std::vector<HloFlow> OpSoftplus(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* input = GetInput(node, 0);
    return {builder_.LogPlusOne(builder_.Exponential(input))};
  }

  std::vector<HloFlow> OpAdd(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto* lhs = GetInput(node, 0);
    auto* rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_.Add(lhs, rhs)};
  }

  std::vector<HloFlow> OpSub(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto* lhs = GetInput(node, 0);
    auto* rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_.Subtract(lhs, rhs)};
  }

  std::vector<HloFlow> OpMul(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto* lhs = GetInput(node, 0);
    auto* rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_.Multiply(lhs, rhs)};
  }

  std::vector<HloFlow> OpSplit(const pblczero::NodeProto& node) {
    if (opset_version_ < 13) {
      throw Exception("Split not supported in ONNX opset < 13");
    }
    CheckKnownAttributes(node, 2, {"axis", "num_outputs"});
    auto* input = GetInput(node, 0);
    auto split = GetConstantInput(node, 1, true);
    const auto* axis_attr = GetAttribute(node, "axis");
    const auto* num_outputs_attr = GetAttribute(node, "num_outputs", true);
    if (!axis_attr) throw Exception("Attribute 'axis' not set");

    if (split && num_outputs_attr) {
      throw Exception("Split cannot have both 'split' and 'num_outputs'");
    }

    const size_t axis = axis_attr->i();
    std::vector<size_t> splits;

    if (split) {
      size_t offset = 0;
      for (size_t i = 0; i < split->s64s().size(); ++i) {
        offset += split->s64s(i);
        splits.push_back(offset);
      }
    } else {
      size_t num_outputs =
          num_outputs_attr ? num_outputs_attr->i() : node.output_size();
      size_t chunk_size =
          (input->shape().dimensions(axis) + num_outputs - 1) / num_outputs;
      int64_t offset = 0;
      for (size_t i = 0; i < num_outputs; ++i) {
        offset += chunk_size;
        if (offset > input->shape().dimensions(axis)) {
          offset = input->shape().dimensions(axis);
        }
        splits.push_back(offset);
      }
      if (offset != input->shape().dimensions(axis)) {
        throw Exception("Split sizes do not add up to input size");
      }
    }

    std::vector<HloFlow> flows;

    auto make_slice_dim = [](int64_t start, int64_t end) {
      pblczero::HloInstructionProto::SliceDimensions slice;
      slice.set_start(start);
      slice.set_limit(end);
      slice.set_stride(1);
      return slice;
    };

    size_t offset = 0;
    for (size_t split : splits) {
      std::vector<pblczero::HloInstructionProto::SliceDimensions> slice;
      for (size_t j = 0; j < input->shape().dimensions_size(); ++j) {
        if (j == axis) {
          size_t begin = offset;
          size_t end = split;
          offset = split;
          slice.push_back(make_slice_dim(begin, end));
        } else {
          slice.push_back(make_slice_dim(0, input->shape().dimensions(j)));
        }
      }
      flows.push_back(builder_.Slice(input, slice));
    }
    return flows;
  }

  std::vector<HloFlow> OpSlice(const pblczero::NodeProto& node) {
    if (opset_version_ < 10) {
      throw Exception("Slice not supported in ONNX opset < 10");
    }
    CheckKnownAttributes(node, 3, {});
    auto* input = GetInput(node, 0);
    auto starts = GetConstantInput(node, 1)->s32s();
    auto ends = GetConstantInput(node, 2)->s32s();
    std::vector<pblczero::HloInstructionProto::SliceDimensions> slices;
    for (size_t i = 0; i < starts.size(); ++i) {
      pblczero::HloInstructionProto::SliceDimensions slice;
      slice.set_start(starts[i]);
      slice.set_limit(std::min<int64_t>(ends[i], input->shape().dimensions(i)));
      slice.set_stride(1);
      slices.push_back(slice);
    }
    if (AllInputsConstant(node)) {
      return {builder_.Constant(ConstOpSlice(input->literal(), slices))};
    }
    return {builder_.Slice(input, slices)};
  }

  std::vector<HloFlow> OpReshape(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto* input = GetInput(node, 0);
    auto new_dims = GetConstantInput(node, 1)->s64s();
    HloTensorType input_shape(input->shape());
    HloTensorType new_shape(input_shape.GetElementType());
    std::optional<int64_t> infer_dim;
    size_t num_elements = 1;
    for (size_t i = 0; i < new_dims.size(); ++i) {
      auto dim = new_dims[i];
      if (dim == -1) {
        if (infer_dim.has_value()) {
          throw Exception("Reshape cannot infer shape when multiple -1s");
        }
        infer_dim = i;
        dim = 1;
      }
      if (dim == 0) {
        if (new_dims.size() != input_shape.Rank()) {
          throw Exception("Reshape cannot infer shape when rank changes");
        }
        dim = input_shape.GetDimension(i);
      }
      num_elements *= dim;
      new_shape.AddDimension(dim);
    }
    if (infer_dim.has_value()) {
      new_shape.SetElementType(input->shape().element_type());
      new_shape.SetDimension(infer_dim.value(),
                             input_shape.NumElements() / num_elements);
    }
    return {builder_.Reshape(input, new_shape)};
  }

  std::vector<HloFlow> OpMatMul(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto* lhs = GetInput(node, 0);
    auto* rhs = GetInput(node, 1);
    HloTensorType lhs_shape(lhs->shape());
    HloTensorType rhs_shape(rhs->shape());
    if (lhs_shape.Rank() == 1 || rhs_shape.Rank() == 1) {
      throw Exception("1D MatMul not yet");
    }
    pblczero::XlaDotDimensionNumbers dn;
    dn.add_lhs_contracting_dimensions(lhs_shape.Rank() - 1);
    dn.add_rhs_contracting_dimensions(rhs_shape.Rank() - 2);
    for (size_t i = 0; i < std::min(lhs_shape.Rank(), rhs_shape.Rank()) - 2;
         ++i) {
      dn.add_lhs_batch_dimensions(i);
      dn.add_rhs_batch_dimensions(i);
    }
    return {builder_.Dot(lhs, rhs, dn)};
  }

  std::vector<HloFlow> OpGlobalAveragePool(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto* lhs = GetInput(node, 0);
    if (lhs->shape().dimensions_size() < 2) {
      throw Exception("GlobalAveragePool requires at least 2D input");
    }
    HloFlow zero = MakeScalar(0, lhs->shape().element_type());
    std::vector<int64_t> reduction_dims;
    size_t num_elements = 1;
    for (size_t i = 2; i < lhs->shape().dimensions_size(); ++i) {
      reduction_dims.push_back(i);
      num_elements *= lhs->shape().dimensions(i);
    }
    auto flow = builder_.Reduce(lhs, zero,
                                MakeAddComputation(lhs->shape().element_type()),
                                reduction_dims);
    auto denominator = MakeScalar(num_elements, lhs->shape().element_type());
    std::tie(flow, denominator) = EqualizeShape(flow, denominator);
    flow = builder_.Divide(flow, denominator);
    HloTensorType output_shape(flow->shape());
    while (output_shape.Rank() < lhs->shape().dimensions_size()) {
      output_shape.AddDimension(1);
    }
    return {builder_.Reshape(flow, output_shape)};
  }

  std::vector<HloFlow> OpSqueeze(const pblczero::NodeProto& node) {
    if (opset_version_ < 13) {
      throw Exception("Squeeze not supported in ONNX opset < 13");
    }
    CheckKnownAttributes(node, 2, {});
    auto* input = GetInput(node, 0);
    auto squeeze_dims = GetConstantInput(node, 1, true);

    HloTensorType new_shape(input->shape().element_type());
    if (squeeze_dims) {
      for (size_t i = 0; i < input->shape().dimensions_size(); ++i) {
        bool should_squeeze =
            std::any_of(squeeze_dims->s64s().begin(),
                        squeeze_dims->s64s().end(), [&](int64_t dim) {
                          return dim == static_cast<int64_t>(i) ||
                                 dim + input->shape().dimensions_size() == i;
                        });
        if (!should_squeeze) {
          new_shape.AddDimension(input->shape().dimensions(i));
        }
      }
    } else {
      for (size_t i = 0; i < input->shape().dimensions_size(); ++i) {
        if (input->shape().dimensions(i) != 1) {
          new_shape.AddDimension(input->shape().dimensions(i));
        }
      }
    }
    return {builder_.Reshape(input, new_shape)};
  }

  /////////////////////////////////////////////////////////////////////////////
  // Helper computations
  /////////////////////////////////////////////////////////////////////////////

  HloComputation MakeAddComputation(pblczero::XlaShapeProto::Type type) {
    std::string name = "add_" + pblczero::XlaShapeProto::Type_Name(type);
    if (auto id = builder_.GetComputationId(name)) return *id;
    auto builder = HloBuilder();
    builder.Add(builder.Parameter(MakeScalarShape(type)),
                builder.Parameter(MakeScalarShape(type)));
    return builder_.AddComputation(name, builder);
  }

  HloComputation MakeMaxComputation(pblczero::XlaShapeProto::Type type) {
    std::string name = "max_" + pblczero::XlaShapeProto::Type_Name(type);
    if (auto id = builder_.GetComputationId(name)) return *id;
    auto builder = HloBuilder();
    builder.Maximum(builder.Parameter(MakeScalarShape(type)),
                    builder.Parameter(MakeScalarShape(type)));
    return builder_.AddComputation(name, builder);
  }

  /////////////////////////////////////////////////////////////////////////////

  HloTensorType MakeScalarShape(pblczero::XlaShapeProto::Type type) {
    return HloTensorType(type);
  }

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

  static std::vector<int64_t> BuildCommonDims(
      const std::vector<int64_t>& lhs_dims,
      const std::vector<int64_t>& rhs_dims) {
    const size_t num_dims = std::max(lhs_dims.size(), rhs_dims.size());
    std::vector<int64_t> common_dims;
    common_dims.reserve(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
      int lhs_idx = i + lhs_dims.size() - num_dims;
      int rhs_idx = i + rhs_dims.size() - num_dims;
      const auto lhs_dim = (lhs_idx < 0) ? 1 : lhs_dims[lhs_idx];
      const auto rhs_dim = (rhs_idx < 0) ? 1 : rhs_dims[rhs_idx];
      if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
        throw Exception("Incompatible shapes for broadcast");
      }
      common_dims.push_back(std::max(lhs_dim, rhs_dim));
    }
    return common_dims;
  }

  HloFlow DoBroadcast(HloFlow flow, const std::vector<int64_t>& target_dims) {
    if (flow->shape().dimensions() == target_dims) return flow;

    HloTensorType src_shape(flow->shape());
    HloTensorType target_shape(flow->shape().element_type(), target_dims);
    HloTensorType intermediate_shape(flow->shape().element_type());

    std::vector<int64_t> broadcast_dims;
    bool need_reshape = false;

    for (size_t i = 0; i < src_shape.Rank(); ++i) {
      int target_idx = i + target_shape.Rank() - src_shape.Rank();
      if (src_shape.GetDimension(i) == 1) {
        need_reshape = true;
      } else {
        intermediate_shape.AddDimension(src_shape.GetDimension(i));
        broadcast_dims.push_back(target_idx);
      }
    }
    if (need_reshape) {
      flow = builder_.Reshape(flow, intermediate_shape);
    }
    return builder_.Broadcast(flow, target_shape, broadcast_dims);
  }

  // Take two inputs and optionally performs numpy-style broadcasting to make
  // them equal shape.
  std::pair<HloFlow, HloFlow> EqualizeShape(HloFlow lhs, HloFlow rhs) {
    auto common_dims =
        BuildCommonDims(lhs->shape().dimensions(), rhs->shape().dimensions());
    return {DoBroadcast(lhs, common_dims), DoBroadcast(rhs, common_dims)};
  }

  // Convert ONNX inputs to HLO parameters.
  void BuildInputs(const std::vector<pblczero::ValueInfoProto>& inputs) {
    for (const auto& input : inputs) {
      auto ctx = HloContext(&builder_);
      ctx.SetOpType("input");
      ctx.SetOpName(input.name());
      auto out_shape = OnnxShapeToHloTensorType(input.type(), batch_size_);
      auto in_shape = out_shape;
      in_shape.SetElementType(options_.io_type);
      const auto* flow =
          MakeParameter(std::string(input.name()), in_shape, false);
      flow = builder_.Convert(flow, out_shape.GetElementType());
      onnx_name_to_hlo_flow_[std::string(input.name())] = flow;
    }
  }

  // Makes a parameter instruction (for inputs or large constants).
  HloFlow MakeParameter(const std::string& name, const HloType& shape,
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
      throw Exception("Error in ONNX op=[" + std::string(node.op_type()) +
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
  size_t opset_version_ = 0;
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