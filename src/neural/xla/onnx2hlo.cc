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
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "neural/backends/stablehlo/stablehlo_backend.h"
#include "neural/backends/stablehlo/semantic/semantic_ir.h"
#include "neural/xla/hlo_builder.h"
#include "neural/xla/onnx_builder_interface.h"
#include "neural/xla/onnx_builder_proto_utils.h"
#include "neural/xla/tensor_literal_utils.h"
#include "utils/bf16_utils.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/fp8_utils.h"

namespace lczero {
namespace {

bool CanConvertConstant(const pblczero::XlaShapeProto::Type& type) {
  switch (type) {
    case pblczero::XlaShapeProto::F32:
    case pblczero::XlaShapeProto::F64:
    case pblczero::XlaShapeProto::S32:
    case pblczero::XlaShapeProto::S64:
      return true;
    default:
      return false;
  }
}

template <typename T>
void FetchConstForType(const pblczero::XlaLiteralProto& literal,
                       pblczero::XlaShapeProto::Type type, T&& func) {
  switch (type) {
    case pblczero::XlaShapeProto::F32:
      func(literal.f32s());
      break;
    case pblczero::XlaShapeProto::F64:
      func(literal.f64s());
      break;
    case pblczero::XlaShapeProto::S32:
      func(literal.s32s());
      break;
    case pblczero::XlaShapeProto::S64:
      func(literal.s64s());
      break;
    default:
      throw Exception(
          "Unsupported type for constant input " +
          pblczero::XlaShapeProto::Type_Name(literal.shape().element_type()));
  }
}

template <typename T>
void FetchMutableForType(pblczero::XlaLiteralProto* literal,
                         pblczero::XlaShapeProto::Type type, T&& func) {
  switch (type) {
    case pblczero::XlaShapeProto::F32:
      func(literal->mutable_f32s());
      break;
    case pblczero::XlaShapeProto::F64:
      func(literal->mutable_f64s());
      break;
    case pblczero::XlaShapeProto::S32:
      func(literal->mutable_s32s());
      break;
    case pblczero::XlaShapeProto::S64:
      func(literal->mutable_s64s());
      break;
    default:
      throw Exception(
          "Unsupported type for constant input " +
          pblczero::XlaShapeProto::Type_Name(literal->shape().element_type()));
  }
}

template <typename T>
std::vector<T> LiteralToVector(const pblczero::XlaLiteralProto& literal) {
  std::vector<T> result;
  FetchConstForType(
      literal, literal.shape().element_type(),
      [&](const auto& values) { result.assign(values.begin(), values.end()); });
  return result;
}

template <typename T>
void LiteralOutInOp(pblczero::XlaLiteralProto* dst,
                    const pblczero::XlaLiteralProto& operand, T&& func) {
  const auto type = operand.shape().element_type();
  FetchMutableForType(dst, type, [&](auto* dst) {
    FetchConstForType(operand, type, [&](const auto& src) { func(dst, src); });
  });
}

template <typename T>
void LiteralOutInInOp(pblczero::XlaLiteralProto* dst,
                      const pblczero::XlaLiteralProto& lhs,
                      const pblczero::XlaLiteralProto& rhs, T&& func) {
  const auto out_type = lhs.shape().element_type();
  const auto other_type = rhs.shape().element_type();
  FetchMutableForType(dst, out_type, [&](auto* d) {
    FetchConstForType(lhs, out_type, [&](const auto& l) {
      FetchConstForType(rhs, other_type, [&](const auto& r) { func(d, l, r); });
    });
  });
}

template <typename T>
void LiteralOutInOpDifferentTypes(pblczero::XlaLiteralProto* dst,
                                  const pblczero::XlaLiteralProto& operand,
                                  T&& func) {
  const auto src_type = operand.shape().element_type();
  const auto dst_type = dst->shape().element_type();
  FetchMutableForType(dst, dst_type, [&](auto* dst) {
    FetchConstForType(operand, src_type,
                      [&](const auto& src) { func(dst, src); });
  });
}

pblczero::XlaLiteralProto ConstOpConvert(
    const pblczero::XlaLiteralProto& input,
    const pblczero::XlaShapeProto::Type& to_type) {
  const auto from_type = input.shape().element_type();
  if (from_type == to_type) return input;
  HloTensorType shape(to_type);
  shape.SetDimensions(HloTensorType(input.shape()).GetDimensions());
  pblczero::XlaLiteralProto result;
  *result.mutable_shape() = shape.ToProto();
  LiteralOutInOpDifferentTypes(&result, input, [](auto* dst, const auto& src) {
    std::copy(src.begin(), src.end(), std::back_inserter(*dst));
  });
  return result;
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
    LiteralOutInOp(&result, input, [](auto* dst, const auto& src) {
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
  *result.mutable_shape() = shape.ToProto();
  LiteralOutInOp(&result, input, [&slice](auto* dst, const auto& src) {
    dst->insert(dst->end(), src.begin() + slice[0].start(),
                src.begin() + slice[0].limit());
  });
  return result;
}

pblczero::XlaLiteralProto ConstOpGather(
    const pblczero::XlaLiteralProto& input,
    const pblczero::XlaLiteralProto& indices, int axis) {
  if (input.shape().dimensions_size() != 1 ||
      indices.shape().dimensions_size() != 1) {
    throw Exception(
        "For constant gather, only 1D inputs are supported for now");
  }
  if (axis != 0) {
    throw Exception("For constant gather, only axis 0 is supported for now");
  }
  HloTensorType shape(input.shape().element_type());
  shape.AddDimension(indices.shape().dimensions(axis));
  pblczero::XlaLiteralProto result;
  *result.mutable_shape() = shape.ToProto();
  LiteralOutInInOp(&result, input, indices,
                   [](auto* dst, const auto& inp, const auto& idxs) {
                     for (size_t i : idxs) {
                       if (i > inp.size()) {
                         throw Exception("Constant gather index out of bounds");
                       }
                       dst->push_back(inp[i]);
                     }
                   });
  return result;
}

void EnsureSameShape(const pblczero::XlaLiteralProto& lhs,
                     const pblczero::XlaLiteralProto& rhs,
                     bool also_check_types = true) {
  if (lhs.shape().dimensions() != rhs.shape().dimensions()) {
    throw Exception("Operands must have the same shape");
  }
  if (also_check_types &&
      lhs.shape().element_type() != rhs.shape().element_type()) {
    throw Exception("Operands must have the same type");
  }
}

pblczero::XlaLiteralProto ConstOpMul(const pblczero::XlaLiteralProto& lhs,
                                     const pblczero::XlaLiteralProto& rhs) {
  EnsureSameShape(lhs, rhs);
  pblczero::XlaLiteralProto result;
  *result.mutable_shape() = lhs.shape();
  LiteralOutInInOp(
      &result, lhs, rhs, [](auto* dst, const auto& lhs, const auto& rhs) {
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
                       std::back_inserter(*dst), std::multiplies<>());
      });
  return result;
}

pblczero::XlaLiteralProto ConstOpMax(const pblczero::XlaLiteralProto& lhs,
                                     const pblczero::XlaLiteralProto& rhs) {
  EnsureSameShape(lhs, rhs);
  pblczero::XlaLiteralProto result;
  *result.mutable_shape() = lhs.shape();
  LiteralOutInInOp(
      &result, lhs, rhs, [](auto* dst, const auto& lhs, const auto& rhs) {
        using T =
            typename std::remove_reference<decltype(lhs)>::type::value_type;
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
                       std::back_inserter(*dst),
                       [](const T &a, const T &b) { return std::max(a, b); });
      });
  return result;
}

pblczero::XlaLiteralProto ConstOpReduceProd(
    const pblczero::XlaLiteralProto& input,
    const std::vector<int64_t>& dimensions) {
  if (input.shape().dimensions_size() != 1) {
    throw Exception(
        "For constant reduce_prod, only 1D inputs are supported for now");
  }
  if (dimensions.size() != 1 || dimensions[0] != 0) {
    throw Exception(
        "For constant reduce_prod, only 1D dimensions are supported for now");
  }
  HloTensorType shape(input.shape().element_type());
  pblczero::XlaLiteralProto result;
  HloTensorType result_shape(input.shape().element_type());
  *result.mutable_shape() = result_shape.ToProto();
  LiteralOutInOp(&result, input, [](auto* dst, const auto& inp) {
    using T = typename std::remove_reference<decltype(inp)>::type::value_type;
    dst->push_back(
        std::accumulate(inp.begin(), inp.end(), T(1), std::multiplies<T>()));
  });
  return result;
}

pblczero::XlaLiteralProto ConstReshape(
    const pblczero::XlaLiteralProto& input,
    const pblczero::XlaShapeProto& new_shape) {
  auto result = input;
  *result.mutable_shape() = new_shape;
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
    case pblczero::TensorProto::FLOAT8E5M2:
      return pblczero::XlaShapeProto::F8E5M2;
    case pblczero::TensorProto::FLOAT8E4M3FN:
      return pblczero::XlaShapeProto::F8E4M3FN;
    case pblczero::TensorProto::FLOAT8E4M3FNUZ:
      return pblczero::XlaShapeProto::F8E4M3FNUZ;
    case pblczero::TensorProto::FLOAT8E5M2FNUZ:
      return pblczero::XlaShapeProto::F8E5M2FNUZ;
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
    if (dim.has_dim_param()) {
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
    case pblczero::TensorProto::FLOAT16:
      literal.set_f16s(tensor.raw_data());
      break;
    case pblczero::TensorProto::BFLOAT16:
      literal.set_bf16s(tensor.raw_data());
      break;
    case pblczero::TensorProto::FLOAT8E5M2:
      literal.set_f8e5m2s(tensor.raw_data());
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
    onnx_op_to_builder_["BatchNormalization"] =
        &Onnx2HloConverter::OpBatchNormalization;
    onnx_op_to_builder_["Cast"] = &Onnx2HloConverter::OpCast;
    onnx_op_to_builder_["Concat"] = &Onnx2HloConverter::OpConcat;
    onnx_op_to_builder_["Conv"] = &Onnx2HloConverter::OpConv;
    onnx_op_to_builder_["Div"] = &Onnx2HloConverter::OpDiv;
    onnx_op_to_builder_["Gather"] = &Onnx2HloConverter::OpGather;
    onnx_op_to_builder_["GlobalAveragePool"] =
        &Onnx2HloConverter::OpGlobalAveragePool;
    onnx_op_to_builder_["Greater"] = &Onnx2HloConverter::OpGreater;
    onnx_op_to_builder_["Exp"] = &Onnx2HloConverter::OpExp;
    onnx_op_to_builder_["Expand"] = &Onnx2HloConverter::OpExpand;
    onnx_op_to_builder_["Identity"] = &Onnx2HloConverter::OpIdentity;
    onnx_op_to_builder_["LayerNormalization"] =
        &Onnx2HloConverter::OpLayerNormalization;
    onnx_op_to_builder_["Max"] = &Onnx2HloConverter::OpMax;
    onnx_op_to_builder_["MatMul"] = &Onnx2HloConverter::OpMatMul;
    onnx_op_to_builder_["Mish"] = &Onnx2HloConverter::OpMish;
    onnx_op_to_builder_["Mul"] = &Onnx2HloConverter::OpMul;
    onnx_op_to_builder_["Reciprocal"] = &Onnx2HloConverter::OpReciprocal;
    onnx_op_to_builder_["ReduceMean"] = &Onnx2HloConverter::OpReduceMean;
    onnx_op_to_builder_["ReduceProd"] = &Onnx2HloConverter::OpReduceProd;
    onnx_op_to_builder_["ReduceSumSquare"] =
        &Onnx2HloConverter::OpReduceSumSquare;
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
    onnx_op_to_builder_["Unsqueeze"] = &Onnx2HloConverter::OpUnsqueeze;
    onnx_op_to_builder_["Where"] = &Onnx2HloConverter::OpWhere;

    auto semantic_builder =
        std::make_unique<stablehlo::semantic::SemanticBuilder>();
    semantic_builder_ = semantic_builder.get();
    builder_ = std::move(semantic_builder);
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
    // Pre-create large initializer-backed parameters for used initializers in
    // deterministic name order. This avoids ValueId/parameter interleaving in
    // SemanticBuilder while keeping frozen-parameter indices reproducible.
    PrecreateInitializerParameters(onnx_model.graph());
    FinalizeInputConversions();
    ValidateParameterValueIds();
    if (!IsParameterMapNameSorted()) {
      throw Exception(
          "Initializer parameter ordering violation: non-monotonic name order");
    }
    Onnx2HloResult result;
    std::vector<ValueId> output_value_ids;
    try {
      BuildGraph(onnx_model.graph());
      // Convert ONNX outputs to HLO result.
      result.outputs = options_.outputs_override.empty()
                           ? BuildOutputs(GetOnnxOutputNodes(onnx_model),
                                          &output_value_ids)
                           : BuildOutputs(options_.outputs_override,
                                          &output_value_ids);

      for (size_t i = 0; i < params_.size(); ++i) {
        const auto& param = params_[i];
        auto& dst = param.is_constant ? result.constants : result.inputs;
        dst.push_back(
            {i, param.name, TensorTypeToProto(builder_->GetType(param.value_id))});
      }

    } catch (Exception& e) {
      if (!options_.debugging_allow_partial_result) throw;
      CERR << "Ignoring error in ONNX to HLO conversion: " << e.what();
    }
    semantic_builder_->Return(output_value_ids);
    auto module = semantic_builder_->BuildModule();
    result.mlirbc_bytes = stablehlo::SemanticModuleToMlirbc(module);
    return result;
  }

 private:
  std::vector<std::string> GetOnnxOutputNodes(
      const pblczero::ModelProto& onnx_model) const {
    std::vector<std::string> result;
    for (const auto& output : onnx_model.graph().output()) {
      result.emplace_back(output.name());
    }
    return result;
  }

  std::vector<Onnx2HloResult::NamedTensor> BuildOutputs(
      const std::vector<std::string>& node_names,
      std::vector<ValueId>* output_value_ids) {
    // Gathers outputs into the root tuple, optionally converting their type if
    // I/O type is different from the instruction output.
    std::vector<Onnx2HloResult::NamedTensor> result;
    for (size_t i = 0; i < node_names.size(); ++i) {
      const auto& output = node_names[i];
      auto flow = GetFlowByName(output);
      if (options_.io_type &&
          ElementTypeToProto(builder_->GetType(flow).element_type) != *options_.io_type) {
        BuilderContext ctx(builder_.get());
        builder_->SetOpType("output");
        builder_->SetOpName(output);
        flow = EmitConvert(flow, *options_.io_type);
      }
      result.push_back({i, output, TensorTypeToProto(builder_->GetType(flow))});
      output_value_ids->push_back(flow);
    }
    return result;
  }

  void BuildInitializerMapping(const pblczero::ModelProto& onnx_model) {
    for (const auto& tensor : onnx_model.graph().initializer()) {
      initializers_[std::string(tensor.name())] = &tensor;
    }
  }

  ValueId EmitConstant(const pblczero::XlaLiteralProto& literal) {
    return builder_->Constant(FromLiteralProto(literal));
  }

  ValueId EmitConvert(ValueId input, pblczero::XlaShapeProto::Type type) {
    return builder_->Convert(input, ElementTypeFromProto(type));
  }

  ValueId EmitConvolution(ValueId input, ValueId filter,
                          const pblczero::XlaWindow& window,
                          const pblczero::XlaConvolutionDimensionNumbers& dn) {
    ConvolutionParams params;
    params.input_batch_dim = dn.input_batch_dimension();
    params.input_feature_dim = dn.input_feature_dimension();
    for (const int64_t dim : dn.input_spatial_dimensions()) {
      params.input_spatial_dims.push_back(dim);
    }
    params.kernel_input_feature_dim = dn.kernel_input_feature_dimension();
    params.kernel_output_feature_dim = dn.kernel_output_feature_dimension();
    for (const int64_t dim : dn.kernel_spatial_dimensions()) {
      params.kernel_spatial_dims.push_back(dim);
    }
    params.output_batch_dim = dn.output_batch_dimension();
    params.output_feature_dim = dn.output_feature_dimension();
    for (const int64_t dim : dn.output_spatial_dimensions()) {
      params.output_spatial_dims.push_back(dim);
    }
    for (const auto& dim : window.dimensions()) {
      params.window_strides.push_back(dim.stride());
      params.padding.emplace_back(dim.padding_low(), dim.padding_high());
      params.lhs_dilation.push_back(dim.base_dilation());
      params.rhs_dilation.push_back(dim.window_dilation());
    }
    return builder_->Convolution(input, filter, params);
  }

  ValueId EmitBroadcast(ValueId input, const HloTensorType& target_shape,
                        const std::vector<int64_t>& broadcast_dimensions) {
    return builder_->Broadcast(input, TensorTypeFromProto(target_shape.ToProto()),
                               broadcast_dimensions);
  }

  ValueId EmitReshape(ValueId input, const HloTensorType& new_shape) {
    return builder_->Reshape(input, TensorTypeFromProto(new_shape.ToProto()));
  }

  ValueId EmitGather(ValueId input, ValueId indices, size_t index_vector_dim,
                     const std::vector<int64_t>& offset_dims,
                     const std::vector<int64_t>& slice_sizes,
                     const std::vector<int64_t>& collapsed_slice_dims,
                     const std::vector<int64_t>& start_index_map,
                     bool indices_are_sorted, bool unique_indices) {
    GatherParams params;
    params.index_vector_dim = index_vector_dim;
    params.offset_dims = offset_dims;
    params.slice_sizes = slice_sizes;
    params.collapsed_slice_dims = collapsed_slice_dims;
    params.start_index_map = start_index_map;
    params.indices_are_sorted = indices_are_sorted;
    params.unique_indices = unique_indices;
    return builder_->Gather(input, indices, params);
  }

  ValueId EmitSlice(
      ValueId input,
      const std::vector<pblczero::HloInstructionProto::SliceDimensions>& dims) {
    SliceParams params;
    for (const auto& dim : dims) {
      params.start_indices.push_back(dim.start());
      params.limit_indices.push_back(dim.limit());
      params.strides.push_back(dim.stride());
    }
    return builder_->Slice(input, params);
  }

  ValueId EmitDot(ValueId lhs, ValueId rhs,
                  const pblczero::XlaDotDimensionNumbers& dn) {
    DotParams params;
    for (const int64_t dim : dn.lhs_batch_dimensions()) {
      params.lhs_batch_dims.push_back(dim);
    }
    for (const int64_t dim : dn.rhs_batch_dimensions()) {
      params.rhs_batch_dims.push_back(dim);
    }
    for (const int64_t dim : dn.lhs_contracting_dimensions()) {
      params.lhs_contracting_dims.push_back(dim);
    }
    for (const int64_t dim : dn.rhs_contracting_dimensions()) {
      params.rhs_contracting_dims.push_back(dim);
    }
    return builder_->Dot(lhs, rhs, params);
  }

  ValueId EmitCompare(ValueId lhs, ValueId rhs, std::string_view direction) {
    return builder_->Compare(lhs, rhs, CompareParams{std::string(direction)});
  }

  ValueId EmitReduce(ValueId input, ValueId initial,
                     ReduceParams::Computation computation,
                     const std::vector<int64_t>& reduction_dimensions) {
    return builder_->Reduce(
        input, initial, ReduceParams{computation, reduction_dimensions});
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

  // Fetches an ValueId by name. If the name is not in the map, check whether
  // there is an initializer for it, and either create a constant or a parameter
  // depending on its size.
  ValueId GetFlowByName(const std::string& name) {
    if (auto iter = onnx_name_to_value_id_.find(name);
        iter != onnx_name_to_value_id_.end()) {
      return iter->second;
    }

    auto iter2 = initializers_.find(name);
    if (iter2 == initializers_.end()) {
      throw Exception("Unknown input " + name);
    }
    BuilderContext ctx(builder_.get());
    builder_->SetOpType("initializer");
    builder_->SetOpName(name);

    ValueId flow = kInvalidValueId;
    if (iter2->second->raw_data().size() <= options_.max_inline_constant_size) {
      flow = EmitConstant(OnnxTensorToXlaLiteral(*iter2->second));
    } else {
      const auto shape = TensorTypeFromProto(OnnxTensorToHloTensorType(*iter2->second).ToProto());
      flow = MakeParameter(name, shape, true);
    }
    onnx_name_to_value_id_[name] = flow;
    return flow;
  }

  // A helper function to fetch an input of ONNX node by index.
  ValueId GetInput(const pblczero::NodeProto& node, size_t idx,
                   bool optional = false) {
    if (idx >= node.input_size()) {
      if (optional) return kInvalidValueId;
      throw Exception("Input " + std::to_string(idx) + " not set");
    }
    return GetFlowByName(std::string(node.input(idx)));
  }

  std::optional<pblczero::XlaLiteralProto> GetConstantInput(
      const pblczero::NodeProto& node, size_t idx, bool optional = false) {
    if (idx >= node.input_size()) {
      if (optional) return std::nullopt;
      throw Exception("Constant input " + std::to_string(idx) + " not set");
    }
    const std::string name(node.input(idx));
    if (auto tensor = initializers_.find(name); tensor != initializers_.end()) {
      return OnnxTensorToXlaLiteral(*tensor->second);
    }
    if (auto iter = onnx_name_to_value_id_.find(name);
        iter != onnx_name_to_value_id_.end()) {
      if (const auto literal = builder_->TryGetLiteral(iter->second)) {
        return ToLiteralProto(*literal);
      }
    }
    throw Exception("Constant input " + std::string(node.input(idx)) +
                    " not found");
  }

  template <typename T>
  std::optional<std::vector<T>> GetConstantInputAsVec(
      const pblczero::NodeProto& node, size_t idx, bool optional = false) {
    if (auto literal = GetConstantInput(node, idx, optional)) {
      return LiteralToVector<T>(*literal);
    }
    return std::nullopt;
  }

  bool AllInputsConstant(const pblczero::NodeProto& node) {
    for (const auto& input : node.input()) {
      const std::string name(input);
      if (initializers_.contains(name)) continue;
      if (auto iter = onnx_name_to_value_id_.find(name);
          iter != onnx_name_to_value_id_.end() &&
          builder_->TryGetLiteral(iter->second).has_value()) {
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

  template <typename T>
  std::optional<T> GetOptionalAttributeAs(const pblczero::NodeProto& node,
                                          std::string_view name,
                                          bool optional = true) {
    if (auto* attribute = GetAttribute(node, name, optional)) {
      switch (attribute->type()) {
        case pblczero::AttributeProto::FLOAT:
          return attribute->f();
        case pblczero::AttributeProto::INT:
          return attribute->i();
        default:
          throw Exception(
              "Unsupported attribute type " +
              pblczero::AttributeProto::AttributeType_Name(attribute->type()));
      }
    }
    return std::nullopt;
  }

  template <typename T>
  T GetAttributeAs(const pblczero::NodeProto& node, std::string_view name) {
    return GetOptionalAttributeAs<T>(node, name, false).value();
  }

  template <typename T>
  std::optional<std::vector<T>> GetOptionalAttributeAsVec(
      const pblczero::NodeProto& node, std::string_view name,
      bool optional = true) {
    if (auto* attribute = GetAttribute(node, name, optional)) {
      std::vector<T> result;
      switch (attribute->type()) {
        case pblczero::AttributeProto::FLOATS:
          result.assign(attribute->floats().begin(), attribute->floats().end());
          break;
        case pblczero::AttributeProto::INTS:
          result.assign(attribute->ints().begin(), attribute->ints().end());
          break;
        default:
          throw Exception(
              "Unsupported attribute type " +
              pblczero::AttributeProto::AttributeType_Name(attribute->type()));
      }
      return result;
    }
    return std::nullopt;
  }

  template <typename T>
  std::vector<T> GetAttributeAsVec(const pblczero::NodeProto& node,
                                   std::string_view name) {
    return GetOptionalAttributeAsVec<T>(node, name, false).value();
  }

  std::vector<int64_t> GetIota(size_t size) {
    std::vector<int64_t> result(size);
    std::iota(result.begin(), result.end(), 0);
    return result;
  };

  uint64_t GetNumberElements(const std::vector<int64_t>& dimensions) {
    return std::accumulate(dimensions.begin(), dimensions.end(), 1,
                           std::multiplies<int64_t>());
  }

  uint64_t GetShapeSize(const pblczero::XlaShapeProto& shape) {
    return GetNumberElements(shape.dimensions()) *
           GetXlaTypeSize(shape.element_type());
  }

  int64_t NormalizeDimension(int64_t dim, int rank) {
    if (dim >= rank || dim < -rank) {
      throw Exception("Invalid dimension " + std::to_string(dim) +
                      " for rank " + std::to_string(rank));
    }
    if (dim < 0) dim += rank;
    return dim;
  }

  void NormalizeDimensions(std::vector<int64_t>* dimensions, int rank) {
    for (auto& dim : *dimensions) {
      dim = NormalizeDimension(dim, rank);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // ONNX operations
  /////////////////////////////////////////////////////////////////////////////

  std::vector<ValueId> OpConv(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 3, {"pads", "kernel_shape"});
    auto input = GetInput(node, 0);
    auto kernel = GetInput(node, 1);
    auto bias = GetInput(node, 2, true);

    pblczero::XlaConvolutionDimensionNumbers dn;
    dn.set_input_batch_dimension(0);
    dn.set_input_feature_dimension(1);
    dn.set_kernel_input_feature_dimension(1);
    dn.set_kernel_output_feature_dimension(0);
    dn.set_output_batch_dimension(0);
    dn.set_output_feature_dimension(1);
    const size_t num_dims = GetShape(input).dimensions_size() - 2;
    for (size_t i = 0; i < num_dims; ++i) {
      dn.add_input_spatial_dimensions(i + 2);
      dn.add_kernel_spatial_dimensions(i + 2);
      dn.add_output_spatial_dimensions(i + 2);
    }

    const auto pads = GetAttributeAsVec<int32_t>(node, "pads");
    const auto kernel_shape = GetAttributeAsVec<int32_t>(node, "kernel_shape");
    if (pads.size() != 2 * num_dims) {
      throw Exception("'pads' attribute must have 2 * num_dims elements");
    }
    if (kernel_shape.size() != num_dims) {
      throw Exception("'kernel_shape' attribute must have num_dims elements");
    }
    pblczero::XlaWindow window;
    for (size_t i = 0; i < GetShape(input).dimensions_size() - 2; ++i) {
      auto* dim = window.add_dimensions();
      dim->set_size(kernel_shape[i]);
      dim->set_stride(1);
      dim->set_padding_low(pads[i]);
      dim->set_padding_high(pads[i + num_dims]);
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
    }

    auto conv = EmitConvolution(input, kernel, window, dn);

    if (bias == kInvalidValueId) return {conv};
    auto flow = EmitBroadcast(bias, HloTensorType(GetShape(conv)), {1});
    return {builder_->Add(conv, flow)};
  }

  std::vector<ValueId> OpRelu(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    auto zero = MakeScalar(0, GetShape(input).element_type());
    zero = EmitBroadcast(zero, HloTensorType(GetShape(input)), {});
    return {builder_->Maximum(input, zero)};
  }

  std::vector<ValueId> OpSelu(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"alpha", "gamma"});
    auto input = GetInput(node, 0);
    double alpha = GetOptionalAttributeAs<double>(node, "alpha")
                       .value_or(1.6732632423543772848170429916717);
    double gamma = GetOptionalAttributeAs<double>(node, "gamma")
                       .value_or(1.0507009873554804934193349852946);
    auto neg = builder_->Multiply(
        EmitBroadcast(MakeScalar(alpha, GetShape(input).element_type()),
                           HloTensorType(GetShape(input)), {}),
        builder_->ExponentialMinusOne(input));
    auto zeros = EmitBroadcast(MakeScalar(0, GetShape(input).element_type()),
                           HloTensorType(GetShape(input)), {});
    auto preds = EmitCompare(input, zeros, "GE");
    auto flow = builder_->Select(preds, input, neg);
    return {builder_->Multiply(
        flow,
        EmitBroadcast(MakeScalar(gamma, GetShape(input).element_type()),
                           HloTensorType(GetShape(input)), {}))};
  }

  std::vector<ValueId> OpIdentity(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    return {GetInput(node, 0)};
  }

  std::vector<ValueId> OpTranspose(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"perm"});
    auto input = GetInput(node, 0);
    auto perm = GetAttributeAsVec<int64_t>(node, "perm");
    return {builder_->Transpose(input, perm)};
  }

  std::vector<ValueId> OpShape(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    const auto input_shape = GetShape(input);

    pblczero::XlaLiteralProto literal;
    HloTensorType shape(pblczero::XlaShapeProto::S64);
    shape.AddDimension(input_shape.dimensions_size());
    *literal.mutable_shape() = shape.ToProto();
    for (auto dim : input_shape.dimensions()) {
      literal.add_s64s(dim == -1 ? batch_size_ : dim);
    }
    return {EmitConstant(literal)};
  }

  std::vector<ValueId> OpExpand(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto input = GetInput(node, 0);
    const auto shape = *GetConstantInputAsVec<int64_t>(node, 1);
    return {DoBroadcast(input, shape)};
  }

  std::vector<ValueId> OpSoftmax(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"axis"});
    auto axis = GetOptionalAttributeAs<int>(node, "axis").value_or(-1);
    auto input = GetInput(node, 0);
    axis = NormalizeDimension(axis, GetShape(input).dimensions_size());

    // Normalize each batch by subtracting the maximum value.
    auto max = EmitReduce(
        input,
        MakeScalar(-std::numeric_limits<float>::infinity(),
                   GetShape(input).element_type()),
        ReduceParams::Computation::kMaximum, {axis});
    std::vector<int64_t> broadcast_dims;
    for (size_t i = 0; i < GetShape(input).dimensions_size(); ++i) {
      if (i != static_cast<size_t>(axis)) broadcast_dims.push_back(i);
    }
    max =
        EmitBroadcast(max, HloTensorType(GetShape(input)), broadcast_dims);
    input = builder_->Subtract(input, max);

    auto exp = builder_->Exponential(input);
    auto sum = EmitReduce(exp, MakeScalar(0, GetShape(input).element_type()),
                          ReduceParams::Computation::kAdd, {axis});
    sum =
        EmitBroadcast(sum, HloTensorType(GetShape(input)), broadcast_dims);
    return {builder_->Divide(exp, sum)};
  }

  std::vector<ValueId> OpGather(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {"axis"});
    auto axis = GetOptionalAttributeAs<int>(node, "axis").value_or(0);
    if (AllInputsConstant(node)) {
      return {EmitConstant(ConstOpGather(
          *GetConstantInput(node, 0), *GetConstantInput(node, 1), axis))};
    }
    auto input = GetInput(node, 0);
    axis = NormalizeDimension(axis, GetShape(input).dimensions_size());
    bool is_sorted = false;
    bool is_unique = false;
    ValueId indices;
    if (auto indices_constant = GetConstantInput(node, 1)) {
      std::tie(is_sorted, is_unique) =
          IsConstantSortedUnique(*indices_constant);
      indices = EmitConstant(*indices_constant);
    } else {
      indices = GetInput(node, 1);
    }
    return {EmitGather(input, indices, axis, {0},
                            {GetShape(input).dimensions(0), 1}, {1}, {1},
                            is_sorted, is_unique)};
  }

  ValueId DoReduceMean(ValueId input, const std::vector<int64_t>& axes,
                       bool keepdims = true) {
    ValueId zero = MakeScalar(0, GetShape(input).element_type());
    auto flow =
        EmitReduce(input, zero, ReduceParams::Computation::kAdd, axes);
    size_t count = std::accumulate(
        axes.begin(), axes.end(), 1, [&](size_t acc, size_t axis) {
          return acc * GetShape(input).dimensions(axis);
        });
    auto denominator =
        DoBroadcast(MakeScalar(count, GetShape(input).element_type()),
                    GetShape(flow).dimensions());
    flow = builder_->Divide(flow, denominator);
    if (!keepdims) return flow;
    HloTensorType target_shape(GetShape(input));
    for (auto axis : axes) target_shape.SetDimension(axis, 1);
    return EmitReshape(flow, target_shape);
  }

  std::vector<ValueId> OpReduceMean(const pblczero::NodeProto& node) {
    if (opset_version_ < 18) {
      CheckKnownAttributes(node, 1, {"axes", "keepdims"});
    } else {
      CheckKnownAttributes(node, 2, {"keepdims", "noop_with_empty_axes"});
    }
    auto input = GetInput(node, 0);
    auto axes = opset_version_ < 18
                    ? GetOptionalAttributeAsVec<int64_t>(node, "axes")
                    : GetConstantInputAsVec<int64_t>(node, 1, true);
    if (!axes) {
      if (GetOptionalAttributeAs<bool>(node, "noop_with_empty_axes")
              .value_or(false)) {
        return {input};
      }
      axes = GetIota(GetShape(input).dimensions_size());
    }
    NormalizeDimensions(&*axes, GetShape(input).dimensions_size());
    bool keepdims =
        GetOptionalAttributeAs<bool>(node, "keepdims").value_or(true);
    return {DoReduceMean(input, *axes, keepdims)};
  }

  std::vector<ValueId> OpReduceProd(const pblczero::NodeProto& node) {
    if (opset_version_ < 18) {
      CheckKnownAttributes(node, 1, {"axes", "keepdims"});
    } else {
      CheckKnownAttributes(node, 2, {"keepdims", "noop_with_empty_axes"});
    }
    auto input = GetInput(node, 0);
    auto axes = opset_version_ < 18
                    ? GetOptionalAttributeAsVec<int64_t>(node, "axes")
                    : GetConstantInputAsVec<int64_t>(node, 1, true);
    if (!axes) {
      if (GetOptionalAttributeAs<bool>(node, "noop_with_empty_axes")
              .value_or(false)) {
        return {input};
      }
      axes = GetIota(GetShape(input).dimensions_size());
    }
    NormalizeDimensions(&*axes, GetShape(input).dimensions_size());
    bool keepdims =
        GetOptionalAttributeAs<bool>(node, "keepdims").value_or(true);
    ValueId flow;
    if (AllInputsConstant(node)) {
      auto literal = ConstOpReduceProd(*GetConstantInput(node, 0), *axes);
      if (!keepdims) return {EmitConstant(literal)};
      HloTensorType target_shape(GetShape(input));
      for (auto axis : *axes) target_shape.SetDimension(axis, 1);
      return {EmitConstant(ConstReshape(literal, target_shape.ToProto()))};
    }
    ValueId one = MakeScalar(1, GetShape(input).element_type());
    flow = EmitReduce(input, one, ReduceParams::Computation::kMultiply, *axes);
    if (!keepdims) return {flow};
    HloTensorType target_shape(GetShape(input));
    for (auto axis : *axes) target_shape.SetDimension(axis, 1);
    return {EmitReshape(flow, target_shape)};
  }

  std::vector<ValueId> OpReduceSumSquare(const pblczero::NodeProto& node) {
    if (opset_version_ < 18) {
      CheckKnownAttributes(node, 1, {"axes", "keepdims"});
    } else {
      CheckKnownAttributes(node, 2, {"keepdims", "noop_with_empty_axes"});
    }
    auto input = GetInput(node, 0);
    auto axes = opset_version_ < 18
                    ? GetOptionalAttributeAsVec<int64_t>(node, "axes")
                    : GetConstantInputAsVec<int64_t>(node, 1, true);
    if (!axes) {
      if (GetOptionalAttributeAs<bool>(node, "noop_with_empty_axes")
              .value_or(false)) {
        return {input};
      }
      axes = GetIota(GetShape(input).dimensions_size());
    }
    NormalizeDimensions(&*axes, GetShape(input).dimensions_size());
    bool keepdims =
        GetOptionalAttributeAs<bool>(node, "keepdims").value_or(true);
    auto flow = builder_->Multiply(input, input);
    flow = EmitReduce(flow, MakeScalar(0, GetShape(input).element_type()),
                      ReduceParams::Computation::kAdd, *axes);
    if (!keepdims) return {flow};
    HloTensorType target_shape(GetShape(input));
    for (auto axis : *axes) target_shape.SetDimension(axis, 1);
    return {EmitReshape(flow, target_shape)};
  }

  std::vector<ValueId> OpCast(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {"to"});
    auto input = GetInput(node, 0);
    const auto onnx_type = static_cast<pblczero::TensorProto::DataType>(
        GetAttributeAs<int>(node, "to"));
    const auto hlo_type = OnnxTypeToXlaType(onnx_type);
    if (GetShape(input).element_type() == hlo_type) return {input};
    // Only convert constants of int64 to int32 as that's what TF does.
    if (AllInputsConstant(node) && CanConvertConstant(hlo_type) &&
        CanConvertConstant(GetShape(input).element_type())) {
      return {EmitConstant(
          ConstOpConvert(*GetConstantInput(node, 0), hlo_type))};
    }
    return {EmitConvert(input, hlo_type)};
  }

  std::vector<ValueId> OpBatchNormalization(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 5, {"epsilon", "momentum", "training_mode"});
    if (GetOptionalAttributeAs<bool>(node, "training_mode").value_or(false)) {
      throw Exception("Training mode not supported");
    }
    auto input = GetInput(node, 0);
    auto scale = GetInput(node, 1);
    auto bias = GetInput(node, 2);
    auto mean = GetInput(node, 3);
    auto variance = GetInput(node, 4);
    const auto epsilon =
        GetOptionalAttributeAs<float>(node, "epsilon").value_or(1e-5);
    std::vector<int64_t> broadcast_dims = {1};
    HloTensorType shape(GetShape(input));
    auto flow = builder_->Subtract(
        input, EmitBroadcast(mean, shape, broadcast_dims));
    flow = builder_->Divide(
        flow, EmitBroadcast(
                  builder_->Sqrt(builder_->Add(
                      variance,
                      EmitBroadcast(
                          MakeScalar(epsilon, GetShape(input).element_type()),
                          HloTensorType(GetShape(variance)), {}))),
                  shape, broadcast_dims));
    flow = builder_->Multiply(flow,
                             EmitBroadcast(scale, shape, broadcast_dims));
    flow = builder_->Add(flow, EmitBroadcast(bias, shape, broadcast_dims));
    return {flow};
  }

  std::vector<ValueId> OpLayerNormalization(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 3, {"axis", "epsilon"});
    auto input = GetInput(node, 0);
    auto axis = GetAttributeAs<int>(node, "axis");
    axis = NormalizeDimension(axis, GetShape(input).dimensions_size());
    const auto epsilon = GetAttributeAs<float>(node, "epsilon");
    auto scale = GetInput(node, 1);
    auto bias = GetInput(node, 2, true);
    constexpr auto kAccType = pblczero::XlaShapeProto::F32;
    const auto input_type = GetShape(input).element_type();
    const bool need_conv = input_type != kAccType;
    input = need_conv ? EmitConvert(input, kAccType) : input;
    auto flow =
        DoBroadcast(DoReduceMean(input, {axis}), GetShape(input).dimensions());
    auto norm = builder_->Subtract(input, flow);
    flow = builder_->Multiply(norm, norm);
    flow = DoReduceMean(flow, {axis});
    flow = builder_->Add(flow, DoBroadcast(MakeScalar(epsilon, kAccType),
                                          GetShape(flow).dimensions()));
    flow = builder_->Rsqrt(flow);
    flow =
        builder_->Multiply(norm, DoBroadcast(flow, GetShape(norm).dimensions()));
    if (need_conv) flow = EmitConvert(flow, input_type);

    flow =
        builder_->Multiply(flow, DoBroadcast(scale, GetShape(flow).dimensions()));
    if (bias != kInvalidValueId) {
      flow = builder_->Add(flow, DoBroadcast(bias, GetShape(flow).dimensions()));
    }
    return {flow};
  }

  std::vector<ValueId> OpConcat(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, std::numeric_limits<size_t>::max(), {"axis"});
    auto axis = GetAttributeAs<int>(node, "axis");
    if (AllInputsConstant(node)) {
      std::vector<pblczero::XlaLiteralProto> constants;
      for (size_t i = 0; i < node.input_size(); ++i) {
        constants.push_back(*GetConstantInput(node, i));
      }
      return {EmitConstant(ConstOpConcat(constants, axis))};
    }
    std::vector<ValueId> inputs;
    for (size_t i = 0; i < node.input_size(); ++i) {
      inputs.push_back(GetInput(node, i));
    }
    axis = NormalizeDimension(axis, GetShape(inputs[0]).dimensions_size());
    return {builder_->Concatenate(inputs, axis)};
  }

  std::vector<ValueId> OpSigmoid(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    auto one = MakeScalar(1, GetShape(input).element_type());
    one = EmitBroadcast(one, HloTensorType(GetShape(input)), {});
    return {builder_->Divide(
        one, builder_->Add(one, builder_->Exponential(builder_->Negate(input))))};
  }

  std::vector<ValueId> OpTanh(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    return {builder_->Tanh(input)};
  }

  std::vector<ValueId> OpSqrt(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    return {builder_->Sqrt(input)};
  }

  std::vector<ValueId> OpReciprocal(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    auto one = MakeScalar(1, GetShape(input).element_type());
    return {
        builder_->Divide(DoBroadcast(one, GetShape(input).dimensions()), input)};
  }

  std::vector<ValueId> OpSoftplus(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    return {builder_->LogPlusOne(builder_->Exponential(input))};
  }

  std::vector<ValueId> OpAdd(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto lhs = GetInput(node, 0);
    auto rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_->Add(lhs, rhs)};
  }

  std::vector<ValueId> OpDiv(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto lhs = GetInput(node, 0);
    auto rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_->Divide(lhs, rhs)};
  }

  std::vector<ValueId> OpSub(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto lhs = GetInput(node, 0);
    auto rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_->Subtract(lhs, rhs)};
  }

  std::vector<ValueId> OpMul(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto lhs = GetInput(node, 0);
    auto rhs = GetInput(node, 1);

    if (AllInputsConstant(node) &&
        GetShapeSize(GetShape(lhs)) <= options_.max_inline_constant_size) {
      return {EmitConstant(
          ConstOpMul(*GetConstantInput(node, 0), *GetConstantInput(node, 1)))};
    }

    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_->Multiply(lhs, rhs)};
  }

  std::vector<ValueId> OpMax(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto lhs = GetInput(node, 0);
    auto rhs = GetInput(node, 1);

    if (AllInputsConstant(node) &&
        GetShapeSize(GetShape(lhs)) <= options_.max_inline_constant_size) {
      return {EmitConstant(
          ConstOpMax(*GetConstantInput(node, 0), *GetConstantInput(node, 1)))};
    }
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {builder_->Maximum(lhs, rhs)};
  }

  std::vector<ValueId> OpSplit(const pblczero::NodeProto& node) {
    if (opset_version_ < 13) {
      throw Exception("Split not supported in ONNX opset < 13");
    }
    CheckKnownAttributes(node, 2, {"axis", "num_outputs"});
    auto input = GetInput(node, 0);
    auto split = GetConstantInputAsVec<int64_t>(node, 1, true);
    size_t axis = GetAttributeAs<size_t>(node, "axis");
    axis = NormalizeDimension(axis, GetShape(input).dimensions_size());
    const auto num_outputs_attr =
        GetOptionalAttributeAs<size_t>(node, "num_outputs");

    if (split && num_outputs_attr) {
      throw Exception("Split cannot have both 'split' and 'num_outputs'");
    }

    std::vector<size_t> splits;

    if (split) {
      size_t offset = 0;
      for (size_t i = 0; i < split->size(); ++i) {
        offset += (*split)[i];
        splits.push_back(offset);
      }
    } else {
      size_t num_outputs = num_outputs_attr.value_or(node.output_size());
      size_t chunk_size =
          (GetShape(input).dimensions(axis) + num_outputs - 1) / num_outputs;
      int64_t offset = 0;
      for (size_t i = 0; i < num_outputs; ++i) {
        offset += chunk_size;
        if (offset > GetShape(input).dimensions(axis)) {
          offset = GetShape(input).dimensions(axis);
        }
        splits.push_back(offset);
      }
      if (offset != GetShape(input).dimensions(axis)) {
        throw Exception("Split sizes do not add up to input size");
      }
    }

    std::vector<ValueId> flows;

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
      for (size_t j = 0; j < GetShape(input).dimensions_size(); ++j) {
        if (j == axis) {
          size_t begin = offset;
          size_t end = split;
          offset = split;
          slice.push_back(make_slice_dim(begin, end));
        } else {
          slice.push_back(make_slice_dim(0, GetShape(input).dimensions(j)));
        }
      }
      flows.push_back(EmitSlice(input, slice));
    }
    return flows;
  }

  std::vector<ValueId> OpSlice(const pblczero::NodeProto& node) {
    if (opset_version_ < 10) {
      throw Exception("Slice not supported in ONNX opset < 10");
    }
    CheckKnownAttributes(node, 4, {});
    auto input = GetInput(node, 0);
    HloTensorType input_shape(GetShape(input));
    auto starts = *GetConstantInputAsVec<int64_t>(node, 1);
    auto ends = *GetConstantInputAsVec<int64_t>(node, 2);
    auto axes_attr = GetConstantInputAsVec<int64_t>(node, 3, true);
    if (starts.size() != ends.size()) {
      throw Exception("Slice starts and ends must have the same size");
    }
    if (axes_attr && axes_attr->size() != starts.size()) {
      throw Exception("Slice axes must have the same size as starts and ends");
    }
    std::vector<int64_t> axes =
        axes_attr.value_or(std::vector<int64_t>(starts.size()));
    if (!axes_attr) std::iota(axes.begin(), axes.end(), 0);
    NormalizeDimensions(&axes, input_shape.Rank());

    std::vector<pblczero::HloInstructionProto::SliceDimensions> slices;
    for (const auto& dim : input_shape.GetDimensions()) {
      pblczero::HloInstructionProto::SliceDimensions slice;
      slice.set_start(0);
      slice.set_limit(dim);
      slice.set_stride(1);
      slices.push_back(slice);
    }

    for (size_t i = 0; i < axes.size(); ++i) {
      pblczero::HloInstructionProto::SliceDimensions slice;
      const auto axis = axes[i];
      const int dim = input_shape.GetDimension(axis);
      int start = starts[i] < 0 ? starts[i] + dim : starts[i];
      start = std::clamp(start, 0, dim);
      int end = ends[i] < 0 ? ends[i] + dim : ends[i];
      end = std::clamp(end, 0, dim);
      slice.set_start(std::min<int64_t>(start, dim));
      slice.set_limit(std::min<int64_t>(end, dim));
      slice.set_stride(1);
      slices[axis] = slice;
    }

    if (AllInputsConstant(node)) {
      return {EmitConstant(ConstOpSlice(*GetConstantInput(node, 0), slices))};
    }
    return {EmitSlice(input, slices)};
  }

  std::vector<ValueId> OpReshape(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto input = GetInput(node, 0);
    auto new_dims = *GetConstantInputAsVec<int64_t>(node, 1);
    HloTensorType input_shape(GetShape(input));
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
      new_shape.SetElementType(GetShape(input).element_type());
      new_shape.SetDimension(infer_dim.value(),
                             input_shape.NumElements() / num_elements);
    }
    if (AllInputsConstant(node) &&
        new_shape.NumElements() <= options_.max_inline_constant_size) {
      return {EmitConstant(
          ConstReshape(*GetConstantInput(node, 0), new_shape.ToProto()))};
    }
    return {EmitReshape(input, new_shape)};
  }

  std::vector<ValueId> OpMatMul(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto lhs = GetInput(node, 0);
    auto rhs = GetInput(node, 1);
    HloTensorType lhs_shape(GetShape(lhs));
    HloTensorType rhs_shape(GetShape(rhs));
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
    return {EmitDot(lhs, rhs, dn)};
  }

  std::vector<ValueId> OpGlobalAveragePool(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto lhs = GetInput(node, 0);
    if (GetShape(lhs).dimensions_size() < 2) {
      throw Exception("GlobalAveragePool requires at least 2D input");
    }
    ValueId zero = MakeScalar(0, GetShape(lhs).element_type());
    std::vector<int64_t> reduction_dims;
    size_t num_elements = 1;
    for (size_t i = 2; i < GetShape(lhs).dimensions_size(); ++i) {
      reduction_dims.push_back(i);
      num_elements *= GetShape(lhs).dimensions(i);
    }
    auto flow = EmitReduce(lhs, zero, ReduceParams::Computation::kAdd,
                           reduction_dims);
    auto denominator = MakeScalar(num_elements, GetShape(lhs).element_type());
    std::tie(flow, denominator) = EqualizeShape(flow, denominator);
    flow = builder_->Divide(flow, denominator);
    HloTensorType output_shape(GetShape(flow));
    while (output_shape.Rank() < GetShape(lhs).dimensions_size()) {
      output_shape.AddDimension(1);
    }
    return {EmitReshape(flow, output_shape)};
  }

  std::vector<ValueId> OpSqueeze(const pblczero::NodeProto& node) {
    if (opset_version_ < 13) {
      throw Exception("Squeeze not supported in ONNX opset < 13");
    }
    CheckKnownAttributes(node, 2, {});
    auto input = GetInput(node, 0);
    auto squeeze_dims = GetConstantInputAsVec<int64_t>(node, 1);

    HloTensorType new_shape(GetShape(input).element_type());
    if (squeeze_dims) {
      for (size_t i = 0; i < GetShape(input).dimensions_size(); ++i) {
        bool should_squeeze = std::any_of(
            squeeze_dims->begin(), squeeze_dims->end(), [&](int64_t dim) {
              return dim == static_cast<int64_t>(i) ||
                     dim + GetShape(input).dimensions_size() == i;
            });
        if (!should_squeeze) {
          new_shape.AddDimension(GetShape(input).dimensions(i));
        }
      }
    } else {
      for (size_t i = 0; i < GetShape(input).dimensions_size(); ++i) {
        if (GetShape(input).dimensions(i) != 1) {
          new_shape.AddDimension(GetShape(input).dimensions(i));
        }
      }
    }
    if (AllInputsConstant(node)) {
      return {EmitConstant(
          ConstReshape(*GetConstantInput(node, 0), new_shape.ToProto()))};
    }
    return {EmitReshape(input, new_shape)};
  }

  std::vector<ValueId> OpUnsqueeze(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto input = GetInput(node, 0);
    auto axes = *GetConstantInputAsVec<int64_t>(node, 1);
    HloTensorType input_shape(GetShape(input));
    HloTensorType new_shape(GetShape(input).element_type());
    const size_t new_num_dims = input_shape.Rank() + axes.size();
    NormalizeDimensions(&axes, new_num_dims);
    size_t src_dim = 0;
    for (size_t i = 0; i < new_num_dims; ++i) {
      if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
        new_shape.AddDimension(1);
      } else {
        new_shape.AddDimension(input_shape.GetDimension(src_dim++));
      }
    }
    if (AllInputsConstant(node)) {
      return {EmitConstant(
          ConstReshape(*GetConstantInput(node, 0), new_shape.ToProto()))};
    }
    return {EmitReshape(input, new_shape)};
  }

  std::vector<ValueId> OpMish(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    auto flow = builder_->Exponential(input);
    flow = builder_->LogPlusOne(flow);
    flow = builder_->Tanh(flow);
    return {builder_->Multiply(flow, input)};
  }

  std::vector<ValueId> OpExp(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 1, {});
    auto input = GetInput(node, 0);
    return {builder_->Exponential(input)};
  }

  std::vector<ValueId> OpGreater(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 2, {});
    auto lhs = GetInput(node, 0);
    auto rhs = GetInput(node, 1);
    std::tie(lhs, rhs) = EqualizeShape(lhs, rhs);
    return {EmitCompare(lhs, rhs, "GT")};
  }

  std::vector<ValueId> OpWhere(const pblczero::NodeProto& node) {
    CheckKnownAttributes(node, 3, {});
    auto pred = GetInput(node, 0);
    auto on_true = GetInput(node, 1);
    auto on_false = GetInput(node, 2);
    std::tie(on_true, on_false) = EqualizeShape(on_true, on_false);
    std::tie(pred, on_true) = EqualizeShape(pred, on_true);
    std::tie(pred, on_false) = EqualizeShape(pred, on_false);
    return {builder_->Select(pred, on_true, on_false)};
  }

  // Makes a scalar constant (usually 0 or 1) of a given type.
  template <typename T>
  ValueId MakeScalar(T value, pblczero::XlaShapeProto::Type type) {
    pblczero::XlaLiteralProto literal;
    literal.mutable_shape()->set_element_type(type);
    literal.mutable_shape()->mutable_layout();
    switch (type) {
      case pblczero::XlaShapeProto::F32:
        literal.add_f32s(value);
        break;
      case pblczero::XlaShapeProto::F16: {
        uint16_t f16 = FP32toFP16(value);
        std::string_view f16_view(reinterpret_cast<const char*>(&f16),
                                  sizeof(f16));
        literal.set_f16s(f16_view);
      } break;
      case pblczero::XlaShapeProto::BF16: {
        uint16_t bf16 = FP32toBF16(value);
        std::string_view bf16_view(reinterpret_cast<const char*>(&bf16),
                                   sizeof(bf16));
        literal.set_bf16s(bf16_view);
      } break;
      case pblczero::XlaShapeProto::F8E5M2: {
        uint8_t f8e5m2 = FP32toFP8E5M2(value);
        std::string_view f8e5m2_view(reinterpret_cast<const char*>(&f8e5m2),
                                     sizeof(f8e5m2));
        literal.set_f8e5m2s(f8e5m2_view);
      } break;
      case pblczero::XlaShapeProto::S32:
        literal.add_s32s(value);
        break;
      case pblczero::XlaShapeProto::S64:
        literal.add_s64s(value);
        break;
      default:
        throw Exception("Unsupported type for a constant: " +
                        pblczero::XlaShapeProto::Type_Name(type));
    }
    return EmitConstant(literal);
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

  ValueId DoBroadcast(ValueId flow, const std::vector<int64_t>& target_dims) {
    if (GetShape(flow).dimensions() == target_dims) return flow;

    HloTensorType src_shape(GetShape(flow));
    HloTensorType target_shape(GetShape(flow).element_type(), target_dims);
    HloTensorType intermediate_shape(GetShape(flow).element_type());

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
      flow = EmitReshape(flow, intermediate_shape);
    }
    return EmitBroadcast(flow, target_shape, broadcast_dims);
  }

  // Take two inputs and optionally performs numpy-style broadcasting to make
  // them equal shape.
  std::pair<ValueId, ValueId> EqualizeShape(ValueId lhs, ValueId rhs) {
    auto common_dims =
        BuildCommonDims(GetShape(lhs).dimensions(), GetShape(rhs).dimensions());
    return {DoBroadcast(lhs, common_dims), DoBroadcast(rhs, common_dims)};
  }

  // Convert ONNX inputs to HLO parameters.
  void BuildInputs(const std::vector<pblczero::ValueInfoProto>& inputs) {
    for (const auto& input : inputs) {
      BuilderContext ctx(builder_.get());
      builder_->SetOpType("input");
      builder_->SetOpName(input.name());
      auto out_shape = OnnxShapeToHloTensorType(input.type(), batch_size_);
      auto in_shape = out_shape;
      if (options_.io_type) in_shape.SetElementType(*options_.io_type);
      const std::string input_name(input.name());
      const auto param_id =
          MakeParameter(input_name, TensorTypeFromProto(in_shape.ToProto()), false);
      onnx_name_to_value_id_[input_name] = param_id;
      if (in_shape.GetElementType() != out_shape.GetElementType()) {
        pending_input_conversions_.push_back(
            {input_name, param_id, out_shape.GetElementType()});
      }
    }
  }

  void FinalizeInputConversions() {
    for (const auto& conversion : pending_input_conversions_) {
      BuilderContext ctx(builder_.get());
      builder_->SetOpType("input");
      builder_->SetOpName(conversion.name);
      const ValueId converted =
          EmitConvert(conversion.parameter_id, conversion.target_element_type);
      onnx_name_to_value_id_[conversion.name] = converted;
    }
    pending_input_conversions_.clear();
  }

  void ValidateParameterValueIds() const {
    for (size_t i = 0; i < params_.size(); ++i) {
      if (params_[i].value_id == i) continue;
      throw Exception("Parameter ValueId ordering violation at index " +
                      std::to_string(i) + ": expected ValueId " +
                      std::to_string(i) + ", got " +
                      std::to_string(params_[i].value_id) + " for name [" +
                      params_[i].name + "]");
    }
  }

  // Makes a parameter instruction (for inputs or large constants).
  ValueId MakeParameter(const std::string& name, const TensorType& shape,
                        bool is_constant) {
    auto res = builder_->Parameter(shape);
    params_.push_back({name, res, is_constant});
    return res;
  }

  void BuildGraph(const pblczero::GraphProto& graph) {
    for (const auto& node : graph.node()) {
      // Set up the context so that nodes have metadata from the original
      // ONNX.
      BuilderContext ctx(builder_.get());
      builder_->SetOpType(node.op_type());
      builder_->SetOpName(node.name());
      DispatchNode(node);
    }
  }

  void PrecreateInitializerParameters(const pblczero::GraphProto& graph) {
    std::unordered_set<std::string> used_initializer_names;
    for (const auto& node : graph.node()) {
      for (const auto& input_name : node.input()) {
        const std::string name(input_name);
        if (onnx_name_to_value_id_.contains(name)) continue;
        const auto iter = initializers_.find(name);
        if (iter == initializers_.end()) continue;
        if (iter->second->raw_data().size() <=
            options_.max_inline_constant_size) {
          continue;
        }
        used_initializer_names.insert(name);
      }
    }

    std::vector<std::string> sorted_initializer_names(
        used_initializer_names.begin(), used_initializer_names.end());
    std::sort(sorted_initializer_names.begin(), sorted_initializer_names.end());

    for (const auto& name : sorted_initializer_names) {
      if (onnx_name_to_value_id_.contains(name)) continue;
      const auto iter = initializers_.find(name);
      if (iter == initializers_.end()) continue;

      BuilderContext ctx(builder_.get());
      builder_->SetOpType("initializer");
      builder_->SetOpName(name);
      const auto shape =
          TensorTypeFromProto(OnnxTensorToHloTensorType(*iter->second).ToProto());
      onnx_name_to_value_id_[name] = MakeParameter(name, shape, true);
    }
  }

  bool IsParameterMapNameSorted() const {
    std::string previous_initializer_name;
    bool seen_first_initializer = false;
    for (const auto& param : params_) {
      if (!param.is_constant) continue;
      if (!seen_first_initializer) {
        previous_initializer_name = param.name;
        seen_first_initializer = true;
        continue;
      }
      if (param.name < previous_initializer_name) return false;
      previous_initializer_name = param.name;
    }
    return true;
  }

  // Calls the correct function to handle the ONNX node, and stores output in
  // the map.
  void DispatchNode(const pblczero::NodeProto& node) {
    try {
      auto iter = onnx_op_to_builder_.find(std::string(node.op_type()));
      if (iter == onnx_op_to_builder_.end()) {
        throw Exception("Unsupported ONNX op.");
      }
      auto outputs = (this->*iter->second)(node);
      if (outputs.size() != node.output_size()) {
        throw Exception("Node produced wrong number of outputs");
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        onnx_name_to_value_id_[std::string(node.output(i))] = outputs[i];
      }
    } catch (Exception& e) {
      std::string inputs;
      for (const auto& input : node.input()) {
        try {
          auto flow = GetFlowByName(input);
          inputs += "\n  input=[" + input +
                    "]  shape=" + GetShape(flow).OutputAsJson();
        } catch (const Exception&) {
          inputs += "\n  input=[" + input + "]  shape=(not found)";
        }
      }
      throw Exception("Error in ONNX op=[" + std::string(node.op_type()) +
                      "] name=[" + std::string(node.name()) + "]: " + e.what() +
                      inputs);
    }
  }

  static constexpr ValueId kInvalidValueId = std::numeric_limits<ValueId>::max();

  pblczero::XlaShapeProto GetShape(ValueId value) const {
    return TensorTypeToProto(builder_->GetType(value));
  }

  std::unordered_map<std::string, ValueId> onnx_name_to_value_id_;
  std::unordered_map<std::string, std::vector<ValueId> (Onnx2HloConverter::*)(
                                      const pblczero::NodeProto&)>
      onnx_op_to_builder_;
  std::unordered_map<std::string, const pblczero::TensorProto*> initializers_;
  std::unique_ptr<IBuilder> builder_;
  stablehlo::semantic::SemanticBuilder* semantic_builder_ = nullptr;
  size_t batch_size_ = 0;
  size_t opset_version_ = 0;
  Onnx2HloOptions options_;
  struct Param {
    std::string name;
    ValueId value_id;
    bool is_constant;
  };
  struct PendingInputConversion {
    std::string name;
    ValueId parameter_id;
    pblczero::XlaShapeProto::Type target_element_type;
  };
  std::vector<Param> params_;
  std::vector<PendingInputConversion> pending_input_conversions_;
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
  return std::make_unique<XlaTensorNotOwned>(
      OnnxTypeToXlaType(onnx_tensor.data_type()), onnx_tensor.dims(),
      onnx_tensor.raw_data());
}

}  // namespace lczero
