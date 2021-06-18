/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "neural/onnx/converter.h"

#include <initializer_list>

#include "neural/onnx/onnx.pb.h"
#include "proto/net.pb.h"
#include "utils/exception.h"
#include "version.h"

namespace lczero {
namespace {

class Converter {
 public:
  Converter(const pblczero::Net& net,
            const WeightsToOnnxConverterOptions& options)
      : src_(net), options_(options) {}

  void Convert(pblczero::Net* dst);

 private:
  void CopyGenericFields(pblczero::Net* dst);
  void GenerateOnnx(pblczero::OnnxModel* onnx);
  void FillValueInfo(pblczero::ValueInfoProto* vip, const std::string& name,
                     std::initializer_list<int> dims);

  pblczero::TensorProto::DataType GetDataType() const;

  const pblczero::Net& src_;
  const WeightsToOnnxConverterOptions& options_;
};

pblczero::TensorProto::DataType Converter::GetDataType() const {
  switch (options_.data_type_) {
    case WeightsToOnnxConverterOptions::DataType::kFloat32:
      return pblczero::TensorProto::FLOAT;
    default:
      return pblczero::TensorProto::UNDEFINED;
  }
}

void Converter::FillValueInfo(pblczero::ValueInfoProto* vip,
                              const std::string& name,
                              std::initializer_list<int> dims) {
  vip->set_name(name);
  auto* type = vip->mutable_type()->mutable_tensor_type();
  type->set_elem_type(GetDataType());
  auto* shape = type->mutable_shape();
  for (const auto d : dims) {
    auto* dim = shape->add_dim();
    if (d < 0) {
      dim->set_dim_param("batch");
    } else {
      dim->set_dim_value(d);
    }
  }
}

void Converter::GenerateOnnx(pblczero::OnnxModel* onnx) {
  pblczero::ModelProto model;

  model.set_ir_version(4);
  model.set_producer_name("Lc0");
  model.set_producer_version(GetVersionStr());
  model.add_opset_import()->set_version(9);

  onnx->set_input_planes(options_.input_planes_name);
  auto* graph = model.mutable_graph();
  FillValueInfo(graph->add_input(), options_.input_planes_name,
                {-1, 112, 8, 8});

  onnx->set_model(model.OutputAsString());
}

void Converter::CopyGenericFields(pblczero::Net* dst) {
  dst->set_license(src_.license());
  dst->set_magic(src_.magic());
  auto* min_version = dst->mutable_min_version();
  min_version->set_minor(28);
  auto* network_format = dst->mutable_format()->mutable_network_format();
  network_format->set_input(src_.format().network_format().input());
  network_format->set_output(src_.format().network_format().output());
  network_format->set_network(pblczero::NetworkFormat::NETWORK_ONNX);
  // We add convolution-to-classical layer to ONNX layers anyway, so from
  // outside they are all POLICY_CLASSICAL.
  network_format->set_policy(pblczero::NetworkFormat::POLICY_CLASSICAL);
  network_format->set_value(src_.format().network_format().value());
  network_format->set_moves_left(src_.format().network_format().moves_left());

  *dst->mutable_training_params() = src_.training_params();
}

void Converter::Convert(pblczero::Net* dst) {
  if (src_.has_onnx_model() && src_.format().network_format().network() ==
                                   pblczero::NetworkFormat::NETWORK_ONNX) {
    *dst = src_;
    return;
  }
  if (!src_.has_weights()) {
    throw Exception("The network doesn't have weights.");
  }
  if (src_.has_onnx_model()) {
    throw Exception("The network already has ONNX section.");
  }
  CopyGenericFields(dst);
  GenerateOnnx(dst->mutable_onnx_model());
}

}  // namespace

const pblczero::Net ConvertWeightsToOnnx(
    pblczero::Net& net, const WeightsToOnnxConverterOptions& options) {
  Converter converter(net, options);
  pblczero::Net dst;
  converter.Convert(&dst);
  return dst;
}

}  // namespace lczero