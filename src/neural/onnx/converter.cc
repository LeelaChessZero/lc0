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

#include <cstddef>
#include <initializer_list>
#include <memory>

#include "neural/network.h"
#include "neural/network_legacy.h"
#include "neural/onnx/adapters.h"
#include "neural/onnx/builder.h"
#include "proto/net.pb.h"
#include "utils/exception.h"

namespace lczero {
namespace {

class Converter {
 public:
  Converter(const pblczero::Net& net,
            const WeightsToOnnxConverterOptions& options)
      : src_(net), options_(options) {}

  void Convert(pblczero::Net* dst);

 private:
  int NumFilters() const {
    return LayerAdapter(src_.weights().input().weights()).size() /
           kInputPlanes / 9;
  }
  size_t NumBlocks() const { return src_.weights().residual_size(); }
  void CopyGenericFields(pblczero::Net* dst);
  void GenerateOnnx(pblczero::OnnxModel* onnx);
  void FillValueInfo(pblczero::ValueInfoProto* vip, const std::string& name,
                     std::initializer_list<int> dims);

  std::string MakeConvBlock(OnnxBuilder* builder,
                            const LegacyWeights::ConvBlock&, int input_channels,
                            int output_channels, const std::string& input,
                            const std::string& name,
                            const LegacyWeights::SEunit* se_unit = nullptr,
                            const std::string& mixin = "", bool relu = true);

  std::string MakeResidualBlock(OnnxBuilder* builder,
                                const LegacyWeights::Residual&,
                                const std::string& input,
                                const std::string& name);

  std::string MakeSqueezeAndExcite(OnnxBuilder* builder,
                                   const LegacyWeights::SEunit& se_unit,
                                   const std::string& input,
                                   const std::string& name);

  void AddStdInitializers(OnnxBuilder* builder);

  pblczero::TensorProto::DataType GetDataType() const;
  std::unique_ptr<OnnxConst> GetWeghtsConverter(
      const std::vector<float>&, std::initializer_list<int> dims,
      std::initializer_list<int> order);

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

std::unique_ptr<OnnxConst> Converter::GetWeghtsConverter(
    const std::vector<float>& weights, std::initializer_list<int> dims,
    std::initializer_list<int> order = {}) {
  switch (options_.data_type_) {
    case WeightsToOnnxConverterOptions::DataType::kFloat32:
      return std::make_unique<FloatOnnxWeightsAdapter>(weights, dims, order);
  }
  throw Exception("Data type " +
                  std::to_string(static_cast<int>(options_.data_type_)) +
                  " is not supported in weights converter");
}

std::string Converter::MakeSqueezeAndExcite(
    OnnxBuilder* builder, const LegacyWeights::SEunit& se_unit,
    const std::string& input, const std::string& name) {
  const int se_filters = se_unit.b1.size();

  auto flow = builder->GlobalAveragePool(name + "/pooled", input);
  flow = builder->Squeeze(name + "/squeeze", flow);
  flow = builder->MatMul(
      name + "/matmul1", flow,
      *GetWeghtsConverter(se_unit.w1, {se_filters, NumFilters()}));
  flow = builder->Add(name + "/add1", flow,
                      *GetWeghtsConverter(se_unit.b1, {se_filters}));
  flow = builder->Relu(name + "/relu", flow);
  flow = builder->MatMul(
      name + "/matmul2", flow,
      *GetWeghtsConverter(se_unit.w2, {se_filters, NumFilters()}));
  flow = builder->Add(name + "/add2", flow,
                      *GetWeghtsConverter(se_unit.b2, {se_filters}));
  flow = builder->Reshape(name + "/reshape", flow, "/const/se_reshape");

  auto splits = builder->Split(name + "/split", flow, 1);

  flow = builder->Sigmoid(name + "/sigmoid", splits.first);
  flow = builder->MatMul(name + "/matmul3", flow, input);
  return builder->Add(name + "/add3", flow, splits.second);
}

std::string Converter::MakeConvBlock(OnnxBuilder* builder,
                                     const LegacyWeights::ConvBlock& weights,
                                     int input_channels, int output_channels,
                                     const std::string& input,
                                     const std::string& name,
                                     const LegacyWeights::SEunit* seunit,
                                     const std::string& mixin, bool relu) {
  auto flow =
      builder->Conv(name, input,
                    *GetWeghtsConverter(weights.weights,
                                        {3, 3, input_channels, output_channels},
                                        {3, 2, 0, 1}),
                    *GetWeghtsConverter(weights.biases, {NumFilters()}));

  if (seunit) flow = MakeSqueezeAndExcite(builder, *seunit, flow, name + "/se");
  if (!mixin.empty()) flow = builder->Add(name + "/mixin", flow, mixin);
  if (relu) flow = builder->Relu(name + "/relu", flow);
  return flow;
}

std::string Converter::MakeResidualBlock(OnnxBuilder* builder,
                                         const LegacyWeights::Residual& res,
                                         const std::string& input,
                                         const std::string& name) {
  auto block1 = MakeConvBlock(builder, res.conv1, NumFilters(), NumFilters(),
                              input, name + "/conv1");
  return MakeConvBlock(builder, res.conv2, NumFilters(), NumFilters(), block1,
                       name + "/conv2", res.has_se ? &res.se : nullptr, input);
}

void Converter::AddStdInitializers(OnnxBuilder* builder) {
  builder->AddInitializer("/const/se_reshape",
                          Int32OnnxConst({-1, NumFilters() * 2, 1, 1}, {4}));
}

void Converter::GenerateOnnx(pblczero::OnnxModel* onnx) {
  LegacyWeights weights(src_.weights());
  OnnxBuilder builder;

  AddStdInitializers(&builder);

  onnx->set_input_planes(options_.input_planes_name);
  builder.AddInput(options_.input_planes_name, {-1, 112, 8, 8}, GetDataType());

  // Input convolution.
  auto flow = MakeConvBlock(&builder, weights.input, kInputPlanes, NumFilters(),
                            options_.input_planes_name, "inputconv");

  // Residual tower.
  for (size_t i = 0; i < NumBlocks(); ++i) {
    flow = MakeResidualBlock(&builder, weights.residual[i], flow,
                             "block" + std::to_string(i));
  }

  onnx->set_model(builder.OutputAsString());
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
    const pblczero::Net& net, const WeightsToOnnxConverterOptions& options) {
  Converter converter(net, options);
  pblczero::Net dst;
  converter.Convert(&dst);
  return dst;
}

}  // namespace lczero