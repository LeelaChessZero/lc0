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

#include <climits>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>

#include "neural/loader.h"
#include "neural/network.h"
#include "neural/network_legacy.h"
#include "neural/onnx/adapters.h"
#include "neural/onnx/builder.h"
#include "neural/shared/activation.h"
#include "neural/shared/attention_policy_map.h"
#include "neural/shared/policy_map.h"
#include "proto/net.pb.h"
#include "utils/bf16_utils.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/fp8_utils.h"

namespace lczero {
namespace {

class Converter {
 public:
  Converter(const pblczero::Net& net,
            const WeightsToOnnxConverterOptions& options)
      : src_(net),
        options_(options),
        default_activation_(
            net.format().network_format().default_activation() ==
                    pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH
                ? ACTIVATION_MISH
                : ACTIVATION_RELU),
        default_eps_(net.format().network_format().input_embedding() ==
                             pblczero::NetworkFormat::INPUT_EMBEDDING_PE_DENSE
                         ? 1e-3
                         : 1e-6) {}

  void Convert(pblczero::Net* dst);

 private:
  int NumFilters() const {
    return LayerAdapter(src_.weights().input().weights()).size() /
           kInputPlanes / 9;
  }
  size_t NumResBlocks() const { return src_.weights().residual_size(); }
  size_t NumEncBlocks() const { return src_.weights().encoder().size(); }
  void CopyGenericFields(pblczero::Net* dst);
  void GenerateOnnx(pblczero::OnnxModel* onnx);
  void FillValueInfo(pblczero::ValueInfoProto* vip, const std::string& name,
                     std::initializer_list<int> dims);

  std::string MakeConvBlock(OnnxBuilder* builder,
                            const MultiHeadWeights::ConvBlock&,
                            int input_channels, int output_channels,
                            const std::string& input, const std::string& name,
                            const MultiHeadWeights::SEunit* se_unit = nullptr,
                            const std::string& mixin = "",
                            bool activation = true, int filters = 3);

  std::string MakeResidualBlock(OnnxBuilder* builder,
                                const MultiHeadWeights::Residual&,
                                const std::string& input,
                                const std::string& name);

  std::string AttentionBodyMapEmbedding(OnnxBuilder* builder,
                                        const std::string& input);

  std::string AttentionBodyDenseEmbedding(OnnxBuilder* builder,
                                          const std::string& input,
                                          const MultiHeadWeights& weights,
                                          int embedding_dense_size);

  std::string MakeAttentionBody(OnnxBuilder* builder, const std::string& input,
                                const MultiHeadWeights& weights);

  std::string MakeSqueezeAndExcite(OnnxBuilder* builder,
                                   const MultiHeadWeights::SEunit& se_unit,
                                   const std::string& input,
                                   const std::string& name);

  std::string MakeMish(OnnxBuilder* builder, const std::string& input,
                       const std::string& name);

  std::string MakeSwish(OnnxBuilder* builder, const std::string& input,
                        const std::string& name);

  std::string MakeActivation(OnnxBuilder* builder, const std::string& input,
                             const std::string& name,
                             ActivationFunction activation);

  std::string MakeSmolgen(OnnxBuilder* builder,
                          const MultiHeadWeights::EncoderLayer& layer,
                          int embedding_size, int heads,
                          const std::string& encoder_in,
                          const std::string& name);

  std::string MakeLayerNorm(OnnxBuilder* builder, const std::string& input,
                            const std::string& name,
                            const lczero::OnnxConst& gammas,
                            const lczero::OnnxConst& betas, float eps = 1e-6);

  std::string MakeFFN(OnnxBuilder* builder, const MultiHeadWeights::FFN& ffn,
                      int embedding_size, const std::string& ffn_in,
                      const std::string& name, ActivationFunction activation,
                      float alpha);

  std::string MakeEncoderLayer(OnnxBuilder* builder,
                               const MultiHeadWeights::EncoderLayer& layer,
                               int embedding_size, int heads,
                               const std::string& encoder_in,
                               const std::string& name,
                               ActivationFunction activation,
                               float alpha = 1.0f);

  std::string MakeAttentionPolicy(OnnxBuilder* builder,
                                  const std::string& input,
                                  const MultiHeadWeights& weights,
                                  const MultiHeadWeights::PolicyHead& head);

  void MakePolicyHead(pblczero::OnnxModel* onnx, OnnxBuilder* builder,
                      const std::string& input,
                      const MultiHeadWeights& weights);

  void MakeValueHead(pblczero::OnnxModel* onnx, OnnxBuilder* builder,
                     const std::string& input, const MultiHeadWeights& weights);

  void MakeMovesLeftHead(pblczero::OnnxModel* onnx, OnnxBuilder* builder,
                         const std::string& input,
                         const MultiHeadWeights& weights);

  void AddStdInitializers(OnnxBuilder* builder);

  pblczero::TensorProto::DataType GetDataType() const;
  std::unique_ptr<OnnxConst> GetWeghtsConverter(
      const std::vector<float>&, std::initializer_list<int> dims,
      std::initializer_list<int> order = {});

  std::unique_ptr<OnnxConst> GetScalarConverter(float in);

  std::string StartOptionalBf16Fix(OnnxBuilder* builder, std::string flow,
                                   std::string name);

  std::string EndOptionalBf16Fix(OnnxBuilder* builder, std::string flow,
                                 std::string name);

  const pblczero::Net& src_;
  const WeightsToOnnxConverterOptions& options_;
  const ActivationFunction default_activation_;
  const float default_eps_;
  bool se_reshape_init_ = false;
};

pblczero::TensorProto::DataType Converter::GetDataType() const {
  switch (options_.data_type) {
    case WeightsToOnnxConverterOptions::DataType::kFloat32:
      return pblczero::TensorProto::FLOAT;
    case WeightsToOnnxConverterOptions::DataType::kFloat16:
      return pblczero::TensorProto::FLOAT16;
    case WeightsToOnnxConverterOptions::DataType::kBFloat16:
      return pblczero::TensorProto::BFLOAT16;
    default:
      return pblczero::TensorProto::UNDEFINED;
  }
}

std::unique_ptr<OnnxConst> Converter::GetWeghtsConverter(
    const std::vector<float>& weights, std::initializer_list<int> dims,
    std::initializer_list<int> order) {
  switch (options_.data_type) {
    case WeightsToOnnxConverterOptions::DataType::kFloat32:
      return std::make_unique<FloatOnnxWeightsAdapter>(weights, dims, order);
    case WeightsToOnnxConverterOptions::DataType::kFloat16:
      return std::make_unique<Float16OnnxWeightsAdapter>(weights, dims, order);
    case WeightsToOnnxConverterOptions::DataType::kBFloat16:
      return std::make_unique<BFloat16OnnxWeightsAdapter>(weights, dims, order);
  }
  throw Exception("Data type " +
                  std::to_string(static_cast<int>(options_.data_type)) +
                  " is not supported in weights converter");
}

std::unique_ptr<OnnxConst> Converter::GetScalarConverter(float in) {
  switch (options_.data_type) {
    case WeightsToOnnxConverterOptions::DataType::kFloat32:
      return std::make_unique<FloatOnnxConst>(FloatOnnxConst({in}, {1}));
    case WeightsToOnnxConverterOptions::DataType::kFloat16:
      return std::make_unique<Float16OnnxConst>(
          Float16OnnxConst({FP32toFP16(in)}, {1}));
    case WeightsToOnnxConverterOptions::DataType::kBFloat16:
      return std::make_unique<BFloat16OnnxConst>(
          BFloat16OnnxConst({FP32toBF16(in)}, {1}));
  }
  throw Exception("Data type " +
                  std::to_string(static_cast<int>(options_.data_type)) +
                  " is not supported in scalar converter");
}

std::string Converter::StartOptionalBf16Fix(OnnxBuilder* builder,
                                            std::string flow,
                                            std::string name) {
  if (options_.relax_op_types ||
      options_.data_type !=
          WeightsToOnnxConverterOptions::DataType::kBFloat16) {
    return flow;
  }
  return builder->Cast(name + "/to_float", flow, pblczero::TensorProto::FLOAT);
}

std::string Converter::EndOptionalBf16Fix(OnnxBuilder* builder,
                                          std::string flow, std::string name) {
  if (options_.relax_op_types ||
      options_.data_type !=
          WeightsToOnnxConverterOptions::DataType::kBFloat16) {
    return flow;
  }
  return builder->Cast(name + "/to_bf16", flow,
                       pblczero::TensorProto::BFLOAT16);
}

std::string Converter::MakeMish(OnnxBuilder* builder, const std::string& input,
                                const std::string& name) {
  if (!options_.alt_mish || options_.opset < 9 ||
      options_.data_type != WeightsToOnnxConverterOptions::DataType::kFloat32) {
    std::string flow = input;
    flow = StartOptionalBf16Fix(builder, flow, name);
    if (options_.opset >= 18) {
      flow = builder->Mish(name, flow);
      return EndOptionalBf16Fix(builder, flow, name);
    }
    flow = builder->Softplus(name + "/softplus", flow);
    flow = EndOptionalBf16Fix(builder, flow, name);
    flow = builder->Tanh(name + "/tanh", flow);
    return builder->Mul(name, flow, input);
  } else {
    const OnnxConst& two =
        static_cast<const OnnxConst&>(FloatOnnxConst({2.0f}, {1}));
    const OnnxConst& zero =
        static_cast<const OnnxConst&>(FloatOnnxConst({0.0f}, {1}));
    auto e = builder->Exp(name + "/exp", input);
    auto flow = builder->Add(name + "/e+2", e, two);
    auto n = builder->Mul(name + "/n", e, flow);
    flow = builder->Add(name + "/n+2", n, two);
    auto d = builder->Div(name + "/d", input, flow);
    auto f = builder->Mul(name + "/n*d", n, d);
    flow = builder->Mul(name + "/2*d", d, two);
    auto t = builder->Sub(name + "/in-2*d", input, flow);
    flow = builder->Greater(name + "/compare", input, zero);
    return builder->Where(name, flow, t, f);
  }
}

std::string Converter::MakeSwish(OnnxBuilder* builder, const std::string& input,
                                 const std::string& name) {
  auto flow = builder->Sigmoid(name + "/sigmoid", input);
  return builder->Mul(name, flow, input);
}

std::string Converter::MakeActivation(OnnxBuilder* builder,
                                      const std::string& input,
                                      const std::string& name,
                                      ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      return builder->Relu(name + "/relu", input);
    case ACTIVATION_MISH:
      return MakeMish(builder, input, name + "/mish");
    case ACTIVATION_SELU: {
      auto flow = input;
      flow = StartOptionalBf16Fix(builder, flow, name);
      flow = builder->Selu(name + "/selu", flow);
      return EndOptionalBf16Fix(builder, flow, name);
    }
    case ACTIVATION_SWISH:
      return MakeSwish(builder, input, name + "/swish");
    case ACTIVATION_RELU_2: {
      auto flow = builder->Relu(name + "/sqrrelu/relu", input);
      return builder->Mul(name + "/sqrrelu/sqr", flow, flow);
    }
    case ACTIVATION_NONE:
      return input;
    default:
      throw Exception("Unsupported activation in " + name);
  }
}

std::string Converter::MakeSqueezeAndExcite(
    OnnxBuilder* builder, const MultiHeadWeights::SEunit& se_unit,
    const std::string& input, const std::string& name) {
  const int se_filters = se_unit.b1.size();

  if (!se_reshape_init_) {
    builder->AddInitializer("/const/se_reshape",
                            Int64OnnxConst({-1, NumFilters() * 2, 1, 1}, {4}));
    se_reshape_init_ = true;
  }
  auto flow = input;
  flow = builder->ReduceMean(name + "/reduce_mean", flow, {2, 3}, false);
  flow = builder->MatMul(
      name + "/matmul1", flow,
      *GetWeghtsConverter(se_unit.w1, {NumFilters(), se_filters}, {1, 0}));
  flow = builder->Add(name + "/add1", flow,
                      *GetWeghtsConverter(se_unit.b1, {se_filters}));
  flow = MakeActivation(builder, flow, name, default_activation_);
  flow = builder->MatMul(
      name + "/matmul2", flow,
      *GetWeghtsConverter(se_unit.w2, {se_filters, 2 * NumFilters()}, {1, 0}));
  flow = builder->Add(name + "/add2", flow,
                      *GetWeghtsConverter(se_unit.b2, {2 * NumFilters()}));
  flow = builder->Reshape(name + "/reshape", flow, "/const/se_reshape");

  auto splits = builder->Split(name + "/split", flow, 1);

  flow = builder->Sigmoid(name + "/sigmoid", splits[0]);
  flow = builder->Mul(name + "/mul", flow, input);
  return builder->Add(name + "/add3", flow, splits[1]);
}

std::string Converter::MakeConvBlock(
    OnnxBuilder* builder, const MultiHeadWeights::ConvBlock& weights,
    int input_channels, int output_channels, const std::string& input,
    const std::string& name, const MultiHeadWeights::SEunit* seunit,
    const std::string& mixin, bool activation, int filters) {
  auto flow = input;
  if (!options_.relax_op_types &&
      options_.data_type ==
          WeightsToOnnxConverterOptions::DataType::kBFloat16) {
    flow =
        builder->Cast(name + "/to_float", flow, pblczero::TensorProto::FLOAT);
    flow = builder->Conv(
        name, flow, *std::make_unique<FloatOnnxWeightsAdapter>(
                        weights.weights, std::initializer_list<int>(
                                             {output_channels, input_channels,
                                              filters, filters})),
        *std::make_unique<FloatOnnxWeightsAdapter>(
            weights.biases, std::initializer_list<int>({output_channels})),
        (filters - 1) / 2);
    flow =
        builder->Cast(name + "/to_bf16", flow, pblczero::TensorProto::BFLOAT16);
  } else {
    flow = builder->Conv(
        name, flow,
        *GetWeghtsConverter(weights.weights, {output_channels, input_channels,
                                              filters, filters}),
        *GetWeghtsConverter(weights.biases, {output_channels}),
        (filters - 1) / 2);
  }
  if (seunit) flow = MakeSqueezeAndExcite(builder, *seunit, flow, name + "/se");
  if (!mixin.empty()) flow = builder->Add(name + "/mixin", flow, mixin);
  if (activation) {
    flow = MakeActivation(builder, flow, name, default_activation_);
  }
  return flow;
}

std::string Converter::MakeResidualBlock(OnnxBuilder* builder,
                                         const MultiHeadWeights::Residual& res,
                                         const std::string& input,
                                         const std::string& name) {
  auto block1 = MakeConvBlock(builder, res.conv1, NumFilters(), NumFilters(),
                              input, name + "/conv1");
  return MakeConvBlock(builder, res.conv2, NumFilters(), NumFilters(), block1,
                       name + "/conv2", res.has_se ? &res.se : nullptr, input);
}

std::string Converter::MakeSmolgen(OnnxBuilder* builder,
                                   const MultiHeadWeights::EncoderLayer& layer,
                                   int embedding_size, int heads,
                                   const std::string& encoder_in,
                                   const std::string& name) {
  const auto smolgen_activation = static_cast<ActivationFunction>(
      src_.format().network_format().smolgen_activation());
  const auto activation = smolgen_activation == ACTIVATION_DEFAULT
                              ? default_activation_
                              : smolgen_activation;
  const int smolgen_hidden_channels =
      layer.mha.smolgen.compress.size() / embedding_size;
  const int smolgen_hidden_sz = layer.mha.smolgen.dense1_b.size();
  const int smolgen_gen_sz = layer.mha.smolgen.dense2_b.size() / heads;
  auto flow = builder->MatMul(
      name + "/smolgen/compress", encoder_in,
      *GetWeghtsConverter(layer.mha.smolgen.compress,
                          {embedding_size, smolgen_hidden_channels}, {1, 0}));
  flow = builder->Reshape(
      name + "/smolgen/compress/reshape", flow,
      builder->AddInitializer(
          "/const" + name + "/smolgen/compress/shape",
          Int64OnnxConst({-1, 64 * smolgen_hidden_channels}, {2})));
  flow = builder->MatMul(
      name + "/smolgen/dense1/w", flow,
      *GetWeghtsConverter(layer.mha.smolgen.dense1_w,
                          {64 * smolgen_hidden_channels, smolgen_hidden_sz},
                          {1, 0}));
  flow = builder->Add(
      name + "/smolgen/dense1/b", flow,
      *GetWeghtsConverter(layer.mha.smolgen.dense1_b, {smolgen_hidden_sz}));
  flow = MakeActivation(builder, flow, name + "/smolgen/dense1", activation);
  flow = MakeLayerNorm(
      builder, flow, name + "/smolgen/ln1",
      *GetWeghtsConverter(layer.mha.smolgen.ln1_gammas, {smolgen_hidden_sz}),
      *GetWeghtsConverter(layer.mha.smolgen.ln1_betas, {smolgen_hidden_sz}),
      1e-3);
  flow = builder->MatMul(
      name + "/smolgen/dense2/w", flow,
      *GetWeghtsConverter(layer.mha.smolgen.dense2_w,
                          {smolgen_hidden_sz, smolgen_gen_sz * heads}, {1, 0}));
  flow = builder->Add(name + "/smolgen/dense2/b", flow,
                      *GetWeghtsConverter(layer.mha.smolgen.dense2_b,
                                          {smolgen_gen_sz * heads}));
  flow = MakeActivation(builder, flow, name + "/smolgen/dense2", activation);
  flow = MakeLayerNorm(builder, flow, name + "/smolgen/ln2",
                       *GetWeghtsConverter(layer.mha.smolgen.ln2_gammas,
                                           {smolgen_gen_sz * heads}),
                       *GetWeghtsConverter(layer.mha.smolgen.ln2_betas,
                                           {smolgen_gen_sz * heads}),
                       1e-3);
  flow =
      builder->Reshape(name + "/smolgen/gen_from/reshape", flow,
                       builder->AddInitializer(
                           "/const" + name + "/smolgen/gen_from/shape",
                           Int64OnnxConst({-1, heads, smolgen_gen_sz}, {3})));
  flow = builder->MatMul(name + "/smolgen/smol_weight_gen", flow,
                         "/const/smolgen_w");
  flow = builder->Reshape(
      name + "/smolgen/out/reshape", flow,
      builder->AddInitializer("/const" + name + "/smolgen/out/shape",
                              Int64OnnxConst({-1, heads, 64, 64}, {4})));
  return flow;
}

std::string Converter::MakeLayerNorm(OnnxBuilder* builder,
                                     const std::string& input,
                                     const std::string& name,
                                     const lczero::OnnxConst& gammas,
                                     const lczero::OnnxConst& betas,
                                     float eps) {
  if (!options_.alt_layernorm) {
    return builder->LayerNormalization(name, input, gammas, betas, 1, eps);
  }
  auto in =
      builder->Cast(name + "/to_float", input, pblczero::TensorProto::FLOAT);
  auto flow = builder->ReduceMean(name + "/mean", in, {1});
  in = builder->Sub(name + "/centered", in, flow);
  flow = builder->Mul(name + "/squared", in, in);
  flow = builder->ReduceMean(name + "/var", flow, {1});
  flow =
      builder->Add(name + "/var_eps", flow,
                   static_cast<const OnnxConst&>(FloatOnnxConst({eps}, {1})));
  flow = builder->Sqrt(name + "/std", flow);
  flow = builder->Reciprocal(name + "/inv_std", flow);
  flow = builder->Mul(name + "/normalized", in, flow);
  flow = builder->Cast(name + "/to_data_type", flow, GetDataType());
  flow = builder->Mul(name + "/gammas", flow, gammas);
  flow = builder->Add(name + "/betas", flow, betas);
  return flow;
}

std::string Converter::MakeFFN(OnnxBuilder* builder,
                               const MultiHeadWeights::FFN& ffn,
                               int embedding_size, const std::string& ffn_in,
                               const std::string& name,
                               ActivationFunction activation, float alpha) {
  const int dff_size = ffn.dense1_b.size();
  auto flow = builder->MatMul(
      name + "/ffn/dense1/w", ffn_in,
      *GetWeghtsConverter(ffn.dense1_w, {embedding_size, dff_size}, {1, 0}));
  flow = builder->Add(name + "/ffn/dense1/b", flow,
                      *GetWeghtsConverter(ffn.dense1_b, {dff_size}));
  flow = MakeActivation(builder, flow, name + "/ffn/dense1", activation);
  flow = builder->MatMul(
      name + "/ffn/dense2/w", flow,
      *GetWeghtsConverter(ffn.dense2_w, {dff_size, embedding_size}, {1, 0}));
  flow = builder->Add(name + "/ffn/dense2/b", flow,
                      *GetWeghtsConverter(ffn.dense2_b, {embedding_size}));
  if (alpha != 1.0) {
    flow = builder->Mul(name + "/ffn/alpha", flow, *GetScalarConverter(alpha));
  }
  flow = builder->Add(name + "/ffn/skip", flow, ffn_in);
  return flow;
}

std::string Converter::MakeEncoderLayer(
    OnnxBuilder* builder, const MultiHeadWeights::EncoderLayer& layer,
    int embedding_size, int heads, const std::string& encoder_in,
    const std::string& name, ActivationFunction activation, float alpha) {
  const int d_model = layer.mha.q_b.size();
  const int depth = d_model / heads;

  auto mha_shape =
      builder->AddInitializer("/const" + name + "/mha/shape",
                              Int64OnnxConst({-1, 64, heads, depth}, {4}));
  auto flow = builder->MatMul(
      name + "/mha/Q/w", encoder_in,
      *GetWeghtsConverter(layer.mha.q_w, {embedding_size, d_model}, {1, 0}));
  flow = builder->Add(name + "/mha/Q/b", flow,
                      *GetWeghtsConverter(layer.mha.q_b, {d_model}));
  flow = builder->Reshape(name + "/mha/Q/reshape", flow, mha_shape);
  auto Q = builder->Transpose(name + "/mha/Q/transpose", flow, {0, 2, 1, 3});
  flow = builder->MatMul(
      name + "/mha/K/w", encoder_in,
      *GetWeghtsConverter(layer.mha.k_w, {embedding_size, d_model}, {1, 0}));
  flow = builder->Add(name + "/mha/K/b", flow,
                      *GetWeghtsConverter(layer.mha.k_b, {d_model}));
  flow = builder->Reshape(name + "/mha/K/reshape", flow, mha_shape);
  auto K = builder->Transpose(name + "/mha/K/transpose", flow, {0, 2, 3, 1});
  flow = builder->MatMul(
      name + "/mha/V/w", encoder_in,
      *GetWeghtsConverter(layer.mha.v_w, {embedding_size, d_model}, {1, 0}));
  flow = builder->Add(name + "/mha/V/b", flow,
                      *GetWeghtsConverter(layer.mha.v_b, {d_model}));
  flow = builder->Reshape(name + "/mha/V/reshape", flow, mha_shape);
  auto V = builder->Transpose(name + "/mha/V/transpose", flow, {0, 2, 1, 3});
  flow = builder->MatMul(name + "/mha/QK/matmul", Q, K);
  flow = builder->Mul(name + "/mha/QK/scale", flow,
                      *GetScalarConverter(1.0f / sqrtf(depth)));
  if (layer.mha.has_smolgen) {
    auto smolgen_weights =
        MakeSmolgen(builder, layer, embedding_size, heads, encoder_in, name);
    flow = builder->Add(name + "/smolgen_weights", flow, smolgen_weights);
  }
  flow = builder->Softmax(name + "/mha/QK/softmax", flow, 3);
  flow = builder->MatMul(name + "/mha/QKV/matmul", flow, V);
  if (heads > 1) {
    flow = builder->Transpose(name + "/mha/out/transpose", flow, {0, 2, 1, 3});
  }
  flow = builder->Reshape(
      name + "/mha/out/reshape", flow,
      builder->AddInitializer("/const" + name + "/mha/out/shape",
                              Int64OnnxConst({-1, d_model}, {2})));
  flow =
      builder->MatMul(name + "/mha/out/dense/w", flow,
                      *GetWeghtsConverter(layer.mha.dense_w,
                                          {d_model, embedding_size}, {1, 0}));
  flow = builder->Add(name + "/mha/out/dense/b", flow,
                      *GetWeghtsConverter(layer.mha.dense_b, {embedding_size}));
  if (alpha != 1.0) {
    flow =
        builder->Mul(name + "/alpha*input", flow, *GetScalarConverter(alpha));
  }
  flow = builder->Add(name + "/mha/out/skip", flow, encoder_in);
  flow = MakeLayerNorm(builder, flow, name + "/ln1",
                       *GetWeghtsConverter(layer.ln1_gammas, {embedding_size}),
                       *GetWeghtsConverter(layer.ln1_betas, {embedding_size}),
                       default_eps_);
  const auto ffn_activation = static_cast<ActivationFunction>(
      src_.format().network_format().ffn_activation());
  flow = MakeFFN(
      builder, layer.ffn, embedding_size, flow, name,
      ffn_activation == ACTIVATION_DEFAULT ? activation : ffn_activation,
      alpha);
  flow = MakeLayerNorm(builder, flow, name + "/ln2",
                       *GetWeghtsConverter(layer.ln2_gammas, {embedding_size}),
                       *GetWeghtsConverter(layer.ln2_betas, {embedding_size}),
                       default_eps_);
  return flow;
}

std::string Converter::AttentionBodyMapEmbedding(OnnxBuilder* builder,
                                                 const std::string& input) {
  auto flow = input;
  flow = builder->Reshape(
      "/attn_body/reshape", flow,
      builder->AddInitializer("/const/att_body_shape",
                              Int64OnnxConst({-1, 64, 112}, {3})));
  std::string pad;
  if (options_.opset < 8 || (options_.no_shape && options_.batch_size < 0)) {
    pad = builder->Slice("/attn_body/pad/slice", flow, {0, 0, 0},
                         {INT_MAX, 1, 1});
    pad =
        builder->Reshape("/attn_body/pad/reshape_in", pad,
                         builder->AddInitializer("/const/pad_in_shape",
                                                 Int64OnnxConst({-1, 1}, {2})));
    pad = builder->Sub("/attn_body/pad/zeros_vec", pad, pad);
    pad =
        builder->Add("/attn_body/pad/one_vec", pad, *GetScalarConverter(1.0f));
    pad = builder->MatMul(
        "/attn_body/pad/expand", pad,
        builder->AddInitializer(
            "/const/pos_encoding",
            *GetWeghtsConverter(
                std::vector<float>(kPosEncoding[0], kPosEncoding[0] + 64 * 64),
                {1, 64 * 64})));

    pad = builder->Reshape(
        "/attn_body/pad/reshape_out", pad,
        builder->AddInitializer("/const/pad_out_shape",
                                Int64OnnxConst({-1, 64, 64}, {3})));
  } else if (options_.batch_size < 0) {
    pad = builder->Shape("/attn_body/shape", flow);
    pad = builder->Slice("/attn_body/batch", pad, {0}, {1});
    pad = builder->Concat(
        "/attn_body/pos_encoding_shape",
        {pad, builder->AddInitializer("/const/pos_encoding_shape",
                                      Int64OnnxConst({64, 64}, {2}))},
        0);
    pad = builder->Expand(
        "/attn_body/expand",
        builder->AddInitializer(
            "/const/pos_encoding",
            *GetWeghtsConverter(
                std::vector<float>(kPosEncoding[0], kPosEncoding[0] + 64 * 64),
                {1, 64, 64})),
        pad);
  } else {
    pad = builder->AddInitializer(
        "/const/pos_encoding_shape",
        Int64OnnxConst({options_.batch_size, 64, 64}, {3}));
    pad = builder->Expand(
        "/attn_body/expand",
        builder->AddInitializer(
            "/const/pos_encoding",
            *GetWeghtsConverter(
                std::vector<float>(kPosEncoding[0], kPosEncoding[0] + 64 * 64),
                {1, 64, 64})),
        pad);
  }
  flow = builder->Concat("/attn_body/padded_input", {flow, pad}, 2);
  flow =
      builder->Reshape("/attn_body/reshape2", flow,
                       builder->AddInitializer("/const/att_body_shape2",
                                               Int64OnnxConst({-1, 176}, {2})));
  return flow;
}

std::string Converter::AttentionBodyDenseEmbedding(
    OnnxBuilder* builder, const std::string& input,
    const MultiHeadWeights& weights, int embedding_dense_size) {
  auto flow = input;

  flow = builder->Reshape(
      "/attn_body/reshape", flow,
      builder->AddInitializer("/const/att_body_shape",
                              Int64OnnxConst({-1, 64, 112}, {3})));
  auto pos_info = builder->Slice("/attn_body/embedding/slice", flow, {0, 0, 0},
                                 {INT_MAX, 64, 12});
  pos_info = builder->Reshape(
      "/attn_body/embedding/reshape", pos_info,
      builder->AddInitializer("/const/pos_info_shape",
                              Int64OnnxConst({-1, 64 * 12}, {2})));

  pos_info = builder->MatMul(
      "/attn_body/embedding/preprocess/matmul", pos_info,
      *GetWeghtsConverter(weights.ip_emb_preproc_w,
                          {64 * 12, 64 * embedding_dense_size}, {1, 0}));
  pos_info = builder->Add("/attn_body/embedding/preprocess/add", pos_info,
                          *GetWeghtsConverter(weights.ip_emb_preproc_b,
                                              {64 * embedding_dense_size}));

  pos_info = builder->Reshape(
      "/attn_body/embedding/preprocess/reshape", pos_info,
      builder->AddInitializer(
          "/const/pos_info_processed_shape",
          Int64OnnxConst({-1, 64, embedding_dense_size}, {3})));

  flow = builder->Concat("/attn_body/embedding/concat", {flow, pos_info}, 2);

  flow = builder->Reshape(
      "/attn_body/embedding/out/reshape", flow,
      builder->AddInitializer(
          "/const/embedding/out_shape",
          Int64OnnxConst({-1, 112 + embedding_dense_size}, {2})));

  return flow;
}

std::string Converter::MakeAttentionBody(OnnxBuilder* builder,
                                         const std::string& input,
                                         const MultiHeadWeights& weights) {
  if (weights.has_smolgen) {
    builder->AddInitializer(
        "/const/smolgen_w",
        *GetWeghtsConverter(
            weights.smolgen_w,
            {static_cast<int>(weights.smolgen_w.size() / 4096), 4096}, {1, 0}));
  }
  auto input_embedding = src_.format().network_format().input_embedding();
  using network_format = pblczero::NetworkFormat;
  auto flow = builder->Transpose("/attn_body/transpose", input, {0, 2, 3, 1});
  int fist_stage_out_C = 0;

  if (NumResBlocks() > 0) {
    flow = builder->Reshape(
        "/attn_body/reshape", flow,
        builder->AddInitializer("/const/att_body_shape",
                                Int64OnnxConst({-1, NumFilters()}, {2})));
    fist_stage_out_C = NumFilters();
  } else if (input_embedding == network_format::INPUT_EMBEDDING_PE_MAP) {
    flow = AttentionBodyMapEmbedding(builder, flow);
    fist_stage_out_C = 176;
  } else if (input_embedding == network_format::INPUT_EMBEDDING_PE_DENSE) {
    int embedding_dense_size = weights.ip_emb_preproc_b.size() / 64;
    flow = AttentionBodyDenseEmbedding(builder, flow, weights,
                                       embedding_dense_size);
    fist_stage_out_C = 112 + embedding_dense_size;
  } else {
    throw Exception("Attention body missing input embedding.");
  }

  int embedding_size = weights.ip_emb_b.size();
  flow = builder->MatMul(
      "/attn_body/matmul", flow,
      *GetWeghtsConverter(weights.ip_emb_w, {fist_stage_out_C, embedding_size},
                          {1, 0}));
  flow = builder->Add("/attn_body/add", flow,
                      *GetWeghtsConverter(weights.ip_emb_b, {embedding_size}));
  flow = MakeActivation(builder, flow, "/attn_body", default_activation_);

  if (input_embedding == network_format::INPUT_EMBEDDING_PE_DENSE) {
    flow = MakeLayerNorm(
        builder, flow, "/attn_body/ln",
        *GetWeghtsConverter(weights.ip_emb_ln_gammas, {embedding_size}),
        *GetWeghtsConverter(weights.ip_emb_ln_betas, {embedding_size}), 1e-3);
  }

  if (weights.ip_mult_gate.size() > 0 || weights.ip_add_gate.size() > 0) {
    flow = builder->Reshape(
        "/attn_body/ma_gating/rehape1", flow,
        builder->AddInitializer("/const/ma_gating/shape1",
                                Int64OnnxConst({-1, 64, embedding_size}, {3})));
    if (weights.ip_mult_gate.size() > 0) {
      flow = builder->Mul("/ip_mul_gate", flow,
                          *GetWeghtsConverter(weights.ip_mult_gate,
                                              {64, embedding_size}, {1, 0}));
    }
    if (weights.ip_add_gate.size() > 0) {
      flow = builder->Add("/ip_add_gate", flow,
                          *GetWeghtsConverter(weights.ip_add_gate,
                                              {64, embedding_size}, {1, 0}));
    }
    flow = builder->Reshape(
        "/attn_body/ma_gating/rehape2", flow,
        builder->AddInitializer("/const/ma_gating/shape2",
                                Int64OnnxConst({-1, embedding_size}, {2})));
  }

  float alpha = std::pow(2.0f * NumEncBlocks(), -0.25f);

  if (input_embedding == network_format::INPUT_EMBEDDING_PE_DENSE) {
    flow = MakeFFN(builder, weights.ip_emb_ffn, embedding_size, flow,
                   "/attn_body", default_activation_, alpha);
    flow = MakeLayerNorm(
        builder, flow, "/attn_body/ln2",
        *GetWeghtsConverter(weights.ip_emb_ffn_ln_gammas, {embedding_size}),
        *GetWeghtsConverter(weights.ip_emb_ffn_ln_betas, {embedding_size}),
        1e-3);
  }

  for (size_t i = 0; i < NumEncBlocks(); i++) {
    flow = MakeEncoderLayer(
        builder, weights.encoder[i], embedding_size, weights.encoder_head_count,
        flow, "/encoder" + std::to_string(i), default_activation_, alpha);
  }
  return flow;
}

namespace {
std::vector<int> MakePolicyMap(const short* map, int size) {
  std::vector<int> policy_map(1858);
  int idx = 0;
  for (int i = 0; i < size; i++) {
    if (map[i] > -1) policy_map[map[i]] = idx;
    idx++;
  }
  return policy_map;
}
}  // namespace

std::string Converter::MakeAttentionPolicy(
    OnnxBuilder* builder, const std::string& input,
    const MultiHeadWeights& weights, const MultiHeadWeights::PolicyHead& head) {
  if (head.ip2_pol_b.empty()) {
    throw Exception("The policy head selected '" + options_.policy_head + "'" +
                    " is empty.");
  }
  const int embedding_size = weights.ip_emb_b.size();
  const int policy_embedding_size = head.ip_pol_b.size();
  const int policy_d_model = head.ip2_pol_b.size();
  auto flow = input;
  auto activation = src_.format().network_format().network() >=
                            pblczero::NetworkFormat::
                                NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
                        ? default_activation_
                        : ACTIVATION_SELU;
  if (NumEncBlocks() == 0) {
    flow = builder->Transpose("/policy/dense1/transpose", flow, {0, 2, 3, 1});

    flow = builder->Reshape(
        "/policy/dense1/reshape", flow,
        builder->AddInitializer("/const/policy_shape",
                                Int64OnnxConst({-1, NumFilters()}, {2})));
  }
  flow = builder->MatMul(
      "/policy/dense1/matmul", flow,
      *GetWeghtsConverter(head.ip_pol_w,
                          {NumEncBlocks() > 0 ? embedding_size : NumFilters(),
                           policy_embedding_size},
                          {1, 0}));
  flow = builder->Add("/policy/dense1/add", flow,
                      *GetWeghtsConverter(head.ip_pol_b,
                                          {policy_embedding_size}));
  flow = MakeActivation(builder, flow, "/policy/dense1", activation);

  for (size_t i = 0; i < head.pol_encoder.size(); i++) {
    std::string name = "/policy/enc_layer_" + std::to_string(i);

    flow =
        MakeEncoderLayer(builder, head.pol_encoder[i], policy_embedding_size,
                         head.pol_encoder_head_count, flow, name, activation);
  }
  auto encoder_out = flow;
  flow = builder->MatMul(
      "/policy/Q/matmul", encoder_out,
      *GetWeghtsConverter(head.ip2_pol_w,
                          {policy_embedding_size, policy_d_model}, {1, 0}));
  flow = builder->Add("/policy/Q/add", flow,
                      *GetWeghtsConverter(head.ip2_pol_b, {policy_d_model}));
  auto Q = builder->Reshape(
      "/policy/Q/reshape", flow,
      builder->AddInitializer("/const/QK_shape",
                              Int64OnnxConst({-1, 64, policy_d_model}, {3})));
  flow = builder->MatMul(
      "/policy/K/matmul", encoder_out,
      *GetWeghtsConverter(head.ip3_pol_w,
                          {policy_embedding_size, policy_d_model}, {1, 0}));
  flow = builder->Add("/policy/K/add", flow,
                      *GetWeghtsConverter(head.ip3_pol_b, {policy_d_model}));
  auto K = builder->Reshape("/policy/K/reshape", flow, "/const/QK_shape");
  flow = builder->Transpose("/policy/K/transpose", K, {0, 2, 1});
  flow = builder->MatMul("/policy/matmul", Q, flow);
  flow = builder->Mul("/policy/scale", flow,
                      *GetScalarConverter(1.0f / sqrtf(policy_d_model)));
  auto prom = builder->Slice("policy/promotion/slice", K, {0, 56, 0},
                             {INT_MAX, 64, policy_d_model});
  prom = builder->MatMul(
      "/policy/promotion/matmul", prom,
      *GetWeghtsConverter(head.ip4_pol_w, {policy_d_model, 4}, {1, 0}));
  prom = builder->Transpose("/policy/promotion/transpose", prom, {0, 2, 1});
  auto prom2 = builder->Split("/policy/promotion/split", prom, 1, {3, 1});
  prom = builder->Add("/policy/promotion/add", prom2[0], prom2[1]);
  prom = builder->Transpose("/policy/promotion/transpose2", prom, {0, 2, 1});
  prom = builder->Reshape(
      "/policy/promotion/reshape", prom,
      builder->AddInitializer("/const/policy_promotion_shape",
                              Int64OnnxConst({-1, 1, 24}, {3})));
  auto sl = builder->Slice("policy/promotion/slice2", flow, {0, 48, 56},
                           {INT_MAX, 56, 64});
  sl = builder->Reshape(
      "/policy/promotion/reshape2", sl,
      builder->AddInitializer("/const/policy_promotion_shape2",
                              Int64OnnxConst({-1, 64, 1}, {3})));
  sl = builder->Concat("/policy/promotion/concat", {sl, sl, sl}, 2);
  sl = builder->Reshape(
      "/policy/promotion/reshape3", sl,
      builder->AddInitializer("/const/policy_promotion_shape3",
                              Int64OnnxConst({-1, 8, 24}, {3})));
  prom = builder->Add("/policy/promotion/add2", sl, prom);
  prom = builder->Reshape(
      "/policy/promotion/reshape4", prom,
      builder->AddInitializer("/const/policy_promotion_shape4",
                              Int64OnnxConst({-1, 3, 64}, {3})));
  flow = builder->Concat("/policy/concat", {flow, prom}, 1);
  flow = builder->Reshape(
      "/policy/reshape", flow,
      builder->AddInitializer("/const/policy_out_shape",
                              Int64OnnxConst({-1, 67 * 64}, {2})));
  return builder->Gather(
      options_.output_policy_head, flow,
      builder->AddInitializer(
          "/const/mapping_table",
          Int32OnnxConst(
              MakePolicyMap(kAttnPolicyMap, std::size(kAttnPolicyMap)),
              {1858})),
      1);
}

void Converter::MakePolicyHead(pblczero::OnnxModel* onnx, OnnxBuilder* builder,
                               const std::string& input,
                               const MultiHeadWeights& weights) {
  // Check that selected policy head exists.
  if (weights.policy_heads.count(options_.policy_head) == 0) {
    throw Exception("The policy head you specified '" + options_.policy_head +
                    "'" + " does not exist in this net.");
  }
  const MultiHeadWeights::PolicyHead& head =
      weights.policy_heads.at(options_.policy_head);
  if (src_.format().network_format().policy() ==
      pblczero::NetworkFormat::POLICY_ATTENTION) {
    auto output = MakeAttentionPolicy(builder, input, weights, head);
    builder->AddOutput(output, {-1, 1858}, GetDataType());
    onnx->set_output_policy(output);
    return;
  } else if (head.policy.weights.empty()) {
    throw Exception("The policy head selected '" + options_.policy_head + "'" +
                    " is empty.");
  } else if (!head.policy1.weights.empty()) {
    // Conv policy head.
    if (NumEncBlocks() > 0) {
      throw Exception(
          "Convolutional policy not supported with attention body.");
    }
    auto flow = MakeConvBlock(builder, head.policy1, NumFilters(), NumFilters(),
                              input, "/policy/conv1");
    flow = MakeConvBlock(builder, head.policy, NumFilters(), 80, flow,
                         "/policy/conv2", nullptr, "", false);
    flow = builder->Reshape(
        "/policy/flatten", flow,
        builder->AddInitializer("/const/policy_shape",
                                Int64OnnxConst({-1, 80 * 8 * 8}, {2})));
    auto output = builder->Gather(
        options_.output_policy_head, flow,
        builder->AddInitializer(
            "/const/mapping_table",
            Int32OnnxConst(
                MakePolicyMap(kConvPolicyMap, std::size(kConvPolicyMap)),
                {1858})),
        1);
    builder->AddOutput(output, {options_.batch_size, 1858}, GetDataType());
    onnx->set_output_policy(output);
  } else {
    // Dense policy head.
    if (NumEncBlocks() > 0) {
      throw Exception("Classical policy not supported with attention body.");
    }
    const int pol_channels = head.policy.biases.size();
    auto flow = MakeConvBlock(builder, head.policy, NumFilters(), pol_channels,
                              input, "/policy/conv", nullptr, "", true, 1);
    flow =
        builder->Reshape("/policy/reshape", flow,
                         builder->AddInitializer(
                             "/const/policy_shape",
                             Int64OnnxConst({-1, pol_channels * 8 * 8}, {2})));
    flow = builder->MatMul(
        "/policy/dense/matmul", flow,
        *GetWeghtsConverter(head.ip_pol_w,
                            {pol_channels * 8 * 8, 1858}, {1, 0}));
    auto output = builder->Add(
        options_.output_policy_head, flow,
        *GetWeghtsConverter(head.ip_pol_b, {1858}));
    builder->AddOutput(output, {options_.batch_size, 1858}, GetDataType());
    onnx->set_output_policy(output);
  }
}

void Converter::MakeValueHead(pblczero::OnnxModel* onnx, OnnxBuilder* builder,
                              const std::string& input,
                              const MultiHeadWeights& weights) {
  // Check that selected value head exists.
  if (weights.value_heads.count(options_.value_head) == 0) {
    throw Exception("The value head you specified '" + options_.value_head +
                    "'" + " does not exist in this net.");
  }
  const MultiHeadWeights::ValueHead& head =
      weights.value_heads.at(options_.value_head);
  if (head.ip1_val_b.empty()) {
    throw Exception("The value head selected '" + options_.value_head + "'" +
                    " is empty.");
  }
  std::string flow;
  const int val_channels = NumEncBlocks() > 0 ? head.ip_val_b.size() : 32;
  if (NumEncBlocks() > 0) {
    int embedding_size = weights.ip_emb_b.size();
    flow = builder->MatMul(
        "/value/embed/matmul", input,
        *GetWeghtsConverter(head.ip_val_w, {embedding_size, val_channels},
                            {1, 0}));
    flow = builder->Add("/value/embed/add", flow,
                        *GetWeghtsConverter(head.ip_val_b, {val_channels}));
    flow = MakeActivation(builder, flow, "/value/embed", default_activation_);
  } else {
    flow = MakeConvBlock(builder, head.value, NumFilters(), val_channels, input,
                         "/value/conv", nullptr, "", true, 1);
  }
  flow = builder->Reshape(
      "/value/reshape", flow,
      builder->AddInitializer("/const/value_shape",
                              Int64OnnxConst({-1, val_channels * 8 * 8}, {2})));
  flow = builder->MatMul(
      "/value/dense1/matmul", flow,
      *GetWeghtsConverter(head.ip1_val_w, {val_channels * 8 * 8, 128}, {1, 0}));
  flow = builder->Add("/value/dense1/add", flow,
                      *GetWeghtsConverter(head.ip1_val_b, {128}));
  flow = MakeActivation(builder, flow, "/value/dense1", default_activation_);

  const bool wdl = src_.format().network_format().value() ==
                   pblczero::NetworkFormat::VALUE_WDL;
  if (wdl) {
    flow =
        builder->MatMul("/value/dense2/matmul", flow,
                        *GetWeghtsConverter(head.ip2_val_w, {128, 3}, {1, 0}));
    flow = builder->Add("/value/dense2/add", flow,
                        *GetWeghtsConverter(head.ip2_val_b, {3}));
    auto output = builder->Softmax(options_.output_wdl, flow);
    builder->AddOutput(output, {options_.batch_size, 3}, GetDataType());
    onnx->set_output_wdl(output);
  } else {
    flow =
        builder->MatMul("/value/dense2/matmul", flow,
                        *GetWeghtsConverter(head.ip2_val_w, {128, 1}, {1, 0}));
    flow = builder->Add("/value/dense2/add", flow,
                        *GetWeghtsConverter(head.ip2_val_b, {1}));
    auto output = builder->Tanh(options_.output_value, flow);
    builder->AddOutput(output, {options_.batch_size, 1}, GetDataType());
    onnx->set_output_value(output);
  }
}

void Converter::MakeMovesLeftHead(pblczero::OnnxModel* onnx,
                                  OnnxBuilder* builder,
                                  const std::string& input,
                                  const MultiHeadWeights& weights) {
  if (src_.format().network_format().moves_left() !=
      pblczero::NetworkFormat::MOVES_LEFT_V1) {
    return;
  }
  const int mlh_channels = NumEncBlocks() > 0
                               ? weights.ip_mov_b.size()
                               : weights.moves_left.biases.size();
  const int mlh_fc1_outputs = weights.ip1_mov_b.size();
  std::string flow;
  if (NumEncBlocks() > 0) {
    int embedding_size = weights.ip_emb_b.size();
    flow = builder->MatMul(
        "/mlh/embed/matmul", input,
        *GetWeghtsConverter(weights.ip_mov_w, {embedding_size, mlh_channels},
                            {1, 0}));
    flow = builder->Add("/mlh/embed/add", flow,
                        *GetWeghtsConverter(weights.ip_mov_b, {mlh_channels}));
    flow = MakeActivation(builder, flow, "/mlh/embed", default_activation_);
  } else {
    flow =
        MakeConvBlock(builder, weights.moves_left, NumFilters(), mlh_channels,
                      input, "/mlh/conv", nullptr, "", true, 1);
  }
  flow = builder->Reshape(
      "/mlh/reshape", flow,
      builder->AddInitializer("/const/mlh_shape",
                              Int64OnnxConst({-1, mlh_channels * 8 * 8}, {2})));
  flow = builder->MatMul(
      "/mlh/dense1/matmul", flow,
      *GetWeghtsConverter(weights.ip1_mov_w,
                          {mlh_channels * 8 * 8, mlh_fc1_outputs}, {1, 0}));
  flow =
      builder->Add("/mlh/dense1/add", flow,
                   *GetWeghtsConverter(weights.ip1_mov_b, {mlh_fc1_outputs}));
  flow = MakeActivation(builder, flow, "/mlh/dense1", default_activation_);
  flow = builder->MatMul(
      "/mlh/dense2/matmul", flow,
      *GetWeghtsConverter(weights.ip2_mov_w, {mlh_fc1_outputs, 1}, {1, 0}));
  flow = builder->Add("/mlh/dense2/add", flow,
                      *GetWeghtsConverter(weights.ip2_mov_b, {1}));
  flow = MakeActivation(builder, flow, "/mlh/dense2", default_activation_);
  auto output = builder->Identity(options_.output_mlh, flow);
  builder->AddOutput(output, {options_.batch_size, 1}, GetDataType());
  onnx->set_output_mlh(output);
}

void Converter::GenerateOnnx(pblczero::OnnxModel* onnx) {
  MultiHeadWeights weights(src_.weights());
  OnnxBuilder builder(options_.opset);

  if (GetDataType() == pblczero::TensorProto::FLOAT16) {
    onnx->set_data_type(pblczero::OnnxModel::FLOAT16);
  } else if (GetDataType() == pblczero::TensorProto::BFLOAT16) {
    onnx->set_data_type(pblczero::OnnxModel::BFLOAT16);
  } else {
    onnx->set_data_type(pblczero::OnnxModel::FLOAT);
  }
  onnx->set_input_planes(options_.input_planes_name);
  builder.AddInput(options_.input_planes_name, {options_.batch_size, 112, 8, 8},
                   GetDataType());

  auto flow = options_.input_planes_name;

  // Input convolution.
  if (NumResBlocks() > 0) {
    flow = MakeConvBlock(&builder, weights.input, kInputPlanes, NumFilters(),
                         flow, "/inputconv");
  }

  // Residual tower.
  for (size_t i = 0; i < NumResBlocks(); ++i) {
    flow = MakeResidualBlock(&builder, weights.residual[i], flow,
                             "/block" + std::to_string(i));
  }

  if (NumEncBlocks() > 0) {
    flow = MakeAttentionBody(&builder, flow, weights);
  }

  // Policy head.
  MakePolicyHead(onnx, &builder, flow, weights);
  // Value head.
  MakeValueHead(onnx, &builder, flow, weights);
  // Moves left head.
  MakeMovesLeftHead(onnx, &builder, flow, weights);

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

void CheckSrcFormat(const pblczero::NetworkFormat& nf) {
  switch (nf.network()) {
    case pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT:
    case pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT:
    case pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT:
    case pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
      break;
    default:
      throw Exception(
          "Network format " +
          pblczero::NetworkFormat::NetworkStructure_Name(nf.network()) +
          " is not supported by the ONNX converter.");
  }
  switch (nf.policy()) {
    case pblczero::NetworkFormat::POLICY_CLASSICAL:
    case pblczero::NetworkFormat::POLICY_CONVOLUTION:
    case pblczero::NetworkFormat::POLICY_ATTENTION:
      break;
    default:
      throw Exception("Policy format " +
                      pblczero::NetworkFormat::PolicyFormat_Name(nf.policy()) +
                      " is not supported by the ONNX converter.");
  }
  switch (nf.value()) {
    case pblczero::NetworkFormat::VALUE_CLASSICAL:
    case pblczero::NetworkFormat::VALUE_WDL:
      break;
    default:
      throw Exception("Value format " +
                      pblczero::NetworkFormat::ValueFormat_Name(nf.value()) +
                      " is not supported by the ONNX converter.");
  }
  switch (nf.default_activation()) {
    case pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU:
    case pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH:
      break;
    default:
      throw Exception("Default activation " +
                      pblczero::NetworkFormat::DefaultActivation_Name(
                          nf.default_activation()) +
                      " is not supported by the ONNX converter.");
  }
  switch (nf.input_embedding()) {
    case pblczero::NetworkFormat::INPUT_EMBEDDING_NONE:
    case pblczero::NetworkFormat::INPUT_EMBEDDING_PE_MAP:
    case pblczero::NetworkFormat::INPUT_EMBEDDING_PE_DENSE:
      break;
    default:
      throw Exception("Input embedding " +
                      pblczero::NetworkFormat::InputEmbeddingFormat_Name(
                          nf.input_embedding()) +
                      " is not supported by the ONNX converter.");
  }
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
  CheckSrcFormat(src_.format().network_format());

  CopyGenericFields(dst);
  GenerateOnnx(dst->mutable_onnx_model());
}

}  // namespace

WeightsToOnnxConverterOptions::DataType
WeightsToOnnxConverterOptions::StringToDataType(const std::string& s) {
  if (s == "f32") return DataType::kFloat32;
  if (s == "f16") return DataType::kFloat16;
  if (s == "bf16") return DataType::kBFloat16;
  throw Exception("Invalid data type: [" + s +
                  "]. Only f32, f16 and bf16 are supported.");
}

pblczero::Net ConvertWeightsToOnnx(
    const pblczero::Net& net, const WeightsToOnnxConverterOptions& options) {
  Converter converter(net, options);
  pblczero::Net dst;
  converter.Convert(&dst);
  return dst;
}

}  // namespace lczero
