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

#include <zlib.h>

#include <algorithm>
#include <fstream>
#include <set>

#include "lc0ctl/describenet.h"
#include "neural/onnx/onnx.pb.h"
#include "proto/net.pb.h"
#include "utils/files.h"
#include "utils/fp16_utils.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

template <typename T, size_t N>
std::vector<std::string> GetAllEnumValues(const std::array<T, N>& vals,
                                          std::string (*func)(T)) {
  std::vector<std::string> res;
  std::transform(vals.begin(), vals.end(), std::back_inserter(res), func);
  return res;
}

template <typename T, size_t N>
T GetEnumValueFromString(const std::string& str_value,
                         const std::array<T, N>& vals, std::string (*func)(T)) {
  auto iter = std::find_if(vals.begin(), vals.end(),
                           [&](T val) { return func(val) == str_value; });
  if (iter == vals.end()) {
    throw Exception("Enum value " + str_value + " is unknown.");
  }
  return *iter;
}

const OptionId kInputFilenameId{"input", "InputFile",
                                "Path of the input Lc0 weights file."};
const OptionId kOutputFilenameId{"output", "OutputFile",
                                 "Path of the output ONNX file."};

const OptionId kInputFormatId(
    "input-format", "InputFormat",
    "Format in which the neural network expects input to be.");
const OptionId kPolicyFormatId(
    "policy-format", "PolicyFormat",
    "Format of the policy head output. Currently the search code does not "
    "distinguish between POLICY_CLASSICAL and POLICY_CONVOLUTION, but maybe "
    "one day for new types it will have new values.");
const OptionId kValueFormatId(
    "value-format", "ValueFormat",
    "Format of the value head output. Currently the search code does not "
    "distinguish between VALUE_CLASSICAL and VALUE_WDL, but maybe one day for "
    "new types it will have new values.");
const OptionId kMovesLeftFormatId("moves-left-format", "MovesLeftFormat",
                                  "Format of the moves left head output.");

// ONNX options.
const OptionId kOnnxInputId{"onnx-input", "OnnxInput",
                            "The name of the input ONNX node."};
const OptionId kOnnxOutputValueId{
    "onnx-output-value", "OnnxOutputValue",
    "The name of the node for a classical value head."};
const OptionId kOnnxOutputWdlId{"onnx-output-wdl", "OnnxOutputWdl",
                                "The name of the node for a wdl value head."};
const OptionId kOnnxOutputPolicyId{"onnx-output-policy", "OnnxOutputPolicy",
                                   "The name of the node for a policy head."};
const OptionId kOnnxOutputMlhId{"onnx-output-mlh", "OnnxOutputMlh",
                                "The name of the node for a moves left head."};

const OptionId kValidateModelId{"validate-weights", "ValidateWeights",
                                "Do a basic check of the provided ONNX file."};
const OptionId kFixRule50Id{
    "fix-rule50", "",
    "Fix tensorflow exported onnx that needs rule50 input scaling."};
const OptionId kFixWdlSoftmaxId{
    "fix-wdl-softmax", "",
    "Fix tensorflow exported onnx that is missing wdl output softmax."};

bool ProcessParameters(OptionsParser* options) {
  using pblczero::NetworkFormat;
  using pblczero::OnnxModel;
  options->Add<StringOption>(kInputFilenameId);
  options->Add<StringOption>(kOutputFilenameId);
  // Data format options.
  options->Add<ChoiceOption>(
      kInputFormatId, GetAllEnumValues(NetworkFormat::InputFormat_AllValues,
                                       NetworkFormat::InputFormat_Name)) =
      NetworkFormat::InputFormat_Name(NetworkFormat::INPUT_CLASSICAL_112_PLANE);
  options->Add<ChoiceOption>(
      kPolicyFormatId, GetAllEnumValues(NetworkFormat::PolicyFormat_AllValues,
                                        NetworkFormat::PolicyFormat_Name)) =
      NetworkFormat::PolicyFormat_Name(NetworkFormat::POLICY_CLASSICAL);
  options->Add<ChoiceOption>(
      kValueFormatId, GetAllEnumValues(NetworkFormat::ValueFormat_AllValues,
                                       NetworkFormat::ValueFormat_Name)) =
      NetworkFormat::ValueFormat_Name(NetworkFormat::VALUE_WDL);
  options->Add<ChoiceOption>(
      kMovesLeftFormatId,
      GetAllEnumValues(NetworkFormat::MovesLeftFormat_AllValues,
                       NetworkFormat::MovesLeftFormat_Name)) =
      NetworkFormat::MovesLeftFormat_Name(NetworkFormat::MOVES_LEFT_V1);
  // Onnx options.
  options->Add<StringOption>(kOnnxInputId);
  options->Add<StringOption>(kOnnxOutputPolicyId);
  options->Add<StringOption>(kOnnxOutputValueId);
  options->Add<StringOption>(kOnnxOutputWdlId);
  options->Add<StringOption>(kOnnxOutputMlhId);

  options->Add<BoolOption>(kValidateModelId) = true;
  options->Add<BoolOption>(kFixRule50Id) = false;
  options->Add<BoolOption>(kFixWdlSoftmaxId) = false;

  if (!options->ProcessAllFlags()) return false;

  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kInputFilenameId);
  dict.EnsureExists<std::string>(kOutputFilenameId);
  return true;
}

bool ValidateNetwork(const pblczero::Net& weights, pblczero::ModelProto& onnx) {
  const auto& onnx_model = weights.onnx_model();

  if (!onnx.has_ir_version()) {
    CERR << "ONNX file doesn't appear to have version specified. Likely not an "
            "ONNX file.";
    return false;
  }
  const auto& onnx_inputs = onnx.graph().input();
  std::set<std::string> inputs;
  std::transform(onnx_inputs.begin(), onnx_inputs.end(),
                 std::inserter(inputs, inputs.end()),
                 [](const auto& x) { return std::string(x.name()); });

  const auto& onnx_outputs = onnx.graph().output();
  std::set<std::string> outputs;
  std::transform(onnx_outputs.begin(), onnx_outputs.end(),
                 std::inserter(outputs, outputs.end()),
                 [](const auto& x) { return std::string(x.name()); });

  auto check_exists = [](std::string_view n, std::set<std::string>* nodes) {
    std::string name(n);
    if (nodes->count(name) == 0) {
      CERR << "Node '" << name << "' doesn't exist in ONNX.";
      return false;
    }
    nodes->erase(name);
    return true;
  };

  if (onnx_model.has_input_planes() &&
      !check_exists(onnx_model.input_planes(), &inputs)) {
    return false;
  }
  if (onnx_model.has_output_value() &&
      !check_exists(onnx_model.output_value(), &outputs)) {
    return false;
  }
  if (onnx_model.has_output_wdl() &&
      !check_exists(onnx_model.output_wdl(), &outputs)) {
    return false;
  }
  if (onnx_model.has_output_policy() &&
      !check_exists(onnx_model.output_policy(), &outputs)) {
    return false;
  }
  if (onnx_model.has_output_mlh() &&
      !check_exists(onnx_model.output_mlh(), &outputs)) {
    return false;
  }
  for (const auto& input : inputs) {
    CERR << "Warning: ONNX input node '" << input << "' not used.";
  }
  for (const auto& output : outputs) {
    CERR << "Warning: ONNX output node '" << output << "' not used.";
  }

  if (!onnx_model.has_input_planes()) {
    CERR << "The --" << kOnnxInputId.long_flag()
         << " must be defined. Typical value for the ONNX networks exported "
            "from Leela is /input/planes.";
    return false;
  }
  if (!onnx_model.has_output_policy()) {
    CERR << "The --" << kOnnxOutputPolicyId.long_flag()
         << " must be defined. Typical value for the ONNX networks exported "
            "from Leela is /output/policy.";
    return false;
  }
  if (!onnx_model.has_output_value() && !onnx_model.has_output_wdl()) {
    CERR << "Either --" << kOnnxOutputValueId.long_flag() << " or --"
         << kOnnxOutputWdlId.long_flag()
         << " must be defined. Typical values for the ONNX networks exported "
            "from Leela are /output/value and /output/wdl.";
    return false;
  }
  if (onnx_model.has_output_value() && onnx_model.has_output_wdl()) {
    CERR << "Both --" << kOnnxOutputValueId.long_flag() << " and --"
         << kOnnxOutputWdlId.long_flag()
         << " are set. Only one of them has to be set.";
    return false;
  }

  return true;
}

void FixRule50(pblczero::ModelProto& model, const std::string& in, bool fp16) {
  std::string name = "rule50fix";

  for (size_t i = 0; i < model.graph().node_size(); i++) {
    auto node = model.graph().node(i);
    for (size_t j = 0; j < node.input_size(); j++) {
      if (node.input(j) == in) {
        CERR << "Inerting scaling between " << in << " and " << node.name();
        model.mutable_graph()->mutable_node(i)->mutable_input()->at(j) =
            std::string(name);
      }
    }
  }

  auto* init = model.mutable_graph()->add_initializer();
  init->set_name(name + "_weights");
  init->add_dims(112);
  init->add_dims(1);
  init->add_dims(1);
  if (fp16) {
    init->set_data_type(pblczero::TensorProto::FLOAT16);
    std::vector<uint16_t> rule50weights(112, FP32toFP16(1.0f));
    rule50weights[109] = FP32toFP16(1.0f / 99);
    init->set_raw_data(
        std::string(reinterpret_cast<const char*>(rule50weights.data()),
                    rule50weights.size() * sizeof(uint16_t)));
  } else {
    init->set_data_type(pblczero::TensorProto::FLOAT);
    std::vector<float> rule50weights(112, 1.0f);
    rule50weights[109] = 1.0f / 99;
    init->set_raw_data(
        std::string(reinterpret_cast<const char*>(rule50weights.data()),
                    rule50weights.size() * sizeof(float)));
  }
  auto* new_node = model.mutable_graph()->add_node();
  new_node->set_name(name);
  new_node->set_op_type("Mul");
  new_node->add_input(in);
  new_node->add_output(name);

  new_node->add_input(name + "_weights");
}

void FixWdlSoftmax(pblczero::ModelProto& model, const std::string& out) {
  std::string name = "softmax_fix";

  for (size_t i = 0; i < model.graph().node_size(); i++) {
    auto node = model.graph().node(i);
    for (size_t j = 0; j < node.output_size(); j++) {
      if (node.output(j) == out) {
        CERR << "Inserting softmax between " << node.name() << " and " << out;
        model.mutable_graph()->mutable_node(i)->mutable_output()->at(j) =
            std::string(name);
        break;
      }
    }
  }

  auto* new_node = model.mutable_graph()->add_node();
  new_node->set_name(name);
  new_node->set_op_type("Softmax");
  new_node->add_input(name);
  new_node->add_output(out);
}

pblczero::OnnxModel_DataType GetDataType(pblczero::ModelProto& model,
                                         const std::string& name) {
  using pblczero::TensorProto;
  using pblczero::OnnxModel;
  for (auto& in : model.graph().input()) {
    if (in.name() == name && in.has_type() && in.type().has_tensor_type() &&
        in.type().tensor_type().has_elem_type()) {
      auto data_type = in.type().tensor_type().elem_type();
      switch (data_type) {
        case TensorProto::FLOAT:
          return OnnxModel::FLOAT;
        case TensorProto::FLOAT16:
          return OnnxModel::FLOAT16;
        default:
          throw Exception("Unsupported data type: " +
                          TensorProto::DataType_Name(data_type));
      }
    }
  }
  return OnnxModel::FLOAT;
}

bool EnsureOutDataType(pblczero::ModelProto& model, const std::string& name,
                       pblczero::OnnxModel_DataType data_type) {
  // Check if output has the correct data type and set it if not.
  for (size_t i = 0; i < model.graph().output_size(); i++) {
    auto out = model.graph().output(i);
    if (out.name() == name) {
      if (!out.has_type()) {
        model.mutable_graph()->mutable_output(i)->mutable_type();
      }
      if (!out.type().has_tensor_type()) {
        model.mutable_graph()
            ->mutable_output(i)
            ->mutable_type()
            ->mutable_tensor_type();
      }
      if (!out.type().tensor_type().has_elem_type() ||
          out.type().tensor_type().elem_type() !=
              static_cast<pblczero::TensorProto_DataType>(data_type)) {
        model.mutable_graph()
            ->mutable_output(i)
            ->mutable_type()
            ->mutable_tensor_type()
            ->set_elem_type(
                static_cast<pblczero::TensorProto_DataType>(data_type));
        break;
      }
      return false;
    }
  }

  // Insert a cast to the correct data type.
  for (size_t i = 0; i < model.graph().node_size(); i++) {
    auto node = model.graph().node(i);
    for (size_t j = 0; j < node.output_size(); j++) {
      if (node.output(j) == name) {
        CERR << "Inserting cast between " << node.name() << " and " << name;
        model.mutable_graph()->mutable_node(i)->mutable_output()->at(j) =
            std::string(name + "/cast");
        break;
      }
    }
  }

  auto* new_node = model.mutable_graph()->add_node();
  new_node->set_name(name + "/cast");
  new_node->set_op_type("Cast");
  new_node->add_input(name + "/cast");
  new_node->add_output(name);
  auto* attr = new_node->add_attribute();
  attr->set_name("to");
  attr->set_type(pblczero::AttributeProto::INT);
  attr->set_i(data_type);
  return true;
}

bool MaybeFixOnnx(pblczero::ModelProto& model, const OptionsDict& dict,
                  pblczero::OnnxModel_DataType data_type) {
  bool updated = false;

  // Input.
  if (dict.OwnExists<std::string>(kOnnxInputId)) {
    if (dict.Get<bool>(kFixRule50Id)) {
      FixRule50(model, dict.Get<std::string>(kOnnxInputId),
                data_type == pblczero::OnnxModel::FLOAT16);
      updated = true;
    }
  }

  // Policy.
  if (dict.OwnExists<std::string>(kOnnxOutputPolicyId)) {
    updated |= EnsureOutDataType(
        model, dict.Get<std::string>(kOnnxOutputPolicyId), data_type);
  }

  // Value.
  if (dict.OwnExists<std::string>(kOnnxOutputValueId)) {
    updated |= EnsureOutDataType(
        model, dict.Get<std::string>(kOnnxOutputValueId), data_type);
  }
  if (dict.OwnExists<std::string>(kOnnxOutputWdlId)) {
    auto out = dict.Get<std::string>(kOnnxOutputWdlId);
    if (dict.Get<bool>(kFixWdlSoftmaxId)) {
      FixWdlSoftmax(model, out);
      updated = true;
    }
    updated |= EnsureOutDataType(model, out, data_type);
  }

  // Mlh.
  if (dict.OwnExists<std::string>(kOnnxOutputMlhId)) {
    updated |= EnsureOutDataType(model, dict.Get<std::string>(kOnnxOutputMlhId),
                                 data_type);
  }

  return updated;
}

}  // namespace

void ConvertOnnxToLeela() {
  using pblczero::NetworkFormat;
  using pblczero::OnnxModel;
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;

  const OptionsDict& dict = options_parser.GetOptionsDict();

  auto onnx_model = ReadFileToString(dict.Get<std::string>(kInputFilenameId));
  pblczero::ModelProto model;
  model.ParseFromString(onnx_model);

  pblczero::Net out_weights;
  out_weights.set_magic(0x1c0);
  // ONNX networks appeared in v0.28.
  out_weights.mutable_min_version()->set_major(0);
  out_weights.mutable_min_version()->set_minor(28);

  auto format = out_weights.mutable_format()->mutable_network_format();
  format->set_network(NetworkFormat::NETWORK_ONNX);
  auto onnx = out_weights.mutable_onnx_model();
  auto data_type = OnnxModel::FLOAT;

  // Input.
  format->set_input(GetEnumValueFromString(
      dict.Get<std::string>(kInputFormatId),
      NetworkFormat::InputFormat_AllValues, NetworkFormat::InputFormat_Name));
  if (dict.OwnExists<std::string>(kOnnxInputId)) {
    auto in = dict.Get<std::string>(kOnnxInputId);
    onnx->set_input_planes(in);
    data_type = GetDataType(model, in);
  }
  onnx->set_data_type(data_type);

  // Policy.
  format->set_policy(GetEnumValueFromString(
      dict.Get<std::string>(kPolicyFormatId),
      NetworkFormat::PolicyFormat_AllValues, NetworkFormat::PolicyFormat_Name));
  if (dict.OwnExists<std::string>(kOnnxOutputPolicyId)) {
    onnx->set_output_policy(dict.Get<std::string>(kOnnxOutputPolicyId));
  }

  // Value.
  format->set_value(GetEnumValueFromString(
      dict.Get<std::string>(kValueFormatId),
      NetworkFormat::ValueFormat_AllValues, NetworkFormat::ValueFormat_Name));
  if (dict.OwnExists<std::string>(kOnnxOutputValueId)) {
    onnx->set_output_value(dict.Get<std::string>(kOnnxOutputValueId));
  }
  if (dict.OwnExists<std::string>(kOnnxOutputWdlId)) {
    onnx->set_output_wdl(dict.Get<std::string>(kOnnxOutputWdlId));
  }

  // Mlh.
  if (dict.OwnExists<std::string>(kOnnxOutputMlhId)) {
    format->set_moves_left(
        GetEnumValueFromString(dict.Get<std::string>(kMovesLeftFormatId),
                               NetworkFormat::MovesLeftFormat_AllValues,
                               NetworkFormat::MovesLeftFormat_Name));
    onnx->set_output_mlh(dict.Get<std::string>(kOnnxOutputMlhId));
  }

  if (MaybeFixOnnx(model, dict, data_type)) {
    onnx->set_model(model.OutputAsString());
  } else {
    onnx->set_model(onnx_model);
  }
  if (dict.Get<bool>(kValidateModelId) &&
      !ValidateNetwork(out_weights, model)) {
    return;
  }
  WriteStringToGzFile(dict.Get<std::string>(kOutputFilenameId),
                      out_weights.OutputAsString());
  ShowNetworkFormatInfo(out_weights);
  ShowNetworkOnnxInfo(out_weights, dict.Get<bool>(kValidateModelId));
  COUT << "Done.";
}

}  // namespace lczero
