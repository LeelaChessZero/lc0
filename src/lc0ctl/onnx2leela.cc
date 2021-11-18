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
const OptionId kOnnxDataTypeId("onnx-data-type", "OnnxDataType",
                               "Data type to feed into the neural network.");
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
  options->Add<ChoiceOption>(kOnnxDataTypeId,
                             GetAllEnumValues(OnnxModel::DataType_AllValues,
                                              OnnxModel::DataType_Name)) =
      OnnxModel::DataType_Name(OnnxModel::FLOAT);
  options->Add<StringOption>(kOnnxInputId);
  options->Add<StringOption>(kOnnxOutputPolicyId);
  options->Add<StringOption>(kOnnxOutputValueId);
  options->Add<StringOption>(kOnnxOutputWdlId);
  options->Add<StringOption>(kOnnxOutputMlhId);

  options->Add<BoolOption>(kValidateModelId) = true;

  if (!options->ProcessAllFlags()) return false;

  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kInputFilenameId);
  dict.EnsureExists<std::string>(kOutputFilenameId);
  return true;
}

bool ValidateNetwork(const pblczero::Net& weights) {
  const auto& onnx_model = weights.onnx_model();
  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_model.model());

  if (!onnx.has_ir_version()) {
    CERR << "ONNX file doesn't appear to have version specified. Likely not an "
            "ONNX file.";
    return false;
  }
  if (!onnx.has_domain()) {
    CERR << "ONNX file doesn't appear to have domain specified. Likely not an "
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

}  // namespace

void ConvertOnnxToLeela() {
  using pblczero::NetworkFormat;
  using pblczero::OnnxModel;
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;

  const OptionsDict& dict = options_parser.GetOptionsDict();

  pblczero::Net out_weights;
  out_weights.set_magic(0x1c0);
  // ONNX networks appeared in v0.28.
  out_weights.mutable_min_version()->set_major(0);
  out_weights.mutable_min_version()->set_minor(28);

  auto format = out_weights.mutable_format()->mutable_network_format();
  format->set_network(NetworkFormat::NETWORK_ONNX);
  auto onnx = out_weights.mutable_onnx_model();
  onnx->set_data_type(GetEnumValueFromString(
      dict.Get<std::string>(kOnnxDataTypeId), OnnxModel::DataType_AllValues,
      OnnxModel::DataType_Name));

  // Input.
  format->set_input(GetEnumValueFromString(
      dict.Get<std::string>(kInputFormatId),
      NetworkFormat::InputFormat_AllValues, NetworkFormat::InputFormat_Name));
  if (dict.OwnExists<std::string>(kOnnxInputId)) {
    onnx->set_input_planes(dict.Get<std::string>(kOnnxInputId));
  }

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

  onnx->set_model(ReadFileToString(dict.Get<std::string>(kInputFilenameId)));
  if (dict.Get<bool>(kValidateModelId) && !ValidateNetwork(out_weights)) {
    return;
  }
  WriteStringToGzFile(dict.Get<std::string>(kOutputFilenameId),
                      out_weights.OutputAsString());
  ShowNetworkFormatInfo(out_weights);
  ShowNetworkOnnxInfo(out_weights, dict.Get<bool>(kValidateModelId));
  COUT << "Done.";
}

}  // namespace lczero