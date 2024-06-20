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

#include <fstream>
#include <iostream>

#include "lc0ctl/describenet.h"
#include "neural/loader.h"
#include "neural/onnx/converter.h"
#include "neural/xla/onnx2hlo.h"
#include "neural/xla/print_hlo.h"
#include "utils/files.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kInputFilenameId{"input", "",
                                "Path of the input Lc0 weights file."};
const OptionId kOutputFilenameId{"output", "", "Path of the output ONNX file."};
const OptionId kHloTextOutputFilenameId = {"hlo-text-output", "",
                                           "Path of the output HLO file."};
const OptionId kHloProtoOutputFilenameId = {
    "hlo-proto-output", "", "Path of the output HLO proto file."};
const OptionId kOnnxBatchSizeId{"onnx-batch-size", "",
                                "Batch size to use for ONNX conversion."};
const OptionId kHloBatchSizeId{"hlo-batch-size", "",
                               "Batch size to use for HLO conversion."};
const OptionId kOnnxDataTypeId{"onnx-data-type", "",
                               "Data type to use in the ONNX model."};
const OptionId kOnnxOpsetId{"onnx-opset", "",
                            "Opset to use in the ONNX model."};
const OptionId kHloAllowPartialResultId = {
    "hlo-allow-partial-result", "",
    "Allow partial result in case of HLO conversion failure (DEBUG ONLY!)."};
const OptionId kRelaxOpTypes{
    "relax-op-types", "", "Use onnx-data-type even if unsuported by operator."};

const OptionId kInputPlanesName{"input-planes-name", "",
                                "ONNX name to use for the input planes node."};
const OptionId kOutputPolicyHead{
    "policy-head-name", "",
    "ONNX name to use for the policy head output node."};
const OptionId kOutputWdl{"wdl-head-name", "WdlHeadName",
                          "ONNX name to use for the WDL head output node."};
const OptionId kOutputValue{
    "value-head-name", "",
    "ONNX name to use for value policy head output node."};
const OptionId kOutputMlh{"mlh-head-name", "MlhHeadName",
                          "ONNX name to use for the MLH head output node."};
const OptionId kOnnxToPytorch{
    "onnx2pytorch", "",
    "Only use layer definitions supported by onnx2pytorch."};
const OptionId kValueHead{
    "value-head", "",
    "Value head to be used in the generated model. Typical values are "
    "'winner', 'q' or 'st', but only 'winner' is always available."};
const OptionId kPolicyHead{
    "policy-head", "",
    "Policy head to be used in the generated model. Typical values are "
    "'vanilla', 'optimistic' or 'soft', but only 'vanilla' is always "
    "available."};

bool ProcessParameters(OptionsParser* options) {
  options->Add<StringOption>(kInputFilenameId);
  options->Add<StringOption>(kOutputFilenameId);
  options->Add<StringOption>(kHloTextOutputFilenameId);
  options->Add<StringOption>(kHloProtoOutputFilenameId);
  options->Add<IntOption>(kOnnxBatchSizeId, -1, 2048) = -1;
  options->Add<IntOption>(kOnnxOpsetId, 7, 18) = 17;
  options->Add<IntOption>(kHloBatchSizeId, 1, 2048) = 333;
  options->Add<ChoiceOption>(
      kOnnxDataTypeId,
      std::vector<std::string>{"f32", "f16", "bf16"}) = "f32";
  options->Add<BoolOption>(kHloAllowPartialResultId);
  options->Add<BoolOption>(kRelaxOpTypes) = false;
  options->HideOption(kOnnxBatchSizeId);
  options->HideOption(kHloAllowPartialResultId);
  options->HideOption(kRelaxOpTypes);

  options->Add<StringOption>(kInputPlanesName) = "/input/planes";
  options->Add<StringOption>(kOutputPolicyHead) = "/output/policy";
  options->Add<StringOption>(kOutputWdl) = "/output/wdl";
  options->Add<StringOption>(kOutputValue) = "/output/value";
  options->Add<StringOption>(kOutputMlh) = "/output/mlh";
  options->Add<BoolOption>(kOnnxToPytorch) = false;
  options->Add<StringOption>(kValueHead) = "winner";
  options->Add<StringOption>(kPolicyHead) = "vanilla";
  if (!options->ProcessAllFlags()) return false;

  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kInputFilenameId);
  if (!dict.OwnExists<std::string>(kOutputFilenameId) &&
      !dict.OwnExists<std::string>(kHloTextOutputFilenameId) &&
      !dict.OwnExists<std::string>(kHloProtoOutputFilenameId)) {
    throw Exception(
        "At least one of --output, --hlo-output or --hlo-proto-output "
        "must be specified.");
  }
  return true;
}

}  // namespace

void ConvertLeelaToOnnx() {
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;

  const OptionsDict& dict = options_parser.GetOptionsDict();
  auto weights_file =
      LoadWeightsFromFile(dict.Get<std::string>(kInputFilenameId));

  ShowNetworkFormatInfo(weights_file);
  if (weights_file.has_onnx_model()) {
    COUT << "The leela network already has ONNX network embedded, extracting.";
  } else {
    ShowNetworkWeightsInfo(weights_file);
    COUT << "Converting Leela network to the ONNX.";
    WeightsToOnnxConverterOptions onnx_options;
    onnx_options.input_planes_name = dict.Get<std::string>(kInputPlanesName);
    onnx_options.output_policy_head = dict.Get<std::string>(kOutputPolicyHead);
    onnx_options.output_wdl = dict.Get<std::string>(kOutputWdl);
    onnx_options.output_value = dict.Get<std::string>(kOutputValue);
    onnx_options.opset = dict.Get<int>(kOnnxOpsetId);
    onnx_options.batch_size = dict.Get<int>(kOnnxBatchSizeId);
    onnx_options.data_type = WeightsToOnnxConverterOptions::StringToDataType(
        dict.Get<std::string>(kOnnxDataTypeId));
    onnx_options.relax_op_types = dict.Get<bool>(kRelaxOpTypes);
    // onnx2pytorch only needs an alternate layernorm-implementation, so it's
    // currently only enables that. Might need to be extended in the future.
    onnx_options.alt_layernorm = dict.Get<bool>(kOnnxToPytorch);
    onnx_options.value_head = dict.Get<std::string>(kValueHead);
    onnx_options.policy_head = dict.Get<std::string>(kPolicyHead);
    weights_file = ConvertWeightsToOnnx(weights_file, onnx_options);
  }

  const auto& onnx = weights_file.onnx_model();
  if (dict.OwnExists<std::string>(kOutputFilenameId)) {
    WriteStringToFile(dict.Get<std::string>(kOutputFilenameId), onnx.model());
  }
  if (dict.OwnExists<std::string>(kHloTextOutputFilenameId) ||
      dict.OwnExists<std::string>(kHloProtoOutputFilenameId)) {
    Onnx2HloOptions hlo_options;
    hlo_options.debugging_allow_partial_result =
        dict.Get<bool>(kHloAllowPartialResultId);
    pblczero::ModelProto onnx_model;
    onnx_model.ParseFromString(onnx.model());
    auto hlo_result = ConvertOnnxToHlo(
        onnx_model, dict.Get<int>(kHloBatchSizeId), hlo_options);
    if (dict.OwnExists<std::string>(kHloTextOutputFilenameId)) {
      std::string filename = dict.Get<std::string>(kHloTextOutputFilenameId);
      if (filename == "-") {
        PrettyPrintHlo(hlo_result.hlo_module, {}, std::cout);
      } else {
        std::ofstream file(filename.c_str());
        PrettyPrintHlo(hlo_result.hlo_module, {}, file);
      }
    }
    if (dict.OwnExists<std::string>(kHloProtoOutputFilenameId)) {
      WriteStringToFile(dict.Get<std::string>(kHloProtoOutputFilenameId),
                        hlo_result.hlo_module.OutputAsString());
    }
  }
  ShowNetworkOnnxInfo(weights_file, false);
  COUT << "Done.";
}

}  // namespace lczero
