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

#include "neural/loader.h"
#include "neural/onnx/converter.h"
#include "utils/files.h"
#include "utils/optionsparser.h"
#include "lc0ctl/describenet.h"

namespace lczero {
namespace {

const OptionId kInputFilenameId{"input", "InputFile",
                                "Path of the input Lc0 weights file."};
const OptionId kOutputFilenameId{"output", "OutputFile",
                                 "Path of the output ONNX file."};

const OptionId kInputPlanesName{"input-planes-name", "InputPlanesName",
                                "ONNX name to use for the input planes node."};
const OptionId kOutputPolicyHead{
    "policy-head-name", "PolicyHeadName",
    "ONNX name to use for the policy head output node."};
const OptionId kOutputWdl{"wdl-head-name", "WdlHeadName",
                          "ONNX name to use for the WDL head output node."};
const OptionId kOutputValue{
    "value-head-name", "ValueHeadName",
    "ONNX name to use for value policy head output node."};
const OptionId kOutputMlh{"mlh-head-name", "MlhHeadName",
                          "ONNX name to use for the MLH head output node."};

bool ProcessParameters(OptionsParser* options) {
  options->Add<StringOption>(kInputFilenameId);
  options->Add<StringOption>(kOutputFilenameId);

  options->Add<StringOption>(kInputPlanesName) = "/input/planes";
  options->Add<StringOption>(kOutputPolicyHead) = "/output/policy";
  options->Add<StringOption>(kOutputWdl) = "/output/wdl";
  options->Add<StringOption>(kOutputValue) = "/output/value";
  options->Add<StringOption>(kOutputMlh) = "/output/mlh";
  if (!options->ProcessAllFlags()) return false;

  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kInputFilenameId);
  dict.EnsureExists<std::string>(kOutputFilenameId);

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
    onnx_options.output_wdl = dict.Get<std::string>(kOutputWdl);
    weights_file = ConvertWeightsToOnnx(weights_file, onnx_options);
  }

  const auto& onnx = weights_file.onnx_model();
  WriteStringToFile(dict.Get<std::string>(kOutputFilenameId), onnx.model());
  ShowNetworkOnnxInfo(weights_file, false);
  COUT << "Done.";
}

}  // namespace lczero