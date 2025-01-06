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

#include "lc0ctl/describenet.h"

#include "neural/loader.h"
#include "neural/onnx/onnx.pb.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kWeightsFilenameId{"weights", "WeightsFile",
                                  "Path of the input Lc0 weights file.", 'w'};

bool ProcessParameters(OptionsParser* options) {
  options->Add<StringOption>(kWeightsFilenameId);
  if (!options->ProcessAllFlags()) return false;
  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kWeightsFilenameId);

  return true;
}

std::string Justify(std::string str, size_t length = 30) {
  if (str.size() + 2 < length) {
    str = std::string(length - 2 - str.size(), ' ') + str;
  }
  str += ": ";
  return str;
}

}  // namespace

void ShowNetworkGenericInfo(const pblczero::Net& weights) {
  const auto& version = weights.min_version();
  COUT << "\nGeneral";
  COUT << "~~~~~~~";
  COUT << Justify("Minimal Lc0 version") << "v" << version.major() << '.'
       << version.minor() << '.' << version.patch();
}

void ShowNetworkFormatInfo(const pblczero::Net& weights) {
  const auto& format = weights.format();
  const auto& net_format = format.network_format();

  using pblczero::Format;
  using pblczero::NetworkFormat;
  COUT << "\nFormat";
  COUT << "~~~~~~";
  if (format.has_weights_encoding()) {
    COUT << Justify("Weights encoding")
         << Format::Encoding_Name(format.weights_encoding());
  }
  if (net_format.has_input()) {
    COUT << Justify("Input")
         << NetworkFormat::InputFormat_Name(net_format.input());
  }
  if (net_format.has_network()) {
    COUT << Justify("Network")
         << NetworkFormat::NetworkStructure_Name(net_format.network());
  }
  if (net_format.has_policy()) {
    COUT << Justify("Policy")
         << NetworkFormat::PolicyFormat_Name(net_format.policy());
  }
  if (net_format.has_value()) {
    COUT << Justify("Value")
         << NetworkFormat::ValueFormat_Name(net_format.value());
  }
  if (net_format.has_moves_left()) {
    COUT << Justify("MLH")
         << NetworkFormat::MovesLeftFormat_Name(net_format.moves_left());
  }
  if (net_format.has_default_activation()) {
    COUT << Justify("Default activation")
         << NetworkFormat::DefaultActivation_Name(
                net_format.default_activation());
  }
  if (net_format.has_smolgen_activation()) {
    COUT << Justify("Smolgen activation")
         << NetworkFormat::ActivationFunction_Name(
                net_format.smolgen_activation());
  }
  if (net_format.has_ffn_activation()) {
    COUT << Justify("FFN activation")
         << NetworkFormat::ActivationFunction_Name(net_format.ffn_activation());
  }
}

void ShowNetworkTrainingInfo(const pblczero::Net& weights) {
  if (!weights.has_training_params()) return;
  COUT << "\nTraining Parameters";
  COUT << "~~~~~~~~~~~~~~~~~~~";
  using pblczero::TrainingParams;
  const auto& params = weights.training_params();
  if (params.has_training_steps()) {
    COUT << Justify("Training steps") << params.training_steps();
  }
  if (params.has_learning_rate()) {
    COUT << Justify("Learning rate") << params.learning_rate();
  }
  if (params.has_mse_loss()) {
    COUT << Justify("MSE loss") << params.mse_loss();
  }
  if (params.has_policy_loss()) {
    COUT << Justify("Policy loss") << params.policy_loss();
  }
  if (params.has_accuracy()) {
    COUT << Justify("Accuracy") << params.accuracy();
  }
  if (params.has_lc0_params()) {
    COUT << Justify("Lc0 Params") << params.lc0_params();
  }
}

void ShowNetworkWeightsBodyInfo(const pblczero::Net& weights) {
  const auto& w = weights.weights();
  if (w.encoder_size() > 0) {
    COUT << Justify("Encoders") << w.encoder_size();
    COUT << Justify("Encoder heads") << w.headcount();
    COUT << Justify("Embedding size") << w.ip_emb_b().params().size() / 2;
    COUT << Justify("Dmodel") << w.encoder(0).mha().q_b().params().size() / 2;
    COUT << Justify("Encoder DFF")
         << w.encoder(0).ffn().dense1_b().params().size() / 2;
  } else {
    COUT << Justify("Blocks") << w.residual_size();
    int se_count = 0;
    for (size_t i = 0; i < w.residual_size(); i++) {
      if (w.residual(i).has_se()) se_count++;
    }
    if (se_count > 0) {
      COUT << Justify("SE blocks") << se_count;
    }
    COUT << Justify("Filters")
         << w.input().weights().params().size() / 2 / 112 / 9;
  }
}

void ShowNetworkWeightsPolicyInfo(const pblczero::Net& weights) {
  using pblczero::NetworkFormat;
  const auto& w = weights.weights();
  const auto& format = weights.format().network_format();
  auto pol_activation = NetworkFormat::ACTIVATION_DEFAULT;
  if (format.policy() == NetworkFormat::POLICY_ATTENTION) {
    // Non-attentionbody nets use hardcoded SELU as policy activation and FFN
    // activations.
    auto ffn_activation = format.ffn_activation();
    if (w.encoder_size() == 0) {
      pol_activation = NetworkFormat::ACTIVATION_SELU;
      ffn_activation = NetworkFormat::ACTIVATION_SELU;
    }

    COUT << Justify("Policy") << "Attention";
    COUT << Justify("Policy activation")
         << NetworkFormat::ActivationFunction_Name(pol_activation);

    if (w.pol_encoder_size() > 0) {
      COUT << Justify("Policy encoders") << w.pol_encoder_size();
      COUT << Justify("Policy encoder heads") << w.pol_headcount();
      COUT << Justify("Policy encoder Dmodel")
           << w.pol_encoder(0).mha().q_b().params().size() / 2;
      COUT << Justify("Policy encoder DFF")
           << w.pol_encoder(0).ffn().dense1_b().params().size() / 2;
      COUT << Justify("Policy FFN activation")
           << NetworkFormat::ActivationFunction_Name(ffn_activation);
    }
    COUT << Justify("Policy Dmodel") << w.ip2_pol_b().params().size() / 2;
  } else {
    COUT << Justify("Policy") << (w.has_policy1() ? "Convolution" : "Dense");
    COUT << Justify("Policy activation")
         << NetworkFormat::ActivationFunction_Name(pol_activation);
    if (!w.has_policy1()) {
      int policy_channels = w.policy().biases().params().size() / 2;
      if (policy_channels == 0) {
        policy_channels = w.policy().bn_means().params().size() / 2;
      }
      COUT << Justify("Policy channels") << policy_channels;
    }
  }
}

void ShowNetworkWeightsInfo(const pblczero::Net& weights) {
  if (!weights.has_weights()) return;
  COUT << "\nWeights";
  COUT << "~~~~~~~";

  ShowNetworkWeightsBodyInfo(weights);
  ShowNetworkWeightsPolicyInfo(weights);

  const auto& w = weights.weights();
  COUT << Justify("Value")
       << (w.ip2_val_w().params().size() / 2 % 3 == 0 ? "WDL" : "Classical");
  COUT << Justify("MLH") << (w.has_ip2_mov_w() ? "Present" : "Absent");
}

void ShowNetworkOnnxInfo(const pblczero::Net& weights,
                         bool show_onnx_internals) {
  if (!weights.has_onnx_model()) return;
  const auto& onnx_model = weights.onnx_model();
  COUT << "\nONNX interface";
  COUT << "~~~~~~~~~~~~~~";
  if (onnx_model.has_data_type()) {
    COUT << Justify("Data type")
         << pblczero::OnnxModel::DataType_Name(onnx_model.data_type());
  }
  if (onnx_model.has_input_planes()) {
    COUT << Justify("Input planes") << onnx_model.input_planes();
  }
  if (onnx_model.has_output_value()) {
    COUT << Justify("Output value") << onnx_model.output_value();
  }
  if (onnx_model.has_output_wdl()) {
    COUT << Justify("Output WDL") << onnx_model.output_wdl();
  }
  if (onnx_model.has_output_policy()) {
    COUT << Justify("Output Policy") << onnx_model.output_policy();
  }
  if (onnx_model.has_output_mlh()) {
    COUT << Justify("Output MLH") << onnx_model.output_mlh();
  }

  if (!show_onnx_internals) return;
  if (!onnx_model.has_model()) return;

  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_model.model());
  COUT << "\nONNX model";
  COUT << "~~~~~~~~~~";

  if (onnx.has_ir_version()) {
    COUT << Justify("IR version") << onnx.ir_version();
  }
  if (onnx.has_producer_name()) {
    COUT << Justify("Producer Name") << onnx.producer_name();
  }
  if (onnx.has_producer_version()) {
    COUT << Justify("Producer Version") << onnx.producer_version();
  }
  if (onnx.has_domain()) {
    COUT << Justify("Domain") << onnx.domain();
  }
  if (onnx.has_model_version()) {
    COUT << Justify("Model Version") << onnx.model_version();
  }
  if (onnx.has_doc_string()) {
    COUT << Justify("Doc String") << onnx.doc_string();
  }
  for (const auto& input : onnx.graph().input()) {
    std::string name(input.name());
    if (input.has_doc_string()) {
      name += " (" + std::string(input.doc_string()) + ")";
    }
    COUT << Justify("Input") << name;
  }
  for (const auto& output : onnx.graph().output()) {
    std::string name(output.name());
    if (output.has_doc_string()) {
      name += " (" + std::string(output.doc_string()) + ")";
    }
    COUT << Justify("Output") << name;
  }
  for (const auto& opset : onnx.opset_import()) {
    std::string name;
    if (opset.has_domain()) name += std::string(opset.domain()) + " ";
    COUT << Justify("Opset") << name << opset.version();
  }
}

void ShowAllNetworkInfo(const pblczero::Net& weights) {
  ShowNetworkGenericInfo(weights);
  ShowNetworkFormatInfo(weights);
  ShowNetworkTrainingInfo(weights);
  ShowNetworkWeightsInfo(weights);
  ShowNetworkOnnxInfo(weights, true);
}

void DescribeNetworkCmd() {
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;

  const OptionsDict& dict = options_parser.GetOptionsDict();
  auto weights_file =
      LoadWeightsFromFile(dict.Get<std::string>(kWeightsFilenameId));
  ShowAllNetworkInfo(weights_file);
}
}  // namespace lczero
