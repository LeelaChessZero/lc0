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
#include <memory>

#include "neural/factory.h"
#include "neural/onnx/converter.h"
#include "onnxruntime_cxx_api.h"

namespace lczero {
namespace {

std::unique_ptr<Network> MakeOnnxNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict&) {
  if (!w) throw Exception("The ONNX backend requires a network file.");

  // DO NOT SUBMIT  begin
  auto x = ConvertWeightsToOnnx(*w, {});
  std::ofstream fo1("/tmp/weights.weights");
  fo1 << x.OutputAsString();
  std::ofstream fo2("/tmp/onnx.onnx");
  fo2 << x.onnx_model().model();
  // DO NOT SUBMIT  end

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lc0");

  Ort::SessionOptions session_options;
  // session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  Ort::Session session(env, nullptr, 0, session_options);

  return nullptr;
}

REGISTER_NETWORK("onnx-cpu", MakeOnnxNetwork, 62)

}  // namespace
}  // namespace lczero