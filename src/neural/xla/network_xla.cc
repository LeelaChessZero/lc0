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

#include "neural/factory.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "neural/xla/onnx2hlo.h"
#include "neural/xla/xla_runner.h"

namespace lczero {
namespace {

class XlaComputation : public NetworkComputation {
 public:
  void AddInput(InputPlanes&& input) override {}
  int GetBatchSize() const override { return 0; }
  void ComputeBlocking() override {}
  float GetQVal(int sample) const override { return 0.0f; }
  float GetDVal(int sample) const override { return 0.0f; }
  float GetPVal(int sample, int move_id) const override { return 0.0f; }
  float GetMVal(int sample) const override { return 0.0f; }
};

class XlaNetwork : public Network {
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<XlaComputation>();
  }

 private:
  NetworkCapabilities capabilities_;
};

std::unique_ptr<Network> MakeXlaNetwork(const std::optional<WeightsFile>& w,
                                        const OptionsDict&) {
  if (!w) throw Exception("The XLA backend requires a network file.");
  auto runner = std::make_unique<XlaRunner>();
  if (w->has_onnx_model()) {
    LoadOnnxIntoXlaRunner(runner.get(), w->onnx_model().model());
  } else {
    CERR << "Converting weights to ONNX first.";
    WeightsToOnnxConverterOptions converter_options;
    auto converted = ConvertWeightsToOnnx(*w, converter_options);
    LoadOnnxIntoXlaRunner(runner.get(), converted.onnx_model().model());
  }

  return std::make_unique<XlaNetwork>();
}

REGISTER_NETWORK("xla", MakeXlaNetwork, 143)

}  // namespace
}  // namespace lczero
