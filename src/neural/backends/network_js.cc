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

#include <emscripten.h>
#include <memory>

#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"

namespace lczero {
namespace {

extern "C" {

EM_JS(int, lc0web_is_cpu, (int id), { return globalThis.lc0web_is_cpu(id) });
EM_JS(int, lc0web_computation, (int id), { return globalThis.lc0web_computation(id) });
EM_JS(float, lc0web_q_val, (int id, int sample), { return globalThis.lc0web_q_val(id, sample) });
EM_JS(float, lc0web_d_val, (int id, int sample), { return globalThis.lc0web_d_val(id, sample) });
EM_JS(float, lc0web_p_val, (int id, int sample, int move_id), { return globalThis.lc0web_p_val(id, sample, move_id) });
EM_JS(float, lc0web_m_val, (int id, int sample), { return globalThis.lc0web_m_val(id, sample) });
EM_JS(int, lc0web_remove, (int id), { return globalThis.lc0web_remove(id) });
EM_JS(void, lc0web_add_input, (int id), { return globalThis.lc0web_add_input(id) });
EM_JS(void, lc0web_add_plane, (int id, int index, uint64_t mask, float value), { return globalThis.lc0web_add_plane(id, index,  mask, value) });
EM_JS(int, lc0web_batch_size, (int id), { return globalThis.lc0web_batch_size(id) });
EM_ASYNC_JS(void, lc0web_compute, (int id), { return globalThis.lc0web_compute(id) });
EM_ASYNC_JS(int, lc0web_network, (const char *data, size_t length), { return globalThis.lc0web_network(data, length) });

}

class JSComputation : public NetworkComputation {
 public:
  JSComputation(int id);
  ~JSComputation() override;
  void AddInput(InputPlanes&& input) override;
  int GetBatchSize() const override;
  void ComputeBlocking() override;
  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;
 private:
  int id;
};

class JSNetwork : public Network {
 public:
  JSNetwork(const WeightsFile& file);
  ~JSNetwork() override;
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities;
  };
  std::unique_ptr<NetworkComputation> NewComputation() override;
  bool IsCpu() const override;
 private:
  int id;
  const NetworkCapabilities capabilities;
};

std::unique_ptr<Network> MakeJSNetwork(
  const std::optional<WeightsFile>& w,
  const OptionsDict& opts) {
  (void) opts;
  if (!w) {
    throw Exception("The JS backend requires a network file.");
  }
  auto weights = *w;
  if (!weights.has_onnx_model()) {
    WeightsToOnnxConverterOptions onnx_options;
    onnx_options.alt_mish = true;
    onnx_options.alt_selu = true;
    weights = ConvertWeightsToOnnx(weights, onnx_options);
  }
  return std::make_unique<JSNetwork>(weights);
}

bool JSNetwork::IsCpu() const {
  return lc0web_is_cpu(id);
}

std::unique_ptr<NetworkComputation> JSNetwork::NewComputation() {
  return std::make_unique<JSComputation>(id);
}

float JSComputation::GetQVal(int sample) const {
  return lc0web_q_val(id, sample);
}

float JSComputation::GetDVal(int sample) const {
  return lc0web_d_val(id, sample);
}

float JSComputation::GetPVal(int sample, int move_id) const {
  return lc0web_p_val(id, sample, move_id);
}

float JSComputation::GetMVal(int sample) const {
  return lc0web_m_val(id, sample);
}

void JSComputation::AddInput(InputPlanes&& input) {
  int i = GetBatchSize();
  lc0web_add_input(id);
  for (auto& plane : input)
    lc0web_add_plane(id, i, plane.mask, plane.value);
}

int JSComputation::GetBatchSize() const {
  return lc0web_batch_size(id);
}

void JSComputation::ComputeBlocking() {
  lc0web_compute(id);
}

JSComputation::JSComputation(int id2) {
  id = lc0web_computation(id2);
}

JSComputation::~JSComputation() {
  lc0web_remove(id);
}

JSNetwork::JSNetwork(const WeightsFile& file)
    : capabilities{file.format().network_format().input(),
                   file.format().network_format().output(),
                   file.format().network_format().moves_left()} {
  const auto& bytes = file.onnx_model().model();
  id = lc0web_network(bytes.data(), bytes.length());
}

JSNetwork::~JSNetwork() {
  lc0web_remove(id);
}

REGISTER_NETWORK("js", MakeJSNetwork, 1000)

}
}
