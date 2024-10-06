#include <emscripten.h>
#include <memory>

#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"

namespace lczero {
namespace {

extern "C" {

EM_JS(int, lc0web_is_cpu, (), { return globalThis.lc0web_is_cpu() });
EM_JS(int, lc0web_id, (), { return globalThis.lc0web_id() });
EM_JS(int, lc0web_q_val, (int id, int sample), { return globalThis.lc0web_q_val(id, sample) });
EM_JS(int, lc0web_d_val, (int id, int sample), { return globalThis.lc0web_d_val(id, sample) });
EM_JS(int, lc0web_p_val, (int id, int sample, int move_id), { return globalThis.lc0web_p_val(id, sample, move_id) });
EM_JS(int, lc0web_m_val, (int id, int sample), { return globalThis.lc0web_m_val(id, sample) });
EM_JS(int, lc0web_remove, (int id), { return globalThis.lc0web_remove(id) });
EM_JS(void, lc0web_add_input, (int id), { return globalThis.lc0web_add_input(id) });
EM_JS(void, lc0web_add_plane, (int id, int index, uint64_t mask, float value), { return globalThis.lc0web_add_plane(id, index,  mask, value) });
EM_JS(int, lc0web_batch_size, (int id), { return globalThis.lc0web_batch_size(id) });
EM_ASYNC_JS(void, lc0web_compute, (int id), { return globalThis.lc0web_compute(id) });

}

class JSComputation : public NetworkComputation {
 public:
  JSComputation();
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
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities;
  };
  std::unique_ptr<NetworkComputation> NewComputation() override;
  bool IsCpu() const override;
 private:
  const NetworkCapabilities capabilities = {
    pblczero::NetworkFormat_InputFormat_INPUT_CLASSICAL_112_PLANE,
    pblczero::NetworkFormat_OutputFormat_OUTPUT_WDL,
    pblczero::NetworkFormat_MovesLeftFormat_MOVES_LEFT_V1,
  };
};

std::unique_ptr<Network> MakeJSNetwork(
  const std::optional<WeightsFile>& w,
  const OptionsDict& opts) {
  (void) w;
  (void) opts;
  return std::make_unique<JSNetwork>();
}

bool JSNetwork::IsCpu() const {
  return lc0web_is_cpu();
}

std::unique_ptr<NetworkComputation> JSNetwork::NewComputation() {
  return std::make_unique<JSComputation>();
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

JSComputation::JSComputation() {
  id = lc0web_id();
}

JSComputation::~JSComputation() {
  lc0web_remove(id);
}

REGISTER_NETWORK("js", MakeJSNetwork, 1000)

}
}
