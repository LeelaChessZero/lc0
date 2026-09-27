/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

// --backend=proxy: evaluates batches in a separate `lc0 backendserver`
// process. Options: name=<server name>, timeout=<ms, 0 waits forever>,
// restart-wait=<ms to wait for a crashed server to be restarted>.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "neural/backends/proxy/ipc.h"
#include "neural/backends/proxy/protocol.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace proxy {
namespace {

constexpr int kPollMs = 100;

class ProxyBackend : public Backend {
 public:
  explicit ProxyBackend(const OptionsDict& options) {
    OptionsDict proxy_options;
    proxy_options.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));
    name_ = proxy_options.GetOrDefault<std::string>("name", "default");
    timeout_ms_ = proxy_options.GetOrDefault<int>("timeout", 0);
    restart_wait_ms_ = proxy_options.GetOrDefault<int>("restart-wait", 30000);

    shm_ = std::make_unique<SharedMemory>(SharedMemory::Open(name_));
    header_ = static_cast<ServerHeader*>(shm_->data());
    if (shm_->size() < sizeof(ServerHeader) ||
        header_->ready.load(std::memory_order_acquire) == 0) {
      throw Exception("Backend server " + name_ + " is not ready");
    }
    if (header_->magic != kMagic || header_->version != kVersion ||
        header_->position_size != sizeof(Position) ||
        shm_->size() < RegionSize(header_->num_slots, header_->max_batch)) {
      throw Exception("Backend server " + name_ +
                      " was built from a different lc0 version");
    }
    for (uint32_t i = 0; i < header_->num_slots; ++i) {
      requests_.push_back(NamedSemaphore::Open(RequestName(name_, i)));
      responses_.push_back(NamedSemaphore::Open(ResponseName(name_, i)));
    }
    pid_ = CurrentProcessId();
    CERR << "Connected to backend server " << name_ << " (pid "
         << header_->server_pid.load() << ", " << header_->num_slots
         << " slots, batch up to " << header_->max_batch << ").";
  }

  BackendAttributes GetAttributes() const override {
    BackendAttributes attributes = header_->attributes;
    attributes.maximum_batch_size = std::min<int>(
        attributes.maximum_batch_size, static_cast<int>(header_->max_batch));
    return attributes;
  }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  uint32_t max_batch() const { return header_->max_batch; }

  // Blocks until a slot is free. Slots are shared with other clients.
  uint32_t AcquireSlot() {
    for (uint32_t spin = 0;; ++spin) {
      for (uint32_t i = 0; i < header_->num_slots; ++i) {
        uint32_t expected = 0;
        if (GetSlot(shm_->data(), i)
                .header->owner_pid.compare_exchange_strong(expected, pid_)) {
          return i;
        }
      }
      // TODO: reclaim slots whose owner died with no request in flight.
      if (spin < 64) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
  }

  void ReleaseSlot(uint32_t index) {
    GetSlot(shm_->data(), index)
        .header->owner_pid.store(0, std::memory_order_release);
  }

  // Posts the slot's batch and waits for the server to answer it.
  void Run(uint32_t index) {
    SlotHeader* slot = GetSlot(shm_->data(), index).header;
    const uint64_t seq = slot->request_seq.load(std::memory_order_relaxed) + 1;
    slot->failed = 0;
    slot->request_seq.store(seq, std::memory_order_release);
    requests_[index].Post();

    const auto start = std::chrono::steady_clock::now();
    auto last_alive = start;
    while (slot->response_seq.load(std::memory_order_acquire) != seq) {
      if (responses_[index].Wait(kPollMs)) continue;
      const auto now = std::chrono::steady_clock::now();
      if (timeout_ms_ > 0 &&
          now - start > std::chrono::milliseconds(timeout_ms_)) {
        throw Exception("Backend server " + name_ + " timed out");
      }
      // A restarted server finds the pending batch and answers it, so a
      // crash only costs the time until someone restarts the server.
      if (IsProcessAlive(header_->server_pid.load())) {
        last_alive = now;
        continue;
      }
      if (now - last_alive > std::chrono::milliseconds(restart_wait_ms_)) {
        throw Exception("Backend server " + name_ + " died");
      }
    }
    if (slot->failed) {
      throw Exception("Backend server " + name_ + " failed to evaluate");
    }
  }

  SlotView Slot(uint32_t index) const { return GetSlot(shm_->data(), index); }

 private:
  std::string name_;
  int timeout_ms_;
  int restart_wait_ms_;
  uint32_t pid_;
  std::unique_ptr<SharedMemory> shm_;
  ServerHeader* header_;
  std::vector<NamedSemaphore> requests_;
  std::vector<NamedSemaphore> responses_;
};

class ProxyComputation : public BackendComputation {
 public:
  explicit ProxyComputation(ProxyBackend* backend) : backend_(backend) {}

  ~ProxyComputation() override {
    // If Run() threw, the server may still be reading the slot, so it leaks.
    if (slot_ >= 0 && !in_flight_) backend_->ReleaseSlot(slot_);
  }

  size_t UsedBatchSize() const override { return results_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    if (slot_ < 0) slot_ = backend_->AcquireSlot();
    if (results_.size() >= backend_->max_batch()) {
      throw Exception("Batch is larger than the backend server allows");
    }
    if (pos.legal_moves.size() > kMaxLegalMoves) {
      throw Exception("Too many legal moves for the backend proxy");
    }
    // The encoder reads at most kMoveHistory positions.
    const auto history =
        pos.pos.last(std::min<size_t>(pos.pos.size(), kMoveHistory));
    PositionRecord& record = backend_->Slot(slot_).positions[results_.size()];
    record.history_size = history.size();
    record.num_moves = pos.legal_moves.size();
    record.want_policy = !result.p.empty();
    std::memcpy(record.history, history.data(),
                history.size() * sizeof(Position));
    std::memcpy(record.moves, pos.legal_moves.data(),
                pos.legal_moves.size() * sizeof(Move));
    results_.push_back(result);
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    if (results_.empty()) return;
    const SlotView slot = backend_->Slot(slot_);
    slot.header->batch_size = results_.size();
    in_flight_ = true;
    backend_->Run(slot_);
    in_flight_ = false;
    for (size_t i = 0; i < results_.size(); ++i) {
      const ResultRecord& src = slot.results[i];
      const EvalResultPtr& dst = results_[i];
      if (dst.q) *dst.q = src.q;
      if (dst.d) *dst.d = src.d;
      if (dst.m) *dst.m = src.m;
      std::copy_n(src.p, std::min<size_t>(dst.p.size(), kMaxLegalMoves),
                  dst.p.begin());
    }
  }

 private:
  ProxyBackend* const backend_;
  int slot_ = -1;
  bool in_flight_ = false;
  std::vector<EvalResultPtr> results_;
};

std::unique_ptr<BackendComputation> ProxyBackend::CreateComputation() {
  return std::make_unique<ProxyComputation>(this);
}

class ProxyBackendFactory : public BackendFactory {
 public:
  int GetPriority() const override { return -1000; }
  std::string_view GetName() const override { return "proxy"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<ProxyBackend>(options);
  }
};

[[maybe_unused]] static BackendManager::Register reg_proxy(
    std::make_unique<ProxyBackendFactory>());

}  // namespace
}  // namespace proxy
}  // namespace lczero
