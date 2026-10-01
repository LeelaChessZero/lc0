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

// The engine side of the backend process. lc0 runs every backend it creates in
// a child process, so a crash in the backend or in the GPU driver costs a
// restart of that process instead of the engine. Batches cross in shared
// memory, see protocol.h.

#include "neural/backends/backend_process/child_process_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "neural/backends/backend_process/interprocess.h"
#include "neural/backends/backend_process/protocol.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace backend_process {
namespace {

// Batches in flight at once; more search threads than this wait for a slot.
constexpr uint32_t kNumSlots = 16;
// No network backend reports a larger maximum batch size.
constexpr uint32_t kMaxBatch = 1024;
constexpr int kPollMilliseconds = 100;
constexpr int kStopWaitMilliseconds = 2000;
// Restarts while one batch is pending before it fails.
constexpr uint32_t kMaxRestarts = 2;

std::string Flag(const OptionId& id, const std::string& value) {
  return std::string("--") + id.long_flag() + "=" + value;
}

// The options the backend process needs, as its command line.
std::vector<std::string> ForwardedFlags(const OptionsDict& options) {
  std::vector<std::string> flags;
  for (const OptionId* id :
       {&SharedBackendParams::kWeightsId, &SharedBackendParams::kBackendId,
        &SharedBackendParams::kBackendOptionsId,
        &SharedBackendParams::kHistoryFill}) {
    flags.push_back(Flag(*id, options.Get<std::string>(*id)));
  }
  std::ostringstream temperature;
  temperature << std::setprecision(9)
              << options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
  flags.push_back(
      Flag(SharedBackendParams::kPolicySoftmaxTemp, temperature.str()));
  return flags;
}

// Unique among the backends of all running lc0 processes.
std::string NewName() {
  static std::atomic<uint32_t> counter{0};
  return std::to_string(CurrentProcessId()) + "-" + std::to_string(counter++);
}

class ChildProcessBackend : public Backend {
 public:
  explicit ChildProcessBackend(const OptionsDict& options)
      : flags_(ForwardedFlags(options)),
        name_(NewName()),
        shared_memory_(
            SharedMemory::Create(name_, RegionSize(kNumSlots, kMaxBatch))),
        header_(static_cast<RegionHeader*>(shared_memory_.data())),
        state_changed_(NamedSemaphore::Create(StateName(name_))),
        slot_results_(kNumSlots, std::vector<EvalResultPtr>(kMaxBatch)) {
    header_->version = kVersion;
    header_->position_size = sizeof(Position);
    header_->num_slots = kNumSlots;
    header_->max_batch = kMaxBatch;
    header_->slot_stride = SlotStride(kMaxBatch);
    header_->magic = kMagic;
    for (uint32_t i = 0; i < kNumSlots; ++i) {
      requests_.push_back(NamedSemaphore::Create(RequestName(name_, i)));
      responses_.push_back(NamedSemaphore::Create(ResponseName(name_, i)));
      free_slots_.push_back(i);
    }
    {
      std::lock_guard lock(process_mutex_);
      StartProcess();
    }
    attributes_ = header_->attributes;
    attributes_.maximum_batch_size =
        std::min<int>(attributes_.maximum_batch_size, kMaxBatch);
    Backend::UpdateConfiguration(options);
    LOGFILE << "Backend process " << name_ << " started.";
  }

  ~ChildProcessBackend() override {
    header_->stop.store(1, std::memory_order_release);
    for (NamedSemaphore& request : requests_) request.Post();
    process_.Stop(kStopWaitMilliseconds);
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    // The backend process got these on its command line, so any change,
    // even to the softmax temperature, takes a new process.
    return ForwardedFlags(options) == flags_ ? UPDATE_OK : NEED_RESTART;
  }

  // Blocks until a slot is free.
  uint32_t AcquireSlot() {
    std::unique_lock lock(slots_mutex_);
    // More batches at once than slots, e.g. from more than kNumSlots search
    // threads, would otherwise cap the parallelism without a word.
    if (free_slots_.empty() && !warned_all_slots_busy_) {
      warned_all_slots_busy_ = true;
      CERR << "All " << kNumSlots << " slots of the backend process are "
           << "busy; further batches wait for a free one.";
    }
    slot_freed_.wait(lock, [&] { return !free_slots_.empty(); });
    const uint32_t index = free_slots_.back();
    free_slots_.pop_back();
    return index;
  }

  void ReleaseSlot(uint32_t index) {
    {
      std::lock_guard lock(slots_mutex_);
      free_slots_.push_back(index);
    }
    slot_freed_.notify_one();
  }

  // Posts the slot's batch and waits for the backend process to answer it,
  // restarting the process if it dies meanwhile.
  void Run(uint32_t index) {
    SlotHeader* slot = Slot(index).header;
    // Only the slot's owner writes request_sequence, and slots change owners
    // under slots_mutex_, so a relaxed load sees the last request.
    const uint64_t sequence =
        slot->request_sequence.load(std::memory_order_relaxed) + 1;
    const uint32_t first_start = starts_.load(std::memory_order_relaxed);
    slot->failed = 0;
    slot->request_sequence.store(sequence, std::memory_order_release);
    requests_[index].Post();

    while (slot->response_sequence.load(std::memory_order_acquire) !=
           sequence) {
      if (responses_[index].Wait(kPollMilliseconds)) continue;
      std::lock_guard lock(process_mutex_);
      if (process_.IsRunning() ||
          slot->response_sequence.load(std::memory_order_acquire) == sequence) {
        continue;
      }
      // A batch that crashes every process it meets must not loop forever.
      // Marking it answered keeps the next process away from it.
      if (starts_.load(std::memory_order_relaxed) - first_start >=
          kMaxRestarts) {
        slot->response_sequence.store(sequence, std::memory_order_relaxed);
        throw Exception("The backend process keeps crashing");
      }
      CERR << "The backend process died, restarting it.";
      try {
        // The new process answers the pending batches, as their sequences
        // differ.
        StartProcess();
      } catch (...) {
        slot->response_sequence.store(sequence, std::memory_order_relaxed);
        throw;
      }
    }
    if (slot->failed) {
      throw Exception("The backend process failed to evaluate a batch");
    }
  }

  SlotView Slot(uint32_t index) const {
    return GetSlot(shared_memory_.data(), index);
  }

  // Kept across batches so a batch allocates nothing once warmed up. Only
  // the computation that owns the slot touches its entry.
  std::vector<EvalResultPtr>& SlotResults(uint32_t index) {
    return slot_results_[index];
  }

 private:
  // Starts the backend process and waits until it has loaded the network.
  // Requires process_mutex_.
  void StartProcess() {
    header_->state.store(ProcessState::kStarting, std::memory_order_relaxed);
    header_->error[0] = '\0';
    std::vector<std::string> arguments = {
        ExecutablePath(), "backendprocess", "--name=" + name_,
        "--parent-process-id=" + std::to_string(CurrentProcessId())};
    arguments.insert(arguments.end(), flags_.begin(), flags_.end());
    process_ = ChildProcess::Spawn(arguments);
    starts_.fetch_add(1, std::memory_order_relaxed);
    while (true) {
      const ProcessState state = header_->state.load(std::memory_order_acquire);
      if (state == ProcessState::kReady) return;
      if (state == ProcessState::kFailed) {
        throw Exception(std::string(
            header_->error, strnlen(header_->error, sizeof(header_->error))));
      }
      if (!process_.IsRunning()) {
        // It may have set the state just before it exited.
        if (header_->state.load(std::memory_order_acquire) !=
            ProcessState::kStarting) {
          continue;
        }
        throw Exception("The backend process exited while loading the network");
      }
      // Woken as soon as the state is set. A post left over from an earlier
      // process only costs one more pass.
      state_changed_.Wait(10);
    }
  }

  const std::vector<std::string> flags_;
  const std::string name_;
  SharedMemory shared_memory_;
  RegionHeader* const header_;
  std::vector<NamedSemaphore> requests_;
  std::vector<NamedSemaphore> responses_;
  NamedSemaphore state_changed_;
  BackendAttributes attributes_;

  std::mutex process_mutex_;
  ChildProcess process_;
  std::atomic<uint32_t> starts_{0};

  std::mutex slots_mutex_;
  std::condition_variable slot_freed_;
  std::vector<uint32_t> free_slots_;
  bool warned_all_slots_busy_ = false;
  std::vector<std::vector<EvalResultPtr>> slot_results_;
};

class ChildProcessComputation : public BackendComputation {
 public:
  explicit ChildProcessComputation(ChildProcessBackend* backend)
      : backend_(backend) {}

  // Also after Run() threw: the slot is answered or its process is dead.
  ~ChildProcessComputation() override {
    if (slot_ >= 0) backend_->ReleaseSlot(slot_);
  }

  size_t UsedBatchSize() const override {
    return std::min<size_t>(size_.load(), kMaxBatch);
  }

  // The search's task workers call this concurrently, as they do
  // NetworkAsBackendComputation::AddInput.
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    if (pos.legal_moves.size() > kMaxLegalMoves) {
      throw Exception("Too many legal moves for the backend process");
    }
    std::call_once(slot_acquired_, [&] {
      slot_ = backend_->AcquireSlot();
      results_ = backend_->SlotResults(slot_).data();
    });
    const size_t index = size_.fetch_add(1);
    if (index >= kMaxBatch) {
      throw Exception("Batch is larger than the backend process allows");
    }
    // Only the positions the encoder reads cross to the backend process.
    std::array<int, kCompactHistory> history;
    const int history_size = CompactHistoryForNN(
        backend_->GetAttributes().input_format, pos.pos, history);
    PositionRecord& record = backend_->Slot(slot_).positions[index];
    record.history_size = history_size;
    record.num_moves = pos.legal_moves.size();
    record.want_policy = !result.p.empty();
    for (int i = 0; i < history_size; ++i) {
      std::memcpy(record.history + i * sizeof(Position), &pos.pos[history[i]],
                  sizeof(Position));
    }
    std::memcpy(record.moves, pos.legal_moves.data(),
                pos.legal_moves.size() * sizeof(Move));
    results_[index] = result;
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    const size_t batch_size = UsedBatchSize();
    if (batch_size == 0) return;
    const SlotView slot = backend_->Slot(slot_);
    slot.header->batch_size = batch_size;
    backend_->Run(slot_);
    for (size_t i = 0; i < batch_size; ++i) {
      const ResultRecord& source = slot.results[i];
      const EvalResultPtr& destination = results_[i];
      if (destination.q) *destination.q = source.q;
      if (destination.d) *destination.d = source.d;
      if (destination.m) *destination.m = source.m;
      std::copy_n(source.p,
                  std::min<size_t>(destination.p.size(), kMaxLegalMoves),
                  destination.p.begin());
    }
  }

 private:
  ChildProcessBackend* const backend_;
  std::once_flag slot_acquired_;
  int slot_ = -1;
  EvalResultPtr* results_ = nullptr;
  std::atomic<size_t> size_ = 0;
};

std::unique_ptr<BackendComputation> ChildProcessBackend::CreateComputation() {
  return std::make_unique<ChildProcessComputation>(this);
}

}  // namespace

std::unique_ptr<Backend> CreateChildProcessBackend(const OptionsDict& options) {
  return std::make_unique<ChildProcessBackend>(options);
}

}  // namespace backend_process
}  // namespace lczero
