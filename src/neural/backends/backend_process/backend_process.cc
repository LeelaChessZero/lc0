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

#include "neural/backends/backend_process/backend_process.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/backends/backend_process/interprocess.h"
#include "neural/backends/backend_process/protocol.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/logging.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

using namespace backend_process;

const OptionId kNameId{"name", "", "Shared memory the engine created."};
const OptionId kParentProcessIdId{"parent-process-id", "",
                                  "Process id of the engine."};

bool Evaluate(Backend* backend, const SlotView& slot, uint32_t max_batch) {
  const uint32_t batch_size = slot.header->batch_size;
  if (batch_size == 0 || batch_size > max_batch) return false;
  try {
    auto computation = backend->CreateComputation();
    for (uint32_t i = 0; i < batch_size; ++i) {
      const PositionRecord& in = slot.positions[i];
      if (in.history_size == 0 || in.history_size > kCompactHistory ||
          in.num_moves > kMaxLegalMoves) {
        return false;
      }
      ResultRecord& out = slot.results[i];
      computation->AddInput(
          EvalPosition{std::span(in.positions(), in.history_size),
                       std::span(in.moves, in.num_moves)},
          EvalResultPtr{&out.q, &out.d, &out.m,
                        in.want_policy ? std::span(out.p, in.num_moves)
                                       : std::span<float>()});
    }
    computation->ComputeBlocking();
  } catch (const std::exception& e) {
    CERR << "Backend error: " << e.what();
    return false;
  }
  return true;
}

void RunSlot(Backend* backend, void* base, uint32_t index,
             NamedSemaphore* request, NamedSemaphore* response) {
  const auto* header = static_cast<const RegionHeader*>(base);
  const SlotView slot = GetSlot(base, index);
  while (!header->stop.load(std::memory_order_acquire)) {
    // Posts can outnumber requests after a restart; the sequences are the
    // truth.
    const uint64_t sequence =
        slot.header->request_sequence.load(std::memory_order_acquire);
    if (sequence ==
        slot.header->response_sequence.load(std::memory_order_relaxed)) {
      request->Wait(-1);
      continue;
    }
    slot.header->failed = !Evaluate(backend, slot, header->max_batch);
    slot.header->response_sequence.store(sequence, std::memory_order_release);
    response->Post();
  }
}

// Evaluates the starting position, so that what a backend sets up on its
// first batch, such as oneMKL's GEMM kernels under SYCL, is done before
// readyok instead of on the first move.
void WarmUp(Backend* backend) {
  const Position position = Position::FromFen(ChessBoard::kStartposFen);
  const MoveList moves = position.GetBoard().GenerateLegalMoves();
  std::vector<float> policy(moves.size());
  float q = 0, d = 0, m = 0;
  auto computation = backend->CreateComputation();
  computation->AddInput(EvalPosition{std::span(&position, 1), moves},
                        EvalResultPtr{&q, &d, &m, policy});
  computation->ComputeBlocking();
}

}  // namespace

void RunBackendProcess() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<StringOption>(kNameId);
  options.Add<IntOption>(kParentProcessIdId, 0,
                         std::numeric_limits<int>::max()) = 0;
  if (!options.ProcessAllFlags()) return;
  const OptionsDict& dict = options.GetOptionsDict();

  // Nothing is left to evaluate for once the engine is gone, and on POSIX
  // nothing else ends this process then.
  const uint32_t parent_process_id = dict.Get<int>(kParentProcessIdId);
  std::thread([parent_process_id] {
    WaitForParentExit(parent_process_id);
    std::_Exit(1);
  }).detach();

  const std::string name = dict.Get<std::string>(kNameId);
  SharedMemory shared_memory = SharedMemory::Open(name);
  // Too small to hold an error message; the engine sees this process exit.
  if (shared_memory.size() < sizeof(RegionHeader)) return;
  auto* header = static_cast<RegionHeader*>(shared_memory.data());

  std::vector<NamedSemaphore> requests;
  std::vector<NamedSemaphore> responses;
  // Opened only after the checks; the engine still sees a state set before,
  // just without being woken for it.
  std::optional<NamedSemaphore> state_changed;
  std::unique_ptr<Backend> backend;
  try {
    if (header->magic != kMagic || header->version != kVersion ||
        header->position_size != sizeof(Position) ||
        shared_memory.size() <
            RegionSize(header->num_slots, header->max_batch)) {
      throw Exception("The backend process is from a different lc0 build");
    }
    state_changed.emplace(NamedSemaphore::Open(StateName(name)));
    for (uint32_t i = 0; i < header->num_slots; ++i) {
      requests.push_back(NamedSemaphore::Open(RequestName(name, i)));
      responses.push_back(NamedSemaphore::Open(ResponseName(name, i)));
    }
    backend = BackendManager::Get()->CreateInProcess(dict);
    WarmUp(backend.get());
  } catch (const std::exception& e) {
    std::snprintf(header->error, sizeof(header->error), "%s", e.what());
    header->state.store(ProcessState::kFailed, std::memory_order_release);
    if (state_changed) state_changed->Post();
    return;
  }
  header->attributes = backend->GetAttributes();
  header->state.store(ProcessState::kReady, std::memory_order_release);
  state_changed->Post();

  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < header->num_slots; ++i) {
    threads.emplace_back(RunSlot, backend.get(), shared_memory.data(), i,
                         &requests[i], &responses[i]);
  }
  for (auto& thread : threads) thread.join();
}

}  // namespace lczero
