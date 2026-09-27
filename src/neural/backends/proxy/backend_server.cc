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

#include "neural/backends/proxy/backend_server.h"

#include <algorithm>
#include <span>
#include <thread>
#include <vector>

#include "neural/backends/proxy/ipc.h"
#include "neural/backends/proxy/protocol.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/logging.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

using namespace proxy;

const OptionId kServerNameId{
    "server-name", "", "Name that clients connect to with --backend=proxy "
                       "--backend-opts=name=<name>."};
const OptionId kSlotsId{
    "slots", "",
    "Batches evaluated concurrently, shared by all clients. Use at least the "
    "total number of search threads across clients."};
const OptionId kMaxBatchId{"max-batch", "",
                           "Largest batch a client may send in one request."};

bool Evaluate(Backend* backend, const SlotView& slot, uint32_t max_batch) {
  const uint32_t batch_size = slot.header->batch_size;
  if (batch_size == 0 || batch_size > max_batch) return false;
  try {
    auto computation = backend->CreateComputation();
    for (uint32_t i = 0; i < batch_size; ++i) {
      const PositionRecord& in = slot.positions[i];
      if (in.history_size == 0 || in.history_size > kMoveHistory ||
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

void ServeSlot(Backend* backend, void* base, uint32_t index,
               NamedSemaphore* request, NamedSemaphore* response) {
  const SlotView slot = GetSlot(base, index);
  const uint32_t max_batch = static_cast<ServerHeader*>(base)->max_batch;
  while (true) {
    // Posts can outnumber requests after a restart; the seqs are the truth.
    const uint64_t seq = slot.header->request_seq.load(std::memory_order_acquire);
    if (seq == slot.header->response_seq.load(std::memory_order_relaxed)) {
      request->Wait(-1);
      continue;
    }
    slot.header->failed = !Evaluate(backend, slot, max_batch);
    slot.header->response_seq.store(seq, std::memory_order_release);
    response->Post();
  }
}

}  // namespace

void RunBackendServer() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<StringOption>(kServerNameId) = "default";
  options.Add<IntOption>(kSlotsId, 1, 256) = 16;
  options.Add<IntOption>(kMaxBatchId, 1, 65536) = 1024;
  if (!options.ProcessAllFlags()) return;

  const OptionsDict& dict = options.GetOptionsDict();
  const std::string name = dict.Get<std::string>(kServerNameId);
  if (dict.Get<std::string>(SharedBackendParams::kBackendId) == "proxy") {
    throw Exception("The backend server cannot itself use --backend=proxy");
  }
  // Loads the weights before the region exists, so clients never see a
  // server that is not ready yet.
  std::unique_ptr<Backend> backend =
      BackendManager::Get()->CreateFromParams(dict);
  const BackendAttributes attributes = backend->GetAttributes();
  const uint32_t num_slots = dict.Get<int>(kSlotsId);
  const uint32_t max_batch = static_cast<uint32_t>(std::min(
      dict.Get<int>(kMaxBatchId), attributes.maximum_batch_size));

  SharedMemory shm =
      SharedMemory::CreateOrAttach(name, RegionSize(num_slots, max_batch));
  auto* header = static_cast<ServerHeader*>(shm.data());
  if (header->magic == kMagic) {
    // Replacing a server that died: its clients are still mapped and
    // waiting, so the layout has to stay exactly as it was.
    if (IsProcessAlive(header->server_pid.load())) {
      throw Exception("A backend server named " + name + " is running");
    }
    if (header->version != kVersion ||
        header->position_size != sizeof(Position) ||
        header->num_slots != num_slots || header->max_batch != max_batch) {
      throw Exception("Backend server " + name +
                      " left clients with a different layout; restart it "
                      "with the same --slots and --max-batch");
    }
  } else {
    header->version = kVersion;
    header->position_size = sizeof(Position);
    header->num_slots = num_slots;
    header->max_batch = max_batch;
    header->slot_stride = SlotStride(max_batch);
    header->magic = kMagic;
  }
  header->attributes = attributes;

  std::vector<NamedSemaphore> requests;
  std::vector<NamedSemaphore> responses;
  for (uint32_t i = 0; i < num_slots; ++i) {
    requests.push_back(NamedSemaphore::CreateOrAttach(RequestName(name, i)));
    responses.push_back(NamedSemaphore::CreateOrAttach(ResponseName(name, i)));
    // Free slots whose client died; their pending batch goes with them.
    SlotHeader* slot = GetSlot(shm.data(), i).header;
    const uint32_t owner = slot->owner_pid.load();
    if (owner != 0 && !IsProcessAlive(owner)) {
      slot->response_seq.store(slot->request_seq.load());
      slot->owner_pid.store(0);
    }
  }

  header->server_pid.store(CurrentProcessId());
  header->ready.store(1, std::memory_order_release);
  CERR << "Backend server " << name << " ready: " << num_slots
       << " slots, batch up to " << max_batch << ".";

  // TODO: reclaim slots of clients that die while this server runs, clean
  // shutdown (unlink the POSIX objects), and a supervisor that restarts the
  // server after a crash.
  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < num_slots; ++i) {
    threads.emplace_back(ServeSlot, backend.get(), shm.data(), i, &requests[i],
                         &responses[i]);
  }
  for (auto& thread : threads) thread.join();
}

}  // namespace lczero
