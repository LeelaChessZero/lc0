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

// Shared memory layout between lc0 and the backend process it starts.
//
// lc0 creates the region and the semaphores, then starts `lc0 backendprocess`,
// which loads the network, fills in the attributes and sets the state to
// kReady. [RegionHeader][slot 0][slot 1]...  Each slot carries one batch at a
// time: [SlotHeader][PositionRecord x max_batch][ResultRecord x max_batch],
// with a request/response semaphore pair. lc0 writes the positions, posts the
// request and waits for the response whose seq matches its own.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "chess/position.h"
#include "neural/backend.h"
#include "neural/encoder.h"

namespace lczero {
namespace proxy {

constexpr uint32_t kMagic = 0x6c63304e;
constexpr uint32_t kVersion = 1;
constexpr uint32_t kMaxLegalMoves = 256;

enum class ProcessState : uint32_t { kStarting, kReady, kFailed };

// Positions cross the boundary as raw bytes, so both sides must be the same
// build; RegionHeader::position_size is a cheap guard against mixing them.
static_assert(std::is_trivially_copyable_v<Position>);
static_assert(std::is_trivially_copyable_v<Move>);
static_assert(std::is_trivially_copyable_v<BackendAttributes>);
static_assert(std::atomic<ProcessState>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);

struct RegionHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t position_size;
  uint32_t num_slots;
  uint32_t max_batch;
  uint64_t slot_stride;
  // Written by the backend process before it sets the state to kReady.
  BackendAttributes attributes;
  std::atomic<ProcessState> state;
  // Set by lc0 to make the backend process exit.
  std::atomic<uint32_t> stop;
  // Why loading failed, when the state is kFailed.
  char error[1024];
};

struct SlotHeader {
  uint32_t batch_size;
  uint32_t failed;
  // A request is pending while these differ, so a restarted backend
  // process picks up the batch its predecessor died on.
  std::atomic<uint64_t> request_seq;
  std::atomic<uint64_t> response_seq;
};

struct PositionRecord {
  uint32_t history_size;  // At most kMoveHistory, newest last.
  uint32_t num_moves;
  uint32_t want_policy;
  alignas(Position) std::byte history[kMoveHistory * sizeof(Position)];
  Move moves[kMaxLegalMoves];

  const Position* positions() const {
    return reinterpret_cast<const Position*>(history);
  }
};

struct ResultRecord {
  float q;
  float d;
  float m;
  float p[kMaxLegalMoves];
};

constexpr size_t RoundUp(size_t size) { return (size + 63) / 64 * 64; }

inline size_t SlotStride(uint32_t max_batch) {
  // GetSlot() starts the positions at RoundUp(sizeof(SlotHeader)).
  return RoundUp(RoundUp(sizeof(SlotHeader)) +
                 max_batch * (sizeof(PositionRecord) + sizeof(ResultRecord)));
}

inline size_t RegionSize(uint32_t num_slots, uint32_t max_batch) {
  return RoundUp(sizeof(RegionHeader)) + num_slots * SlotStride(max_batch);
}

struct SlotView {
  SlotHeader* header;
  PositionRecord* positions;
  ResultRecord* results;
};

inline SlotView GetSlot(void* base, uint32_t index) {
  const auto* header = static_cast<const RegionHeader*>(base);
  char* slot = static_cast<char*>(base) + RoundUp(sizeof(RegionHeader)) +
               index * header->slot_stride;
  auto* positions =
      reinterpret_cast<PositionRecord*>(slot + RoundUp(sizeof(SlotHeader)));
  auto* results =
      reinterpret_cast<ResultRecord*>(positions + header->max_batch);
  return {reinterpret_cast<SlotHeader*>(slot), positions, results};
}

inline std::string RequestName(const std::string& name, uint32_t slot) {
  return name + "-q" + std::to_string(slot);
}

inline std::string ResponseName(const std::string& name, uint32_t slot) {
  return name + "-r" + std::to_string(slot);
}

}  // namespace proxy
}  // namespace lczero
