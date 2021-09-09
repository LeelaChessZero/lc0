/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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
#pragma once

#include "mcts/node.h"
#include "neural/encoder.h"
#include "neural/network.h"
#include "utils/cache.h"

namespace lczero {

struct CachedNNRequest {
  std::shared_ptr<LowNode> low_node;
};

typedef HashKeyedCache<CachedNNRequest> NNCache;
typedef HashKeyedCacheLock<CachedNNRequest> NNCacheLock;

// Wraps around NetworkComputation and caches result.
// While it mostly repeats NetworkComputation interface, it's not derived
// from it, as AddInput() needs hash and index of probabilities to store.
class CachingComputation {
 public:
  CachingComputation(std::unique_ptr<NetworkComputation> parent,
                     pblczero::NetworkFormat::InputFormat input_format,
                     FillEmptyHistory history_fill, NNCache* cache);

  // How many inputs are not found in cache and will be forwarded to a wrapped
  // computation.
  int GetCacheMisses() const;
  // Total number of times AddInput/AddInputByHash were (successfully) called.
  int GetBatchSize() const;
  // Adds input by hash only. If that hash is not in cache, returns false
  // and does nothing. Otherwise adds.
  bool AddInputByHash(uint64_t hash);
  // Adds input by hash with existing lock. Assumes the given lock holds a real
  // reference.
  void AddInputByHash(uint64_t hash, NNCacheLock&& lock);
  // Adds a sample to the batch. Also calls EncodePositionForNN() if needed.
  // @hash is a hash to store/lookup it in the cache.
  void AddInput(uint64_t hash, const PositionHistory& history,
                const Node* node);
  // Undos last AddInput. If it was a cache miss, then it's actually not removed
  // from parent's batch.
  void PopLastInputHit();
  // Do the computation.
  void ComputeBlocking(float softmax_temp);
  // Returns Q value of @sample.
  float GetQVal(int sample) const;
  // Returns probability of draw if NN has WDL value head.
  float GetDVal(int sample) const;
  // Returns estimated remaining moves.
  float GetMVal(int sample) const;
  // Returns compressed P value @move_id of @sample.
  uint16_t GetPVal(int sample, int move_ct) const;
  Move GetMove(int sample, int move_ct) const;
  // Pops last input from the computation. Only allowed for inputs which were
  // cached.
  void PopCacheHit();

  // Can be used to avoid repeated reallocations internally while adding itemms.
  void Reserve(int batch_size) { batch_.reserve(batch_size); }

 private:
  struct WorkItem {
    uint64_t hash;
    NNCacheLock lock;
    int idx_in_parent = -1;
    std::shared_ptr<LowNode> low_node;
    int transform;
  };

  std::unique_ptr<NetworkComputation> parent_;
  pblczero::NetworkFormat::InputFormat input_format_;
  FillEmptyHistory history_fill_;
  NNCache* cache_;
  std::vector<WorkItem> batch_;
};

}  // namespace lczero
