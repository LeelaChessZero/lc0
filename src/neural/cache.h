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

#include "neural/network.h"
#include "utils/cache.h"
#include "utils/smallarray.h"

namespace lczero {

struct CachedNNRequest {
  CachedNNRequest(size_t size) : p(size) {}
  typedef std::pair<uint16_t, float> IdxAndProb;
  float q;
  float d;
  // TODO(mooskagh) Don't really need index if using perfect hash.
  SmallArray<IdxAndProb> p;
};

typedef LruCache<uint64_t, CachedNNRequest> NNCache;
typedef LruCacheLock<uint64_t, CachedNNRequest> NNCacheLock;

// Wraps around NetworkComputation and caches result.
// While it mostly repeats NetworkComputation interface, it's not derived
// from it, as AddInput() needs hash and index of probabilities to store.
class CachingComputation {
 public:
  CachingComputation(std::unique_ptr<NetworkComputation> parent,
                     NNCache* cache);

  // How many inputs are not found in cache and will be forwarded to a wrapped
  // computation.
  int GetCacheMisses() const;
  // Total number of times AddInput/AddInputByHash were (successfully) called.
  int GetBatchSize() const;
  // Adds input by hash only. If that hash is not in cache, returns false
  // and does nothing. Otherwise adds.
  bool AddInputByHash(uint64_t hash);
  // Adds a sample to the batch.
  // @hash is a hash to store/lookup it in the cache.
  // @probabilities_to_cache is which indices of policy head to store.
  void AddInput(uint64_t hash, InputPlanes&& input,
                std::vector<uint16_t>&& probabilities_to_cache);
  // Undos last AddInput. If it was a cache miss, the it's actually not removed
  // from parent's batch.
  void PopLastInputHit();
  // Do the computation.
  void ComputeBlocking();
  // Returns Q value of @sample.
  float GetQVal(int sample) const;
  // Returns probability of draw if NN has WDL value head
  float GetDVal(int sample) const;
  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const;
  // Pops last input from the computation. Only allowed for inputs which were
  // cached.
  void PopCacheHit();

 private:
  struct WorkItem {
    uint64_t hash;
    NNCacheLock lock;
    int idx_in_parent = -1;
    std::vector<uint16_t> probabilities_to_cache;
    mutable int last_idx = 0;
  };

  std::unique_ptr<NetworkComputation> parent_;
  NNCache* cache_;
  std::vector<WorkItem> batch_;
};

}  // namespace lczero
