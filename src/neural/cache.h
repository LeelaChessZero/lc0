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
*/
#pragma once

#include "neural/network.h"
#include "utils/cache.h"

namespace lczero {

struct CachedNNRequest {
  typedef std::pair<uint16_t, float> IdxAndProb;
  float q;
  std::vector<IdxAndProb> p;
};

typedef LruCache<uint64_t, CachedNNRequest> NNCache;
typedef LruCacheLock<uint64_t, CachedNNRequest> NNCacheLock;

class CachingComputation {
 public:
  CachingComputation(std::unique_ptr<NetworkComputation> parent,
                     NNCache* cache);

  int GetCacheMisses() const;
  int GetBatchSize() const;
  bool AddInputByHash(uint64_t hash);
  // Adds a sample to the batch.
  void AddInput(uint64_t hash, InputPlanes&& input,
                std::vector<uint16_t>&& probabilities_to_cache);
  void PopLastInputHit();
  // Do the computation.
  void ComputeBlocking();
  // Returns Q value of @sample.
  float GetQVal(int sample) const;
  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const;

 private:
  struct WorkItem {
    uint64_t hash;
    NNCacheLock lock;
    int idx_in_parent = -1;
    std::vector<uint16_t> probabilities_to_cache;
  };

  std::unique_ptr<NetworkComputation> parent_;
  NNCache* cache_;
  std::vector<WorkItem> batch_;
};

}  // namespace lczero