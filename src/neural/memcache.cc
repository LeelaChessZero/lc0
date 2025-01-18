/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "neural/memcache.h"

#include "neural/cache.h"
#include "utils/atomic_vector.h"
#include "utils/smallarray.h"

namespace lczero {
namespace {

// TODO For now it uses the hash of the current position, ignoring repetitions
// and history. We'll likely need to have configurable hash function that we'll
// also reuse as a tree hash key.
uint64_t ComputeEvalPositionHash(const EvalPosition& pos) {
  return pos.pos.back().Hash();
}

struct CachedValue {
  float q;
  float d;
  float m;
  std::unique_ptr<float[]> p;
};

void CachedValueToEvalResult(const CachedValue& cv, const EvalResultPtr& ptr) {
  if (ptr.d) *ptr.d = cv.d;
  if (ptr.q) *ptr.q = cv.q;
  if (ptr.m) *ptr.m = cv.m;
  std::copy(cv.p.get(), cv.p.get() + ptr.p.size(), ptr.p.begin());
}

class MemCache : public Backend {
 public:
  MemCache(std::unique_ptr<Backend> wrapped, size_t capacity)
      : wrapped_backend_(std::move(wrapped)),
        cache_(capacity),
        max_batch_size_(wrapped_backend_->GetAttributes().maximum_batch_size) {}

  BackendAttributes GetAttributes() const override {
    return wrapped_backend_->GetAttributes();
  }
  std::unique_ptr<BackendComputation> CreateComputation() override;
  std::optional<EvalResult> GetCachedEvaluation(const EvalPosition&) override;

 private:
  std::unique_ptr<Backend> wrapped_backend_;
  HashKeyedCache<CachedValue> cache_;
  const size_t max_batch_size_;
  friend class MemCacheComputation;
};

class MemCacheComputation : public BackendComputation {
 public:
  MemCacheComputation(std::unique_ptr<BackendComputation> wrapped_computation,
                      MemCache* memcache)
      : wrapped_computation_(std::move(wrapped_computation)),
        memcache_(memcache),
        entries_(memcache->max_batch_size_) {}

 private:
  size_t UsedBatchSize() const override {
    return wrapped_computation_->UsedBatchSize();
  }
  virtual AddInputResult AddInput(const EvalPosition& pos,
                                  EvalResultPtr result) override {
    const uint64_t hash = ComputeEvalPositionHash(pos);
    {
      HashKeyedCacheLock<CachedValue> lock(&memcache_->cache_, hash);
      if (lock.holds_value()) {
        CachedValueToEvalResult(**lock, result);
        return AddInputResult::FETCHED_IMMEDIATELY;
      }
    }
    size_t entry_idx = entries_.emplace_back(
        Entry{hash, std::make_unique<CachedValue>(), result});
    auto& value = entries_[entry_idx].value;
    value->p.reset(pos.legal_moves.empty() ? nullptr
                                           : new float[pos.legal_moves.size()]);
    return wrapped_computation_->AddInput(
        pos, EvalResultPtr{&value->q,
                           &value->d,
                           &value->m,
                           {value->p.get(), pos.legal_moves.size()}});
  }

  virtual void ComputeBlocking() override {
    wrapped_computation_->ComputeBlocking();
    for (auto& entry : entries_) {
      CachedValueToEvalResult(*entry.value, entry.result_ptr);
      memcache_->cache_.Insert(entry.key, std::move(entry.value));
    }
  }

  struct Entry {
    uint64_t key;
    std::unique_ptr<CachedValue> value;
    EvalResultPtr result_ptr;
  };

  std::unique_ptr<BackendComputation> wrapped_computation_;
  MemCache* memcache_;
  AtomicVector<Entry> entries_;
};

std::unique_ptr<BackendComputation> MemCache::CreateComputation() {
  return std::make_unique<MemCacheComputation>(
      wrapped_backend_->CreateComputation(), this);
}
std::optional<EvalResult> MemCache::GetCachedEvaluation(
    const EvalPosition& pos) {
  const uint64_t hash = ComputeEvalPositionHash(pos);
  HashKeyedCacheLock<CachedValue> lock(&cache_, hash);
  if (!lock.holds_value()) return std::nullopt;
  EvalResult result;
  result.d = lock->d;
  result.q = lock->q;
  result.m = lock->m;
  std::copy(lock->p.get(), lock->p.get() + pos.legal_moves.size(),
            result.p.begin());
  return result;
}

}  // namespace

std::unique_ptr<Backend> CreateMemCache(std::unique_ptr<Backend> wrapped,
                                        size_t capacity) {
  return std::make_unique<MemCache>(std::move(wrapped), capacity);
}

}  // namespace lczero