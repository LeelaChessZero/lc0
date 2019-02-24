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
#include "neural/cache.h"
#include <cassert>
#include <iostream>

namespace lczero {
CachingComputation::CachingComputation(
    std::unique_ptr<NetworkComputation> parent, NNCache* cache)
    : parent_(std::move(parent)), cache_(cache) {}

int CachingComputation::GetCacheMisses() const {
  return parent_->GetBatchSize();
}

int CachingComputation::GetBatchSize() const { return batch_.size(); }

bool CachingComputation::AddInputByHash(uint64_t hash) {
  NNCacheLock lock(cache_, hash);
  if (!lock) return false;
  batch_.emplace_back();
  batch_.back().lock = std::move(lock);
  batch_.back().hash = hash;
  return true;
}

void CachingComputation::PopCacheHit() {
  assert(!batch_.empty());
  assert(batch_.back().lock);
  assert(batch_.back().idx_in_parent == -1);
  batch_.pop_back();
}

void CachingComputation::AddInput(
    uint64_t hash, InputPlanes&& input,
    std::vector<uint16_t>&& probabilities_to_cache) {
  if (AddInputByHash(hash)) return;
  batch_.emplace_back();
  batch_.back().hash = hash;
  batch_.back().idx_in_parent = parent_->GetBatchSize();
  batch_.back().probabilities_to_cache = probabilities_to_cache;
  parent_->AddInput(std::move(input));
}

void CachingComputation::PopLastInputHit() {
  assert(!batch_.empty());
  assert(batch_.back().idx_in_parent == -1);
  batch_.pop_back();
}

void CachingComputation::ComputeBlocking() {
  if (parent_->GetBatchSize() == 0) return;
  parent_->ComputeBlocking();

  // Fill cache with data from NN.
  for (const auto& item : batch_) {
    if (item.idx_in_parent == -1) continue;
    auto req =
        std::make_unique<CachedNNRequest>(item.probabilities_to_cache.size());
    req->q = parent_->GetQVal(item.idx_in_parent);
    req->d = parent_->GetDVal(item.idx_in_parent);
    int idx = 0;
    for (auto x : item.probabilities_to_cache) {
      req->p[idx++] =
          std::make_pair(x, parent_->GetPVal(item.idx_in_parent, x));
    }
    cache_->Insert(item.hash, std::move(req));
  }
}

float CachingComputation::GetQVal(int sample) const {
  const auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) return parent_->GetQVal(item.idx_in_parent);
  return item.lock->q;
}

float CachingComputation::GetDVal(int sample) const {
  const auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) return parent_->GetDVal(item.idx_in_parent);
  return item.lock->d;
}

float CachingComputation::GetPVal(int sample, int move_id) const {
  auto& item = batch_[sample];
  if (item.idx_in_parent >= 0)
    return parent_->GetPVal(item.idx_in_parent, move_id);
  const auto& moves = item.lock->p;

  int total_count = 0;
  while (total_count < moves.size()) {
    // Optimization: usually moves are stored in the same order as queried.
    const auto& move = moves[item.last_idx++];
    if (item.last_idx == moves.size()) item.last_idx = 0;
    if (move.first == move_id) return move.second;
    ++total_count;
  }
  assert(false);  // Move not found.
  return 0;
}

}  // namespace lczero
