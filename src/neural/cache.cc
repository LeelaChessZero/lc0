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

#include <array>
#include <cassert>
#include <iostream>

#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/pfloat16.h"

namespace lczero {
CachingComputation::CachingComputation(
    std::unique_ptr<NetworkComputation> parent,
    pblczero::NetworkFormat::InputFormat input_format,
    lczero::FillEmptyHistory history_fill, float softmax_temp,
    int history_length, NNCache* cache)
    : parent_(std::move(parent)),
      input_format_(input_format),
      history_fill_(history_fill),
      softmax_temp_(softmax_temp),
      history_length_(history_length),
      cache_(cache) {}

int CachingComputation::GetCacheMisses() const {
  return parent_->GetBatchSize();
}

int CachingComputation::GetBatchSize() const { return batch_.size(); }

void CachingComputation::PopCacheHit() {
  assert(!batch_.empty());
  assert(batch_.back().lock);
  assert(batch_.back().idx_in_parent == -1);
  batch_.pop_back();
}

bool CachingComputation::CacheLookup(const PositionHistory& history,
                                     const MoveList& moves,
                                     CachedNNRequest* entry) {
  const auto hash = history.HashLast(history_length_);
  NNCacheLock lock(cache_, hash);
  if (!lock) return false;
  if (entry != nullptr) {
    if (moves.size() != lock->p.size()) return false;
    entry->q = lock->q;
    entry->d = lock->d;
    entry->m = lock->m;
    entry->p.clear();
    entry->p.resize(lock->p.size());
    for (size_t i = 0; i < lock->p.size(); i++) entry->p[i] = lock->p[i];
  }
  return true;
}

void CachingComputation::AddInput(const PositionHistory& history,
                                  const MoveList& moves) {
  const auto hash = history.HashLast(history_length_);
  NNCacheLock lock(cache_, hash);
  if (lock && moves.size() == lock->p.size()) {
    batch_.emplace_back();
    batch_.back().lock = std::move(lock);
    batch_.back().hash = hash;
    return;
  }

  int transform;
  auto input =
      EncodePositionForNN(input_format_, history, 8, history_fill_, &transform);
  std::vector<uint16_t> moves_as_nn_index;
  moves_as_nn_index.reserve(moves.size());
  for (auto iter = moves.begin(), end = moves.end(); iter != end; ++iter) {
    moves_as_nn_index.emplace_back(iter->as_nn_index(transform));
  }
  batch_.emplace_back();
  batch_.back().hash = hash;
  batch_.back().idx_in_parent = parent_->GetBatchSize();
  batch_.back().probabilities_to_cache = std::move(moves_as_nn_index);
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

  // Intermediate array to store values when processing policy.
  // There are never more than 256 valid legal moves in any legal position.
  std::array<float, 256> intermediate;

  // Fill cache with data from NN.
  for (auto& item : batch_) {
    if (item.idx_in_parent == -1) continue;
    auto req =
        std::make_unique<CachedNNRequest>(item.probabilities_to_cache.size());
    req->q = parent_->GetQVal(item.idx_in_parent);
    req->d = parent_->GetDVal(item.idx_in_parent);
    req->m = parent_->GetMVal(item.idx_in_parent);

    // Calculate maximum first.
    float max_p = -std::numeric_limits<float>::infinity();
    int counter = 0;
    for (auto x : item.probabilities_to_cache) {
      float p = parent_->GetPVal(item.idx_in_parent, x);
      intermediate[counter++] = p;
      max_p = std::max(max_p, p);
    }
    float total = 0.0;
    for (int i = 0; i < counter; i++) {
      // Perform softmax and take into account policy softmax temperature T.
      // Note that we want to calculate (exp(p-max_p))^(1/T) = exp((p-max_p)/T).
      float p = FastExp((intermediate[i] - max_p) / softmax_temp_);
      intermediate[i] = p;
      total += p;
    }
    // Normalize P values to add up to 1.0.
    const float scale = total > 0.0f ? 1.0f / total : 1.0f;
    for (size_t ct = 0; ct < item.probabilities_to_cache.size(); ct++) {
      pfloat16 p = intermediate[ct] * scale;
      req->p[ct] = p;
      std::memcpy(&item.probabilities_to_cache[ct], &p, sizeof(pfloat16));
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

float CachingComputation::GetMVal(int sample) const {
  const auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) return parent_->GetMVal(item.idx_in_parent);
  return item.lock->m;
}

pfloat16 CachingComputation::GetPVal(int sample, int move_ct) const {
  auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) {
    pfloat16 r;
    std::memcpy(&r, &item.probabilities_to_cache[move_ct], sizeof(pfloat16));
    return r;
  }
  return item.lock->p[move_ct];
}

}  // namespace lczero
