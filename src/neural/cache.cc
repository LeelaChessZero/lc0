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

#include "utils/fastmath.h"

namespace lczero {
CachingComputation::CachingComputation(
    std::unique_ptr<NetworkComputation> parent,
    pblczero::NetworkFormat::InputFormat input_format,
    lczero::FillEmptyHistory history_fill, NNCache* cache)
    : parent_(std::move(parent)),
      input_format_(input_format),
      history_fill_(history_fill),
      cache_(cache) {}

int CachingComputation::GetCacheMisses() const {
  return parent_->GetBatchSize();
}

int CachingComputation::GetBatchSize() const { return batch_.size(); }

bool CachingComputation::AddInputByHash(uint64_t hash) {
  NNCacheLock lock(cache_, hash);
  if (!lock) return false;
  AddInputByHash(hash, std::move(lock));
  return true;
}

void CachingComputation::AddInputByHash(uint64_t hash, NNCacheLock&& lock) {
  assert(lock);
  batch_.emplace_back();
  batch_.back().lock = std::move(lock);
  batch_.back().hash = hash;
}

void CachingComputation::PopCacheHit() {
  assert(!batch_.empty());
  assert(batch_.back().lock);
  assert(batch_.back().idx_in_parent == -1);
  batch_.pop_back();
}

void CachingComputation::AddInput(uint64_t hash, const PositionHistory& history,
                                  const Node* node) {
  if (AddInputByHash(hash)) {
    return;
  }
  int transform;
  auto input =
      EncodePositionForNN(input_format_, history, 8, history_fill_, &transform);
  std::vector<Move> moves;
  if (node && node->HasChildren()) {
    // Legal moves are known, use them.
    moves.reserve(node->GetNumEdges());
    for (const auto& edge : node->Edges()) {
      moves.emplace_back(edge.GetMove());
    }
  } else {
    // Cache legal moves.
    moves = history.Last().GetBoard().GenerateLegalMoves();
  }
  batch_.emplace_back();
  batch_.back().hash = hash;
  batch_.back().idx_in_parent = parent_->GetBatchSize();
  batch_.back().edges = Edge::FromMovelist(moves);
  batch_.back().transform = transform;
  batch_.back().num_edges = moves.size();
  parent_->AddInput(std::move(input));
  return;
}

void CachingComputation::PopLastInputHit() {
  assert(!batch_.empty());
  assert(batch_.back().idx_in_parent == -1);
  batch_.pop_back();
}

void CachingComputation::ComputeBlocking(float softmax_temp) {
  if (parent_->GetBatchSize() == 0) return;
  parent_->ComputeBlocking();

  // Fill cache with data from NN.
  for (auto& item : batch_) {
    if (item.idx_in_parent == -1) continue;
    auto req =
        std::make_unique<CachedNNRequest>();
    req->q = parent_->GetQVal(item.idx_in_parent);
    req->d = parent_->GetDVal(item.idx_in_parent);
    req->m = parent_->GetMVal(item.idx_in_parent);

    // Calculate maximum first.
    float max_p = -std::numeric_limits<float>::infinity();
    // Intermediate array to store values when processing policy.
    // There are never more than 256 valid legal moves in any legal position.
    std::array<float, 256> intermediate;
    int transform = item.transform;
    for (int ct = 0; ct < item.num_edges; ct++) {
      auto move = item.edges[ct].GetMove();
      float p =
          parent_->GetPVal(item.idx_in_parent, move.as_nn_index(transform));
      intermediate[ct] = p;
      max_p = std::max(max_p, p);
    }
    float total = 0.0;
    for (int ct = 0; ct < item.num_edges; ct++) {
      // Perform softmax and take into account policy softmax temperature T.
      // Note that we want to calculate (exp(p-max_p))^(1/T) = exp((p-max_p)/T).
      float p = FastExp((intermediate[ct] - max_p) / softmax_temp);
      intermediate[ct] = p;
      total += p;
    }
    // Normalize P values to add up to 1.0.
    const float scale = total > 0.0f ? 1.0f / total : 1.0f;
    for (int ct = 0; ct < item.num_edges; ct++) {
      item.edges[ct].SetP(intermediate[ct] * scale);
    }

    Edge::SortEdges(item.edges.get(), item.num_edges);

    std::vector<Move> moves;
    moves.resize(item.num_edges);
    for (int ct = 0; ct < item.num_edges; ct++) {
      moves[ct] = item.edges[ct].GetMove();
    }
    req->edges = Edge::FromMovelist(moves);
    for (int ct = 0; ct < item.num_edges; ct++) {
      req->edges[ct].SetPCompressed(item.edges[ct].GetPCompressed());
    }
    req->num_edges = item.num_edges;
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

uint16_t CachingComputation::GetPVal(int sample, int move_ct) const {
  auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) {
    return item.edges[move_ct].GetPCompressed();
  }
  return item.lock->edges[move_ct].GetPCompressed();
}

Move CachingComputation::GetMove(int sample, int move_ct) const {
  auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) {
    return item.edges[move_ct].GetMove();
  }
  return item.lock->edges[move_ct].GetMove();
}

}  // namespace lczero
