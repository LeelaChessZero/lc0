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

#include "mcts/batch_collector.h"

#include "utils/fastmath.h"
#include "utils/mutex.h"

namespace lczero {

namespace {
constexpr int kNodePickFanout = 5;
}  // namespace

BatchCollector::BatchCollector(int num_threads) : idle_workers_(0) {
  for (int i = 0; i < num_threads; ++i)
    threads_.emplace_back([this]() { Worker(); });
}

BatchCollector::~BatchCollector() {
  stop_.store(true);
  cv_.notify_all();
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

void BatchCollector::EnqueueWork(Node* root, int limit, int depth) {
  queue_.enqueue(CollectionItem(root, limit, depth));
  if (idle_workers_.load(std::memory_order_acquire) > 0) {
    cv_.notify_one();
  }
}

namespace {
void IncrementN(Node* node, Node* root, int amount) {
  while (true) {
    node->IncrementN(amount);
    if (node == root) break;
    node = node->GetParent();
  }
}
}  // namespace

std::vector<BatchCollector::NodeToProcess> BatchCollector::Collect(
    Node* root, int limit, const SearchParams& params) {
  work_done_ = false;
  // assert(results_.size_approx() == 0);  // DO NOT SUBMIT
  assert(queue_.size_approx() == 0);
  assert(static_cast<int>(threads_.size()) == idle_workers_);
  params_ = &params;
  EnqueueWork(root, limit, 0);
  std::unique_lock<std::mutex> lock(mutex_);
  work_done_cv_.wait(lock, [this]() { return work_done_; });

  std::vector<NodeToProcess> result;
  NodeToProcess node;
  while (results_.try_dequeue(node)) {
    IncrementN(node.node, root, node.multivisit);
    result.emplace_back(std::move(node));
  }
  return result;
}

void BatchCollector::Worker() {
  while (!stop_.load(std::memory_order_acquire)) {
    CollectionItem item;
    if (!queue_.try_dequeue(item)) {
      std::unique_lock<std::mutex> lock(mutex_);
      ++idle_workers_;
      if (idle_workers_ == static_cast<int>(threads_.size())) {
        work_done_ = true;
        work_done_cv_.notify_one();
      }
      // TODO(crem) can size_approx() return 0 when it's not empty?..
      cv_.wait(lock, [this]() {
        return stop_.load(std::memory_order_acquire) ||
               queue_.size_approx() > 0;
      });
      --idle_workers_;
      continue;
    }

    results_.enqueue(
        PickNodeToExtend(item.node, item.remaining_batch_size, item.depth));
  }
}

// DO NOT SUBMIT
// This is copy of functions from search.cc. Make sure there's only one copy.
namespace {
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node) {
  return params.GetFpuAbsolute()
             ? params.GetFpuValue()
             : ((is_root_node && params.GetNoise()) ||
                !params.GetFpuReduction())
                   ? -node->GetQ()
                   : -node->GetQ() - params.GetFpuReduction() *
                                         std::sqrt(node->GetVisitedPolicy());
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N) {
  const float init = params.GetCpuct();
  const float k = params.GetCpuctFactor();
  const float base = params.GetCpuctBase();
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}
}  // namespace

BatchCollector::NodeToProcess BatchCollector::PickNodeToExtend(Node* node,
                                                               int batch_size,
                                                               int depth) {
  constexpr int kEdges = kNodePickFanout + 1;

  while (true) {
    ++depth;

    // First, terminate if we find collisions or leaf nodes.
    if (node->IsBeingExtended()) {
      return NodeToProcess::Collision(node, depth, 1);  // DO NOT SUBMIT
    }

    // Either terminal or unexamined leaf node -- the end of this playout.
    if (!node->HasChildren()) {
      if (node->IsTerminal()) {
        return NodeToProcess::TerminalHit(node, depth, 1);  // DO NOT SUBMIT
      } else {
        node->SetBeingExtended();
        return NodeToProcess::Extension(node, depth);
      }
    }

    std::array<Node::Iterator, kEdges> best_edges;
    std::array<float, kEdges> best_scores;
    best_scores.fill(std::numeric_limits<float>::lowest());

    // Find kEdges best edges.
    const float cpuct = ComputeCpuct(*params_, node->GetN());
    const float fpu = GetFpu(*params_, node, depth == 0);

    float puct_mult =
        cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));

    for (auto child : node->Edges()) {
      float Q = child.GetQ(fpu);
      const float score = child.GetU(puct_mult) + Q;

      int new_index = kEdges;
      for (; new_index > 0; --new_index) {
        if (score <= best_scores[new_index - 1]) break;
        if (new_index < kEdges) {
          best_scores[new_index] = best_scores[new_index - 1];
          best_edges[new_index] = best_edges[new_index - 1];
        }
      }
      if (new_index < kEdges) {
        best_scores[new_index] = score;
        best_edges[new_index] = child;
      }
    }

    Node* parent = node;
    int visits_left = batch_size;
    for (int i = 0; i < kNodePickFanout; ++i) {
      auto& edge = best_edges[i];
      const auto& next_edge = best_edges[i + 1];
      Node* new_node = edge.GetOrSpawnNode(parent, nullptr);
      // Best node will be looked further in this thread, other nodes are
      // queued to be picked in parallel.

      int estimated_visits = batch_size;
      if (estimated_visits > 1 && next_edge) {
        estimated_visits =
            edge.GetVisitsToReachU(best_scores[i + 1], puct_mult, fpu);
      }

      if (i == 0) {
        node = new_node;
        batch_size = estimated_visits;
      } else {
        EnqueueWork(new_node, estimated_visits, depth);
      }
      visits_left -= estimated_visits;
      if (visits_left == 0) break;
    }
  }
}

}  // namespace lczero