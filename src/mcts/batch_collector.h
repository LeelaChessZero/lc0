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

#include <concurrentqueue.h>
#include <atomic>
#include <thread>
#include <vector>

#include "mcts/node.h"
#include "mcts/params.h"
#include "utils/logging.h"

namespace lczero {

// Not thread safe! Can only be used by one thread at a time.
class BatchCollector {
 public:
  struct NodeToProcess {
    NodeToProcess() = default;
    bool IsExtendable() const { return !is_collision && !node->IsTerminal(); }
    bool IsCollision() const { return is_collision; }
    bool CanEvalOutOfOrder() const {
      return is_cache_hit || node->IsTerminal();
    }

    // The node to extend.
    Node* node;
    // Value from NN's value head, or -1/0/1 for terminal nodes.
    float v;
    int multivisit = 0;
    uint16_t depth;
    bool nn_queried = false;
    bool is_cache_hit = false;
    bool is_collision = false;

    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count) {
      return NodeToProcess(node, depth, true, collision_count);
    }
    static NodeToProcess Extension(Node* node, uint16_t depth) {
      return NodeToProcess(node, depth, false, 1);
    }
    static NodeToProcess TerminalHit(Node* node, uint16_t depth,
                                     int visit_count) {
      return NodeToProcess(node, depth, false, visit_count);
    }

   private:
    NodeToProcess(Node* node, uint16_t depth, bool is_collision, int multivisit)
        : node(node),
          multivisit(multivisit),
          depth(depth),
          is_collision(is_collision) {}
  };

  BatchCollector(int num_threads);
  ~BatchCollector();
  std::vector<NodeToProcess> Collect(Node* root, int limit,
                                     const SearchParams& params);

 private:
  NodeToProcess PickNodeToExtend(Node* node, int batch_size, int depth);
  void EnqueueWork(Node* root, int limit, int depth);
  void Worker();

  struct CollectionItem {
    CollectionItem() = default;
    CollectionItem(Node* node, int batch, int depth)
        : node(node), remaining_batch_size(batch), depth(depth) {}
    Node* node;
    int remaining_batch_size;
    int depth;
  };

  std::atomic<bool> stop_;
  std::vector<std::thread> threads_;
  moodycamel::ConcurrentQueue<CollectionItem> queue_;
  moodycamel::ConcurrentQueue<NodeToProcess> results_;
  const SearchParams* params_;

  std::atomic<int> idle_workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable work_done_cv_;
  bool work_done_ = false;
};

}  // namespace lczero