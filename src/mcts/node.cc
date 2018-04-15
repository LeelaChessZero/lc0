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

#include "mcts/node.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include "utils/hashcat.h"

namespace lczero {

namespace {
const int kAllocationSize = 1024 * 64;
}

Node* NodePool::GetNode() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pool_.empty()) {
    AllocateNewBatch();
  }

  Node* result = pool_.back();
  pool_.pop_back();
  std::memset(result, 0, sizeof(Node));
  return result;
}

void NodePool::ReleaseNode(Node* node) {
  std::lock_guard<std::mutex> lock(mutex_);
  pool_.push_back(node);
}

void NodePool::ReleaseChildren(Node* node) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (Node* iter = node->child; iter; iter = iter->sibling) {
    ReleaseSubtree(iter);
  }
  node->child = nullptr;
}

void NodePool::ReleaseAllChildrenExceptOne(Node* root, Node* subtree) {
  std::lock_guard<std::mutex> lock(mutex_);
  Node* child = nullptr;
  for (Node* iter = root->child; iter; iter = iter->sibling) {
    if (iter == subtree) {
      child = iter;
    } else {
      ReleaseSubtree(iter);
    }
  }
  root->child = child;
  if (child) {
    child->sibling = nullptr;
  }
}

uint64_t NodePool::GetAllocatedNodeCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return kAllocationSize * allocations_.size() - pool_.size();
}

// Mutex must be hold.
void NodePool::ReleaseSubtree(Node* node) {
  for (Node* iter = node->child; iter; iter = iter->sibling) {
    ReleaseSubtree(iter);
    pool_.push_back(iter);
  }
}

// Mutex must be hold.
void NodePool::AllocateNewBatch() {
  allocations_.emplace_back(std::make_unique<Node[]>(kAllocationSize));
  for (int i = 0; i < kAllocationSize; ++i) {
    pool_.push_back(allocations_.back().get() + i);
  }
}

uint64_t Node::BoardHash() const {
  return board.Hash();
  // return HashCat({board.Hash(), no_capture_ply, repetitions});
}

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move.as_string() << "\n"
      << board.DebugString() << "Term:" << is_terminal << " Parent:" << parent
      << " child:" << child << " sibling:" << sibling << " P:" << p
      << " Q:" << q << " W:" << w << " N:" << n << " N_:" << n_in_flight
      << " Rep:" << (int)repetitions;
  return oss.str();
}

int ComputeRepetitions(const Node* ref_node) {
  // TODO(crem) implement some sort of caching.
  if (ref_node->no_capture_ply < 2) return 0;

  const Node* node = ref_node;
  while (true) {
    node = node->parent;
    if (!node) break;
    node = node->parent;
    if (!node) break;

    if (node->board == ref_node->board) {
      return 1 + node->repetitions;
    }
    if (node->no_capture_ply < 2) return 0;
  }
  return 0;
}

}  // namespace lczero