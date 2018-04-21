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
  Mutex::Lock lock(mutex_);
  if (pool_.empty()) {
    AllocateNewBatch();
  }

  Node* result = pool_.back();
  pool_.pop_back();
  std::memset(result, 0, sizeof(Node));
  return result;
}

void NodePool::ReleaseNode(Node* node) {
  Mutex::Lock lock(mutex_);
  pool_.push_back(node);
}

void NodePool::ReleaseChildren(Node* node) {
  Mutex::Lock lock(mutex_);
  for (Node* iter = node->child; iter; iter = iter->sibling) {
    ReleaseSubtreeInternal(iter);
  }
  node->child = nullptr;
}

void NodePool::ReleaseAllChildrenExceptOne(Node* root, Node* subtree) {
  Mutex::Lock lock(mutex_);
  Node* child = nullptr;
  for (Node* iter = root->child; iter; iter = iter->sibling) {
    if (iter == subtree) {
      child = iter;
    } else {
      ReleaseSubtreeInternal(iter);
    }
  }
  root->child = child;
  if (child) {
    child->sibling = nullptr;
  }
}

uint64_t NodePool::GetAllocatedNodeCount() const {
  Mutex::Lock lock(mutex_);
  return kAllocationSize * allocations_.size() - pool_.size();
}

void NodePool::ReleaseSubtree(Node* node) {
  Mutex::Lock lock(mutex_);
  ReleaseSubtreeInternal(node);
}

void NodePool::ReleaseSubtreeInternal(Node* node) REQUIRES(mutex_) {
  for (Node* iter = node->child; iter; iter = iter->sibling) {
    ReleaseSubtreeInternal(iter);
    pool_.push_back(iter);
  }
}

void NodePool::AllocateNewBatch() REQUIRES(mutex_) {
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

int Node::ComputeRepetitions() {
  // TODO(crem) implement some sort of caching.
  if (no_capture_ply < 2) return 0;

  const Node* node = this;
  while (true) {
    node = node->parent;
    if (!node) break;
    node = node->parent;
    if (!node) break;

    if (node->board == board) {
      return 1 + node->repetitions;
    }
    if (node->no_capture_ply < 2) return 0;
  }
  return 0;
}

void NodeTree::MakeMove(Move move) {
  if (current_head_->board.flipped()) move.Mirror();

  Node* new_head = nullptr;
  for (Node* n = current_head_->child; n; n = n->sibling) {
    if (n->move == move) {
      new_head = n;
      break;
    }
  }
  node_pool_->ReleaseAllChildrenExceptOne(current_head_, new_head);
  if (!new_head) {
    new_head = node_pool_->GetNode();
    current_head_->child = new_head;
    new_head->parent = current_head_;
    new_head->board = current_head_->board;
    const bool capture = new_head->board.ApplyMove(move);
    new_head->board.Mirror();
    new_head->ply_count = current_head_->ply_count + 1;
    new_head->no_capture_ply = capture ? 0 : current_head_->no_capture_ply + 1;
    new_head->repetitions = new_head->ComputeRepetitions();
    new_head->move = move;
  }
  current_head_ = new_head;
}

void NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<Move>& moves) {
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);
  if (gamebegin_node_ && gamebegin_node_->board != starting_board) {
    // Completely different position.
    DeallocateTree();
    current_head_ = nullptr;
    gamebegin_node_ = nullptr;
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = node_pool_->GetNode();
    gamebegin_node_->board = starting_board;
    gamebegin_node_->no_capture_ply = no_capture_ply;
    gamebegin_node_->ply_count =
        full_moves * 2 - (starting_board.flipped() ? 1 : 2);
  }

  current_head_ = gamebegin_node_;
  for (const auto& move : moves) {
    MakeMove(move);
  }
  node_pool_->ReleaseChildren(current_head_);
}

void NodeTree::DeallocateTree() {
  node_pool_->ReleaseSubtree(gamebegin_node_);
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}

}  // namespace lczero