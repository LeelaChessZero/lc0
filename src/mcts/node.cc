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
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include "neural/encoder.h"
#include "neural/network.h"
#include "utils/hashcat.h"

namespace lczero {

/////////////////////////////////////////////////////////////////////////
// Node::Pool
/////////////////////////////////////////////////////////////////////////

namespace {
// How many nodes to allocate a once.
const int kAllocationSize = 1024 * 64;
// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;
}  // namespace

class Node::Pool {
 public:
  Pool();
  ~Pool();

  // Allocates a new node and initializes it with all zeros.
  Node* AllocateNode();

  // Release* function don't release trees immediately but rather schedule
  // release until when GarbageCollect() is called.
  // Releases all children of the node, except specified. Also updates pointers
  // accordingly.
  void ReleaseAllChildrenExceptOne(Node* root, Node* subtree);
  // Releases all children, but doesn't release the node isself.
  void ReleaseChildren(Node*);
  // Releases all children and the node itself;
  void ReleaseSubtree(Node*);
  // Really releases subtrees makerd for release earlier.
  void GarbageCollect();

 private:
  // Runs garbabe collection every kGCIntervalMs milliseconds.
  void GarbageCollectThread();
  // Allocates a new set of nodes of size kAllocationSize and puts it into
  // reserve_list_.
  void AllocateNewBatch();

  union FreeNode {
    FreeNode* next;
    Node node;

    FreeNode() {}
  };

  static FreeNode* UnrollNodeTree(FreeNode* node);

  mutable Mutex mutex_;
  // Linked list of free nodes.
  FreeNode* free_list_ GUARDED_BY(mutex_) = nullptr;

  // Mutex for slow but rare operations.
  mutable Mutex allocations_mutex_ ACQUIRED_AFTER(mutex_);
  FreeNode* reserve_list_ GUARDED_BY(allocations_mutex_) = nullptr;
  std::vector<std::unique_ptr<FreeNode[]>> allocations_
      GUARDED_BY(allocations_mutex_);

  mutable Mutex gc_mutex_;
  std::vector<Node*> subtrees_to_gc_ GUARDED_BY(gc_mutex_);

  // Should garbage colletion thread stop?
  volatile bool stop_ = false;
  std::thread gc_thread_;
};

Node::Pool::Pool() : gc_thread_([this]() { GarbageCollectThread(); }) {}
Node::Pool::~Pool() {
  stop_ = true;
  gc_thread_.join();
}

void Node::Pool::GarbageCollectThread() {
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
    GarbageCollect();
  };
}

Node::Pool::FreeNode* Node::Pool::UnrollNodeTree(FreeNode* node) {
  if (!node->node.child_) return node;
  FreeNode* prev = node;
  for (Node* child = node->node.child_; child; child = child->sibling_) {
    FreeNode* next = reinterpret_cast<FreeNode*>(child);
    prev->next = next;
    prev = UnrollNodeTree(next);
  }
  return prev;
}

void Node::Pool::GarbageCollect() {
  while (true) {
    Node* node_to_gc = nullptr;
    {
      Mutex::Lock lock(gc_mutex_);
      if (subtrees_to_gc_.empty()) return;
      node_to_gc = subtrees_to_gc_.back();
      subtrees_to_gc_.pop_back();
    }
    FreeNode* head = reinterpret_cast<FreeNode*>(node_to_gc);
    FreeNode* tail = UnrollNodeTree(head);
    {
      Mutex::Lock lock(mutex_);
      tail->next = free_list_;
      free_list_ = head;
    }
  }
}

Node* Node::Pool::AllocateNode() {
  while (true) {
    Node* result = nullptr;
    {
      Mutex::Lock lock(mutex_);
      // Try to pick from a head of the freelist.
      if (free_list_) {
        result = &free_list_->node;
        free_list_ = free_list_->next;
      } else {
        // Free list empty. Trying to make reserve list free list.
        Mutex::Lock lock(allocations_mutex_);
        if (reserve_list_) {
          free_list_ = reserve_list_;
          reserve_list_ = nullptr;
        }
      }
    }

    // Have node! Return.
    if (result) {
      std::memset(reinterpret_cast<void*>(result), 0, sizeof(Node));
      return result;
    }

    {
      Mutex::Lock lock(allocations_mutex_);
      // Reserve is empty now, so unless another thread did that, we have to
      // rebuild a new reserve.
      if (!reserve_list_) AllocateNewBatch();
    }
    // Repeat again, now as we have reserve list and (possibly) free list.
  }
}

void Node::Pool::AllocateNewBatch() REQUIRES(allocations_mutex_) {
  allocations_.emplace_back(std::make_unique<FreeNode[]>(kAllocationSize));

  FreeNode* new_nodes = allocations_.back().get();
  for (int i = 0; i < kAllocationSize; ++i) {
    FreeNode* n = new_nodes + i;
    n->next = reserve_list_;
    reserve_list_ = n;
  }
}

void Node::Pool::ReleaseChildren(Node* node) {
  Node* next = node->child_;
  // Iterating manually rather than with iterator, as node is released in the
  // middle and can be taken by other threads, so we have to be careful.
  while (next) {
    Node* iter = next;
    // Getting next after releasing node, as otherwise it can be reallocated
    // and overwritten.
    next = next->sibling_;
    ReleaseSubtree(iter);
  }
  node->child_ = nullptr;
}

void Node::Pool::ReleaseAllChildrenExceptOne(Node* root, Node* subtree) {
  Node* child = nullptr;
  Node* next = root->child_;
  while (next) {
    Node* iter = next;
    // Getting next after releasing node, as otherwise it can be reallocated
    // and overwritten.
    next = next->sibling_;
    if (iter == subtree) {
      child = iter;
    } else {
      ReleaseSubtree(iter);
    }
  }
  root->child_ = child;
  if (child) {
    child->sibling_ = nullptr;
  }
}

void Node::Pool::ReleaseSubtree(Node* node) {
  Mutex::Lock lock(gc_mutex_);
  subtrees_to_gc_.push_back(node);
}

Node::Pool gNodePool;

/////////////////////////////////////////////////////////////////////////
// Node
/////////////////////////////////////////////////////////////////////////

Node* Node::CreateChild(Move m) {
  Node* new_node = gNodePool.AllocateNode();
  new_node->parent_ = this;
  new_node->sibling_ = child_;
  new_node->move_ = m;
  child_ = new_node;
  return new_node;
}

float Node::GetVisitedPolicy() const {
  float res = 0.0f;
  for (const Node* node : Children()) {
    if (node->GetNStarted() > 0) res += node->GetP();
  }
  return res;
}

void Node::ResetStats() {
  n_in_flight_ = 0;
  n_ = 0;
  v_ = 0.0;
  q_ = 0.0;
  w_ = 0.0;
  p_ = 0.0;
  max_depth_ = 0;
  full_depth_ = 0;
  is_terminal_ = false;
}

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.as_string() << " Term:" << is_terminal_
      << " This:" << this << " Parent:" << parent_ << " child:" << child_
      << " sibling:" << sibling_ << " P:" << p_ << " Q:" << q_ << " W:" << w_
      << " N:" << n_ << " N_:" << n_in_flight_;
  return oss.str();
}

Move Node::GetMove(bool flip) const {
  if (!flip) return move_;
  Move m = move_;
  m.Mirror();
  return m;
}

void Node::MakeTerminal(GameResult result) {
  is_terminal_ = true;
  v_ = (result == GameResult::DRAW) ? 0.0f : 1.0f;
}

bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}

void Node::CancelScoreUpdate() { --n_in_flight_; }

void Node::FinalizeScoreUpdate(float v) {
  // Add new value to W.
  w_ += v;
  // Increment N.
  ++n_;
  // Decrement virtual loss.
  --n_in_flight_;
  // Recompute Q.
  q_ = w_ / n_;
}

void Node::UpdateMaxDepth(int depth) {
  if (depth > max_depth_) max_depth_ = depth;
}

bool Node::UpdateFullDepth(uint16_t* depth) {
  if (full_depth_ > *depth) return false;
  for (Node* child : Children()) {
    if (*depth > child->full_depth_) *depth = child->full_depth_;
  }
  if (*depth >= full_depth_) {
    full_depth_ = ++*depth;
    return true;
  }
  return false;
}

namespace {
// Reverse bits in every byte of a number
uint64_t ReverseBitsInBytes(uint64_t v) {
  v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
  v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
  v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
  return v;
}
}  // namespace

V3TrainingData Node::GetV3TrainingData(GameResult game_result,
                                       const PositionHistory& history) const {
  V3TrainingData result;

  // Set version.
  result.version = 3;

  // Populate probabilities.
  float total_n =
      static_cast<float>(n_ - 1);  // First visit was expansion of it inself.
  std::memset(result.probabilities, 0, sizeof(result.probabilities));
  for (Node* child : Children()) {
    result.probabilities[child->move_.as_nn_index()] = child->n_ / total_n;
  }

  // Populate planes.
  InputPlanes planes = EncodePositionForNN(history, 8);
  int plane_idx = 0;
  for (auto& plane : result.planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  const auto& position = history.Last();
  // Populate castlings.
  result.castling_us_ooo = position.CanCastle(Position::WE_CAN_OOO) ? 1 : 0;
  result.castling_us_oo = position.CanCastle(Position::WE_CAN_OO) ? 1 : 0;
  result.castling_them_ooo = position.CanCastle(Position::THEY_CAN_OOO) ? 1 : 0;
  result.castling_them_oo = position.CanCastle(Position::THEY_CAN_OO) ? 1 : 0;

  // Other params.
  result.side_to_move = position.IsBlackToMove() ? 1 : 0;
  result.move_count = 0;
  result.rule50_count = position.GetNoCapturePly();

  // Game result.
  if (game_result == GameResult::WHITE_WON) {
    result.result = position.IsBlackToMove() ? -1 : 1;
  } else if (game_result == GameResult::BLACK_WON) {
    result.result = position.IsBlackToMove() ? 1 : -1;
  } else {
    result.result = 0;
  }

  return result;
}

void NodeTree::MakeMove(Move move) {
  if (HeadPosition().IsBlackToMove()) move.Mirror();

  Node* new_head = nullptr;
  for (Node* n : current_head_->Children()) {
    if (n->GetMove() == move) {
      new_head = n;
      break;
    }
  }
  gNodePool.ReleaseAllChildrenExceptOne(current_head_, new_head);
  current_head_ = new_head ? new_head : current_head_->CreateChild(move);
  history_.Append(move);
}

void NodeTree::TrimTreeAtHead() {
  gNodePool.ReleaseChildren(current_head_);
  current_head_->ResetStats();
}

void NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<Move>& moves) {
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);
  if (gamebegin_node_ && history_.Starting().GetBoard() != starting_board) {
    // Completely different position.
    DeallocateTree();
    current_head_ = nullptr;
    gamebegin_node_ = nullptr;
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = gNodePool.AllocateNode();
  }

  history_.Reset(starting_board, no_capture_ply,
                 full_moves * 2 - (starting_board.flipped() ? 1 : 2));

  Node* old_head = current_head_;
  current_head_ = gamebegin_node_;
  bool seen_old_head = (gamebegin_node_ == old_head);
  for (const auto& move : moves) {
    MakeMove(move);
    if (old_head == current_head_) seen_old_head = true;
  }

  // If we didn't see old head, it means that new position is shorter.
  // As we killed the search tree already, trim it to redo the search.
  if (!seen_old_head) {
    assert(!current_head_->sibling_);
    gNodePool.ReleaseChildren(current_head_);
    current_head_->ResetStats();
  }
}

void NodeTree::DeallocateTree() {
  gNodePool.ReleaseSubtree(gamebegin_node_);
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}

}  // namespace lczero
