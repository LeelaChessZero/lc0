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
  // return board.Hash();
  return HashCat({board.Hash(), no_capture_ply, repetitions});
}

void Node::ResetStats() {
  n_in_flight = 0;
  n = 0;
  v = 0.0;
  q = 0.0;
  w = 0.0;
  p = 0.0;
  max_depth = 0;
  full_depth = 0;
  is_terminal = 0;
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

Move Node::GetMoveAsWhite() const {
  Move m = move;
  if (!board.flipped()) m.Mirror();
  return m;
}

namespace {
const int kMoveHistory = 8;
const int kPlanesPerBoard = 13;
const int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
}  // namespace

InputPlanes Node::EncodeForNN() const {
  InputPlanes result(kAuxPlaneBase + 8);

  const bool we_are_black = board.flipped();
  if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
  if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
  if (board.castlings().they_can_000()) result[kAuxPlaneBase + 2].SetAll();
  if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
  if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
  result[kAuxPlaneBase + 5].Fill(no_capture_ply);

  const Node* node = this;
  bool flip = false;
  for (int i = 0; i < kMoveHistory; ++i, flip = !flip) {
    if (!node) break;
    ChessBoard board = node->board;
    if (flip) board.Mirror();

    const int base = i * kPlanesPerBoard;
    result[base + 0].mask = (board.ours() * board.pawns()).as_int();
    result[base + 1].mask = (board.our_knights()).as_int();
    result[base + 2].mask = (board.ours() * board.bishops()).as_int();
    result[base + 3].mask = (board.ours() * board.rooks()).as_int();
    result[base + 4].mask = (board.ours() * board.queens()).as_int();
    result[base + 5].mask = (board.our_king()).as_int();

    result[base + 6].mask = (board.theirs() * board.pawns()).as_int();
    result[base + 7].mask = (board.their_knights()).as_int();
    result[base + 8].mask = (board.theirs() * board.bishops()).as_int();
    result[base + 9].mask = (board.theirs() * board.rooks()).as_int();
    result[base + 10].mask = (board.theirs() * board.queens()).as_int();
    result[base + 11].mask = (board.their_king()).as_int();

    const int repetitions = node->repetitions;
    if (repetitions >= 1) result[base + 12].SetAll();

    node = node->parent;
  }

  return result;
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

V3TrainingData Node::GetV3TrainingData(GameInfo::GameResult game_result) const {
  V3TrainingData result;

  // Set version.
  result.version = 3;

  // Populate probabilities.
  float total_n = n - 1;  // First visit was expansion of it inself.
  std::memset(result.probabilities, 0, sizeof(result.probabilities));
  for (Node* iter = child; iter; iter = iter->sibling) {
    result.probabilities[iter->move.as_nn_index()] = iter->n / total_n;
  }

  // Populate planes.
  InputPlanes planes = EncodeForNN();
  int plane_idx = 0;
  for (auto& plane : result.planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  // Populate castlings.
  result.castling_us_ooo = board.castlings().we_can_000() ? 1 : 0;
  result.castling_us_oo = board.castlings().we_can_00() ? 1 : 0;
  result.castling_them_ooo = board.castlings().they_can_000() ? 1 : 0;
  result.castling_them_oo = board.castlings().they_can_00() ? 1 : 0;

  // Other params.
  result.side_to_move = board.flipped() ? 1 : 0;
  result.move_count = 0;
  result.rule50_count = no_capture_ply;

  // Game result.
  if (game_result == GameInfo::WHITE_WON) {
    result.result = board.flipped() ? -1 : 1;
  } else if (game_result == GameInfo::BLACK_WON) {
    result.result = board.flipped() ? 1 : -1;
  } else {
    result.result = 0;
  }

  return result;
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
    assert(!current_head_->sibling);
    node_pool_->ReleaseChildren(current_head_);
    current_head_->ResetStats();
  }
}

void NodeTree::DeallocateTree() {
  node_pool_->ReleaseSubtree(gamebegin_node_);
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}

}  // namespace lczero