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
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include "neural/encoder.h"
#include "neural/network.h"
#include "utils/hashcat.h"

namespace lczero {

/////////////////////////////////////////////////////////////////////////
// Node
/////////////////////////////////////////////////////////////////////////

Node* Node::CreateChild(Move m) {
  auto new_node = std::make_unique<Node>();
  new_node->parent_ = this;
  new_node->sibling_ = std::move(child_);
  new_node->move_ = m;
  child_ = std::move(new_node);
  return child_.get();
}

float Node::GetVisitedPolicy() const {
  float res = 0.0f;
  for (const Node* node : Children()) {
    if (node->GetNStarted() > 0) res += node->GetP();
  }
  return res;
}

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.as_string() << " Term:" << is_terminal_
      << " This:" << this << " Parent:" << parent_ << " child:" << child_.get()
      << " sibling:" << sibling_.get() << " P:" << p_ << " Q:" << q_
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
  q_ = (result == GameResult::DRAW) ? 0.0f : 1.0f;
}

bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}

void Node::CancelScoreUpdate() { --n_in_flight_; }

void Node::FinalizeScoreUpdate(float v, float gamma, float beta) {
  // Recompute Q.
  q_ += (v - q_) / (std::pow(static_cast<float>(n_), gamma) * beta + 1);
  // Increment N.
  ++n_;
  // Decrement virtual loss.
  --n_in_flight_;
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

void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
  // Stores node which will have to survive (or nullptr if it's not found).
  std::unique_ptr<Node> saved_node;
  // Pointer to unique_ptr, so that we could move from it.
  for (std::unique_ptr<Node>* node = &child_; *node;
       node = &(*node)->sibling_) {
    // If current node is the one that we have to save.
    if (node->get() == node_to_save) {
      // Kill all remaining siblings.
      (*node)->sibling_.reset();
      // Save the node, and take the ownership from the unique_ptr.
      saved_node = std::move(*node);
      break;
    }
  }
  // Make saved node the only child. (kills previous siblings).
  child_ = std::move(saved_node);
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
  current_head_->ReleaseChildrenExceptOne(new_head);
  current_head_ = new_head ? new_head : current_head_->CreateChild(move);
  history_.Append(move);
}

void NodeTree::TrimTreeAtHead() { *current_head_ = Node(); }

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
    gamebegin_node_ = std::make_unique<Node>();
  }

  history_.Reset(starting_board, no_capture_ply,
                 full_moves * 2 - (starting_board.flipped() ? 1 : 2));

  Node* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const auto& move : moves) {
    MakeMove(move);
    if (old_head == current_head_) seen_old_head = true;
  }

  // If we didn't see old head, it means that new position is shorter.
  // As we killed the search tree already, trim it to redo the search.
  if (!seen_old_head) {
    assert(!current_head_->sibling_);
    TrimTreeAtHead();
  }
}

void NodeTree::DeallocateTree() {
  gamebegin_node_.reset();
  current_head_ = nullptr;
}

}  // namespace lczero
