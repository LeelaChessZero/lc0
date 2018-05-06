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

#pragma once

#include <memory>
#include <mutex>
#include "chess/board.h"
#include "chess/position.h"
#include "mcts/callbacks.h"
#include "neural/writer.h"
#include "utils/mutex.h"

namespace lczero {

class Node;
class Node_Iterator {
 public:
  Node_Iterator(Node* node) : node_(node) {}
  Node* operator*() { return node_; }
  Node* operator->() { return node_; }
  bool operator==(Node_Iterator& other) { return node_ == other.node_; }
  bool operator!=(Node_Iterator& other) { return node_ != other.node_; }
  void operator++();

 private:
  Node* node_;
};

// TODO(mooskagh) This interface is ugly as a result of quick
// incaptulatiotation. Will fix.
class Node {
 public:
  // Allocates a new node and adds it to front of the children list.
  Node* CreateChild(Move m);
  V3TrainingData GetV3TrainingData(GameResult result,
                                   const PositionHistory& history) const;
  // Returns move from the point of new of player BEFORE the position.
  Move GetMove() const { return move_; }
  // Returns move, with optional flip.
  Move GetMove(bool flip) const;
  std::string DebugString() const;
  void ResetStats();

  Node* GetParent() const { return parent_; }

  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const { return n_in_flight_; }
  int GetNStarted() const { return n_ + n_in_flight_; }
  float GetQ() const { return n_ ? q_ : -parent_->q_; }
  // Returns U / (Puct * N[parent])
  float GetU() const { return p_ / (1 + n_ + n_in_flight_); }
  float GetV() const { return v_; }
  float GetP() const { return p_; }

  void SetV(float val) { v_ = val; }
  void SetP(float val) { p_ = val; }
  void SetTerminal(bool val) { is_terminal_ = val; }

  bool TryStartScoreUpdate() {
    if (n_ == 0 && n_in_flight_ > 0) return false;
    ++n_in_flight_;
    return true;
  }
  void CancelScoreUpdate() { --n_in_flight_; }
  void FinalizeScoreUpdate(float v) {
    // Add new value to W.
    w_ += v;
    // Increment N.
    ++n_;
    // Decrement virtual loss.
    --n_in_flight_;
    // Recompute Q.
    q_ = w_ / n_;
  }

  void UpdateMaxDepth(int depth) {
    if (depth > max_depth_) max_depth_ = depth;
  }

  bool UpdateFullDepth(uint16_t* depth) {
    if (full_depth_ > *depth) return false;
    for (Node* iter : Children()) {
      if (*depth > iter->full_depth_) *depth = iter->full_depth_;
    }
    if (*depth >= full_depth_) {
      full_depth_ = ++*depth;
      return true;
    }
    return false;
  }

  class NodeRange {
   public:
    Node_Iterator begin() { return Node_Iterator(node_); }
    Node_Iterator end() { return Node_Iterator(nullptr); }

   private:
    NodeRange(Node* node) : node_(node) {}
    Node* node_;
    friend class Node;
  };

  bool HasChildren() const { return child_ != nullptr; }
  NodeRange Children() const { return child_; }
  void AddNoise(float eps, float val) { p_ = p_ * (1 - eps) + val * eps; }
  void ScaleP(float scale) { p_ *= scale; }
  bool IsTerminal() const { return is_terminal_; }
  uint16_t GetFullDepth() const { return full_depth_; }
  uint16_t GetMaxDepth() const { return max_depth_; }

  class Pool;

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Q value fetched from neural network.
  float v_;
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. Terminal nodes (which lead to checkmate or draw) may be visited
  // several times, those are counted several times. q = w / n
  float q_;
  // Sum of values of all visited nodes in a subtree. Used to compute an
  // average.
  float w_;
  // Probabality that this move will be made. From policy head of the neural
  // network.
  float p_;
  // How many completed visits this node had.
  uint32_t n_;
  // (aka virtual loss). How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move.
  uint16_t n_in_flight_;

  // Maximum depth any subnodes of this node were looked at.
  uint16_t max_depth_;
  // Complete depth all subnodes of this node were fully searched.
  uint16_t full_depth_;
  // Does this node end game (with a winning of either sides or draw).
  bool is_terminal_;

  // Pointer to a parent node. nullptr for the root.
  Node* parent_;
  // Pointer to a first child. nullptr for leave node.
  Node* child_;
  // Pointer to a next sibling. nullptr if there are no further siblings.
  Node* sibling_;

  // TODO(mooskagh) Unfriend both NodeTree and Node::Pool.
  friend class NodeTree;
  friend class Node_Iterator;
};

inline void Node_Iterator::operator++() { node_ = node_->sibling_; }

class NodeTree {
 public:
  ~NodeTree() { DeallocateTree(); }
  // Adds a move to current_head_;
  void MakeMove(Move move);
  // Sets the position in a tree, trying to reuse the tree.
  void ResetToPosition(const std::string& starting_fen,
                       const std::vector<Move>& moves);
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_; }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  void DeallocateTree();
  Node* current_head_ = nullptr;
  Node* gamebegin_node_ = nullptr;
  PositionHistory history_;
};
}  // namespace lczero