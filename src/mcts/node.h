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
#include "chess/callbacks.h"
#include "chess/position.h"
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

class Node {
 public:
  // Resets all values (but not links to parents/children/siblings) to zero.
  void ResetStats();

  // Allocates a new node and adds it to front of the children list.
  // Not thread-friendly.
  Node* CreateChild(Move m);

  // Gets parent node.
  Node* GetParent() const { return parent_; }

  // Returns whether a node has children.
  bool HasChildren() const { return child_ != nullptr; }

  // Returns move from the point of view of player BEFORE the position.
  Move GetMove() const { return move_; }

  // Returns move, with optional flip (false == player BEFORE the position).
  Move GetMove(bool flip) const;

  // Returns sum of policy priors which have had at least one playout.
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const { return n_in_flight_; }
  uint32_t GetChildrenVisits() const { return n_ > 0 ? n_ - 1 : 1; }
  // Returns n = n_if_flight.
  int GetNStarted() const { return n_ + n_in_flight_; }
  // Returns Q if number of visits is more than 0,
  float GetQ(float default_q, float virtual_loss) const {
    if (n_ == 0) return default_q;
    if (virtual_loss && n_in_flight_) {
      return (w_ - n_in_flight_ * virtual_loss) /
             (n_ + n_in_flight_ * virtual_loss);
    } else {
      return q_;
    }
  }
  // Returns p / N, which is equal to U / (cpuct * sqrt(N[parent])) by the MCTS
  // equation. So it's really more of a "reduced U" than raw U.
  float GetU() const { return p_ / (1 + n_ + n_in_flight_); }
  // Returns value of Value Head returned from the neural net.
  float GetV() const { return v_; }
  // Returns value of Move probability returned from the neural net
  // (but can be changed by adding Dirichlet noise).
  float GetP() const { return p_; }
  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return is_terminal_; }
  uint16_t GetFullDepth() const { return full_depth_; }
  uint16_t GetMaxDepth() const { return max_depth_; }

  // Sets node own value (from neural net or win/draw/lose adjudication).
  void SetV(float val) { v_ = val; }
  // Sets move probability.
  void SetP(float val) { p_ = val; }
  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result);

  // If this node is not in the process of being expanded by another thread
  // (which can happen only if n==0 and n-in-flight==1), mark the node as
  // "being updated" by incrementing n-in-flight, and return true.
  // Otherwise return false.
  bool TryStartScoreUpdate();
  // Decrements n-in-flight back.
  void CancelScoreUpdate();
  // Updates the node with newly computed value v.
  // Updates:
  // * N (+=1)
  // * N-in-flight (-=1)
  // * W (+= v)
  // * Q (=w/n)
  void FinalizeScoreUpdate(float v);

  // Updates max depth, if new depth is larger.
  void UpdateMaxDepth(int depth);

  // Calculates the full depth if new depth is larger, updates it, returns
  // in depth parameter, and returns true if it was indeed updated.
  bool UpdateFullDepth(uint16_t* depth);

  V3TrainingData GetV3TrainingData(GameResult result,
                                   const PositionHistory& history) const;

  class NodeRange {
   public:
    Node_Iterator begin() { return Node_Iterator(node_); }
    Node_Iterator end() { return Node_Iterator(nullptr); }

   private:
    NodeRange(Node* node) : node_(node) {}
    Node* node_;
    friend class Node;
  };

  // Returns range for iterating over children.
  NodeRange Children() const { return child_; }

  // Debug information about the node.
  std::string DebugString() const;

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
  // Adds a move to current_head_.
  void MakeMove(Move move);
  // Resets the current head to ensure it doesn't carry over details from a
  // previous search.
  void TrimTreeAtHead();
  // Sets the position in a tree, trying to reuse the tree.
  // If @auto_garbage_collect, old tree is garbage collected immediately. (may
  // take some milliseconds)
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
