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

#include <iterator>
#include <memory>
#include <mutex>
#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/position.h"
#include "neural/writer.h"
#include "utils/mutex.h"

namespace lczero {

class Node;
class Edge {
 public:
  // Returns move from the point of view of player BEFORE the position.
  Move GetMove() const { return move_; }

  // Returns move, with optional flip (false == player BEFORE the position).
  Move GetMove(bool flip) const;

  void SetMove(Move move) { move_ = move; }

  // Returns value of Move probability returned from the neural net
  // (but can be changed by adding Dirichlet noise).
  float GetP() const { return p_; }

  // Sets move probability.
  void SetP(float val) { p_ = val; }

  // Returns whether the edge is extended, i.e. it has corresponding node.
  bool HasNode() const { return has_node_; }

  // Debug information about the edge.
  std::string DebugString() const;

  Node* SpawnNode(Node* parent, std::unique_ptr<Node>* ptr);

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Probability that this move will be made. From policy head of the neural
  // network.
  float p_ = 0.0;

  // If true, there is a Node instance somewhere which this edge leads to.
  // If false, this edge is dangling.
  bool has_node_ = false;

  friend class Node;
};

class EdgeList {
 public:
  EdgeList() {}
  EdgeList(MoveList moves);
  void operator=(EdgeList&& other);
  ~EdgeList() { delete[] edges_; }
  Edge* get() const { return edges_; }
  Edge& operator[](size_t idx) { return edges_[idx]; }
  operator bool() const { return edges_ != nullptr; }
  uint16_t size() const { return size_; }

 private:
  Edge* edges_ = nullptr;
  uint16_t size_ = 0;
};

class EdgeAndNode;
class Node {
 public:
  Node(Node* parent) : parent_(parent) {}

  // Allocates a new edge and a new node. The node has to be no edges before
  // that. Not thread-friendly.
  Node* CreateSingleChildNode(Move m);

  void CreateEdges(const MoveList& moves) {
    assert(!edges_);
    edges_ = EdgeList(moves);
  }

  // Gets parent node.
  Node* GetParent() const { return parent_; }

  // Returns whether a node has children.
  bool HasChildren() const { return edges_; }

  // Returns sum of policy priors which have had at least one playout.
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const { return n_in_flight_; }
  uint32_t GetChildrenVisits() const { return n_ > 0 ? n_ - 1 : 1; }
  // Returns n = n_if_flight.
  int GetNStarted() const { return n_ + n_in_flight_; }
  // Returns Q if number of visits is more than 0.
  // TODO(crem) When node exists, n_ always > 0 (I guess?) so remove that
  //            parameter default_q.
  float GetQ(float default_q) const {
    if (n_ == 0) return default_q;
    return q_;
  }

  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return is_terminal_; }
  float GetTerminalNodeValue() const { return q_; }
  uint16_t GetFullDepth() const { return full_depth_; }
  uint16_t GetMaxDepth() const { return max_depth_; }
  uint16_t GetNumEdges() const { return edges_.size(); }

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
  // * Q (weighted average of all V in a subtree)
  // * N (+=1)
  // * N-in-flight (-=1)
  void FinalizeScoreUpdate(float v, float gamma, float beta);

  // Updates max depth, if new depth is larger.
  void UpdateMaxDepth(int depth);

  // Calculates the full depth if new depth is larger, updates it, returns
  // in depth parameter, and returns true if it was indeed updated.
  bool UpdateFullDepth(uint16_t* depth);

  V3TrainingData GetV3TrainingData(GameResult result,
                                   const PositionHistory& history) const;

  class EdgeRange;
  class NodeRange;
  class SpawnableEdgeRange;

  // Returns range for iterating over edges.
  EdgeRange Edges() const;
  SpawnableEdgeRange SpawnableEdges();
  uint16_t NumEdges() const { return edges_.size(); }

  // Returns range for iterating over edges.
  NodeRange ChildNodes() const;

  // Deletes all children.
  void ReleaseChildren();

  // Deletes all children except one.
  void ReleaseChildrenExceptOne(Node* node);

  EdgeAndNode GetEdgeToNode(const Node* node) const;

  // Debug information about the node.
  std::string DebugString() const;

 private:
  // Edges.
  EdgeList edges_;

  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored.
  float q_ = 0.0f;
  // How many completed visits this node had.
  uint32_t n_ = 0;
  // (aka virtual loss). How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move.
  uint16_t n_in_flight_ = 0;

  // Maximum depth any subnodes of this node were looked at.
  uint16_t max_depth_ = 0;
  // Complete depth all subnodes of this node were fully searched.
  uint16_t full_depth_ = 0;
  // Does this node end game (with a winning of either sides or draw).
  bool is_terminal_ = false;

  // Pointer to a parent node. nullptr for the root.
  Node* parent_ = nullptr;
  // Pointer to a first child. nullptr for leave node.
  std::unique_ptr<Node> child_;
  // Pointer to a next sibling. nullptr if there are no further siblings.
  std::unique_ptr<Node> sibling_;

  // TODO(mooskagh) Unfriend NodeTree.
  friend class NodeTree;
  friend class Edge_Iterator;
  friend class Node_Iterator;
  friend class Edge_SpawnableIterator;
  friend class Edge;
};

class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : node_(node), edge_(edge) {}
  operator bool() const { return edge_ != nullptr; }

  bool HasNode() const { return edge_->HasNode(); }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  // Proxy functions for easier access to node/edge.
  float GetQ(float default_q) const {
    return node_ ? node_->GetQ(default_q) : default_q;
  }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  float GetP() const { return edge_->GetP(); }
  Move GetMove() const { return edge_->GetMove(); }
  Move GetMove(bool flip) const { return edge_->GetMove(flip); }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }

  // Returns p / N, which is equal to U / (cpuct * sqrt(N[parent])) by the MCTS
  // equation. So it's really more of a "reduced U" than raw U.
  float GetU(float numerator) const {
    return numerator * GetP() / (1 + GetNStarted());
  }

 protected:
  Node* node_ = nullptr;

 private:
  Edge* edge_ = nullptr;
};

class Edge_Iterator
    : public std::iterator<std::forward_iterator_tag, EdgeAndNode> {
 public:
  Edge_Iterator(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  bool operator==(Edge_Iterator& other) { return edge_ == other.edge_; }
  bool operator!=(Edge_Iterator& other) { return edge_ != other.edge_; }
  void operator++() {
    if (edge_->HasNode()) node_ = node_->sibling_.get();
    ++edge_;
  }
  EdgeAndNode operator*() {
    return {edge_, edge_->HasNode() ? node_ : nullptr};
  }
  EdgeAndNode operator->() {
    return {edge_, edge_->HasNode() ? node_ : nullptr};
  }

 private:
  Edge* edge_;
  Node* node_;
};

class Node::EdgeRange {
 public:
  Edge_Iterator begin() { return {node_->edges_.get(), node_->child_.get()}; }
  Edge_Iterator end() {
    return {node_->edges_.get() + node_->NumEdges(), nullptr};
  }

 private:
  const Node* node_;
  EdgeRange(const Node* node) : node_(node) {}
  friend class Node;
};

class SpawnableEdgeAndNode : public EdgeAndNode {
 public:
  SpawnableEdgeAndNode() = default;
  SpawnableEdgeAndNode(Edge* edge, Node* node, std::unique_ptr<Node>* node_ptr)
      : EdgeAndNode(edge, node), node_ptr_(node_ptr) {}

  Node* SpawnNode(Node* parent) {
    node_ = edge()->SpawnNode(parent, node_ptr_);
    return node_;
  }

 private:
  std::unique_ptr<Node>* node_ptr_ = nullptr;
};

class Edge_SpawnableIterator
    : public std::iterator<std::forward_iterator_tag, EdgeAndNode> {
 public:
  Edge_SpawnableIterator(Edge* edge, std::unique_ptr<Node>* node_ptr)
      : edge_(edge), node_ptr_(node_ptr) {}
  bool operator==(Edge_SpawnableIterator& other) {
    return edge_ == other.edge_;
  }
  bool operator!=(Edge_SpawnableIterator& other) {
    return edge_ != other.edge_;
  }
  void operator++() {
    if (edge_->HasNode()) node_ptr_ = &(*node_ptr_)->sibling_;
    ++edge_;
  }
  SpawnableEdgeAndNode operator*() {
    return {edge_, edge_->HasNode() ? (*node_ptr_).get() : nullptr, node_ptr_};
  }
  SpawnableEdgeAndNode operator->() {
    return {edge_, edge_->HasNode() ? (*node_ptr_).get() : nullptr, node_ptr_};
  }

 private:
  Edge* edge_;
  std::unique_ptr<Node>* node_ptr_;
};

class Node::SpawnableEdgeRange {
 public:
  Edge_SpawnableIterator begin() {
    return {node_->edges_.get(), &node_->child_};
  }
  Edge_SpawnableIterator end() {
    return {node_->edges_.get() + node_->NumEdges(), nullptr};
  }

 private:
  SpawnableEdgeRange(Node* node) : node_(node) {}
  Node* node_;
  friend class Node;
};

class Node_Iterator {
 public:
  Node_Iterator(Node* node) : node_(node) {}
  Node* operator*() { return node_; }
  Node* operator->() { return node_; }
  bool operator==(Node_Iterator& other) { return node_ == other.node_; }
  bool operator!=(Node_Iterator& other) { return node_ != other.node_; }
  void operator++() { node_ = node_->sibling_.get(); }

 private:
  Node* node_;
};

class Node::NodeRange {
 public:
  Node_Iterator begin() { return Node_Iterator(node_); }
  Node_Iterator end() { return Node_Iterator(nullptr); }

 private:
  NodeRange(Node* node) : node_(node) {}
  Node* node_;
  friend class Node;
};

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
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  void DeallocateTree();
  // A node which to start search from.
  Node* current_head_ = nullptr;
  // Root node of a game tree.
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
};

}  // namespace lczero
