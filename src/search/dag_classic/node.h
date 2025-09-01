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

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>

#if __cpp_lib_atomic_wait < 201907L
#define NO_STD_ATOMIC_WAIT 1
#include <condition_variable>
#endif

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/gamestate.h"
#include "chess/position.h"
#include "neural/backend.h"
#include "utils/mutex.h"

namespace lczero {
namespace dag_classic {

// Terminology:
// * Edge - a potential edge with a move and policy information.
// * Node - an existing edge with number of visits and evaluation.
// * LowNode - a node with number of visits, evaluation and edges.
//
// Storage:
// * Potential edges are stored in a simple array inside the LowNode as edges_.
// * Existing edges are stored in a linked list starting with a child_ pointer
//   in the LowNode and continuing with a sibling_ pointer in each Node.
// * Existing edges have a copy of their potential edge counterpart, index_
//   among potential edges and are linked to the target LowNode via the
//   low_node_ pointer.
//
// Example:
//                                 LowNode
//                                    |
//        +-------------+-------------+----------------+--------------+
//        |              |            |                |              |
//   Edge 0(Nf3)    Edge 1(Bc5)     Edge 2(a4)     Edge 3(Qxf7)    Edge 4(a3)
//    (dangling)         |           (dangling)        |           (dangling)
//                   Node, Q=0.5                    Node, Q=-0.2
//
//  Is represented as:
// +-----------------+
// | LowNode         |
// +-----------------+                                        +--------+
// | edges_          | -------------------------------------> | Edge[] |
// |                 |    +------------+                      +--------+
// | child_          | -> | Node       |                      | Nf3    |
// |                 |    +------------+                      | Bc5    |
// | ...             |    | edge_      |                      | a4     |
// |                 |    | index_ = 1 |                      | Qxf7   |
// |                 |    | q_ = 0.5   |    +------------+    | a3     |
// |                 |    | sibling_   | -> | Node       |    +--------+
// |                 |    +------------+    +------------+
// |                 |                      | edge_      |
// +-----------------+                      | index_ = 3 |
//                                          | q_ = -0.2  |
//                                          | sibling_   | -> nullptr
//                                          +------------+

// Define __i386__  or __arm__ also for 32 bit Windows.
#if defined(_M_IX86)
#define __i386__
#endif
#if defined(_M_ARM) && !defined(_M_AMD64)
#define __arm__
#endif

// Atomic unique_ptr based on the public domain code from
// https://stackoverflow.com/a/42811152 .
template <class T>
class atomic_unique_ptr {
  using pointer = T*;
  using unique_pointer = std::unique_ptr<T>;

 public:
  // Manage no pointer.
  constexpr atomic_unique_ptr() noexcept : ptr() {}

  // Make pointer @p managed.
  explicit atomic_unique_ptr(pointer p) noexcept : ptr(p) {}

  // Move the managed pointer ownership from another atomic_unique_ptr.
  atomic_unique_ptr(atomic_unique_ptr&& p) noexcept : ptr(p.release()) {}
  // Move the managed pointer ownership from another atomic_unique_ptr.
  atomic_unique_ptr& operator=(atomic_unique_ptr&& p) noexcept {
    reset(p.release());
    return *this;
  }

  // Move the managed object ownership from a unique_ptr.
  atomic_unique_ptr(unique_pointer&& p) noexcept : ptr(p.release()) {}
  // Move the managed object ownership from a unique_ptr.
  atomic_unique_ptr& operator=(unique_pointer&& p) noexcept {
    reset(p.release());
    return *this;
  }

  // Replace the managed pointer, deleting the old one.
  void reset(pointer p = pointer()) noexcept {
    auto old = ptr.exchange(p, std::memory_order_acq_rel);
    if (old) delete old;
  }
  // Release ownership of and delete the owned pointer.
  ~atomic_unique_ptr() { reset(); }

  // Returns the managed pointer.
  operator pointer() const noexcept { return get(); }
  // Returns the managed pointer.
  pointer operator->() const noexcept { return get(); }
  // Returns the managed pointer.
  pointer get() const noexcept {
    return ptr.load(std::memory_order_acquire);
  }

  // Checks whether there is a managed pointer.
  explicit operator bool() const noexcept { return get() != pointer(); }

  // Replace the managed pointer, only releasing returning the old one.
  pointer set(pointer p = pointer()) noexcept {
    return ptr.exchange(p, std::memory_order_acq_rel);
  }
  // Return the managed pointer and release its ownership.
  pointer release() noexcept { return set(pointer()); }

  // Move managed pointer from @source, iff the managed pointer equals
  // @expected.
  bool compare_exchange(pointer& expected,
                        atomic_unique_ptr<T>& source) noexcept {
    if (ptr.compare_exchange_strong(expected, source.get(),
                                    std::memory_order_acq_rel)) {
      source.release();
      return true;
    } else {
      return false;
    }
  }

 private:
  std::atomic<pointer> ptr;
};

class Node;
class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  // Debug information about the edge.
  std::string DebugString() const;

  static void SortEdges(Edge* edges, int num_edges);

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;
  friend class Node;
};

struct Eval {
  float wl;
  float d;
  float ml;
};

struct NNEval {
  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Array of edges.
  std::unique_ptr<Edge[]> edges;

  // 4 byte fields.
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;

  // 1 byte fields.
  // Number of edges in @edges.
  uint8_t num_edges = 0;
};

typedef std::pair<GameResult, GameResult> Bounds;

enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase };

class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;

template <bool is_const>
class VisitedNode_Iterator;

class NodeGarbageCollector;
class ReleaseNodesWork;

class LowNode;
class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  // Takes own @index in the parent.
  Node(uint16_t index)
      : index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        repetition_(false) {}
  // Takes own @edge and @index in the parent.
  Node(const Edge& edge, uint16_t index)
      : edge_(edge),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        repetition_(false) {}
  ~Node();

  // Trim node, resetting everything except parent, sibling, edge and index.
  void Trim();

  // Allocates a new edge and a new node. The node has to be without edges
  // before that.
  Node* CreateSingleChildNode(Move move) {
    assert(!low_node_);
    auto low_node = std::make_shared<LowNode>(MoveList({move}), 0);
    SetLowNode(low_node);
    return GetChild();
  }

  // Get first child.
  Node* GetChild() const;
  // Get next sibling.
  atomic_unique_ptr<Node>* GetSibling() { return &sibling_; }
  // Moves sibling in.
  void MoveSiblingIn(std::unique_ptr<Node>& sibling) {
    sibling_ = std::move(sibling);
  }

  // Returns whether a node has children.
  bool HasChildren() const;

  // Returns sum of policy priors which have had at least one playout.
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const;
  uint32_t GetChildrenVisits() const;
  uint32_t GetTotalVisits() const;
  // Returns n + n_in_flight.
  int GetNStarted() const { return n_ + GetNInFlight(); }

  float GetQ(float draw_score) const { return wl_ + draw_score * d_; }
  // Returns node eval, i.e. average subtree V for non-terminal node and -1/0/1
  // for terminal nodes.
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }

  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }

  uint8_t GetNumEdges() const;

  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result, float plies_left = 1.0f,
                    Terminal type = Terminal::EndOfGame);
  // Makes the node not terminal and recomputes bounds, visits and values.
  // Changes low node as well unless @also_low_node is false.
  void MakeNotTerminal(bool also_low_node = true);
  void SetBounds(GameResult lower, GameResult upper);

  // If this node is not in the process of being expanded by another thread
  // (which can happen only if n==0 and n-in-flight==1), mark the node as
  // "being updated" by incrementing n-in-flight, and return true.
  // Otherwise return false.
  bool TryStartScoreUpdate();
  // Decrements n-in-flight back.
  void CancelScoreUpdate(uint32_t multivisit);
  // Updates the node with newly computed value v.
  // Updates:
  // * Q (weighted average of all V in a subtree)
  // * N (+=multivisit)
  // * N-in-flight (-=multivisit)
  void FinalizeScoreUpdate(float v, float d, float m, uint32_t multivisit);
  // Like FinalizeScoreUpdate, but it updates n existing visits by delta amount.
  void AdjustForTerminal(float v, float d, float m, uint32_t multivisit);
  // When search decides to treat one visit as several (in case of collisions
  // or visiting terminal nodes several times), it amplifies the visit by
  // incrementing n_in_flight.
  void IncrementNInFlight(uint32_t multivisit);

  // Returns range for iterating over edges.
  ConstIterator Edges() const;
  Iterator Edges();

  // Returns range for iterating over child nodes with N > 0.
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();

  // Deletes all children except one.
  // The node provided may be moved, so should not be relied upon to exist
  // afterwards.
  void ReleaseChildrenExceptOne(Node* node_to_save) const;

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const {
    return edge_.GetMove(as_opponent);
  }
  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise or when turning terminal).
  // Must be in [0,1].
  float GetP() const { return edge_.GetP(); }
  void SetP(float val) { edge_.SetP(val); }

  const std::shared_ptr<LowNode>& GetLowNode() const { return low_node_; }

  void SetLowNode(std::shared_ptr<LowNode> low_node);
  void UnsetLowNode();

  // Debug information about the node.
  std::string DebugString() const;
  // Return string describing the edge from node's parent to its low node in the
  // Graphviz dot format.
  void DotEdgeString(std::ofstream& file,
                     bool as_opponent = false,
                     const LowNode* parent = nullptr) const;
  // Return string describing the graph starting at this node in the Graphviz
  // dot format.
  void DotGraphString(std::ofstream& file, bool as_opponent = false) const;

  // Returns true if graph under this node has every n_in_flight_ == 0 and
  // prints offending nodes and low nodes and stats to cerr otherwise.
  bool ZeroNInFlight() const;

  void SortEdges() const;

  // Index in parent's edges - useful for correlated ordering.
  uint16_t Index() const { return index_; }

  void SetRepetition() { repetition_ = true; }
  bool IsRepetition() const { return repetition_; }

  bool WLDMInvariantsHold() const;

#ifndef NDEBUG
  // RAII holder was a visitor. It will automatically release the reservation
  // when going out of scope. It is possible to use visitor for branches. There
  // must be a full tree walk before id value wraps arround or walk will ignore
  // some nodes.
  // It doesn't support concurrent access currently. API emulates mutexes which
  // makes it possible to add limited number of concurrent access and waiting
  // for free resources if needed.
  struct VisitorId {
    using type = uint32_t;
    using storage = uint32_t;

    VisitorId(const VisitorId&) = delete;

    explicit VisitorId();
    ~VisitorId();

    operator type() const {
      return id_;
    }

    friend class Node;
    friend class LowNode;
  private:
    type id_;
  };
#endif

 private:
  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // 16 byte fields on 64-bit platforms, 8 byte on 32-bit.
  // Shared pointer to the low node.
  std::shared_ptr<LowNode> low_node_;

  // 8 byte fields.
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored. This is from the perspective
  // of the player who "just" moved to reach this position, rather than from
  // the perspective of the player-to-move for the position. WL stands for "W
  // minus L". Is equal to Q if draw score is 0.
  double wl_ = 0.0f;
  // Averaged draw probability. Works similarly to WL, except that D is not
  // flipped depending on the side to move.
  double d_ = 0.0f;

  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Pointer to a next sibling. nullptr if there are no further siblings.
  atomic_unique_ptr<Node> sibling_;

  // 4 byte fields.
  // Estimated remaining plies.
  float m_ = 0.0f;
  // How many completed visits this node had.
  uint32_t n_ = 0;
  // (AKA virtual loss.) How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move.
  std::atomic<uint32_t> n_in_flight_ = 0;

  // Move and policy for this edge.
  Edge edge_;

  // 2 byte fields.
  // Index of this node is parent's edge list.
  uint16_t index_;

  // 1 byte fields.
  // Bit fields using parts of uint8_t fields initialized in the constructor.
  // Whether or not this node end game (with a winning of either sides or
  // draw).
  Terminal terminal_type_ : 2;
  // Best and worst result for this node.
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  // Edge was handled as a repetition at some point.
  bool repetition_ : 1;
};

// Check that Node still fits into an expected cache line size.
static_assert(sizeof(Node) <= 64, "Node is too large");

class LowNode {
 public:
  LowNode()
      : terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON) {}
  // Init from from another low node, but use it for NNEval only.
  LowNode(const LowNode& p)
      : wl_(p.wl_),
        d_(p.d_),
        m_(p.m_),
        num_edges_(p.num_edges_),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON) {
    assert(p.edges_);
    edges_ = std::make_unique<Edge[]>(num_edges_);
    std::memcpy(edges_.get(), p.edges_.get(), num_edges_ * sizeof(Edge));
  }
  // Init @edges_ with moves from @moves and 0 policy.
  LowNode(const MoveList& moves)
      : num_edges_(moves.size()),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON) {
    edges_ = Edge::FromMovelist(moves);
  }
  // Init @edges_ with moves from @moves and 0 policy.
  // Also create the first child at @index.
  LowNode(const MoveList& moves, uint16_t index)
      : num_edges_(moves.size()),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON) {
    edges_ = Edge::FromMovelist(moves);
    child_ = std::make_unique<Node>(edges_[index], index);
  }
  ~LowNode();

  void SetNNEval(const EvalResult* eval) {
    assert(n_ == 0);
    assert(!child_);

    for (size_t idx = 0; idx < num_edges_; idx++) {
      edges_.get()[idx].SetP(eval->p[idx]);
    }

    wl_ = eval->q;
    d_ = eval->d;
    m_ = eval->m;

    assert(WLDMInvariantsHold());
  }

  // Gets the first child.
  atomic_unique_ptr<Node>* GetChild() { return &child_; }

  // Returns whether a node has children.
  bool HasChildren() const { return num_edges_ > 0; }

  uint32_t GetN() const { return n_; }
  uint32_t GetChildrenVisits() const { return n_ - 1; }

  // Returns node eval, i.e. average subtree V for non-terminal node and -1/0/1
  // for terminal nodes.
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }

  // Returns whether the node is known to be draw/loss/win.
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  Terminal GetTerminalType() const { return terminal_type_; }

  uint8_t GetNumEdges() const { return num_edges_; }
  // Gets pointer to the start of the edge array.
  Edge* GetEdges() const { return edges_.get(); }

  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  // Makes the low node not terminal and recomputes bounds, visits and values
  // using incoming @node.
  void MakeNotTerminal(const Node* node);
  void SetBounds(GameResult lower, GameResult upper);

  // Decrements n-in-flight back.
  void CancelScoreUpdate(uint32_t multivisit);
  // Updates the node with newly computed value v.
  // Updates:
  // * Q (weighted average of all V in a subtree)
  // * N (+=multivisit)
  // * N-in-flight (-=multivisit)
  void FinalizeScoreUpdate(float v, float d, float m, uint32_t multivisit);
  // Like FinalizeScoreUpdate, but it updates n existing visits by delta amount.
  void AdjustForTerminal(float v, float d, float m, uint32_t multivisit);

  // Deletes all children.
  void ReleaseChildren();

  // Deletes all children except one.
  // The node provided may be moved, so should not be relied upon to exist
  // afterwards.
  void ReleaseChildrenExceptOne(Node* node_to_save);

  // Return move policy for edge/node at @index.
  const Edge& GetEdgeAt(uint16_t index) const;

  // Debug information about the node.
  std::string DebugString() const;
  // Return string describing this node in the Graphviz dot format.
  void DotNodeString(std::ofstream& file) const;

  void SortEdges() {
    assert(edges_);
    assert(!child_);
    Edge::SortEdges(edges_.get(), num_edges_);
  }

  // Add new parent with @n_in_flight visits.
  void AddParent() {
    num_parents_.fetch_add(1, std::memory_order_acq_rel);

    assert(num_parents_ > 0);
  }
  // Remove parent and its first visit.
  void RemoveParent() {
    assert(num_parents_ > 0);
    num_parents_.fetch_sub(1, std::memory_order_acq_rel);
  }
  bool IsTransposition() const {
    return num_parents_.load(std::memory_order_acquire) > 1;
  }

  bool WLDMInvariantsHold() const;

#ifndef NDEBUG
  bool Visit(Node::VisitorId::type id);
#endif

 private:
  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // 8 byte fields.
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored. This is from the perspective
  // of the player who "just" moved to reach this position, rather than from the
  // perspective of the player-to-move for the position.
  // WL stands for "W minus L". Is equal to Q if draw score is 0.
  double wl_ = 0.0f;
  // Averaged draw probability. Works similarly to WL, except that D is not
  // flipped depending on the side to move.
  double d_ = 0.0f;

  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Array of edges.
  std::unique_ptr<Edge[]> edges_;
  // Pointer to the first child. nullptr when no children.
  atomic_unique_ptr<Node> child_;

  // 4 byte fields.
  // Estimated remaining plies.
  float m_ = 0.0f;
  // How many completed visits this node had.
  uint32_t n_ = 0;

  // 2 byte fields.
  // Number of parents.
  std::atomic<uint16_t> num_parents_ = {};

  // 1 byte fields.
  // Number of edges in @edges_.
  uint8_t num_edges_ = 0;
  // Bit fields using parts of uint8_t fields initialized in the constructor.
  // Whether or not this node end game (with a winning of either sides or draw).
  Terminal terminal_type_ : 2;
  // Best and worst result for this node.
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  // Debug only id as the last to avoid taking place of actively used variables
  // in the cache.
#ifndef NDEBUG
  Node::VisitorId::storage visitor_id_ = {};
#endif
};

// Check that LowNode still fits into an expected cache line size.
static_assert(sizeof(LowNode) <= 64, "LowNode is too large");

// Contains Edge and Node pair and set of proxy functions to simplify access
// to them.
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
  bool operator==(const EdgeAndNode& other) const {
    return edge_ == other.edge_;
  }
  bool operator!=(const EdgeAndNode& other) const {
    return edge_ != other.edge_;
  }
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  // Proxy functions for easier access to node/edge.
  float GetQ(float default_q, float draw_score) const {
    return (node_ && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q;
  }
  float GetWL(float default_wl) const {
    return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl;
  }
  float GetD(float default_d) const {
    return (node_ && node_->GetN() > 0) ? node_->GetD() : default_d;
  }
  float GetM(float default_m) const {
    return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m;
  }
  // N-related getters, from Node (if exists).
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }

  // Whether the node is known to be terminal.
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Bounds GetBounds() const {
    return node_ ? node_->GetBounds()
                 : Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON};
  }

  // Edge related getters.
  float GetP() const {
    return node_ != nullptr ? node_->GetP() : edge_->GetP();
  }
  Move GetMove(bool flip = false) const {
    return edge_ ? edge_->GetMove(flip) : Move();
  }

  // Returns U = numerator * p / N.
  // Passed numerator is expected to be equal to (cpuct * sqrt(N[parent])).
  float GetU(float numerator) const {
    return numerator * GetP() / (1 + GetNStarted());
  }

  std::string DebugString() const;

 protected:
  // nullptr means that the whole pair is "null". (E.g. when search for a node
  // didn't find anything, or as end iterator signal).
  Edge* edge_ = nullptr;
  // nullptr means that the edge doesn't yet have node extended.
  Node* node_ = nullptr;
};

// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// * Element, pointed by iterator (EdgeAndNode class mainly, but Edge_Iterator
//   is useful too when client wants to call GetOrSpawnNode).
//   It's safe to slice EdgeAndNode off Edge_Iterator.
// It's more customary to have those as three classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized), but
// it's fine if GetOrSpawnNode is called between calls to functions of the
// iterator (e.g. advancing the iterator). Other functions that manipulate
// child_ of parent or the sibling chain are not safe to call while iterating.
template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const atomic_unique_ptr<Node>*,
                                 atomic_unique_ptr<Node>*>;
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  // Creates "end()" iterator.
  Edge_Iterator() {}

  // Creates "begin()" iterator.
  Edge_Iterator(LowNode* parent_node)
      : EdgeAndNode(parent_node != nullptr ? parent_node->GetEdges() : nullptr,
                    nullptr) {
    if (parent_node != nullptr) {
      node_ptr_ = parent_node->GetChild();
      total_count_ = parent_node->GetNumEdges();
      if (edge_) Actualize();
    }
  }

  // Function to support range interface.
  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }

  // Functions to support iterator interface.
  // Equality comparison operators are inherited from EdgeAndNode.
  void operator++() {
    // If it was the last edge in array, become end(), otherwise advance.
    if (++current_idx_ == total_count_) {
      edge_ = nullptr;
    } else {
      ++edge_;
      Actualize();
    }
  }
  Edge_Iterator& operator*() { return *this; }

  // If there is node, return it. Otherwise spawn a new one and return it.
  Node* GetOrSpawnNode(Node* parent) {
    if (node_) return node_;  // If there is already a node, return it.

    // We likely need to add a new node, prepare it now.
    auto low_parent = parent->GetLowNode()->GetEdgeAt(current_idx_);
    atomic_unique_ptr<Node> new_node =
        std::make_unique<Node>(low_parent, current_idx_);
    while (true) {
      auto node = Actualize();  // But maybe other thread already did that.
      if (node_) return node_;  // If it did, return.

      // New node needs to be added, but we might be in a race with another
      // thread doing what we do or adding a different index to the same
      // sibling.

      // Suppose there are nodes with idx 3 and 7, and we want to insert one
      // with idx 5. Here is how it looks like:
      //    node_ptr_ -> &Node(idx_.3).sibling_  ->  Node(idx_.7)
      // Here is how we do that:
      // 1. Store pointer to a node idx_.7:
      //    node_ptr_ -> &Node(idx_.3).sibling_  ->  nullptr
      //    tmp -> Node(idx_.7)
      // 2. Create fresh Node(idx_.5):
      //    node_ptr_ -> &Node(idx_.3).sibling_  ->  Node(idx_.5)
      //    tmp -> Node(idx_.7)
      // 3. Attach stored pointer back to a list:
      //    node_ptr_ ->
      //         &Node(idx_.3).sibling_ -> Node(idx_.5).sibling_ -> Node(idx_.7)

      // Atomically add the new node into the right place.
      // Set new node's sibling to the expected sibling seen by Actualize in
      // node_ptr_.
      auto new_sibling = new_node->GetSibling();
      new_sibling->set(node);
      // Try to atomically insert the new node and stop if it works.
      if (node_ptr_->compare_exchange(node, new_node)) break;
      // Recover from failure and try again.
      // Release expected sibling to avoid double free.
      new_sibling->release();
    }
    // 4. Actualize:
    //    node_ -> &Node(idx_.5)
    //    node_ptr_ -> &Node(idx_.5).sibling_ -> Node(idx_.7)
    Actualize();
    return node_;
  }

 private:
  // Moves node_ptr_ as close as possible to the target index and returns the
  // contents of node_ptr_ for use by atomic insert in GetOrSpawnNode.
  Node* Actualize() {
    // If node_ptr_ is behind, advance it.
    // This is needed (and has to be 'while' rather than 'if') as other threads
    // could spawn new nodes between &node_ptr_ and *node_ptr_ while we didn't
    // see.
    // Read the direct pointer just once as other threads may change it between
    // uses.
    auto node = node_ptr_->get();
    while (node != nullptr && node->Index() < current_idx_) {
      node_ptr_ = node->GetSibling();
      node = node_ptr_->get();
    }
    // If in the end node_ptr_ points to the node that we need, populate node_
    // and advance node_ptr_.
    if (node != nullptr && node->Index() == current_idx_) {
      node_ = node;
      node_ptr_ = node->GetSibling();
    } else {
      node_ = nullptr;
    }

    return node;
  }

  // Pointer to a pointer to the next node. Has to be a pointer to pointer
  // as we'd like to update it when spawning a new node.
  Ptr node_ptr_;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

inline Node::ConstIterator Node::Edges() const {
  return {this->GetLowNode().get()};
}
inline Node::Iterator Node::Edges() { return {this->GetLowNode().get()}; }

// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// It's more customary to have those as two classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized).
template <bool is_const>
class VisitedNode_Iterator {
 public:
  // Creates "end()" iterator.
  VisitedNode_Iterator() {}

  // Creates "begin()" iterator.
  VisitedNode_Iterator(LowNode* parent_node) {
    if (parent_node != nullptr) {
      node_ptr_ = parent_node->GetChild()->get();
      total_count_ = parent_node->GetNumEdges();
      if (node_ptr_ != nullptr && node_ptr_->GetN() == 0) {
        operator++();
      }
    }
  }

  // These are technically wrong, but are usable to compare with end().
  bool operator==(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ == other.node_ptr_;
  }
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ != other.node_ptr_;
  }

  // Function to support range interface.
  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }

  // Functions to support iterator interface.
  // Equality comparison operators are inherited from EdgeAndNode.
  void operator++() {
    do {
      node_ptr_ = node_ptr_->GetSibling()->get();
      // If n started is 0, can jump direct to end due to sorted policy
      // ensuring that each time a new edge becomes best for the first time,
      // it is always the first of the section at the end that has NStarted of
      // 0.
      if (node_ptr_ != nullptr && node_ptr_->GetN() == 0 &&
          node_ptr_->GetNInFlight() == 0) {
        node_ptr_ = nullptr;
        break;
      }
    } while (node_ptr_ != nullptr && node_ptr_->GetN() == 0);
  }
  Node* operator*() { return node_ptr_; }

 private:
  // Pointer to current node.
  Node* node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

inline VisitedNode_Iterator<true> Node::VisitedNodes() const {
  return {this->GetLowNode().get()};
}
inline VisitedNode_Iterator<false> Node::VisitedNodes() {
  return {this->GetLowNode().get()};
}

// Transposition Table type for holding references to all low nodes in DAG.
typedef absl::flat_hash_map<uint64_t, std::weak_ptr<LowNode>>
    TranspositionTable;

class NodeTree {
 public:
  ~NodeTree();
  // Adds a move to current_head_.
  void MakeMove(Move move);
  // Resets the current head to ensure it doesn't carry over details from a
  // previous search.
  void TrimTreeAtHead();
  // Sets the position in the tree, trying to reuse the tree.
  // If @auto_garbage_collect, old tree is garbage collected immediately. (may
  // take some milliseconds)
  // Returns whether the new position is the same game as the old position (with
  // some moves added). Returns false, if the position is completely different,
  // or if it's shorter than before.
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<std::string>& moves);
  bool ResetToPosition(const GameState& pos);
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }
  const std::vector<Move>& GetMoves() const { return moves_; }

 private:
  void DeallocateTree();
  // A node which to start search from.
  Node* current_head_ = nullptr;
  // Root node of a game tree.
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
  std::vector<Move> moves_;
};

// Implement thread local queues. It tracks GC thread to allow faster removal in
// the thread.
class ReleaseNodesWork {
  static constexpr size_t kCapacity = 32;
public:
  ReleaseNodesWork(bool gc_thread = false);
  ~ReleaseNodesWork();
  bool IsWorker() const;

  // A limited vector like interface to operate on the container.
  void emplace_back(std::unique_ptr<Node>&& node);
  bool empty() const;

  // Swap is used to transfer queue into a new stack variable. The stack
  // variable will flush the queue in the desctructor.
  void swap(ReleaseNodesWork &other);
private:
  // Flush the local queue to the shared queue.
  void Submit();

  // No locks required because only one thread can access this object.
  std::vector<std::unique_ptr<Node>> released_nodes_;
  bool is_gc_thread_;
};

class NodeGarbageCollector {
  NodeGarbageCollector();
  ~NodeGarbageCollector();
public:
  enum State {
    Running,
    GoToSleep,
    Sleeping,
    Exit,
  };

  // Access to the singleton which is only created on the demand.
  static NodeGarbageCollector& Instance() {
    static NodeGarbageCollector singleton;
    return singleton;
  }
  // Delays node destruction until GC thread activates.
  template<typename UniquePtr>
  void AddToGcQueue(UniquePtr& node);

  // Allow search to control when garbage collection runs.
  void Start();
  void Stop();
  State Wait() const;
  void Abort();

  // Moves thread local GC queue to the shared queue. This avoid case where a
  // thread frees only a few branches which will be stuck in the thread local
  // queue. A few big branches can have a major memory impact. If thread exits,
  // there is no need to call this.
  void NotifyThreadGoingSleep();

private:
  // Helper to transition between states safely
  bool SetState(State& old, State desired);
  bool IsActive() const;
  bool ShouldQueue(std::unique_ptr<Node>& node) const;
  // The collection thread implementation.
  void GCThread();
  // Thread local collection queue. Local queues flush to the shared queue
  // in batches to avoid lock contention.
  static ReleaseNodesWork& LocalWork(bool gc_thread = false) {
    static thread_local ReleaseNodesWork shared{gc_thread};
    return shared;
  }

  std::atomic<State> state_ = {Sleeping};
#ifdef NO_STD_ATOMIC_WAIT
  // Fallback conditional variable when c++ library doesn't implement
  // std::atomic::wait().
  mutable Mutex state_mutex_;
  mutable std::condition_variable state_signal_;
#endif
  std::thread gc_thread_;
  SpinMutex mutex_;
  std::deque<std::vector<std::unique_ptr<Node>>> released_nodes_ GUARDED_BY(mutex_);

  friend class ReleaseNodesWork;
};

}  // namespace dag_classic
}  // namespace lczero
