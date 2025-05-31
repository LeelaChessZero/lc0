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

#include "search/classic/node.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>

#include "neural/encoder.h"
#include "neural/network.h"
#include "utils/exception.h"
#include "utils/hashcat.h"

namespace lczero {
namespace classic {

/////////////////////////////////////////////////////////////////////////
// Node garbage collector
/////////////////////////////////////////////////////////////////////////

namespace {
// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollector {
 public:
  NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

  // Takes ownership of a subtree, to dispose it in a separate thread when
  // it has time.
  void AddToGcQueue(std::unique_ptr<Node> node, size_t solid_size = 0) {
    if (!node) return;
    Mutex::Lock lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
    subtrees_to_gc_solid_size_.push_back(solid_size);
  }

  ~NodeGarbageCollector() {
    // Flips stop flag and waits for a worker thread to stop.
    stop_.store(true);
    gc_thread_.join();
  }

 private:
  void GarbageCollect() {
    while (!stop_.load()) {
      // Node will be released in destructor when mutex is not locked.
      std::unique_ptr<Node> node_to_gc;
      size_t solid_size = 0;
      {
        // Lock the mutex and move last subtree from subtrees_to_gc_ into
        // node_to_gc.
        Mutex::Lock lock(gc_mutex_);
        if (subtrees_to_gc_.empty()) return;
        node_to_gc = std::move(subtrees_to_gc_.back());
        subtrees_to_gc_.pop_back();
        solid_size = subtrees_to_gc_solid_size_.back();
        subtrees_to_gc_solid_size_.pop_back();
      }
      // Solid is a hack...
      if (solid_size != 0) {
        for (size_t i = 0; i < solid_size; i++) {
          node_to_gc.get()[i].~Node();
        }
        std::allocator<Node> alloc;
        alloc.deallocate(node_to_gc.release(), solid_size);
      }
    }
  }

  void Worker() {
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    };
  }

  mutable Mutex gc_mutex_;
  std::vector<std::unique_ptr<Node>> subtrees_to_gc_ GUARDED_BY(gc_mutex_);
  std::vector<size_t> subtrees_to_gc_solid_size_ GUARDED_BY(gc_mutex_);

  // When true, Worker() should stop and exit.
  std::atomic<bool> stop_{false};
  std::thread gc_thread_;
};

NodeGarbageCollector gNodeGc;
}  // namespace

/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Flip();
  return m;
}

// Policy priors (P) are stored in a compressed 16-bit format.
//
// Source values are 32-bit floats:
// * bit 31 is sign (zero means positive)
// * bit 30 is sign of exponent (zero means nonpositive)
// * bits 29..23 are value bits of exponent
// * bits 22..0 are significand bits (plus a "virtual" always-on bit: s ∈ [1,2))
// The number is then sign * 2^exponent * significand, usually.
// See https://www.h-schmidt.net/FloatConverter/IEEE754.html for details.
//
// In compressed 16-bit value we store bits 27..12:
// * bit 31 is always off as values are always >= 0
// * bit 30 is always off as values are always < 2
// * bits 29..28 are only off for values < 4.6566e-10, assume they are always on
// * bits 11..0 are for higher precision, they are dropped leaving only 11 bits
//     of precision
//
// When converting to compressed format, bit 11 is added to in order to make it
// a rounding rather than truncation.
//
// Out of 65556 possible values, 2047 are outside of [0,1] interval (they are in
// interval (1,2)). This is fine because the values in [0,1] are skewed towards
// 0, which is also exactly how the components of policy tend to behave (since
// they add up to 1).

// If the two assumed-on exponent bits (3<<28) are in fact off, the input is
// rounded up to the smallest value with them on. We accomplish this by
// subtracting the two bits from the input and checking for a negative result
// (the subtraction works despite crossing from exponent to significand). This
// is combined with the round-to-nearest addition (1<<11) into one op.
void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.ToString(true) << " p_: " << p_
      << " GetP: " << GetP();
  return oss.str();
}

std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (const auto move : moves) edge++->move_ = move;
  return edges;
}

/////////////////////////////////////////////////////////////////////////
// Node
/////////////////////////////////////////////////////////////////////////

Node* Node::CreateSingleChildNode(Move move) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist({move});
  num_edges_ = 1;
  child_ = std::make_unique<Node>(this, 0);
  return child_.get();
}

void Node::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist(moves);
  num_edges_ = moves.size();
}

Node::ConstIterator Node::Edges() const {
  return {*this, !solid_children_ ? &child_ : nullptr};
}
Node::Iterator Node::Edges() {
  return {*this, !solid_children_ ? &child_ : nullptr};
}

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += GetEdgeToNode(node)->GetP();
  return sum;
}

Edge* Node::GetEdgeToNode(const Node* node) const {
  assert(node->parent_ == this);
  assert(node->index_ < num_edges_);
  return &edges_[node->index_];
}

Edge* Node::GetOwnEdge() const { return GetParent()->GetEdgeToNode(this); }

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << static_cast<int>(terminal_type_) << " This:" << this
      << " Parent:" << parent_ << " Index:" << index_
      << " Child:" << child_.get() << " Sibling:" << sibling_.get()
      << " WL:" << wl_ << " N:" << n_ << " N_:" << n_in_flight_
      << " Edges:" << static_cast<int>(num_edges_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << " Solid:" << solid_children_;
  return oss.str();
}

bool Node::MakeSolid() {
  if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;
  // Can only make solid if no immediate leaf children are in flight since we
  // allow the search code to hold references to leaf nodes across locks.
  Node* old_child_to_check = child_.get();
  uint32_t total_in_flight = 0;
  while (old_child_to_check != nullptr) {
    if (old_child_to_check->GetN() <= 1 &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    if (old_child_to_check->IsTerminal() &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    total_in_flight += old_child_to_check->GetNInFlight();
    old_child_to_check = old_child_to_check->sibling_.get();
  }
  // If the total of children in flight is not the same as self, then there are
  // collisions against immediate children (which don't update the GetNInFlight
  // of the leaf) and its not safe.
  if (total_in_flight != GetNInFlight()) {
    return false;
  }
  std::allocator<Node> alloc;
  auto* new_children = alloc.allocate(num_edges_);
  for (int i = 0; i < num_edges_; i++) {
    new (&(new_children[i])) Node(this, i);
  }
  std::unique_ptr<Node> old_child = std::move(child_);
  while (old_child) {
    int index = old_child->index_;
    new_children[index] = std::move(*old_child.get());
    // This isn't needed, but it helps crash things faster if something has gone
    // wrong.
    old_child->parent_ = nullptr;
    gNodeGc.AddToGcQueue(std::move(old_child));
    new_children[index].UpdateChildrenParents();
    old_child = std::move(new_children[index].sibling_);
  }
  // This is a hack.
  child_ = std::unique_ptr<Node>(new_children);
  solid_children_ = true;
  return true;
}

void Node::SortEdges() {
  assert(edges_);
  assert(!child_);
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges_.get(), (edges_.get() + num_edges_),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  if (type != Terminal::TwoFold) SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    // Terminal losses have no uncertainty and no reason for their U value to be
    // comparable to another non-loss choice. Force this by clearing the policy.
    if (GetParent() != nullptr) GetOwnEdge()->SetP(0.0f);
  }
}

void Node::MakeNotTerminal() {
  terminal_type_ = Terminal::NonTerminal;
  n_ = 0;

  // If we have edges, we've been extended (1 visit), so include children too.
  if (edges_) {
    n_++;
    for (const auto& child : Edges()) {
      const auto n = child.GetN();
      if (n > 0) {
        n_ += n;
        // Flip Q for opponent.
        // Default values don't matter as n is > 0.
        wl_ += -child.GetWL(0.0f) * n;
        d_ += child.GetD(0.0f) * n;
      }
    }

    // Recompute with current eval (instead of network's) and children's eval.
    wl_ /= n_;
    d_ /= n_;
  }
}

void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}

void Node::CancelScoreUpdate(int multivisit) { n_in_flight_ -= multivisit; }

void Node::FinalizeScoreUpdate(float v, float d, float m, int multivisit) {
  // Recompute Q.
  wl_ += multivisit * (v - wl_) / (n_ + multivisit);
  d_ += multivisit * (d - d_) / (n_ + multivisit);
  m_ += multivisit * (m - m_) / (n_ + multivisit);

  // Increment N.
  n_ += multivisit;
  // Decrement virtual loss.
  n_in_flight_ -= multivisit;
}

void Node::AdjustForTerminal(float v, float d, float m, int multivisit) {
  // Recompute Q.
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;
}

void Node::RevertTerminalVisits(float v, float d, float m, int multivisit) {
  // Compute new n_ first, as reducing a node to 0 visits is a special case.
  const int n_new = n_ - multivisit;
  if (n_new <= 0) {
    // If n_new == 0, reset all relevant values to 0.
    wl_ = 0.0;
    d_ = 1.0;
    m_ = 0.0;
    n_ = 0;
  } else {
    // Recompute Q and M.
    wl_ -= multivisit * (v - wl_) / n_new;
    d_ -= multivisit * (d - d_) / n_new;
    m_ -= multivisit * (m - m_) / n_new;
    // Decrement N.
    n_ -= multivisit;
  }
}

void Node::UpdateChildrenParents() {
  if (!solid_children_) {
    Node* cur_child = child_.get();
    while (cur_child != nullptr) {
      cur_child->parent_ = this;
      cur_child = cur_child->sibling_.get();
    }
  } else {
    Node* child_array = child_.get();
    for (int i = 0; i < num_edges_; i++) {
      child_array[i].parent_ = this;
    }
  }
}

void Node::ReleaseChildren() {
  gNodeGc.AddToGcQueue(std::move(child_), solid_children_ ? num_edges_ : 0);
}

void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
  if (solid_children_) {
    std::unique_ptr<Node> saved_node;
    if (node_to_save != nullptr) {
      saved_node = std::make_unique<Node>(this, node_to_save->index_);
      *saved_node = std::move(*node_to_save);
    }
    gNodeGc.AddToGcQueue(std::move(child_), num_edges_);
    child_ = std::move(saved_node);
    if (child_) {
      child_->UpdateChildrenParents();
    }
    solid_children_ = false;
  } else {
    // Stores node which will have to survive (or nullptr if it's not found).
    std::unique_ptr<Node> saved_node;
    // Pointer to unique_ptr, so that we could move from it.
    for (std::unique_ptr<Node>* node = &child_; *node;
         node = &(*node)->sibling_) {
      // If current node is the one that we have to save.
      if (node->get() == node_to_save) {
        // Kill all remaining siblings.
        gNodeGc.AddToGcQueue(std::move((*node)->sibling_));
        // Save the node, and take the ownership from the unique_ptr.
        saved_node = std::move(*node);
        break;
      }
    }
    // Make saved node the only child. (kills previous siblings).
    gNodeGc.AddToGcQueue(std::move(child_));
    child_ = std::move(saved_node);
  }
  if (!child_) {
    num_edges_ = 0;
    edges_.reset();  // Clear edges list.
  }
}

/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

void NodeTree::MakeMove(Move move) {
  Node* new_head = nullptr;
  for (auto& n : current_head_->Edges()) {
    if (n.GetMove() == move) {
      new_head = n.GetOrSpawnNode(current_head_);
      // Ensure head is not terminal, so search can extend or visit children of
      // "terminal" positions, e.g., WDL hits, converted terminals, 3-fold draw.
      if (new_head->IsTerminal()) new_head->MakeNotTerminal();
      break;
    }
  }
  current_head_->ReleaseChildrenExceptOne(new_head);
  new_head = current_head_->child_.get();
  current_head_ =
      new_head ? new_head : current_head_->CreateSingleChildNode(move);
  history_.Append(move);
}

void NodeTree::TrimTreeAtHead() {
  // If solid, this will be empty before move and will be moved back empty
  // afterwards which is fine.
  auto tmp = std::move(current_head_->sibling_);
  // Send dependent nodes for GC instead of destroying them immediately.
  current_head_->ReleaseChildren();
  *current_head_ = Node(current_head_->GetParent(), current_head_->index_);
  current_head_->sibling_ = std::move(tmp);
}

bool NodeTree::ResetToPosition(const GameState& pos) {
  if (gamebegin_node_ && (history_.Starting() != pos.startpos)) {
    // Completely different position.
    DeallocateTree();
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<Node>(nullptr, 0);
  }

  history_.Reset(pos.startpos);

  Node* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const Move m : pos.moves) {
    MakeMove(m);
    if (old_head == current_head_) seen_old_head = true;
  }

  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  if (!seen_old_head) TrimTreeAtHead();
  return seen_old_head;
}

bool NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(starting_fen);
  ChessBoard cur_board = state.startpos.GetBoard();
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    Move m = cur_board.ParseMove(move);
    state.moves.push_back(m);
    cur_board.ApplyMove(m);
    cur_board.Mirror();
  }
  return ResetToPosition(state);
}

void NodeTree::DeallocateTree() {
  // Same as gamebegin_node_.reset(), but actual deallocation will happen in
  // GC thread.
  gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}

}  // namespace classic
}  // namespace lczero
