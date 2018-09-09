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
#include "utils/exception.h"
#include "utils/hashcat.h"

namespace lczero {

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
  void AddToGcQueue(std::unique_ptr<Node> node) {
    if (!node) return;
    Mutex::Lock lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
  }

  ~NodeGarbageCollector() {
    // Flips stop flag and waits for a worker thread to stop.
    stop_ = true;
    gc_thread_.join();
  }

 private:
  void GarbageCollect() {
    while (!stop_) {
      // Node will be released in destructor when mutex is not locked.
      std::unique_ptr<Node> node_to_gc;
      {
        // Lock the mutex and move last subtree from subtrees_to_gc_ into
        // node_to_gc.
        Mutex::Lock lock(gc_mutex_);
        if (subtrees_to_gc_.empty()) return;
        node_to_gc = std::move(subtrees_to_gc_.back());
        subtrees_to_gc_.pop_back();
      }
    }
  }

  void Worker() {
    while (!stop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    };
  }

  mutable Mutex gc_mutex_;
  std::vector<std::unique_ptr<Node>> subtrees_to_gc_ GUARDED_BY(gc_mutex_);

  // When true, Worker() should stop and exit.
  volatile bool stop_ = false;
  std::thread gc_thread_;
};  // namespace

NodeGarbageCollector gNodeGc;
}  // namespace

/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Mirror();
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
  oss << "Move: " << move_.as_string() << " p_: " << p_ << " GetP: " << GetP();
  return oss.str();
}

/////////////////////////////////////////////////////////////////////////
// EdgeList
/////////////////////////////////////////////////////////////////////////

EdgeList::EdgeList(MoveList moves)
    : edges_(std::make_unique<Edge[]>(moves.size())), size_(moves.size()) {
  auto* edge = edges_.get();
  for (auto move : moves) edge++->SetMove(move);
}

EdgeList::EdgeList(EdgeList&& other)
    : edges_(std::move(other.edges_)), size_(other.size_) {
  other.size_ = 0;
}

EdgeList& EdgeList::operator=(EdgeList&& other) {
  edges_ = std::move(other.edges_);
  size_ = other.size_;
  other.size_ = 0;
  return *this;
}

/////////////////////////////////////////////////////////////////////////
// Node
/////////////////////////////////////////////////////////////////////////

Node* Node::CreateSingleChildNode(Move move) {
  assert(!edges_);
  assert(!child_);
  edges_ = EdgeList({move});
  child_ = std::make_unique<Node>(this, 0);
  return child_.get();
}

void Node::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_);
  edges_ = EdgeList(moves);
}

Node::ConstIterator Node::Edges() const { return {edges_, &child_}; }
Node::Iterator Node::Edges() { return {edges_, &child_}; }

float Node::GetVisitedPolicy() const { return visited_policy_; }

Edge* Node::GetEdgeToNode(const Node* node) const {
  assert(node->parent_ == this);
  assert(node->index_ < edges_.size());
  return &edges_[node->index_];
}

Edge* Node::GetEdgeToSelf() const {
  assert(parent_);
  return parent_->GetEdgeToNode(this);
}

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << is_terminal_ << " This:" << this << " Parent:" << parent_
      << " Index:" << index_ << " Child:" << child_.get()
      << " Sibling:" << sibling_.get() << " Q:" << q_ << " N:" << n_
      << " N_:" << n_in_flight_ << " Edges:" << edges_.size()
      << " SubTree:" << subtree_.get();
  return oss.str();
}

void Node::MakeTerminal(GameResult result) {
  is_terminal_ = true;
  if (result == GameResult::DRAW) {
    q_ = 0.0f;
  } else if (result == GameResult::WHITE_WON) {
    q_ = 1.0f;
  } else if (result == GameResult::BLACK_WON) {
    q_ = -1.0f;
  }
}

bool Node::TryStartUpdateFromSubtree() {
  assert(subtree_);
  if (subtree_->GetN() <= n_) {
    subtree_->ReportDeficiency();
    return false;
  }
  ++n_in_flight_;
  return true;
}

bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}

void Node::CancelScoreUpdate() {
  assert(n_in_flight_ > 0);
  --n_in_flight_;
}

void Node::FinalizeScoreUpdate(float v) {
  // Recompute Q.
  q_ += (v - q_) / (n_ + 1);
  // If first visit, update parent's sum of policies visited at least once.
  if (n_ == 0 && parent_ != nullptr) {
    parent_->visited_policy_ += parent_->edges_[index_].GetP();
  }
  // Increment N.
  ++n_;
  // Decrement virtual loss.
  assert(n_in_flight_ > 0);
  --n_in_flight_;
}

Node::NodeRange Node::ChildNodes() const { return child_.get(); }

void Node::ReleaseChildren() { gNodeGc.AddToGcQueue(std::move(child_)); }

void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
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
  float total_n = static_cast<float>(GetChildrenVisits());
  // Prevent garbage/invalid training data from being uploaded to server.
  if (total_n <= 0.0f) throw Exception("Search generated invalid data!");
  std::memset(result.probabilities, 0, sizeof(result.probabilities));
  for (const auto& child : Edges()) {
    result.probabilities[child.edge()->GetMove().as_nn_index()] =
        child.GetN() / total_n;
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
  result.rule50_count = position.GetNoCaptureNoPawnPly();

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

void Node::FixChildrenParents() {
  for (auto* node : ChildNodes()) node->parent_ = this;
}

SubTree* Node::DetachSubtree() {
  assert(!HasDetachedSubtree());

  std::unique_ptr<Node> subtree_root = std::make_unique<Node>(nullptr, 0);
  subtree_root->edges_ = std::move(edges_);
  subtree_root->q_ = q_;
  subtree_root->n_ = n_;
  assert(n_in_flight_ == 0);
  subtree_root->visited_policy_ = visited_policy_;
  assert(!is_terminal_);
  subtree_root->child_ = std::move(child_);
  // `sibling_` is NOT to be moved moved to detached subtree.
  subtree_root->FixChildrenParents();

  subtree_ = std::make_unique<SubTree>(this, std::move(subtree_root));
  return subtree_.get();
}

void Node::ReattachSubtree() {
  // Check that there is a subtree to attach.
  assert(HasDetachedSubtree());
  // Check that no search worker is using this subtree.
  assert(!subtree_->HasWorker());
  // Check that root of a subtree was root indeed.
  assert(nullptr == subtree_->GetRootNode()->parent_);
  assert(0 == subtree_->GetRootNode()->index_);

  // Take ownership of a subtree.
  std::unique_ptr<SubTree> subtree = std::move(subtree_);

  auto index = index_;
  auto parent = parent_;
  std::unique_ptr<Node> sibling = std::move(subtree->GetRootNode()->sibling_);
  *this = std::move(*subtree->GetRootNode());
  index_ = index;
  parent_ = parent;
  sibling_ = std::move(sibling);
  FixChildrenParents();
}

/////////////////////////////////////////////////////////////////////////
// SubTree
/////////////////////////////////////////////////////////////////////////

SubTree::SubTree(Node* parent_node, std::unique_ptr<Node> detached_node)
    : root_(std::move(detached_node)),
      parent_node_(parent_node),
      q_(root_->GetQ()),
      n_(root_->GetN()),
      parent_n_(root_->GetN()) {}

bool SubTree::HasWorker() const {
  return is_used_.load(std::memory_order_acquire);
}

void SubTree::SetHasAssignedWorker() {
  assert(!HasWorker());
  is_used_.store(true, std::memory_order_release);
}

void SubTree::ResetHasAssignedWorker() {
  assert(HasWorker());
  is_used_.store(false, std::memory_order_release);
}

void SubTree::UpdateNQ(uint32_t n, float q) {
  q_.store(q, std::memory_order_release);
  n_.store(n, std::memory_order_release);
}

uint32_t SubTree::GetN() const { return n_.load(std::memory_order_acquire); }
float SubTree::GetQ() const { return q_.load(std::memory_order_acquire); }

void SubTree::ReportDeficiency() {
  // std::cerr << " Def:" << this;
  typical_deficiency_.fetch_add(1, std::memory_order_release);
}

void SubTree::PullStatsFromParent() {
  assert(parent_node_);
  parent_n_.store(parent_node_->GetN(), std::memory_order_release);
}

int SubTree::GetRecommendedBatchSize() const {
  return parent_n_.load(std::memory_order_acquire) +
         // typical_deficiency_.load(std::memory_order_acquire) -
         100 - n_.load(std::memory_order_acquire);
}

bool SubTree::IsBehind() const {
  return parent_n_.load(std::memory_order_acquire) >=
         n_.load(std::memory_order_acquire);
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
  if (HeadPosition().IsBlackToMove()) move.Mirror();

  Node* old_head = current_head_->GetRootNode();
  Node* new_head = nullptr;
  // Search through existing children whether that move already has node.
  for (auto& n : old_head->Edges()) {
    if (n.GetMove() == move) {
      new_head = n.GetOrSpawnNode(old_head);
      break;
    }
  }
  // Deallocate nodes from other moves.
  old_head->ReleaseChildrenExceptOne(new_head);

  // If a node for a move was not there, create it.
  if (!new_head) new_head = old_head->CreateSingleChildNode(move);

  // Detach node for a new head if it's not already detached.
  if (!new_head->HasDetachedSubtree()) new_head->DetachSubtree();

  // Reattach old head to its parent.
  if (current_head_->HasParent()) current_head_->Reattach();

  // Update current head to a detached subtree.
  current_head_ = new_head->GetDetachedSubtree();
  history_.Append(move);
}

void NodeTree::TrimTreeAtHead() {
  Node* head = current_head_->GetRootNode();
  assert(!head->sibling_);
  // Send dependent nodes for GC instead of destroying them immediately.
  gNodeGc.AddToGcQueue(std::move(head->child_));
  *head = Node(head->GetParent(), head->index_);
}

void NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<Move>& moves) {
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);
  if (game_tree_ && history_.Starting().GetBoard() != starting_board) {
    // Completely different position.
    DeallocateTree();
  }

  if (!game_tree_) {
    game_tree_ =
        std::make_unique<SubTree>(nullptr, std::make_unique<Node>(nullptr, 0));
  }

  history_.Reset(starting_board, no_capture_ply,
                 full_moves * 2 - (starting_board.flipped() ? 1 : 2));

  SubTree* old_head = current_head_;
  bool seen_old_head = (game_tree_.get() == old_head);
  current_head_ = game_tree_.get();

  for (const auto& move : moves) {
    MakeMove(move);
    if (old_head == current_head_) seen_old_head = true;
  }

  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  // Also, if the current_head_ is terminal, reset that as well to allow forced
  // analysis of WDL hits, or possibly 3 fold or 50 move "draws", etc.
  if (!seen_old_head || current_head_->GetRootNode()->IsTerminal()) {
    TrimTreeAtHead();
  }
}

void NodeTree::DeallocateTree() {
  // Same as game_tree_.reset(), but actual deallocation will happen in
  // GC thread.
  gNodeGc.AddToGcQueue(std::move(game_tree_->GetRootNode()->child_));
  game_tree_ = nullptr;
  current_head_ = nullptr;
}

}  // namespace lczero
