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

#include "search/dag_classic/node.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "utils/exception.h"
#include "utils/hashcat.h"

namespace lczero {
namespace dag_classic {

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
// LowNode + Node
/////////////////////////////////////////////////////////////////////////

void Node::Trim() {
  wl_ = 0.0f;

  UnsetLowNode();
  // sibling_

  d_ = 0.0f;
  m_ = 0.0f;
  n_ = 0;
  n_in_flight_ = 0;

  // edge_

  // index_

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  repetition_ = false;
}

Node* Node::GetChild() const {
  if (!low_node_) return nullptr;
  return low_node_->GetChild()->get();
}

bool Node::HasChildren() const { return low_node_ && low_node_->HasChildren(); }

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += node->GetP();
  return sum;
}

uint32_t Node::GetNInFlight() const {
  return n_in_flight_;
}

uint32_t Node::GetChildrenVisits() const {
  return low_node_ ? low_node_->GetChildrenVisits() : 0;
}

uint32_t Node::GetTotalVisits() const {
  return low_node_ ? low_node_->GetN() : 0;
}

const Edge& LowNode::GetEdgeAt(uint16_t index) const { return edges_[index]; }

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " <Node> This:" << this << " LowNode:" << low_node_.get()
      << " Index:" << index_ << " Move:" << GetMove().ToString(true)
      << " Sibling:" << sibling_.get() << " P:" << GetP() << " WL:" << wl_
      << " D:" << d_ << " M:" << m_ << " N:" << n_ << " N_:" << n_in_flight_
      << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2;
  return oss.str();
}

std::string LowNode::DebugString() const {
  std::ostringstream oss;
  oss << " <LowNode> This:" << this << " Edges:" << edges_.get()
      << " NumEdges:" << static_cast<int>(num_edges_)
      << " Child:" << child_.get() << " WL:" << wl_ << " D:" << d_
      << " M:" << m_ << " N:" << n_ << " NP:" << num_parents_
      << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2
      << " IsTransposition:" << is_transposition;
  return oss.str();
}

void Edge::SortEdges(Edge* edges, int num_edges) {
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges, (edges + num_edges),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void LowNode::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
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
  }

  assert(WLDMInvariantsHold());
}

void LowNode::MakeNotTerminal(const Node* node) {
  assert(edges_);
  if (!IsTerminal()) return;

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  n_ = 0;
  wl_ = 0.0;
  d_ = 0.0;
  m_ = 0.0;

  // Include children too.
  if (node->GetNumEdges() > 0) {
    for (const auto& child : node->Edges()) {
      const auto n = child.GetN();
      if (n > 0) {
        n_ += n;
        // Flip Q for opponent.
        // Default values don't matter as n is > 0.
        wl_ += child.GetWL(0.0f) * n;
        d_ += child.GetD(0.0f) * n;
        m_ += child.GetM(0.0f) * n;
      }
    }

    // Recompute with current eval (instead of network's) and children's eval.
    wl_ /= n_;
    d_ /= n_;
    m_ /= n_;
  }

  assert(WLDMInvariantsHold());
}

void LowNode::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

uint8_t Node::GetNumEdges() const {
  return low_node_ ? low_node_->GetNumEdges() : 0;
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
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
    SetP(0.0f);
  }

  assert(WLDMInvariantsHold());
}

void Node::MakeNotTerminal(bool also_low_node) {
  // At least one of node and low node pair needs to be a terminal.
  if (!IsTerminal() &&
      (!also_low_node || !low_node_ || !low_node_->IsTerminal()))
    return;

  terminal_type_ = Terminal::NonTerminal;
  repetition_ = false;
  if (low_node_) {  // Two-fold or derived terminal.
    // Revert low node first.
    if (also_low_node && low_node_) low_node_->MakeNotTerminal(this);

    auto [lower_bound, upper_bound] = low_node_->GetBounds();
    lower_bound_ = -upper_bound;
    upper_bound_ = -lower_bound;
    n_ = low_node_->GetN();
    wl_ = -low_node_->GetWL();
    d_ = low_node_->GetD();
    m_ = low_node_->GetM() + 1;
  } else {  // Real terminal.
    lower_bound_ = GameResult::BLACK_WON;
    upper_bound_ = GameResult::WHITE_WON;
    n_ = 0.0f;
    wl_ = 0.0f;
    d_ = 0.0f;
    m_ = 0.0f;
  }

  assert(WLDMInvariantsHold());
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

void Node::CancelScoreUpdate(uint32_t multivisit) {
  assert(n_in_flight_ >= (uint32_t)multivisit);
  n_in_flight_ -= multivisit;
}

void LowNode::FinalizeScoreUpdate(float v, float d, float m,
                                  uint32_t multivisit) {
  assert(edges_);
  // Recompute Q.
  wl_ += multivisit * (v - wl_) / (n_ + multivisit);
  d_ += multivisit * (d - d_) / (n_ + multivisit);
  m_ += multivisit * (m - m_) / (n_ + multivisit);

  assert(WLDMInvariantsHold());

  // Increment N.
  n_ += multivisit;
}

void LowNode::AdjustForTerminal(float v, float d, float m,
                                uint32_t multivisit) {
  assert(static_cast<uint32_t>(multivisit) <= n_);

  // Recompute Q.
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;

  assert(WLDMInvariantsHold());
}

void Node::FinalizeScoreUpdate(float v, float d, float m, uint32_t multivisit) {
  // Recompute Q.
  wl_ += multivisit * (v - wl_) / (n_ + multivisit);
  d_ += multivisit * (d - d_) / (n_ + multivisit);
  m_ += multivisit * (m - m_) / (n_ + multivisit);

  assert(WLDMInvariantsHold());

  // Increment N.
  n_ += multivisit;
  // Decrement virtual loss.
  assert(n_in_flight_ >= (uint32_t)multivisit);
  n_in_flight_ -= multivisit;
}

void Node::AdjustForTerminal(float v, float d, float m, uint32_t multivisit) {
  assert(static_cast<uint32_t>(multivisit) <= n_);

  // Recompute Q.
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;

  assert(WLDMInvariantsHold());
}

void Node::IncrementNInFlight(uint32_t multivisit) {
  n_in_flight_ += multivisit;
}

void LowNode::ReleaseChildren(
    std::vector<std::unique_ptr<Node>>& released_nodes) {
  released_nodes.emplace_back(std::move(child_));
}

void LowNode::ReleaseChildrenExceptOne(
    Node* node_to_save, std::vector<std::unique_ptr<Node>>& released_nodes) {
  // Stores node which will have to survive (or nullptr if it's not found).
  std::unique_ptr<Node> saved_node;
  // Pointer to unique_ptr, so that we could move from it.
  for (std::unique_ptr<Node>* node = &child_; *node;
       node = (*node)->GetSibling()) {
    // If current node is the one that we have to save.
    if (node->get() == node_to_save) {
      // Kill all remaining siblings.
      released_nodes.emplace_back(std::move(*(*node)->GetSibling()));
      // Save the node, and take the ownership from the unique_ptr.
      saved_node = std::move(*node);
      break;
    }
  }
  // Make saved node the only child. (kills previous siblings).
  released_nodes.emplace_back(std::move(child_));
  child_ = std::move(saved_node);
}

void Node::ReleaseChildrenExceptOne(
    Node* node_to_save,
    std::vector<std::unique_ptr<Node>>& released_nodes) const {
  // Sometime we have no graph yet or a reverted terminal without low node.
  if (low_node_) {
    low_node_->ReleaseChildrenExceptOne(node_to_save, released_nodes);
  }
}

void Node::SetLowNode(std::shared_ptr<LowNode> low_node) {
  assert(!low_node_);
  low_node->AddParent();
  low_node_ = low_node;
}
void Node::UnsetLowNode() {
  if (low_node_) low_node_->RemoveParent();
  low_node_.reset();
}

static std::string PtrToNodeName(const void* ptr) {
  std::ostringstream oss;
  oss << "n_" << ptr;
  return oss.str();
}

std::string LowNode::DotNodeString() const {
  std::ostringstream oss;
  oss << PtrToNodeName(this) << " ["
      << "shape=box";
  // Adjust formatting to limit node size.
  oss << std::fixed << std::setprecision(3);
  oss << ",label=\""     //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\lD=" << d_ << "\\lM=" << m_ << "\\lN=" << n_ << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::showpos << std::setprecision(5);
  oss << ",tooltip=\""   //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << n_
      << "\\nNP=" << num_parents_
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2
      << "\\nIsTransposition=" << is_transposition  //
      << std::noshowpos                             //
      << "\\n\\nThis=" << this << "\\nEdges=" << edges_.get()
      << "\\nNumEdges=" << static_cast<int>(num_edges_)
      << "\\nChild=" << child_.get() << "\\n\"";
  oss << "];";
  return oss.str();
}

std::string Node::DotEdgeString(bool as_opponent, const LowNode* parent) const {
  std::ostringstream oss;
  oss << (parent == nullptr ? "top" : PtrToNodeName(parent)) << " -> "
      << (low_node_ ? PtrToNodeName(low_node_.get()) : PtrToNodeName(this))
      << " [";
  oss << "label=\""
      << (parent == nullptr ? "N/A" : GetMove(as_opponent).ToString(true))
      << "\\lN=" << n_ << "\\lN_=" << n_in_flight_;
  oss << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::setprecision(5);
  oss << ",labeltooltip=\""
      << "P=" << (parent == nullptr ? 0.0f : GetP())  //
      << std::showpos                                 //
      << "\\nWL= " << wl_                             //
      << std::noshowpos                               //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << n_
      << "\\nN_=" << n_in_flight_
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << "\\n\\nThis=" << this  //
      << std::noshowpos                                               //
      << "\\nLowNode=" << low_node_.get() << "\\nParent=" << parent
      << "\\nIndex=" << index_ << "\\nSibling=" << sibling_.get() << "\\n\"";
  oss << "];";
  return oss.str();
}

std::string Node::DotGraphString(bool as_opponent) const {
  std::ostringstream oss;
  std::unordered_set<const LowNode*> seen;
  std::list<std::pair<const Node*, bool>> unvisited_fifo;

  oss << "strict digraph {" << std::endl;
  oss << "edge ["
      << "headport=n"
      << ",tooltip=\" \""  // Remove default tooltips from edge parts.
      << "];" << std::endl;
  oss << "node ["
      << "shape=point"    // For fake nodes.
      << ",style=filled"  // Show tooltip everywhere on the node.
      << ",fillcolor=ivory"
      << "];" << std::endl;
  oss << "ranksep=" << 4.0f * std::log10(GetN()) << std::endl;

  oss << DotEdgeString(!as_opponent) << std::endl;
  if (low_node_) {
    seen.insert(low_node_.get());
    unvisited_fifo.push_back(std::pair(this, as_opponent));
  }

  while (!unvisited_fifo.empty()) {
    auto [parent_node, parent_as_opponent] = unvisited_fifo.front();
    unvisited_fifo.pop_front();

    auto parent_low_node = parent_node->GetLowNode().get();
    seen.insert(parent_low_node);
    oss << parent_low_node->DotNodeString() << std::endl;

    for (auto& child_edge : parent_node->Edges()) {
      auto child = child_edge.node();
      if (child == nullptr) break;

      oss << child->DotEdgeString(parent_as_opponent) << std::endl;
      auto child_low_node = child->GetLowNode().get();
      if (child_low_node != nullptr &&
          (seen.find(child_low_node) == seen.end())) {
        seen.insert(child_low_node);
        unvisited_fifo.push_back(std::pair(child, !parent_as_opponent));
      }
    }
  }

  oss << "}" << std::endl;

  return oss.str();
}

bool Node::ZeroNInFlight() const {
  std::unordered_set<const LowNode*> seen;
  std::list<const Node*> unvisited_fifo;
  size_t nonzero_node_count = 0;

  if (GetNInFlight() > 0) {
    std::cerr << DebugString() << std::endl;
    ++nonzero_node_count;
  }
  if (low_node_) {
    seen.insert(low_node_.get());
    unvisited_fifo.push_back(this);
  }

  while (!unvisited_fifo.empty()) {
    auto parent_node = unvisited_fifo.front();
    unvisited_fifo.pop_front();

    for (auto& child_edge : parent_node->Edges()) {
      auto child = child_edge.node();
      if (child == nullptr) break;

      if (child->GetNInFlight() > 0) {
        std::cerr << child->DebugString() << std::endl;
        ++nonzero_node_count;
      }

      auto child_low_node = child->GetLowNode().get();
      if (child_low_node != nullptr &&
          (seen.find(child_low_node) == seen.end())) {
        seen.insert(child_low_node);
        unvisited_fifo.push_back(child);
      }
    }
  }

  if (nonzero_node_count > 0) {
    std::cerr << "GetNInFlight() is nonzero on " << nonzero_node_count
              << " nodes" << std::endl;
    return false;
  }

  return true;
}

void Node::SortEdges() const {
  assert(low_node_);
  low_node_->SortEdges();
}

static constexpr float wld_tolerance = 0.000001f;
static constexpr float m_tolerance = 0.000001f;

static bool WLDMInvariantsHold(float wl, float d, float m) {
  return -(1.0f + wld_tolerance) < wl && wl < (1.0f + wld_tolerance) &&  //
         -(0.0f + wld_tolerance) < d && d < (1.0f + wld_tolerance) &&    //
         -(0.0f + m_tolerance) < m &&                                    //
         std::abs(wl) + std::abs(d) < (1.0f + wld_tolerance);
}

bool Node::WLDMInvariantsHold() const {
  if (dag_classic::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
}

bool LowNode::WLDMInvariantsHold() const {
  if (dag_classic::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
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
  // Free old released nodes before adding new.
  released_nodes_.clear();
  // Release nodes from last move if any.
  current_head_->ReleaseChildrenExceptOne(new_head, released_nodes_);
  new_head = current_head_->GetChild();
  current_head_ =
      new_head ? new_head : current_head_->CreateSingleChildNode(move);
  history_.Append(move);
  moves_.push_back(move);
}

void NodeTree::TrimTreeAtHead() {
  current_head_->Trim();
}

bool NodeTree::ResetToPosition(const GameState& pos) {
  if (gamebegin_node_ && (history_.Starting() != pos.startpos)) {
    // Completely different position.
    DeallocateTree();
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<Node>(0);
  }

  history_.Reset(pos.startpos);
  moves_.clear();

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
  released_nodes_.emplace_back(std::move(gamebegin_node_));
  // Free all released nodes.
  released_nodes_.clear();
  current_head_ = nullptr;
}

}  // namespace dag_classic
}  // namespace lczero
